import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { ConvexHull } from "three/examples/jsm/math/ConvexHull.js";

import type { ModelDef } from "./models";

export interface LoadedModel {
  geometry: THREE.BufferGeometry;
  unitVolume: number;
  unitInertia: THREE.Vector3;
  unitHalfExtents: THREE.Vector3;
  hullPoints: Float32Array;
  hullCount: number;
}

export async function loadModel(def: ModelDef): Promise<LoadedModel> {
  const raw = def.build ? def.build() : await loadGLTFMerged(def.url!);

  if (!raw.attributes.normal) raw.computeVertexNormals();

  raw.computeBoundingBox();
  raw.computeBoundingSphere();

  const center = new THREE.Vector3();
  raw.boundingBox!.getCenter(center);
  raw.translate(-center.x, -center.y, -center.z);

  const r = Math.max(raw.boundingSphere!.radius, 1e-4);
  raw.scale(1 / r, 1 / r, 1 / r);
  raw.computeBoundingBox();
  raw.computeBoundingSphere();

  const size = new THREE.Vector3();
  raw.boundingBox!.getSize(size);

  const unitHalfExtents = size.clone().multiplyScalar(0.5);
  const unitVolume = size.x * size.y * size.z * 0.52;
  const i = 0.4 * unitVolume;
  const unitInertia = new THREE.Vector3(i, i, i);

  const hullPoints = bakeHullPoints(raw);
  const hullCount = hullPoints.length / 3;

  return {
    geometry: raw,
    unitVolume,
    unitInertia,
    unitHalfExtents,
    hullPoints,
    hullCount,
  };
}

function bakeHullPoints(geo: THREE.BufferGeometry): Float32Array {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const count = pos.count;

  const stride = Math.max(1, Math.ceil(count / 2048));
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < count; i += stride) {
    pts.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)));
  }

  try {
    const hull = new ConvexHull().setFromPoints(pts);
    const uniq = new Set<THREE.Vector3>();
    for (const face of hull.faces) {
      let e = face.edge;
      do {
        uniq.add(e.head().point);
        e = e.next;
      } while (e !== face.edge);
    }
    const out = new Float32Array(uniq.size * 3);
    let i = 0;
    for (const v of uniq) {
      out[i++] = v.x;
      out[i++] = v.y;
      out[i++] = v.z;
    }
    return out;
  } catch {
    const out = new Float32Array(pts.length * 3);
    for (let i = 0; i < pts.length; i++) {
      out[i * 3 + 0] = pts[i].x;
      out[i * 3 + 1] = pts[i].y;
      out[i * 3 + 2] = pts[i].z;
    }
    return out;
  }
}

async function loadGLTFMerged(url: string): Promise<THREE.BufferGeometry> {
  const gltf = await new GLTFLoader().loadAsync(url);
  gltf.scene.updateMatrixWorld(true);

  const parts: THREE.BufferGeometry[] = [];
  gltf.scene.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    const g = sanitize(m.geometry.clone());
    g.applyMatrix4(m.matrixWorld);
    if (!g.attributes.normal) g.computeVertexNormals();
    parts.push(g);
  });

  if (parts.length === 0) throw new Error(`No meshes found at ${url}`);
  if (parts.length === 1) return parts[0];

  const merged = mergeGeometries(parts, false);
  if (!merged) throw new Error(`Failed to merge geometries from ${url}`);
  return merged;
}

function sanitize(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const keep = new Set(["position", "normal"]);
  for (const name of Object.keys(g.attributes)) {
    if (!keep.has(name)) g.deleteAttribute(name);
  }
  g.morphAttributes = {};
  return g;
}
