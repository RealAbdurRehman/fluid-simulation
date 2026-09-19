import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { ConvexHull } from "three/examples/jsm/math/ConvexHull.js";

import type { ModelDef } from "./models";
import {
  createDefaultMaterial,
  type ModelMaterial,
  type TextureSource,
} from "./materials";

export type { ModelMaterial } from "./materials";

export interface LoadedModel {
  geometry: THREE.BufferGeometry;
  materials: ModelMaterial[];
  unitVolume: number;
  unitInertia: THREE.Vector3;
  unitHalfExtents: THREE.Vector3;
  hullPoints: Float32Array;
  hullCount: number;
}

export async function loadModel(def: ModelDef): Promise<LoadedModel> {
  let raw: THREE.BufferGeometry;
  let materials: ModelMaterial[];

  if (def.build) {
    raw = def.build();
    materials = [createDefaultMaterial()];

    raw.clearGroups();
    ensureUv(raw);
    raw.addGroup(
      0,
      raw.index ? raw.index.count : raw.attributes.position.count,
      0,
    );
  } else ({ geometry: raw, materials } = await loadGLTFMerged(def.url!));

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
    materials,
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
  for (let i = 0; i < count; i += stride)
    pts.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)));

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

async function loadGLTFMerged(
  url: string,
): Promise<{ geometry: THREE.BufferGeometry; materials: ModelMaterial[] }> {
  const gltf = await new GLTFLoader().loadAsync(url);
  gltf.scene.updateMatrixWorld(true);

  const parts: THREE.BufferGeometry[] = [];
  const materials: ModelMaterial[] = [];
  const materialCache = new Map<THREE.Material, ModelMaterial>();

  const convert = (m: THREE.Material | undefined): ModelMaterial => {
    if (!m) return createDefaultMaterial();

    let c = materialCache.get(m);
    if (!c) {
      c = toModelMaterial(m);
      materialCache.set(m, c);
    }

    return c;
  };

  gltf.scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;

    const multi = Array.isArray(mesh.material);
    const matList = multi
      ? (mesh.material as THREE.Material[])
      : [mesh.material as THREE.Material];

    const ranges =
      multi && mesh.geometry.groups.length > 0
        ? mesh.geometry.groups.map((g) => ({
            start: g.start,
            count: g.count,
            materialIndex: g.materialIndex ?? 0,
          }))
        : [{ start: 0, count: Infinity, materialIndex: 0 }];

    const base = sanitize(mesh.geometry.clone());

    for (const r of ranges) {
      const part =
        r.count === Infinity ? base : sliceGeometry(base, r.start, r.count);
      part.applyMatrix4(mesh.matrixWorld);

      if (!part.attributes.normal) part.computeVertexNormals();
      ensureIndexed(part);

      parts.push(part);
      materials.push(convert(matList[r.materialIndex]));
    }
  });

  if (parts.length === 0) throw new Error(`No meshes found at ${url}`);

  let geometry: THREE.BufferGeometry;
  if (parts.length === 1) {
    geometry = parts[0];
    geometry.clearGroups();
    geometry.addGroup(0, geometry.index!.count, 0);
  } else {
    const merged = mergeGeometries(parts, true);
    if (!merged) throw new Error(`Failed to merge geometries from ${url}`);

    geometry = merged;
  }

  return { geometry, materials };
}

function toModelMaterial(mat: THREE.Material): ModelMaterial {
  const m = mat as THREE.MeshStandardMaterial;

  const rgb = { r: 1, g: 1, b: 1 };
  if (m.color) m.color.getRGB(rgb, THREE.SRGBColorSpace);

  const tex = m.map ?? null;
  const usable =
    tex && !(tex as any).isCompressedTexture && tex.image ? tex : null;
  const map = usable ? (usable.image as TextureSource) : null;

  const plainWhite = !map && rgb.r > 0.99 && rgb.g > 0.99 && rgb.b > 0.99;

  return {
    color: [rgb.r, rgb.g, rgb.b],
    map,
    wrapS: toAddressMode(usable?.wrapS ?? THREE.RepeatWrapping),
    wrapT: toAddressMode(usable?.wrapT ?? THREE.RepeatWrapping),

    tintWithObjectColor: plainWhite,
  };
}

function toAddressMode(w: THREE.Wrapping): GPUAddressMode {
  if (w === THREE.ClampToEdgeWrapping) return "clamp-to-edge";
  if (w === THREE.MirroredRepeatWrapping) return "mirror-repeat";
  return "repeat";
}

function sanitize(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const keep = ["position", "normal", "uv"];
  for (const name of Object.keys(g.attributes))
    if (!keep.includes(name)) g.deleteAttribute(name);

  for (const name of keep) {
    const a = g.attributes[name];
    if (a) g.setAttribute(name, toFloat32(a));
  }

  ensureUv(g);
  g.morphAttributes = {};
  g.clearGroups();

  return g;
}

function toFloat32(
  a: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
): THREE.BufferAttribute {
  const n = a.itemSize;
  const out = new Float32Array(a.count * n);
  for (let i = 0; i < a.count; i++) {
    out[i * n] = a.getX(i);
    if (n > 1) out[i * n + 1] = a.getY(i);
    if (n > 2) out[i * n + 2] = a.getZ(i);
  }

  return new THREE.BufferAttribute(out, n);
}

function ensureUv(g: THREE.BufferGeometry): void {
  if (g.attributes.uv) return;

  const count = g.attributes.position.count;
  g.setAttribute(
    "uv",
    new THREE.BufferAttribute(new Float32Array(count * 2), 2),
  );
}

function ensureIndexed(g: THREE.BufferGeometry): void {
  if (g.index) return;

  const n = g.attributes.position.count;
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;

  g.setIndex(new THREE.BufferAttribute(idx, 1));
}

function sliceGeometry(
  src: THREE.BufferGeometry,
  start: number,
  count: number,
): THREE.BufferGeometry {
  const out = new THREE.BufferGeometry();

  if (src.index) {
    const arr = src.index.array as Uint16Array | Uint32Array;
    const end = Math.min(arr.length, start + count);
    for (const name of Object.keys(src.attributes))
      out.setAttribute(name, src.attributes[name].clone());

    out.setIndex(new THREE.BufferAttribute(arr.slice(start, end), 1));
    return out;
  }

  for (const name of Object.keys(src.attributes)) {
    const a = src.attributes[name] as THREE.BufferAttribute;
    const end = Math.min(a.count, start + count);
    out.setAttribute(
      name,
      new THREE.BufferAttribute(
        (a.array as Float32Array).slice(start * a.itemSize, end * a.itemSize),
        a.itemSize,
      ),
    );
  }

  return out;
}
