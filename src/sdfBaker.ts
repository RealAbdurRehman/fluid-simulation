import * as THREE from "three";
import {
  MeshBVH,
  computeBoundsTree,
  disposeBoundsTree,
  acceleratedRaycast,
} from "three-mesh-bvh";

(THREE.BufferGeometry.prototype as any).computeBoundsTree = computeBoundsTree;
(THREE.BufferGeometry.prototype as any).disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export interface BakedSDF {
  data: Float32Array;
  dims: THREE.Vector3;
  origin: THREE.Vector3;
  cellSize: number;
}

export function bakeSignedDistanceField(
  geometry: THREE.BufferGeometry,
  resolution = 64,
  padding = 0.5,
): BakedSDF {
  const geo = geometry.index ? geometry : geometry.toNonIndexed();
  if (!(geo as any).boundsTree) (geo as any).computeBoundsTree();
  const bvh: MeshBVH = (geo as any).boundsTree;

  geo.computeBoundingBox();
  const bbox = geo.boundingBox!.clone();
  bbox.min.subScalar(padding);
  bbox.max.addScalar(padding);

  const size = new THREE.Vector3();
  bbox.getSize(size);
  const longestAxis = Math.max(size.x, size.y, size.z);
  const cellSize = longestAxis / resolution;

  const dims = new THREE.Vector3(
    Math.max(2, Math.ceil(size.x / cellSize) + 1),
    Math.max(2, Math.ceil(size.y / cellSize) + 1),
    Math.max(2, Math.ceil(size.z / cellSize) + 1),
  );

  const data = new Float32Array(dims.x * dims.y * dims.z);

  const positions = geo.attributes.position as THREE.BufferAttribute;
  const indexAttr = geo.index;
  const indices = indexAttr ? (indexAttr.array as ArrayLike<number>) : null;

  const point = new THREE.Vector3();
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const edge1 = new THREE.Vector3();
  const edge2 = new THREE.Vector3();
  const faceNormal = new THREE.Vector3();
  const toPoint = new THREE.Vector3();

  let i = 0;
  for (let z = 0; z < dims.z; z++) {
    const pz = bbox.min.z + z * cellSize;
    for (let y = 0; y < dims.y; y++) {
      const py = bbox.min.y + y * cellSize;
      for (let x = 0; x < dims.x; x++, i++) {
        point.set(bbox.min.x + x * cellSize, py, pz);

        const hit = bvh.closestPointToPoint(point);
        if (!hit || hit.faceIndex < 0) {
          data[i] = 1e6;
          continue;
        }

        const dist = hit.distance;
        const closest = hit.point;
        const f = hit.faceIndex;

        let i0: number, i1: number, i2: number;
        if (indices) {
          i0 = indices[f * 3 + 0];
          i1 = indices[f * 3 + 1];
          i2 = indices[f * 3 + 2];
        } else {
          i0 = f * 3 + 0;
          i1 = f * 3 + 1;
          i2 = f * 3 + 2;
        }

        vA.fromBufferAttribute(positions, i0);
        vB.fromBufferAttribute(positions, i1);
        vC.fromBufferAttribute(positions, i2);
        edge1.subVectors(vB, vA);
        edge2.subVectors(vC, vA);
        faceNormal.crossVectors(edge1, edge2);

        toPoint.subVectors(point, closest);
        const signed = toPoint.dot(faceNormal);

        data[i] = signed < 0 ? -dist : dist;
      }
    }
  }

  return { data, dims, origin: bbox.min.clone(), cellSize };
}

export function bakeSignedDistanceFieldAsync(
  geometry: THREE.BufferGeometry,
  resolution = 64,
  padding = 0.5,
): Promise<BakedSDF> {
  const geo = geometry.index ? geometry : geometry.toNonIndexed();
  const posAttr = geo.attributes.position as THREE.BufferAttribute;

  const positions = new Float32Array(posAttr.array as ArrayLike<number>);
  let indices: Uint32Array | null = null;
  if (geo.index) {
    const arr = geo.index.array as ArrayLike<number>;
    indices =
      arr instanceof Uint32Array ? new Uint32Array(arr) : new Uint32Array(arr);
  }

  return new Promise((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./sdfWorker.ts", import.meta.url), {
        type: "module",
      });
    } catch (err) {
      console.warn("Worker unavailable, falling back to sync bake.", err);
      try {
        const result = bakeSignedDistanceField(geometry, resolution, padding);
        resolve(result);
      } catch (e) {
        reject(e);
      }

      return;
    }

    worker.onmessage = (e) => {
      const { data, dims, origin, cellSize } = e.data;
      resolve({
        data,
        dims: new THREE.Vector3(dims[0], dims[1], dims[2]),
        origin: new THREE.Vector3(origin[0], origin[1], origin[2]),
        cellSize,
      });
      worker.terminate();
    };

    worker.onerror = (err) => {
      reject(err);
      worker.terminate();
    };

    const transfer: Transferable[] = [positions.buffer];
    if (indices) transfer.push(indices.buffer);

    worker.postMessage({ positions, indices, resolution, padding }, transfer);
  });
}
