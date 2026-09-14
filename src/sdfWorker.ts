import * as THREE from "three";
import { bakeSignedDistanceField } from "./sdfBaker";

interface BakeRequest {
  positions: Float32Array;
  indices: Uint32Array | null;
  resolution: number;
  padding: number;
}

self.onmessage = (e: MessageEvent<BakeRequest>) => {
  const { positions, indices, resolution, padding } = e.data;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (indices) geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  const result = bakeSignedDistanceField(geometry, resolution, padding);
  (self as any).postMessage(
    {
      data: result.data,
      dims: [result.dims.x, result.dims.y, result.dims.z],
      origin: [result.origin.x, result.origin.y, result.origin.z],
      cellSize: result.cellSize,
    },
    [result.data.buffer],
  );
};
