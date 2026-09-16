import * as THREE from "three";

export interface TerrainData {
  resolution: number;
  extentX: number;
  extentZ: number;
  heightScale: number;
  heights: Float32Array;
}

function hash2(x: number, y: number, seed: number): number {
  let h = (x | 0) * 374761393 + (y | 0) * 668265263 + (seed | 0) * 69069;
  h = (h ^ (h >> 13)) * 1274126177;
  h = h ^ (h >> 16);

  return (h >>> 0) / 4294967296;
}

function smoothNoise(x: number, y: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);

  const xf = x - xi;
  const yf = y - yi;

  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);

  const a = hash2(xi, yi, seed);
  const b = hash2(xi + 1, yi, seed);
  const c = hash2(xi, yi + 1, seed);
  const d = hash2(xi + 1, yi + 1, seed);

  const ab = a + (b - a) * u;
  const cd = c + (d - c) * u;

  return ab + (cd - ab) * v;
}

function fbm(x: number, y: number, octaves: number, seed: number): number {
  let amp = 0.5;
  let freq = 1.0;

  let sum = 0;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * smoothNoise(x * freq, y * freq, seed + o * 17);
    norm += amp;
    amp *= 0.5;
    freq *= 2.0;
  }

  return sum / Math.max(norm, 1e-6);
}

export function generateTerrain(
  resolution: number,
  extentX: number,
  extentZ: number,
  heightScale: number,
  seed: number,
): TerrainData {
  const N = Math.max(4, Math.floor(resolution));
  const heights = new Float32Array(N * N);
  const scale = 2.5;

  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const u = i / (N - 1);
      const v = j / (N - 1);

      let h = fbm(u * scale, v * scale, 6, seed);

      const ridge =
        1.0 -
        Math.abs(fbm(u * scale * 2, v * scale * 2, 4, seed + 100) * 2 - 1);
      h = h * 0.7 + ridge * 0.3 * h;

      const cx = u - 0.5;
      const cz = v - 0.5;
      const r = Math.sqrt(cx * cx + cz * cz);
      const bowl = THREE.MathUtils.smoothstep(r, 0.05, 0.45);

      h *= bowl;
      heights[j * N + i] = Math.pow(THREE.MathUtils.clamp(h, 0, 1), 1.5);
    }
  }

  return { resolution: N, extentX, extentZ, heightScale, heights };
}

export function terrainToGeometry(
  terrain: TerrainData,
  baseY: number,
  depth: number = 3.0,
): THREE.BufferGeometry {
  const { resolution: N, extentX, extentZ, heightScale, heights } = terrain;
  const geo = new THREE.BufferGeometry();
  const halfX = extentX * 0.5;
  const halfZ = extentZ * 0.5;
  const cellX = extentX / (N - 1);
  const cellZ = extentZ / (N - 1);
  const bottomY = baseY - depth;

  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  const sampleTopY = (i: number, j: number) => {
    const ii = Math.max(0, Math.min(N - 1, i));
    const jj = Math.max(0, Math.min(N - 1, j));
    return baseY + heights[jj * N + ii] * heightScale;
  };

  const topBase = 0;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const x = -halfX + i * cellX;
      const z = -halfZ + j * cellZ;
      const y = sampleTopY(i, j);
      positions.push(x, y, z);

      const dx = (sampleTopY(i + 1, j) - sampleTopY(i - 1, j)) / (2 * cellX);
      const dz = (sampleTopY(i, j + 1) - sampleTopY(i, j - 1)) / (2 * cellZ);
      const nx = -dx;
      const ny = 1.0;
      const nz = -dz;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      normals.push(nx / len, ny / len, nz / len);
    }
  }

  for (let j = 0; j < N - 1; j++) {
    for (let i = 0; i < N - 1; i++) {
      const a = topBase + j * N + i;
      const b = a + 1;
      const c = a + N;
      const d = c + 1;
      indices.push(a, c, b);
      indices.push(b, c, d);
    }
  }

  const botBase = positions.length / 3;
  positions.push(-halfX, bottomY, -halfZ);
  positions.push(halfX, bottomY, -halfZ);
  positions.push(halfX, bottomY, halfZ);
  positions.push(-halfX, bottomY, halfZ);
  for (let k = 0; k < 4; k++) normals.push(0, -1, 0);

  indices.push(botBase + 0, botBase + 1, botBase + 2);
  indices.push(botBase + 0, botBase + 2, botBase + 3);

  const sides: {
    pointAt: (k: number) => [number, number, number];
    normal: [number, number, number];
  }[] = [
    {
      pointAt: (k) => [-halfX + k * cellX, sampleTopY(k, 0), -halfZ],
      normal: [0, 0, -1],
    },
    {
      pointAt: (k) => [halfX, sampleTopY(N - 1, k), -halfZ + k * cellZ],
      normal: [1, 0, 0],
    },
    {
      pointAt: (k) => [halfX - k * cellX, sampleTopY(N - 1 - k, N - 1), halfZ],
      normal: [0, 0, 1],
    },
    {
      pointAt: (k) => [-halfX, sampleTopY(0, N - 1 - k), halfZ - k * cellZ],
      normal: [-1, 0, 0],
    },
  ];

  for (const side of sides) {
    const sideBase = positions.length / 3;

    for (let k = 0; k < N; k++) {
      const p = side.pointAt(k);
      positions.push(p[0], p[1], p[2]);
      normals.push(side.normal[0], side.normal[1], side.normal[2]);
    }

    for (let k = 0; k < N; k++) {
      const p = side.pointAt(k);
      positions.push(p[0], bottomY, p[2]);
      normals.push(side.normal[0], side.normal[1], side.normal[2]);
    }

    for (let k = 0; k < N - 1; k++) {
      const tk = sideBase + k;
      const tk1 = sideBase + k + 1;
      const bk = sideBase + N + k;
      const bk1 = sideBase + N + k + 1;

      indices.push(bk, tk, tk1);
      indices.push(bk, tk1, bk1);
    }
  }

  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geo.setIndex(indices);
  geo.computeBoundingSphere();

  return geo;
}
