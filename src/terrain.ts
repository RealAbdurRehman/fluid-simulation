import * as THREE from "three";

export interface TerrainData {
  resolution: number;
  extent: number;
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
  extent: number,
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

  return { resolution: N, extent, heightScale, heights };
}

export function terrainToGeometry(
  terrain: TerrainData,
  baseY: number,
): THREE.BufferGeometry {
  const { resolution: N, extent, heightScale, heights } = terrain;
  const geo = new THREE.BufferGeometry();

  const positions = new Float32Array(N * N * 3);
  const normals = new Float32Array(N * N * 3);
  const uvs = new Float32Array(N * N * 2);

  const cell = extent / (N - 1);
  const half = extent * 0.5;

  const sample = (i: number, j: number) => {
    const ii = Math.max(0, Math.min(N - 1, i));
    const jj = Math.max(0, Math.min(N - 1, j));
    return heights[jj * N + ii] * heightScale;
  };

  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const idx = j * N + i;
      const x = -half + i * cell;
      const z = -half + j * cell;
      const y = heights[idx] * heightScale;

      positions[idx * 3 + 0] = x;
      positions[idx * 3 + 1] = baseY + y;
      positions[idx * 3 + 2] = z;

      const dx = (sample(i + 1, j) - sample(i - 1, j)) / (2 * cell);
      const dz = (sample(i, j + 1) - sample(i, j - 1)) / (2 * cell);

      const nx = -dx;
      const ny = 1.0;
      const nz = -dz;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);

      normals[idx * 3 + 0] = nx / len;
      normals[idx * 3 + 1] = ny / len;
      normals[idx * 3 + 2] = nz / len;

      uvs[idx * 2 + 0] = i / (N - 1);
      uvs[idx * 2 + 1] = j / (N - 1);
    }
  }

  const indices = new Uint32Array((N - 1) * (N - 1) * 6);

  let p = 0;
  for (let j = 0; j < N - 1; j++)
    for (let i = 0; i < N - 1; i++) {
      const a = j * N + i;
      const b = a + 1;
      const c = a + N;
      const d = c + 1;
      indices[p++] = a;
      indices[p++] = c;
      indices[p++] = b;
      indices[p++] = b;
      indices[p++] = c;
      indices[p++] = d;
    }

  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.computeBoundingSphere();

  return geo;
}
