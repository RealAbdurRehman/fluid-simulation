import * as THREE from "three";

export interface PlanetData {
  radius: number;
  heightScale: number;
  noiseScale: number;
  resolution: number;
  heights: Float32Array;
}

export interface PlanetSampler {
  radius: number;
  heightScale: number;
  heightAt(dir: THREE.Vector3): number;
}

function hash3(x: number, y: number, z: number, seed: number): number {
  let h =
    (x | 0) * 374761393 +
    (y | 0) * 668265263 +
    (z | 0) * 2147483647 +
    (seed | 0) * 69069;
  h = (h ^ (h >> 13)) * 1274126177;
  h = h ^ (h >> 16);

  return (h >>> 0) / 4294967296;
}

function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x),
    yi = Math.floor(y),
    zi = Math.floor(z);
  const xf = x - xi,
    yf = y - yi,
    zf = z - zi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const w = zf * zf * (3 - 2 * zf);

  const c000 = hash3(xi, yi, zi, seed);
  const c100 = hash3(xi + 1, yi, zi, seed);
  const c010 = hash3(xi, yi + 1, zi, seed);
  const c110 = hash3(xi + 1, yi + 1, zi, seed);
  const c001 = hash3(xi, yi, zi + 1, seed);
  const c101 = hash3(xi + 1, yi, zi + 1, seed);
  const c011 = hash3(xi, yi + 1, zi + 1, seed);
  const c111 = hash3(xi + 1, yi + 1, zi + 1, seed);

  const x00 = c000 + (c100 - c000) * u;
  const x10 = c010 + (c110 - c010) * u;
  const x01 = c001 + (c101 - c001) * u;
  const x11 = c011 + (c111 - c011) * u;

  const y0 = x00 + (x10 - x00) * v;
  const y1 = x01 + (x11 - x01) * v;

  return y0 + (y1 - y0) * w;
}

function fbm3(
  x: number,
  y: number,
  z: number,
  octaves: number,
  seed: number,
): number {
  let sum = 0,
    amp = 0.5,
    freq = 1,
    norm = 0;

  for (let o = 0; o < octaves; o++) {
    sum += amp * valueNoise3(x * freq, y * freq, z * freq, seed + o * 17);
    norm += amp;
    amp *= 0.5;
    freq *= 2.0;
  }

  return sum / Math.max(norm, 1e-6);
}

export function faceUVToDir(
  face: number,
  s: number,
  t: number,
  out: THREE.Vector3 = new THREE.Vector3(),
): THREE.Vector3 {
  switch (face) {
    case 0:
      out.set(1, -t, -s);
      break;
    case 1:
      out.set(-1, -t, s);
      break;
    case 2:
      out.set(s, 1, t);
      break;
    case 3:
      out.set(s, -1, -t);
      break;
    case 4:
      out.set(s, -t, 1);
      break;
    case 5:
      out.set(-s, -t, -1);
      break;
    default:
      out.set(0, 1, 0);
  }

  return out.normalize();
}

export function dirToFaceUV(dir: THREE.Vector3): {
  face: number;
  u: number;
  v: number;
} {
  const ax = Math.abs(dir.x),
    ay = Math.abs(dir.y),
    az = Math.abs(dir.z);

  let face: number, s: number, t: number;
  if (ax >= ay && ax >= az) {
    if (dir.x > 0) {
      face = 0;
      s = -dir.z / ax;
      t = -dir.y / ax;
    } else {
      face = 1;
      s = dir.z / ax;
      t = -dir.y / ax;
    }
  } else if (ay >= az) {
    if (dir.y > 0) {
      face = 2;
      s = dir.x / ay;
      t = dir.z / ay;
    } else {
      face = 3;
      s = dir.x / ay;
      t = -dir.z / ay;
    }
  } else {
    if (dir.z > 0) {
      face = 4;
      s = dir.x / az;
      t = -dir.y / az;
    } else {
      face = 5;
      s = -dir.x / az;
      t = -dir.y / az;
    }
  }

  return { face, u: (s + 1) * 0.5, v: (t + 1) * 0.5 };
}

function shapeHeight(
  dir: THREE.Vector3,
  noiseScale: number,
  seed: number,
): number {
  const nx = dir.x * noiseScale;
  const ny = dir.y * noiseScale;
  const nz = dir.z * noiseScale;

  const ridge1 =
    1.0 - Math.abs(fbm3(nx * 1.2, ny * 1.2, nz * 1.2, 5, seed) * 2.0 - 1.0);
  const ridge2 =
    1.0 -
    Math.abs(fbm3(nx * 3.0, ny * 3.0, nz * 3.0, 5, seed + 100) * 2.0 - 1.0);
  const detail = fbm3(nx * 7.0, ny * 7.0, nz * 7.0, 4, seed + 200);

  let h = ridge1 * 0.45 + ridge2 * 0.35 + detail * 0.2;
  h = (h - 0.65) * 2.5 + 0.5;

  return THREE.MathUtils.clamp(h, 0.0, 1.0);
}

export function generatePlanet(
  radius: number,
  heightScale: number,
  noiseScale: number,
  resolution: number,
  seed: number,
): PlanetData {
  const N = Math.max(4, Math.floor(resolution));
  const heights = new Float32Array(6 * N * N);
  const dir = new THREE.Vector3();

  for (let face = 0; face < 6; face++) {
    const base = face * N * N;
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const s = (i / (N - 1)) * 2 - 1;
        const t = (j / (N - 1)) * 2 - 1;

        faceUVToDir(face, s, t, dir);
        heights[base + j * N + i] = shapeHeight(dir, noiseScale, seed);
      }
    }
  }

  return { radius, heightScale, noiseScale, resolution: N, heights };
}

export function createPlanetSampler(planet: PlanetData): PlanetSampler {
  const N = planet.resolution;
  const hs = planet.heightScale;
  const heights = planet.heights;

  return {
    radius: planet.radius,
    heightScale: hs,
    heightAt(dir: THREE.Vector3): number {
      const { face, u, v } = dirToFaceUV(dir);
      const uu = THREE.MathUtils.clamp(u, 0, 1) * (N - 1);
      const vv = THREE.MathUtils.clamp(v, 0, 1) * (N - 1);
      const i0 = Math.floor(uu),
        j0 = Math.floor(vv);
      const i1 = Math.min(i0 + 1, N - 1),
        j1 = Math.min(j0 + 1, N - 1);
      const fx = uu - i0,
        fy = vv - j0;

      const base = face * N * N;
      const h00 = heights[base + j0 * N + i0];
      const h10 = heights[base + j0 * N + i1];
      const h01 = heights[base + j1 * N + i0];
      const h11 = heights[base + j1 * N + i1];

      const top = h00 + (h10 - h00) * fx;
      const bot = h01 + (h11 - h01) * fx;
      return (top + (bot - top) * fy) * hs;
    },
  };
}

export function planetToGeometry(planet: PlanetData): THREE.BufferGeometry {
  const N = planet.resolution;
  const R = planet.radius;
  const hs = planet.heightScale;
  const perFace = N * N;
  const totalVerts = 6 * perFace;

  const positions = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const dir = new THREE.Vector3();

  for (let face = 0; face < 6; face++) {
    const base = face * perFace;
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const u = i / (N - 1);
        const v = j / (N - 1);
        const s = u * 2 - 1;
        const t = v * 2 - 1;
        faceUVToDir(face, s, t, dir);

        const h = planet.heights[base + j * N + i] * hs;
        const r = R + h;
        const idx = base + j * N + i;

        positions[idx * 3 + 0] = dir.x * r;
        positions[idx * 3 + 1] = dir.y * r;
        positions[idx * 3 + 2] = dir.z * r;

        uvs[idx * 2 + 0] = u;
        uvs[idx * 2 + 1] = v;
      }
    }
  }

  const indicesPerFace = (N - 1) * (N - 1) * 6;
  const indices = new Uint32Array(6 * indicesPerFace);
  let o = 0;

  for (let face = 0; face < 6; face++) {
    const base = face * perFace;
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const a = base + j * N + i;
        const b = a + 1;
        const c = a + N;
        const d = c + 1;
        indices[o++] = a;
        indices[o++] = b;
        indices[o++] = c;
        indices[o++] = b;
        indices[o++] = d;
        indices[o++] = c;
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.computeVertexNormals();

  const normals = geo.attributes.normal as THREE.BufferAttribute;
  const colors = new Float32Array(totalVerts * 3);
  const nrm = new THREE.Vector3();
  const radial = new THREE.Vector3();

  const rock: [number, number, number] = [0.34, 0.29, 0.26];
  const grass: [number, number, number] = [0.2, 0.42, 0.16];
  const sand: [number, number, number] = [0.72, 0.63, 0.39];
  const snow: [number, number, number] = [0.92, 0.94, 0.97];
  const deep: [number, number, number] = [0.13, 0.2, 0.24];

  const seaLevel = 0.12 * hs;
  const snowLine = 0.75 * hs;

  for (let face = 0; face < 6; face++) {
    const base = face * perFace;
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const idx = base + j * N + i;

        nrm
          .set(normals.getX(idx), normals.getY(idx), normals.getZ(idx))
          .normalize();
        radial
          .set(
            positions[idx * 3 + 0],
            positions[idx * 3 + 1],
            positions[idx * 3 + 2],
          )
          .normalize();

        const slope = nrm.dot(radial);
        const h = planet.heights[idx] * hs;

        let c: [number, number, number];

        if (h < seaLevel) {
          c = deep;
        } else if (h > snowLine && slope > 0.55) c = snow;
        else if (slope < 0.75) {
          const t = THREE.MathUtils.clamp((0.75 - slope) / 0.4, 0, 1);
          c = [
            sand[0] + (rock[0] - sand[0]) * t,
            sand[1] + (rock[1] - sand[1]) * t,
            sand[2] + (rock[2] - sand[2]) * t,
          ];
        } else if (h < seaLevel + 0.2 * hs) c = sand;
        else c = grass;

        colors[idx * 3 + 0] = c[0];
        colors[idx * 3 + 1] = c[1];
        colors[idx * 3 + 2] = c[2];
      }
  }

  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeBoundingSphere();

  return geo;
}
