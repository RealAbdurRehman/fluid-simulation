import * as THREE from "three";
import { type TerrainData, type DirtyRect, markDirty } from "./terrain";

export interface SculptState {
  active: boolean;
  mode: "raise" | "lower" | "smooth" | "flatten";
  radius: number;
  strength: number;
  minHeight: number;
  maxHeight: number;
  flattenTarget: number;
  cursorX: number;
  cursorZ: number;
  cursorValid: boolean;
  lastX: number;
  lastZ: number;
  hasLast: boolean;
}

export function createSculptState(): SculptState {
  return {
    active: false,
    mode: "raise",
    radius: 2.0,
    strength: 3.0,
    minHeight: 0.0,
    maxHeight: 2.0,
    flattenTarget: 0,
    cursorX: 0,
    cursorZ: 0,
    cursorValid: false,
    lastX: 0,
    lastZ: 0,
    hasLast: false,
  };
}

const _inv = new THREE.Quaternion();
const _o = new THREE.Vector3();
const _d = new THREE.Vector3();

export function raycastTerrain(
  rayOriginWorld: THREE.Vector3,
  rayDirWorld: THREE.Vector3,
  containerQuat: THREE.Quaternion,
  terrain: TerrainData,
  baseY: number,
): { x: number; z: number } | null {
  _inv.copy(containerQuat).invert();
  _o.copy(rayOriginWorld).applyQuaternion(_inv);
  _d.copy(rayDirWorld).applyQuaternion(_inv);

  const N = terrain.resolution;
  const halfX = terrain.extentX * 0.5;
  const halfZ = terrain.extentZ * 0.5;
  const hs = terrain.heightScale;
  const heights = terrain.heights;

  const sample = (x: number, z: number): number => {
    const cellX = terrain.extentX / (N - 1);
    const cellZ = terrain.extentZ / (N - 1);
    const u = Math.min(Math.max((x + halfX) / cellX, 0), N - 1);
    const v = Math.min(Math.max((z + halfZ) / cellZ, 0), N - 1);
    const i0 = u | 0;
    const j0 = v | 0;
    const i1 = Math.min(i0 + 1, N - 1);
    const j1 = Math.min(j0 + 1, N - 1);
    const fx = u - i0;
    const fy = v - j0;
    const h00 = heights[j0 * N + i0];
    const h10 = heights[j0 * N + i1];
    const h01 = heights[j1 * N + i0];
    const h11 = heights[j1 * N + i1];
    return (
      baseY +
      ((h00 * (1 - fx) + h10 * fx) * (1 - fy) +
        (h01 * (1 - fx) + h11 * fx) * fy) *
        hs
    );
  };

  const STEPS = 64;
  const STEP = 4.0;
  let tPrev = 0;
  let abovePrev = _o.y > sample(_o.x, _o.z);

  for (let i = 1; i <= STEPS; i++) {
    const t = i * STEP;
    const px = _o.x + _d.x * t;
    const py = _o.y + _d.y * t;
    const pz = _o.z + _d.z * t;

    if (
      px < -halfX - 4 ||
      px > halfX + 4 ||
      pz < -halfZ - 4 ||
      pz > halfZ + 4
    ) {
      if (py < baseY - 1) return null;
      tPrev = t;
      abovePrev = false;
      continue;
    }

    const above = py > sample(px, pz);
    if (abovePrev && !above) {
      let lo = tPrev;
      let hi = t;
      for (let k = 0; k < 10; k++) {
        const m = (lo + hi) * 0.5;
        const mx = _o.x + _d.x * m;
        const my = _o.y + _d.y * m;
        const mz = _o.z + _d.z * m;
        if (my > sample(mx, mz)) lo = m;
        else hi = m;
      }
      const tHit = (lo + hi) * 0.5;
      return { x: _o.x + _d.x * tHit, z: _o.z + _d.z * tHit };
    }
    tPrev = t;
    abovePrev = above;
  }
  return null;
}

export function sculptFrame(
  state: SculptState,
  terrain: TerrainData,
  dt: number,
  dirty: DirtyRect,
): void {
  if (!state.active || !state.cursorValid) {
    state.hasLast = false;
    return;
  }
  const d = Math.min(dt, 1 / 30);

  if (!state.hasLast) {
    applyBrush(
      terrain,
      state.cursorX,
      state.cursorZ,
      state.strength * d,
      state,
      dirty,
    );
    state.lastX = state.cursorX;
    state.lastZ = state.cursorZ;
    state.hasLast = true;
    return;
  }

  const dx = state.cursorX - state.lastX;
  const dz = state.cursorZ - state.lastZ;
  const dist = Math.hypot(dx, dz);
  const spacing = state.radius * 0.35;
  const samples = Math.max(1, Math.ceil(dist / spacing));
  const subDelta = (state.strength * d) / samples;

  for (let s = 1; s <= samples; s++) {
    const t = s / samples;
    applyBrush(
      terrain,
      state.lastX + dx * t,
      state.lastZ + dz * t,
      subDelta,
      state,
      dirty,
    );
  }
  state.lastX = state.cursorX;
  state.lastZ = state.cursorZ;
}

function applyBrush(
  terrain: TerrainData,
  cx: number,
  cz: number,
  delta: number,
  state: SculptState,
  dirty: DirtyRect,
): void {
  const N = terrain.resolution;
  const halfX = terrain.extentX * 0.5;
  const halfZ = terrain.extentZ * 0.5;
  const cellX = terrain.extentX / (N - 1);
  const cellZ = terrain.extentZ / (N - 1);
  const fc = (cx + halfX) / cellX;
  const gc = (cz + halfZ) / cellZ;
  const radius = state.radius;
  const ri = Math.ceil(radius / cellX) + 1;
  const rj = Math.ceil(radius / cellZ) + 1;
  const i0 = Math.max(0, Math.floor(fc - ri));
  const i1 = Math.min(N - 1, Math.ceil(fc + ri));
  const j0 = Math.max(0, Math.floor(gc - rj));
  const j1 = Math.min(N - 1, Math.ceil(gc + rj));

  const r2 = radius * radius;
  const invH = 1 / Math.max(terrain.heightScale, 1e-4);
  const heights = terrain.heights;
  const lo = state.minHeight;
  const hi = Math.max(state.maxHeight, lo + 1e-4);

  for (let j = j0; j <= j1; j++) {
    const wz = -halfZ + j * cellZ;
    const dz = wz - cz;
    const dz2 = dz * dz;
    for (let i = i0; i <= i1; i++) {
      const wx = -halfX + i * cellX;
      const dx = wx - cx;
      const d2 = dx * dx + dz2;
      if (d2 > r2) continue;

      const t = Math.sqrt(d2) / radius;
      const w = 1 - t * t * (3 - 2 * t);
      const idx = j * N + i;
      const h = heights[idx];
      let nh = h;

      if (state.mode === "raise") {
        nh = h + delta * w * invH;
      } else if (state.mode === "lower") {
        nh = h - delta * w * invH;
      } else if (state.mode === "smooth") {
        const iL = i > 0 ? i - 1 : i;
        const iR = i < N - 1 ? i + 1 : i;
        const jU = j > 0 ? j - 1 : j;
        const jD = j < N - 1 ? j + 1 : j;
        const avg =
          (heights[j * N + iL] +
            heights[j * N + iR] +
            heights[jU * N + i] +
            heights[jD * N + i]) *
          0.25;
        nh = h + (avg - h) * Math.min(delta * w, 1);
      } else {
        nh = h + (state.flattenTarget - h) * Math.min(delta * w, 1);
      }

      if (nh < lo) nh = lo;
      else if (nh > hi) nh = hi;

      if (nh !== h) {
        heights[idx] = nh;
        markDirty(dirty, i, j);
      }
    }
  }
}
