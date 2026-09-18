import * as THREE from "three";

import { config, type ObjectSlotConfig } from "./config";
import type { FluidSimulationGPU, ObjectDescriptor } from "./simulation";
import type { SceneRenderer, ObjectVisual } from "./sceneRenderer";
import type { MeshRegistry } from "./meshRegistry";
import type { LoadedModel } from "./modelLoader";
import type { TerrainData } from "./terrain";
import { RigidBody } from "./rigidBody";

type RGBA = [number, number, number, number];

interface BodyTransform {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  linearVelocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;
}

interface ActiveBody {
  index: number;
  slot: ObjectSlotConfig;
  body: RigidBody;
  hullPoints: Float32Array;
  hullCount: number;
}

const MAX_HULL_PROBES = 48;

const ADDED_MASS = 0.5;

const MAX_LINEAR_SPEED = 40.0;
const MAX_ANGULAR_SPEED = 4.0;

const AIR_LINEAR_DAMPING = 0.4;
const AIR_ANGULAR_DAMPING = 1.2;

const WATER_ANGULAR_DAMPING = 6.0;
const BUOYANCY_TORQUE_CAP = 0.15;
const RIGHTING_STRENGTH = 0.4;

const TERRAIN_RESTITUTION = 0.08;
const TERRAIN_FRICTION = 0.78;
const PAIR_RESTITUTION = 0.2;
const PAIR_FRICTION = 0.85;

const TERRAIN_CONTACT_EPS = 0.5;
const PAIR_ITERATIONS = 3;

const _hullTmp = new THREE.Vector3();
const _localPos = new THREE.Vector3();
const _localVel = new THREE.Vector3();
const _localQuat = new THREE.Quaternion();
const _invContainerQuat = new THREE.Quaternion();
const _pairAxis = new THREE.Vector3();
const _pairAxisNeg = new THREE.Vector3();
const _relVel = new THREE.Vector3();
const _tangent = new THREE.Vector3();
const _tmpVecA = new THREE.Vector3();
const _tmpVecB = new THREE.Vector3();
const _tmpVecC = new THREE.Vector3();

const colorCache = new Map<string, RGBA>();

export interface SimUpdaters {
  updateBoundsRotation(dt: number): void;
  updateObjects(dt: number): void;
  requestBakeForSlot(index: number): void;
  spawnObject(index: number): void;
  setTerrain(data: TerrainData | null, baseY: number): void;
}

export function createSimUpdaters(
  simulation: FluidSimulationGPU,
  sceneRenderer: SceneRenderer,
  meshRegistry: MeshRegistry,
): SimUpdaters {
  return {
    ...createBoundsUpdater(simulation, sceneRenderer),
    ...createObjectsUpdater(simulation, sceneRenderer, meshRegistry),
  };
}

function colorForModel(id: string): RGBA {
  const cached = colorCache.get(id);
  if (cached) return cached;

  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;

  const hue = (((h % 360) + 360) % 360) / 360;
  const c = new THREE.Color().setHSL(hue, 0.65, 0.55);
  const rgba: RGBA = [c.r, c.g, c.b, 1.0];

  colorCache.set(id, rgba);
  return rgba;
}

function sampleTerrainHeight(
  data: TerrainData,
  baseY: number,
  x: number,
  z: number,
): number {
  const N = data.resolution;
  const cellX = data.extentX / (N - 1);
  const cellZ = data.extentZ / (N - 1);

  const u = THREE.MathUtils.clamp((x + data.extentX * 0.5) / cellX, 0, N - 1);
  const v = THREE.MathUtils.clamp((z + data.extentZ * 0.5) / cellZ, 0, N - 1);

  const i0x = Math.floor(u);
  const i0y = Math.floor(v);
  const i1x = Math.min(i0x + 1, N - 1);
  const i1y = Math.min(i0y + 1, N - 1);
  const fx = u - i0x;
  const fy = v - i0y;

  const h00 = data.heights[i0x + i0y * N];
  const h10 = data.heights[i1x + i0y * N];
  const h01 = data.heights[i0x + i1y * N];
  const h11 = data.heights[i1x + i1y * N];

  const h =
    (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy;

  return baseY + h * data.heightScale;
}

function createBoundsUpdater(
  simulation: FluidSimulationGPU,
  sceneRenderer: SceneRenderer,
): { updateBoundsRotation(dt: number): void } {
  const quaternion = new THREE.Quaternion();
  let tumbleTime = 0;

  function updateBoundsRotation(dt: number): void {
    let euler: THREE.Euler;
    if (config.boundsAutoTumble) {
      tumbleTime += dt;
      euler = new THREE.Euler(
        Math.sin(tumbleTime * 0.4) * 0.5,
        tumbleTime * 0.3,
        Math.cos(tumbleTime * 0.3) * 0.4,
      );
    } else
      euler = new THREE.Euler(
        THREE.MathUtils.degToRad(config.boundsRotationX),
        THREE.MathUtils.degToRad(config.boundsRotationY),
        THREE.MathUtils.degToRad(config.boundsRotationZ),
      );

    quaternion.setFromEuler(euler);
    simulation.setBoundsRotation(quaternion);
    sceneRenderer.setBoundsRotationQuat([
      quaternion.x,
      quaternion.y,
      quaternion.z,
      quaternion.w,
    ]);
  }

  return { updateBoundsRotation };
}

function createObjectsUpdater(
  simulation: FluidSimulationGPU,
  sceneRenderer: SceneRenderer,
  meshRegistry: MeshRegistry,
) {
  const slotCount = config.objects.length;

  const bodies: (RigidBody | null)[] = new Array(slotCount).fill(null);
  const hulls: (Float32Array | null)[] = new Array(slotCount).fill(null);
  const hullCounts: number[] = new Array(slotCount).fill(0);
  const lastModelId: string[] = config.objects.map((s) => s.modelId);
  const lastSize: number[] = config.objects.map((s) => s.size);
  const pendingTransform: (BodyTransform | null)[] = new Array(slotCount).fill(
    null,
  );

  const probePositions = new Float32Array(1024 * 4);
  let probeCount = 0;

  let terrainData: TerrainData | null = null;
  let terrainBaseY = 0;

  const visuals: ObjectVisual[] = Array.from({ length: slotCount }, () => ({
    visible: false,
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
    scale: 1,
    color: [1, 1, 1, 0],
  }));

  const active: ActiveBody[] = [];

  function setTerrain(data: TerrainData | null, baseY: number): void {
    terrainData = data;
    terrainBaseY = baseY;
  }

  function requestBakeForSlot(index: number): void {
    const slot = config.objects[index];
    if (slot.modelId === "none") return;

    meshRegistry.ensure(slot.modelId).catch(() => {});
  }

  function captureTransform(body: RigidBody): BodyTransform {
    return {
      position: body.position.clone(),
      quaternion: body.quaternion.clone(),
      linearVelocity: body.linearVelocity.clone(),
      angularVelocity: body.angularVelocity.clone(),
    };
  }

  function applyModelToSlot(
    index: number,
    model: LoadedModel,
    slot: ObjectSlotConfig,
    localRho: number,
    body: RigidBody,
  ): void {
    const s = slot.size;
    const volume = model.unitVolume * s * s * s;
    const mass = Math.max(slot.densityRatio * localRho * volume, 1e-4);
    const k = mass * s * s;

    body.volume = volume;
    body.mass = mass;
    body.inertiaLocal.set(
      model.unitInertia.x * k,
      model.unitInertia.y * k,
      model.unitInertia.z * k,
    );

    hulls[index] = model.hullPoints;
    hullCounts[index] = model.hullCount;
  }

  function spawnBody(index: number, transform: BodyTransform | null): void {
    const slot = config.objects[index];
    if (slot.modelId === "none") return;

    const model = meshRegistry.getLoaded(slot.modelId);
    if (!model) return;

    const refDensity = Math.max(config.targetDensity, 1e-4);
    const s = slot.size;
    const volume = model.unitVolume * s * s * s;
    const mass = Math.max(slot.densityRatio * refDensity * volume, 1e-4);
    const k = mass * s * s;

    const pos = transform
      ? transform.position
      : new THREE.Vector3(slot.posX, slot.posY, slot.posZ);

    const quat = transform
      ? transform.quaternion
      : new THREE.Quaternion().setFromEuler(
          new THREE.Euler(
            THREE.MathUtils.degToRad(slot.rotX),
            THREE.MathUtils.degToRad(slot.rotY),
            THREE.MathUtils.degToRad(slot.rotZ),
          ),
        );

    const body = new RigidBody({
      position: pos,
      quaternion: quat,
      mass,
      volume,
      densityRatio: slot.densityRatio,
      inertiaLocal: new THREE.Vector3(
        model.unitInertia.x * k,
        model.unitInertia.y * k,
        model.unitInertia.z * k,
      ),
      drag: slot.drag,
      angularDrag: slot.angularDrag,
    });

    if (transform) {
      body.linearVelocity.copy(transform.linearVelocity);
      body.angularVelocity.copy(transform.angularVelocity);
    }

    bodies[index] = body;
    hulls[index] = model.hullPoints;
    hullCounts[index] = model.hullCount;
  }

  function writeVisual(
    index: number,
    slot: ObjectSlotConfig,
    body: RigidBody,
  ): void {
    const visual = visuals[index];
    visual.visible = true;
    visual.position = [body.position.x, body.position.y, body.position.z];
    visual.quaternion = [
      body.quaternion.x,
      body.quaternion.y,
      body.quaternion.z,
      body.quaternion.w,
    ];
    visual.scale = slot.size;
    visual.color = colorForModel(slot.modelId);
    visual.meshId = slot.modelId;
  }

  function buildDescriptor(
    slot: ObjectSlotConfig,
    body: RigidBody,
  ): ObjectDescriptor | null {
    const handle = meshRegistry.get(slot.modelId);
    if (!handle) return null;

    return {
      type: "mesh",
      position: body.position,
      quaternion: body.quaternion,
      velocity: body.linearVelocity,
      restitution: 0.5,
      size: new THREE.Vector3(slot.size, 0, 0),
      mesh: handle,
    };
  }

  function placeProbesForBody(
    body: RigidBody,
    slot: ObjectSlotConfig,
    hullPoints: Float32Array,
    hullCount: number,
  ): number {
    const start = probeCount;
    const size = slot.size;

    {
      const o = probeCount * 4;
      probePositions[o + 0] = body.position.x;
      probePositions[o + 1] = body.position.y;
      probePositions[o + 2] = body.position.z;
      probePositions[o + 3] = 0;
      probeCount++;
    }

    if (hullCount === 0) return probeCount - start;

    const target = Math.min(hullCount, MAX_HULL_PROBES);
    const stride = Math.max(1, Math.floor(hullCount / target));

    let placed = 0;
    for (let j = 0; j < hullCount && placed < target; j += stride) {
      _hullTmp
        .set(
          hullPoints[j * 3 + 0] * size,
          hullPoints[j * 3 + 1] * size,
          hullPoints[j * 3 + 2] * size,
        )
        .applyQuaternion(body.quaternion)
        .add(body.position);

      const o = probeCount * 4;
      probePositions[o + 0] = _hullTmp.x;
      probePositions[o + 1] = _hullTmp.y;
      probePositions[o + 2] = _hullTmp.z;
      probePositions[o + 3] = 0;
      probeCount++;
      placed++;
    }

    return probeCount - start;
  }

  function applyFluidForces(
    body: RigidBody,
    slot: ObjectSlotConfig,
    localRho: number,
    g: number,
    dt: number,
    probeResults: Float32Array | null,
    probeBase: number,
    probeTotal: number,
  ): void {
    let submergedSum = 0;
    const centerSub = _tmpVecA.set(0, 0, 0);
    const avgFluidVel = _tmpVecB.set(0, 0, 0);

    if (probeResults) {
      for (let p = 0; p < probeTotal; p++) {
        const o = (probeBase + p) * 4;
        const density = probeResults[o + 0];
        const sub = THREE.MathUtils.clamp(density / localRho, 0, 1);

        _hullTmp.set(
          probePositions[o + 0],
          probePositions[o + 1],
          probePositions[o + 2],
        );

        centerSub.addScaledVector(_hullTmp, sub);
        avgFluidVel.x += probeResults[o + 1] * sub;
        avgFluidVel.y += probeResults[o + 2] * sub;
        avgFluidVel.z += probeResults[o + 3] * sub;
        submergedSum += sub;
      }
    }

    const submergedFraction = submergedSum / Math.max(probeTotal, 1);
    const Vsub = body.volume * submergedFraction;
    const fluidMass = localRho * Vsub;
    const mEff = body.mass + ADDED_MASS * fluidMass;
    const fs = body.mass / mEff;

    const dragSubmerged = Math.sqrt(submergedFraction);
    const dragFluidMass = localRho * body.volume * dragSubmerged;

    body.applyForceWorld(
      new THREE.Vector3(0, -body.mass * g * fs, 0),
      body.position,
      dt,
    );

    body.linearVelocity.multiplyScalar(
      Math.max(0, 1 - AIR_LINEAR_DAMPING * dt),
    );
    body.angularVelocity.multiplyScalar(
      Math.max(0, 1 - AIR_ANGULAR_DAMPING * dt),
    );

    if (submergedFraction > 1e-3 && probeResults && submergedSum > 1e-6) {
      centerSub.divideScalar(submergedSum);
      avgFluidVel.divideScalar(submergedSum);

      const buoyancy = _tmpVecC.set(0, fluidMass * g * fs, 0);

      body.applyForceWorld(buoyancy, body.position, dt);

      const rSub = _hullTmp.subVectors(centerSub, body.position);
      const torque = new THREE.Vector3().crossVectors(rSub, buoyancy);

      const torqueCap = body.mass * g * slot.size * BUOYANCY_TORQUE_CAP;
      const torqueLen = torque.length();
      if (torqueLen > torqueCap) torque.multiplyScalar(torqueCap / torqueLen);

      body.applyTorqueWorld(torque, dt);

      const relVel = new THREE.Vector3().subVectors(
        body.linearVelocity,
        avgFluidVel,
      );

      const dragForce = relVel.multiplyScalar(-slot.drag * dragFluidMass * fs);
      body.applyForceWorld(dragForce, body.position, dt);

      body.angularVelocity.multiplyScalar(
        Math.max(0, 1 - WATER_ANGULAR_DAMPING * submergedFraction * dt),
      );

      const localUp = new THREE.Vector3(0, 1, 0).applyQuaternion(
        body.quaternion,
      );
      const worldUp = new THREE.Vector3(0, 1, 0);
      const axis = new THREE.Vector3().crossVectors(localUp, worldUp);
      const sinAngle = axis.length();

      if (sinAngle > 1e-4) {
        axis.divideScalar(sinAngle);
        const angle = Math.asin(Math.min(sinAngle, 1.0));
        const strength = submergedFraction * body.mass * g * RIGHTING_STRENGTH;
        axis.multiplyScalar(angle * strength);
        body.applyTorqueWorld(axis, dt);
      }
    }

    const linSpeed = body.linearVelocity.length();
    if (linSpeed > MAX_LINEAR_SPEED)
      body.linearVelocity.multiplyScalar(MAX_LINEAR_SPEED / linSpeed);

    const angSpeed = body.angularVelocity.length();
    if (angSpeed > MAX_ANGULAR_SPEED)
      body.angularVelocity.multiplyScalar(MAX_ANGULAR_SPEED / angSpeed);
  }

  function clampToBounds(
    body: RigidBody,
    slot: ObjectSlotConfig,
    hullPoints: Float32Array,
    hullCount: number,
  ): void {
    const containerQuat = simulation.getBoundsQuaternion();
    _invContainerQuat.copy(containerQuat).invert();

    _localPos.copy(body.position).applyQuaternion(_invContainerQuat);
    _localVel.copy(body.linearVelocity).applyQuaternion(_invContainerQuat);
    _localQuat.copy(_invContainerQuat).multiply(body.quaternion);

    const wallClearance = config.particleSize * 3.0;
    const hx = config.boundsWidth / 2 - wallClearance;
    const hy = config.boundsHeight / 2 - wallClearance;
    const hz = config.boundsDepth / 2 - wallClearance;

    const size = slot.size;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;

    for (let i = 0; i < hullCount; i++) {
      _hullTmp
        .set(
          hullPoints[i * 3 + 0] * size,
          hullPoints[i * 3 + 1] * size,
          hullPoints[i * 3 + 2] * size,
        )
        .applyQuaternion(_localQuat);

      if (_hullTmp.x < minX) minX = _hullTmp.x;
      if (_hullTmp.x > maxX) maxX = _hullTmp.x;
      if (_hullTmp.y < minY) minY = _hullTmp.y;
      if (_hullTmp.y > maxY) maxY = _hullTmp.y;
      if (_hullTmp.z < minZ) minZ = _hullTmp.z;
      if (_hullTmp.z > maxZ) maxZ = _hullTmp.z;
    }

    if (_localPos.x + minX < -hx) {
      _localPos.x = -hx - minX;
      _localVel.x *= -config.collisionDamping;
    } else if (_localPos.x + maxX > hx) {
      _localPos.x = hx - maxX;
      _localVel.x *= -config.collisionDamping;
    }

    if (_localPos.y + minY < -hy) {
      _localPos.y = -hy - minY;
      _localVel.y *= -config.collisionDamping;
    } else if (_localPos.y + maxY > hy) {
      _localPos.y = hy - maxY;
      _localVel.y *= -config.collisionDamping;
    }

    if (_localPos.z + minZ < -hz) {
      _localPos.z = -hz - minZ;
      _localVel.z *= -config.collisionDamping;
    } else if (_localPos.z + maxZ > hz) {
      _localPos.z = hz - maxZ;
      _localVel.z *= -config.collisionDamping;
    }

    body.position.copy(_localPos).applyQuaternion(containerQuat);
    body.linearVelocity.copy(_localVel).applyQuaternion(containerQuat);
  }

  function resolveBodyTerrain(entry: ActiveBody): void {
    if (!terrainData) return;

    const { body, slot, hullPoints, hullCount } = entry;
    const size = slot.size;
    const containerQuat = simulation.getBoundsQuaternion();

    _invContainerQuat.copy(containerQuat).invert();
    _localPos.copy(body.position).applyQuaternion(_invContainerQuat);
    _localQuat.copy(_invContainerQuat).multiply(body.quaternion);

    let bestPen = 0;
    let bestX = 0;
    let bestZ = 0;

    for (let i = 0; i < hullCount; i++) {
      const hx = hullPoints[i * 3 + 0] * size;
      const hy = hullPoints[i * 3 + 1] * size;
      const hz = hullPoints[i * 3 + 2] * size;

      _hullTmp.set(hx, hy, hz).applyQuaternion(_localQuat);

      const wx = _hullTmp.x + _localPos.x;
      const wy = _hullTmp.y + _localPos.y;
      const wz = _hullTmp.z + _localPos.z;

      const terrainY = sampleTerrainHeight(terrainData, terrainBaseY, wx, wz);
      const pen = terrainY - wy;

      if (pen > bestPen) {
        bestPen = pen;
        bestX = wx;
        bestZ = wz;
      }
    }

    if (bestPen <= 0) return;

    const eps = TERRAIN_CONTACT_EPS;
    const hxp = sampleTerrainHeight(
      terrainData,
      terrainBaseY,
      bestX + eps,
      bestZ,
    );
    const hxm = sampleTerrainHeight(
      terrainData,
      terrainBaseY,
      bestX - eps,
      bestZ,
    );
    const hzp = sampleTerrainHeight(
      terrainData,
      terrainBaseY,
      bestX,
      bestZ + eps,
    );
    const hzm = sampleTerrainHeight(
      terrainData,
      terrainBaseY,
      bestX,
      bestZ - eps,
    );

    const dxH = hxp - hxm;
    const dzH = hzp - hzm;

    const nx = -dxH;
    const ny = 2 * eps;
    const nz = -dzH;
    const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    const ux = nx / nLen;
    const uy = ny / nLen;
    const uz = nz / nLen;

    _localPos.y += bestPen;
    body.position.copy(_localPos).applyQuaternion(containerQuat);

    _localVel.copy(body.linearVelocity).applyQuaternion(_invContainerQuat);

    const vn = _localVel.x * ux + _localVel.y * uy + _localVel.z * uz;

    if (vn < 0) {
      const vTx = _localVel.x - ux * vn;
      const vTy = _localVel.y - uy * vn;
      const vTz = _localVel.z - uz * vn;

      const bounce = -vn * TERRAIN_RESTITUTION;

      _localVel.set(
        vTx * TERRAIN_FRICTION + ux * bounce,
        vTy * TERRAIN_FRICTION + uy * bounce,
        vTz * TERRAIN_FRICTION + uz * bounce,
      );

      body.linearVelocity.copy(_localVel).applyQuaternion(containerQuat);
      body.angularVelocity.multiplyScalar(TERRAIN_FRICTION);
    }
  }

  function hullSupportAlong(
    hullPoints: Float32Array,
    hullCount: number,
    size: number,
    quat: THREE.Quaternion,
    axis: THREE.Vector3,
  ): number {
    let best = -Infinity;
    for (let i = 0; i < hullCount; i++) {
      _hullTmp
        .set(
          hullPoints[i * 3 + 0] * size,
          hullPoints[i * 3 + 1] * size,
          hullPoints[i * 3 + 2] * size,
        )
        .applyQuaternion(quat);

      const d = _hullTmp.x * axis.x + _hullTmp.y * axis.y + _hullTmp.z * axis.z;
      if (d > best) best = d;
    }
    return best;
  }

  function resolveBodyPairs(): void {
    for (let iteration = 0; iteration < PAIR_ITERATIONS; iteration++) {
      for (let a = 0; a < active.length; a++) {
        const A = active[a];
        const sizeA = A.slot.size;

        for (let b = a + 1; b < active.length; b++) {
          const B = active[b];
          const sizeB = B.slot.size;

          _pairAxis.subVectors(B.body.position, A.body.position);
          const dist = _pairAxis.length();
          if (dist < 1e-6) continue;
          if (dist > sizeA + sizeB) continue;

          _pairAxis.divideScalar(dist);
          _pairAxisNeg.copy(_pairAxis).negate();

          const supA = hullSupportAlong(
            A.hullPoints,
            A.hullCount,
            sizeA,
            A.body.quaternion,
            _pairAxis,
          );
          const supB = hullSupportAlong(
            B.hullPoints,
            B.hullCount,
            sizeB,
            B.body.quaternion,
            _pairAxisNeg,
          );

          const overlap = supA + supB - dist;
          if (overlap <= 0) continue;

          const invMA = 1 / Math.max(A.body.mass, 1e-6);
          const invMB = 1 / Math.max(B.body.mass, 1e-6);
          const totalInvM = invMA + invMB;

          const corr = (overlap * 0.8) / totalInvM;
          A.body.position.addScaledVector(_pairAxis, -corr * invMA);
          B.body.position.addScaledVector(_pairAxis, corr * invMB);

          _relVel.subVectors(B.body.linearVelocity, A.body.linearVelocity);
          const vn = _relVel.dot(_pairAxis);
          if (vn > 0) continue;

          const jImp = (-(1 + PAIR_RESTITUTION) * vn) / totalInvM;

          A.body.linearVelocity.addScaledVector(_pairAxis, -jImp * invMA);
          B.body.linearVelocity.addScaledVector(_pairAxis, jImp * invMB);

          _tangent.copy(_relVel).addScaledVector(_pairAxis, -vn);
          const tLen = _tangent.length();
          if (tLen > 1e-6) {
            _tangent.divideScalar(tLen);
            const vt = _relVel.dot(_tangent);
            const jT = (-vt / totalInvM) * PAIR_FRICTION;
            A.body.linearVelocity.addScaledVector(_tangent, -jT * invMA);
            B.body.linearVelocity.addScaledVector(_tangent, jT * invMB);
          }
        }
      }
    }
  }

  function updateObjects(dt: number): void {
    probeCount = 0;
    active.length = 0;

    const probeResults = simulation.getLatestProbes();
    const g = config.gravity;
    const localRho = Math.max(config.targetDensity, 1e-4);

    for (let i = 0; i < slotCount; i++) {
      visuals[i].visible = false;

      const slot = config.objects[i];

      if (lastModelId[i] !== slot.modelId) {
        lastModelId[i] = slot.modelId;
        pendingTransform[i] = bodies[i] ? captureTransform(bodies[i]!) : null;
        bodies[i] = null;
        hulls[i] = null;
        hullCounts[i] = 0;
      }

      if (slot.modelId === "none") {
        bodies[i] = null;
        pendingTransform[i] = null;
        continue;
      }

      let body = bodies[i];

      if (!body) {
        const model = meshRegistry.getLoaded(slot.modelId);
        if (!model) {
          meshRegistry.ensure(slot.modelId).catch(() => {});
          continue;
        }

        spawnBody(i, pendingTransform[i]);
        pendingTransform[i] = null;
        body = bodies[i];

        if (!body) continue;
      }

      if (lastSize[i] !== slot.size) {
        lastSize[i] = slot.size;
        const model = meshRegistry.getLoaded(slot.modelId);
        if (model) applyModelToSlot(i, model, slot, localRho, body);
      }

      body.drag = slot.drag;
      body.angularDrag = slot.angularDrag;

      if (body.densityRatio !== slot.densityRatio) {
        body.densityRatio = slot.densityRatio;
        const model = meshRegistry.getLoaded(slot.modelId);
        if (model) applyModelToSlot(i, model, slot, localRho, body);
      }

      const probeBase = probeCount;
      const placedProbes = placeProbesForBody(
        body,
        slot,
        hulls[i]!,
        hullCounts[i],
      );

      applyFluidForces(
        body,
        slot,
        localRho,
        g,
        dt,
        probeResults,
        probeBase,
        placedProbes,
      );

      body.integrate(dt);

      clampToBounds(body, slot, hulls[i]!, hullCounts[i]);

      active.push({
        index: i,
        slot,
        body,
        hullPoints: hulls[i]!,
        hullCount: hullCounts[i],
      });
    }

    resolveBodyPairs();

    for (let k = 0; k < active.length; k++) {
      resolveBodyTerrain(active[k]);
    }

    const descriptors: ObjectDescriptor[] = [];

    for (let k = 0; k < active.length; k++) {
      const entry = active[k];
      writeVisual(entry.index, entry.slot, entry.body);

      const desc = buildDescriptor(entry.slot, entry.body);
      if (desc) descriptors.push(desc);
      else requestBakeForSlot(entry.index);
    }

    simulation.setProbes(probePositions, probeCount);
    simulation.setObjects(descriptors);
    sceneRenderer.setObjectVisuals(visuals);
  }

  return {
    updateObjects,
    requestBakeForSlot,
    setTerrain,
    spawnObject: (index: number) => spawnBody(index, null),
  };
}
