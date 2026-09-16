import * as THREE from "three";

import { config, type ObjectSlotConfig } from "./config";
import type { FluidSimulationGPU, ObjectDescriptor } from "./simulation";
import type { SceneRenderer, ObjectVisual } from "./sceneRenderer";
import type { MeshRegistry } from "./meshRegistry";
import { RigidBody, boxInertia, sphereInertia } from "./rigidBody";

type RGBA = [number, number, number, number];
type ActiveShape = Exclude<ObjectSlotConfig["type"], "none">;

const OBJECT_COLORS: Record<ObjectSlotConfig["type"], RGBA> = {
  none: [1, 1, 1, 0],
  sphere: [0.2, 0.7, 1.0, 0.9],
  box: [1.0, 0.5, 0.2, 0.9],
  torusKnot: [0.5, 1.0, 0.3, 0.9],
};

const PROBE_OFFSETS: [number, number, number][] = (() => {
  const out: [number, number, number][] = [];
  for (let x = -1; x <= 1; x++)
    for (let y = -1; y <= 1; y++)
      for (let z = -1; z <= 1; z++) out.push([x, y, z]);
  return out;
})();

const PROBES_PER_BODY = PROBE_OFFSETS.length;

const ADDED_MASS = 0.5;

const MAX_LINEAR_SPEED = 40.0;
const MAX_ANGULAR_SPEED = 6.0;

const AIR_LINEAR_DAMPING = 0.4;
const AIR_ANGULAR_DAMPING = 1.2;

const _tmpLocalPos = new THREE.Vector3();
const _tmpLocalVel = new THREE.Vector3();
const _tmpInvContainerQuat = new THREE.Quaternion();

export interface SimUpdaters {
  updateBoundsRotation(dt: number): void;
  updateObjects(dt: number): void;
  requestBakeForSlot(index: number): void;
  spawnObject(index: number): void;
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

function bodyVolume(shape: ActiveShape, size: number): number {
  if (shape === "sphere") return (4 / 3) * Math.PI * size * size * size;
  return Math.pow(size * 2, 3);
}

function bodyInertia(
  shape: ActiveShape,
  mass: number,
  size: number,
): THREE.Vector3 {
  if (shape === "sphere") return sphereInertia(mass, size);
  const s = new THREE.Vector3(size * 2, size * 2, size * 2);
  return boxInertia(mass, s);
}

function createObjectsUpdater(
  simulation: FluidSimulationGPU,
  sceneRenderer: SceneRenderer,
  meshRegistry: MeshRegistry,
) {
  const slotCount = config.objects.length;

  const spinAngles = new Array<number>(slotCount).fill(0);
  const prevPositions = Array.from(
    { length: slotCount },
    () => new THREE.Vector3(),
  );
  const velocities = Array.from(
    { length: slotCount },
    () => new THREE.Vector3(),
  );
  const quaternions = Array.from(
    { length: slotCount },
    () => new THREE.Quaternion(),
  );

  const bodies: (RigidBody | null)[] = new Array(slotCount).fill(null);

  const probePositions = new Float32Array(256 * 4);
  let probeCount = 0;

  const visuals: ObjectVisual[] = Array.from({ length: slotCount }, () => ({
    visible: false,
    shape: "box",
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
    scale: 1,
    color: OBJECT_COLORS.none,
  }));

  function requestBakeForSlot(index: number): void {
    const slot = config.objects[index];
    if (slot.type === "none") return;

    meshRegistry.ensure(slot.type);
  }

  function spawnBody(index: number): void {
    const slot = config.objects[index];
    if (slot.type === "none" || !slot.physics) return;

    const shape: ActiveShape = slot.type;
    const volume = bodyVolume(shape, slot.size);
    const refDensity = Math.max(config.targetDensity, 1e-4);
    const mass = Math.max(slot.densityRatio * refDensity * volume, 1e-4);

    const pos = new THREE.Vector3(slot.posX, slot.posY, slot.posZ);
    const quat = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(
        THREE.MathUtils.degToRad(slot.rotX),
        THREE.MathUtils.degToRad(slot.rotY),
        THREE.MathUtils.degToRad(slot.rotZ),
      ),
    );

    bodies[index] = new RigidBody({
      position: pos,
      quaternion: quat,
      mass,
      volume,
      densityRatio: slot.densityRatio,
      inertiaLocal: bodyInertia(shape, mass, slot.size),
      drag: slot.drag,
      angularDrag: slot.angularDrag,
    });

    prevPositions[index].copy(pos);
    velocities[index].set(0, 0, 0);
    quaternions[index].copy(quat);
    spinAngles[index] = 0;
  }

  function updateSlotTransform(
    index: number,
    slot: ObjectSlotConfig,
    dt: number,
  ): THREE.Vector3 {
    if (slot.autoSpin)
      spinAngles[index] += THREE.MathUtils.degToRad(slot.spinSpeed) * dt;

    const euler = new THREE.Euler(
      THREE.MathUtils.degToRad(slot.rotX),
      THREE.MathUtils.degToRad(slot.rotY) +
        (slot.autoSpin ? spinAngles[index] : 0),
      THREE.MathUtils.degToRad(slot.rotZ),
    );
    quaternions[index].setFromEuler(euler);

    const position = new THREE.Vector3(slot.posX, slot.posY, slot.posZ);
    velocities[index]
      .subVectors(position, prevPositions[index])
      .divideScalar(Math.max(dt, 1e-4));
    prevPositions[index].copy(position);
    return position;
  }

  function writeVisual(
    index: number,
    shape: ActiveShape,
    slot: ObjectSlotConfig,
    position: THREE.Vector3,
  ): void {
    const visual = visuals[index];
    visual.visible = true;
    visual.position = [position.x, position.y, position.z];
    visual.quaternion = [
      quaternions[index].x,
      quaternions[index].y,
      quaternions[index].z,
      quaternions[index].w,
    ];

    visual.scale = slot.size;
    visual.color = OBJECT_COLORS[shape];
    visual.meshId = shape;
  }

  function buildDescriptor(
    index: number,
    shape: ActiveShape,
    slot: ObjectSlotConfig,
    position: THREE.Vector3,
  ): ObjectDescriptor | null {
    const base: ObjectDescriptor = {
      type: shape === "torusKnot" ? "mesh" : shape,
      position,
      quaternion: quaternions[index],
      velocity: velocities[index],
      restitution: 0.5,
    };

    switch (shape) {
      case "sphere":
        base.size = new THREE.Vector3(slot.size, 0, 0);
        return base;
      case "box":
        base.size = new THREE.Vector3(slot.size, slot.size, slot.size);
        return base;
      case "torusKnot": {
        const mesh = meshRegistry.get("torusKnot");
        if (!mesh) return null;

        base.mesh = mesh;
        base.size = new THREE.Vector3(slot.size, 0, 0);

        return base;
      }
    }
  }

  function updateObjects(dt: number): void {
    probeCount = 0;
    const descriptors: ObjectDescriptor[] = [];
    const probeResults = simulation.getLatestProbes();

    const g = config.gravity;
    const localRho = Math.max(config.targetDensity, 1e-4);
    for (let i = 0; i < slotCount; i++) {
      const slot = config.objects[i];
      if (slot.type === "none") {
        bodies[i] = null;
        visuals[i].visible = false;
        continue;
      }
      const shape: ActiveShape = slot.type;

      if (!slot.physics) {
        bodies[i] = null;
        const position = updateSlotTransform(i, slot, dt);
        writeVisual(i, shape, slot, position);
        const desc = buildDescriptor(i, shape, slot, position);
        if (desc) descriptors.push(desc);
        else requestBakeForSlot(i);
        continue;
      }

      let body = bodies[i];
      if (!body) {
        spawnBody(i);
        body = bodies[i]!;
      }

      body.drag = slot.drag;
      body.angularDrag = slot.angularDrag;
      if (body.densityRatio !== slot.densityRatio) {
        body.densityRatio = slot.densityRatio;
        body.mass = Math.max(slot.densityRatio * localRho * body.volume, 1e-4);
        body.inertiaLocal.copy(bodyInertia(shape, body.mass, slot.size));
      }

      const probeBase = probeCount;
      const half = slot.size;
      for (const [ox, oy, oz] of PROBE_OFFSETS) {
        const local = new THREE.Vector3(ox * half, oy * half, oz * half)
          .applyQuaternion(body.quaternion)
          .add(body.position);
        const o = probeCount * 4;
        probePositions[o + 0] = local.x;
        probePositions[o + 1] = local.y;
        probePositions[o + 2] = local.z;
        probePositions[o + 3] = 0;
        probeCount++;
      }

      let submergedSum = 0;
      const centerSub = new THREE.Vector3();
      const avgFluidVel = new THREE.Vector3();

      if (probeResults) {
        for (let p = 0; p < PROBES_PER_BODY; p++) {
          const o = (probeBase + p) * 4;
          const density = probeResults[o + 0];
          const sub = THREE.MathUtils.clamp(density / localRho, 0, 1);

          const worldPos = new THREE.Vector3(
            probePositions[o + 0],
            probePositions[o + 1],
            probePositions[o + 2],
          );

          centerSub.addScaledVector(worldPos, sub);
          avgFluidVel.addScaledVector(
            new THREE.Vector3(
              probeResults[o + 1],
              probeResults[o + 2],
              probeResults[o + 3],
            ),
            sub,
          );
          submergedSum += sub;
        }
      }

      const submergedFraction = submergedSum / PROBES_PER_BODY;
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

        const buoyancy = new THREE.Vector3(0, fluidMass * g * fs, 0);
        body.applyForceWorld(buoyancy, centerSub, dt);

        const relVel = new THREE.Vector3().subVectors(
          body.linearVelocity,
          avgFluidVel,
        );

        const dragForce = relVel.multiplyScalar(
          -slot.drag * dragFluidMass * fs,
        );
        body.applyForceWorld(dragForce, body.position, dt);

        const angDrag = body.angularVelocity
          .clone()
          .multiplyScalar(-slot.angularDrag * fluidMass);
        body.applyTorqueWorld(angDrag, dt);
      }

      const linSpeed = body.linearVelocity.length();
      if (linSpeed > MAX_LINEAR_SPEED)
        body.linearVelocity.multiplyScalar(MAX_LINEAR_SPEED / linSpeed);

      const angSpeed = body.angularVelocity.length();
      if (angSpeed > MAX_ANGULAR_SPEED)
        body.angularVelocity.multiplyScalar(MAX_ANGULAR_SPEED / angSpeed);

      body.integrate(dt);

      const containerQuat = simulation.getBoundsQuaternion();
      const invQuat = _tmpInvContainerQuat.copy(containerQuat).invert();

      _tmpLocalPos.copy(body.position).applyQuaternion(invQuat);
      _tmpLocalVel.copy(body.linearVelocity).applyQuaternion(invQuat);

      const hx = config.boundsWidth / 2 - slot.size;
      const hy = config.boundsHeight / 2 - slot.size;
      const hz = config.boundsDepth / 2 - slot.size;

      if (Math.abs(_tmpLocalPos.x) > hx) {
        _tmpLocalPos.x = Math.sign(_tmpLocalPos.x) * hx;
        _tmpLocalVel.x *= -config.collisionDamping;
      }
      if (Math.abs(_tmpLocalPos.y) > hy) {
        _tmpLocalPos.y = Math.sign(_tmpLocalPos.y) * hy;
        _tmpLocalVel.y *= -config.collisionDamping;
      }
      if (Math.abs(_tmpLocalPos.z) > hz) {
        _tmpLocalPos.z = Math.sign(_tmpLocalPos.z) * hz;
        _tmpLocalVel.z *= -config.collisionDamping;
      }

      body.position.copy(_tmpLocalPos).applyQuaternion(containerQuat);
      body.linearVelocity.copy(_tmpLocalVel).applyQuaternion(containerQuat);

      quaternions[i].copy(body.quaternion);
      velocities[i].copy(body.linearVelocity);
      prevPositions[i].copy(body.position);

      writeVisual(i, shape, slot, body.position);

      const desc = buildDescriptor(i, shape, slot, body.position);
      if (desc) {
        desc.quaternion = body.quaternion;
        desc.velocity = body.linearVelocity;
        descriptors.push(desc);
      } else requestBakeForSlot(i);
    }

    simulation.setProbes(probePositions, probeCount);
    simulation.setObjects(descriptors);
    sceneRenderer.setObjectVisuals(visuals);
  }

  return { updateObjects, requestBakeForSlot, spawnObject: spawnBody };
}
