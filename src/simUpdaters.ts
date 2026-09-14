import * as THREE from "three";

import { config, type ObjectSlotConfig } from "./config";
import type { FluidSimulationGPU, ObjectDescriptor } from "./simulation";
import type { SceneRenderer, ObjectVisual } from "./sceneRenderer";
import type { MeshRegistry } from "./meshRegistry";

type RGBA = [number, number, number, number];
type ActiveShape = Exclude<ObjectSlotConfig["type"], "none">;

const OBJECT_COLORS: Record<ObjectSlotConfig["type"], RGBA> = {
  none: [1, 1, 1, 0],
  sphere: [0.2, 0.7, 1.0, 0.9],
  box: [1.0, 0.5, 0.2, 0.9],
  torusKnot: [0.5, 1.0, 0.3, 0.9],
};

const VISUAL_SHAPE: Record<ActiveShape, ObjectVisual["shape"]> = {
  sphere: "sphere",
  box: "box",
  torusKnot: "mesh",
};

export interface SimUpdaters {
  updateBoundsRotation(dt: number): void;
  updateObjects(dt: number): void;
  requestBakeForSlot(index: number): void;
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
    visual.shape = VISUAL_SHAPE[shape];
    visual.position = [position.x, position.y, position.z];
    visual.quaternion = [
      quaternions[index].x,
      quaternions[index].y,
      quaternions[index].z,
      quaternions[index].w,
    ];

    visual.scale = slot.size;
    visual.color = OBJECT_COLORS[shape];
    visual.meshId = shape === "torusKnot" ? "torusKnot" : undefined;
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
    const descriptors: ObjectDescriptor[] = [];
    for (let i = 0; i < slotCount; i++) {
      const slot = config.objects[i];
      if (slot.type === "none") {
        visuals[i].visible = false;
        continue;
      }

      const shape: ActiveShape = slot.type;

      const position = updateSlotTransform(i, slot, dt);
      writeVisual(i, shape, slot, position);

      const descriptor = buildDescriptor(i, shape, slot, position);
      if (descriptor) descriptors.push(descriptor);
      else requestBakeForSlot(i);
    }

    simulation.setObjects(descriptors);
    sceneRenderer.setObjectVisuals(visuals);
  }

  return { updateObjects, requestBakeForSlot };
}
