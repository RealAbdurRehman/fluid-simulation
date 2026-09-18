import * as THREE from "three";

import GUI from "lil-gui";
import { MODEL_IDS } from "./models";

export interface SimulationPreset {
  name: string;
  gravity: number;
  collisionDamping: number;
  targetDensity: number;
  pressureMultiplier: number;
  nearDensityMultiplier: number;
  smoothingRadius: number;
  viscosityStrength: number;
  particleSpacing: number;
  substeps: number;
}

export const PRESETS: Record<string, SimulationPreset> = {
  "Standard Water": {
    name: "Water",
    gravity: 16.0,
    collisionDamping: 0,
    targetDensity: 6.0,
    pressureMultiplier: 65.0,
    nearDensityMultiplier: 30.0,
    smoothingRadius: 0.95,
    viscosityStrength: 0.4,
    particleSpacing: 0.05,
    substeps: 1,
  },
  "Slimey Liquid": {
    name: "Slimey Liquid",
    gravity: 9.81,
    collisionDamping: 0,
    targetDensity: 8.0,
    pressureMultiplier: 80.0,
    nearDensityMultiplier: 40.0,
    smoothingRadius: 1.0,
    viscosityStrength: 18.0,
    particleSpacing: 0.04,
    substeps: 1,
  },
  "Zero-G Fluid Bubble": {
    name: "Zero-G Fluid Bubble",
    gravity: 0.0,
    collisionDamping: 0,
    targetDensity: 5.5,
    pressureMultiplier: 45.0,
    nearDensityMultiplier: 50.0,
    smoothingRadius: 1.1,
    viscosityStrength: 2.0,
    particleSpacing: 0.04,
    substeps: 1,
  },
};

export interface ObjectSlotConfig {
  modelId: string;
  posX: number;
  posY: number;
  posZ: number;
  rotX: number;
  rotY: number;
  rotZ: number;
  size: number;
  densityRatio: number;
  drag: number;
  angularDrag: number;
}

function makeObjectSlot(
  overrides: Partial<ObjectSlotConfig> = {},
): ObjectSlotConfig {
  return {
    modelId: "none",
    posX: 0,
    posY: 0,
    posZ: 0,
    rotX: 0,
    rotY: 0,
    rotZ: 0,
    size: 1.5,
    densityRatio: 0.5,
    drag: 8.0,
    angularDrag: 4.0,
    ...overrides,
  };
}

export const config = {
  preset: "Standard Water",
  paused: false,
  gravity: 16.0,

  substeps: 1,
  collisionDamping: 0,
  targetDensity: 6.0,
  pressureMultiplier: 65.0,
  nearDensityMultiplier: 30.0,
  smoothingRadius: 0.95,
  viscosityStrength: 0.4,
  particleSpacing: 0.05,
  xsphStrength: 0.15,

  numParticles: 22768,
  maxParticles: 42768,
  particleSize: 0.4,
  renderSplatScale: 1.35,
  waterColor: new THREE.Color(0.07, 0.16, 0.2),
  foamEnabled: true,
  foamIntensity: 1.0,
  foamBaseGeneration: 12.0,
  get foamGenerationRate(): number {
    return this.foamEnabled
      ? this.foamBaseGeneration * this.foamIntensity
      : 0.0;
  },
  foamDecayRate: 1.4,
  foamMinSpeed: 1.5,
  foamMaxSpeed: 9.0,
  foamNoiseScale: 7.5,
  foamThreshold: 0.25,
  foamSoftness: 0.35,

  causticsEnabled: true,
  causticsIntensity: 1.6,
  causticsScale: 0.3,
  causticsSpeed: 1.0,

  waterAbsorb: new THREE.Color(0.22, 0.12, 0.07),
  waterAbsorbStrength: 0.1,
  waterDeepTint: 0.22,
  refractionStrength: 1.4,
  refractionTravelMax: 30.0,
  iorWater: 1.33,
  fresnelF0: 0.02,
  specularPower: 200.0,
  specularIntensity: 0.6,
  specColor: new THREE.Color(1.0, 0.98, 0.92),
  foamColor: new THREE.Color(0.93, 0.96, 0.99),
  reflSky: new THREE.Color(0.42, 0.55, 0.68),
  reflHorizon: new THREE.Color(0.14, 0.24, 0.34),
  lightAbsorb: new THREE.Color(0.85, 0.45, 0.28),

  depthMixCoeff: 0.22,
  thicknessGateStart: 2.0,
  thicknessGateEnd: 6.0,
  alphaThicknessCoeff: 1.5,
  normalNoiseScale: 2.5,
  normalNoiseAmp: 0.6,
  flatnessStart: 0.3,
  flatnessEnd: 0.7,

  bilateralSigmaWorld: 2.0,
  bilateralSigmaDepth: 10.0,
  bilateralMaxRadiusPx: 10,

  boundsWidth: 24,
  boundsHeight: 30,
  boundsDepth: 24,
  boundsRotationX: 0,
  boundsRotationY: 0,
  boundsRotationZ: 0,
  boundsAutoTumble: false,

  interactionRadius: 5.0,
  interactionStrength: 80.0,

  objects: [
    makeObjectSlot({
      modelId: "sphere",
      posX: -3.5,
      posY: 0.5,
      posZ: 0.5,
      size: 1.2,
      densityRatio: 2.5,
      drag: 3.0,
      angularDrag: 2.0,
    }),
    makeObjectSlot({
      modelId: "box",
      posX: 3.0,
      posY: 2.5,
      posZ: -1.0,
      size: 0.9,
      densityRatio: 0.25,
      drag: 6.0,
      angularDrag: 3.0,
    }),
  ] as ObjectSlotConfig[],

  terrainEnabled: true,
  terrainResolution: 128,
  terrainHeightScale: 2.5,
  terrainSeed: 1,
};

export function setupGUI(
  onResetParticles: () => void,
  onUpdateBounds: () => void,
  onUpdateParticleSize: (size: number) => void,
  onObjectChanged: (index: number) => void,
  onSpawnObject: (index: number) => void,
  onRegenerateTerrain: () => void,
): GUI {
  const gui = new GUI({ title: "Fluid Simulation" });

  addPresetFolder(gui, onResetParticles);
  addSimulationFolder(gui);
  addFluidFolder(gui, onResetParticles, onUpdateParticleSize);
  addPhysicsFolder(gui);
  addInteractionFolder(gui);
  addSceneFolder(gui, onUpdateBounds, onRegenerateTerrain);
  addObjectFolders(gui, onObjectChanged, onSpawnObject);

  return gui;
}

function addPresetFolder(gui: GUI, onResetParticles: () => void): void {
  const folder = gui.addFolder("Presets");
  folder
    .add(config, "preset", Object.keys(PRESETS))
    .name("Load Preset")
    .onChange((key: string) => {
      const p = PRESETS[key];
      if (!p) return;
      Object.assign(config, {
        gravity: p.gravity,
        collisionDamping: p.collisionDamping,
        targetDensity: p.targetDensity,
        pressureMultiplier: p.pressureMultiplier,
        nearDensityMultiplier: p.nearDensityMultiplier,
        smoothingRadius: p.smoothingRadius,
        viscosityStrength: p.viscosityStrength,
        particleSpacing: p.particleSpacing,
        substeps: p.substeps,
      });
      gui.controllersRecursive().forEach((c) => c.updateDisplay());
      onResetParticles();
    });
}

function addSimulationFolder(gui: GUI): void {
  const folder = gui.addFolder("Simulation");
  folder.add(config, "paused").name("Paused").listen();
  folder.add(config, "gravity", -40, 40, 0.5).name("Gravity");
}

function addFluidFolder(
  gui: GUI,
  onResetParticles: () => void,
  onUpdateParticleSize: (size: number) => void,
): void {
  const folder = gui.addFolder("Fluid");
  folder
    .add(config, "numParticles", 256, config.maxParticles, 256)
    .name("Particle Count")
    .onChange(onResetParticles);
  folder
    .add(config, "particleSize", 0.05, 0.6, 0.01)
    .name("Visual Radius")
    .onChange(onUpdateParticleSize);
  folder.add(config, "renderSplatScale", 0.5, 3.0, 0.05).name("Splat Scale");
  folder.addColor(config, "waterColor").name("Water Color");
  folder.add(config, "foamEnabled").name("Foam");
  folder.add(config, "foamIntensity", 0, 3, 0.05).name("Foam Intensity");
  folder.add(config, "causticsEnabled").name("Caustics");
  folder
    .add(config, "causticsIntensity", 0, 4, 0.05)
    .name("Caustics Intensity");
}

function addPhysicsFolder(gui: GUI): void {
  const folder = gui.addFolder("Physics");
  folder.add(config, "viscosityStrength", 0.0, 30.0, 0.1).name("Viscosity");
}

function addInteractionFolder(gui: GUI): void {
  const folder = gui.addFolder("Interaction");
  folder.add(config, "interactionRadius", 1, 15, 0.5).name("Mouse Radius");
  folder.add(config, "interactionStrength", 5, 300, 5).name("Mouse Strength");
}

function addSceneFolder(
  gui: GUI,
  onUpdateBounds: () => void,
  onRegenerateTerrain: () => void,
): void {
  const folder = gui.addFolder("Scene");
  folder
    .add(config, "terrainEnabled")
    .name("Terrain")
    .onChange(onRegenerateTerrain);

  const terrain = folder.addFolder("Terrain Detail");
  terrain
    .add(config, "terrainResolution", 32, 512, 32)
    .name("Resolution")
    .onChange(onRegenerateTerrain);
  terrain
    .add(config, "terrainHeightScale", 0.5, 30, 0.1)
    .name("Height Scale")
    .onChange(onRegenerateTerrain);
  terrain
    .add(config, "terrainSeed", 0, 100000, 1)
    .name("Seed")
    .onChange(onRegenerateTerrain);
  terrain
    .add(
      {
        regen: () => {
          config.terrainSeed = Math.floor(Math.random() * 100000);
          gui.controllersRecursive().forEach((c) => c.updateDisplay());
          onRegenerateTerrain();
        },
      },
      "regen",
    )
    .name("Randomize");

  folder
    .add(config, "boundsWidth", 4, 40, 1)
    .name("Width")
    .onChange(onUpdateBounds);
  folder
    .add(config, "boundsHeight", 4, 40, 1)
    .name("Height")
    .onChange(onUpdateBounds);
  folder
    .add(config, "boundsDepth", 4, 40, 1)
    .name("Depth")
    .onChange(onUpdateBounds);
}

function addObjectFolders(
  gui: GUI,
  onObjectChanged: (index: number) => void,
  onSpawnObject: (index: number) => void,
): void {
  const modelOptions = ["none", ...MODEL_IDS];
  const maxSize =
    Math.min(config.boundsWidth, config.boundsHeight, config.boundsDepth) * 0.2;
  const minSize = Math.max(config.particleSize * 2.0, 0.5);

  config.objects.forEach((slot, index) => {
    const folder = gui.addFolder(`Object ${index + 1}`);
    folder
      .add(slot, "modelId", modelOptions)
      .name("Model")
      .onChange(() => onObjectChanged(index));
    folder.add(slot, "size", minSize, maxSize, 0.05).name("Size");
    folder
      .add(slot, "densityRatio", 0.05, 4.0, 0.05)
      .name("Density (rel. fluid)");
    folder.add({ spawn: () => onSpawnObject(index) }, "spawn").name("Spawn");
  });
}
