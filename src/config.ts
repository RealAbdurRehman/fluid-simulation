import GUI from "lil-gui";

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

export type ObjectShapeType = "none" | "sphere" | "box" | "torusKnot";

export interface ObjectSlotConfig {
  type: ObjectShapeType;
  posX: number;
  posY: number;
  posZ: number;
  rotX: number;
  rotY: number;
  rotZ: number;
  size: number;
  autoSpin: boolean;
  spinSpeed: number;

  physics: boolean;
  densityRatio: number;
  drag: number;
  angularDrag: number;
}

function makeObjectSlot(
  overrides: Partial<ObjectSlotConfig> = {},
): ObjectSlotConfig {
  return {
    type: "none",
    posX: 0,
    posY: 0,
    posZ: 0,
    rotX: 0,
    rotY: 0,
    rotZ: 0,
    size: 1.5,
    autoSpin: false,
    spinSpeed: 30,
    physics: false,
    densityRatio: 0.5,
    drag: 8.0,
    angularDrag: 4.0,
    ...overrides,
  };
}

export const config = {
  preset: "Standard Water",

  paused: false,
  substeps: 1,
  gravity: 16.0,
  collisionDamping: 0,

  targetDensity: 6.0,
  pressureMultiplier: 65.0,
  nearDensityMultiplier: 30.0,
  smoothingRadius: 0.95,
  viscosityStrength: 0.4,

  foamEnabled: true,
  foamGenerationRate: 12.0,
  foamDecayRate: 1.4,
  foamMinSpeed: 1.5,
  foamMaxSpeed: 9.0,
  foamNoiseScale: 7.5,
  foamThreshold: 0.25,
  foamSoftness: 0.35,

  numParticles: 22768,
  maxParticles: 42768,
  particleSize: 0.4,
  particleSpacing: 0.05,
  xsphStrength: 0.15,

  boundsWidth: 24,
  boundsHeight: 30,
  boundsDepth: 24,

  boundsRotationX: 0,
  boundsRotationY: 0,
  boundsRotationZ: 0,
  boundsAutoTumble: false,

  renderSplatScale: 1.35,

  objects: [
    makeObjectSlot({
      type: "sphere",
      posX: -3.5,
      posY: 0.5,
      posZ: 0.5,
      size: 1.2,
      physics: true,
      densityRatio: 2.5,
      drag: 3.0,
      angularDrag: 2.0,
    }),
    makeObjectSlot({
      type: "box",
      posX: 3.0,
      posY: 2.5,
      posZ: -1.0,
      size: 0.9,
      physics: true,
      densityRatio: 0.25,
      drag: 6.0,
      angularDrag: 3.0,
    }),
  ] as ObjectSlotConfig[],

  interactionRadius: 5.0,
  interactionStrength: 80.0,

  minSpeed: 0.0,
  maxSpeed: 10.0,

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
  addContainerFolder(gui, onUpdateBounds);
  addSPHFolder(gui);
  addFoamFolder(gui);
  addParticleFolder(gui, onResetParticles, onUpdateParticleSize);
  addObjectFolders(gui, onObjectChanged, onSpawnObject);
  addTerrainFolder(gui, onRegenerateTerrain);
  addVisualsFolder(gui);

  return gui;
}

function addTerrainFolder(gui: GUI, onRegen: () => void): void {
  const folder = gui.addFolder("Terrain");
  folder.add(config, "terrainEnabled").name("Enabled").onChange(onRegen);
  folder
    .add(config, "terrainResolution", 32, 512, 32)
    .name("Resolution")
    .onChange(onRegen);
  folder
    .add(config, "terrainHeightScale", 0.5, 30, 0.1)
    .name("Height Scale")
    .onChange(onRegen);
  folder
    .add(
      {
        regen: () => {
          config.terrainSeed = Math.floor(Math.random() * 100000);
          onRegen();
        },
      },
      "regen",
    )
    .name("Regenerate");
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
  const folder = gui.addFolder("Simulation Engine");
  folder.add(config, "paused").name("Paused").listen();
  folder.add(config, "substeps", 1, 5, 1).name("Substeps");
  folder.add(config, "gravity", -40, 40, 0.5).name("Gravity");
  folder.add(config, "collisionDamping", 0, 1, 0.05).name("Wall Restitution");
}

function addContainerFolder(gui: GUI, onUpdateBounds: () => void): void {
  const folder = gui.addFolder("Container");
  folder.add(config, "boundsAutoTumble").name("Auto rotate");
  folder.add(config, "boundsRotationX", -180, 180, 1).name("Tilt X");
  folder.add(config, "boundsRotationY", -180, 180, 1).name("Tilt Y");
  folder.add(config, "boundsRotationZ", -180, 180, 1).name("Tilt Z");
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

function addSPHFolder(gui: GUI): void {
  const folder = gui.addFolder("SPH Parameters");
  folder.add(config, "targetDensity", 1, 30, 0.5).name("Target Density");
  folder
    .add(config, "pressureMultiplier", 10, 300, 5)
    .name("Pressure Multiplier");
  folder.add(config, "nearDensityMultiplier", 0, 200, 5).name("Near Pressure");
  folder
    .add(config, "smoothingRadius", 0.3, 2.0, 0.02)
    .name("Smoothing Radius");
  folder.add(config, "viscosityStrength", 0.0, 30.0, 0.1).name("Viscosity");
}

function addFoamFolder(gui: GUI): void {
  const folder = gui.addFolder("Foam");
  folder.add(config, "foamEnabled").name("Enabled");
  folder.add(config, "foamGenerationRate", 0, 20, 0.5).name("Generation");
  folder.add(config, "foamDecayRate", 0.1, 5, 0.1).name("Decay Rate");
  folder.add(config, "foamMinSpeed", 0, 10, 0.5).name("Min Speed Gate");
  folder.add(config, "foamMaxSpeed", 1, 20, 0.5).name("Max Speed Gate");
  folder.add(config, "foamNoiseScale", 1, 15, 0.5).name("Noise Scale");
  folder.add(config, "foamThreshold", 0, 1, 0.02).name("Threshold");
  folder.add(config, "foamSoftness", 0.02, 0.8, 0.02).name("Softness");
}

function addParticleFolder(
  gui: GUI,
  onResetParticles: () => void,
  onUpdateParticleSize: (size: number) => void,
): void {
  const folder = gui.addFolder("Particle Configuration");
  folder
    .add(config, "numParticles", 256, config.maxParticles, 256)
    .name("Particle Count")
    .onChange(onResetParticles);
  folder
    .add(config, "particleSize", 0.05, 0.6, 0.01)
    .name("Visual Radius")
    .onChange(onUpdateParticleSize);
  folder
    .add(config, "particleSpacing", 0.0, 0.2, 0.005)
    .name("Initial Spacing")
    .onChange(onResetParticles);
  folder.add(config, "interactionRadius", 1, 15, 0.5).name("Mouse Radius");
  folder.add(config, "interactionStrength", 5, 300, 5).name("Mouse Strength");
}

function addObjectFolders(
  gui: GUI,
  onObjectChanged: (index: number) => void,
  onSpawnObject: (index: number) => void,
): void {
  config.objects.forEach((slot, index) => {
    const folder = gui.addFolder(`Object ${index + 1}`);
    folder
      .add(slot, "type", ["none", "sphere", "box", "torusKnot"])
      .name("Shape")
      .onChange(() => onObjectChanged(index));
    folder.add(slot, "size", 0.5, 4, 0.1).name("Size");
    folder.add(slot, "posX", -24, 24, 0.1).name("Position X");
    folder.add(slot, "posY", -24, 24, 0.1).name("Position Y");
    folder.add(slot, "posZ", -24, 24, 0.1).name("Position Z");
    folder.add(slot, "rotX", -180, 180, 1).name("Rotation X");
    folder.add(slot, "rotY", -180, 180, 1).name("Rotation Y");
    folder.add(slot, "rotZ", -180, 180, 1).name("Rotation Z");
    folder.add(slot, "autoSpin").name("Rotate");
    folder.add(slot, "spinSpeed", 0, 180, 1).name("Rotation Speed");

    folder
      .add(slot, "physics")
      .name("Physics")
      .onChange(() => onObjectChanged(index));
    folder
      .add(slot, "densityRatio", 0.05, 4.0, 0.05)
      .name("Density (rel. fluid)");
    folder.add(slot, "drag", 0, 20, 0.1).name("Drag");
    folder.add(slot, "angularDrag", 0, 20, 0.1).name("Angular Drag");
    folder.add({ spawn: () => onSpawnObject(index) }, "spawn").name("Spawn");
  });
}

function addVisualsFolder(gui: GUI): void {
  const folder = gui.addFolder("Visuals");
  folder.add(config, "minSpeed", 0, 10, 0.5).name("Color Min Speed");
  folder.add(config, "maxSpeed", 2, 30, 0.5).name("Color Max Speed");
}
