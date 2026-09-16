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
  Water: {
    name: "Water",
    gravity: 14.0,
    collisionDamping: 0,
    targetDensity: 6.0,
    pressureMultiplier: 5.0,
    nearDensityMultiplier: 20.0,
    smoothingRadius: 0.35,
    viscosityStrength: 0.18,
    particleSpacing: 0.01,
    substeps: 1,
  },
  "Slimey Liquid": {
    name: "Slimey Liquid",
    gravity: 9.81,
    collisionDamping: 0,
    targetDensity: 7.0,
    pressureMultiplier: 10.0,
    nearDensityMultiplier: 25.0,
    smoothingRadius: 0.38,
    viscosityStrength: 3.5,
    particleSpacing: 0.01,
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
  preset: "Water",

  paused: false,
  substeps: 1,
  gravity: 14.0,
  collisionDamping: 0,

  targetDensity: 6.0,
  pressureMultiplier: 5.0,
  nearDensityMultiplier: 20.0,
  smoothingRadius: 0.35,
  viscosityStrength: 0.18,

  numParticles: 8192,
  maxParticles: 42768,
  particleSize: 0.1,
  particleSpacing: 0.01,
  xsphStrength: 0.15,

  planetRadius: 20.0,
  planetHeightScale: 2.5,
  planetNoiseScale: 2.5,
  planetFaceResolution: 128,
  planetSeed: 1,
  planetCenterX: 0,
  planetCenterY: 0,
  planetCenterZ: 0,

  objects: [makeObjectSlot({ type: "none", posX: 0, posY: 25, size: 1.5 })],

  interactionRadius: 5.0,
  interactionStrength: 10.0,

  minSpeed: 0.0,
  maxSpeed: 12.0,
};

export function setupGUI(
  onResetParticles: () => void,
  onUpdateParticleSize: (size: number) => void,
  onObjectChanged: (index: number) => void,
  onSpawnObject: (index: number) => void,
  onRegeneratePlanet: () => void,
): GUI {
  const gui = new GUI({ title: "Fluid Simulation" });

  addPresetFolder(gui, onResetParticles);
  addSimulationFolder(gui);
  addSPHFolder(gui);
  addParticleFolder(gui, onResetParticles, onUpdateParticleSize);
  addObjectFolders(gui, onObjectChanged, onSpawnObject);
  addPlanetFolder(gui, onRegeneratePlanet);
  addVisualsFolder(gui);

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
  const folder = gui.addFolder("Simulation Engine");
  folder.add(config, "paused").name("Paused").listen();
  folder.add(config, "substeps", 1, 5, 1).name("Substeps");
  folder.add(config, "gravity", 0, 40, 0.5).name("Gravity");
  folder.add(config, "collisionDamping", 0, 1, 0.05).name("Restitution");
}

function addSPHFolder(gui: GUI): void {
  const folder = gui.addFolder("SPH Parameters");
  folder.add(config, "targetDensity", 1, 30, 0.5).name("Target Density");
  folder
    .add(config, "pressureMultiplier", 5, 300, 5)
    .name("Pressure Multiplier");
  folder.add(config, "nearDensityMultiplier", 0, 200, 5).name("Near Pressure");
  folder
    .add(config, "smoothingRadius", 0.1, 2.0, 0.01)
    .name("Smoothing Radius");
  folder.add(config, "viscosityStrength", 0.0, 10.0, 0.05).name("Viscosity");
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
    .add(config, "particleSize", 0.02, 0.4, 0.005)
    .name("Visual Radius")
    .onChange(onUpdateParticleSize);
  folder
    .add(config, "particleSpacing", 0.0, 0.1, 0.005)
    .name("Particle Spacing")
    .onChange(onResetParticles);
  folder.add(config, "interactionRadius", 1, 10, 0.5).name("Mouse Radius");
  folder.add(config, "interactionStrength", 1, 100, 5).name("Mouse Strength");
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
    folder.add(slot, "posX", -50, 50, 0.1).name("Position X");
    folder.add(slot, "posY", -50, 50, 0.1).name("Position Y");
    folder.add(slot, "posZ", -50, 50, 0.1).name("Position Z");
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

function addPlanetFolder(gui: GUI, onRegen: () => void): void {
  const folder = gui.addFolder("Planet");
  folder
    .add(config, "planetRadius", 5, 60, 0.5)
    .name("Radius")
    .onChange(onRegen);
  folder
    .add(config, "planetHeightScale", 0.5, 8, 0.1)
    .name("Relief")
    .onChange(onRegen);
  folder
    .add(config, "planetNoiseScale", 0.5, 8, 0.1)
    .name("Feature Scale")
    .onChange(onRegen);
  folder
    .add(config, "planetFaceResolution", 32, 512, 32)
    .name("Face Res")
    .onChange(onRegen);
  folder
    .add(
      {
        regen: () => {
          config.planetSeed = Math.floor(Math.random() * 100000);
          onRegen();
        },
      },
      "regen",
    )
    .name("Regenerate");
}

function addVisualsFolder(gui: GUI): void {
  const folder = gui.addFolder("Visuals");
  folder.add(config, "minSpeed", 0, 10, 0.5).name("Color Min Speed");
  folder.add(config, "maxSpeed", 2, 30, 0.5).name("Color Max Speed");
}
