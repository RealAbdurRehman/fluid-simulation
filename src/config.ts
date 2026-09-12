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
    gravity: 12.0,
    collisionDamping: 0.5,
    targetDensity: 24.0,
    pressureMultiplier: 120.0,
    nearDensityMultiplier: 60.0,
    smoothingRadius: 0.65,
    viscosityStrength: 0.8,
    particleSpacing: 0.02,
    substeps: 1,
  },
  "Slimey Liquid": {
    name: "Slimey Liquid",
    gravity: 9.81,
    collisionDamping: 0.1,
    targetDensity: 30.0,
    pressureMultiplier: 180.0,
    nearDensityMultiplier: 90.0,
    smoothingRadius: 0.75,
    viscosityStrength: 18.0,
    particleSpacing: 0.01,
    substeps: 1,
  },
  "Zero-G Fluid Bubble": {
    name: "Zero-G Fluid Bubble",
    gravity: 0.0,
    collisionDamping: 0.8,
    targetDensity: 20.0,
    pressureMultiplier: 80.0,
    nearDensityMultiplier: 140.0,
    smoothingRadius: 0.8,
    viscosityStrength: 1.5,
    particleSpacing: 0.02,
    substeps: 1,
  },
  "High Pressure": {
    name: "High Pressure",
    gravity: 15.0,
    collisionDamping: 0.7,
    targetDensity: 18.0,
    pressureMultiplier: 350.0,
    nearDensityMultiplier: 120.0,
    smoothingRadius: 0.6,
    viscosityStrength: 0.2,
    particleSpacing: 0.04,
    substeps: 1,
  },
};

export const config = {
  preset: "Water",

  paused: false,
  substeps: 1,
  gravity: 12.0,
  collisionDamping: 0.5,

  targetDensity: 24.0,
  pressureMultiplier: 120.0,
  nearDensityMultiplier: 60.0,
  smoothingRadius: 0.65,
  viscosityStrength: 0.8,

  numParticles: 4096,
  maxParticles: 42768,
  particleSize: 0.08,
  particleSpacing: 0.02,

  boundsWidth: 28,
  boundsHeight: 14,
  cameraDistance: 16,

  interactionRadius: 4.5,
  interactionStrength: 60.0,

  minSpeed: 0.0,
  maxSpeed: 10.0,
};

export function setupGUI(
  onResetParticles: () => void,
  onUpdateBounds: () => void,
  onUpdateCamera: (dist: number) => void,
  onUpdateParticleSize: (size: number) => void,
): GUI {
  const gui = new GUI({ title: "Fluid Simulation" });

  const presetFolder = gui.addFolder("Presets");
  presetFolder
    .add(config, "preset", Object.keys(PRESETS))
    .name("Load Preset")
    .onChange((presetKey: string) => {
      const p = PRESETS[presetKey];
      if (!p) return;
      config.gravity = p.gravity;
      config.collisionDamping = p.collisionDamping;
      config.targetDensity = p.targetDensity;
      config.pressureMultiplier = p.pressureMultiplier;
      config.nearDensityMultiplier = p.nearDensityMultiplier;
      config.smoothingRadius = p.smoothingRadius;
      config.viscosityStrength = p.viscosityStrength;
      config.particleSpacing = p.particleSpacing;
      config.substeps = p.substeps;
      gui.controllersRecursive().forEach((c) => c.updateDisplay());
      onResetParticles();
    });

  const simFolder = gui.addFolder("Simulation Engine");
  simFolder.add(config, "paused").name("Paused [Space]");
  simFolder.add(config, "substeps", 1, 3, 1).name("Substeps");
  simFolder.add(config, "gravity", -40, 40, 0.5).name("Gravity");
  simFolder
    .add(config, "collisionDamping", 0, 1, 0.05)
    .name("Wall Restitution");
  simFolder
    .add(config, "boundsWidth", 8, 100, 1)
    .name("Bounds Width")
    .onChange(() => onUpdateBounds());
  simFolder
    .add(config, "boundsHeight", 6, 100, 1)
    .name("Bounds Height")
    .onChange(() => onUpdateBounds());

  const sphFolder = gui.addFolder("SPH Parameters");
  sphFolder.add(config, "targetDensity", 2, 60, 0.5).name("Target Density");
  sphFolder
    .add(config, "pressureMultiplier", 10, 600, 5)
    .name("Pressure Multiplier");
  sphFolder
    .add(config, "nearDensityMultiplier", 0, 300, 5)
    .name("Near Pressure Multiplier");
  sphFolder
    .add(config, "smoothingRadius", 0.2, 2.0, 0.02)
    .name("Smoothing Radius");
  sphFolder.add(config, "viscosityStrength", 0.0, 30.0, 0.1).name("Viscosity");

  const particleFolder = gui.addFolder("Particle Configuration");
  particleFolder
    .add(config, "numParticles", 256, config.maxParticles, 256)
    .name("Particle Count")
    .onChange(() => onResetParticles());
  particleFolder
    .add(config, "particleSize", 0.02, 0.3, 0.01)
    .name("Visual Radius")
    .onChange((s: number) => onUpdateParticleSize(s));
  particleFolder
    .add(config, "particleSpacing", 0.0, 0.2, 0.005)
    .name("Initial Spacing")
    .onChange(() => onResetParticles());
  particleFolder
    .add(config, "interactionRadius", 1, 15, 0.5)
    .name("Mouse Radius");
  particleFolder
    .add(config, "interactionStrength", 5, 200, 5)
    .name("Mouse Strength");

  const visualsFolder = gui.addFolder("Visuals & Camera");
  visualsFolder.add(config, "minSpeed", 0, 10, 0.5).name("Color Min Speed");
  visualsFolder.add(config, "maxSpeed", 2, 30, 0.5).name("Color Max Speed");
  visualsFolder
    .add(config, "cameraDistance", 5, 100, 1)
    .name("Camera Zoom")
    .onChange((d: number) => onUpdateCamera(d));

  return gui;
}
