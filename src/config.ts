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
    gravity: 14.0,
    collisionDamping: 0.4,
    targetDensity: 6.0,
    pressureMultiplier: 60.0,
    nearDensityMultiplier: 25.0,
    smoothingRadius: 0.95,
    viscosityStrength: 1.2,
    particleSpacing: 0.04,
    substeps: 1,
  },
  "Viscous Slime": {
    name: "Viscous Slime",
    gravity: 9.81,
    collisionDamping: 0.1,
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
    collisionDamping: 0.8,
    targetDensity: 5.5,
    pressureMultiplier: 45.0,
    nearDensityMultiplier: 50.0,
    smoothingRadius: 1.1,
    viscosityStrength: 2.0,
    particleSpacing: 0.04,
    substeps: 1,
  },
};

export const config = {
  preset: "Water",

  paused: false,
  substeps: 1,
  gravity: 14.0,
  collisionDamping: 0.4,

  targetDensity: 6.0,
  pressureMultiplier: 60.0,
  nearDensityMultiplier: 25.0,
  smoothingRadius: 0.95,
  viscosityStrength: 1.2,

  numParticles: 4096,
  maxParticles: 42768,
  particleSize: 0.22,
  particleSpacing: 0.04,

  boundsWidth: 14,
  boundsHeight: 14,
  boundsDepth: 14,

  interactionRadius: 5.0,
  interactionStrength: 80.0,

  minSpeed: 0.0,
  maxSpeed: 12.0,
};

export function setupGUI(
  onResetParticles: () => void,
  onUpdateBounds: () => void,
  onUpdateParticleSize: (size: number) => void,
): GUI {
  const gui = new GUI({ title: "3D Fluid Simulation" });

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
  simFolder.add(config, "paused").name("Paused").listen();
  simFolder.add(config, "substeps", 1, 5, 1).name("Substeps");
  simFolder.add(config, "gravity", -40, 40, 0.5).name("Gravity");
  simFolder
    .add(config, "collisionDamping", 0, 1, 0.05)
    .name("Wall Restitution");
  simFolder
    .add(config, "boundsWidth", 4, 40, 1)
    .name("Bounds Width")
    .onChange(() => onUpdateBounds());
  simFolder
    .add(config, "boundsHeight", 4, 40, 1)
    .name("Bounds Height")
    .onChange(() => onUpdateBounds());
  simFolder
    .add(config, "boundsDepth", 4, 40, 1)
    .name("Bounds Depth")
    .onChange(() => onUpdateBounds());

  const sphFolder = gui.addFolder("SPH Parameters");
  sphFolder.add(config, "targetDensity", 1, 30, 0.5).name("Target Density");
  sphFolder
    .add(config, "pressureMultiplier", 10, 300, 5)
    .name("Pressure Multiplier");
  sphFolder
    .add(config, "nearDensityMultiplier", 0, 200, 5)
    .name("Near Pressure");
  sphFolder
    .add(config, "smoothingRadius", 0.3, 2.0, 0.02)
    .name("Smoothing Radius");
  sphFolder.add(config, "viscosityStrength", 0.0, 30.0, 0.1).name("Viscosity");

  const particleFolder = gui.addFolder("Particle Configuration");
  particleFolder
    .add(config, "numParticles", 256, config.maxParticles, 256)
    .name("Particle Count")
    .onChange(() => onResetParticles());
  particleFolder
    .add(config, "particleSize", 0.05, 0.6, 0.01)
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
    .add(config, "interactionStrength", 5, 300, 5)
    .name("Mouse Strength");

  const visualsFolder = gui.addFolder("Visuals");
  visualsFolder.add(config, "minSpeed", 0, 10, 0.5).name("Color Min Speed");
  visualsFolder.add(config, "maxSpeed", 2, 30, 0.5).name("Color Max Speed");

  return gui;
}
