import GUI from "lil-gui";

import { setCameraDistance, setBoundsSize } from "./scene";
import { setParticleSize, setParticleGridPosition, start } from "./simulation";
import { setSpeedRange, MAX_PARTICLES } from "./particles";

const gui = new GUI();

const config = {
  paused: true,

  gravity: 9.81,
  collisionDamping: 0.3,

  targetDensity: 16,
  nearDensityMultiplier: 70,
  pressureMultiplier: 40,
  smoothingRadius: 0.8,
  viscosityStrength: 3,
  interactionRadius: 4,
  interactionStrength: 50,

  numParticles: 1000,
  maxParticles: MAX_PARTICLES,
  particleSize: 0.1,
  particleSpacing: 0,

  boundsWidth: 28,
  boundsHeight: 14,
  cameraDistance: 15,

  minSpeed: 0,
  maxSpeed: 6,
};

const simulationFolder = gui.addFolder("Simulation");
simulationFolder.add(config, "paused");
simulationFolder.add(config, "gravity", -20, 20);
simulationFolder.add(config, "collisionDamping", 0, 1);
simulationFolder
  .add(config, "boundsWidth", 5, 30)
  .onChange(() => setBoundsSize(config.boundsWidth, config.boundsHeight));
simulationFolder
  .add(config, "boundsHeight", 5, 30)
  .onChange(() => setBoundsSize(config.boundsWidth, config.boundsHeight));
simulationFolder.add(config, "targetDensity", 8, 32);
simulationFolder.add(config, "nearDensityMultiplier", 35, 150);
simulationFolder.add(config, "pressureMultiplier", 20, 85);

const particleFolder = gui.addFolder("Particles");
particleFolder
  .add(config, "numParticles", 1, config.maxParticles, 1)
  .onChange(() => start());
particleFolder
  .add(config, "particleSpacing", 0, 0.5)
  .onChange(() => setParticleGridPosition());
particleFolder
  .add(config, "particleSize", 0.02, 0.3)
  .onChange((size: number) => {
    setParticleSize(size);
    setParticleGridPosition();
  });
particleFolder.add(config, "smoothingRadius", 0.4, 1.6);
particleFolder.add(config, "viscosityStrength", 1, 6);
particleFolder.add(config, "interactionRadius", 0.1, 10);
particleFolder.add(config, "interactionStrength", 0.1, 100);

const visualsFolder = gui.addFolder("Visuals");
visualsFolder
  .add(config, "minSpeed", 0, 10)
  .onChange(() => setSpeedRange(config.minSpeed, config.maxSpeed));
visualsFolder
  .add(config, "maxSpeed", 0.1, 20)
  .onChange(() => setSpeedRange(config.minSpeed, config.maxSpeed));

const cameraFolder = gui.addFolder("Camera");
cameraFolder
  .add(config, "cameraDistance", 5, 50)
  .onChange((distance: number) => setCameraDistance(distance));

setCameraDistance(config.cameraDistance);
setBoundsSize(config.boundsWidth, config.boundsHeight);

export { config };
