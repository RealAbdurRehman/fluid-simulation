import GUI from "lil-gui";

import { setCameraDistance, setBoundsSize } from "./scene";
import { setParticleSize, setParticleGridPosition, start } from "./simulation";
import { setSpeedRange, MAX_PARTICLES } from "./particles";

const gui = new GUI();

const config = {
  paused: true,

  gravity: 0,
  collisionDamping: 0.5,

  targetDensity: 1.5,
  pressureMultiplier: 0.1,
  smoothingRadius: 0.4,

  numParticles: 1000,
  maxParticles: MAX_PARTICLES,
  particleSize: 0.1,
  particleSpacing: 0,

  boundsWidth: 16,
  boundsHeight: 9,
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
simulationFolder.add(config, "targetDensity", 0.5, 5);
simulationFolder.add(config, "pressureMultiplier", 0, 20);

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
particleFolder.add(config, "smoothingRadius", 0.1, 1);

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
