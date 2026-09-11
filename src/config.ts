import GUI from "lil-gui";

import { setCameraDistance, setBoundsSize } from "./scene";
import { setParticleSize, setParticleGridPosition, start } from "./simulation";

const gui = new GUI();

const config = {
  paused: true,

  gravity: 0,
  collisionDamping: 0.5,

  targetDensity: 1.5,
  pressureMultiplier: 0.1,
  smoothingRadius: 0.4,

  numParticles: 1000,
  particleSize: 0.1,
  particleSpacing: 0,

  boundsSize: 16,
  cameraDistance: 15,
};

const simulationFolder = gui.addFolder("Simulation");
simulationFolder.add(config, "paused");
simulationFolder.add(config, "gravity", -20, 20);
simulationFolder.add(config, "collisionDamping", 0, 1);
simulationFolder
  .add(config, "boundsSize", 5, 30)
  .onChange((size: number) => setBoundsSize(size));
simulationFolder.add(config, "targetDensity", 0.5, 5);
simulationFolder.add(config, "pressureMultiplier", 0, 2);

const particleFolder = gui.addFolder("Particles");
particleFolder.add(config, "numParticles", 1, 4000, 1).onChange(() => start());
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

const cameraFolder = gui.addFolder("Camera");
cameraFolder
  .add(config, "cameraDistance", 5, 50)
  .onChange((distance: number) => setCameraDistance(distance));

export { config };
