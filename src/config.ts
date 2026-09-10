import GUI from "lil-gui";

import { setCameraDistance, setBoundsSize } from "./scene";
import { setParticleSize, setParticleGridPosition, start } from "./simulation";

const gui = new GUI();

const config = {
  paused: true,

  gravity: 0,
  collisionDamping: 0.5,

  targetDensity: 2.75,
  pressureMultiplier: 10,
  smoothingRadius: 1.2,

  numParticles: 100,
  particleSize: 0.5,
  particleSpacing: 0.1,

  boundsSize: 30,
  cameraDistance: 50,
};

const simulationFolder = gui.addFolder("Simulation");
simulationFolder.add(config, "paused");
simulationFolder.add(config, "gravity", -50, 50);
simulationFolder.add(config, "collisionDamping", 0, 1);
simulationFolder
  .add(config, "boundsSize", 10, 100)
  .onChange((size: number) => setBoundsSize(size));
simulationFolder.add(config, "targetDensity", 0.5, 6);
simulationFolder.add(config, "pressureMultiplier", 0, 50);

const particleFolder = gui.addFolder("Particles");
particleFolder.add(config, "numParticles", 1, 300, 1).onChange(() => start());
particleFolder
  .add(config, "particleSpacing", 0, 2)
  .onChange(() => setParticleGridPosition());
particleFolder.add(config, "particleSize", 0.1, 3).onChange((size: number) => {
  setParticleSize(size);
  setParticleGridPosition();
});
particleFolder.add(config, "smoothingRadius", 0.5, 5);

const cameraFolder = gui.addFolder("Camera");
cameraFolder
  .add(config, "cameraDistance", 10, 200)
  .onChange((distance: number) => setCameraDistance(distance));

export { config };
