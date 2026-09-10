import GUI from "lil-gui";

import { setCameraDistance, setBoundsSize } from "./scene";
import { setParticleSize, setParticleGridPosition, start } from "./simulation";

const gui = new GUI();

const config = {
  paused: false,
  gravity: 0,
  collisionDamping: 1,
  particleSize: 0.5,
  cameraDistance: 50,
  boundsSize: 50,
  numParticles: 1,
  particleSpacing: 0,
  smoothingRadius: 1,
};

const simulationFolder = gui.addFolder("Simulation");
simulationFolder.add(config, "paused");
simulationFolder.add(config, "gravity", -100, 100);
simulationFolder.add(config, "collisionDamping", 0, 1);
simulationFolder
  .add(config, "boundsSize", 30, 150)
  .onChange((size: number) => setBoundsSize(size));

const particleFolder = gui.addFolder("Particles");
particleFolder.add(config, "numParticles", 1, 500, 1).onChange(() => start());
particleFolder
  .add(config, "particleSpacing", 0, 4)
  .onChange(() => setParticleGridPosition());
particleFolder.add(config, "particleSize", 0.1, 10).onChange((size: number) => {
  setParticleSize(size);
  setParticleGridPosition();
});
particleFolder.add(config, "smoothingRadius", 1, 5);

const cameraFolder = gui.addFolder("Camera");
cameraFolder
  .add(config, "cameraDistance", 10, 200)
  .onChange((distance: number) => setCameraDistance(distance));

export { config };
