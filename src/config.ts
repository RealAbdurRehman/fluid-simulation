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
  boundsSize: 30,
  numParticles: 1,
  particleSpacing: 0,
};

const simulationFolder = gui.addFolder("Simulation");
simulationFolder.add(config, "paused");
simulationFolder.add(config, "gravity", -100, 100);
simulationFolder.add(config, "collisionDamping", 0, 1);
simulationFolder
  .add(config, "boundsSize", 10, 80)
  .onChange((size: number) => setBoundsSize(size));

const particleFolder = gui.addFolder("Particles");
particleFolder.add(config, "numParticles", 1, 100, 1).onChange(() => start());
particleFolder
  .add(config, "particleSpacing", 0, 4)
  .onChange(() => setParticleGridPosition());
particleFolder.add(config, "particleSize", 0.1, 10).onChange((size: number) => {
  setParticleSize(size);
  setParticleGridPosition();
});

const cameraFolder = gui.addFolder("Camera");
cameraFolder
  .add(config, "cameraDistance", 10, 100)
  .onChange((distance: number) => setCameraDistance(distance));

export { config };
