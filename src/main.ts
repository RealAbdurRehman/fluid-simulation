import * as THREE from "three";
import { config, setupGUI } from "./config";
import {
  scene,
  camera,
  renderer,
  setCameraDistance,
  setBoundsSize,
} from "./scene";
import { FluidSimulationGPU } from "./simulation";
import { GPUParticleRenderer } from "./renderer";

async function bootstrap() {
  const gpuCanvas = document.createElement("canvas");
  gpuCanvas.width = window.innerWidth * window.devicePixelRatio;
  gpuCanvas.height = window.innerHeight * window.devicePixelRatio;
  gpuCanvas.style.position = "absolute";
  gpuCanvas.style.top = "0";
  gpuCanvas.style.left = "0";
  gpuCanvas.style.width = "100vw";
  gpuCanvas.style.height = "100vh";
  gpuCanvas.style.pointerEvents = "none";
  document.getElementById("app")!.appendChild(gpuCanvas);

  const simulation = new FluidSimulationGPU();
  const initialized = await simulation.initialize();
  if (!initialized) return;

  const gpuRenderer = new GPUParticleRenderer(
    simulation.getDevice(),
    gpuCanvas,
    simulation.getParticlesBuffer(),
  );

  const viewProj = new THREE.Matrix4();
  const viewProjArray = new Float32Array(16);

  setupGUI(
    () => simulation.updateParticleCount(),
    () => setBoundsSize(config.boundsWidth, config.boundsHeight),
    (dist: number) => setCameraDistance(dist),
    () => {},
  );

  setBoundsSize(config.boundsWidth, config.boundsHeight);
  setCameraDistance(config.cameraDistance);

  window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio, 2);

    renderer.setSize(width, height);
    renderer.setPixelRatio(dpr);

    gpuCanvas.width = width * dpr;
    gpuCanvas.height = height * dpr;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });

  let mouseButton: number | null = null;
  const mouse = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();
  const simulationPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const mouseWorld3D = new THREE.Vector3();
  const mouseWorld = new THREE.Vector2();

  function updatePointer(e: PointerEvent) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    raycaster.ray.intersectPlane(simulationPlane, mouseWorld3D);
    mouseWorld.set(mouseWorld3D.x, mouseWorld3D.y);
  }

  window.addEventListener("contextmenu", (e) => e.preventDefault());

  window.addEventListener("pointerdown", (e) => {
    mouseButton = e.button;
    updatePointer(e);
    if (mouseButton === 0)
      simulation.setInteraction(mouseWorld, -config.interactionStrength);
    if (mouseButton === 2)
      simulation.setInteraction(mouseWorld, config.interactionStrength);
  });

  window.addEventListener("pointermove", (e) => {
    updatePointer(e);
    if (mouseButton === 0)
      simulation.setInteraction(mouseWorld, -config.interactionStrength);
    if (mouseButton === 2)
      simulation.setInteraction(mouseWorld, config.interactionStrength);
  });

  window.addEventListener("pointerup", () => {
    mouseButton = null;
    simulation.setInteraction(mouseWorld, 0);
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") config.paused = !config.paused;
    if (e.code === "KeyR") simulation.initParticleGrid();
  });

  const timer = new THREE.Timer();
  const FIXED_DELTA = 1 / 60;
  let accumulator = 0;

  function animate(timestamp: number) {
    timer.update(timestamp);
    accumulator += Math.min(timer.getDelta(), 0.1);

    camera.updateMatrixWorld();
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    viewProj.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    );
    viewProj.toArray(viewProjArray);

    const commandEncoder = simulation.getDevice().createCommandEncoder();
    while (accumulator >= FIXED_DELTA) {
      simulation.recordStepCommands(commandEncoder, FIXED_DELTA);
      accumulator -= FIXED_DELTA;
    }

    renderer.render(scene, camera);

    gpuRenderer.render(commandEncoder, viewProjArray, config.numParticles);
    simulation.getDevice().queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

bootstrap();
