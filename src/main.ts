import * as THREE from "three";

import { config, setupGUI } from "./config";
import { FluidSimulationGPU } from "./simulation";
import { GPUParticleRenderer } from "./renderer";
import { scene, camera, controls, renderer, setBoundsSize } from "./scene";

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

  const viewMatrixArray = new Float32Array(16);
  const projMatrixArray = new Float32Array(16);

  setupGUI(
    () => simulation.updateParticleCount(),
    () =>
      setBoundsSize(
        config.boundsWidth,
        config.boundsHeight,
        config.boundsDepth,
      ),
    () => {},
  );

  setBoundsSize(config.boundsWidth, config.boundsHeight, config.boundsDepth);

  window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio, 2);

    renderer.setSize(width, height);
    renderer.setPixelRatio(dpr);

    gpuCanvas.width = width * dpr;
    gpuCanvas.height = height * dpr;
    gpuRenderer.resize(gpuCanvas.width, gpuCanvas.height);

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });

  const mouse = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();

  let isInteracting = false;
  let interactionMode: "push" | "pull" | null = null;

  function updateRaycast(e: MouseEvent) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    if (isInteracting && interactionMode) {
      const strength =
        interactionMode === "push"
          ? config.interactionStrength
          : -config.interactionStrength;
      simulation.setInteraction(
        raycaster.ray.origin,
        raycaster.ray.direction,
        strength,
      );
    }
  }

  renderer.domElement.addEventListener("contextmenu", (e) =>
    e.preventDefault(),
  );

  renderer.domElement.addEventListener("mousedown", (e) => {
    if (e.button === 2 || (e.button === 0 && e.shiftKey)) {
      controls.enabled = false;
      isInteracting = true;
      interactionMode = "push";
      updateRaycast(e);
    } else if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
      controls.enabled = false;
      isInteracting = true;
      interactionMode = "pull";
      updateRaycast(e);
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (isInteracting) updateRaycast(e);
  });

  window.addEventListener("mouseup", () => {
    if (isInteracting) {
      isInteracting = false;
      interactionMode = null;
      controls.enabled = true;
      simulation.setInteraction(new THREE.Vector3(), new THREE.Vector3(), 0);
    }
  });

  let stepOnce = false;
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") config.paused = !config.paused;
    if (e.code === "KeyR") simulation.initParticleGrid();
    if (e.code === "Period") {
      config.paused = true;
      stepOnce = true;
    }
  });

  let accumulator = 0;
  const timer = new THREE.Timer();
  const FIXED_DELTA = 1 / 60;

  const cameraRight = new THREE.Vector3();
  const cameraUp = new THREE.Vector3();

  function animate(timestamp: number) {
    timer.update(timestamp);
    accumulator += Math.min(timer.getDelta(), 0.1);

    controls.update();

    camera.updateMatrixWorld();
    camera.matrixWorld.extractBasis(cameraRight, cameraUp, new THREE.Vector3());
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    camera.matrixWorldInverse.toArray(viewMatrixArray);
    camera.projectionMatrix.toArray(projMatrixArray);

    const commandEncoder = simulation.getDevice().createCommandEncoder();
    if (stepOnce) {
      simulation.recordStepCommands(commandEncoder, FIXED_DELTA, true);
      stepOnce = false;
    } else {
      while (accumulator >= FIXED_DELTA) {
        simulation.recordStepCommands(commandEncoder, FIXED_DELTA);
        accumulator -= FIXED_DELTA;
      }
    }

    renderer.render(scene, camera);

    gpuRenderer.render(
      commandEncoder,
      viewMatrixArray,
      projMatrixArray,
      [cameraRight.x, cameraRight.y, cameraRight.z],
      [cameraUp.x, cameraUp.y, cameraUp.z],
      config.numParticles,
    );

    simulation.getDevice().queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

bootstrap();
