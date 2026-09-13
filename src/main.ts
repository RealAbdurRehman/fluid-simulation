import * as THREE from "three";

import { config, setupGUI } from "./config";
import { FluidSimulationGPU } from "./simulation";
import { GPUParticleRenderer } from "./renderer";
import { SceneRenderer } from "./sceneRenderer";
import { camera, attachControls, resizeCamera } from "./scene";

async function bootstrap() {
  const simulation = new FluidSimulationGPU();
  const initialized = await simulation.initialize();
  if (!initialized) {
    console.error("Simulation failed to initialize (no WebGPU?)");
    return;
  }

  const device = simulation.getDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();

  const canvas = document.createElement("canvas");
  canvas.style.position = "fixed";
  canvas.style.top = "0";
  canvas.style.left = "0";
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
  canvas.style.display = "block";
  canvas.style.zIndex = "0";
  document.body.appendChild(canvas);

  const contextOrNull = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!contextOrNull) {
    console.error("Failed to acquire WebGPU canvas context");
    return;
  }
  const context: GPUCanvasContext = contextOrNull;
  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  const controls = attachControls(canvas);

  const sceneRenderer = new SceneRenderer(device, format);
  const particleRenderer = new GPUParticleRenderer(
    device,
    simulation.getParticlesBuffer(),
    format,
  );

  const viewMatrixArray = new Float32Array(16);
  const projMatrixArray = new Float32Array(16);
  const viewProjArray = new Float32Array(16);

  const zRemap = new THREE.Matrix4().set(
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0.5,
    0.5,
    0,
    0,
    0,
    1,
  );
  const tmpMat = new THREE.Matrix4();

  setupGUI(
    () => simulation.updateParticleCount(),
    () => {},
    () => {},
  );

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio, 2);

    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    resizeCamera();
    sceneRenderer.resize(canvas.width, canvas.height);
  }

  resize();
  window.addEventListener("resize", resize);

  const mouse = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();

  let isInteracting = false;
  let interactionMode: "push" | "pull" | null = null;

  function updateRaycast(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
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

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("mousedown", (e) => {
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

    tmpMat.copy(camera.projectionMatrix).premultiply(zRemap);
    tmpMat.toArray(projMatrixArray);

    tmpMat.multiply(camera.matrixWorldInverse);
    tmpMat.toArray(viewProjArray);

    const encoder = device.createCommandEncoder();

    if (stepOnce) {
      simulation.recordStepCommands(encoder, FIXED_DELTA, true);
      stepOnce = false;
    } else {
      while (accumulator >= FIXED_DELTA) {
        simulation.recordStepCommands(encoder, FIXED_DELTA);
        accumulator -= FIXED_DELTA;
      }
    }

    sceneRenderer.updateFrame(viewProjArray, [
      camera.position.x,
      camera.position.y,
      camera.position.z,
    ]);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: sceneRenderer.getMSAAView(),
          resolveTarget: context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "discard",
        },
      ],
      depthStencilAttachment: {
        view: sceneRenderer.getDepthView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    });

    sceneRenderer.encode(pass);
    particleRenderer.encode(
      pass,
      viewMatrixArray,
      projMatrixArray,
      [cameraRight.x, cameraRight.y, cameraRight.z],
      [cameraUp.x, cameraUp.y, cameraUp.z],
      config.numParticles,
    );

    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

bootstrap();
