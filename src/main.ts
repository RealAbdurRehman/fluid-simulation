import * as THREE from "three";

import { config, setupGUI } from "./config";
import { FluidSimulationGPU } from "./simulation";
import { GPUParticleRenderer } from "./renderer";
import { SceneRenderer } from "./sceneRenderer";
import { camera, attachControls, resizeCamera } from "./scene";
import { MeshRegistry } from "./meshRegistry";
import { createSimUpdaters } from "./simUpdaters";

async function bootstrap(): Promise<void> {
  const simulation = new FluidSimulationGPU();
  if (!(await simulation.initialize())) {
    console.error("Simulation failed to initialize (no WebGPU?)");
    return;
  }

  const device = simulation.getDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();

  const canvas = createCanvas();
  const context = configureContext(canvas, device, format);
  const controls = attachControls(canvas);

  const sceneRenderer = new SceneRenderer(device, format);
  const particleRenderer = new GPUParticleRenderer(
    device,
    simulation.getParticlesBuffer(),
    format,
  );

  const meshRegistry = new MeshRegistry(simulation, sceneRenderer);
  meshRegistry.register(
    "torusKnot",
    new THREE.TorusKnotGeometry(1, 0.35, 160, 24),
  );

  const updaters = createSimUpdaters(simulation, sceneRenderer, meshRegistry);

  setupGUI(
    () => simulation.updateParticleCount(),
    () => {},
    () => {},
    (index) => updaters.requestBakeForSlot(index),
    (index) => updaters.spawnObject(index),
  );

  for (let i = 0; i < config.objects.length; i++) {
    if (config.objects[i].physics && config.objects[i].type !== "none") {
      updaters.requestBakeForSlot(i);
      updaters.spawnObject(i);
    }
  }

  const viewState = createViewState();
  const interaction = setupInteraction({
    canvas,
    camera,
    controls,
    simulation,
  });

  function resize(): void {
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

  let stepOnce = false;
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") config.paused = !config.paused;
    if (e.code === "KeyR") simulation.initParticleGrid();
    if (e.code === "Period") {
      config.paused = true;
      stepOnce = true;
    }
  });

  const timer = new THREE.Timer();
  const FIXED_DELTA = 1 / 60;
  let accumulator = 0;

  function animate(timestamp: number): void {
    timer.update(timestamp);

    const frameDelta = Math.min(timer.getDelta(), 0.1);
    accumulator += frameDelta;

    simulation.pollProbeResults();

    controls.update();
    camera.updateMatrixWorld();
    viewState.update(camera);

    const encoder = device.createCommandEncoder();

    if (stepOnce) {
      stepOnce = false;
      accumulator = 0;
      updaters.updateBoundsRotation(FIXED_DELTA);
      updaters.updateObjects(FIXED_DELTA);
      simulation.recordStepCommands(encoder, FIXED_DELTA);
    } else if (!config.paused)
      while (accumulator >= FIXED_DELTA) {
        accumulator -= FIXED_DELTA;
        updaters.updateBoundsRotation(FIXED_DELTA);
        updaters.updateObjects(FIXED_DELTA);
        simulation.recordStepCommands(encoder, FIXED_DELTA);
      }
    else accumulator = 0;

    sceneRenderer.updateFrame(viewState.viewProj, [
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
      viewState.view,
      viewState.proj,
      [
        viewState.cameraRight.x,
        viewState.cameraRight.y,
        viewState.cameraRight.z,
      ],
      [viewState.cameraUp.x, viewState.cameraUp.y, viewState.cameraUp.z],
      config.numParticles,
    );

    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
  interaction.attach();
}

bootstrap();

function createCanvas(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, {
    position: "fixed",
    top: "0",
    left: "0",
    width: "100vw",
    height: "100vh",
    display: "block",
    zIndex: "0",
  } satisfies Partial<CSSStyleDeclaration>);
  document.body.appendChild(canvas);
  return canvas;
}

function configureContext(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  format: GPUTextureFormat,
): GPUCanvasContext {
  const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!context) throw new Error("Failed to acquire WebGPU canvas context");

  context.configure({ device, format, alphaMode: "opaque" });
  return context;
}

function createViewState() {
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
  const tmp = new THREE.Matrix4();

  const view = new Float32Array(16);
  const proj = new Float32Array(16);
  const viewProj = new Float32Array(16);

  const cameraRight = new THREE.Vector3();
  const cameraUp = new THREE.Vector3();
  const scratch = new THREE.Vector3();

  function update(cam: THREE.PerspectiveCamera): void {
    cam.matrixWorld.extractBasis(cameraRight, cameraUp, scratch);
    cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
    cam.matrixWorldInverse.toArray(view);

    tmp.copy(cam.projectionMatrix).premultiply(zRemap);
    tmp.toArray(proj);

    tmp.multiply(cam.matrixWorldInverse);
    tmp.toArray(viewProj);
  }

  return { view, proj, viewProj, cameraRight, cameraUp, update };
}

function setupInteraction(args: {
  canvas: HTMLCanvasElement;
  camera: THREE.PerspectiveCamera;
  controls: ReturnType<typeof attachControls>;
  simulation: FluidSimulationGPU;
}) {
  const { canvas, camera, controls, simulation } = args;

  const mouse = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();

  let isInteracting = false;
  let mode: "push" | "pull" | null = null;

  function updateRay(e: MouseEvent): void {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    if (!isInteracting || !mode) return;

    const strength =
      mode === "push"
        ? config.interactionStrength
        : -config.interactionStrength;
    simulation.setInteraction(
      raycaster.ray.origin,
      raycaster.ray.direction,
      strength,
    );
  }

  function begin(nextMode: "push" | "pull", e: MouseEvent): void {
    controls.enabled = false;
    isInteracting = true;
    mode = nextMode;
    updateRay(e);
  }

  function end(): void {
    if (!isInteracting) return;
    isInteracting = false;
    mode = null;
    controls.enabled = true;
    simulation.setInteraction(new THREE.Vector3(), new THREE.Vector3(), 0);
  }

  function attach(): void {
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("mousedown", (e) => {
      if (e.button === 2 || (e.button === 0 && e.shiftKey)) begin("push", e);
      else if (e.button === 1 || (e.button === 0 && e.ctrlKey))
        begin("pull", e);
    });

    window.addEventListener("mousemove", (e) => {
      if (isInteracting) updateRay(e);
    });

    window.addEventListener("mouseup", end);
  }

  return { attach };
}
