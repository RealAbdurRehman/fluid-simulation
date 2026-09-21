import * as THREE from "three";

import { config, setupGUI } from "./config";
import { FluidSimulationGPU } from "./simulation";
import { SceneRenderer } from "./sceneRenderer";
import { SSFRRenderer } from "./ssfrRenderer";
import { camera, attachControls, resizeCamera } from "./scene";
import { MeshRegistry } from "./meshRegistry";
import { createSimUpdaters } from "./simUpdaters";
import {
  generateTerrain,
  terrainToGeometry,
  createTerrainMeshPatch,
  patchTerrainMesh,
  createDirtyRect,
  resetDirtyRect,
  type TerrainData,
  type TerrainMeshPatch,
} from "./terrain";
import { createSculptState, raycastTerrain, sculptFrame } from "./sculpt";
import { AudioEngine } from "./audio/audioEngine";
import { FluidStatsGPU } from "./audio/fluidStats";
import { SplashDetector } from "./audio/splashDetector";
import { SimAudio } from "./audio/audio";

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

  const ssfr = new SSFRRenderer(
    device,
    format,
    simulation.getParticlesBuffer(),
  );

  sceneRenderer.setLightMaps(
    ssfr.getLightDepthView(),
    ssfr.getLightThicknessView(),
    ssfr.getCausticsView(),
  );

  const meshRegistry = new MeshRegistry(simulation, sceneRenderer);
  meshRegistry.preloadAll();

  const audioEngine = new AudioEngine();
  const audio = new SimAudio(audioEngine);
  const fluidStats = new FluidStatsGPU(device, simulation.getParticlesBuffer());
  const splashDetector = new SplashDetector(
    device,
    simulation.getParticlesBuffer(),
  );

  audioEngine.armAutoResume();
  void audioEngine.load().then(() => audio.initBeds());

  const updaters = createSimUpdaters(
    simulation,
    sceneRenderer,
    meshRegistry,
    audio,
  );

  const heldKeys = new Set<string>();

  function isTypingTarget(t: EventTarget | null): boolean {
    const el = t as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
  }

  window.addEventListener("keydown", (e) => {
    if (isTypingTarget(e.target)) return;
    heldKeys.add(e.code);
  });
  window.addEventListener("keyup", (e) => {
    heldKeys.delete(e.code);
  });
  window.addEventListener("blur", () => {
    heldKeys.clear();
  });

  const isSculpting = (): boolean =>
    config.sculpt.enabled && heldKeys.has("KeyB");

  let terrainMeshCounter = 0;
  let terrainData: TerrainData | null = null;
  let terrainBaseY = 0;
  let terrainPatch: TerrainMeshPatch | null = null;

  const sculptState = createSculptState();
  const sculptDirty = createDirtyRect();
  const sculptNDC = new THREE.Vector2(0, 0);
  const sculptRaycaster = new THREE.Raycaster();

  function sampleTerrainNormalized(x: number, z: number): number {
    if (!terrainData) return 0;
    const N = terrainData.resolution;
    const halfX = terrainData.extentX * 0.5;
    const halfZ = terrainData.extentZ * 0.5;
    const cellX = terrainData.extentX / (N - 1);
    const cellZ = terrainData.extentZ / (N - 1);

    const u = Math.max(0, Math.min(N - 1, (x + halfX) / cellX));
    const v = Math.max(0, Math.min(N - 1, (z + halfZ) / cellZ));

    const i0 = Math.floor(u);
    const j0 = Math.floor(v);
    const i1 = Math.min(i0 + 1, N - 1);
    const j1 = Math.min(j0 + 1, N - 1);
    const fx = u - i0;
    const fy = v - j0;

    const h00 = terrainData.heights[j0 * N + i0];
    const h10 = terrainData.heights[j0 * N + i1];
    const h01 = terrainData.heights[j1 * N + i0];
    const h11 = terrainData.heights[j1 * N + i1];

    return (
      (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy
    );
  }

  function regenerateTerrain(): void {
    terrainData = null;
    terrainPatch = null;

    if (!config.terrainEnabled) {
      simulation.setTerrain(null);
      sceneRenderer.setTerrainMesh(null);
      updaters.setTerrain(null, 0);
      return;
    }

    const baseY = -config.boundsHeight / 2;
    const data = generateTerrain(
      config.terrainResolution,
      config.boundsWidth,
      config.boundsDepth,
      config.terrainHeightScale,
      config.terrainSeed,
    );

    terrainData = data;
    terrainBaseY = baseY;

    simulation.setTerrain(data);
    updaters.setTerrain(data, baseY);

    const id = `terrain_${terrainMeshCounter++}`;
    meshRegistry.registerTerrain(id, terrainToGeometry(data, baseY));
    sceneRenderer.setTerrainMesh(id);

    const vbuf = sceneRenderer.getMeshVertexBuffer(id);
    terrainPatch = vbuf ? createTerrainMeshPatch(vbuf, data, baseY) : null;
  }

  setupGUI(
    () => {
      simulation.updateParticleCount();
      ssfr.setParticleCount(config.numParticles);
    },
    regenerateTerrain,
    () => {},
    (index) => updaters.requestBakeForSlot(index),
    (index) => updaters.spawnObject(index),
    regenerateTerrain,
  );

  ssfr.setParticleCount(config.numParticles);

  regenerateTerrain();

  for (let i = 0; i < config.objects.length; i++) {
    if (config.objects[i].modelId !== "none") {
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
    audio,
    isSculpting,
  });

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    sculptNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    sculptNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  });

  canvas.addEventListener("mousedown", (e) => {
    if (!isSculpting()) return;
    if (e.button !== 0) return;
    if (!terrainData) return;

    e.preventDefault();
    e.stopPropagation();

    sculptState.mode = config.sculpt.mode;
    sculptState.active = true;
    sculptState.hasLast = false;
    sculptState.cursorValid = false;

    if (sculptState.mode === "flatten") {
      sculptRaycaster.setFromCamera(sculptNDC, camera);
      const hit = raycastTerrain(
        sculptRaycaster.ray.origin,
        sculptRaycaster.ray.direction,
        simulation.getBoundsQuaternion(),
        terrainData,
        terrainBaseY,
      );
      if (hit) {
        sculptState.flattenTarget = sampleTerrainNormalized(hit.x, hit.z);
      }
    }

    controls.enabled = false;
  });

  window.addEventListener("mouseup", () => {
    if (!sculptState.active) return;
    sculptState.active = false;
    sculptState.hasLast = false;
    if (isSculpting()) controls.enabled = true;
  });

  function resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    resizeCamera();
    sceneRenderer.resize(canvas.width, canvas.height);
    ssfr.resize(canvas.width, canvas.height);
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
  const MAX_STEPS_PER_FRAME = 2;
  let accumulator = 0;
  let simTime = 0;

  function animate(timestamp: number): void {
    timer.update(timestamp);

    const frameDelta = Math.min(timer.getDelta(), 0.1);
    accumulator += frameDelta;

    simulation.pollProbeResults();
    fluidStats.poll();
    splashDetector.poll();

    controls.update();
    camera.updateMatrixWorld();
    viewState.update(camera);
    updaters.setListenerPosition(camera.position);

    if (isSculpting() && terrainData && terrainPatch) {
      sculptRaycaster.setFromCamera(sculptNDC, camera);
      const hit = raycastTerrain(
        sculptRaycaster.ray.origin,
        sculptRaycaster.ray.direction,
        simulation.getBoundsQuaternion(),
        terrainData,
        terrainBaseY,
      );

      if (hit) {
        sculptState.cursorX = hit.x;
        sculptState.cursorZ = hit.z;
        sculptState.cursorValid = true;
      } else {
        sculptState.cursorValid = false;
        sculptState.hasLast = false;
      }

      sculptState.radius = config.sculpt.radius;
      sculptState.strength = config.sculpt.strength;
      sculptState.minHeight = config.sculpt.minHeight;
      sculptState.maxHeight = config.sculpt.maxHeight;

      sculptFrame(sculptState, terrainData, frameDelta, sculptDirty);

      if (sculptDirty.active) {
        simulation.updateTerrainRegion(
          terrainData,
          sculptDirty.i0,
          sculptDirty.j0,
          sculptDirty.i1,
          sculptDirty.j1,
        );
        patchTerrainMesh(device, terrainPatch, sculptDirty);
        resetDirtyRect(sculptDirty);
      }
    } else if (sculptState.active) {
      sculptState.active = false;
      sculptState.hasLast = false;
      if (controls.enabled === false) controls.enabled = true;
    }

    const encoder = device.createCommandEncoder();

    if (stepOnce) {
      stepOnce = false;
      accumulator = 0;
      updaters.updateBoundsRotation(FIXED_DELTA);
      updaters.updateObjects(FIXED_DELTA);
      simulation.recordStepCommands(encoder, FIXED_DELTA);
    } else if (!config.paused) {
      simTime += frameDelta;
      let steps = 0;
      while (accumulator >= FIXED_DELTA && steps < MAX_STEPS_PER_FRAME) {
        accumulator -= FIXED_DELTA;
        updaters.updateBoundsRotation(FIXED_DELTA);
        updaters.updateObjects(FIXED_DELTA);
        simulation.recordStepCommands(encoder, FIXED_DELTA);
        steps++;
      }

      const maxBacklog = FIXED_DELTA * (MAX_STEPS_PER_FRAME - 1);
      if (accumulator > maxBacklog) accumulator = maxBacklog;
    } else accumulator = 0;

    if (!config.paused && config.audio.enabled)
      fluidStats.record(encoder, config.numParticles, config.targetDensity);

    if (!config.paused && config.numParticles > 0)
      splashDetector.record(
        encoder,
        config.numParticles,
        config.boundsWidth,
        config.boundsHeight,
        config.boundsDepth,
        config.targetDensity,
      );

    ssfr.time = simTime;

    const aspect = canvas.width / canvas.height;
    const tanHalfFovY = Math.tan((camera.fov * Math.PI) / 360);

    ssfr.updateFrame(
      viewState.view,
      viewState.proj,
      [
        viewState.cameraRight.x,
        viewState.cameraRight.y,
        viewState.cameraRight.z,
      ],
      [viewState.cameraUp.x, viewState.cameraUp.y, viewState.cameraUp.z],
      camera.near,
      camera.far,
      tanHalfFovY,
      aspect,
    );

    sceneRenderer.updateLight(
      ssfr.getLightViewProj(),
      ssfr.getLightView(),
      ssfr.lightAbsorb,
    );

    sceneRenderer.time = timer.getElapsed();
    sceneRenderer.waterLevel = -1.0;
    sceneRenderer.updateFrame(viewState.viewProj, [
      camera.position.x,
      camera.position.y,
      camera.position.z,
    ]);

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ssfr.getLightDepthView(),
            clearValue: { r: 1e6, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: ssfr.getLightDepthStencilView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      ssfr.encodeLightDepth(pass, config.numParticles);
      pass.end();
    }

    {
      const pass = encoder.beginComputePass();
      ssfr.encodeCaustics(pass);
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ssfr.getLightDepthView(),
            clearValue: { r: 1e6, g: 0, b: 0, a: 0 },
            loadOp: "load",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: ssfr.getLightDepthStencilView(),
          depthClearValue: 1.0,
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
      });
      sceneRenderer.encodeLightObjects(pass);
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ssfr.getLightThicknessView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      ssfr.encodeLightThickness(pass, config.numParticles);
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ssfr.getSceneColorView(),
            clearValue: { r: 0.02, g: 0.05, b: 0.1, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: ssfr.getSceneDepthView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      sceneRenderer.encode(pass);
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ssfr.getFluidDepthView(),
            clearValue: { r: ssfr.getFarDepth(), g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          },
          {
            view: ssfr.getFluidWorldPosView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: ssfr.getFluidDepthStencilView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "discard",
        },
      });
      ssfr.encodeParticleDepth(pass, config.numParticles);
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ssfr.getFluidThicknessView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      ssfr.encodeParticleThickness(pass, config.numParticles);
      pass.end();
    }

    {
      const pass = encoder.beginComputePass();
      ssfr.encodeBilateralH(pass);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass();
      ssfr.encodeBilateralV(pass);
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      ssfr.encodeComposite(pass);
      pass.end();
    }

    audio.feedSplashes(splashDetector.drain(), frameDelta);
    audio.update(frameDelta, {
      camera,
      paused: config.paused,
      fluid: fluidStats.latest,
      listenerSubmersion: updaters.getListenerSubmersion(),
      particleCount: config.numParticles,
    });

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

interface SetupInteractionArgs {
  canvas: HTMLCanvasElement;
  camera: THREE.PerspectiveCamera;
  controls: ReturnType<typeof attachControls>;
  simulation: FluidSimulationGPU;
  audio: SimAudio;
  isSculpting?: () => boolean;
}

function setupInteraction(args: SetupInteractionArgs) {
  const { canvas, camera, controls, simulation, audio, isSculpting } = args;

  const mouse = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();

  let isInteracting = false;
  let mode: "push" | "pull" | "vortex" | null = null;

  const heldKeys = new Set<string>();

  function isTypingTarget(t: EventTarget | null): boolean {
    const el = t as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
  }

  function onKeyDown(e: KeyboardEvent): void {
    if (isTypingTarget(e.target)) return;
    heldKeys.add(e.code);
  }

  function onKeyUp(e: KeyboardEvent): void {
    heldKeys.delete(e.code);
  }

  function onBlur(): void {
    heldKeys.clear();
  }

  function resolveMouseMode(e: MouseEvent): "push" | "pull" | "vortex" | null {
    if (isSculpting && isSculpting()) return null;
    if (e.button === 1) return "pull";
    if (e.button === 2) return "push";

    if (e.button === 0) {
      if (e.shiftKey) return "push";
      if (e.ctrlKey || e.metaKey) return "pull";
      if (heldKeys.has("KeyV")) return "vortex";
      return null;
    }

    return null;
  }

  function updateRay(e: MouseEvent): void {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    if (!isInteracting || !mode) return;

    let strength: number;
    if (mode === "vortex") strength = config.interactionStrength;
    else
      strength =
        mode === "push"
          ? config.interactionStrength
          : -config.interactionStrength;

    simulation.setInteraction(
      raycaster.ray.origin,
      raycaster.ray.direction,
      strength,
      mode,
    );
  }

  const boundsBox = new THREE.Box3();
  const localRay = new THREE.Ray();
  const invQuat = new THREE.Quaternion();
  const rayOriginLocal = new THREE.Vector3();
  const rayDirLocal = new THREE.Vector3();

  function rayHitsFluidBounds(ray: THREE.Ray): boolean {
    const hx = config.boundsWidth / 2;
    const hy = config.boundsHeight / 2;
    const hz = config.boundsDepth / 2;
    boundsBox.min.set(-hx, -hy, -hz);
    boundsBox.max.set(hx, hy, hz);

    invQuat.copy(simulation.getBoundsQuaternion()).invert();
    rayOriginLocal.copy(ray.origin).applyQuaternion(invQuat);
    rayDirLocal.copy(ray.direction).applyQuaternion(invQuat);
    localRay.set(rayOriginLocal, rayDirLocal);

    return localRay.intersectsBox(boundsBox);
  }

  function begin(nextMode: "push" | "pull" | "vortex", e: MouseEvent): void {
    controls.enabled = false;
    isInteracting = true;
    mode = nextMode;
    updateRay(e);
    if (rayHitsFluidBounds(raycaster.ray)) audio.onInteractionStart();
  }

  function end(): void {
    if (!isInteracting) return;
    isInteracting = false;
    mode = null;
    controls.enabled = true;
    simulation.setInteraction(
      new THREE.Vector3(),
      new THREE.Vector3(),
      0,
      "push",
    );
  }

  function attach(): void {
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("mousedown", (e) => {
      const m = resolveMouseMode(e);
      if (m) begin(m, e);
    });

    window.addEventListener("mousemove", (e) => {
      if (isInteracting) updateRay(e);
    });

    window.addEventListener("mouseup", end);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
  }

  return { attach };
}
