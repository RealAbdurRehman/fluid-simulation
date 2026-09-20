import * as THREE from "three";

import { config, setupGUI } from "./config";
import { FluidSimulationGPU } from "./simulation";
import { SceneRenderer } from "./sceneRenderer";
import { SSFRRenderer } from "./ssfrRenderer";
import { camera, attachControls, resizeCamera } from "./scene";
import { MeshRegistry } from "./meshRegistry";
import { createSimUpdaters } from "./simUpdaters";
import { generateTerrain, terrainToGeometry } from "./terrain";
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

  let terrainMeshCounter = 0;

  function regenerateTerrain(): void {
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

    simulation.setTerrain(data);
    updaters.setTerrain(data, baseY);

    const id = `terrain_${terrainMeshCounter++}`;
    meshRegistry.registerTerrain(id, terrainToGeometry(data, baseY));
    sceneRenderer.setTerrainMesh(id);
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
      if (steps === MAX_STEPS_PER_FRAME) accumulator = 0;
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

function setupInteraction(args: {
  canvas: HTMLCanvasElement;
  camera: THREE.PerspectiveCamera;
  controls: ReturnType<typeof attachControls>;
  simulation: FluidSimulationGPU;
  audio: SimAudio;
}) {
  const { canvas, camera, controls, simulation, audio } = args;

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
    audio.onInteractionStart();
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
