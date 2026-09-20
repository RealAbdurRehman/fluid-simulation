import { GPUBufferUsage, GPUMapMode } from "../types";

const GRID_SIZE = 8;
const CELL_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE;
const WORDS_PER_CELL = 2;
const BUFFER_WORDS = CELL_COUNT * WORDS_PER_CELL;
const BUFFER_BYTES = BUFFER_WORDS * 4;

const WGSL = /* wgsl */ `
struct Particle {
  position: vec4<f32>,
  predictedPosition: vec4<f32>,
  velocity: vec4<f32>,
  density: vec4<f32>,
};

struct SplashParams {
  numParticles: u32,
  gridSize: u32,
  vyThreshold: f32,
  targetDensity: f32,
  boundsMin: vec4<f32>,
  boundsInvSize: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sp: SplashParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> cells: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sp.numParticles) { return; }
  let p = particles[i];

  let vy = p.velocity.y;
  if (vy > -sp.vyThreshold) { return; }

  let rel = p.density.x / max(sp.targetDensity, 0.0001);
  if (rel < 0.6) { return; }

  let local = (p.position.xyz - sp.boundsMin.xyz) * sp.boundsInvSize.x;
  if (any(local < vec3<f32>(0.0)) || any(local >= vec3<f32>(1.0))) { return; }

  let gs = sp.gridSize;
  let ci = vec3<u32>(local * f32(gs));
  let idx = ci.x + ci.y * gs + ci.z * gs * gs;
  if (idx >= gs * gs * gs) { return; }

  let speed = length(p.velocity.xyz);
  let energy = u32(clamp(speed * speed * 100.0, 0.0, 65535.0));

  atomicAdd(&cells[idx * 2u], 1u);
  atomicAdd(&cells[idx * 2u + 1u], energy);
}
`;

export interface SplashEvent {
  x: number;
  y: number;
  z: number;
  energy: number;
  count: number;
}

export class SplashDetector {
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly uniformBuffer: GPUBuffer;
  private readonly cellsBuffer: GPUBuffer;
  private readonly readback: GPUBuffer;

  private readonly uniformData = new ArrayBuffer(48);
  private readonly uniformU32 = new Uint32Array(this.uniformData);
  private readonly uniformF32 = new Float32Array(this.uniformData);

  private hasFresh = false;
  private mapInFlight = false;
  private frame = 0;

  private boundsW = 1;
  private boundsH = 1;
  private boundsD = 1;

  private pendingEvents: SplashEvent[] = [];

  public sampleInterval = 3;
  public vyThreshold = 7.5;
  public minCellCount = 14;
  public minEnergy = 8.0;
  public maxEventsPerPoll = 4;
  constructor(device: GPUDevice, particlesBuffer: GPUBuffer) {
    this.device = device;

    const module = device.createShaderModule({ code: WGSL });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    this.uniformBuffer = device.createBuffer({
      label: "splashParams",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.cellsBuffer = device.createBuffer({
      label: "splashCells",
      size: BUFFER_BYTES,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    this.readback = device.createBuffer({
      label: "splashReadback",
      size: BUFFER_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: particlesBuffer } },
        { binding: 2, resource: { buffer: this.cellsBuffer } },
      ],
    });
  }
  public record(
    encoder: GPUCommandEncoder,
    numParticles: number,
    boundsW: number,
    boundsH: number,
    boundsD: number,
    targetDensity: number,
  ): void {
    if (this.mapInFlight) return;
    if (++this.frame % this.sampleInterval !== 0) return;

    this.uniformU32[0] = numParticles;
    this.uniformU32[1] = GRID_SIZE;
    this.uniformF32[2] = this.vyThreshold;
    this.uniformF32[3] = targetDensity;

    this.uniformF32[4] = -boundsW * 0.5;
    this.uniformF32[5] = -boundsH * 0.5;
    this.uniformF32[6] = -boundsD * 0.5;
    this.uniformF32[7] = 0;

    this.uniformF32[8] = 1 / Math.max(boundsW, 1e-4);
    this.uniformF32[9] = 1 / Math.max(boundsH, 1e-4);
    this.uniformF32[10] = 1 / Math.max(boundsD, 1e-4);
    this.uniformF32[11] = 0;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

    encoder.clearBuffer(this.cellsBuffer);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(numParticles / 256));
    pass.end();

    encoder.copyBufferToBuffer(
      this.cellsBuffer,
      0,
      this.readback,
      0,
      BUFFER_BYTES,
    );

    this.boundsW = boundsW;
    this.boundsH = boundsH;
    this.boundsD = boundsD;
    this.hasFresh = true;
  }
  public poll(): void {
    if (!this.hasFresh || this.mapInFlight) return;

    this.hasFresh = false;
    this.mapInFlight = true;

    this.readback
      .mapAsync(GPUMapMode.READ, 0, BUFFER_BYTES)
      .then(() => {
        const raw = new Uint32Array(
          this.readback.getMappedRange(0, BUFFER_BYTES).slice(0),
        );

        this.readback.unmap();
        this.mapInFlight = false;

        this.processCells(raw);
      })
      .catch((err) => {
        console.warn("[splash] readback failed:", err);

        try {
          this.readback.unmap();
        } catch {}

        this.mapInFlight = false;
      });
  }
  private processCells(raw: Uint32Array): void {
    const w = this.boundsW;
    const h = this.boundsH;
    const d = this.boundsD;
    const cellW = w / GRID_SIZE;
    const cellH = h / GRID_SIZE;
    const cellD = d / GRID_SIZE;

    const events: SplashEvent[] = [];

    for (let idx = 0; idx < CELL_COUNT; idx++) {
      const count = raw[idx * 2];
      if (count < this.minCellCount) continue;

      const energy = raw[idx * 2 + 1] / 100;
      if (energy < this.minEnergy) continue;

      const cx = idx % GRID_SIZE;
      const cy = Math.floor(idx / GRID_SIZE) % GRID_SIZE;
      const cz = Math.floor(idx / (GRID_SIZE * GRID_SIZE));

      const x = -w * 0.5 + (cx + 0.5) * cellW;
      const y = -h * 0.5 + (cy + 0.5) * cellH;
      const z = -d * 0.5 + (cz + 0.5) * cellD;

      events.push({ x, y, z, energy, count });
    }

    if (events.length === 0) return;

    events.sort((a, b) => b.energy - a.energy);
    this.pendingEvents = events.slice(0, this.maxEventsPerPoll);
  }
  public drain(): SplashEvent[] {
    if (this.pendingEvents.length === 0) return [];

    const out = this.pendingEvents;
    this.pendingEvents = [];

    return out;
  }
}
