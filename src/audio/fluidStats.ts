import { GPUBufferUsage, GPUMapMode } from "../types";

const STATS_WGSL = /* wgsl */ `
struct Particle {
  position: vec4<f32>,
  predictedPosition: vec4<f32>,
  velocity: vec4<f32>,
  density: vec4<f32>,
};

struct StatsParams {
  numParticles: u32,
  targetDensity: f32,
  surfaceThreshold: f32,
  sprayThreshold: f32,
};

@group(0) @binding(0) var<uniform> sp: StatsParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;

var<workgroup> shSum: array<vec4<f32>, 256>;
var<workgroup> shAux: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  var a = vec4<f32>(0.0);
  var b = vec2<f32>(0.0);

  if (gid.x < sp.numParticles) {
    let p = particles[gid.x];
    var speed = length(p.velocity.xyz);
    if (!(speed == speed) || speed > 500.0) { speed = 0.0; }

    let rel = p.density.x / max(sp.targetDensity, 0.0001);
    let foam = clamp(p.density.z, 0.0, 1.0);

    var isSurface = 0.0;
    if (rel < sp.surfaceThreshold) { isSurface = 1.0; }
    var isSpray = 0.0;
    if (rel < sp.sprayThreshold) { isSpray = 1.0; }

    a = vec4<f32>(speed, foam, isSurface, speed * isSurface);
    b = vec2<f32>(isSpray, speed);
  }

  shSum[lid.x] = a;
  shAux[lid.x] = b;
  workgroupBarrier();

  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      shSum[lid.x] = shSum[lid.x] + shSum[lid.x + s];

      let o = shAux[lid.x + s];
      let m = shAux[lid.x];
      shAux[lid.x] = vec2<f32>(m.x + o.x, max(m.y, o.y));
    }

    workgroupBarrier();
  }

  if (lid.x == 0u) {
    let t = shSum[0];
    let u = shAux[0];

    atomicAdd(&stats[0], u32(t.x * 256.0));
    atomicAdd(&stats[1], u32(t.y * 1024.0));
    atomicAdd(&stats[2], u32(t.z));
    atomicAdd(&stats[3], u32(t.w * 256.0));
    atomicAdd(&stats[4], u32(u.x));
    atomicMax(&stats[5], u32(u.y * 256.0));
  }
}
`;

const STATS_WORDS = 6;
const STATS_BYTES = STATS_WORDS * 4;

export interface FluidStats {
  avgSpeed: number;
  maxSpeed: number;
  avgFoam: number;
  surfaceFraction: number;
  avgSurfaceSpeed: number;
  sprayCount: number;
}

export class FluidStatsGPU {
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly uniformBuffer: GPUBuffer;
  private readonly statsBuffer: GPUBuffer;
  private readonly readback: GPUBuffer;

  private readonly uniformData = new ArrayBuffer(16);
  private readonly uniformU32 = new Uint32Array(this.uniformData);
  private readonly uniformF32 = new Float32Array(this.uniformData);

  private hasFresh = false;
  private mapInFlight = false;
  private frame = 0;
  private lastCount = 1;

  public sampleInterval = 6;
  public surfaceThreshold = 0.8;
  public sprayThreshold = 0.3;

  public latest: FluidStats | null = null;
  constructor(device: GPUDevice, particlesBuffer: GPUBuffer) {
    this.device = device;

    const module = device.createShaderModule({ code: STATS_WGSL });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    this.uniformBuffer = device.createBuffer({
      label: "audioStatsParams",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.statsBuffer = device.createBuffer({
      label: "audioStats",
      size: STATS_BYTES,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    this.readback = device.createBuffer({
      label: "audioStatsReadback",
      size: STATS_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: particlesBuffer } },
        { binding: 2, resource: { buffer: this.statsBuffer } },
      ],
    });
  }
  public poll(): void {
    if (!this.hasFresh || this.mapInFlight) return;

    this.hasFresh = false;
    this.mapInFlight = true;

    const n = Math.max(this.lastCount, 1);

    this.readback
      .mapAsync(GPUMapMode.READ, 0, STATS_BYTES)
      .then(() => {
        const w = new Uint32Array(
          this.readback.getMappedRange(0, STATS_BYTES).slice(0),
        );

        this.readback.unmap();
        this.mapInFlight = false;

        const surfaceCount = w[2];
        this.latest = {
          avgSpeed: w[0] / 256 / n,
          maxSpeed: w[5] / 256,
          avgFoam: w[1] / 1024 / n,
          surfaceFraction: surfaceCount / n,
          avgSurfaceSpeed: w[3] / 256 / Math.max(surfaceCount, 1),
          sprayCount: w[4],
        };
      })
      .catch((err) => {
        console.warn("[audio] fluid stats readback failed:", err);
        try {
          this.readback.unmap();
        } catch {}

        this.mapInFlight = false;
      });
  }
  public record(
    encoder: GPUCommandEncoder,
    numParticles: number,
    targetDensity: number,
  ): void {
    if (this.mapInFlight) return;
    if (++this.frame % this.sampleInterval !== 0) return;

    this.uniformU32[0] = numParticles;
    this.uniformF32[1] = targetDensity;
    this.uniformF32[2] = this.surfaceThreshold;
    this.uniformF32[3] = this.sprayThreshold;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

    encoder.clearBuffer(this.statsBuffer);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(numParticles / 256));
    pass.end();

    encoder.copyBufferToBuffer(
      this.statsBuffer,
      0,
      this.readback,
      0,
      STATS_BYTES,
    );

    this.lastCount = numParticles;
    this.hasFresh = true;
  }
}
