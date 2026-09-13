import { config } from "./config";
import { acesFilmicWGSL } from "./shader/common.wgsl";
import { GPUBufferUsage, GPUShaderStage } from "./types";

const renderShaderWGSL = /* wgsl */ `
struct Particle {
  position: vec4<f32>,
  predictedPosition: vec4<f32>,
  velocity: vec4<f32>,
  density: vec4<f32>,
};

struct RenderUniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  particleScale: f32,
  minSpeed: f32,
  maxSpeed: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) viewPos: vec3<f32>,
};

struct FOut {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

fn speedToColor(tRaw: f32) -> vec3<f32> {
  let t = smoothstep(0.0, 1.0, clamp(tRaw, 0.0, 1.0));
  let c0 = vec3<f32>(0.05, 0.20, 0.85);
  let c1 = vec3<f32>(0.10, 0.70, 0.95);
  let c2 = vec3<f32>(0.15, 0.95, 0.60);
  let c3 = vec3<f32>(1.00, 0.65, 0.10);
  let c4 = vec3<f32>(0.95, 0.10, 0.10);
  if (t < 0.25) { return mix(c0, c1, t / 0.25); }
  if (t < 0.50) { return mix(c1, c2, (t - 0.25) / 0.25); }
  if (t < 0.75) { return mix(c2, c3, (t - 0.50) / 0.25); }
  return mix(c3, c4, (t - 0.75) / 0.25);
}

${acesFilmicWGSL}

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VOut {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
  );

  let p = particles[instanceIndex];
  let uvOffset = quad[vertexIndex] * uniforms.particleScale;
  let worldPos = p.position.xyz
               + uniforms.cameraRight.xyz * uvOffset.x
               + uniforms.cameraUp.xyz * uvOffset.y;
  let viewPos = (uniforms.viewMatrix * vec4<f32>(worldPos, 1.0)).xyz;

  let speed = length(p.velocity.xyz);
  let t = (speed - uniforms.minSpeed) / max(uniforms.maxSpeed - uniforms.minSpeed, 0.001);

  var out: VOut;
  out.position = uniforms.projectionMatrix * vec4<f32>(viewPos, 1.0);
  out.uv = quad[vertexIndex];
  out.color = speedToColor(t);
  out.viewPos = viewPos;
  return out;
}

@fragment
fn fs_main(input: VOut) -> FOut {
  let distSqr = dot(input.uv, input.uv);
  if (distSqr > 1.0) { discard; }

  let sphereZ = sqrt(1.0 - distSqr) * uniforms.particleScale;
  let sphereViewPos = vec3<f32>(input.viewPos.x, input.viewPos.y, input.viewPos.z + sphereZ);
  let clip = uniforms.projectionMatrix * vec4<f32>(sphereViewPos, 1.0);

  let dist = sqrt(distSqr);
  let glow = 1.0 - dist * 0.35;
  let shaded = acesFilmic(input.color * glow * 1.4);

  var out: FOut;
  out.color = vec4<f32>(shaded, 1.0);
  out.depth = clip.z / clip.w;
  return out;
}
`;

export class GPUParticleRenderer {
  private device: GPUDevice;
  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  constructor(
    device: GPUDevice,
    particlesBuffer: GPUBuffer,
    format: GPUTextureFormat,
  ) {
    this.device = device;

    const mod = this.device.createShaderModule({ code: renderShaderWGSL });
    const layout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    this.uniformBuffer = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.bindGroup = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: particlesBuffer } },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module: mod, entryPoint: "vs_main" },
      fragment: { module: mod, entryPoint: "fs_main", targets: [{ format }] },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
        depthBias: 1000,
        depthBiasSlopeScale: 2.0,
        depthBiasClamp: 0,
      },
      multisample: { count: 4 },
      primitive: { topology: "triangle-list" },
    });
  }
  public encode(
    pass: GPURenderPassEncoder,
    viewMatrix: Float32Array,
    projMatrix: Float32Array,
    cameraRight: [number, number, number],
    cameraUp: [number, number, number],
    particleCount: number,
  ): void {
    const data = new Float32Array(44);
    data.set(viewMatrix, 0);
    data.set(projMatrix, 16);

    data[32] = cameraRight[0];
    data[33] = cameraRight[1];
    data[34] = cameraRight[2];
    data[35] = 0;
    data[36] = cameraUp[0];
    data[37] = cameraUp[1];
    data[38] = cameraUp[2];
    data[39] = 0;
    data[40] = config.particleSize * 0.92;
    data[41] = config.minSpeed;
    data[42] = config.maxSpeed;
    data[43] = 0;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, particleCount, 0, 0);
  }
}
