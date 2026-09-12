import { config } from "./config";
import { GPUBufferUsage, GPUShaderStage } from "./types";

const renderShaderWGSL = /* wgsl */ `
struct Particle {
  position: vec2<f32>,
  predictedPosition: vec2<f32>,
  velocity: vec2<f32>,
  density: vec2<f32>,
};

struct RenderUniforms {
  viewProjectionMatrix: mat4x4<f32>,
  particleScale: f32,
  minSpeed: f32,
  maxSpeed: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
};

fn speedToColor(tRaw: f32) -> vec3<f32> {
  let t = clamp(tRaw, 0.0, 1.0);
  let c0 = vec3<f32>(0.05, 0.18, 0.85); 
  let c1 = vec3<f32>(0.10, 0.75, 0.95); 
  let c2 = vec3<f32>(0.15, 0.95, 0.55); 
  let c3 = vec3<f32>(1.00, 0.60, 0.10); 
  let c4 = vec3<f32>(0.90, 0.05, 0.05);

  if (t < 0.25) { return mix(c0, c1, t / 0.25); }
  if (t < 0.50) { return mix(c1, c2, (t - 0.25) / 0.25); }
  if (t < 0.75) { return mix(c2, c3, (t - 0.50) / 0.25); }
  return mix(c3, c4, (t - 0.75) / 0.25);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
  );

  let p = particles[instanceIndex];
  let quadOffset = quad[vertexIndex] * uniforms.particleScale;
  let worldPos = vec4<f32>(p.position + quadOffset, 0.0, 1.0);

  let speed = length(p.velocity);
  let t = (speed - uniforms.minSpeed) / max(uniforms.maxSpeed - uniforms.minSpeed, 0.001);

  var output: VertexOutput;
  output.position = uniforms.viewProjectionMatrix * worldPos;
  output.uv = quad[vertexIndex];
  output.color = speedToColor(t);
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let dist = length(input.uv);
  let aa = fwidth(dist);
  let alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);
  if (alpha <= 0.001) { discard; }

  let glow = 1.0 - dist * 0.35;
  return vec4<f32>(input.color * glow, alpha * 0.95);
}
`;

export class GPUParticleRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    particlesBuffer: GPUBuffer,
  ) {
    this.device = device;
    this.context = canvas.getContext("webgpu") as GPUCanvasContext;
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "premultiplied",
    });

    this.initPipeline(particlesBuffer);
  }
  private initPipeline(particlesBuffer: GPUBuffer): void {
    const shaderModule = this.device.createShaderModule({
      code: renderShaderWGSL,
    });

    this.uniformBuffer = this.device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
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

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: particlesBuffer } },
      ],
    });
  }
  public render(
    commandEncoder: GPUCommandEncoder,
    viewProjMatrix: Float32Array,
    particleCount: number,
  ): void {
    const uniformData = new Float32Array(20);
    uniformData.set(viewProjMatrix, 0);
    uniformData[16] = config.particleSize;
    uniformData[17] = config.minSpeed;
    uniformData[18] = config.maxSpeed;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.bindGroup);
    renderPass.draw(6, particleCount, 0, 0);
    renderPass.end();
  }
}
