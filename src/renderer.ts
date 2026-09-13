import { config } from "./config";
import { GPUBufferUsage, GPUShaderStage, GPUTextureUsage } from "./types";

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

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) viewPos: vec3<f32>,
};

struct FragmentOutput {
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
  let uvOffset = quad[vertexIndex] * uniforms.particleScale;
  let worldPos = p.position.xyz 
               + uniforms.cameraRight.xyz * uvOffset.x 
               + uniforms.cameraUp.xyz * uvOffset.y;
  let viewPos = (uniforms.viewMatrix * vec4<f32>(worldPos, 1.0)).xyz;
  
  let speed = length(p.velocity.xyz);
  let t = (speed - uniforms.minSpeed) / max(uniforms.maxSpeed - uniforms.minSpeed, 0.001);

  var output: VertexOutput;
  output.position = uniforms.projectionMatrix * vec4<f32>(viewPos, 1.0);
  output.uv = quad[vertexIndex];
  output.color = speedToColor(t);
  output.viewPos = viewPos;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
  let distSqr = dot(input.uv, input.uv);
  if (distSqr > 1.0) { 
    discard; 
  }
  
  let sphereZ = sqrt(1.0 - distSqr) * uniforms.particleScale;
  let sphereViewPos = vec3<f32>(input.viewPos.x, input.viewPos.y, input.viewPos.z + sphereZ);
  let clipPos = uniforms.projectionMatrix * vec4<f32>(sphereViewPos, 1.0);
  let realDepth = clipPos.z / clipPos.w;

  let dist = sqrt(distSqr);
  let glow = 1.0 - dist * 0.35;

  var output: FragmentOutput;
  output.color = vec4<f32>(input.color * glow, 1.0);
  output.depth = realDepth;
  return output;
}
`;

export class GPUParticleRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private depthTexture!: GPUTexture;
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

    this.createDepthTexture(canvas.width, canvas.height);
    this.initPipeline(particlesBuffer);
  }
  public resize(width: number, height: number): void {
    if (this.depthTexture) this.depthTexture.destroy();
    this.createDepthTexture(width, height);
  }
  private createDepthTexture(width: number, height: number): void {
    this.depthTexture = this.device.createTexture({
      size: [Math.max(1, width), Math.max(1, height)],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  private initPipeline(particlesBuffer: GPUBuffer): void {
    const shaderModule = this.device.createShaderModule({
      code: renderShaderWGSL,
    });

    this.uniformBuffer = this.device.createBuffer({
      size: 256,
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
          },
        ],
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
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
    viewMatrix: Float32Array,
    projMatrix: Float32Array,
    cameraRight: [number, number, number],
    cameraUp: [number, number, number],
    particleCount: number,
  ): void {
    const uniformData = new Float32Array(64);
    uniformData.set(viewMatrix, 0);
    uniformData.set(projMatrix, 16);

    uniformData[32] = cameraRight[0];
    uniformData[33] = cameraRight[1];
    uniformData[34] = cameraRight[2];
    uniformData[35] = 0.0;

    uniformData[36] = cameraUp[0];
    uniformData[37] = cameraUp[1];
    uniformData[38] = cameraUp[2];
    uniformData[39] = 0.0;

    uniformData[40] = config.particleSize;
    uniformData[41] = config.minSpeed;
    uniformData[42] = config.maxSpeed;
    uniformData[43] = 0.0;

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
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.bindGroup);
    renderPass.draw(6, particleCount, 0, 0);
    renderPass.end();
  }
}
