import { config } from "./config";
import { GPUBufferUsage, GPUShaderStage, GPUTextureUsage } from "./types";

const FAR_DEPTH = 1e6;
const BILATERAL_RADIUS = 4;

const particleDepthWGSL = /* wgsl */ `
struct Particle {
  position: vec4<f32>,
  predictedPosition: vec4<f32>,
  velocity: vec4<f32>,
  density: vec4<f32>,
};

struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  particleScale: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) viewZ: f32,
};

@vertex
fn vs(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VOut {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let p = particles[ii];
  let uvOffset = quad[vi] * u.particleScale;
  let worldPos = p.position.xyz
               + u.cameraRight.xyz * uvOffset.x
               + u.cameraUp.xyz * uvOffset.y;
  let viewPos = (u.viewMatrix * vec4<f32>(worldPos, 1.0)).xyz;
  var out: VOut;
  out.clip = u.projectionMatrix * vec4<f32>(viewPos, 1.0);
  out.uv = quad[vi];
  out.viewZ = -viewPos.z;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }
  let r = u.particleScale;
  let sphereZ = r * (1.0 - sqrt(max(1.0 - d2, 0.0)));
  return vec4<f32>(in.viewZ - sphereZ, 0.0, 0.0, 0.0);
}
`;

const particleThicknessWGSL = /* wgsl */ `
struct Particle {
  position: vec4<f32>,
  predictedPosition: vec4<f32>,
  velocity: vec4<f32>,
  density: vec4<f32>,
};

struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  particleScale: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VOut {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let p = particles[ii];
  let uvOffset = quad[vi] * u.particleScale;
  let worldPos = p.position.xyz
               + u.cameraRight.xyz * uvOffset.x
               + u.cameraUp.xyz * uvOffset.y;
  let viewPos = (u.viewMatrix * vec4<f32>(worldPos, 1.0)).xyz;
  var out: VOut;
  out.clip = u.projectionMatrix * vec4<f32>(viewPos, 1.0);
  out.uv = quad[vi];
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }
  let chord = 2.0 * u.particleScale * sqrt(max(1.0 - d2, 0.0));
  return vec4<f32>(chord, 0.0, 0.0, 0.0);
}
`;

const bilateralWGSL = /* wgsl */ `
struct Params {
  axis: u32,
  radius: u32,
  sigmaSpatial: f32,
  sigmaDepth: f32,
};

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(inputTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let center = textureLoad(inputTex, coord, 0).r;

  if (center >= 1e5) {
    textureStore(outputTex, coord, vec4<f32>(center, 0.0, 0.0, 0.0));
    return;
  }

  var sum = 0.0;
  var wsum = 0.0;
  let radius = i32(params.radius);
  let twoSS = 2.0 * params.sigmaSpatial * params.sigmaSpatial;
  let twoSD = 2.0 * params.sigmaDepth * params.sigmaDepth;

  for (var i = -radius; i <= radius; i++) {
    var offset: vec2<i32>;
    if (params.axis == 0u) {
      offset = vec2<i32>(i, 0);
    } else {
      offset = vec2<i32>(0, i);
    }

    let sc = coord + offset;
    if (sc.x < 0 || sc.y < 0 || sc.x >= i32(dims.x) || sc.y >= i32(dims.y)) {
      continue;
    }

    let s = textureLoad(inputTex, sc, 0).r;
    if (s >= 1e5) { continue; }

    let fi = f32(i);
    let dd = s - center;
    let w = exp(-(fi * fi) / twoSS) * exp(-(dd * dd) / twoSD);

    sum += s * w;
    wsum += w;
  }

  let result = select(center, sum / wsum, wsum > 1e-5);
  textureStore(outputTex, coord, vec4<f32>(result, 0.0, 0.0, 0.0));
}
`;

const compositeWGSL = /* wgsl */ `
struct Uniforms {
  sunDirView: vec4<f32>,
  fluidColor: vec4<f32>,
  
  params0: vec4<f32>,

  params1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sceneColor: texture_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_2d;
@group(0) @binding(3) var fluidDepth: texture_2d<f32>;
@group(0) @binding(4) var fluidThickness: texture_2d<f32>;
@group(0) @binding(5) var linSamp: sampler;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: VOut;
  out.pos = vec4<f32>(p[vi], 0.0, 1.0);
  out.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 1.0 - (p[vi].y * 0.5 + 0.5));
  return out;
}

fn linearizeDepth(z: f32, near: f32, far: f32) -> f32 {
  return (near * far) / (far - z * (far - near));
}

fn reconstructViewPos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
  let th = u.params0.x;
  let aspect = u.params0.y;
  let x = (uv.x * 2.0 - 1.0) * aspect * th * depth;
  let y = (1.0 - uv.y * 2.0) * th * depth;
  return vec3<f32>(x, y, -depth);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let dimsI = vec2<i32>(textureDimensions(fluidDepth));
  let dimsF = vec2<f32>(dimsI);
  let coordF = in.uv * dimsF;
  let coord = vec2<i32>(clamp(vec2<i32>(coordF), vec2<i32>(0), dimsI - vec2<i32>(1)));

  let sceneCol = textureSampleLevel(sceneColor, linSamp, in.uv, 0.0).rgb;
  let dC = textureLoad(fluidDepth, coord, 0).r;

  if (dC >= 1e5) {
    return vec4<f32>(sceneCol, 1.0);
  }

  
  let near = u.params0.w;
  let far = u.params1.x;
  let sceneRaw = textureLoad(sceneDepth, coord, 0);
  let sceneLinear = linearizeDepth(sceneRaw, near, far);
  if (sceneRaw < 1.0 && dC > sceneLinear + 0.05) {
    return vec4<f32>(sceneCol, 1.0);
  }

  let dL = textureLoad(fluidDepth, coord + vec2<i32>(-1, 0), 0).r;
  let dR = textureLoad(fluidDepth, coord + vec2<i32>( 1, 0), 0).r;
  let dU = textureLoad(fluidDepth, coord + vec2<i32>(0, -1), 0).r;
  let dD = textureLoad(fluidDepth, coord + vec2<i32>(0,  1), 0).r;

  let dLv = select(dC, dL, dL < 1e5);
  let dRv = select(dC, dR, dR < 1e5);
  let dUv = select(dC, dU, dU < 1e5);
  let dDv = select(dC, dD, dD < 1e5);

  let texel = vec2<f32>(1.0) / dimsF;
  let pL = reconstructViewPos(in.uv - vec2<f32>(texel.x, 0.0), dLv);
  let pR = reconstructViewPos(in.uv + vec2<f32>(texel.x, 0.0), dRv);
  let pU = reconstructViewPos(in.uv - vec2<f32>(0.0, texel.y), dUv);
  let pD = reconstructViewPos(in.uv + vec2<f32>(0.0, texel.y), dDv);

  var normalVS = normalize(cross(pR - pL, pD - pU));
  if (normalVS.z > 0.0) { normalVS = -normalVS; }

  let thickness = textureLoad(fluidThickness, coord, 0).r;
  let viewDir = normalize(-reconstructViewPos(in.uv, dC));

  let cosTheta = clamp(dot(normalVS, viewDir), 0.0, 1.0);
  let fresnel = u.params1.w + (1.0 - u.params1.w) * pow(1.0 - cosTheta, 5.0);
  
  let refrStr = u.params0.z;
  let tFactor = smoothstep(0.0, 1.5, thickness);
  let refrUV = clamp(
    in.uv + normalVS.xy * refrStr * tFactor,
    vec2<f32>(0.0),
    vec2<f32>(1.0),
  );
  let refracted = textureSampleLevel(sceneColor, linSamp, refrUV, 0.0).rgb;

  let fluidCol = u.fluidColor.rgb;
  let absorb = exp(-fluidCol * thickness * u.params1.y);
  var col = refracted * absorb + fluidCol * (1.0 - absorb);

  let skyCol = vec3<f32>(0.30, 0.45, 0.65);
  col = mix(col, skyCol, fresnel * 0.5);
  
  let sunVS = normalize(u.sunDirView.xyz);
  let halfVS = normalize(sunVS + viewDir);
  let spec = pow(max(dot(normalVS, halfVS), 0.0), u.params1.z);
  col += vec3<f32>(1.0, 0.98, 0.92) * spec * 1.5;

  let alpha = smoothstep(0.0, 0.10, thickness);
  col = mix(sceneCol, col, alpha);

  return vec4<f32>(col, 1.0);
}
`;

export class SSFRRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private depthPipeline!: GPURenderPipeline;
  private thicknessPipeline!: GPURenderPipeline;
  private bilateralHPipeline!: GPUComputePipeline;
  private bilateralVPipeline!: GPUComputePipeline;
  private compositePipeline!: GPURenderPipeline;

  private particleUniform!: GPUBuffer;
  private bilateralParams!: GPUBuffer;
  private compositeUniform!: GPUBuffer;

  private depthBindGroup!: GPUBindGroup;
  private thicknessBindGroup!: GPUBindGroup;
  private bilateralHBind!: GPUBindGroup;
  private bilateralVBind!: GPUBindGroup;
  private compositeBindGroup!: GPUBindGroup;

  private sceneColorTexture!: GPUTexture;
  private sceneDepthTexture!: GPUTexture;
  private fluidDepthTexture!: GPUTexture;
  private fluidDepthStencil!: GPUTexture;
  private fluidDepthTemp!: GPUTexture;
  private fluidDepthSmooth!: GPUTexture;
  private fluidThicknessTexture!: GPUTexture;

  private linearSampler!: GPUSampler;

  private width = 1;
  private height = 1;
  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    particlesBuffer: GPUBuffer,
  ) {
    this.device = device;
    this.format = format;

    this.createSamplers();
    this.createUniformBuffers();
    this.createPipelines(particlesBuffer);
  }
  private createSamplers(): void {
    this.linearSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
  }
  private createUniformBuffers(): void {
    this.particleUniform = this.device.createBuffer({
      size: 176,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.bilateralParams = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.compositeUniform = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  private createPipelines(particlesBuffer: GPUBuffer): void {
    const depthMod = this.device.createShaderModule({
      code: particleDepthWGSL,
    });
    const thickMod = this.device.createShaderModule({
      code: particleThicknessWGSL,
    });
    const bilatMod = this.device.createShaderModule({ code: bilateralWGSL });
    const compMod = this.device.createShaderModule({ code: compositeWGSL });

    const particleLayout = this.device.createBindGroupLayout({
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

    this.depthBindGroup = this.device.createBindGroup({
      layout: particleLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particleUniform } },
        { binding: 1, resource: { buffer: particlesBuffer } },
      ],
    });

    this.thicknessBindGroup = this.device.createBindGroup({
      layout: particleLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particleUniform } },
        { binding: 1, resource: { buffer: particlesBuffer } },
      ],
    });

    this.depthPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [particleLayout],
      }),
      vertex: { module: depthMod, entryPoint: "vs" },
      fragment: {
        module: depthMod,
        entryPoint: "fs",
        targets: [{ format: "r32float" }],
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      primitive: { topology: "triangle-list" },
    });

    this.thicknessPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [particleLayout],
      }),
      vertex: { module: thickMod, entryPoint: "vs" },
      fragment: {
        module: thickMod,
        entryPoint: "fs",
        targets: [
          {
            format: "r16float",
            blend: {
              color: {
                srcFactor: "one",
                dstFactor: "one",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    const bilatLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "r32float" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.bilateralHPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bilatLayout],
      }),
      compute: { module: bilatMod, entryPoint: "main" },
    });
    this.bilateralVPipeline = this.bilateralHPipeline;

    const compLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" },
        },
      ],
    });

    this.compositePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [compLayout],
      }),
      vertex: { module: compMod, entryPoint: "vs" },
      fragment: {
        module: compMod,
        entryPoint: "fs",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.compLayout = compLayout;
    this.bilatLayout = bilatLayout;
  }

  private compLayout!: GPUBindGroupLayout;
  private bilatLayout!: GPUBindGroupLayout;
  public resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);

    this.sceneColorTexture?.destroy();
    this.sceneDepthTexture?.destroy();
    this.fluidDepthTexture?.destroy();
    this.fluidDepthStencil?.destroy();
    this.fluidDepthTemp?.destroy();
    this.fluidDepthSmooth?.destroy();
    this.fluidThicknessTexture?.destroy();

    const size = [this.width, this.height];

    this.sceneColorTexture = this.device.createTexture({
      size,
      format: this.format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.sceneDepthTexture = this.device.createTexture({
      size,
      format: "depth32float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.fluidDepthTexture = this.device.createTexture({
      size,
      format: "r32float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.fluidDepthStencil = this.device.createTexture({
      size,
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.fluidDepthTemp = this.device.createTexture({
      size,
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.fluidDepthSmooth = this.device.createTexture({
      size,
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.fluidThicknessTexture = this.device.createTexture({
      size,
      format: "r16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.bilateralHBind = this.device.createBindGroup({
      layout: this.bilatLayout,
      entries: [
        {
          binding: 0,
          resource: this.fluidDepthTexture.createView(),
        },
        {
          binding: 1,
          resource: this.fluidDepthTemp.createView(),
        },
        { binding: 2, resource: { buffer: this.bilateralParams } },
      ],
    });

    this.bilateralVBind = this.device.createBindGroup({
      layout: this.bilatLayout,
      entries: [
        {
          binding: 0,
          resource: this.fluidDepthTemp.createView(),
        },
        {
          binding: 1,
          resource: this.fluidDepthSmooth.createView(),
        },
        { binding: 2, resource: { buffer: this.bilateralParams } },
      ],
    });

    this.compositeBindGroup = this.device.createBindGroup({
      layout: this.compLayout,
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniform } },
        { binding: 1, resource: this.sceneColorTexture.createView() },
        { binding: 2, resource: this.sceneDepthTexture.createView() },
        { binding: 3, resource: this.fluidDepthSmooth.createView() },
        { binding: 4, resource: this.fluidThicknessTexture.createView() },
        { binding: 5, resource: this.linearSampler },
      ],
    });
  }
  public getSceneColorView(): GPUTextureView {
    return this.sceneColorTexture.createView();
  }
  public getSceneDepthView(): GPUTextureView {
    return this.sceneDepthTexture.createView();
  }
  public getFluidDepthView(): GPUTextureView {
    return this.fluidDepthTexture.createView();
  }
  public getFluidDepthStencilView(): GPUTextureView {
    return this.fluidDepthStencil.createView();
  }
  public getFluidThicknessView(): GPUTextureView {
    return this.fluidThicknessTexture.createView();
  }
  public getFarDepth(): number {
    return FAR_DEPTH;
  }
  public updateFrame(
    viewMatrix: Float32Array,
    projectionMatrix: Float32Array,
    cameraRight: [number, number, number],
    cameraUp: [number, number, number],
    near: number,
    far: number,
    tanHalfFovY: number,
    aspect: number,
  ): void {
    const pu = new Float32Array(44);
    pu.set(viewMatrix, 0);
    pu.set(projectionMatrix, 16);
    pu[32] = cameraRight[0];
    pu[33] = cameraRight[1];
    pu[34] = cameraRight[2];
    pu[35] = 0;
    pu[36] = cameraUp[0];
    pu[37] = cameraUp[1];
    pu[38] = cameraUp[2];
    pu[39] = 0;

    pu[40] = config.particleSize * 1.4;
    pu[41] = 0;
    pu[42] = 0;
    pu[43] = 0;

    this.device.queue.writeBuffer(this.particleUniform, 0, pu);

    const [sx, sy, sz] = this.sunDirView(viewMatrix);
    const cu = new Float32Array(16);
    cu[0] = sx;
    cu[1] = sy;
    cu[2] = sz;
    cu[3] = 0;
    cu[4] = 0.1;
    cu[5] = 0.35;
    cu[6] = 0.45;
    cu[7] = 1.0;
    cu[8] = tanHalfFovY;
    cu[9] = aspect;
    cu[10] = 0.06;
    cu[11] = near;
    cu[12] = far;
    cu[13] = 2.5;
    cu[14] = 100.0;
    cu[15] = 0.02;

    this.device.queue.writeBuffer(this.compositeUniform, 0, cu);
  }
  private sunDirView(viewMatrix: Float32Array): [number, number, number] {
    const sx = 0.45,
      sy = 1.0,
      sz = 0.35;
    const len = Math.hypot(sx, sy, sz);
    const wx = sx / len,
      wy = sy / len,
      wz = sz / len;
    const vx = viewMatrix[0] * wx + viewMatrix[4] * wy + viewMatrix[8] * wz;
    const vy = viewMatrix[1] * wx + viewMatrix[5] * wy + viewMatrix[9] * wz;
    const vz = viewMatrix[2] * wx + viewMatrix[6] * wy + viewMatrix[10] * wz;
    return [vx, vy, vz];
  }
  public encodeParticleDepth(
    pass: GPURenderPassEncoder,
    particleCount: number,
  ): void {
    pass.setPipeline(this.depthPipeline);
    pass.setBindGroup(0, this.depthBindGroup);
    pass.draw(6, particleCount, 0, 0);
  }
  public encodeParticleThickness(
    pass: GPURenderPassEncoder,
    particleCount: number,
  ): void {
    pass.setPipeline(this.thicknessPipeline);
    pass.setBindGroup(0, this.thicknessBindGroup);
    pass.draw(6, particleCount, 0, 0);
  }
  public encodeBilateralH(pass: GPUComputePassEncoder): void {
    const params = new ArrayBuffer(16);
    const u = new Uint32Array(params);
    const f = new Float32Array(params);
    u[0] = 0;
    u[1] = BILATERAL_RADIUS;
    f[2] = 3.0;
    f[3] = 0.5;
    this.device.queue.writeBuffer(this.bilateralParams, 0, params);

    pass.setPipeline(this.bilateralHPipeline);
    pass.setBindGroup(0, this.bilateralHBind);
    pass.dispatchWorkgroups(
      Math.ceil(this.width / 8),
      Math.ceil(this.height / 8),
    );
  }
  public encodeBilateralV(pass: GPUComputePassEncoder): void {
    const params = new ArrayBuffer(16);
    const u = new Uint32Array(params);
    const f = new Float32Array(params);
    u[0] = 1;
    u[1] = BILATERAL_RADIUS;
    f[2] = 3.0;
    f[3] = 0.5;
    this.device.queue.writeBuffer(this.bilateralParams, 0, params);

    pass.setPipeline(this.bilateralVPipeline);
    pass.setBindGroup(0, this.bilateralVBind);
    pass.dispatchWorkgroups(
      Math.ceil(this.width / 8),
      Math.ceil(this.height / 8),
    );
  }
  public encodeComposite(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, this.compositeBindGroup);
    pass.draw(3);
  }
}
