import * as THREE from "three";

import { config } from "./config";
import { GPUBufferUsage, GPUShaderStage, GPUTextureUsage } from "./types";

const FAR_DEPTH = 1e6;
const MAX_BILATERAL_RADIUS_PX = 20;

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

struct FOut {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }

  let sphereZ = u.particleScale * sqrt(max(1.0 - d2, 0.0));
  let surfaceViewZ = in.viewZ - sphereZ;

  out.color = vec4<f32>(surfaceViewZ, 0.0, 0.0, 0.0);

  let viewSpaceZ = -surfaceViewZ;
  let clip = u.projectionMatrix * vec4<f32>(0.0, 0.0, viewSpaceZ, 1.0);
  out.depth = clip.z / clip.w;

  return out;
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
  maxRadiusPx: u32,
  sigmaWorld: f32,
  sigmaDepthWorld: f32,
  tanHalfFovY: f32,
  viewportHeight: f32,
  _pad0: f32,
  _pad1: f32,
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

  let worldPerPixel = (2.0 * params.tanHalfFovY * center) / params.viewportHeight;
  let sigmaPx = clamp(
    params.sigmaWorld / max(worldPerPixel, 1e-6),
    1.0,
    f32(params.maxRadiusPx),
  );
  let radiusPx = i32(min(ceil(sigmaPx * 2.5), f32(params.maxRadiusPx)));

  let twoSS = 2.0 * sigmaPx * sigmaPx;
  let twoSD = 2.0 * params.sigmaDepthWorld * params.sigmaDepthWorld;

  var sum = 0.0;
  var wsum = 0.0;

  for (var i = -radiusPx; i <= radiusPx; i++) {
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
  fluidColor: vec4<f32>,   // rgb = per-channel absorption coefficient
  params0: vec4<f32>,
  params1: vec4<f32>,
  params2: vec4<f32>,
  reflSky: vec4<f32>,
  reflHorizon: vec4<f32>,
  params3: vec4<f32>,      // x = splat world radius, yzw = deep-water scatter color
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

fn waveGradient(p: vec2<f32>, t: f32) -> vec2<f32> {
  var d = vec2<f32>(0.0);
  let a1 = 1.3; let b1 = 0.7;
  d += vec2<f32>(a1, b1) * cos(p.x * a1 + p.y * b1 + t * 0.9);
  let a2 = -0.9; let b2 = 1.4;
  d += vec2<f32>(a2, b2) * cos(p.x * a2 + p.y * b2 - t * 0.7) * 0.7;
  let a3 = 2.6; let b3 = -1.8;
  d += vec2<f32>(a3, b3) * cos(p.x * a3 + p.y * b3 + t * 1.6) * 0.35;
  let a4 = 3.3; let b4 = 2.9;
  d += vec2<f32>(a4, b4) * cos(p.x * a4 + p.y * b4 - t * 1.1) * 0.2;
  return d * 0.25;
}

fn sampleFakeEnv(ry: f32) -> vec3<f32> {
  let up = clamp(ry, 0.0, 1.0);
  let skyToHorizon = mix(u.reflHorizon.rgb, u.reflSky.rgb, up);
  let below = clamp(-ry, 0.0, 1.0);
  let groundCol = u.reflHorizon.rgb * 0.6;
  return mix(skyToHorizon, groundCol, below);
}

// Overlapping SPH particle splats stack along every view ray through the
// body of the fluid, so the accumulated "thickness" sampled below can be
// far larger than the fluid's real physical depth, and that path length
// itself grows a lot at steep viewing angles. Left unclamped, Beer-
// Lambert absorption on a value like that decays to ~0 for most viewing
// angles, which is why the pool used to read as solid black/opaque and
// hid anything submerged in it. Clamping keeps a believable floor on
// transmittance so deep water gets more tinted/murky instead of fully
// opaque, and submerged objects stay visible from any angle.
const MAX_OPTICAL_DEPTH: f32 = 3.0;

// Fresnel reflectance legitimately climbs toward a full mirror at
// grazing angles - that's real water behavior - but letting it reach
// ~1.0 means the reflection can fully replace the refracted image at
// some angles, hiding submerged objects no matter how shallow they are.
// Capping it guarantees a minimum amount of the refracted/underwater
// image always shows through, regardless of viewing angle.
const MAX_FRESNEL: f32 = 0.35;

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
  if (sceneRaw < 0.9999) {
    let sceneLinear = linearizeDepth(sceneRaw, near, far);
    if (sceneLinear < dC - 0.05) {
      return vec4<f32>(sceneCol, 1.0);
    }
  }

  let dL = textureLoad(fluidDepth, coord + vec2<i32>(-3, 0), 0).r;
  let dR = textureLoad(fluidDepth, coord + vec2<i32>( 3, 0), 0).r;
  let dU = textureLoad(fluidDepth, coord + vec2<i32>(0, -3), 0).r;
  let dD = textureLoad(fluidDepth, coord + vec2<i32>(0,  3), 0).r;

  let validL = dL < 1e5;
  let validR = dR < 1e5;
  let validU = dU < 1e5;
  let validD = dD < 1e5;
  let edgeMask = select(0.0, 1.0, validL && validR && validU && validD);

  let dLv = select(dC, dL, validL);
  let dRv = select(dC, dR, validR);
  let dUv = select(dC, dU, validU);
  let dDv = select(dC, dD, validD);

  let texel = vec2<f32>(3.0) / dimsF;
  let pC = reconstructViewPos(in.uv, dC);
  let pL = reconstructViewPos(in.uv - vec2<f32>(texel.x, 0.0), dLv);
  let pR = reconstructViewPos(in.uv + vec2<f32>(texel.x, 0.0), dRv);
  let pU = reconstructViewPos(in.uv - vec2<f32>(0.0, texel.y), dUv);
  let pD = reconstructViewPos(in.uv + vec2<f32>(0.0, texel.y), dDv);

  let dxFwd = pR - pC;
  let dxBwd = pC - pL;
  let ddx = select(dxFwd, dxBwd, abs(dxFwd.z) > abs(dxBwd.z));
  let dyFwd = pD - pC;
  let dyBwd = pC - pU;
  let ddy = select(dyFwd, dyBwd, abs(dyFwd.z) > abs(dyBwd.z));

  let worldUpInView = normalize(u.params2.yzw);
  var normalSmooth: vec3<f32>;
  if (length(ddx) < 1e-4 || length(ddy) < 1e-4) {
    normalSmooth = worldUpInView;
  } else {
    normalSmooth = normalize(cross(ddx, ddy));
    if (normalSmooth.z < 0.0) { normalSmooth = -normalSmooth; }
  }

  let pView = reconstructViewPos(in.uv, dC);
  let flatness = smoothstep(0.3, 0.7, dot(normalSmooth, worldUpInView));
  let wave = waveGradient(pView.xy, u.params2.x);
  var normalShade = normalize(normalSmooth + vec3<f32>(wave * 0.03 * flatness * edgeMask, 0.0));

  let thickness = textureLoad(fluidThickness, coord, 0).r;
  let viewDir = normalize(-reconstructViewPos(in.uv, dC));

  let cosTheta = clamp(dot(normalShade, viewDir), 0.0, 1.0);
  let rawFresnel = u.params1.w + (1.0 - u.params1.w) * pow(1.0 - cosTheta, 5.0);
  let silhouetteFade = smoothstep(0.3, 0.6, normalShade.z);
  // Real Fresnel reflectance does climb toward a full mirror at grazing
  // angles, which is why a straight-down look (near-normal incidence,
  // low reflectance) reads as more transparent than a shallow, off-axis
  // look (near-grazing incidence, high reflectance) in most physically
  // based water shaders. But letting it run all the way to ~1.0 means
  // the reflection can fully replace the refracted image at some angles
  // - nothing submerged is visible no matter how shallow it is. Capping
  // it keeps a believable sheen without ever fully hiding what's below.
  let fresnel = min(rawFresnel * silhouetteFade * edgeMask, MAX_FRESNEL);

  let refrStr = u.params0.z;
  let thicknessGate = smoothstep(2.0, 6.0, thickness);
  let sinTheta = sqrt(max(1.0 - normalSmooth.z * normalSmooth.z, 0.0));
  let refrUV = clamp(
    in.uv + normalSmooth.xy * refrStr * thicknessGate * sinTheta * edgeMask,
    vec2<f32>(0.0),
    vec2<f32>(1.0),
  );

  let refrCoordF = refrUV * dimsF;
  let refrCoord = vec2<i32>(clamp(vec2<i32>(refrCoordF), vec2<i32>(0), dimsI - vec2<i32>(1)));

  let refrThickness = textureLoad(fluidThickness, refrCoord, 0).r;
  let refrSceneRaw = textureLoad(sceneDepth, refrCoord, 0);
  let refrValid = refrThickness > 1.0 && refrSceneRaw < 0.9999;

  var refracted: vec3<f32>;
  if (refrValid) {
    refracted = textureSampleLevel(sceneColor, linSamp, refrUV, 0.0).rgb;
  } else {
    refracted = sceneCol;
  }

  // These used to be the same vector, reused for two different jobs:
  // (1) how fast each color channel gets absorbed with depth (physically,
  // red should fade fastest and blue slowest - that's why oceans look
  // blue), and (2) the color of fully opaque water. Job (2) was the bug:
  // fluidColor is (0.35, 0.15, 0.05) - mostly red, barely any blue - so
  // as soon as absorb dropped toward 0 the water settled on that dim
  // reddish-brown instead of a blue. absorptionCoeff keeps the physical
  // per-channel falloff (unchanged); scatterColor is the actual visible
  // color of deep/opaque water - the blue you want - and is independent
  // of the absorption tuning.
  let absorptionCoeff = u.fluidColor.rgb;
  let scatterColor = u.params3.yzw;
  let opticalDepth = min(sqrt(max(thickness, 0.0)) * u.params1.y, MAX_OPTICAL_DEPTH);
  let absorb = exp(-absorptionCoeff * opticalDepth);
  var col = mix(scatterColor, refracted, absorb);

  let R = reflect(-viewDir, normalShade);
  let worldRy = max(dot(R, worldUpInView), 0.0);
  let reflection = sampleFakeEnv(worldRy);
  col = mix(col, reflection, fresnel);

  let sunVS = normalize(u.sunDirView.xyz);
  let halfVS = normalize(sunVS + viewDir);
  let spec = pow(max(dot(normalShade, halfVS), 0.0), u.params1.z);
  col = mix(col, vec3<f32>(1.0, 0.98, 0.92), spec * 0.3 * edgeMask * silhouetteFade * silhouetteFade);

  let alpha = smoothstep(0.0, 1.0, thickness);
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
  private bilateralParamsH!: GPUBuffer;
  private bilateralParamsV!: GPUBuffer;
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

  private compLayout!: GPUBindGroupLayout;
  private bilatLayout!: GPUBindGroupLayout;

  private width = 1;
  private height = 1;

  public time = 0;

  // Visible color of fully opaque/deep water (the "in-scattered ambient
  // light" look). Independent of fluidColor's absorption coefficients
  // below - tweak this to change the water's overall blue tint without
  // touching how quickly it absorbs light with depth.
  public scatterColor: [number, number, number] = [0.04, 0.3, 0.5];

  private readonly tmpMat4 = new THREE.Matrix4();
  private readonly tmpVec3 = new THREE.Vector3();

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

    this.bilateralParamsH = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bilateralParamsV = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.compositeUniform = this.device.createBuffer({
      size: 128,
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
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
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
        { binding: 0, resource: this.fluidDepthTexture.createView() },
        { binding: 1, resource: this.fluidDepthTemp.createView() },
        { binding: 2, resource: { buffer: this.bilateralParamsH } },
      ],
    });

    this.bilateralVBind = this.device.createBindGroup({
      layout: this.bilatLayout,
      entries: [
        { binding: 0, resource: this.fluidDepthTemp.createView() },
        { binding: 1, resource: this.fluidDepthSmooth.createView() },
        { binding: 2, resource: { buffer: this.bilateralParamsV } },
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
    const splatWorldRadius = config.particleSize * config.renderSplatScale;

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
    pu[40] = splatWorldRadius;
    pu[41] = 0;
    pu[42] = 0;
    pu[43] = 0;
    this.device.queue.writeBuffer(this.particleUniform, 0, pu);

    // const sigmaWorld = splatWorldRadius;
    // const sigmaDepthWorld = splatWorldRadius;
    const sigmaWorld = splatWorldRadius * 1.5;
    const sigmaDepthWorld = splatWorldRadius * 6.0;

    const writeParams = (buffer: GPUBuffer, axis: 0 | 1) => {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab);
      const f = new Float32Array(ab);
      u[0] = axis;
      u[1] = MAX_BILATERAL_RADIUS_PX;
      f[2] = sigmaWorld;
      f[3] = sigmaDepthWorld;
      f[4] = tanHalfFovY;
      f[5] = this.height;
      f[6] = 0;
      f[7] = 0;
      this.device.queue.writeBuffer(buffer, 0, ab);
    };
    writeParams(this.bilateralParamsH, 0);
    writeParams(this.bilateralParamsV, 1);

    this.tmpMat4.fromArray(viewMatrix).invert();
    this.tmpVec3.set(0, 1, 0).transformDirection(this.tmpMat4);

    const [sx, sy, sz] = this.sunDirView(viewMatrix);
    const cu = new Float32Array(32);

    cu[0] = sx;
    cu[1] = sy;
    cu[2] = sz;
    cu[3] = 0;

    cu[4] = 0.35;
    cu[5] = 0.15;
    cu[6] = 0.05;
    cu[7] = 1.0;

    cu[8] = tanHalfFovY;
    cu[9] = aspect;
    cu[10] = 0.15;
    cu[11] = near;

    cu[12] = far;
    cu[13] = 0.15;
    cu[14] = 200.0;
    cu[15] = 0.06;

    cu[16] = this.time;
    cu[17] = this.tmpVec3.x;
    cu[18] = this.tmpVec3.y;
    cu[19] = this.tmpVec3.z;

    cu[20] = 0.55;
    cu[21] = 0.75;
    cu[22] = 0.95;
    cu[23] = 1.0;

    cu[24] = 0.2;
    cu[25] = 0.35;
    cu[26] = 0.55;
    cu[27] = 1.0;

    cu[28] = splatWorldRadius;
    cu[29] = this.scatterColor[0];
    cu[30] = this.scatterColor[1];
    cu[31] = this.scatterColor[2];

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
    pass.setPipeline(this.bilateralHPipeline);
    pass.setBindGroup(0, this.bilateralHBind);
    pass.dispatchWorkgroups(
      Math.ceil(this.width / 8),
      Math.ceil(this.height / 8),
    );
  }
  public encodeBilateralV(pass: GPUComputePassEncoder): void {
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
