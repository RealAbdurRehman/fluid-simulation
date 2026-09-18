import * as THREE from "three";

import { config } from "./config";
import { GPUBufferUsage, GPUShaderStage, GPUTextureUsage } from "./types";

const FAR_DEPTH = 1e6;
const LIGHT_MAP_SIZE = 1024;

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
  @location(2) worldPos: vec3<f32>,
  @location(3) foam: f32,
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
  out.worldPos = worldPos;
  out.foam = p.density.z;
  return out;
}

struct FOut {
  @location(0) viewZ: vec4<f32>,
  @location(1) world: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }

  let sphereZ = u.particleScale * sqrt(max(1.0 - d2, 0.0));
  let surfaceViewZ = in.viewZ - sphereZ;

  out.viewZ = vec4<f32>(surfaceViewZ, 0.0, 0.0, 0.0);
  out.world = vec4<f32>(in.worldPos, in.foam);

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

const lightDepthWGSL = /* wgsl */ `
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
  @location(0) viewZ: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  let d2 = dot(in.uv, in.uv);
  if (d2 > 1.0) { discard; }
  let sphereZ = u.particleScale * sqrt(max(1.0 - d2, 0.0));
  let surfaceViewZ = in.viewZ - sphereZ;
  out.viewZ = vec4<f32>(surfaceViewZ, 0.0, 0.0, 0.0);
  let viewSpaceZ = -surfaceViewZ;
  let clip = u.projectionMatrix * vec4<f32>(0.0, 0.0, viewSpaceZ, 1.0);
  out.depth = clip.z / clip.w;
  return out;
}
`;

const causticsWGSL = /* wgsl */ `
struct CausticParams {
  time: f32,
  intensity: f32,
  patternScale: f32,
  _pad0: f32,
};

@group(0) @binding(0) var fluidLightDepth: texture_2d<f32>;
@group(0) @binding(1) var causticOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: CausticParams;

fn hash21(p: vec2<f32>) -> f32 {
  var q = fract(p * vec2<f32>(127.1, 311.7));
  q += dot(q, q + 34.23);
  return fract(q.x * q.y);
}

fn noise2(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2<f32>(1.0, 0.0));
  let c = hash21(i + vec2<f32>(0.0, 1.0));
  let d = hash21(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p0: vec2<f32>) -> f32 {
  var v = 0.0;
  var a = 0.5;
  var p = p0;
  for (var i = 0; i < 3; i = i + 1) {
    v += a * noise2(p);
    p = p * 2.07 + vec2<f32>(1.7, 9.2);
    a *= 0.5;
  }
  return v;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(fluidLightDepth);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));

  let dC = textureLoad(fluidLightDepth, coord, 0).r;

  if (dC >= 1e5) {
    textureStore(causticOut, coord, vec4<f32>(1.0, 1e6, 0.0, 0.0));
    return;
  }

  let dimsI = vec2<i32>(dims);
  let maxIdx = dimsI - vec2<i32>(1);
  let cL = clamp(coord + vec2<i32>(-4, 0), vec2<i32>(0), maxIdx);
  let cR = clamp(coord + vec2<i32>( 4, 0), vec2<i32>(0), maxIdx);
  let cU = clamp(coord + vec2<i32>(0, -4), vec2<i32>(0), maxIdx);
  let cD = clamp(coord + vec2<i32>(0,  4), vec2<i32>(0), maxIdx);

  let dL = textureLoad(fluidLightDepth, cL, 0).r;
  let dR = textureLoad(fluidLightDepth, cR, 0).r;
  let dU = textureLoad(fluidLightDepth, cU, 0).r;
  let dD = textureLoad(fluidLightDepth, cD, 0).r;

  let validL = dL < 1e5;
  let validR = dR < 1e5;
  let validU = dU < 1e5;
  let validD = dD < 1e5;

  let dx = select(0.0, dR - dL, validL && validR);
  let dy = select(0.0, dD - dU, validU && validD);

  let scale = params.patternScale;
  let time = params.time;

  let uvBase = vec2<f32>(f32(coord.x), f32(coord.y));

  let warpX = fbm2(uvBase * 0.006 * scale + vec2<f32>(time * 0.21, time * 0.13));
  let warpY = fbm2(uvBase * 0.006 * scale + vec2<f32>(-time * 0.17, time * 0.29) + vec2<f32>(113.0, 71.0));

  let uvWarp = uvBase * 0.03 * scale + vec2<f32>(warpX, warpY) * 2.5;
  let depthWarp = vec2<f32>(dx, dy) * 0.35;
  let warped = uvWarp + depthWarp;

  let n1 = fbm2(warped + vec2<f32>(time * 0.5, 0.0));
  let n2 = fbm2(warped * 1.17 + vec2<f32>(0.0, time * 0.4) + vec2<f32>(37.0, 19.0));

  let r1 = abs(sin(n1 * 12.566));
  let r2 = abs(sin(n2 * 12.566));
  let ridged = 1.0 - min(r1, r2);

  let caustic = pow(ridged, 5.0);
  let slope = sqrt(dx * dx + dy * dy);
  let slopeBoost = 1.0 + min(slope * 2.0, 1.5);

  let finalVal = clamp(1.0 + caustic * params.intensity * slopeBoost, 1.0, 3.5);
  textureStore(causticOut, coord, vec4<f32>(finalVal, dC, 0.0, 0.0));
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
  fluidColor: vec4<f32>,
  params0: vec4<f32>,
  params1: vec4<f32>,
  params2: vec4<f32>,
  reflSky: vec4<f32>,
  reflHorizon: vec4<f32>,
  params3: vec4<f32>,
  foamParams: vec4<f32>,
  water0: vec4<f32>,
  water1: vec4<f32>,
  water2: vec4<f32>,
  foamColor: vec4<f32>,
  specColor: vec4<f32>,
  invView: mat4x4<f32>,
};

struct LightShadowU {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  absorb: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sceneColor: texture_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_2d;
@group(0) @binding(3) var fluidDepth: texture_2d<f32>;
@group(0) @binding(4) var fluidThickness: texture_2d<f32>;
@group(0) @binding(5) var linSamp: sampler;
@group(0) @binding(6) var fluidWorldPos: texture_2d<f32>;
@group(1) @binding(0) var<uniform> lightU: LightShadowU;
@group(1) @binding(1) var lightDepthTex: texture_2d<f32>;
@group(1) @binding(2) var lightThicknessTex: texture_2d<f32>;

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

fn viewToWorld(vp: vec3<f32>) -> vec3<f32> {
  return (u.invView * vec4<f32>(vp, 1.0)).xyz;
}

fn sampleFoamDilated(coord: vec2<i32>, dims: vec2<i32>) -> f32 {
  var best = 0.0;
  for (var dy = -2; dy <= 2; dy = dy + 1) {
    for (var dx = -2; dx <= 2; dx = dx + 1) {
      let c = clamp(coord + vec2<i32>(dx, dy),
                    vec2<i32>(0), dims - vec2<i32>(1));
      best = max(best, textureLoad(fluidWorldPos, c, 0).w);
    }
  }
  return best;
}

fn projectToUV(viewPos: vec3<f32>) -> vec3<f32> {
  let th = u.params0.x;
  let aspect = u.params0.y;
  let depth = -viewPos.z;
  let safeDepth = max(depth, 1e-4);
  let ndcX = viewPos.x / (aspect * th * safeDepth);
  let ndcY = viewPos.y / (th * safeDepth);
  return vec3<f32>(ndcX * 0.5 + 0.5, 1.0 - (ndcY * 0.5 + 0.5), depth);
}

fn computeFluidShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(vec3<f32>(0.45, 1.0, 0.35));
  let nDotL = max(dot(normal, lightDir), 0.0);

  let normalOffset = 0.06 + 0.18 * (1.0 - nDotL);
  let offsetPos = worldPos + normal * normalOffset;

  let lp = lightU.viewProj * vec4<f32>(offsetPos, 1.0);
  let w = max(abs(lp.w), 1e-6);
  let ndc = lp.xyz / w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return vec3<f32>(1.0);
  }

  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let dims = vec2<f32>(textureDimensions(lightDepthTex));
  let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));

  let mapZ = textureLoad(lightDepthTex, px, 0).r;
  let lv = lightU.view * vec4<f32>(offsetPos, 1.0);
  let recZ = -lv.z;

  let slopeBias = 0.10 + 0.85 * (1.0 - nDotL) * (1.0 - nDotL);
  let diff = recZ - mapZ;
  let shadowAmt = smoothstep(slopeBias, slopeBias + 0.45, diff);

  let rawThickness = textureLoad(lightThicknessTex, px, 0).r;
  let thickness = min(rawThickness, 3.0);
  let absorb = exp(-lightU.absorb.rgb * thickness);

  let opaqueAmount = 1.0 - smoothstep(0.0, 0.5, thickness);
  let occluderColor = mix(absorb, vec3<f32>(0.0), opaqueAmount);

  let tinted = mix(vec3<f32>(1.0), occluderColor, shadowAmt);
  return max(tinted, vec3<f32>(0.40));
}

fn hash31(p: vec3<f32>) -> f32 {
  var q = fract(p * vec3<f32>(127.1, 311.7, 74.7));
  q += dot(q, q.yzx + 34.23);
  return fract(q.x * q.y * q.z);
}

fn valueNoise3(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  let c000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
  let c100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
  let c010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
  let c110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
  let c001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
  let c101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
  let c011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
  let c111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

  let c00 = mix(c000, c100, u.x);
  let c10 = mix(c010, c110, u.x);
  let c01 = mix(c001, c101, u.x);
  let c11 = mix(c011, c111, u.x);

  return mix(mix(c00, c10, u.y), mix(c01, c11, u.y), u.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
  var v = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < 3; i = i + 1) {
    v += a * valueNoise3(q);
    q = q * 2.07 + vec3<f32>(1.7, 9.2, 3.3);
    a *= 0.5;
  }
  return v;
}

fn sampleFakeEnv(ry: f32) -> vec3<f32> {
  let up = clamp(ry, 0.0, 1.0);
  let skyToHorizon = mix(u.reflHorizon.rgb, u.reflSky.rgb, up);
  let below = clamp(-ry, 0.0, 1.0);
  let groundCol = u.reflHorizon.rgb * 0.55;
  return mix(skyToHorizon, groundCol, below);
}

fn traceSSR(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
  if (dir.z >= 0.0) { return vec4<f32>(0.0); }

  let STEPS: i32 = 6;
  let MAX_DIST: f32 = 25.0;
  let stepSize = MAX_DIST / f32(STEPS);
  let dims = vec2<f32>(textureDimensions(sceneDepth));
  let dimsMax = dims - vec2<f32>(1.0);
  let near = u.params0.w;
  let far = u.params1.x;

  var t = stepSize;
  for (var i = 0; i < STEPS; i = i + 1) {
    let samplePos = origin + dir * t;
    if (samplePos.z >= -1e-3) { break; }

    let proj = projectToUV(samplePos);
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) { break; }

    let px = vec2<i32>(clamp(proj.xy * dims, vec2<f32>(0.0), dimsMax));
    let raw = textureLoad(sceneDepth, px, 0);

    if (raw < 0.9999) {
      let sceneLin = linearizeDepth(raw, near, far);
      if (sceneLin <= proj.z) {
        let col = textureSampleLevel(sceneColor, linSamp, proj.xy, 0.0).rgb;
        let fade = 1.0 - f32(i) / f32(STEPS);
        return vec4<f32>(col, fade * 0.9);
      }
    }

    t = t + stepSize;
  }

  return vec4<f32>(0.0);
}

const MAX_OPTICAL_DEPTH: f32 = 3.0;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let sceneDimsI = vec2<i32>(textureDimensions(sceneDepth));
  let sceneDimsF = vec2<f32>(sceneDimsI);
  let fluidDimsI = vec2<i32>(textureDimensions(fluidDepth));
  let fluidDimsF = vec2<f32>(fluidDimsI);

  let sceneCoord = clamp(
    vec2<i32>(in.uv * sceneDimsF),
    vec2<i32>(0),
    sceneDimsI - vec2<i32>(1),
  );
  let fluidCoord = clamp(
    vec2<i32>(in.uv * fluidDimsF),
    vec2<i32>(0),
    fluidDimsI - vec2<i32>(1),
  );

  let sceneCol = textureSampleLevel(sceneColor, linSamp, in.uv, 0.0).rgb;

  let dC = textureLoad(fluidDepth, fluidCoord, 0).r;

  if (dC >= 1e5) {
    return vec4<f32>(sceneCol, 1.0);
  }

  let near = u.params0.w;
  let far = u.params1.x;

  let sceneRaw = textureLoad(sceneDepth, sceneCoord, 0);
  if (sceneRaw < 0.9999) {
    let sceneLinear = linearizeDepth(sceneRaw, near, far);
    if (sceneLinear < dC - 0.05) {
      return vec4<f32>(sceneCol, 1.0);
    }
  }

  let pC = reconstructViewPos(in.uv, dC);
  let surfaceWS = viewToWorld(pC);

  let fluidMax = fluidDimsI - vec2<i32>(1);
  let sL = textureLoad(fluidDepth, clamp(fluidCoord + vec2<i32>(-2, 0), vec2<i32>(0), fluidMax), 0).r;
  let sR = textureLoad(fluidDepth, clamp(fluidCoord + vec2<i32>( 2, 0), vec2<i32>(0), fluidMax), 0).r;
  let sU = textureLoad(fluidDepth, clamp(fluidCoord + vec2<i32>(0, -2), vec2<i32>(0), fluidMax), 0).r;
  let sD = textureLoad(fluidDepth, clamp(fluidCoord + vec2<i32>(0,  2), vec2<i32>(0), fluidMax), 0).r;

  let validL = sL < 1e5;
  let validR = sR < 1e5;
  let validU = sU < 1e5;
  let validD = sD < 1e5;

  var validCount = 0.0;
  if (validL) { validCount = validCount + 1.0; }
  if (validR) { validCount = validCount + 1.0; }
  if (validU) { validCount = validCount + 1.0; }
  if (validD) { validCount = validCount + 1.0; }
  let edgeFactor = validCount / 4.0;

  let dLv = select(dC, sL, validL);
  let dRv = select(dC, sR, validR);
  let dUv = select(dC, sU, validU);
  let dDv = select(dC, sD, validD);

  let texel = vec2<f32>(2.0) / fluidDimsF;
  let pL = reconstructViewPos(in.uv - vec2<f32>(texel.x, 0.0), dLv);
  let pR = reconstructViewPos(in.uv + vec2<f32>(texel.x, 0.0), dRv);
  let pU = reconstructViewPos(in.uv - vec2<f32>(0.0, texel.y), dUv);
  let pD = reconstructViewPos(in.uv + vec2<f32>(0.0, texel.y), dDv);

  let ddx = pR - pL;
  let ddy = pD - pU;

  let worldUpInView = normalize(u.params2.yzw);
  var normalSmooth: vec3<f32>;
  if (length(ddx) < 1e-4 || length(ddy) < 1e-4) {
    normalSmooth = worldUpInView;
  } else {
    normalSmooth = normalize(cross(ddx, ddy));
    if (normalSmooth.z < 0.0) { normalSmooth = -normalSmooth; }
  }

  let facing = clamp(abs(normalSmooth.z), 0.0, 1.0);

  let foamRaw = sampleFoamDilated(fluidCoord, fluidDimsI);

  let n1 = fbm3(surfaceWS * u.water1.y);
  let n2 = fbm3(surfaceWS * u.water1.y + vec3<f32>(17.3, 5.1, 11.7));
  let perturb = vec2<f32>(n1 - 0.5, n2 - 0.5) * u.water1.z;

  let flatness = smoothstep(u.water1.w, u.water2.x, dot(normalSmooth, worldUpInView));

  let perturbAmt = flatness * edgeFactor * facing;
  var normalShade = normalize(
    normalSmooth + vec3<f32>(perturb.x, perturb.y, 0.0) * perturbAmt,
  );

  let foamDetail = fbm3(surfaceWS * u.foamParams.y);
  let foamEdge = fbm3(surfaceWS * u.foamParams.y * 3.3 + vec3<f32>(9.1, 2.4, 5.6));
  let foamShaped = foamRaw * mix(0.5, 1.4, foamDetail) - foamEdge * 0.12;
  let foamMask = smoothstep(
    u.foamParams.z,
    u.foamParams.z + u.foamParams.w,
    foamShaped
  ) * u.foamParams.x;

  let thickness = textureLoad(fluidThickness, fluidCoord, 0).r;
  let viewDir = normalize(-reconstructViewPos(in.uv, dC));

  let shadowTint = computeFluidShadow(surfaceWS, normalSmooth);

  let cosTheta = clamp(dot(normalShade, viewDir), 0.0, 1.0);
  let F0 = u.water1.x;
  let fresnel = min(F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0), 0.85);

  let baseScatter = u.params3.yzw;
  let deepScatter = baseScatter * u.water2.w;
  let depthMix = 1.0 - exp(-thickness * u.water0.x);
  var scatterColor = mix(baseScatter, deepScatter, depthMix);
  scatterColor *= mix(vec3<f32>(0.5), vec3<f32>(1.0), shadowTint);

  let refrStr = u.params0.z;
  let thicknessGate = smoothstep(u.water0.y, u.water0.z, thickness);
  let incident = -viewDir;
  let refrDir = refract(incident, normalSmooth, u.water2.z);

  var refrUV = in.uv;
  if (dot(refrDir, refrDir) > 1e-6) {
    let travel = clamp(
      sqrt(max(thickness, 0.0)) * refrStr,
      0.0,
      u.water2.y,
    ) * thicknessGate * edgeFactor * facing;
    let refrPoint = pC + refrDir * travel;
    let proj = projectToUV(refrPoint);
    refrUV = clamp(proj.xy, vec2<f32>(0.0), vec2<f32>(1.0));
  }

  let refrCoordF = refrUV * fluidDimsF;
  let refrCoord = clamp(
    vec2<i32>(refrCoordF),
    vec2<i32>(0),
    fluidDimsI - vec2<i32>(1),
  );

  let refrThickness = textureLoad(fluidThickness, refrCoord, 0).r;
  let refrSceneCoord = clamp(
    vec2<i32>(refrUV * sceneDimsF),
    vec2<i32>(0),
    sceneDimsI - vec2<i32>(1),
  );
  let refrSceneRaw = textureLoad(sceneDepth, refrSceneCoord, 0);
  let refrValid = refrThickness > 1.0 && refrSceneRaw < 0.9999;

  var refracted: vec3<f32>;
  if (refrValid) {
    refracted = textureSampleLevel(sceneColor, linSamp, refrUV, 0.0).rgb;
  } else {
    let edgeTint = smoothstep(0.0, 3.0, thickness);
    refracted = mix(sceneCol, scatterColor, edgeTint);
  }

  let absorptionCoeff = u.fluidColor.rgb;
  let opticalDepth = min(sqrt(max(thickness, 0.0)) * u.params1.y, MAX_OPTICAL_DEPTH);
  let absorb = exp(-absorptionCoeff * opticalDepth);
  var col = mix(scatterColor, refracted, absorb);

  let R = reflect(-viewDir, normalShade);
  let worldRy = max(dot(R, worldUpInView), 0.0);
  let fakeRefl = sampleFakeEnv(worldRy);

  var reflection = fakeRefl;
  if (fresnel > 0.04) {
    let ssr = traceSSR(pC, R);
    reflection = mix(fakeRefl, ssr.rgb, ssr.a * 0.85);
  }
  col = mix(col, reflection, fresnel);

  let sunVS = normalize(u.sunDirView.xyz);
  let halfVS = normalize(sunVS + viewDir);
  let spec = pow(max(dot(normalShade, halfVS), 0.0), u.params1.z);
  let specFacing = smoothstep(0.0, 0.4, facing);
  col = col + u.specColor.rgb * spec * u.params1.w * edgeFactor * specFacing * shadowTint;

  let alpha = clamp((1.0 - exp(-thickness * u.water0.w)) * edgeFactor, 0.0, 1.0);
  col = mix(sceneCol, col, alpha);

  let foamVisible = foamMask * alpha * facing;
  col = mix(col, u.foamColor.rgb, foamVisible);
  col = col + u.specColor.rgb * spec * foamVisible * 0.35 * shadowTint;

  return vec4<f32>(col, 1.0);
}
`;

const Z_REMAP = new THREE.Matrix4().set(
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

export class SSFRRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private depthPipeline!: GPURenderPipeline;
  private thicknessPipeline!: GPURenderPipeline;
  private lightDepthPipeline!: GPURenderPipeline;
  private bilateralHPipeline!: GPUComputePipeline;
  private bilateralVPipeline!: GPUComputePipeline;
  private compositePipeline!: GPURenderPipeline;
  private causticsPipeline!: GPUComputePipeline;

  private particleUniform!: GPUBuffer;
  private lightUniform!: GPUBuffer;
  private causticsParams!: GPUBuffer;
  private bilateralParamsH!: GPUBuffer;
  private bilateralParamsV!: GPUBuffer;
  private compositeUniform!: GPUBuffer;
  private compositeLightUniform!: GPUBuffer;

  private depthBindGroup!: GPUBindGroup;
  private thicknessBindGroup!: GPUBindGroup;
  private lightDepthBind!: GPUBindGroup;
  private lightThicknessBind!: GPUBindGroup;
  private bilateralHBind!: GPUBindGroup;
  private bilateralVBind!: GPUBindGroup;
  private compositeBindGroup!: GPUBindGroup;
  private compositeLightBind!: GPUBindGroup;
  private causticsBind!: GPUBindGroup;

  private sceneColorTexture!: GPUTexture;
  private sceneDepthTexture!: GPUTexture;
  private fluidDepthTexture!: GPUTexture;
  private fluidDepthStencil!: GPUTexture;
  private fluidDepthTemp!: GPUTexture;
  private fluidDepthSmooth!: GPUTexture;
  private fluidThicknessTexture!: GPUTexture;
  private fluidWorldPosTexture!: GPUTexture;

  private lightDepthTexture!: GPUTexture;
  private lightThicknessTexture!: GPUTexture;
  private lightDepthStencil!: GPUTexture;
  private causticsTexture!: GPUTexture;

  private linearSampler!: GPUSampler;

  private compLayout!: GPUBindGroupLayout;
  private bilatLayout!: GPUBindGroupLayout;
  private causticsLayout!: GPUBindGroupLayout;

  private width = 1;
  private height = 1;
  private fluidWidth = 1;
  private fluidHeight = 1;

  private fluidScale = 0.5;

  private lightViewProjArray = new Float32Array(16);
  private lightViewArray = new Float32Array(16);

  private readonly compositeData = new Float32Array(72);

  public time = 0;
  public get lightAbsorb(): [number, number, number] {
    return [config.lightAbsorb.r, config.lightAbsorb.g, config.lightAbsorb.b];
  }

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

    this.lightUniform = this.device.createBuffer({
      size: 176,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.causticsParams = this.device.createBuffer({
      size: 16,
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
      size: 288,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.compositeLightUniform = this.device.createBuffer({
      size: 144,
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
    const lightDepthMod = this.device.createShaderModule({
      code: lightDepthWGSL,
    });
    const causticsMod = this.device.createShaderModule({
      code: causticsWGSL,
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

    this.lightDepthBind = this.device.createBindGroup({
      layout: particleLayout,
      entries: [
        { binding: 0, resource: { buffer: this.lightUniform } },
        { binding: 1, resource: { buffer: particlesBuffer } },
      ],
    });

    this.lightThicknessBind = this.device.createBindGroup({
      layout: particleLayout,
      entries: [
        { binding: 0, resource: { buffer: this.lightUniform } },
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
        targets: [{ format: "r32float" }, { format: "rgba16float" }],
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

    this.lightDepthPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [particleLayout],
      }),
      vertex: { module: lightDepthMod, entryPoint: "vs" },
      fragment: {
        module: lightDepthMod,
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

    this.causticsLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "rg32float" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.causticsPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.causticsLayout],
      }),
      compute: { module: causticsMod, entryPoint: "main" },
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
        {
          binding: 6,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float" },
        },
      ],
    });

    const compositeLightLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float" },
        },
      ],
    });

    this.compositePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [compLayout, compositeLightLayout],
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

    this.lightDepthTexture = this.device.createTexture({
      size: [LIGHT_MAP_SIZE, LIGHT_MAP_SIZE],
      format: "r32float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.lightThicknessTexture = this.device.createTexture({
      size: [LIGHT_MAP_SIZE, LIGHT_MAP_SIZE],
      format: "r16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.lightDepthStencil = this.device.createTexture({
      size: [LIGHT_MAP_SIZE, LIGHT_MAP_SIZE],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.causticsTexture = this.device.createTexture({
      size: [LIGHT_MAP_SIZE, LIGHT_MAP_SIZE],
      format: "rg32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.compositeLightBind = this.device.createBindGroup({
      layout: compositeLightLayout,
      entries: [
        { binding: 0, resource: { buffer: this.compositeLightUniform } },
        { binding: 1, resource: this.lightDepthTexture.createView() },
        { binding: 2, resource: this.lightThicknessTexture.createView() },
      ],
    });

    this.causticsBind = this.device.createBindGroup({
      layout: this.causticsLayout,
      entries: [
        { binding: 0, resource: this.lightDepthTexture.createView() },
        { binding: 1, resource: this.causticsTexture.createView() },
        { binding: 2, resource: { buffer: this.causticsParams } },
      ],
    });
  }
  public resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.fluidWidth = Math.max(1, Math.floor(this.width * this.fluidScale));
    this.fluidHeight = Math.max(1, Math.floor(this.height * this.fluidScale));

    this.sceneColorTexture?.destroy();
    this.sceneDepthTexture?.destroy();
    this.fluidDepthTexture?.destroy();
    this.fluidDepthStencil?.destroy();
    this.fluidDepthTemp?.destroy();
    this.fluidDepthSmooth?.destroy();
    this.fluidThicknessTexture?.destroy();
    this.fluidWorldPosTexture?.destroy();

    const sceneSize = [this.width, this.height];
    const fluidSize = [this.fluidWidth, this.fluidHeight];

    this.sceneColorTexture = this.device.createTexture({
      size: sceneSize,
      format: this.format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sceneDepthTexture = this.device.createTexture({
      size: sceneSize,
      format: "depth32float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fluidDepthTexture = this.device.createTexture({
      size: fluidSize,
      format: "r32float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fluidDepthStencil = this.device.createTexture({
      size: fluidSize,
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.fluidDepthTemp = this.device.createTexture({
      size: fluidSize,
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fluidDepthSmooth = this.device.createTexture({
      size: fluidSize,
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fluidThicknessTexture = this.device.createTexture({
      size: fluidSize,
      format: "r16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fluidWorldPosTexture = this.device.createTexture({
      size: fluidSize,
      format: "rgba16float",
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
        { binding: 6, resource: this.fluidWorldPosTexture.createView() },
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
  public getFluidWorldPosView(): GPUTextureView {
    return this.fluidWorldPosTexture.createView();
  }
  public getCausticsView(): GPUTextureView {
    return this.causticsTexture.createView();
  }
  public getLightDepthView(): GPUTextureView {
    return this.lightDepthTexture.createView();
  }
  public getLightThicknessView(): GPUTextureView {
    return this.lightThicknessTexture.createView();
  }
  public getLightDepthStencilView(): GPUTextureView {
    return this.lightDepthStencil.createView();
  }
  public getLightViewProj(): Float32Array {
    return this.lightViewProjArray;
  }
  public getLightView(): Float32Array {
    return this.lightViewArray;
  }
  public getFarDepth(): number {
    return FAR_DEPTH;
  }
  public setParticleCount(n: number): void {
    const desired =
      n <= 10000 ? 0.75 : n <= 24000 ? 0.6 : n <= 48000 ? 0.5 : 0.4;
    if (desired !== this.fluidScale) {
      this.fluidScale = desired;
      if (this.width > 1 && this.height > 1)
        this.resize(this.width, this.height);
    }
  }
  private updateLightUniform(): void {
    const dir = new THREE.Vector3(0.45, 1.0, 0.35).normalize();
    const radius =
      Math.sqrt(
        config.boundsWidth * config.boundsWidth +
          config.boundsHeight * config.boundsHeight +
          config.boundsDepth * config.boundsDepth,
      ) *
        0.5 +
      4;

    const eye = dir.clone().multiplyScalar(radius * 3);
    const camWorld = new THREE.Matrix4().lookAt(
      eye,
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 1, 0),
    );
    camWorld.setPosition(eye);
    const view = camWorld.clone().invert();

    const proj = new THREE.Matrix4().makeOrthographic(
      -radius,
      radius,
      radius,
      -radius,
      radius * 1.5,
      radius * 5,
    );
    proj.premultiply(Z_REMAP);

    const viewProj = new THREE.Matrix4().multiplyMatrices(proj, view);

    const e = camWorld.elements;
    const rightX = e[0];
    const rightY = e[1];
    const rightZ = e[2];
    const upX = e[4];
    const upY = e[5];
    const upZ = e[6];

    const splatWorldRadius = config.particleSize * config.renderSplatScale;

    const u = new Float32Array(44);
    u.set(view.elements, 0);
    u.set(proj.elements, 16);
    u[32] = rightX;
    u[33] = rightY;
    u[34] = rightZ;
    u[35] = 0;
    u[36] = upX;
    u[37] = upY;
    u[38] = upZ;
    u[39] = 0;
    u[40] = splatWorldRadius;
    u[41] = 0;
    u[42] = 0;
    u[43] = 0;
    this.device.queue.writeBuffer(this.lightUniform, 0, u);

    this.lightViewProjArray.set(viewProj.elements);
    this.lightViewArray.set(view.elements);

    const shadowData = new Float32Array(36);
    shadowData.set(viewProj.elements, 0);
    shadowData.set(view.elements, 16);
    shadowData[32] = config.lightAbsorb.r;
    shadowData[33] = config.lightAbsorb.g;
    shadowData[34] = config.lightAbsorb.b;
    shadowData[35] = 0;
    this.device.queue.writeBuffer(this.compositeLightUniform, 0, shadowData);
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

    const causticU = new Float32Array(4);
    causticU[0] = this.time * config.causticsSpeed;
    causticU[1] = config.causticsEnabled ? config.causticsIntensity : 0.0;
    causticU[2] = config.causticsScale;
    causticU[3] = 0;
    this.device.queue.writeBuffer(this.causticsParams, 0, causticU);

    this.updateLightUniform();

    const sigmaWorld = splatWorldRadius * config.bilateralSigmaWorld;
    const sigmaDepthWorld = splatWorldRadius * config.bilateralSigmaDepth;

    const writeParams = (buffer: GPUBuffer, axis: 0 | 1) => {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab);
      const f = new Float32Array(ab);
      u[0] = axis;
      u[1] = config.bilateralMaxRadiusPx;
      f[2] = sigmaWorld;
      f[3] = sigmaDepthWorld;
      f[4] = tanHalfFovY;
      f[5] = this.fluidHeight;
      f[6] = 0;
      f[7] = 0;
      this.device.queue.writeBuffer(buffer, 0, ab);
    };
    writeParams(this.bilateralParamsH, 0);
    writeParams(this.bilateralParamsV, 1);

    this.tmpMat4.fromArray(viewMatrix).invert();
    this.tmpVec3.set(0, 1, 0).transformDirection(this.tmpMat4);

    const [sx, sy, sz] = this.sunDirView(viewMatrix);

    const cu = this.compositeData;

    cu[0] = sx;
    cu[1] = sy;
    cu[2] = sz;
    cu[3] = 0;

    cu[4] = config.waterAbsorb.r;
    cu[5] = config.waterAbsorb.g;
    cu[6] = config.waterAbsorb.b;
    cu[7] = 1.0;

    cu[8] = tanHalfFovY;
    cu[9] = aspect;
    cu[10] = config.refractionStrength;
    cu[11] = near;

    cu[12] = far;
    cu[13] = config.waterAbsorbStrength;
    cu[14] = config.specularPower;
    cu[15] = config.specularIntensity;

    cu[16] = this.time;
    cu[17] = this.tmpVec3.x;
    cu[18] = this.tmpVec3.y;
    cu[19] = this.tmpVec3.z;

    cu[20] = config.reflSky.r;
    cu[21] = config.reflSky.g;
    cu[22] = config.reflSky.b;
    cu[23] = 1.0;

    cu[24] = config.reflHorizon.r;
    cu[25] = config.reflHorizon.g;
    cu[26] = config.reflHorizon.b;
    cu[27] = 1.0;

    cu[28] = splatWorldRadius;
    cu[29] = config.waterColor.r;
    cu[30] = config.waterColor.g;
    cu[31] = config.waterColor.b;

    cu[32] = config.foamEnabled ? 1.0 : 0.0;
    cu[33] = config.foamNoiseScale;
    cu[34] = config.foamThreshold;
    cu[35] = config.foamSoftness;

    cu[36] = config.depthMixCoeff;
    cu[37] = config.thicknessGateStart;
    cu[38] = config.thicknessGateEnd;
    cu[39] = config.alphaThicknessCoeff;

    cu[40] = config.fresnelF0;
    cu[41] = config.normalNoiseScale;
    cu[42] = config.normalNoiseAmp;
    cu[43] = config.flatnessStart;

    cu[44] = config.flatnessEnd;
    cu[45] = config.refractionTravelMax;
    cu[46] = 1.0 / Math.max(config.iorWater, 1e-4);
    cu[47] = config.waterDeepTint;

    cu[48] = config.foamColor.r;
    cu[49] = config.foamColor.g;
    cu[50] = config.foamColor.b;
    cu[51] = 1.0;

    cu[52] = config.specColor.r;
    cu[53] = config.specColor.g;
    cu[54] = config.specColor.b;
    cu[55] = 1.0;

    cu.set(this.tmpMat4.elements, 56);

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
  public encodeLightDepth(
    pass: GPURenderPassEncoder,
    particleCount: number,
  ): void {
    pass.setPipeline(this.lightDepthPipeline);
    pass.setBindGroup(0, this.lightDepthBind);
    pass.draw(6, particleCount, 0, 0);
  }
  public encodeLightThickness(
    pass: GPURenderPassEncoder,
    particleCount: number,
  ): void {
    pass.setPipeline(this.thicknessPipeline);
    pass.setBindGroup(0, this.lightThicknessBind);
    pass.draw(6, particleCount, 0, 0);
  }
  public encodeCaustics(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.causticsPipeline);
    pass.setBindGroup(0, this.causticsBind);
    pass.dispatchWorkgroups(
      Math.ceil(LIGHT_MAP_SIZE / 8),
      Math.ceil(LIGHT_MAP_SIZE / 8),
    );
  }
  public encodeBilateralH(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.bilateralHPipeline);
    pass.setBindGroup(0, this.bilateralHBind);
    pass.dispatchWorkgroups(
      Math.ceil(this.fluidWidth / 8),
      Math.ceil(this.fluidHeight / 8),
    );
  }
  public encodeBilateralV(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.bilateralVPipeline);
    pass.setBindGroup(0, this.bilateralVBind);
    pass.dispatchWorkgroups(
      Math.ceil(this.fluidWidth / 8),
      Math.ceil(this.fluidHeight / 8),
    );
  }
  public encodeComposite(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, this.compositeBindGroup);
    pass.setBindGroup(1, this.compositeLightBind);
    pass.draw(3);
  }
}
