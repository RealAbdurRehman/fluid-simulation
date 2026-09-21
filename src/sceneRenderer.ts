import * as THREE from "three";

import { acesFilmicWGSL } from "./shader/common.wgsl";
import { GPUBufferUsage, GPUShaderStage, GPUTextureUsage } from "./types";
import {
  createDefaultMaterial,
  type ModelMaterial,
  type TextureSource,
} from "./materials";

const SAMPLE_COUNT = 1;
const MAX_OBJECT_SLOTS = 16;

const sceneShaderWGSL = /* wgsl */ `
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
};

struct ObjectUniforms {
  scale: vec4<f32>,
  translate: vec4<f32>,
  color: vec4<f32>,
  rotation: vec4<f32>,
};

struct LightUniforms {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;
@group(2) @binding(0) var<uniform> light: LightUniforms;
@group(2) @binding(1) var lightDepth: texture_2d<f32>;
@group(2) @binding(2) var lightThickness: texture_2d<f32>;
@group(2) @binding(3) var causticTex: texture_2d<f32>;

fn qRotateVec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let qv = q.xyz;
  let uv = cross(qv, v);
  let uuv = cross(qv, uv);
  return v + ((uv * q.w) + uuv) * 2.0;
}

fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn computeShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(vec3<f32>(0.45, 1.0, 0.35));
  let nDotL = max(dot(normal, lightDir), 0.0);

  let normalOffset = 0.06 + 0.18 * (1.0 - nDotL);
  let offsetPos = worldPos + normal * normalOffset;

  let lp = light.viewProj * vec4<f32>(offsetPos, 1.0);
  let w = max(abs(lp.w), 1e-6);
  let ndc = lp.xyz / w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return vec3<f32>(1.0);
  }

  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let dims = vec2<f32>(textureDimensions(lightDepth));
  let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));

  let mapZ = textureLoad(lightDepth, px, 0).r;
  let lv = light.view * vec4<f32>(offsetPos, 1.0);
  let recZ = -lv.z;

  let slopeBias = 0.08 + 0.85 * (1.0 - nDotL) * (1.0 - nDotL);
  let diff = recZ - mapZ;
  let shadowAmt = smoothstep(slopeBias, slopeBias + 0.40, diff);

  let rawThickness = textureLoad(lightThickness, px, 0).r;
  let thickness = min(rawThickness, 3.0);
  let absorb = exp(-light.params.rgb * thickness);

  let opaqueAmount = 1.0 - smoothstep(0.0, 0.5, thickness);
  let occluderColor = mix(absorb, vec3<f32>(0.0), opaqueAmount);

  let tinted = mix(vec3<f32>(1.0), occluderColor, shadowAmt);
  let backfaceMask = smoothstep(0.0, 0.20, nDotL);
  return mix(vec3<f32>(1.0), max(tinted, vec3<f32>(0.35)), backfaceMask);
}

fn computeCaustics(worldPos: vec3<f32>, worldNormal: vec3<f32>) -> f32 {
  if (worldNormal.y < 0.1) { return 1.0; }

  let lp = light.viewProj * vec4<f32>(worldPos, 1.0);
  let w = max(abs(lp.w), 1e-6);
  let ndc = lp.xyz / w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
    return 1.0;
  }

  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let dims = vec2<f32>(textureDimensions(causticTex));
  let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));

  let sample2 = textureLoad(causticTex, px, 0);
  let fluidDepth = sample2.g;
  if (fluidDepth >= 1e5) { return 1.0; }

  let lv = light.view * vec4<f32>(worldPos, 1.0);
  let recZ = -lv.z;
  if (recZ < fluidDepth - 0.15) { return 1.0; }

  return sample2.r;
}

struct MaterialUniforms {
  baseColor: vec4<f32>,
  params: vec4<f32>,
};

@group(3) @binding(0) var<uniform> material: MaterialUniforms;
@group(3) @binding(1) var baseSampler: sampler;
@group(3) @binding(2) var baseTex: texture_2d<f32>;

struct MeshVIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};
struct MeshVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn mesh_vs(input: MeshVIn) -> MeshVOut {
  let scaled = input.position * obj.scale.xyz;
  let rotatedPos = qRotateVec(obj.rotation, scaled);
  let rotatedNormal = qRotateVec(obj.rotation, input.normal);
  let world = rotatedPos + obj.translate.xyz;
  var out: MeshVOut;
  out.position = frame.viewProj * vec4<f32>(world, 1.0);
  out.worldPos = world;
  out.worldNormal = rotatedNormal;
  out.uv = input.uv;
  return out;
}

@fragment
fn mesh_fs(input: MeshVOut) -> @location(0) vec4<f32> {
  let texel = textureSample(baseTex, baseSampler, input.uv);
  let tint = mix(vec3<f32>(1.0), obj.color.rgb, material.params.x);
  let albedo = tint * material.baseColor.rgb * texel.rgb;

  let n = normalize(input.worldNormal);
  let lightDir = normalize(vec3<f32>(0.45, 1.0, 0.35));
  let ndl = max(dot(n, lightDir), 0.0);
  let viewDir = normalize(frame.cameraPos.xyz - input.worldPos);

  let shadow = computeShadow(input.worldPos, n);
  let caustic = computeCaustics(input.worldPos, n);

  let ambient = vec3<f32>(0.30, 0.36, 0.44);
  let direct = albedo * (ndl * 0.95) * shadow * caustic;

  var lit = albedo * ambient + direct;

  let rim = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0) * 0.55;
  lit += vec3<f32>(0.35, 0.65, 1.0) * rim;

  let halfDir = normalize(lightDir + viewDir);
  lit += vec3<f32>(pow(max(dot(n, halfDir), 0.0), 48.0) * 0.4) * shadow;

  return vec4<f32>(acesFilmic(lit * 1.3), obj.color.a);
}
`;

const lightObjectWGSL = /* wgsl */ `
struct LightUniforms {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  params: vec4<f32>,
};

struct ObjectUniforms {
  scale: vec4<f32>,
  translate: vec4<f32>,
  color: vec4<f32>,
  rotation: vec4<f32>,
};

@group(0) @binding(0) var<uniform> light: LightUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

fn qRotateVec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let qv = q.xyz;
  let uv = cross(qv, v);
  let uuv = cross(qv, uv);
  return v + ((uv * q.w) + uuv) * 2.0;
}

struct VIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
};
struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@vertex
fn light_vs(input: VIn) -> VOut {
  let scaled = input.position * obj.scale.xyz;
  let rotated = qRotateVec(obj.rotation, scaled);
  let world = rotated + obj.translate.xyz;
  var out: VOut;
  out.clip = light.viewProj * vec4<f32>(world, 1.0);
  out.worldPos = world;
  return out;
}

@fragment
fn light_fs(in: VOut) -> @location(0) vec4<f32> {
  let lv = light.view * vec4<f32>(in.worldPos, 1.0);
  return vec4<f32>(-lv.z, 0.0, 0.0, 0.0);
}
`;

const skyShaderWGSL = /* wgsl */ `
struct SkyUniforms {
  invViewProj: mat4x4<f32>,
  sunDir: vec4<f32>,
  sunColor: vec4<f32>,
  topColor: vec4<f32>,
  horizonColor: vec4<f32>,
  bottomColor: vec4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sky: SkyUniforms;

${acesFilmicWGSL}

struct SkyVOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn sky_vs(@builtin(vertex_index) vi: u32) -> SkyVOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: SkyVOut;
  out.pos = vec4<f32>(p[vi], 1.0, 1.0);
  out.uv = p[vi] * 0.5 + 0.5;
  return out;
}

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
  for (var i = 0; i < 5; i = i + 1) {
    v += a * noise2(p);
    p = p * 2.07 + vec2<f32>(1.7, 9.2);
    a *= 0.5;
  }
  return v;
}

@fragment
fn sky_fs(in: SkyVOut) -> @location(0) vec4<f32> {
  let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, in.uv.y * 2.0 - 1.0);
  let nW = sky.invViewProj * vec4<f32>(ndc, 0.0, 1.0);
  let fW = sky.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  let dir = normalize(fW.xyz / fW.w - nW.xyz / nW.w);
  let up = dir.y;

  var col: vec3<f32>;
  if (up > 0.0) {
    let t = pow(clamp(up, 0.0, 1.0), 0.55);
    col = mix(sky.horizonColor.rgb, sky.topColor.rgb, t);
  } else {
    let t = pow(clamp(-up, 0.0, 1.0), 0.55);
    col = mix(sky.horizonColor.rgb, sky.bottomColor.rgb, t);
  }

  let sunDot = max(dot(dir, sky.sunDir.xyz), 0.0);
  let sunDisk = smoothstep(0.9997, 0.99995, sunDot);
  let sunGlow = pow(sunDot, 128.0) * 0.5 + pow(sunDot, 12.0) * 0.10;
  col += sky.sunColor.rgb * (sunDisk * 20.0 + sunGlow);

  let haze = exp(-abs(up) * 6.0) * 0.5;
  let hazeCol = mix(sky.horizonColor.rgb, sky.sunColor.rgb, pow(sunDot, 3.0) * 0.7);
  col = mix(col, hazeCol, haze * 0.35);

  if (up > 0.01) {
    let plane = dir.xz / max(dir.y, 0.08) * 0.5;
    let t = sky.params.x;
    let p = plane + vec2<f32>(t * 0.006, t * 0.003);
    let base = fbm2(p * 0.55);
    let detail = fbm2(p * 1.90 + vec2<f32>(11.0, 5.0));
    let shape = base * 0.75 + detail * 0.25;
    let cov = sky.params.y;
    var cloud = smoothstep(cov, cov + 0.22, shape);
    cloud = cloud * smoothstep(0.0, 0.25, up);
    let lit = 0.55 + 0.45 * sunDot;
    let cloudCol = mix(sky.horizonColor.rgb * 0.9, vec3<f32>(1.0, 0.98, 0.94), lit);
    col = mix(col, cloudCol, cloud * 0.85);
  }

  col *= 1.0 - clamp(-up, 0.0, 1.0) * 0.35;
  return vec4<f32>(acesFilmic(col * sky.params.z), 1.0);
}
`;

const terrainShaderWGSL = /* wgsl */ `
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
};

struct TerrainUniforms {
  rotation: vec4<f32>,
  time: f32,
  waterLevel: f32,
  causticStrength: f32,
  fogDensity: f32,
};

struct LightUniforms {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> terrain: TerrainUniforms;
@group(2) @binding(0) var<uniform> light: LightUniforms;
@group(2) @binding(1) var lightDepth: texture_2d<f32>;
@group(2) @binding(2) var lightThickness: texture_2d<f32>;
@group(2) @binding(3) var causticTex: texture_2d<f32>;

fn qRotateVec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let qv = q.xyz;
  let uv = cross(qv, v);
  let uuv = cross(qv, uv);
  return v + ((uv * q.w) + uuv) * 2.0;
}

fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn computeShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(vec3<f32>(0.45, 1.0, 0.35));
  let nDotL = max(dot(normal, lightDir), 0.0);

  let normalOffset = 0.06 + 0.18 * (1.0 - nDotL);
  let offsetPos = worldPos + normal * normalOffset;

  let lp = light.viewProj * vec4<f32>(offsetPos, 1.0);
  let w = max(abs(lp.w), 1e-6);
  let ndc = lp.xyz / w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return vec3<f32>(1.0);
  }

  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let dims = vec2<f32>(textureDimensions(lightDepth));
  let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));

  let mapZ = textureLoad(lightDepth, px, 0).r;
  let lv = light.view * vec4<f32>(offsetPos, 1.0);
  let recZ = -lv.z;

  let slopeBias = 0.08 + 0.85 * (1.0 - nDotL) * (1.0 - nDotL);
  let diff = recZ - mapZ;
  let shadowAmt = smoothstep(slopeBias, slopeBias + 0.40, diff);

  let rawThickness = textureLoad(lightThickness, px, 0).r;
  let thickness = min(rawThickness, 3.0);
  let absorb = exp(-light.params.rgb * thickness);

  let opaqueAmount = 1.0 - smoothstep(0.0, 0.5, thickness);
  let occluderColor = mix(absorb, vec3<f32>(0.0), opaqueAmount);

  let tinted = mix(vec3<f32>(1.0), occluderColor, shadowAmt);
  let backfaceMask = smoothstep(0.0, 0.20, nDotL);
  return mix(vec3<f32>(1.0), max(tinted, vec3<f32>(0.35)), backfaceMask);
}

fn computeCaustics(worldPos: vec3<f32>, worldNormal: vec3<f32>) -> f32 {
  if (worldNormal.y < 0.1) { return 1.0; }

  let lp = light.viewProj * vec4<f32>(worldPos, 1.0);
  let w = max(abs(lp.w), 1e-6);
  let ndc = lp.xyz / w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
    return 1.0;
  }

  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let dims = vec2<f32>(textureDimensions(causticTex));
  let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));

  let sample2 = textureLoad(causticTex, px, 0);
  let fluidDepth = sample2.g;
  if (fluidDepth >= 1e5) { return 1.0; }

  let lv = light.view * vec4<f32>(worldPos, 1.0);
  let recZ = -lv.z;
  if (recZ < fluidDepth - 0.15) { return 1.0; }

  return sample2.r;
}

fn hash21(p: vec2<f32>) -> f32 {
  var q = fract(p * vec2<f32>(127.1, 311.7));
  q += dot(q, q + 34.23);
  return fract(q.x * q.y);
}

fn valueNoise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2<f32>(1.0, 0.0));
  let c = hash21(i + vec2<f32>(0.0, 1.0));
  let d = hash21(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p0: vec2<f32>, octaves: i32) -> f32 {
  var v = 0.0;
  var a = 0.5;
  var p = p0;
  for (var i = 0; i < octaves; i++) {
    v += a * valueNoise(p);
    p = p * 2.07 + vec2<f32>(1.7, 9.2);
    a *= 0.5;
  }
  return v;
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

fn fbm3(p0: vec3<f32>, octaves: i32) -> f32 {
  var v = 0.0;
  var a = 0.5;
  var p = p0;
  for (var i = 0; i < octaves; i++) {
    v += a * valueNoise3(p);
    p = p * 2.07 + vec3<f32>(1.7, 9.2, 3.3);
    a *= 0.5;
  }
  return v;
}

struct VIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

struct VOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) localPos: vec3<f32>,
  @location(3) localNormal: vec3<f32>,
};

@vertex
fn terrain_vs(input: VIn) -> VOut {
  let world = qRotateVec(terrain.rotation, input.position);
  let worldN = qRotateVec(terrain.rotation, input.normal);
  var out: VOut;
  out.position = frame.viewProj * vec4<f32>(world, 1.0);
  out.worldPos = world;
  out.worldNormal = worldN;
  out.localPos = input.position;
  out.localNormal = input.normal;
  return out;
}

@fragment
fn terrain_fs(input: VOut) -> @location(0) vec4<f32> {
  let n = normalize(input.worldNormal);
  let localN = normalize(input.localNormal);
  let up = max(localN.y, 0.0);
  let slope = 1.0 - up;

  let wp3 = input.localPos;
  let wp = wp3.xz;

  let sandLight = vec3<f32>(0.96, 0.90, 0.76);
  let sandMid   = vec3<f32>(0.86, 0.77, 0.60);
  let sandDeep  = vec3<f32>(0.62, 0.54, 0.42);

  let drift = fbm3(wp3 * 0.22, 4);
  var albedo = mix(sandDeep, sandLight, drift);

  let lum = dot(albedo, vec3<f32>(0.299, 0.587, 0.114));
  albedo = max(vec3<f32>(0.0), mix(vec3<f32>(lum), albedo, 1.18));

  let tintWarmField = fbm3(wp3 * 0.42 + vec3<f32>(51.3, 12.7, 33.1), 3);
  let tintCoolField = fbm3(wp3 * 0.31 + vec3<f32>(-8.9, 47.1, 27.4), 3);
  let warmTint = vec3<f32>(1.07, 1.00, 0.88);
  let coolTint = vec3<f32>(0.90, 0.96, 1.08);
  let warmAmt = smoothstep(0.35, 0.75, tintWarmField) * 0.40;
  let coolAmt = smoothstep(0.35, 0.75, tintCoolField) * 0.35;
  albedo *= mix(vec3<f32>(1.0), warmTint, warmAmt);
  albedo *= mix(vec3<f32>(1.0), coolTint, coolAmt);

  let warpA = fbm(wp * 0.55, 3);
  let phaseA = (wp.x * 0.85 + wp.y * 0.50) * 8.0 + warpA * 10.0;
  let rippleA = sin(phaseA) * 0.5 + 0.5;

  let warpB = fbm(wp * 0.80 + 30.0, 3);
  let phaseB = (wp.x * -0.45 + wp.y * 0.95) * 13.0 + warpB * 7.0;
  let rippleB = sin(phaseB) * 0.5 + 0.5;

  let regionMask = smoothstep(
    0.35,
    0.65,
    fbm(wp * 0.06 + 100.0, 3)
  );
  let ripple = mix(rippleA, rippleB, regionMask);

  let rippleMask = smoothstep(0.35, 0.90, up);
  albedo *= 0.94 + ripple * 0.10 * rippleMask;

  let grain = fbm3(wp3 * 8.0, 2);
  albedo *= 0.96 + grain * 0.08;

  albedo = mix(albedo, sandMid, slope * 0.5);

  let sunDir = normalize(vec3<f32>(0.45, 1.0, 0.35));
  let ndl = max(dot(n, sunDir), 0.0);
  let halfLambert = ndl * 0.7 + 0.30;
  let sunCol = vec3<f32>(1.02, 0.98, 0.92);

  let shadow = computeShadow(input.worldPos, n);
  let caustic = computeCaustics(input.worldPos, n);

  let ambientTerm = vec3<f32>(0.30, 0.34, 0.42);
  let directTerm = vec3<f32>(halfLambert) * sunCol * 1.65;
  var lit = albedo * (ambientTerm + directTerm * shadow * caustic);

  let depth = clamp(-input.localPos.y * 0.10, 0.0, 1.0);
  lit = mix(lit, lit * vec3<f32>(0.72, 0.66, 0.56), depth * 0.40);

  let toCam = frame.cameraPos.xyz - input.worldPos;
  let viewDir = toCam / max(length(toCam), 1e-4);
  let rim = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0) * 0.05;
  lit += vec3<f32>(0.85, 0.80, 0.72) * rim;

  return vec4<f32>(acesFilmic(lit), 1.0);
}
`;

export interface ObjectVisual {
  visible: boolean;
  meshId?: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  scale: number;
  color: [number, number, number, number];
}

const MESH_VERTEX_BUFFERS: GPUVertexBufferLayout[] = [
  {
    arrayStride: 32,
    stepMode: "vertex",
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
      { shaderLocation: 2, offset: 24, format: "float32x2" },
    ],
  },
];

interface RegisteredSubmesh {
  indexStart: number;
  indexCount: number;
  materialBindGroup: GPUBindGroup;
}

interface RegisteredMesh {
  vertex: GPUBuffer;
  index: GPUBuffer;
  indexCount: number;
  submeshes: RegisteredSubmesh[];
}

export class SceneRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private meshPipeline!: GPURenderPipeline;
  private skyPipeline!: GPURenderPipeline;
  private terrainPipeline!: GPURenderPipeline;
  private lightMeshPipeline!: GPURenderPipeline;

  private frameUniform!: GPUBuffer;
  private frameBindGroup!: GPUBindGroup;
  private objectUniform!: GPUBuffer;
  private objectBindGroup!: GPUBindGroup;
  private objectStride = 256;

  private skyUniform!: GPUBuffer;
  private skyBindGroup!: GPUBindGroup;

  private terrainUniform!: GPUBuffer;
  private terrainBindGroup!: GPUBindGroup;

  private lightUniform!: GPUBuffer;
  private lightBindGroup: GPUBindGroup | null = null;
  private lightLayout!: GPUBindGroupLayout;

  private lightObjectFrameLayout!: GPUBindGroupLayout;
  private lightObjectFrameBind!: GPUBindGroup;

  private meshRegistry = new Map<string, RegisteredMesh>();

  private materialLayout!: GPUBindGroupLayout;
  private whiteTexture!: GPUTexture;
  private defaultMaterialBindGroup!: GPUBindGroup;
  private textureCache = new Map<TextureSource, GPUTexture>();
  private materialCache = new Map<ModelMaterial, GPUBindGroup>();

  private depthTexture!: GPUTexture;
  private msaaTexture!: GPUTexture;
  private depthWidth = 1;
  private depthHeight = 1;

  private objectVisuals: ObjectVisual[] = [];
  private boundsQuat: [number, number, number, number] = [0, 0, 0, 1];

  private terrainMeshId: string | null = null;

  public time = 0;
  public waterLevel = 0;
  public causticStrength = 0.7;
  public fogDensity = 0.6;
  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
    this.objectStride = Math.max(
      256,
      this.device.limits.minUniformBufferOffsetAlignment || 256,
    );
    this.createPipelines();
  }
  public setObjectVisuals(v: ObjectVisual[]): void {
    this.objectVisuals = v;
  }
  public setBoundsRotationQuat(q: [number, number, number, number]): void {
    this.boundsQuat = q;
  }
  public setTerrainMesh(id: string | null): void {
    this.terrainMeshId = id;
  }
  public setLightMaps(
    depthView: GPUTextureView,
    thicknessView: GPUTextureView,
    causticView: GPUTextureView,
  ): void {
    if (!this.lightUniform) return;
    this.lightBindGroup = this.device.createBindGroup({
      layout: this.lightLayout,
      entries: [
        { binding: 0, resource: { buffer: this.lightUniform } },
        { binding: 1, resource: depthView },
        { binding: 2, resource: thicknessView },
        { binding: 3, resource: causticView },
      ],
    });
  }
  public updateLight(
    viewProj: Float32Array,
    view: Float32Array,
    absorb: [number, number, number],
  ): void {
    if (!this.lightUniform) return;

    const data = new Float32Array(36);
    data.set(viewProj, 0);
    data.set(view, 16);
    data[32] = absorb[0];
    data[33] = absorb[1];
    data[34] = absorb[2];
    data[35] = 0;

    this.device.queue.writeBuffer(this.lightUniform, 0, data);
  }
  public registerMesh(
    name: string,
    geometry: THREE.BufferGeometry,
    materials: ModelMaterial[] = [],
  ): void {
    if (this.meshRegistry.has(name)) return;
    if (!geometry.attributes.normal) geometry.computeVertexNormals();

    const posAttr = geometry.attributes.position as THREE.BufferAttribute;
    const normAttr = geometry.attributes.normal as THREE.BufferAttribute;
    const uvAttr = geometry.attributes.uv as THREE.BufferAttribute | undefined;
    const count = posAttr.count;

    const data = new Float32Array(count * 8);
    for (let i = 0; i < count; i++) {
      data[i * 8 + 0] = posAttr.getX(i);
      data[i * 8 + 1] = posAttr.getY(i);
      data[i * 8 + 2] = posAttr.getZ(i);
      data[i * 8 + 3] = normAttr.getX(i);
      data[i * 8 + 4] = normAttr.getY(i);
      data[i * 8 + 5] = normAttr.getZ(i);
      data[i * 8 + 6] = uvAttr ? uvAttr.getX(i) : 0;
      data[i * 8 + 7] = uvAttr ? uvAttr.getY(i) : 0;
    }

    const vertexBuf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuf, 0, data);

    let indexData: Uint32Array;
    if (geometry.index) {
      const raw = geometry.index.array as ArrayLike<number>;
      indexData = raw instanceof Uint32Array ? raw : new Uint32Array(raw);
    } else {
      indexData = new Uint32Array(count);
      for (let i = 0; i < count; i++) indexData[i] = i;
    }

    const indexBuf = this.device.createBuffer({
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(indexBuf, 0, indexData);

    const groups =
      geometry.groups.length > 0
        ? geometry.groups
        : [{ start: 0, count: indexData.length, materialIndex: 0 }];

    const submeshes: RegisteredSubmesh[] = [];
    for (const g of groups) {
      const indexCount = Math.min(g.count, indexData.length - g.start);
      if (indexCount <= 0) continue;

      const mat = materials[g.materialIndex ?? 0];
      submeshes.push({
        indexStart: g.start,
        indexCount,
        materialBindGroup: mat
          ? this.getMaterialBindGroup(mat)
          : this.defaultMaterialBindGroup,
      });
    }

    this.meshRegistry.set(name, {
      vertex: vertexBuf,
      index: indexBuf,
      indexCount: indexData.length,
      submeshes,
    });
  }
  private getMaterialBindGroup(mat: ModelMaterial): GPUBindGroup {
    const cached = this.materialCache.get(mat);
    if (cached) return cached;

    const tex = mat.map ? this.getTexture(mat.map) : null;

    const uniform = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      uniform,
      0,
      new Float32Array([
        mat.color[0],
        mat.color[1],
        mat.color[2],
        1,
        mat.tintWithObjectColor ? 1 : 0,
        0,
        0,
        0,
      ]),
    );

    const sampler = this.device.createSampler({
      addressModeU: mat.wrapS,
      addressModeV: mat.wrapT,
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      maxAnisotropy: 8,
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.materialLayout,
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: (tex ?? this.whiteTexture).createView() },
      ],
    });

    this.materialCache.set(mat, bindGroup);
    return bindGroup;
  }
  private getTexture(src: TextureSource): GPUTexture | null {
    const cached = this.textureCache.get(src);
    if (cached) return cached;

    const anySrc = src as any;
    const w: number = anySrc.naturalWidth || anySrc.width || 0;
    const h: number = anySrc.naturalHeight || anySrc.height || 0;
    const maxDim = this.device.limits.maxTextureDimension2D;
    if (!w || !h || w > maxDim || h > maxDim) {
      console.warn(`Skipping texture with unsupported size ${w}x${h}`);
      return null;
    }

    try {
      const mipCount = Math.floor(Math.log2(Math.max(w, h))) + 1;
      const texture = this.device.createTexture({
        size: [w, h],
        format: "rgba8unorm",
        mipLevelCount: mipCount,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });

      this.device.queue.copyExternalImageToTexture(
        { source: src },
        { texture, mipLevel: 0 },
        [w, h],
      );

      let prev: CanvasImageSource = src;
      let mw = w;
      let mh = h;
      for (let level = 1; level < mipCount; level++) {
        mw = Math.max(1, mw >> 1);
        mh = Math.max(1, mh >> 1);

        const canvas: HTMLCanvasElement | OffscreenCanvas =
          typeof OffscreenCanvas !== "undefined"
            ? new OffscreenCanvas(mw, mh)
            : Object.assign(document.createElement("canvas"), {
                width: mw,
                height: mh,
              });
        const ctx = canvas.getContext("2d") as
          | CanvasRenderingContext2D
          | OffscreenCanvasRenderingContext2D
          | null;
        if (!ctx) break;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(prev, 0, 0, mw, mh);

        this.device.queue.copyExternalImageToTexture(
          { source: canvas },
          { texture, mipLevel: level },
          [mw, mh],
        );
        prev = canvas;
      }

      this.textureCache.set(src, texture);
      return texture;
    } catch (err) {
      console.warn("Failed to upload model texture:", err);
      return null;
    }
  }
  private createPipelines(): void {
    const mod = this.device.createShaderModule({ code: sceneShaderWGSL });
    const skyMod = this.device.createShaderModule({ code: skyShaderWGSL });
    const terrainMod = this.device.createShaderModule({
      code: terrainShaderWGSL,
    });
    const lightMeshMod = this.device.createShaderModule({
      code: lightObjectWGSL,
    });

    const frameLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    const objectLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true,
            minBindingSize: 64,
          },
        },
      ],
    });

    const lightLayout = this.device.createBindGroupLayout({
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
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float" },
        },
      ],
    });
    this.lightLayout = lightLayout;

    this.lightObjectFrameLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.materialLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        },
      ],
    });

    this.whiteTexture = this.device.createTexture({
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.whiteTexture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );
    this.defaultMaterialBindGroup = this.getMaterialBindGroup(
      createDefaultMaterial(),
    );

    const layout = this.device.createPipelineLayout({
      bindGroupLayouts: [
        frameLayout,
        objectLayout,
        lightLayout,
        this.materialLayout,
      ],
    });

    this.frameUniform = this.device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.frameBindGroup = this.device.createBindGroup({
      layout: frameLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameUniform } }],
    });

    this.objectUniform = this.device.createBuffer({
      size: this.objectStride * MAX_OBJECT_SLOTS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.objectBindGroup = this.device.createBindGroup({
      layout: objectLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.objectUniform, offset: 0, size: 64 },
        },
      ],
    });

    this.lightUniform = this.device.createBuffer({
      size: 144,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.lightObjectFrameBind = this.device.createBindGroup({
      layout: this.lightObjectFrameLayout,
      entries: [{ binding: 0, resource: { buffer: this.lightUniform } }],
    });

    const skyLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.skyUniform = this.device.createBuffer({
      size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.skyBindGroup = this.device.createBindGroup({
      layout: skyLayout,
      entries: [{ binding: 0, resource: { buffer: this.skyUniform } }],
    });

    const terrainLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.terrainUniform = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.terrainBindGroup = this.device.createBindGroup({
      layout: terrainLayout,
      entries: [{ binding: 0, resource: { buffer: this.terrainUniform } }],
    });

    const depthStencil: GPUDepthStencilState = {
      format: "depth32float",
      depthWriteEnabled: true,
      depthCompare: "less",
    };

    const baseTargets: GPUColorTargetState[] = [{ format: this.format }];

    this.meshPipeline = this.device.createRenderPipeline({
      layout,
      vertex: {
        module: mod,
        entryPoint: "mesh_vs",
        buffers: MESH_VERTEX_BUFFERS,
      },
      fragment: { module: mod, entryPoint: "mesh_fs", targets: baseTargets },
      depthStencil,
      multisample: { count: SAMPLE_COUNT },
      primitive: { topology: "triangle-list", cullMode: "back" },
    });

    this.skyPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [skyLayout],
      }),
      vertex: { module: skyMod, entryPoint: "sky_vs" },
      fragment: {
        module: skyMod,
        entryPoint: "sky_fs",
        targets: [{ format: this.format }],
      },
      depthStencil: {
        format: "depth32float",
        depthWriteEnabled: false,
        depthCompare: "always",
      },
      multisample: { count: SAMPLE_COUNT },
      primitive: { topology: "triangle-list" },
    });

    this.terrainPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [frameLayout, terrainLayout, lightLayout],
      }),
      vertex: {
        module: terrainMod,
        entryPoint: "terrain_vs",
        buffers: MESH_VERTEX_BUFFERS,
      },
      fragment: {
        module: terrainMod,
        entryPoint: "terrain_fs",
        targets: baseTargets,
      },
      depthStencil,
      multisample: { count: SAMPLE_COUNT },
      primitive: { topology: "triangle-list", cullMode: "back" },
    });

    this.lightMeshPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.lightObjectFrameLayout, objectLayout],
      }),
      vertex: {
        module: lightMeshMod,
        entryPoint: "light_vs",
        buffers: MESH_VERTEX_BUFFERS,
      },
      fragment: {
        module: lightMeshMod,
        entryPoint: "light_fs",
        targets: [{ format: "r32float" }],
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
    });
  }
  public resize(width: number, height: number): void {
    this.depthWidth = Math.max(1, Math.floor(width));
    this.depthHeight = Math.max(1, Math.floor(height));

    this.depthTexture?.destroy();
    this.msaaTexture?.destroy();

    this.depthTexture = this.device.createTexture({
      size: [this.depthWidth, this.depthHeight],
      format: "depth24plus",
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.msaaTexture = this.device.createTexture({
      size: [this.depthWidth, this.depthHeight],
      format: this.format,
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  public getDepthView(): GPUTextureView {
    return this.depthTexture.createView();
  }
  public getMSAAView(): GPUTextureView {
    return this.msaaTexture.createView();
  }
  public getMeshVertexBuffer(id: string): GPUBuffer | null {
    return this.meshRegistry.get(id)?.vertex ?? null;
  }
  public updateFrame(
    viewProjMatrix: Float32Array,
    cameraPos: [number, number, number],
  ): void {
    const data = new Float32Array(24);
    data.set(viewProjMatrix, 0);
    data[16] = cameraPos[0];
    data[17] = cameraPos[1];
    data[18] = cameraPos[2];
    this.device.queue.writeBuffer(this.frameUniform, 0, data);

    const sunLen = Math.hypot(0.45, 1.0, 0.35);
    const invVP = new THREE.Matrix4().fromArray(viewProjMatrix).invert();

    const sky = new Float32Array(40);
    sky.set(invVP.elements, 0);
    sky[16] = 0.45 / sunLen;
    sky[17] = 1.0 / sunLen;
    sky[18] = 0.35 / sunLen;
    sky[19] = 0.0;
    sky[20] = 1.0;
    sky[21] = 0.96;
    sky[22] = 0.86;
    sky[23] = 0.0;
    sky[24] = 0.1;
    sky[25] = 0.22;
    sky[26] = 0.45;
    sky[27] = 1.0;
    sky[28] = 0.55;
    sky[29] = 0.68;
    sky[30] = 0.82;
    sky[31] = 1.0;
    sky[32] = 0.18;
    sky[33] = 0.2;
    sky[34] = 0.22;
    sky[35] = 1.0;
    sky[36] = this.time;
    sky[37] = 0.55;
    sky[38] = 1.15;
    sky[39] = 0.0;

    this.device.queue.writeBuffer(this.skyUniform, 0, sky);
  }
  private writeObject(
    slot: number,
    scale: [number, number, number],
    translate: [number, number, number],
    color: [number, number, number, number],
    rotation: [number, number, number, number] = [0, 0, 0, 1],
  ): void {
    const data = new Float32Array([
      scale[0],
      scale[1],
      scale[2],
      0,
      translate[0],
      translate[1],
      translate[2],
      0,
      color[0],
      color[1],
      color[2],
      color[3],
      rotation[0],
      rotation[1],
      rotation[2],
      rotation[3],
    ]);

    this.device.queue.writeBuffer(
      this.objectUniform,
      slot * this.objectStride,
      data,
    );
  }
  public encodeLightObjects(pass: GPURenderPassEncoder): void {
    for (let i = 0; i < this.objectVisuals.length; i++) {
      const o = this.objectVisuals[i];
      if (!o.visible) continue;
      const slot = 3 + i;
      this.writeObject(
        slot,
        [o.scale, o.scale, o.scale],
        o.position,
        o.color,
        o.quaternion,
      );
    }

    pass.setPipeline(this.lightMeshPipeline);
    pass.setBindGroup(0, this.lightObjectFrameBind);

    for (let i = 0; i < this.objectVisuals.length; i++) {
      const o = this.objectVisuals[i];
      if (!o.visible) continue;
      const slot = 3 + i;
      pass.setBindGroup(1, this.objectBindGroup, [slot * this.objectStride]);

      const mesh = o.meshId ? this.meshRegistry.get(o.meshId) : null;
      if (!mesh) continue;

      pass.setVertexBuffer(0, mesh.vertex);
      pass.setIndexBuffer(mesh.index, "uint32");
      pass.drawIndexed(mesh.indexCount);
    }
  }
  public encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.skyPipeline);
    pass.setBindGroup(0, this.skyBindGroup);
    pass.draw(3);

    for (let i = 0; i < this.objectVisuals.length; i++) {
      const o = this.objectVisuals[i];
      if (!o.visible) continue;

      const slot = 3 + i;
      this.writeObject(
        slot,
        [o.scale, o.scale, o.scale],
        o.position,
        o.color,
        o.quaternion,
      );
    }

    pass.setBindGroup(0, this.frameBindGroup);

    if (this.terrainMeshId) {
      const mesh = this.meshRegistry.get(this.terrainMeshId);
      if (mesh) {
        const tData = new Float32Array([
          this.boundsQuat[0],
          this.boundsQuat[1],
          this.boundsQuat[2],
          this.boundsQuat[3],
          this.time,
          this.waterLevel,
          this.causticStrength,
          this.fogDensity,
        ]);
        this.device.queue.writeBuffer(this.terrainUniform, 0, tData);

        pass.setPipeline(this.terrainPipeline);
        pass.setBindGroup(1, this.terrainBindGroup);
        if (this.lightBindGroup) pass.setBindGroup(2, this.lightBindGroup);
        pass.setVertexBuffer(0, mesh.vertex);
        pass.setIndexBuffer(mesh.index, "uint32");
        pass.drawIndexed(mesh.indexCount);
      }
    }

    for (let i = 0; i < this.objectVisuals.length; i++) {
      const o = this.objectVisuals[i];
      if (!o.visible) continue;

      const slot = 3 + i;
      pass.setBindGroup(1, this.objectBindGroup, [slot * this.objectStride]);

      const mesh = o.meshId ? this.meshRegistry.get(o.meshId) : null;
      if (!mesh) continue;

      pass.setPipeline(this.meshPipeline);
      if (this.lightBindGroup) pass.setBindGroup(2, this.lightBindGroup);
      pass.setVertexBuffer(0, mesh.vertex);
      pass.setIndexBuffer(mesh.index, "uint32");
      for (const sm of mesh.submeshes) {
        pass.setBindGroup(3, sm.materialBindGroup);
        pass.drawIndexed(sm.indexCount, 1, sm.indexStart);
      }
    }
  }
}
