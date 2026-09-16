import * as THREE from "three";

import { acesFilmicWGSL } from "./shader/common.wgsl";
import { GPUBufferUsage, GPUShaderStage, GPUTextureUsage } from "./types";

const SAMPLE_COUNT = 4;
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

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

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

struct VIn { @location(0) position: vec3<f32>, };
struct VOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@vertex
fn vs_main(input: VIn) -> VOut {
  let scaled = input.position * obj.scale.xyz;
  let rotated = qRotateVec(obj.rotation, scaled);
  let world = rotated + obj.translate.xyz;
  var out: VOut;
  out.position = frame.viewProj * vec4<f32>(world, 1.0);
  out.worldPos = world;
  return out;
}

@fragment
fn fs_main(input: VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(acesFilmic(obj.color.rgb * 1.1), obj.color.a);
}

struct MeshVIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
};
struct MeshVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
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
  return out;
}

@fragment
fn mesh_fs(input: MeshVOut) -> @location(0) vec4<f32> {
  let n = normalize(input.worldNormal);
  let lightDir = normalize(vec3<f32>(0.45, 1.0, 0.35));
  let ndl = max(dot(n, lightDir), 0.0);

  var lit = obj.color.rgb * (0.28 + ndl * 0.95);

  let viewDir = normalize(frame.cameraPos.xyz - input.worldPos);
  let rim = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0) * 0.55;
  lit += vec3<f32>(0.35, 0.65, 1.0) * rim;

  let halfDir = normalize(lightDir + viewDir);
  lit += vec3<f32>(pow(max(dot(n, halfDir), 0.0), 48.0) * 0.4);

  return vec4<f32>(acesFilmic(lit * 1.3), obj.color.a);
}
`;

const skyShaderWGSL = /* wgsl */ `
struct SkyUniforms {
  topColor: vec4<f32>,
  horizonColor: vec4<f32>,
  bottomColor: vec4<f32>,
  resolution: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sky: SkyUniforms;

@vertex
fn sky_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  return vec4<f32>(pos[vi], 1.0, 1.0);
}

${acesFilmicWGSL}

@fragment
fn sky_fs(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
  let t = clamp(fragPos.y / sky.resolution.y, 0.0, 1.0);
  var c: vec3<f32>;
  if (t < 0.5) {
    c = mix(sky.topColor.rgb, sky.horizonColor.rgb, t * 2.0);
  } else {
    c = mix(sky.horizonColor.rgb, sky.bottomColor.rgb, (t - 0.5) * 2.0);
  }
  return vec4<f32>(acesFilmic(c * 1.3), 1.0);
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

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> terrain: TerrainUniforms;

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
  return out;
}

@fragment
fn terrain_fs(input: VOut) -> @location(0) vec4<f32> {
  let n = normalize(input.worldNormal);
  let up = max(n.y, 0.0);
  let slope = 1.0 - up;

  let wp3 = input.localPos;
  let wp = wp3.xz;

  let t = terrain.time;

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
  
  let warpA = fbm(wp * 0.55 + vec2<f32>(t * 0.06, t * 0.04), 3);
  let phaseA = (wp.x * 0.85 + wp.y * 0.50) * 8.0 + warpA * 10.0 + t * 0.20;
  let rippleA = sin(phaseA) * 0.5 + 0.5;

  let warpB = fbm(wp * 0.80 + vec2<f32>(-t * 0.05, t * 0.07) + 30.0, 3);
  let phaseB = (wp.x * -0.45 + wp.y * 0.95) * 13.0 + warpB * 7.0 + t * 0.14;
  let rippleB = sin(phaseB) * 0.5 + 0.5;

  let regionMask = smoothstep(
    0.35,
    0.65,
    fbm(wp * 0.06 + vec2<f32>(t * 0.008, -t * 0.006) + 100.0, 3)
  );
  let ripple = mix(rippleA, rippleB, regionMask);

  let rippleMask = smoothstep(0.35, 0.90, up);
  albedo *= 0.94 + ripple * 0.10 * rippleMask;

  let grain = fbm3(wp3 * 8.0, 2);
  albedo *= 0.96 + grain * 0.08;

  albedo = mix(albedo, sandMid, slope * 0.5);

  let sunDir = normalize(vec3<f32>(0.35, 0.75, 0.30));
  let ndl = max(dot(n, sunDir), 0.0);
  let halfLambert = ndl * 0.7 + 0.30;
  let sunCol = vec3<f32>(1.02, 0.98, 0.92);
  var lit = albedo * halfLambert * sunCol * 1.65;

  let depth = clamp(-input.localPos.y * 0.10, 0.0, 1.0);
  lit = mix(lit, lit * vec3<f32>(0.72, 0.66, 0.56), depth * 0.40);

  let toCam = frame.cameraPos.xyz - input.worldPos;
  let viewDir = toCam / max(length(toCam), 1e-4);
  let rim = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0) * 0.05;
  lit += vec3<f32>(0.85, 0.80, 0.72) * rim;

  return vec4<f32>(acesFilmic(lit), 1.0);
}
`;

function buildWireBox(): Float32Array {
  const c = [
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5,
    -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
  ];
  const edges = [
    0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
  ];
  const out = new Float32Array(edges.length * 3);
  let p = 0;
  for (const i of edges) {
    out[p++] = c[i * 3];
    out[p++] = c[i * 3 + 1];
    out[p++] = c[i * 3 + 2];
  }
  return out;
}

function buildWireSphere(segments = 16, rings = 10): Float32Array {
  const lines: number[] = [];
  for (let r = 1; r < rings; r++) {
    const phi = (r / rings) * Math.PI;
    const y = Math.cos(phi) * 0.5;
    const rad = Math.sin(phi) * 0.5;
    for (let s = 0; s < segments; s++) {
      const a0 = (s / segments) * Math.PI * 2;
      const a1 = ((s + 1) / segments) * Math.PI * 2;
      lines.push(Math.cos(a0) * rad, y, Math.sin(a0) * rad);
      lines.push(Math.cos(a1) * rad, y, Math.sin(a1) * rad);
    }
  }
  for (let s = 0; s < segments; s++) {
    const th = (s / segments) * Math.PI * 2;
    for (let r = 0; r < rings; r++) {
      const phi0 = (r / rings) * Math.PI;
      const phi1 = ((r + 1) / rings) * Math.PI;
      lines.push(
        Math.sin(phi0) * Math.cos(th) * 0.5,
        Math.cos(phi0) * 0.5,
        Math.sin(phi0) * Math.sin(th) * 0.5,
      );
      lines.push(
        Math.sin(phi1) * Math.cos(th) * 0.5,
        Math.cos(phi1) * 0.5,
        Math.sin(phi1) * Math.sin(th) * 0.5,
      );
    }
  }
  return new Float32Array(lines);
}

export interface ObjectVisual {
  visible: boolean;
  shape: "box" | "sphere" | "mesh";
  meshId?: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  scale: number;
  color: [number, number, number, number];
}

const BLEND: GPUBlendState = {
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
};

const LINE_VERTEX_BUFFERS: GPUVertexBufferLayout[] = [
  {
    arrayStride: 12,
    stepMode: "vertex",
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
  },
];

const MESH_VERTEX_BUFFERS: GPUVertexBufferLayout[] = [
  {
    arrayStride: 24,
    stepMode: "vertex",
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
    ],
  },
];

interface RegisteredMesh {
  vertex: GPUBuffer;
  index: GPUBuffer;
  indexCount: number;
}

export class SceneRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private linePipeline!: GPURenderPipeline;
  private meshPipeline!: GPURenderPipeline;
  private skyPipeline!: GPURenderPipeline;
  private terrainPipeline!: GPURenderPipeline;

  private frameUniform!: GPUBuffer;
  private frameBindGroup!: GPUBindGroup;
  private objectUniform!: GPUBuffer;
  private objectBindGroup!: GPUBindGroup;
  private objectStride = 256;

  private skyUniform!: GPUBuffer;
  private skyBindGroup!: GPUBindGroup;

  private terrainUniform!: GPUBuffer;
  private terrainBindGroup!: GPUBindGroup;

  private boxBuf!: GPUBuffer;
  private sphereBuf!: GPUBuffer;
  private boxVerts = 0;
  private sphereVerts = 0;

  private meshRegistry = new Map<string, RegisteredMesh>();

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
    this.createGeometries();
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
  public registerMesh(name: string, geometry: THREE.BufferGeometry): void {
    if (this.meshRegistry.has(name)) return;
    if (!geometry.attributes.normal) geometry.computeVertexNormals();

    const posAttr = geometry.attributes.position as THREE.BufferAttribute;
    const normAttr = geometry.attributes.normal as THREE.BufferAttribute;
    const count = posAttr.count;

    const data = new Float32Array(count * 6);
    for (let i = 0; i < count; i++) {
      data[i * 6 + 0] = posAttr.getX(i);
      data[i * 6 + 1] = posAttr.getY(i);
      data[i * 6 + 2] = posAttr.getZ(i);
      data[i * 6 + 3] = normAttr.getX(i);
      data[i * 6 + 4] = normAttr.getY(i);
      data[i * 6 + 5] = normAttr.getZ(i);
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

    this.meshRegistry.set(name, {
      vertex: vertexBuf,
      index: indexBuf,
      indexCount: indexData.length,
    });
  }
  private createPipelines(): void {
    const mod = this.device.createShaderModule({ code: sceneShaderWGSL });
    const skyMod = this.device.createShaderModule({ code: skyShaderWGSL });
    const terrainMod = this.device.createShaderModule({
      code: terrainShaderWGSL,
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

    const layout = this.device.createPipelineLayout({
      bindGroupLayouts: [frameLayout, objectLayout],
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
      size: 64,
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
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    };

    const baseTargets: GPUColorTargetState[] = [
      { format: this.format, blend: BLEND },
    ];

    this.linePipeline = this.device.createRenderPipeline({
      layout,
      vertex: {
        module: mod,
        entryPoint: "vs_main",
        buffers: LINE_VERTEX_BUFFERS,
      },
      fragment: { module: mod, entryPoint: "fs_main", targets: baseTargets },
      depthStencil,
      multisample: { count: SAMPLE_COUNT },
      primitive: { topology: "line-list" },
    });

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
        format: "depth24plus",
        depthWriteEnabled: false,
        depthCompare: "always",
      },
      multisample: { count: SAMPLE_COUNT },
      primitive: { topology: "triangle-list" },
    });

    this.terrainPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [frameLayout, terrainLayout],
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
  }
  private createGeometries(): void {
    const box = buildWireBox();
    this.boxVerts = box.length / 3;
    this.boxBuf = this.makeVertexBuffer(box);

    const sphere = buildWireSphere(16, 10);
    this.sphereVerts = sphere.length / 3;
    this.sphereBuf = this.makeVertexBuffer(sphere);
  }
  private makeVertexBuffer(data: Float32Array): GPUBuffer {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
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

    const sky = new Float32Array([
      0.015,
      0.025,
      0.055,
      1,
      0.045,
      0.075,
      0.12,
      1,
      0.09,
      0.11,
      0.15,
      1,
      this.depthWidth,
      this.depthHeight,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.skyUniform, 0, sky);
  }
  public getDepthView(): GPUTextureView {
    return this.depthTexture.createView();
  }
  public getMSAAView(): GPUTextureView {
    return this.msaaTexture.createView();
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
  public encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.skyPipeline);
    pass.setBindGroup(0, this.skyBindGroup);
    pass.draw(3);

    for (let i = 0; i < this.objectVisuals.length; i++) {
      const o = this.objectVisuals[i];
      if (!o.visible) continue;

      const slot = 3 + i;
      const s = o.shape === "mesh" ? o.scale : o.scale * 2;
      this.writeObject(slot, [s, s, s], o.position, o.color, o.quaternion);
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

      const mesh =
        o.shape === "mesh" && o.meshId ? this.meshRegistry.get(o.meshId) : null;

      if (mesh) {
        pass.setPipeline(this.meshPipeline);
        pass.setVertexBuffer(0, mesh.vertex);
        pass.setIndexBuffer(mesh.index, "uint32");
        pass.drawIndexed(mesh.indexCount);
      } else if (o.shape === "sphere") {
        pass.setPipeline(this.linePipeline);
        pass.setVertexBuffer(0, this.sphereBuf);
        pass.draw(this.sphereVerts);
      } else {
        pass.setPipeline(this.linePipeline);
        pass.setVertexBuffer(0, this.boxBuf);
        pass.draw(this.boxVerts);
      }
    }
  }
}
