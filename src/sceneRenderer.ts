import { config } from "./config";
import { acesFilmicWGSL } from "./shader/common.wgsl";
import { GPUBufferUsage, GPUShaderStage, GPUTextureUsage } from "./types";

const SAMPLE_COUNT = 4;

const sceneShaderWGSL = /* wgsl */ `
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
};

struct ObjectUniforms {
  scale: vec4<f32>,
  translate: vec4<f32>,
  color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

struct VIn {
  @location(0) position: vec3<f32>,
};

struct VOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@vertex
fn vs_main(input: VIn) -> VOut {
  let world = input.position * obj.scale.xyz + obj.translate.xyz;
  var out: VOut;
  out.position = frame.viewProj * vec4<f32>(world, 1.0);
  out.worldPos = world;
  return out;
}

fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input: VOut) -> @location(0) vec4<f32> {
  let d = distance(input.worldPos, frame.cameraPos.xyz);
  let fade = clamp((90.0 - d) / 70.0, 0.0, 1.0);
  let linear = obj.color.rgb * 1.1;
  let mapped = acesFilmic(linear);
  return vec4<f32>(mapped, obj.color.a * fade);
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

function buildGrid(size: number, divisions: number): Float32Array {
  const half = size / 2;
  const step = size / divisions;
  const lines: number[] = [];
  for (let i = 0; i <= divisions; i++) {
    const t = -half + i * step;
    lines.push(-half, 0, t, half, 0, t);
    lines.push(t, 0, -half, t, 0, half);
  }
  return new Float32Array(lines);
}

function buildPlate(): Float32Array {
  return new Float32Array([
    -0.5, 0, -0.5, 0.5, 0, -0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, 0.5, -0.5,
    0, 0.5,
  ]);
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

const VERTEX_BUFFERS: GPUVertexBufferLayout[] = [
  {
    arrayStride: 12,
    stepMode: "vertex",
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
  },
];

export class SceneRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private linePipeline!: GPURenderPipeline;
  private triPipeline!: GPURenderPipeline;
  private skyPipeline!: GPURenderPipeline;

  private frameUniform!: GPUBuffer;
  private frameBindGroup!: GPUBindGroup;
  private objectUniform!: GPUBuffer;
  private objectBindGroup!: GPUBindGroup;
  private objectStride = 256;

  private skyUniform!: GPUBuffer;
  private skyBindGroup!: GPUBindGroup;

  private boxBuf!: GPUBuffer;
  private gridBuf!: GPUBuffer;
  private plateBuf!: GPUBuffer;
  private boxVerts = 0;
  private gridVerts = 0;
  private plateVerts = 0;

  private depthTexture!: GPUTexture;
  private msaaTexture!: GPUTexture;
  private depthWidth = 1;
  private depthHeight = 1;
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
  private createPipelines(): void {
    const mod = this.device.createShaderModule({ code: sceneShaderWGSL });
    const skyMod = this.device.createShaderModule({ code: skyShaderWGSL });

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
            minBindingSize: 48,
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
      size: this.objectStride * 3,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.objectBindGroup = this.device.createBindGroup({
      layout: objectLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.objectUniform, offset: 0, size: 48 },
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
        buffers: VERTEX_BUFFERS,
      },
      fragment: { module: mod, entryPoint: "fs_main", targets: baseTargets },
      depthStencil,
      multisample: { count: SAMPLE_COUNT },
      primitive: { topology: "line-list" },
    });

    this.triPipeline = this.device.createRenderPipeline({
      layout,
      vertex: {
        module: mod,
        entryPoint: "vs_main",
        buffers: VERTEX_BUFFERS,
      },
      fragment: { module: mod, entryPoint: "fs_main", targets: baseTargets },
      depthStencil,
      multisample: { count: SAMPLE_COUNT },
      primitive: { topology: "triangle-list" },
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
  }
  private createGeometries(): void {
    const box = buildWireBox();
    this.boxVerts = box.length / 3;
    this.boxBuf = this.makeVertexBuffer(box);

    const grid = buildGrid(60, 40);
    this.gridVerts = grid.length / 3;
    this.gridBuf = this.makeVertexBuffer(grid);

    const plate = buildPlate();
    this.plateVerts = plate.length / 3;
    this.plateBuf = this.makeVertexBuffer(plate);
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

    if (this.depthTexture) this.depthTexture.destroy();
    if (this.msaaTexture) this.msaaTexture.destroy();

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

    const sky = new Float32Array(16);
    sky[0] = 0.015;
    sky[1] = 0.025;
    sky[2] = 0.055;
    sky[3] = 1;
    sky[4] = 0.045;
    sky[5] = 0.075;
    sky[6] = 0.12;
    sky[7] = 1;
    sky[8] = 0.09;
    sky[9] = 0.11;
    sky[10] = 0.15;
    sky[11] = 1;
    sky[12] = this.depthWidth;
    sky[13] = this.depthHeight;
    sky[14] = 0;
    sky[15] = 0;
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
    data[19] = 0;
    this.device.queue.writeBuffer(this.frameUniform, 0, data);
  }
  private writeObject(
    slot: number,
    scale: [number, number, number],
    translate: [number, number, number],
    color: [number, number, number, number],
  ): void {
    const data = new Float32Array(12);
    data[0] = scale[0];
    data[1] = scale[1];
    data[2] = scale[2];
    data[3] = 0;
    data[4] = translate[0];
    data[5] = translate[1];
    data[6] = translate[2];
    data[7] = 0;
    data[8] = color[0];
    data[9] = color[1];
    data[10] = color[2];
    data[11] = color[3];
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

    const bw = config.boundsWidth;
    const bh = config.boundsHeight;
    const bd = config.boundsDepth;

    this.writeObject(0, [bw, bh, bd], [0, 0, 0], [0.0, 0.95, 1.0, 0.9]);
    this.writeObject(1, [bw, 1, bd], [0, -bh / 2, 0], [0.03, 0.09, 0.18, 0.7]);
    this.writeObject(
      2,
      [60, 1, 60],
      [0, -bh / 2 - 0.05, 0],
      [0.0, 0.85, 1.0, 0.5],
    );

    pass.setBindGroup(0, this.frameBindGroup);

    pass.setPipeline(this.triPipeline);
    pass.setBindGroup(1, this.objectBindGroup, [1 * this.objectStride]);
    pass.setVertexBuffer(0, this.plateBuf);
    pass.draw(this.plateVerts);

    pass.setPipeline(this.linePipeline);
    pass.setBindGroup(1, this.objectBindGroup, [2 * this.objectStride]);
    pass.setVertexBuffer(0, this.gridBuf);
    pass.draw(this.gridVerts);

    pass.setBindGroup(1, this.objectBindGroup, [0 * this.objectStride]);
    pass.setVertexBuffer(0, this.boxBuf);
    pass.draw(this.boxVerts);
  }
}
