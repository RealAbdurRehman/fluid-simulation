import * as THREE from "three";

import { config } from "./config";
import { fluidComputeShaderWGSL } from "./shader/fluidCompute.wgsl";
import { GPUShaderStage, GPUBufferUsage, GPUMapMode } from "./types";
import { bakeSignedDistanceFieldAsync } from "./sdfBaker";

interface BitonicStep {
  k: number;
  j: number;
}

export interface BakedMeshHandle {
  dataOffset: number;
  dims: THREE.Vector3;
  origin: THREE.Vector3;
  cellSize: number;
}

export interface ObjectDescriptor {
  type: "box" | "sphere" | "mesh";
  position: THREE.Vector3;
  quaternion?: THREE.Quaternion;
  size?: THREE.Vector3;
  mesh?: BakedMeshHandle;
  velocity?: THREE.Vector3;
  restitution?: number;
}

const IDENTITY_QUAT = new THREE.Quaternion();
const ZERO_VEC = new THREE.Vector3();

const MAX_COLLIDERS = 9;
const FLOATS_PER_COLLIDER = 24;
const SDF_BUFFER_FLOATS = 3_000_000;

const MAX_PROBES = 128;
const PROBE_BYTES = 16;

const COMPUTE_ENTRY_POINTS = [
  "externalForces",
  "updateSpatialHash",
  "bitonicSort",
  "clearStartIndices",
  "calculateStartIndices",
  "calculateDensities",
  "calculateForces",
  "integratePositions",
  "sampleProbes",
] as const;

type ComputeEntryPoint = (typeof COMPUTE_ENTRY_POINTS)[number];

export class FluidSimulationGPU {
  private device!: GPUDevice;
  private isInitialized = false;

  private maxPaddedCount = 0;
  private paddedParticlesCount = 0;
  private bitonicSteps: BitonicStep[] = [];
  private uniformAlignment = 256;

  private particlesBuffer!: GPUBuffer;
  private spatialLookupBuffer!: GPUBuffer;
  private startIndicesBuffer!: GPUBuffer;
  private simParamsBuffer!: GPUBuffer;
  private bitonicParamsBuffer!: GPUBuffer;
  private collidersBuffer!: GPUBuffer;
  private collidersData = new Float32Array(MAX_COLLIDERS * FLOATS_PER_COLLIDER);

  private objects: ObjectDescriptor[] = [];
  private numColliders = 1;
  private boundsQuaternion = new THREE.Quaternion();

  private sdfDataBuffer!: GPUBuffer;
  private sdfCursor = 0;
  private meshBakeCache = new Map<string, BakedMeshHandle>();
  private meshBakePromises = new Map<string, Promise<BakedMeshHandle>>();

  private probeInputBuffer!: GPUBuffer;
  private probeOutputBuffer!: GPUBuffer;
  private probeReadbackBuffer!: GPUBuffer;

  private probeHasFreshData = false;
  private probeMapInFlight = false;
  private probeCount = 0;
  private latestProbes: Float32Array | null = null;

  private pipelines: Record<ComputeEntryPoint, GPUComputePipeline> =
    {} as never;

  private mainBindGroup!: GPUBindGroup;
  private bitonicBindGroup!: GPUBindGroup;
  private bitonicBindGroupLayout!: GPUBindGroupLayout;

  private rayOrigin = new THREE.Vector3();
  private rayDir = new THREE.Vector3();
  private interactionStrength = 0;
  public async initialize(): Promise<boolean> {
    if (!navigator.gpu) {
      console.error("WebGPU is not supported by your browser/device.");
      return false;
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      console.error("Failed to acquire WebGPU adapter.");
      return false;
    }

    this.device = await adapter.requestDevice();
    this.uniformAlignment = Math.max(
      256,
      this.device.limits.minUniformBufferOffsetAlignment || 256,
    );

    this.createPipelines();
    this.allocateFixedBuffers();
    this.updateParticleCount();
    this.isInitialized = true;
    return true;
  }
  private nextPowerOfTwo(n: number): number {
    return Math.pow(2, Math.ceil(Math.log2(Math.max(n, 2))));
  }
  private createPipelines(): void {
    const shaderModule = this.device.createShaderModule({
      code: fluidComputeShaderWGSL,
    });

    const mainLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 7,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    this.bitonicBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true,
            minBindingSize: 16,
          },
        },
      ],
    });

    const mainPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [mainLayout],
    });

    const bitonicPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [mainLayout, this.bitonicBindGroupLayout],
    });

    for (const entry of COMPUTE_ENTRY_POINTS) {
      const layout =
        entry === "bitonicSort" ? bitonicPipelineLayout : mainPipelineLayout;
      this.pipelines[entry] = this.device.createComputePipeline({
        layout,
        compute: { module: shaderModule, entryPoint: entry },
      });
    }
  }
  private allocateFixedBuffers(): void {
    this.maxPaddedCount = this.nextPowerOfTwo(config.maxParticles);

    this.particlesBuffer = this.createBuffer(
      this.maxPaddedCount * 64,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "particles",
    );
    this.spatialLookupBuffer = this.createBuffer(
      this.maxPaddedCount * 8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "spatialLookup",
    );
    this.startIndicesBuffer = this.createBuffer(
      this.maxPaddedCount * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "startIndices",
    );
    this.simParamsBuffer = this.createBuffer(
      160,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      "simParams",
    );

    const maxBitonicPasses =
      (Math.log2(this.maxPaddedCount) * (Math.log2(this.maxPaddedCount) + 1)) /
      2;
    this.bitonicParamsBuffer = this.createBuffer(
      Math.max(256, maxBitonicPasses * this.uniformAlignment),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      "bitonicParams",
    );

    this.collidersBuffer = this.createBuffer(
      MAX_COLLIDERS * FLOATS_PER_COLLIDER * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "colliders",
    );

    this.sdfDataBuffer = this.createBuffer(
      SDF_BUFFER_FLOATS * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "sdfData",
    );

    this.probeInputBuffer = this.createBuffer(
      MAX_PROBES * PROBE_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "probeInput",
    );
    this.probeOutputBuffer = this.createBuffer(
      MAX_PROBES * PROBE_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      "probeOutput",
    );
    this.probeReadbackBuffer = this.createBuffer(
      MAX_PROBES * PROBE_BYTES,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      "probeReadback",
    );

    this.mainBindGroup = this.device.createBindGroup({
      layout: this.pipelines.externalForces.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simParamsBuffer } },
        { binding: 1, resource: { buffer: this.particlesBuffer } },
        { binding: 2, resource: { buffer: this.spatialLookupBuffer } },
        { binding: 3, resource: { buffer: this.startIndicesBuffer } },
        { binding: 4, resource: { buffer: this.collidersBuffer } },
        { binding: 5, resource: { buffer: this.sdfDataBuffer } },
        { binding: 6, resource: { buffer: this.probeInputBuffer } },
        { binding: 7, resource: { buffer: this.probeOutputBuffer } },
      ],
    });

    this.bitonicBindGroup = this.device.createBindGroup({
      layout: this.bitonicBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.bitonicParamsBuffer, offset: 0, size: 16 },
        },
      ],
    });
  }
  private createBuffer(
    size: number,
    usage: GPUBufferUsageFlags,
    label?: string,
  ): GPUBuffer {
    return this.device.createBuffer({
      label,
      size,
      usage,
    });
  }
  public updateParticleCount(): void {
    this.paddedParticlesCount = this.nextPowerOfTwo(config.numParticles);

    this.bitonicSteps = [];
    for (let k = 2; k <= this.paddedParticlesCount; k <<= 1)
      for (let j = k >> 1; j > 0; j >>= 1) this.bitonicSteps.push({ k, j });

    const bitonicDataArray = new Uint32Array(
      (this.bitonicSteps.length * this.uniformAlignment) / 4,
    );

    for (let s = 0; s < this.bitonicSteps.length; s++) {
      const u32Offset = (s * this.uniformAlignment) / 4;
      bitonicDataArray[u32Offset + 0] = this.bitonicSteps[s].k;
      bitonicDataArray[u32Offset + 1] = this.bitonicSteps[s].j;
      bitonicDataArray[u32Offset + 2] = this.paddedParticlesCount;
      bitonicDataArray[u32Offset + 3] = 0;
    }

    this.device.queue.writeBuffer(
      this.bitonicParamsBuffer,
      0,
      bitonicDataArray,
    );

    this.initParticleGrid();
  }
  public initParticleGrid(): void {
    const n = config.numParticles;
    const initialData = new Float32Array(this.paddedParticlesCount * 16);

    const margin = config.particleSize * 1.5;
    const usableWidth = config.boundsWidth - margin * 2;
    const usableHeight = config.boundsHeight - margin * 2;
    const usableDepth = config.boundsDepth - margin * 2;

    const perAxis = Math.max(1, Math.ceil(Math.cbrt(n)));
    const countX = perAxis;
    const countY = perAxis;
    const countZ = perAxis;

    const baseSpacing = config.particleSize * 1.8 + config.particleSpacing;
    const spacing = Math.min(
      baseSpacing,
      countX > 1 ? usableWidth / (countX - 1) : baseSpacing,
      countY > 1 ? usableHeight / (countY - 1) : baseSpacing,
      countZ > 1 ? usableDepth / (countZ - 1) : baseSpacing,
    );

    const gridWidthX = (countX - 1) * spacing;
    const gridWidthZ = (countZ - 1) * spacing;

    const startX = -gridWidthX / 2;
    const startY = -config.boundsHeight / 2 + margin;
    const startZ = -gridWidthZ / 2;

    const halfBoundX = config.boundsWidth / 2 - margin;
    const halfBoundY = config.boundsHeight / 2 - margin;
    const halfBoundZ = config.boundsDepth / 2 - margin;

    for (let i = 0; i < n; i++) {
      const xIdx = i % countX;
      const zIdx = Math.floor(i / countX) % countZ;
      const yIdx = Math.floor(i / (countX * countZ));

      const x = THREE.MathUtils.clamp(
        startX + xIdx * spacing,
        -halfBoundX,
        halfBoundX,
      );
      const y = THREE.MathUtils.clamp(
        startY + yIdx * spacing,
        -halfBoundY,
        halfBoundY,
      );
      const z = THREE.MathUtils.clamp(
        startZ + zIdx * spacing,
        -halfBoundZ,
        halfBoundZ,
      );

      const offset = i * 16;
      initialData[offset + 0] = x;
      initialData[offset + 1] = y;
      initialData[offset + 2] = z;
      initialData[offset + 3] = 1.0;

      initialData[offset + 4] = x;
      initialData[offset + 5] = y;
      initialData[offset + 6] = z;
      initialData[offset + 7] = 1.0;
    }

    this.device.queue.writeBuffer(this.particlesBuffer, 0, initialData);
  }
  public setBoundsRotation(quaternion: THREE.Quaternion): void {
    this.boundsQuaternion.copy(quaternion);
  }
  public getBoundsQuaternion(): THREE.Quaternion {
    return this.boundsQuaternion;
  }
  public setObjects(objects: ObjectDescriptor[]): void {
    this.objects = objects.slice(0, MAX_COLLIDERS - 1);
  }
  public setInteraction(
    rayOrigin: THREE.Vector3,
    rayDir: THREE.Vector3,
    strength: number,
  ): void {
    this.rayOrigin.copy(rayOrigin);
    this.rayDir.copy(rayDir).normalize();
    this.interactionStrength = strength;
  }
  public getDevice(): GPUDevice {
    return this.device;
  }
  public getParticlesBuffer(): GPUBuffer {
    return this.particlesBuffer;
  }
  public setProbes(positions: Float32Array, count: number): void {
    this.probeCount = Math.min(count, MAX_PROBES);
    if (this.probeCount === 0) return;
    this.device.queue.writeBuffer(
      this.probeInputBuffer,
      0,
      positions,
      0,
      this.probeCount * 4,
    );
  }
  public getLatestProbes(): Float32Array | null {
    return this.latestProbes;
  }
  public pollProbeResults(): void {
    if (this.probeCount === 0) return;
    if (this.probeMapInFlight) return;
    if (!this.probeHasFreshData) return;

    const byteCount = this.probeCount * PROBE_BYTES;

    this.probeHasFreshData = false;
    this.probeMapInFlight = true;

    this.probeReadbackBuffer
      .mapAsync(GPUMapMode.READ, 0, byteCount)
      .then(() => {
        const mapped = this.probeReadbackBuffer.getMappedRange(0, byteCount);

        this.latestProbes = new Float32Array(mapped).slice();
        this.probeReadbackBuffer.unmap();
        this.probeMapInFlight = false;
      })
      .catch((err) => {
        console.warn("Probe readback failed:", err);

        try {
          this.probeReadbackBuffer.unmap();
        } catch {}
        this.probeMapInFlight = false;
      });
  }
  public bakeMesh(
    geometry: THREE.BufferGeometry,
    resolution = 48,
    padding = 0.5,
  ): Promise<BakedMeshHandle> {
    const key = `${geometry.uuid}:${resolution}:${padding}`;

    const cached = this.meshBakeCache.get(key);
    if (cached) return Promise.resolve(cached);

    const inflight = this.meshBakePromises.get(key);
    if (inflight) return inflight;

    const promise = this.runBake(geometry, resolution, padding, key);
    this.meshBakePromises.set(key, promise);
    return promise;
  }
  public getCachedMesh(
    geometry: THREE.BufferGeometry,
    resolution = 48,
    padding = 0.5,
  ): BakedMeshHandle | null {
    return (
      this.meshBakeCache.get(`${geometry.uuid}:${resolution}:${padding}`) ??
      null
    );
  }
  private async runBake(
    geometry: THREE.BufferGeometry,
    resolution: number,
    padding: number,
    key: string,
  ): Promise<BakedMeshHandle> {
    try {
      console.log(`Baking SDF for mesh (resolution ${resolution})...`);
      const t0 = performance.now();
      const baked = await bakeSignedDistanceFieldAsync(
        geometry,
        resolution,
        padding,
      );
      const floatCount = baked.data.length;

      if (this.sdfCursor + floatCount > SDF_BUFFER_FLOATS) {
        throw new Error(
          `SDF data buffer exhausted (need ${floatCount} floats, ` +
            `${SDF_BUFFER_FLOATS - this.sdfCursor} left). Lower the bake ` +
            `resolution or increase SDF_BUFFER_FLOATS in simulation.ts.`,
        );
      }

      const handle: BakedMeshHandle = {
        dataOffset: this.sdfCursor,
        dims: baked.dims,
        origin: baked.origin,
        cellSize: baked.cellSize,
      };

      this.device.queue.writeBuffer(
        this.sdfDataBuffer,
        this.sdfCursor * 4,
        baked.data.buffer,
        baked.data.byteOffset,
        baked.data.byteLength,
      );

      this.sdfCursor += floatCount;
      this.meshBakeCache.set(key, handle);
      console.log(
        `SDF bake complete (${floatCount} cells, ${(performance.now() - t0).toFixed(0)}ms).`,
      );
      return handle;
    } finally {
      this.meshBakePromises.delete(key);
    }
  }
  private writeColliders(): void {
    this.packCollider(
      0,
      0,
      ZERO_VEC,
      this.boundsQuaternion,
      new THREE.Vector3(
        config.boundsWidth / 2,
        config.boundsHeight / 2,
        config.boundsDepth / 2,
      ),
      config.collisionDamping,
      ZERO_VEC,
    );

    let count = 1;
    for (const object of this.objects) {
      const shapeType =
        object.type === "sphere" ? 2 : object.type === "mesh" ? 3 : 1;
      this.packCollider(
        count,
        shapeType,
        object.position,
        object.quaternion ?? IDENTITY_QUAT,
        object.size ?? ZERO_VEC,
        object.restitution ?? 0.5,
        object.velocity ?? ZERO_VEC,
        object.type === "mesh" ? object.mesh : undefined,
      );
      count++;
    }

    this.numColliders = count;
    this.device.queue.writeBuffer(
      this.collidersBuffer,
      0,
      this.collidersData,
      0,
      count * FLOATS_PER_COLLIDER,
    );
  }
  private packCollider(
    index: number,
    shapeType: number,
    position: THREE.Vector3,
    quaternion: THREE.Quaternion,
    size: THREE.Vector3,
    restitution: number,
    velocity: THREE.Vector3,
    mesh?: BakedMeshHandle,
  ): void {
    const o = index * FLOATS_PER_COLLIDER;
    const d = this.collidersData;

    d[o + 0] = position.x;
    d[o + 1] = position.y;
    d[o + 2] = position.z;
    d[o + 3] = shapeType;

    d[o + 4] = quaternion.x;
    d[o + 5] = quaternion.y;
    d[o + 6] = quaternion.z;
    d[o + 7] = quaternion.w;

    d[o + 8] = size.x;
    d[o + 9] = size.y;
    d[o + 10] = size.z;
    d[o + 11] = restitution;

    d[o + 12] = velocity.x;
    d[o + 13] = velocity.y;
    d[o + 14] = velocity.z;
    d[o + 15] = 0.0;

    if (mesh) {
      d[o + 16] = mesh.origin.x;
      d[o + 17] = mesh.origin.y;
      d[o + 18] = mesh.origin.z;
      d[o + 19] = mesh.cellSize;

      d[o + 20] = mesh.dims.x;
      d[o + 21] = mesh.dims.y;
      d[o + 22] = mesh.dims.z;
      d[o + 23] = mesh.dataOffset;
    } else for (let i = 16; i < 24; i++) d[o + i] = 0;
  }
  public updateUniforms(subDelta: number): void {
    this.writeColliders();

    const r = config.smoothingRadius;
    const r6 = Math.pow(r, 6);
    const r9 = Math.pow(r, 9);

    const buffer = new ArrayBuffer(160);
    const f32 = new Float32Array(buffer);
    const u32 = new Uint32Array(buffer);

    f32[0] = config.boundsWidth;
    f32[1] = config.boundsHeight;
    f32[2] = config.boundsDepth;
    f32[3] = config.gravity;

    f32[4] = config.collisionDamping;
    f32[5] = config.targetDensity;
    f32[6] = config.pressureMultiplier;
    f32[7] = config.nearDensityMultiplier;

    f32[8] = config.viscosityStrength;
    f32[9] = config.smoothingRadius;
    f32[10] = 1.0;
    f32[11] = config.particleSize;

    f32[12] = subDelta;
    u32[13] = config.numParticles;
    u32[14] = this.paddedParticlesCount;
    f32[15] = config.interactionRadius;

    f32[16] = this.interactionStrength;
    f32[17] = 315 / (64 * Math.PI * r9);
    f32[18] = 45 / (Math.PI * r6);
    f32[19] = 45 / (Math.PI * r6);

    f32[20] = 45 / (Math.PI * r6);
    u32[21] = this.numColliders;
    u32[22] = this.probeCount;
    f32[23] = config.xsphStrength;

    f32[24] = this.rayOrigin.x;
    f32[25] = this.rayOrigin.y;
    f32[26] = this.rayOrigin.z;
    f32[27] = 0.0;

    f32[28] = this.rayDir.x;
    f32[29] = this.rayDir.y;
    f32[30] = this.rayDir.z;
    f32[31] = 0.0;

    this.device.queue.writeBuffer(this.simParamsBuffer, 0, buffer);
  }
  public recordStepCommands(
    commandEncoder: GPUCommandEncoder,
    deltaTime: number,
  ): void {
    if (!this.isInitialized) return;

    const substeps = Math.max(1, config.substeps);
    const subDelta = deltaTime / substeps;
    const workgroupCount = Math.ceil(this.paddedParticlesCount / 256);
    for (let step = 0; step < substeps; step++) {
      this.updateUniforms(subDelta);

      const pass = commandEncoder.beginComputePass();
      pass.setBindGroup(0, this.mainBindGroup);

      pass.setPipeline(this.pipelines.externalForces);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.pipelines.updateSpatialHash);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.pipelines.bitonicSort);
      for (let s = 0; s < this.bitonicSteps.length; s++) {
        pass.setBindGroup(1, this.bitonicBindGroup, [
          s * this.uniformAlignment,
        ]);
        pass.dispatchWorkgroups(workgroupCount);
      }

      pass.setPipeline(this.pipelines.clearStartIndices);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.pipelines.calculateStartIndices);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.pipelines.calculateDensities);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.pipelines.calculateForces);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.pipelines.integratePositions);
      pass.dispatchWorkgroups(workgroupCount);

      pass.end();
    }

    if (this.probeCount > 0) {
      const pass = commandEncoder.beginComputePass();
      pass.setBindGroup(0, this.mainBindGroup);
      pass.setPipeline(this.pipelines.sampleProbes);
      pass.dispatchWorkgroups(Math.ceil(this.probeCount / 64));
      pass.end();

      if (!this.probeMapInFlight) {
        commandEncoder.copyBufferToBuffer(
          this.probeOutputBuffer,
          0,
          this.probeReadbackBuffer,
          0,
          this.probeCount * PROBE_BYTES,
        );
        this.probeHasFreshData = true;
      }
    }
  }
}
