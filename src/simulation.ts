import * as THREE from "three";
import { config } from "./config";
import { fluidComputeShaderWGSL } from "./shader/fluidCompute.wgsl";
import { GPUShaderStage, GPUBufferUsage } from "./types";

interface BitonicStep {
  k: number;
  j: number;
}

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

  private externalForcesPipeline!: GPUComputePipeline;
  private updateSpatialHashPipeline!: GPUComputePipeline;
  private bitonicSortPipeline!: GPUComputePipeline;
  private clearStartIndicesPipeline!: GPUComputePipeline;
  private calculateStartIndicesPipeline!: GPUComputePipeline;
  private calculateDensitiesPipeline!: GPUComputePipeline;
  private calculateForcesPipeline!: GPUComputePipeline;
  private integratePositionsPipeline!: GPUComputePipeline;

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
      console.error("Failed to acquire high-performance WebGPU adapter.");
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

    const mainBindGroupLayout = this.device.createBindGroupLayout({
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
      bindGroupLayouts: [mainBindGroupLayout],
    });

    const bitonicPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [mainBindGroupLayout, this.bitonicBindGroupLayout],
    });

    this.externalForcesPipeline = this.device.createComputePipeline({
      layout: mainPipelineLayout,
      compute: { module: shaderModule, entryPoint: "externalForces" },
    });
    this.updateSpatialHashPipeline = this.device.createComputePipeline({
      layout: mainPipelineLayout,
      compute: { module: shaderModule, entryPoint: "updateSpatialHash" },
    });
    this.bitonicSortPipeline = this.device.createComputePipeline({
      layout: bitonicPipelineLayout,
      compute: { module: shaderModule, entryPoint: "bitonicSort" },
    });
    this.clearStartIndicesPipeline = this.device.createComputePipeline({
      layout: mainPipelineLayout,
      compute: { module: shaderModule, entryPoint: "clearStartIndices" },
    });
    this.calculateStartIndicesPipeline = this.device.createComputePipeline({
      layout: mainPipelineLayout,
      compute: { module: shaderModule, entryPoint: "calculateStartIndices" },
    });
    this.calculateDensitiesPipeline = this.device.createComputePipeline({
      layout: mainPipelineLayout,
      compute: { module: shaderModule, entryPoint: "calculateDensities" },
    });
    this.calculateForcesPipeline = this.device.createComputePipeline({
      layout: mainPipelineLayout,
      compute: { module: shaderModule, entryPoint: "calculateForces" },
    });
    this.integratePositionsPipeline = this.device.createComputePipeline({
      layout: mainPipelineLayout,
      compute: { module: shaderModule, entryPoint: "integratePositions" },
    });
  }
  private allocateFixedBuffers(): void {
    this.maxPaddedCount = this.nextPowerOfTwo(config.maxParticles);
    const particleByteSize = 64;

    this.particlesBuffer = this.device.createBuffer({
      size: this.maxPaddedCount * particleByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.spatialLookupBuffer = this.device.createBuffer({
      size: this.maxPaddedCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.startIndicesBuffer = this.device.createBuffer({
      size: this.maxPaddedCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.simParamsBuffer = this.device.createBuffer({
      size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const maxBitonicPasses =
      (Math.log2(this.maxPaddedCount) * (Math.log2(this.maxPaddedCount) + 1)) /
      2;
    const totalBitonicBytes = Math.max(
      256,
      maxBitonicPasses * this.uniformAlignment,
    );

    this.bitonicParamsBuffer = this.device.createBuffer({
      size: totalBitonicBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.mainBindGroup = this.device.createBindGroup({
      layout: this.externalForcesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.simParamsBuffer } },
        { binding: 1, resource: { buffer: this.particlesBuffer } },
        { binding: 2, resource: { buffer: this.spatialLookupBuffer } },
        { binding: 3, resource: { buffer: this.startIndicesBuffer } },
      ],
    });

    this.bitonicBindGroup = this.device.createBindGroup({
      layout: this.bitonicBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.bitonicParamsBuffer,
            offset: 0,
            size: 16,
          },
        },
      ],
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
    const numParticles = config.numParticles;
    const initialData = new Float32Array(this.paddedParticlesCount * 16);

    const margin = config.particleSize * 1.5;
    const usableWidth = config.boundsWidth - margin * 2;
    const usableHeight = config.boundsHeight - margin * 2;
    const usableDepth = config.boundsDepth - margin * 2;

    const perAxis = Math.max(1, Math.ceil(Math.cbrt(numParticles)));
    const countX = perAxis;
    const countY = perAxis;
    const countZ = perAxis;

    const baseSpacing = config.particleSize * 1.8 + config.particleSpacing;
    const maxSpacingX = countX > 1 ? usableWidth / (countX - 1) : baseSpacing;
    const maxSpacingY = countY > 1 ? usableHeight / (countY - 1) : baseSpacing;
    const maxSpacingZ = countZ > 1 ? usableDepth / (countZ - 1) : baseSpacing;
    const spacing = Math.min(
      baseSpacing,
      maxSpacingX,
      maxSpacingY,
      maxSpacingZ,
    );

    const gridWidthX = (countX - 1) * spacing;
    const gridWidthZ = (countZ - 1) * spacing;

    const startX = -gridWidthX / 2;
    const startY = -config.boundsHeight / 2 + margin;
    const startZ = -gridWidthZ / 2;

    const halfBoundX = config.boundsWidth / 2 - margin;
    const halfBoundY = config.boundsHeight / 2 - margin;
    const halfBoundZ = config.boundsDepth / 2 - margin;

    for (let i = 0; i < numParticles; i++) {
      const xIdx = i % countX;
      const zIdx = Math.floor(i / countX) % countZ;
      const yIdx = Math.floor(i / (countX * countZ));

      let x = startX + xIdx * spacing;
      let y = startY + yIdx * spacing;
      let z = startZ + zIdx * spacing;

      x = THREE.MathUtils.clamp(x, -halfBoundX, halfBoundX);
      y = THREE.MathUtils.clamp(y, -halfBoundY, halfBoundY);
      z = THREE.MathUtils.clamp(z, -halfBoundZ, halfBoundZ);

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
  public updateUniforms(subDelta: number): void {
    const r = config.smoothingRadius;
    const r6 = Math.pow(r, 6);
    const r9 = Math.pow(r, 9);

    const poly6Factor = 315 / (64 * Math.PI * r9);
    const spikyGradFactor = 45 / (Math.PI * r6);
    const nearSpikyGradFactor = 45 / (Math.PI * r6);
    const viscFactor = 45 / (Math.PI * r6);

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
    f32[17] = poly6Factor;
    f32[18] = spikyGradFactor;
    f32[19] = nearSpikyGradFactor;

    f32[20] = viscFactor;
    f32[21] = 0;
    f32[22] = 0;
    f32[23] = 0;

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
  public recordStepCommands(
    commandEncoder: GPUCommandEncoder,
    deltaTime: number,
    force: boolean = false,
  ): void {
    if (!this.isInitialized || (config.paused && !force)) return;

    const substeps = Math.max(1, config.substeps);
    const subDelta = deltaTime / substeps;

    for (let step = 0; step < substeps; step++) {
      this.updateUniforms(subDelta);
      const pass = commandEncoder.beginComputePass();
      const workgroupCount = Math.ceil(this.paddedParticlesCount / 256);

      pass.setBindGroup(0, this.mainBindGroup);

      pass.setPipeline(this.externalForcesPipeline);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.updateSpatialHashPipeline);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.bitonicSortPipeline);
      for (let s = 0; s < this.bitonicSteps.length; s++) {
        pass.setBindGroup(1, this.bitonicBindGroup, [
          s * this.uniformAlignment,
        ]);
        pass.dispatchWorkgroups(workgroupCount);
      }

      pass.setPipeline(this.clearStartIndicesPipeline);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.calculateStartIndicesPipeline);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.calculateDensitiesPipeline);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.calculateForcesPipeline);
      pass.dispatchWorkgroups(workgroupCount);

      pass.setPipeline(this.integratePositionsPipeline);
      pass.dispatchWorkgroups(workgroupCount);

      pass.end();
    }
  }
}
