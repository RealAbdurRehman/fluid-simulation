import * as THREE from "three";

import { config } from "./config";
import { fluidComputeShaderWGSL } from "./shader/fluidCompute.wgsl";
import { GPUShaderStage, GPUBufferUsage, GPUMapMode } from "./types";

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
  private stagingBuffer!: GPUBuffer;

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

  private isMapping = false;
  private cachedParticleData = new Float32Array(0);

  private interactionPosition = new THREE.Vector2();
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
    const particleByteSize = 32;

    this.particlesBuffer = this.device.createBuffer({
      size: this.maxPaddedCount * particleByteSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
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
      size: 112,
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

    this.stagingBuffer = this.device.createBuffer({
      size: this.maxPaddedCount * particleByteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.cachedParticleData = new Float32Array(this.maxPaddedCount * 8);

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
    const initialData = new Float32Array(this.paddedParticlesCount * 8);

    const halfBoundX = config.boundsWidth / 2 - config.particleSize * 1.5;
    const halfBoundY = config.boundsHeight / 2 - config.particleSize * 1.5;

    const availableWidth = halfBoundX * 2;
    const availableHeight = halfBoundY * 2;

    const aspectRatio = Math.max(0.1, availableWidth / availableHeight);
    let particlesPerRow = Math.max(
      1,
      Math.round(Math.sqrt(numParticles * aspectRatio)),
    );
    let particlesPerCol = Math.ceil(numParticles / particlesPerRow);

    const maxSpacingX = availableWidth / Math.max(1, particlesPerRow);
    const maxSpacingY = availableHeight / Math.max(1, particlesPerCol);
    const spacing = Math.min(
      config.particleSize * 2 + config.particleSpacing,
      Math.min(maxSpacingX, maxSpacingY) * 0.95,
    );

    const startX = -((particlesPerRow - 1) * spacing) / 2;
    const startY = -((particlesPerCol - 1) * spacing) / 2;

    for (let i = 0; i < numParticles; i++) {
      const col = i % particlesPerRow;
      const row = Math.floor(i / particlesPerRow);

      let x = startX + col * spacing;
      let y = startY + row * spacing;

      x = THREE.MathUtils.clamp(x, -halfBoundX, halfBoundX);
      y = THREE.MathUtils.clamp(y, -halfBoundY, halfBoundY);

      const offset = i * 8;
      initialData[offset + 0] = x;
      initialData[offset + 1] = y;
      initialData[offset + 2] = x;
      initialData[offset + 3] = y;
      initialData[offset + 4] = 0;
      initialData[offset + 5] = 0;
      initialData[offset + 6] = 0;
      initialData[offset + 7] = 0;
    }

    this.device.queue.writeBuffer(this.particlesBuffer, 0, initialData);
  }
  public updateUniforms(subDelta: number): void {
    const r = config.smoothingRadius;
    const r4 = Math.pow(r, 4);
    const r5 = Math.pow(r, 5);

    const poly6Factor = 6 / (Math.PI * r4);
    const spikyGradFactor = 12 / (Math.PI * r4);
    const nearSpikyGradFactor = 30 / (Math.PI * r5);
    const viscFactor = 6 / (Math.PI * r4);

    const buffer = new ArrayBuffer(112);
    const f32 = new Float32Array(buffer);
    const u32 = new Uint32Array(buffer);

    f32[0] = config.boundsWidth;
    f32[1] = config.boundsHeight;
    f32[2] = config.gravity;
    f32[3] = config.collisionDamping;

    f32[4] = config.targetDensity;
    f32[5] = config.pressureMultiplier;
    f32[6] = config.nearDensityMultiplier;
    f32[7] = config.viscosityStrength;

    f32[8] = config.smoothingRadius;
    f32[9] = 1.0;
    f32[10] = config.particleSize;
    f32[11] = subDelta;

    u32[12] = config.numParticles;
    u32[13] = this.paddedParticlesCount;
    f32[14] = config.interactionRadius;
    f32[15] = this.interactionStrength;

    f32[16] = this.interactionPosition.x;
    f32[17] = this.interactionPosition.y;
    f32[18] = poly6Factor;
    f32[19] = spikyGradFactor;

    f32[20] = nearSpikyGradFactor;
    f32[21] = viscFactor;
    f32[22] = 0;
    f32[23] = 0;

    this.device.queue.writeBuffer(this.simParamsBuffer, 0, buffer);
  }
  public step(deltaTime: number): void {
    if (!this.isInitialized || config.paused) return;

    const substeps = Math.max(1, config.substeps);
    const subDelta = deltaTime / substeps;
    for (let step = 0; step < substeps; step++) {
      this.updateUniforms(subDelta);

      const commandEncoder = this.device.createCommandEncoder();
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
      this.device.queue.submit([commandEncoder.finish()]);
    }
  }
  public async fetchParticleData(): Promise<Float32Array> {
    if (!this.isInitialized || this.isMapping) return this.cachedParticleData;

    this.isMapping = true;
    try {
      const particleByteSize = 32;
      const readSize = config.numParticles * particleByteSize;

      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(
        this.particlesBuffer,
        0,
        this.stagingBuffer,
        0,
        readSize,
      );
      this.device.queue.submit([commandEncoder.finish()]);

      await this.stagingBuffer.mapAsync(GPUMapMode.READ, 0, readSize);

      if (this.stagingBuffer.mapState === "mapped") {
        const mappedArray = new Float32Array(
          this.stagingBuffer.getMappedRange(0, readSize),
        );
        this.cachedParticleData.set(mappedArray);
        this.stagingBuffer.unmap();
      }
    } catch {
    } finally {
      this.isMapping = false;
    }

    return this.cachedParticleData;
  }
  public setInteraction(pos: THREE.Vector2, strength: number): void {
    this.interactionPosition.copy(pos);
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
  ): void {
    if (!this.isInitialized || config.paused) return;

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
