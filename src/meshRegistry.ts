import * as THREE from "three";
import type { FluidSimulationGPU, BakedMeshHandle } from "./simulation";
import type { SceneRenderer } from "./sceneRenderer";

export const MESH_BAKE_RESOLUTION = 64;
export const MESH_BAKE_PADDING = 0.75;

export class MeshRegistry {
  private readonly simulation: FluidSimulationGPU;
  private readonly sceneRenderer: SceneRenderer;
  private geometries = new Map<string, THREE.BufferGeometry>();
  constructor(simulation: FluidSimulationGPU, sceneRenderer: SceneRenderer) {
    this.simulation = simulation;
    this.sceneRenderer = sceneRenderer;
  }
  register(id: string, geometry: THREE.BufferGeometry): void {
    if (this.geometries.has(id)) return;

    this.geometries.set(id, geometry);
    this.sceneRenderer.registerMesh(id, geometry);
  }
  ensure(id: string): void {
    const geometry = this.geometries.get(id);
    if (!geometry) return;

    this.simulation
      .bakeMesh(geometry, MESH_BAKE_RESOLUTION, MESH_BAKE_PADDING)
      .catch((err) => console.error(`SDF bake failed for "${id}":`, err));
  }
  get(id: string): BakedMeshHandle | null {
    const geometry = this.geometries.get(id);
    if (!geometry) return null;

    return this.simulation.getCachedMesh(
      geometry,
      MESH_BAKE_RESOLUTION,
      MESH_BAKE_PADDING,
    );
  }
}
