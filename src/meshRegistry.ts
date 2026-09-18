import * as THREE from "three";
import type { FluidSimulationGPU, BakedMeshHandle } from "./simulation";
import type { SceneRenderer } from "./sceneRenderer";
import { MODELS, type ModelDef } from "./models";
import { loadModel, type LoadedModel } from "./modelLoader";

export const MESH_BAKE_RESOLUTION = 96;
export const MESH_BAKE_PADDING = 0.5;

export class MeshRegistry {
  private readonly simulation: FluidSimulationGPU;
  private readonly sceneRenderer: SceneRenderer;
  private geometries = new Map<string, THREE.BufferGeometry>();
  private loaded = new Map<string, LoadedModel>();
  private loadPromises = new Map<string, Promise<LoadedModel | null>>();
  constructor(simulation: FluidSimulationGPU, sceneRenderer: SceneRenderer) {
    this.simulation = simulation;
    this.sceneRenderer = sceneRenderer;
  }
  ensure(id: string): Promise<LoadedModel | null> {
    const existing = this.loaded.get(id);
    if (existing) return Promise.resolve(existing);

    const inflight = this.loadPromises.get(id);
    if (inflight) return inflight;

    const def = MODELS[id];
    if (!def) return Promise.resolve(null);

    const promise = this.runLoad(id, def);
    this.loadPromises.set(id, promise);
    return promise;
  }
  private async runLoad(
    id: string,
    def: ModelDef,
  ): Promise<LoadedModel | null> {
    try {
      const model = await loadModel(def);
      this.loaded.set(id, model);
      this.geometries.set(id, model.geometry);
      this.sceneRenderer.registerMesh(id, model.geometry);

      this.simulation
        .bakeMesh(model.geometry, MESH_BAKE_RESOLUTION, MESH_BAKE_PADDING)
        .catch((err) => console.error(`SDF bake failed for "${id}":`, err));

      return model;
    } catch (err) {
      console.error(`Failed to load model "${id}":`, err);
      return null;
    } finally {
      this.loadPromises.delete(id);
    }
  }
  async preloadAll(): Promise<void> {
    await Promise.all(Object.keys(MODELS).map((id) => this.ensure(id)));
  }
  registerTerrain(id: string, geometry: THREE.BufferGeometry): void {
    if (this.geometries.has(id)) return;
    this.geometries.set(id, geometry);
    this.sceneRenderer.registerMesh(id, geometry);
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
  getLoaded(id: string): LoadedModel | null {
    return this.loaded.get(id) ?? null;
  }
}
