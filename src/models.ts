import * as THREE from "three";

export interface ModelDef {
  id: string;
  label?: string;
  build?: () => THREE.BufferGeometry;
  url?: string;
}

export const MODELS: Record<string, ModelDef> = {
  sphere: {
    id: "sphere",
    label: "Sphere",
    build: () => new THREE.SphereGeometry(1, 32, 16),
  },
  box: {
    id: "box",
    label: "Box",
    build: () => new THREE.BoxGeometry(1, 1, 1),
  },
  torusKnot: {
    id: "torusKnot",
    label: "Torus Knot",
    build: () => new THREE.TorusKnotGeometry(1, 0.35, 160, 24),
  },
  bunny: { id: "bunny", label: "Bunny", url: "/models/bunny.glb" },
  boat: { id: "boat", label: "Boat", url: "/models/boat.glb" },
  rock: { id: "rock", label: "Rock", url: "/models/rock.glb" },
  crate: { id: "crate", label: "Crate", url: "/models/crate.glb" },
};

export const MODEL_IDS = Object.keys(MODELS);

export interface ModelDefaults {
  densityRatio: number;
  drag: number;
  angularDrag: number;
  wobbleDamping?: number;
}

export const MODEL_DEFAULTS: Record<string, ModelDefaults> = {
  sphere: { densityRatio: 2.5, drag: 3.0, angularDrag: 2.0 },
  box: { densityRatio: 0.25, drag: 6.0, angularDrag: 3.0 },
  torusKnot: { densityRatio: 1.0, drag: 12.0, angularDrag: 8.0 },
  bunny: { densityRatio: 0.6, drag: 5.0, angularDrag: 3.5 },
  boat: {
    densityRatio: 0.35,
    drag: 2.0,
    angularDrag: 1.2,
    wobbleDamping: 20.0,
  },
  rock: { densityRatio: 2.2, drag: 10.0, angularDrag: 6.0 },
  crate: { densityRatio: 0.4, drag: 7.0, angularDrag: 4.0 },
};
