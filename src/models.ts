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
};

export const MODEL_IDS = Object.keys(MODELS);
