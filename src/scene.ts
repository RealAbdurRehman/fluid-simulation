import * as THREE from "three";

import { config } from "./config";

const app = document.getElementById("app")!;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x090b12);

const camera = new THREE.PerspectiveCamera(
  70,
  innerWidth / innerHeight,
  1,
  1000,
);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
  precision: "highp",
});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

app.appendChild(renderer.domElement);

window.addEventListener("resize", () => {
  const width = innerWidth,
    height = innerHeight;

  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});

const boundsGeometry = new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(-1, -1, 0),
  new THREE.Vector3(1, -1, 0),
  new THREE.Vector3(1, 1, 0),
  new THREE.Vector3(-1, 1, 0),
  new THREE.Vector3(-1, -1, 0),
]);
const boundsMaterial = new THREE.LineBasicMaterial({
  color: 0x71b579,
});
const bounds = new THREE.Line(boundsGeometry, boundsMaterial);
scene.add(bounds);

const particle = new THREE.Mesh(
  new THREE.CircleGeometry(),
  new THREE.MeshBasicMaterial({ color: 0x0278fd }),
);

function setCameraDistance(distance: number): void {
  camera.position.z = distance;
}

function setBoundsSize(width: number, height: number): void {
  bounds.scale.set(width / 2, height / 2, 1);
}

setCameraDistance(config.cameraDistance);
setBoundsSize(config.boundsWidth, config.boundsHeight);

export { scene, renderer, camera, particle, setCameraDistance, setBoundsSize };
