import * as THREE from "three";

const app = document.getElementById("app")!;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x090b12);

const camera = new THREE.PerspectiveCamera(
  70,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
  precision: "highp",
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

renderer.domElement.style.position = "absolute";
renderer.domElement.style.top = "0";
renderer.domElement.style.left = "0";
renderer.domElement.style.width = "100vw";
renderer.domElement.style.height = "100vh";
renderer.domElement.style.zIndex = "0";

app.appendChild(renderer.domElement);

window.addEventListener("resize", () => {
  const width = window.innerWidth;
  const height = window.innerHeight;

  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
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
  linewidth: 2,
});

const bounds = new THREE.Line(boundsGeometry, boundsMaterial);
bounds.frustumCulled = false;
scene.add(bounds);

function setCameraDistance(distance: number): void {
  camera.position.z = distance;
}

function setBoundsSize(width: number, height: number): void {
  bounds.scale.set(width / 2, height / 2, 1);
}

export { scene, renderer, camera, setCameraDistance, setBoundsSize };
