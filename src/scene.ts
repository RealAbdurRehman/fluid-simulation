import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const app = document.getElementById("app")!;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x515151);
scene.fog = new THREE.FogExp2(0x06080e, 0.015);

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);
camera.position.set(0, 10, 26);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
  precision: "highp",
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.3;

renderer.domElement.style.position = "absolute";
renderer.domElement.style.top = "0";
renderer.domElement.style.left = "0";
renderer.domElement.style.width = "100vw";
renderer.domElement.style.height = "100vh";
renderer.domElement.style.zIndex = "0";

app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxDistance = 80;
controls.minDistance = 5;

const ambientLight = new THREE.AmbientLight(0x223355, 1.5);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0x66ccff, 2.0);
dirLight.position.set(15, 25, 15);
scene.add(dirLight);

const blueFill = new THREE.PointLight(0x0088ff, 3.0, 50);
blueFill.position.set(-10, -5, -10);
scene.add(blueFill);

const containerGroup = new THREE.Group();
scene.add(containerGroup);

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const wireframeGeo = new THREE.EdgesGeometry(boxGeo);
const boundsMaterial = new THREE.LineBasicMaterial({
  color: 0x00e5ff,
  transparent: true,
  opacity: 0.75,
  linewidth: 2,
});
const bounds = new THREE.LineSegments(wireframeGeo, boundsMaterial);
containerGroup.add(bounds);

const basePlaneGeo = new THREE.PlaneGeometry(1, 1);
const basePlaneMat = new THREE.MeshBasicMaterial({
  color: 0x071526,
  transparent: true,
  opacity: 0.6,
  side: THREE.DoubleSide,
});
const basePlate = new THREE.Mesh(basePlaneGeo, basePlaneMat);
basePlate.rotation.x = Math.PI / 2;
containerGroup.add(basePlate);

const grid = new THREE.GridHelper(60, 40, 0x00e5ff, 0x112233);
grid.position.y = -8;
(grid.material as THREE.Material).transparent = true;
(grid.material as THREE.Material).opacity = 0.4;
scene.add(grid);

window.addEventListener("resize", () => {
  const width = window.innerWidth;
  const height = window.innerHeight;

  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});

function setCameraDistance(distance: number): void {
  const dir = camera.position.clone().normalize();
  camera.position.copy(dir.multiplyScalar(distance));
  controls.update();
}

function setBoundsSize(width: number, height: number, depth: number): void {
  bounds.scale.set(width, height, depth);
  basePlate.scale.set(width, depth, 1);
  basePlate.position.y = -height / 2;
  grid.position.y = -height / 2 - 0.05;
}

export { scene, renderer, camera, controls, setCameraDistance, setBoundsSize };
