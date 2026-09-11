import * as THREE from "three";

import { config } from "./config";
import { scene, camera, renderer } from "./scene";
import { update, start, syncVisuals, setInteraction } from "./simulation";

let accumulator = 0;
const FIXED_DELTA = 1 / 60;
const MAX_FRAME_DELTA = 0.1;

const time = new THREE.Timer();

let interactionButton: number | null = null;

const mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
const simulationPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const mouseWorld3D = new THREE.Vector3();
const mouseWorld = new THREE.Vector2();

function updateMousePosition(event: PointerEvent): void {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  raycaster.ray.intersectPlane(simulationPlane, mouseWorld3D);
  mouseWorld.set(mouseWorld3D.x, mouseWorld3D.y);
}

start();

function animate(timestamp: number): void {
  time.update(timestamp);

  const delta = Math.min(time.getDelta(), MAX_FRAME_DELTA);
  accumulator += delta;

  while (accumulator >= FIXED_DELTA) {
    update(FIXED_DELTA);
    accumulator -= FIXED_DELTA;
  }

  syncVisuals();

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);

window.addEventListener("contextmenu", (event) => event.preventDefault());

window.addEventListener("pointermove", (event) => {
  updateMousePosition(event);
  if (interactionButton === 0)
    setInteraction(mouseWorld, -config.interactionStrength);
  else if (interactionButton === 2)
    setInteraction(mouseWorld, config.interactionStrength);
});

window.addEventListener("pointerdown", (event) => {
  interactionButton = event.button;
  updateMousePosition(event);

  if (event.button === 0)
    setInteraction(mouseWorld, -config.interactionStrength);
  else if (event.button === 2)
    setInteraction(mouseWorld, config.interactionStrength);
});

window.addEventListener("pointerup", () => {
  interactionButton = null;
  setInteraction(mouseWorld, 0);
});
