import * as THREE from "three";

import { scene, camera, renderer } from "./scene";
import { update, start } from "./simulation";

let accumulator = 0;
const FIXED_DELTA = 1 / 120;
const MAX_FRAME_DELTA = 0.1;

const time = new THREE.Timer();

start();

function animate(timestamp: number): void {
  time.update(timestamp);

  const delta = Math.min(time.getDelta(), MAX_FRAME_DELTA);
  accumulator += delta;

  while (accumulator >= FIXED_DELTA) {
    update(FIXED_DELTA);
    accumulator -= FIXED_DELTA;
  }

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);
