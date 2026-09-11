import * as THREE from "three";

import { scene } from "./scene";

const MAX_PARTICLES = 4000;

const vertexShader = /* glsl */ `
  attribute float aSpeed;

  uniform float uMinSpeed;
  uniform float uMaxSpeed;

  varying vec2 vUv;
  varying vec3 vColor;
 
  vec3 speedToColor(float t) {
    t = clamp(t, 0.0, 1.0);

    float r = clamp(1.35 - abs(4.0 * t - 3.0), 0.0, 1.0);
    float g = clamp(1.35 - abs(4.0 * t - 2.0), 0.0, 1.0);
    float b = clamp(1.9 - abs(4.0 * t - 1.0), 0.0, 1.0);

    g += 0.45 * (1.0 - smoothstep(0.0, 0.3, t));
    g = clamp(g, 0.0, 1.0);

    r += 0.65 * smoothstep(0.75, 1.0, t);
    r = clamp(r, 0.0, 1.0);

    vec3 color = vec3(r, g, b);
    color = mix(color, vec3(1.0), 0.25);

    return color;
  }

  void main() {
    vUv = uv;

    float t = (aSpeed - uMinSpeed) / max(uMaxSpeed - uMinSpeed, 1e-5);
    vColor = speedToColor(t);

    vec4 worldPosition = instanceMatrix * vec4(position, 1.0);
    vec4 mvPosition = modelViewMatrix * worldPosition;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vColor;

  void main() {
    vec2 centered = vUv * 2.0 - 1.0;
    float dist = length(centered);

    float aa = fwidth(dist);
    float alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);
    if (alpha <= 0.001) discard;

    gl_FragColor = vec4(vColor, alpha);
  }
`;

const geometry = new THREE.PlaneGeometry(2, 2);

const speeds = new Float32Array(MAX_PARTICLES);
const speedAttribute = new THREE.InstancedBufferAttribute(speeds, 1);
speedAttribute.setUsage(THREE.DynamicDrawUsage);
geometry.setAttribute("aSpeed", speedAttribute);

const material = new THREE.ShaderMaterial({
  vertexShader,
  fragmentShader,
  uniforms: {
    uMinSpeed: { value: 0 },
    uMaxSpeed: { value: 6 },
  },
  transparent: true,
  depthWrite: false,
});

const instancedMesh = new THREE.InstancedMesh(
  geometry,
  material,
  MAX_PARTICLES,
);
instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
instancedMesh.count = 0;
instancedMesh.frustumCulled = false;
scene.add(instancedMesh);

const dummy = new THREE.Object3D();
let particleScale = 0.1;

function setParticleCount(count: number): void {
  instancedMesh.count = Math.min(count, MAX_PARTICLES);
}

function setParticleSize(size: number): void {
  particleScale = size;
}

function setSpeedRange(min: number, max: number): void {
  material.uniforms.uMinSpeed.value = min;
  material.uniforms.uMaxSpeed.value = max;
}

function updateParticle(
  index: number,
  x: number,
  y: number,
  speed: number,
): void {
  dummy.position.set(x, y, 0);
  dummy.scale.setScalar(particleScale);
  dummy.updateMatrix();
  instancedMesh.setMatrixAt(index, dummy.matrix);
  speeds[index] = speed;
}

function commitParticles(): void {
  instancedMesh.instanceMatrix.needsUpdate = true;
  speedAttribute.needsUpdate = true;
}

export {
  MAX_PARTICLES,
  setParticleCount,
  setParticleSize,
  setSpeedRange,
  updateParticle,
  commitParticles,
};
