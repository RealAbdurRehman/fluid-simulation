import * as THREE from "three";

import { config } from "./config";
import { scene, particle } from "./scene";

const DOWN: THREE.Vector2 = new THREE.Vector2(0, -1);

const positions: THREE.Vector2[] = [];
const velocities: THREE.Vector2[] = [];

const particles: THREE.Mesh[] = [];

function getHalfBounds(): THREE.Vector2 {
  return new THREE.Vector2().setScalar(
    config.boundsSize / 2 - config.particleSize,
  );
}

function setParticleGridPosition(): void {
  const numParticles = config.numParticles;
  const particlesPerRow = Math.floor(Math.sqrt(numParticles));
  const particlesPerCol = Math.ceil(numParticles / particlesPerRow);
  const spacing = config.particleSize * 2 + config.particleSpacing;

  for (let i = 0; i < numParticles; i++) {
    const column = i % particlesPerRow;
    const row = Math.floor(i / particlesPerRow);

    let x = (column - particlesPerRow / 2 + 0.5) * spacing;
    let y = (row - particlesPerCol / 2 + 0.5) * spacing;

    const halfBounds = getHalfBounds();
    x = THREE.MathUtils.clamp(x, -halfBounds.x, halfBounds.x);
    y = THREE.MathUtils.clamp(y, -halfBounds.y, halfBounds.y);

    positions[i].set(x, y);
    particles[i].position.set(x, y);
  }
}

function clearParticles(): void {
  for (const particle of particles) scene.remove(particle);

  particles.length = 0;
  positions.length = 0;
  velocities.length = 0;
}

function start(): void {
  clearParticles();
  for (let i = 0; i < config.numParticles; i++) {
    velocities.push(new THREE.Vector2());
    positions.push(new THREE.Vector2());

    const particleInstance = particle.clone();
    particleInstance.scale.setScalar(config.particleSize);
    particles.push(particleInstance);
    scene.add(particleInstance);
  }

  setParticleGridPosition();
}

function resolveCollisions(
  position: THREE.Vector2,
  velocity: THREE.Vector2,
): void {
  const halfBounds = getHalfBounds();
  if (Math.abs(position.x) > halfBounds.x) {
    position.x = halfBounds.x * Math.sign(position.x);
    velocity.x *= -1 * config.collisionDamping;
  }
  if (Math.abs(position.y) > halfBounds.y) {
    position.y = halfBounds.y * Math.sign(position.y);
    velocity.y *= -1 * config.collisionDamping;
  }
}

function update(delta: number): void {
  if (config.paused) return;

  for (let i = 0; i < config.numParticles; i++) {
    const velocity = velocities[i],
      position = positions[i];
    velocity.addScaledVector(DOWN, config.gravity * delta);
    position.addScaledVector(velocity, delta);
    resolveCollisions(position, velocity);

    particles[i].position.set(position.x, position.y);
  }
}

function setParticleSize(size: number): void {
  for (const particle of particles) particle.scale.setScalar(size);
}

export { update, start, setParticleSize, setParticleGridPosition };
