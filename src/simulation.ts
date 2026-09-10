import * as THREE from "three";

import { config } from "./config";
import { scene, particle } from "./scene";

const DOWN: THREE.Vector2 = new THREE.Vector2(0, -1);

const positions: THREE.Vector2[] = [];
const velocities: THREE.Vector2[] = [];

const particleMass = 1;

const particles: THREE.Mesh[] = [];
const densities: number[] = [];

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

  updateDensities();
}

function clearParticles(): void {
  for (const particle of particles) scene.remove(particle);

  particles.length = 0;
  positions.length = 0;
  velocities.length = 0;
  densities.length = 0;
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

function smoothingKernel(radius: number, distance: number): number {
  if (distance >= radius) return 0;

  const volume = (Math.PI * Math.pow(radius, 4)) / 6;
  return ((radius - distance) * (radius - distance)) / volume;
}

function smoothingKernelDerivative(radius: number, distance: number): number {
  if (distance >= radius) return 0;

  const scale = 12 / (Math.PI * Math.pow(radius, 4));
  return (distance - radius) * scale;
}

function calculateDensity(point: THREE.Vector2): number {
  let density = 0;
  for (const position of positions) {
    const distance = position.distanceTo(point);
    const influence = smoothingKernel(config.smoothingRadius, distance);
    density += particleMass * influence;
  }

  return density;
}

function updateDensities(): void {
  for (let i = 0; i < config.numParticles; i++)
    densities[i] = calculateDensity(positions[i]);
}

function convertDensityToPressure(density: number): number {
  const densityError = density - config.targetDensity;
  const pressure = densityError * config.pressureMultiplier;
  return pressure;
}

function calculateSharedPressure(densityA: number, densityB: number): number {
  const pressureA = convertDensityToPressure(densityA);
  const pressureB = convertDensityToPressure(densityB);
  return (pressureA + pressureB) / 2;
}

function calculatePressureForce(particleIndex: number): THREE.Vector2 {
  const pressureForce = new THREE.Vector2();
  for (let i = 0; i < config.numParticles; i++) {
    if (i === particleIndex) continue;

    const distance = positions[i].distanceTo(positions[particleIndex]);
    if (distance === 0) continue;

    const direction = positions[i]
      .clone()
      .sub(positions[particleIndex])
      .divideScalar(distance);
    const slope = smoothingKernelDerivative(config.smoothingRadius, distance);

    const density = densities[i];
    if (density <= 0) continue;

    const sharedPressure = calculateSharedPressure(
      density,
      densities[particleIndex],
    );
    const scalar = (-sharedPressure * slope * particleMass) / density;
    pressureForce.addScaledVector(direction, scalar);
  }

  return pressureForce;
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
    velocities[i].addScaledVector(DOWN, config.gravity * delta);
    densities[i] = calculateDensity(positions[i]);
  }

  for (let i = 0; i < config.numParticles; i++) {
    const density = densities[i];
    if (density <= 0) continue;

    const pressureForce = calculatePressureForce(i);
    const pressureAcceleration = pressureForce.divideScalar(density);
    velocities[i].addScaledVector(pressureAcceleration, delta);
  }

  for (let i = 0; i < config.numParticles; i++) {
    positions[i].addScaledVector(velocities[i], delta);
    resolveCollisions(positions[i], velocities[i]);
    particles[i].position.set(positions[i].x, positions[i].y);
  }
}

function setParticleSize(size: number): void {
  for (const particle of particles) particle.scale.setScalar(size);
}

export { update, start, setParticleSize, setParticleGridPosition };
