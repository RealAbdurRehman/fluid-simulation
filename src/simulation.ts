import * as THREE from "three";

import { config } from "./config";
import * as particles from "./particles";

const DOWN: THREE.Vector2 = new THREE.Vector2(0, -1);

const positions: THREE.Vector2[] = [];
const predictedPositions: THREE.Vector2[] = [];
const velocities: THREE.Vector2[] = [];

const particleMass = 1;

const densities: number[] = [];

class Entry {
  index: number;
  key: number;
  constructor(index: number, key: number) {
    this.index = index;
    this.key = key;
  }
}

const spatialLookup: Entry[] = [];
const startIndices: number[] = [];

let interactionStrength = 0;
const interactionPosition = new THREE.Vector2();

const cellOffsets: [number, number][] = [];
for (let y = -1; y <= 1; y++)
  for (let x = -1; x <= 1; x++) cellOffsets.push([x, y]);

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
  foreachPointWithinRadius(point, (particleIndex) => {
    const distance = predictedPositions[particleIndex].distanceTo(point);
    const influence = smoothingKernel(config.smoothingRadius, distance);
    density += particleMass * influence;
  });

  return density;
}

function updateDensities(): void {
  for (let i = 0; i < config.numParticles; i++)
    densities[i] = calculateDensity(predictedPositions[i]);
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
  const point = predictedPositions[particleIndex];

  foreachPointWithinRadius(point, (neighborIndex) => {
    if (neighborIndex === particleIndex) return;
    const neighbor = predictedPositions[neighborIndex];

    const dx = neighbor.x - point.x;
    const dy = neighbor.y - point.y;

    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance === 0) return;

    const direction = new THREE.Vector2(dx / distance, dy / distance);
    const slope = smoothingKernelDerivative(config.smoothingRadius, distance);

    const density = densities[neighborIndex];
    if (density <= 0) return;

    const sharedPressure = calculateSharedPressure(
      density,
      densities[particleIndex],
    );
    const scalar = (sharedPressure * slope * particleMass) / density;
    pressureForce.addScaledVector(direction, scalar);
  });

  return pressureForce;
}

function getHalfBounds(): THREE.Vector2 {
  return new THREE.Vector2(
    config.boundsWidth / 2 - config.particleSize,
    config.boundsHeight / 2 - config.particleSize,
  );
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

function positionToCellCoord(
  point: THREE.Vector2,
  radius: number,
): [number, number] {
  const cellX = Math.floor(point.x / radius);
  const cellY = Math.floor(point.y / radius);
  return [cellX, cellY];
}

function hashCell(cellX: number, cellY: number): number {
  const a = (cellX >>> 0) * 15823;
  const b = (cellY >>> 0) * 9737333;
  return (a + b) >>> 0;
}

function getKeyFromHash(hash: number, tableSize: number): number {
  return hash % tableSize;
}

function updateSpatialLookup(points: THREE.Vector2[], radius: number): void {
  for (let i = 0; i < points.length; i++) {
    const [cellX, cellY] = positionToCellCoord(points[i], radius);
    const cellKey = getKeyFromHash(hashCell(cellX, cellY), points.length);
    spatialLookup[i] = new Entry(i, cellKey);
    startIndices[i] = Number.MAX_SAFE_INTEGER;
  }

  spatialLookup.sort((a, b) => a.key - b.key);

  for (let i = 0; i < points.length; i++) {
    const key = spatialLookup[i].key;
    const keyPrev =
      i === 0 ? Number.MAX_SAFE_INTEGER : spatialLookup[i - 1].key;
    if (key !== keyPrev) startIndices[key] = i;
  }
}

function foreachPointWithinRadius(
  point: THREE.Vector2,
  callback: (particleIndex: number) => void,
): void {
  const radius = config.smoothingRadius;
  const [centerX, centerY] = positionToCellCoord(point, radius);
  const radiusSqr = radius * radius;

  for (const [offsetX, offsetY] of cellOffsets) {
    const key = getKeyFromHash(
      hashCell(centerX + offsetX, centerY + offsetY),
      spatialLookup.length,
    );
    const cellStartIndex = startIndices[key];

    for (let i = cellStartIndex; i < spatialLookup.length; i++) {
      if (spatialLookup[i].key !== key) break;

      const particleIndex = spatialLookup[i].index;
      const dx = predictedPositions[particleIndex].x - point.x;
      const dy = predictedPositions[particleIndex].y - point.y;

      const distanceSqr = dx * dx + dy * dy;
      if (distanceSqr <= radiusSqr) callback(particleIndex);
    }
  }
}

function interactionForce(
  inputPos: THREE.Vector2,
  radius: number,
  strength: number,
  particleIndex: number,
): THREE.Vector2 {
  const interactionForce = new THREE.Vector2();
  const offset = inputPos.clone().sub(positions[particleIndex]);
  const distanceSqr = offset.dot(offset);

  if (distanceSqr < radius * radius) {
    const distance = Math.sqrt(distanceSqr);
    const directionToInputPoint =
      distance <= 0 ? new THREE.Vector2() : offset.divideScalar(distance);
    const centerT = 1 - distance / radius;
    interactionForce.addScaledVector(
      directionToInputPoint
        .multiplyScalar(strength)
        .sub(velocities[particleIndex]),
      centerT,
    );
  }

  return interactionForce;
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
  }
}

function clearParticles(): void {
  positions.length = 0;
  predictedPositions.length = 0;
  velocities.length = 0;
  densities.length = 0;
  spatialLookup.length = 0;
  startIndices.length = 0;
}

function start(): void {
  clearParticles();
  for (let i = 0; i < config.numParticles; i++) {
    velocities.push(new THREE.Vector2());
    positions.push(new THREE.Vector2());
    predictedPositions.push(new THREE.Vector2());
  }

  particles.setParticleCount(config.numParticles);
  setParticleGridPosition();
}

function update(delta: number): void {
  if (config.paused) return;

  for (let i = 0; i < config.numParticles; i++) {
    velocities[i].addScaledVector(DOWN, config.gravity * delta);

    if (interactionStrength !== 0)
      velocities[i].addScaledVector(
        interactionForce(
          interactionPosition,
          config.interactionRadius,
          interactionStrength,
          i,
        ),
        delta,
      );

    predictedPositions[i]
      .copy(positions[i])
      .addScaledVector(velocities[i], delta);
  }

  updateSpatialLookup(predictedPositions, config.smoothingRadius);
  updateDensities();

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
  }
}

function syncVisuals(): void {
  for (let i = 0; i < config.numParticles; i++) {
    const p = positions[i];
    const speed = velocities[i].length();
    particles.updateParticle(i, p.x, p.y, speed);
  }

  particles.commitParticles();
}

function setParticleSize(size: number): void {
  particles.setParticleSize(size);
}

function setInteraction(position: THREE.Vector2, strength: number): void {
  interactionPosition.copy(position);
  interactionStrength = strength;
}

export {
  update,
  start,
  syncVisuals,
  setParticleSize,
  setParticleGridPosition,
  setInteraction,
};
