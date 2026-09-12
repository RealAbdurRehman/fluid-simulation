import * as THREE from "three";

import { config } from "./config";
import * as particles from "./particles";

const DOWN: THREE.Vector2 = new THREE.Vector2(0, -1);

const halfBounds = new THREE.Vector2();

const positions: THREE.Vector2[] = [];
const predictedPositions: THREE.Vector2[] = [];
const velocities: THREE.Vector2[] = [];

const particleMass = 1;

const densities: number[] = [];
const nearDensities: number[] = [];

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

function nearSmoothingKernel(radius: number, distance: number): number {
  if (distance >= radius) return 0;

  const volume = (Math.PI * Math.pow(radius, 5)) / 10;
  return Math.pow(radius - distance, 3) / volume;
}

function nearSmoothingKernelDerivative(
  radius: number,
  distance: number,
): number {
  if (distance >= radius) return 0;

  const scale = 30 / (Math.PI * Math.pow(radius, 5));
  return -(radius - distance) * (radius - distance) * scale;
}

function calculateDensity(point: THREE.Vector2): [number, number] {
  let density = 0;
  let nearDensity = 0;
  foreachPointWithinRadius(point, (particleIndex) => {
    const distance = predictedPositions[particleIndex].distanceTo(point);
    density += particleMass * smoothingKernel(config.smoothingRadius, distance);
    nearDensity +=
      particleMass * nearSmoothingKernel(config.smoothingRadius, distance);
  });

  return [density, nearDensity];
}

function updateDensities(): void {
  for (let i = 0; i < config.numParticles; i++) {
    const [density, nearDensity] = calculateDensity(predictedPositions[i]);
    densities[i] = density;
    nearDensities[i] = nearDensity;
  }
}

function convertDensityToPressure(density: number): number {
  const densityError = density - config.targetDensity;
  return densityError * config.pressureMultiplier;
}

function convertNearDensityToPressure(nearDensity: number): number {
  return nearDensity * config.nearDensityMultiplier;
}

function calculateSharedPressure(densityA: number, densityB: number): number {
  return (
    (convertDensityToPressure(densityA) + convertDensityToPressure(densityB)) /
    2
  );
}

function calculateSharedNearPressure(
  nearDensityA: number,
  nearDensityB: number,
): number {
  return (
    (convertNearDensityToPressure(nearDensityA) +
      convertNearDensityToPressure(nearDensityB)) /
    2
  );
}

function viscositySmoothingKernel(radius: number, distance: number): number {
  if (distance >= radius) return 0;

  const volume = (Math.PI * Math.pow(radius, 4)) / 6;
  return ((radius - distance) * (radius - distance)) / volume;
}

function calculatePressureAndViscosityForces(
  particleIndex: number,
): THREE.Vector2 {
  const totalForce = new THREE.Vector2();
  const point = predictedPositions[particleIndex];
  const density = densities[particleIndex];
  const nearDensity = nearDensities[particleIndex];
  const velocity = velocities[particleIndex];

  foreachPointWithinRadius(point, (neighborIndex) => {
    if (neighborIndex === particleIndex) return;
    const neighbor = predictedPositions[neighborIndex];

    const dx = neighbor.x - point.x;
    const dy = neighbor.y - point.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance === 0) return;

    const dirX = dx / distance;
    const dirY = dy / distance;

    const neighborDensity = densities[neighborIndex];
    const neighborNearDensity = nearDensities[neighborIndex];

    if (neighborDensity > 0) {
      const slope = smoothingKernelDerivative(config.smoothingRadius, distance);
      const sharedPressure = calculateSharedPressure(neighborDensity, density);
      const scalar = (sharedPressure * slope * particleMass) / neighborDensity;
      totalForce.x += dirX * scalar;
      totalForce.y += dirY * scalar;
    }

    if (neighborNearDensity > 0) {
      const nearSlope = nearSmoothingKernelDerivative(
        config.smoothingRadius,
        distance,
      );
      const sharedNearPressure = calculateSharedNearPressure(
        neighborNearDensity,
        nearDensity,
      );
      const nearScalar =
        (sharedNearPressure * nearSlope * particleMass) / neighborNearDensity;
      totalForce.x += dirX * nearScalar;
      totalForce.y += dirY * nearScalar;
    }

    const influence = viscositySmoothingKernel(
      config.smoothingRadius,
      distance,
    );
    const neighborVel = velocities[neighborIndex];
    const scalarVisc = influence * config.viscosityStrength;
    totalForce.x += (neighborVel.x - velocity.x) * scalarVisc;
    totalForce.y += (neighborVel.y - velocity.y) * scalarVisc;
  });

  return totalForce;
}

function updateHalfBounds(): void {
  halfBounds.set(
    config.boundsWidth / 2 - config.particleSize,
    config.boundsHeight / 2 - config.particleSize,
  );
}

function resolveCollisions(
  position: THREE.Vector2,
  velocity: THREE.Vector2,
): void {
  if (Math.abs(position.x) > halfBounds.x) {
    position.x = halfBounds.x * Math.sign(position.x);
    velocity.x *= -1 * config.collisionDamping;
  }
  if (Math.abs(position.y) > halfBounds.y) {
    position.y = halfBounds.y * Math.sign(position.y);
    velocity.y *= -1 * config.collisionDamping;
  }
}

function updateSpatialLookup(points: THREE.Vector2[], radius: number): void {
  for (let i = 0; i < points.length; i++) {
    const [cellX, cellY] = positionToCellCoord(points[i], radius);
    const cellKey = getKeyFromHash(hashCell(cellX, cellY), points.length);

    if (spatialLookup[i]) {
      spatialLookup[i].index = i;
      spatialLookup[i].key = cellKey;
    } else spatialLookup[i] = new Entry(i, cellKey);

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

function applyInteractionForce(
  inputPos: THREE.Vector2,
  radius: number,
  strength: number,
  particleIndex: number,
  delta: number,
): void {
  const px = positions[particleIndex].x;
  const py = positions[particleIndex].y;
  const offsetX = inputPos.x - px;
  const offsetY = inputPos.y - py;
  const distanceSqr = offsetX * offsetX + offsetY * offsetY;
  if (distanceSqr >= radius * radius) return;

  const distance = Math.sqrt(distanceSqr);
  const dirX = distance <= 0 ? 0 : offsetX / distance;
  const dirY = distance <= 0 ? 0 : offsetY / distance;
  const centerT = 1 - distance / radius;

  const velocity = velocities[particleIndex];
  const fx = (dirX * strength - velocity.x) * centerT;
  const fy = (dirY * strength - velocity.y) * centerT;
  velocity.x += fx * delta;
  velocity.y += fy * delta;
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
  updateHalfBounds();
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

  updateHalfBounds();

  for (let i = 0; i < config.numParticles; i++) {
    velocities[i].addScaledVector(DOWN, config.gravity * delta);

    if (interactionStrength !== 0)
      applyInteractionForce(
        interactionPosition,
        config.interactionRadius,
        interactionStrength,
        i,
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

    const totalForce = calculatePressureAndViscosityForces(i);
    const acceleration = totalForce.divideScalar(density);
    velocities[i].addScaledVector(acceleration, delta);
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
