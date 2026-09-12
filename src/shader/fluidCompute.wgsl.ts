export const fluidComputeShaderWGSL = /* wgsl */ `
struct Particle {
  position: vec2<f32>,
  predictedPosition: vec2<f32>,
  velocity: vec2<f32>,
  density: vec2<f32>,
};

struct SpatialEntry {
  particleIndex: u32,
  cellKey: u32,
};

struct SimParams {
  boundsWidth: f32,
  boundsHeight: f32,
  gravity: f32,
  collisionDamping: f32,

  targetDensity: f32,
  pressureMultiplier: f32,
  nearPressureMultiplier: f32,
  viscosityStrength: f32,

  smoothingRadius: f32,
  particleMass: f32,
  particleRadius: f32,
  deltaTime: f32,

  numParticles: u32,
  tableSize: u32,
  interactionRadius: f32,
  interactionStrength: f32,

  interactionPos: vec2<f32>,
  poly6Factor: f32,
  spikyGradFactor: f32,

  nearSpikyGradFactor: f32,
  viscFactor: f32,
  _pad0: f32,
  _pad1: f32,
};

struct BitonicParams {
  k: u32,
  j: u32,
  numEntries: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> spatialLookup: array<SpatialEntry>;
@group(0) @binding(3) var<storage, read_write> startIndices: array<u32>;

@group(1) @binding(0) var<uniform> bitonicParams: BitonicParams;


fn positionToCellCoord(pos: vec2<f32>, radius: f32) -> vec2<i32> {
  return vec2<i32>(floor(pos / radius));
}

fn hashCell(cell: vec2<i32>) -> u32 {
  let ux = u32(cell.x * 73856093);
  let uy = u32(cell.y * 19349663);
  return ux ^ uy;
}

fn getKeyFromHash(hash: u32, tableSize: u32) -> u32 {
  return hash % tableSize;
}

fn getDisplacementJitter(indexA: u32, indexB: u32) -> vec2<f32> {
  let angle = f32((indexA * 15823u + indexB * 9737333u) % 6283u) * 0.001;
  return vec2<f32>(cos(angle), sin(angle));
}

@compute @workgroup_size(256)
fn externalForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  var vel = particles[index].velocity;
  let pos = particles[index].position;

  vel.y += -params.gravity * params.deltaTime;

  if (params.interactionStrength != 0.0) {
    let offset = params.interactionPos - pos;
    let distSqr = dot(offset, offset);
    let radiusSqr = params.interactionRadius * params.interactionRadius;

    if (distSqr < radiusSqr && distSqr > 0.0001) {
      let dist = sqrt(distSqr);
      let dir = offset / dist;
      let centerT = 1.0 - (dist / params.interactionRadius);
      let force = (dir * params.interactionStrength - vel) * centerT;
      vel += force * params.deltaTime;
    }
  }

  particles[index].velocity = vel;
  particles[index].predictedPosition = pos + vel * (1.0 / 60.0);
}

@compute @workgroup_size(256)
fn updateSpatialHash(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.tableSize) { return; }

  if (index < params.numParticles) {
    let cellCoord = positionToCellCoord(particles[index].predictedPosition, params.smoothingRadius);
    let key = getKeyFromHash(hashCell(cellCoord), params.tableSize);
    spatialLookup[index] = SpatialEntry(index, key);
  } else {
    spatialLookup[index] = SpatialEntry(0xFFFFFFFFu, 0xFFFFFFFFu);
  }
}

@compute @workgroup_size(256)
fn bitonicSort(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= bitonicParams.numEntries) { return; }

  let k = bitonicParams.k;
  let j = bitonicParams.j;

  let l = i ^ j;
  if (l > i && l < bitonicParams.numEntries) {
    let ascending = (i & k) == 0u;
    let entryA = spatialLookup[i];
    let entryB = spatialLookup[l];

    if ((entryA.cellKey > entryB.cellKey) == ascending) {
      spatialLookup[i] = entryB;
      spatialLookup[l] = entryA;
    }
  }
}

@compute @workgroup_size(256)
fn clearStartIndices(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index < params.tableSize) {
    startIndices[index] = 0xFFFFFFFFu;
  }
}

@compute @workgroup_size(256)
fn calculateStartIndices(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  let key = spatialLookup[index].cellKey;
  let prevKey = select(spatialLookup[index - 1u].cellKey, 0xFFFFFFFFu, index == 0u);

  if (key != prevKey && key < params.tableSize) {
    startIndices[key] = index;
  }
}

fn densityKernel(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return v * v * params.poly6Factor;
}

fn nearDensityKernel(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return v * v * v * params.nearSpikyGradFactor;
}

@compute @workgroup_size(256)
fn calculateDensities(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  let pos = particles[index].predictedPosition;
  let centerCell = positionToCellCoord(pos, params.smoothingRadius);
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;

  var density = 0.0;
  var nearDensity = 0.0;

  for (var offsetY = -1; offsetY <= 1; offsetY++) {
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
      let neighborCell = centerCell + vec2<i32>(offsetX, offsetY);
      let key = getKeyFromHash(hashCell(neighborCell), params.tableSize);
      let startIndex = startIndices[key];

      if (startIndex != 0xFFFFFFFFu) {
        for (var i = startIndex; i < params.numParticles; i++) {
          let entry = spatialLookup[i];
          if (entry.cellKey != key) { break; }

          let neighborIndex = entry.particleIndex;
          let neighborPos = particles[neighborIndex].predictedPosition;
          let diff = neighborPos - pos;
          let distSqr = dot(diff, diff);

          if (distSqr <= radiusSqr) {
            let dist = sqrt(distSqr);
            density += params.particleMass * densityKernel(radius, dist);
            nearDensity += params.particleMass * nearDensityKernel(radius, dist);
          }
        }
      }
    }
  }

  particles[index].density = vec2<f32>(density, nearDensity);
}

fn densityToPressure(density: f32) -> f32 {
  return (density - params.targetDensity) * params.pressureMultiplier;
}

fn nearDensityToPressure(nearDensity: f32) -> f32 {
  return nearDensity * params.nearPressureMultiplier;
}

fn densityKernelDerivative(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  return (dist - radius) * params.spikyGradFactor;
}

fn nearDensityKernelDerivative(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return -v * v * params.nearSpikyGradFactor;
}

fn viscosityKernel(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return v * v * params.viscFactor;
}

@compute @workgroup_size(256)
fn calculateForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  let pos = particles[index].predictedPosition;
  let vel = particles[index].velocity;
  let densityData = particles[index].density;
  let density = densityData.x;
  let nearDensity = densityData.y;

  let pressure = densityToPressure(density);
  let nearPressure = nearDensityToPressure(nearDensity);

  let centerCell = positionToCellCoord(pos, params.smoothingRadius);
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;

  var pressureForce = vec2<f32>(0.0);
  var viscosityForce = vec2<f32>(0.0);

  for (var offsetY = -1; offsetY <= 1; offsetY++) {
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
      let neighborCell = centerCell + vec2<i32>(offsetX, offsetY);
      let key = getKeyFromHash(hashCell(neighborCell), params.tableSize);
      let startIndex = startIndices[key];

      if (startIndex != 0xFFFFFFFFu) {
        for (var i = startIndex; i < params.numParticles; i++) {
          let entry = spatialLookup[i];
          if (entry.cellKey != key) { break; }

          let neighborIndex = entry.particleIndex;
          if (neighborIndex == index) { continue; }

          let neighborPos = particles[neighborIndex].predictedPosition;
          let diff = neighborPos - pos;
          let distSqr = dot(diff, diff);

          if (distSqr <= radiusSqr) {
            let rawDist = sqrt(distSqr);
           
           
            let isOverlapping = rawDist < 0.0001;
            let dist = select(rawDist, 0.0001, isOverlapping);
            let dir = select(diff / dist, getDisplacementJitter(index, neighborIndex), isOverlapping);

            let neighborDensityData = particles[neighborIndex].density;
            let neighborDensity = neighborDensityData.x;
            let neighborNearDensity = neighborDensityData.y;

            if (neighborDensity > 0.0) {
              let neighborPressure = densityToPressure(neighborDensity);
              let sharedPressure = (pressure + neighborPressure) * 0.5;
              let slope = densityKernelDerivative(radius, dist);
              pressureForce += dir * (sharedPressure * slope * params.particleMass / neighborDensity);
            }

            if (neighborNearDensity > 0.0) {
              let neighborNearPressure = nearDensityToPressure(neighborNearDensity);
              let sharedNearPressure = (nearPressure + neighborNearPressure) * 0.5;
              let nearSlope = nearDensityKernelDerivative(radius, dist);
              pressureForce += dir * (sharedNearPressure * nearSlope * params.particleMass / neighborNearDensity);
            }

            let neighborVel = particles[neighborIndex].velocity;
            let viscWeight = viscosityKernel(radius, dist);
            viscosityForce += (neighborVel - vel) * (viscWeight * params.viscosityStrength);
          }
        }
      }
    }
  }

  let totalAcceleration = select((pressureForce + viscosityForce) / max(density, 0.001), vec2<f32>(0.0), density <= 0.0);
  particles[index].velocity += totalAcceleration * params.deltaTime;
}


@compute @workgroup_size(256)
fn integratePositions(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  var pos = particles[index].position;
  var vel = particles[index].velocity;

  pos += vel * params.deltaTime;

  let halfBoundX = params.boundsWidth * 0.5 - params.particleRadius;
  let halfBoundY = params.boundsHeight * 0.5 - params.particleRadius;
  if (abs(pos.x) > halfBoundX) {
    pos.x = sign(pos.x) * halfBoundX;
    vel.x *= -params.collisionDamping;
  }
  if (abs(pos.y) > halfBoundY) {
    pos.y = sign(pos.y) * halfBoundY;
    vel.y *= -params.collisionDamping;
  }

  particles[index].position = pos;
  particles[index].velocity = vel;
}
`;
