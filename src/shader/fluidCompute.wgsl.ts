export const fluidComputeShaderWGSL = /* wgsl */ `
struct Particle {
  position: vec4<f32>,
  predictedPosition: vec4<f32>,
  velocity: vec4<f32>,
  density: vec4<f32>,
};

struct SpatialEntry {
  particleIndex: u32,
  cellKey: u32,
};

struct Collider {
  data0: vec4<f32>,
  rotation: vec4<f32>,
  data2: vec4<f32>,
  velocity: vec4<f32>,
  sdfOrigin: vec4<f32>,
  sdfMeta: vec4<f32>,
};

struct CollisionResult {
  position: vec3<f32>,
  velocity: vec3<f32>,
};

struct MeshQuery {
  dist: f32,
  worldNormal: vec3<f32>,
  valid: bool,
};

struct SimParams {
  boundsWidth: f32, boundsHeight: f32, boundsDepth: f32, gravity: f32,
  collisionDamping: f32, targetDensity: f32, pressureMultiplier: f32, nearPressureMultiplier: f32,
  viscosityStrength: f32, smoothingRadius: f32, particleMass: f32, particleRadius: f32,
  deltaTime: f32, numParticles: u32, tableSize: u32, interactionRadius: f32,
  interactionStrength: f32, poly6Factor: f32, spikyGradFactor: f32, nearSpikyGradFactor: f32,
  viscFactor: f32, numColliders: u32, numProbes: u32, terrainExtentZ: f32,
  interactionRayOrigin: vec4<f32>,
  interactionRayDir: vec4<f32>,
  terrainMeta: vec4<f32>,
};

struct BitonicParams { k: u32, j: u32, numEntries: u32, _pad: u32 };

struct ProbeSample {
  density: f32,
  velX: f32,
  velY: f32,
  velZ: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> spatialLookup: array<SpatialEntry>;
@group(0) @binding(3) var<storage, read_write> startIndices: array<u32>;
@group(0) @binding(4) var<storage, read> colliders: array<Collider>;
@group(0) @binding(5) var<storage, read> sdfData: array<f32>;
@group(0) @binding(6) var<storage, read> probePositions: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> probeSamples: array<ProbeSample>;
@group(0) @binding(8) var<storage, read> terrain: array<f32>;

@group(1) @binding(0) var<uniform> bitonicParams: BitonicParams;

fn qConjugate(q: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn qRotateVec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let qv = q.xyz;
  let uv = cross(qv, v);
  let uuv = cross(qv, uv);
  return v + ((uv * q.w) + uuv) * 2.0;
}

fn positionToCellCoord(pos: vec3<f32>, radius: f32) -> vec3<i32> {
  return vec3<i32>(floor(pos / radius));
}

fn hashCell(cell: vec3<i32>) -> u32 {
  let ux = u32(cell.x) * 73856093u;
  let uy = u32(cell.y) * 19349663u;
  let uz = u32(cell.z) * 83492791u;
  return ux ^ uy ^ uz;
}

fn getKeyFromHash(hash: u32, tableSize: u32) -> u32 {
  return hash % tableSize;
}

@compute @workgroup_size(256)
fn externalForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  var vel = particles[index].velocity.xyz;
  let pos = particles[index].position.xyz;

  vel.y += -params.gravity * params.deltaTime;

  if (params.interactionStrength != 0.0) {
    let rayOrigin = params.interactionRayOrigin.xyz;
    let rayDir = params.interactionRayDir.xyz;
    let v = pos - rayOrigin;
    let projDist = dot(v, rayDir);
    let closestPointOnRay = rayOrigin + rayDir * max(0.0, projDist);

    let offset = closestPointOnRay - pos;
    let distSqr = dot(offset, offset);
    let radiusSqr = params.interactionRadius * params.interactionRadius;

    if (distSqr < radiusSqr && distSqr > 0.0001) {
      let dist = sqrt(distSqr);
      let dir = offset / dist;
      let centerT = 1.0 - (dist / params.interactionRadius);
      let force = (dir * params.interactionStrength - vel) * (centerT * centerT);
      vel += force * params.deltaTime;
    }
  }

  particles[index].velocity = vec4<f32>(vel, 0.0);
  particles[index].predictedPosition = vec4<f32>(pos + vel * (1.0 / 60.0), 0.0);
}

@compute @workgroup_size(256)
fn updateSpatialHash(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.tableSize) { return; }

  if (index < params.numParticles) {
    let cellCoord = positionToCellCoord(particles[index].predictedPosition.xyz, params.smoothingRadius);
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
  if (index < params.tableSize) { startIndices[index] = 0xFFFFFFFFu; }
}

@compute @workgroup_size(256)
fn calculateStartIndices(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  let key = spatialLookup[index].cellKey;
  let prevKey = select(spatialLookup[index - 1u].cellKey, 0xFFFFFFFFu, index == 0u);
  if (key != prevKey && key < params.tableSize) { startIndices[key] = index; }
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
fn calculateDensities(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  let pos = particles[index].predictedPosition.xyz;
  let centerCell = positionToCellCoord(pos, params.smoothingRadius);
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;

  var density = 0.0;
  var nearDensity = 0.0;

  for (var offsetZ = -1; offsetZ <= 1; offsetZ++) {
    for (var offsetY = -1; offsetY <= 1; offsetY++) {
      for (var offsetX = -1; offsetX <= 1; offsetX++) {
        let neighborCell = centerCell + vec3<i32>(offsetX, offsetY, offsetZ);
        let key = getKeyFromHash(hashCell(neighborCell), params.tableSize);
        let startIndex = startIndices[key];

        if (startIndex != 0xFFFFFFFFu) {
          for (var i = startIndex; i < params.numParticles; i++) {
            let entry = spatialLookup[i];
            if (entry.cellKey != key) { break; }

            let neighborIndex = entry.particleIndex;
            let neighborPos = particles[neighborIndex].predictedPosition.xyz;
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
  }

  particles[index].density = vec4<f32>(density, nearDensity, 0.0, 0.0);
}

fn containerRepulsionForce(pos: vec3<f32>, radius: f32) -> vec3<f32> {
  let container = colliders[0];
  let invRot = qConjugate(container.rotation);
  let localPos = qRotateVec(invRot, pos - container.data0.xyz);
  let halfExtents = container.data2.xyz - vec3<f32>(radius);
  let margin = radius;

  var localForce = vec3<f32>(0.0);
  let dx = halfExtents.x - abs(localPos.x);
  if (dx < margin) { localForce.x += -sign(localPos.x) * (margin - dx) / margin; }
  let dy = halfExtents.y - abs(localPos.y);
  if (dy < margin) { localForce.y += -sign(localPos.y) * (margin - dy) / margin; }
  let dz = halfExtents.z - abs(localPos.z);
  if (dz < margin) { localForce.z += -sign(localPos.z) * (margin - dz) / margin; }

  return qRotateVec(container.rotation, localForce) * params.pressureMultiplier;
}

fn resolveContainer(posIn: vec3<f32>, velIn: vec3<f32>, collider: Collider, radius: f32) -> CollisionResult {
  let invRot = qConjugate(collider.rotation);
  var localPos = qRotateVec(invRot, posIn - collider.data0.xyz);
  var localVel = qRotateVec(invRot, velIn - collider.velocity.xyz);
  let halfExtents = collider.data2.xyz - vec3<f32>(radius);
  let restitution = collider.data2.w;

  if (abs(localPos.x) > halfExtents.x) { localPos.x = sign(localPos.x) * halfExtents.x; localVel.x *= -restitution; }
  if (abs(localPos.y) > halfExtents.y) { localPos.y = sign(localPos.y) * halfExtents.y; localVel.y *= -restitution; }
  if (abs(localPos.z) > halfExtents.z) { localPos.z = sign(localPos.z) * halfExtents.z; localVel.z *= -restitution; }

  var result: CollisionResult;
  result.position = qRotateVec(collider.rotation, localPos) + collider.data0.xyz;
  result.velocity = qRotateVec(collider.rotation, localVel) + collider.velocity.xyz;
  return result;
}

fn resolveBox(posIn: vec3<f32>, velIn: vec3<f32>, collider: Collider, radius: f32) -> CollisionResult {
  let invRot = qConjugate(collider.rotation);
  let localPos = qRotateVec(invRot, posIn - collider.data0.xyz);
  let localVel = qRotateVec(invRot, velIn - collider.velocity.xyz);
  let halfExtents = collider.data2.xyz + vec3<f32>(radius);
  let restitution = collider.data2.w;

  var result: CollisionResult;
  result.position = posIn;
  result.velocity = velIn;

  let outside = abs(localPos.x) > halfExtents.x || abs(localPos.y) > halfExtents.y || abs(localPos.z) > halfExtents.z;
  if (outside) { return result; }

  let penX = halfExtents.x - abs(localPos.x);
  let penY = halfExtents.y - abs(localPos.y);
  let penZ = halfExtents.z - abs(localPos.z);

  var newLocalPos = localPos;
  var newLocalVel = localVel;

  if (penX <= penY && penX <= penZ) {
    newLocalPos.x = sign(localPos.x) * halfExtents.x;
    newLocalVel.x *= -restitution;
  } else if (penY <= penX && penY <= penZ) {
    newLocalPos.y = sign(localPos.y) * halfExtents.y;
    newLocalVel.y *= -restitution;
  } else {
    newLocalPos.z = sign(localPos.z) * halfExtents.z;
    newLocalVel.z *= -restitution;
  }

  result.position = qRotateVec(collider.rotation, newLocalPos) + collider.data0.xyz;
  result.velocity = qRotateVec(collider.rotation, newLocalVel) + collider.velocity.xyz;
  return result;
}

fn resolveSphere(posIn: vec3<f32>, velIn: vec3<f32>, collider: Collider, radius: f32) -> CollisionResult {
  var result: CollisionResult;
  result.position = posIn;
  result.velocity = velIn;

  let center = collider.data0.xyz;
  let sphereRadius = collider.data2.x + radius;
  let restitution = collider.data2.w;

  let offset = posIn - center;
  let dist = length(offset);
  if (dist >= sphereRadius || dist < 0.0001) { return result; }

  let normal = offset / dist;
  result.position = center + normal * sphereRadius;

  let relVel = velIn - collider.velocity.xyz;
  let velAlongNormal = dot(relVel, normal);
  var newRelVel = relVel;
  if (velAlongNormal < 0.0) {
    newRelVel -= normal * velAlongNormal * (1.0 + restitution);
  }
  result.velocity = newRelVel + collider.velocity.xyz;
  return result;
}

fn sdfCellIndex(dims: vec3<u32>, offset: u32, p: vec3<u32>) -> u32 {
  return offset + p.x + p.y * dims.x + p.z * dims.x * dims.y;
}

fn sampleSDF(collider: Collider, localPos: vec3<f32>) -> f32 {
  let dims = vec3<u32>(collider.sdfMeta.xyz);
  let offset = u32(collider.sdfMeta.w);
  let cellSize = collider.sdfOrigin.w;

  let gridPosF = (localPos - collider.sdfOrigin.xyz) / cellSize;
  let maxIdx = vec3<f32>(dims) - vec3<f32>(1.0001);
  let clamped = clamp(gridPosF, vec3<f32>(0.0), maxIdx);

  let i0 = vec3<u32>(floor(clamped));
  let frac = clamped - vec3<f32>(i0);
  let i1 = min(i0 + vec3<u32>(1u), dims - vec3<u32>(1u));

  let c000 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i0.x, i0.y, i0.z))];
  let c100 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i1.x, i0.y, i0.z))];
  let c010 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i0.x, i1.y, i0.z))];
  let c110 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i1.x, i1.y, i0.z))];
  let c001 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i0.x, i0.y, i1.z))];
  let c101 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i1.x, i0.y, i1.z))];
  let c011 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i0.x, i1.y, i1.z))];
  let c111 = sdfData[sdfCellIndex(dims, offset, vec3<u32>(i1.x, i1.y, i1.z))];

  let c00 = mix(c000, c100, frac.x);
  let c10 = mix(c010, c110, frac.x);
  let c01 = mix(c001, c101, frac.x);
  let c11 = mix(c011, c111, frac.x);

  return mix(mix(c00, c10, frac.y), mix(c01, c11, frac.y), frac.z);
}

fn sdfGradient(collider: Collider, localPos: vec3<f32>) -> vec3<f32> {
  let eps = max(collider.sdfOrigin.w * 1.5, 0.0001);
  let dx = sampleSDF(collider, localPos + vec3<f32>(eps, 0.0, 0.0)) - sampleSDF(collider, localPos - vec3<f32>(eps, 0.0, 0.0));
  let dy = sampleSDF(collider, localPos + vec3<f32>(0.0, eps, 0.0)) - sampleSDF(collider, localPos - vec3<f32>(0.0, eps, 0.0));
  let dz = sampleSDF(collider, localPos + vec3<f32>(0.0, 0.0, eps)) - sampleSDF(collider, localPos - vec3<f32>(0.0, 0.0, eps));
  let grad = vec3<f32>(dx, dy, dz);
  let len = length(grad);
  return select(grad / len, vec3<f32>(0.0, 1.0, 0.0), len < 0.00001);
}

fn queryMesh(collider: Collider, worldPos: vec3<f32>) -> MeshQuery {
  var q: MeshQuery;
  q.valid = false;
  q.dist = 1e9;
  q.worldNormal = vec3<f32>(0.0, 1.0, 0.0);

  let invRot = qConjugate(collider.rotation);
  let scale = max(collider.data2.x, 0.0001);
  let localPos = qRotateVec(invRot, worldPos - collider.data0.xyz) / scale;

  let dims = collider.sdfMeta.xyz;
  let cellSize = collider.sdfOrigin.w;
  let gridPos = (localPos - collider.sdfOrigin.xyz) / cellSize;
  if (any(gridPos < vec3<f32>(0.0)) || any(gridPos > dims - vec3<f32>(1.0))) {
    return q;
  }

  q.valid = true;
  q.dist = sampleSDF(collider, localPos) * scale;
  q.worldNormal = normalize(qRotateVec(collider.rotation, sdfGradient(collider, localPos)));
  return q;
}

fn resolveMesh(
  posIn: vec3<f32>, velIn: vec3<f32>, collider: Collider, radius: f32,
) -> CollisionResult {
  var r: CollisionResult;
  r.position = posIn;
  r.velocity = velIn;

  let q = queryMesh(collider, posIn);
  if (!q.valid || q.dist >= radius) { return r; }

  r.position = posIn + q.worldNormal * (radius - q.dist);

  let restitution = collider.data2.w;
  let relVel = velIn - collider.velocity.xyz;
  let velAlongNormal = dot(relVel, q.worldNormal);
  var newRelVel = relVel;
  if (velAlongNormal < 0.0) {
    newRelVel -= q.worldNormal * velAlongNormal * (1.0 + restitution);
  }

  r.velocity = newRelVel + collider.velocity.xyz;
  return r;
}

fn terrainHeightAt(x: f32, z: f32) -> f32 {
  let N = i32(params.terrainMeta.x);
  let extX = params.terrainMeta.y;
  let extZ = params.terrainExtentZ;
  let hs = params.terrainMeta.z;
  if (params.terrainMeta.w < 0.5 || N < 2) { return -1e9; }

  let cellX = extX / f32(N - 1);
  let cellZ = extZ / f32(N - 1);
  let u = clamp((x + extX * 0.5) / cellX, 0.0, f32(N - 1));
  let v = clamp((z + extZ * 0.5) / cellZ, 0.0, f32(N - 1));

  let i0x = i32(floor(u));
  let i0y = i32(floor(v));
  let i1x = min(i0x + 1, N - 1);
  let i1y = min(i0y + 1, N - 1);
  let fx = u - f32(i0x);
  let fy = v - f32(i0y);

  let h00 = terrain[i0x + i0y * N];
  let h10 = terrain[i1x + i0y * N];
  let h01 = terrain[i0x + i1y * N];
  let h11 = terrain[i1x + i1y * N];

  let h = mix(mix(h00, h10, fx), mix(h01, h11, fx), fy);
  return h * hs;
}

fn resolveTerrain(
  posIn: vec3<f32>,
  velIn: vec3<f32>,
  radius: f32,
) -> CollisionResult {
  var r: CollisionResult;
  r.position = posIn;
  r.velocity = velIn;
  if (params.terrainMeta.w < 0.5) { return r; }

  let container = colliders[0];
  let invRot = qConjugate(container.rotation);
  var localPos = qRotateVec(invRot, posIn - container.data0.xyz);
  var localVel = qRotateVec(invRot, velIn - container.velocity.xyz);

  let baseY = -params.boundsHeight * 0.5;
  let surfY = baseY + terrainHeightAt(localPos.x, localPos.z);

  let extX = params.terrainMeta.y;
  let extZ = params.terrainExtentZ;
  let cellX = extX / max(params.terrainMeta.x - 1.0, 1.0);
  let cellZ = extZ / max(params.terrainMeta.x - 1.0, 1.0);
  let epsX = max(cellX, 0.01);
  let epsZ = max(cellZ, 0.01);
  let dxH = terrainHeightAt(localPos.x + epsX, localPos.z)
          - terrainHeightAt(localPos.x - epsX, localPos.z);
  let dzH = terrainHeightAt(localPos.x, localPos.z + epsZ)
          - terrainHeightAt(localPos.x, localPos.z - epsZ);
  let n = normalize(vec3<f32>(-dxH * epsZ, 2.0 * epsX * epsZ, -dzH * epsX));

  let signedDist = localPos.y - surfY;
  if (signedDist < radius) {
    let push = radius - signedDist;
    localPos += n * push;

    let vn = dot(localVel, n);
    if (vn < 0.0) {
      let restitution = container.data2.w;
      localVel -= n * vn * (1.0 + restitution);
    }
  }

  r.position = qRotateVec(container.rotation, localPos) + container.data0.xyz;
  r.velocity = qRotateVec(container.rotation, localVel) + container.velocity.xyz;
  return r;
}

@compute @workgroup_size(256)
fn calculateForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  let pos = particles[index].predictedPosition.xyz;
  let vel = particles[index].velocity.xyz;
  let densityData = particles[index].density;
  let density = densityData.x;
  let nearDensity = densityData.y;

  let pressure = densityToPressure(density);
  let nearPressure = nearDensityToPressure(nearDensity);

  let centerCell = positionToCellCoord(pos, params.smoothingRadius);
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;

  var pressureForce = vec3<f32>(0.0);
  var viscosityForce = vec3<f32>(0.0);

  for (var offsetZ = -1; offsetZ <= 1; offsetZ++) {
    for (var offsetY = -1; offsetY <= 1; offsetY++) {
      for (var offsetX = -1; offsetX <= 1; offsetX++) {
        let neighborCell = centerCell + vec3<i32>(offsetX, offsetY, offsetZ);
        let key = getKeyFromHash(hashCell(neighborCell), params.tableSize);
        let startIndex = startIndices[key];

        if (startIndex != 0xFFFFFFFFu) {
          for (var i = startIndex; i < params.numParticles; i++) {
            let entry = spatialLookup[i];
            if (entry.cellKey != key) { break; }

            let neighborIndex = entry.particleIndex;
            if (neighborIndex == index) { continue; }

            let neighborPos = particles[neighborIndex].predictedPosition.xyz;
            let diff = neighborPos - pos;
            let distSqr = dot(diff, diff);

            if (distSqr <= radiusSqr) {
              let dist = max(sqrt(distSqr), 0.001);
              let dir = diff / dist;

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

              let neighborVel = particles[neighborIndex].velocity.xyz;
              let viscWeight = viscosityKernel(radius, dist);
              let viscDamping = min(viscWeight * params.viscosityStrength * params.deltaTime, 0.40) / max(params.deltaTime, 0.0001);
              viscosityForce += (neighborVel - vel) * viscDamping;
            }
          }
        }
      }
    }
  }

  var pressureAccel = pressureForce / max(density, 0.001);
  let maxUpwardAccel = max(params.gravity * 3.5, 60.0);
  pressureAccel.y = min(pressureAccel.y, maxUpwardAccel);

  let totalAcceleration = select(
    pressureAccel + (viscosityForce / max(density, 0.001)),
    vec3<f32>(0.0),
    density <= 0.0,
  );

  let wallAccel = containerRepulsionForce(pos, params.particleRadius);
  particles[index].velocity = vec4<f32>(vel + (totalAcceleration + wallAccel) * params.deltaTime, 0.0);
}

@compute @workgroup_size(256)
fn integratePositions(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  var pos = particles[index].position.xyz;
  var vel = particles[index].velocity.xyz;

  pos += vel * params.deltaTime;

  for (var c = 0u; c < params.numColliders; c++) {
    let collider = colliders[c];
    let shapeType = collider.data0.w;
    var result: CollisionResult;

    if (shapeType < 0.5) {
      result = resolveContainer(pos, vel, collider, params.particleRadius);
    } else if (shapeType < 1.5) {
      result = resolveBox(pos, vel, collider, params.particleRadius);
    } else if (shapeType < 2.5) {
      result = resolveSphere(pos, vel, collider, params.particleRadius);
    } else {
      result = resolveMesh(pos, vel, collider, params.particleRadius);
    }

    pos = result.position;
    vel = result.velocity;
  }

  let tr = resolveTerrain(pos, vel, params.particleRadius);
  pos = tr.position;
  vel = tr.velocity;

  particles[index].position = vec4<f32>(pos, 1.0);
  particles[index].velocity = vec4<f32>(vel, 0.0);
}

@compute @workgroup_size(64)
fn sampleProbes(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= params.numProbes) { return; }

  let pos = probePositions[idx].xyz;
  let centerCell = positionToCellCoord(pos, params.smoothingRadius);
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;

  var density = 0.0;
  var weightedVel = vec3<f32>(0.0);
  var weightSum = 0.0;

  for (var oz = -1; oz <= 1; oz++) {
    for (var oy = -1; oy <= 1; oy++) {
      for (var ox = -1; ox <= 1; ox++) {
        let neighborCell = centerCell + vec3<i32>(ox, oy, oz);
        let key = getKeyFromHash(hashCell(neighborCell), params.tableSize);
        let startIndex = startIndices[key];
        if (startIndex == 0xFFFFFFFFu) { continue; }

        for (var i = startIndex; i < params.numParticles; i++) {
          let entry = spatialLookup[i];
          if (entry.cellKey != key) { break; }

          let nIdx = entry.particleIndex;
          let nPos = particles[nIdx].predictedPosition.xyz;
          let diff = nPos - pos;
          let d2 = dot(diff, diff);
          if (d2 <= radiusSqr) {
            let d = sqrt(d2);
            let w = densityKernel(radius, d) * params.particleMass;
            density += w;
            weightedVel += particles[nIdx].velocity.xyz * w;
            weightSum += w;
          }
        }
      }
    }
  }

  var s: ProbeSample;
  s.density = density;

  let vel = select(vec3<f32>(0.0), weightedVel / weightSum, weightSum > 0.0001);
  s.velX = vel.x;
  s.velY = vel.y;
  s.velZ = vel.z;

  probeSamples[idx] = s;
}
`;
