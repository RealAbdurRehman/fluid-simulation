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
  planetCenter: vec4<f32>,     
  planetParams: vec4<f32>,     

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
  poly6Factor: f32,
  spikyGradFactor: f32,

  nearSpikyGradFactor: f32,
  viscFactor: f32,
  numColliders: u32,
  numProbes: u32,

  interactionRayOrigin: vec4<f32>,
  interactionRayDir: vec4<f32>,
  _pad0: vec4<f32>,
  _pad1: vec4<f32>,
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
@group(0) @binding(8) var<storage, read> planetHeights: array<f32>;

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

fn densityKernel(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return v * v * v * params.poly6Factor;
}

fn nearDensityKernel(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return v * v * v * v * params.nearSpikyGradFactor;
}

fn densityToPressure(density: f32) -> f32 {
  return (density - params.targetDensity) * params.pressureMultiplier;
}

fn nearDensityToPressure(nearDensity: f32) -> f32 {
  return nearDensity * params.nearPressureMultiplier;
}

fn densityKernelDerivative(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return -v * v * params.spikyGradFactor;
}

fn nearDensityKernelDerivative(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return -v * v * v * params.nearSpikyGradFactor;
}

fn viscosityKernel(radius: f32, dist: f32) -> f32 {
  if (dist >= radius) { return 0.0; }
  let v = radius - dist;
  return v * v * params.viscFactor;
}

fn dirToFaceUV(d: vec3<f32>) -> vec3<f32> {
  let a = abs(d);
  var face: f32;
  var s: f32;
  var t: f32;

  if (a.x >= a.y && a.x >= a.z) {
    if (d.x > 0.0) { face = 0.0; s = -d.z / a.x; t = -d.y / a.x; }
    else           { face = 1.0; s =  d.z / a.x; t = -d.y / a.x; }
  } else if (a.y >= a.z) {
    if (d.y > 0.0) { face = 2.0; s = d.x / a.y; t =  d.z / a.y; }
    else           { face = 3.0; s = d.x / a.y; t = -d.z / a.y; }
  } else {
    if (d.z > 0.0) { face = 4.0; s =  d.x / a.z; t = -d.y / a.z; }
    else           { face = 5.0; s = -d.x / a.z; t = -d.y / a.z; }
  }

  return vec3<f32>(face, (s + 1.0) * 0.5, (t + 1.0) * 0.5);
}

fn samplePlanetHeight(dir: vec3<f32>) -> f32 {
  let N = i32(params.planetParams.z);
  let fuv = dirToFaceUV(dir);
  let face = i32(fuv.x);

  let uu = clamp(fuv.y, 0.0, 1.0) * f32(N - 1);
  let vv = clamp(fuv.z, 0.0, 1.0) * f32(N - 1);

  let i0 = vec2<i32>(i32(floor(uu)), i32(floor(vv)));
  let i1 = vec2<i32>(min(i0.x + 1, N - 1), min(i0.y + 1, N - 1));
  let f = vec2<f32>(uu - floor(uu), vv - floor(vv));

  let base = face * N * N;
  let h00 = planetHeights[base + i0.y * N + i0.x];
  let h10 = planetHeights[base + i0.y * N + i1.x];
  let h01 = planetHeights[base + i1.y * N + i0.x];
  let h11 = planetHeights[base + i1.y * N + i1.x];

  let top = mix(h00, h10, f.x);
  let bot = mix(h01, h11, f.x);

  return mix(top, bot, f.y);
}

fn samplePlanetPoint(dir: vec3<f32>) -> vec3<f32> {
  let h = samplePlanetHeight(dir) * params.planetParams.y;
  return params.planetCenter.xyz + dir * (params.planetParams.x + h);
}

fn planetSurfaceNormal(dir: vec3<f32>) -> vec3<f32> {
  let eps = 0.004;
  var t1 = cross(dir, vec3<f32>(0.0, 1.0, 0.0));
  if (length(t1) < 0.01) { t1 = cross(dir, vec3<f32>(1.0, 0.0, 0.0)); }
  t1 = normalize(t1);
  let t2 = normalize(cross(dir, t1));

  var n = vec3<f32>(0.0);
  for (var i = 0u; i < 4u; i++) {
    let a = f32(i) * 1.5707963;
    let ca = cos(a);
    let sa = sin(a);
    let u = t1 * ca + t2 * sa;
    let v = t1 * (-sa) + t2 * ca;

    let pu = samplePlanetPoint(normalize(dir + u * eps));
    let pm = samplePlanetPoint(normalize(dir - u * eps));
    let qu = samplePlanetPoint(normalize(dir + v * eps));
    let qm = samplePlanetPoint(normalize(dir - v * eps));

    let du = (pu - pm) * 0.5;
    let dv = (qu - qm) * 0.5;
    n += cross(du, dv);
  }

  let len = length(n);
  if (len < 0.0001) { return dir; }
  n = n / len;
  if (dot(n, dir) < 0.0) { n = -n; }
  return n;
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

fn resolveMesh(posIn: vec3<f32>, velIn: vec3<f32>, collider: Collider, radius: f32) -> CollisionResult {
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

fn resolvePlanetTerrain(posIn: vec3<f32>, velIn: vec3<f32>, radius: f32) -> CollisionResult {
  var r: CollisionResult;
  r.position = posIn;
  r.velocity = velIn;

  let center = params.planetCenter.xyz;
  let R = params.planetParams.x;
  let hs = params.planetParams.y;

  let delta = posIn - center;
  let dist = length(delta);
  if (dist < 0.0001) { return r; }

  let dir = delta / dist;
  let h = samplePlanetHeight(dir) * hs;
  let floorR = R + h + radius;

  if (dist < floorR) {
    let n = planetSurfaceNormal(dir);

    r.position = center + dir * floorR;

    let vn = dot(velIn, n);
    var v = velIn;
    if (vn < 0.0) {
      v = velIn - n * vn * (1.0 + params.collisionDamping);
    }

    let vt = v - n * dot(v, n);
    let friction = clamp(0.05 * params.deltaTime, 0.0, 0.01);
    v = v - vt * friction;

    r.velocity = v;
  }

  return r;
}

fn resolveAllCollisions(posIn: vec3<f32>, velIn: vec3<f32>, radius: f32) -> CollisionResult {
  var r: CollisionResult;
  r.position = posIn;
  r.velocity = velIn;

  r = resolvePlanetTerrain(r.position, r.velocity, radius);

  for (var c = 0u; c < params.numColliders; c++) {
    let collider = colliders[c];
    let shapeType = collider.data0.w;
    var hit: CollisionResult;

    if (shapeType < 1.5) {
      hit = resolveBox(r.position, r.velocity, collider, radius);
    } else if (shapeType < 2.5) {
      hit = resolveSphere(r.position, r.velocity, collider, radius);
    } else {
      hit = resolveMesh(r.position, r.velocity, collider, radius);
    }

    r.position = hit.position;
    r.velocity = hit.velocity;
  }

  return r;
}

@compute @workgroup_size(256)
fn externalForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  var vel = particles[index].velocity.xyz;
  let pos = particles[index].position.xyz;

  let delta = pos - params.planetCenter.xyz;
  let dist = length(delta);
  if (dist > 0.0001) {
    let dir = delta / dist;
    vel += -dir * params.gravity * params.deltaTime;
  }

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
  particles[index].predictedPosition = vec4<f32>(pos + vel * params.deltaTime, 0.0);
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
  var viscForce = vec3<f32>(0.0);

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
              let w = viscosityKernel(radius, dist);
              viscForce += (neighborVel - vel) * w * (params.particleMass / max(neighborDensity, 0.001));
            }
          }
        }
      }
    }
  }

  var pressureAccel = pressureForce / max(density, 0.001);
  let accelLen = length(pressureAccel);
  let maxAccel = 800.0;
  if (accelLen > maxAccel) {
    pressureAccel *= maxAccel / accelLen;
  }

  let totalAcceleration = select(
    pressureAccel + viscForce * params.viscosityStrength,
    vec3<f32>(0.0),
    density <= 0.0,
  );

  var newVel = vel + totalAcceleration * params.deltaTime;

  particles[index].velocity = vec4<f32>(newVel, 0.0);
}

@compute @workgroup_size(256)
fn integratePositions(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  var pos = particles[index].position.xyz;
  var vel = particles[index].velocity.xyz;

  pos += vel * params.deltaTime;

  let r = resolveAllCollisions(pos, vel, params.particleRadius);
  pos = r.position;
  vel = r.velocity;

  let center = params.planetCenter.xyz;
  let R = params.planetParams.x;
  let hs = params.planetParams.y;
  let delta = pos - center;
  let dist = length(delta);

  let outer = R + hs * 6.0 + 20.0;
  let inner = R - 2.0;

  if (dist > outer || dist < inner || !(dist == dist)) {
    let seed = f32(index) * 0.6180339887;
    let phi = acos(clamp(2.0 * fract(seed * 2.71828) - 1.0, -1.0, 1.0));
    let theta = fract(seed * 1.61803) * 6.2831853;
    let dir = vec3<f32>(
      sin(phi) * cos(theta),
      cos(phi),
      sin(phi) * sin(theta),
    );
    let alt = R + hs * 3.0 + fract(seed * 3.17) * 2.0;
    pos = center + dir * alt;
    vel = vec3<f32>(0.0);
  }

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
