export const fluidComputeShaderWGSL = /* wgsl */ `
struct Particle {
  position: vec4<f32>,
  predictedPosition: vec4<f32>,
  velocity: vec4<f32>,
  density: vec4<f32>,
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
  interactionExtra: vec4<f32>,
  vortexFalloff:    vec4<f32>,
  terrainMeta: vec4<f32>,
  gridInfo: vec4<u32>,
  gridInfo2: vec4<f32>,
  foamParams: vec4<f32>,
  windDir:    vec4<f32>,
  windParams: vec4<f32>,
};

struct ProbeSample {
  density: f32,
  velX: f32,
  velY: f32,
  velZ: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read> sdfData: array<f32>;
@group(0) @binding(4) var<storage, read> probePositions: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> probeSamples: array<ProbeSample>;
@group(0) @binding(6) var<storage, read> terrain: array<f32>;
@group(0) @binding(7) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> sortedIndices: array<u32>;

var<workgroup> blockSums: array<u32, 256>;

fn hash33(p: vec3<f32>) -> f32 {
  var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + 33.33);

  return fract((q.x + q.y) * q.z);
}

fn valueNoise3(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  let c000 = hash33(i);
  let c100 = hash33(i + vec3<f32>(1.0, 0.0, 0.0));
  let c010 = hash33(i + vec3<f32>(0.0, 1.0, 0.0));
  let c110 = hash33(i + vec3<f32>(1.0, 1.0, 0.0));
  let c001 = hash33(i + vec3<f32>(0.0, 0.0, 1.0));
  let c101 = hash33(i + vec3<f32>(1.0, 0.0, 1.0));
  let c011 = hash33(i + vec3<f32>(0.0, 1.0, 1.0));
  let c111 = hash33(i + vec3<f32>(1.0, 1.0, 1.0));

  let c00 = mix(c000, c100, u.x);
  let c10 = mix(c010, c110, u.x);
  let c01 = mix(c001, c101, u.x);
  let c11 = mix(c011, c111, u.x);

  return mix(mix(c00, c10, u.y), mix(c01, c11, u.y), u.z);
}

fn qConjugate(q: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn qRotateVec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let qv = q.xyz;
  let uv = cross(qv, v);
  let uuv = cross(qv, uv);
  return v + ((uv * q.w) + uuv) * 2.0;
}

fn gridNumCells() -> u32 { return params.gridInfo.x; }
fn gridSizeU() -> u32 { return params.gridInfo.y; }
fn gridSearchRadius() -> i32 { return i32(params.gridInfo.z); }
fn gridCellSize() -> f32 { return params.gridInfo2.x; }

fn worldToLocal(worldPos: vec3<f32>) -> vec3<f32> {
  let container = colliders[0];
  let invRot = qConjugate(container.rotation);
  return qRotateVec(invRot, worldPos - container.data0.xyz);
}

fn localToCell(localPos: vec3<f32>) -> vec3<i32> {
  let gs = f32(gridSizeU());
  let half = gs * 0.5;
  let c = floor(localPos / gridCellSize() + vec3<f32>(half));
  return vec3<i32>(clamp(c, vec3<f32>(0.0), vec3<f32>(gs - 1.0)));
}

fn cellToIndex(cell: vec3<i32>) -> u32 {
  let gs = gridSizeU();
  return u32(cell.x) + u32(cell.y) * gs + u32(cell.z) * gs * gs;
}

fn gridRead(i: u32) -> u32 { return atomicLoad(&grid[i]); }

@compute @workgroup_size(256)
fn externalForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index < gridNumCells()) {
    atomicStore(&grid[index], 0u);
  }

  if (index >= params.numParticles) { return; }

  var vel = particles[index].velocity.xyz;
  let pos = particles[index].position.xyz;

  vel.y += -params.gravity * params.deltaTime;

  if (params.windDir.w > 0.0001) {
    let windDir    = params.windDir.xyz;
    let strength   = params.windDir.w;
    let t          = params.windParams.x;
    let turbulence = params.windParams.y;

    var windAccel = windDir * strength;

    if (turbulence > 0.001) {
      let phase = pos * (0.35 + turbulence * 0.6)
                + vec3<f32>(t * 0.9, t * 0.7, t * 1.1);

      let tx = valueNoise3(phase) - 0.5;
      let ty = valueNoise3(phase + vec3<f32>(31.4, 17.2,  9.8)) - 0.5;
      let tz = valueNoise3(phase + vec3<f32>(11.1,  5.5, 27.3)) - 0.5;

      let turbAmp = turbulence * 0.5;
      windAccel += vec3<f32>(tx, ty * 0.4, tz) * strength * turbAmp;
    }

    vel += windAccel * params.deltaTime;
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

      if (params.interactionExtra.x < 0.5) {
        let force = (dir * params.interactionStrength - vel) * (centerT * centerT);
        vel += force * params.deltaTime;
      } else {
        let falloff = max(params.vortexFalloff.x, 0.01);
        let w = pow(centerT, falloff);

        let swirlRaw = cross(rayDir, dir);
        let sl = length(swirlRaw);
        var tHat = vec3<f32>(0.0);
        if (sl > 1e-4) {
          tHat = swirlRaw / sl;
        }

        let VORTEX_SWIRL_SCALE  = 3.0; 
        let VORTEX_INWARD_SCALE = 0.15;
        let VORTEX_LIFT_SCALE   = 0.35;
        let swirlSpeed  = params.interactionExtra.y * VORTEX_SWIRL_SCALE;
        let inwardSpeed = params.interactionExtra.z * VORTEX_INWARD_SCALE;
        let axialSpeed  = params.interactionExtra.w * VORTEX_LIFT_SCALE;

        let rFrac = clamp(dist / params.interactionRadius, 0.0, 1.0);
        let tangentialSpeed = swirlSpeed * rFrac;

        let targetVel = tHat    * tangentialSpeed * w
                      + dir     * inwardSpeed    * w
                      + rayDir  * axialSpeed     * w;

        let responsiveness = 10.0;
        let k = (1.0 - exp(-responsiveness * params.deltaTime)) * w;
        vel = mix(vel, targetVel, k);
      }
    }
  }

  particles[index].velocity = vec4<f32>(vel, 0.0);
  particles[index].predictedPosition =
    vec4<f32>(pos + vel * params.deltaTime, 0.0);
}

@compute @workgroup_size(256)
fn countParticles(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.numParticles) { return; }
  let pos = particles[i].predictedPosition.xyz;
  let ci = cellToIndex(localToCell(worldToLocal(pos)));
  atomicAdd(&grid[ci], 1u);
}

@compute @workgroup_size(256)
fn prefixSumGrid(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = gridNumCells();
  let chunk = (n + 255u) / 256u;
  let start = tid * chunk;
  let end = min(start + chunk, n);

  var sum: u32 = 0u;
  for (var i = start; i < end; i = i + 1u) {
    let v = atomicLoad(&grid[i]);
    atomicStore(&grid[i], sum);
    sum = sum + v;
  }
  blockSums[tid] = sum;
  workgroupBarrier();

  var step: u32 = 1u;
  for (var s = 0u; s < 8u; s = s + 1u) {
    var v: u32 = 0u;
    if (tid >= step) { v = blockSums[tid - step]; }
    workgroupBarrier();
    blockSums[tid] = blockSums[tid] + v;
    workgroupBarrier();
    step = step * 2u;
  }

  let offset = select(0u, blockSums[tid - 1u], tid > 0u);

  for (var i = start; i < end; i = i + 1u) {
    let v = atomicLoad(&grid[i]) + offset;
    atomicStore(&grid[i], v);
    atomicStore(&grid[n + 1u + i], v);
  }

  if (tid == 0u) {
    atomicStore(&grid[n], params.numParticles);
  }
}

@compute @workgroup_size(256)
fn scatterParticles(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.numParticles) { return; }
  let pos = particles[i].predictedPosition.xyz;
  let ci = cellToIndex(localToCell(worldToLocal(pos)));
  let slot = atomicAdd(&grid[gridNumCells() + 1u + ci], 1u);
  sortedIndices[slot] = i;
}

@compute @workgroup_size(256)
fn calculateDensities(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.numParticles) { return; }

  let prevFoam = particles[index].density.z;

  let pos = particles[index].predictedPosition.xyz;
  let centerCell = localToCell(worldToLocal(pos));
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;
  let search = gridSearchRadius();
  let gs = i32(gridSizeU());
  let gsSq = gs * gs;
  let ciCenter = i32(cellToIndex(centerCell));

  let massPoly6 = params.particleMass * params.poly6Factor;
  let massNear = params.particleMass * params.nearSpikyGradFactor;

  var density = 0.0;
  var nearDensity = 0.0;

  for (var oz = -search; oz <= search; oz++) {
    let zc = centerCell.z + oz;
    if (zc < 0 || zc >= gs) { continue; }
    let zOff = oz * gsSq;
    for (var oy = -search; oy <= search; oy++) {
      let yc = centerCell.y + oy;
      if (yc < 0 || yc >= gs) { continue; }
      let yOff = oy * gs;
      for (var ox = -search; ox <= search; ox++) {
        let xc = centerCell.x + ox;
        if (xc < 0 || xc >= gs) { continue; }
        let ci = u32(ciCenter + ox + yOff + zOff);
        let s = gridRead(ci);
        let e = gridRead(ci + 1u);

        for (var i = s; i < e; i = i + 1u) {
          let ni = sortedIndices[i];
          let nPos = particles[ni].predictedPosition.xyz;
          let diff = nPos - pos;
          let d2 = dot(diff, diff);
          if (d2 <= radiusSqr) {
            let d = sqrt(d2);
            let v = radius - d;
            let v2 = v * v;
            density += v2 * massPoly6;
            nearDensity += v2 * v * massNear;
          }
        }
      }
    }
  }

  particles[index].density = vec4<f32>(density, nearDensity, prevFoam, 0.0);
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

fn terrainRepulsionForce(pos: vec3<f32>, radius: f32) -> vec3<f32> {
  if (params.terrainMeta.w < 0.5) { return vec3<f32>(0.0); }

  let container = colliders[0];
  let localPos = qRotateVec(qConjugate(container.rotation), pos - container.data0.xyz);
  let baseY = -params.boundsHeight * 0.5;

  let ts = terrainSample(localPos.x, localPos.z);
  let n = ts.yzw;
  let perp = (localPos.y - (baseY + ts.x)) * n.y;
  let d = max(perp - radius, -radius);       // distance past the contact plane
  if (d >= radius) { return vec3<f32>(0.0); }

  let localForce = n * ((radius - d) / radius);
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

  let center = collider.data0.xyz;
  if (!(center.x == center.x) || !(center.y == center.y) || !(center.z == center.z)) {
    return q;
  }
  if (!(worldPos.x == worldPos.x) || !(worldPos.y == worldPos.y) || !(worldPos.z == worldPos.z)) {
    return q;
  }

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
  if (!q.valid) { return r; }

  let skin = collider.sdfOrigin.w * max(collider.data2.x, 0.0001);
  let dist = q.dist - skin;
  if (dist >= radius) { return r; }

  let push = min(radius - dist, radius);
  r.position = posIn + q.worldNormal * push;

  let relVel = velIn - collider.velocity.xyz;
  let vn = dot(relVel, q.worldNormal);
  var newRelVel = relVel;
  if (vn < 0.0) { newRelVel -= q.worldNormal * vn * (1.0 + collider.data2.w); }
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

fn terrainSample(x: f32, z: f32) -> vec4<f32> {
  let N = i32(params.terrainMeta.x);
  let extX = params.terrainMeta.y;
  let extZ = params.terrainExtentZ;
  let hs = params.terrainMeta.z;
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

  let h = mix(mix(h00, h10, fx), mix(h01, h11, fx), fy) * hs;
  let dhdx = mix(h10 - h00, h11 - h01, fy) * hs / cellX;
  let dhdz = mix(h01 - h00, h11 - h10, fx) * hs / cellZ;
  return vec4<f32>(h, normalize(vec3<f32>(-dhdx, 1.0, -dhdz)));
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

  let baseY = -params.boundsHeight * 0.5;

  let ts = terrainSample(localPos.x, localPos.z);
  let n = ts.yzw;
  let surfY = baseY + ts.x;
  let signedDist = localPos.y - surfY;
  let perpDist = signedDist * n.y;
  if (perpDist >= radius) { return r; }

  var localVel = qRotateVec(invRot, velIn - container.velocity.xyz);

  if (perpDist < -radius) {
    localPos.y = surfY + radius;        
  } else {
    localPos += n * (radius - perpDist);
  }

  let vn = dot(localVel, n);
  if (vn < 0.0) {
    localVel -= n * vn * (1.0 + container.data2.w);
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

  let targetDensity = params.targetDensity;
  let pressureMul = params.pressureMultiplier;
  let nearPressureMul = params.nearPressureMultiplier;

  let pressure = (density - targetDensity) * pressureMul;
  let nearPressure = nearDensity * nearPressureMul;

  let centerCell = localToCell(worldToLocal(pos));
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;
  let search = gridSearchRadius();
  let gs = i32(gridSizeU());
  let gsSq = gs * gs;
  let ciCenter = i32(cellToIndex(centerCell));

  let mass = params.particleMass;
  let spiky = params.spikyGradFactor;
  let nearSpiky = params.nearSpikyGradFactor;
  let viscFactor = params.viscFactor;
  let dt = params.deltaTime;
  let invDt = 1.0 / max(dt, 0.0001);
  let viscScale = params.viscosityStrength * dt;

  var pressureForce = vec3<f32>(0.0);
  var viscosityForce = vec3<f32>(0.0);
  var trappedAir = 0.0;

  for (var oz = -search; oz <= search; oz++) {
    let zc = centerCell.z + oz;
    if (zc < 0 || zc >= gs) { continue; }
    let zOff = oz * gsSq;
    for (var oy = -search; oy <= search; oy++) {
      let yc = centerCell.y + oy;
      if (yc < 0 || yc >= gs) { continue; }
      let yOff = oy * gs;
      for (var ox = -search; ox <= search; ox++) {
        let xc = centerCell.x + ox;
        if (xc < 0 || xc >= gs) { continue; }
        let ci = u32(ciCenter + ox + yOff + zOff);
        let s = gridRead(ci);
        let e = gridRead(ci + 1u);

        for (var i = s; i < e; i = i + 1u) {
          let neighborIndex = sortedIndices[i];
          if (neighborIndex == index) { continue; }

          let neighborPos = particles[neighborIndex].predictedPosition.xyz;
          let diff = neighborPos - pos;
          let distSqr = dot(diff, diff);

          if (distSqr <= radiusSqr) {
            let dist = max(sqrt(distSqr), 0.001);
            let dir = diff / dist;

            let nd = particles[neighborIndex].density;
            let neighborDensity = nd.x;
            let neighborNearDensity = nd.y;

            if (neighborDensity > 0.0) {
              let neighborPressure = (neighborDensity - targetDensity) * pressureMul;
              let sharedPressure = (pressure + neighborPressure) * 0.5;
              let slope = (dist - radius) * spiky;
              pressureForce += dir * (sharedPressure * slope * mass / neighborDensity);
            }

            if (neighborNearDensity > 0.0) {
              let neighborNearPressure = neighborNearDensity * nearPressureMul;
              let sharedNearPressure = (nearPressure + neighborNearPressure) * 0.5;
              let vNear = radius - dist;
              let nearSlope = -vNear * vNear * nearSpiky;
              pressureForce += dir * (sharedNearPressure * nearSlope * mass / neighborNearDensity);
            }

            let neighborVel = particles[neighborIndex].velocity.xyz;
            let vVisc = radius - dist;
            let viscWeight = vVisc * vVisc * viscFactor;
            let viscDamping = min(viscWeight * viscScale, 0.40) * invDt;
            viscosityForce += (neighborVel - vel) * viscDamping;

            let velDiff = vel - neighborVel;
            let velDiffLen = length(velDiff);
            if (velDiffLen > 1e-4) {
              let velDiffDir = velDiff / velDiffLen;
              let align = clamp(1.0 - dot(velDiffDir, -dir), 0.0, 2.0);
              trappedAir += vVisc * vVisc * align * velDiffLen;
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

  let wallAccel = containerRepulsionForce(pos, params.particleRadius) + terrainRepulsionForce(pos, params.particleRadius);
  particles[index].velocity = vec4<f32>(vel + (totalAcceleration + wallAccel) * params.deltaTime, 0.0);

  let speed = length(vel);
  let kinetic = smoothstep(params.foamParams.z, params.foamParams.w, speed);
  let surfaceExposure = clamp(1.0 - density / max(targetDensity, 0.0001), 0.0, 1.0);
  let foamGain = trappedAir * kinetic * surfaceExposure * params.foamParams.x * dt;

  let d = particles[index].density;
  let decayedFoam = d.z * exp(-params.foamParams.y * dt);
  let newFoam = clamp(decayedFoam + foamGain, 0.0, 1.0);
  particles[index].density = vec4<f32>(d.x, d.y, newFoam, d.w);
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
  let centerCell = localToCell(worldToLocal(pos));
  let radius = params.smoothingRadius;
  let radiusSqr = radius * radius;
  let search = gridSearchRadius();
  let gs = i32(gridSizeU());
  let gsSq = gs * gs;
  let ciCenter = i32(cellToIndex(centerCell));
  let massPoly6 = params.particleMass * params.poly6Factor;

  var density = 0.0;
  var weightedVel = vec3<f32>(0.0);
  var weightSum = 0.0;

  for (var oz = -search; oz <= search; oz++) {
    let zc = centerCell.z + oz;
    if (zc < 0 || zc >= gs) { continue; }
    let zOff = oz * gsSq;
    for (var oy = -search; oy <= search; oy++) {
      let yc = centerCell.y + oy;
      if (yc < 0 || yc >= gs) { continue; }
      let yOff = oy * gs;
      for (var ox = -search; ox <= search; ox++) {
        let xc = centerCell.x + ox;
        if (xc < 0 || xc >= gs) { continue; }
        let ci = u32(ciCenter + ox + yOff + zOff);
        let s = gridRead(ci);
        let e = gridRead(ci + 1u);

        for (var i = s; i < e; i = i + 1u) {
          let nIdx = sortedIndices[i];
          let nPos = particles[nIdx].predictedPosition.xyz;
          let diff = nPos - pos;
          let d2 = dot(diff, diff);
          if (d2 <= radiusSqr) {
            let d = sqrt(d2);
            let v = radius - d;
            let w = v * v * massPoly6;
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
