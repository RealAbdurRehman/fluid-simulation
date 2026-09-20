import * as THREE from "three";

export interface RigidBodyInit {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  mass: number;
  volume: number;
  densityRatio: number;
  inertiaLocal: THREE.Vector3;
  drag: number;
  angularDrag: number;
}

export class RigidBody {
  position = new THREE.Vector3();
  quaternion = new THREE.Quaternion();
  linearVelocity = new THREE.Vector3();
  angularVelocity = new THREE.Vector3();
  cobArm = new THREE.Vector3();
  angVelSlow = new THREE.Vector3();

  // Written by applyFluidForces each step; read by the audio layer.
  submergedFraction = 0;
  supportRatio = 0;
  fluidVelocity = new THREE.Vector3();

  mass: number;
  volume: number;
  densityRatio: number;
  inertiaLocal = new THREE.Vector3();
  drag: number;
  angularDrag: number;

  private invInertiaWorld = new THREE.Matrix3();
  private tmpR = new THREE.Vector3();
  private tmpTorque = new THREE.Vector3();
  private tmpMat = new THREE.Matrix4();
  constructor(init: RigidBodyInit) {
    this.position.copy(init.position);
    this.quaternion.copy(init.quaternion);
    this.mass = init.mass;
    this.volume = init.volume;
    this.densityRatio = init.densityRatio;
    this.inertiaLocal.copy(init.inertiaLocal);
    this.drag = init.drag;
    this.angularDrag = init.angularDrag;
  }
  private updateInvInertiaWorld(): void {
    this.tmpMat.makeRotationFromQuaternion(this.quaternion);
    const e = this.tmpMat.elements;
    const ix = 1 / Math.max(this.inertiaLocal.x, 1e-8);
    const iy = 1 / Math.max(this.inertiaLocal.y, 1e-8);
    const iz = 1 / Math.max(this.inertiaLocal.z, 1e-8);
    const o = this.invInertiaWorld.elements;

    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        o[i * 3 + j] =
          e[0 * 4 + i] * ix * e[0 * 4 + j] +
          e[1 * 4 + i] * iy * e[1 * 4 + j] +
          e[2 * 4 + i] * iz * e[2 * 4 + j];
      }
  }
  public applyForceWorld(
    force: THREE.Vector3,
    worldPoint: THREE.Vector3,
    dt: number,
  ): void {
    this.linearVelocity.addScaledVector(force, dt / Math.max(this.mass, 1e-8));

    this.tmpR.subVectors(worldPoint, this.position);
    this.tmpTorque.crossVectors(this.tmpR, force);
    this.applyTorqueWorld(this.tmpTorque, dt);
  }
  public applyTorqueWorld(torque: THREE.Vector3, dt: number): void {
    this.updateInvInertiaWorld();
    const dOmega = torque.clone().applyMatrix3(this.invInertiaWorld);
    this.angularVelocity.addScaledVector(dOmega, dt);
  }
  public integrate(dt: number): void {
    this.position.addScaledVector(this.linearVelocity, dt);

    const ox = this.angularVelocity.x;
    const oy = this.angularVelocity.y;
    const oz = this.angularVelocity.z;
    const qx = this.quaternion.x;
    const qy = this.quaternion.y;
    const qz = this.quaternion.z;
    const qw = this.quaternion.w;

    const dx = 0.5 * (ox * qw + oy * qz - oz * qy);
    const dy = 0.5 * (-ox * qz + oy * qw + oz * qx);
    const dz = 0.5 * (ox * qy - oy * qx + oz * qw);
    const dw = 0.5 * (-ox * qx - oy * qy - oz * qz);

    this.quaternion
      .set(qx + dx * dt, qy + dy * dt, qz + dz * dt, qw + dw * dt)
      .normalize();
  }
}

export function boxInertia(mass: number, s: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3(
    (mass / 12) * (s.y * s.y + s.z * s.z),
    (mass / 12) * (s.x * s.x + s.z * s.z),
    (mass / 12) * (s.x * s.x + s.y * s.y),
  );
}

export function sphereInertia(mass: number, r: number): THREE.Vector3 {
  const i = 0.4 * mass * r * r;
  return new THREE.Vector3(i, i, i);
}
