import * as THREE from "three";

import { config, type ObjectSlotConfig } from "../config";
import type { RigidBody } from "../rigidBody";

import {
  AudioEngine,
  clampToOne,
  lerp,
  rand,
  smoothAsym,
  smoothstep,
  type CrossfadeLoop,
  type PositionalBus,
} from "./audioEngine";

import type { FluidStats } from "./fluidStats";

import {
  DRIP_IDS,
  SPLASH_TIERS,
  THUD_IDS,
  WATER_SPLASH_IDS,
  type SoundId,
} from "./sounds";

import type { SplashEvent } from "./splashDetector";

interface BodyRef {
  index: number;
  body: RigidBody;
  slot: ObjectSlotConfig;
}

export interface SimAudioHooks {
  onStepBegin(dt: number): void;
  onBodyUpdate(
    index: number,
    body: RigidBody,
    slot: ObjectSlotConfig,
    dt: number,
  ): void;
  onTerrainImpact(
    index: number,
    body: RigidBody,
    slot: ObjectSlotConfig,
    normalSpeed: number,
  ): void;
  onWallImpact(
    index: number,
    body: RigidBody,
    slot: ObjectSlotConfig,
    speed: number,
  ): void;
  onPairImpact(a: BodyRef, b: BodyRef, closingSpeed: number): void;
}

export interface AudioFrame {
  camera: THREE.Camera;
  paused: boolean;
  fluid: FluidStats | null;
  listenerSubmersion: number | null;
  particleCount: number;
}

const WARMUP = 0.75;
const ENTER_SUB = 0.1;
const EXIT_SUB = 0.02;
const TRANSITION_COOLDOWN = 0.35;
const IMPACT_COOLDOWN = 0.12;
const WHOOSH_SPEED = 3.0;
const SIZE_REF = 4.0;
const HIST_LEN = 12;
const STALE_STEPS = 30;

const _rel = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _drip = new THREE.Vector3();

interface BodyVoice {
  bus: PositionalBus;
  rumble: CrossfadeLoop | null;
  bubbles: CrossfadeLoop | null;
  lapping: CrossfadeLoop | null;
  pos: THREE.Vector3;
  lastStep: number;
  age: number;
  inWater: boolean;
  lastTransition: number;
  lastImpact: number;
  lastWhoosh: number;
  prevRel: number;
  bubblePulse: number;
  hist: Float32Array;
  histI: number;
}

interface Beds {
  water: CrossfadeLoop | null;
  flowSlow: CrossfadeLoop | null;
  flowMedium: CrossfadeLoop | null;
  flowFast: CrossfadeLoop | null;
  foam: CrossfadeLoop | null;
  lapping: CrossfadeLoop | null;
  underwater: CrossfadeLoop | null;
  bubbles: CrossfadeLoop | null;
  flowFilter: BiquadFilterNode;
}

export class SimAudio implements SimAudioHooks {
  private voices = new Map<number, BodyVoice>();
  private beds: Beds | null = null;

  private step = 0;
  private simTime = 0;
  private time = 0;

  private sSpeed = 0;
  private sFoam = 0;
  private sSurface = 0;
  private sSpray = 0;
  private dripAcc = 0;

  private sprayBaseline = 0;
  private foamBaseline = 0;
  private surfaceBaseline = 0;

  private splashCooldown = 0;

  private uw = 0;
  private camUnder = false;
  private lastCamEvent = -10;

  private globalPulse = 0;
  private readonly engine: AudioEngine;
  constructor(engine: AudioEngine) {
    this.engine = engine;
  }
  public initBeds(): void {
    const e = this.engine;
    const ctx = e.ctx;

    const flowFilter = ctx.createBiquadFilter();
    flowFilter.type = "lowpass";
    flowFilter.frequency.value = 2000;
    flowFilter.Q.value = 0.5;
    flowFilter.connect(e.bedBus);

    this.beds = {
      water: e.createLoop("water", e.bedBus, 1.5),
      flowSlow: e.createLoop("flowSlow", flowFilter, 1.5),
      flowMedium: e.createLoop("flowMedium", flowFilter, 1.5),
      flowFast: e.createLoop("flowFast", flowFilter, 1.5),
      foam: e.createLoop("foam", e.bedBus, 1.0),
      lapping: e.createLoop("lapping", e.bedBus, 1.5),
      underwater: e.createLoop("underwater", e.uwBus, 2.0),
      bubbles: e.createLoop("bubbles", e.uwBus, 1.0),
      flowFilter,
    };
  }
  public onStepBegin(dt: number): void {
    this.step++;
    this.simTime += dt;
  }
  public onBodyUpdate(
    index: number,
    body: RigidBody,
    slot: ObjectSlotConfig,
    dt: number,
  ): void {
    const v = this.voice(index);
    v.lastStep = this.step;
    v.age += dt;
    v.pos.copy(body.position);
    v.bus.setPosition(v.pos.x, v.pos.y, v.pos.z);

    const vel = body.linearVelocity;
    const speed = vel.length();
    v.hist[v.histI] = speed;
    v.histI = (v.histI + 1) % HIST_LEN;

    _rel.copy(vel).sub(body.fluidVelocity);
    const relSpeed = _rel.length();
    const relVy = _rel.y;
    const sub = body.submergedFraction;
    const sizeN = clampToOne(slot.size / SIZE_REF);

    const armed = v.age > WARMUP;
    const cooled = this.simTime - v.lastTransition > TRANSITION_COOLDOWN;
    if (!armed) {
      v.inWater = sub > ENTER_SUB;
    } else if (!v.inWater && sub > ENTER_SUB && cooled) {
      v.inWater = true;
      v.lastTransition = this.simTime;
      this.onEnterWater(v, body, this.peakSpeed(v), sizeN);
    } else if (v.inWater && sub < EXIT_SUB && cooled) {
      v.inWater = false;
      v.lastTransition = this.simTime;
      this.onExitWater(v, this.peakSpeed(v), sizeN);
    }

    if (
      armed &&
      sub > 0.2 &&
      relSpeed > WHOOSH_SPEED &&
      v.prevRel <= WHOOSH_SPEED &&
      this.simTime - v.lastWhoosh > 0.8
    ) {
      v.lastWhoosh = this.simTime;
      const k = clampToOne((relSpeed - WHOOSH_SPEED) / 9);
      this.engine.play("whoosh", {
        gain: 0.2 + 0.5 * k,
        rate: lerp(1.15, 0.85, sizeN) * rand(0.95, 1.05),
        position: v.pos,
      });
    }
    v.prevRel = relSpeed;

    v.bubblePulse *= Math.exp(-dt / 1.3);

    const heavy = clampToOne((body.densityRatio - 0.6) / 2.0);
    const motion = clampToOne(relSpeed / 5);
    const sinkF = smoothstep(0.2, 2.5, -relVy);

    const rumble =
      smoothstep(0.35, 0.9, sub) * heavy * Math.max(motion * 0.6, sinkF);
    const sinkBubbles =
      smoothstep(0.3, 0.9, sub) *
      (0.75 * sinkF + 0.25 * motion) *
      (1 - 0.5 * heavy);

    const surfaceBand = 4 * sub * (1 - sub);
    const bob = clampToOne(Math.abs(vel.y) / 1.5 + relSpeed * 0.15);
    const floatAct = surfaceBand * (0.2 + 0.8 * bob);

    const bubbles = Math.max(
      sinkBubbles,
      v.bubblePulse * smoothstep(0.05, 0.5, sub),
      floatAct * 0.35,
    );
    const lapping = floatAct * 0.8;

    this.drive(v, "rumble", "rumble", rumble * 0.9);
    this.drive(v, "bubbles", "bubbles", bubbles * 0.7);
    this.drive(v, "lapping", "lapping", lapping);
  }
  public onTerrainImpact(
    index: number,
    body: RigidBody,
    slot: ObjectSlotConfig,
    normalSpeed: number,
  ): void {
    const v = this.voices.get(index);
    if (!v || v.age < WARMUP) return;
    if (normalSpeed < config.audio.minImpactSpeed) return;
    if (this.simTime - v.lastImpact < IMPACT_COOLDOWN) return;
    v.lastImpact = this.simTime;
    this.playImpact(
      v.pos,
      body.mass,
      body.submergedFraction,
      slot.size,
      normalSpeed,
      0.92,
    );
  }
  public onWallImpact(
    index: number,
    body: RigidBody,
    slot: ObjectSlotConfig,
    speed: number,
  ): void {
    const v = this.voices.get(index);
    if (!v || v.age < WARMUP) return;
    if (speed < config.audio.minImpactSpeed) return;
    if (this.simTime - v.lastImpact < IMPACT_COOLDOWN) return;
    v.lastImpact = this.simTime;
    this.playImpact(
      v.pos,
      body.mass,
      body.submergedFraction,
      slot.size,
      speed,
      1.12,
    );
  }
  public onPairImpact(a: BodyRef, b: BodyRef, closingSpeed: number): void {
    if (closingSpeed < config.audio.minImpactSpeed) return;
    const va = this.voices.get(a.index);
    const vb = this.voices.get(b.index);
    if (!va || !vb || va.age < WARMUP || vb.age < WARMUP) return;
    if (
      this.simTime - va.lastImpact < IMPACT_COOLDOWN ||
      this.simTime - vb.lastImpact < IMPACT_COOLDOWN
    )
      return;
    va.lastImpact = vb.lastImpact = this.simTime;

    const mA = a.body.mass;
    const mB = b.body.mass;
    const reduced = (mA * mB) / Math.max(mA + mB, 1e-6);
    _mid.addVectors(va.pos, vb.pos).multiplyScalar(0.5);
    const sub = 0.5 * (a.body.submergedFraction + b.body.submergedFraction);
    this.playImpact(
      _mid,
      reduced,
      sub,
      0.5 * (a.slot.size + b.slot.size),
      closingSpeed,
      1.0,
    );
  }
  public onInteractionStart(): void {
    this.engine.play("whoosh", { gain: 0.35, rate: rand(0.9, 1.1) });
  }
  public feedSplashes(events: SplashEvent[], dt: number): void {
    this.splashCooldown -= dt;

    if (this.splashCooldown > 0) return;
    if (this.uw > 0.7) return;
    if (!this.engine.canPlay) return;
    if (events.length === 0) return;

    let energy = 0;
    let count = 0;
    const pos = new THREE.Vector3();
    for (const ev of events) {
      const total = energy + ev.energy;
      const w = total > 0 ? ev.energy / total : 0.5;
      pos.lerp(new THREE.Vector3(ev.x, ev.y, ev.z), w);
      energy = total;
      count += ev.count;
    }

    const k = clampToOne(Math.sqrt(energy) / 18);
    const gain = (0.22 + 0.68 * Math.pow(k, 0.75)) * (1 - 0.55 * this.uw);
    if (gain < 0.03) return;

    const id = this.engine.pickVariant("waterSplash", WATER_SPLASH_IDS);
    const sizeN = clampToOne(k);

    this.engine.play(id, {
      gain,
      rate: rand(0.9, 1.1) * lerp(1.15, 0.85, sizeN),
      position: pos,
    });

    this.splashCooldown = 0.2 + Math.random() * 0.1;
    this.globalPulse = Math.max(this.globalPulse, k * 0.4);
  }
  public update(dt: number, f: AudioFrame): void {
    const e = this.engine;
    const a = config.audio;
    this.time += dt;

    e.setListener(f.camera);
    e.setMuted(f.paused || !a.enabled);
    e.setVolumes(a.master, a.ambience, a.effects);

    const rawSub = f.listenerSubmersion ?? 0;
    const canEvent =
      this.beds !== null &&
      this.time - this.lastCamEvent > 0.8 &&
      this.time > 1.5;
    if (!this.camUnder && rawSub > 0.6) {
      this.camUnder = true;
      if (canEvent) {
        this.lastCamEvent = this.time;
        e.play("splashSmall", { gain: 0.35, rate: 0.9, lowpassHz: 3500 });
        this.globalPulse = Math.max(this.globalPulse, 0.8);
      }
    } else if (this.camUnder && rawSub < 0.25) {
      this.camUnder = false;
      if (canEvent) {
        this.lastCamEvent = this.time;
        e.play("splashSmall", { gain: 0.18, rate: 1.2 });
        e.play(e.pickVariant("drip", DRIP_IDS), {
          gain: 0.2,
          delay: 0.15,
          pan: rand(-0.4, 0.4),
        });
      }
    }

    this.uw = smoothAsym(this.uw, smoothstep(0.3, 0.8, rawSub), dt, 0.08, 0.2);
    e.setUnderwater(this.uw, a.underwaterCutoff);

    const s = f.fluid;
    if (s) {
      this.sSpeed = smoothAsym(this.sSpeed, s.avgSpeed, dt, 0.15, 0.2);
      this.sFoam = smoothAsym(this.sFoam, s.avgFoam, dt, 0.2, 0.35);
      this.sSurface = smoothAsym(
        this.sSurface,
        s.avgSurfaceSpeed,
        dt,
        0.2,
        0.3,
      );
      this.sSpray = smoothAsym(this.sSpray, s.sprayCount, dt, 0.1, 0.4);
    }

    const invCount = 1 / Math.max(f.particleCount, 1);
    const sprayNorm = s ? this.sSpray * invCount : 0;

    this.sprayBaseline = smoothAsym(
      this.sprayBaseline,
      sprayNorm,
      dt,
      1.0,
      3.0,
    );
    this.foamBaseline = smoothAsym(this.foamBaseline, this.sFoam, dt, 1.0, 3.0);
    this.surfaceBaseline = smoothAsym(
      this.surfaceBaseline,
      this.sSurface,
      dt,
      1.0,
      3.0,
    );
    const flowF = smoothstep(a.flowMinSpeed, a.flowMaxSpeed, this.sSpeed);
    const foamG = smoothstep(a.foamMin, a.foamMax, this.sFoam);
    const lapG =
      smoothstep(0.1, 0.8, this.sSurface) *
      (1 - 0.6 * smoothstep(2.5, 5.0, this.sSurface));
    const air = 1 - 0.75 * this.uw;
    const hasFluid = f.particleCount > 0 ? 1 : 0;

    const b = this.beds;
    if (b) {
      const wS = 1 - smoothstep(0.0, 0.5, flowF);
      const wM = 1 - Math.abs(2 * flowF - 1);
      const wF = smoothstep(0.45, 0.95, flowF);
      const norm = 1 / Math.max(Math.hypot(wS, wM, wF), 1e-4);
      const flowGain = smoothstep(0.02, 0.25, flowF) * 0.9 * air;

      const FLOW_ATK = 0.1;
      const FLOW_REL = 0.22;
      b.flowSlow?.setTarget(flowGain * wS * norm, FLOW_ATK, FLOW_REL);
      b.flowMedium?.setTarget(flowGain * wM * norm, FLOW_ATK, FLOW_REL);
      b.flowFast?.setTarget(flowGain * wF * norm, FLOW_ATK, FLOW_REL);
      b.flowFilter.frequency.setTargetAtTime(
        1800 * Math.pow(14000 / 1800, flowF),
        e.ctx.currentTime,
        0.1,
      );

      b.water?.setTarget(
        0.3 * (0.55 + 0.45 * flowF) * air * hasFluid,
        0.3,
        0.5,
      );
      b.foam?.setTarget(foamG * 0.35 * air, 0.35, 0.6);
      b.lapping?.setTarget(
        lapG * 0.7 * smoothstep(0.02, 0.1, s?.surfaceFraction ?? 0) * air,
        0.3,
        0.5,
      );

      b.underwater?.setTarget(this.uw * 0.75, 0.15, 0.3);
      b.bubbles?.setTarget(
        this.uw * (0.12 + 0.55 * foamG) +
          0.6 * this.globalPulse * (0.4 + 0.6 * this.uw),
        0.1,
        0.5,
      );
    }
    this.globalPulse *= Math.exp(-dt / 1.5);

    if (s && this.uw < 0.5 && !f.paused) {
      this.dripAcc = Math.min(
        this.dripAcc + clampToOne(this.sSpray / 250) * 8 * dt,
        2,
      );

      for (let n = 0; this.dripAcc >= 1 && n < 3; n++) {
        this.dripAcc -= 1;
        e.play(e.pickVariant("drip", DRIP_IDS), {
          gain: rand(0.05, 0.2),
          rate: rand(0.85, 1.3),
          pan: rand(-0.7, 0.7),
          delay: rand(0, 0.05),
        });
      }
    }

    for (const [index, v] of this.voices)
      if (this.step - v.lastStep > STALE_STEPS) {
        e.disposeLoop(v.rumble);
        e.disposeLoop(v.bubbles);
        e.disposeLoop(v.lapping);
        v.bus.dispose();
        this.voices.delete(index);
      }

    e.update(dt);
  }
  private voice(index: number): BodyVoice {
    let v = this.voices.get(index);
    if (!v) {
      v = {
        bus: this.engine.createPositionalBus(),
        rumble: null,
        bubbles: null,
        lapping: null,
        pos: new THREE.Vector3(),
        lastStep: this.step,
        age: 0,
        inWater: false,
        lastTransition: -10,
        lastImpact: -10,
        lastWhoosh: -10,
        prevRel: 0,
        bubblePulse: 0,
        hist: new Float32Array(HIST_LEN),
        histI: 0,
      };

      this.voices.set(index, v);
    }

    return v;
  }
  private drive(
    v: BodyVoice,
    key: "rumble" | "bubbles" | "lapping",
    id: SoundId,
    target: number,
  ): void {
    let loop = v[key];
    if (!loop) {
      if (target < 0.01) return;
      loop = this.engine.createLoop(id, v.bus.input, 1.0);
      v[key] = loop;
    }

    loop?.setTarget(target, 0.1, 0.4);
  }
  private peakSpeed(v: BodyVoice): number {
    let m = 0;
    for (let i = 0; i < HIST_LEN; i++) if (v.hist[i] > m) m = v.hist[i];

    return m;
  }
  private onEnterWater(
    v: BodyVoice,
    body: RigidBody,
    peak: number,
    sizeN: number,
  ): void {
    if (peak < config.audio.minImpactSpeed) return;

    const energy = 0.5 * body.mass * peak * peak;
    const k = clampToOne(Math.sqrt(energy / config.audio.splashEnergyRef));

    const x = k * 2;
    const i = Math.min(1, Math.floor(x));
    const f = x - i;
    const base = 0.25 + 0.75 * Math.pow(k, 0.8);
    const rate = lerp(1.15, 0.8, sizeN) * rand(0.94, 1.06);

    this.engine.play(SPLASH_TIERS[i], {
      gain: base * Math.cos(f * Math.PI * 0.5),
      rate,
      position: v.pos,
    });
    if (f > 0.02)
      this.engine.play(SPLASH_TIERS[i + 1], {
        gain: base * Math.sin(f * Math.PI * 0.5),
        rate,
        position: v.pos,
      });

    if (k > 0.5)
      this.engine.play("splashSmall", {
        gain: 0.3 * k,
        rate: rand(1.05, 1.25),
        delay: rand(0.06, 0.14),
        position: v.pos,
      });
    this.scheduleDrips(v.pos, Math.round(1 + k * 5), 0.15, 1.2, 0.4, sizeN);

    v.bubblePulse = Math.max(v.bubblePulse, k);
    this.globalPulse = Math.max(this.globalPulse, k * 0.7);
  }
  private onExitWater(v: BodyVoice, peak: number, sizeN: number): void {
    this.scheduleDrips(
      v.pos,
      1 + Math.floor(sizeN * 3),
      0.03,
      0.8,
      0.28,
      sizeN,
    );

    if (peak > 2.0) {
      const k = clampToOne((peak - 2.0) / 8);
      this.engine.play("splashSmall", {
        gain: 0.12 + 0.25 * k,
        rate: lerp(1.35, 1.05, sizeN) * rand(0.95, 1.05),
        position: v.pos,
      });
    }
  }
  private scheduleDrips(
    origin: THREE.Vector3,
    count: number,
    minDelay: number,
    maxDelay: number,
    maxGain: number,
    sizeN: number,
  ): void {
    for (let n = 0; n < count; n++) {
      _drip
        .set(rand(-1, 1), rand(-1, 0), rand(-1, 1))
        .multiplyScalar(0.5 + sizeN * 1.5)
        .add(origin);
      this.engine.play(this.engine.pickVariant("drip", DRIP_IDS), {
        gain: rand(0.35, 1.0) * maxGain,
        rate: rand(0.85, 1.3) * lerp(1.1, 0.9, sizeN),
        delay: rand(minDelay, maxDelay),
        position: _drip,
      });
    }
  }
  private playImpact(
    pos: THREE.Vector3,
    mass: number,
    submerged: number,
    size: number,
    speed: number,
    pitchMul: number,
  ): void {
    const energy = 0.5 * mass * speed * speed;
    const k = clampToOne(Math.sqrt(energy / config.audio.impactEnergyRef));
    const sizeN = clampToOne(size / SIZE_REF);

    this.engine.play(this.engine.pickVariant("thud", THUD_IDS), {
      gain: (0.12 + 0.88 * Math.pow(k, 0.85)) * lerp(1, 0.3, submerged),
      rate: lerp(1.3, 0.75, sizeN) * pitchMul * rand(0.94, 1.06),
      lowpassHz: 16000 * Math.pow(800 / 16000, submerged),
      position: pos,
    });
  }
}
