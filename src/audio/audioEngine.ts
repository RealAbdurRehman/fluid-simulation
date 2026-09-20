import * as THREE from "three";

import { AUDIO_BASE, SOUND_FILES, type SoundId } from "./sounds";

export const clampToOne = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);

export const lerp = (a: number, b: number, t: number): number =>
  a + (b - a) * t;

export const rand = (a: number, b: number): number =>
  a + Math.random() * (b - a);

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = clampToOne((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

export function smoothAsym(
  current: number,
  target: number,
  dt: number,
  tauUp: number,
  tauDown: number,
): number {
  const tau = target > current ? tauUp : tauDown;
  return (
    current + (target - current) * (1 - Math.exp(-dt / Math.max(tau, 1e-4)))
  );
}

const REF_DISTANCE = 10;
const ROLLOFF = 0.5;

const MAX_ONE_SHOTS = 40;
const LOOP_LOOKAHEAD = 1.0;
const LOOP_IDLE_STOP = 0.6;
const LOOP_EPS = 0.0015;

const FADE_STEPS = 64;
const FADE_IN = new Float32Array(FADE_STEPS);
const FADE_OUT = new Float32Array(FADE_STEPS);

for (let i = 0; i < FADE_STEPS; i++) {
  const t = i / (FADE_STEPS - 1);
  FADE_IN[i] = Math.sin(t * Math.PI * 0.5);
  FADE_OUT[i] = Math.cos(t * Math.PI * 0.5);
}

function setPannerPosition(
  p: PannerNode,
  x: number,
  y: number,
  z: number,
): void {
  p.positionX.value = x;
  p.positionY.value = y;
  p.positionZ.value = z;
}

export interface OneShotOptions {
  gain?: number;
  rate?: number;
  delay?: number;
  pan?: number;
  position?: THREE.Vector3;
  lowpassHz?: number;
  highpassHz?: number;
}

export interface PositionalBus {
  input: GainNode;
  setPosition(x: number, y: number, z: number): void;
  dispose(): void;
}

interface Segment {
  src: AudioBufferSourceNode;
  gain: GainNode;
}

export class CrossfadeLoop {
  private readonly out: GainNode;
  private readonly xfade: number;
  private segments: Segment[] = [];
  private running = false;
  private nextStart = 0;
  private first = true;
  private idle = 0;
  private disposed = false;

  private level = 0;
  private target = 0;
  private attack = 0.12;
  private release = 0.5;

  private readonly ctx: AudioContext;
  private readonly buffer: AudioBuffer;
  constructor(
    ctx: AudioContext,
    buffer: AudioBuffer,
    dest: AudioNode,
    crossfadeSeconds: number,
  ) {
    this.ctx = ctx;
    this.buffer = buffer;

    this.xfade = Math.min(crossfadeSeconds, buffer.duration / 4);
    this.out = ctx.createGain();
    this.out.gain.value = 0;
    this.out.connect(dest);
  }
  public setTarget(v: number, attack = 0.12, release = 0.5): void {
    this.target = v;
    this.attack = attack;
    this.release = release;
  }
  get currentLevel(): number {
    return this.level;
  }
  public update(dt: number): void {
    if (this.disposed) return;
    const now = this.ctx.currentTime;

    this.level = smoothAsym(
      this.level,
      this.target,
      dt,
      this.attack,
      this.release,
    );

    this.out.gain.setTargetAtTime(this.level, now, 0.03);

    const audible = this.target > LOOP_EPS || this.level > LOOP_EPS;
    if (audible) {
      this.idle = 0;
      if (!this.running) {
        this.running = true;
        this.first = true;
        this.nextStart = now + 0.03;
      }
      this.schedule(now);
    } else if (this.running) {
      this.idle += dt;
      if (this.idle > LOOP_IDLE_STOP) this.stopAll(now);
    }
  }
  public dispose(): void {
    if (this.disposed) return;

    this.stopAll(this.ctx.currentTime);
    this.out.disconnect();
    this.disposed = true;
  }
  private schedule(now: number): void {
    while (this.nextStart < now + LOOP_LOOKAHEAD) {
      const t = Math.max(this.nextStart, now + 0.01);
      const dur = this.buffer.duration;

      const maxOffset = Math.max(0, dur - 3 * this.xfade);
      const offset = this.first ? Math.random() * maxOffset : 0;
      this.first = false;

      const segLen = dur - offset;
      const g = this.ctx.createGain();
      g.gain.value = 0;
      g.gain.setValueCurveAtTime(FADE_IN, t, this.xfade);
      g.gain.setValueCurveAtTime(FADE_OUT, t + segLen - this.xfade, this.xfade);
      g.connect(this.out);

      const src = this.ctx.createBufferSource();
      src.buffer = this.buffer;
      src.connect(g);
      src.start(t, offset);
      src.stop(t + segLen + 0.05);

      const seg: Segment = { src, gain: g };
      src.onended = () => {
        src.disconnect();
        g.disconnect();

        const i = this.segments.indexOf(seg);
        if (i >= 0) this.segments.splice(i, 1);
      };

      this.segments.push(seg);
      this.nextStart = t + segLen - this.xfade;
    }
  }
  private stopAll(now: number): void {
    for (const seg of this.segments)
      try {
        seg.src.stop(now + 0.05);
      } catch {}

    this.running = false;
  }
}

export class AudioEngine {
  public readonly ctx: AudioContext;

  public readonly bedBus: GainNode;
  public readonly fxBus: GainNode;

  private readonly sfxIn: GainNode;
  private readonly uwFilter: BiquadFilterNode;
  private readonly sfxOut: GainNode;

  public readonly uwBus: GainNode;

  private readonly master: GainNode;
  private readonly compressor: DynamicsCompressorNode;

  private buffers = new Map<SoundId, AudioBuffer>();
  private loops = new Set<CrossfadeLoop>();
  private oneShots = 0;
  private lastVariant = new Map<string, number>();

  private muted = true;
  private masterVol = 0.8;

  private tmpPos = new THREE.Vector3();
  private tmpFwd = new THREE.Vector3();
  private tmpUp = new THREE.Vector3();
  private tmpQuat = new THREE.Quaternion();
  constructor() {
    const Ctor: typeof AudioContext =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    this.ctx = new Ctor({ latencyHint: "interactive" });

    const ctx = this.ctx;

    this.master = ctx.createGain();
    this.master.gain.value = 0;

    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -14;
    this.compressor.knee.value = 12;
    this.compressor.ratio.value = 4;
    this.compressor.attack.value = 0.005;
    this.compressor.release.value = 0.2;

    this.sfxIn = ctx.createGain();
    this.uwFilter = ctx.createBiquadFilter();
    this.uwFilter.type = "lowpass";
    this.uwFilter.frequency.value = 20000;
    this.uwFilter.Q.value = 0.7;
    this.sfxOut = ctx.createGain();

    this.bedBus = ctx.createGain();
    this.fxBus = ctx.createGain();
    this.uwBus = ctx.createGain();

    this.bedBus.connect(this.sfxIn);
    this.fxBus.connect(this.sfxIn);
    this.sfxIn.connect(this.uwFilter);
    this.uwFilter.connect(this.sfxOut);
    this.sfxOut.connect(this.master);
    this.uwBus.connect(this.master);
    this.master.connect(this.compressor);
    this.compressor.connect(ctx.destination);
  }
  public armAutoResume(target: Window = window): void {
    const events = ["pointerdown", "keydown", "touchend"] as const;
    const unlock = (): void => {
      void this.ctx.resume().then(() => {
        if (this.ctx.state === "running")
          for (const e of events) target.removeEventListener(e, unlock);
      });
    };

    for (const e of events)
      target.addEventListener(e, unlock, { passive: true });
  }
  public async load(): Promise<void> {
    const entries = Object.entries(SOUND_FILES) as [SoundId, string][];
    const results = await Promise.allSettled(
      entries.map(async ([id, file]) => {
        const res = await fetch(AUDIO_BASE + file);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

        const data = await res.arrayBuffer();
        const buf = await this.ctx.decodeAudioData(data);
        this.buffers.set(id, buf);
      }),
    );

    results.forEach((r, i) => {
      if (r.status === "rejected")
        console.warn(`[audio] failed to load ${entries[i][1]}:`, r.reason);
    });
  }
  public has(id: SoundId): boolean {
    return this.buffers.has(id);
  }
  public get canPlay(): boolean {
    return this.ctx.state === "running" && !this.muted;
  }
  public setMuted(muted: boolean): void {
    this.muted = muted;
    this.master.gain.setTargetAtTime(
      muted ? 0 : this.masterVol,
      this.ctx.currentTime,
      0.08,
    );
  }
  public setVolumes(master: number, ambience: number, effects: number): void {
    this.masterVol = master;
    if (!this.muted)
      this.master.gain.setTargetAtTime(master, this.ctx.currentTime, 0.05);

    this.bedBus.gain.setTargetAtTime(ambience, this.ctx.currentTime, 0.05);
    this.fxBus.gain.setTargetAtTime(effects, this.ctx.currentTime, 0.05);
    this.uwBus.gain.setTargetAtTime(ambience, this.ctx.currentTime, 0.05);
  }
  public setUnderwater(amount: number, cutoffHz: number): void {
    const now = this.ctx.currentTime;
    const freq = 20000 * Math.pow(cutoffHz / 20000, clampToOne(amount));
    this.uwFilter.frequency.setTargetAtTime(freq, now, 0.06);
    this.sfxOut.gain.setTargetAtTime(1 - 0.3 * amount, now, 0.06);
  }
  public setListener(camera: THREE.Camera): void {
    camera.getWorldPosition(this.tmpPos);
    camera.getWorldDirection(this.tmpFwd);
    this.tmpUp
      .set(0, 1, 0)
      .applyQuaternion(camera.getWorldQuaternion(this.tmpQuat));

    const l = this.ctx.listener;
    const t = this.ctx.currentTime;

    l.positionX.setValueAtTime(this.tmpPos.x, t);
    l.positionY.setValueAtTime(this.tmpPos.y, t);
    l.positionZ.setValueAtTime(this.tmpPos.z, t);

    l.forwardX.setValueAtTime(this.tmpFwd.x, t);
    l.forwardY.setValueAtTime(this.tmpFwd.y, t);
    l.forwardZ.setValueAtTime(this.tmpFwd.z, t);

    l.upX.setValueAtTime(this.tmpUp.x, t);
    l.upY.setValueAtTime(this.tmpUp.y, t);
    l.upZ.setValueAtTime(this.tmpUp.z, t);
  }
  public update(dt: number): void {
    for (const loop of this.loops) loop.update(dt);
  }
  public createLoop(
    id: SoundId,
    dest: AudioNode,
    crossfadeSeconds = 1.0,
  ): CrossfadeLoop | null {
    const buf = this.buffers.get(id);
    if (!buf) return null;

    const loop = new CrossfadeLoop(this.ctx, buf, dest, crossfadeSeconds);
    this.loops.add(loop);

    return loop;
  }
  public disposeLoop(loop: CrossfadeLoop | null): void {
    if (!loop) return;

    this.loops.delete(loop);
    loop.dispose();
  }
  public createPositionalBus(): PositionalBus {
    const input = this.ctx.createGain();
    const panner = this.ctx.createPanner();

    panner.panningModel = "equalpower";
    panner.distanceModel = "inverse";
    panner.refDistance = REF_DISTANCE;
    panner.rolloffFactor = ROLLOFF;
    panner.maxDistance = 10000;

    input.connect(panner);
    panner.connect(this.fxBus);

    return {
      input,
      setPosition: (x, y, z) => setPannerPosition(panner, x, y, z),
      dispose: () => {
        input.disconnect();
        panner.disconnect();
      },
    };
  }
  public pickVariant(group: string, ids: SoundId[]): SoundId {
    const last = this.lastVariant.get(group) ?? -1;

    let i = Math.floor(Math.random() * ids.length);
    if (ids.length > 1 && i === last) i = (i + 1) % ids.length;

    this.lastVariant.set(group, i);
    return ids[i];
  }
  public play(id: SoundId, o: OneShotOptions = {}): void {
    if (!this.canPlay) return;

    const buf = this.buffers.get(id);
    if (!buf) return;

    const gain = o.gain ?? 1;
    if (gain < 0.004 || this.oneShots >= MAX_ONE_SHOTS) return;

    const ctx = this.ctx;
    const nodes: AudioNode[] = [];

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = o.rate ?? 1;

    let tail: AudioNode = src;
    const chain = (n: AudioNode): void => {
      tail.connect(n);
      nodes.push(n);
      tail = n;
    };

    if (o.highpassHz) {
      const hp = ctx.createBiquadFilter();
      hp.type = "highpass";
      hp.frequency.value = o.highpassHz;

      chain(hp);
    }

    if (o.lowpassHz) {
      const lp = ctx.createBiquadFilter();
      lp.type = "lowpass";
      lp.frequency.value = Math.min(o.lowpassHz, ctx.sampleRate * 0.45);

      chain(lp);
    }

    const g = ctx.createGain();
    g.gain.value = gain;
    chain(g);

    if (o.position) {
      const p = ctx.createPanner();
      p.panningModel = "equalpower";
      p.distanceModel = "inverse";
      p.refDistance = REF_DISTANCE;
      p.rolloffFactor = ROLLOFF;
      p.maxDistance = 10000;

      setPannerPosition(p, o.position.x, o.position.y, o.position.z);
      chain(p);
    } else if (o.pan) {
      const sp = ctx.createStereoPanner();
      sp.pan.value = Math.max(-1, Math.min(1, o.pan));

      chain(sp);
    }

    tail.connect(this.fxBus);

    this.oneShots++;
    src.onended = () => {
      this.oneShots--;
      src.disconnect();

      for (const n of nodes) n.disconnect();
    };

    src.start(ctx.currentTime + (o.delay ?? 0));
  }
}
