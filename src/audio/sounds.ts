export const AUDIO_BASE = "/audio/";

export const SOUND_FILES = {
  water: "water.mp3",
  flowSlow: "flow-slow.wav",
  flowMedium: "flow-medium.wav",
  flowFast: "flow-fast.flac",
  foam: "foam.flac",
  lapping: "lapping.mp3",
  bubbles: "bubbles.mp3",
  rumble: "rumble.mp3",
  underwater: "under-water.mp3",
  splashWater1: "splash-water1.mp3",
  splashWater2: "splash-water2.mp3",
  splashWater3: "splash-water3.mp3",
  splashWater4: "splash-water4.mp3",
  splashWater5: "splash-water5.mp3",
  splashWater6: "splash-water6.mp3",
  splashWater7: "splash-water7.mp3",
  splashWater8: "splash-water8.mp3",
  splashSmall: "splash-small.mp3",
  splashMedium: "splash-medium.mp3",
  splashLarge: "splash-large.mp3",
  drip1: "drip1.wav",
  drip2: "drip2.mp3",
  drip3: "drip3.mp3",
  thud1: "thud1.mp3",
  thud2: "thud2.mp3",
  thud3: "thud3.mp3",
  whoosh: "whoosh.mp3",
} as const;

export type SoundId = keyof typeof SOUND_FILES;

export const SPLASH_TIERS: SoundId[] = [
  "splashSmall",
  "splashMedium",
  "splashLarge",
];

export const DRIP_IDS: SoundId[] = ["drip1", "drip2", "drip3"];

export const THUD_IDS: SoundId[] = ["thud1", "thud2", "thud3"];

export const WATER_SPLASH_IDS: SoundId[] = [
  "splashWater1",
  "splashWater2",
  "splashWater3",
  "splashWater4",
  "splashWater5",
  "splashWater6",
  "splashWater7",
  "splashWater8",
];
