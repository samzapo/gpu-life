export const SYNTH_RATE = 22050;
export const SYNTH_BPM = 120;

export const DRUM_BASE_NOTES = {
  kick: 36,
  hat: 60,
  snare: 60,
  rim: 45,
  crystal: 48,
  anvil: 36,
} as const;

const DRUM_SPECS: ReadonlyArray<{ id: 0 | 1 | 2 | 3 | 14 | 15; baseNote: number; dur: number }> = [
  { id: 0, baseNote: DRUM_BASE_NOTES.kick, dur: 0.42 },
  { id: 1, baseNote: DRUM_BASE_NOTES.hat, dur: 0.075 },
  { id: 2, baseNote: DRUM_BASE_NOTES.snare, dur: 0.24 },
  { id: 3, baseNote: DRUM_BASE_NOTES.rim, dur: 0.18 },
  { id: 14, baseNote: DRUM_BASE_NOTES.crystal, dur: 0.72 },
  { id: 15, baseNote: DRUM_BASE_NOTES.anvil, dur: 1.15 },
];

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return (s / 4294967296) * 2 - 1;
  };
}

/**
 * Deterministically renders the percussion and mineral-hit bank
 * (kick, dark hi-hat, gated retro-industrial snare, deep industrial tom, warm crystal pulse, sulfide anvil).
 * Pitched bass, arpeggio, lead, and pad voices are synthesized live via Web Audio nodes.
 */
export function buildCyberDrumBank(ctx: AudioContext): Map<number, AudioBuffer> {
  const bank = new Map<number, AudioBuffer>();
  const rand = makePrng(2026120);
  const TAU = Math.PI * 2;
  const sampleRate = Math.max(22050, Math.min(96000, Math.round(ctx.sampleRate || 44100)));

  for (const spec of DRUM_SPECS) {
    const { id, baseNote, dur } = spec;
    const count = Math.ceil(dur * sampleRate);
    const buffer = ctx.createBuffer(1, count, sampleRate);
    const out = buffer.getChannelData(0);
    const baseFreq = 440 * Math.pow(2, (baseNote - 69) / 12);

    let filtered = 0;
    let prevNoise = 0;

    for (let i = 0; i < count; i++) {
      const t = i / sampleRate;
      const p = TAU * baseFreq * t;
      const noise = rand();
      let value = 0;
      let attack = Math.min(1, t / 0.004);
      const release = Math.min(1, (dur - t) / 0.028);

      switch (id) {
        case 0: {
          // 4-on-the-floor kick: tight sub sweep + saturated punch + warm transient
          const sweepPhase = TAU * (42 * t + 86 * 0.022 * (1 - Math.exp(-t / 0.022)));
          const body = Math.tanh(Math.sin(sweepPhase) * 1.55) * Math.exp(-t * 7.8);
          const click = (noise - prevNoise * 0.25) * Math.exp(-t * 170) * 0.14;
          value = body * 0.94 + click;
          attack = Math.min(1, t / 0.0012);
          break;
        }
        case 1: {
          // Crisp dark 16th hi-hat (filtered so highs never turn brittle)
          const hp = noise - prevNoise * 0.78;
          const darkMetal = Math.sin(p * 3.2 + 0.9 * Math.sin(p * 4.7));
          value = (hp * 0.60 + darkMetal * 0.14) * Math.exp(-t * 58) * 0.36;
          attack = Math.min(1, t / 0.001);
          break;
        }
        case 2: {
          // Gated retro-industrial snare / clap over dark 168 Hz punch
          let burst = Math.exp(-t * 48);
          if (t >= 0.009) burst += Math.exp(-(t - 0.009) * 52) * 0.65;
          if (t >= 0.020) burst += Math.exp(-(t - 0.020) * 44) * 0.5;
          const body = Math.sin(TAU * (168 - 44 * (1 - Math.exp(-t * 36))) * t) * Math.exp(-t * 22);
          const gateEnv = t < 0.14 ? 1.0 : Math.exp(-(t - 0.14) * 32);
          value = ((noise - prevNoise * 0.35) * burst * 0.52 + body * 0.44) * gateEnv;
          attack = Math.min(1, t / 0.0012);
          break;
        }
        case 3: {
          // Deep dark industrial tom / tectonic thud (replaces high metallic rim tink)
          const pitchEnv = Math.exp(-t * 28);
          const sweep = TAU * (baseFreq * (0.65 + 0.55 * pitchEnv)) * t;
          value = Math.tanh((Math.sin(sweep) + 0.35 * Math.sin(sweep * 0.5)) * 1.25) * Math.exp(-t * 18) * 0.68;
          attack = Math.min(1, t / 0.0015);
          break;
        }
        case 14: {
          // Warm dark crystal fifth pulse (sub-warm analog glass without high inharmonic tinks)
          value =
            (0.54 * Math.sin(p) +
              0.34 * Math.sin(p * 0.5) +
              0.16 * Math.sin(p * 1.498) * Math.exp(-t * 7.5)) *
            Math.exp(-t * 4.2) *
            0.55;
          attack = Math.min(1, t / 0.008);
          break;
        }
        case 15:
        default: {
          // Deep sulfide / tectonic sub anvil thump
          const body =
            0.58 * Math.sin(p) +
            0.34 * Math.sin(p * 0.5) +
            0.20 * Math.sin(p * 1.498) * Math.exp(-t * 4.0);
          value = Math.tanh(body * 1.25) * Math.exp(-t * 2.5) * 0.72;
          break;
        }
      }

      const smoothAlpha = id === 0 || id === 2 ? 0.75 : id === 1 ? 0.55 : 0.48;
      filtered += (value - filtered) * smoothAlpha;
      out[i] = Math.max(-1, Math.min(1, filtered * attack * Math.max(0, Math.min(1, release))));
      prevNoise = noise;
    }
    bank.set(id, buffer);
  }

  return bank;
}
