export const SYNTH_RATE = 22050;
export const SYNTH_BPM = 120;

export const DRUM_BASE_NOTES = {
  kick: 36,
  hat: 60,
  snare: 60,
  rim: 60,
  crystal: 60,
  anvil: 36,
} as const;

const DRUM_SPECS: ReadonlyArray<{ id: 0 | 1 | 2 | 3 | 14 | 15; baseNote: number; dur: number }> = [
  { id: 0, baseNote: DRUM_BASE_NOTES.kick, dur: 0.42 },
  { id: 1, baseNote: DRUM_BASE_NOTES.hat, dur: 0.085 },
  { id: 2, baseNote: DRUM_BASE_NOTES.snare, dur: 0.24 },
  { id: 3, baseNote: DRUM_BASE_NOTES.rim, dur: 0.16 },
  { id: 14, baseNote: DRUM_BASE_NOTES.crystal, dur: 1.05 },
  { id: 15, baseNote: DRUM_BASE_NOTES.anvil, dur: 1.25 },
];

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return (s / 4294967296) * 2 - 1;
  };
}

/**
 * Deterministically renders the dark cyberpunk percussion and mineral-hit bank
 * (kick, hi-hat, gated retro-industrial snare, metallic rim, crystal bell, sulfide anvil).
 * Pitched bass, arpeggio, lead, and pad voices are synthesized live via Web Audio nodes.
 */
export function buildCyberDrumBank(ctx: AudioContext): Map<number, AudioBuffer> {
  const bank = new Map<number, AudioBuffer>();
  const rand = makePrng(2026120);
  const TAU = Math.PI * 2;

  for (const spec of DRUM_SPECS) {
    const { id, baseNote, dur } = spec;
    const count = Math.ceil(dur * SYNTH_RATE);
    const buffer = ctx.createBuffer(1, count, SYNTH_RATE);
    const out = buffer.getChannelData(0);
    const baseFreq = 440 * Math.pow(2, (baseNote - 69) / 12);

    let filtered = 0;
    let prevNoise = 0;

    for (let i = 0; i < count; i++) {
      const t = i / SYNTH_RATE;
      const p = TAU * baseFreq * t;
      const noise = rand();
      let value = 0;
      let attack = Math.min(1, t / 0.004);
      const release = Math.min(1, (dur - t) / 0.025);

      switch (id) {
        case 0: {
          // 4-on-the-floor cyberpunk kick: tight sub sweep + saturated punch + crisp click
          const sweepPhase = TAU * (42 * t + 86 * 0.022 * (1 - Math.exp(-t / 0.022)));
          const body = Math.tanh(Math.sin(sweepPhase) * 1.55) * Math.exp(-t * 7.8);
          const click = (noise - prevNoise * 0.4) * Math.exp(-t * 210) * 0.24;
          value = body * 0.92 + click;
          attack = Math.min(1, t / 0.001);
          break;
        }
        case 1: {
          // Crisp dark 16th hi-hat: high-passed FM metallic noise tick
          const hp = noise - prevNoise * 0.88;
          const metal = Math.sin(p * 7.1 + 1.5 * Math.sin(p * 11.3));
          value = (hp * 0.68 + metal * 0.24) * Math.exp(-t * 54) * 0.52;
          attack = Math.min(1, t / 0.0008);
          break;
        }
        case 2: {
          // Gated retro-industrial snare / clap over dark 168 Hz punch
          let burst = Math.exp(-t * 48);
          if (t >= 0.009) burst += Math.exp(-(t - 0.009) * 52) * 0.65;
          if (t >= 0.020) burst += Math.exp(-(t - 0.020) * 44) * 0.5;
          const body = Math.sin(TAU * (168 - 44 * (1 - Math.exp(-t * 36))) * t) * Math.exp(-t * 22);
          const gateEnv = t < 0.14 ? 1.0 : Math.exp(-(t - 0.14) * 32);
          value = ((noise - prevNoise * 0.35) * burst * 0.56 + body * 0.42) * gateEnv;
          attack = Math.min(1, t / 0.0012);
          break;
        }
        case 3: {
          // Metallic tectonic rim / cyber tick
          const fm = Math.sin(p * 3.1 + 2.2 * Math.sin(p * 4.8) * Math.exp(-t * 25));
          value = fm * Math.exp(-t * 28) * 0.72;
          attack = Math.min(1, t / 0.001);
          break;
        }
        case 14: {
          // Crystal mineral bell chime
          value =
            (0.56 * Math.sin(p) +
              0.26 * Math.sin(p * 2.41) * Math.exp(-t * 3.5) +
              0.14 * Math.sin(p * 4.18) * Math.exp(-t * 8.0)) *
            Math.exp(-t * 2.9) *
            0.62;
          break;
        }
        case 15:
        default: {
          // Deep sulfide / tectonic sub anvil strike
          const body =
            0.55 * Math.sin(p) +
            0.30 * Math.sin(p * 1.47) * Math.exp(-t * 2.8) +
            0.20 * Math.sin(p * 2.13) * Math.exp(-t * 5.2);
          value = Math.tanh(body * 1.25) * Math.exp(-t * 2.2) * 0.74;
          break;
        }
      }

      const smoothAlpha = id <= 3 ? 0.80 : 0.62;
      filtered += (value - filtered) * smoothAlpha;
      out[i] = Math.max(-1, Math.min(1, filtered * attack * Math.max(0, Math.min(1, release))));
      prevNoise = noise;
    }
    bank.set(id, buffer);
  }

  return bank;
}
