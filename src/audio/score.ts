import { SYNTH_BPM, SYNTH_RATE } from './synthBank.ts';
import type { ScreenAudioObservation } from './observer.ts';

export type ScheduledAudioEvent =
  | {
      kind: 'drum';
      offset: number;
      instrument: 0 | 1 | 2 | 3 | 14 | 15;
      lane: number;
      note: number;
      gain: number;
    }
  | {
      kind: 'synth';
      offset: number;
      voiceType: 'bass' | 'arp' | 'lead' | 'pad';
      lane: number;
      note: number;
      gain: number;
      durationSec: number;
      baseCutoffHz: number;
      envPeakHz: number;
      filterQ: number;
      subMix: number;
      detuneCents: number;
    };

const PHRYGIAN_DEGREES = [0, 1, 3, 5, 7, 8, 10, 12] as const;

// Sequenced cyberpunk / synthwave bassline contours (octave jumps, b7, 5th, m3)
const SEQ_BASS_CONTOURS: readonly (readonly number[])[] = [
  [0, 12, 0, 10, 0, 7, 0, 3, 0, 12, 0, 10, 7, 12, 10, 3],
  [0, 0, 12, 0, 10, 0, 7, 10, 0, 0, 12, 3, 7, 10, 12, 7],
  [0, 12, 7, 0, 12, 10, 7, 3, 0, 12, 7, 0, 1, 13, 10, 7],
  [0, 0, 7, 12, 0, 10, 0, 3, 0, 0, 7, 10, 12, 7, 3, 0],
];

// Harmonic roots: i (0) -> IV (+5 "up to the 4th" transpose) -> bVI (+8) -> bVII (+10) / bII (+1)
const PROG_CYBER = [0, 5, 8, 10] as const;
const PROG_PHRYGIAN = [0, 5, 1, 8] as const;

const MOTIF_BEATS = new Set([0, 3, 6, 10, 12, 15]);
const MOTIF_ORDER = [0, 3, 6, 10, 12, 15] as const;

export function lineageMotif(seedHash: number, speciesId: number): number[] {
  let s = ((seedHash * 65537 + speciesId * 104729) >>> 0) || 1337;
  const nextRand = () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s;
  };
  const scale = [0, 1, 3, 5, 7, 8, 10, 12];
  const motif = [0, 7, 10, 3, 1, 0];
  for (let i = 1; i < 5; i++) {
    motif[i] = scale[nextRand() % scale.length];
  }
  return motif;
}

export function cyberSection(bar: number): 'emerge' | 'drive' | 'surge' | 'void' | 'build' | 'climax' {
  const pos = ((bar % 64) + 64) % 64;
  if (pos < 8) return 'emerge';
  if (pos < 24) return 'drive';
  if (pos < 32) return 'surge';
  if (pos < 40) return 'void';
  if (pos < 48) return 'build';
  return 'climax';
}

export function generateScoreEvents(
  startFrame: number,
  frameCount: number,
  obs: ScreenAudioObservation,
): ScheduledAudioEvent[] {
  const out: ScheduledAudioEvent[] = [];
  if (frameCount <= 0) return out;

  const interval = (SYNTH_RATE * 60) / SYNTH_BPM / 4;
  const firstStep = Math.max(0, Math.floor(startFrame / interval) - 1);
  const lastStep = Math.ceil((startFrame + frameCount) / interval);

  for (let step = firstStep; step <= lastStep; step++) {
    const swing = step % 2 === 1 ? 0.035 : 0;
    const sample = Math.round((step + swing) * interval);
    if (sample < startFrame || sample >= startFrame + frameCount) continue;
    emitStepEvents(out, step, sample - startFrame, obs);
  }

  out.sort((a, b) => a.offset - b.offset);
  return out;
}

function emitStepEvents(
  out: ScheduledAudioEvent[],
  step: number,
  offset: number,
  obs: ScreenAudioObservation,
): void {
  const bar = Math.floor(step / 16);
  const beat = ((step % 16) + 16) % 16;
  const part = cyberSection(bar);
  const env = obs.env;
  const stepSec = 60 / SYNTH_BPM / 4;

  const darkBias = obs.halogenStress * 0.45 + env.nightFactor * 0.45 + obs.mineralSulfideRatio * 0.35;
  const usePhrygian = darkBias > 0.36;
  const prog = usePhrygian ? PROG_PHRYGIAN : PROG_CYBER;
  const chordIdx = Math.floor(bar / 2) % prog.length;

  const massShift = Math.min(12, Math.max(0, Math.floor(Math.log2(Math.max(1, obs.totalBioMass / 25))) * 3));
  const nightDrop = env.nightFactor > 0.55 ? 12 : 0;
  const transpose = -Math.min(12, Math.max(nightDrop, massShift > 6 ? 12 : 0));

  const bassRoot = 38 + prog[chordIdx] + (transpose <= -12 ? -12 : 0);
  const leadRoot = 50 + prog[chordIdx] + transpose;
  const sparse = part === 'void' || !obs.active;
  const opening = part === 'emerge' && bar % 8 < 4;

  // 1. Driving Cyber Kick
  if (!sparse && beat % 4 === 0) {
    out.push({
      kind: 'drum',
      offset,
      instrument: 0,
      lane: 4,
      note: 36,
      gain: obs.active ? 0.80 + 0.14 * obs.intensity : 0.42,
    });
  } else if (sparse && (beat === 0 || beat === 8)) {
    out.push({
      kind: 'drum',
      offset,
      instrument: 0,
      lane: 4,
      note: 36,
      gain: 0.46,
    });
  }

  // 2. Gated Industrial Synthwave Snare
  if (!sparse && !opening) {
    if (beat === 4 || beat === 12) {
      out.push({
        kind: 'drum',
        offset,
        instrument: 2,
        lane: 4,
        note: 60,
        gain: 0.48 + 0.16 * (1 - env.oceanFraction * 0.4) + 0.14 * obs.consumerRatio,
      });
    } else if (beat === 15 && obs.consumerRatio > 0.30) {
      out.push({
        kind: 'drum',
        offset,
        instrument: 2,
        lane: 4,
        note: 58,
        gain: 0.22 * obs.consumerRatio,
      });
    }
  }

  // 3. Dark Hi-Hats driven by wind speed & chemical reaction activity
  const hatEnergy = Math.min(1, env.windSpeedNorm * 0.65 + obs.chemBondActivity * 0.55);
  if (!sparse) {
    const offbeat = beat % 4 === 2;
    const roll = hatEnergy > 0.42 && beat % 2 === 1 && (step + bar) % 3 !== 0;
    if (offbeat || roll) {
      out.push({
        kind: 'drum',
        offset,
        instrument: 1,
        lane: 4,
        note: offbeat ? 60 : 63,
        gain: (offbeat ? 0.25 : 0.13) * (0.45 + 0.72 * hatEnergy),
      });
    }
  }

  // 4. Tectonic Metallic Rim & Sulfide Sub Anvil
  const ridgeActivity = Math.max(env.ridgeMask, env.elevKmNorm, obs.mineralSulfideRatio);
  if (ridgeActivity > 0.20 && (beat === 3 || beat === 11 || (beat === 9 && ridgeActivity > 0.5))) {
    out.push({
      kind: 'drum',
      offset,
      instrument: beat === 3 ? 15 : 3,
      lane: 4,
      note: beat === 3 ? Math.max(24, bassRoot) : 60 + prog[chordIdx],
      gain: Math.min(0.40, 0.15 + 0.26 * ridgeActivity),
    });
  }

  // 5. Core Cyberpunk Sequenced Saw+Sub Bassline with Per-Note Filter Pluck Envelope
  const grooveIdx =
    obs.consumerRatio > 0.28
      ? 0
      : obs.chemBondActivity > 0.38
        ? 1
        : usePhrygian
          ? 2
          : 3;
  const contour = SEQ_BASS_CONTOURS[grooveIdx];
  const bassPlayStep = (!sparse && !opening) || beat % 2 === 0;
  if (bassPlayStep) {
    const deg = contour[beat];
    const bassNote = Math.max(26, Math.min(60, bassRoot + deg));
    const accent = beat % 4 === 0 || deg >= 10;
    const nightDark = 1 - 0.52 * env.nightFactor;
    const baseCutoff = (170 + 260 * env.tempNorm * nightDark + 180 * obs.producerRatio) * (sparse ? 0.72 : 1.0);
    const envPeakHz =
      (accent ? 2100 : 1280) *
      (0.55 + 0.55 * obs.intensity + 0.45 * obs.chemBondActivity + 0.35 * obs.halogenStress) *
      (0.65 + 0.45 * nightDark);
    const filterQ = 2.6 + 4.2 * Math.min(1, obs.halogenStress * 0.6 + obs.chemBondActivity * 0.5);
    const subMix = Math.min(0.95, 0.55 + 0.25 * env.oceanFraction + 0.20 * env.nightFactor);
    const detuneCents = 9 + 14 * Math.max(obs.complexRatio, obs.chemVaporRatio);

    out.push({
      kind: 'synth',
      offset,
      voiceType: 'bass',
      lane: 4,
      note: bassNote,
      gain: (sparse ? 0.35 : accent ? 0.54 : 0.42) * (0.85 + 0.25 * obs.intensity),
      durationSec: accent ? stepSec * 1.15 : stepSec * 0.85,
      baseCutoffHz: baseCutoff,
      envPeakHz: Math.min(4800, envPeakHz),
      filterQ,
      subMix,
      detuneCents,
    });
  }

  // 6. Dark Vapor Atmosphere Pad on bar downbeats
  if (beat === 0 || (beat === 8 && (obs.chemVaporRatio > 0.35 || env.rhNorm > 0.52))) {
    out.push({
      kind: 'synth',
      offset,
      voiceType: 'pad',
      lane: beat === 0 ? 1 : 2,
      note: leadRoot + (beat === 8 ? 7 : 0),
      gain: 0.14 + 0.14 * Math.max(obs.chemVaporRatio, env.rhNorm, env.oceanFraction * 0.35),
      durationSec: stepSec * 6.5,
      baseCutoffHz: 240 + 380 * (1 - env.nightFactor * 0.5),
      envPeakHz: 620 + 680 * env.tempNorm,
      filterQ: 1.4,
      subMix: 0.35,
      detuneCents: 18,
    });
  }

  // 7. Dominant Bio Lineage Synth Lead
  if (obs.totalBioMass > 1 && MOTIF_BEATS.has(beat) && (!sparse || beat === 0 || beat === 10)) {
    const motif = lineageMotif(obs.seedHash, obs.dominantBioSp);
    const idx = MOTIF_ORDER.indexOf(beat as (typeof MOTIF_ORDER)[number]);
    const degree = motif[idx >= 0 ? idx : 0] ?? 0;
    const isComplex = obs.complexRatio > 0.35;
    out.push({
      kind: 'synth',
      offset,
      voiceType: 'lead',
      lane: obs.dominantBioSp % 4,
      note: Math.max(42, Math.min(80, leadRoot + 12 + degree)),
      gain: (part === 'surge' || part === 'climax' ? 0.22 : 0.16) * Math.min(1.2, 0.5 + obs.avgEnergy * 0.7),
      durationSec: isComplex ? stepSec * 2.4 : stepSec * 1.4,
      baseCutoffHz: 320 + 420 * (1 - env.nightFactor * 0.55),
      envPeakHz: 1400 + 1800 * obs.consumerRatio + 900 * obs.complexRatio,
      filterQ: isComplex ? 3.2 : 2.2,
      subMix: isComplex ? 0.55 : 0.25,
      detuneCents: isComplex ? 20 : 11,
    });
  }

  // 8. Spatial Stereo Lanes across Left-to-Right Screen Regions (Chemistry Arps + Mineral Chimes/Anvils)
  for (let l = 0; l < 4; l++) {
    const laneObs = obs.lanes[l];

    if (laneObs.chemGain > 0.08 && beat % 4 === (l % 4)) {
      const deg = PHRYGIAN_DEGREES[(bar + beat + l * 2) % PHRYGIAN_DEGREES.length];
      out.push({
        kind: 'synth',
        offset,
        voiceType: 'arp',
        lane: l,
        note: Math.max(48, Math.min(86, leadRoot + 24 + deg)),
        gain: 0.11 * laneObs.chemGain * (0.55 + 0.65 * obs.chemBondActivity),
        durationSec: stepSec * 0.78,
        baseCutoffHz: 420,
        envPeakHz: Math.min(4200, 1200 + 2400 * (1 - env.nightFactor * 0.6) * obs.chemBondActivity),
        filterQ: 2.0 + 3.5 * laneObs.acidDrive,
        subMix: 0.15,
        detuneCents: 10,
      });
    }

    if (laneObs.mineralGain > 0.08 && ((beat === 6 && l % 2 === 0) || (beat === 14 && l % 2 === 1))) {
      const isSulfide = laneObs.mineralSulfideFrac > 0.42;
      out.push({
        kind: 'drum',
        offset,
        instrument: isSulfide ? 15 : 14,
        lane: l,
        note: isSulfide ? bassRoot + 12 : leadRoot + 24 + (usePhrygian ? 1 : 7),
        gain: (isSulfide ? 0.21 : 0.15) * laneObs.mineralGain,
      });
    }
  }
}
