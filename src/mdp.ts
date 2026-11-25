import { createSeededRng } from './matrix.ts';

export const MDP_NUM_STATES = 4;
export const MDP_NUM_ACTIONS = 4;
export const MDP_SPECIES_STRIDE = MDP_NUM_STATES * MDP_NUM_ACTIONS; // 16 floats per bioSpecies
export const MDP_STATE_BYTES = 32; // vec4u qPack (16B) + vec4f ctrl (16B)

export const MDP_STATE_STARVING = 0;
export const MDP_STATE_THREATENED = 1;
export const MDP_STATE_THRIVING = 2;
export const MDP_STATE_DORMANT = 3;

export const MDP_ACTION_FORAGE = 0;
export const MDP_ACTION_HUNT = 1;
export const MDP_ACTION_EVADE = 2;
export const MDP_ACTION_CONSOLIDATE = 3;

export type MdpStateId = 0 | 1 | 2 | 3;
export type MdpActionId = 0 | 1 | 2 | 3;

export const MDP_ACTION_LABELS: readonly string[] = [
  'Foraging',
  'Hunting',
  'Evading',
  'Consolidating',
] as const;

export const MDP_STATE_LABELS: readonly string[] = [
  'Starving',
  'Threatened',
  'Thriving',
  'Dormant',
] as const;

const Q_MIN = -1.5;
const Q_MAX = 1.5;

export function quantizeMdpQ(q: number): number {
  const norm = (q - Q_MIN) / (Q_MAX - Q_MIN);
  return Math.max(0, Math.min(255, Math.round(norm * 255)));
}

export function dequantizeMdpQ(qByte: number): number {
  return ((qByte & 255) / 255) * (Q_MAX - Q_MIN) + Q_MIN;
}

export function unpackMdpActionFromPad(padVal: number): MdpActionId {
  if (!(padVal >= 2097151.5)) return MDP_ACTION_FORAGE;
  const packed = Math.round(padVal);
  if ((packed & 2097152) === 0) return MDP_ACTION_FORAGE;
  return (((packed >>> 22) & 3) as MdpActionId);
}

export interface DecodedMdpCtrl {
  prevState: MdpStateId;
  action: MdpActionId;
  dwell: number;
  initialized: boolean;
}

export function unpackMdpCtrl(ctrlW: number): DecodedMdpCtrl {
  const bits = Math.round(ctrlW);
  return {
    prevState: (bits & 3) as MdpStateId,
    action: ((bits >>> 2) & 3) as MdpActionId,
    dwell: (bits >>> 4) & 255,
    initialized: ((bits >>> 12) & 1) === 1,
  };
}

/**
 * Generates evolutionary per-species MDP Q-table priors (`nBio * 16` floats) appended
 * to `worldRulesBuffer`. Autotroph-leaning cohesive species favor consolidation/foraging;
 * asymmetric motile species favor hunting/evasion while preserving room for online TD adaptation.
 */
export function buildSpeciesMdpPriors(
  traitsTable: readonly (readonly number[])[],
  nBio: number,
  seed: string | number,
): Float32Array<ArrayBuffer> {
  const n = Math.max(1, nBio);
  const out = new Float32Array(n * MDP_SPECIES_STRIDE);
  const rng = createSeededRng(`${seed}:mdpPriors`);

  const clampPrior = (v: number) => Math.max(-1.2, Math.min(1.2, v));

  for (let sp = 0; sp < n; sp++) {
    const tr = traitsTable[sp] ?? [0.45, 0.25, 0.25, 0.40];
    const coh = Math.max(0, Math.min(1, tr[0] ?? 0.45));
    const chase = Math.max(0, Math.min(1, tr[1] ?? 0.25));
    const flee = Math.max(0, Math.min(1, tr[2] ?? 0.25));
    const asym = Math.max(0, Math.min(1, tr[3] ?? 0.40));
    const predTendency = Math.max(0, Math.min(1, 0.55 * asym + 0.45 * chase - 0.25 * coh));
    const producerTendency = Math.max(0, Math.min(1, 0.65 * coh + 0.20 * (1 - asym)));

    const j = () => (rng() - 0.5) * 0.14;
    const base = sp * MDP_SPECIES_STRIDE;

    // State 0: STARVING -> producers bias FORAGE/CONSOLIDATE; predators bias HUNT/FORAGE
    out[base + 0] = clampPrior(0.42 + 0.24 * producerTendency + j()); // FORAGE
    out[base + 1] = clampPrior(-0.08 + 0.72 * predTendency + j());    // HUNT
    out[base + 2] = clampPrior(-0.28 + 0.12 * flee + j());            // EVADE
    out[base + 3] = clampPrior(0.06 + 0.28 * producerTendency + j()); // CONSOLIDATE

    // State 1: THREATENED -> everyone values EVADE or CONSOLIDATE (armored species shelter more)
    out[base + 4] = clampPrior(-0.18 + j());                           // FORAGE
    out[base + 5] = clampPrior(-0.24 + 0.28 * predTendency + j());     // HUNT
    out[base + 6] = clampPrior(0.46 + 0.32 * flee + 0.16 * asym + j()); // EVADE
    out[base + 7] = clampPrior(0.24 + 0.38 * producerTendency + j());  // CONSOLIDATE

    // State 2: THRIVING -> balanced exploration between FORAGE, HUNT, and shell growth
    out[base + 8] = clampPrior(0.32 + 0.26 * producerTendency + j());  // FORAGE
    out[base + 9] = clampPrior(0.08 + 0.62 * predTendency + j());      // HUNT
    out[base + 10] = clampPrior(-0.34 + j());                          // EVADE
    out[base + 11] = clampPrior(0.16 + 0.22 * producerTendency + j()); // CONSOLIDATE

    // State 3: DORMANT / NIGHT / CROWDED -> strong CONSOLIDATE bias vs exploratory FORAGE/HUNT
    out[base + 12] = clampPrior(-0.02 + 0.16 * predTendency + j());    // FORAGE
    out[base + 13] = clampPrior(-0.16 + 0.34 * predTendency + j());    // HUNT
    out[base + 14] = clampPrior(-0.10 + 0.20 * flee + j());            // EVADE
    out[base + 15] = clampPrior(0.52 + 0.30 * producerTendency + j()); // CONSOLIDATE
  }

  return out;
}
