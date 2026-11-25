import { createSeededRng } from './matrix.ts';

export const MDP_NUM_STATES = 4;
export const MDP_NUM_ACTIONS = 4;
export const MDP_SPECIES_STRIDE = MDP_NUM_STATES * MDP_NUM_ACTIONS; // 16 floats per bioSpecies
export const LIFE_MDP_STRIDE = 64; // bytes: life(16) + qPack(16) + ctrl(16) + blob(16)
export const MDP_STATE_BYTES = 48; // qPack(16) + ctrl(16) + blob(16)
export const SPECIES_MDP_SLOT_WORDS = 20; // 16 i32 Q_fixed (x10000) + 4 u32 bestAct = 80B
export const BLOB_POOL_SLOT_WORDS = 8; // 32B atomic accumulator per leader
export const BLOB_MDP_SLOT_WORDS = 8; // 32B (qPack + ctrl) per leader

export const MDP_STATE_LOW_ENERGY = 0;
export const MDP_STATE_HAZARD = 1;
export const MDP_STATE_ABUNDANT = 2;
export const MDP_STATE_DARK_CROWDED = 3;

export const MDP_ACTION_EXPAND = 0;
export const MDP_ACTION_PROPEL = 1;
export const MDP_ACTION_DISPERSE = 2;
export const MDP_ACTION_COMPACT = 3;

export const MDP_PAD_TAG = 2097152; // 1 << 21

export type MdpStateId = 0 | 1 | 2 | 3;
export type MdpActionId = 0 | 1 | 2 | 3;

export const MDP_ACTION_LABELS: readonly string[] = [
  'Expand / Absorb',
  'Swirl / Propel',
  'Disperse / Repel',
  'Compact / Shell',
] as const;

export const MDP_ACTION_SHORT_LABELS: readonly string[] = [
  'Expand',
  'Propel',
  'Disperse',
  'Compact',
] as const;

export const MDP_STATE_LABELS: readonly string[] = [
  'Low Energy',
  'Hazard',
  'Abundant',
  'Dark / Dense',
] as const;

export interface MdpLeverProfile {
  dSameKin: number;
  dForeignBio: number;
  dChemNutr: number;
  dChemHalogen: number;
  dLipid: number;
  dMineral: number;
  dBeta: number;
  dDrag: number;
  chiralTwist: number;
  agitation: number;
  lipidAlloc: number;
  torpor: number;
}

const ACTION_LEVER_PROFILES: readonly MdpLeverProfile[] = [
  {
    // 0: Expand / Absorb
    dSameKin: 0.18,
    dForeignBio: 0.45,
    dChemNutr: 0.62,
    dChemHalogen: -0.30,
    dLipid: 0.18,
    dMineral: 0.32,
    dBeta: -0.05,
    dDrag: -0.08,
    chiralTwist: 0.22,
    agitation: 0.25,
    lipidAlloc: 0.25,
    torpor: 0.0,
  },
  {
    // 1: Swirl / Propel
    dSameKin: 0.05,
    dForeignBio: 0.70,
    dChemNutr: 0.30,
    dChemHalogen: -0.55,
    dLipid: -0.05,
    dMineral: 0.10,
    dBeta: -0.08,
    dDrag: -0.22,
    chiralTwist: 0.48,
    agitation: 0.42,
    lipidAlloc: 0.12,
    torpor: 0.0,
  },
  {
    // 2: Disperse / Repel
    dSameKin: -0.28,
    dForeignBio: -0.75,
    dChemNutr: 0.15,
    dChemHalogen: -0.85,
    dLipid: -0.22,
    dMineral: -0.35,
    dBeta: 0.12,
    dDrag: -0.28,
    chiralTwist: -0.30,
    agitation: 0.78,
    lipidAlloc: 0.05,
    torpor: 0.0,
  },
  {
    // 3: Compact / Shell
    dSameKin: 0.52,
    dForeignBio: -0.22,
    dChemNutr: 0.28,
    dChemHalogen: -0.45,
    dLipid: 0.62,
    dMineral: 0.38,
    dBeta: -0.04,
    dDrag: 0.42,
    chiralTwist: 0.0,
    agitation: 0.0,
    lipidAlloc: 0.85,
    torpor: 0.45,
  },
] as const;

export function evalActionLeverProfileJs(act: MdpActionId): MdpLeverProfile {
  return ACTION_LEVER_PROFILES[act & 3] ?? ACTION_LEVER_PROFILES[0];
}

export function evalCombinedMdpPolicyJs(
  pAct: MdpActionId,
  bAct: MdpActionId,
  sAct: MdpActionId,
  hasBlob: boolean,
): MdpLeverProfile {
  const lp = evalActionLeverProfileJs(pAct);
  const lb = evalActionLeverProfileJs(bAct);
  const ls = evalActionLeverProfileJs(sAct);
  const wp = hasBlob ? 0.35 : 0.65;
  const wb = hasBlob ? 0.40 : 0.0;
  const ws = hasBlob ? 0.25 : 0.35;
  return {
    dSameKin: wp * lp.dSameKin + wb * lb.dSameKin + ws * ls.dSameKin,
    dForeignBio: wp * lp.dForeignBio + wb * lb.dForeignBio + ws * ls.dForeignBio,
    dChemNutr: wp * lp.dChemNutr + wb * lb.dChemNutr + ws * ls.dChemNutr,
    dChemHalogen: wp * lp.dChemHalogen + wb * lb.dChemHalogen + ws * ls.dChemHalogen,
    dLipid: wp * lp.dLipid + wb * lb.dLipid + ws * ls.dLipid,
    dMineral: wp * lp.dMineral + wb * lb.dMineral + ws * ls.dMineral,
    dBeta: wp * lp.dBeta + wb * lb.dBeta + ws * ls.dBeta,
    dDrag: wp * lp.dDrag + wb * lb.dDrag + ws * ls.dDrag,
    chiralTwist: wp * lp.chiralTwist + wb * lb.chiralTwist + ws * ls.chiralTwist,
    agitation: wp * lp.agitation + wb * lb.agitation + ws * ls.agitation,
    lipidAlloc: wp * lp.lipidAlloc + wb * lb.lipidAlloc + ws * ls.lipidAlloc,
    torpor: wp * lp.torpor + wb * lb.torpor + ws * ls.torpor,
  };
}

const Q_MIN = -1.5;
const Q_MAX = 1.5;

export function quantizeMdpQ(q: number): number {
  const norm = (q - Q_MIN) / (Q_MAX - Q_MIN);
  return Math.max(0, Math.min(255, Math.round(norm * 255)));
}

export function dequantizeMdpQ(qByte: number): number {
  return ((qByte & 255) / 255) * (Q_MAX - Q_MIN) + Q_MIN;
}

export interface DecodedThreeMdp {
  valid: boolean;
  pAct: MdpActionId;
  bAct: MdpActionId;
  sAct: MdpActionId;
  pState: MdpStateId;
  bState: MdpStateId;
  hasBlob: boolean;
  blobSizeBucket: number;
}

export function packThreeMdpPad(
  pAct: MdpActionId,
  bAct: MdpActionId,
  sAct: MdpActionId,
  hasBlob: boolean,
  pState: MdpStateId,
  bState: MdpStateId,
  blobSizeBucket: number = 0,
): number {
  const packed =
    MDP_PAD_TAG |
    (pAct & 3) |
    ((bAct & 3) << 2) |
    ((sAct & 3) << 4) |
    ((pState & 3) << 6) |
    ((bState & 3) << 8) |
    ((hasBlob ? 1 : 0) << 10) |
    ((Math.max(0, Math.min(7, Math.floor(blobSizeBucket))) & 7) << 11);
  return Math.fround(packed);
}

export function unpackThreeMdpFromPad(padVal: number): DecodedThreeMdp {
  if (!(padVal >= 2097151.5)) {
    return {
      valid: false,
      pAct: MDP_ACTION_EXPAND,
      bAct: MDP_ACTION_EXPAND,
      sAct: MDP_ACTION_EXPAND,
      pState: MDP_STATE_ABUNDANT,
      bState: MDP_STATE_ABUNDANT,
      hasBlob: false,
      blobSizeBucket: 0,
    };
  }
  const packed = Math.round(padVal);
  if ((packed & MDP_PAD_TAG) === 0) {
    return {
      valid: false,
      pAct: MDP_ACTION_EXPAND,
      bAct: MDP_ACTION_EXPAND,
      sAct: MDP_ACTION_EXPAND,
      pState: MDP_STATE_ABUNDANT,
      bState: MDP_STATE_ABUNDANT,
      hasBlob: false,
      blobSizeBucket: 0,
    };
  }
  return {
    valid: true,
    pAct: (packed & 3) as MdpActionId,
    bAct: ((packed >>> 2) & 3) as MdpActionId,
    sAct: ((packed >>> 4) & 3) as MdpActionId,
    pState: ((packed >>> 6) & 3) as MdpStateId,
    bState: ((packed >>> 8) & 3) as MdpStateId,
    hasBlob: ((packed >>> 10) & 1) === 1,
    blobSizeBucket: (packed >>> 11) & 7,
  };
}

export function unpackMdpActionFromPad(padVal: number): MdpActionId {
  const dec = unpackThreeMdpFromPad(padVal);
  if (!dec.valid) return MDP_ACTION_EXPAND;
  return dec.hasBlob ? dec.bAct : dec.pAct;
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
 * Generates near-neutral tabula-rasa Q-table priors (`nBio * 16` floats in `[-0.05, +0.05]`)
 * with tiny deterministic tie-breaking micro-noise so cells, blobs, and species genuinely learn online.
 */
export function buildSpeciesMdpPriors(
  nBio: number,
  seed: string | number,
): Float32Array<ArrayBuffer> {
  const n = Math.max(1, nBio);
  const out = new Float32Array(n * MDP_SPECIES_STRIDE);
  const rng = createSeededRng(`${seed}:mdpPriors`);
  for (let i = 0; i < out.length; i++) {
    out[i] = (rng() - 0.5) * 0.1;
  }
  return out;
}

/**
 * Initializes the live GPU `speciesMdpBuffer` (`nBio * 20` i32/u32 words = `80B` per species):
 * 16 fixed-point i32 Q-values scaled by 10000 + 4 cached u32 argmax `bestAct[s]` words.
 */
export function buildInitialSpeciesMdpBuffer(
  nBio: number,
  seed: string | number,
): Int32Array<ArrayBuffer> {
  const n = Math.max(1, nBio);
  const priors = buildSpeciesMdpPriors(n, seed);
  const out = new Int32Array(n * SPECIES_MDP_SLOT_WORDS);
  for (let sp = 0; sp < n; sp++) {
    const pBase = sp * MDP_SPECIES_STRIDE;
    const oBase = sp * SPECIES_MDP_SLOT_WORDS;
    for (let s = 0; s < MDP_NUM_STATES; s++) {
      let bestAct = 0;
      let bestVal = -Infinity;
      for (let a = 0; a < MDP_NUM_ACTIONS; a++) {
        const val = priors[pBase + s * MDP_NUM_ACTIONS + a] ?? 0;
        const qFixed = Math.round(val * 10000);
        out[oBase + s * MDP_NUM_ACTIONS + a] = qFixed;
        if (qFixed > bestVal) {
          bestVal = qFixed;
          bestAct = a;
        }
      }
      out[oBase + MDP_SPECIES_STRIDE + s] = bestAct;
    }
  }
  return out;
}
