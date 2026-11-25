import type * as d3Geo from 'd3-geo';
import {
  buildD3RotationMatrixColMajor,
  type GlobeViewState,
} from '../earth/projections.ts';
import type { Element2D } from '../periodicTable2D.ts';

export interface ScreenLaneObservation {
  pan: number;
  bioGain: number;
  chemGain: number;
  mineralGain: number;
  cutoffHz: number;
  acidDrive: number;
  dominantBioSp: number;
  producerFrac: number;
  consumerFrac: number;
  complexFrac: number;
  mineralSulfideFrac: number;
}

export interface ScreenEnvObservation {
  windSpeedNorm: number;
  oceanFraction: number;
  oceanSpeedNorm: number;
  tempNorm: number;
  nightFactor: number;
  rhNorm: number;
  precipNorm: number;
  elevKmNorm: number;
  ridgeMask: number;
}

export interface ScreenAudioObservation {
  active: boolean;
  seedHash: number;
  intensity: number;
  centerDensity: number;
  totalBioMass: number;
  producerRatio: number;
  consumerRatio: number;
  decomposerRatio: number;
  complexRatio: number;
  chemBondActivity: number;
  chemVaporRatio: number;
  chemLiquidRatio: number;
  chemIceRatio: number;
  halogenStress: number;
  mineralCrystalRatio: number;
  mineralSulfideRatio: number;
  dominantBioSp: number;
  dominantChemElem: number;
  avgEnergy: number;
  env: ScreenEnvObservation;
  lanes: [
    ScreenLaneObservation,
    ScreenLaneObservation,
    ScreenLaneObservation,
    ScreenLaneObservation,
  ];
}

const LANE_PANS = [-0.68, -0.22, 0.22, 0.68] as const;

export function createEmptyAudioObservation(): ScreenAudioObservation {
  const mkLane = (pan: number): ScreenLaneObservation => ({
    pan,
    bioGain: 0,
    chemGain: 0,
    mineralGain: 0,
    cutoffHz: 1400,
    acidDrive: 0.1,
    dominantBioSp: 0,
    producerFrac: 0.5,
    consumerFrac: 0.25,
    complexFrac: 0,
    mineralSulfideFrac: 0,
  });
  return {
    active: false,
    seedHash: 0,
    intensity: 0.25,
    centerDensity: 0,
    totalBioMass: 0,
    producerRatio: 0,
    consumerRatio: 0,
    decomposerRatio: 0,
    complexRatio: 0,
    chemBondActivity: 0,
    chemVaporRatio: 0,
    chemLiquidRatio: 0,
    chemIceRatio: 0,
    halogenStress: 0,
    mineralCrystalRatio: 0,
    mineralSulfideRatio: 0,
    dominantBioSp: 0,
    dominantChemElem: 5,
    avgEnergy: 0.65,
    env: {
      windSpeedNorm: 0.35,
      oceanFraction: 0.5,
      oceanSpeedNorm: 0.2,
      tempNorm: 0.55,
      nightFactor: 0.2,
      rhNorm: 0.5,
      precipNorm: 0.1,
      elevKmNorm: 0.1,
      ridgeMask: 0,
    },
    lanes: [
      mkLane(LANE_PANS[0]),
      mkLane(LANE_PANS[1]),
      mkLane(LANE_PANS[2]),
      mkLane(LANE_PANS[3]),
    ],
  };
}

const bioSpeciesAcc = new Float32Array(256);
const laneBioSpAcc = [
  new Float32Array(64),
  new Float32Array(64),
  new Float32Array(64),
  new Float32Array(64),
];
const laneBioWeight = new Float32Array(4);
const laneBioProd = new Float32Array(4);
const laneBioCons = new Float32Array(4);
const laneBioComp = new Float32Array(4);
const laneChemWeight = new Float32Array(4);
const laneChemAcid = new Float32Array(4);
const laneMinWeight = new Float32Array(4);
const laneMinSulf = new Float32Array(4);

/**
 * Observes life, chemistry, and mineral state near screen center without mutating world state.
 */
export function observeScreenAudioState(
  particlesF32: Float32Array,
  lifeStateF32: Float32Array,
  count: number,
  elements: Element2D[],
  view: GlobeViewState,
  proj: d3Geo.GeoProjection | null,
  env: ScreenEnvObservation,
  seedHash: number,
  paused: boolean,
): ScreenAudioObservation {
  const obs = createEmptyAudioObservation();
  obs.active = !paused;
  obs.seedHash = seedHash;
  obs.env = env;

  const n = Math.min(
    count,
    Math.floor(particlesF32.length / 12),
    Math.floor(lifeStateF32.length / 4),
  );
  if (n <= 0 || view.width <= 0 || view.height <= 0) return obs;

  const isOrtho = view.projectionId === 'orthographic';
  const rot = buildD3RotationMatrixColMajor(view.rotate);
  const r00 = rot[0], r01 = rot[1], r02 = rot[2];
  const r10 = rot[4], r11 = rot[5], r12 = rot[6];
  const r20 = rot[8], r21 = rot[9], r22 = rot[10];

  const cx = view.width * 0.5 + view.pan[0];
  const cy = view.height * 0.5 + view.pan[1];
  const halfW = Math.max(view.width * 0.5, 1);
  const halfH = Math.max(view.height * 0.5, 1);
  const scalePx = Math.max(view.scale, 1);
  const rad2Deg = 180 / Math.PI;

  bioSpeciesAcc.fill(0);
  laneBioWeight.fill(0);
  laneBioProd.fill(0);
  laneBioCons.fill(0);
  laneBioComp.fill(0);
  laneChemWeight.fill(0);
  laneChemAcid.fill(0);
  laneMinWeight.fill(0);
  laneMinSulf.fill(0);
  for (let l = 0; l < 4; l++) laneBioSpAcc[l].fill(0);

  let totalW = 0;
  let bioW = 0;
  let bioProdW = 0;
  let bioConsW = 0;
  let bioDecompW = 0;
  let bioComplexW = 0;
  let bioEnergySum = 0;
  let chemW = 0;
  let chemActW = 0;
  let chemVaporW = 0;
  let chemLiqW = 0;
  let chemIceW = 0;
  let halogenW = 0;
  let minW = 0;
  let minCrystalW = 0;
  let minSulfideW = 0;
  let chemElemCarbonW = 0;
  let domChemElem = 5;
  let bestChemScore = -1;
  const lifeStride =
    lifeStateF32.length >= count * 16
      ? 16
      : lifeStateF32.length >= count * 12
        ? 12
        : 4;

  const stride = n > 4000 ? 2 : 1;
  for (let i = 0; i < n; i += stride) {
    const pBase = i * 12;
    const px = particlesF32[pBase];
    const py = particlesF32[pBase + 1];
    const pz = particlesF32[pBase + 2];

    let sx = 0;
    let sy = 0;
    if (isOrtho) {
      const vx = r00 * px + r10 * py + r20 * pz;
      if (vx < 0.04) continue;
      const vy = r01 * px + r11 * py + r21 * pz;
      const vz = r02 * px + r12 * py + r22 * pz;
      sx = cx + vy * scalePx;
      sy = cy - vz * scalePx;
    } else if (proj) {
      const lon = Math.atan2(py, px) * rad2Deg;
      const lat = Math.asin(Math.max(-1, Math.min(1, pz))) * rad2Deg;
      const pt = proj([lon, lat]);
      if (!pt || !Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) continue;
      sx = pt[0];
      sy = pt[1];
    } else {
      continue;
    }

    const dxNorm = (sx - view.width * 0.5) / halfW;
    const dyNorm = (sy - view.height * 0.5) / halfH;
    const dist = Math.hypot(dxNorm, dyNorm);
    if (dist > 1.05) continue;

    // Emphasize viewport center smoothly (matching primordia's proximity windowing)
    const w = Math.pow(Math.max(0, 1 - dist / 1.05), 1.55) * stride;
    if (w < 0.015) continue;

    const laneIdx = Math.max(0, Math.min(3, Math.floor((dxNorm * 0.5 + 0.5) * 4)));
    totalW += w;

    const lBase = i * lifeStride;
    const ls0 = lifeStateF32[lBase];
    const ls1 = lifeStateF32[lBase + 1];
    const ls2 = lifeStateF32[lBase + 2];
    const ls3 = lifeStateF32[lBase + 3];

    if (ls3 < 0.5) {
      // Chemistry (phase 0)
      chemW += w;
      laneChemWeight[laneIdx] += w;
      const actGlowAbs = Math.min(1, Math.abs(particlesF32[pBase + 11] || 0));
      chemActW += w * (0.25 + 0.75 * actGlowAbs);

      if (ls1 >= 1.5) chemIceW += w;
      else if (ls1 >= 0.5) chemLiqW += w;
      else chemVaporW += w;

      const elemIdx = Math.max(0, Math.min(elements.length - 1, Math.floor(particlesF32[pBase + 3] || 0)));
      const el = elements[elemIdx];
      const anum = el?.atomicNumber ?? 5;
      if (anum === 7 || anum === 13) {
        halogenW += w;
        laneChemAcid[laneIdx] += w;
      } else if (anum === 5) {
        chemElemCarbonW += w;
      }
      if (chemElemCarbonW > bestChemScore) {
        bestChemScore = chemElemCarbonW;
        domChemElem = anum;
      }
      continue;
    }

    const packed = Math.round(ls3);
    const phase = packed & 3;
    if (phase === 3) {
      // Mineral crystal (phase 3)
      const mType = (packed >>> 2) & 255;
      const integ = Math.max(0.1, Math.min(1.2, ls0));
      const mw = w * integ;
      minW += mw;
      laneMinWeight[laneIdx] += mw;
      if (mType === 3 || mType === 4) {
        minSulfideW += mw;
        laneMinSulf[laneIdx] += mw;
        if (ls2 > 0.35) laneChemAcid[laneIdx] += mw * 0.5;
      } else {
        minCrystalW += mw;
      }
      continue;
    }

    // Living biomass or lipid membrane (phase 1 | 2)
    const bioSp = (packed >>> 2) & 255;
    const grad = ((packed >>> 23) & 1) === 1;
    const bw = w * (grad ? 1.4 : phase === 2 ? 1.15 : 1.0);
    bioW += bw;
    bioEnergySum += w * Math.max(0, Math.min(1.5, ls0));
    laneBioWeight[laneIdx] += bw;
    if (bioSp < bioSpeciesAcc.length) bioSpeciesAcc[bioSp] += bw;
    laneBioSpAcc[laneIdx][bioSp % 64] += bw;

    if (grad || phase === 2) {
      bioComplexW += bw * (grad ? 1.0 : 0.45);
      laneBioComp[laneIdx] += bw;
    }

    const metaBias = particlesF32[pBase + 11] || 0;
    if (metaBias >= 2097151.5) {
      const packedNiche = Math.round(metaBias);
      if (((packedNiche >>> 20) & 1) === 1) {
        bioProdW += bw;
        laneBioProd[laneIdx] += bw;
      } else {
        bioConsW += bw;
        laneBioCons[laneIdx] += bw;
      }
    } else if (metaBias < -0.08) {
      bioProdW += bw;
      laneBioProd[laneIdx] += bw;
    } else if (metaBias > 0.14) {
      bioConsW += bw;
      laneBioCons[laneIdx] += bw;
    }
    if (ls2 > 0.18 || ls0 < 0.35) {
      bioDecompW += bw;
    }
  }

  // Dominant bio species across screen center
  let bestSp = 0;
  let bestSpW = 0;
  for (let s = 0; s < bioSpeciesAcc.length; s++) {
    if (bioSpeciesAcc[s] > bestSpW) {
      bestSpW = bioSpeciesAcc[s];
      bestSp = s;
    }
  }

  const bioDenom = Math.max(bioW, 1e-4);
  const chemDenom = Math.max(chemW, 1e-4);
  const minDenom = Math.max(minW, 1e-4);

  obs.centerDensity = Math.min(1, totalW / 120);
  obs.totalBioMass = bioW;
  obs.producerRatio = bioW > 0 ? Math.min(1, bioProdW / bioDenom) : 0;
  obs.consumerRatio = bioW > 0 ? Math.min(1, bioConsW / bioDenom) : 0;
  obs.decomposerRatio = bioW > 0 ? Math.min(1, bioDecompW / bioDenom) : 0;
  obs.complexRatio = bioW > 0 ? Math.min(1, bioComplexW / bioDenom) : 0;
  obs.avgEnergy = bioW > 0 ? Math.min(1, bioEnergySum / Math.max(bioW, 1)) : 0.5;
  obs.chemBondActivity = chemW > 0 ? Math.min(1, chemActW / chemDenom) : 0;
  obs.chemVaporRatio = chemW > 0 ? Math.min(1, chemVaporW / chemDenom) : 0;
  obs.chemLiquidRatio = chemW > 0 ? Math.min(1, chemLiqW / chemDenom) : 0;
  obs.chemIceRatio = chemW > 0 ? Math.min(1, chemIceW / chemDenom) : 0;
  obs.halogenStress = chemW > 0 ? Math.min(1, halogenW / (chemDenom * 0.35)) : 0;
  obs.mineralCrystalRatio = minW > 0 ? Math.min(1, minCrystalW / minDenom) : 0;
  obs.mineralSulfideRatio = minW > 0 ? Math.min(1, minSulfideW / minDenom) : 0;
  obs.dominantBioSp = bestSp;
  obs.dominantChemElem = domChemElem;

  // Overall synth drive/intensity climbs with predation, complex blooms, wind, ridges & acid chemistry
  obs.intensity = Math.max(
    0.18,
    Math.min(
      1,
      0.24 +
        0.28 * obs.complexRatio +
        0.25 * obs.consumerRatio +
        0.22 * obs.chemBondActivity +
        0.16 * env.windSpeedNorm +
        0.16 * env.ridgeMask -
        0.10 * env.nightFactor * 0.4,
    ),
  );

  // Deeper nocturnal filter cutoff curve: colder temperatures & night side plunge cutoff deeper
  const baseCutoff =
    320 +
    1900 * Math.pow(Math.max(0.05, env.tempNorm * (1 - 0.52 * env.nightFactor)), 1.35) +
    1500 * obs.chemBondActivity +
    950 * (1 - env.nightFactor) * env.windSpeedNorm;

  for (let l = 0; l < 4; l++) {
    const lane = obs.lanes[l];
    const lwBio = laneBioWeight[l];
    const lwChem = laneChemWeight[l];
    const lwMin = laneMinWeight[l];

    lane.bioGain = Math.min(1, lwBio / 45);
    lane.chemGain = Math.min(1, lwChem / 65);
    lane.mineralGain = Math.min(1, lwMin / 35);
    lane.acidDrive = Math.min(
      1,
      0.14 + 0.55 * (lwChem > 0 ? laneChemAcid[l] / Math.max(lwChem * 0.25, 1) : obs.halogenStress) +
        0.28 * obs.consumerRatio +
        0.22 * env.ridgeMask,
    );
    lane.cutoffHz = Math.max(
      280,
      Math.min(5400, baseCutoff * (0.78 + 0.44 * lane.bioGain + 0.25 * obs.chemVaporRatio)),
    );
    lane.producerFrac = lwBio > 0 ? Math.min(1, laneBioProd[l] / lwBio) : obs.producerRatio;
    lane.consumerFrac = lwBio > 0 ? Math.min(1, laneBioCons[l] / lwBio) : obs.consumerRatio;
    lane.complexFrac = lwBio > 0 ? Math.min(1, laneBioComp[l] / lwBio) : obs.complexRatio;
    lane.mineralSulfideFrac = lwMin > 0 ? Math.min(1, laneMinSulf[l] / lwMin) : obs.mineralSulfideRatio;

    let laneSp = bestSp;
    let laneSpMax = 0;
    const spArr = laneBioSpAcc[l];
    for (let s = 0; s < spArr.length; s++) {
      if (spArr[s] > laneSpMax) {
        laneSpMax = spArr[s];
        laneSp = s;
      }
    }
    lane.dominantBioSp = laneSp;
  }

  return obs;
}
