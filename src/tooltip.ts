import type * as d3Geo from 'd3-geo';
import { lonLatToUnitVec } from './earth/projections.ts';
import type { LocalUvSample } from './earth/uvData.ts';
import type { LocalHydroSample } from './earth/hydroData.ts';
import { getElementName, getElementSymbol, type Element2D } from './periodicTable2D.ts';
import type { ViewLayerMode } from './options.ts';
import {
  MDP_ACTION_FORAGE,
  MDP_ACTION_LABELS,
  type MdpActionId,
  unpackMdpActionFromPad,
} from './mdp.ts';

export interface HoverSimSnapshot {
  elements: Element2D[];
  particles: number[][];
  valence: number[][];
  lifeState: number[][];
  bioForce: number[][];
  speciesTraits: number[][];
  capturedAt: number;
}

export interface LocalTempsSample {
  airC: number;
  sfcAirC: number;
  sstC: number | null;
  surfaceC: number;
  airLabel: string;
}

export interface HoverInspectConfig {
  isEnabled: () => boolean;
  inspectionPaused: () => boolean;
  getProj: () => d3Geo.GeoProjection | null;
  getViewLayer: () => ViewLayerMode;
  isBioLinked: () => boolean;
  getAirRMax: () => number;
  getOceanRMax: () => number;
  getWorldSize: () => number;
  getBeta: () => number;
  getTemperature: () => number;
  getScale: () => number;
  getOceanMask?: (unitVec: [number, number, number]) => number;
  getOceanSpeedMs?: (unitVec: [number, number, number]) => number;
  getWindSpeedMs?: (unitVec: [number, number, number]) => number;
  getLocalTemps?: (unitVec: [number, number, number]) => LocalTempsSample | null;
  getLocalUv?: (unitVec: [number, number, number]) => LocalUvSample | null;
  getLocalHydro?: (unitVec: [number, number, number]) => LocalHydroSample | null;
  getEnvColorRgb?: (kind: 'air' | 'sst' | 'uv' | 'rh' | 'precip', value: number) => [number, number, number];
  getMineralOpacity?: () => number;
  requestSnapshot: () => Promise<HoverSimSnapshot | null>;
}

export type TooltipTheme =
  | 'menu'
  | 'anabolic'
  | 'catabolic'
  | 'toxic'
  | 'cofactor'
  | 'chem'
  | 'ocean'
  | 'lipid'
  | 'complex'
  | 'mineral'
  | 'mineralWindfall'
  | 'mineralTrap';

export interface TooltipTarget {
  idx: number;
}

export interface TooltipLensState {
  active: boolean;
  x: number;
  y: number;
  radius: number;
  mag: number;
}

export interface TooltipLensGpuConfig {
  kind: 0 | 1;
  idx: number;
  radius: number;
  mag: number;
  color: [number, number, number, number];
  generation: number;
}

const LENS_THEME_COLORS: Record<TooltipTheme, [number, number, number, number]> = {
  menu: [0.57, 0.76, 1.0, 1.0],
  chem: [0.57, 0.76, 1.0, 1.0],
  anabolic: [0.20, 0.91, 0.72, 1.0],
  catabolic: [1.0, 0.57, 0.28, 1.0],
  toxic: [0.89, 0.34, 0.98, 1.0],
  cofactor: [0.35, 0.88, 1.0, 1.0],
  ocean: [0.22, 0.74, 0.97, 1.0],
  lipid: [0.52, 0.92, 0.96, 1.0],
  complex: [0.98, 0.84, 0.36, 1.0],
  mineral: [0.83, 0.71, 0.46, 1.0],
  mineralWindfall: [0.29, 0.91, 0.81, 1.0],
  mineralTrap: [0.96, 0.31, 0.58, 1.0],
};

export interface TooltipSelectedElement {
  symbol: string;
  name: string;
  atomicNumber: number;
  subshellLabel: string;
  roleLabel: string;
  outerElectrons: number;
  outerCapacity: number;
  valence: number;
  color: [number, number, number];
}

export interface TooltipEnvChip {
  label: string;
  dotRgb?: [number, number, number];
}

export interface TooltipContent {
  theme: TooltipTheme;
  badge: string;
  tempTag?: string;
  envChips?: TooltipEnvChip[];
  title: string;
  body: string;
  meta?: string;
  element?: TooltipSelectedElement;
  targetPhase?: 0 | 1 | 2 | 3;
  targetIdx?: number;
  targetX?: number;
  targetY?: number;
}

const SUBSCRIPTS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];

function toSubscript(n: number): string {
  if (n <= 1) return '';
  return String(n)
    .split('')
    .map((d) => SUBSCRIPTS[Number(d)] ?? d)
    .join('');
}

function escapeHtml(s: string): string {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function bioKind(atomicNum: number, valence: number): number {
  if (valence === 0) return 0; // Noble
  if (atomicNum === 7 || atomicNum === 13) return 4; // F, Cl Poison
  if (atomicNum === 3 || atomicNum === 4 || atomicNum === 10) return 3; // Li, Be, Mg Cofactor
  if (atomicNum === 5) return 2; // C Food
  if (atomicNum === 1 || atomicNum === 12) return 1; // H, S Fuel
  return 5; // O, Si, Na
}

function unpackActGlow(packed: number): [number, number] {
  if (packed <= 0.5) return [1.0, 0.0];
  const v = Math.max(packed - 1.0, 0.0);
  const dQ = Math.floor(v / 2048.0);
  const gQ = v - dQ * 2048.0;
  return [
    Math.max(0.05, Math.min(1.0, 1.0 - dQ / 1000.0)),
    Math.max(-1.0, Math.min(1.0, gQ / 1000.0 - 1.0)),
  ];
}

function formatFormula(counts: Map<string, number>): string {
  const keys = [...counts.keys()].sort((a, b) => {
    if (a === 'C' && b !== 'C') return -1;
    if (b === 'C' && a !== 'C') return 1;
    if (a === 'H' && b !== 'H') return -1;
    if (b === 'H' && a !== 'H') return 1;
    return a.localeCompare(b);
  });
  return keys.map((k) => `${k}${toSubscript(counts.get(k) ?? 1)}`).join('');
}

function buildSelectedElement(el: Element2D): TooltipSelectedElement {
  const roleLabel =
    el.valence === 0
      ? 'Noble gas'
      : el.role === 0
        ? `Covalent (${el.valence} ${el.valence === 1 ? 'bond' : 'bonds'})`
        : el.role > 0
          ? `Donor (+${el.valence})`
          : `Acceptor (-${el.valence})`;
  return {
    symbol: getElementSymbol(el.atomicNumber),
    name: getElementName(el.atomicNumber),
    atomicNumber: el.atomicNumber,
    subshellLabel: el.subshellLabel,
    roleLabel,
    outerElectrons: el.outerElectrons,
    outerCapacity: el.outerCapacity,
    valence: el.valence,
    color: el.color,
  };
}

export type EmergentBlobRole =
  | 'producer'
  | 'consumer'
  | 'mixotroph'
  | 'decomposer'
  | 'halophile';

export interface EmergentBlobSummary {
  size: number;
  cytoplasmCount: number;
  lipidCount: number;
  graduatedCount: number;
  graduated: boolean;
  encapsulated: boolean;
  distinctSpecies: number;
  formula: string;
  avgEnergy: number;
  avgVit: number;
  avgPoi: number;
  avgCoh: number;
  avgAsym: number;
  avgSolarFrac: number;
  avgRedoxFrac: number;
  avgMotility: number;
  avgEncapGene: number;
  encapStrategy: 'early-armor' | 'balanced' | 'late-complex';
  autoCap: number;
  predCap: number;
  decompCap: number;
  haloTol: number;
  role: EmergentBlobRole;
  mdpAction: MdpActionId;
  mdpPolicyLabel: string;
}

export function kinAngularDist(a: number, b: number): number {
  return 0.5 * (1 - Math.cos(a - b));
}

export function elemSolarWeight(atomicNum: number): number {
  if (atomicNum === 5) return 1.0;
  if (atomicNum === 6) return 0.8;
  if (atomicNum === 1) return 0.25;
  if (atomicNum === 12) return 0.15;
  return 0;
}

export function elemRedoxWeight(atomicNum: number): number {
  if (atomicNum === 12) return 1.0;
  if (atomicNum === 1) return 0.85;
  if (atomicNum === 5) return 0.25;
  if (atomicNum === 6) return 0.1;
  return 0;
}

export function summarizeAndClassifyBlob(
  memberIndices: number[],
  particles: number[][],
  lifeState: number[][],
  elements: Element2D[],
  speciesTraits: number[][],
  nearbyHalogens: number = 0,
): EmergentBlobSummary {
  const n = Math.max(1, memberIndices.length);
  const formulaCounts = new Map<string, number>();
  const spSet = new Set<number>();
  let cytoplasmCount = 0;
  let lipidCount = 0;
  let graduatedCount = 0;
  let eSum = 0;
  let vitSum = 0;
  let poiSum = 0;
  let cohSum = 0;
  let asymSum = 0;
  let solarSum = 0;
  let redoxSum = 0;
  let speedSum = 0;
  let geneSum = 0;
  const mdpActionCounts = [0, 0, 0, 0];

  for (let idxPos = 0; idxPos < memberIndices.length; idxPos++) {
    const id = memberIndices[idxPos];
    const p = particles[id];
    const ls = lifeState[id] ?? [0.65, 0, 0, 1];
    const meta = unpackLifeMeta(ls);
    if (meta.graduated) graduatedCount++;
    const el = elements[Math.max(0, Math.min(elements.length - 1, Math.floor(p?.[6] ?? 0)))];
    if (el) {
      const s = getElementSymbol(el.atomicNumber);
      formulaCounts.set(s, (formulaCounts.get(s) ?? 0) + 1);
    }
    eSum += ls[0] ?? 0;
    vitSum += ls[1] ?? 0;
    poiSum += ls[2] ?? 0;
    geneSum += meta.encapGene;

    if (meta.phase === 2) {
      lipidCount++;
    } else {
      cytoplasmCount++;
      mdpActionCounts[unpackMdpActionFromPad(p?.[11] ?? 0)]++;
      spSet.add(meta.bioSpecies);
      const tr = speciesTraits[meta.bioSpecies] ?? [0.5, 0.25, 0.25, 0.35];
      if (el) {
        solarSum += elemSolarWeight(el.atomicNumber);
        redoxSum += elemRedoxWeight(el.atomicNumber);
      }
      cohSum += tr[0] ?? 0.5;
      asymSum += tr[3] ?? 0.35;
      if (p) {
        speedSum += Math.hypot(p[3] ?? 0, p[4] ?? 0, p[5] ?? 0);
      }
    }
  }

  const bioN = Math.max(1, cytoplasmCount);
  const avgEnergy = eSum / n;
  const avgVit = vitSum / n;
  const avgPoi = poiSum / n;
  const avgEncapGene = geneSum / n;
  const encapStrategy: 'early-armor' | 'balanced' | 'late-complex' =
    avgEncapGene <= 4.5
      ? 'early-armor'
      : avgEncapGene <= 10.5
        ? 'balanced'
        : 'late-complex';
  const avgCoh = cytoplasmCount > 0 ? cohSum / bioN : 0.5;
  const avgAsym = cytoplasmCount > 0 ? asymSum / bioN : 0.35;
  const avgSolarFrac = cytoplasmCount > 0 ? solarSum / bioN : 0.65;
  const avgRedoxFrac = cytoplasmCount > 0 ? redoxSum / bioN : 0.35;
  const avgMotility = cytoplasmCount > 0 ? Math.max(0, Math.min(1.5, speedSum / bioN / 0.16)) : 0;
  const lightExposure = Math.max(0.64, Math.min(1.0, 1.16 - 0.008 * Math.min(cytoplasmCount, 65)));

  const autoCap = Math.max(
    0.12,
    Math.min(1.55, ((0.42 + 0.9 * avgCoh) * (0.5 + 0.85 * avgSolarFrac) * lightExposure) / (1 + 0.45 * avgMotility)),
  );
  const predCap = Math.max(
    0.05,
    Math.min(1.65, (0.24 + 0.95 * avgAsym) * (0.40 + 0.85 * avgMotility) * (0.68 + 0.52 * avgRedoxFrac)),
  );
  const decompCap = Math.max(
    0.1,
    Math.min(1.4, (0.3 + 0.9 * avgRedoxFrac) * (0.55 + 0.55 * (1 - Math.min(avgMotility, 1)))),
  );
  const haloTol = Math.max(0, Math.min(1, 0.25 + 0.65 * avgRedoxFrac + 0.25 * avgCoh));

  let role: EmergentBlobRole = 'producer';
  if (avgPoi > 0.12 || (haloTol >= 0.66 && nearbyHalogens >= 4)) {
    role = 'halophile';
  } else if (
    (avgEnergy < 0.44 && decompCap >= 0.45) ||
    (decompCap > autoCap * 1.15 && decompCap > predCap * 1.12 && avgRedoxFrac >= 0.46 && avgMotility < 0.30)
  ) {
    role = 'decomposer';
  } else if (predCap > autoCap * 0.94 && avgMotility >= 0.28 && avgAsym >= 0.32) {
    role = autoCap >= 0.46 && autoCap >= predCap * 0.78 ? 'mixotroph' : 'consumer';
  } else if (autoCap >= 0.46 && predCap >= 0.46 && avgAsym >= 0.36 && avgMotility >= 0.30) {
    role = 'mixotroph';
  }

  let dominantAct: MdpActionId = MDP_ACTION_FORAGE;
  let bestActCount = -1;
  for (let a = 0; a < 4; a++) {
    if (mdpActionCounts[a] > bestActCount) {
      bestActCount = mdpActionCounts[a];
      dominantAct = a as MdpActionId;
    }
  }

  return {
    size: memberIndices.length,
    cytoplasmCount,
    lipidCount,
    graduatedCount,
    graduated: graduatedCount >= 4 && cytoplasmCount >= 28,
    encapsulated: lipidCount >= 4,
    distinctSpecies: spSet.size,
    formula: formatFormula(formulaCounts),
    avgEnergy,
    avgVit,
    avgPoi,
    avgCoh,
    avgAsym,
    avgSolarFrac,
    avgRedoxFrac,
    avgMotility,
    avgEncapGene,
    encapStrategy,
    autoCap,
    predCap,
    decompCap,
    haloTol,
    role,
    mdpAction: dominantAct,
    mdpPolicyLabel: MDP_ACTION_LABELS[dominantAct] ?? 'Foraging',
  };
}

function unitDistChord2(a: number[], b: number[]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function projectUnitVecToScreen(
  v: number[],
  proj: d3Geo.GeoProjection,
): [number, number] | null {
  const x = v[0] ?? 0;
  const y = v[1] ?? 0;
  const z = v[2] ?? 0;
  const r = Math.hypot(x, y, z);
  if (r < 1e-6) return null;
  const lat = Math.asin(Math.max(-1, Math.min(1, z / r))) * (180 / Math.PI);
  const lon = Math.atan2(y, x) * (180 / Math.PI);
  const pt = proj([lon, lat]);
  if (!pt || !Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) return null;
  return [pt[0], pt[1]];
}

export function unpackLifeMeta(ls: number[] | undefined): {
  phase: number;
  bioSpecies: number;
  kinPhase: number;
  encapGene: number;
  graduated: boolean;
} {
  const packed = ls?.[3] ?? 0;
  if (packed < 0.5) {
    return { phase: 0, bioSpecies: 0, kinPhase: 0, encapGene: 7, graduated: false };
  }
  const u = Math.round(packed);
  return {
    phase: u & 3,
    bioSpecies: (u >>> 2) & 255,
    kinPhase: (((u >>> 10) & 255) / 256) * Math.PI * 2,
    encapGene: (u >>> 19) & 15,
    graduated: ((u >>> 23) & 1) === 1,
  };
}

export function analyzeReactionAt(
  clientX: number,
  clientY: number,
  snap: HoverSimSnapshot,
  cfg: HoverInspectConfig,
  pickPx: number = 20,
  includeSingletons: boolean = false,
  pinnedTarget: TooltipTarget | null = null,
): TooltipContent | null {
  const proj = cfg.getProj();
  if (!proj?.invert) return null;

  const viewLayer = cfg.getViewLayer();
  const showChem = viewLayer !== 'ocean';
  const showLife = viewLayer !== 'air';
  const showMinerals = cfg.getMineralOpacity ? cfg.getMineralOpacity() > 0.001 : true;
  const R = Math.max(cfg.getWorldSize(), 0.25);
  const airRMax = cfg.getAirRMax();
  const oceanRMax = cfg.getOceanRMax();
  const beta = cfg.getBeta();
  const catRadius = Math.max(oceanRMax * 1.35, 0.065);
  const scalePx = Math.max(cfg.getScale(), 120);

  // Pick radius in unit-sphere chord units matching pointer target size on screen & physical interaction radius
  const pxChord = pickPx / scalePx;
  const chemPickChord = Math.min(0.18, Math.max((airRMax / R) * 1.05, pxChord));
  const chemPickChord2 = chemPickChord * chemPickChord;
  const lifePickChord = Math.min(0.16, Math.max((oceanRMax / R) * 1.3, pxChord));
  const lifePickChord2 = lifePickChord * lifePickChord;
  const maxPickChord = Math.max(chemPickChord, lifePickChord);

  let uCur: [number, number, number] | null = null;
  let bestIdx = -1;
  let bestPhase: 0 | 1 | 2 | 3 = 0;
  let bestBioSp = 0;
  let bestScore = 1.0;
  let pinnedPt: [number, number] | null = null;

  const nParticles = snap.particles.length;
  if (pinnedTarget) {
    if (pinnedTarget.idx < 0 || pinnedTarget.idx >= nParticles) {
      return null;
    }
    const p = snap.particles[pinnedTarget.idx];
    const meta = unpackLifeMeta(snap.lifeState[pinnedTarget.idx]);
    if (
      (meta.phase === 0 && !showChem) ||
      ((meta.phase === 1 || meta.phase === 2) && !showLife) ||
      (meta.phase === 3 && !showMinerals)
    ) {
      return null;
    }
    pinnedPt = projectUnitVecToScreen(p, proj);
    if (!pinnedPt) return null;
    uCur = [p[0], p[1], p[2]];
    bestIdx = pinnedTarget.idx;
    bestPhase = meta.phase === 0 ? 0 : meta.phase === 2 ? 2 : meta.phase === 3 ? 3 : 1;
    bestBioSp = meta.bioSpecies;
  }

  if (!uCur) {
    const ll = proj.invert([clientX, clientY]);
    if (!ll || !Number.isFinite(ll[0]) || !Number.isFinite(ll[1])) return null;
    const chk = proj(ll);
    if (
      !chk ||
      !Number.isFinite(chk[0]) ||
      !Number.isFinite(chk[1]) ||
      Math.hypot(chk[0] - clientX, chk[1] - clientY) > 12
    ) {
      return null;
    }
    uCur = lonLatToUnitVec(ll[0], ll[1]);
  }

  const localChemIdx: number[] = [];
  const localPatchR = Math.max((airRMax * 3.0) / R, maxPickChord * 2.4);
  const localPatchR2 = localPatchR * localPatchR;

  if (!pinnedTarget) {
    for (let i = 0; i < nParticles; i++) {
      const meta = unpackLifeMeta(snap.lifeState[i]);
      if (
        (meta.phase === 0 && !showChem) ||
        ((meta.phase === 1 || meta.phase === 2) && !showLife) ||
        (meta.phase === 3 && !showMinerals)
      ) {
        continue;
      }
      const p = snap.particles[i];
      const d2 = unitDistChord2(p, uCur);
      const thr2 = meta.phase === 0 ? chemPickChord2 : lifePickChord2;
      if (d2 < thr2) {
        const score = d2 / Math.max(thr2, 1e-12);
        if (score < bestScore) {
          bestScore = score;
          bestIdx = i;
          bestPhase = meta.phase === 0 ? 0 : meta.phase === 2 ? 2 : meta.phase === 3 ? 3 : 1;
          bestBioSp = meta.bioSpecies;
        }
      }
    }
  }

  if (bestIdx < 0) {
    return null;
  }

  const centerRef = snap.particles[bestIdx];
  const catChord = catRadius / R;
  const catRef2 = Math.max(catChord * catChord, lifePickChord2);
  const blobPatchR = Math.min(0.28, Math.max(catChord * 4.5, lifePickChord * 3.0));
  const blobPatchR2 = blobPatchR * blobPatchR;
  const nearbyLife: number[] = [];
  const localLifePool: number[] = [];
  const localMinerals: number[] = [];

  for (let i = 0; i < nParticles; i++) {
    const meta = unpackLifeMeta(snap.lifeState[i]);
    if (meta.phase === 0 && showChem) {
      if (unitDistChord2(snap.particles[i], centerRef) < localPatchR2) {
        localChemIdx.push(i);
      }
    } else if ((meta.phase === 1 || meta.phase === 2) && showLife) {
      const d2 = unitDistChord2(snap.particles[i], centerRef);
      if (d2 < catRef2) {
        nearbyLife.push(i);
      }
      if (d2 < blobPatchR2) {
        localLifePool.push(i);
      }
    } else if (meta.phase === 3 && showMinerals) {
      if (unitDistChord2(snap.particles[i], centerRef) < blobPatchR2) {
        localMinerals.push(i);
      }
    }
  }

  const els = snap.elements;
  const p0 = snap.particles[bestIdx];
  const targetPt = pinnedPt ?? projectUnitVecToScreen(p0, proj);
  const targetX = targetPt?.[0] ?? clientX;
  const targetY = targetPt?.[1] ?? clientY;
  const c0 = Math.max(0, Math.min(els.length - 1, Math.floor(p0[6] ?? 0)));
  const el0 = els[c0];
  const selectedElement = buildSelectedElement(el0);
  const sym0 = selectedElement.symbol;

  if (bestPhase === 0) {
    const valVec = snap.valence[bestIdx] ?? [0, 0, 0, 0];
    const atomLoad = (valVec[0] ?? 0) + (valVec[1] ?? 0) + (valVec[2] ?? 0);

    // Build connected component & direct neighbors inside local Chemistry patch
    const bondChord = (airRMax * 0.78) / R;
    const bondChord2 = bondChord * bondChord;
    const maxChord = airRMax / R;
    const maxChord2 = maxChord * maxChord;

    const compIds = [bestIdx];
    const visited = new Set<number>([bestIdx]);
    const directInteract: { idx: number; dist: number; vRelDot: number }[] = [];

    if (el0.valence > 0) {
      let head = 0;
      while (head < compIds.length && compIds.length < 18) {
        const u = compIds[head++];
        const pu = snap.particles[u];
        const eu = els[Math.max(0, Math.min(els.length - 1, Math.floor(pu[6] ?? 0)))];
        if (!eu || eu.valence === 0) continue;

        for (const j of localChemIdx) {
          const pj = snap.particles[j];
          const ej = els[Math.max(0, Math.min(els.length - 1, Math.floor(pj[6] ?? 0)))];
          if (!ej || ej.valence === 0) continue;
          const d2 = unitDistChord2(pu, pj);
          if (u === bestIdx && j !== bestIdx && d2 < maxChord2) {
            const dist = 2 * R * Math.asin(Math.min(1, 0.5 * Math.sqrt(d2)));
            const dx = pj[0] - pu[0];
            const dy = pj[1] - pu[1];
            const dz = pj[2] - pu[2];
            const dvx = pj[3] - pu[3];
            const dvy = pj[4] - pu[4];
            const dvz = pj[5] - pu[5];
            const invL = 1 / Math.max(Math.sqrt(d2), 1e-6);
            const vRelDot = (dvx * dx + dvy * dy + dvz * dz) * invL;
            directInteract.push({ idx: j, dist, vRelDot });
          }
          if (d2 < bondChord2 && !visited.has(j)) {
            const valU =
              (snap.valence[u]?.[0] ?? 0) +
              (snap.valence[u]?.[1] ?? 0) +
              (snap.valence[u]?.[2] ?? 0);
            const valJ =
              (snap.valence[j]?.[0] ?? 0) +
              (snap.valence[j]?.[1] ?? 0) +
              (snap.valence[j]?.[2] ?? 0);
            if (valU > 0.22 && valJ > 0.22) {
              visited.add(j);
              compIds.push(j);
            }
          }
        }
      }
    }

    const formulaCounts = new Map<string, number>();
    let hasDonor = false;
    let hasAcceptor = false;
    let hasCovalent = false;
    for (const id of compIds) {
      const pi = snap.particles[id];
      const ei = els[Math.max(0, Math.min(els.length - 1, Math.floor(pi[6] ?? 0)))];
      const s = getElementSymbol(ei.atomicNumber);
      formulaCounts.set(s, (formulaCounts.get(s) ?? 0) + 1);
      if (ei.role > 0) hasDonor = true;
      else if (ei.role < 0) hasAcceptor = true;
      else if (ei.valence > 0) hasCovalent = true;
    }

    const formula = formatFormula(formulaCounts);
    const bio = snap.bioForce[bestIdx] ?? [0, 0, 0, 0];
    const [actScale, glow] = unpackActGlow(bio[3] ?? 0);
    const kickMag = Math.hypot(bio[0] ?? 0, bio[1] ?? 0, bio[2] ?? 0);
    const localTChem = cfg.getLocalTemps ? cfg.getLocalTemps([p0[0], p0[1], p0[2]]) : null;
    const localUvChem = cfg.getLocalUv ? cfg.getLocalUv([p0[0], p0[1], p0[2]]) : null;
    const airTempPart = localTChem
      ? `${localTChem.airLabel.replace(' Air', '')} ${localTChem.airC >= 0 ? '+' : ''}${localTChem.airC.toFixed(1)}°C`
      : '';
    const uvPartChem = localUvChem ? `UV ${localUvChem.uviNow.toFixed(1)}` : '';
    const airTempChip =
      airTempPart && uvPartChem
        ? `${airTempPart} · ${uvPartChem}`
        : airTempPart || uvPartChem || undefined;
    const airEnvChips: TooltipEnvChip[] = [];
    if (localTChem && airTempPart) {
      airEnvChips.push({
        label: airTempPart,
        dotRgb: cfg.getEnvColorRgb?.('air', localTChem.airC),
      });
    }
    if (localUvChem && uvPartChem) {
      airEnvChips.push({
        label: uvPartChem,
        dotRgb: cfg.getEnvColorRgb?.('uv', localUvChem.uviNow),
      });
    }

    const bioActive =
      cfg.isBioLinked() &&
      showLife &&
      (kickMag > 0.02 || actScale < 0.985 || Math.abs(glow) > 0.06);

    const nearbySummary =
      nearbyLife.length > 0
        ? summarizeAndClassifyBlob(
            nearbyLife,
            snap.particles,
            snap.lifeState,
            els,
            snap.speciesTraits ?? [],
            0,
          )
        : null;
    const avgE = nearbySummary?.avgEnergy ?? 0;
    const avgVit = nearbySummary?.avgVit ?? 0;
    const avgPoi = nearbySummary?.avgPoi ?? 0;
    const nearbyRoleLabel =
      nearbySummary?.role === 'halophile'
        ? 'Halophile Biofilm'
        : nearbySummary?.role === 'decomposer'
          ? 'Decomposer Colony'
          : nearbySummary?.role === 'mixotroph'
            ? 'Mixotroph Blob'
            : nearbySummary?.role === 'consumer'
              ? 'Consumer Glider'
              : 'Producer Blob';

    const hasOngoingChem = compIds.length >= 2 || directInteract.length > 0;
    if (bioActive) {
      const kind = bioKind(el0.atomicNumber, el0.valence);
      const dropPct = Math.round((1 - actScale) * 100);
      const radicalSuffix =
        compIds.length === 1
          ? `${sym0} monomer`
          : `${formula} (${compIds.length}${compIds.length >= 18 ? '+' : ''} atoms)`;

      let theme: TooltipTheme = glow >= 0 ? 'anabolic' : 'catabolic';
      let badge = 'BIO-CHEM CATALYSIS';
      let body = '';

      if (kind === 3) {
        theme = 'cofactor';
        badge = 'VITAMIN COFACTOR';
        body = `Abiotic ${sym0} cofactor stabilizing nearby emergent blobs (${nearbyLife.length} living particles), calming thermal jitter (+vit ${(avgVit * 100).toFixed(0)}%).`;
      } else if (kind === 4) {
        if ((nearbySummary?.haloTol ?? 0) > 0.58 && glow < -0.05) {
          theme = 'cofactor';
          badge = 'HALOPHILE DETOX';
          body = `Halotolerant biofilm detoxifying abiotic ${sym0} halide into benign salts; barrier lowered ${dropPct}%.`;
        } else {
          theme = 'toxic';
          badge = 'HALOGEN TOXIN';
          body = `Abiotic ${sym0} stressing nearby emergent blobs (avg poison ${(avgPoi * 100).toFixed(0)}%, draining energy).`;
        }
      } else if (kind === 2) {
        if (glow >= 0) {
          theme = 'anabolic';
          badge = 'CARBON FIXATION';
          body = `Cohesive ${nearbyRoleLabel} fixing biogenic ${radicalSuffix}: barrier cut ${dropPct}% while recruiting carbon into living tissue.`;
        } else {
          theme = 'catabolic';
          badge = 'CARBON METABOLISM';
          body = `Motile ${nearbyRoleLabel} cleaving ${radicalSuffix} carbon bonds, harvesting chemical energy (+ΔE).`;
        }
      } else if (kind === 1) {
        if (glow >= 0) {
          theme = 'anabolic';
          badge = 'HYDRIDE FIXATION';
          body = `Autotrophic blob funneling biogenic ${sym0} fuel into lowered-barrier bond synthesis (-${dropPct}% activation).`;
        } else {
          theme = 'catabolic';
          badge = 'FUEL HARVEST';
          body = `Emergent blob harvesting ${sym0} chemical fuel: cleaving weak bonds with exothermic kinetic kick.`;
        }
      } else {
        theme = glow >= 0 ? 'anabolic' : 'catabolic';
        badge = glow < -0.05 && atomLoad >= 1.35 ? 'OXIDE STRIPPING' : 'RADICAL STABILIZATION';
        body =
          glow < -0.05 && atomLoad >= 1.35
            ? `Blobs stripping excess bound ${sym0} from saturated oxides to free reactive bonding sites.`
            : `Marine symbionts guiding ${sym0} radicals toward stable aqueous equilibrium.`;
      }

      const meta = `Nearby ecology: ${nearbyRoleLabel} · Blob avg E=${avgE.toFixed(2)} · Valence load ${atomLoad.toFixed(2)}/${el0.valence}`;
      return {
        theme,
        badge,
        tempTag: airTempChip,
        envChips: airEnvChips,
        title: `${badge === 'HALOGEN TOXIN' ? 'Toxic' : 'Catalyzed'} ${radicalSuffix}`,
        body,
        meta,
        element: selectedElement,
        targetPhase: 0,
        targetIdx: bestIdx,
        targetX,
        targetY,
      };
    }

    if (hasOngoingChem) {
      directInteract.sort((a, b) => a.dist - b.dist);
      const satPct = Math.min(100, Math.round((atomLoad / Math.max(el0.valence, 1)) * 100));
      const bondingCount = compIds.length;
      const approaching = directInteract.filter(
        (n) => n.dist >= airRMax * Math.max(beta, 0.35) && n.vRelDot < -0.08,
      ).length;
      const separating = directInteract.filter(
        (n) => n.dist < airRMax * 0.85 && n.vRelDot > 0.14,
      ).length;

      let title = bondingCount >= 2 ? `${formula} Complex` : `${sym0} Collision Pair`;
      let badge = 'CHEMICAL REACTION';
      let body = '';

      if (bondingCount === 1) {
        const nb = snap.particles[directInteract[0].idx];
        const eNb = els[Math.max(0, Math.min(els.length - 1, Math.floor(nb[6] ?? 0)))];
        const symNb = getElementSymbol(eNb.atomicNumber);
        const pairCounts = new Map<string, number>([[sym0, 1]]);
        pairCounts.set(symNb, (pairCounts.get(symNb) ?? 0) + 1);
        const targetFormula = formatFormula(pairCounts);
        if (approaching > 0) {
          title = `Forming ${targetFormula}`;
          badge = 'BOND ASSEMBLY';
          body = `${sym0} (${el0.subshellLabel}) and ${symNb} (${eNb.subshellLabel}) converging across the activation barrier toward orbital overlap.`;
        } else if (separating > 0) {
          title = `Scattering ${sym0} + ${symNb}`;
          badge = 'COLLISION REBOUND';
          body = `${sym0} (${el0.subshellLabel}) and ${symNb} (${eNb.subshellLabel}) rebounding after short-range core repulsion (${directInteract.length} neighbors).`;
        } else {
          title = `${sym0} + ${symNb} Alignment`;
          badge = 'ORBITAL ALIGNMENT';
          body = `${sym0} and ${symNb} rotating valence orbitals within reaction range toward a ${targetFormula} encounter (${directInteract.length} nearby atoms).`;
        }
      } else if (separating > 0 && atomLoad > 0.15) {
        badge = 'BOND DISSOCIATION';
        body = `Thermal agitation / wind shear pulling apart ${formula} bonds (${cfg.getTemperature() > 0.05 ? 'endothermic cleavage' : 'mechanical shear'}).`;
      } else if (hasDonor && hasAcceptor) {
        badge = 'IONIC TRANSFER';
        body = `Donor → acceptor valence charge transfer stabilizing ${formula} (${bondingCount}${bondingCount >= 18 ? '+' : ''} atoms, ${satPct}% shell filled).`;
      } else if (satPct < 75 && hasCovalent) {
        badge = 'REACTIVE RADICAL';
        body = `Unsaturated ${formula} radical with open covalent orbital slots (${satPct}% filled), attracting free H/O/C donors.`;
      } else {
        badge = bondingCount >= 7 ? 'MACROMOLECULE' : 'COVALENT BOND';
        body = `Directional valence lobes sharing electrons in a ${bondingCount}${bondingCount >= 18 ? '+' : ''}-atom ${formula} network (${satPct}% valence filled).`;
      }

      return {
        theme: 'chem',
        badge,
        tempTag: airTempChip,
        envChips: airEnvChips,
        title,
        body,
        meta: `Shell [${el0.outerElectrons}/${el0.outerCapacity}] · Valence load ${atomLoad.toFixed(2)}/${el0.valence} · Nearby pairs: ${directInteract.length}`,
        element: selectedElement,
        targetPhase: 0,
        targetIdx: bestIdx,
        targetX,
        targetY,
      };
    }

    if (includeSingletons) {
      const inert = el0.valence === 0;
      const ls0 = snap.lifeState[bestIdx] ?? [0, 0, 0, 0];
      const comp = ls0[1] ?? 0;
      const alt01 = ls0[2] ?? 0;
      const ocMaskChem = cfg.getOceanMask ? cfg.getOceanMask([p0[0], p0[1], p0[2]]) : 1;
      const mediumLabel =
        comp >= 1.5
          ? 'Polar Sea Ice'
          : comp >= 0.5
            ? ocMaskChem >= 0.42
              ? 'Ocean / Aqueous Water'
              : 'Continental Runoff / Soil'
            : `Atmospheric Gas (${Math.round(1000 - alt01 * 750)} hPa)`;
      return {
        theme: comp >= 0.5 ? 'ocean' : 'chem',
        badge: inert ? 'NOBLE GAS' : comp >= 1.5 ? 'CRYOSPHERE ICE' : comp >= 0.5 ? 'AQUEOUS / HYDROSPHERE' : 'ATMOSPHERIC GAS',
        tempTag: airTempChip,
        envChips: airEnvChips,
        title: `${selectedElement.name} (${sym0}) Atom`,
        body: inert
          ? `Inert ${sym0} (${el0.subshellLabel}) drifting across upper and surface wind layers (${mediumLabel}); permanently abiotic.`
          : `Unbonded ${sym0} (${el0.subshellLabel}) monomer with ${el0.valence} open valence slot${el0.valence === 1 ? '' : 's'} cycling through ${mediumLabel}.`,
        meta: `${mediumLabel} · Valence [${el0.outerElectrons}/${el0.outerCapacity}] · Nearby atoms: ${Math.max(0, localChemIdx.length - 1)}`,
        element: selectedElement,
        targetPhase: 0,
        targetIdx: bestIdx,
        targetX,
        targetY,
      };
    }

    return null;
  }

  const ocMask = cfg.getOceanMask ? cfg.getOceanMask([p0[0], p0[1], p0[2]]) : 1;
  const overLand = ocMask < 0.42;
  const localTSurf = cfg.getLocalTemps ? cfg.getLocalTemps([p0[0], p0[1], p0[2]]) : null;
  const localUvSurf = cfg.getLocalUv ? cfg.getLocalUv([p0[0], p0[1], p0[2]]) : null;
  const localHydroSurf = cfg.getLocalHydro ? cfg.getLocalHydro([p0[0], p0[1], p0[2]]) : null;
  const surfTempPart = localTSurf
    ? overLand || localTSurf.sstC === null
      ? `Sfc ${localTSurf.sfcAirC >= 0 ? '+' : ''}${localTSurf.sfcAirC.toFixed(1)}°C`
      : `SST ${localTSurf.sstC >= 0 ? '+' : ''}${localTSurf.sstC.toFixed(1)}°C`
    : '';
  const uvPartSurf = localUvSurf ? `UV ${localUvSurf.uviNow.toFixed(1)}` : '';
  const hydroPartSurf = localHydroSurf
    ? `RH ${localHydroSurf.rhPct}%${localHydroSurf.precip3hMm >= 0.05 ? ` · 3HPS ${localHydroSurf.precip3hMm.toFixed(1)}mm` : ''}`
    : '';
  const surfTempTag =
    [surfTempPart, uvPartSurf, hydroPartSurf].filter(Boolean).join(' · ') || undefined;
  const surfEnvChips: TooltipEnvChip[] = [];
  if (localTSurf && surfTempPart) {
    const useSst = !overLand && localTSurf.sstC !== null;
    const tC = useSst ? (localTSurf.sstC as number) : localTSurf.sfcAirC;
    surfEnvChips.push({
      label: surfTempPart,
      dotRgb: cfg.getEnvColorRgb?.(useSst ? 'sst' : 'air', tC),
    });
  }
  if (localUvSurf && uvPartSurf) {
    surfEnvChips.push({
      label: uvPartSurf,
      dotRgb: cfg.getEnvColorRgb?.('uv', localUvSurf.uviNow),
    });
  }
  if (localHydroSurf) {
    surfEnvChips.push({
      label: `RH ${localHydroSurf.rhPct}%`,
      dotRgb: cfg.getEnvColorRgb?.('rh', localHydroSurf.rhPct),
    });
    if (localHydroSurf.precip3hMm >= 0.05) {
      surfEnvChips.push({
        label: `3HPS ${localHydroSurf.precip3hMm.toFixed(1)}mm`,
        dotRgb: cfg.getEnvColorRgb?.('precip', localHydroSurf.precip3hMm),
      });
    }
  }

  // Inorganic Crust-Pinned Mineral (`bestPhase === 3`): cluster matching crystal vein & inspect risk/reward
  if (bestPhase === 3) {
    const minConnectChord2 = Math.pow(Math.max(0.02, (oceanRMax * 1.45) / R), 2);
    const meta0 = unpackLifeMeta(snap.lifeState[bestIdx]);
    const minType = Math.min(4, Math.max(0, meta0.bioSpecies));

    const veinIds = [bestIdx];
    const veinSeen = new Set<number>([bestIdx]);
    let headMin = 0;
    while (headMin < veinIds.length && veinIds.length < 180) {
      const u = veinIds[headMin++];
      const pu = snap.particles[u];
      for (const j of localMinerals) {
        if (veinSeen.has(j)) continue;
        if (unitDistChord2(pu, snap.particles[j]) < minConnectChord2) {
          const mj = unpackLifeMeta(snap.lifeState[j]);
          if (mj.bioSpecies === minType) {
            veinSeen.add(j);
            veinIds.push(j);
          }
        }
      }
    }

    const counts = new Map<string, number>();
    let integSum = 0;
    let nutrSum = 0;
    let hazSum = 0;
    let trapCount = 0;
    let beHit = false;
    let halHit = false;
    for (const id of veinIds) {
      const cIdx = Math.max(0, Math.min(els.length - 1, Math.floor(snap.particles[id]?.[6] ?? 0)));
      const el = els[cIdx];
      if (el) {
        const s = getElementSymbol(el.atomicNumber);
        counts.set(s, (counts.get(s) ?? 0) + 1);
        if (el.atomicNumber === 4) beHit = true;
        if (el.atomicNumber === 7 || el.atomicNumber === 13) halHit = true;
      }
      const lsi = snap.lifeState[id] ?? [0.85, 0, 0, 3];
      const mi = unpackLifeMeta(lsi);
      integSum += lsi[0] ?? 0.85;
      nutrSum += lsi[1] ?? 0;
      hazSum += lsi[2] ?? 0;
      if (mi.graduated) trapCount++;
    }

    const formula = [...counts.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([s, c]) => `${s}${toSubscript(c)}`)
      .join('');
    const avgInteg = Math.round(Math.min(1, integSum / Math.max(1, veinIds.length)) * 100);
    const avgHaz = hazSum / Math.max(1, veinIds.length);
    const avgNutr = nutrSum / Math.max(1, veinIds.length);
    const isTrap = meta0.graduated || trapCount > 0 || beHit || avgHaz >= 0.50;
    const windMs = cfg.getWindSpeedMs ? cfg.getWindSpeedMs([p0[0], p0[1], p0[2]]) : 0;
    const curMs = cfg.getOceanSpeedMs ? cfg.getOceanSpeedMs([p0[0], p0[1], p0[2]]) : 0;
    const crystWord = veinIds.length === 1 ? 'crystal' : 'crystals';

    let theme: TooltipTheme = 'mineral';
    let badge = 'SILICATE BEDROCK · CRUSTAL REEF';
    let title = `Quartz-Olivine Reef (${veinIds.length} ${crystWord} · ${formula})`;
    let stabilityStr = 'Insoluble in air & water · Storm/surf mechanical shear only';
    let encounterStr =
      'Physical shelter against currents with mild halogen detoxification, though fast impact abrades unarmored cytoplasm.';

    if (minType === 1) {
      theme = isTrap || halHit ? 'mineralTrap' : 'mineral';
      badge = isTrap ? 'EVAPORITE FLUORITE · OSMOTIC HAZARD' : 'EVAPORITE HALITE · WATER-SOLUBLE';
      title = `${isTrap ? 'Halite-Fluorite Trap' : 'Halite Salt Crust'} (${veinIds.length} ${crystWord} · ${formula})`;
      stabilityStr = overLand
        ? 'Stable on dry land · Slowly abraded into salt dust by wind'
        : 'Dissolving rapidly in ocean water & coastal surf!';
      encounterStr = isTrap
        ? 'Death sentence for naked blobs (Cl/F halogen shock drains energy & poison spikes)—only Halophiles & Lipid Protocells tolerate it.'
        : 'Leaches osmotic salts that stress ordinary blobs while nourishing Halophile biofilms & armored Protocells.';
    } else if (minType === 2) {
      theme = isTrap || beHit ? 'mineralTrap' : 'mineralWindfall';
      badge = isTrap ? 'BERYL-CARBONATE TRAP · POISON' : 'CARBONATE DOLOMITE · CARBON VAULT';
      title = `${isTrap ? 'Beryl-Doped Carbonate' : 'Dolomite Carbonate Vein'} (${veinIds.length} ${crystWord} · ${formula})`;
      stabilityStr = overLand
        ? 'Moderately stable on land · Weathered by winds'
        : 'Moderately stable underwater · Slow turbulent dissolution';
      encounterStr = isTrap
        ? `TRAP VEIN (Be inclusion): lures cells toward ${formula}, then releases toxic beryllium poisoning when bio-mined!`
        : 'Carbon windfall! Chelation releases C/O fuel + cofactors that boost biomass & rapid Lipid Bilayer assembly.';
    } else if (minType === 3) {
      theme = overLand || isTrap ? 'mineralTrap' : 'mineralWindfall';
      badge = overLand || isTrap ? 'SULFIDE VEIN · ACID OXIDIZING' : 'SULFIDE CHIMNEY · CHEMOSYNTHESIS';
      title = `${overLand || isTrap ? 'Oxidizing Pyrite Crust' : 'Hydrothermal Sulfide Chimney'} (${veinIds.length} ${crystWord} · ${formula})`;
      stabilityStr = overLand
        ? 'Air-unstable! Atmospheric O + wind rapidly oxidize sulfide into acid plumes'
        : 'Stable in ocean water · Fuels upwelling chemosynthetic vents';
      encounterStr = overLand || isTrap
        ? 'Corroding in windy air: triggers acidic sulfur flash-burns on exposed cells (partially scavenged by Decomposers).'
        : 'Major chemosynthetic windfall: pumps reducing S/Mg ATP energy even under polar night, deep clouds, or low UV!';
    } else if (minType === 4) {
      theme = 'mineralWindfall';
      badge = 'HYDRATED CLAY · CATALYTIC MATRIX';
      title = `Montmorillonite Clay (${veinIds.length} ${crystWord} · ${formula})`;
      stabilityStr = overLand
        ? 'Inland dehydration! Cracking & flaking into mineral dust under dry winds'
        : 'Stable in oceans & moist coastal shelves';
      encounterStr =
        'Prebiotic evolutionary windfall: detoxifies poisons, boosts enzyme cofactors, and doubles Lipid Bilayer & complex trait evolution!';
    }

    const habitatLabel = overLand ? 'Land Crust' : 'Marine Shelf/Floor';
    const bioMineNote =
      nearbyLife.length > 0 ? ` Active bio-mining by ${nearbyLife.length} nearby cell(s).` : '';
    return {
      theme,
      badge,
      tempTag: surfTempTag,
      envChips: surfEnvChips,
      title,
      body: `${stabilityStr}. ${encounterStr}${bioMineNote}`,
      meta: `Crystal #${bestIdx} (${sym0}) · ${habitatLabel} · Integrity ${avgInteg}% · Wind ${windMs.toFixed(0)} m/s · Current ${curMs.toFixed(2)} m/s · Nutr ${Math.round(avgNutr * 100)}% / Haz ${Math.round(avgHaz * 100)}%`,
      element: selectedElement,
      targetPhase: 3,
      targetIdx: bestIdx,
      targetX,
      targetY,
    };
  }

  // Living Biomass or Lipid Bilayer (phase === 1 or 2): assemble emergent multi-particle protocell via BFS
  const blobConnectChord = Math.max(0.02, (oceanRMax * 1.35) / R);
  const blobConnectChord2 = blobConnectChord * blobConnectChord;
  const tightChord2 = Math.pow((oceanRMax * 0.72) / R, 2);

  const blobIds = [bestIdx];
  const blobVisited = new Set<number>([bestIdx]);
  let head = 0;
  while (head < blobIds.length && blobIds.length < 480) {
    const u = blobIds[head++];
    const pu = snap.particles[u];
    const metaU = unpackLifeMeta(snap.lifeState[u]);
    for (const j of localLifePool) {
      if (blobVisited.has(j)) continue;
      const pj = snap.particles[j];
      const d2 = unitDistChord2(pu, pj);
      if (d2 >= blobConnectChord2) continue;
      const metaJ = unpackLifeMeta(snap.lifeState[j]);
      if (kinAngularDist(metaU.kinPhase, metaJ.kinPhase) < 0.30 || d2 < tightChord2) {
        blobVisited.add(j);
        blobIds.push(j);
      }
    }
  }

  let nearbyHalogens = 0;
  for (const ci of localChemIdx) {
    const eC = els[Math.max(0, Math.min(els.length - 1, Math.floor(snap.particles[ci]?.[6] ?? 0)))];
    if (eC && (eC.atomicNumber === 7 || eC.atomicNumber === 13)) {
      nearbyHalogens++;
    }
  }

  const blob = summarizeAndClassifyBlob(
    blobIds,
    snap.particles,
    snap.lifeState,
    els,
    snap.speciesTraits ?? [],
    nearbyHalogens,
  );
  const ls = snap.lifeState[bestIdx] ?? [0.65, 0, 0, 1];
  const vit = ls[1] ?? 0;
  const poi = ls[2] ?? 0;
  const strategyLabel =
    blob.encapStrategy === 'early-armor'
      ? 'Early-Armor'
      : blob.encapStrategy === 'late-complex'
        ? 'Late-Complex'
        : 'Balanced';
  const strategyGeneStr = `${strategyLabel} (gene ${Math.round(blob.avgEncapGene)}/15)`;
  const habitatTag = overLand ? 'Terrestrial (calm · +chem · -desicc)' : 'Marine (currents)';

  let nearWindfallMinerals = 0;
  let nearTrapMinerals = 0;
  for (const mi of localMinerals) {
    if (unitDistChord2(snap.particles[mi], centerRef) < catRef2) {
      const mm = unpackLifeMeta(snap.lifeState[mi]);
      if (mm.graduated || mm.bioSpecies === 1) nearTrapMinerals++;
      if (mm.bioSpecies === 4 || (!mm.graduated && (mm.bioSpecies === 2 || mm.bioSpecies === 3))) {
        nearWindfallMinerals++;
      }
    }
  }
  const mineralNote =
    nearTrapMinerals > 0 && nearWindfallMinerals > 0
      ? ' Grazing mixed crustal veins (catalytic mineral windfall + hazardous Be/halide leach).'
      : nearTrapMinerals > 0
        ? ' Exposed to corrosive halide/Be/acid mineral leachate!'
        : nearWindfallMinerals > 0
          ? ' Anchored near catalytic Clay/Carbonate/Sulfide mineral windfall.'
          : '';

  if (bestPhase === 2) {
    const theme: TooltipTheme = poi > 0.25 ? 'toxic' : blob.graduated ? 'complex' : 'lipid';
    const badge = blob.graduated
      ? 'COMPLEX BILAYER · MULTICELLULAR ENVELOPE'
      : 'LIPID BILAYER · PROTOCELL MEMBRANE';
    const title = blob.graduated
      ? `Complex Organism Envelope (${blob.lipidCount} lipids · ${blob.cytoplasmCount} somatic cells, ${blob.distinctSpecies} ${blob.distinctSpecies === 1 ? 'color' : 'colors'})`
      : `Biogenic Lipid Shell (${blob.lipidCount} lipids · ${blob.cytoplasmCount} internals, ${blob.distinctSpecies} ${blob.distinctSpecies === 1 ? 'color' : 'colors'})`;
    const landNote = overLand
      ? ' Sheltering internals over calm land by cutting evaporative desiccation cost while harvesting abundant air chemistry.'
      : '';
    const gradNote = blob.graduated
      ? ' Graduated post-engulfment bilayer suppressing core crowding Oxygen dissolution.'
      : '';
    const body = `Oriented ${sym0}-bearing amphiphile built by its host blob under the ${strategyLabel} strategy (gene ${Math.round(blob.avgEncapGene)}/15): hydrophilic heads anchor inner ${blob.formula} cytoplasm and import C/H/O/S nutrients while hydrophobic tails block halogens, harsh UV & grazers.${gradNote}${landNote}${mineralNote}`;
    return {
      theme,
      badge,
      tempTag: surfTempTag,
      envChips: surfEnvChips,
      title,
      body,
      meta: `Lipid #${bestIdx} (${sym0}) · Host #${bestBioSp} · ${habitatTag} · Encap ${strategyGeneStr} · Shell E=${(ls[0] ?? 0).toFixed(2)} · Motility ${blob.avgMotility.toFixed(2)}`,
      element: selectedElement,
      targetPhase: 2,
      targetIdx: bestIdx,
      targetX,
      targetY,
    };
  }

  const hasActivity =
    blob.size >= 2 ||
    vit > 0.04 ||
    poi > 0.04 ||
    blob.avgEnergy < 0.35 ||
    blob.avgEnergy > 0.65;

  if (hasActivity || includeSingletons) {
    const cellWord = blob.cytoplasmCount === 1 ? 'cell' : 'cells';
    const colorWord = blob.distinctSpecies === 1 ? 'color' : 'colors';
    let theme: TooltipTheme = 'ocean';
    let badge = 'PRODUCER BLOB · AUTOTROPH';
    let title = `Producer Blob (${blob.cytoplasmCount} ${cellWord} · ${blob.distinctSpecies} ${colorWord})`;
    let statusDesc = '';
    const encapTag = blob.graduated
      ? `Graduated (${blob.lipidCount} lipids)`
      : blob.encapsulated
        ? `Encapsulated (${blob.lipidCount} lipids)`
        : 'Naked Colony';

    if (blob.cytoplasmCount <= 1 && blob.lipidCount === 0) {
      theme = poi > 0.15 ? 'toxic' : 'ocean';
      badge = 'FREE LIVING CELL';
      title = `Solitary #${bestBioSp} · Living ${selectedElement.name} (${sym0})`;
      statusDesc = `Free single cell carrying conserved ${sym0} (${strategyGeneStr}), drifting until joining a colony or nucleating a heterogeneous blob.`;
    } else if (blob.role === 'halophile') {
      theme = poi > 0.15 ? 'cofactor' : blob.graduated ? 'complex' : 'ocean';
      badge = blob.graduated
        ? 'COMPLEX MULTICELLULAR · HALOPHILE'
        : blob.encapsulated
          ? 'ENCAPSULATED HALOPHILE PROTOCELL'
          : 'HALOPHILE BIOFILM';
      title = `${blob.graduated ? 'Complex Halophile Organism' : `Halophile ${blob.encapsulated ? 'Protocell' : 'Biofilm'}`} (${blob.cytoplasmCount} ${cellWord} · ${blob.distinctSpecies} ${colorWord})`;
      statusDesc = `Halotolerant ${blob.formula} ${blob.graduated ? 'complex organism' : 'colony'} (${encapTag}, ${strategyGeneStr}) clearing dissolved Cl/F halides while shielding embodied ${sym0}.`;
    } else if (blob.role === 'decomposer') {
      theme = blob.graduated ? 'complex' : 'cofactor';
      badge = blob.graduated
        ? 'COMPLEX MULTICELLULAR · SAPROTROPH'
        : blob.encapsulated
          ? 'ENCAPSULATED DECOMPOSER PROTOCELL'
          : 'DECOMPOSER BLOB · SAPROTROPH';
      title = `${blob.graduated ? 'Complex Saprotroph Organism' : `Decomposer ${blob.encapsulated ? 'Protocell' : 'Colony'}`} (${blob.cytoplasmCount} ${cellWord} · ${blob.distinctSpecies} ${colorWord})`;
      statusDesc = `Redox-rich (${Math.round(blob.avgRedoxFrac * 100)}% S/H) ${blob.formula} saprotroph (${encapTag}, ${strategyGeneStr}) remineralizing stressed biomass back into free chemistry.`;
    } else if (blob.role === 'mixotroph') {
      theme = blob.graduated ? 'complex' : 'anabolic';
      badge = blob.graduated
        ? 'COMPLEX MULTICELLULAR · MIXOTROPH'
        : blob.encapsulated
          ? 'ENCAPSULATED MIXOTROPH PROTOCELL'
          : 'MIXOTROPH BLOB';
      title = `${blob.graduated ? 'Complex Mixotroph Organism' : `Mixotroph ${blob.encapsulated ? 'Protocell' : 'Swimmer'}`} (${blob.cytoplasmCount} ${cellWord} · ${blob.distinctSpecies} ${colorWord})`;
      statusDesc = `Hybrid ${blob.formula} anatomy (${encapTag}, ${strategyGeneStr}) pairing a cohesive C/O photosynthetic core (coh=${blob.avgCoh.toFixed(2)}) with a motile tail (asym=${blob.avgAsym.toFixed(2)}).`;
    } else if (blob.role === 'consumer') {
      theme = blob.graduated ? 'complex' : 'catabolic';
      badge = blob.graduated
        ? 'COMPLEX MULTICELLULAR · CONSUMER'
        : blob.encapsulated
          ? 'ENCAPSULATED CONSUMER PROTOCELL'
          : 'CONSUMER BLOB · HETEROTROPH';
      title = `${blob.graduated ? 'Complex Consumer Organism' : `Consumer ${blob.encapsulated ? 'Protocell' : 'Glider'}`} (${blob.cytoplasmCount} ${cellWord} · ${blob.distinctSpecies} ${colorWord})`;
      statusDesc = `Asymmetric ${blob.formula} swimmer (${encapTag}, ${strategyGeneStr}; asym=${blob.avgAsym.toFixed(2)}, motility=${blob.avgMotility.toFixed(2)}) grazing foreign blobs with ~22% organelle endosymbiosis.`;
    } else {
      theme = blob.graduated ? 'complex' : 'anabolic';
      badge = blob.graduated
        ? 'COMPLEX MULTICELLULAR · PRODUCER'
        : blob.encapsulated
          ? 'ENCAPSULATED PRODUCER PROTOCELL'
          : 'PRODUCER BLOB · AUTOTROPH';
      title = `${blob.graduated ? 'Complex Producer Organism' : `Producer ${blob.encapsulated ? 'Protocell' : 'Blob'}`} (${blob.cytoplasmCount} ${cellWord} · ${blob.distinctSpecies} ${colorWord})`;
      statusDesc = `Calm cohesive ${blob.formula} ${blob.graduated ? 'organism' : 'colony'} (${encapTag}, ${strategyGeneStr}; coh=${blob.avgCoh.toFixed(2)}, solar=${Math.round(blob.avgSolarFrac * 100)}%) fixing C/H/O/S atoms across sunlight, currents & atmospheric chemistry.`;
    }

    const gradDetail = blob.graduated
      ? ' Graduated via lipid bilayer + post-consumption endosymbiosis to resist core crowding Oxygen dissolution.'
      : '';
    const landDetail = overLand
      ? ' Roaming calm current-free terrain to forage atmospheric chemicals at a modest desiccation cost.'
      : '';
    return {
      theme,
      badge,
      tempTag: surfTempTag,
      envChips: surfEnvChips,
      title,
      body: `${statusDesc}${gradDetail}${landDetail}${mineralNote}`,
      meta: `Atom #${bestIdx} (${sym0}) · Tissue #${bestBioSp} · ${habitatTag} · Policy: ${blob.mdpPolicyLabel} · ${encapTag} · ${strategyGeneStr} · Blob E=${blob.avgEnergy.toFixed(2)} · Motility ${blob.avgMotility.toFixed(2)}`,
      element: selectedElement,
      targetPhase: 1,
      targetIdx: bestIdx,
      targetX,
      targetY,
    };
  }

  return null;
}

export class TooltipManager {
  private el: HTMLElement | null;
  private lensEl: HTMLElement | null;
  private cfg: HoverInspectConfig;
  private mode: 'none' | 'menu' | 'canvas' = 'none';
  private hoverCanvas = false;
  private touchPinned = false;
  private pinnedTarget: TooltipTarget | null = null;
  private lensActive = false;
  private lensX = 0;
  private lensY = 0;
  private lensRadius = 64;
  private readonly lensMag = 2.0;
  private lensTheme: TooltipTheme = 'chem';
  private lensGeneration = 0;
  private gpuLensSynced = false;
  private offGlobeStreak = 0;
  private tipAnchorX = 0;
  private tipAnchorY = 0;
  private tipAnchorInit = false;
  private smoothBoxX = 0;
  private smoothBoxY = 0;
  private smoothBoxInit = false;
  private lastTickAt = 0;
  private lastInfo: TooltipContent | null = null;
  private hoverX = 0;
  private hoverY = 0;
  private cachedSnapshot: HoverSimSnapshot | null = null;
  private snapshotPromise: Promise<HoverSimSnapshot | null> | null = null;
  private lastSampleTime = 0;
  private inspectionRevision = 0;
  private hasPausedSnapshot = false;
  private lastTouchAt = -10000;
  private lastHtml = '';
  private activeTouchPointers = new Set<number>();
  private dismissUntilAllTouchUp = false;
  private pendingTap: { x: number; y: number } | null = null;
  private activeTouchTap: {
    id: number;
    startX: number;
    startY: number;
    startAt: number;
    moved: boolean;
    multi: boolean;
  } | null = null;

  constructor(cfg: HoverInspectConfig) {
    this.el = document.getElementById('hoverTooltip');
    this.lensEl = document.getElementById('particleLens');
    this.cfg = cfg;
    this.bindEvents();
  }

  private computeLensRadius(): number {
    return Math.round(
      Math.max(54, Math.min(72, Math.min(window.innerWidth, window.innerHeight) * 0.165)),
    );
  }

  public getLensState(): TooltipLensState {
    return {
      active: this.lensActive,
      x: this.lensX,
      y: this.lensY,
      radius: this.lensRadius,
      mag: this.lensMag,
    };
  }

  public getLensGpuConfig(): TooltipLensGpuConfig {
    const kind: 0 | 1 = !this.lensActive || !this.pinnedTarget ? 0 : 1;
    return {
      kind,
      idx: this.pinnedTarget?.idx ?? 0,
      radius: this.lensActive ? this.lensRadius : 0,
      mag: this.lensActive ? this.lensMag : 1.0,
      color: LENS_THEME_COLORS[this.lensTheme] ?? LENS_THEME_COLORS.chem,
      generation: this.lensGeneration,
    };
  }

  public syncGpuLensPos(cx: number, cy: number, visibleOnGlobe: boolean): void {
    if (!this.lensActive || !this.touchPinned) return;
    if (!visibleOnGlobe) {
      this.offGlobeStreak++;
      if (this.gpuLensSynced && this.offGlobeStreak >= 4) {
        this.hide();
      }
      return;
    }
    this.offGlobeStreak = 0;
    this.gpuLensSynced = true;
    this.lensX = cx;
    this.lensY = cy;
    this.hoverX = cx;
    this.hoverY = cy;
    if (!this.tipAnchorInit) {
      this.tipAnchorX = cx;
      this.tipAnchorY = cy;
      this.tipAnchorInit = true;
    }
    this.syncLensDom(this.lensTheme);
  }

  public isTouchPinned(): boolean {
    return this.touchPinned;
  }

  private ensureSnapshot(): Promise<HoverSimSnapshot | null> {
    if (this.snapshotPromise) return this.snapshotPromise;
    this.lastSampleTime = performance.now();
    const paused = this.cfg.inspectionPaused();
    if (paused && this.hasPausedSnapshot && this.cachedSnapshot)
      return Promise.resolve(this.cachedSnapshot);
    const revision = this.inspectionRevision;
    this.snapshotPromise = this.cfg
      .requestSnapshot()
      .then((snap) => {
        if (revision !== this.inspectionRevision) return null;
        if (snap) {
          this.hasPausedSnapshot = paused && this.cfg.inspectionPaused();
          this.cachedSnapshot = snap;
        }
        return snap;
      })
      .catch(() => null)
      .finally(() => {
        this.snapshotPromise = null;
      });
    return this.snapshotPromise;
  }

  private bindEvents(): void {
    const menuEl = document.querySelector<HTMLElement>('.tp-dfwv');
    if (menuEl) {
      menuEl.addEventListener('pointermove', (ev) => {
        if (ev.pointerType !== 'mouse') return;
        if (performance.now() - this.lastTouchAt < 700) return;
        if (!this.cfg.isEnabled()) {
          this.hide();
          return;
        }
        const t = ev.target;
        const item =
          t instanceof Element ? t.closest<HTMLElement>('[data-tip-title]') : null;
        if (!item) {
          if (this.mode === 'menu') this.hide();
          return;
        }
        this.hide();
        this.hoverCanvas = false;
        this.mode = 'menu';
        const title = item.dataset.tipTitle ?? '';
        const body = item.dataset.tipBody ?? '';
        this.renderCard(
          {
            theme: 'menu',
            badge: 'PARAMETER',
            title,
            body,
          },
          ev.clientX,
          ev.clientY,
          item,
          false,
        );
      });

      menuEl.addEventListener('pointerleave', () => {
        if (this.mode === 'menu') this.hide();
      });
    }

    // Touching anywhere while a particle tooltip/lens is active clears it first
    // before a subsequent touch may inspect another particle.
    window.addEventListener(
      'pointerdown',
      (ev) => {
        if (ev.pointerType === 'mouse') return;
        this.lastTouchAt = performance.now();
        this.activeTouchPointers.add(ev.pointerId);
        if (
          this.touchPinned ||
          this.lensActive ||
          this.mode !== 'none' ||
          this.pendingTap !== null
        ) {
          this.dismissUntilAllTouchUp = true;
          this.pendingTap = null;
          this.activeTouchTap = null;
          this.hide();
        }
      },
      { capture: true },
    );

    const endTouchPointer = (ev: PointerEvent) => {
      if (ev.pointerType === 'mouse') return;
      this.lastTouchAt = performance.now();
      this.activeTouchPointers.delete(ev.pointerId);
      if (this.activeTouchPointers.size === 0) {
        this.dismissUntilAllTouchUp = false;
      }
    };
    window.addEventListener('pointerup', endTouchPointer);
    window.addEventListener('pointercancel', endTouchPointer);

    const simCanvas = document.getElementById('canvas');
    if (simCanvas) {
      simCanvas.addEventListener('pointermove', (ev) => {
        if (ev.pointerType !== 'mouse') {
          if (
            this.activeTouchTap &&
            ev.pointerId === this.activeTouchTap.id &&
            Math.hypot(
              ev.clientX - this.activeTouchTap.startX,
              ev.clientY - this.activeTouchTap.startY,
            ) > 12
          ) {
            this.activeTouchTap.moved = true;
          }
          return;
        }
        if (performance.now() - this.lastTouchAt < 700 || this.touchPinned) return;
        if (!this.cfg.isEnabled() || ev.buttons !== 0) {
          if (this.mode === 'canvas') this.hide();
          this.hoverCanvas = false;
          return;
        }
        this.hoverCanvas = true;
        this.hoverX = ev.clientX;
        this.hoverY = ev.clientY;
        this.mode = 'canvas';
        this.refreshCanvasTooltip(ev.clientX, ev.clientY, false, false);
      });

      simCanvas.addEventListener('pointerleave', (ev) => {
        if (ev.pointerType !== 'mouse') return;
        this.hoverCanvas = false;
        if (this.mode === 'canvas' && !this.touchPinned) this.hide();
      });

      simCanvas.addEventListener('pointerdown', (ev) => {
        this.hoverCanvas = false;
        if (ev.pointerType !== 'mouse') {
          if (!this.cfg.isEnabled() || this.dismissUntilAllTouchUp) return;
          if (!this.activeTouchTap && ev.isPrimary) {
            this.activeTouchTap = {
              id: ev.pointerId,
              startX: ev.clientX,
              startY: ev.clientY,
              startAt: performance.now(),
              moved: false,
              multi: false,
            };
            if (!this.cachedSnapshot || performance.now() - this.lastSampleTime > 120) {
              void this.ensureSnapshot();
            }
          } else if (this.activeTouchTap && ev.pointerId !== this.activeTouchTap.id) {
            // Two-finger pinch/zoom gesture: do not trigger tap tooltip
            this.activeTouchTap.multi = true;
          }
          return;
        }
        if (this.mode === 'canvas' || this.touchPinned || this.lensActive) {
          this.hide();
        }
      });

      simCanvas.addEventListener('pointerup', (ev) => {
        if (ev.pointerType === 'mouse') return;
        if (this.dismissUntilAllTouchUp) {
          this.activeTouchTap = null;
          return;
        }
        if (!this.activeTouchTap || ev.pointerId !== this.activeTouchTap.id) return;
        const tapInfo = this.activeTouchTap;
        this.activeTouchTap = null;

        const dist = Math.hypot(
          ev.clientX - tapInfo.startX,
          ev.clientY - tapInfo.startY,
        );
        const dt = performance.now() - tapInfo.startAt;
        if (
          tapInfo.moved ||
          tapInfo.multi ||
          dist > 12 ||
          dt > 700 ||
          !this.cfg.isEnabled()
        ) {
          return;
        }

        const tapX = ev.clientX;
        const tapY = ev.clientY;
        const hitNow = this.cachedSnapshot
          ? analyzeReactionAt(tapX, tapY, this.cachedSnapshot, this.cfg, 38, true)
          : null;
        if (hitNow) {
          this.activateTouchInspection(hitNow, tapX, tapY);
          // Refresh asynchronously with latest GPU state
          void this.ensureSnapshot().then((snap) => {
            if (
              snap &&
              this.touchPinned &&
              this.mode === 'canvas' &&
              this.cfg.isEnabled()
            ) {
              this.refreshCanvasTooltip(this.lensX, this.lensY, true, true);
            }
          });
          return;
        }

        // Snapshot may still be initializing or slightly behind; inspect fresh GPU state
        this.pendingTap = { x: tapX, y: tapY };
        void this.ensureSnapshot().then((snap) => {
          const pending = this.pendingTap;
          this.pendingTap = null;
          if (
            !snap ||
            !pending ||
            this.dismissUntilAllTouchUp ||
            this.activeTouchTap ||
            !this.cfg.isEnabled()
          ) {
            return;
          }
          const freshHit = analyzeReactionAt(pending.x, pending.y, snap, this.cfg, 38, true);
          if (freshHit) {
            this.activateTouchInspection(freshHit, pending.x, pending.y);
          }
        });
      });

      simCanvas.addEventListener('pointercancel', (ev) => {
        if (this.activeTouchTap && ev.pointerId === this.activeTouchTap.id) {
          this.activeTouchTap = null;
        }
      });
    }

    window.addEventListener('pointerdown', (ev) => {
      if (ev.pointerType !== 'mouse') return;
      const t = ev.target;
      if (t instanceof Node && simCanvas?.contains(t)) return;
      if (this.mode === 'canvas' || this.touchPinned || this.lensActive) {
        this.hide();
      }
    });
  }

  private activateTouchInspection(info: TooltipContent, fallbackX: number, fallbackY: number): void {
    const cx = info.targetX ?? fallbackX;
    const cy = info.targetY ?? fallbackY;
    this.mode = 'canvas';
    this.touchPinned = true;
    this.pinnedTarget =
      info.targetIdx !== undefined ? { idx: info.targetIdx } : null;
    this.lensRadius = this.computeLensRadius();
    this.lensX = cx;
    this.lensY = cy;
    this.tipAnchorX = cx;
    this.tipAnchorY = cy;
    this.tipAnchorInit = true;
    this.lastTickAt = performance.now();
    this.lensTheme = info.theme;
    this.lastInfo = info;
    this.lensActive = true;
    this.gpuLensSynced = false;
    this.offGlobeStreak = 0;
    this.lensGeneration++;
    this.hoverX = cx;
    this.hoverY = cy;
    this.syncLensDom(info.theme);
    this.renderCard(info, cx, cy, null, true);
  }

  private syncLensDom(theme: TooltipTheme): void {
    const lensEl = this.lensEl;
    if (!lensEl) return;
    if (!this.lensActive) {
      lensEl.classList.remove('visible');
      lensEl.setAttribute('aria-hidden', 'true');
      return;
    }
    const d = Math.round(this.lensRadius * 2);
    lensEl.dataset.theme = theme;
    lensEl.style.width = `${d}px`;
    lensEl.style.height = `${d}px`;
    lensEl.style.transform = `translate3d(${this.lensX.toFixed(2)}px, ${this.lensY.toFixed(2)}px, 0) translate(-50%, -50%)`;
    lensEl.classList.add('visible');
    lensEl.setAttribute('aria-hidden', 'false');
  }

  public invalidateInspectionFrame(): void {
    this.inspectionRevision++;
    this.hasPausedSnapshot = false;
    this.cachedSnapshot = null;
    this.lastSampleTime = -Infinity;
  }

  public cancelInspection(): void {
    this.activeTouchTap = null;
    this.hoverCanvas = false;
    this.hide();
  }

  public syncEnabledState(): void {
    if (!this.cfg.isEnabled()) {
      this.hide();
    }
  }

  public tick(): void {
    if (!this.cfg.isEnabled()) {
      this.hide();
      return;
    }
    if ((!this.hoverCanvas && !this.touchPinned) || this.mode !== 'canvas') return;

    const isTouch = this.touchPinned;
    const now = performance.now();
    const dtSec = this.lastTickAt > 0 ? Math.min(0.08, Math.max(0.001, (now - this.lastTickAt) / 1000)) : 1 / 60;
    this.lastTickAt = now;

    // Smoothly track #hoverTooltip toward GPU lens position every frame via sub-pixel translate3d
    if (isTouch && this.lensActive && this.lastInfo) {
      const alpha = 1 - Math.exp(-dtSec * 12.0);
      this.tipAnchorX += (this.lensX - this.tipAnchorX) * alpha;
      this.tipAnchorY += (this.lensY - this.tipAnchorY) * alpha;
      this.renderCard(this.lastInfo, this.tipAnchorX, this.tipAnchorY, null, true, true);
    }

    const intervalMs = isTouch ? 350 : 200;
    if (!this.snapshotPromise && now - this.lastSampleTime >= intervalMs) {
      void this.ensureSnapshot().then((snap) => {
        if (
          snap &&
          (this.hoverCanvas || this.touchPinned) &&
          this.mode === 'canvas' &&
          this.cfg.isEnabled()
        ) {
          this.refreshCanvasTooltip(this.hoverX, this.hoverY, isTouch, isTouch);
        }
      });
    } else if (this.cachedSnapshot && this.hoverCanvas && !this.touchPinned) {
      this.refreshCanvasTooltip(this.hoverX, this.hoverY, false, false);
    }
  }

  public inspectAt(
    clientX: number,
    clientY: number,
    pickPx: number = 20,
    includeSingletons: boolean = false,
  ): TooltipContent | null {
    if (!this.cachedSnapshot || !this.cfg.isEnabled()) return null;
    return analyzeReactionAt(
      clientX,
      clientY,
      this.cachedSnapshot,
      this.cfg,
      pickPx,
      includeSingletons,
    );
  }

  public setSnapshot(snap: HoverSimSnapshot): void {
    this.cachedSnapshot = snap;
  }

  private refreshCanvasTooltip(
    clientX: number,
    clientY: number,
    isTouch: boolean = false,
    keepIfPinned: boolean = false,
  ): void {
    if (!this.cachedSnapshot) return;
    let info = analyzeReactionAt(
      clientX,
      clientY,
      this.cachedSnapshot,
      this.cfg,
      isTouch ? 38 : 20,
      isTouch || this.cfg.inspectionPaused(),
      isTouch ? this.pinnedTarget : null,
    );
    if (!info && isTouch && this.pinnedTarget && !this.gpuLensSynced) {
      info = analyzeReactionAt(
        this.lensX,
        this.lensY,
        this.cachedSnapshot,
        this.cfg,
        38,
        true,
        null,
      );
      if (info && info.targetIdx !== undefined) {
        this.pinnedTarget = { idx: info.targetIdx };
        this.lensGeneration++;
      }
    }
    if (!info) {
      if (!keepIfPinned && this.mode === 'canvas') {
        this.hide();
      }
      return;
    }

    if (isTouch) {
      this.lensRadius = this.computeLensRadius();
      this.lensTheme = info.theme;
      this.lastInfo = info;
      if (!this.gpuLensSynced) {
        this.lensX = info.targetX ?? clientX;
        this.lensY = info.targetY ?? clientY;
        this.hoverX = this.lensX;
        this.hoverY = this.lensY;
      }
      this.lensActive = true;
      if (info.targetIdx !== undefined) {
        if (!this.pinnedTarget || this.pinnedTarget.idx !== info.targetIdx) {
          this.pinnedTarget = { idx: info.targetIdx };
          this.lensGeneration++;
        }
      }
      this.syncLensDom(info.theme);
      this.renderCard(info, this.tipAnchorX, this.tipAnchorY, null, true);
    } else {
      this.renderCard(info, clientX, clientY, null, false);
    }
  }

  private renderCard(
    info: TooltipContent,
    clientX: number,
    clientY: number,
    menuRowEl: HTMLElement | null,
    isTouch: boolean = false,
    repositionOnly: boolean = false,
  ): void {
    const el = this.el;
    if (!el) return;

    if (!repositionOnly) {
      const elHtml = info.element
        ? `<div class="tip-element">` +
          `<span class="tip-el-swatch" style="background:rgb(${Math.round(Math.max(0, Math.min(1, info.element.color[0])) * 255)},${Math.round(Math.max(0, Math.min(1, info.element.color[1])) * 255)},${Math.round(Math.max(0, Math.min(1, info.element.color[2])) * 255)})"></span>` +
          `<span class="tip-el-label">Element:</span> ` +
          `<strong class="tip-el-name">${escapeHtml(info.element.name)} (${escapeHtml(info.element.symbol)})</strong>` +
          `<span class="tip-el-detail">Z=${info.element.atomicNumber} · ${escapeHtml(info.element.subshellLabel)} · ${escapeHtml(info.element.roleLabel)}</span>` +
          `</div>`
        : '';

      const envHtml =
        info.envChips && info.envChips.length > 0
          ? `<div class="tip-temp">${info.envChips
              .map((chip) => {
                const dot = chip.dotRgb
                  ? `<span class="tip-env-dot" style="background:rgb(${Math.round(Math.max(0, Math.min(255, chip.dotRgb[0])))},${Math.round(Math.max(0, Math.min(255, chip.dotRgb[1])))},${Math.round(Math.max(0, Math.min(255, chip.dotRgb[2])))})"></span>`
                  : '';
                return `<span class="tip-env-chip">${dot}<span>${escapeHtml(chip.label)}</span></span>`;
              })
              .join('')}</div>`
          : info.tempTag
            ? `<span class="tip-temp">${escapeHtml(info.tempTag)}</span>`
            : '';

      const html =
        `<div class="tip-head"><span class="tip-badge tip-badge-${info.theme}">${escapeHtml(info.badge)}</span>` +
        envHtml +
        `</div>` +
        `<div class="tip-title">${escapeHtml(info.title)}</div>` +
        elHtml +
        `<div class="tip-body">${escapeHtml(info.body)}</div>` +
        (info.meta ? `<div class="tip-meta">${escapeHtml(info.meta)}</div>` : '');

      if (html !== this.lastHtml) {
        el.innerHTML = html;
        this.lastHtml = html;
      }
      el.dataset.theme = info.theme;
      el.classList.add('visible');
      el.setAttribute('aria-hidden', 'false');
    }

    const pad = 10;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const rect = el.getBoundingClientRect();
    const tipW = rect.width || 250;
    const tipH = rect.height || 84;

    let left = clientX + 16;
    let top = clientY + 16;

    if (menuRowEl) {
      const paneRect =
        document.querySelector('.tp-dfwv')?.getBoundingClientRect() ??
        menuRowEl.getBoundingClientRect();
      const rowRect = menuRowEl.getBoundingClientRect();
      if (paneRect.left >= tipW + pad * 2) {
        left = paneRect.left - tipW - 12;
        top = rowRect.top - 4;
      } else {
        left = Math.max(pad, Math.min(vw - tipW - pad, paneRect.left));
        top =
          rowRect.bottom + tipH + pad < vh
            ? rowRect.bottom + 8
            : Math.max(pad, rowRect.top - tipH - 8);
      }
    } else if (isTouch) {
      left = clientX - tipW * 0.5;
      const clearance = (this.lensActive ? this.lensRadius : 0) + 18;
      const aboveTop = clientY - clearance - tipH;
      const belowTop = clientY + clearance;
      if (aboveTop >= pad) {
        top = aboveTop;
      } else if (belowTop + tipH <= vh - pad) {
        top = belowTop;
      } else {
        top = aboveTop > vh - pad - (belowTop + tipH) ? aboveTop : belowTop;
      }
    } else {
      if (left + tipW + pad > vw) {
        left = clientX - tipW - 16;
      }
      if (top + tipH + pad > vh) {
        top = clientY - tipH - 16;
      }
    }

    let clampedL = Math.max(pad, Math.min(vw - tipW - pad, left));
    let clampedT = Math.max(pad, Math.min(vh - tipH - pad, top));
    if (isTouch && !menuRowEl) {
      if (!this.smoothBoxInit) {
        this.smoothBoxX = clampedL;
        this.smoothBoxY = clampedT;
        this.smoothBoxInit = true;
      } else {
        const dx = clampedL - this.smoothBoxX;
        const dy = clampedT - this.smoothBoxY;
        const dist = Math.hypot(dx, dy);
        const maxStep = 2.4;
        if (dist > maxStep && dist > 1e-4) {
          const s = maxStep / dist;
          this.smoothBoxX += dx * s;
          this.smoothBoxY += dy * s;
        } else {
          this.smoothBoxX = clampedL;
          this.smoothBoxY = clampedT;
        }
        clampedL = this.smoothBoxX;
        clampedT = this.smoothBoxY;
      }
    } else {
      this.smoothBoxInit = false;
    }
    el.style.transform = `translate3d(${clampedL.toFixed(2)}px, ${clampedT.toFixed(2)}px, 0)`;
  }

  public hide(): void {
    const el = this.el;
    this.touchPinned = false;
    this.pinnedTarget = null;
    this.lensActive = false;
    this.gpuLensSynced = false;
    this.offGlobeStreak = 0;
    this.tipAnchorInit = false;
    this.smoothBoxInit = false;
    this.lastInfo = null;
    this.pendingTap = null;
    this.lensGeneration++;
    this.syncLensDom('chem');
    if (!el) return;
    if (this.mode !== 'none' || el.classList.contains('visible')) {
      el.classList.remove('visible');
      el.setAttribute('aria-hidden', 'true');
      this.mode = 'none';
    }
  }
}
