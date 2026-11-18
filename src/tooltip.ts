import type * as d3Geo from 'd3-geo';
import { lonLatToUnitVec } from './earth/projections';
import { getElementName, getElementSymbol, type Element2D } from './periodicTable2D';
import type { ViewLayerMode } from './options';

export interface HoverSimSnapshot {
  elements: Element2D[];
  particles: number[][];
  valence: number[][];
  lifeState: number[][];
  bioForce: number[][];
  capturedAt: number;
}

export interface HoverInspectConfig {
  isEnabled: () => boolean;
  getProj: () => d3Geo.GeoProjection | null;
  getViewLayer: () => ViewLayerMode;
  isBioLinked: () => boolean;
  getAirRMax: () => number;
  getOceanRMax: () => number;
  getWorldSize: () => number;
  getBeta: () => number;
  getTemperature: () => number;
  getScale: () => number;
  requestSnapshot: () => Promise<HoverSimSnapshot | null>;
}

export type TooltipTheme =
  | 'menu'
  | 'anabolic'
  | 'catabolic'
  | 'toxic'
  | 'cofactor'
  | 'chem'
  | 'ocean';

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

export interface TooltipContent {
  theme: TooltipTheme;
  badge: string;
  title: string;
  body: string;
  meta?: string;
  element?: TooltipSelectedElement;
  targetPhase?: 0 | 1;
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

const OCEAN_ROLE_NAMES = [
  'Autotroph (Producer)',
  'Consumer (Heterotroph)',
  'Halophile (Detoxifier)',
  'Decomposer (Remineralizer)',
];

const OCEAN_ROLE_SHORT = [
  'Producer',
  'Consumer',
  'Halophile',
  'Decomposer',
];

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

function unpackLifeTuple(ls: number[] | undefined): [0 | 1, number] {
  const w = ls?.[3] ?? 0;
  if (w < 0.5) return [0, 0];
  return [1, Math.max(0, Math.round(w - 1.0))];
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
  let bestPhase: 0 | 1 = 0;
  let bestBioSp = 0;
  let bestScore = 1.0;
  let pinnedPt: [number, number] | null = null;

  const nParticles = snap.particles.length;
  if (pinnedTarget) {
    if (pinnedTarget.idx < 0 || pinnedTarget.idx >= nParticles) {
      return null;
    }
    const p = snap.particles[pinnedTarget.idx];
    const [phase, bioSp] = unpackLifeTuple(snap.lifeState[pinnedTarget.idx]);
    if ((phase === 0 && !showChem) || (phase === 1 && !showLife)) {
      return null;
    }
    pinnedPt = projectUnitVecToScreen(p, proj);
    if (!pinnedPt) return null;
    uCur = [p[0], p[1], p[2]];
    bestIdx = pinnedTarget.idx;
    bestPhase = phase;
    bestBioSp = bioSp;
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
      const [phase, bioSp] = unpackLifeTuple(snap.lifeState[i]);
      if ((phase === 0 && !showChem) || (phase === 1 && !showLife)) {
        continue;
      }
      const p = snap.particles[i];
      const d2 = unitDistChord2(p, uCur);
      const thr2 = phase === 0 ? chemPickChord2 : lifePickChord2;
      if (d2 < thr2) {
        const score = d2 / Math.max(thr2, 1e-12);
        if (score < bestScore) {
          bestScore = score;
          bestIdx = i;
          bestPhase = phase;
          bestBioSp = bioSp;
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
  const nearbyLife: number[] = [];

  for (let i = 0; i < nParticles; i++) {
    const [phase] = unpackLifeTuple(snap.lifeState[i]);
    if (phase === 0 && showChem) {
      if (unitDistChord2(snap.particles[i], centerRef) < localPatchR2) {
        localChemIdx.push(i);
      }
    } else if (phase === 1 && showLife) {
      if (unitDistChord2(snap.particles[i], centerRef) < catRef2) {
        nearbyLife.push(i);
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
    const bioActive =
      cfg.isBioLinked() &&
      showLife &&
      (kickMag > 0.02 || actScale < 0.985 || Math.abs(glow) > 0.06);

    const roleCounts = [0, 0, 0, 0];
    let avgE = 0;
    let avgVit = 0;
    let avgPoi = 0;
    for (const oi of nearbyLife) {
      const lsO = snap.lifeState[oi] ?? [0.65, 0, 0, 1];
      const [, oiSp] = unpackLifeTuple(lsO);
      roleCounts[oiSp % 4]++;
      avgE += lsO[0] ?? 0;
      avgVit += lsO[1] ?? 0;
      avgPoi += lsO[2] ?? 0;
    }
    if (nearbyLife.length > 0) {
      avgE /= nearbyLife.length;
      avgVit /= nearbyLife.length;
      avgPoi /= nearbyLife.length;
    }

    const hasOngoingChem = compIds.length >= 2 || directInteract.length > 0;
    if (bioActive) {
      const kind = bioKind(el0.atomicNumber, el0.valence);
      const dropPct = Math.round((1 - actScale) * 100);
      const dominantRoleIdx = roleCounts.indexOf(Math.max(...roleCounts));
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
        body = `Abiotic ${sym0} cofactor stabilizing nearby marine enzymes (${nearbyLife.length} organisms), calming thermal jitter and boosting halide detox (+vit ${(avgVit * 100).toFixed(0)}%).`;
      } else if (kind === 4) {
        if (roleCounts[2] > 0 && glow < -0.05) {
          theme = 'cofactor';
          badge = 'HALOPHILE DETOX';
          body = `Detoxifying halophiles catalyzing abiotic ${sym0} halide into benign salts; barrier lowered ${dropPct}%.`;
        } else {
          theme = 'toxic';
          badge = 'HALOGEN TOXIN';
          body = `Abiotic ${sym0} stressing nearby Ocean Life (avg poison ${(avgPoi * 100).toFixed(0)}%, draining vitality).`;
        }
      } else if (kind === 2) {
        if (glow >= 0) {
          theme = 'anabolic';
          badge = 'CARBON FIXATION';
          body = `Producers fixing biogenic ${radicalSuffix}: barrier cut ${dropPct}% while recruiting carbon toward living Producer biomass.`;
        } else {
          theme = 'catabolic';
          badge = 'CARBON METABOLISM';
          body = `Consumers cleaving ${radicalSuffix} carbon bonds, harvesting chemical energy (+ΔE) near living biomass.`;
        }
      } else if (kind === 1) {
        if (glow >= 0) {
          theme = 'anabolic';
          badge = 'HYDRIDE FIXATION';
          body = `Producers funneling biogenic ${sym0} fuel into lowered-barrier bond synthesis (-${dropPct}% activation).`;
        } else {
          theme = 'catabolic';
          badge = 'FUEL HARVEST';
          body = `Marine life harvesting ${sym0} chemical fuel: cleaving weak bonds with exothermic kinetic kick.`;
        }
      } else {
        theme = glow >= 0 ? 'anabolic' : 'catabolic';
        badge = glow < -0.05 && atomLoad >= 1.35 ? 'OXIDE STRIPPING' : 'RADICAL STABILIZATION';
        body =
          glow < -0.05 && atomLoad >= 1.35
            ? `Autotrophs stripping excess bound ${sym0} from saturated oxides to free reactive bonding sites.`
            : `Marine symbionts guiding ${sym0} radicals toward stable aqueous equilibrium.`;
      }

      const meta = `Species nearby: ${OCEAN_ROLE_NAMES[dominantRoleIdx]} · Biomass avg E=${avgE.toFixed(2)} · Valence load ${atomLoad.toFixed(2)}/${el0.valence}`;
      return {
        theme,
        badge,
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
      return {
        theme: 'chem',
        badge: inert ? 'NOBLE GAS' : 'FREE ATOM',
        title: `${selectedElement.name} (${sym0}) Atom`,
        body: inert
          ? `Inert ${sym0} (${el0.subshellLabel}) drifting on atmospheric wind currents; permanently abiotic (filled outer shell).`
          : `Unbonded ${sym0} (${el0.subshellLabel}) monomer with ${el0.valence} open valence slot${el0.valence === 1 ? '' : 's'}, drifting in Chemistry phase.`,
        meta: `Valence [${el0.outerElectrons}/${el0.outerCapacity}] · Nearby atoms: ${Math.max(0, localChemIdx.length - 1)}`,
        element: selectedElement,
        targetPhase: 0,
        targetIdx: bestIdx,
        targetX,
        targetY,
      };
    }

    return null;
  }

  // Living Biomass (phase === 1): conserved biogenic element currently fixed in a living guild
  const ls = snap.lifeState[bestIdx] ?? [0.65, 0, 0, 1];
  const energy = ls[0] ?? 0;
  const vit = ls[1] ?? 0;
  const poi = ls[2] ?? 0;
  const roleIdx = bestBioSp % 4;
  const hasActivity =
    nearbyLife.length >= 2 ||
    vit > 0.04 ||
    poi > 0.04 ||
    energy < 0.35 ||
    energy > 0.65;

  if (hasActivity || includeSingletons) {
    let theme: TooltipTheme = 'ocean';
    let statusDesc = `Conserved ${selectedElement.name} (${sym0}) cycling in ${OCEAN_ROLE_NAMES[roleIdx]} biomass.`;

    if (poi > 0.15) {
      theme = roleIdx === 2 ? 'cofactor' : 'toxic';
      statusDesc =
        roleIdx === 2
          ? `Halophile clearing dissolved halide toxins while protecting embodied ${sym0} biomass.`
          : `Biomass containing ${sym0} stressed by halogen toxins (halide injury increases decomposition susceptibility).`;
    } else if (vit > 0.12) {
      theme = 'cofactor';
      statusDesc = `Cofactor-stabilized ${OCEAN_ROLE_SHORT[roleIdx].toLowerCase()} holding conserved ${sym0} with elevated halide resilience.`;
    } else if (energy < 0.25) {
      theme = 'catabolic';
      statusDesc = `Low-energy ${sym0} biomass — vulnerable to Decomposer remineralization back into free ${sym0} atoms.`;
    } else if (roleIdx === 0) {
      theme = 'anabolic';
      statusDesc = `Autotroph harvesting solar & coastal upwelling energy to fix nearby C/H/O/S atoms alongside bound ${sym0}.`;
    } else if (roleIdx === 1) {
      theme = 'catabolic';
      statusDesc = `Consumer grazing Producer & Decomposer biomass (~65% assimilated, ~35% respired back to free chemistry).`;
    } else if (roleIdx === 2) {
      theme = 'ocean';
      statusDesc = `Halophile extremophile thriving inside saline halogen refuges around ${sym0} biopolymer.`;
    } else {
      theme = 'cofactor';
      statusDesc = `Decomposer remineralizing stressed/dense biomass (~80% returned as free atoms, ~20% Decomposer biomass).`;
    }

    return {
      theme,
      badge: `${OCEAN_ROLE_NAMES[roleIdx].toUpperCase()} · LIVING`,
      title: `${OCEAN_ROLE_SHORT[roleIdx]} #${bestBioSp} · Living ${selectedElement.name} (${sym0})`,
      body: statusDesc,
      meta: `Atom #${bestIdx} (${sym0}) · Energy ${energy.toFixed(2)} · Vitamin ${(vit * 100).toFixed(0)}% · Poison ${(poi * 100).toFixed(0)}% · Cluster: ${nearbyLife.length}`,
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
  private lastTickAt = 0;
  private lastInfo: TooltipContent | null = null;
  private hoverX = 0;
  private hoverY = 0;
  private cachedSnapshot: HoverSimSnapshot | null = null;
  private snapshotPromise: Promise<HoverSimSnapshot | null> | null = null;
  private lastSampleTime = 0;
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
    this.snapshotPromise = this.cfg
      .requestSnapshot()
      .then((snap) => {
        if (snap) {
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
          if (this.dismissUntilAllTouchUp) return;
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
      isTouch,
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

      const html =
        `<div class="tip-head"><span class="tip-badge tip-badge-${info.theme}">${escapeHtml(info.badge)}</span></div>` +
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
      const clearance = (this.lensActive ? this.lensRadius : 0) + 14;
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

    const clampedL = Math.max(pad, Math.min(vw - tipW - pad, left));
    const clampedT = Math.max(pad, Math.min(vh - tipH - pad, top));
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
