import type * as d3Geo from 'd3-geo';
import { lonLatToUnitVec } from './earth/projections';
import { getElementSymbol, type Element2D } from './periodicTable2D';
import type { ViewLayerMode } from './options';

export interface HoverSimSnapshot {
  elements: Element2D[];
  air: number[][];
  valence: number[][];
  bioReact: number[][];
  ocean: number[][];
  oceanState: number[][];
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

export interface TooltipContent {
  theme: TooltipTheme;
  badge: string;
  title: string;
  body: string;
  meta?: string;
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

const OCEAN_ROLE_NAMES = [
  'Autotroph (Producer)',
  'Consumer (Heterotroph)',
  'Halophile (Detoxifier)',
  'Decomposer (Symbiont)',
];

function unitDistChord2(a: number[], b: number[]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

export function analyzeReactionAt(
  clientX: number,
  clientY: number,
  snap: HoverSimSnapshot,
  cfg: HoverInspectConfig,
): TooltipContent | null {
  const proj = cfg.getProj();
  if (!proj?.invert) return null;
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

  const uCur = lonLatToUnitVec(ll[0], ll[1]);
  const viewLayer = cfg.getViewLayer();
  const showAir = viewLayer !== 'ocean';
  const showOcean = viewLayer !== 'air';
  const R = Math.max(cfg.getWorldSize(), 0.25);
  const airRMax = cfg.getAirRMax();
  const oceanRMax = cfg.getOceanRMax();
  const beta = cfg.getBeta();
  const catRadius = Math.max(oceanRMax * 1.35, 0.065);
  const scalePx = Math.max(cfg.getScale(), 120);

  // Pick radius in unit-sphere chord units matching pointer target size on screen & physical interaction radius
  const pxChord = 20 / scalePx;
  const airPickChord = Math.min(0.18, Math.max((airRMax / R) * 1.05, pxChord));
  const airPickChord2 = airPickChord * airPickChord;
  const oceanPickChord = Math.min(0.16, Math.max((oceanRMax / R) * 1.3, pxChord));
  const oceanPickChord2 = oceanPickChord * oceanPickChord;

  let bestAir = -1;
  let bestAirD2 = airPickChord2;
  const localAirIdx: number[] = [];
  const localPatchR = Math.max((airRMax * 3.0) / R, airPickChord * 2.4);
  const localPatchR2 = localPatchR * localPatchR;

  if (showAir && snap.air.length > 0) {
    for (let i = 0; i < snap.air.length; i++) {
      const p = snap.air[i];
      const d2 = unitDistChord2(p, uCur);
      if (d2 < localPatchR2) {
        localAirIdx.push(i);
        if (d2 < bestAirD2) {
          bestAirD2 = d2;
          bestAir = i;
        }
      }
    }
  }

  let bestOcean = -1;
  let bestOceanD2 = oceanPickChord2;
  const nearbyOcean: number[] = [];
  const catChord = catRadius / R;
  const catRef = bestAir >= 0 ? snap.air[bestAir] : uCur;
  const catRef2 = Math.max(catChord * catChord, oceanPickChord2);

  if (showOcean && snap.ocean.length > 0) {
    for (let i = 0; i < snap.ocean.length; i++) {
      const p = snap.ocean[i];
      const dc2 = unitDistChord2(p, uCur);
      if (dc2 < bestOceanD2) {
        bestOceanD2 = dc2;
        bestOcean = i;
      }
      if (unitDistChord2(p, catRef) < catRef2) {
        nearbyOcean.push(i);
      }
    }
  }

  if (bestAir >= 0) {
    const p0 = snap.air[bestAir];
    const els = snap.elements;
    const c0 = Math.max(0, Math.min(els.length - 1, Math.floor(p0[6])));
    const el0 = els[c0];
    const sym0 = getElementSymbol(el0.atomicNumber);
    const valVec = snap.valence[bestAir] ?? [0, 0, 0, 0];
    const atomLoad = valVec[3] ?? 0;

    // Build connected component & direct neighbors inside local patch
    const bondChord = (airRMax * 0.78) / R;
    const bondChord2 = bondChord * bondChord;
    const maxChord = airRMax / R;
    const maxChord2 = maxChord * maxChord;

    const compIds = [bestAir];
    const visited = new Set<number>([bestAir]);
    const directInteract: { idx: number; dist: number; vRelDot: number }[] = [];

    if (el0.valence > 0) {
      let head = 0;
      while (head < compIds.length && compIds.length < 18) {
        const u = compIds[head++];
        const pu = snap.air[u];
        const eu = els[Math.max(0, Math.min(els.length - 1, Math.floor(pu[6])))];
        if (!eu || eu.valence === 0) continue;

        for (const j of localAirIdx) {
          const pj = snap.air[j];
          const ej = els[Math.max(0, Math.min(els.length - 1, Math.floor(pj[6])))];
          if (!ej || ej.valence === 0) continue;
          const d2 = unitDistChord2(pu, pj);
          if (u === bestAir && j !== bestAir && d2 < maxChord2) {
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
            const valU = snap.valence[u]?.[3] ?? 0;
            const valJ = snap.valence[j]?.[3] ?? 0;
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
      const pi = snap.air[id];
      const ei = els[Math.max(0, Math.min(els.length - 1, Math.floor(pi[6])))];
      const s = getElementSymbol(ei.atomicNumber);
      formulaCounts.set(s, (formulaCounts.get(s) ?? 0) + 1);
      if (ei.role > 0) hasDonor = true;
      else if (ei.role < 0) hasAcceptor = true;
      else if (ei.valence > 0) hasCovalent = true;
    }

    const formula = formatFormula(formulaCounts);
    const bio = snap.bioReact[bestAir] ?? [0, 0, 0, 0];
    const [actScale, glow] = unpackActGlow(bio[3] ?? 0);
    const kickMag = Math.hypot(bio[0] ?? 0, bio[1] ?? 0, bio[2] ?? 0);
    const bioActive =
      cfg.isBioLinked() &&
      showOcean &&
      (kickMag > 0.02 || actScale < 0.985 || Math.abs(glow) > 0.06);

    const roleCounts = [0, 0, 0, 0];
    let avgE = 0;
    let avgVit = 0;
    let avgPoi = 0;
    for (const oi of nearbyOcean) {
      const op = snap.ocean[oi];
      const os = snap.oceanState[oi] ?? [0.65, 0, 0, 0];
      roleCounts[Math.floor(Math.max(0, op[6])) % 4]++;
      avgE += os[0];
      avgVit += os[1];
      avgPoi += os[2];
    }
    if (nearbyOcean.length > 0) {
      avgE /= nearbyOcean.length;
      avgVit /= nearbyOcean.length;
      avgPoi /= nearbyOcean.length;
    }

    // Check if an actual reaction, bonded complex, or bio-catalysis is occurring here
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
        body = `Trace ${sym0} stabilizing nearby marine enzymes (${nearbyOcean.length} organisms), calming thermal jitter and boosting halide detox (+vit ${(avgVit * 100).toFixed(0)}%).`;
      } else if (kind === 4) {
        if (roleCounts[2] > 0 && glow < -0.05) {
          theme = 'cofactor';
          badge = 'HALOPHILE DETOX';
          body = `Detoxifying halophiles catalyzing ${sym0} halide into benign salts; barrier lowered ${dropPct}%.`;
        } else {
          theme = 'toxic';
          badge = 'HALOGEN TOXIN';
          body = `Reactive ${sym0} stressing nearby Ocean Life (avg poison ${(avgPoi * 100).toFixed(0)}%, draining vitality).`;
        }
      } else if (kind === 2) {
        if (glow >= 0) {
          theme = 'anabolic';
          badge = 'CARBON FIXATION';
          body = `Autotrophs fixing ${radicalSuffix}: barrier cut ${dropPct}% and thermal noise damped to assemble organic chains.`;
        } else {
          theme = 'catabolic';
          badge = 'CARBON METABOLISM';
          body = `Heterotrophs cleaving ${radicalSuffix} carbon bonds, harvesting chemical energy (+ΔE) from food substrate.`;
        }
      } else if (kind === 1) {
        if (glow >= 0) {
          theme = 'anabolic';
          badge = 'HYDRIDE SYNTHESIS';
          body = `Autotrophs funneling ${sym0} fuel into lowered-barrier bond synthesis (-${dropPct}% activation).`;
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
            : `Marine symbionts guiding ${sym0} hydroxyl radicals toward stable H₂O equilibrium.`;
      }

      const meta = `Species nearby: ${OCEAN_ROLE_NAMES[dominantRoleIdx]} · Ocean avg E=${avgE.toFixed(2)} · Valence load ${atomLoad.toFixed(2)}/${el0.valence}`;
      return {
        theme,
        badge,
        title: `${badge === 'HALOGEN TOXIN' ? 'Toxic' : 'Catalyzed'} ${radicalSuffix}`,
        body,
        meta,
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
        const nb = snap.air[directInteract[0].idx];
        const eNb = els[Math.max(0, Math.min(els.length - 1, Math.floor(nb[6])))];
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
        meta: `Anchor: Z=${el0.atomicNumber} ${sym0} [${el0.outerElectrons}/${el0.outerCapacity}] · Nearby pairs: ${directInteract.length}`,
      };
    }
  }

  // Fallback check: hovering over active Ocean Life swarm/metabolism
  if (bestOcean >= 0) {
    const op = snap.ocean[bestOcean];
    const os = snap.oceanState[bestOcean] ?? [0.65, 0, 0, 0];
    const sp = Math.floor(Math.max(0, op[6]));
    const roleIdx = sp % 4;
    const energy = os[0];
    const vit = os[1];
    const poi = os[2];
    const flux = os[3];
    const hasActivity =
      nearbyOcean.length >= 2 ||
      Math.abs(flux) > 0.015 ||
      vit > 0.04 ||
      poi > 0.04;
    if (hasActivity) {
      let theme: TooltipTheme = 'ocean';
      let statusDesc = 'Foraging along ocean current filaments.';
      if (poi > 0.15) {
        theme = 'toxic';
        statusDesc =
          roleIdx === 2
            ? 'Halophile organism actively detoxifying local chloride/fluoride halides.'
            : 'Organism stressed by dissolved halogen toxins (reduced motility).';
      } else if (flux > 0.04) {
        theme = 'anabolic';
        statusDesc = 'Actively harvesting dissolved carbon & hydride chemical energy (+ΔE).';
      } else if (vit > 0.12) {
        theme = 'cofactor';
        statusDesc = 'Stabilized by trace Mg/Li/Be cofactors (enhanced toxin clearance).';
      } else if (energy < 0.25) {
        theme = 'catabolic';
        statusDesc = 'Low energy reserve — seeking atmospheric H/C/S nutrient pockets.';
      }

      return {
        theme,
        badge: OCEAN_ROLE_NAMES[roleIdx].toUpperCase(),
        title: `Ocean Species #${sp}`,
        body: statusDesc,
        meta: `Energy ${energy.toFixed(2)} · Vitamin ${(vit * 100).toFixed(0)}% · Poison ${(poi * 100).toFixed(0)}% · Cluster: ${nearbyOcean.length}`,
      };
    }
  }

  return null;
}

export class TooltipManager {
  private el: HTMLElement | null;
  private cfg: HoverInspectConfig;
  private mode: 'none' | 'menu' | 'canvas' = 'none';
  private hoverCanvas = false;
  private hoverX = 0;
  private hoverY = 0;
  private cachedSnapshot: HoverSimSnapshot | null = null;
  private snapshotInFlight = false;
  private lastSampleTime = 0;
  private lastHtml = '';

  constructor(cfg: HoverInspectConfig) {
    this.el = document.getElementById('hoverTooltip');
    this.cfg = cfg;
    this.bindEvents();
  }

  private bindEvents(): void {
    const menuEl = document.querySelector<HTMLElement>('.tp-dfwv');
    if (menuEl) {
      menuEl.addEventListener('pointermove', (ev) => {
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
        );
      });

      menuEl.addEventListener('pointerleave', () => {
        if (this.mode === 'menu') this.hide();
      });
    }

    const simCanvas = document.getElementById('canvas');
    if (simCanvas) {
      simCanvas.addEventListener('pointermove', (ev) => {
        if (!this.cfg.isEnabled() || ev.buttons !== 0) {
          if (this.mode === 'canvas') this.hide();
          this.hoverCanvas = false;
          return;
        }
        this.hoverCanvas = true;
        this.hoverX = ev.clientX;
        this.hoverY = ev.clientY;
        this.mode = 'canvas';
        this.refreshCanvasTooltip(ev.clientX, ev.clientY);
      });

      simCanvas.addEventListener('pointerleave', () => {
        this.hoverCanvas = false;
        if (this.mode === 'canvas') this.hide();
      });

      simCanvas.addEventListener('pointerdown', () => {
        this.hoverCanvas = false;
        if (this.mode === 'canvas') this.hide();
      });
    }
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
    if (!this.hoverCanvas || this.mode !== 'canvas') return;

    const now = performance.now();
    if (!this.snapshotInFlight && now - this.lastSampleTime >= 200) {
      this.snapshotInFlight = true;
      this.lastSampleTime = now;
      this.cfg
        .requestSnapshot()
        .then((snap) => {
          if (snap) {
            this.cachedSnapshot = snap;
            if (this.hoverCanvas && this.mode === 'canvas' && this.cfg.isEnabled()) {
              this.refreshCanvasTooltip(this.hoverX, this.hoverY);
            }
          }
        })
        .catch(() => {
          // Ignore snapshot errors during resize/reset
        })
        .finally(() => {
          this.snapshotInFlight = false;
        });
    } else if (this.cachedSnapshot) {
      this.refreshCanvasTooltip(this.hoverX, this.hoverY);
    }
  }

  public inspectAt(clientX: number, clientY: number): TooltipContent | null {
    if (!this.cachedSnapshot || !this.cfg.isEnabled()) return null;
    return analyzeReactionAt(clientX, clientY, this.cachedSnapshot, this.cfg);
  }

  public setSnapshot(snap: HoverSimSnapshot): void {
    this.cachedSnapshot = snap;
  }

  private refreshCanvasTooltip(clientX: number, clientY: number): void {
    if (!this.cachedSnapshot) return;
    const info = analyzeReactionAt(clientX, clientY, this.cachedSnapshot, this.cfg);
    if (!info) {
      if (this.mode === 'canvas') this.hide();
      return;
    }
    this.renderCard(info, clientX, clientY, null);
  }

  private renderCard(
    info: TooltipContent,
    clientX: number,
    clientY: number,
    menuRowEl: HTMLElement | null,
  ): void {
    const el = this.el;
    if (!el) return;

    const html =
      `<div class="tip-head"><span class="tip-badge tip-badge-${info.theme}">${escapeHtml(info.badge)}</span></div>` +
      `<div class="tip-title">${escapeHtml(info.title)}</div>` +
      `<div class="tip-body">${escapeHtml(info.body)}</div>` +
      (info.meta ? `<div class="tip-meta">${escapeHtml(info.meta)}</div>` : '');

    if (html !== this.lastHtml) {
      el.innerHTML = html;
      this.lastHtml = html;
    }
    el.dataset.theme = info.theme;
    el.classList.add('visible');
    el.setAttribute('aria-hidden', 'false');

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
    } else {
      if (left + tipW + pad > vw) {
        left = clientX - tipW - 16;
      }
      if (top + tipH + pad > vh) {
        top = clientY - tipH - 16;
      }
    }

    el.style.left = `${Math.max(pad, Math.min(vw - tipW - pad, Math.round(left)))}px`;
    el.style.top = `${Math.max(pad, Math.min(vh - tipH - pad, Math.round(top)))}px`;
  }

  public hide(): void {
    const el = this.el;
    if (!el) return;
    if (this.mode !== 'none' || el.classList.contains('visible')) {
      el.classList.remove('visible');
      el.setAttribute('aria-hidden', 'true');
      this.mode = 'none';
    }
  }
}
