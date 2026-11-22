import {
  SPECTROGRAM_BINS,
  type SynthSpectrogramSnapshot,
} from './engine.ts';

export interface SynthEffectSpec {
  id: 'bass' | 'acid' | 'chorus' | 'delay' | 'poly' | 'pulse';
  name: string;
  color: string;
  coreColor: string;
  glow: string;
  lineWidth: number;
}

export const SYNTH_EFFECT_SPECS: readonly SynthEffectSpec[] = [
  {
    id: 'pulse',
    name: 'Kick / Sidechain Pulse',
    color: '#38bdf8',
    coreColor: '#e0f2fe',
    glow: 'rgba(56, 189, 248, 0.92)',
    lineWidth: 1.4,
  },
  {
    id: 'bass',
    name: 'Sub Bass + Filter Pluck',
    color: '#ff2a85',
    coreColor: '#ffd6eb',
    glow: 'rgba(255, 42, 133, 0.96)',
    lineWidth: 1.9,
  },
  {
    id: 'acid',
    name: 'Acid Waveshaper Drive',
    color: '#ffb000',
    coreColor: '#fff3c4',
    glow: 'rgba(255, 176, 0, 0.92)',
    lineWidth: 1.5,
  },
  {
    id: 'chorus',
    name: 'Stereo LFO Chorus',
    color: '#00f0ff',
    coreColor: '#d9feff',
    glow: 'rgba(0, 240, 255, 0.95)',
    lineWidth: 1.6,
  },
  {
    id: 'delay',
    name: 'Hydro Dotted-8th Delay',
    color: '#2effb5',
    coreColor: '#d8ffef',
    glow: 'rgba(46, 255, 181, 0.92)',
    lineWidth: 1.45,
  },
  {
    id: 'poly',
    name: 'Bio/Chem Motif & Arps',
    color: '#c084fc',
    coreColor: '#f5e8ff',
    glow: 'rgba(192, 132, 252, 0.94)',
    lineWidth: 1.65,
  },
] as const;

export interface EffectProvenanceEntry {
  spec: SynthEffectSpec;
  level: number;
  metric: string;
  provenance: string;
}

export function describeSynthEffects(snap: SynthSpectrogramSnapshot): EffectProvenanceEntry[] {
  const obs = snap.observation;
  const env = obs.env;
  const pct = (v: number) => `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;

  return [
    {
      spec: SYNTH_EFFECT_SPECS[1],
      level: snap.bassLevel,
      metric: `Cutoff ${Math.round(snap.bassCutoffHz)} Hz · Sub ${Math.round((0.55 + 0.25 * env.oceanFraction + 0.20 * env.nightFactor) * 100)}%`,
      provenance: `Root/transpose from biomass (${Math.round(obs.totalBioMass)}) & night (${pct(env.nightFactor)}); filter envelope from temp (${pct(env.tempNorm)}) & reaction glow.`,
    },
    {
      spec: SYNTH_EFFECT_SPECS[2],
      level: snap.acidLevel,
      metric: `Drive ${pct(snap.acidDrive)}`,
      provenance: `Tanh saturation & Q from halogen stress (${pct(obs.halogenStress)}), sulfide minerals (${pct(obs.mineralSulfideRatio)}), & predation (${pct(obs.consumerRatio)}).`,
    },
    {
      spec: SYNTH_EFFECT_SPECS[3],
      level: snap.chorusLevel,
      metric: `Spread ${pct(snap.chorusWidth)}`,
      provenance: `0.42 Hz stereo LFO chorus driven by complex life (${pct(obs.complexRatio)}), vapor chemistry (${pct(obs.chemVaporRatio)}), & ocean cover (${pct(env.oceanFraction)}).`,
    },
    {
      spec: SYNTH_EFFECT_SPECS[4],
      level: snap.delayLevel,
      metric: `Echo ${pct(snap.delayFeedback)}`,
      provenance: `Dotted-8th delay feedback & dispersion driven by humidity (RH ${pct(env.rhNorm)}), rain (${pct(env.precipNorm)}), & crystals (${pct(obs.mineralCrystalRatio)}).`,
    },
    {
      spec: SYNTH_EFFECT_SPECS[5],
      level: snap.polyLevel,
      metric: `Sp #${obs.dominantBioSp} · Chem ${pct(obs.chemBondActivity)}`,
      provenance: `Stereo L/R lanes from viewport particles; deterministic lineage motif from Sp #${obs.dominantBioSp} & chemical bond activity.`,
    },
    {
      spec: SYNTH_EFFECT_SPECS[0],
      level: snap.pulseLevel,
      metric: `Duck ${pct(snap.sidechainDepth)} · ${snap.bpm} BPM`,
      provenance: `Kick sidechain pump & dark sub-percussion driven by wind (${pct(env.windSpeedNorm)}), tectonic ridges (${pct(env.ridgeMask)}), & elevation.`,
    },
  ];
}

// Curve indices matching SYNTH_EFFECT_SPECS order above:
// 0: pulse -> curves[5], 1: bass -> curves[0], 2: acid -> curves[1],
// 3: chorus -> curves[2], 4: delay -> curves[3], 5: poly -> curves[4]
const CURVE_INDEX_ORDER = [5, 0, 1, 2, 3, 4] as const;

export class SynthSpectrogramOverlay {
  private container: HTMLElement | null;
  private canvas: HTMLCanvasElement | null;
  private ctx: CanvasRenderingContext2D | null;
  private tooltip: HTMLElement | null;
  private hovered = false;
  private lastTooltipMs = 0;
  private lastBoundsMs = -1000;
  private tipTopPx = 82;

  constructor(
    containerId = 'synth-spec',
    canvasId = 'synth-spec-canvas',
    tooltipId = 'synth-spec-tooltip',
  ) {
    this.container = document.getElementById(containerId);
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.tooltip = document.getElementById(tooltipId);

    if (this.container) {
      this.container.addEventListener('pointerenter', () => {
        this.hovered = true;
        if (this.tooltip) this.tooltip.hidden = false;
      });
      this.container.addEventListener('pointerleave', () => {
        this.hovered = false;
        if (this.tooltip) this.tooltip.hidden = true;
      });
    }
  }

  private syncClearanceBounds(nowMs: number): void {
    if (!this.container || nowMs - this.lastBoundsMs < 140) return;
    this.lastBoundsMs = nowMs;

    let safeTop = 30;
    const switchesEl = document.getElementById('globe-switches');
    if (switchesEl) {
      const buttons = switchesEl.querySelectorAll('.game-switch');
      for (const btn of buttons) {
        const r = btn.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
          safeTop = Math.max(safeTop, Math.ceil(r.bottom) + 6);
        }
      }
    }
    const githubEl = document.getElementById('githubBtn');
    if (githubEl && getComputedStyle(githubEl).display !== 'none') {
      const r = githubEl.getBoundingClientRect();
      if (r.height > 0 && r.bottom < 80) {
        safeTop = Math.max(safeTop, Math.ceil(r.bottom) + 6);
      }
    }

    let safeLeft = 10;
    const captionEl = document.getElementById('tour-caption');
    if (captionEl && !captionEl.hidden) {
      const cr = captionEl.getBoundingClientRect();
      if (cr.width > 0 && cr.bottom > safeTop && cr.top < safeTop + 50) {
        safeLeft = Math.max(safeLeft, Math.ceil(cr.right) + 10);
      }
    }

    let safeRight = 10;
    const tpEl = document.querySelector('.tp-dfwv') as HTMLElement | null;
    if (tpEl) {
      const tr = tpEl.getBoundingClientRect();
      if (tr.width > 0 && tr.bottom > safeTop) {
        safeRight = Math.max(safeRight, Math.ceil(window.innerWidth - tr.left) + 10);
      }
    }

    const specHeight = 48;
    if (safeLeft + safeRight + 80 >= window.innerWidth) {
      safeLeft = 10;
      safeRight = 10;
    }

    this.container.style.top = `${safeTop}px`;
    this.container.style.left = `${safeLeft}px`;
    this.container.style.right = `${safeRight}px`;
    this.container.style.height = `${specHeight}px`;
    this.tipTopPx = safeTop + specHeight + 6;
    if (this.tooltip) {
      this.tooltip.style.top = `${this.tipTopPx}px`;
    }
  }

  private traceSmoothCurve(
    ctx: CanvasRenderingContext2D,
    curve: Float32Array,
    cssW: number,
    yBase: number,
    ampSpan: number,
  ): { peakX: number; peakY: number; peakAmp: number } {
    ctx.beginPath();
    let prevX = 0;
    let peakAmp = curve[0] ?? 0;
    let prevY = yBase - Math.max(0, Math.min(1, peakAmp)) * ampSpan;
    let peakX = 0;
    let peakY = prevY;
    ctx.moveTo(prevX, prevY);

    for (let i = 1; i < SPECTROGRAM_BINS; i++) {
      const amp = Math.max(0, Math.min(1, curve[i] ?? 0));
      const x = (i / (SPECTROGRAM_BINS - 1)) * cssW;
      const y = yBase - amp * ampSpan;
      if (amp > peakAmp) {
        peakAmp = amp;
        peakX = x;
        peakY = y;
      }
      const midX = (prevX + x) * 0.5;
      const midY = (prevY + y) * 0.5;
      ctx.quadraticCurveTo(prevX, prevY, midX, midY);
      prevX = x;
      prevY = y;
    }
    ctx.lineTo(cssW, prevY);
    return { peakX, peakY, peakAmp };
  }

  render(snap: SynthSpectrogramSnapshot, nowMs: number): void {
    if (!this.container || !this.canvas || !this.ctx) return;

    this.container.hidden = !snap.enabled;
    if (!snap.enabled) {
      this.hovered = false;
      if (this.tooltip) this.tooltip.hidden = true;
      return;
    }

    this.syncClearanceBounds(nowMs);

    const cssW = Math.max(100, this.container.clientWidth || window.innerWidth - 20);
    const cssH = Math.max(34, this.container.clientHeight || 48);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const targetW = Math.round(cssW * dpr);
    const targetH = Math.round(cssH * dpr);

    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }

    const ctx = this.ctx;
    ctx.save();
    ctx.scale(dpr, dpr);
    // Strict current frame only: full transparent clear with zero temporal fade and no background
    ctx.clearRect(0, 0, cssW, cssH);

    const yBase = cssH - 4;
    const ampSpan = Math.max(12, cssH - 10);
    const kickBoost = snap.running ? snap.sidechainDepth : 0;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.globalCompositeOperation = 'lighter';

    for (let sIdx = 0; sIdx < SYNTH_EFFECT_SPECS.length; sIdx++) {
      const spec = SYNTH_EFFECT_SPECS[sIdx];
      const curve = snap.curves[CURVE_INDEX_ORDER[sIdx]];
      const crest = this.traceSmoothCurve(ctx, curve, cssW, yBase, ampSpan);

      // Pass 1: Wide outer neon bloom halo
      ctx.strokeStyle = spec.color;
      ctx.shadowColor = spec.glow;
      ctx.shadowBlur = snap.running ? 18 + kickBoost * 10 : 6;
      ctx.lineWidth = spec.lineWidth + 5.0;
      ctx.globalAlpha = snap.running ? 0.16 + kickBoost * 0.08 : 0.05;
      ctx.stroke();

      // Pass 2: Mid glowing neon aura
      ctx.shadowBlur = snap.running ? 9 + kickBoost * 4 : 4;
      ctx.lineWidth = spec.lineWidth + 1.6;
      ctx.globalAlpha = snap.running ? 0.46 : 0.14;
      ctx.stroke();

      // Pass 3: White-hot laser filament core
      ctx.strokeStyle = spec.coreColor;
      ctx.shadowColor = spec.color;
      ctx.shadowBlur = snap.running ? 4 : 2;
      ctx.lineWidth = Math.max(0.85, spec.lineWidth * 0.62);
      ctx.globalAlpha = snap.running ? 0.96 : 0.34;
      ctx.stroke();

      // Instantaneous spectral crest flare (strictly current frame peaks, no fade)
      if (snap.running && crest.peakAmp > 0.16) {
        const sparkR = 1.6 + crest.peakAmp * 1.9;
        ctx.fillStyle = spec.coreColor;
        ctx.shadowColor = spec.glow;
        ctx.shadowBlur = 12;
        ctx.globalAlpha = Math.min(1, 0.55 + crest.peakAmp * 0.45);
        ctx.beginPath();
        ctx.arc(crest.peakX, crest.peakY, sparkR, 0, Math.PI * 2);
        ctx.fill();

        // Crisp vertical harmonic spark whisker downward from crest toward horizon
        ctx.strokeStyle = spec.color;
        ctx.shadowBlur = 8;
        ctx.lineWidth = 1.0;
        ctx.globalAlpha = 0.45 * crest.peakAmp;
        ctx.beginPath();
        ctx.moveTo(crest.peakX, crest.peakY);
        ctx.lineTo(crest.peakX, Math.min(yBase, crest.peakY + (yBase - crest.peakY) * 0.45));
        ctx.stroke();
      }
    }

    // Current-frame kick edge pulse ticks along baseline edges
    if (snap.running && kickBoost > 0.08) {
      ctx.strokeStyle = '#00f0ff';
      ctx.shadowColor = 'rgba(0, 240, 255, 0.95)';
      ctx.shadowBlur = 12;
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = Math.min(0.85, kickBoost * 0.9);
      const tickSpan = Math.min(cssW * 0.12, 16 + kickBoost * 42);
      ctx.beginPath();
      ctx.moveTo(0, yBase);
      ctx.lineTo(tickSpan, yBase);
      ctx.moveTo(cssW, yBase);
      ctx.lineTo(cssW - tickSpan, yBase);
      ctx.stroke();
    }

    ctx.restore();

    if (this.hovered && this.tooltip) {
      this.tooltip.hidden = false;
      this.tooltip.style.top = `${this.tipTopPx}px`;
      if (nowMs - this.lastTooltipMs >= 90) {
        this.lastTooltipMs = nowMs;
        this.updateTooltip(snap);
      }
    }
  }

  private updateTooltip(snap: SynthSpectrogramSnapshot): void {
    if (!this.tooltip) return;
    const entries = describeSynthEffects(snap);
    const statusText = snap.running
      ? `${snap.bpm} BPM · ${snap.mode} · ${snap.section.toUpperCase()}`
      : `${snap.bpm} BPM · CLICK OR TAP GLOBE TO WAKE AUDIO`;

    this.tooltip.replaceChildren();

    const header = document.createElement('div');
    header.className = 'synth-tip-head';
    const titleSpan = document.createElement('strong');
    titleSpan.textContent = 'CYBERPUNK SYNTH SPECTROGRAM';
    const stateSpan = document.createElement('span');
    stateSpan.className = 'synth-tip-status';
    stateSpan.textContent = statusText;
    header.append(titleSpan, stateSpan);
    this.tooltip.appendChild(header);

    const list = document.createElement('div');
    list.className = 'synth-tip-list';
    for (const entry of entries) {
      const row = document.createElement('div');
      row.className = 'synth-tip-row';

      const top = document.createElement('div');
      top.className = 'synth-tip-row-top';

      const swatch = document.createElement('i');
      swatch.className = 'synth-tip-swatch';
      swatch.style.backgroundColor = entry.spec.color;
      swatch.style.boxShadow = `0 0 8px ${entry.spec.glow}`;

      const name = document.createElement('span');
      name.className = 'synth-tip-name';
      name.style.color = entry.spec.color;
      name.textContent = entry.spec.name;

      const metric = document.createElement('span');
      metric.className = 'synth-tip-metric';
      metric.textContent = entry.metric;

      top.append(swatch, name, metric);

      const prov = document.createElement('div');
      prov.className = 'synth-tip-prov';
      prov.textContent = entry.provenance;

      row.append(top, prov);
      list.appendChild(row);
    }
    this.tooltip.appendChild(list);
  }
}
