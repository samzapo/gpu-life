import {
  SPECTROGRAM_BINS,
  type SynthSpectrogramSnapshot,
} from './engine.ts';

export interface SynthEffectSpec {
  id: 'bass' | 'acid' | 'chorus' | 'delay' | 'poly' | 'pulse';
  name: string;
  color: string;
  glow: string;
  lineWidth: number;
}

export const SYNTH_EFFECT_SPECS: readonly SynthEffectSpec[] = [
  {
    id: 'pulse',
    name: 'Kick / Sidechain Pulse',
    color: '#38bdf8',
    glow: 'rgba(56, 189, 248, 0.85)',
    lineWidth: 1.35,
  },
  {
    id: 'bass',
    name: 'Sub Bass + Filter Pluck',
    color: '#ff2a85',
    glow: 'rgba(255, 42, 133, 0.92)',
    lineWidth: 2.0,
  },
  {
    id: 'acid',
    name: 'Acid Waveshaper Drive',
    color: '#ffb000',
    glow: 'rgba(255, 176, 0, 0.88)',
    lineWidth: 1.5,
  },
  {
    id: 'chorus',
    name: 'Stereo LFO Chorus',
    color: '#00f0ff',
    glow: 'rgba(0, 240, 255, 0.90)',
    lineWidth: 1.6,
  },
  {
    id: 'delay',
    name: 'Hydro Dotted-8th Delay',
    color: '#2effb5',
    glow: 'rgba(46, 255, 181, 0.88)',
    lineWidth: 1.45,
  },
  {
    id: 'poly',
    name: 'Bio/Chem Motif & Arps',
    color: '#c084fc',
    glow: 'rgba(192, 132, 252, 0.90)',
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

  render(snap: SynthSpectrogramSnapshot, nowMs: number): void {
    if (!this.container || !this.canvas || !this.ctx) return;

    this.container.hidden = !snap.enabled;
    if (!snap.enabled) {
      this.hovered = false;
      if (this.tooltip) this.tooltip.hidden = true;
      return;
    }

    const cssW = Math.max(120, this.container.clientWidth || window.innerWidth);
    const cssH = Math.max(36, this.container.clientHeight || 56);
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
    // Strict current frame only: full transparent clear with zero temporal fade/trail and no background
    ctx.clearRect(0, 0, cssW, cssH);

    const yBase = cssH - 4;
    const ampSpan = Math.max(12, cssH - 9);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.globalCompositeOperation = 'lighter';

    for (let sIdx = 0; sIdx < SYNTH_EFFECT_SPECS.length; sIdx++) {
      const spec = SYNTH_EFFECT_SPECS[sIdx];
      const curve = snap.curves[CURVE_INDEX_ORDER[sIdx]];

      ctx.beginPath();
      ctx.strokeStyle = spec.color;
      ctx.shadowColor = spec.glow;
      ctx.shadowBlur = snap.running ? 9 : 4;
      ctx.lineWidth = spec.lineWidth;
      ctx.globalAlpha = snap.running ? 0.92 : 0.28;

      let prevX = 0;
      let prevY = yBase - (curve[0] ?? 0) * ampSpan;
      ctx.moveTo(prevX, prevY);

      for (let i = 1; i < SPECTROGRAM_BINS; i++) {
        const x = (i / (SPECTROGRAM_BINS - 1)) * cssW;
        const y = yBase - Math.max(0, Math.min(1, curve[i] ?? 0)) * ampSpan;
        const midX = (prevX + x) * 0.5;
        const midY = (prevY + y) * 0.5;
        ctx.quadraticCurveTo(prevX, prevY, midX, midY);
        prevX = x;
        prevY = y;
      }
      ctx.lineTo(cssW, prevY);
      ctx.stroke();
    }

    ctx.restore();

    if (this.hovered && this.tooltip) {
      this.tooltip.hidden = false;
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
