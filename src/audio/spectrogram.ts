import {
  SPECTROGRAM_BINS,
  type SynthSpectrogramSnapshot,
} from './engine.ts';

export interface SynthEffectSpec {
  id: 'bass' | 'acid' | 'chorus' | 'delay' | 'poly' | 'pulse';
  name: string;
  rgb: readonly [number, number, number];
  color: string;
  coreColor: string;
  glow: string;
  lineWidth: number;
}

export const SYNTH_EFFECT_SPECS: readonly SynthEffectSpec[] = [
  {
    id: 'pulse',
    name: 'Kick / Sidechain Pulse',
    rgb: [56, 189, 248],
    color: '#38bdf8',
    coreColor: '#e0f2fe',
    glow: 'rgba(56, 189, 248, 0.90)',
    lineWidth: 2.3,
  },
  {
    id: 'bass',
    name: 'Sub Bass + Filter Pluck',
    rgb: [255, 42, 133],
    color: '#ff2a85',
    coreColor: '#ffd6eb',
    glow: 'rgba(255, 42, 133, 0.96)',
    lineWidth: 2.9,
  },
  {
    id: 'acid',
    name: 'Acid Waveshaper Drive',
    rgb: [255, 176, 0],
    color: '#ffb000',
    coreColor: '#fff3c4',
    glow: 'rgba(255, 176, 0, 0.92)',
    lineWidth: 2.4,
  },
  {
    id: 'chorus',
    name: 'Stereo LFO Chorus',
    rgb: [0, 240, 255],
    color: '#00f0ff',
    coreColor: '#d9feff',
    glow: 'rgba(0, 240, 255, 0.95)',
    lineWidth: 2.6,
  },
  {
    id: 'delay',
    name: 'Hydro Dotted-8th Delay',
    rgb: [46, 255, 181],
    color: '#2effb5',
    coreColor: '#d8ffef',
    glow: 'rgba(46, 255, 181, 0.92)',
    lineWidth: 2.4,
  },
  {
    id: 'poly',
    name: 'Bio/Chem Motif & Arps',
    rgb: [192, 132, 252],
    color: '#c084fc',
    coreColor: '#f5e8ff',
    glow: 'rgba(192, 132, 252, 0.94)',
    lineWidth: 2.5,
  },
] as const;

const INTENSITY_BUCKETS = 5;

function buildStreamStyles(rgb: readonly [number, number, number]): string[] {
  const styles: string[] = [];
  for (let i = 0; i < INTENSITY_BUCKETS; i++) {
    const t = i / (INTENSITY_BUCKETS - 1);
    const scale = (120 + 100 * t) / 255;
    const alpha = (0.34 + 0.44 * t).toFixed(2);
    const r = Math.min(255, Math.round(rgb[0] * scale));
    const g = Math.min(255, Math.round(rgb[1] * scale));
    const b = Math.min(255, Math.round(rgb[2] * scale));
    styles.push(`rgba(${r}, ${g}, ${b}, ${alpha})`);
  }
  return styles;
}

const STREAM_BUCKET_STYLES: readonly string[][] = SYNTH_EFFECT_SPECS.map((s) => buildStreamStyles(s.rgb));

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
  private toggleBtn: HTMLButtonElement | null;
  private streamCheckbox: HTMLInputElement | null;
  private streamToggleCb: ((enabled: boolean) => void) | null = null;
  private tooltip: HTMLElement | null;
  private hovered = false;
  private minimized = false;
  private lastTooltipMs = 0;
  private lastBoundsMs = -1000;
  private tipTopPx = 108;
  private segPools: number[][] = Array.from({ length: INTENSITY_BUCKETS }, () => []);

  constructor(
    containerId = 'synth-spec',
    canvasId = 'synth-spec-canvas',
    tooltipId = 'synth-spec-tooltip',
    toggleId = 'synth-spec-toggle',
    streamCheckboxId = 'synth-stream-toggle',
  ) {
    this.container = document.getElementById(containerId);
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.tooltip = document.getElementById(tooltipId);
    this.toggleBtn = document.getElementById(toggleId) as HTMLButtonElement | null;
    this.streamCheckbox = document.getElementById(streamCheckboxId) as HTMLInputElement | null;

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

    if (this.streamCheckbox) {
      this.streamCheckbox.addEventListener('change', () => {
        this.streamToggleCb?.(this.streamCheckbox?.checked ?? false);
      });
      this.streamCheckbox.addEventListener('click', (ev) => ev.stopPropagation());
    }

    if (this.toggleBtn) {
      this.toggleBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this.setMinimized(!this.minimized);
      });
    }
    this.syncToggleUi();
    this.syncClearanceBounds(0);
    window.addEventListener('resize', () => {
      this.lastBoundsMs = -1000;
      this.syncClearanceBounds(performance.now());
    });
  }

  onStreamlinesToggle(cb: (enabled: boolean) => void): void {
    this.streamToggleCb = cb;
    cb(this.isStreamlinesEnabled());
  }

  isStreamlinesEnabled(): boolean {
    return this.streamCheckbox ? this.streamCheckbox.checked : true;
  }

  isMinimized(): boolean {
    return this.minimized;
  }

  setMinimized(min: boolean): void {
    this.minimized = min;
    this.syncToggleUi();
    this.lastBoundsMs = -1000;
  }

  private syncToggleUi(): void {
    if (!this.container || !this.toggleBtn) return;
    this.container.classList.toggle('minimized', this.minimized);
    this.toggleBtn.textContent = this.minimized ? '∿ +' : '−';
    this.toggleBtn.setAttribute('aria-expanded', String(!this.minimized));
    const label = this.minimized ? 'Expand synth spectrogram' : 'Minimize synth spectrogram';
    this.toggleBtn.setAttribute('aria-label', label);
    this.toggleBtn.title = label;
  }

  private syncClearanceBounds(nowMs: number): void {
    if (!this.container || nowMs - this.lastBoundsMs < 140) return;
    this.lastBoundsMs = nowMs;

    const winW = Math.max(240, window.innerWidth || 1280);
    let specW = this.minimized
      ? (this.container.getBoundingClientRect().width || 68)
      : Math.min(winW * 0.5, winW - 24);
    let centerLeft = (winW - specW) * 0.5;
    let centerRight = (winW + specW) * 0.5;

    let safeTop = 6;
    const checkOverlap = (el: Element | null, maxBottom = 140) => {
      if (!el) return;
      if ((el as HTMLElement).hidden || getComputedStyle(el).display === 'none') return;
      const r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0 || r.bottom > maxBottom) return;
      const overlapsX = r.right + 8 > centerLeft && r.left - 8 < centerRight;
      if (overlapsX && r.top < 140) {
        safeTop = Math.max(safeTop, Math.ceil(r.bottom) + 6);
      }
    };

    const switchesEl = document.getElementById('globe-switches');
    if (switchesEl) {
      for (const btn of switchesEl.querySelectorAll('.game-switch')) {
        checkOverlap(btn);
      }
      checkOverlap(document.getElementById('tour-caption'));
    }
    checkOverlap(document.getElementById('githubBtn'));

    const tpEl = document.querySelector('.tp-dfwv');
    if (tpEl && getComputedStyle(tpEl).display !== 'none') {
      const tr = tpEl.getBoundingClientRect();
      if (tr.width > 0 && tr.height > 0) {
        if (tr.bottom <= 72) {
          checkOverlap(tpEl, 72);
        } else if (tr.left - 8 < centerRight && tr.bottom > safeTop) {
          // Keep right edge strictly left of an expanded Display panel so toggles remain clickable
          const maxRight = Math.max(centerLeft + 110, tr.left - 8);
          specW = Math.min(specW, Math.max(96, maxRight - 10));
          const maxLeft = Math.max(10, maxRight - specW);
          centerLeft = Math.min(centerLeft, maxLeft);
          centerRight = centerLeft + specW;
          this.container.style.left = `${Math.round(centerLeft)}px`;
          this.container.style.transform = 'none';
          if (!this.minimized) this.container.style.width = `${Math.round(specW)}px`;
        }
      }
    }
    if (!tpEl || tpEl.getBoundingClientRect().bottom <= 72 || tpEl.getBoundingClientRect().left - 8 >= (winW + specW) * 0.5) {
      this.container.style.left = '50%';
      this.container.style.transform = 'translateX(-50%)';
      if (!this.minimized) this.container.style.width = '';
    }

    this.container.style.top = `${safeTop}px`;
    const activeHeight = this.minimized ? 22 : 96;
    this.tipTopPx = safeTop + activeHeight + 6;
    if (this.tooltip) {
      this.tooltip.style.top = `${this.tipTopPx}px`;
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

    this.syncClearanceBounds(nowMs);

    if (!this.minimized) {
      const cssW = Math.max(120, this.container.clientWidth || window.innerWidth * 0.5);
      const cssH = Math.max(64, this.container.clientHeight || 96);
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
      // Zero fade trail, zero background: strictly current-frame streamline wave segments
      ctx.clearRect(0, 0, cssW, cssH);

      const yBase = cssH - 6;
      const ampSpan = Math.max(28, cssH - 14);
      const kickBoost = snap.running ? snap.sidechainDepth : 0;
      const flowShift = snap.running ? (nowMs * 0.022) : 0;

      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.globalCompositeOperation = 'source-over';

      for (let sIdx = 0; sIdx < SYNTH_EFFECT_SPECS.length; sIdx++) {
        const spec = SYNTH_EFFECT_SPECS[sIdx];
        const curve = snap.curves[CURVE_INDEX_ORDER[sIdx]];
        const bucketStyles = STREAM_BUCKET_STYLES[sIdx];
        for (let b = 0; b < INTENSITY_BUCKETS; b++) this.segPools[b].length = 0;

        let prevX = 0;
        let prevAmp = Math.max(0, Math.min(1, (curve[0] ?? 0) * 0.72));
        let prevY = yBase - prevAmp * ampSpan;

        for (let i = 1; i < SPECTROGRAM_BINS; i++) {
          const amp = Math.max(0, Math.min(1, (curve[i] ?? 0) * 0.72));
          const x = (i / (SPECTROGRAM_BINS - 1)) * cssW;
          const y = yBase - amp * ampSpan;

          const segAmp = Math.max(prevAmp, amp);
          const dashPhase = (i * 0.72 + flowShift * (0.65 + sIdx * 0.14) + sIdx * 1.9) % 3.0;
          const drawGap = snap.running && segAmp < 0.08 && dashPhase > 2.25;

          if (!drawGap) {
            const slopeBoost = Math.min(0.24, Math.abs(amp - prevAmp) * 1.4);
            const energy = Math.min(1, segAmp * 0.92 + slopeBoost);
            const bIdx = Math.min(
              INTENSITY_BUCKETS - 1,
              Math.floor((snap.running ? energy : 0.25) * (INTENSITY_BUCKETS - 1)),
            );
            this.segPools[bIdx].push(prevX, prevY, x, y);
          }

          prevX = x;
          prevY = y;
          prevAmp = amp;
        }

        // Render muted color-preserving segments without additive blowout
        ctx.lineWidth = spec.lineWidth;
        for (let b = 0; b < INTENSITY_BUCKETS; b++) {
          const segs = this.segPools[b];
          if (segs.length === 0) continue;
          ctx.beginPath();
          ctx.strokeStyle = bucketStyles[b];
          if (b >= INTENSITY_BUCKETS - 2 && snap.running) {
            ctx.shadowColor = spec.glow;
            ctx.shadowBlur = b === INTENSITY_BUCKETS - 1 ? 5 + kickBoost * 3 : 2.5;
          } else {
            ctx.shadowBlur = 0;
          }
          for (let k = 0; k < segs.length; k += 4) {
            ctx.moveTo(segs[k], segs[k + 1]);
            ctx.lineTo(segs[k + 2], segs[k + 3]);
          }
          ctx.stroke();
        }
      }

      // Current-frame kick sidechain anchor ticks on edges
      if (snap.running && kickBoost > 0.1) {
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.85)';
        ctx.shadowColor = 'rgba(0, 240, 255, 0.95)';
        ctx.shadowBlur = 8;
        ctx.lineWidth = 1.2;
        const tickSpan = Math.min(cssW * 0.14, 10 + kickBoost * 24);
        ctx.beginPath();
        ctx.moveTo(0, yBase);
        ctx.lineTo(tickSpan, yBase);
        ctx.moveTo(cssW, yBase);
        ctx.lineTo(cssW - tickSpan, yBase);
        ctx.stroke();
      }

      ctx.restore();
    }

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
    titleSpan.textContent = 'GLOBE SYNTH SPECTROGRAM';
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

    const streamRow = document.createElement('div');
    streamRow.className = 'synth-tip-row';
    const streamTop = document.createElement('div');
    streamTop.className = 'synth-tip-row-top';
    const streamSwatch = document.createElement('i');
    streamSwatch.className = 'synth-tip-swatch';
    streamSwatch.style.background = 'linear-gradient(90deg, #ff2a85, #00f0ff)';
    streamSwatch.style.boxShadow = '0 0 8px rgba(0, 240, 255, 0.8)';
    const streamName = document.createElement('span');
    streamName.className = 'synth-tip-name';
    streamName.style.color = snap.streamlinesEnabled ? '#00f0ff' : '#94a3b8';
    streamName.textContent = 'Wind & Ocean Streamlines';
    const streamMetric = document.createElement('span');
    streamMetric.className = 'synth-tip-metric';
    streamMetric.textContent = snap.streamlinesEnabled
      ? `ON · ${Math.round(snap.streamlineEnergy * 100)}% energy`
      : 'OFF (unchecked)';
    streamTop.append(streamSwatch, streamName, streamMetric);
    const streamProv = document.createElement('div');
    streamProv.className = 'synth-tip-prov';
    streamProv.textContent =
      '16-band AudioWorklet harmonic resonator + live wave ripple modulation along all visible wind & ocean lines.';
    streamRow.append(streamTop, streamProv);
    list.appendChild(streamRow);

    this.tooltip.appendChild(list);
  }
}
