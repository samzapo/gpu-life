import type { SynthSpectrogramSnapshot } from './engine.ts';

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

export class SynthSpectrogramOverlay {
  private container: HTMLElement | null;
  private toggleBtn: HTMLButtonElement | null;
  private streamCheckbox: HTMLInputElement | null;
  private streamToggleCb: ((enabled: boolean) => void) | null = null;
  private tooltip: HTMLElement | null;
  private infoOpen = false;
  private lastTooltipMs = 0;
  private lastBoundsMs = -1000;
  private tipTopPx = 36;
  private lastSnap: SynthSpectrogramSnapshot | null = null;

  constructor(
    containerId = 'synth-spec',
    tooltipId = 'synth-spec-tooltip',
    toggleId = 'synth-spec-toggle',
    streamCheckboxId = 'synth-stream-toggle',
  ) {
    this.container = document.getElementById(containerId);
    this.tooltip = document.getElementById(tooltipId);
    this.toggleBtn = document.getElementById(toggleId) as HTMLButtonElement | null;
    this.streamCheckbox = document.getElementById(streamCheckboxId) as HTMLInputElement | null;

    if (this.streamCheckbox) {
      this.streamCheckbox.addEventListener('change', () => {
        this.streamToggleCb?.(this.streamCheckbox?.checked ?? false);
      });
      this.streamCheckbox.addEventListener('click', (ev) => ev.stopPropagation());
    }

    if (this.toggleBtn) {
      this.toggleBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this.setInfoOpen(!this.infoOpen);
      });
    }

    document.addEventListener('pointerdown', (ev) => {
      if (!this.infoOpen) return;
      const target = ev.target;
      if (target instanceof Node) {
        if (this.container?.contains(target) || this.tooltip?.contains(target)) return;
      }
      this.setInfoOpen(false);
    });

    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && this.infoOpen) {
        this.setInfoOpen(false);
      }
    });

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

  isInfoOpen(): boolean {
    return this.infoOpen;
  }

  setInfoOpen(open: boolean): void {
    this.infoOpen = open;
    this.syncToggleUi();
    if (this.tooltip) {
      this.tooltip.hidden = !open;
      this.tooltip.style.top = `${this.tipTopPx}px`;
      if (open && this.lastSnap) {
        this.lastTooltipMs = performance.now();
        this.updateTooltip(this.lastSnap);
      }
    }
  }

  private syncToggleUi(): void {
    if (!this.toggleBtn) return;
    this.toggleBtn.textContent = this.infoOpen ? '∿ ×' : '∿ ?';
    this.toggleBtn.setAttribute('aria-expanded', String(this.infoOpen));
    const label = this.infoOpen
      ? 'Hide synth motif source explanations'
      : 'Show synth motif source explanations';
    this.toggleBtn.setAttribute('aria-label', label);
    this.toggleBtn.title = label;
  }

  private syncClearanceBounds(nowMs: number): void {
    if (!this.container || nowMs - this.lastBoundsMs < 140) return;
    this.lastBoundsMs = nowMs;

    const winW = Math.max(240, window.innerWidth || 1280);
    const pillW = this.container.getBoundingClientRect().width || 68;
    const centerLeft = (winW - pillW) * 0.5;
    const centerRight = (winW + pillW) * 0.5;

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
    let shiftedForPanel = false;
    if (tpEl && getComputedStyle(tpEl).display !== 'none') {
      const tr = tpEl.getBoundingClientRect();
      if (tr.width > 0 && tr.height > 0) {
        if (tr.bottom <= 72) {
          checkOverlap(tpEl, 72);
        } else if (tr.left - 8 < centerRight && tr.bottom > safeTop) {
          const maxRight = Math.max(pillW + 10, tr.left - 8);
          const shiftedLeft = Math.max(10, Math.min(centerLeft, maxRight - pillW));
          this.container.style.left = `${Math.round(shiftedLeft)}px`;
          this.container.style.transform = 'none';
          shiftedForPanel = true;
        }
      }
    }
    if (!shiftedForPanel) {
      this.container.style.left = '50%';
      this.container.style.transform = 'translateX(-50%)';
    }

    this.container.style.top = `${safeTop}px`;
    this.tipTopPx = safeTop + 28;
    if (this.tooltip) {
      this.tooltip.style.top = `${this.tipTopPx}px`;
    }
  }

  render(snap: SynthSpectrogramSnapshot, nowMs: number): void {
    this.lastSnap = snap;
    if (!this.container) return;

    this.container.hidden = !snap.enabled;
    if (!snap.enabled) {
      if (this.infoOpen) this.setInfoOpen(false);
      return;
    }

    this.syncClearanceBounds(nowMs);

    if (this.infoOpen && this.tooltip) {
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
    titleSpan.textContent = 'GLOBE SYNTH MOTIF & SOURCES';
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
      'Narrow-band beat motion (~68 Hz kick/sub ocean swells & ~4.2 kHz transient wind ripples) + 16-band AudioWorklet harmonic resonator.';
    streamRow.append(streamTop, streamProv);
    list.appendChild(streamRow);

    this.tooltip.appendChild(list);
  }
}
