import {
  buildCyberDrumBank,
  DRUM_BASE_NOTES,
  SYNTH_BPM,
  SYNTH_RATE,
} from './synthBank.ts';
import {
  createEmptyAudioObservation,
  type ScreenAudioObservation,
} from './observer.ts';
import { cyberSection, generateScoreEvents, type ScheduledAudioEvent } from './score.ts';
import { STREAMLINE_WORKLET_NAME, STREAMLINE_WORKLET_SOURCE } from './streamWorklet.ts';

interface AudioLaneNodes {
  input: GainNode;
  shaper: WaveShaperNode;
  panner: StereoPannerNode;
  chorusSend: GainNode;
  delaySend: GainNode;
  lastAcidBucket: number;
}

export const SPECTROGRAM_BINS = 72;

export interface SynthSpectrogramSnapshot {
  running: boolean;
  enabled: boolean;
  bpm: number;
  mode: 'CYBER' | 'PHRYGIAN';
  section: string;
  rms: number;
  bassLevel: number;
  acidLevel: number;
  chorusLevel: number;
  delayLevel: number;
  polyLevel: number;
  pulseLevel: number;
  bassCutoffHz: number;
  chorusWidth: number;
  delayFeedback: number;
  acidDrive: number;
  sidechainDepth: number;
  downbeatPulse: number;
  offbeatPulse: number;
  lowBassPhase: number;
  lowBassEnergy: number;
  streamlinesEnabled: boolean;
  streamlineEnergy: number;
  /** 6 distinct effect rows × SPECTROGRAM_BINS normalized [0..1] current spectral amplitudes */
  curves: [
    Float32Array,
    Float32Array,
    Float32Array,
    Float32Array,
    Float32Array,
    Float32Array,
  ];
  observation: ScreenAudioObservation;
}

export function createSpectrogramSnapshot(): SynthSpectrogramSnapshot {
  return {
    running: false,
    enabled: false,
    bpm: SYNTH_BPM,
    mode: 'CYBER',
    section: 'emerge',
    rms: 0,
    bassLevel: 0,
    acidLevel: 0,
    chorusLevel: 0,
    delayLevel: 0,
    polyLevel: 0,
    pulseLevel: 0,
    bassCutoffHz: 900,
    chorusWidth: 0.42,
    delayFeedback: 0.24,
    acidDrive: 0.18,
    sidechainDepth: 0,
    downbeatPulse: 0,
    offbeatPulse: 0,
    lowBassPhase: 0,
    lowBassEnergy: 0,
    streamlinesEnabled: true,
    streamlineEnergy: 0,
    curves: [
      new Float32Array(SPECTROGRAM_BINS),
      new Float32Array(SPECTROGRAM_BINS),
      new Float32Array(SPECTROGRAM_BINS),
      new Float32Array(SPECTROGRAM_BINS),
      new Float32Array(SPECTROGRAM_BINS),
      new Float32Array(SPECTROGRAM_BINS),
    ],
    observation: createEmptyAudioObservation(),
  };
}

const MAX_VOICES = 48;
const LOOKAHEAD_SEC = 0.28;
const NUM_LANES = 5;

function makeDriveCurve(amount: number): Float32Array<ArrayBuffer> {
  const curve = new Float32Array(512);
  const k = 1 + amount * 3.0;
  const norm = Math.tanh(k) || 1;
  for (let i = 0; i < curve.length; i++) {
    const x = (i / (curve.length - 1)) * 2 - 1;
    curve[i] = Math.tanh(x * k) / norm;
  }
  return curve;
}

function midiToHz(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function makeInstantAnalyser(ctx: AudioContext): { node: AnalyserNode; bins: Uint8Array<ArrayBuffer> } {
  const node = ctx.createAnalyser();
  node.fftSize = 256;
  // Near-zero smoothing so display reflects strictly the instantaneous current frame (no fade trails)
  node.smoothingTimeConstant = 0.06;
  node.minDecibels = -86;
  node.maxDecibels = -12;
  return { node, bins: new Uint8Array(node.frequencyBinCount) };
}

function sampleLogBins(
  src: Uint8Array,
  sampleRate: number,
  out: Float32Array,
  gainMul: number,
): number {
  const nyquist = Math.max(8000, sampleRate * 0.5);
  const binHz = nyquist / Math.max(1, src.length);
  const minHz = 36;
  const maxHz = Math.min(9200, nyquist * 0.85);
  const ratio = maxHz / minHz;
  let peak = 0;

  for (let i = 0; i < SPECTROGRAM_BINS; i++) {
    const t = i / (SPECTROGRAM_BINS - 1);
    const freq = minHz * Math.pow(ratio, t);
    const exactBin = freq / binHz;
    const b0 = Math.max(0, Math.min(src.length - 1, Math.floor(exactBin)));
    const b1 = Math.max(0, Math.min(src.length - 1, b0 + 1));
    const frac = Math.max(0, Math.min(1, exactBin - b0));
    const v = ((src[b0] * (1 - frac) + src[b1] * frac) / 255) * gainMul;
    const clamped = Math.max(0, Math.min(1, v));
    out[i] = clamped;
    if (clamped > peak) peak = clamped;
  }
  return peak;
}

export class GlobeCyberSynth {
  private ctx: AudioContext | null = null;
  private drumBank: Map<number, AudioBuffer> = new Map();
  private masterGain: GainNode | null = null;
  private duckGain: GainNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private delayNode: DelayNode | null = null;
  private delayFeedback: GainNode | null = null;
  private delayFilter: BiquadFilterNode | null = null;
  private chorusL: DelayNode | null = null;
  private chorusR: DelayNode | null = null;
  private chorusMix: GainNode | null = null;
  private polyBus: GainNode | null = null;

  private analyser: AnalyserNode | null = null;
  private masterBins: Uint8Array<ArrayBuffer> = new Uint8Array(128);
  private bassAnalyser: AnalyserNode | null = null;
  private bassBins: Uint8Array<ArrayBuffer> = new Uint8Array(128);
  private polyAnalyser: AnalyserNode | null = null;
  private polyBins: Uint8Array<ArrayBuffer> = new Uint8Array(128);
  private chorusAnalyser: AnalyserNode | null = null;
  private chorusBins: Uint8Array<ArrayBuffer> = new Uint8Array(128);
  private delayAnalyser: AnalyserNode | null = null;
  private delayBins: Uint8Array<ArrayBuffer> = new Uint8Array(128);
  private rmsScratch = new Float32Array(256);

  private lanes: AudioLaneNodes[] = [];
  private activeVoices = 0;
  private originTime = 0;
  private frameCursor = 0;
  private lastScheduleMs = -1000;
  private obs: ScreenAudioObservation = createEmptyAudioObservation();

  private enabled = false;
  private volume = 0.65;
  private unlocked = false;
  private armedOnGesture = false;

  private curBassCutoffHz = 900;
  private curChorusWidth = 0.42;
  private curDelayFeedback = 0.24;
  private curAcidDrive = 0.18;

  private streamlinesEnabled = true;
  private streamWorkletNode: AudioWorkletNode | null = null;
  private streamWorkletLoading = false;
  private streamBands16 = new Float32Array(16);
  private streamFreqs16 = new Float32Array(16);
  private streamEnergyAvg = 0;
  private lastStreamSendMs = -1000;

  isEnabled(): boolean {
    return this.enabled;
  }

  isStreamlinesEnabled(): boolean {
    return this.streamlinesEnabled;
  }

  setStreamlinesEnabled(on: boolean): void {
    this.streamlinesEnabled = on;
    if (this.streamWorkletNode) {
      this.streamWorkletNode.port.postMessage({
        masterAmp: this.streamlinesEnabled && this.isRunning() ? 1 : 0,
      });
    }
  }

  isRunning(): boolean {
    return Boolean(this.enabled && this.unlocked && this.ctx && this.ctx.state === 'running');
  }

  getVolume(): number {
    return this.volume;
  }

  getState(): AudioContextState | 'uninitialized' {
    return this.ctx ? this.ctx.state : 'uninitialized';
  }

  private initStreamWorklet(ctx: AudioContext): void {
    if (this.streamWorkletLoading || this.streamWorkletNode || !ctx.audioWorklet || typeof AudioWorkletNode === 'undefined') {
      return;
    }
    this.streamWorkletLoading = true;
    const blob = new Blob([STREAMLINE_WORKLET_SOURCE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    void ctx.audioWorklet
      .addModule(url)
      .then(() => {
        if (this.ctx !== ctx) return;
        const node = new AudioWorkletNode(ctx, STREAMLINE_WORKLET_NAME, {
          numberOfInputs: 0,
          numberOfOutputs: 1,
          outputChannelCount: [2],
        });
        const workletGain = ctx.createGain();
        workletGain.gain.value = 0.28;
        node.connect(workletGain);
        if (this.polyBus) workletGain.connect(this.polyBus);
        if (this.chorusMix) {
          const cSend = ctx.createGain();
          cSend.gain.value = 0.18;
          workletGain.connect(cSend);
          cSend.connect(this.chorusMix);
        }
        if (this.duckGain) workletGain.connect(this.duckGain);
        this.streamWorkletNode = node;
        node.port.postMessage({
          masterAmp: this.streamlinesEnabled && this.isRunning() ? 1 : 0,
          drive: this.curAcidDrive,
        });
      })
      .catch(() => {
        // Gracefully degrade if AudioWorklet is unavailable
      })
      .finally(() => {
        this.streamWorkletLoading = false;
        URL.revokeObjectURL(url);
      });
  }

  updateStreamlineTelemetry(nowMs: number, windBands?: Float32Array, oceanBands?: Float32Array): void {
    if (!this.streamlinesEnabled || !this.isRunning()) {
      if (this.streamWorkletNode && nowMs - this.lastStreamSendMs >= 120) {
        this.lastStreamSendMs = nowMs;
        this.streamWorkletNode.port.postMessage({ masterAmp: 0 });
      }
      this.streamEnergyAvg = 0;
      return;
    }

    let sum = 0;
    for (let i = 0; i < 8; i++) {
      const w = windBands ? Math.max(0, Math.min(1, windBands[i] || 0)) : 0;
      const o = oceanBands ? Math.max(0, Math.min(1, oceanBands[i] || 0)) : 0;
      this.streamBands16[i] = w;
      this.streamBands16[8 + i] = o;
      sum += w + o;
    }
    this.streamEnergyAvg = Math.min(1, sum / 16);

    if (!this.streamWorkletNode || nowMs - this.lastStreamSendMs < 65) return;
    this.lastStreamSendMs = nowMs;

    const obs = this.obs;
    const env = obs.env;
    const darkBias = obs.halogenStress * 0.45 + env.nightFactor * 0.45 + obs.mineralSulfideRatio * 0.35;
    const usePhrygian = darkBias > 0.36;
    const bar = Math.floor(this.frameCursor / ((SYNTH_RATE * 60) / SYNTH_BPM / 4 * 16));
    const prog = usePhrygian ? [0, 5, 1, 8] : [0, 5, 8, 10];
    const rootMidi = 50 + prog[Math.floor(bar / 2) % prog.length] + (env.nightFactor > 0.55 ? -12 : 0);

    const windDegrees = usePhrygian ? [0, 1, 7, 10, 12, 13, 19, 22] : [0, 3, 7, 10, 12, 15, 19, 22];
    const oceanDegrees = usePhrygian ? [-12, -5, -11, 0, 1, 7, 8, 12] : [-12, -5, 0, 3, 7, 10, 12, 15];
    for (let i = 0; i < 8; i++) {
      this.streamFreqs16[i] = midiToHz(Math.max(36, Math.min(80, rootMidi + windDegrees[i])));
      this.streamFreqs16[8 + i] = midiToHz(Math.max(28, Math.min(72, rootMidi + oceanDegrees[i])));
    }

    this.streamWorkletNode.port.postMessage({
      bands: this.streamBands16,
      freqs: this.streamFreqs16,
      drive: this.curAcidDrive,
      masterAmp: 1,
    });
  }

  private syncMasterGain(): void {
    if (!this.masterGain || !this.ctx) return;
    const active = this.enabled && this.ctx.state === 'running';
    const target = active ? this.volume * 0.85 : 0;
    this.masterGain.gain.setTargetAtTime(target, this.ctx.currentTime, 0.02);
    this.streamWorkletNode?.port.postMessage({
      masterAmp: active && this.unlocked && this.streamlinesEnabled ? 1 : 0,
    });
  }

  setVolume(vol: number): void {
    this.volume = Math.max(0, Math.min(1, Number.isFinite(vol) ? vol : 0.65));
    this.syncMasterGain();
  }

  setEnabled(on: boolean, fromUserGesture = false): void {
    const wasEnabled = this.enabled;
    this.enabled = on;
    if (!on) {
      this.armedOnGesture = false;
      this.syncMasterGain();
      return;
    }
    this.armedOnGesture = true;
    if (!wasEnabled && this.ctx?.state === 'running') {
      this.resetClock();
    }
    if (fromUserGesture) {
      void this.unlock();
    } else if (this.ctx?.state === 'running') {
      this.unlocked = true;
      this.syncMasterGain();
    }
  }

  setObservation(obs: ScreenAudioObservation): void {
    this.obs = obs;
  }

  async unlock(): Promise<void> {
    if (this.ctx && this.ctx.state === 'running') {
      this.unlocked = true;
      this.syncMasterGain();
      return;
    }
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    if (!this.ctx) {
      const ctx = new Ctor({ latencyHint: 'interactive' });
      this.ctx = ctx;
      this.buildGraph(ctx);
      this.drumBank = buildCyberDrumBank(ctx);
      this.initStreamWorklet(ctx);
      this.originTime = ctx.currentTime;
      this.frameCursor = 0;
      ctx.addEventListener('statechange', () => {
        if (this.ctx === ctx && ctx.state === 'running') {
          this.unlocked = true;
          this.originTime = ctx.currentTime;
          this.frameCursor = 0;
          this.syncMasterGain();
        }
      });
    }
    if (this.ctx.state === 'suspended') {
      try {
        await this.ctx.resume();
      } catch {
        // Ignore until next user gesture
      }
    }
    if (this.ctx.state === 'running') {
      this.unlocked = true;
      this.originTime = this.ctx.currentTime;
      this.frameCursor = 0;
      this.syncMasterGain();
    }
  }

  private buildGraph(ctx: AudioContext): void {
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0;

    this.duckGain = ctx.createGain();
    this.duckGain.gain.value = 1;

    this.polyBus = ctx.createGain();
    this.polyBus.gain.value = 1;

    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -13;
    this.compressor.knee.value = 10;
    this.compressor.ratio.value = 4.5;
    this.compressor.attack.value = 0.004;
    this.compressor.release.value = 0.13;

    const mTap = makeInstantAnalyser(ctx);
    this.analyser = mTap.node;
    this.masterBins = mTap.bins;

    const bTap = makeInstantAnalyser(ctx);
    this.bassAnalyser = bTap.node;
    this.bassBins = bTap.bins;

    const pTap = makeInstantAnalyser(ctx);
    this.polyAnalyser = pTap.node;
    this.polyBins = pTap.bins;

    const cTap = makeInstantAnalyser(ctx);
    this.chorusAnalyser = cTap.node;
    this.chorusBins = cTap.bins;

    const dTap = makeInstantAnalyser(ctx);
    this.delayAnalyser = dTap.node;
    this.delayBins = dTap.bins;

    // Tempo-synced dotted-8th stereo delay
    this.delayNode = ctx.createDelay(2.0);
    this.delayNode.delayTime.value = (60 / SYNTH_BPM) * 0.75;
    this.delayFeedback = ctx.createGain();
    this.delayFeedback.gain.value = 0.24;
    this.delayFilter = ctx.createBiquadFilter();
    this.delayFilter.type = 'lowpass';
    this.delayFilter.frequency.value = 1950;

    this.delayNode.connect(this.delayFilter);
    this.delayFilter.connect(this.delayFeedback);
    this.delayFeedback.connect(this.delayNode);
    this.delayFilter.connect(this.duckGain);
    this.delayFilter.connect(this.delayAnalyser);

    // Wide stereo modulated analog Chorus ("makes everything sound HUGE")
    this.chorusMix = ctx.createGain();
    this.chorusMix.gain.value = 0.42;
    this.chorusL = ctx.createDelay(0.1);
    this.chorusR = ctx.createDelay(0.1);
    this.chorusL.delayTime.value = 0.015;
    this.chorusR.delayTime.value = 0.022;

    const panL = ctx.createStereoPanner();
    const panR = ctx.createStereoPanner();
    panL.pan.value = -0.82;
    panR.pan.value = 0.82;

    const lfo = ctx.createOscillator();
    lfo.type = 'sine';
    lfo.frequency.value = 0.42;
    const lfoDepthL = ctx.createGain();
    const lfoDepthR = ctx.createGain();
    lfoDepthL.gain.value = 0.0032;
    lfoDepthR.gain.value = -0.0032;
    lfo.connect(lfoDepthL);
    lfo.connect(lfoDepthR);
    lfoDepthL.connect(this.chorusL.delayTime);
    lfoDepthR.connect(this.chorusR.delayTime);
    lfo.start();

    this.chorusMix.connect(this.chorusL);
    this.chorusMix.connect(this.chorusR);
    this.chorusMix.connect(this.chorusAnalyser);
    this.chorusL.connect(panL);
    this.chorusR.connect(panR);
    panL.connect(this.duckGain);
    panR.connect(this.duckGain);

    this.polyBus.connect(this.polyAnalyser);

    this.duckGain.connect(this.masterGain);
    this.masterGain.connect(this.compressor);
    this.compressor.connect(this.analyser);
    this.analyser.connect(ctx.destination);

    this.lanes = [];
    for (let i = 0; i < NUM_LANES; i++) {
      const input = ctx.createGain();
      const shaper = ctx.createWaveShaper();
      shaper.curve = makeDriveCurve(0.18);

      const panner = ctx.createStereoPanner();
      panner.pan.value = i < 4 ? this.obs.lanes[i].pan : 0;

      const chorusSend = ctx.createGain();
      chorusSend.gain.value = i === 4 ? 0.28 : 0.45;

      const delaySend = ctx.createGain();
      delaySend.gain.value = i === 4 ? 0.05 : 0.20;

      input.connect(shaper);
      shaper.connect(panner);
      panner.connect(this.duckGain);
      panner.connect(chorusSend);
      chorusSend.connect(this.chorusMix);
      panner.connect(delaySend);
      delaySend.connect(this.delayNode);

      if (i === 4) {
        panner.connect(this.bassAnalyser);
      } else {
        panner.connect(this.polyBus);
      }

      this.lanes.push({
        input,
        shaper,
        panner,
        chorusSend,
        delaySend,
        lastAcidBucket: 15,
      });
    }
  }

  resetClock(): void {
    if (!this.ctx) return;
    this.originTime = this.ctx.currentTime;
    this.frameCursor = 0;
  }

  tick(nowMs: number): number {
    if (!this.enabled || !this.unlocked || !this.ctx || this.ctx.state !== 'running') {
      if (this.armedOnGesture && this.enabled && this.ctx?.state === 'running') {
        this.unlocked = true;
        this.syncMasterGain();
      } else {
        return 0;
      }
    }
    if (this.volume <= 0.001) return 0;

    if (nowMs - this.lastScheduleMs >= 45) {
      this.lastScheduleMs = nowMs;
      const now = this.ctx.currentTime;
      const nowFrame = Math.max(0, Math.floor((now - this.originTime) * SYNTH_RATE));
      if (this.frameCursor < nowFrame) {
        this.frameCursor = nowFrame;
      }
      const targetFrame = nowFrame + Math.round(LOOKAHEAD_SEC * SYNTH_RATE);
      if (targetFrame > this.frameCursor) {
        this.applyDspModulation(now);
        const events = generateScoreEvents(
          this.frameCursor,
          targetFrame - this.frameCursor,
          this.obs,
        );
        this.scheduleEvents(this.frameCursor, events, now);
        this.frameCursor = targetFrame;
      }
    }

    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.rmsScratch);
    let sumSq = 0;
    for (let i = 0; i < this.rmsScratch.length; i++) {
      const s = this.rmsScratch[i];
      sumSq += s * s;
    }
    return Math.sqrt(sumSq / this.rmsScratch.length);
  }

  readSpectrogram(out: SynthSpectrogramSnapshot): void {
    const obs = this.obs;
    const env = obs.env;
    const darkBias = obs.halogenStress * 0.45 + env.nightFactor * 0.45 + obs.mineralSulfideRatio * 0.35;
    out.enabled = this.enabled;
    out.running = this.isRunning();
    out.bpm = SYNTH_BPM;
    out.mode = darkBias > 0.36 ? 'PHRYGIAN' : 'CYBER';
    out.observation = obs;
    out.bassCutoffHz = this.curBassCutoffHz;
    out.chorusWidth = this.curChorusWidth;
    out.delayFeedback = this.curDelayFeedback;
    out.acidDrive = this.curAcidDrive;
    out.streamlinesEnabled = this.streamlinesEnabled;
    out.streamlineEnergy = this.streamEnergyAvg;

    const currentStep = Math.floor(this.frameCursor / ((SYNTH_RATE * 60) / SYNTH_BPM / 4));
    out.section = cyberSection(Math.floor(currentStep / 16));

    if (!out.running || !this.ctx || !this.analyser || !this.bassAnalyser || !this.polyAnalyser || !this.chorusAnalyser || !this.delayAnalyser) {
      out.rms = 0;
      out.bassLevel = 0;
      out.acidLevel = 0;
      out.chorusLevel = 0;
      out.delayLevel = 0;
      out.polyLevel = 0;
      out.pulseLevel = 0;
      out.sidechainDepth = 0;
      out.downbeatPulse = 0;
      out.offbeatPulse = 0;
      out.lowBassPhase = 0;
      out.lowBassEnergy = 0;
      for (let c = 0; c < out.curves.length; c++) out.curves[c].fill(0);
      return;
    }

    const sr = this.ctx.sampleRate || 44100;
    this.bassAnalyser.getByteFrequencyData(this.bassBins);
    this.chorusAnalyser.getByteFrequencyData(this.chorusBins);
    this.delayAnalyser.getByteFrequencyData(this.delayBins);
    this.polyAnalyser.getByteFrequencyData(this.polyBins);
    this.analyser.getByteFrequencyData(this.masterBins);

    out.bassLevel = sampleLogBins(this.bassBins, sr, out.curves[0], 1.12);
    out.chorusLevel = sampleLogBins(this.chorusBins, sr, out.curves[2], 1.25);
    out.delayLevel = sampleLogBins(this.delayBins, sr, out.curves[3], 1.32);
    out.polyLevel = sampleLogBins(this.polyBins, sr, out.curves[4], 1.22);
    out.pulseLevel = sampleLogBins(this.masterBins, sr, out.curves[5], 1.0);

    // Acid waveshaper curve: highlight nonlinear upper-harmonic saturation energy directly
    const acidMul = 0.45 + 0.95 * this.curAcidDrive;
    let acidPeak = 0;
    for (let i = 0; i < SPECTROGRAM_BINS; i++) {
      const t = i / (SPECTROGRAM_BINS - 1);
      const harmonicBand = Math.sin(Math.PI * Math.min(1, Math.max(0.08, t * 1.15))) * 0.85 + 0.15;
      const v = Math.min(1, (out.curves[0][i] * 0.55 + out.curves[4][i] * 0.65) * acidMul * harmonicBand);
      out.curves[1][i] = v;
      if (v > acidPeak) acidPeak = v;
    }
    out.acidLevel = acidPeak;

    const duckVal = this.duckGain ? this.duckGain.gain.value : 1;
    out.sidechainDepth = Math.max(0, Math.min(1, (1 - duckVal) / 0.62));
    out.rms = Math.max(out.pulseLevel * 0.6, out.bassLevel * 0.5);

    // Pure low-tempo BPM beat timing + low-end bass energy (< ~220 Hz)
    const elapsedSec = Math.max(0, this.ctx.currentTime - this.originTime);
    const beatCount = (elapsedSec * SYNTH_BPM) / 60;
    const qPhase = beatCount % 1;
    const barBeat = Math.floor(beatCount) % 4;
    const downEmphasis = barBeat === 0 || barBeat === 2 ? 1.0 : 0.62;
    out.downbeatPulse = Math.exp(-qPhase * 4.2) * downEmphasis;

    const offPhase = (beatCount + 0.5) % 1;
    const offBeatIdx = Math.floor(beatCount + 0.5) % 4;
    const backEmphasis = offBeatIdx === 1 || offBeatIdx === 3 ? 1.0 : 0.68;
    out.offbeatPulse = Math.exp(-offPhase * 4.6) * backEmphasis;

    out.lowBassPhase = beatCount * Math.PI;
    let subSum = 0;
    const subBins = 16;
    for (let i = 0; i < subBins; i++) subSum += out.curves[0][i] ?? 0;
    out.lowBassEnergy = Math.max(0, Math.min(1, (subSum / subBins) * 1.35));
  }

  private applyDspModulation(now: number): void {
    if (!this.ctx) return;
    const obs = this.obs;
    const env = obs.env;

    if (this.delayFeedback && this.delayFilter) {
      const hydroWet = Math.min(0.40, 0.14 + 0.16 * env.rhNorm + 0.14 * env.precipNorm + 0.08 * obs.mineralCrystalRatio);
      this.curDelayFeedback = hydroWet;
      this.delayFeedback.gain.setTargetAtTime(hydroWet, now, 0.08);
      this.delayFilter.frequency.setTargetAtTime(
        1000 + 1400 * (1 - env.nightFactor) + 450 * obs.mineralCrystalRatio,
        now,
        0.08,
      );
    }

    if (this.chorusMix) {
      const chorusAmount = Math.min(0.68, 0.24 + 0.24 * obs.chemVaporRatio + 0.22 * obs.complexRatio + 0.16 * env.oceanFraction);
      this.curChorusWidth = chorusAmount;
      this.chorusMix.gain.setTargetAtTime(chorusAmount, now, 0.08);
    }

    let driveSum = 0;
    for (let i = 0; i < 4; i++) {
      const laneNode = this.lanes[i];
      const laneObs = obs.lanes[i];
      if (!laneNode || !laneObs) continue;

      driveSum += laneObs.acidDrive;
      laneNode.panner.pan.setTargetAtTime(laneObs.pan, now, 0.04);
      laneNode.delaySend.gain.setTargetAtTime(
        Math.min(0.38, 0.12 + 0.16 * env.rhNorm + 0.12 * laneObs.mineralGain + 0.08 * laneObs.chemGain),
        now,
        0.06,
      );
      laneNode.chorusSend.gain.setTargetAtTime(
        Math.min(0.65, 0.25 + 0.28 * laneObs.bioGain + 0.18 * laneObs.chemGain),
        now,
        0.06,
      );

      const bucket = Math.round(laneObs.acidDrive * 25);
      if (bucket !== laneNode.lastAcidBucket) {
        laneNode.lastAcidBucket = bucket;
        laneNode.shaper.curve = makeDriveCurve(bucket / 25);
      }
    }

    const bassLane = this.lanes[4];
    const bassDrive = Math.min(1, 0.20 + 0.42 * obs.halogenStress + 0.30 * env.ridgeMask + 0.24 * obs.chemBondActivity);
    this.curAcidDrive = (driveSum + bassDrive) / 5;
    if (bassLane) {
      bassLane.chorusSend.gain.setTargetAtTime(
        Math.min(0.55, 0.18 + 0.22 * obs.complexRatio + 0.18 * env.oceanFraction),
        now,
        0.08,
      );
      const bucket = Math.round(bassDrive * 25);
      if (bucket !== bassLane.lastAcidBucket) {
        bassLane.lastAcidBucket = bucket;
        bassLane.shaper.curve = makeDriveCurve(bucket / 25);
      }
    }

    this.curBassCutoffHz = obs.lanes[1]?.cutoffHz ?? 900;
  }

  private scheduleEvents(startFrame: number, events: ScheduledAudioEvent[], now: number): void {
    const ctx = this.ctx;
    if (!ctx) return;

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const when = Math.max(now, this.originTime + (startFrame + ev.offset) / SYNTH_RATE);
      if (when < now - 0.025 || !Number.isFinite(when)) continue;
      if (this.activeVoices >= MAX_VOICES) continue;

      if (ev.kind === 'drum') {
        const buf = this.drumBank.get(ev.instrument);
        const laneNode = this.lanes[ev.lane];
        if (!buf || !laneNode) continue;

        const src = ctx.createBufferSource();
        src.buffer = buf;
        const base =
          ev.instrument === 0
            ? DRUM_BASE_NOTES.kick
            : ev.instrument === 1
              ? DRUM_BASE_NOTES.hat
              : ev.instrument === 2
                ? DRUM_BASE_NOTES.snare
                : ev.instrument === 3
                  ? DRUM_BASE_NOTES.rim
                  : ev.instrument === 14
                    ? DRUM_BASE_NOTES.crystal
                    : DRUM_BASE_NOTES.anvil;
        const rate = Math.pow(2, (ev.note - base) / 12);
        src.playbackRate.value = Number.isFinite(rate) ? Math.max(0.25, Math.min(2.5, rate)) : 1;

        const gainNode = ctx.createGain();
        gainNode.gain.value = Number.isFinite(ev.gain) ? Math.max(0, Math.min(1.1, ev.gain)) : 0.3;
        src.connect(gainNode);

        if (ev.instrument === 0 && this.duckGain && this.masterGain) {
          this.duckGain.gain.cancelScheduledValues(when);
          this.duckGain.gain.setValueAtTime(1.0, Math.max(now, when - 0.002));
          this.duckGain.gain.linearRampToValueAtTime(0.38, when + 0.008);
          this.duckGain.gain.exponentialRampToValueAtTime(1.0, when + 0.195);
          gainNode.connect(this.masterGain);
        } else {
          gainNode.connect(laneNode.input);
        }

        this.activeVoices++;
        src.onended = () => {
          this.activeVoices = Math.max(0, this.activeVoices - 1);
          src.disconnect();
          gainNode.disconnect();
        };
        src.start(when);
        continue;
      }

      const laneNode = this.lanes[ev.lane];
      if (!laneNode) continue;

      const freq = midiToHz(ev.note);
      if (!Number.isFinite(freq) || freq <= 20) continue;
      const durSec = Number.isFinite(ev.durationSec) ? Math.max(0.05, ev.durationSec) : 0.25;
      const endWhen = when + durSec + 0.04;

      const oscMain = ctx.createOscillator();
      oscMain.type = 'sawtooth';
      oscMain.frequency.setValueAtTime(freq, when);
      oscMain.detune.setValueAtTime(-ev.detuneCents * 0.5, when);

      const oscDetune = ctx.createOscillator();
      oscDetune.type = 'sawtooth';
      oscDetune.frequency.setValueAtTime(freq, when);
      oscDetune.detune.setValueAtTime(ev.detuneCents * 0.5, when);

      const oscSub = ctx.createOscillator();
      oscSub.type = ev.voiceType === 'bass' ? 'triangle' : 'sine';
      oscSub.frequency.setValueAtTime(freq * 0.5, when);

      const subGain = ctx.createGain();
      subGain.gain.value = Number.isFinite(ev.subMix) ? ev.subMix : 0.4;

      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.Q.setValueAtTime(Number.isFinite(ev.filterQ) ? Math.min(12, ev.filterQ) : 2.0, when);
      const rawPeak = (Number.isFinite(ev.baseCutoffHz) ? ev.baseCutoffHz : 300) +
        (Number.isFinite(ev.envPeakHz) ? ev.envPeakHz : 1200);
      const peakFreq = Math.max(120, Math.min(8500, rawPeak));
      const baseFreq = Math.max(80, Math.min(peakFreq - 20, Number.isFinite(ev.baseCutoffHz) ? ev.baseCutoffHz : 250));

      if (ev.voiceType === 'pad') {
        filter.frequency.setValueAtTime(baseFreq, when);
        filter.frequency.exponentialRampToValueAtTime(peakFreq, when + Math.min(0.28, durSec * 0.45));
        filter.frequency.exponentialRampToValueAtTime(baseFreq, endWhen);
      } else {
        filter.frequency.setValueAtTime(peakFreq, when);
        filter.frequency.exponentialRampToValueAtTime(baseFreq, when + Math.max(0.05, durSec * 0.85));
      }

      const amp = ctx.createGain();
      const peakGain = Math.max(0.001, Math.min(0.85, (Number.isFinite(ev.gain) ? ev.gain : 0.2) * 0.55));
      const att = ev.voiceType === 'pad' ? 0.16 : 0.005;
      amp.gain.setValueAtTime(0.0001, when);
      amp.gain.linearRampToValueAtTime(peakGain, when + att);
      amp.gain.exponentialRampToValueAtTime(0.0005, endWhen);

      oscMain.connect(filter);
      oscDetune.connect(filter);
      oscSub.connect(subGain);
      subGain.connect(filter);
      filter.connect(amp);
      amp.connect(laneNode.input);

      this.activeVoices++;
      oscMain.onended = () => {
        this.activeVoices = Math.max(0, this.activeVoices - 1);
        oscMain.disconnect();
        oscDetune.disconnect();
        oscSub.disconnect();
        subGain.disconnect();
        filter.disconnect();
        amp.disconnect();
      };

      oscMain.start(when);
      oscDetune.start(when);
      oscSub.start(when);
      oscMain.stop(endWhen + 0.01);
      oscDetune.stop(endWhen + 0.01);
      oscSub.stop(endWhen + 0.01);
    }
  }
}
