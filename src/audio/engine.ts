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
import { generateScoreEvents, type ScheduledAudioEvent } from './score.ts';

interface AudioLaneNodes {
  input: GainNode;
  shaper: WaveShaperNode;
  panner: StereoPannerNode;
  chorusSend: GainNode;
  delaySend: GainNode;
  lastAcidBucket: number;
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
  private analyser: AnalyserNode | null = null;
  private rmsScratch = new Float32Array(256);

  private lanes: AudioLaneNodes[] = [];
  private activeVoices = 0;
  private originTime = 0;
  private frameCursor = 0;
  private lastScheduleMs = 0;
  private obs: ScreenAudioObservation = createEmptyAudioObservation();

  private enabled = false;
  private volume = 0.65;
  private unlocked = false;
  private armedOnGesture = false;

  isEnabled(): boolean {
    return this.enabled;
  }

  getVolume(): number {
    return this.volume;
  }

  getState(): AudioContextState | 'uninitialized' {
    return this.ctx ? this.ctx.state : 'uninitialized';
  }

  setVolume(vol: number): void {
    this.volume = Math.max(0, Math.min(1, vol));
    if (this.masterGain && this.ctx) {
      const target = this.enabled ? this.volume * 0.85 : 0;
      this.masterGain.gain.setTargetAtTime(target, this.ctx.currentTime, 0.02);
    }
  }

  setEnabled(on: boolean, fromUserGesture = false): void {
    this.enabled = on;
    if (!on) {
      this.armedOnGesture = false;
      if (this.masterGain && this.ctx) {
        this.masterGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.025);
      }
      return;
    }
    this.armedOnGesture = true;
    if (fromUserGesture) {
      void this.unlock();
    }
  }

  setObservation(obs: ScreenAudioObservation): void {
    this.obs = obs;
  }

  async unlock(): Promise<void> {
    if (this.ctx && this.ctx.state === 'running') {
      this.unlocked = true;
      return;
    }
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    if (!this.ctx) {
      this.ctx = new Ctor({ latencyHint: 'interactive' });
      this.buildGraph(this.ctx);
      this.drumBank = buildCyberDrumBank(this.ctx);
      this.originTime = this.ctx.currentTime;
      this.frameCursor = 0;
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
      if (this.masterGain) {
        this.masterGain.gain.setTargetAtTime(
          this.enabled ? this.volume * 0.85 : 0,
          this.ctx.currentTime,
          0.025,
        );
      }
      this.originTime = this.ctx.currentTime;
      this.frameCursor = 0;
    }
  }

  private buildGraph(ctx: AudioContext): void {
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0;

    this.duckGain = ctx.createGain();
    this.duckGain.gain.value = 1;

    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -13;
    this.compressor.knee.value = 10;
    this.compressor.ratio.value = 4.5;
    this.compressor.attack.value = 0.004;
    this.compressor.release.value = 0.13;

    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 256;

    // Tempo-synced dotted-8th stereo delay
    this.delayNode = ctx.createDelay(2.0);
    this.delayNode.delayTime.value = (60 / SYNTH_BPM) * 0.75;
    this.delayFeedback = ctx.createGain();
    this.delayFeedback.gain.value = 0.26;
    this.delayFilter = ctx.createBiquadFilter();
    this.delayFilter.type = 'lowpass';
    this.delayFilter.frequency.value = 2100;

    this.delayNode.connect(this.delayFilter);
    this.delayFilter.connect(this.delayFeedback);
    this.delayFeedback.connect(this.delayNode);
    this.delayFilter.connect(this.duckGain);

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
    this.chorusL.connect(panL);
    this.chorusR.connect(panR);
    panL.connect(this.duckGain);
    panR.connect(this.duckGain);

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

  private applyDspModulation(now: number): void {
    if (!this.ctx) return;
    const obs = this.obs;
    const env = obs.env;

    if (this.delayFeedback && this.delayFilter) {
      const hydroWet = Math.min(0.42, 0.14 + 0.18 * env.rhNorm + 0.14 * env.precipNorm + 0.08 * obs.mineralCrystalRatio);
      this.delayFeedback.gain.setTargetAtTime(hydroWet, now, 0.08);
      this.delayFilter.frequency.setTargetAtTime(
        1000 + 1700 * (1 - env.nightFactor) + 600 * obs.mineralCrystalRatio,
        now,
        0.08,
      );
    }

    if (this.chorusMix) {
      const chorusAmount = Math.min(0.68, 0.24 + 0.24 * obs.chemVaporRatio + 0.22 * obs.complexRatio + 0.16 * env.oceanFraction);
      this.chorusMix.gain.setTargetAtTime(chorusAmount, now, 0.08);
    }

    for (let i = 0; i < 4; i++) {
      const laneNode = this.lanes[i];
      const laneObs = obs.lanes[i];
      if (!laneNode || !laneObs) continue;

      laneNode.panner.pan.setTargetAtTime(laneObs.pan, now, 0.04);
      laneNode.delaySend.gain.setTargetAtTime(
        Math.min(0.40, 0.12 + 0.16 * env.rhNorm + 0.14 * laneObs.mineralGain + 0.08 * laneObs.chemGain),
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
    if (bassLane) {
      const bassDrive = Math.min(1, 0.20 + 0.42 * obs.halogenStress + 0.30 * env.ridgeMask + 0.24 * obs.chemBondActivity);
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
  }

  private scheduleEvents(startFrame: number, events: ScheduledAudioEvent[], now: number): void {
    const ctx = this.ctx;
    if (!ctx) return;

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const when = Math.max(now, this.originTime + (startFrame + ev.offset) / SYNTH_RATE);
      if (when < now - 0.025) continue;
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
        src.playbackRate.value = Math.pow(2, (ev.note - base) / 12);

        const gainNode = ctx.createGain();
        gainNode.gain.value = Math.max(0, Math.min(1.1, ev.gain));
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
      const endWhen = when + ev.durationSec + 0.04;

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
      subGain.gain.value = ev.subMix;

      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.Q.setValueAtTime(ev.filterQ, when);
      const peakFreq = Math.max(120, Math.min(9500, ev.baseCutoffHz + ev.envPeakHz));
      const baseFreq = Math.max(80, Math.min(peakFreq - 20, ev.baseCutoffHz));

      if (ev.voiceType === 'pad') {
        filter.frequency.setValueAtTime(baseFreq, when);
        filter.frequency.exponentialRampToValueAtTime(peakFreq, when + Math.min(0.28, ev.durationSec * 0.45));
        filter.frequency.exponentialRampToValueAtTime(baseFreq, endWhen);
      } else {
        filter.frequency.setValueAtTime(peakFreq, when);
        filter.frequency.exponentialRampToValueAtTime(baseFreq, when + Math.max(0.05, ev.durationSec * 0.85));
      }

      const amp = ctx.createGain();
      const peakGain = Math.max(0.001, Math.min(0.85, ev.gain * 0.55));
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
