/**
 * Self-contained, zero-allocation AudioWorkletProcessor synthesizer and 120 BPM sequencer.
 * Moves all timing, voice allocation, filtering, waveshaping, stereo chorus, dotted-8th delay,
 * sidechain ducking, drum playback, and 16-band streamline synthesis onto the audio thread.
 */
export const GLOBE_SYNTH_WORKLET_NAME = 'globe-synth-processor';

export const GLOBE_SYNTH_CORE_SOURCE = `'use strict';

var GLOBE_SYNTH_BPM = 120;
var MAX_DRUM_VOICES = 8;
var MAX_SYNTH_VOICES = 16;
var CHORUS_BUF_LEN = 4096;
var CHORUS_MASK = CHORUS_BUF_LEN - 1;
var DELAY_BUF_LEN = 131072;
var DELAY_MASK = DELAY_BUF_LEN - 1;

var SIN_TABLE = new Float32Array(1025);
(function initSinTable() {
  var tau = Math.PI * 2;
  for (var i = 0; i < 1024; i++) {
    SIN_TABLE[i] = Math.sin((i / 1024) * tau);
  }
  SIN_TABLE[1024] = 0;
})();

function fastSin01(phase01) {
  var p = phase01 - Math.floor(phase01);
  var idxF = p * 1024;
  var i0 = idxF | 0;
  var frac = idxF - i0;
  var s0 = SIN_TABLE[i0];
  return s0 + (SIN_TABLE[i0 + 1] - s0) * frac;
}

function fastTanh(x) {
  if (x < -3.0) return -1.0;
  if (x > 3.0) return 1.0;
  var x2 = x * x;
  return (x * (27.0 + x2)) / (27.0 + 9.0 * x2);
}

function clamp01(v) {
  return v > 0 ? (v < 1 ? v : 1) : 0;
}

function clampNum(v, lo, hi) {
  return v > lo ? (v < hi ? v : hi) : lo;
}

function midiToHz(midi) {
  return 440.0 * Math.pow(2.0, (midi - 69.0) / 12.0);
}

function polyBlep(t, dt) {
  if (t < dt) {
    var x = t / dt;
    return x + x - x * x - 1.0;
  }
  if (t > 1.0 - dt) {
    var y = (t - 1.0) / dt;
    return y * y + y + y + 1.0;
  }
  return 0.0;
}

var PROG_CYBER = [0, 5, 8, 10];
var PROG_PHRYGIAN = [0, 5, 1, 8];
var PHRYGIAN_DEGREES = [0, 1, 3, 5, 7, 8, 10, 12];
var MOTIF_SCALE = [0, 1, 3, 5, 7, 8, 10, 12];
var MOTIF_SLOT_FOR_BEAT = new Int8Array([-1, -1, -1, 1, -1, -1, 2, -1, -1, -1, 3, -1, 4, -1, -1, 5]);
MOTIF_SLOT_FOR_BEAT[0] = 0;

var SEQ_BASS_CONTOURS = [
  new Int8Array([0, 12, 0, 10, 0, 7, 0, 3, 0, 12, 0, 10, 7, 12, 10, 3]),
  new Int8Array([0, 0, 12, 0, 10, 0, 7, 10, 0, 0, 12, 3, 7, 10, 12, 7]),
  new Int8Array([0, 12, 7, 0, 12, 10, 7, 3, 0, 12, 7, 0, 1, 13, 10, 7]),
  new Int8Array([0, 0, 7, 12, 0, 10, 0, 3, 0, 0, 7, 10, 12, 7, 3, 0]),
];

var WIND_DEGREES_AEOLIAN = [0, 3, 7, 10, 12, 15, 19, 22];
var WIND_DEGREES_PHRYGIAN = [0, 1, 7, 10, 12, 13, 19, 22];
var OCEAN_DEGREES_AEOLIAN = [-12, -5, 0, 3, 7, 10, 12, 15];
var OCEAN_DEGREES_PHRYGIAN = [-12, -5, -11, 0, 1, 7, 8, 12];

var LANE_DEFAULT_PAN = [-0.68, -0.22, 0.22, 0.68, 0.0];
var DRUM_BASE_NOTES = [36, 60, 60, 45, 48, 36];

function buildDrumBankBuffers(sr) {
  var TAU = Math.PI * 2.0;
  var randState = 2026120 >>> 0;
  function nextRand() {
    randState = (Math.imul(1664525, randState) + 1013904223) >>> 0;
    return (randState / 4294967296.0) * 2.0 - 1.0;
  }

  var ids = [0, 1, 2, 3, 14, 15];
  var baseNotes = [36, 60, 60, 45, 48, 36];
  var durs = [0.42, 0.075, 0.24, 0.18, 0.72, 1.15];
  var bank = new Array(6);

  for (var s = 0; s < 6; s++) {
    var id = ids[s];
    var dur = durs[s];
    var count = Math.ceil(dur * sr);
    var out = new Float32Array(count);
    var baseFreq = midiToHz(baseNotes[s]);
    var filtered = 0.0;
    var prevNoise = 0.0;
    var smoothAlpha = id === 0 || id === 2 ? 0.75 : id === 1 ? 0.55 : 0.48;

    for (var i = 0; i < count; i++) {
      var t = i / sr;
      var p = TAU * baseFreq * t;
      var noise = nextRand();
      var value = 0.0;
      var attack = Math.min(1.0, t / 0.004);
      var release = Math.min(1.0, (dur - t) / 0.028);

      if (id === 0) {
        var sweepPhase = TAU * (42.0 * t + 86.0 * 0.022 * (1.0 - Math.exp(-t / 0.022)));
        var body = Math.tanh(Math.sin(sweepPhase) * 1.55) * Math.exp(-t * 7.8);
        var click = (noise - prevNoise * 0.25) * Math.exp(-t * 170.0) * 0.14;
        value = body * 0.94 + click;
        attack = Math.min(1.0, t / 0.0012);
      } else if (id === 1) {
        var hp = noise - prevNoise * 0.78;
        var darkMetal = Math.sin(p * 3.2 + 0.9 * Math.sin(p * 4.7));
        value = (hp * 0.60 + darkMetal * 0.14) * Math.exp(-t * 58.0) * 0.36;
        attack = Math.min(1.0, t / 0.001);
      } else if (id === 2) {
        var burst = Math.exp(-t * 48.0);
        if (t >= 0.009) burst += Math.exp(-(t - 0.009) * 52.0) * 0.65;
        if (t >= 0.020) burst += Math.exp(-(t - 0.020) * 44.0) * 0.5;
        var sBody = Math.sin(TAU * (168.0 - 44.0 * (1.0 - Math.exp(-t * 36.0))) * t) * Math.exp(-t * 22.0);
        var gateEnv = t < 0.14 ? 1.0 : Math.exp(-(t - 0.14) * 32.0);
        value = ((noise - prevNoise * 0.35) * burst * 0.52 + sBody * 0.44) * gateEnv;
        attack = Math.min(1.0, t / 0.0012);
      } else if (id === 3) {
        var pitchEnv = Math.exp(-t * 28.0);
        var sweep = TAU * (baseFreq * (0.65 + 0.55 * pitchEnv)) * t;
        value = Math.tanh((Math.sin(sweep) + 0.35 * Math.sin(sweep * 0.5)) * 1.25) * Math.exp(-t * 18.0) * 0.68;
        attack = Math.min(1.0, t / 0.0015);
      } else if (id === 14) {
        value =
          (0.54 * Math.sin(p) +
            0.34 * Math.sin(p * 0.5) +
            0.16 * Math.sin(p * 1.498) * Math.exp(-t * 7.5)) *
          Math.exp(-t * 4.2) *
          0.55;
        attack = Math.min(1.0, t / 0.008);
      } else {
        var aBody =
          0.58 * Math.sin(p) +
          0.34 * Math.sin(p * 0.5) +
          0.20 * Math.sin(p * 1.498) * Math.exp(-t * 4.0);
        value = Math.tanh(aBody * 1.25) * Math.exp(-t * 2.5) * 0.72;
      }

      filtered += (value - filtered) * smoothAlpha;
      var env = attack * (release > 0 ? release : 0);
      var clamped = filtered * env;
      out[i] = clamped > 1.0 ? 1.0 : clamped < -1.0 ? -1.0 : clamped;
      prevNoise = noise;
    }
    bank[s] = out;
  }
  return bank;
}

class SynthDspCore {
  constructor(sampleRate) {
    this.sampleRate = clampNum(Math.round(sampleRate || 48000), 8000, 192000);
    this.invSr = 1.0 / this.sampleRate;
    this.samplesPerStep = (this.sampleRate * 60.0) / GLOBE_SYNTH_BPM / 4.0;
    this.stepSec = 60.0 / GLOBE_SYNTH_BPM / 4.0;

    this.drumBuffers = buildDrumBankBuffers(this.sampleRate);

    // Pre-allocated drum voice pool (8 voices)
    this.dActive = new Uint8Array(MAX_DRUM_VOICES);
    this.dSlot = new Uint8Array(MAX_DRUM_VOICES);
    this.dLane = new Uint8Array(MAX_DRUM_VOICES);
    this.dIsKick = new Uint8Array(MAX_DRUM_VOICES);
    this.dPos = new Float64Array(MAX_DRUM_VOICES);
    this.dRate = new Float32Array(MAX_DRUM_VOICES);
    this.dGain = new Float32Array(MAX_DRUM_VOICES);
    this.dAge = new Int32Array(MAX_DRUM_VOICES);

    // Pre-allocated polyphonic synth voice pool (16 voices)
    this.vActive = new Uint8Array(MAX_SYNTH_VOICES);
    this.vLane = new Uint8Array(MAX_SYNTH_VOICES);
    this.vType = new Uint8Array(MAX_SYNTH_VOICES);
    this.vStage = new Uint8Array(MAX_SYNTH_VOICES);
    this.vPhase1 = new Float64Array(MAX_SYNTH_VOICES);
    this.vPhase2 = new Float64Array(MAX_SYNTH_VOICES);
    this.vSubPhase = new Float64Array(MAX_SYNTH_VOICES);
    this.vInc1 = new Float32Array(MAX_SYNTH_VOICES);
    this.vInc2 = new Float32Array(MAX_SYNTH_VOICES);
    this.vSubInc = new Float32Array(MAX_SYNTH_VOICES);
    this.vDualSaw = new Uint8Array(MAX_SYNTH_VOICES);
    this.vSubIsTri = new Uint8Array(MAX_SYNTH_VOICES);
    this.vGain = new Float32Array(MAX_SYNTH_VOICES);
    this.vSubMix = new Float32Array(MAX_SYNTH_VOICES);
    this.vAge = new Int32Array(MAX_SYNTH_VOICES);
    this.vAttackSamples = new Int32Array(MAX_SYNTH_VOICES);
    this.vHoldSamples = new Int32Array(MAX_SYNTH_VOICES);
    this.vCutRiseSamples = new Int32Array(MAX_SYNTH_VOICES);
    this.vAttStep = new Float32Array(MAX_SYNTH_VOICES);
    this.vDecayMul = new Float32Array(MAX_SYNTH_VOICES);
    this.vReleaseMul = new Float32Array(MAX_SYNTH_VOICES);
    this.vAmp = new Float32Array(MAX_SYNTH_VOICES);
    this.vBaseCut = new Float32Array(MAX_SYNTH_VOICES);
    this.vPeakCut = new Float32Array(MAX_SYNTH_VOICES);
    this.vCurCut = new Float32Array(MAX_SYNTH_VOICES);
    this.vCutRiseStep = new Float32Array(MAX_SYNTH_VOICES);
    this.vCutDecayMul = new Float32Array(MAX_SYNTH_VOICES);
    this.vFilterK = new Float32Array(MAX_SYNTH_VOICES);
    this.vIc1eq = new Float32Array(MAX_SYNTH_VOICES);
    this.vIc2eq = new Float32Array(MAX_SYNTH_VOICES);
    this.vA1 = new Float32Array(MAX_SYNTH_VOICES);
    this.vA2 = new Float32Array(MAX_SYNTH_VOICES);
    this.vA3 = new Float32Array(MAX_SYNTH_VOICES);

    // Per-lane DSP parameters & scratch
    this.laneAcc = new Float32Array(5);
    this.laneK = new Float32Array([1.4, 1.4, 1.4, 1.4, 1.6]);
    this.laneInvNorm = new Float32Array(5);
    this.lanePan = new Float32Array([-0.68, -0.22, 0.22, 0.68, 0.0]);
    this.laneTargetPan = new Float32Array([-0.68, -0.22, 0.22, 0.68, 0.0]);
    this.lanePanL = new Float32Array(5);
    this.lanePanR = new Float32Array(5);
    this.laneChorusSend = new Float32Array([0.45, 0.45, 0.45, 0.45, 0.28]);
    this.laneTargetChorusSend = new Float32Array([0.45, 0.45, 0.45, 0.45, 0.28]);
    this.laneDelaySend = new Float32Array([0.20, 0.20, 0.20, 0.20, 0.05]);
    this.laneTargetDelaySend = new Float32Array([0.20, 0.20, 0.20, 0.20, 0.05]);
    for (var l = 0; l < 5; l++) {
      this.laneInvNorm[l] = 1.0 / (fastTanh(this.laneK[l]) || 1.0);
      var a = (this.lanePan[l] + 1.0) * 0.25 * Math.PI;
      this.lanePanL[l] = Math.cos(a);
      this.lanePanR[l] = Math.sin(a);
    }

    // FX ring buffers (zero heap allocation after construction)
    this.chorusBuf = new Float32Array(CHORUS_BUF_LEN);
    this.chorusWrite = 0;
    this.chorusLfoPhase = 0.0;
    this.chorusPanLAngle = (-0.82 + 1.0) * 0.25 * Math.PI;
    this.chorusPanRAngle = (0.82 + 1.0) * 0.25 * Math.PI;
    this.chorusGainLL = Math.cos(this.chorusPanLAngle);
    this.chorusGainLR = Math.sin(this.chorusPanLAngle);
    this.chorusGainRL = Math.cos(this.chorusPanRAngle);
    this.chorusGainRR = Math.sin(this.chorusPanRAngle);

    this.delayBuf = new Float32Array(DELAY_BUF_LEN);
    this.delayWrite = 0;
    this.delaySamples = clampNum(Math.round(this.sampleRate * 0.375), 1, DELAY_BUF_LEN - 2);
    this.delayLpState = 0.0;
    this.delayLpAlpha = 0.22;

    // 16-band streamline resonator state
    this.streamPhases = new Float32Array(16);
    this.streamDetunePhases = new Float32Array(16);
    this.streamAmps = new Float32Array(16);
    this.streamTargetAmps = new Float32Array(16);
    this.streamFreqs = new Float32Array([
      146.83, 220.0, 293.66, 349.23, 440.0, 523.25, 587.33, 659.25,
      73.42, 110.0, 146.83, 174.61, 220.0, 261.63, 293.66, 329.63
    ]);
    this.streamTargetFreqs = new Float32Array(this.streamFreqs);
    this.streamPanL = new Float32Array(16);
    this.streamPanR = new Float32Array(16);
    for (var s = 0; s < 8; s++) {
      var sp = -0.82 + (s / 7.0) * 1.64;
      var lG = 0.5 * (1.0 - sp * 0.72);
      var rG = 0.5 * (1.0 + sp * 0.72);
      this.streamPanL[s] = lG;
      this.streamPanR[s] = rG;
      this.streamPanL[8 + s] = lG;
      this.streamPanR[8 + s] = rG;
    }

    // Sidechain & pause ramp envelopes
    this.duckVal = 1.0;
    this.duckStage = 0; // 0: idle (1.0), 1: attack down, 2: release up
    this.duckAttStep = (1.0 - 0.36) / Math.max(1, Math.round(0.007 * this.sampleRate));
    this.duckRelCoeff = 1.0 - Math.exp(-1.0 / Math.max(1, 0.14 * this.sampleRate));

    this.playGain = 0.0;
    this.playRampCoeff = 1.0 - Math.exp(-1.0 / Math.max(1, 0.015 * this.sampleRate));
    this.streamMasterAmp = 0.0;
    this.active = false;
    this.streamlinesEnabled = true;
    this.streamEnergyAvg = 0.0;

    // Sequencer cursor state
    this.sampleCursor = 0;
    this.nextStep = 0;
    this.nextStepSample = 0;
    this.currentBar = 0;
    this.currentStep = 0;
    this.motifBuf = new Int8Array([0, 7, 10, 3, 1, 0]);
    this.motifSeedHash = -1;
    this.motifSpeciesId = -1;

    // Flat observation state (pre-allocated)
    this.obsActive = false;
    this.obsSeedHash = 0;
    this.obsIntensity = 0.25;
    this.obsTotalBioMass = 0.0;
    this.obsProducerRatio = 0.0;
    this.obsConsumerRatio = 0.0;
    this.obsComplexRatio = 0.0;
    this.obsChemBondActivity = 0.0;
    this.obsChemVaporRatio = 0.0;
    this.obsHalogenStress = 0.0;
    this.obsMineralCrystalRatio = 0.0;
    this.obsMineralSulfideRatio = 0.0;
    this.obsDominantBioSp = 0;
    this.obsAvgEnergy = 0.65;

    this.envWindSpeedNorm = 0.35;
    this.envOceanFraction = 0.5;
    this.envTempNorm = 0.55;
    this.envNightFactor = 0.2;
    this.envRhNorm = 0.5;
    this.envPrecipNorm = 0.1;
    this.envElevKmNorm = 0.1;
    this.envRidgeMask = 0.0;

    this.laneBioGain = new Float32Array(4);
    this.laneChemGain = new Float32Array(4);
    this.laneMineralGain = new Float32Array(4);
    this.laneCutoffHz = new Float32Array([1400, 1400, 1400, 1400]);
    this.laneAcidDrive = new Float32Array([0.1, 0.1, 0.1, 0.1]);
    this.laneMinSulfFrac = new Float32Array(4);

    // Smoothed macro parameters
    this.curBassCutoffHz = 900.0;
    this.curChorusWidth = 0.42;
    this.curDelayFeedback = 0.24;
    this.curAcidDrive = 0.18;

    // Telemetry accumulators & smoothed RMS gauges
    this.rms = 0.0;
    this.bassLevel = 0.0;
    this.polyLevel = 0.0;
    this.chorusLevel = 0.0;
    this.delayLevel = 0.0;
    this.lowBassEnergy = 0.0;
    this.sidechainDepth = 0.0;

    this.syncModulationTargets();
    this.updateStreamTargetFreqs();
  }

  applyMessage(msg) {
    if (!msg) return;
    if (msg.type === 'obs' && msg.obs) {
      this.setObservationData(msg.obs);
      return;
    }
    if (msg.type === 'streams') {
      if (typeof msg.enabled === 'boolean') {
        this.streamlinesEnabled = msg.enabled;
      }
      if (msg.bands && msg.bands.length >= 16) {
        var sum = 0.0;
        for (var i = 0; i < 16; i++) {
          var v = clamp01(msg.bands[i] || 0);
          this.streamTargetAmps[i] = v;
          sum += v;
        }
        this.streamEnergyAvg = Math.min(1.0, sum / 16.0);
      } else if (!this.streamlinesEnabled) {
        this.streamTargetAmps.fill(0);
        this.streamEnergyAvg = 0.0;
      }
      return;
    }
    if (msg.type === 'state') {
      if (typeof msg.active === 'boolean') {
        this.active = msg.active;
      }
      if (typeof msg.streamlinesEnabled === 'boolean') {
        this.streamlinesEnabled = msg.streamlinesEnabled;
      }
      if (msg.clearFx) {
        this.chorusBuf.fill(0);
        this.delayBuf.fill(0);
        this.delayLpState = 0.0;
      }
      if (msg.resetClock) {
        this.resetClockState();
      }
    }
  }

  resetClockState() {
    this.sampleCursor = 0;
    this.nextStep = 0;
    this.nextStepSample = 0;
    this.currentBar = 0;
    this.currentStep = 0;
    this.dActive.fill(0);
    this.vActive.fill(0);
    this.chorusBuf.fill(0);
    this.delayBuf.fill(0);
    this.delayLpState = 0.0;
    this.duckVal = 1.0;
    this.duckStage = 0;
    this.streamAmps.fill(0);
    this.streamMasterAmp = 0.0;
    this.playGain = 0.0;
    this.updateStreamTargetFreqs();
  }

  setObservationData(o) {
    this.obsActive = Boolean(o.active);
    this.obsSeedHash = (o.seedHash >>> 0) || 0;
    this.obsIntensity = clampNum(o.intensity || 0.25, 0.05, 1.0);
    this.obsTotalBioMass = Math.max(0.0, Number(o.totalBioMass) || 0.0);
    this.obsProducerRatio = clamp01(o.producerRatio || 0);
    this.obsConsumerRatio = clamp01(o.consumerRatio || 0);
    this.obsComplexRatio = clamp01(o.complexRatio || 0);
    this.obsChemBondActivity = clamp01(o.chemBondActivity || 0);
    this.obsChemVaporRatio = clamp01(o.chemVaporRatio || 0);
    this.obsHalogenStress = clamp01(o.halogenStress || 0);
    this.obsMineralCrystalRatio = clamp01(o.mineralCrystalRatio || 0);
    this.obsMineralSulfideRatio = clamp01(o.mineralSulfideRatio || 0);
    this.obsDominantBioSp = (o.dominantBioSp | 0) & 255;
    this.obsAvgEnergy = clamp01(o.avgEnergy != null ? o.avgEnergy : 0.65);

    var env = o.env;
    if (env) {
      this.envWindSpeedNorm = clamp01(env.windSpeedNorm || 0);
      this.envOceanFraction = clamp01(env.oceanFraction != null ? env.oceanFraction : 0.5);
      this.envTempNorm = clamp01(env.tempNorm != null ? env.tempNorm : 0.55);
      this.envNightFactor = clamp01(env.nightFactor || 0);
      this.envRhNorm = clamp01(env.rhNorm || 0);
      this.envPrecipNorm = clamp01(env.precipNorm || 0);
      this.envElevKmNorm = clamp01(env.elevKmNorm || 0);
      this.envRidgeMask = clamp01(env.ridgeMask || 0);
    }

    if (o.lanes && o.lanes.length >= 4) {
      for (var i = 0; i < 4; i++) {
        var ln = o.lanes[i];
        if (!ln) continue;
        this.laneTargetPan[i] = clampNum(ln.pan != null ? ln.pan : LANE_DEFAULT_PAN[i], -1.0, 1.0);
        this.laneBioGain[i] = clamp01(ln.bioGain || 0);
        this.laneChemGain[i] = clamp01(ln.chemGain || 0);
        this.laneMineralGain[i] = clamp01(ln.mineralGain || 0);
        this.laneCutoffHz[i] = clampNum(ln.cutoffHz || 1400, 180, 7200);
        this.laneAcidDrive[i] = clamp01(ln.acidDrive || 0.1);
        this.laneMinSulfFrac[i] = clamp01(ln.mineralSulfideFrac || 0);
      }
    }

    this.syncModulationTargets();
    this.updateStreamTargetFreqs();
  }

  syncModulationTargets() {
    this.curDelayFeedback = Math.min(
      0.40,
      0.14 + 0.16 * this.envRhNorm + 0.14 * this.envPrecipNorm + 0.08 * this.obsMineralCrystalRatio
    );
    var lpFreq = clampNum(
      1000.0 + 1400.0 * (1.0 - this.envNightFactor) + 450.0 * this.obsMineralCrystalRatio,
      250.0,
      this.sampleRate * 0.40
    );
    this.delayLpAlpha = Math.min(0.95, 1.0 - Math.exp(-Math.PI * 2.0 * lpFreq * this.invSr));

    this.curChorusWidth = Math.min(
      0.68,
      0.24 + 0.24 * this.obsChemVaporRatio + 0.22 * this.obsComplexRatio + 0.16 * this.envOceanFraction
    );

    var driveSum = 0.0;
    for (var i = 0; i < 4; i++) {
      var drive = this.laneAcidDrive[i];
      driveSum += drive;
      this.laneTargetDelaySend[i] = Math.min(
        0.38,
        0.12 + 0.16 * this.envRhNorm + 0.12 * this.laneMineralGain[i] + 0.08 * this.laneChemGain[i]
      );
      this.laneTargetChorusSend[i] = Math.min(
        0.65,
        0.25 + 0.28 * this.laneBioGain[i] + 0.18 * this.laneChemGain[i]
      );
      this.laneK[i] = 1.0 + drive * 2.4;
      this.laneInvNorm[i] = 1.0 / (fastTanh(this.laneK[i]) || 1.0);
    }

    var bassDrive = Math.min(
      1.0,
      0.20 + 0.42 * this.obsHalogenStress + 0.30 * this.envRidgeMask + 0.24 * this.obsChemBondActivity
    );
    this.curAcidDrive = (driveSum + bassDrive) * 0.2;
    this.laneK[4] = 1.0 + bassDrive * 2.4;
    this.laneInvNorm[4] = 1.0 / (fastTanh(this.laneK[4]) || 1.0);
    this.laneTargetDelaySend[4] = 0.05;
    this.laneTargetChorusSend[4] = Math.min(
      0.55,
      0.18 + 0.22 * this.obsComplexRatio + 0.18 * this.envOceanFraction
    );
    this.curBassCutoffHz = this.laneCutoffHz[1] || 900.0;
  }

  updateStreamTargetFreqs() {
    var darkBias = this.obsHalogenStress * 0.45 + this.envNightFactor * 0.45 + this.obsMineralSulfideRatio * 0.35;
    var usePhrygian = darkBias > 0.36;
    var prog = usePhrygian ? PROG_PHRYGIAN : PROG_CYBER;
    var chordOffset = prog[((this.currentBar >> 1) & 3)];
    var rootMidi = 50 + chordOffset + (this.envNightFactor > 0.55 ? -12 : 0);
    var wDeg = usePhrygian ? WIND_DEGREES_PHRYGIAN : WIND_DEGREES_AEOLIAN;
    var oDeg = usePhrygian ? OCEAN_DEGREES_PHRYGIAN : OCEAN_DEGREES_AEOLIAN;
    for (var i = 0; i < 8; i++) {
      this.streamTargetFreqs[i] = midiToHz(clampNum(rootMidi + wDeg[i], 36, 80));
      this.streamTargetFreqs[8 + i] = midiToHz(clampNum(rootMidi + oDeg[i], 28, 72));
    }
  }

  updateMotifIfNeeded() {
    var sh = this.obsSeedHash;
    var sp = this.obsDominantBioSp;
    if (sh === this.motifSeedHash && sp === this.motifSpeciesId) return;
    this.motifSeedHash = sh;
    this.motifSpeciesId = sp;
    var s = ((sh * 65537 + sp * 104729) >>> 0) || 1337;
    this.motifBuf[0] = 0;
    this.motifBuf[5] = 0;
    for (var i = 1; i < 5; i++) {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      this.motifBuf[i] = MOTIF_SCALE[s % MOTIF_SCALE.length];
    }
  }

  spawnDrum(inst, lane, note, gain) {
    var slot = inst === 0 ? 0 : inst === 1 ? 1 : inst === 2 ? 2 : inst === 3 ? 3 : inst === 14 ? 4 : 5;
    var chosen = -1;
    var maxAge = -1;
    for (var i = 0; i < MAX_DRUM_VOICES; i++) {
      if (!this.dActive[i]) {
        chosen = i;
        break;
      }
      if (this.dAge[i] > maxAge) {
        maxAge = this.dAge[i];
        chosen = i;
      }
    }
    if (chosen < 0) chosen = 0;

    var rate = Math.pow(2.0, (note - DRUM_BASE_NOTES[slot]) / 12.0);
    this.dActive[chosen] = 1;
    this.dSlot[chosen] = slot;
    this.dLane[chosen] = clampNum(lane, 0, 4) | 0;

    this.dIsKick[chosen] = inst === 0 ? 1 : 0;
    this.dPos[chosen] = 0.0;
    this.dRate[chosen] = Number.isFinite(rate) ? clampNum(rate, 0.25, 2.5) : 1.0;
    this.dGain[chosen] = clampNum(gain || 0.3, 0.0, 1.1);
    this.dAge[chosen] = 0;

    if (inst === 0) {
      this.duckStage = 1;
    }
  }

  spawnSynth(voiceType, lane, note, gain, durSec, baseCutHz, envPeakHz, filterQ, subMix, detuneCents) {
    var freq = midiToHz(note);
    if (!Number.isFinite(freq) || freq <= 20.0) return;

    // Keep monophonic bassline tight: accelerate release on any overlapping active bass voice
    if (voiceType === 0) {
      var fastRelMul = Math.exp(-1.0 / Math.max(1.0, 0.008 * this.sampleRate));
      for (var b = 0; b < MAX_SYNTH_VOICES; b++) {
        if (this.vActive[b] && this.vType[b] === 0) {
          this.vStage[b] = 2;
          if (this.vReleaseMul[b] > fastRelMul) {
            this.vReleaseMul[b] = fastRelMul;
          }
        }
      }
    }

    // Steal order: inactive -> quietest releasing non-bass -> quietest non-bass -> quietest voice
    var chosen = -1;
    var bestScore = 1e9;
    for (var i = 0; i < MAX_SYNTH_VOICES; i++) {
      if (!this.vActive[i]) {
        chosen = i;
        break;
      }
      var score = this.vAmp[i] + (this.vType[i] === 0 ? 10.0 : 0.0) + (this.vStage[i] === 2 ? -2.0 : 0.0);
      if (score < bestScore) {
        bestScore = score;
        chosen = i;
      }
    }
    if (chosen < 0) chosen = 0;

    var dualSaw = voiceType !== 1 ? 1 : 0;
    var detSemi = detuneCents / 1200.0;
    var detMul1 = dualSaw ? Math.pow(2.0, -0.5 * detSemi) : 1.0;
    var detMul2 = dualSaw ? Math.pow(2.0, 0.5 * detSemi) : 1.0;
    var peakGain = clampNum(gain * (dualSaw ? 0.50 : 0.68), 0.001, 0.80);

    var safeDur = clampNum(durSec, 0.05, 4.0);
    var attSec = voiceType === 3 ? 0.16 : 0.005;
    var attSamples = Math.max(4, Math.round(attSec * this.sampleRate));
    var holdSamples = Math.max(attSamples + 4, Math.round(safeDur * this.sampleRate));
    var relSec = Math.max(0.035, safeDur * 0.25);

    var peakFreq = clampNum(baseCutHz + envPeakHz, 120.0, 0.42 * this.sampleRate);
    var baseFreq = clampNum(baseCutHz, 60.0, peakFreq - 20.0);
    var safeQ = clampNum(filterQ || 2.0, 0.7, 6.5);

    this.vActive[chosen] = 1;
    this.vLane[chosen] = clampNum(lane, 0, 4) | 0;
    this.vType[chosen] = voiceType;
    this.vStage[chosen] = 0;
    this.vPhase1[chosen] = 0.0;
    this.vPhase2[chosen] = 0.23; // slight offset prevents phase-lock spike on attack
    this.vSubPhase[chosen] = 0.0;
    this.vInc1[chosen] = clampNum(freq * detMul1 * this.invSr, 0.0001, 0.45);
    this.vInc2[chosen] = clampNum(freq * detMul2 * this.invSr, 0.0001, 0.45);
    this.vSubInc[chosen] = clampNum(freq * 0.5 * this.invSr, 0.00005, 0.25);
    this.vDualSaw[chosen] = dualSaw;
    this.vSubIsTri[chosen] = voiceType === 0 ? 1 : 0;
    this.vGain[chosen] = peakGain;
    this.vSubMix[chosen] = clampNum(subMix, 0.0, 1.0);
    this.vAge[chosen] = 0;
    this.vAttackSamples[chosen] = attSamples;
    this.vHoldSamples[chosen] = holdSamples;
    this.vAttStep[chosen] = peakGain / attSamples;
    this.vDecayMul[chosen] = Math.exp(-1.0 / Math.max(16.0, safeDur * 1.45 * this.sampleRate));
    this.vReleaseMul[chosen] = Math.exp(-3.5 / Math.max(16.0, relSec * this.sampleRate));
    this.vAmp[chosen] = 0.0001;

    this.vBaseCut[chosen] = baseFreq;
    this.vPeakCut[chosen] = peakFreq;
    if (voiceType === 3) {
      var riseSec = Math.min(0.28, safeDur * 0.45);
      var riseControlTicks = Math.max(1, Math.round((riseSec * this.sampleRate) / 8.0));
      this.vCutRiseSamples[chosen] = riseControlTicks * 8;
      this.vCurCut[chosen] = baseFreq;
      this.vCutRiseStep[chosen] = (peakFreq - baseFreq) / riseControlTicks;
      this.vCutDecayMul[chosen] = Math.exp(-8.0 / Math.max(32.0, (safeDur - riseSec + 0.1) * 0.55 * this.sampleRate));
    } else {
      this.vCutRiseSamples[chosen] = 0;
      this.vCurCut[chosen] = peakFreq;
      this.vCutRiseStep[chosen] = 0.0;
      this.vCutDecayMul[chosen] = Math.exp(-8.0 / Math.max(32.0, safeDur * 0.42 * this.sampleRate));
    }

    this.vFilterK[chosen] = 1.0 / safeQ;
    this.vIc1eq[chosen] = 0.0;
    this.vIc2eq[chosen] = 0.0;
    this.updateVoiceSvfCoeffs(chosen);
  }

  updateVoiceSvfCoeffs(v) {
    var fc = clampNum(this.vCurCut[v], 40.0, 0.42 * this.sampleRate);
    var g = Math.tan(Math.PI * fc * this.invSr);
    var k = this.vFilterK[v];
    var a1 = 1.0 / (1.0 + g * (g + k));
    this.vA1[v] = a1;
    this.vA2[v] = g * a1;
    this.vA3[v] = g * this.vA2[v];
  }

  triggerStep(step) {
    var bar = Math.floor(step / 16);
    var beat = ((step % 16) + 16) % 16;
    if (bar !== this.currentBar) {
      this.currentBar = bar;
      this.updateStreamTargetFreqs();
    }
    this.currentStep = step;

    var pos = ((bar % 64) + 64) % 64;
    var sparse = (pos >= 32 && pos < 40) || !this.obsActive;
    var opening = pos < 8 && (bar % 8) < 4;
    var surgeOrClimax = (pos >= 24 && pos < 32) || pos >= 48;

    var darkBias = this.obsHalogenStress * 0.45 + this.envNightFactor * 0.45 + this.obsMineralSulfideRatio * 0.35;
    var usePhrygian = darkBias > 0.36;
    var prog = usePhrygian ? PROG_PHRYGIAN : PROG_CYBER;
    var chordOffset = prog[((bar >> 1) & 3)];

    var massRatio = Math.max(1.0, this.obsTotalBioMass / 25.0);
    var massShift = Math.min(12, Math.max(0, Math.floor(Math.log2(massRatio)) * 3));
    var nightDrop = this.envNightFactor > 0.55 ? 12 : 0;
    var transpose = -Math.min(12, Math.max(nightDrop, massShift > 6 ? 12 : 0));

    var bassRoot = 38 + chordOffset + (transpose <= -12 ? -12 : 0);
    var leadRoot = 50 + chordOffset + transpose;
    var stepSec = this.stepSec;

    // 1. Driving Cyber Kick
    if (!sparse && (beat & 3) === 0) {
      this.spawnDrum(0, 4, 36, this.obsActive ? 0.80 + 0.14 * this.obsIntensity : 0.42);
    } else if (sparse && (beat === 0 || beat === 8)) {
      this.spawnDrum(0, 4, 36, 0.46);
    }

    // 2. Gated Snare
    if (!sparse && !opening) {
      if (beat === 4 || beat === 12) {
        this.spawnDrum(2, 4, 60, 0.48 + 0.16 * (1.0 - this.envOceanFraction * 0.4) + 0.14 * this.obsConsumerRatio);
      } else if (beat === 15 && this.obsConsumerRatio > 0.30) {
        this.spawnDrum(2, 4, 58, 0.22 * this.obsConsumerRatio);
      }
    }

    // 3. Dark Hi-Hats
    var hatEnergy = Math.min(1.0, this.envWindSpeedNorm * 0.65 + this.obsChemBondActivity * 0.55);
    if (!sparse) {
      var offbeat = (beat & 3) === 2;
      var roll = hatEnergy > 0.42 && (beat & 1) === 1 && ((step + bar) % 3) !== 0;
      if (offbeat || roll) {
        this.spawnDrum(1, 4, offbeat ? 60 : 63, (offbeat ? 0.25 : 0.13) * (0.45 + 0.72 * hatEnergy));
      }
    }

    // 4. Tectonic Dark Tom & Sulfide Anvil
    var ridgeActivity = Math.max(this.envRidgeMask, this.envElevKmNorm, this.obsMineralSulfideRatio);
    if (ridgeActivity > 0.24 && (beat === 3 || beat === 11)) {
      var note4 = beat === 3 ? clampNum(bassRoot, 26, 46) : clampNum(bassRoot + 12, 38, 54);
      this.spawnDrum(beat === 3 ? 15 : 3, 4, note4, Math.min(0.36, 0.14 + 0.22 * ridgeActivity));
    }

    // 5. Sequenced Saw+Sub Bassline with Per-Note Filter Pluck
    var grooveIdx =
      this.obsConsumerRatio > 0.28
        ? 0
        : this.obsChemBondActivity > 0.38
          ? 1
          : usePhrygian
            ? 2
            : 3;
    var contour = SEQ_BASS_CONTOURS[grooveIdx];
    var bassPlayStep = (!sparse && !opening) || (beat & 1) === 0;
    if (bassPlayStep) {
      var deg = contour[beat];
      var bassNote = clampNum(bassRoot + deg, 26, 60);
      var accent = (beat & 3) === 0 || deg >= 10;
      var nightDark = 1.0 - 0.52 * this.envNightFactor;
      var baseCutoff = (170.0 + 260.0 * this.envTempNorm * nightDark + 180.0 * this.obsProducerRatio) * (sparse ? 0.72 : 1.0);
      var envPeakHz =
        (accent ? 2100.0 : 1280.0) *
        (0.55 + 0.55 * this.obsIntensity + 0.45 * this.obsChemBondActivity + 0.35 * this.obsHalogenStress) *
        (0.65 + 0.45 * nightDark);
      var filterQ = 2.6 + 4.2 * Math.min(1.0, this.obsHalogenStress * 0.6 + this.obsChemBondActivity * 0.5);
      var subMix = Math.min(0.95, 0.55 + 0.25 * this.envOceanFraction + 0.20 * this.envNightFactor);
      var detuneCents = 9.0 + 14.0 * Math.max(this.obsComplexRatio, this.obsChemVaporRatio);

      this.spawnSynth(
        0,
        4,
        bassNote,
        (sparse ? 0.35 : accent ? 0.54 : 0.42) * (0.85 + 0.25 * this.obsIntensity),
        accent ? stepSec * 1.15 : stepSec * 0.85,
        baseCutoff,
        Math.min(4800.0, envPeakHz),
        filterQ,
        subMix,
        detuneCents
      );
    }

    // 6. Dark Vapor Atmosphere Pad
    if (beat === 0 || (beat === 8 && (this.obsChemVaporRatio > 0.35 || this.envRhNorm > 0.52))) {
      this.spawnSynth(
        3,
        beat === 0 ? 1 : 2,
        leadRoot + (beat === 8 ? 7 : 0),
        0.14 + 0.14 * Math.max(this.obsChemVaporRatio, this.envRhNorm, this.envOceanFraction * 0.35),
        stepSec * 6.5,
        240.0 + 380.0 * (1.0 - this.envNightFactor * 0.5),
        620.0 + 680.0 * this.envTempNorm,
        1.4,
        0.35,
        18.0
      );
    }

    // 7. Dominant Bio Lineage Synth Lead
    var motifSlot = MOTIF_SLOT_FOR_BEAT[beat];
    if (this.obsTotalBioMass > 1.0 && motifSlot >= 0 && (!sparse || beat === 0 || beat === 10)) {
      this.updateMotifIfNeeded();
      var leadDeg = this.motifBuf[motifSlot];
      var isComplex = this.obsComplexRatio > 0.35;
      this.spawnSynth(
        2,
        this.obsDominantBioSp & 3,
        clampNum(leadRoot + 12 + leadDeg, 40, 72),
        (surgeOrClimax ? 0.22 : 0.16) * Math.min(1.2, 0.5 + this.obsAvgEnergy * 0.7),
        isComplex ? stepSec * 2.4 : stepSec * 1.4,
        280.0 + 380.0 * (1.0 - this.envNightFactor * 0.55),
        1100.0 + 1400.0 * this.obsConsumerRatio + 700.0 * this.obsComplexRatio,
        isComplex ? 2.8 : 2.0,
        isComplex ? 0.58 : 0.28,
        isComplex ? 20.0 : 11.0
      );
    }

    // 8. Spatial Stereo Lanes (Chemistry Arps + Mineral Pulses)
    for (var l = 0; l < 4; l++) {
      var chemG = this.laneChemGain[l];
      if (chemG > 0.08 && (beat & 3) === l) {
        var arpDeg = PHRYGIAN_DEGREES[(bar + beat + l * 2) & 7];
        this.spawnSynth(
          1,
          l,
          clampNum(leadRoot + arpDeg, 43, 68),
          0.13 * chemG * (0.55 + 0.65 * this.obsChemBondActivity),
          stepSec * 0.95,
          260.0,
          Math.min(1850.0, 780.0 + 1250.0 * (1.0 - this.envNightFactor * 0.55) * this.obsChemBondActivity),
          1.8 + 2.4 * this.laneAcidDrive[l],
          0.32,
          11.0
        );
      }

      var minG = this.laneMineralGain[l];
      if (minG > 0.08 && ((beat === 6 && (l & 1) === 0) || (beat === 14 && (l & 1) === 1))) {
        var isSulf = this.laneMinSulfFrac[l] > 0.42;
        this.spawnDrum(
          isSulf ? 15 : 14,
          l,
          isSulf ? clampNum(bassRoot, 26, 48) : clampNum(leadRoot + (usePhrygian ? 1 : 7), 40, 62),
          (isSulf ? 0.20 : 0.14) * minG
        );
      }
    }
  }

  process(outL, outR) {
    var len = outL.length;
    var stereo = outR !== outL;

    if (!this.active && this.playGain <= 0.0001) {
      outL.fill(0);
      if (stereo) outR.fill(0);
      this.rms *= 0.7;
      this.bassLevel *= 0.7;
      this.polyLevel *= 0.7;
      this.chorusLevel *= 0.7;
      this.delayLevel *= 0.7;
      this.lowBassEnergy *= 0.7;
      this.sidechainDepth *= 0.7;
      return;
    }

    // Smooth lane pan & send parameters once per quantum
    for (var l = 0; l < 5; l++) {
      var p = this.lanePan[l] + (this.laneTargetPan[l] - this.lanePan[l]) * 0.12;
      this.lanePan[l] = p;
      var ang = (p + 1.0) * 0.25 * Math.PI;
      this.lanePanL[l] = Math.cos(ang);
      this.lanePanR[l] = Math.sin(ang);
      this.laneChorusSend[l] += (this.laneTargetChorusSend[l] - this.laneChorusSend[l]) * 0.12;
      this.laneDelaySend[l] += (this.laneTargetDelaySend[l] - this.laneDelaySend[l]) * 0.12;
    }

    var masterSqAcc = 0.0;
    var bassSqAcc = 0.0;
    var polySqAcc = 0.0;
    var chorusSqAcc = 0.0;
    var delaySqAcc = 0.0;

    var invSr = this.invSr;
    var sr = this.sampleRate;
    var chorusWidth = this.curChorusWidth;
    var delayFb = clampNum(this.curDelayFeedback, 0.0, 0.44);
    var delayLpAlpha = this.delayLpAlpha;
    var chorusLfoInc = 0.42 * invSr;
    var streamSat = 1.0 + this.curAcidDrive * 0.65;
    var streamHarmWWind = 0.36 + 0.30 * this.curAcidDrive;
    var streamHarmWOcean = 0.20 + 0.16 * this.curAcidDrive;
    var targetPlayGain = this.active ? 1.0 : 0.0;
    var targetStreamMaster = (this.active && this.streamlinesEnabled) ? 1.0 : 0.0;

    for (var n = 0; n < len; n++) {
      // Sample-accurate 120 BPM sequencer trigger
      if (this.sampleCursor >= this.nextStepSample) {
        this.triggerStep(this.nextStep);
        this.nextStep++;
        var nextSwing = (this.nextStep & 1) === 1 ? 0.035 : 0.0;
        this.nextStepSample = Math.round((this.nextStep + nextSwing) * this.samplesPerStep);
      }

      // Kick sidechain ducking envelope
      if (this.duckStage === 1) {
        this.duckVal -= this.duckAttStep;
        if (this.duckVal <= 0.36) {
          this.duckVal = 0.36;
          this.duckStage = 2;
        }
      } else if (this.duckStage === 2) {
        this.duckVal += (1.0 - this.duckVal) * this.duckRelCoeff;
        if (this.duckVal >= 0.998) {
          this.duckVal = 1.0;
          this.duckStage = 0;
        }
      }

      this.playGain += (targetPlayGain - this.playGain) * this.playRampCoeff;
      this.streamMasterAmp += (targetStreamMaster - this.streamMasterAmp) * 0.004;

      // Clear 5 mono lane accumulators
      this.laneAcc[0] = 0.0;
      this.laneAcc[1] = 0.0;
      this.laneAcc[2] = 0.0;
      this.laneAcc[3] = 0.0;
      this.laneAcc[4] = 0.0;
      var dryKick = 0.0;

      // 1. Render active drum voices
      for (var d = 0; d < MAX_DRUM_VOICES; d++) {
        if (!this.dActive[d]) continue;
        this.dAge[d]++;
        var buf = this.drumBuffers[this.dSlot[d]];
        var pos = this.dPos[d];
        var idx = pos | 0;
        if (idx >= buf.length - 1) {
          this.dActive[d] = 0;
          continue;
        }
        var frac = pos - idx;
        var ds = (buf[idx] + (buf[idx + 1] - buf[idx]) * frac) * this.dGain[d];
        this.dPos[d] = pos + this.dRate[d];
        if (this.dIsKick[d]) {
          dryKick += ds;
        } else {
          this.laneAcc[this.dLane[d]] += ds;
        }
      }

      // 2. Render active polyphonic synth voices
      var ctrlTick = (n & 7) === 0;
      for (var v = 0; v < MAX_SYNTH_VOICES; v++) {
        if (!this.vActive[v]) continue;
        var age = this.vAge[v]++;
        var stage = this.vStage[v];
        var amp = this.vAmp[v];

        if (stage === 0) {
          amp += this.vAttStep[v];
          if (age >= this.vAttackSamples[v]) {
            amp = this.vGain[v];
            this.vStage[v] = 1;
          }
        } else if (stage === 1) {
          amp *= this.vDecayMul[v];
          if (age >= this.vHoldSamples[v]) {
            this.vStage[v] = 2;
          }
        } else {
          amp *= this.vReleaseMul[v];
          if (amp < 0.0001) {
            this.vActive[v] = 0;
            this.vAmp[v] = 0.0;
            continue;
          }
        }
        this.vAmp[v] = amp;

        if (ctrlTick && n > 0) {
          if (this.vType[v] === 3 && age < this.vCutRiseSamples[v]) {
            this.vCurCut[v] = Math.min(this.vPeakCut[v], this.vCurCut[v] + this.vCutRiseStep[v]);
          } else {
            this.vCurCut[v] = this.vBaseCut[v] + (this.vCurCut[v] - this.vBaseCut[v]) * this.vCutDecayMul[v];
          }
          this.updateVoiceSvfCoeffs(v);
        }

        // Anti-aliased sawtooth 1
        var p1 = this.vPhase1[v] + this.vInc1[v];
        if (p1 >= 1.0) p1 -= 1.0;
        this.vPhase1[v] = p1;
        var saw = (2.0 * p1 - 1.0) - polyBlep(p1, this.vInc1[v]);

        if (this.vDualSaw[v]) {
          var p2 = this.vPhase2[v] + this.vInc2[v];
          if (p2 >= 1.0) p2 -= 1.0;
          this.vPhase2[v] = p2;
          var saw2 = (2.0 * p2 - 1.0) - polyBlep(p2, this.vInc2[v]);
          saw = (saw + saw2) * 0.55;
        }

        // Sub oscillator (triangle for bass, sine for lead/pad/arp)
        var sp = this.vSubPhase[v] + this.vSubInc[v];
        if (sp >= 1.0) sp -= 1.0;
        this.vSubPhase[v] = sp;
        var subSig = this.vSubIsTri[v]
          ? (4.0 * Math.abs(sp - 0.5) - 1.0)
          : fastSin01(sp);

        var oscIn = saw + subSig * this.vSubMix[v] * 0.72;

        // Cytomic Trapezoidal SVF Lowpass
        var ic1 = this.vIc1eq[v];
        var ic2 = this.vIc2eq[v];
        var v3 = oscIn - ic2;
        var v1 = this.vA1[v] * ic1 + this.vA2[v] * v3;
        var v2 = ic2 + this.vA2[v] * ic1 + this.vA3[v] * v3;
        var nextIc1 = 2.0 * v1 - ic1;
        var nextIc2 = 2.0 * v2 - ic2;
        // Flush denormals/NaNs cleanly
        this.vIc1eq[v] = (nextIc1 > 1e-18 || nextIc1 < -1e-18) ? nextIc1 : 0.0;
        this.vIc2eq[v] = (nextIc2 > 1e-18 || nextIc2 < -1e-18) ? nextIc2 : 0.0;

        this.laneAcc[this.vLane[v]] += v2 * amp;
      }

      // 3. Per-lane waveshaping + constant-power stereo panning + FX sends
      var duckBusL = 0.0;
      var duckBusR = 0.0;
      var chorusIn = 0.0;
      var delayIn = 0.0;

      for (var lane = 0; lane < 5; lane++) {
        var raw = this.laneAcc[lane];
        var shaped = raw !== 0.0 ? fastTanh(raw * this.laneK[lane]) * this.laneInvNorm[lane] : 0.0;
        var outLL = shaped * this.lanePanL[lane];
        var outLR = shaped * this.lanePanR[lane];
        duckBusL += outLL;
        duckBusR += outLR;

        var mono = 0.5 * (outLL + outLR);
        chorusIn += mono * this.laneChorusSend[lane];
        delayIn += mono * this.laneDelaySend[lane];

        if (lane === 4) {
          bassSqAcc += mono * mono;
        } else {
          polySqAcc += 0.5 * (outLL * outLL + outLR * outLR);
        }
      }

      // 4. 16-band Streamline Resonator (wavetable + fastTanh, zero Math.sin/Math.tanh per sample)
      if (this.streamMasterAmp > 0.0002) {
        var stSumL = 0.0;
        var stSumR = 0.0;
        for (var si = 0; si < 16; si++) {
          var stAmp = this.streamAmps[si] + (this.streamTargetAmps[si] - this.streamAmps[si]) * 0.0035;
          this.streamAmps[si] = stAmp;
          if (stAmp < 0.001) continue;

          var stFreq = this.streamFreqs[si] + (this.streamTargetFreqs[si] - this.streamFreqs[si]) * 0.003;
          this.streamFreqs[si] = stFreq;

          var detRate = (0.0022 + (si & 3) * 0.0008) * 0.0095;
          var dp = this.streamDetunePhases[si] + detRate;
          if (dp >= 1.0) dp -= 1.0;
          this.streamDetunePhases[si] = dp;

          var df = stFreq * (1.0 + 0.0045 * fastSin01(dp));
          var sp2 = this.streamPhases[si] + df * invSr;
          if (sp2 >= 1.0) sp2 -= 1.0;
          this.streamPhases[si] = sp2;

          var tri = 4.0 * Math.abs(sp2 - 0.5) - 1.0;
          var fund = fastSin01(sp2);
          var hw = si < 8 ? streamHarmWWind : streamHarmWOcean;
          var sSig = fastTanh((fund * (1.0 - hw) + tri * hw) * streamSat) * stAmp;
          stSumL += sSig * this.streamPanL[si];
          stSumR += sSig * this.streamPanR[si];
        }
        var stNorm = 0.042 * this.streamMasterAmp * 0.28;
        var stL = stSumL * stNorm;
        var stR = stSumR * stNorm;
        duckBusL += stL;
        duckBusR += stR;
        chorusIn += 0.5 * (stL + stR) * 0.18;
        polySqAcc += 0.5 * (stL * stL + stR * stR);
      }

      // 5. Stereo LFO Modulated Chorus
      this.chorusLfoPhase += chorusLfoInc;
      if (this.chorusLfoPhase >= 1.0) this.chorusLfoPhase -= 1.0;
      var lfoVal = fastSin01(this.chorusLfoPhase);

      this.chorusBuf[this.chorusWrite] = chorusIn;
      var dSamL = (0.015 + 0.0032 * lfoVal) * sr;
      var dSamR = (0.022 - 0.0032 * lfoVal) * sr;

      var cIdxLF = this.chorusWrite - dSamL;
      var cIdxL0 = Math.floor(cIdxLF);
      var cFracL = cIdxLF - cIdxL0;
      var tapL = this.chorusBuf[cIdxL0 & CHORUS_MASK] * (1.0 - cFracL) + this.chorusBuf[(cIdxL0 + 1) & CHORUS_MASK] * cFracL;

      var cIdxRF = this.chorusWrite - dSamR;
      var cIdxR0 = Math.floor(cIdxRF);
      var cFracR = cIdxRF - cIdxR0;
      var tapR = this.chorusBuf[cIdxR0 & CHORUS_MASK] * (1.0 - cFracR) + this.chorusBuf[(cIdxR0 + 1) & CHORUS_MASK] * cFracR;

      this.chorusWrite = (this.chorusWrite + 1) & CHORUS_MASK;

      var cWetL = tapL * chorusWidth;
      var cWetR = tapR * chorusWidth;
      duckBusL += cWetL * this.chorusGainLL + cWetR * this.chorusGainRL;
      duckBusR += cWetL * this.chorusGainLR + cWetR * this.chorusGainRR;
      chorusSqAcc += 0.5 * (cWetL * cWetL + cWetR * cWetR);

      // 6. Dotted-8th Lowpass Feedback Delay
      var delayOut = this.delayBuf[(this.delayWrite - this.delaySamples) & DELAY_MASK];
      this.delayLpState += (delayOut - this.delayLpState) * delayLpAlpha;
      var lpSig = this.delayLpState;
      var fbWrite = delayIn + lpSig * delayFb;
      this.delayBuf[this.delayWrite] = fbWrite > 2.0 ? 2.0 : fbWrite < -2.0 ? -2.0 : fbWrite;
      this.delayWrite = (this.delayWrite + 1) & DELAY_MASK;

      duckBusL += lpSig;
      duckBusR += lpSig;
      delaySqAcc += lpSig * lpSig;

      // 7. Sidechain Pump + Dry Kick + Soft Limiter
      var duck = this.duckVal;
      var mixL = duckBusL * duck + dryKick * 0.7071;
      var mixR = duckBusR * duck + dryKick * 0.7071;

      var sOutL = fastTanh(mixL * 1.15) * 0.92 * this.playGain;
      var sOutR = fastTanh(mixR * 1.15) * 0.92 * this.playGain;

      if (stereo) {
        outL[n] = sOutL;
        outR[n] = sOutR;
      } else {
        outL[n] = 0.5 * (sOutL + sOutR);
      }

      masterSqAcc += 0.5 * (sOutL * sOutL + sOutR * sOutR);
      this.sampleCursor++;
    }

    // Update smoothed per-bus meters (scaled consistently into [0..1])
    var invLen = 1.0 / Math.max(1, len);
    var instRms = Math.sqrt(masterSqAcc * invLen);
    var instBass = Math.min(1.0, Math.sqrt(bassSqAcc * invLen) * 2.6);
    var instPoly = Math.min(1.0, Math.sqrt(polySqAcc * invLen) * 3.8);
    var instChorus = Math.min(1.0, Math.sqrt(chorusSqAcc * invLen) * 3.4);
    var instDelay = Math.min(1.0, Math.sqrt(delaySqAcc * invLen) * 4.0);
    var instDuck = clamp01((1.0 - this.duckVal) / 0.64);
    var instLowBass = Math.min(1.0, instBass * 0.72 + instDuck * 0.35);

    this.rms += (instRms - this.rms) * 0.35;
    this.bassLevel += (instBass - this.bassLevel) * 0.38;
    this.polyLevel += (instPoly - this.polyLevel) * 0.35;
    this.chorusLevel += (instChorus - this.chorusLevel) * 0.35;
    this.delayLevel += (instDelay - this.delayLevel) * 0.35;
    this.sidechainDepth += (instDuck - this.sidechainDepth) * 0.45;
    this.lowBassEnergy += (instLowBass - this.lowBassEnergy) * 0.38;
  }

  fillTelemetry(out) {
    out[0] = this.rms;
    out[1] = this.bassLevel;
    out[2] = this.polyLevel;
    out[3] = this.chorusLevel;
    out[4] = this.delayLevel;
    out[5] = this.sidechainDepth;
    out[6] = this.lowBassEnergy;
    out[7] = this.sampleCursor;
    out[8] = this.sampleRate;
    out[9] = this.curBassCutoffHz;
    out[10] = this.curChorusWidth;
    out[11] = this.curDelayFeedback;
    out[12] = this.curAcidDrive;
    out[13] = this.streamEnergyAvg;
    out[14] = this.currentBar;
    out[15] = this.currentStep;
  }
}
`;

export const GLOBE_SYNTH_WORKLET_SOURCE = `${GLOBE_SYNTH_CORE_SOURCE}
class GlobeSynthProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.core = new SynthDspCore(sampleRate);
    this.telemetry = new Float32Array(16);
    this.quantumCount = 0;
    var po = options && options.processorOptions;
    if (po) {
      if (po.obs) this.core.applyMessage({ type: 'obs', obs: po.obs });
      if (po.streams) {
        this.core.applyMessage({
          type: 'streams',
          enabled: po.streams.enabled,
          bands: po.streams.bands,
        });
      }
      if (po.state) {
        this.core.applyMessage({
          type: 'state',
          active: po.state.active,
          streamlinesEnabled: po.state.streamlinesEnabled,
        });
      }
    }
    this.port.onmessage = (ev) => {
      this.core.applyMessage(ev.data);
    };
  }

  process(_inputs, outputs) {
    var output = outputs[0];
    if (!output || output.length === 0) return true;
    var outL = output[0];
    var outR = output[1] || outL;
    this.core.process(outL, outR);
    this.quantumCount++;
    if ((this.quantumCount & 31) === 0) {
      this.core.fillTelemetry(this.telemetry);
      this.port.postMessage({ type: 'telemetry', data: this.telemetry });
    }
    return true;
  }
}
registerProcessor('${GLOBE_SYNTH_WORKLET_NAME}', GlobeSynthProcessor);
`;
