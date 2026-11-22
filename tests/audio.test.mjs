import test from 'node:test';
import assert from 'node:assert/strict';
import { observeScreenAudioState } from '../src/audio/observer.ts';
import { generateScoreEvents, lineageMotif } from '../src/audio/score.ts';
import { SYNTH_RATE } from '../src/audio/synthBank.ts';
import { createSpectrogramSnapshot, GlobeCyberSynth, SPECTROGRAM_BINS } from '../src/audio/engine.ts';
import { describeSynthEffects, SYNTH_EFFECT_SPECS } from '../src/audio/spectrogram.ts';
import { buildPackedChemistryBuffer } from '../src/periodicTable2D.ts';

function packMeta(phase, bioSp, graduated = 0) {
  if (phase === 0) return 0;
  return ((phase & 3) | ((bioSp & 255) << 2) | (7 << 19) | ((graduated ? 1 : 0) << 23)) >>> 0;
}

test('observeScreenAudioState tracks center life, chemistry halogens, minerals, and dark night filter', () => {
  const elements = buildPackedChemistryBuffer(14).elements;
  const halIdx = Math.max(0, elements.findIndex((e) => e.atomicNumber === 7 || e.atomicNumber === 13));
  const cIdx = Math.max(0, elements.findIndex((e) => e.atomicNumber === 5));

  const view = {
    projectionId: 'orthographic',
    rotate: [0, 0, 0],
    scale: 1200,
    pan: [0, 0],
    width: 1200,
    height: 800,
  };

  const count = 4;
  const particles = new Float32Array(count * 12);
  const lifeState = new Float32Array(count * 4);

  // Particle 0: Graduated consumer near center
  particles[0] = 1; particles[1] = 0; particles[2] = 0;
  particles[11] = 0.45;
  lifeState[0] = 0.9;
  lifeState[3] = packMeta(1, 5, 1);

  // Particle 1: Reactive halogen chemistry slightly left of center
  particles[12] = 0.996; particles[13] = -0.08; particles[14] = 0;
  particles[15] = halIdx;
  particles[23] = -0.8;
  lifeState[4] = 0.1; lifeState[5] = 0; lifeState[6] = 0.4; lifeState[7] = 0;

  // Particle 2: Sulfide mineral on right lane
  particles[24] = 0.96; particles[25] = 0.25; particles[26] = 0;
  particles[27] = cIdx;
  lifeState[8] = 1.0; lifeState[10] = 0.6; lifeState[11] = packMeta(3, 3, 0);

  // Particle 3: On far side of globe (should be occluded cleanly)
  particles[36] = -1; particles[37] = 0; particles[38] = 0;
  lifeState[12] = 1.0; lifeState[15] = packMeta(1, 2, 1);

  const dayEnv = {
    windSpeedNorm: 0.75,
    oceanFraction: 0.2,
    oceanSpeedNorm: 0.1,
    tempNorm: 0.72,
    nightFactor: 0.0,
    rhNorm: 0.4,
    precipNorm: 0.2,
    elevKmNorm: 0.4,
    ridgeMask: 0.6,
  };
  const nightEnv = { ...dayEnv, nightFactor: 0.95 };

  const obsDay = observeScreenAudioState(particles, lifeState, count, elements, view, null, dayEnv, 1234, false);
  const obsNight = observeScreenAudioState(particles, lifeState, count, elements, view, null, nightEnv, 1234, false);

  assert.equal(obsDay.active, true);
  assert.equal(obsDay.dominantBioSp, 5);
  assert.ok(obsDay.complexRatio > 0.5, 'expected graduated consumer near center to set complexRatio');
  assert.ok(obsDay.consumerRatio > 0.5, 'expected positive consumer metaBias near center');
  assert.ok(obsDay.halogenStress > 0.4, 'expected reactive halogen to lift halogenStress');
  assert.ok(obsDay.mineralSulfideRatio > 0.5, 'expected mType=3 to register as sulfide mineral');
  assert.ok(obsDay.lanes[1].cutoffHz > obsNight.lanes[1].cutoffHz, 'expected night hemisphere to darken filter cutoff');
});

test('generateScoreEvents emits sorted cyberpunk kick, per-note filtered saw+sub bassline, and ridge percussion', () => {
  const obs = observeScreenAudioState(
    new Float32Array(12),
    new Float32Array([0.85, 0.1, 0, packMeta(1, 3, 1)]),
    1,
    buildPackedChemistryBuffer(14).elements,
    { projectionId: 'orthographic', rotate: [0, 0, 0], scale: 1200, pan: [0, 0], width: 1200, height: 800 },
    null,
    {
      windSpeedNorm: 0.85,
      oceanFraction: 0.4,
      oceanSpeedNorm: 0.5,
      tempNorm: 0.6,
      nightFactor: 0.8,
      rhNorm: 0.6,
      precipNorm: 0.3,
      elevKmNorm: 0.5,
      ridgeMask: 0.8,
    },
    9911,
    false,
  );
  obs.active = true;
  obs.totalBioMass = 120;
  obs.chemBondActivity = 0.7;
  obs.halogenStress = 0.6;
  obs.mineralSulfideRatio = 0.7;
  obs.lanes[0].mineralGain = 0.6;
  obs.lanes[0].mineralSulfideFrac = 0.8;
  obs.lanes[1].chemGain = 0.8;
  obs.lanes[2].bioGain = 0.8;

  const frames = Math.round(SYNTH_RATE * 2.0);
  const events = generateScoreEvents(0, frames, obs);

  assert.ok(events.length >= 20, `expected rich cyberpunk groove events, got ${events.length}`);
  let sawKick = false;
  let sawFilteredBass = false;
  let sawAnvilOrRim = false;
  let prevOffset = -1;
  for (const ev of events) {
    assert.ok(ev.offset >= 0 && ev.offset < frames, `offset ${ev.offset} out of [0, ${frames})`);
    assert.ok(ev.offset >= prevOffset, 'events must be sorted by sample offset');
    prevOffset = ev.offset;
    assert.ok(ev.lane >= 0 && ev.lane <= 4);
    assert.ok(ev.gain > 0);
    assert.ok(ev.note <= 72, `expected dark cyberpunk register without high tinks, got MIDI note ${ev.note}`);
    if (ev.kind === 'drum' && ev.instrument === 0) sawKick = true;
    if (ev.kind === 'drum' && (ev.instrument === 3 || ev.instrument === 15)) sawAnvilOrRim = true;
    if (ev.kind === 'synth' && ev.voiceType === 'bass') {
      sawFilteredBass = true;
      assert.ok(ev.envPeakHz > 300, 'expected automatic filter envelope sweep on bass note');
      assert.ok(ev.subMix >= 0.5, 'expected lower sub wave blended underneath saw bassline');
    }
  }
  assert.ok(sawKick, 'expected driving cyber kick');
  assert.ok(sawFilteredBass, 'expected sequenced saw+sub bassline with filter envelope');
  assert.ok(sawAnvilOrRim, 'expected tectonic ridge/sulfide percussion');

  const snap = createSpectrogramSnapshot();
  snap.observation = obs;
  const entries = describeSynthEffects(snap);
  assert.equal(entries.length, SYNTH_EFFECT_SPECS.length);
  for (const entry of entries) {
    assert.ok(entry.spec.color.startsWith('#'));
    assert.ok(entry.provenance.length > 10);
  }

  assert.deepEqual(lineageMotif(9911, 3), lineageMotif(9911, 3));
});

test('GlobeCyberSynth restores masterGain after mute/unmute on already-running AudioContext and reads spectrogram', async () => {
  const targetValues = [];
  let ctxState = 'running';

  class FakeAudioContext {
    constructor() {
      this.currentTime = 1.0;
      this.sampleRate = 48000;
      this.destination = { connect() {} };
    }
    get state() {
      return ctxState;
    }
    addEventListener() {}
    resume() {
      ctxState = 'running';
      return Promise.resolve();
    }
    createBuffer(channels, length, sampleRate) {
      return {
        numberOfChannels: channels,
        length,
        sampleRate,
        getChannelData: () => new Float32Array(length),
      };
    }
    createGain() {
      const node = {
        connect() {},
        gain: {
          value: 1,
          setValueAtTime(v) { node.gain.value = v; },
          linearRampToValueAtTime() {},
          exponentialRampToValueAtTime() {},
          cancelScheduledValues() {},
          setTargetAtTime(v) {
            node.gain.value = v;
            targetValues.push(v);
          },
        },
      };
      return node;
    }
    createDynamicsCompressor() {
      return {
        connect() {},
        threshold: { value: 0 },
        knee: { value: 0 },
        ratio: { value: 0 },
        attack: { value: 0 },
        release: { value: 0 },
      };
    }
    createAnalyser() {
      return {
        connect() {},
        fftSize: 256,
        frequencyBinCount: 128,
        smoothingTimeConstant: 0,
        minDecibels: -86,
        maxDecibels: -12,
        getFloatTimeDomainData(arr) { arr.fill(0); },
        getByteFrequencyData(arr) { arr.fill(96); },
      };
    }
    createDelay() {
      return { connect() {}, delayTime: { value: 0 } };
    }
    createBiquadFilter() {
      return { connect() {}, type: 'lowpass', frequency: { value: 1000 } };
    }
    createStereoPanner() {
      return { connect() {}, pan: { value: 0, setTargetAtTime() {} } };
    }
    createOscillator() {
      return {
        connect() {},
        start() {},
        stop() {},
        type: 'sine',
        frequency: { value: 0 },
        detune: { value: 0 },
      };
    }
    createWaveShaper() {
      return { connect() {}, curve: null };
    }
  }

  const prevWindow = globalThis.window;
  globalThis.window = { AudioContext: FakeAudioContext };
  try {
    const synth = new GlobeCyberSynth();
    synth.setVolume(0.8);
    synth.setEnabled(true, false);
    assert.equal(synth.isRunning(), false);

    await synth.unlock();
    assert.equal(synth.isRunning(), true);
    assert.ok(Math.abs(targetValues.at(-1) - 0.8 * 0.85) < 1e-6);

    synth.setEnabled(false, true);
    assert.equal(synth.isRunning(), false);
    assert.equal(targetValues.at(-1), 0);

    synth.setEnabled(true, true);
    await synth.unlock();
    assert.equal(synth.isRunning(), true);
    assert.ok(Math.abs(targetValues.at(-1) - 0.8 * 0.85) < 1e-6);

    const snap = createSpectrogramSnapshot();
    synth.readSpectrogram(snap);
    assert.equal(snap.running, true);
    assert.equal(snap.curves.length, 6);
    assert.equal(snap.curves[0].length, SPECTROGRAM_BINS);
    assert.ok(snap.bassLevel > 0);
    assert.equal(snap.streamlinesEnabled, true);

    synth.updateStreamlineTelemetry(
      2000,
      new Float32Array([0.4, 0.6, 0.2, 0.8, 0.5, 0.3, 0.7, 0.9]),
      new Float32Array([0.3, 0.5, 0.1, 0.4, 0.6, 0.2, 0.5, 0.8]),
    );
    synth.readSpectrogram(snap);
    assert.ok(snap.streamlineEnergy > 0.3);

    synth.setStreamlinesEnabled(false);
    synth.updateStreamlineTelemetry(2100);
    synth.readSpectrogram(snap);
    assert.equal(snap.streamlinesEnabled, false);
    assert.equal(snap.streamlineEnergy, 0);
  } finally {
    globalThis.window = prevWindow;
  }
});
