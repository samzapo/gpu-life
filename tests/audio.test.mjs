import test from 'node:test';
import assert from 'node:assert/strict';
import { observeScreenAudioState } from '../src/audio/observer.ts';
import { generateScoreEvents, lineageMotif } from '../src/audio/score.ts';
import { SYNTH_RATE } from '../src/audio/synthBank.ts';
import { createSpectrogramSnapshot, GlobeCyberSynth, SPECTROGRAM_BINS } from '../src/audio/engine.ts';
import { GLOBE_SYNTH_CORE_SOURCE, GLOBE_SYNTH_WORKLET_NAME } from '../src/audio/synthWorklet.ts';
import { describeSynthEffects, SYNTH_EFFECT_SPECS, SynthSpectrogramOverlay } from '../src/audio/spectrogram.ts';
import {
  computeSynthStreamWaveAmplitude,
  SYNTH_OCEAN_WAVE_BASE_RAD,
  SYNTH_WIND_WAVE_BASE_RAD,
} from '../src/audio/streamWave.ts';
import { buildPackedChemistryBuffer } from '../src/periodicTable2D.ts';
import {
  buildUrlPartsFromConfig,
  CONFIG_STORAGE_KEY,
  FACTORY_DEFAULTS,
  invalidateLegacyBrowserCaches,
  loadCachedConfig,
  parseConfigInput,
} from '../src/config.ts';
import {
  isFullscreenActive,
  isFullscreenSupported,
  toggleFullscreen,
  wireFullscreenToggle,
} from '../src/fullscreen.ts';

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

test('generateScoreEvents emits sorted kick, per-note filtered saw+sub bassline, and ridge percussion', () => {
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

  assert.ok(events.length >= 20, `expected rich groove events, got ${events.length}`);
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
    assert.ok(ev.note <= 72, `expected dark register without high tinks, got MIDI note ${ev.note}`);
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

  let suspendCalls = 0;
  let resumeCalls = 0;
  const ctxOpts = [];

  class FakeAudioContext {
    constructor(opts) {
      ctxOpts.push(opts);
      this.currentTime = 1.0;
      this.sampleRate = 48000;
      this.destination = { connect() {} };
    }
    get state() {
      return ctxState;
    }
    addEventListener() {}
    suspend() {
      suspendCalls++;
      ctxState = 'suspended';
      return Promise.resolve();
    }
    resume() {
      resumeCalls++;
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
  const prevDocument = globalThis.document;
  let fakeHidden = false;
  globalThis.window = {
    AudioContext: FakeAudioContext,
    setTimeout: (fn) => {
      fn();
      return 1;
    },
    clearTimeout: () => {},
  };
  globalThis.document = {
    get hidden() {
      return fakeHidden;
    },
  };
  try {
    const synth = new GlobeCyberSynth();
    synth.setVolume(0.8);
    synth.setEnabled(true, false);
    assert.equal(synth.isRunning(), false);

    await synth.unlock();
    assert.equal(ctxOpts[0]?.latencyHint, 'playback');
    assert.equal(synth.isRunning(), true);
    assert.ok(Math.abs(targetValues.at(-1) - 0.8 * 0.85) < 1e-6);

    // Background tab hide -> suspends clean audio sink; return -> cycles & restores running state cleanly
    fakeHidden = true;
    synth.handleVisibilityChange(true);
    assert.equal(suspendCalls, 1);
    assert.equal(ctxState, 'suspended');

    fakeHidden = false;
    synth.handleVisibilityChange(false);
    await new Promise((r) => setTimeout(r, 10));
    assert.ok(resumeCalls >= 1);
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
    globalThis.document = prevDocument;
  }
});

test('invalidateLegacyBrowserCaches removes stale v1/v2 config entries on epoch upgrade', () => {
  const store = new Map([
    ['complexLife:config:v1', '{"seed":"old1"}'],
    ['complexLife:config:v2', '{"seed":"old2"}'],
    ['complexLife:otherOld', 'stale'],
    ['unrelatedKey', 'keep'],
  ]);
  const prevWindow = globalThis.window;
  globalThis.window = {
    localStorage: {
      getItem: (k) => store.get(k) ?? null,
      setItem: (k, v) => { store.set(k, String(v)); },
      removeItem: (k) => { store.delete(k); },
      get length() { return store.size; },
      key: (i) => Array.from(store.keys())[i] ?? null,
    },
  };
  const origKeys = Object.keys;
  Object.keys = (obj) => (obj === globalThis.window.localStorage ? Array.from(store.keys()) : origKeys(obj));
  try {
    invalidateLegacyBrowserCaches();
    assert.equal(CONFIG_STORAGE_KEY, 'complexLife:config:v3');
    assert.equal(loadCachedConfig(), null);
    assert.equal(store.has('complexLife:config:v1'), false);
    assert.equal(store.has('complexLife:config:v2'), false);
    assert.equal(store.has('complexLife:otherOld'), false);
    assert.equal(store.get('complexLife:cacheEpoch'), '3');
    assert.equal(store.get('unrelatedKey'), 'keep');
  } finally {
    Object.keys = origKeys;
    globalThis.window = prevWindow;
  }
});

test('GlobeCyberSynth preserves bar continuity across long frame stalls and background resume without scheduling past notes', async () => {
  const startedEvents = [];
  const activeMockVoices = [];
  let ctxState = 'running';
  let fakeTime = 1.0;

  function advanceFakeTime(dtSec) {
    fakeTime += dtSec;
    for (let i = activeMockVoices.length - 1; i >= 0; i--) {
      const v = activeMockVoices[i];
      if (v.started && fakeTime >= v.endAt) {
        activeMockVoices.splice(i, 1);
        v.node.onended?.();
      }
    }
  }

  function makeParam(initial = 0) {
    return {
      value: initial,
      setValueAtTime(v) { this.value = v; },
      linearRampToValueAtTime() {},
      exponentialRampToValueAtTime() {},
      cancelScheduledValues() {},
      setTargetAtTime(v) { this.value = v; },
    };
  }

  class StallTestAudioContext {
    constructor() {
      this.sampleRate = 48000;
      this.outputLatency = 0.08;
      this.destination = { connect() {} };
    }
    get currentTime() { return fakeTime; }
    get state() { return ctxState; }
    addEventListener() {}
    suspend() { ctxState = 'suspended'; return Promise.resolve(); }
    resume() { ctxState = 'running'; return Promise.resolve(); }
    createBuffer(channels, length, sampleRate) {
      return {
        numberOfChannels: channels,
        length,
        sampleRate,
        getChannelData: () => new Float32Array(length),
      };
    }
    createBufferSource() {
      const voice = { node: null, started: false, endAt: Infinity };
      const node = {
        connect() {},
        disconnect() {},
        buffer: null,
        loop: false,
        playbackRate: makeParam(1),
        onended: null,
        start(when = 0) {
          startedEvents.push({ kind: 'buf', when, now: fakeTime });
          voice.started = true;
          voice.endAt = when + 0.35;
          activeMockVoices.push(voice);
        },
        stop(when = fakeTime) {
          voice.endAt = Math.min(voice.endAt, when);
        },
      };
      voice.node = node;
      return node;
    }
    createGain() {
      return { connect() {}, disconnect() {}, gain: makeParam(1) };
    }
    createDynamicsCompressor() {
      return {
        connect() {},
        disconnect() {},
        threshold: makeParam(0),
        knee: makeParam(0),
        ratio: makeParam(0),
        attack: makeParam(0),
        release: makeParam(0),
      };
    }
    createAnalyser() {
      return {
        connect() {},
        disconnect() {},
        fftSize: 256,
        frequencyBinCount: 128,
        smoothingTimeConstant: 0,
        minDecibels: -86,
        maxDecibels: -12,
        getFloatTimeDomainData(arr) { arr.fill(0); },
        getByteFrequencyData(arr) { arr.fill(64); },
      };
    }
    createDelay() {
      return { connect() {}, disconnect() {}, delayTime: makeParam(0) };
    }
    createBiquadFilter() {
      return { connect() {}, disconnect() {}, type: 'lowpass', frequency: makeParam(1000), Q: makeParam(1) };
    }
    createStereoPanner() {
      return { connect() {}, disconnect() {}, pan: makeParam(0) };
    }
    createOscillator() {
      const voice = { node: null, started: false, endAt: Infinity };
      const node = {
        connect() {},
        disconnect() {},
        type: 'sine',
        frequency: makeParam(0),
        detune: makeParam(0),
        onended: null,
        start(when = 0) {
          startedEvents.push({ kind: 'osc', when, now: fakeTime });
          voice.started = true;
          activeMockVoices.push(voice);
        },
        stop(when = fakeTime + 0.25) {
          voice.endAt = when;
        },
      };
      voice.node = node;
      return node;
    }
    createWaveShaper() {
      return { connect() {}, disconnect() {}, curve: null };
    }
  }

  const prevWindow = globalThis.window;
  const prevDocument = globalThis.document;
  let fakeHidden = false;
  let timerSeq = 1;
  globalThis.window = {
    AudioContext: StallTestAudioContext,
    setTimeout: (fn) => { fn(); return timerSeq++; },
    clearTimeout: () => {},
    setInterval: () => timerSeq++,
    clearInterval: () => {},
  };
  globalThis.document = {
    get hidden() { return fakeHidden; },
  };

  let synth;
  try {
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

    synth = new GlobeCyberSynth();
    synth.setObservation(obs);
    synth.setEnabled(true, true);
    await synth.unlock();

    // Advance normal frames past bar 8 (~17s -> 'drive' section, cursor well above 0)
    for (let i = 0; i < 86; i++) {
      advanceFakeTime(0.2);
      synth.tick(1000 + i * 200);
    }

    const cursorBeforeStall = synth.frameCursor;
    const snapBeforeStall = createSpectrogramSnapshot();
    synth.readSpectrogram(snapBeforeStall);
    assert.ok(cursorBeforeStall > SYNTH_RATE * 15, `expected frameCursor > 15s, got ${cursorBeforeStall}`);
    assert.equal(snapBeforeStall.section, 'drive');

    // Simulate a 1.8s frame stall (e.g., mobile frame hang): cursor must NOT jump back to 0 / 'emerge',
    // and overdue notes must NOT be scheduled at or before `now` in a past-time burst.
    const countBeforeStall = startedEvents.length;
    advanceFakeTime(1.8);
    synth.tick(19000);

    const cursorAfterStall = synth.frameCursor;
    const snapAfterStall = createSpectrogramSnapshot();
    synth.readSpectrogram(snapAfterStall);
    assert.ok(
      cursorAfterStall > cursorBeforeStall,
      `frameCursor regressed after stall: ${cursorBeforeStall} -> ${cursorAfterStall}`,
    );
    assert.equal(snapAfterStall.section, 'drive');

    const postStallEvents = startedEvents.slice(countBeforeStall);
    assert.ok(postStallEvents.length > 0, 'expected new lookahead events scheduled after stall');
    for (const ev of postStallEvents) {
      assert.ok(ev.when >= ev.now, `scheduled note in the past: when=${ev.when} < now=${ev.now}`);
    }

    // Background hide -> resume should preserve musical bar cursor rather than resetting to bar 0 / 'emerge'
    fakeHidden = true;
    synth.handleVisibilityChange(true);
    advanceFakeTime(3.0);
    fakeHidden = false;
    synth.handleVisibilityChange(false);
    await new Promise((r) => setTimeout(r, 10));
    synth.tick(22200);

    const cursorAfterResume = synth.frameCursor;
    const snapAfterResume = createSpectrogramSnapshot();
    synth.readSpectrogram(snapAfterResume);
    assert.ok(
      cursorAfterResume >= cursorAfterStall,
      `frameCursor reset after background resume: ${cursorAfterStall} -> ${cursorAfterResume}`,
    );
    assert.equal(snapAfterResume.section, 'drive');
  } finally {
    synth?.setEnabled(false, true);
    globalThis.window = prevWindow;
    globalThis.document = prevDocument;
  }
});

function loadSynthDspCoreClass() {
  return new Function(`${GLOBE_SYNTH_CORE_SOURCE}\nreturn SynthDspCore;`)();
}

function makeRichTestObs() {
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
      nightFactor: 0.35,
      rhNorm: 0.65,
      precipNorm: 0.4,
      elevKmNorm: 0.5,
      ridgeMask: 0.6,
    },
    9911,
    false,
  );
  obs.active = true;
  obs.totalBioMass = 180;
  obs.consumerRatio = 0.35;
  obs.chemBondActivity = 0.6;
  obs.chemVaporRatio = 0.45;
  obs.halogenStress = 0.45;
  obs.mineralCrystalRatio = 0.4;
  obs.mineralSulfideRatio = 0.35;
  obs.lanes[0].chemGain = 0.7;
  obs.lanes[0].bioGain = 0.6;
  obs.lanes[0].mineralGain = 0.5;
  obs.lanes[1].chemGain = 0.8;
  obs.lanes[1].bioGain = 0.7;
  obs.lanes[2].bioGain = 0.75;
  obs.lanes[3].chemGain = 0.6;
  obs.lanes[3].mineralGain = 0.5;
  return obs;
}

test('SynthDspCore renders continuous sample-accurate stereo audio below real-time budget with zero allocations', () => {
  const SynthDspCore = loadSynthDspCoreClass();
  const sampleRate = 48000;
  const core = new SynthDspCore(sampleRate);
  const obs = makeRichTestObs();

  core.applyMessage({ type: 'obs', obs });
  core.applyMessage({
    type: 'streams',
    enabled: true,
    bands: new Float32Array([0.4, 0.6, 0.3, 0.7, 0.5, 0.3, 0.8, 0.6, 0.3, 0.5, 0.4, 0.6, 0.7, 0.2, 0.5, 0.8]),
  });
  core.applyMessage({ type: 'state', active: true, streamlinesEnabled: true });

  const blockLen = 128;
  const outL = new Float32Array(blockLen);
  const outR = new Float32Array(blockLen);
  const telemetry = new Float32Array(20);
  const quanta = 2500; // 2500 * 128 = 320,000 samples (6.667s across bars 0..3)

  // Warmup 16 quanta before benchmarking JIT speed
  for (let q = 0; q < 16; q++) {
    core.process(outL, outR);
  }
  core.applyMessage({ type: 'state', active: true, streamlinesEnabled: true, clearFx: true, resetClock: true });

  let sumSq = 0;
  let maxAbs = 0;
  let maxAbsDiff = 0;
  let lowTrMin = 1;
  let lowTrMax = 0;
  let highTrMin = 1;
  let highTrMax = 0;
  const t0 = performance.now();
  for (let q = 0; q < quanta; q++) {
    core.process(outL, outR);
    if ((q & 7) === 0 && q > 64) {
      core.fillTelemetry(telemetry);
      lowTrMin = Math.min(lowTrMin, telemetry[17]);
      lowTrMax = Math.max(lowTrMax, telemetry[17]);
      highTrMin = Math.min(highTrMin, telemetry[18]);
      highTrMax = Math.max(highTrMax, telemetry[18]);
    }
    for (let i = 0; i < blockLen; i++) {
      const l = outL[i];
      const r = outR[i];
      assert.ok(Number.isFinite(l) && Number.isFinite(r), `non-finite audio sample at q=${q}, i=${i}`);
      const al = Math.abs(l);
      const ar = Math.abs(r);
      if (al > maxAbs) maxAbs = al;
      if (ar > maxAbs) maxAbs = ar;
      const diff = Math.abs(l - r);
      if (diff > maxAbsDiff) maxAbsDiff = diff;
      sumSq += 0.5 * (l * l + r * r);
    }
  }
  const elapsedMs = performance.now() - t0;
  const msPerQuantum = elapsedMs / quanta;
  const rms = Math.sqrt(sumSq / (quanta * blockLen));

  core.fillTelemetry(telemetry);
  const expectedSamples = quanta * blockLen;
  const expectedStep = Math.floor(expectedSamples / ((sampleRate * 60) / 120 / 4));

  assert.ok(maxAbs <= 1.0, `peak exceeded 1.0: ${maxAbs}`);
  assert.ok(rms > 0.04, `expected strong audible stereo RMS > 0.04, got ${rms}`);
  assert.ok(maxAbsDiff > 0.01, `expected true stereo separation > 0.01, got ${maxAbsDiff}`);
  assert.ok(lowTrMax - lowTrMin > 0.45, `expected crisp narrow low-band beat contrast, got [${lowTrMin}, ${lowTrMax}]`);
  assert.ok(highTrMax - highTrMin > 0.30, `expected crisp narrow high-band beat contrast, got [${highTrMin}, ${highTrMax}]`);
  assert.equal(telemetry[7], expectedSamples, `expected exact sampleCursor=${expectedSamples}`);
  assert.equal(telemetry[8], sampleRate);
  assert.equal(telemetry[15], expectedStep, `expected sample-accurate step=${expectedStep}`);
  assert.ok(telemetry[1] > 0, `expected non-zero bassLevel, got ${telemetry[1]}`);
  assert.ok(telemetry[2] > 0, `expected non-zero polyLevel, got ${telemetry[2]}`);
  assert.ok(telemetry[3] > 0, `expected non-zero chorusLevel, got ${telemetry[3]}`);
  assert.ok(telemetry[4] > 0, `expected non-zero delayLevel, got ${telemetry[4]}`);
  assert.ok(telemetry[5] >= 0, `expected valid sidechainDepth, got ${telemetry[5]}`);
  assert.ok(telemetry[6] > 0, `expected non-zero lowBassEnergy, got ${telemetry[6]}`);
  assert.ok(
    msPerQuantum < 0.8,
    `expected < 0.8 ms per 128-sample quantum in Node V8, got ${msPerQuantum.toFixed(4)} ms`,
  );

  // Verify pause ramps cleanly to silence and preserves bar/step across clearFx without resetClock
  const barBeforePause = telemetry[14];
  const stepBeforePause = telemetry[15];
  core.applyMessage({ type: 'state', active: false, streamlinesEnabled: false });
  for (let q = 0; q < 60; q++) {
    core.process(outL, outR);
  }
  let tailSq = 0;
  for (let i = 0; i < blockLen; i++) {
    tailSq += 0.5 * (outL[i] * outL[i] + outR[i] * outR[i]);
  }
  const tailRms = Math.sqrt(tailSq / blockLen);
  assert.ok(tailRms < 1e-4, `expected paused tail to fade smoothly to silence, got ${tailRms}`);

  core.fillTelemetry(telemetry);
  const barAfterPause = telemetry[14];
  const stepAfterPause = telemetry[15];
  assert.ok(barAfterPause >= barBeforePause);
  assert.ok(stepAfterPause >= stepBeforePause);

  core.applyMessage({ type: 'state', active: true, streamlinesEnabled: true, clearFx: true, resetClock: false });
  core.fillTelemetry(telemetry);
  assert.equal(telemetry[14], barAfterPause);
  assert.equal(telemetry[15], stepAfterPause);

  core.applyMessage({ type: 'state', active: true, streamlinesEnabled: true, clearFx: true, resetClock: true });
  core.fillTelemetry(telemetry);
  assert.equal(telemetry[7], 0);
  assert.equal(telemetry[14], 0);
  assert.equal(telemetry[15], 0);
});

test('GlobeCyberSynth uses unified single AudioWorklet pipeline with zero main-thread node churn and bar continuity', async () => {
  const SynthDspCore = loadSynthDspCoreClass();
  const prevWindow = globalThis.window;
  const prevDocument = globalThis.document;
  const prevAudioWorkletNode = globalThis.AudioWorkletNode;

  let fakeHidden = false;
  let fakeTime = 1.0;
  let ctxState = 'running';
  let addedModules = [];
  let createdNode = null;
  let postedToWorklet = [];
  const counts = {
    osc: 0,
    shaper: 0,
    delay: 0,
    comp: 0,
    bufSrc: 0,
    analyser: 0,
    setInterval: 0,
  };
  const masterTargets = [];

  class WorkletMockAudioContext {
    constructor() {
      this.sampleRate = 48000;
      this.outputLatency = 0.04;
      this.destination = { connect() {} };
      this.audioWorklet = {
        addModule: async (url) => {
          addedModules.push(url);
        },
      };
    }
    get currentTime() { return fakeTime; }
    get state() { return ctxState; }
    addEventListener() {}
    suspend() { ctxState = 'suspended'; return Promise.resolve(); }
    resume() { ctxState = 'running'; return Promise.resolve(); }
    createGain() {
      const node = {
        connect() {},
        disconnect() {},
        gain: {
          value: 0,
          setValueAtTime(v) { node.gain.value = v; },
          linearRampToValueAtTime() {},
          exponentialRampToValueAtTime() {},
          cancelScheduledValues() {},
          setTargetAtTime(v) {
            node.gain.value = v;
            masterTargets.push(v);
          },
        },
      };
      return node;
    }
    createAnalyser() {
      counts.analyser++;
      return {
        connect() {},
        disconnect() {},
        fftSize: 256,
        frequencyBinCount: 128,
        smoothingTimeConstant: 0,
        minDecibels: -86,
        maxDecibels: -12,
        getFloatTimeDomainData(arr) { arr.fill(0); },
        getByteFrequencyData(arr) { arr.fill(48); },
      };
    }
    createOscillator() { counts.osc++; throw new Error('must not create oscillators in worklet mode'); }
    createWaveShaper() { counts.shaper++; throw new Error('must not create waveshapers in worklet mode'); }
    createDelay() { counts.delay++; throw new Error('must not create DelayNodes in worklet mode'); }
    createDynamicsCompressor() { counts.comp++; throw new Error('must not create DynamicsCompressor in worklet mode'); }
    createBufferSource() { counts.bufSrc++; throw new Error('must not create BufferSourceNodes in worklet mode'); }
    createBuffer() { throw new Error('must not build fallback drum buffers in worklet mode'); }
  }

  class MockAudioWorkletNode {
    constructor(ctx, name) {
      this.ctx = ctx;
      this.name = name;
      this.core = new SynthDspCore(ctx.sampleRate);
      this.outL = new Float32Array(128);
      this.outR = new Float32Array(128);
      this.telemetry = new Float32Array(20);
      this.quantum = 0;
      this.port = {
        onmessage: null,
        postMessage: (msg) => {
          postedToWorklet.push(msg);
          this.core.applyMessage(msg);
        },
      };
      createdNode = this;
    }
    connect() {}
    disconnect() {}
    stepQuanta(n) {
      for (let i = 0; i < n; i++) {
        this.core.process(this.outL, this.outR);
        fakeTime += 128 / this.ctx.sampleRate;
        this.quantum++;
        if ((this.quantum & 7) === 0) {
          this.core.fillTelemetry(this.telemetry);
          this.port.onmessage?.({ data: { type: 'telemetry', data: new Float32Array(this.telemetry) } });
        }
      }
    }
  }

  globalThis.AudioWorkletNode = MockAudioWorkletNode;
  globalThis.window = {
    AudioContext: WorkletMockAudioContext,
    setTimeout: (fn) => { fn(); return 1; },
    clearTimeout: () => {},
    setInterval: () => { counts.setInterval++; return 1; },
    clearInterval: () => {},
  };
  globalThis.document = {
    get hidden() { return fakeHidden; },
  };

  let synth;
  try {
    const obs = makeRichTestObs();
    synth = new GlobeCyberSynth();
    synth.setObservation(obs);
    synth.setVolume(0.75);
    synth.setEnabled(true, true);
    await synth.unlock();

    assert.equal(synth.getAudioMode(), 'worklet');
    assert.equal(synth.isWorkletActive(), true);
    assert.equal(addedModules.length, 1);
    assert.equal(createdNode.name, GLOBE_SYNTH_WORKLET_NAME);
    assert.equal(counts.analyser, 1);
    assert.equal(counts.osc, 0);
    assert.equal(counts.shaper, 0);
    assert.equal(counts.delay, 0);
    assert.equal(counts.comp, 0);
    assert.equal(counts.bufSrc, 0);
    assert.equal(counts.setInterval, 0);

    // Advance ~18 seconds (18 * 48000 / 128 = 6750 quanta) -> 'drive' section (> bar 8)
    createdNode.stepQuanta(6750);

    const rms = synth.tick(18000);
    assert.ok(rms > 0.04, `expected live worklet RMS > 0.04, got ${rms}`);

    const snap = createSpectrogramSnapshot();
    synth.readSpectrogram(snap, false);
    assert.equal(snap.running, true);
    assert.equal(snap.section, 'drive');
    assert.ok(snap.rms > 0.04);
    assert.ok(snap.bassLevel > 0);
    assert.ok(snap.polyLevel > 0);
    assert.ok(snap.chorusLevel > 0);
    assert.ok(snap.delayLevel > 0);
    assert.ok(snap.lowBassEnergy > 0);
    assert.ok(snap.highBandEnergy > 0);
    assert.ok(snap.downbeatPulse > 0);
    assert.equal(snap.curves.length, 6);
    assert.equal(snap.curves[0].length, SPECTROGRAM_BINS);

    const barBeforeHide = createdNode.telemetry[14];

    // Background hide -> 3s pause -> resume must preserve bar continuity (no resetClock)
    fakeHidden = true;
    synth.handleVisibilityChange(true);
    fakeTime += 3.0;
    fakeHidden = false;
    synth.handleVisibilityChange(false);
    await new Promise((r) => setTimeout(r, 10));

    createdNode.stepQuanta(64);
    const barAfterResume = createdNode.telemetry[14];
    assert.ok(barAfterResume >= barBeforeHide, `bar regressed after resume: ${barBeforeHide} -> ${barAfterResume}`);

    // Mute -> unmute cleanly toggles active state & masterGain
    synth.setEnabled(false, true);
    assert.equal(masterTargets.at(-1), 0);
    assert.equal(postedToWorklet.at(-1)?.active, false);

    synth.setEnabled(true, true);
    await synth.unlock();
    assert.ok(Math.abs(masterTargets.at(-1) - 0.75 * 0.85) < 1e-6);
  } finally {
    synth?.setEnabled(false, true);
    globalThis.window = prevWindow;
    globalThis.document = prevDocument;
    globalThis.AudioWorkletNode = prevAudioWorkletNode;
  }
});

test('GlobeCyberSynth unlocks cleanly on touchend/click even after an earlier non-activating pointerdown resume stalls', async () => {
  const prevWindow = globalThis.window;
  const prevDocument = globalThis.document;
  const prevAudioWorkletNode = globalThis.AudioWorkletNode;

  let ctxState = 'suspended';
  let resumeCalls = 0;
  let resumeCallDuringTouchStart = 0;
  const masterTargets = [];
  const postedMsgs = [];

  class AutoplayPolicyMockAudioContext {
    constructor() {
      this.sampleRate = 48000;
      this.outputLatency = 0.04;
      this.currentTime = 0.25;
      this.destination = { connect() {} };
      this.audioWorklet = {
        addModule: async () => {},
      };
    }
    get state() { return ctxState; }
    addEventListener() {}
    suspend() { ctxState = 'suspended'; return Promise.resolve(); }
    resume() {
      resumeCalls++;
      if (resumeCalls === 1) {
        resumeCallDuringTouchStart = resumeCalls;
        // Simulate real browser behavior on non-activating pointerdown/touchstart:
        // AudioContext remains 'suspended' and resume() Promise never settles.
        return new Promise(() => {});
      }
      ctxState = 'running';
      return Promise.resolve();
    }
    createGain() {
      const node = {
        connect() {},
        disconnect() {},
        gain: {
          value: 0,
          setValueAtTime(v) { node.gain.value = v; },
          linearRampToValueAtTime() {},
          exponentialRampToValueAtTime() {},
          cancelScheduledValues() {},
          setTargetAtTime(v) {
            node.gain.value = v;
            masterTargets.push(v);
          },
        },
      };
      return node;
    }
    createAnalyser() {
      return {
        connect() {},
        disconnect() {},
        fftSize: 256,
        frequencyBinCount: 128,
        smoothingTimeConstant: 0,
        minDecibels: -86,
        maxDecibels: -12,
        getFloatTimeDomainData(arr) { arr.fill(0); },
        getByteFrequencyData(arr) { arr.fill(0); },
      };
    }
  }

  class SimpleMockWorkletNode {
    constructor() {
      this.port = {
        onmessage: null,
        postMessage: (msg) => { postedMsgs.push(msg); },
      };
    }
    connect() {}
    disconnect() {}
  }

  globalThis.AudioWorkletNode = SimpleMockWorkletNode;
  globalThis.window = {
    AudioContext: AutoplayPolicyMockAudioContext,
    setTimeout: (fn) => { fn(); return 1; },
    clearTimeout: () => {},
    setInterval: () => 1,
    clearInterval: () => {},
  };
  globalThis.document = {
    get hidden() { return false; },
  };

  let synth;
  try {
    synth = new GlobeCyberSynth();
    synth.setVolume(0.65);
    synth.setEnabled(true, false);

    // 1. Non-activating touchstart/pointerdown starts unlock() and hangs resume() while suspended
    void synth.unlock();
    await new Promise((r) => setTimeout(r, 10));
    assert.equal(resumeCallDuringTouchStart, 1);
    assert.equal(ctxState, 'suspended');
    assert.equal(synth.isRunning(), false);

    // 2. Activating touchend/click must still invoke ctx.resume() synchronously and unlock audio
    await synth.unlock();
    assert.equal(resumeCalls, 2);
    assert.equal(ctxState, 'running');
    assert.equal(synth.isRunning(), true);
    assert.equal(synth.isWorkletActive(), true);
    assert.ok(Math.abs(masterTargets.at(-1) - 0.65 * 0.85) < 1e-6);
    assert.equal(postedMsgs.filter((m) => m.type === 'state').at(-1)?.active, true);
  } finally {
    synth?.setEnabled(false, true);
    globalThis.window = prevWindow;
    globalThis.document = prevDocument;
    globalThis.AudioWorkletNode = prevAudioWorkletNode;
  }
});

test('SynthSpectrogramOverlay removes canvas display and toggles synth motif source explanation popup on button click', () => {
  const prevWindow = globalThis.window;
  const prevDocument = globalThis.document;
  const prevNode = globalThis.Node;
  const prevGetComputedStyle = globalThis.getComputedStyle;

  class MockNode {
    constructor(tag = 'div') {
      this.tag = tag;
      this.children = [];
      this.listeners = new Map();
      this.attrs = new Map();
      this.style = {};
      this.hidden = true;
      this.checked = true;
      this.textContent = '';
      this.className = '';
      this.title = '';
    }
    append(...nodes) {
      this.children.push(...nodes);
    }
    appendChild(n) {
      this.children.push(n);
      return n;
    }
    replaceChildren(...nodes) {
      this.children = [...nodes];
    }
    addEventListener(type, fn) {
      const list = this.listeners.get(type) || [];
      list.push(fn);
      this.listeners.set(type, list);
    }
    dispatch(type, ev = {}) {
      for (const fn of this.listeners.get(type) || []) fn({ stopPropagation() {}, ...ev });
    }
    setAttribute(k, v) {
      this.attrs.set(k, String(v));
    }
    getAttribute(k) {
      return this.attrs.get(k) ?? null;
    }
    contains(target) {
      return target === this || this.children.some((c) => c === target || c.contains?.(target));
    }
    getBoundingClientRect() {
      return { left: 100, right: 180, top: 8, bottom: 28, width: 80, height: 20 };
    }
    querySelectorAll() {
      return [];
    }
  }

  const container = new MockNode('div');
  const tooltip = new MockNode('div');
  const toggleBtn = new MockNode('button');
  const streamCheckbox = new MockNode('input');
  const docListeners = new Map();

  globalThis.Node = MockNode;
  globalThis.getComputedStyle = () => ({ display: 'block' });
  globalThis.window = {
    innerWidth: 1280,
    addEventListener() {},
  };
  globalThis.document = {
    hidden: false,
    getElementById(id) {
      if (id === 'synth-spec') return container;
      if (id === 'synth-spec-tooltip') return tooltip;
      if (id === 'synth-spec-toggle') return toggleBtn;
      if (id === 'synth-stream-toggle') return streamCheckbox;
      return null;
    },
    querySelector() {
      return null;
    },
    createElement(tag) {
      return new MockNode(tag);
    },
    addEventListener(type, fn) {
      const list = docListeners.get(type) || [];
      list.push(fn);
      docListeners.set(type, list);
    },
  };

  try {
    const overlay = new SynthSpectrogramOverlay();
    const snap = createSpectrogramSnapshot();
    snap.enabled = true;
    snap.running = true;
    overlay.render(snap, 200);

    assert.equal(container.hidden, false);
    assert.equal(tooltip.hidden, true);
    assert.equal(overlay.isInfoOpen(), false);
    assert.equal(toggleBtn.getAttribute('aria-expanded'), 'false');

    // Click button -> shows motif source explanations popover
    toggleBtn.dispatch('click');
    assert.equal(overlay.isInfoOpen(), true);
    assert.equal(tooltip.hidden, false);
    assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
    assert.ok(tooltip.children.length >= 2, 'expected populated header + source rows inside motif popover');

    // Clicking outside closes popover cleanly
    const outsideTarget = new MockNode('canvas');
    for (const fn of docListeners.get('pointerdown') || []) fn({ target: outsideTarget });
    assert.equal(overlay.isInfoOpen(), false);
    assert.equal(tooltip.hidden, true);

    // Clicking button toggles back open, and Escape dismisses it
    toggleBtn.dispatch('click');
    assert.equal(overlay.isInfoOpen(), true);
    for (const fn of docListeners.get('keydown') || []) fn({ key: 'Escape' });
    assert.equal(overlay.isInfoOpen(), false);
    assert.equal(tooltip.hidden, true);
  } finally {
    globalThis.window = prevWindow;
    globalThis.document = prevDocument;
    globalThis.Node = prevNode;
    globalThis.getComputedStyle = prevGetComputedStyle;
  }
});

test('windAmp and oceanAmp persist cleanly across config JSON and share URL parameters', () => {
  const prevWindow = globalThis.window;
  globalThis.window = { location: { origin: 'https://example.test' } };
  try {
    assert.equal(FACTORY_DEFAULTS.earth.windAmp, 1.0);
    assert.equal(FACTORY_DEFAULTS.earth.oceanAmp, 1.0);

    const parsed = parseConfigInput({
      v: 1,
      earth: { windAmp: 2.25, oceanAmp: 0.45 },
    });
    assert.equal(parsed.earth?.windAmp, 2.25);
    assert.equal(parsed.earth?.oceanAmp, 0.45);

    const clamped = parseConfigInput({
      v: 1,
      earth: { windAmp: 99, oceanAmp: -5 },
    });
    assert.equal(clamped.earth?.windAmp, 3);
    assert.equal(clamped.earth?.oceanAmp, 0);

    const parts = buildUrlPartsFromConfig(parsed);
    assert.ok(parts.search.includes('windAmp=2.25'));
    assert.ok(parts.search.includes('oceanAmp=0.45'));

    const roundTripped = parseConfigInput(`${parts.search}${parts.hash}`);
    assert.equal(roundTripped.earth?.windAmp, 2.25);
    assert.equal(roundTripped.earth?.oceanAmp, 0.45);
  } finally {
    globalThis.window = prevWindow;
  }
});

test('fullscreen helper toggles standard/webkit fullscreen, falls back safely without options, and syncs toggle button', async () => {
  const docListeners = new Map();
  let fullscreenElement = null;
  const calls = [];
  const doc = {
    fullscreenEnabled: true,
    get fullscreenElement() { return fullscreenElement; },
    addEventListener(type, fn) {
      const arr = docListeners.get(type) || [];
      arr.push(fn);
      docListeners.set(type, arr);
    },
    async exitFullscreen() {
      calls.push('exit');
      fullscreenElement = null;
    },
  };
  const target = {
    async requestFullscreen(opts) {
      calls.push(['req', opts]);
      if (opts) throw new TypeError('options unsupported');
      fullscreenElement = target;
    },
  };
  const labelEl = { textContent: 'Fullscreen' };
  const btnListeners = new Map();
  const attrs = new Map([['title', 'old']]);
  const btn = {
    hidden: false,
    dataset: {},
    title: 'old',
    textContent: '⛶',
    querySelector(sel) { return sel === '.switch-label' ? labelEl : null; },
    setAttribute(k, v) { attrs.set(k, String(v)); },
    getAttribute(k) { return attrs.get(k); },
    removeAttribute(k) { attrs.delete(k); },
    addEventListener(type, fn) {
      const arr = btnListeners.get(type) || [];
      arr.push(fn);
      btnListeners.set(type, arr);
    },
  };

  const changeEvents = [];
  wireFullscreenToggle(btn, { doc, target, onChange: (a) => changeEvents.push(a) });
  assert.equal(btn.hidden, false);
  assert.equal(attrs.get('aria-pressed'), 'false');
  assert.equal(btn.dataset.fullscreen, 'off');
  assert.equal(labelEl.textContent, '');
  assert.equal(btn.title, '');
  assert.equal(attrs.has('title'), false);
  assert.equal(btn.textContent, '⛶');

  // Click triggers requestFullscreen -> retries without options when options throws -> sets active
  for (const fn of btnListeners.get('click') || []) fn();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(calls.length, 2);
  assert.equal(isFullscreenActive(doc), true);
  assert.equal(attrs.get('aria-pressed'), 'true');
  assert.equal(btn.dataset.fullscreen, 'on');
  assert.equal(labelEl.textContent, '');
  assert.equal(btn.title, '');
  assert.equal(attrs.has('title'), false);
  assert.equal(btn.textContent, '⛶');

  for (const fn of docListeners.get('fullscreenchange') || []) fn();
  assert.deepEqual(changeEvents, [true]);

  await toggleFullscreen(doc, target);
  assert.equal(calls.at(-1), 'exit');
  assert.equal(isFullscreenActive(doc), false);

  // WebKit fallback + unsupported check
  let wkActive = false;
  const wkDoc = {
    webkitFullscreenEnabled: true,
    get webkitFullscreenElement() { return wkActive ? wkTarget : null; },
    webkitExitFullscreen() { wkActive = false; },
    addEventListener() {},
  };
  const wkTarget = {
    webkitRequestFullscreen() { wkActive = true; },
  };
  assert.equal(isFullscreenSupported(wkDoc, wkTarget), true);
  await toggleFullscreen(wkDoc, wkTarget);
  assert.equal(isFullscreenActive(wkDoc), true);
  await toggleFullscreen(wkDoc, wkTarget);
  assert.equal(isFullscreenActive(wkDoc), false);

  const unsupportedBtn = { ...btn, hidden: false, dataset: {}, setAttribute() {}, querySelector() { return null; }, addEventListener() {} };
  wireFullscreenToggle(unsupportedBtn, { doc: { fullscreenEnabled: false, addEventListener() {} }, target: {} });
  assert.equal(unsupportedBtn.hidden, true);
});

test('computeSynthStreamWaveAmplitude uses 5x nominal base radii for wind and ocean streamlines', () => {
  assert.ok(Math.abs(SYNTH_OCEAN_WAVE_BASE_RAD - 0.00072 * 5) < 1e-12);
  assert.ok(Math.abs(SYNTH_WIND_WAVE_BASE_RAD - 0.00060 * 5) < 1e-12);

  const oceanNominal = computeSynthStreamWaveAmplitude(1000, 1.0, 1.0, 1.0, true);
  const windNominal = computeSynthStreamWaveAmplitude(1000, 1.0, 1.0, 1.0, false);
  assert.ok(Math.abs(oceanNominal - 3.6) < 1e-9);
  assert.ok(Math.abs(windNominal - 3.0) < 1e-9);
  assert.equal(computeSynthStreamWaveAmplitude(1000, 1.0, 0.004, 1.0, true), 0);
  assert.equal(computeSynthStreamWaveAmplitude(1000, 1.0, 1.0, 0, false), 0);
});
