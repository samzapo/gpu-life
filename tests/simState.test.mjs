import test from 'node:test';
import assert from 'node:assert/strict';
import {
  encodeBytesToBase64,
  decodeBase64ToBytes,
  isSerializedSimStateInput,
  serializeSimState,
  parseSimStateInput,
  PARTICLE_BYTES,
  LIFE_MDP_BYTES,
  VEC4_BYTES,
  GAME_TRACKING_BYTES,
  BLOB_MDP_BYTES,
  SPECIES_MDP_BYTES,
} from '../src/simState.ts';

function makePatternBytes(byteLength, seedOffset) {
  const out = new Uint8Array(byteLength);
  for (let i = 0; i < byteLength; i++) {
    out[i] = (i * 17 + seedOffset) & 0xff;
  }
  return out;
}

test('base64 helpers round-trip arbitrary binary slices beyond stack call limits', () => {
  const src = makePatternBytes(96 * 1024, 13);
  const encoded = encodeBytesToBase64(src);
  const decoded = decodeBase64ToBytes(encoded);
  assert.deepEqual(decoded, src);
});

test('serializeSimState + parseSimStateInput round-trip gzip-base64 buffers bit-for-bit', async () => {
  const n = 128;
  const particles = makePatternBytes(n * PARTICLE_BYTES, 1);
  const lifeMdp = makePatternBytes(n * LIFE_MDP_BYTES, 3);
  const bioForce = makePatternBytes(n * VEC4_BYTES, 7);
  const valence = makePatternBytes(n * VEC4_BYTES, 11);
  const gameTracking = makePatternBytes(n * GAME_TRACKING_BYTES, 19);
  const blobMdp = makePatternBytes(n * BLOB_MDP_BYTES, 23);
  const speciesMdp = makePatternBytes(12 * SPECIES_MDP_BYTES, 29);

  const snapshot = await serializeSimState({
    capturedAt: '2026-11-24T12:00:00.000Z',
    config: {
      v: 1,
      engine: 'cells',
      particles: n,
      colours: 12,
      cells: 20,
      seed: 'TestSeed',
      oceanSeed: 'OceanSeed',
      view: {
        rotate: [-42.5, 18.25, 0],
        pan: [0, 0],
        scale: 423,
      },
    },
    runtime: {
      engine: 'cells',
      particleCount: n,
      alternate: 1,
      frameCount: 4321,
      uvUtcMs: 1795521600000,
      audio: {
        enabled: true,
        state: 'running',
        volume: 0.72,
      },
    },
    particles,
    lifeMdp,
    bioForce,
    valence,
    gameTracking,
    blobMdp,
    speciesMdp,
  });

  const json = JSON.stringify(snapshot);
  assert.equal(isSerializedSimStateInput(json), true);
  assert.equal(isSerializedSimStateInput('{"v":1,"particles":10000}'), false);
  assert.equal(snapshot.buffers.particles.encoding, 'gzip-base64');

  const decoded = await parseSimStateInput(`\`\`\`json\n${json}\n\`\`\``);
  assert.equal(decoded.runtime.particleCount, n);
  assert.equal(decoded.runtime.alternate, 1);
  assert.equal(decoded.runtime.frameCount, 4321);
  assert.equal(decoded.runtime.uvUtcMs, 1795521600000);
  assert.equal(decoded.config.seed, 'TestSeed');
  assert.deepEqual(decoded.particles, particles);
  assert.deepEqual(decoded.lifeMdp, lifeMdp);
  assert.deepEqual(decoded.bioForce, bioForce);
  assert.deepEqual(decoded.valence, valence);
  assert.deepEqual(decoded.gameTracking, gameTracking);
  assert.deepEqual(decoded.blobMdp, blobMdp);
  assert.deepEqual(decoded.speciesMdp, speciesMdp);
});

test('serializeSimState + parseSimStateInput supports raw-base64 fallback encoding', async () => {
  const n = 16;
  const particles = makePatternBytes(n * PARTICLE_BYTES, 5);
  const lifeMdp = makePatternBytes(n * LIFE_MDP_BYTES, 9);

  const snapshot = await serializeSimState({
    encoding: 'raw-base64',
    config: { v: 1, particles: n },
    runtime: {
      engine: 'slime',
      particleCount: n,
      alternate: 0,
      frameCount: 77,
    },
    particles,
    lifeMdp,
  });

  assert.equal(snapshot.buffers.particles.encoding, 'raw-base64');
  const decoded = await parseSimStateInput(snapshot);
  assert.deepEqual(decoded.particles, particles);
  assert.deepEqual(decoded.lifeMdp, lifeMdp);
  assert.equal(decoded.bioForce, undefined);
});

test('parseSimStateInput rejects corrupted byteLength or particle count mismatch', async () => {
  const n = 8;
  const snapshot = await serializeSimState({
    encoding: 'raw-base64',
    config: { v: 1, particles: n },
    runtime: {
      engine: 'cells',
      particleCount: n,
      alternate: 0,
      frameCount: 1,
    },
    particles: makePatternBytes(n * PARTICLE_BYTES, 2),
    lifeMdp: makePatternBytes(n * LIFE_MDP_BYTES, 4),
  });

  snapshot.runtime.particleCount = n + 1;
  await assert.rejects(() => parseSimStateInput(snapshot), /byteLength mismatch/);
});
