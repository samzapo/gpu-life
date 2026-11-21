import test from 'node:test';
import assert from 'node:assert/strict';
import { summarizeAndClassifyBlob, unpackLifeMeta } from '../src/tooltip.ts';
import { buildGlobalTerrainTexture } from '../src/earth/terrainData.ts';
import { buildPackedChemistryBuffer } from '../src/periodicTable2D.ts';

function packMetaJs(phase, bioSp, kinAngle, encapGene = 7, graduated = 0) {
  if (phase === 0) return 0;
  const twoPi = Math.PI * 2;
  let a = kinAngle % twoPi;
  if (a < 0) a += twoPi;
  const k = Math.floor((a / twoPi) * 256) & 255;
  const g = Math.max(0, Math.min(15, Math.floor(encapGene)));
  const gr = graduated ? 1 : 0;
  return (
    (phase & 3) |
    ((bioSp & 255) << 2) |
    (k << 10) |
    (g << 19) |
    (gr << 23)
  ) >>> 0;
}

test('unpackLifeMeta preserves 8-bit kinPhase, encapGene, and graduated bit cleanly within f32', () => {
  const angle = 2.75;
  const packed = packMetaJs(1, 13, angle, 11, 1);
  // Check exact IEEE-754 float32 round-trip
  const f32 = new Float32Array([packed])[0];
  const meta = unpackLifeMeta([0.9, 0.2, 0.0, f32]);
  assert.equal(meta.phase, 1);
  assert.equal(meta.bioSpecies, 13);
  assert.equal(meta.encapGene, 11);
  assert.equal(meta.graduated, true);
  assert.ok(Math.abs(meta.kinPhase - angle) < (Math.PI * 2) / 200);
});

test('summarizeAndClassifyBlob classifies low-motility solar colonies as producers even under high matrix asym', () => {
  const elements = buildPackedChemistryBuffer(14).elements;
  const cIdx = elements.findIndex((e) => e.atomicNumber === 5);
  const oIdx = elements.findIndex((e) => e.atomicNumber === 6);
  const hIdx = elements.findIndex((e) => e.atomicNumber === 1);
  assert.ok(cIdx >= 0 && oIdx >= 0 && hIdx >= 0);

  // Typical random-matrix trait spread: coh=0.42, asym=0.68
  const speciesTraits = [
    [0.42, 0.5, 0.5, 0.68],
    [0.48, 0.45, 0.45, 0.62],
  ];
  const n = 24;
  const particles = [];
  const lifeState = [];
  const members = [];
  for (let i = 0; i < n; i++) {
    members.push(i);
    const col = i % 3 === 0 ? cIdx : i % 3 === 1 ? oIdx : hIdx;
    // Slow stationary producer velocity (~0.04)
    particles.push([1, 0, 0, 0.03, 0.02, 0, col, 0, 1, 0, 0, -0.3]);
    lifeState.push([0.85, 0.12, 0.0, packMetaJs(1, i % 2, 1.0, 8, 0)]);
  }
  const summary = summarizeAndClassifyBlob(members, particles, lifeState, elements, speciesTraits, 0);
  assert.equal(summary.role, 'producer', `expected producer, got ${summary.role} (auto=${summary.autoCap}, pred=${summary.predCap})`);
});

test('buildGlobalTerrainTexture outputs finite elevation, slopes, and bounded tectonic ridge mask', () => {
  const tex = buildGlobalTerrainTexture(null, null, 72, 37);
  assert.equal(tex.length, 72 * 37 * 4);
  let maxRidge = 0;
  for (let i = 0; i < 72 * 37; i++) {
    const o = i * 4;
    for (let c = 0; c < 4; c++) {
      assert.ok(Number.isFinite(tex[o + c]), `non-finite terrain channel at ${o + c}`);
    }
    assert.ok(tex[o + 3] >= 0 && tex[o + 3] <= 1);
    if (tex[o + 3] > maxRidge) maxRidge = tex[o + 3];
  }
  assert.ok(maxRidge > 0.4, `expected active tectonic ridge belt in static mask, got ${maxRidge}`);
});
