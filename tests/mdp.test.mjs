import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildSpeciesMdpPriors,
  dequantizeMdpQ,
  MDP_ACTION_CONSOLIDATE,
  MDP_ACTION_EVADE,
  MDP_ACTION_FORAGE,
  MDP_ACTION_HUNT,
  MDP_SPECIES_STRIDE,
  MDP_STATE_DORMANT,
  MDP_STATE_STARVING,
  MDP_STATE_THREATENED,
  MDP_STATE_THRIVING,
  quantizeMdpQ,
  unpackMdpActionFromPad,
  unpackMdpCtrl,
} from '../src/mdp.ts';
import { buildPackedChemistryBuffer } from '../src/periodicTable2D.ts';
import { summarizeAndClassifyBlob } from '../src/tooltip.ts';

function packMetaJs(phase, bioSp, kinAngle, encapGene, graduated = 0) {
  const k = Math.floor((((kinAngle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)) / (Math.PI * 2) * 256) & 255;
  return (phase & 3) | ((bioSp & 255) << 2) | (k << 10) | ((encapGene & 15) << 19) | ((graduated & 1) << 23);
}

test('buildSpeciesMdpPriors is deterministic and encodes distinct autotroph vs predator instincts', () => {
  const traits = [
    // sp 0: cohesive producer-leaning
    [0.85, 0.15, 0.25, 0.18],
    // sp 1: asymmetric predatory-leaning
    [0.22, 0.88, 0.32, 0.84],
  ];
  const a = buildSpeciesMdpPriors(traits, 2, 'seed-42');
  const b = buildSpeciesMdpPriors(traits, 2, 'seed-42');
  assert.deepEqual(a, b);
  assert.equal(a.length, 2 * MDP_SPECIES_STRIDE);

  for (let i = 0; i < a.length; i++) {
    assert.ok(Number.isFinite(a[i]) && a[i] >= -1.2 && a[i] <= 1.2);
  }

  // Producer species prefers CONSOLIDATE more strongly when dormant than HUNT
  const pDormBase = 0 * MDP_SPECIES_STRIDE + MDP_STATE_DORMANT * 4;
  assert.ok(a[pDormBase + MDP_ACTION_CONSOLIDATE] > a[pDormBase + MDP_ACTION_HUNT]);

  // Predator species prefers HUNT in starving state much more strongly than producer does
  const pStarvHunt = a[0 * MDP_SPECIES_STRIDE + MDP_STATE_STARVING * 4 + MDP_ACTION_HUNT];
  const cStarvHunt = a[1 * MDP_SPECIES_STRIDE + MDP_STATE_STARVING * 4 + MDP_ACTION_HUNT];
  assert.ok(cStarvHunt > pStarvHunt + 0.3);

  // Under threat, predators value EVADE over foraging
  const cThreatBase = 1 * MDP_SPECIES_STRIDE + MDP_STATE_THREATENED * 4;
  assert.ok(a[cThreatBase + MDP_ACTION_EVADE] > a[cThreatBase + MDP_ACTION_FORAGE]);
});

test('MDP quantization, pad action bits, and control bit packing round-trip within exact precision', () => {
  for (const val of [-1.5, -0.75, 0.0, 0.42, 1.5]) {
    const q = dequantizeMdpQ(quantizeMdpQ(val));
    assert.ok(Math.abs(q - val) < 0.015, `expected ${val} ~= ${q}`);
  }

  for (const act of [MDP_ACTION_FORAGE, MDP_ACTION_HUNT, MDP_ACTION_EVADE, MDP_ACTION_CONSOLIDATE]) {
    const packedNiche = 2097152 | (7 << 0) | (9 << 4) | (12 << 8) | (11 << 12) | (8 << 16) | (1 << 20) | (act << 22);
    assert.ok(packedNiche < 16777216, 'must fit exact f32 integer bits');
    const asF32 = Math.fround(packedNiche);
    assert.equal(asF32, packedNiche);
    assert.equal(unpackMdpActionFromPad(asF32), act);
  }

  const ctrlBits = MDP_STATE_THREATENED | (MDP_ACTION_EVADE << 2) | (24 << 4) | (1 << 12);
  const decoded = unpackMdpCtrl(ctrlBits);
  assert.equal(decoded.prevState, MDP_STATE_THREATENED);
  assert.equal(decoded.action, MDP_ACTION_EVADE);
  assert.equal(decoded.dwell, 24);
  assert.equal(decoded.initialized, true);
});

test('summarizeAndClassifyBlob classifies active asymmetric swimmers as consumers and tracks dominant MDP action', () => {
  const elements = buildPackedChemistryBuffer(14).elements;
  const cIdx = elements.findIndex((e) => e.atomicNumber === 5);
  const sIdx = elements.findIndex((e) => e.atomicNumber === 12);
  const hIdx = elements.findIndex((e) => e.atomicNumber === 1);
  assert.ok(cIdx >= 0 && sIdx >= 0 && hIdx >= 0);

  const speciesTraits = [
    [0.24, 0.80, 0.28, 0.78],
  ];
  const members = [];
  const particles = [];
  const lifeState = [];
  const padHunt = 2097152 | (MDP_ACTION_HUNT << 22);

  for (let i = 0; i < 18; i++) {
    members.push(i);
    const col = i % 3 === 0 ? sIdx : i % 3 === 1 ? hIdx : cIdx;
    // Active hunting velocity (~0.32 tangentially -> motility ~0.71)
    particles.push([1, 0, 0, 0.24, 0.21, 0, col, 0, 1, 0, 0, padHunt]);
    lifeState.push([0.82, 0.14, 0.0, packMetaJs(1, 0, 1.2, 6, 0)]);
  }

  const summary = summarizeAndClassifyBlob(members, particles, lifeState, elements, speciesTraits, 0);
  assert.equal(summary.role, 'consumer', `expected consumer, got ${summary.role} (auto=${summary.autoCap}, pred=${summary.predCap})`);
  assert.equal(summary.mdpAction, MDP_ACTION_HUNT);
  assert.equal(summary.mdpPolicyLabel, 'Hunting');
});
