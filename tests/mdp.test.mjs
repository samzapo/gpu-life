import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildInitialSpeciesMdpBuffer,
  buildSpeciesMdpPriors,
  dequantizeMdpQ,
  evalActionLeverProfileJs,
  evalCombinedMdpPolicyJs,
  MDP_ACTION_COMPACT,
  MDP_ACTION_DISPERSE,
  MDP_ACTION_EXPAND,
  MDP_ACTION_PROPEL,
  MDP_PAD_TAG,
  MDP_SPECIES_STRIDE,
  MDP_STATE_ABUNDANT,
  MDP_STATE_DARK_CROWDED,
  MDP_STATE_HAZARD,
  MDP_STATE_LOW_ENERGY,
  packThreeMdpPad,
  quantizeMdpQ,
  SPECIES_MDP_SLOT_WORDS,
  unpackMdpActionFromPad,
  unpackMdpCtrl,
  unpackThreeMdpFromPad,
} from '../src/mdp.ts';
import { buildPackedChemistryBuffer } from '../src/periodicTable2D.ts';
import { summarizeAndClassifyBlob } from '../src/tooltip.ts';

function packMetaJs(phase, bioSp, kinAngle, encapGene, graduated = 0) {
  const k = Math.floor((((kinAngle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)) / (Math.PI * 2) * 256) & 255;
  return (phase & 3) | ((bioSp & 255) << 2) | (k << 10) | ((encapGene & 15) << 19) | ((graduated & 1) << 23);
}

test('buildSpeciesMdpPriors and buildInitialSpeciesMdpBuffer initialize near-neutral tabula-rasa priors', () => {
  const a = buildSpeciesMdpPriors(4, 'seed-42');
  const b = buildSpeciesMdpPriors(4, 'seed-42');
  assert.deepEqual(a, b);
  assert.equal(a.length, 4 * MDP_SPECIES_STRIDE);

  for (let i = 0; i < a.length; i++) {
    assert.ok(Number.isFinite(a[i]) && a[i] >= -0.05 && a[i] <= 0.05);
  }

  const spBuf = buildInitialSpeciesMdpBuffer(4, 'seed-42');
  assert.equal(spBuf.length, 4 * SPECIES_MDP_SLOT_WORDS);
  for (let sp = 0; sp < 4; sp++) {
    const oBase = sp * SPECIES_MDP_SLOT_WORDS;
    for (let s = 0; s < 4; s++) {
      const bestAct = spBuf[oBase + 16 + s];
      assert.ok(bestAct >= 0 && bestAct < 4);
      const qVal = spBuf[oBase + s * 4 + bestAct];
      for (let act = 0; act < 4; act++) {
        assert.ok(qVal >= spBuf[oBase + s * 4 + act]);
      }
    }
  }
});

test('3-MDP pad bit packing and combined lever blend math round-trip accurately within f32 precision', () => {
  for (const val of [-1.5, -0.75, 0.0, 0.42, 1.5]) {
    const q = dequantizeMdpQ(quantizeMdpQ(val));
    assert.ok(Math.abs(q - val) < 0.015, `expected ${val} ~= ${q}`);
  }

  for (const pAct of [MDP_ACTION_EXPAND, MDP_ACTION_PROPEL, MDP_ACTION_DISPERSE, MDP_ACTION_COMPACT]) {
    for (const bAct of [MDP_ACTION_EXPAND, MDP_ACTION_PROPEL]) {
      for (const hasBlob of [false, true]) {
        const packedF32 = packThreeMdpPad(
          pAct,
          bAct,
          MDP_ACTION_DISPERSE,
          hasBlob,
          MDP_STATE_HAZARD,
          MDP_STATE_ABUNDANT,
          5,
        );
        assert.ok(packedF32 >= MDP_PAD_TAG && packedF32 < 16777216);
        const decoded = unpackThreeMdpFromPad(packedF32);
        assert.equal(decoded.valid, true);
        assert.equal(decoded.pAct, pAct);
        assert.equal(decoded.bAct, bAct);
        assert.equal(decoded.sAct, MDP_ACTION_DISPERSE);
        assert.equal(decoded.hasBlob, hasBlob);
        assert.equal(decoded.pState, MDP_STATE_HAZARD);
        assert.equal(decoded.bState, MDP_STATE_ABUNDANT);
        assert.equal(decoded.blobSizeBucket, 5);
        assert.equal(unpackMdpActionFromPad(packedF32), hasBlob ? bAct : pAct);
      }
    }
  }

  // Verify weighted blend math (solitary 0.65P + 0.35S vs blob 0.35P + 0.40B + 0.25S)
  const solitaryMix = evalCombinedMdpPolicyJs(MDP_ACTION_COMPACT, MDP_ACTION_PROPEL, MDP_ACTION_DISPERSE, false);
  const pC = evalActionLeverProfileJs(MDP_ACTION_COMPACT);
  const sD = evalActionLeverProfileJs(MDP_ACTION_DISPERSE);
  assert.ok(Math.abs(solitaryMix.dSameKin - (0.65 * pC.dSameKin + 0.35 * sD.dSameKin)) < 1e-9);

  const blobMix = evalCombinedMdpPolicyJs(MDP_ACTION_EXPAND, MDP_ACTION_PROPEL, MDP_ACTION_COMPACT, true);
  const eP = evalActionLeverProfileJs(MDP_ACTION_EXPAND);
  const prP = evalActionLeverProfileJs(MDP_ACTION_PROPEL);
  assert.ok(Math.abs(blobMix.chiralTwist - (0.35 * eP.chiralTwist + 0.40 * prP.chiralTwist + 0.25 * pC.chiralTwist)) < 1e-9);
  assert.ok(blobMix.dForeignBio > 0.25);

  const ctrlBits = MDP_STATE_DARK_CROWDED | (MDP_ACTION_COMPACT << 2) | (20 << 4) | (1 << 12);
  const decodedCtrl = unpackMdpCtrl(ctrlBits);
  assert.equal(decodedCtrl.prevState, MDP_STATE_DARK_CROWDED);
  assert.equal(decodedCtrl.action, MDP_ACTION_COMPACT);
  assert.equal(decodedCtrl.dwell, 20);
  assert.equal(decodedCtrl.initialized, true);
});

test('summarizeAndClassifyBlob tracks all 3 MDP levels cleanly', () => {
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
  const padTriplet = packThreeMdpPad(
    MDP_ACTION_EXPAND,
    MDP_ACTION_PROPEL,
    MDP_ACTION_COMPACT,
    true,
    MDP_STATE_LOW_ENERGY,
    MDP_STATE_ABUNDANT,
    4,
  );

  for (let i = 0; i < 18; i++) {
    members.push(i);
    const col = i % 3 === 0 ? sIdx : i % 3 === 1 ? hIdx : cIdx;
    particles.push([1, 0, 0, 0.24, 0.21, 0, col, 0, 1, 0, 0, padTriplet]);
    lifeState.push([0.82, 0.14, 0.0, packMetaJs(1, 0, 1.2, 6, 0)]);
  }

  const summary = summarizeAndClassifyBlob(members, particles, lifeState, elements, speciesTraits, 0);
  assert.equal(summary.role, 'consumer', `expected consumer, got ${summary.role}`);
  assert.equal(summary.blobMdpAction, MDP_ACTION_PROPEL);
  assert.equal(summary.speciesMdpAction, MDP_ACTION_COMPACT);
  assert.equal(summary.cellMdpAction, MDP_ACTION_EXPAND);
  assert.equal(summary.mdpAction, MDP_ACTION_PROPEL);
  assert.equal(summary.mdpTripletLabel, 'Blob:Propel · Sp:Compact · Cell:Expand');
});
