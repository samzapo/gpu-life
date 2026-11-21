import test from 'node:test';
import assert from 'node:assert/strict';
import { colonyLosses } from '../src/temporal/losses.ts';
import { CameraFlight, copyView } from '../src/earth/cameraFlight.ts';
import { GlobeTour } from '../src/earth/globeTour.ts';
import { findColonies } from '../src/temporal/world.ts';

function snapshot(tick, cells) {
  const life = new Float32Array(cells.length * 4);
  const episodes = new Uint32Array(cells.length);
  const deathHistory = new Uint32Array(cells.length);
  cells.forEach(([episode, alive, history], index) => {
    life[index * 4 + 3] = Number(alive);
    episodes[index] = episode;
    deathHistory[index] = history;
  });
  return { tick, life, episodes, deathHistory };
}

test('starvation and crowding alone pass quietly; a real mixed collapse is reported', () => {
  const cohort = [0, 1, 2];
  const healthy = snapshot(
    10,
    cohort.map(() => [0, true, 0]),
  );
  const routine = snapshot(13, [
    [0, false, 1],
    [0, false, 2],
    [0, false, 1],
  ]);
  assert.deepEqual(colonyLosses(cohort, healthy, routine, []), {
    qualifies: false,
    summary: 'Starvation 2 · Overcrowding 1',
  });
  const mixed = snapshot(13, [
    [0, false, 10],
    [0, false, 2],
    [0, true, 0],
  ]);
  assert.deepEqual(colonyLosses(cohort, healthy, mixed, []), {
    qualifies: true,
    summary: 'Overcrowding 1 · Toxic bloom 1',
  });
});

test('a later birth cannot donate its death to an earlier cohort', () => {
  const healthy = snapshot(10, [[1, true, 0]]);
  const checkpoint = snapshot(11, [[2, false, (3 << 4) | (1 << 8)]]);
  const recycled = snapshot(40, [[9, false, 2 << 4]]);
  assert.equal(
    colonyLosses([0], healthy, checkpoint, []).summary,
    'Predation 1',
  );
  assert.equal(
    colonyLosses([0], healthy, recycled, [healthy, checkpoint]).summary,
    'Predation 1',
  );
  assert.equal(
    colonyLosses([0], healthy, recycled, []).summary,
    'Unattributed 1',
  );
  assert.equal(
    colonyLosses([0], healthy, snapshot(40, [[0, false, 1]]), []).summary,
    'Did not form 1',
  );
});

test('return flights start at the displayed pose, cross the dateline and survive a suspended tab', () => {
  const flight = new CameraFlight();
  const view = {
    rotate: [177, 50, 0],
    pan: [30, 20],
    scale: 1600,
    width: 1100,
    height: 700,
    projectionId: 'orthographic',
  };
  const from = copyView(view);
  const to = {
    ...view,
    rotate: [-177, 20, 0],
    pan: [0, 0],
    scale: 400,
    projectionId: 'equal_earth',
  };
  flight.move(view, to, 2000);
  flight.frame(100, view);
  assert.deepEqual(view, from);
  flight.frame(1_000_000, view);
  assert.ok(view.scale > 1500);
  let switched = false;
  for (let frame = 1; frame <= 130; frame++) {
    const previous = view.projectionId;
    flight.frame(1_000_000 + frame * 20, view);
    if (previous !== view.projectionId) {
      switched = true;
      assert.equal(flight.opacity, 1);
    }
    assert.ok(view.rotate[0] >= 177 && view.rotate[0] <= 183);
  }
  assert.ok(switched);
  assert.equal(flight.active, false);
  assert.equal(flight.opacity, 0);
  assert.ok(Math.abs(view.scale - to.scale) < 1e-8);
  assert.deepEqual(view.pan, to.pan);
});

test('globe tour zooms closer onto large colonies and moves dynamically', () => {
  const tour = new GlobeTour();
  tour.setEnabled(true);
  const episodes = new Uint32Array(30);
  const large = {
    members: Array.from({ length: 24 }, (_, i) => i),
    center: [1, 0, 0],
    radius: 0.06,
  };
  const small = {
    members: [24, 25, 26, 27, 28, 29],
    center: [0, 1, 0],
    radius: 0.09,
  };
  tour.observe([large, small], episodes);

  const view = {
    rotate: [0, 0, 0],
    pan: [0, 0],
    scale: 400,
    width: 1000,
    height: 800,
    projectionId: 'orthographic',
  };

  let timeMs = 1000;
  // Advance past initial orbit transit (4.5s) into colony visit
  for (let i = 0; i < 160; i++) {
    timeMs += 50;
    tour.frame(timeMs, view);
  }
  // Should be zoomed in closely onto the large colony (scaleRatio >= 2.2 => scale >= 1760 on minDim=800)
  assert.ok(view.scale > 1500, `expected close-up scale > 1500, got ${view.scale}`);
  assert.ok(tour.caption.includes('24 cells'), `expected tracking large colony, got ${tour.caption}`);
  // Dynamic camera bank/roll should be non-zero
  assert.ok(Math.abs(view.rotate[2]) > 0.05, `expected dynamic roll, got ${view.rotate[2]}`);
});

test('findColonies computes colony angular radius', () => {
  const count = 8;
  const particles = new Float32Array(count * 12);
  const life = new Float32Array(count * 4);
  const episodes = new Uint32Array(count);
  const deathHistory = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const angle = (i - 3.5) * 0.01;
    particles[i * 12] = Math.cos(angle);
    particles[i * 12 + 1] = Math.sin(angle);
    particles[i * 12 + 2] = 0;
    life[i * 4 + 3] = 1;
  }
  const colonies = findColonies(
    { tick: 0, frame: 0, particles, life, episodes, deathHistory },
    0.04,
  );
  assert.equal(colonies.length, 1);
  assert.equal(colonies[0].members.length, 8);
  assert.ok(colonies[0].radius > 0.01 && colonies[0].radius < 0.05);
});
