import type { GlobeViewState } from './projections';
import type { Colony, Point } from '../temporal/world';

type Coordinates = [number, number];
type Visit = 'orbit' | 'colony' | 'landmark' | 'rescue';
type Membership = Map<number, number>;

const landmarks: { name: string; center: Coordinates }[] = [
  { name: 'Gulf Stream', center: [-64, 38] },
  { name: 'Cape Agulhas', center: [20, -35] },
  { name: 'Kuroshio Current', center: [140, 35] },
  { name: 'Drake Passage', center: [-64, -58] },
];
const rescueDwellSeconds = 22;
const colonyDwellSeconds = 18;
const landmarkDwellSeconds = 10;
const orbitTransitSeconds = 4.5;
const radiansToDegrees = 180 / Math.PI;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const wrapDegrees = (value: number) => (((value % 360) + 540) % 360) - 180;
export const coordinates = (point: Point): Coordinates => [
  Math.atan2(point[1], point[0]) * radiansToDegrees,
  Math.asin(Math.max(-1, Math.min(1, point[2]))) * radiansToDegrees,
];

function overlap(colony: Colony, episodes: Uint32Array, members: Membership) {
  let count = 0;
  for (const index of colony.members)
    if (members.get(index) === episodes[index]) count++;
  return count;
}

export class GlobeTour {
  enabled = false;
  private visit: Visit = 'orbit';
  private elapsedSeconds = 0;
  private phaseSeconds = 0;
  private lastFrameMs?: number;
  private orbitDegrees?: number;
  private nextLandmark = 0;
  private colonyStopsSinceLandmark = 0;
  private destination: Coordinates = [0, 0];
  private colonyRadius = 0.08;
  private landmarkName = '';
  private colonies: Colony[] = [];
  private episodes = new Uint32Array();
  private members: Membership = new Map();
  private lastColony: Membership = new Map();

  get caption() {
    if (this.visit === 'rescue') return 'Rescue site';
    if (!this.enabled) return '';
    if (this.visit === 'landmark') return this.landmarkName;
    if (this.visit === 'colony')
      return this.members.size
        ? 'Colony · ' + this.members.size + ' cells'
        : 'Colony site';
    return 'Sweeping globe';
  }

  pauseClock() {
    this.lastFrameMs = undefined;
  }

  setEnabled(enabled: boolean) {
    this.enabled = enabled;
    if (this.visit !== 'rescue') this.startOrbit();
    this.pauseClock();
  }

  resetWorld() {
    this.colonies = [];
    this.episodes = new Uint32Array();
    this.lastColony.clear();
    this.colonyStopsSinceLandmark = 0;
    this.startOrbit();
    this.pauseClock();
  }

  manualMove() {
    this.enabled = false;
    this.resetWorld();
  }

  rescue() {
    this.startOrbit();
    this.visit = 'rescue';
    this.pauseClock();
  }

  observe(colonies: Colony[], episodes: Uint32Array<ArrayBuffer>) {
    if (!this.enabled) return;
    this.colonies = colonies;
    // A checkpoint's views share a large backing buffer. Retain only birth IDs.
    this.episodes = episodes.slice();
    if (this.visit !== 'colony' || !this.members.size) return;
    let best: Colony | undefined;
    let bestCount = 0;
    for (const colony of colonies) {
      const count = overlap(colony, episodes, this.members);
      if (count > bestCount) {
        best = colony;
        bestCount = count;
      }
    }
    if (best) this.track(best);
    else this.members.clear();
  }

  private track(colony: Colony) {
    this.destination = coordinates(colony.center);
    if (colony.radius !== undefined) {
      this.colonyRadius = colony.radius;
    }
    this.members = new Map(
      colony.members.map((index) => [index, this.episodes[index]]),
    );
    this.lastColony = this.members;
  }

  private startOrbit() {
    this.visit = 'orbit';
    this.elapsedSeconds = 0;
    this.orbitDegrees = undefined;
    this.members = new Map();
  }

  private chooseVisit() {
    this.elapsedSeconds = 0;
    const largest = this.colonies[0];
    const hasLargeColony = largest && largest.members.length >= 6;
    const visitColony =
      hasLargeColony && this.colonyStopsSinceLandmark < 3;

    if (visitColony) {
      const minSize = Math.max(8, Math.floor(largest.members.length * 0.52));
      const pool = this.colonies.filter((c) => c.members.length >= minSize);
      const candidates = pool.length ? pool : [largest];
      const next = candidates.find(
        (colony) =>
          overlap(colony, this.episodes, this.lastColony) <
          colony.members.length * 0.25,
      );
      this.visit = 'colony';
      this.track(next ?? candidates[0]);
      this.colonyStopsSinceLandmark++;
    } else {
      const landmark = landmarks[this.nextLandmark++ % landmarks.length];
      this.visit = 'landmark';
      this.landmarkName = landmark.name;
      this.destination = landmark.center;
      this.colonyStopsSinceLandmark = 0;
    }
  }

  // Only active, visible viewing time advances a visit. Manual movement owns the
  // camera immediately, including during the post-rescue pullback.
  frame(nowMs: number, view: GlobeViewState): boolean {
    if (!this.enabled && this.visit !== 'rescue') return false;
    const deltaSeconds =
      this.lastFrameMs === undefined
        ? 0
        : Math.max(0, (nowMs - this.lastFrameMs) / 1000);
    this.lastFrameMs = nowMs;
    this.elapsedSeconds += deltaSeconds;
    this.phaseSeconds += deltaSeconds;
    const stepSeconds = Math.min(0.06, deltaSeconds);
    const dimension = Math.min(view.width, view.height);

    if (this.visit === 'rescue') {
      if (this.enabled && this.elapsedSeconds >= rescueDwellSeconds)
        this.startOrbit();
      return false;
    }

    if (this.visit === 'orbit' && this.elapsedSeconds >= orbitTransitSeconds) {
      this.chooseVisit();
    } else if (
      (this.visit === 'colony' && this.elapsedSeconds >= colonyDwellSeconds) ||
      (this.visit === 'landmark' && this.elapsedSeconds >= landmarkDwellSeconds)
    ) {
      if (this.colonies.length && this.visit !== 'colony') {
        this.chooseVisit();
      } else {
        this.startOrbit();
      }
    }

    let center: Coordinates = [this.destination[0], this.destination[1]];
    let scaleRatio = 2.4;
    let targetRoll = 0;
    let tau = 0.9;

    if (this.visit === 'orbit') {
      this.orbitDegrees ??= -view.rotate[0];
      this.orbitDegrees += stepSeconds * 11.5;
      center = [
        this.orbitDegrees,
        26 * Math.sin(this.orbitDegrees / radiansToDegrees),
      ];
      scaleRatio = 1.08 + 0.22 * Math.sin(this.phaseSeconds * 0.9);
      targetRoll = 9 * Math.sin(this.phaseSeconds * 0.75);
      tau = 1.15;
    } else if (this.visit === 'colony') {
      const baseCloseRatio = clamp(
        0.24 / Math.max(this.colonyRadius, 0.055),
        2.2,
        3.8,
      );
      const t = this.elapsedSeconds;
      const arrival = clamp(t / 2.8, 0, 1);
      const smoothArrival = arrival * arrival * (3 - 2 * arrival);
      const breath = 1 + 0.07 * Math.sin(t * 0.55);
      const departPull =
        t > colonyDwellSeconds - 2.0
          ? 1 - 0.16 * ((t - (colonyDwellSeconds - 2.0)) / 2.0)
          : 1;
      scaleRatio =
        baseCloseRatio * (0.76 + 0.28 * smoothArrival) * breath * departPull;
      const swayDeg = 2.2 / scaleRatio;
      center = [
        this.destination[0] + swayDeg * Math.sin(t * 0.52),
        clamp(this.destination[1] + swayDeg * 0.7 * Math.cos(t * 0.41), -82, 82),
      ];
      targetRoll = 6.5 * Math.sin(t * 0.36);
      tau = t < 2.5 ? 0.85 : 0.65;
    } else if (this.visit === 'landmark') {
      const t = this.elapsedSeconds;
      scaleRatio = 1.55 + 0.24 * Math.sin(t * 0.55);
      center = [
        this.destination[0] + 3.2 * Math.sin(t * 0.35),
        clamp(this.destination[1] + 2.0 * Math.cos(t * 0.3), -82, 82),
      ];
      targetRoll = 6 * Math.sin(t * 0.4);
      tau = 1.05;
    }

    const blend = 1 - Math.exp(-stepSeconds / tau);
    const longitudeDelta = wrapDegrees(-center[0] - view.rotate[0]);
    const latitudeDelta = -center[1] - view.rotate[1];
    const rollDelta = wrapDegrees(targetRoll - view.rotate[2]);
    const scaleDelta = Math.log((dimension * scaleRatio) / view.scale);
    if (
      view.projectionId === 'orthographic' &&
      Math.abs(longitudeDelta) < 0.002 &&
      Math.abs(latitudeDelta) < 0.002 &&
      Math.abs(rollDelta) < 0.002 &&
      Math.abs(scaleDelta) < 0.0001 &&
      Math.hypot(...view.pan) < 0.1
    )
      return false;
    view.rotate = [
      wrapDegrees(view.rotate[0] + longitudeDelta * blend),
      view.rotate[1] + latitudeDelta * blend,
      wrapDegrees(view.rotate[2] + rollDelta * blend),
    ];
    view.scale *= Math.exp(scaleDelta * blend);
    view.pan = [view.pan[0] * (1 - blend), view.pan[1] * (1 - blend)];
    return true;
  }
}
