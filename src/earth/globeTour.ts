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
const followSeconds = 60;
const approachSeconds = 7;
const radiansToDegrees = 180 / Math.PI;
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
  private lastFrameMs?: number;
  private orbitDegrees?: number;
  private nextLandmark = 0;
  private preferColony = true;
  private destination: Coordinates = [0, 0];
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
    return 'Orbiting the globe';
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
    this.preferColony = true;
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
    if (this.preferColony && this.colonies.length) {
      const largest = this.colonies[0];
      const next = this.colonies.find(
        (colony) =>
          colony.members.length >= largest.members.length * 0.4 &&
          overlap(colony, this.episodes, this.lastColony) <
            colony.members.length * 0.25,
      );
      this.visit = 'colony';
      this.track(next ?? largest);
      this.preferColony = false;
    } else {
      const landmark = landmarks[this.nextLandmark++ % landmarks.length];
      this.visit = 'landmark';
      this.landmarkName = landmark.name;
      this.destination = landmark.center;
      this.preferColony = true;
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
    const stepSeconds = Math.min(0.06, deltaSeconds);
    const dimension = Math.min(view.width, view.height);

    if (this.visit === 'rescue') {
      if (this.enabled && this.elapsedSeconds >= followSeconds)
        this.startOrbit();
      return false;
    }

    if (this.visit === 'orbit' && this.elapsedSeconds >= 7) this.chooseVisit();
    else if (
      (this.visit === 'colony' &&
        this.elapsedSeconds >= followSeconds + approachSeconds) ||
      (this.visit === 'landmark' && this.elapsedSeconds >= 18)
    )
      this.startOrbit();

    let center = this.destination;
    let scaleRatio = 1.04;
    if (this.visit === 'orbit') {
      this.orbitDegrees ??= -view.rotate[0];
      this.orbitDegrees += stepSeconds * 4;
      center = [
        this.orbitDegrees,
        18 * Math.sin(this.orbitDegrees / radiansToDegrees),
      ];
      scaleRatio = 0.46;
    } else if (this.visit === 'landmark') scaleRatio = 0.78;

    const blend =
      1 - Math.exp(-stepSeconds / (this.visit === 'orbit' ? 1.8 : 1.35));
    const longitudeDelta = wrapDegrees(-center[0] - view.rotate[0]);
    const latitudeDelta = -center[1] - view.rotate[1];
    const scaleDelta = Math.log((dimension * scaleRatio) / view.scale);
    if (
      view.projectionId === 'orthographic' &&
      Math.abs(longitudeDelta) < 0.002 &&
      Math.abs(latitudeDelta) < 0.002 &&
      Math.abs(view.rotate[2]) < 0.002 &&
      Math.abs(scaleDelta) < 0.0001 &&
      Math.hypot(...view.pan) < 0.1
    )
      return false;
    view.rotate = [
      wrapDegrees(view.rotate[0] + longitudeDelta * blend),
      view.rotate[1] + latitudeDelta * blend,
      view.rotate[2] * (1 - blend),
    ];
    view.scale *= Math.exp(scaleDelta * blend);
    view.pan = [view.pan[0] * (1 - blend), view.pan[1] * (1 - blend)];
    return true;
  }
}
