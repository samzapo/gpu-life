import { colonyLosses } from './losses';
import { modeEnabled, saveMode } from './modes';
import panel from './panel.html?raw';
import './game.css';
import {
  countSurvivors,
  findColonies,
  ticksPerSecond,
  type Bloom,
  type Colony,
  type Force,
  type Point,
  type Snapshot,
} from './world';

interface Globe {
  ready(): boolean;
  snapshot(tick: number): Promise<Snapshot>;
  restore(state: Snapshot): void;
  step(force: Force | undefined, bloom: Bloom | undefined): void;
  focus(point: Point): void;
  focusing(): boolean;
  forceStarted(): void;
  leaveFocus(): void;
  continueRescue(colony: Colony, baseline: Snapshot, outcome: Snapshot): void;
  observeColonies(colonies: Colony[], state: Snapshot): void;
  highlight(colony: Colony, baseline: Snapshot): void;
  project(point: Point): { x: number; y: number; scale: number } | null;
  pointer(x: number, y: number): Point | null;
  groupRadius(): number;
  settingsLocked(locked: boolean, incident?: boolean): void;
}

type Mode =
  | 'off'
  | 'watch'
  | 'film'
  | 'countdown'
  | 'play'
  | 'between'
  | 'result';

type Observation = { state: Snapshot; colonies: Colony[] };

type AttemptResult = 'pending' | 'active' | 'used' | 'cleared';

type Incident = {
  colony: Colony;
  healthy: Snapshot;
  original: Snapshot;
  startState: Snapshot;
  start: number;
  attempt: number;
  maxAttempts: number;
  attemptResults: AttemptResult[];
  bestSurvivors: number;
  bestScore: number;
  bestCleared: boolean;
  bestOutcome?: Snapshot;
  originalLosses: string;
  lastLosses: string;
};

interface SwipeNode {
  x: number;
  y: number;
  bornAt: number;
  strength: number;
}

const MAX_ATTEMPTS = 3;
const COUNTDOWN_STEP_MS = 850;
const COUNTDOWN_TOTAL_MS = 3 * COUNTDOWN_STEP_MS;
const BETWEEN_PAUSE_MS = 1600;
const RESULT_HOLD_MS = 2600;
const SWIPE_RADIUS = 0.17;
const SWIPE_DECAY_MS = 115;
const SWIPE_TRAIL_MS = 420;

const seconds = (ticks: number) => (ticks / ticksPerSecond).toFixed(1) + 's';
const element = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;
const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, value));

export class TemporalGame {
  mode: Mode = 'off';
  private world: Globe;
  private tick = 0;
  private epoch = 0;
  private waiting = false;
  private observationStart = 0;
  private history: Observation[] = [];
  private attemptCheckpoints: Snapshot[] = [];
  private blooms: Bloom[] = [];
  private exposed?: { colony: Colony; baseline: Snapshot; bloom: Bloom };
  private incident?: Incident;
  private swipeTicks = new Set<number>();
  private activeSweep?: Force & { updatedAt: number };
  private swipeTrail: SwipeNode[] = [];
  private pointerId?: number;
  private lastClient?: [number, number];
  private lastSphere?: Point;
  private lastHealthyCount = 0;
  private nextStepMs = 0;
  private filmStartMs = 0;
  private lastFilmTick = -1;
  private phaseEndMs = 0;
  private lastCountdownDigit = 0;
  private lastDisplayedScore = -1;
  private intervalTicks = ticksPerSecond;
  private points = 0;
  private disabledReason = '';
  private panel: HTMLElement;
  private timeline: HTMLInputElement;
  private swipeCanvas: HTMLCanvasElement;
  private swipeCtx: CanvasRenderingContext2D | null;

  constructor(world: Globe) {
    this.world = world;
    document.body.insertAdjacentHTML('beforeend', panel);
    this.panel = element('temporal-panel');
    this.timeline = element('game-timeline');
    this.swipeCanvas = element<HTMLCanvasElement>('swipe-canvas');
    this.swipeCtx = this.swipeCanvas.getContext('2d');
    try {
      this.points = Math.max(
        0,
        Number(localStorage.getItem('gpu-life:rescue-points')) || 0,
      );
    } catch {
      /* private mode */
    }
    element('game-toggle').onclick = () => {
      if (this.mode === 'off') this.watch();
      else this.close();
      saveMode('game', this.mode !== 'off');
    };

    const canvas = element<HTMLCanvasElement>('canvas');
    canvas.addEventListener(
      'pointerdown',
      (event) => {
        if (
          this.mode !== 'play' ||
          (event.pointerType === 'mouse' && event.button !== 0) ||
          this.pointerId !== undefined
        )
          return;
        event.preventDefault();
        event.stopPropagation();
        this.world.forceStarted();
        this.pointerId = event.pointerId;
        this.lastClient = [event.clientX, event.clientY];
        this.lastSphere =
          this.world.pointer(event.clientX, event.clientY) ?? undefined;
        try {
          canvas.setPointerCapture(event.pointerId);
        } catch {
          /* synthetic event */
        }
      },
      true,
    );

    canvas.addEventListener('pointermove', (event) => {
      if (this.mode !== 'play' || event.pointerId !== this.pointerId) return;
      const prevClient = this.lastClient;
      const prevSphere = this.lastSphere;
      const currClient: [number, number] = [event.clientX, event.clientY];
      const currSphere =
        this.world.pointer(event.clientX, event.clientY) ?? undefined;
      this.lastClient = currClient;
      if (currSphere) this.lastSphere = currSphere;

      if (!prevClient || !prevSphere || !currSphere) return;
      const pixelDelta = Math.hypot(
        currClient[0] - prevClient[0],
        currClient[1] - prevClient[1],
      );
      if (pixelDelta < 1.5) return;

      const dx = currSphere[0] - prevSphere[0];
      const dy = currSphere[1] - prevSphere[1];
      const dz = currSphere[2] - prevSphere[2];
      const radialDot =
        dx * currSphere[0] + dy * currSphere[1] + dz * currSphere[2];
      const tx = dx - currSphere[0] * radialDot;
      const ty = dy - currSphere[1] * radialDot;
      const tz = dz - currSphere[2] * radialDot;
      const tLen = Math.hypot(tx, ty, tz);
      if (tLen <= 1e-5) return;

      const now = performance.now();
      const rawStrength = clamp(pixelDelta / 14, 0.45, 2.2);
      const prevStrength =
        this.activeSweep && now - this.activeSweep.updatedAt < SWIPE_DECAY_MS
          ? this.activeSweep.strength * 0.45
          : 0;
      const strength = clamp(Math.max(rawStrength, prevStrength), 0.45, 2.2);

      this.activeSweep = {
        point: currSphere,
        direction: [tx / tLen, ty / tLen, tz / tLen],
        strength,
        radius: SWIPE_RADIUS,
        updatedAt: now,
      };
      this.swipeTrail.push({
        x: currClient[0],
        y: currClient[1],
        bornAt: now,
        strength,
      });
      this.swipeTicks.add(this.tick + 1);
    });

    const release = (event: PointerEvent) => {
      if (this.pointerId === event.pointerId) this.releasePointerOnly();
    };
    for (const type of [
      'pointerup',
      'pointercancel',
      'lostpointercapture',
    ] as const) {
      canvas.addEventListener(type, release);
    }
    window.addEventListener('blur', () => this.release());
    document.addEventListener('visibilitychange', () => {
      this.release();
      this.nextStepMs = 0;
    });

    if (modeEnabled('game')) this.watch();
    else this.display();
  }

  unavailable(message: string) {
    this.disabledReason = message;
    this.display();
  }

  reset() {
    if (this.mode !== 'off') this.watch();
  }

  get canInspect() {
    return (
      this.mode === 'off' || this.mode === 'watch' || this.inspectionPaused
    );
  }

  get inspectionPaused() {
    return (
      this.mode === 'countdown' ||
      this.mode === 'between' ||
      this.mode === 'result'
    );
  }

  private releasePointerOnly() {
    this.pointerId = undefined;
    this.lastClient = undefined;
    this.lastSphere = undefined;
  }

  private release() {
    this.releasePointerOnly();
    this.activeSweep = undefined;
    this.swipeTrail.length = 0;
    this.clearSwipeCanvas();
  }

  private clearSwipeCanvas() {
    if (!this.swipeCtx) return;
    this.swipeCtx.clearRect(
      0,
      0,
      this.swipeCanvas.width,
      this.swipeCanvas.height,
    );
  }

  private clear() {
    this.epoch++;
    this.waiting = false;
    this.release();
    this.history = [];
    this.attemptCheckpoints = [];
    this.swipeTicks.clear();
    this.blooms = [];
    this.exposed = undefined;
    this.incident = undefined;
    this.nextStepMs = 0;
    this.phaseEndMs = 0;
    this.lastCountdownDigit = 0;
    this.lastHealthyCount = 0;
    this.world.leaveFocus();
  }

  private watch() {
    if (this.disabledReason.startsWith('Checkpoint memory'))
      this.disabledReason = '';
    this.clear();
    this.mode = 'watch';
    this.world.settingsLocked(true);
    this.tick = 0;
    this.observationStart = 0;
    this.display();
  }

  private close() {
    if (this.incident && this.mode !== 'result')
      this.world.restore(this.incident.original);
    this.clear();
    this.mode = 'off';
    this.world.settingsLocked(false);
    this.display();
  }

  private startTick() {
    return this.incident?.start ?? 0;
  }

  private endTick() {
    return this.incident?.original.tick ?? this.tick;
  }

  private activeBloom(tick: number) {
    return this.blooms.find(
      (bloom) => tick >= bloom.startTick && tick <= bloom.endTick,
    );
  }

  frame(nowMs: number) {
    this.renderSwipeTrail(nowMs);
    if (
      this.mode === 'off' ||
      this.disabledReason ||
      !this.world.ready() ||
      document.hidden
    )
      return;
    this.positionMarkers();

    if (this.mode === 'film') {
      this.film(nowMs);
      return;
    }

    if (this.mode === 'countdown') {
      const remaining = Math.max(0, this.phaseEndMs - nowMs);
      const digit = Math.max(1, Math.ceil(remaining / COUNTDOWN_STEP_MS));
      if (digit !== this.lastCountdownDigit) {
        this.lastCountdownDigit = digit;
        this.display();
      }
      if (remaining <= 0) {
        this.mode = 'play';
        this.nextStepMs = 0;
        this.swipeTicks.clear();
        this.activeSweep = undefined;
        this.attemptCheckpoints = [this.incident!.startState];
        this.display();
      }
      return;
    }

    if (this.mode === 'between') {
      if (nowMs >= this.phaseEndMs && this.incident) {
        this.startAttemptCountdown(nowMs, this.incident.attempt + 1);
      }
      return;
    }

    if (this.mode === 'result') {
      if (nowMs >= this.phaseEndMs && this.incident) {
        const incident = this.incident;
        if (incident.bestCleared && incident.bestOutcome) {
          this.world.continueRescue(
            incident.colony,
            incident.healthy,
            incident.bestOutcome,
          );
        } else {
          this.world.restore(incident.original);
        }
        this.watch();
      }
      return;
    }

    if (this.waiting) return;

    if (this.mode === 'watch' && !this.history.length) {
      this.capture();
      return;
    }

    if (nowMs < this.nextStepMs) return;
    const intervalMs = 1000 / ticksPerSecond;
    if (!this.nextStepMs) this.nextStepMs = nowMs;
    this.nextStepMs +=
      (1 + Math.floor((nowMs - this.nextStepMs) / intervalMs)) * intervalMs;

    if (this.incident && this.tick >= this.endTick()) {
      return;
    }

    const nextTick = this.tick + 1;
    let force: Force | undefined;
    if (
      this.mode === 'play' &&
      this.activeSweep &&
      nowMs - this.activeSweep.updatedAt <= SWIPE_DECAY_MS &&
      this.activeSweep.strength > 0.08
    ) {
      force = {
        point: [...this.activeSweep.point],
        direction: [...this.activeSweep.direction],
        strength: this.activeSweep.strength,
        radius: this.activeSweep.radius,
      };
      this.activeSweep.strength *= 0.68;
    }

    this.world.step(force, this.activeBloom(nextTick));
    this.tick = nextTick;

    if (
      this.tick % this.intervalTicks === 0 ||
      (this.incident && this.tick === this.endTick())
    ) {
      this.capture();
    } else if (this.mode === 'play') {
      this.display();
    }
  }

  private capture() {
    const epoch = this.epoch;
    this.waiting = true;
    void this.world
      .snapshot(this.tick)
      .then((state) => {
        if (this.epoch !== epoch) return;
        this.waiting = false;
        if (this.mode === 'watch') {
          this.observe(state);
        } else if (this.mode === 'play' && this.incident) {
          this.attemptCheckpoints.push(state);
          this.sample(state);
          if (state.tick >= this.endTick()) {
            this.finishAttempt(state);
          }
        }
        this.display();
      })
      .catch(() => {
        if (this.epoch !== epoch) return;
        this.clear();
        this.world.settingsLocked(false);
        this.mode = 'watch';
        this.unavailable(
          'Checkpoint memory is unavailable. Try fewer particles and enable game mode again.',
        );
      });
  }

  private observe(state: Snapshot) {
    if (!this.history.length) {
      const checkpointBytes =
        state.life.byteLength +
        state.particles.byteLength +
        state.episodes.byteLength +
        state.deathHistory.byteLength +
        (state.inspection?.valence.byteLength ?? 0) +
        (state.inspection?.bioForce.byteLength ?? 0);
      const slots = Math.max(
        3,
        Math.min(41, Math.floor((48 << 20) / checkpointBytes)),
      );
      this.intervalTicks = Math.ceil(20 / (slots - 1)) * ticksPerSecond;
    }
    const colonies = findColonies(state, this.world.groupRadius());
    this.world.observeColonies(colonies, state);
    this.history.push({ state, colonies });
    while (
      this.history.length > 2 &&
      this.history[1].state.tick < state.tick - 22 * ticksPerSecond
    )
      this.history.shift();
    this.lastHealthyCount = colonies[0]?.members.length ?? 0;
    if (state.tick - this.observationStart < 8 * ticksPerSecond) return;
    const reference = this.history
      .filter((row) => row.state.tick <= state.tick - 3 * ticksPerSecond)
      .at(-1);
    if (!reference) return;
    const minimum = Math.max(
      5,
      Math.min(30, Math.floor(state.life.length / 4 / 140)),
    );
    const exposed = this.exposed;
    if (exposed && state.tick >= exposed.bloom.endTick) {
      this.exposed = undefined;
      if (
        countSurvivors(state, exposed.colony.members, exposed.baseline) <=
        exposed.colony.members.length * 0.42
      ) {
        if (this.beginIncident(exposed.colony, exposed.baseline, state)) return;
      }
    }
    const previous = this.history.at(-2)!.state;
    for (const colony of reference.colonies) {
      if (colony.members.length < minimum) break;
      if (exposed && state.tick <= exposed.bloom.endTick) continue;
      if (
        countSurvivors(state, colony.members, reference.state) <=
          colony.members.length * 0.42 &&
        countSurvivors(previous, colony.members, reference.state) <=
          colony.members.length * 0.7
      ) {
        if (this.beginIncident(colony, reference.state, state)) return;
      }
    }
    const latest = this.blooms.at(-1);
    if (latest && state.tick <= latest.endTick + 3 * ticksPerSecond) return;
    for (const colony of colonies) {
      if (colony.members.length < minimum) break;
      const distances = colony.members
        .map((index) => {
          const offset = index * 12;
          return Math.hypot(
            state.particles[offset] - colony.center[0],
            state.particles[offset + 1] - colony.center[1],
            state.particles[offset + 2] - colony.center[2],
          );
        })
        .sort((left, right) => left - right);
      const radius = Math.max(
        0.075,
        distances[Math.floor(distances.length * 0.85)] * 1.4,
      );
      if (radius > 0.17) continue;
      const bloom = {
        point: colony.center,
        radius,
        startTick: state.tick + ticksPerSecond,
        endTick: state.tick + 4 * ticksPerSecond,
      };
      this.blooms = this.blooms.filter(
        (item) => item.endTick >= this.history[0].state.tick,
      );
      this.blooms.push(bloom);
      this.exposed = { colony, baseline: state, bloom };
      break;
    }
  }

  private beginIncident(colony: Colony, healthy: Snapshot, original: Snapshot) {
    const losses = colonyLosses(
      colony.members,
      healthy,
      original,
      this.history.map((row) => row.state),
    );
    if (!losses.qualifies) return false;
    const anchor =
      this.history
        .filter((row) => row.state.tick <= healthy.tick - ticksPerSecond)
        .at(-1) ?? this.history[0];
    this.incident = {
      colony,
      healthy,
      original,
      startState: anchor.state,
      start: anchor.state.tick,
      attempt: 1,
      maxAttempts: MAX_ATTEMPTS,
      attemptResults: ['pending', 'pending', 'pending'],
      bestSurvivors: 0,
      bestScore: 0,
      bestCleared: false,
      originalLosses: losses.summary,
      lastLosses: '',
    };
    this.lastHealthyCount = countSurvivors(original, colony.members, healthy);
    this.mode = 'film';
    this.filmStartMs = 0;
    this.lastFilmTick = -1;
    this.world.settingsLocked(true, true);
    this.world.highlight(colony, healthy);
    this.world.focus(colony.center);
    this.display();
    return true;
  }

  private film(nowMs: number) {
    const incident = this.incident!;
    if (this.world.focusing()) {
      this.display();
      return;
    }
    if (!this.filmStartMs) this.filmStartMs = nowMs;
    const elapsed = (nowMs - this.filmStartMs) / 1000;
    const span = incident.original.tick - incident.start;
    const phase = Math.min(1, elapsed / 1.8);
    const target = Math.round(incident.start + phase * span);
    this.tick = target;
    if (target !== this.lastFilmTick) {
      const before = this.history
        .filter((row) => row.state.tick <= target)
        .at(-1)!.state;
      const after = this.history.find((row) => row.state.tick > target)?.state;
      if (after && target > before.tick) {
        const fraction = (target - before.tick) / (after.tick - before.tick);
        const particles = before.particles.slice();
        for (let offset = 0; offset < particles.length; offset += 12) {
          for (let axis = 0; axis < 3; axis++)
            particles[offset + axis] +=
              (after.particles[offset + axis] - particles[offset + axis]) *
              fraction;
        }
        this.world.restore({
          ...before,
          particles,
          life: fraction < 0.5 ? before.life : after.life,
        });
      } else this.world.restore(before);
      this.lastFilmTick = target;
      this.display();
    }
    if (elapsed >= 2.1) {
      this.startAttemptCountdown(nowMs, 1);
    }
  }

  private startAttemptCountdown(nowMs: number, attemptNumber: number) {
    const incident = this.incident;
    if (!incident) return;
    this.release();
    incident.attempt = attemptNumber;
    for (let i = 0; i < incident.maxAttempts; i++) {
      if (i + 1 === attemptNumber) incident.attemptResults[i] = 'active';
      else if (i + 1 > attemptNumber) incident.attemptResults[i] = 'pending';
    }
    this.world.restore(incident.startState);
    this.tick = incident.start;
    this.sample(incident.startState);
    this.swipeTicks.clear();
    this.mode = 'countdown';
    this.phaseEndMs = nowMs + COUNTDOWN_TOTAL_MS;
    this.lastCountdownDigit = 3;
    this.display();
  }

  private sample(state: Snapshot) {
    if (this.incident)
      this.lastHealthyCount = countSurvivors(
        state,
        this.incident.colony.members,
        this.incident.healthy,
      );
  }

  private finishAttempt(outcome: Snapshot) {
    const incident = this.incident;
    if (!incident) return;
    this.release();
    const survivors = countSurvivors(
      outcome,
      incident.colony.members,
      incident.healthy,
    );
    this.lastHealthyCount = survivors;
    const target = Math.ceil(incident.colony.members.length * 0.6);
    const cleared = survivors >= target;
    incident.lastLosses = colonyLosses(
      incident.colony.members,
      incident.healthy,
      outcome,
      this.attemptCheckpoints,
    ).summary;

    const score =
      survivors * 10 +
      (cleared ? 100 * (incident.maxAttempts + 1 - incident.attempt) : 0);
    const gain = Math.max(0, score - incident.bestScore);
    incident.bestScore = Math.max(incident.bestScore, score);
    incident.bestSurvivors = Math.max(incident.bestSurvivors, survivors);
    this.points += gain;
    try {
      localStorage.setItem('gpu-life:rescue-points', String(this.points));
    } catch {
      /* session score remains */
    }

    const now = performance.now();
    if (cleared) {
      incident.attemptResults[incident.attempt - 1] = 'cleared';
      incident.bestCleared = true;
      incident.bestOutcome = outcome;
      this.mode = 'result';
      this.phaseEndMs = now + RESULT_HOLD_MS;
    } else {
      incident.attemptResults[incident.attempt - 1] = 'used';
      if (incident.attempt < incident.maxAttempts) {
        this.mode = 'between';
        this.phaseEndMs = now + BETWEEN_PAUSE_MS;
      } else {
        this.mode = 'result';
        this.phaseEndMs = now + RESULT_HOLD_MS;
      }
    }
  }

  private renderSwipeTrail(nowMs: number) {
    const ctx = this.swipeCtx;
    if (!ctx) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (this.swipeCanvas.width !== w || this.swipeCanvas.height !== h) {
      this.swipeCanvas.width = w;
      this.swipeCanvas.height = h;
    }
    while (
      this.swipeTrail.length &&
      nowMs - this.swipeTrail[0].bornAt > SWIPE_TRAIL_MS
    ) {
      this.swipeTrail.shift();
    }
    ctx.clearRect(0, 0, w, h);
    if (this.mode !== 'play' || this.swipeTrail.length < 2) return;

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let i = 1; i < this.swipeTrail.length; i++) {
      const a = this.swipeTrail[i - 1];
      const b = this.swipeTrail[i];
      const age = (nowMs - b.bornAt) / SWIPE_TRAIL_MS;
      const life = Math.max(0, 1 - age);
      const width = (4 + b.strength * 8) * life;

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(255, 42, 133, ${0.38 * life})`;
      ctx.lineWidth = width + 8 * life;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(0, 240, 255, ${0.88 * life})`;
      ctx.lineWidth = width;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(230, 255, 255, ${0.9 * life})`;
      ctx.lineWidth = Math.max(1.5, width * 0.35);
      ctx.stroke();
    }

    const head = this.swipeTrail.at(-1)!;
    const headLife = Math.max(
      0,
      1 - (nowMs - head.bornAt) / (SWIPE_TRAIL_MS * 0.65),
    );
    if (headLife > 0.05) {
      ctx.beginPath();
      ctx.arc(head.x, head.y, 12 + (1 - headLife) * 26, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(0, 240, 255, ${0.65 * headLife})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  private positionMarkers() {
    const bloom =
      this.activeBloom(this.tick) ??
      this.blooms.find(
        (item) =>
          item.startTick >= this.tick &&
          (!this.incident || item.startTick <= this.incident.original.tick),
      );
    const point = bloom?.point ?? this.incident?.colony.center;
    const marker = element('event-marker');
    const showMarker =
      this.mode === 'film' ||
      this.mode === 'countdown' ||
      this.mode === 'play' ||
      this.mode === 'between';
    const projected = showMarker && point && this.world.project(point);
    marker.hidden = !projected;
    if (projected) {
      const size = Math.max(24, 2 * (bloom?.radius ?? 0.06) * projected.scale);
      marker.style.cssText =
        'left:' +
        projected.x +
        'px;top:' +
        projected.y +
        'px;width:' +
        size +
        'px;height:' +
        size +
        'px';
      marker.classList.toggle('active', Boolean(this.activeBloom(this.tick)));
    }
  }

  private display() {
    const mode = this.mode;
    const incident = this.incident;
    this.panel.hidden = mode === 'off';
    document.body.dataset.temporal = mode;
    element('game-toggle').setAttribute('aria-pressed', String(mode !== 'off'));

    if (mode === 'off') {
      element('event-marker').hidden = true;
      element('cyber-hud').hidden = true;
      return;
    }

    if (this.points !== this.lastDisplayedScore) {
      element('game-score').textContent = this.points + ' PTS';
      this.lastDisplayedScore = this.points;
    }

    const pips = element('attempt-pips').querySelectorAll('i');
    pips.forEach((pip, idx) => {
      const state = incident?.attemptResults[idx] ?? 'pending';
      pip.className = state === 'pending' ? '' : state;
    });

    const hud = element('cyber-hud');
    const hudKicker = element('cyber-kicker');
    const hudMain = element('cyber-main');
    const hudSub = element('cyber-sub');
    hud.classList.remove('hazard', 'success', 'countdown-mode');

    const statusTag = element('game-status-tag');
    const survivorReadout = element('survivor-readout');
    const clock = element('event-clock');
    const track = element('plan-track');
    const wrap = this.panel.querySelector<HTMLElement>('.timeline-wrap')!;

    if (mode === 'watch' || !incident) {
      hud.hidden = true;
      survivorReadout.hidden = true;
      statusTag.textContent = this.disabledReason
        ? 'LINK OFFLINE'
        : !this.world.ready()
          ? 'SYNCING GLOBE'
          : this.lastHealthyCount
            ? `SCANNING // ${this.lastHealthyCount} CELLS`
            : 'SCANNING BIOSPHERE';
      clock.textContent = this.history.length
        ? seconds(this.tick - (this.history[0]?.state.tick ?? 0))
        : '0.0s';
      this.timeline.min = '0';
      this.timeline.max = '100';
      this.timeline.value = '0';
      wrap.style.setProperty('--progress', '0%');
      track.replaceChildren();
      return;
    }

    const start = this.startTick();
    const end = this.endTick();
    const span = Math.max(1, end - start);
    const pct = clamp(((this.tick - start) / span) * 100, 0, 100);
    this.timeline.min = String(start);
    this.timeline.max = String(end);
    this.timeline.value = String(this.tick);
    wrap.style.setProperty('--progress', `${pct.toFixed(2)}%`);
    clock.textContent = seconds(this.tick - end);

    const total = incident.colony.members.length;
    const target = Math.ceil(total * 0.6);
    survivorReadout.hidden = false;
    survivorReadout.textContent = `${this.lastHealthyCount}/${total}`;
    survivorReadout.classList.toggle('clear', this.lastHealthyCount >= target);

    // Render bloom danger zone & live swipe activity on the bottom progress bar
    track.replaceChildren();
    for (const bloom of this.blooms) {
      if (bloom.endTick < start || bloom.startTick > end) continue;
      const bStart = clamp(bloom.startTick, start, end);
      const bEnd = clamp(bloom.endTick, start, end);
      const zone = document.createElement('i');
      zone.className = 'bloom-zone';
      zone.style.left = `${((bStart - start) / span) * 100}%`;
      zone.style.width = `${((bEnd - bStart) / span) * 100}%`;
      track.append(zone);
    }
    for (const sTick of this.swipeTicks) {
      if (sTick < start || sTick > end) continue;
      const mark = document.createElement('i');
      mark.className = 'swipe-mark';
      mark.style.left = `${((sTick - start) / span) * 100}%`;
      mark.style.width = `${Math.max(1, 100 / span)}%`;
      track.append(mark);
    }

    if (mode === 'film') {
      statusTag.textContent = 'HAZARD DETECTED';
      hud.hidden = false;
      hud.classList.add('hazard');
      hudKicker.textContent = 'BIO-HAZARD ALERT';
      hudMain.textContent = this.world.focusing()
        ? 'LOCKING TARGET'
        : 'COLONY COLLAPSE';
      hudSub.textContent = incident.originalLosses;
    } else if (mode === 'countdown') {
      statusTag.textContent = `TRY ${incident.attempt}/${incident.maxAttempts} // READY`;
      hud.hidden = false;
      hud.classList.add('countdown-mode');
      hudKicker.textContent = `ATTEMPT ${incident.attempt} OF ${incident.maxAttempts}`;
      hudMain.textContent = String(this.lastCountdownDigit || 1);
      hudSub.textContent = 'SWIPE TO SWEEP CELLS AWAY FROM HAZARD';
    } else if (mode === 'play') {
      statusTag.textContent = `TRY ${incident.attempt}/${incident.maxAttempts} // SWIPE TO DEFLECT`;
      hud.hidden = true;
    } else if (mode === 'between') {
      statusTag.textContent = `TRY ${incident.attempt}/${incident.maxAttempts} FAILED`;
      hud.hidden = false;
      hud.classList.add('hazard');
      hudKicker.textContent = `ATTEMPT ${incident.attempt} OF ${incident.maxAttempts} FAILED`;
      hudMain.textContent = `${this.lastHealthyCount} / ${total} SAVED`;
      hudSub.textContent = `NEED ${target} (60%) · NEXT ATTEMPT INCOMING`;
    } else if (mode === 'result') {
      hud.hidden = false;
      if (incident.bestCleared) {
        statusTag.textContent = 'COLONY RESCUED';
        hud.classList.add('success');
        hudKicker.textContent = `CLEARED ON TRY ${incident.attempt}`;
        hudMain.textContent = 'TIMELINE SECURED';
        hudSub.textContent = `${incident.bestSurvivors} / ${total} SURVIVORS · +${incident.bestScore} PTS`;
      } else {
        statusTag.textContent = 'TIMELINE LOST';
        hud.classList.add('hazard');
        hudKicker.textContent = '3 OF 3 ATTEMPTS EXHAUSTED';
        hudMain.textContent = 'COLONY COLLAPSED';
        hudSub.textContent = `BEST ${incident.bestSurvivors} / ${total} SAVED (NEED ${target})`;
      }
    }
  }
}
