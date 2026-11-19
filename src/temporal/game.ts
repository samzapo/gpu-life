import { colonyLosses } from './losses';
import { modeEnabled, saveMode } from './modes';
import panel from './panel.html?raw';
import './game.css';
import {
  budgetTicks,
  countSurvivors,
  earlierSecondsPerForceSecond,
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
type Mode = 'off' | 'watch' | 'film' | 'confirm' | 'edit' | 'commit' | 'result';
type Observation = { state: Snapshot; colonies: Colony[] };
type Incident = {
  colony: Colony;
  healthy: Snapshot;
  original: Snapshot;
  start: number;
  best: number;
  originalLosses: string;
  committedLosses: string;
};
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
  private branch = new Map<number, Snapshot>();
  private blooms: Bloom[] = [];
  private exposed?: { colony: Colony; baseline: Snapshot; bloom: Bloom };
  private incident?: Incident;
  private plan = new Map<number, Force>();
  private held?: Force;
  private pointerId?: number;
  private pendingHold?: { id: number; x: number; y: number; timer: number };
  private sign = 1;
  private nextStroke = 0;
  private lastHealthyCount = 0;
  private extraTicks = 0;
  private seekTo?: number;
  private running = true;
  private speed = 1;
  private nextStepMs = 0;
  private filmStartMs = 0;
  private lastFilmTick = -1;
  private lastDisplayedScore = -1;
  private intervalTicks = ticksPerSecond;
  private points = 0;
  private disabledReason = '';
  private panel: HTMLElement;
  private timeline: HTMLInputElement;
  private arrival: HTMLInputElement;
  private speedSelect: HTMLSelectElement;
  private strokes: HTMLSelectElement;

  constructor(world: Globe) {
    this.world = world;
    document.body.insertAdjacentHTML('beforeend', panel);
    this.panel = element('temporal-panel');
    this.timeline = element('game-timeline');
    this.arrival = element('game-arrival');
    this.speedSelect = element('game-speed');
    this.strokes = element('game-strokes');
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
    this.panel.addEventListener('click', (event) => {
      const action = (event.target as HTMLElement).closest('button')?.dataset
        .action;
      if (action) this.action(action);
    });
    this.timeline.oninput = () => {
      if (this.mode !== 'edit') return;
      this.running = false;
      this.seek(Number(this.timeline.value));
      this.display();
    };
    this.arrival.oninput = () => this.setArrival(Number(this.arrival.value));
    this.speedSelect.onchange = () => {
      this.speed = Number(this.speedSelect.value);
      this.nextStepMs = 0;
    };
    const canvas = element<HTMLCanvasElement>('canvas');
    canvas.addEventListener(
      'pointerdown',
      (event) => {
        if (this.pendingHold) {
          this.cancelHold();
          return;
        }
        if (
          this.mode !== 'edit' ||
          event.button !== 0 ||
          this.seekTo !== undefined ||
          this.pointerId !== undefined
        )
          return;
        const point = this.world.pointer(event.clientX, event.clientY);
        if (!point || this.tick >= this.endTick() || this.leftTicks() <= 0)
          return;
        const startForce = () => {
          this.world.forceStarted();
          this.pointerId = event.pointerId;
          this.held = { point, sign: this.sign, stroke: ++this.nextStroke };
          this.running = true;
          this.nextStepMs = 0;
        };
        canvas.setPointerCapture(event.pointerId);
        if (event.pointerType !== 'mouse' && !this.running) {
          // Leave a short tap to the ordinary inspector. A sustained press starts
          // playback and charges only the ticks that actually receive a force.
          if (!event.isPrimary) return;
          const timer = window.setTimeout(() => {
            this.pendingHold = undefined;
            startForce();
          }, 320);
          this.pendingHold = {
            id: event.pointerId,
            x: event.clientX,
            y: event.clientY,
            timer,
          };
        } else {
          event.preventDefault();
          event.stopPropagation();
          startForce();
        }
      },
      true,
    );
    canvas.addEventListener('pointermove', (event) => {
      const pending = this.pendingHold;
      if (
        pending?.id === event.pointerId &&
        Math.hypot(event.clientX - pending.x, event.clientY - pending.y) > 12
      )
        this.cancelHold();
      if (event.pointerId !== this.pointerId || !this.held) return;
      const point = this.world.pointer(event.clientX, event.clientY);
      if (point) this.held.point = point;
    });
    const release = (event: PointerEvent) => {
      if (this.pendingHold?.id === event.pointerId) this.cancelHold();
      if (this.pointerId === event.pointerId) this.release();
    };
    for (const type of [
      'pointerup',
      'pointercancel',
      'lostpointercapture',
    ] as const)
      canvas.addEventListener(type, release);
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
      this.mode === 'confirm' ||
      this.mode === 'result' ||
      ((this.mode === 'edit' || this.mode === 'commit') &&
        !this.running &&
        this.seekTo === undefined)
    );
  }
  private cancelHold() {
    if (this.pendingHold) window.clearTimeout(this.pendingHold.timer);
    this.pendingHold = undefined;
  }
  private release() {
    this.cancelHold();
    this.held = undefined;
    this.pointerId = undefined;
    element('force-marker').hidden = true;
  }
  private clear() {
    this.epoch++;
    this.waiting = false;
    this.release();
    this.strokes.dataset.key = '';
    this.history = [];
    this.branch.clear();
    this.plan.clear();
    this.blooms = [];
    this.exposed = undefined;
    this.incident = undefined;
    this.seekTo = undefined;
    this.extraTicks = 0;
    this.nextStepMs = 0;
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
    this.running = true;
    this.speed = 1;
    this.speedSelect.value = '1';
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
  private arrivalTick() {
    return (this.incident?.start ?? 0) - this.extraTicks;
  }
  private endTick() {
    return this.incident?.original.tick ?? this.tick;
  }
  private leftTicks() {
    return (
      budgetTicks -
      this.plan.size -
      this.extraTicks / earlierSecondsPerForceSecond
    );
  }
  private activeBloom(tick: number) {
    return this.blooms.find(
      (bloom) => tick >= bloom.startTick && tick <= bloom.endTick,
    );
  }

  frame(nowMs: number) {
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
    if (this.waiting || this.mode === 'confirm' || this.mode === 'result')
      return;
    if (this.mode === 'watch' && !this.history.length) {
      this.capture();
      return;
    }
    const seeking = this.seekTo !== undefined;
    if (!seeking && (!this.running || nowMs < this.nextStepMs)) return;
    // A late frame advances once. The next deadline stays on the original clock.
    const speed = this.mode === 'watch' ? 1 : this.speed;
    const batch = Math.ceil(speed);
    const intervalMs = (1000 * batch) / (ticksPerSecond * speed);
    if (!seeking) {
      if (!this.nextStepMs) this.nextStepMs = nowMs;
      this.nextStepMs +=
        (1 + Math.floor((nowMs - this.nextStepMs) / intervalMs)) * intervalMs;
    }
    const steps = seeking ? Math.min(6, this.seekTo! - this.tick) : batch;
    for (let step = 0; step < steps; step++) {
      if (this.incident && this.tick >= this.endTick()) break;
      const nextTick = this.tick + 1;
      if (
        this.mode === 'edit' &&
        !seeking &&
        this.held &&
        (this.plan.has(nextTick) || this.leftTicks() >= 1)
      ) {
        this.plan.set(nextTick, { ...this.held, point: [...this.held.point] });
        this.invalidate(nextTick);
      }
      const cached = this.incident ? this.saved(nextTick) : undefined;
      if (cached) this.world.restore(cached);
      else
        this.world.step(
          this.mode === 'watch' ? undefined : this.plan.get(nextTick),
          this.activeBloom(nextTick),
        );
      this.tick = nextTick;
      if (
        this.tick % this.intervalTicks === 0 ||
        (this.incident && this.tick === this.endTick())
      ) {
        if (cached) this.sample(cached);
        else {
          this.capture();
          break;
        }
      }
    }
    if (this.seekTo !== undefined && this.tick >= this.seekTo)
      this.seekTo = undefined;
    if (!this.waiting && this.incident && this.tick >= this.endTick())
      this.atEnd();
    if (this.mode !== 'watch') this.display();
  }

  private capture() {
    const epoch = this.epoch;
    this.waiting = true;
    void this.world
      .snapshot(this.tick)
      .then((state) => {
        if (this.epoch !== epoch) return;
        this.waiting = false;
        if (this.mode === 'watch') this.observe(state);
        else {
          this.branch.set(state.tick, state);
          this.sample(state);
          if (state.tick === this.endTick()) this.atEnd();
        }
        this.display();
      })
      .catch(() => {
        if (this.epoch !== epoch) return;
        this.clear();
        this.world.settingsLocked(false);
        this.mode = 'watch';
        this.running = false;
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
      // Space for both the original recording and an edited branch.
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
    // Seed a geographically fixed bloom near a compact, living colony.
    // A rewind is offered only after measured losses, never on the bloom timer.
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
      start: anchor.state.tick,
      best: 0,
      originalLosses: losses.summary,
      committedLosses: '',
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
      element('game-message').textContent = 'Locating the colony…';
      return;
    }
    if (!this.filmStartMs) this.filmStartMs = nowMs;
    const elapsed = (nowMs - this.filmStartMs) / 1000;
    const caption =
      elapsed < 2
        ? 'Original timeline · observing the collapse'
        : 'Rewinding to the intervention window…';
    const message = element('game-message');
    if (message.textContent !== caption) message.textContent = caption;
    const span = incident.original.tick - incident.start;
    const phase =
      elapsed < 2 ? elapsed / 2 : Math.max(0, 1 - (elapsed - 2) / 1.7);
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
    }
    if (elapsed >= 3.7) {
      this.mode = 'confirm';
      this.seek(incident.start);
      this.display();
    }
  }

  private invalidate(from: number) {
    for (const tick of this.branch.keys())
      if (tick >= from) this.branch.delete(tick);
  }
  private saved(tick: number) {
    const edited = this.branch.get(tick);
    if (edited) return edited;
    for (const planned of this.plan.keys())
      if (planned <= tick) return undefined;
    return this.history.find((row) => row.state.tick === tick)?.state;
  }
  private seek(target: number) {
    if (!this.incident) return;
    this.epoch++;
    this.waiting = false;
    this.release();
    target = clamp(target, this.arrivalTick(), this.endTick());
    let base = this.history[0].state;
    for (const { state } of this.history) {
      if (state.tick > target) break;
      const available = this.saved(state.tick);
      if (available) base = available;
    }
    for (const state of this.branch.values())
      if (state.tick <= target && state.tick > base.tick) base = state;
    this.world.restore(base);
    this.tick = base.tick;
    this.sample(base);
    this.seekTo = target > base.tick ? target : undefined;
    this.nextStepMs = 0;
  }
  private sample(state: Snapshot) {
    if (this.incident)
      this.lastHealthyCount = countSurvivors(
        state,
        this.incident.colony.members,
        this.incident.healthy,
      );
  }
  private setArrival(extraSeconds: number) {
    if (this.mode !== 'edit') return;
    const proposed = extraSeconds * ticksPerSecond;
    const firstForce = Math.min(...this.plan.keys(), Infinity);
    if (
      proposed / earlierSecondsPerForceSecond + this.plan.size > budgetTicks ||
      firstForce <= this.incident!.start - proposed
    ) {
      this.display();
      return;
    }
    this.extraTicks = proposed;
    this.running = false;
    this.seek(this.arrivalTick());
    this.display();
  }
  private atEnd() {
    this.running = false;
    this.release();
    if (this.mode !== 'commit') return;
    this.mode = 'result';
    const incident = this.incident!;
    const outcome = this.saved(this.endTick())!;
    const checkpoints = this.history
      .map((row) => this.saved(row.state.tick))
      .filter((state): state is Snapshot => !!state);
    incident.committedLosses = colonyLosses(
      incident.colony.members,
      incident.healthy,
      outcome,
      checkpoints,
    ).summary;
    const score = this.runScore();
    const gain = Math.max(0, score - this.incident!.best);
    this.incident!.best = Math.max(this.incident!.best, score);
    this.points += gain;
    try {
      localStorage.setItem('gpu-life:rescue-points', String(this.points));
    } catch {
      /* session score remains */
    }
  }
  private runScore() {
    return (
      this.lastHealthyCount * 10 +
      Math.round(
        (((50 * this.leftTicks()) / budgetTicks) * this.lastHealthyCount) /
          (this.incident?.colony.members.length || 1),
      )
    );
  }
  private action(action: string) {
    const incident = this.incident;
    if (!incident || this.mode === 'film') return;
    if (action === 'skip' || action === 'continue') {
      if (action === 'skip') this.world.restore(incident.original);
      else {
        const outcome = this.saved(this.endTick());
        if (this.mode !== 'result' || !outcome) return;
        this.world.continueRescue(incident.colony, incident.healthy, outcome);
      }
      this.watch();
      return;
    }
    if (action === 'intervene' || action === 'retry' || action === 'commit') {
      this.mode = action === 'commit' ? 'commit' : 'edit';
      this.running = action === 'commit';
      this.seek(this.arrivalTick());
    } else if (action === 'play') {
      if (this.tick === this.endTick()) this.seek(this.arrivalTick());
      this.running = !this.running;
      this.nextStepMs = 0;
      this.release();
    } else if (action === 'attract' || action === 'repel')
      this.sign = action === 'attract' ? 1 : -1;
    else if (action === 'delete') {
      const stroke = Number(this.strokes.value);
      let earliest = Infinity;
      for (const [tick, force] of this.plan)
        if (force.stroke === stroke) {
          this.plan.delete(tick);
          earliest = Math.min(earliest, tick);
        }
      this.invalidate(earliest);
      this.running = false;
      this.seek(this.tick);
    }
    this.display();
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
    const projected =
      this.mode === 'film' && point && this.world.project(point);
    marker.hidden = !projected;
    if (projected) {
      const size = Math.max(20, 2 * (bloom?.radius ?? 0.04) * projected.scale);
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
    const force = this.held ?? this.plan.get(this.tick);
    const location = force && this.world.project(force.point);
    const indicator = element('force-marker');
    indicator.hidden = !location;
    if (location) {
      indicator.style.left = location.x + 'px';
      indicator.style.top = location.y + 'px';
      indicator.textContent = force!.sign === 1 ? '+' : '−';
    }
  }

  private display() {
    const mode = this.mode;
    const editing = mode === 'edit',
      committed = mode === 'commit';
    const incident = this.incident;
    this.panel.hidden = mode === 'off';
    const report = element('game-losses');
    report.hidden = mode !== 'confirm' && mode !== 'result';
    if (!report.hidden && incident) {
      element('original-losses').textContent = incident.originalLosses;
      element('committed-report').hidden = mode !== 'result';
      if (mode === 'result')
        element('committed-losses').textContent = incident.committedLosses;
    }
    document.body.dataset.temporal = mode;
    element('game-toggle').setAttribute('aria-pressed', String(mode !== 'off'));
    element('event-flash').hidden = mode !== 'film';
    if (mode !== 'film') element('event-marker').hidden = true;
    if (mode === 'off') {
      element('event-marker').hidden = true;
      return;
    }
    if (this.points !== this.lastDisplayedScore) {
      element('game-score').textContent = this.points + ' pts';
      this.lastDisplayedScore = this.points;
    }
    element('game-stats').hidden = !(editing || committed || mode === 'result');
    element('game-confirm').hidden = mode !== 'confirm';
    element('game-editor').hidden = !(editing || committed);
    element('game-planning').hidden = !editing;
    element('game-tools').hidden = !editing;
    element('game-result').hidden = mode !== 'result';
    const title = element('game-title'),
      message = element('game-message');
    if (mode === 'watch') {
      title.textContent = this.disabledReason
        ? 'Game unavailable'
        : 'Observing colonies';
      message.textContent =
        this.disabledReason ||
        (!this.world.ready()
          ? 'Preparing the globe…'
          : this.lastHealthyCount
            ? 'Rewind buffer ' +
              seconds(this.tick - (this.history[0]?.state.tick ?? 0)) +
              ' · largest colony ' +
              this.lastHealthyCount +
              ' cells'
            : 'Recording the globe · looking for living colonies');
      return;
    }
    if (!incident) return;
    if (mode === 'film') {
      title.textContent = 'Colony collapse';
      message.textContent = 'Tracing the incident back to its origin…';
      return;
    }
    const originalSurvivors = countSurvivors(
      incident.original,
      incident.colony.members,
      incident.healthy,
    );
    if (mode === 'confirm') {
      title.textContent = 'Intervene in this collapse?';
      message.textContent =
        originalSurvivors +
        ' of ' +
        incident.colony.members.length +
        ' cells survived. Save 60% to clear. Spend six seconds on force, or trade one for two seconds of earlier access.';
      return;
    }
    element('survivors').textContent =
      this.lastHealthyCount + ' / ' + incident.colony.members.length;
    element('remaining').textContent = seconds(this.leftTicks());
    element('event-time').textContent = seconds(this.tick - this.endTick());
    if (mode === 'result') {
      const clear =
        this.lastHealthyCount >=
        Math.ceil(incident.colony.members.length * 0.6);
      this.panel.querySelector<HTMLElement>(
        '[data-action="continue"]',
      )!.hidden = !clear;
      title.textContent = clear ? 'Colony rescued' : 'Intervention complete';
      message.textContent =
        this.lastHealthyCount * 10 +
        ' survival pts + ' +
        (this.runScore() - this.lastHealthyCount * 10) +
        ' reserve pts · original survivors ' +
        originalSurvivors +
        '. ' +
        (clear
          ? 'Continue to watch the survivors. Move the camera to dismiss their rings.'
          : 'Save 60% to clear. Your next attempt is free.');
      return;
    }
    title.textContent = committed
      ? 'Committed timeline'
      : 'Edit the intervention';
    if (committed)
      message.textContent = this.running
        ? 'Playing your recorded forces.'
        : 'Paused · hover or tap to inspect.';
    else
      message.textContent = this.running
        ? 'Hold to steer the ringed cells away from danger. Scrub to edit.'
        : 'Hover or tap to inspect. Hold to steer the ringed cells away from danger.';
    this.timeline.min = String(this.arrivalTick());
    this.timeline.max = String(this.endTick());
    this.timeline.value = String(this.seekTo ?? this.tick);
    this.timeline.disabled = committed;
    element('range-label').textContent =
      seconds(this.arrivalTick() - this.endTick()) + ' to 0.0s';
    this.panel.querySelector<HTMLElement>('[data-action="play"]')!.textContent =
      this.running ? 'Pause' : 'Play';
    this.panel
      .querySelector('[data-action="attract"]')!
      .setAttribute('aria-pressed', String(this.sign === 1));
    this.panel
      .querySelector('[data-action="repel"]')!
      .setAttribute('aria-pressed', String(this.sign === -1));
    this.arrival.title =
      'One second of force buys two seconds of earlier arrival';
    this.arrival.max = String(
      Math.floor(
        Math.min(
          incident.start - this.history[0].state.tick,
          budgetTicks * earlierSecondsPerForceSecond,
        ) / ticksPerSecond,
      ),
    );
    this.arrival.value = String(this.extraTicks / ticksPerSecond);
    element('arrival-label').textContent =
      this.extraTicks / ticksPerSecond +
      's · costs ' +
      seconds(this.extraTicks / earlierSecondsPerForceSecond);
    const strokes = new Map<
      number,
      { start: number; end: number; sign: number }
    >();
    for (const [tick, force] of this.plan) {
      const stroke = strokes.get(force.stroke);
      if (stroke) {
        stroke.start = Math.min(stroke.start, tick);
        stroke.end = Math.max(stroke.end, tick);
      } else
        strokes.set(force.stroke, { start: tick, end: tick, sign: force.sign });
    }
    const key =
      [...strokes].map(([id, s]) => [id, s.start, s.end].join('/')).join(',') +
      ';' +
      this.extraTicks;
    if (this.strokes.dataset.key !== key) {
      this.strokes.dataset.key = key;
      const selected = this.strokes.value;
      this.strokes.replaceChildren();
      const track = element('plan-track');
      track.replaceChildren();
      for (const [id, stroke] of strokes) {
        this.strokes.add(
          new Option(
            (stroke.sign === 1 ? 'Attract ' : 'Repel ') +
              seconds(stroke.start - this.endTick()),
            String(id),
          ),
        );
      }
      // Draw contiguous runs so overwriting the middle of a stroke leaves a visible cut.
      const ticks = [...this.plan.keys()].sort((left, right) => left - right);
      const duration = Math.max(1, this.endTick() - this.arrivalTick());
      for (let cursor = 0; cursor < ticks.length; ) {
        const startTick = ticks[cursor];
        const force = this.plan.get(startTick)!;
        let endTick = startTick;
        while (
          ++cursor < ticks.length &&
          ticks[cursor] === endTick + 1 &&
          this.plan.get(ticks[cursor])!.stroke === force.stroke
        )
          endTick++;
        const bar = document.createElement('i');
        bar.style.left =
          (100 * (startTick - 1 - this.arrivalTick())) / duration + '%';
        bar.style.width = (100 * (endTick - startTick + 1)) / duration + '%';
        bar.className = force.sign === 1 ? 'attract' : 'repel';
        track.append(bar);
      }
      if (!strokes.size) this.strokes.add(new Option('No recorded forces', ''));
      if (strokes.has(Number(selected))) this.strokes.value = selected;
    }
    this.panel.querySelector<HTMLButtonElement>(
      '[data-action="delete"]',
    )!.disabled = !strokes.size;
  }
}
