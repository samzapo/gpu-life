import type { GlobeViewState } from './projections';

export function copyView(view: GlobeViewState): GlobeViewState {
  return { ...view, rotate: [...view.rotate], pan: [...view.pan] };
}

const ease = (value: number) => value * value * (3 - 2 * value);
const shortestDegrees = (delta: number) => (((delta % 360) + 540) % 360) - 180;

// Endpoints retain their viewport so rotation of a phone during a flight
// preserves the same geographic framing for zoom and pan.
export class CameraFlight {
  private flight?: {
    start: GlobeViewState;
    end: GlobeViewState;
    durationMs: number;
    elapsedMs: number;
  };
  private lastFrameMs?: number;
  opacity = 0;
  get active() {
    return !!this.flight;
  }
  get target() {
    return this.flight?.end;
  }

  move(view: GlobeViewState, destination: GlobeViewState, durationMs: number) {
    if (!this.flight) this.pauseClock();
    this.flight = {
      start: copyView(view),
      end: copyView(destination),
      durationMs,
      elapsedMs: 0,
    };
  }

  cancel() {
    this.flight = undefined;
    this.pauseClock();
  }
  pauseClock() {
    this.lastFrameMs = undefined;
  }

  frame(nowMs: number, view: GlobeViewState) {
    const stepMs = Math.min(
      60,
      Math.max(0, nowMs - (this.lastFrameMs ?? nowMs)),
    );
    this.lastFrameMs = nowMs;
    const flight = this.flight;
    if (!flight && this.opacity === 0) return false;

    // Different map projections cannot share a geometry. Fade the globe layers
    // through the scene background and exchange projections at full coverage.
    const switching = flight && view.projectionId !== flight.end.projectionId;
    if (switching && this.opacity === 1)
      view.projectionId = flight.end.projectionId;
    this.opacity = switching
      ? Math.min(1, this.opacity + stepMs / 200)
      : Math.max(0, this.opacity - stepMs / 200);
    if (!flight) return true;
    flight.elapsedMs += stepMs;
    const fraction = Math.min(1, flight.elapsedMs / flight.durationMs);
    const blend = ease(fraction);
    const { start, end } = flight;
    view.rotate = [
      start.rotate[0] +
        shortestDegrees(end.rotate[0] - start.rotate[0]) * blend,
      start.rotate[1] + (end.rotate[1] - start.rotate[1]) * blend,
      start.rotate[2] +
        shortestDegrees(end.rotate[2] - start.rotate[2]) * blend,
    ];
    const width = view.width,
      height = view.height;
    const fromScale = start.scale / Math.min(start.width, start.height);
    const toScale = end.scale / Math.min(end.width, end.height);
    view.scale =
      Math.min(width, height) *
      fromScale *
      Math.pow(toScale / fromScale, blend);
    view.pan = [
      width *
        ((start.pan[0] / start.width) * (1 - blend) +
          (end.pan[0] / end.width) * blend),
      height *
        ((start.pan[1] / start.height) * (1 - blend) +
          (end.pan[1] / end.height) * blend),
    ];
    if (fraction === 1 && !switching && this.opacity === 0)
      this.flight = undefined;
    return true;
  }
}
