import type * as d3Geo from 'd3-geo';
import {
  lonLatToUnitVec,
  PROJECTIONS,
  type GlobeViewState,
} from './earth/projections';

export const mouseSphere = { x: 0, y: 0, z: 1, mode: 0 };
export const keys: Record<string, boolean> = {};

const urlParams = new URLSearchParams(window.location.search);
export const isSample = urlParams.has('sample');

const githubBtn = document.getElementById('githubBtn') as HTMLAnchorElement | null;
if (githubBtn) {
  if (!isSample) githubBtn.style.display = 'block';
  githubBtn.addEventListener('pointerdown', (event) => {
    event.stopPropagation();
  });
}

const controlsHud = document.getElementById('controls');
const hudHeader = document.getElementById('hudHeader');
const hudToggleBtn = document.getElementById('hudToggleBtn') as HTMLButtonElement | null;

function setHudMinimized(minimized: boolean) {
  if (!controlsHud || !hudToggleBtn) return;
  controlsHud.classList.toggle('minimized', minimized);
  hudToggleBtn.textContent = minimized ? '+' : '−';
  hudToggleBtn.setAttribute('aria-expanded', String(!minimized));
  hudToggleBtn.setAttribute(
    'aria-label',
    minimized ? 'Expand display' : 'Minimize display',
  );
}

if (hudHeader && controlsHud) {
  hudHeader.addEventListener('click', (event) => {
    event.stopPropagation();
    setHudMinimized(!controlsHud.classList.contains('minimized'));
  });
}

let activeViewState: GlobeViewState | null = null;
let activeD3Proj: d3Geo.GeoProjection | null = null;
let onViewMutated: (() => void) | null = null;
let onWheelZoom: ((factor: number, x: number, y: number) => void) | undefined;
let cameraLocked = false;
let currentMouseTool: 'globe' | 'attract' | 'repel' = 'globe';

export function bindGlobeInput(
  viewState: GlobeViewState,
  getProj: () => d3Geo.GeoProjection | null,
  notifyViewChange: () => void,
  wheelZoom: (factor: number, x: number, y: number) => void,
) {
  activeViewState = viewState;
  onViewMutated = notifyViewChange;
  onWheelZoom = wheelZoom;
  activeD3Proj = getProj();
}

export function refreshBoundProjection(proj: d3Geo.GeoProjection) {
  activeD3Proj = proj;
}

export function setMouseToolMode(tool: 'globe' | 'attract' | 'repel') {
  currentMouseTool = tool;
}

function updateMouseSphere(clientX: number, clientY: number) {
  if (!activeD3Proj?.invert) return;
  const ll = activeD3Proj.invert([clientX, clientY]);
  if (ll && Number.isFinite(ll[0]) && Number.isFinite(ll[1])) {
    const v = lonLatToUnitVec(ll[0], ll[1]);
    mouseSphere.x = v[0];
    mouseSphere.y = v[1];
    mouseSphere.z = v[2];
  }
}

function isUiTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return Boolean(target.closest('.tp-dfwv, #controls, #githubBtn, #configModal, #globe-switches, #temporal-panel'));
}

function applyDragDelta(dx: number, dy: number, forcePan: boolean) {
  if (!activeViewState || cameraLocked || (!dx && !dy)) return;
  const spec = PROJECTIONS[activeViewState.projectionId];
  if (spec.sphericalDrag && !forcePan) {
    const sensitivity = 60 / Math.max(activeViewState.scale, 120);
    activeViewState.rotate[0] += dx * sensitivity;
    activeViewState.rotate[1] = Math.max(
      -88,
      Math.min(88, activeViewState.rotate[1] - dy * sensitivity),
    );
  } else {
    activeViewState.pan[0] += dx;
    activeViewState.pan[1] += dy;
  }
  onViewMutated?.();
}

export function zoomViewAt(view: GlobeViewState, factor: number, anchorX: number, anchorY: number) {
  if (!Number.isFinite(factor) || factor <= 0) return false;
  const oldScale = view.scale;
  const newScale = Math.max(120, Math.min(30000, oldScale * factor));
  if (oldScale === newScale) return false;
  const ratio = newScale / oldScale;

  if (
    !PROJECTIONS[view.projectionId].sphericalDrag ||
    Math.abs(view.pan[0]) > 1 ||
    Math.abs(view.pan[1]) > 1
  ) {
    const cx = view.width * 0.5 + view.pan[0];
    const cy = view.height * 0.5 + view.pan[1];
    view.pan[0] = anchorX - (anchorX - cx) * ratio - view.width * 0.5;
    view.pan[1] = anchorY - (anchorY - cy) * ratio - view.height * 0.5;
  }
  view.scale = newScale;
  return true;
}

interface PointerPt {
  x: number;
  y: number;
}

const mainCanvas = document.getElementById('canvas') as HTMLCanvasElement | null;
const activePointers = new Map<number, PointerPt>();
let draggingMode: 'none' | 'globe' | 'chem' | 'pinch' = 'none';
let lastX = 0;
let lastY = 0;
let prevPinchDist = 0;
let prevPinchMidX = 0;
let prevPinchMidY = 0;

export function lockGlobeInput(locked: boolean) {
  cameraLocked = locked;
  activePointers.clear();
  draggingMode = 'none';
  mouseSphere.mode = 0;
}

function initPinchFromPointers() {
  const pts = [...activePointers.values()];
  if (pts.length < 2) return;
  const [a, b] = pts;
  prevPinchDist = Math.hypot(b.x - a.x, b.y - a.y);
  prevPinchMidX = (a.x + b.x) * 0.5;
  prevPinchMidY = (a.y + b.y) * 0.5;
  draggingMode = 'pinch';
  mouseSphere.mode = 0;
}

function releaseActivePointer(pointerId: number) {
  if (!activePointers.has(pointerId)) return;
  activePointers.delete(pointerId);

  if (activePointers.size >= 2) {
    initPinchFromPointers();
  } else if (activePointers.size === 1) {
    const [rem] = activePointers.values();
    lastX = rem.x;
    lastY = rem.y;
    draggingMode = currentMouseTool === 'globe' ? 'globe' : 'none';
    mouseSphere.mode = 0;
  } else {
    draggingMode = 'none';
    mouseSphere.mode = 0;
  }
}

window.addEventListener('pointerdown', (event) => {
  if (cameraLocked || isUiTarget(event.target)) return;
  if (event.pointerType === 'mouse' && event.button !== 0 && event.button !== 2) {
    return;
  }

  activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (mainCanvas && event.target === mainCanvas) {
    try {
      mainCanvas.setPointerCapture(event.pointerId);
    } catch {
      // Ignore synthetic events without active pointer capture registration
    }
  }

  if (activePointers.size >= 2) {
    initPinchFromPointers();
    return;
  }

  updateMouseSphere(event.clientX, event.clientY);
  lastX = event.clientX;
  lastY = event.clientY;

  const isRepel =
    event.altKey ||
    keys.AltLeft ||
    keys.AltRight ||
    (currentMouseTool === 'repel' && !event.shiftKey);
  const isAttract =
    event.shiftKey ||
    keys.ShiftLeft ||
    keys.ShiftRight ||
    (currentMouseTool === 'attract' && !isRepel);

  if ((event.pointerType !== 'mouse' || event.button === 0) && (isRepel || isAttract)) {
    draggingMode = 'chem';
    mouseSphere.mode = isRepel ? -1 : 1;
  } else {
    draggingMode = 'globe';
    mouseSphere.mode = 0;
  }
});

window.addEventListener('pointermove', (event) => {
  if (!activePointers.has(event.pointerId)) {
    if (event.pointerType === 'mouse') {
      updateMouseSphere(event.clientX, event.clientY);
    }
    return;
  }

  activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

  if (activePointers.size >= 2) {
    const [a, b] = [...activePointers.values()];
    const midX = (a.x + b.x) * 0.5;
    const midY = (a.y + b.y) * 0.5;
    const dist = Math.hypot(b.x - a.x, b.y - a.y);

    const dx = midX - prevPinchMidX;
    const dy = midY - prevPinchMidY;
    if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01) {
      applyDragDelta(dx, dy, false);
    }
    if (prevPinchDist > 12 && dist > 12) {
      const factor = Math.max(0.75, Math.min(1.35, dist / prevPinchDist));
      if (!cameraLocked && activeViewState && zoomViewAt(activeViewState, factor, midX, midY)) onViewMutated?.();
    }

    prevPinchDist = dist;
    prevPinchMidX = midX;
    prevPinchMidY = midY;
    return;
  }

  updateMouseSphere(event.clientX, event.clientY);
  if (draggingMode === 'globe') {
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    applyDragDelta(dx, dy, event.ctrlKey);
  }
});

window.addEventListener('pointerup', (event) => {
  releaseActivePointer(event.pointerId);
});

window.addEventListener('pointercancel', (event) => {
  releaseActivePointer(event.pointerId);
});

window.addEventListener('blur', () => {
  activePointers.clear();
  draggingMode = 'none';
  mouseSphere.mode = 0;
});

// Prevent iOS Safari / Android native pinch-zoom and pull-to-refresh on canvas touches
for (const evName of ['touchstart', 'touchmove', 'gesturestart', 'gesturechange'] as const) {
  window.addEventListener(
    evName,
    (event) => {
      if (!isUiTarget(event.target)) {
        event.preventDefault();
      }
    },
    { passive: false },
  );
}

window.addEventListener('contextmenu', (event) => {
  if (!isUiTarget(event.target)) {
    event.preventDefault();
  }
});

window.addEventListener(
  'wheel',
  (event) => {
    if (isUiTarget(event.target)) return;
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.0011);
    if (!cameraLocked) onWheelZoom?.(factor, event.clientX, event.clientY);
  },
  { passive: false },
);

window.addEventListener('keydown', (event) => {
  keys[event.code] = true;
});

window.addEventListener('keyup', (event) => {
  keys[event.code] = false;
});
