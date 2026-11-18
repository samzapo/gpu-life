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
  githubBtn.onmousedown = (event) => {
    event.stopPropagation();
  };
}

let activeViewState: GlobeViewState | null = null;
let activeD3Proj: d3Geo.GeoProjection | null = null;
let onViewMutated: (() => void) | null = null;
let currentMouseTool: 'globe' | 'attract' | 'repel' = 'globe';

export function bindGlobeInput(
  viewState: GlobeViewState,
  getProj: () => d3Geo.GeoProjection | null,
  notifyViewChange: () => void,
) {
  activeViewState = viewState;
  onViewMutated = notifyViewChange;
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
  return Boolean(target.closest('.tp-dfwv, #controls, #githubBtn'));
}

let draggingMode: 'none' | 'globe' | 'chem' = 'none';
let lastX = 0;
let lastY = 0;

window.addEventListener('mousedown', (event) => {
  if (isUiTarget(event.target)) return;
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

  if (event.button === 0 && (isRepel || isAttract)) {
    draggingMode = 'chem';
    mouseSphere.mode = isRepel ? -1 : 1;
  } else if (event.button === 0 || event.button === 2) {
    draggingMode = 'globe';
    mouseSphere.mode = 0;
  }
});

window.addEventListener('mouseup', () => {
  draggingMode = 'none';
  mouseSphere.mode = 0;
});

window.addEventListener('mousemove', (event) => {
  updateMouseSphere(event.clientX, event.clientY);
  if (draggingMode === 'globe' && activeViewState) {
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    const spec = PROJECTIONS[activeViewState.projectionId];
    if (spec.sphericalDrag && !event.ctrlKey) {
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
});

window.addEventListener('contextmenu', (event) => {
  event.preventDefault();
});

window.addEventListener(
  'wheel',
  (event) => {
    if (isUiTarget(event.target) || !activeViewState) return;
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.0011);
    const oldScale = activeViewState.scale;
    const newScale = Math.max(120, Math.min(30000, oldScale * factor));
    const ratio = newScale / oldScale;

    // Anchor zoom smoothly at cursor when zooming non-orthographic or already panned
    if (
      !PROJECTIONS[activeViewState.projectionId].sphericalDrag ||
      Math.abs(activeViewState.pan[0]) > 1 ||
      Math.abs(activeViewState.pan[1]) > 1
    ) {
      const cx = activeViewState.width * 0.5 + activeViewState.pan[0];
      const cy = activeViewState.height * 0.5 + activeViewState.pan[1];
      activeViewState.pan[0] = event.clientX - (event.clientX - cx) * ratio - activeViewState.width * 0.5;
      activeViewState.pan[1] = event.clientY - (event.clientY - cy) * ratio - activeViewState.height * 0.5;
    }
    activeViewState.scale = newScale;
    onViewMutated?.();
  },
  { passive: false },
);

window.addEventListener('keydown', (event) => {
  keys[event.code] = true;
});

window.addEventListener('keyup', (event) => {
  keys[event.code] = false;
});
