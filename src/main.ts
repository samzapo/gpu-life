import {
  bindGlobeInput,
  mouseSphere,
  refreshBoundProjection,
} from './input';
import { createSeededRng, generateRandomSeed } from './matrix';
import {
  buildPackedChemistryBuffer,
  computeAbundanceDistribution,
  type Element2D,
} from './periodicTable2D';
import structsShader from './shaders/structs.wgsl?raw';
import projectionShader from './shaders/projection.wgsl?raw';
import bgRenderShader from './shaders/bgRender.wgsl?raw';
import renderShader from './render.wgsl?raw';
import {
  buildConfiguredD3Projection,
  buildD3RotationMatrixColMajor,
  buildWatermanLutData,
  fitScaleForProjection,
  PROJECTIONS,
  type GlobeViewState,
  type ProjectionId,
} from './earth/projections';
import {
  loadEarthFeatures,
  renderMapOverlaySvg,
  type EarthTopoFeatures,
} from './earth/mapOverlay';
import {
  buildDivergenceFreeWind,
  extendedSinebowColor,
  fetchLiveGfsWind,
  loadFallbackWind,
  WIND_LEVELS,
  type WindGrid,
  type WindLevelId,
} from './earth/windData';
import {
  buildDivergenceFreeOcean,
  loadOceanCurrents,
  OCEAN_CURRENTS_SPEC,
  sampleOceanLonLat,
  type OceanGrid,
} from './earth/oceanData';
import { oceanGridToSampler, WindStreamRenderer } from './earth/windLines';
import {
  hslToRgb,
  linkRenderTimestamp,
  readTimestamp,
  requestTimestamps,
  resolveTimestamp,
  setupTimestamp,
} from './utils';
import {
  bindOptions,
  bioLinkBinding,
  bioRateBinding,
  coastlinesBinding,
  colourAmt as colourAmtOptn,
  earthParams,
  engineSelect,
  globalPerformanceParams,
  initialOrientation,
  newOceanSimBtn,
  newSimBtn,
  oceanColorsBinding,
  oceanColourAmt as oceanColourAmtOptn,
  oceanLinesBinding,
  oceanOptionParams,
  oceanParticleAmt as oceanParticleAmtOptn,
  oceanStatusBinding,
  optionParams,
  params,
  particleAmt as particleAmtOptn,
  performanceParams,
  projectionSelect,
  randomizeBtn,
  randomizeOceanRulesBtn,
  resetViewBtn,
  seedInput,
  setBioLink,
  setEngineDisplay,
  setOceanSim,
  setSim,
  updatePeriodicLegend,
  viewLayerSelect,
  type ViewLayerMode,
  windColorsBinding,
  windLevelSelect,
  windLinesBinding,
  windStatusBinding,
} from './options';

import * as linkedList from './linkedList/main';
import * as countingSort from './countingSort/main';
import * as prefixSum from './countingSortOptimized/main';

const bgCanvas = document.getElementById('bgCanvas') as HTMLCanvasElement;
const mapSvg = document.getElementById('mapSvg') as unknown as SVGSVGElement;
const oceanCanvas = document.getElementById('oceanCanvas') as HTMLCanvasElement;
const windCanvas = document.getElementById('windCanvas') as HTMLCanvasElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const scaleBar = document.getElementById('scaleBar') as HTMLElement;
const windHudGroup = document.getElementById('windHudGroup') as HTMLElement;
const oceanHudGroup = document.getElementById('oceanHudGroup') as HTMLElement;
const windLevelBadge = document.getElementById('windLevelBadge') as HTMLElement;
const windTimeBadge = document.getElementById('windTimeBadge') as HTMLElement;
const oceanLevelBadge = document.getElementById('oceanLevelBadge') as HTMLElement;
const oceanTimeBadge = document.getElementById('oceanTimeBadge') as HTMLElement;
const scaleMaxLabel = document.getElementById('scaleMaxLabel') as HTMLElement;

const oceanCtx = oceanCanvas.getContext('2d', { willReadFrequently: true });
const windCtx = windCanvas.getContext('2d', { willReadFrequently: true });
const oceanStreamRenderer = new WindStreamRenderer();
const windStreamRenderer = new WindStreamRenderer();

const uniformsFloats = 32; // 128 bytes
const uniformData = new Float32Array(uniformsFloats);
const oceanUniformData = new Float32Array(uniformsFloats);
const simSize = 20;
const simData = new Float32Array(simSize);
const oceanSimSize = 20;
const oceanSimData = new Float32Array(oceanSimSize);
const bioUniformFloats = 12;
const bioUniformData = new Float32Array(bioUniformFloats);
const frameBuf = new Float32Array(1);

let colourAmt = colourAmtOptn;
let elements: Element2D[] = [];
let chemistryData: Float32Array<ArrayBuffer> = new Float32Array(0);

function refreshChemistryDefinition() {
  const built = buildPackedChemistryBuffer(colourAmt);
  elements = built.elements;
  chemistryData = built.packedData;
}
refreshChemistryDefinition();

let oceanColourAmt = oceanColourAmtOptn;
let oceanSeed = generateRandomSeed();
let oceanMatrixData: Float32Array<ArrayBuffer> = new Float32Array(0);

function buildRandomOceanMatrix(nColours: number, seed: string): Float32Array<ArrayBuffer> {
  const rng = createSeededRng(`${seed}:oceanMatrix`);
  const mat = new Float32Array(nColours * nColours);
  for (let i = 0; i < mat.length; i++) {
    mat[i] = rng() * 2 - 1;
  }
  return mat;
}
oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);

const particleStride = 48; // 12 floats: pos(3), colour(1), vel(3), omega(1), orient(3), pad(1)
const valenceStride = 16;
const GRID_TEX_W = 360;
const GRID_TEX_H = 181;
const WATERMAN_INV_W = 640;
const WATERMAN_INV_H = 360;

let particleAmt = particleAmtOptn;
let oceanParticleAmt = oceanParticleAmtOptn;

let device: GPUDevice | undefined;
let bgContext: GPUCanvasContext | undefined;
let context: GPUCanvasContext | undefined;

let uniformBuffer: GPUBuffer | undefined;
let oceanUniformBuffer: GPUBuffer | undefined;
let simBuffer: GPUBuffer | undefined;
let oceanSimBuffer: GPUBuffer | undefined;
let bioUniformBuffer: GPUBuffer | undefined;
let bgRenderPipeline: GPURenderPipeline | undefined;
let renderPipeline: GPURenderPipeline | undefined;

let matrixBuffer: GPUBuffer | undefined;
let colourBuffer: GPUBuffer | undefined;
let valenceBuffer: GPUBuffer | undefined;
let airBioReactBuffer: GPUBuffer | undefined;
let particleBuffers: [GPUBuffer, GPUBuffer] | undefined;

let oceanMatrixBuffer: GPUBuffer | undefined;
let oceanColourBuffer: GPUBuffer | undefined;
let oceanValenceBuffer: GPUBuffer | undefined;
let oceanSteerBuffer: GPUBuffer | undefined;
let oceanParticleBuffers: [GPUBuffer, GPUBuffer] | undefined;
let bioWasActive = false;

let windTexture: GPUTexture | undefined;
let oceanTexture: GPUTexture | undefined;
let watermanFwdTexture: GPUTexture | undefined;
let watermanInvTexture: GPUTexture | undefined;

let bgBindGroup: GPUBindGroup | undefined;
let renderBindGroup: GPUBindGroup | undefined;
let oceanRenderBindGroup: GPUBindGroup | undefined;

let alternate = 0;
let oceanAlternate = 0;
let fpsc = 0;
let frameCount = 0;
let suppressSeedChange = false;

const engines: Record<
  string,
  typeof linkedList | typeof countingSort | typeof prefixSum
> = {
  linkedList,
  countingSort,
};

let engine: string =
  params.engine in performanceParams ? params.engine : 'prefixSum';
setEngineDisplay(engine);

const initSpec = PROJECTIONS[earthParams.projection];
const initScale =
  initialOrientation && initialOrientation[2] > 50
    ? initialOrientation[2]
    : earthParams.projection === 'orthographic'
      ? Math.round(586 * Math.min(window.innerWidth / 1280, window.innerHeight / 800) || 586)
      : fitScaleForProjection(
          earthParams.projection,
          initSpec.defaultRotate,
          window.innerWidth,
          window.innerHeight,
        );

const initRotate: [number, number, number] = initialOrientation
  ? [-initialOrientation[0], -initialOrientation[1], 0]
  : [...initSpec.defaultRotate];

const viewState: GlobeViewState = {
  projectionId: earthParams.projection,
  rotate: initRotate,
  scale: Math.max(140, initScale),
  pan: [0, 0],
  width: window.innerWidth,
  height: window.innerHeight,
};

let d3Proj = buildConfiguredD3Projection(viewState);
let earthFeatures: EarthTopoFeatures | null = null;
let currentWindGrid: WindGrid = createEmptyWindGrid();
let currentOceanGrid: OceanGrid | null = null;
let watermanDirty = true;
let mapDirty = true;
let viewSignature = '';
let hashSyncTimer: number | undefined;

function createEmptyWindGrid(): WindGrid {
  return {
    nx: GRID_TEX_W,
    ny: GRID_TEX_H,
    u: new Float32Array(GRID_TEX_W * GRID_TEX_H),
    v: new Float32Array(GRID_TEX_W * GRID_TEX_H),
    level: earthParams.windLevel,
    refTime: 'initializing',
    validTime: '',
    source: 'GFS',
    isLive: false,
  };
}

function drawScaleBar(maxSpeed: number) {
  const stops: string[] = [];
  const steps = 16;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const [r, g, b] = extendedSinebowColor(t);
    stops.push(`rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}) ${(t * 100).toFixed(0)}%`);
  }
  scaleBar.style.background = `linear-gradient(to right, ${stops.join(', ')})`;
  scaleMaxLabel.textContent = `${maxSpeed} m/s`;
}

function refreshHudVisibility() {
  const showAir = earthParams.viewLayer !== 'ocean';
  const showOcean = earthParams.viewLayer !== 'air';
  windHudGroup.style.display = showAir ? '' : 'none';
  oceanHudGroup.style.display = showOcean ? '' : 'none';
}

function refreshHudBadges(grid: WindGrid) {
  const spec = WIND_LEVELS[grid.level];
  windLevelBadge.textContent = `Wind @ ${spec.label}`;
  windTimeBadge.textContent = grid.validTime
    ? `${grid.validTime} · ${grid.source}`
    : grid.source;
  earthParams.windStatus = grid.validTime
    ? `${spec.label} · ${grid.validTime}`
    : grid.source;
  windStatusBinding.refresh();
  drawScaleBar(spec.scaleMax);
  refreshHudVisibility();
}

function refreshOceanHud(grid: OceanGrid) {
  const dateLabel = grid.refTime.replace('T00:00:00Z', '').replace(':00Z', 'Z');
  oceanLevelBadge.textContent = `Ocean @ ${OCEAN_CURRENTS_SPEC.label}`;
  oceanTimeBadge.textContent = `${dateLabel} · ${grid.source}`;
  earthParams.oceanStatus = `${dateLabel} · CoastWatch`;
  oceanStatusBinding.refresh();
  refreshHudVisibility();
}

function syncUrlHashDebounced() {
  if (hashSyncTimer !== undefined) {
    window.clearTimeout(hashSyncTimer);
  }
  hashSyncTimer = window.setTimeout(() => {
    const spec = WIND_LEVELS[earthParams.windLevel];
    const r = viewState.rotate;
    const lon = (-r[0]).toFixed(2);
    const lat = (-r[1]).toFixed(2);
    const sc = Math.round(viewState.scale);
    const nextHash =
      earthParams.viewLayer === 'ocean'
        ? `#current/ocean/surface/currents/${viewState.projectionId}=${lon},${lat},${sc}`
        : earthParams.viewLayer === 'air'
          ? `#current/air/${spec.surfaceType === 'surface' ? 'surface/level' : `isobaric/${spec.id}`}/${viewState.projectionId}=${lon},${lat},${sc}`
          : `#current/both/${spec.surfaceType === 'surface' ? 'surface/level' : `isobaric/${spec.id}`}/${viewState.projectionId}=${lon},${lat},${sc}`;
    if (window.location.hash !== nextHash) {
      window.history.replaceState(
        null,
        '',
        `${window.location.pathname}${window.location.search}${nextHash}`,
      );
    }
  }, 150);
}

function updateProjectionState() {
  d3Proj = buildConfiguredD3Projection(viewState);
  refreshBoundProjection(d3Proj);
  mapDirty = true;
  if (viewState.projectionId === 'waterman') {
    watermanDirty = true;
  }
  viewSignature = `${viewState.projectionId}:${viewState.rotate.map((v) => v.toFixed(2)).join(',')}:${viewState.scale.toFixed(1)}:${viewState.pan.map((v) => v.toFixed(1)).join(',')}:${viewState.width}x${viewState.height}`;
  syncUrlHashDebounced();
}

bindGlobeInput(viewState, () => d3Proj, updateProjectionState);

function uploadWindToGpu(grid: WindGrid) {
  if (!device || !windTexture) return;
  const texData = buildDivergenceFreeWind(grid, GRID_TEX_W, GRID_TEX_H);
  device.queue.writeTexture(
    { texture: windTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
}

function uploadOceanToGpu(grid: OceanGrid) {
  if (!device || !oceanTexture) return;
  const texData = buildDivergenceFreeOcean(grid, GRID_TEX_W, GRID_TEX_H);
  device.queue.writeTexture(
    { texture: oceanTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
}

let activeWindLoadId = 0;

async function loadWindLevel(levelId: WindLevelId) {
  const reqId = ++activeWindLoadId;
  const spec = WIND_LEVELS[levelId];
  earthParams.windStatus = `Fetching ${spec.label}…`;
  windStatusBinding.refresh();

  if (levelId === 'surface' && !currentWindGrid.isLive) {
    try {
      const fb = await loadFallbackWind();
      if (reqId === activeWindLoadId) {
        currentWindGrid = fb;
        uploadWindToGpu(fb);
        refreshHudBadges(fb);
      }
    } catch {
      // Live fetch still runs below
    }
  }

  try {
    const live = await fetchLiveGfsWind(levelId);
    if (reqId === activeWindLoadId) {
      currentWindGrid = live;
      uploadWindToGpu(live);
      refreshHudBadges(live);
    }
  } catch {
    if (reqId === activeWindLoadId && !currentWindGrid.isLive && levelId === 'surface') {
      try {
        const fb = await loadFallbackWind();
        currentWindGrid = fb;
        uploadWindToGpu(fb);
        refreshHudBadges(fb);
      } catch {
        // Ignore if both live & fallback fail
      }
    } else if (reqId === activeWindLoadId) {
      earthParams.windStatus = `${spec.label} · cached fallback`;
      windStatusBinding.refresh();
    }
  }
}

function writeChemistryBuffers() {
  if (!device || !matrixBuffer || !colourBuffer) return;
  device.queue.writeBuffer(matrixBuffer, 0, chemistryData.buffer);

  const colourData = new Float32Array(colourAmt * 3);
  for (let c = 0; c < colourAmt; c++) {
    colourData[c * 3] = elements[c].color[0];
    colourData[c * 3 + 1] = elements[c].color[1];
    colourData[c * 3 + 2] = elements[c].color[2];
  }
  device.queue.writeBuffer(colourBuffer, 0, colourData.buffer);
}

function writeOceanRuleBuffers() {
  if (!device || !oceanMatrixBuffer || !oceanColourBuffer) return;
  device.queue.writeBuffer(oceanMatrixBuffer, 0, oceanMatrixData.buffer);

  const colourData = new Float32Array(oceanColourAmt * 3);
  for (let i = 0; i < oceanColourAmt; i++) {
    const [r, g, b] = hslToRgb((i / oceanColourAmt) * 360, 1, 0.5);
    colourData[i * 3] = r;
    colourData[i * 3 + 1] = g;
    colourData[i * 3 + 2] = b;
  }
  device.queue.writeBuffer(oceanColourBuffer, 0, colourData.buffer);
}

declare global {
  interface Window {
    complexLife?: {
      getElements: () => Element2D[];
      randomizeMatrix: (seed?: string) => void;
      getSeed: () => string;
      getEngine: () => string;
      setEngine: (name: string) => boolean;
      getParticleCount: () => number;
      getColourCount: () => number;
      getOceanParticleCount: () => number;
      getOceanColourCount: () => number;
      getViewLayer: () => ViewLayerMode;
      setViewLayer: (layer: ViewLayerMode) => void;
      getBioLink: () => boolean;
      setBioLink: (on: boolean) => void;
      getBioRate: () => number;
      setBioRate: (rate: number) => void;
      setOceanColors: (on: boolean) => void;
      setOceanLines: (on: boolean) => void;
      randomizeOceanMatrix: (seed?: string) => void;
      getOceanInfo: () => {
        nx: number;
        ny: number;
        refTime: string;
        source: string;
      } | null;
      getProjection: () => ProjectionId;
      setProjection: (id: ProjectionId) => void;
      getWindLevel: () => WindLevelId;
      setWindLevel: (lvl: WindLevelId) => Promise<void>;
      getWindInfo: () => {
        level: WindLevelId;
        nx: number;
        ny: number;
        validTime: string;
        source: string;
        isLive: boolean;
      };
      getViewState: () => GlobeViewState;
      setViewState: (patch: Partial<GlobeViewState>) => void;
      readParticles: (count?: number) => Promise<number[][]>;
      readOceanParticles: (count?: number) => Promise<number[][]>;
      readValence: (count?: number) => Promise<number[][]>;
      readOceanState: (count?: number) => Promise<number[][]>;
      readAirBioReact: (count?: number) => Promise<number[][]>;
    };
  }
}

async function readParticleBuffer(
  buffers: [GPUBuffer, GPUBuffer] | undefined,
  altIdx: number,
  total: number,
  count: number,
): Promise<number[][]> {
  if (!device || !buffers) return [];
  const n = Math.min(count, total);
  const byteLength = n * particleStride;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffers[altIdx], 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const f32 = new Float32Array(staging.getMappedRange());
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const b = i * 12;
    out.push([
      f32[b + 0],
      f32[b + 1],
      f32[b + 2],
      f32[b + 4],
      f32[b + 5],
      f32[b + 6],
      f32[b + 3],
      f32[b + 8],
      f32[b + 9],
      f32[b + 10],
      f32[b + 7],
    ]);
  }
  staging.unmap();
  staging.destroy();
  return out;
}

async function readVec4Buffer(
  buf: GPUBuffer | undefined,
  totalAmt: number,
  count: number = 16,
): Promise<number[][]> {
  if (!device || !buf) return [];
  const n = Math.min(count, totalAmt);
  const byteLength = n * valenceStride;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const f32 = new Float32Array(staging.getMappedRange());
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    out.push([
      f32[i * 4],
      f32[i * 4 + 1],
      f32[i * 4 + 2],
      f32[i * 4 + 3],
    ]);
  }
  staging.unmap();
  staging.destroy();
  return out;
}

function writeNeutralOceanState() {
  if (!device || !oceanValenceBuffer) return;
  const neutral = new Float32Array(oceanParticleAmt * 4);
  for (let i = 0; i < oceanParticleAmt; i++) {
    neutral[i * 4] = 0.65;
  }
  device.queue.writeBuffer(oceanValenceBuffer, 0, neutral);
}

function resetNeutralBioBuffers(resetMetabolism: boolean = true) {
  if (!device) return;
  if (airBioReactBuffer) {
    device.queue.writeBuffer(airBioReactBuffer, 0, new Float32Array(particleAmt * 4));
  }
  if (oceanSteerBuffer) {
    device.queue.writeBuffer(oceanSteerBuffer, 0, new Float32Array(oceanParticleAmt * 4));
  }
  if (resetMetabolism) {
    writeNeutralOceanState();
  }
}

function syncBioUniformState() {
  if (!device || !bioUniformBuffer) return;
  const active =
    earthParams.viewLayer === 'both' && earthParams.bioLink && earthParams.bioRate > 0;
  setBioLink(device, bioUniformData, bioUniformBuffer, active);
  if (!active && (bioWasActive || !earthParams.bioLink)) {
    resetNeutralBioBuffers(!earthParams.bioLink);
  }
  bioWasActive = active;
}

function refreshEngineCrossBio() {
  if (!device || !bioUniformBuffer || !airBioReactBuffer || !oceanSteerBuffer) return;
  for (const engineName in engines) {
    engines[engineName].bindCrossBio(
      device,
      bioUniformBuffer,
      airBioReactBuffer,
      oceanSteerBuffer,
    );
  }
}

function unitTangentFromSphere(
  nx: number,
  ny: number,
  nz: number,
  alpha: number,
): [number, number, number] {
  const refX = Math.abs(nz) > 0.9 ? 1 : 0;
  const refZ = Math.abs(nz) > 0.9 ? 0 : 1;
  let e1x = 0 * nz - refZ * ny;
  let e1y = refZ * nx - refX * nz;
  let e1z = refX * ny - 0 * nx;
  const e1Len = Math.hypot(e1x, e1y, e1z) || 1;
  e1x /= e1Len;
  e1y /= e1Len;
  e1z /= e1Len;
  const e2x = ny * e1z - nz * e1y;
  const e2y = nz * e1x - nx * e1z;
  const e2z = nx * e1y - ny * e1x;
  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  return [ca * e1x + sa * e2x, ca * e1y + sa * e2y, ca * e1z + sa * e2z];
}

function applySeed(seed: string, updateInput: boolean = true) {
  params.seed = seed;
  if (updateInput) {
    suppressSeedChange = true;
    seedInput.refresh();
    suppressSeedChange = false;
  }
  startParticles();
}

function setViewLayerChoice(layer: ViewLayerMode) {
  earthParams.viewLayer = layer;
  viewLayerSelect.refresh();
  refreshHudVisibility();
  if (layer === 'air' && oceanCtx) {
    oceanStreamRenderer.clear(oceanCtx, viewState.width, viewState.height);
  } else if (layer === 'ocean' && windCtx) {
    windStreamRenderer.clear(windCtx, viewState.width, viewState.height);
  }
  syncBioUniformState();
  syncUrlHashDebounced();
}

function setProjectionChoice(projId: ProjectionId) {
  earthParams.projection = projId;
  projectionSelect.refresh();
  const spec = PROJECTIONS[projId];
  viewState.projectionId = projId;
  viewState.rotate = [...spec.defaultRotate];
  viewState.pan = [0, 0];
  viewState.scale = fitScaleForProjection(
    projId,
    viewState.rotate,
    viewState.width,
    viewState.height,
  );
  updateProjectionState();
}

window.complexLife = {
  getElements: () => elements,
  randomizeMatrix: (seed?: string) => {
    applySeed(seed ?? generateRandomSeed(), true);
  },
  getSeed: () => params.seed,
  getEngine: () => engine,
  setEngine: (name: string) => {
    if (!(name in engines)) return false;
    engine = name;
    params.engine = name;
    setEngineDisplay(name);
    engineSelect.refresh();
    return true;
  },
  getParticleCount: () => particleAmt,
  getColourCount: () => colourAmt,
  getOceanParticleCount: () => oceanParticleAmt,
  getOceanColourCount: () => oceanColourAmt,
  getViewLayer: () => earthParams.viewLayer,
  setViewLayer: setViewLayerChoice,
  getBioLink: () => earthParams.bioLink,
  setBioLink: (on: boolean) => {
    earthParams.bioLink = on;
    bioLinkBinding.refresh();
    syncBioUniformState();
  },
  getBioRate: () => earthParams.bioRate,
  setBioRate: (rate: number) => {
    earthParams.bioRate = rate;
    bioRateBinding.refresh();
    syncBioUniformState();
  },
  setOceanColors: (on: boolean) => {
    earthParams.oceanColors = on;
    oceanColorsBinding.refresh();
  },
  setOceanLines: (on: boolean) => {
    earthParams.oceanLines = on;
    oceanLinesBinding.refresh();
    if (!on && oceanCtx) {
      oceanStreamRenderer.clear(oceanCtx, viewState.width, viewState.height);
    }
  },
  randomizeOceanMatrix: (seed?: string) => {
    oceanSeed = seed ?? generateRandomSeed();
    oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);
    writeOceanRuleBuffers();
  },
  getOceanInfo: () =>
    currentOceanGrid
      ? {
          nx: currentOceanGrid.nx,
          ny: currentOceanGrid.ny,
          refTime: currentOceanGrid.refTime,
          source: currentOceanGrid.source,
        }
      : null,
  getProjection: () => viewState.projectionId,
  setProjection: setProjectionChoice,
  getWindLevel: () => earthParams.windLevel,
  setWindLevel: async (lvl: WindLevelId) => {
    earthParams.windLevel = lvl;
    windLevelSelect.refresh();
    await loadWindLevel(lvl);
  },
  getWindInfo: () => ({
    level: currentWindGrid.level,
    nx: currentWindGrid.nx,
    ny: currentWindGrid.ny,
    validTime: currentWindGrid.validTime,
    source: currentWindGrid.source,
    isLive: currentWindGrid.isLive,
  }),
  getViewState: () => ({
    ...viewState,
    rotate: [...viewState.rotate],
    pan: [...viewState.pan],
  }),
  setViewState: (patch: Partial<GlobeViewState>) => {
    if (patch.projectionId) viewState.projectionId = patch.projectionId;
    if (patch.rotate) viewState.rotate = [...patch.rotate];
    if (patch.pan) viewState.pan = [...patch.pan];
    if (typeof patch.scale === 'number') viewState.scale = patch.scale;
    earthParams.projection = viewState.projectionId;
    projectionSelect.refresh();
    updateProjectionState();
  },
  readParticles: (count: number = 16) =>
    readParticleBuffer(particleBuffers, alternate, particleAmt, count),
  readOceanParticles: (count: number = 16) =>
    readParticleBuffer(oceanParticleBuffers, oceanAlternate, oceanParticleAmt, count),
  readValence: (count: number = 16) => readVec4Buffer(valenceBuffer, particleAmt, count),
  readOceanState: (count: number = 16) =>
    readVec4Buffer(oceanValenceBuffer, oceanParticleAmt, count),
  readAirBioReact: (count: number = 16) =>
    readVec4Buffer(airBioReactBuffer, particleAmt, count),
};

function syncCanvasDimensions() {
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  for (const c of [bgCanvas, oceanCanvas, windCanvas, canvas]) {
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }
  mapSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  viewState.width = w;
  viewState.height = h;
}

const oceanBootPromise = loadOceanCurrents().catch(() => null);

(async () => {
  loadEarthFeatures()
    .then((feats) => {
      earthFeatures = feats;
      mapDirty = true;
    })
    .catch(() => {
      // Non-fatal if coastline JSON fails to load
    });

  let adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'compatibility',
  });
  for (let retry = 0; retry < 5 && !adapter; retry++) {
    await new Promise((r) => setTimeout(r, 200));
    adapter = await navigator.gpu.requestAdapter({
      featureLevel: 'compatibility',
    });
  }
  if (!adapter) return;

  device = await requestTimestamps(adapter);

  if (device?.features.has('subgroups')) {
    engines['prefixSum'] = prefixSum;
  }

  if (params.engine in engines) {
    engine = params.engine;
  } else {
    engine = 'linkedList';
    params.engine = engine;
  }
  setEngineDisplay(engine);
  engineSelect.refresh();

  bgContext = bgCanvas.getContext('webgpu') ?? undefined;
  context = canvas.getContext('webgpu') ?? undefined;
  if (!bgContext || !context) return;

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  bgContext.configure({ device, format: presentationFormat, alphaMode: 'opaque' });
  context.configure({ device, format: presentationFormat, alphaMode: 'premultiplied' });

  uniformBuffer = device.createBuffer({
    size: uniformsFloats * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'uniformBuffer',
  });
  oceanUniformBuffer = device.createBuffer({
    size: uniformsFloats * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'oceanUniformBuffer',
  });

  simBuffer = device.createBuffer({
    size: simSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'simBuffer',
  });
  oceanSimBuffer = device.createBuffer({
    size: oceanSimSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'oceanSimBuffer',
  });
  bioUniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'bioUniformBuffer',
  });

  windTexture = device.createTexture({
    size: { width: GRID_TEX_W, height: GRID_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'windTexture',
  });
  oceanTexture = device.createTexture({
    size: { width: GRID_TEX_W, height: GRID_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'oceanTexture',
  });
  watermanFwdTexture = device.createTexture({
    size: { width: GRID_TEX_W, height: GRID_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'watermanFwdTexture',
  });
  watermanInvTexture = device.createTexture({
    size: { width: WATERMAN_INV_W, height: WATERMAN_INV_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'watermanInvTexture',
  });

  for (const engineName in engines) {
    engines[engineName].setup(device);
  }

  const bgModule = device.createShaderModule({
    code: `${structsShader}\n${projectionShader}\n${bgRenderShader}`,
  });

  bgRenderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: bgModule,
      entryPoint: 'vertex',
    },
    fragment: {
      module: bgModule,
      entryPoint: 'fragment',
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

  bgBindGroup = device.createBindGroup({
    layout: bgRenderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: windTexture.createView() },
      { binding: 2, resource: watermanInvTexture.createView() },
      { binding: 3, resource: oceanTexture.createView() },
    ],
  });

  const renderModule = device.createShaderModule({
    code: `${structsShader}\n${projectionShader}\n${renderShader}`,
  });

  renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vertex',
      buffers: [
        {
          arrayStride: particleStride,
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32' },
            { shaderLocation: 2, offset: 32, format: 'float32x3' },
            { shaderLocation: 4, offset: 44, format: 'float32' },
          ],
        },
        {
          arrayStride: valenceStride,
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 3, offset: 0, format: 'float32x4' },
          ],
        },
      ],
    },
    fragment: {
      module: renderModule,
      entryPoint: 'fragment',
      targets: [
        {
          format: presentationFormat,
          blend: {
            color: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-strip' },
  });

  syncCanvasDimensions();
  updateProjectionState();
  setupTimestamp(device, 'render');

  drawScaleBar(WIND_LEVELS[earthParams.windLevel].scaleMax);
  refreshHudVisibility();

  const loadedOcean = await oceanBootPromise;
  if (loadedOcean) {
    currentOceanGrid = loadedOcean;
    uploadOceanToGpu(loadedOcean);
    refreshOceanHud(loadedOcean);
  } else {
    earthParams.oceanStatus = 'Ocean data unavailable';
    oceanStatusBinding.refresh();
  }

  startParticles();
  startOceanParticles();
  refreshEngineCrossBio();
  syncBioUniformState();
  bindOptions(
    device,
    simData,
    simBuffer,
    oceanSimData,
    oceanSimBuffer,
    syncBioUniformState,
  );
  loadWindLevel(earthParams.windLevel);
})();

function tick(commandEncoder: GPUCommandEncoder) {
  if (!device) return;
  const runAir = earthParams.viewLayer !== 'ocean';
  const runOcean = earthParams.viewLayer !== 'air';
  const curEngine = engines[engine];

  if (runAir && runOcean) {
    curEngine.buildIndex(device, commandEncoder, alternate, particleAmt, 'air', true);
    curEngine.buildIndex(
      device,
      commandEncoder,
      oceanAlternate,
      oceanParticleAmt,
      'ocean',
      false,
    );
    if (earthParams.bioLink && earthParams.bioRate > 0) {
      curEngine.stepBio(
        commandEncoder,
        alternate,
        oceanAlternate,
        particleAmt,
        oceanParticleAmt,
      );
    }
    curEngine.stepSim(device, commandEncoder, alternate, particleAmt, 'air', true);
    alternate = (alternate + 1) % 2;
    curEngine.stepSim(
      device,
      commandEncoder,
      oceanAlternate,
      oceanParticleAmt,
      'ocean',
      false,
    );
    oceanAlternate = (oceanAlternate + 1) % 2;
    return;
  }

  if (runAir) {
    curEngine.tick(device, commandEncoder, alternate, particleAmt, 'air', true);
    alternate = (alternate + 1) % 2;
  }
  if (runOcean) {
    curEngine.tick(
      device,
      commandEncoder,
      oceanAlternate,
      oceanParticleAmt,
      'ocean',
      true,
    );
    oceanAlternate = (oceanAlternate + 1) % 2;
  }
}

function render(commandEncoder: GPUCommandEncoder) {
  if (
    !device ||
    !bgContext ||
    !context ||
    !bgRenderPipeline ||
    !renderPipeline ||
    !bgBindGroup
  )
    return;

  const bgPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: bgContext.getCurrentTexture().createView(),
        clearValue: [0, 0, 0.0196, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });
  bgPass.setPipeline(bgRenderPipeline);
  bgPass.setBindGroup(0, bgBindGroup);
  bgPass.draw(3, 1, 0, 0);
  bgPass.end();

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

  const passEncoder = commandEncoder.beginRenderPass(
    linkRenderTimestamp(device, renderPassDescriptor, 'render'),
  );
  passEncoder.setPipeline(renderPipeline);

  if (
    earthParams.viewLayer !== 'air' &&
    oceanRenderBindGroup &&
    oceanParticleBuffers &&
    oceanValenceBuffer
  ) {
    passEncoder.setVertexBuffer(0, oceanParticleBuffers[oceanAlternate]);
    passEncoder.setVertexBuffer(1, oceanValenceBuffer);
    passEncoder.setBindGroup(0, oceanRenderBindGroup);
    passEncoder.draw(6, oceanParticleAmt, 0, 0);
  }

  if (
    earthParams.viewLayer !== 'ocean' &&
    renderBindGroup &&
    particleBuffers &&
    valenceBuffer
  ) {
    passEncoder.setVertexBuffer(0, particleBuffers[alternate]);
    passEncoder.setVertexBuffer(1, valenceBuffer);
    passEncoder.setBindGroup(0, renderBindGroup);
    passEncoder.draw(6, particleAmt, 0, 0);
  }

  passEncoder.end();
  resolveTimestamp(commandEncoder, 'render');
}

function startParticles() {
  if (
    !device ||
    !uniformBuffer ||
    !renderPipeline ||
    !simBuffer ||
    !windTexture ||
    !watermanFwdTexture
  )
    return;

  if (particleBuffers) {
    particleBuffers[0].destroy();
    particleBuffers[1].destroy();
  }
  matrixBuffer?.destroy();
  colourBuffer?.destroy();
  valenceBuffer?.destroy();
  airBioReactBuffer?.destroy();

  refreshChemistryDefinition();

  const bufferSize = particleAmt * particleStride;
  particleBuffers = [
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    }),
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    }),
  ];

  valenceBuffer = device.createBuffer({
    size: particleAmt * valenceStride,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.VERTEX |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    label: 'valenceBuffer',
  });
  airBioReactBuffer = device.createBuffer({
    size: particleAmt * valenceStride,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: 'airBioReactBuffer',
  });

  alternate = 0;
  frameCount = 0;
  const rng = createSeededRng(`${params.seed}:chem`);
  const probs = computeAbundanceDistribution(colourAmt);
  const cdf: number[] = [];
  let accW = 0;
  for (let c = 0; c < colourAmt; c++) {
    accW += probs[c];
    cdf.push(accW);
  }

  const pickElement = () => {
    const u = rng();
    for (let c = 0; c < colourAmt; c++) {
      if (u <= cdf[c]) return c;
    }
    return colourAmt - 1;
  };

  const data = new Float32Array(bufferSize / 4);
  for (let pi = 0; pi < particleAmt; pi++) {
    const z = rng() * 2 - 1;
    const lon = rng() * Math.PI * 2;
    const rxy = Math.sqrt(Math.max(0, 1 - z * z));
    const nx = rxy * Math.cos(lon);
    const ny = rxy * Math.sin(lon);
    const nz = z;
    const [ux, uy, uz] = unitTangentFromSphere(nx, ny, nz, rng() * Math.PI * 2);

    const b = pi * 12;
    data[b + 0] = nx;
    data[b + 1] = ny;
    data[b + 2] = nz;
    data[b + 3] = pickElement();
    data[b + 8] = ux;
    data[b + 9] = uy;
    data[b + 10] = uz;
  }

  device.queue.writeBuffer(particleBuffers[0], 0, data.buffer);
  setSim(device, simData, simBuffer, frameCount);

  matrixBuffer = device.createBuffer({
    size: chemistryData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  colourBuffer = device.createBuffer({
    size: colourAmt * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  writeChemistryBuffers();

  const windView = windTexture.createView();
  for (const engineName in engines) {
    engines[engineName].start(
      device,
      uniformBuffer,
      simBuffer,
      matrixBuffer,
      valenceBuffer,
      particleBuffers,
      particleAmt,
      params.cells,
      windView,
      'air',
      airBioReactBuffer,
    );
  }
  refreshEngineCrossBio();
  syncBioUniformState();

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: watermanFwdTexture.createView() },
      { binding: 2, resource: { buffer: colourBuffer } },
      { binding: 3, resource: { buffer: matrixBuffer } },
    ],
  });
}

function startOceanParticles() {
  if (
    !device ||
    !oceanUniformBuffer ||
    !renderPipeline ||
    !oceanSimBuffer ||
    !oceanTexture ||
    !watermanFwdTexture
  )
    return;

  if (oceanParticleBuffers) {
    oceanParticleBuffers[0].destroy();
    oceanParticleBuffers[1].destroy();
  }
  oceanMatrixBuffer?.destroy();
  oceanColourBuffer?.destroy();
  oceanValenceBuffer?.destroy();
  oceanSteerBuffer?.destroy();

  oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);

  const bufferSize = oceanParticleAmt * particleStride;
  oceanParticleBuffers = [
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: 'oceanParticles0',
    }),
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: 'oceanParticles1',
    }),
  ];

  oceanValenceBuffer = device.createBuffer({
    size: oceanParticleAmt * valenceStride,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.VERTEX |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    label: 'oceanStateBuffer',
  });
  oceanSteerBuffer = device.createBuffer({
    size: oceanParticleAmt * valenceStride,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: 'oceanSteerBuffer',
  });
  writeNeutralOceanState();

  oceanAlternate = 0;
  const rng = createSeededRng(`${oceanSeed}:spawn`);
  const data = new Float32Array(bufferSize / 4);
  const DEG2RAD = Math.PI / 180;

  for (let pi = 0; pi < oceanParticleAmt; pi++) {
    let lonDeg = (rng() - 0.5) * 360;
    let latDeg = Math.asin(rng() * 2 - 1) * (180 / Math.PI);

    if (currentOceanGrid) {
      for (let t = 0; t < 40; t++) {
        const candLon = (rng() - 0.5) * 360;
        const candLat = Math.asin(rng() * 2 - 1) * (180 / Math.PI);
        if (sampleOceanLonLat(currentOceanGrid, candLon, candLat)[3] >= 0.55) {
          lonDeg = candLon;
          latDeg = candLat;
          break;
        }
      }
    }

    const lonR = lonDeg * DEG2RAD;
    const latR = latDeg * DEG2RAD;
    const cLat = Math.cos(latR);
    const nx = cLat * Math.cos(lonR);
    const ny = cLat * Math.sin(lonR);
    const nz = Math.sin(latR);
    const [ux, uy, uz] = unitTangentFromSphere(nx, ny, nz, rng() * Math.PI * 2);

    const b = pi * 12;
    data[b + 0] = nx;
    data[b + 1] = ny;
    data[b + 2] = nz;
    data[b + 3] = Math.floor(rng() * oceanColourAmt);
    data[b + 8] = ux;
    data[b + 9] = uy;
    data[b + 10] = uz;
  }

  device.queue.writeBuffer(oceanParticleBuffers[0], 0, data.buffer);
  setOceanSim(device, oceanSimData, oceanSimBuffer);

  oceanMatrixBuffer = device.createBuffer({
    size: Math.max(oceanMatrixData.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'oceanMatrixBuffer',
  });
  oceanColourBuffer = device.createBuffer({
    size: oceanColourAmt * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'oceanColourBuffer',
  });
  writeOceanRuleBuffers();

  const oceanView = oceanTexture.createView();
  for (const engineName in engines) {
    engines[engineName].start(
      device,
      oceanUniformBuffer,
      oceanSimBuffer,
      oceanMatrixBuffer,
      oceanValenceBuffer,
      oceanParticleBuffers,
      oceanParticleAmt,
      oceanOptionParams.cells,
      oceanView,
      'ocean',
      oceanSteerBuffer,
    );
  }
  refreshEngineCrossBio();
  syncBioUniformState();

  oceanRenderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: oceanUniformBuffer } },
      { binding: 1, resource: watermanFwdTexture.createView() },
      { binding: 2, resource: { buffer: oceanColourBuffer } },
      { binding: 3, resource: { buffer: oceanMatrixBuffer } },
    ],
  });
}

let lastTime = 0;
const deltaVs: number[] = [];

function writeUniforms() {
  if (!device || !uniformBuffer || !oceanUniformBuffer) return;
  const spec = WIND_LEVELS[earthParams.windLevel];
  const R = Math.max(optionParams.worldSize, 0.25);
  const showAir = earthParams.viewLayer !== 'ocean';
  const showOcean = earthParams.viewLayer !== 'air';

  uniformData[0] = viewState.width;
  uniformData[1] = viewState.height;
  uniformData[2] = viewState.width / Math.max(viewState.height, 1);
  uniformData[3] = ((1 / optionParams.r) / R) * optionParams.beta * 0.34;

  uniformData[4] = mouseSphere.x;
  uniformData[5] = mouseSphere.y;
  uniformData[6] = mouseSphere.z;
  uniformData[7] = mouseSphere.mode;

  uniformData[8] = viewState.scale;
  uniformData[9] = viewState.pan[0];
  uniformData[10] = viewState.pan[1];
  uniformData[11] = PROJECTIONS[viewState.projectionId].shaderType;

  uniformData[12] = optionParams.showOrbitals ? 1 : 0;
  uniformData[13] = showAir && earthParams.windColors ? 1 : 0;
  uniformData[14] = spec.scaleMax;
  uniformData[15] = spec.streamMax;

  const rotCols = buildD3RotationMatrixColMajor(viewState.rotate);
  uniformData.set(rotCols, 16);

  uniformData[28] = showOcean && earthParams.oceanColors ? 1 : 0;
  uniformData[29] = OCEAN_CURRENTS_SPEC.scaleMax;
  uniformData[30] = earthParams.oceanStrength;
  uniformData[31] = 0;

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Ocean uniform copy renders classic smooth discs (`flags.x = 0`) sized to ocean interaction radius
  oceanUniformData.set(uniformData);
  const oceanRMax = 1 / Math.max(oceanOptionParams.r, 0.5);
  oceanUniformData[3] = (oceanRMax / R) * Math.max(oceanOptionParams.beta, 0.25) * 0.42;
  oceanUniformData[12] = 0;
  device.queue.writeBuffer(oceanUniformBuffer, 0, oceanUniformData);
}

function update() {
  requestAnimationFrame(update);
  if (!device || !context || !bgContext) return;

  const start = performance.now();
  deltaVs.push(start - lastTime);
  if (deltaVs.length > 1000) {
    deltaVs.splice(0, 1);
  }
  lastTime = start;

  syncCanvasDimensions();

  if (
    viewState.projectionId === 'waterman' &&
    watermanDirty &&
    watermanFwdTexture &&
    watermanInvTexture
  ) {
    watermanDirty = false;
    const luts = buildWatermanLutData(viewState, d3Proj);
    device.queue.writeTexture(
      { texture: watermanFwdTexture },
      luts.fwdData,
      { bytesPerRow: luts.fwdW * 16 },
      { width: luts.fwdW, height: luts.fwdH, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: watermanInvTexture },
      luts.invData,
      { bytesPerRow: luts.invW * 16 },
      { width: luts.invW, height: luts.invH, depthOrArrayLayers: 1 },
    );
  }

  if (mapDirty) {
    mapDirty = false;
    renderMapOverlaySvg(
      mapSvg,
      d3Proj,
      earthFeatures,
      earthParams.coastlines,
      viewState.projectionId === 'orthographic',
    );
  }

  const showAir = earthParams.viewLayer !== 'ocean';
  const showOcean = earthParams.viewLayer !== 'air';

  if (oceanCtx && currentOceanGrid) {
    const oceanTint: [number, number, number] =
      showAir && earthParams.windLines ? [115, 235, 255] : [235, 250, 255];
    oceanStreamRenderer.stepAndDraw(
      oceanCtx,
      viewState.width,
      viewState.height,
      d3Proj,
      oceanGridToSampler(currentOceanGrid, oceanTint),
      viewSignature,
      showOcean && earthParams.oceanLines,
    );
  }

  if (windCtx) {
    windStreamRenderer.stepAndDraw(
      windCtx,
      viewState.width,
      viewState.height,
      d3Proj,
      currentWindGrid,
      viewSignature,
      showAir && earthParams.windLines,
    );
  }

  frameCount = (frameCount + 1) % 1000000;
  if (simBuffer && optionParams.temperature > 0) {
    frameBuf[0] = frameCount;
    simData[18] = frameCount;
    device.queue.writeBuffer(simBuffer, 18 * 4, frameBuf);
  }

  writeUniforms();

  const commandEncoder = device.createCommandEncoder();
  tick(commandEncoder);
  render(commandEncoder);

  const commands = commandEncoder.finish();
  device.queue.submit([commands]);

  device.popErrorScope().then((error) => {
    if (error && !window.location.search.includes('noTimestamp')) {
      const url = new URL(window.location.href);
      url.searchParams.set('noTimestamp', '');
      window.location.replace(url.toString());
    }
  });

  const cpuTime = performance.now() - start;
  globalPerformanceParams.cpu = cpuTime;
  updateTotal();

  for (const engine2 in engines) {
    if (engine == engine2) {
      engines[engine2].updateDisplays(
        performanceParams[engine2 as keyof typeof performanceParams],
      );
    }
  }

  readTimestamp('render').then((time) => {
    globalPerformanceParams.render = time;
    updateTotal();
  });

  fpsc++;
}

requestAnimationFrame(update);

newSimBtn.on('click', () => {
  colourAmt = Math.max(1, Math.min(62, Math.floor(optionParams.colours)));
  params.colours = colourAmt;
  updatePeriodicLegend(colourAmt);

  params.cells = Math.max(1, Math.floor(optionParams.cells));
  particleAmt = Math.max(1, Math.floor(params.particles));

  startParticles();
});

randomizeBtn.on('click', () => {
  if (!device || !matrixBuffer) return;
  applySeed(generateRandomSeed(), true);
});

seedInput.on('change', () => {
  if (suppressSeedChange || !device || !matrixBuffer) return;
  applySeed(params.seed, false);
});

newOceanSimBtn.on('click', () => {
  oceanColourAmt = Math.max(1, Math.min(200, Math.floor(oceanOptionParams.colours)));
  oceanParticleAmt = Math.max(1, Math.floor(oceanOptionParams.particles));
  startOceanParticles();
});

randomizeOceanRulesBtn.on('click', () => {
  oceanColourAmt = Math.max(1, Math.min(200, Math.floor(oceanOptionParams.colours)));
  oceanSeed = generateRandomSeed();
  if (!oceanMatrixBuffer || oceanMatrixData.length !== oceanColourAmt * oceanColourAmt) {
    startOceanParticles();
  } else {
    oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);
    writeOceanRuleBuffers();
  }
});

viewLayerSelect.on('change', (ev) => {
  setViewLayerChoice(ev.value);
  syncBioUniformState();
});

bioLinkBinding.on('change', () => {
  syncBioUniformState();
});

bioRateBinding.on('change', () => {
  syncBioUniformState();
});

projectionSelect.on('change', (ev) => {
  setProjectionChoice(ev.value);
});

windLevelSelect.on('change', (ev) => {
  loadWindLevel(ev.value);
});

windLinesBinding.on('change', () => {
  if (!earthParams.windLines && windCtx) {
    windStreamRenderer.clear(windCtx, viewState.width, viewState.height);
  }
});

oceanLinesBinding.on('change', () => {
  if (!earthParams.oceanLines && oceanCtx) {
    oceanStreamRenderer.clear(oceanCtx, viewState.width, viewState.height);
  }
});

windColorsBinding.on('change', () => {
  // Picked up immediately by bgRender uniform
});

oceanColorsBinding.on('change', () => {
  // Picked up immediately by bgRender uniform
});

coastlinesBinding.on('change', () => {
  mapDirty = true;
});

resetViewBtn.on('click', () => {
  earthParams.projection = 'orthographic';
  projectionSelect.refresh();
  viewState.projectionId = 'orthographic';
  viewState.rotate = [...PROJECTIONS.orthographic.defaultRotate];
  viewState.pan = [0, 0];
  viewState.scale = Math.max(
    140,
    Math.round(586 * Math.min(viewState.width / 1280, viewState.height / 800) || 586),
  );
  updateProjectionState();
});

setInterval(() => {
  params.fps = fpsc;
  fpsc = 0;
}, 1000);

engineSelect.on('change', (event) => {
  if (event.value in engines) {
    engine = event.value;
    setEngineDisplay(engine);
  }
});

function updateTotal() {
  let deltaTotal = 0;
  for (const time of deltaVs) {
    deltaTotal += time;
  }
  deltaTotal /= deltaVs.length;
  if (isNaN(deltaTotal)) deltaTotal = 0;

  let total = 0;
  const enginePerf =
    performanceParams[engine as keyof typeof performanceParams];
  for (const pass in enginePerf) {
    total += enginePerf[pass as keyof typeof enginePerf];
  }
  total += globalPerformanceParams.cpu;
  total += globalPerformanceParams.render;
  globalPerformanceParams.total = total;
}
