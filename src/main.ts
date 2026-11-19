import {
  bindGlobeInput,
  mouseSphere,
  refreshBoundProjection,
} from './input';
import { createSeededRng, generateRandomSeed } from './matrix';
import {
  buildPackedChemistryBuffer,
  computeAbundanceDistribution,
  getElementSymbol,
  isBiogenicAtomicNumber,
  type Element2D,
} from './periodicTable2D';
import type { UnifiedEngineRefs } from './engineTypes';
import structsShader from './shaders/structs.wgsl?raw';
import projectionShader from './shaders/projection.wgsl?raw';
import bgRenderShader from './shaders/bgRender.wgsl?raw';
import lensTrackShader from './shaders/lensTrack.wgsl?raw';
import lensOverlayShader from './shaders/lensOverlay.wgsl?raw';
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
  bindConfigPersistence,
  bindOptions,
  bioLinkBinding,
  bioRateBinding,
  buildFactoryConfig,
  chemOpacityBinding,
  coastlinesBinding,
  colourAmt as colourAmtOptn,
  copyConfigBtn,
  earthParams,
  engineSelect,
  getCachedConfig,
  globalPerformanceParams,
  hasExplicitUrlConfig,
  hasExplicitUrlHash,
  initialOrientation,
  initialUrlPan,
  lifeOpacityBinding,
  mouseToolBinding,
  newOceanSimBtn,
  newSimBtn,
  oceanColorsBinding,
  oceanColourAmt as oceanColourAmtOptn,
  oceanLinesBinding,
  oceanOptionParams,
  oceanStatusBinding,
  optionParams,
  params,
  particleAmt as particleAmtOptn,
  pasteConfigBtn,
  performanceParams,
  projectionSelect,
  randomizeBtn,
  randomizeOceanRulesBtn,
  refreshAllPane,
  resetViewBtn,
  seedInput,
  setBioLink,
  setEngineDisplay,
  setOceanSim,
  setSim,
  tooltipsBinding,
  updatePeriodicLegend,
  viewLayerSelect,
  type ViewLayerMode,
  windColorsBinding,
  windLevelSelect,
  windLinesBinding,
  windStatusBinding,
} from './options';
import { setMouseToolMode } from './input';
import {
  buildUrlPartsFromConfig,
  clearCachedConfig,
  parseConfigInput,
  saveCachedConfig,
  type SimConfigV1,
} from './config';
import {
  kinAngularDist,
  summarizeAndClassifyBlob,
  TooltipManager,
  unpackLifeMeta,
  type HoverSimSnapshot,
  type TooltipContent,
  type TooltipLensState,
} from './tooltip';

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

const uniformsFloats = 40; // 160 bytes
const uniformData = new Float32Array(uniformsFloats);
const simSize = 24;
const simData = new Float32Array(simSize);
const oceanSimSize = 24;
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
const cachedBootCfg = getCachedConfig();
const urlOceanSeed = new URLSearchParams(window.location.search).get('oceanSeed');
let oceanSeed =
  urlOceanSeed ??
  (hasExplicitUrlConfig ? generateRandomSeed() : cachedBootCfg?.oceanSeed) ??
  generateRandomSeed();
let oceanMatrixData: Float32Array<ArrayBuffer> = new Float32Array(0);

function computeBioSeedHash(seed: string, oSeed: string): number {
  const s = `${seed}:${oSeed}`;
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) & 0xffff;
}

let speciesTraits: number[][] = [];

function buildRandomOceanMatrix(nColours: number, seed: string): Float32Array<ArrayBuffer> {
  const rng = createSeededRng(`${seed}:oceanMatrix`);
  const mat = new Float32Array(nColours * nColours);
  for (let i = 0; i < nColours; i++) {
    for (let j = 0; j < nColours; j++) {
      mat[i * nColours + j] =
        i === j
          ? Math.max(-1, Math.min(1, rng() * 2 - 0.72))
          : rng() * 2 - 1;
    }
  }
  return mat;
}

function computeSpeciesTraitsFromMatrix(
  mat: Float32Array,
  nBio: number,
): { flat: Float32Array<ArrayBuffer>; table: number[][] } {
  const n = Math.max(1, nBio);
  const flat = new Float32Array(n * 4);
  const table: number[][] = [];
  for (let i = 0; i < n; i++) {
    let posMutualSum = 0;
    let posDiffSum = 0;
    let negDiffSum = 0;
    for (let j = 0; j < n; j++) {
      const aIJ = mat[i * n + j] ?? 0;
      const aJI = mat[j * n + i] ?? 0;
      posMutualSum += Math.max(0, 0.5 * (aIJ + aJI));
      if (j !== i) {
        posDiffSum += Math.max(0, aIJ - aJI);
        negDiffSum += Math.max(0, aJI - aIJ);
      }
    }
    const denom = Math.max(1, n - 1);
    const coh =
      n <= 1
        ? Math.max(0, Math.min(1, (mat[0] ?? 0) * 0.5 + 0.5))
        : Math.max(0, Math.min(1, (posMutualSum / n) * 2.2));
    const chase = n <= 1 ? 0 : Math.max(0, Math.min(1, (posDiffSum / denom) * 1.6));
    const flee = n <= 1 ? 0 : Math.max(0, Math.min(1, (negDiffSum / denom) * 1.6));
    const asym = Math.max(0, Math.min(1, (chase + flee) * 0.65));
    flat[i * 4 + 0] = coh;
    flat[i * 4 + 1] = chase;
    flat[i * 4 + 2] = flee;
    flat[i * 4 + 3] = asym;
    table.push([coh, chase, flee, asym]);
  }
  return { flat, table };
}

oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);
speciesTraits = computeSpeciesTraitsFromMatrix(oceanMatrixData, oceanColourAmt).table;

const particleStride = 48; // 12 floats: pos(3), colour(1), vel(3), omega(1), orient(3), pad(1)
const valenceStride = 16;
const GRID_TEX_W = 360;
const GRID_TEX_H = 181;
const WATERMAN_INV_W = 640;
const WATERMAN_INV_H = 360;

let particleAmt = particleAmtOptn;

let device: GPUDevice | undefined;
let bgContext: GPUCanvasContext | undefined;
let context: GPUCanvasContext | undefined;

let uniformBuffer: GPUBuffer | undefined;
let simBuffer: GPUBuffer | undefined;
let oceanSimBuffer: GPUBuffer | undefined;
let bioUniformBuffer: GPUBuffer | undefined;
let bgRenderPipeline: GPURenderPipeline | undefined;
let renderPipeline: GPURenderPipeline | undefined;

let worldRulesBuffer: GPUBuffer | undefined;
let worldColoursBuffer: GPUBuffer | undefined;
let valenceBuffer: GPUBuffer | undefined;
let bioForceBuffer: GPUBuffer | undefined;
let particleBuffers: [GPUBuffer, GPUBuffer] | undefined;
let lifeStateBuffers: [GPUBuffer, GPUBuffer] | undefined;
let bioWasActive = false;

let windTexture: GPUTexture | undefined;
let oceanTexture: GPUTexture | undefined;
let watermanFwdTexture: GPUTexture | undefined;
let watermanInvTexture: GPUTexture | undefined;

let bgBindGroup: GPUBindGroup | undefined;
let renderBindGroup: GPUBindGroup | undefined;

let lensTrackPipeline: GPUComputePipeline | undefined;
let lensOverlayPipeline: GPURenderPipeline | undefined;
let lensTrackUniformBuffer: GPUBuffer | undefined;
let lensGpuStateBuffer: GPUBuffer | undefined;
let lensOverlayBindGroup: GPUBindGroup | undefined;
const lensTrackBindGroups: [GPUBindGroup | undefined, GPUBindGroup | undefined] = [
  undefined,
  undefined,
];
const lensTrackUniformData = new Float32Array(12);
let prevGpuLensGen = -1;
let prevGpuLensViewSig = '';
let lensReadbackBuffers: [GPUBuffer, GPUBuffer] | undefined;
const lensReadbackInFlight: [boolean, boolean] = [false, false];
let lensReadbackToggle = 0;
let pendingLensReadbackIdx: number | null = null;

let alternate = 0;
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
const useCachedView =
  !hasExplicitUrlHash && !hasExplicitUrlConfig && Boolean(cachedBootCfg?.view);
const initScale =
  initialOrientation && initialOrientation[2] > 50
    ? initialOrientation[2]
    : useCachedView && typeof cachedBootCfg?.view?.scale === 'number'
      ? cachedBootCfg.view.scale
      : earthParams.projection === 'orthographic'
        ? Math.round(586 * Math.min(window.innerWidth / 1280, window.innerHeight / 800) || 586)
        : fitScaleForProjection(
            earthParams.projection,
            initSpec.defaultRotate,
            window.innerWidth,
            window.innerHeight,
          );

const initRotate: [number, number, number] = initialOrientation
  ? [-initialOrientation[0], -initialOrientation[1], initSpec.defaultRotate[2]]
  : useCachedView && cachedBootCfg?.view?.rotate
    ? [...cachedBootCfg.view.rotate]
    : [...initSpec.defaultRotate];

const initPan: [number, number] =
  initialUrlPan ??
  (useCachedView && cachedBootCfg?.view?.pan ? [...cachedBootCfg.view.pan] : [0, 0]);

const viewState: GlobeViewState = {
  projectionId: earthParams.projection,
  rotate: initRotate,
  scale: Math.max(140, initScale),
  pan: initPan,
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
let configSaveTimer: number | undefined;
let configPersistenceReady = false;
let suppressPaneEvents = false;

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

function getCurrentShareUrl(): string {
  const { search, hash } = buildUrlPartsFromConfig(
    exportCurrentConfig(),
    window.location.search,
  );
  return `${window.location.origin}${window.location.pathname}${search}${hash}`;
}

function syncUrlStateNow() {
  if (hashSyncTimer !== undefined) {
    window.clearTimeout(hashSyncTimer);
    hashSyncTimer = undefined;
  }
  const { search, hash } = buildUrlPartsFromConfig(
    exportCurrentConfig(),
    window.location.search,
  );
  const nextRel = `${window.location.pathname}${search}${hash}`;
  const curRel = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (curRel !== nextRel) {
    window.history.replaceState(null, '', nextRel);
  }
}

function syncUrlStateDebounced() {
  if (hashSyncTimer !== undefined) {
    window.clearTimeout(hashSyncTimer);
  }
  hashSyncTimer = window.setTimeout(() => {
    hashSyncTimer = undefined;
    syncUrlStateNow();
  }, 180);
}

function exportCurrentConfig(): SimConfigV1 {
  return {
    v: 1,
    engine,
    particles: particleAmt,
    colours: colourAmt,
    cells: params.cells,
    seed: params.seed,
    oceanSeed,
    air: {
      r: optionParams.r,
      beta: optionParams.beta,
      force: optionParams.force,
      delta: optionParams.delta,
      friction: optionParams.friction,
      torque: optionParams.torque,
      angularFriction: optionParams.angularFriction,
      valency: optionParams.valency,
      temperature: optionParams.temperature,
      activation: optionParams.activation,
      orbitalSharpness: optionParams.orbitalSharpness,
      bondStability: optionParams.bondStability,
      exchangeDrive: optionParams.exchangeDrive,
      shearBreak: optionParams.shearBreak,
      shearFloor: optionParams.shearFloor,
      showOrbitals: optionParams.showOrbitals,
      avoidance: optionParams.avoidance,
      worldSize: optionParams.worldSize,
    },
    ocean: {
      colours: oceanColourAmt,
      r: oceanOptionParams.r,
      beta: oceanOptionParams.beta,
      force: oceanOptionParams.force,
      delta: oceanOptionParams.delta,
      friction: oceanOptionParams.friction,
      avoidance: oceanOptionParams.avoidance,
    },
    earth: {
      viewLayer: earthParams.viewLayer,
      projection: earthParams.projection,
      windLevel: earthParams.windLevel,
      vortex: earthParams.vortex,
      windStrength: earthParams.windStrength,
      windLines: earthParams.windLines,
      windColors: earthParams.windColors,
      chemOpacity: earthParams.chemOpacity,
      oceanMix: earthParams.oceanMix,
      oceanStrength: earthParams.oceanStrength,
      oceanLines: earthParams.oceanLines,
      oceanColors: earthParams.oceanColors,
      lifeOpacity: earthParams.lifeOpacity,
      coastlines: earthParams.coastlines,
      bioLink: earthParams.bioLink,
      bioRate: earthParams.bioRate,
      tooltips: earthParams.tooltips,
      mouseTool: earthParams.mouseTool,
    },
    view: {
      rotate: [...viewState.rotate],
      pan: [...viewState.pan],
      scale: viewState.scale,
    },
  };
}

let skipAutoSave = false;

async function resetToFactoryDefaults(): Promise<SimConfigV1> {
  skipAutoSave = true;
  if (configSaveTimer !== undefined) {
    window.clearTimeout(configSaveTimer);
    configSaveTimer = undefined;
  }
  clearCachedConfig();
  try {
    const factory = buildFactoryConfig(generateRandomSeed(), generateRandomSeed());
    const res = await importConfiguration(factory);
    if (configSaveTimer !== undefined) {
      window.clearTimeout(configSaveTimer);
      configSaveTimer = undefined;
    }
    clearCachedConfig();
    syncUrlStateNow();
    return res;
  } finally {
    skipAutoSave = false;
  }
}

function saveConfigNow() {
  syncUrlStateNow();
  if (!configPersistenceReady || skipAutoSave) return;
  saveCachedConfig(exportCurrentConfig());
}

function scheduleSaveConfig() {
  if (!configPersistenceReady || skipAutoSave) return;
  syncUrlStateDebounced();
  if (configSaveTimer !== undefined) {
    window.clearTimeout(configSaveTimer);
  }
  configSaveTimer = window.setTimeout(() => {
    configSaveTimer = undefined;
    if (!skipAutoSave) {
      saveCachedConfig(exportCurrentConfig());
    }
  }, 220);
}

async function importConfiguration(rawInput: string | SimConfigV1): Promise<SimConfigV1> {
  const cfg = parseConfigInput(rawInput);

  let resetWorld = false;
  let nextWindLevel: WindLevelId | null = null;
  let nextViewLayer: ViewLayerMode | null = null;
  let projUpdated = false;

  if (cfg.engine && cfg.engine in engines) {
    engine = cfg.engine;
    params.engine = cfg.engine;
    setEngineDisplay(engine);
  }

  if (cfg.particles !== undefined) {
    const p = Math.max(1, Math.floor(cfg.particles));
    if (p !== particleAmt) resetWorld = true;
    particleAmt = p;
    params.particles = p;
  }
  if (cfg.colours !== undefined) {
    const c = Math.max(1, Math.min(62, Math.floor(cfg.colours)));
    if (c !== colourAmt) resetWorld = true;
    colourAmt = c;
    params.colours = c;
    optionParams.colours = c;
  }
  if (cfg.cells !== undefined) {
    const cl = Math.max(1, Math.floor(cfg.cells));
    if (cl !== params.cells) resetWorld = true;
    params.cells = cl;
    optionParams.cells = cl;
  }
  if (cfg.seed !== undefined && cfg.seed !== params.seed) {
    params.seed = cfg.seed;
    resetWorld = true;
  }
  if (cfg.oceanSeed !== undefined && cfg.oceanSeed !== oceanSeed) {
    oceanSeed = cfg.oceanSeed;
    resetWorld = true;
  }

  if (cfg.air) {
    const a = cfg.air;
    if (a.r !== undefined) optionParams.r = a.r;
    if (a.beta !== undefined) optionParams.beta = a.beta;
    if (a.force !== undefined) optionParams.force = a.force;
    if (a.delta !== undefined) optionParams.delta = a.delta;
    if (a.friction !== undefined) optionParams.friction = a.friction;
    if (a.torque !== undefined) optionParams.torque = a.torque;
    if (a.angularFriction !== undefined) optionParams.angularFriction = a.angularFriction;
    if (a.valency !== undefined) optionParams.valency = a.valency;
    if (a.temperature !== undefined) optionParams.temperature = a.temperature;
    if (a.activation !== undefined) optionParams.activation = a.activation;
    if (a.orbitalSharpness !== undefined) optionParams.orbitalSharpness = a.orbitalSharpness;
    if (a.bondStability !== undefined) optionParams.bondStability = a.bondStability;
    if (a.exchangeDrive !== undefined) optionParams.exchangeDrive = a.exchangeDrive;
    if (a.shearBreak !== undefined) optionParams.shearBreak = a.shearBreak;
    if (a.shearFloor !== undefined) optionParams.shearFloor = a.shearFloor;
    if (a.showOrbitals !== undefined) optionParams.showOrbitals = a.showOrbitals;
    if (a.avoidance !== undefined) optionParams.avoidance = a.avoidance;
    if (a.worldSize !== undefined) {
      optionParams.worldSize = a.worldSize;
      oceanOptionParams.worldSize = a.worldSize;
    }
  }

  if (cfg.ocean) {
    const o = cfg.ocean;
    if (o.colours !== undefined) {
      const oc = Math.max(1, Math.min(200, Math.floor(o.colours)));
      if (oc !== oceanColourAmt) resetWorld = true;
      oceanColourAmt = oc;
      oceanOptionParams.colours = oc;
    }
    if (o.r !== undefined) oceanOptionParams.r = o.r;
    if (o.beta !== undefined) oceanOptionParams.beta = o.beta;
    if (o.force !== undefined) oceanOptionParams.force = o.force;
    if (o.delta !== undefined) oceanOptionParams.delta = o.delta;
    if (o.friction !== undefined) oceanOptionParams.friction = o.friction;
    if (o.avoidance !== undefined) oceanOptionParams.avoidance = o.avoidance;
  }

  if (cfg.earth) {
    const e = cfg.earth;
    if (e.viewLayer && e.viewLayer !== earthParams.viewLayer) {
      nextViewLayer = e.viewLayer;
    }
    if (e.projection && e.projection !== earthParams.projection) {
      earthParams.projection = e.projection;
      viewState.projectionId = e.projection;
      projUpdated = true;
    }
    if (e.windLevel && e.windLevel !== earthParams.windLevel) {
      earthParams.windLevel = e.windLevel;
      nextWindLevel = e.windLevel;
    }
    if (e.vortex !== undefined) earthParams.vortex = e.vortex;
    if (e.windStrength !== undefined) earthParams.windStrength = e.windStrength;
    if (e.windLines !== undefined) earthParams.windLines = e.windLines;
    if (e.windColors !== undefined) earthParams.windColors = e.windColors;
    if (e.chemOpacity !== undefined) earthParams.chemOpacity = e.chemOpacity;
    if (e.oceanMix !== undefined) earthParams.oceanMix = e.oceanMix;
    if (e.oceanStrength !== undefined) earthParams.oceanStrength = e.oceanStrength;
    if (e.oceanLines !== undefined) earthParams.oceanLines = e.oceanLines;
    if (e.oceanColors !== undefined) earthParams.oceanColors = e.oceanColors;
    if (e.lifeOpacity !== undefined) earthParams.lifeOpacity = e.lifeOpacity;
    if (e.coastlines !== undefined) {
      earthParams.coastlines = e.coastlines;
      mapDirty = true;
    }
    if (e.bioLink !== undefined) earthParams.bioLink = e.bioLink;
    if (e.bioRate !== undefined) earthParams.bioRate = e.bioRate;
    if (e.tooltips !== undefined) earthParams.tooltips = e.tooltips;
    if (e.mouseTool !== undefined) {
      earthParams.mouseTool = e.mouseTool;
      setMouseToolMode(e.mouseTool);
    }
  }

  if (cfg.view) {
    if (cfg.view.rotate) {
      viewState.rotate = [...cfg.view.rotate];
      projUpdated = true;
    }
    if (cfg.view.pan) {
      viewState.pan = [...cfg.view.pan];
      projUpdated = true;
    }
    if (cfg.view.scale !== undefined) {
      viewState.scale = Math.max(120, Math.min(30000, cfg.view.scale));
      projUpdated = true;
    }
  }

  syncStreamCanvasOpacity();
  if (earthParams.windLines <= 0.001 && windCtx) {
    windStreamRenderer.clear(windCtx, viewState.width, viewState.height);
  }
  if (earthParams.oceanLines <= 0.001 && oceanCtx) {
    oceanStreamRenderer.clear(oceanCtx, viewState.width, viewState.height);
  }
  tooltipManager.syncEnabledState();

  if (nextViewLayer) {
    setViewLayerChoice(nextViewLayer);
  } else {
    refreshHudVisibility();
  }

  if (device && simBuffer) setSim(device, simData, simBuffer, frameCount);
  if (device && oceanSimBuffer) setOceanSim(device, oceanSimData, oceanSimBuffer);
  syncBioUniformState();

  if (projUpdated) {
    updateProjectionState();
  }

  updatePeriodicLegend(colourAmt);
  if (resetWorld) {
    refreshChemistryDefinition();
    startWorld();
  }
  if (nextWindLevel) {
    await loadWindLevel(nextWindLevel);
  }

  suppressPaneEvents = true;
  try {
    refreshAllPane();
    mouseToolBinding.refresh();
  } finally {
    suppressPaneEvents = false;
  }
  saveConfigNow();
  syncUrlStateNow();
  return exportCurrentConfig();
}

function updateProjectionState() {
  d3Proj = buildConfiguredD3Projection(viewState);
  refreshBoundProjection(d3Proj);
  mapDirty = true;
  if (viewState.projectionId === 'waterman') {
    watermanDirty = true;
  }
  viewSignature = `${viewState.projectionId}:${viewState.rotate.map((v) => v.toFixed(2)).join(',')}:${viewState.scale.toFixed(1)}:${viewState.pan.map((v) => v.toFixed(1)).join(',')}:${viewState.width}x${viewState.height}`;
  syncUrlStateDebounced();
  scheduleSaveConfig();
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

function writeWorldRuleAndColourBuffers() {
  if (!device || !worldRulesBuffer || !worldColoursBuffer) return;
  const { flat: traitsFlat, table } = computeSpeciesTraitsFromMatrix(
    oceanMatrixData,
    oceanColourAmt,
  );
  speciesTraits = table;
  const rules = new Float32Array(
    chemistryData.length + oceanMatrixData.length + traitsFlat.length,
  );
  rules.set(chemistryData, 0);
  rules.set(oceanMatrixData, chemistryData.length);
  rules.set(traitsFlat, chemistryData.length + oceanMatrixData.length);
  device.queue.writeBuffer(worldRulesBuffer, 0, rules.buffer);

  const totalPalette = colourAmt + oceanColourAmt;
  const colourData = new Float32Array(totalPalette * 3);
  for (let c = 0; c < colourAmt; c++) {
    colourData[c * 3] = elements[c].color[0];
    colourData[c * 3 + 1] = elements[c].color[1];
    colourData[c * 3 + 2] = elements[c].color[2];
  }
  for (let i = 0; i < oceanColourAmt; i++) {
    const hue = (i * 137.508) % 360;
    const [r, g, b] = hslToRgb(hue, 0.9, 0.54);
    const outIdx = (colourAmt + i) * 3;
    colourData[outIdx] = r;
    colourData[outIdx + 1] = g;
    colourData[outIdx + 2] = b;
  }
  device.queue.writeBuffer(worldColoursBuffer, 0, colourData.buffer);
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
      getTooltips: () => boolean;
      setTooltips: (on: boolean) => void;
      getOpacities: () => {
        windLines: number;
        windColors: number;
        chemOpacity: number;
        oceanLines: number;
        oceanColors: number;
        lifeOpacity: number;
      };
      setChemOpacity: (val: number) => void;
      setLifeOpacity: (val: number) => void;
      setOceanColors: (val: boolean | number) => void;
      setOceanLines: (val: boolean | number) => void;
      setWindColors: (val: boolean | number) => void;
      setWindLines: (val: boolean | number) => void;
      refreshTooltipSnapshot: () => Promise<HoverSimSnapshot | null>;
      inspectReactionAt: (clientX: number, clientY: number) => TooltipContent | null;
      getTooltipLens: () => TooltipLensState;
      exportConfig: () => SimConfigV1;
      getShareUrl: () => string;
      importConfig: (input: string | SimConfigV1) => Promise<SimConfigV1>;
      resetDefaultConfig: () => Promise<SimConfigV1>;
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
      readLifeState: (count?: number) => Promise<number[][]>;
      readBioForce: (count?: number) => Promise<number[][]>;
      readWorldState: (count?: number) => Promise<{
        particles: number[][];
        valence: number[][];
        lifeState: number[][];
        bioForce: number[][];
      }>;
      getWorldStats: () => Promise<{
        total: number;
        chemCount: number;
        bioCount: number;
        cytoplasmCount: number;
        lipidCount: number;
        guildCounts: [number, number, number, number];
        abioticBioViolations: number;
        lipidAbioticViolations: number;
        chemElementCounts: Record<string, number>;
        bioElementCounts: Record<string, number>;
        avgBioEnergy: number;
        oceanBioCount: number;
        landBioCount: number;
        blobCount: number;
        multiSpeciesBlobs: number;
        complexBlobs: number;
        encapsulatedBlobs: number;
        encapsulatedSimpleViolations: number;
        nakedBlobs: number;
        avgBlobSize: number;
        maxBlobSize: number;
        avgBlobDistinctSpecies: number;
        maxBlobDistinctSpecies: number;
        avgEncapsulatedSpecies: number;
        avgNakedSpecies: number;
        avgEncapGene: number;
        encapStrategies: {
          earlyArmorBlobs: number;
          balancedBlobs: number;
          lateComplexBlobs: number;
        };
        emergentRoles: {
          producerBlobs: number;
          consumerBlobs: number;
          mixotrophBlobs: number;
          decomposerBlobs: number;
          halophileBlobs: number;
          producerParticles: number;
          consumerParticles: number;
          mixotrophParticles: number;
          decomposerParticles: number;
          halophileParticles: number;
        };
      }>;
    };
  }
}

function unpackParticlesF32(f32: Float32Array, n: number): number[][] {
  const out: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const b = i * 12;
    out[i] = [
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
    ];
  }
  return out;
}

function unpackVec4F32(f32: Float32Array, n: number): number[][] {
  const out: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const b = i * 4;
    out[i] = [f32[b], f32[b + 1], f32[b + 2], f32[b + 3]];
  }
  return out;
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
  const out = unpackParticlesF32(new Float32Array(staging.getMappedRange()), n);
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
  const out = unpackVec4F32(new Float32Array(staging.getMappedRange()), n);
  staging.unmap();
  staging.destroy();
  return out;
}

async function readUnifiedWorldBuffers(
  count: number,
  includeExtras: boolean,
): Promise<{
  particles: number[][];
  valence: number[][];
  lifeState: number[][];
  bioForce: number[][];
} | null> {
  if (!device || !particleBuffers || !lifeStateBuffers) return null;
  const n = Math.max(0, Math.min(count, particleAmt));
  if (n === 0) {
    return { particles: [], valence: [], lifeState: [], bioForce: [] };
  }

  const enc = device.createCommandEncoder();
  const stg: Record<string, { buf: GPUBuffer; count: number; isVec4: boolean }> = {};
  const queueRead = (name: string, src: GPUBuffer, byteLength: number, isVec4: boolean) => {
    const buf = device!.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    enc.copyBufferToBuffer(src, 0, buf, 0, byteLength);
    stg[name] = { buf, count: n, isVec4 };
  };

  queueRead('particles', particleBuffers[alternate], n * particleStride, false);
  queueRead('lifeState', lifeStateBuffers[alternate], n * valenceStride, true);
  if (includeExtras) {
    if (valenceBuffer) {
      queueRead('valence', valenceBuffer, n * valenceStride, true);
    }
    if (bioForceBuffer) {
      queueRead('bioForce', bioForceBuffer, n * valenceStride, true);
    }
  }

  device.queue.submit([enc.finish()]);
  const keys = Object.keys(stg);
  try {
    await Promise.all(keys.map((k) => stg[k].buf.mapAsync(GPUMapMode.READ)));
    const readArr = (k: string): number[][] => {
      const entry = stg[k];
      if (!entry) return [];
      const f32 = new Float32Array(entry.buf.getMappedRange());
      return entry.isVec4 ? unpackVec4F32(f32, entry.count) : unpackParticlesF32(f32, entry.count);
    };
    return {
      particles: readArr('particles'),
      valence: readArr('valence'),
      lifeState: readArr('lifeState'),
      bioForce: readArr('bioForce'),
    };
  } finally {
    for (const k of keys) {
      try {
        stg[k].buf.unmap();
      } catch {
        // Already unmapped
      }
      stg[k].buf.destroy();
    }
  }
}

async function captureHoverSnapshot(): Promise<HoverSimSnapshot | null> {
  const data = await readUnifiedWorldBuffers(particleAmt, true);
  if (!data) return null;
  return {
    elements,
    particles: data.particles,
    valence: data.valence,
    lifeState: data.lifeState,
    bioForce: data.bioForce,
    speciesTraits,
    capturedAt: performance.now(),
  };
}

async function computeWorldStats() {
  const data = await readUnifiedWorldBuffers(particleAmt, false);
  const chemElementCounts: Record<string, number> = {};
  const bioElementCounts: Record<string, number> = {};
  const emptyRoles = {
    producerBlobs: 0,
    consumerBlobs: 0,
    mixotrophBlobs: 0,
    decomposerBlobs: 0,
    halophileBlobs: 0,
    producerParticles: 0,
    consumerParticles: 0,
    mixotrophParticles: 0,
    decomposerParticles: 0,
    halophileParticles: 0,
  };
  const emptyStrategies = {
    earlyArmorBlobs: 0,
    balancedBlobs: 0,
    lateComplexBlobs: 0,
  };
  if (!data) {
    return {
      total: 0,
      chemCount: 0,
      bioCount: 0,
      cytoplasmCount: 0,
      lipidCount: 0,
      guildCounts: [0, 0, 0, 0] as [number, number, number, number],
      abioticBioViolations: 0,
      lipidAbioticViolations: 0,
      chemElementCounts,
      bioElementCounts,
      avgBioEnergy: 0,
      oceanBioCount: 0,
      landBioCount: 0,
      blobCount: 0,
      multiSpeciesBlobs: 0,
      complexBlobs: 0,
      encapsulatedBlobs: 0,
      encapsulatedSimpleViolations: 0,
      nakedBlobs: 0,
      avgBlobSize: 0,
      maxBlobSize: 0,
      avgBlobDistinctSpecies: 0,
      maxBlobDistinctSpecies: 0,
      avgEncapsulatedSpecies: 0,
      avgNakedSpecies: 0,
      avgEncapGene: 0,
      encapStrategies: emptyStrategies,
      emergentRoles: emptyRoles,
    };
  }

  const total = data.particles.length;
  let chemCount = 0;
  let bioCount = 0;
  let cytoplasmCount = 0;
  let lipidCount = 0;
  let abioticBioViolations = 0;
  let lipidAbioticViolations = 0;
  let bioEnergySum = 0;
  let oceanBioCount = 0;
  let landBioCount = 0;
  const RAD2DEG = 180 / Math.PI;
  const bioIndices: number[] = [];
  const bioMeta = new Array<{
    phase: number;
    bioSpecies: number;
    kinPhase: number;
    encapGene: number;
  }>(total);
  const halogenIndices: number[] = [];

  for (let i = 0; i < total; i++) {
    const p = data.particles[i];
    const ls = data.lifeState[i] ?? [0, 0, 0, 0];
    const meta = unpackLifeMeta(ls);
    bioMeta[i] = meta;
    const cIdx = Math.max(0, Math.min(elements.length - 1, Math.floor(p[6] ?? 0)));
    const el = elements[cIdx];
    const sym = getElementSymbol(el.atomicNumber);
    if (meta.phase !== 0) {
      bioCount++;
      if (meta.phase === 2) {
        lipidCount++;
        const validLipid =
          el.valence > 0 &&
          (el.atomicNumber === 1 || el.atomicNumber === 5 || el.atomicNumber === 6);
        if (!validLipid) lipidAbioticViolations++;
      } else {
        cytoplasmCount++;
      }
      bioIndices.push(i);
      bioEnergySum += ls[0] ?? 0;
      bioElementCounts[sym] = (bioElementCounts[sym] ?? 0) + 1;
      if (!isBiogenicAtomicNumber(el.atomicNumber, el.valence)) {
        abioticBioViolations++;
      }
      if (currentOceanGrid) {
        const z = Math.max(-1, Math.min(1, p[2] ?? 0));
        const latDeg = Math.asin(z) * RAD2DEG;
        const lonDeg = Math.atan2(p[1] ?? 0, p[0] ?? 1) * RAD2DEG;
        const m = sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg)[3];
        if (m >= 0.42) oceanBioCount++;
        else landBioCount++;
      }
    } else {
      chemCount++;
      chemElementCounts[sym] = (chemElementCounts[sym] ?? 0) + 1;
      if (el.atomicNumber === 7 || el.atomicNumber === 13) {
        halogenIndices.push(i);
      }
    }
  }

  // Build spatial grid hash over living particles to assemble connected emergent blobs & protocells
  const R = Math.max(optionParams.worldSize, 0.25);
  const oceanRMax = 1 / Math.max(oceanOptionParams.r, 0.5);
  const connectChord = Math.max(0.02, (oceanRMax * 1.35) / R);
  const connectChord2 = connectChord * connectChord;
  const tightChord2 = Math.pow((oceanRMax * 0.72) / R, 2);

  const keyForPos = (p: number[]) => {
    const gx = Math.floor((p[0] + 1.5) / connectChord);
    const gy = Math.floor((p[1] + 1.5) / connectChord);
    const gz = Math.floor((p[2] + 1.5) / connectChord);
    return `${gx},${gy},${gz}`;
  };

  const grid = new Map<string, number[]>();
  for (const idx of bioIndices) {
    const k = keyForPos(data.particles[idx]);
    let arr = grid.get(k);
    if (!arr) {
      arr = [];
      grid.set(k, arr);
    }
    arr.push(idx);
  }
  const halGrid = new Map<string, number[]>();
  for (const idx of halogenIndices) {
    const k = keyForPos(data.particles[idx]);
    let arr = halGrid.get(k);
    if (!arr) {
      arr = [];
      halGrid.set(k, arr);
    }
    arr.push(idx);
  }

  const visited = new Uint8Array(total);
  const emergentRoles = { ...emptyRoles };
  const encapStrategies = { ...emptyStrategies };
  let blobCount = 0;
  let multiSpeciesBlobs = 0;
  let complexBlobs = 0;
  let encapsulatedBlobs = 0;
  let encapsulatedSimpleViolations = 0;
  let nakedBlobs = 0;
  let blobSizeSum = 0;
  let maxBlobSize = 0;
  let blobSpeciesSum = 0;
  let maxBlobDistinctSpecies = 0;
  let encapSpeciesSum = 0;
  let nakedSpeciesSum = 0;
  let encapGeneSum = 0;

  for (const seedIdx of bioIndices) {
    if (visited[seedIdx]) continue;
    visited[seedIdx] = 1;
    const comp = [seedIdx];
    let head = 0;
    const nearbyHalogenSet = new Set<number>();

    while (head < comp.length && comp.length < 640) {
      const u = comp[head++];
      const pu = data.particles[u];
      const mu = bioMeta[u];
      const gx = Math.floor((pu[0] + 1.5) / connectChord);
      const gy = Math.floor((pu[1] + 1.5) / connectChord);
      const gz = Math.floor((pu[2] + 1.5) / connectChord);

      for (let ox = -1; ox <= 1; ox++) {
        for (let oy = -1; oy <= 1; oy++) {
          for (let oz = -1; oz <= 1; oz++) {
            const key = `${gx + ox},${gy + oy},${gz + oz}`;
            const bucket = grid.get(key);
            if (bucket) {
              for (const v of bucket) {
                if (visited[v]) continue;
                const pv = data.particles[v];
                const dx = pv[0] - pu[0];
                const dy = pv[1] - pu[1];
                const dz = pv[2] - pu[2];
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 >= connectChord2) continue;
                const mv = bioMeta[v];
                if (kinAngularDist(mu.kinPhase, mv.kinPhase) < 0.3 || d2 < tightChord2) {
                  visited[v] = 1;
                  comp.push(v);
                }
              }
            }
            const hBucket = halGrid.get(key);
            if (hBucket) {
              for (const hi of hBucket) {
                const ph = data.particles[hi];
                const dx = ph[0] - pu[0];
                const dy = ph[1] - pu[1];
                const dz = ph[2] - pu[2];
                if (dx * dx + dy * dy + dz * dz < connectChord2) {
                  nearbyHalogenSet.add(hi);
                }
              }
            }
          }
        }
      }
    }

    if (comp.length < 3) continue;
    const summary = summarizeAndClassifyBlob(
      comp,
      data.particles,
      data.lifeState,
      elements,
      speciesTraits,
      nearbyHalogenSet.size,
    );
    if (summary.lipidCount >= 6 && summary.cytoplasmCount >= 1 && summary.distinctSpecies <= 2) {
      encapsulatedSimpleViolations++;
    }
    if (summary.cytoplasmCount < 2) continue;

    blobCount++;
    blobSizeSum += comp.length;
    if (comp.length > maxBlobSize) maxBlobSize = comp.length;
    blobSpeciesSum += summary.distinctSpecies;
    if (summary.distinctSpecies > maxBlobDistinctSpecies) {
      maxBlobDistinctSpecies = summary.distinctSpecies;
    }
    encapGeneSum += summary.avgEncapGene;
    if (summary.distinctSpecies >= 2) multiSpeciesBlobs++;
    if (summary.distinctSpecies >= 4) complexBlobs++;
    if (summary.encapsulated) {
      encapsulatedBlobs++;
      encapSpeciesSum += summary.distinctSpecies;
    } else {
      nakedBlobs++;
      nakedSpeciesSum += summary.distinctSpecies;
    }
    if (summary.encapStrategy === 'early-armor') encapStrategies.earlyArmorBlobs++;
    else if (summary.encapStrategy === 'balanced') encapStrategies.balancedBlobs++;
    else encapStrategies.lateComplexBlobs++;

    if (summary.role === 'producer') {
      emergentRoles.producerBlobs++;
      emergentRoles.producerParticles += comp.length;
    } else if (summary.role === 'consumer') {
      emergentRoles.consumerBlobs++;
      emergentRoles.consumerParticles += comp.length;
    } else if (summary.role === 'mixotroph') {
      emergentRoles.mixotrophBlobs++;
      emergentRoles.mixotrophParticles += comp.length;
    } else if (summary.role === 'decomposer') {
      emergentRoles.decomposerBlobs++;
      emergentRoles.decomposerParticles += comp.length;
    } else {
      emergentRoles.halophileBlobs++;
      emergentRoles.halophileParticles += comp.length;
    }
  }

  return {
    total,
    chemCount,
    bioCount,
    cytoplasmCount,
    lipidCount,
    guildCounts: [
      emergentRoles.producerParticles,
      emergentRoles.consumerParticles,
      emergentRoles.halophileParticles,
      emergentRoles.decomposerParticles,
    ] as [number, number, number, number],
    abioticBioViolations,
    lipidAbioticViolations,
    chemElementCounts,
    bioElementCounts,
    avgBioEnergy: bioCount > 0 ? bioEnergySum / bioCount : 0,
    oceanBioCount,
    landBioCount,
    blobCount,
    multiSpeciesBlobs,
    complexBlobs,
    encapsulatedBlobs,
    encapsulatedSimpleViolations,
    nakedBlobs,
    avgBlobSize: blobCount > 0 ? blobSizeSum / blobCount : 0,
    maxBlobSize,
    avgBlobDistinctSpecies: blobCount > 0 ? blobSpeciesSum / blobCount : 0,
    maxBlobDistinctSpecies,
    avgEncapsulatedSpecies: encapsulatedBlobs > 0 ? encapSpeciesSum / encapsulatedBlobs : 0,
    avgNakedSpecies: nakedBlobs > 0 ? nakedSpeciesSum / nakedBlobs : 0,
    avgEncapGene: blobCount > 0 ? encapGeneSum / blobCount : 7,
    encapStrategies,
    emergentRoles,
  };
}

function syncBioUniformState() {
  if (!device || !bioUniformBuffer) return;
  const active = earthParams.bioLink && earthParams.bioRate > 0;
  setBioLink(
    device,
    bioUniformData,
    bioUniformBuffer,
    active,
    frameCount,
    computeBioSeedHash(params.seed, oceanSeed),
  );
  if (!active && bioWasActive && bioForceBuffer) {
    device.queue.writeBuffer(bioForceBuffer, 0, new Float32Array(particleAmt * 4));
  }
  bioWasActive = active;
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
  startWorld();
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
  syncUrlStateDebounced();
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

const clampOpacity = (val: boolean | number) =>
  typeof val === 'number' ? Math.max(0, Math.min(1, val)) : val ? 0.2 : 0;

function syncStreamCanvasOpacity() {
  oceanCanvas.style.opacity = String(earthParams.oceanLines);
  windCanvas.style.opacity = String(earthParams.windLines);
}
syncStreamCanvasOpacity();

const tooltipManager = new TooltipManager({
  isEnabled: () => earthParams.tooltips,
  getProj: () => d3Proj,
  getViewLayer: () => earthParams.viewLayer,
  isBioLinked: () => earthParams.bioLink && earthParams.bioRate > 0,
  getAirRMax: () => 1 / Math.max(optionParams.r, 0.5),
  getOceanRMax: () => 1 / Math.max(oceanOptionParams.r, 0.5),
  getWorldSize: () => Math.max(optionParams.worldSize, 0.25),
  getBeta: () => optionParams.beta,
  getTemperature: () => optionParams.temperature,
  getScale: () => viewState.scale,
  requestSnapshot: captureHoverSnapshot,
});

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
  getOceanParticleCount: () => particleAmt,
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
  getTooltips: () => earthParams.tooltips,
  setTooltips: (on: boolean) => {
    earthParams.tooltips = on;
    tooltipsBinding.refresh();
    tooltipManager.syncEnabledState();
  },
  getOpacities: () => ({
    windLines: earthParams.windLines,
    windColors: earthParams.windColors,
    chemOpacity: earthParams.chemOpacity,
    oceanLines: earthParams.oceanLines,
    oceanColors: earthParams.oceanColors,
    lifeOpacity: earthParams.lifeOpacity,
  }),
  setChemOpacity: (val: number) => {
    earthParams.chemOpacity = clampOpacity(val);
    chemOpacityBinding.refresh();
  },
  setLifeOpacity: (val: number) => {
    earthParams.lifeOpacity = clampOpacity(val);
    lifeOpacityBinding.refresh();
  },
  setOceanColors: (val: boolean | number) => {
    earthParams.oceanColors = clampOpacity(val);
    oceanColorsBinding.refresh();
  },
  setOceanLines: (val: boolean | number) => {
    earthParams.oceanLines = clampOpacity(val);
    oceanLinesBinding.refresh();
    syncStreamCanvasOpacity();
    if (earthParams.oceanLines <= 0.001 && oceanCtx) {
      oceanStreamRenderer.clear(oceanCtx, viewState.width, viewState.height);
    }
  },
  setWindColors: (val: boolean | number) => {
    earthParams.windColors = clampOpacity(val);
    windColorsBinding.refresh();
  },
  setWindLines: (val: boolean | number) => {
    earthParams.windLines = clampOpacity(val);
    windLinesBinding.refresh();
    syncStreamCanvasOpacity();
    if (earthParams.windLines <= 0.001 && windCtx) {
      windStreamRenderer.clear(windCtx, viewState.width, viewState.height);
    }
  },
  refreshTooltipSnapshot: async () => {
    const snap = await captureHoverSnapshot();
    if (snap) {
      tooltipManager.setSnapshot(snap);
    }
    return snap;
  },
  inspectReactionAt: (clientX: number, clientY: number) =>
    tooltipManager.inspectAt(clientX, clientY),
  getTooltipLens: () => tooltipManager.getLensState(),
  exportConfig: exportCurrentConfig,
  getShareUrl: getCurrentShareUrl,
  importConfig: importConfiguration,
  resetDefaultConfig: resetToFactoryDefaults,
  randomizeOceanMatrix: (seed?: string) => {
    oceanSeed = seed ?? generateRandomSeed();
    oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);
    writeWorldRuleAndColourBuffers();
    syncBioUniformState();
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
    readParticleBuffer(particleBuffers, alternate, particleAmt, count),
  readValence: (count: number = 16) => readVec4Buffer(valenceBuffer, particleAmt, count),
  readOceanState: (count: number = 16) =>
    readVec4Buffer(lifeStateBuffers?.[alternate], particleAmt, count),
  readAirBioReact: (count: number = 16) =>
    readVec4Buffer(bioForceBuffer, particleAmt, count),
  readLifeState: (count: number = 16) =>
    readVec4Buffer(lifeStateBuffers?.[alternate], particleAmt, count),
  readBioForce: (count: number = 16) =>
    readVec4Buffer(bioForceBuffer, particleAmt, count),
  readWorldState: async (count: number = 16) =>
    (await readUnifiedWorldBuffers(count, true)) ?? {
      particles: [],
      valence: [],
      lifeState: [],
      bioForce: [],
    },
  getWorldStats: computeWorldStats,
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
        {
          arrayStride: valenceStride,
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 5, offset: 0, format: 'float32x4' },
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

  lensTrackUniformBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'lensTrackUniformBuffer',
  });
  lensGpuStateBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'lensGpuStateBuffer',
  });
  device.queue.writeBuffer(
    lensGpuStateBuffer,
    0,
    new Float32Array([0, 0, 0, 1, 0, 0, 0, 0]),
  );
  lensReadbackBuffers = [
    device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'lensReadbackBuffer0',
    }),
    device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'lensReadbackBuffer1',
    }),
  ];

  const lensTrackModule = device.createShaderModule({
    code: `${structsShader}\n${projectionShader}\n${lensTrackShader}`,
  });
  lensTrackPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: lensTrackModule,
      entryPoint: 'main',
    },
  });

  const lensOverlayModule = device.createShaderModule({
    code: `${structsShader}\n${lensOverlayShader}`,
  });
  lensOverlayPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: lensOverlayModule,
      entryPoint: 'vertex',
    },
    fragment: {
      module: lensOverlayModule,
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
    primitive: { topology: 'triangle-list' },
  });
  lensOverlayBindGroup = device.createBindGroup({
    layout: lensOverlayPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: lensTrackUniformBuffer } },
    ],
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

  startWorld();
  bindOptions(
    device,
    simData,
    simBuffer,
    oceanSimData,
    oceanSimBuffer,
    syncBioUniformState,
  );
  bindConfigPersistence(() => {
    if (!suppressPaneEvents) scheduleSaveConfig();
  });
  configPersistenceReady = true;
  syncUrlStateNow();
  loadWindLevel(earthParams.windLevel);
})();

function tick(commandEncoder: GPUCommandEncoder) {
  if (!device || !particleBuffers) return;
  engines[engine].tick(device, commandEncoder, alternate, particleAmt);
  alternate = (alternate + 1) % 2;
}

function rebuildLensTrackBindGroups() {
  if (
    !device ||
    !lensTrackPipeline ||
    !uniformBuffer ||
    !watermanFwdTexture ||
    !lensTrackUniformBuffer ||
    !lensGpuStateBuffer ||
    !particleBuffers
  ) {
    return;
  }

  const fwdView = watermanFwdTexture.createView();
  for (let i = 0; i < 2; i++) {
    lensTrackBindGroups[i] = device.createBindGroup({
      layout: lensTrackPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: fwdView },
        { binding: 2, resource: { buffer: lensTrackUniformBuffer } },
        { binding: 3, resource: { buffer: particleBuffers[i] } },
        { binding: 4, resource: { buffer: lensGpuStateBuffer } },
      ],
    });
  }
}

function updateGpuLens(commandEncoder: GPUCommandEncoder, frameDtMs: number) {
  pendingLensReadbackIdx = null;
  if (
    !device ||
    !uniformBuffer ||
    !lensTrackPipeline ||
    !lensTrackUniformBuffer ||
    !lensGpuStateBuffer
  ) {
    return;
  }

  const bg = lensTrackBindGroups[alternate];
  if (!bg) return;

  const cfg = tooltipManager.getLensGpuConfig();
  const snapNow =
    cfg.generation !== prevGpuLensGen || viewSignature !== prevGpuLensViewSig;
  prevGpuLensGen = cfg.generation;
  prevGpuLensViewSig = viewSignature;

  const dtSec = Math.min(0.08, Math.max(0.001, frameDtMs / 1000));
  const smoothAlpha = Math.min(0.45, Math.max(0.14, 1 - Math.exp(-dtSec * 14.0)));

  lensTrackUniformData[0] = cfg.kind;
  lensTrackUniformData[1] = cfg.idx;
  lensTrackUniformData[2] = cfg.radius;
  lensTrackUniformData[3] = cfg.mag;
  lensTrackUniformData[4] = snapNow ? 1 : 0;
  lensTrackUniformData[5] = smoothAlpha;
  lensTrackUniformData[6] = 0;
  lensTrackUniformData[7] = 0;
  lensTrackUniformData[8] = cfg.color[0];
  lensTrackUniformData[9] = cfg.color[1];
  lensTrackUniformData[10] = cfg.color[2];
  lensTrackUniformData[11] = cfg.color[3];
  device.queue.writeBuffer(lensTrackUniformBuffer, 0, lensTrackUniformData);

  const cPass = commandEncoder.beginComputePass();
  cPass.setPipeline(lensTrackPipeline);
  cPass.setBindGroup(0, bg);
  cPass.dispatchWorkgroups(1);
  cPass.end();

  commandEncoder.copyBufferToBuffer(lensGpuStateBuffer, 0, uniformBuffer, 32 * 4, 16);

  if (cfg.kind > 0 && lensReadbackBuffers) {
    const slot = lensReadbackToggle;
    if (!lensReadbackInFlight[slot]) {
      commandEncoder.copyBufferToBuffer(
        lensGpuStateBuffer,
        0,
        lensReadbackBuffers[slot],
        0,
        16,
      );
      pendingLensReadbackIdx = slot;
      lensReadbackToggle = (slot + 1) % 2;
    }
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
  ) {
    return;
  }

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

  const showChem = earthParams.viewLayer !== 'ocean' && earthParams.chemOpacity > 0.001;
  const showLife = earthParams.viewLayer !== 'air' && earthParams.lifeOpacity > 0.001;
  if (
    (showChem || showLife) &&
    renderBindGroup &&
    particleBuffers &&
    valenceBuffer &&
    lifeStateBuffers
  ) {
    passEncoder.setPipeline(renderPipeline);
    passEncoder.setVertexBuffer(0, particleBuffers[alternate]);
    passEncoder.setVertexBuffer(1, valenceBuffer);
    passEncoder.setVertexBuffer(2, lifeStateBuffers[alternate]);
    passEncoder.setBindGroup(0, renderBindGroup);
    passEncoder.draw(6, particleAmt, 0, 0);
  }

  if (lensOverlayPipeline && lensOverlayBindGroup) {
    passEncoder.setPipeline(lensOverlayPipeline);
    passEncoder.setBindGroup(0, lensOverlayBindGroup);
    passEncoder.draw(3, 1, 0, 0);
  }

  passEncoder.end();
  resolveTimestamp(commandEncoder, 'render');
}

function packLifeMetaJs(
  phase: number,
  bioSp: number,
  kinAngle: number,
  encapGene: number = 7,
): number {
  if (phase === 0) return 0;
  const twoPi = Math.PI * 2;
  let a = kinAngle % twoPi;
  if (a < 0) a += twoPi;
  const k = Math.floor((a / twoPi) * 512) & 511;
  const g = Math.max(0, Math.min(15, Math.floor(encapGene)));
  return (
    (phase & 3) |
    ((Math.min(255, Math.max(0, bioSp)) & 255) << 2) |
    (k << 10) |
    (g << 19)
  ) >>> 0;
}

function startWorld() {
  if (
    !device ||
    !uniformBuffer ||
    !renderPipeline ||
    !simBuffer ||
    !oceanSimBuffer ||
    !bioUniformBuffer ||
    !windTexture ||
    !oceanTexture ||
    !watermanFwdTexture
  ) {
    return;
  }

  particleBuffers?.[0].destroy();
  particleBuffers?.[1].destroy();
  lifeStateBuffers?.[0].destroy();
  lifeStateBuffers?.[1].destroy();
  valenceBuffer?.destroy();
  bioForceBuffer?.destroy();
  worldRulesBuffer?.destroy();
  worldColoursBuffer?.destroy();

  refreshChemistryDefinition();
  oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);
  speciesTraits = computeSpeciesTraitsFromMatrix(oceanMatrixData, oceanColourAmt).table;

  const bufferSize = particleAmt * particleStride;
  particleBuffers = [
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: 'worldParticles0',
    }),
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: 'worldParticles1',
    }),
  ];

  lifeStateBuffers = [
    device.createBuffer({
      size: particleAmt * valenceStride,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: 'lifeState0',
    }),
    device.createBuffer({
      size: particleAmt * valenceStride,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: 'lifeState1',
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
  bioForceBuffer = device.createBuffer({
    size: particleAmt * valenceStride,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: 'bioForceBuffer',
  });

  worldRulesBuffer = device.createBuffer({
    size: Math.max((chemistryData.length + oceanMatrixData.length + oceanColourAmt * 4) * 4, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'worldRulesBuffer',
  });
  worldColoursBuffer = device.createBuffer({
    size: Math.max((colourAmt + oceanColourAmt) * 3 * 4, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'worldColoursBuffer',
  });

  alternate = 0;
  frameCount = 0;
  const rng = createSeededRng(`${params.seed}:world`);
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

  const elemChoice = new Uint16Array(particleAmt);
  const biogenicPool: number[] = [];
  const abioticPool: number[] = [];
  for (let pi = 0; pi < particleAmt; pi++) {
    const cIdx = pickElement();
    elemChoice[pi] = cIdx;
    const el = elements[cIdx];
    if (el && isBiogenicAtomicNumber(el.atomicNumber, el.valence)) {
      biogenicPool.push(pi);
    } else {
      abioticPool.push(pi);
    }
  }
  for (let i = biogenicPool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = biogenicPool[i];
    biogenicPool[i] = biogenicPool[j];
    biogenicPool[j] = tmp;
  }

  const particleData = new Float32Array(bufferSize / 4);
  const lifeData = new Float32Array(particleAmt * 4);
  const worldR = Math.max(optionParams.worldSize, 0.25);
  const nBio = Math.max(1, oceanColourAmt);
  const cohSorted = Array.from({ length: nBio }, (_, i) => i).sort(
    (a, b) => (speciesTraits[b]?.[0] ?? 0) - (speciesTraits[a]?.[0] ?? 0),
  );
  const asymSorted = Array.from({ length: nBio }, (_, i) => i).sort(
    (a, b) => (speciesTraits[b]?.[3] ?? 0) - (speciesTraits[a]?.[3] ?? 0),
  );

  const pickHeterogeneousSpeciesSet = (isGliderCommunity: boolean): number[] => {
    if (nBio <= 1) return [0];
    if (nBio < 6) {
      return Array.from({ length: nBio }, (_, i) => i);
    }
    const desired = Math.min(nBio, 6 + Math.floor(rng() * 5));
    const poolLen = Math.max(2, Math.ceil(nBio * 0.55));
    const s0 = isGliderCommunity
      ? asymSorted[Math.floor(rng() * poolLen)]
      : cohSorted[Math.floor(rng() * poolLen)];
    const selected = [s0];
    const chosen = new Set<number>([s0]);

    while (selected.length < desired) {
      const cur = selected[selected.length - 1];
      const candidates = Array.from({ length: nBio }, (_, j) => j)
        .filter((j) => !chosen.has(j))
        .map((j) => {
          const aCurJ = oceanMatrixData[cur * nBio + j] ?? 0;
          const aJCur = oceanMatrixData[j * nBio + cur] ?? 0;
          const aJS0 = oceanMatrixData[j * nBio + s0] ?? 0;
          const aS0J = oceanMatrixData[s0 * nBio + j] ?? 0;
          const score = isGliderCommunity
            ? Math.abs(aCurJ - aJCur) + 0.45 * aCurJ + 0.35 * aJS0
            : aCurJ + aJCur + 0.5 * (aS0J + aJS0) + 0.25 * Math.abs(aCurJ - aJCur);
          return { j, score };
        })
        .sort((a, b) => b.score - a.score);
      if (candidates.length === 0) break;
      const pick =
        candidates.length > 1 && rng() < 0.28 ? candidates[1].j : candidates[0].j;
      chosen.add(pick);
      selected.push(pick);
    }
    return selected;
  };

  const targetBio = Math.min(
    Math.round(particleAmt * 0.28),
    Math.floor(biogenicPool.length * 0.72),
  );
  const numBlobs =
    targetBio >= 24 ? Math.max(12, Math.min(110, Math.round(targetBio / 88))) : 0;
  const DEG2RAD = Math.PI / 180;

  const sampleOceanCenter = (): [number, number, number] => {
    let lonDeg = (rng() - 0.5) * 360;
    let latDeg = Math.asin(rng() * 2 - 1) * (180 / Math.PI);
    if (currentOceanGrid) {
      for (let t = 0; t < 60; t++) {
        const candLon = (rng() - 0.5) * 360;
        const candLat = Math.asin(rng() * 2 - 1) * (180 / Math.PI);
        if (sampleOceanLonLat(currentOceanGrid, candLon, candLat)[3] >= 0.58) {
          lonDeg = candLon;
          latDeg = candLat;
          break;
        }
      }
    }
    const lonR = lonDeg * DEG2RAD;
    const latR = latDeg * DEG2RAD;
    const cLat = Math.cos(latR);
    return [cLat * Math.cos(lonR), cLat * Math.sin(lonR), Math.sin(latR)];
  };

  let bioCursor = 0;
  if (numBlobs > 0) {
    for (let bi = 0; bi < numBlobs; bi++) {
      const endCursor =
        bi === numBlobs - 1
          ? targetBio
          : Math.min(targetBio, Math.round(((bi + 1) * targetBio) / numBlobs));
      const [cx, cy, cz] = sampleOceanCenter();
      const [e1x, e1y, e1z] = unitTangentFromSphere(cx, cy, cz, rng() * Math.PI * 2);
      const e2x = cy * e1z - cz * e1y;
      const e2y = cz * e1x - cx * e1z;
      const e2z = cx * e1y - cy * e1x;
      const kinPhase = rng() * Math.PI * 2;
      const blobSpecies = pickHeterogeneousSpeciesSet(bi % 2 === 1);
      const patchGene =
        bi % 3 === 0
          ? 1 + Math.floor(rng() * 4)
          : bi % 3 === 1
            ? 5 + Math.floor(rng() * 6)
            : 11 + Math.floor(rng() * 4);

      while (bioCursor < endCursor) {
        const pi = biogenicPool[bioCursor++];
        const angR = (0.012 + Math.sqrt(rng()) * 0.044) / worldR;
        const th = rng() * Math.PI * 2;
        const ct = Math.cos(th);
        const st = Math.sin(th);
        const dx = ct * e1x + st * e2x;
        const dy = ct * e1y + st * e2y;
        const dz = ct * e1z + st * e2z;
        const ca = Math.cos(angR);
        const sa = Math.sin(angR);
        let nx = ca * cx + sa * dx;
        let ny = ca * cy + sa * dy;
        let nz = ca * cz + sa * dz;
        const invL = 1 / (Math.hypot(nx, ny, nz) || 1);
        nx *= invL;
        ny *= invL;
        nz *= invL;
        const [ux, uy, uz] = unitTangentFromSphere(nx, ny, nz, rng() * Math.PI * 2);

        const b = pi * 12;
        particleData[b + 0] = nx;
        particleData[b + 1] = ny;
        particleData[b + 2] = nz;
        particleData[b + 3] = elemChoice[pi];
        particleData[b + 8] = ux;
        particleData[b + 9] = uy;
        particleData[b + 10] = uz;

        const sp = blobSpecies[Math.floor(rng() * blobSpecies.length)];
        const lBase = pi * 4;
        lifeData[lBase + 0] = 0.72 + 0.23 * rng();
        lifeData[lBase + 1] = 0.05;
        lifeData[lBase + 2] = 0.0;
        lifeData[lBase + 3] = packLifeMetaJs(1, sp, kinPhase, patchGene);
      }
    }
  }

  const placeFreeAtom = (pi: number) => {
    const z = rng() * 2 - 1;
    const lon = rng() * Math.PI * 2;
    const rxy = Math.sqrt(Math.max(0, 1 - z * z));
    const nx = rxy * Math.cos(lon);
    const ny = rxy * Math.sin(lon);
    const nz = z;
    const [ux, uy, uz] = unitTangentFromSphere(nx, ny, nz, rng() * Math.PI * 2);
    const b = pi * 12;
    particleData[b + 0] = nx;
    particleData[b + 1] = ny;
    particleData[b + 2] = nz;
    particleData[b + 3] = elemChoice[pi];
    particleData[b + 8] = ux;
    particleData[b + 9] = uy;
    particleData[b + 10] = uz;
  };

  while (bioCursor < biogenicPool.length) {
    placeFreeAtom(biogenicPool[bioCursor++]);
  }
  for (const pi of abioticPool) {
    placeFreeAtom(pi);
  }

  device.queue.writeBuffer(particleBuffers[0], 0, particleData.buffer);
  device.queue.writeBuffer(lifeStateBuffers[0], 0, lifeData.buffer);
  device.queue.writeBuffer(lifeStateBuffers[1], 0, new Float32Array(particleAmt * 4));
  device.queue.writeBuffer(valenceBuffer, 0, new Float32Array(particleAmt * 4));
  device.queue.writeBuffer(bioForceBuffer, 0, new Float32Array(particleAmt * 4));

  writeWorldRuleAndColourBuffers();
  setSim(device, simData, simBuffer, frameCount);
  setOceanSim(device, oceanSimData, oceanSimBuffer);
  syncBioUniformState();

  const refs: UnifiedEngineRefs = {
    uniformBuffer,
    simBuffer,
    oceanSimBuffer,
    bioUniformBuffer,
    rulesMatrixBuffer: worldRulesBuffer,
    particleBuffers,
    lifeStateBuffers,
    valenceBuffer,
    bioForceBuffer,
    windTexView: windTexture.createView(),
    oceanTexView: oceanTexture.createView(),
    particleAmt,
    cellAmt: params.cells,
  };
  for (const engineName in engines) {
    engines[engineName].start(device, refs);
  }

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: watermanFwdTexture.createView() },
      { binding: 2, resource: { buffer: worldColoursBuffer } },
      { binding: 3, resource: { buffer: worldRulesBuffer } },
    ],
  });
  rebuildLensTrackBindGroups();
}

let lastTime = 0;
const deltaVs: number[] = [];

function writeUniforms() {
  if (!device || !uniformBuffer) return;
  const spec = WIND_LEVELS[earthParams.windLevel];
  const R = Math.max(optionParams.worldSize, 0.25);
  const showAir = earthParams.viewLayer !== 'ocean';
  const showOcean = earthParams.viewLayer !== 'air';

  uniformData[0] = viewState.width;
  uniformData[1] = viewState.height;
  uniformData[2] = viewState.width / Math.max(viewState.height, 1);
  uniformData[3] = ((1 / optionParams.r) / R) * optionParams.beta * 0.39;

  uniformData[4] = mouseSphere.x;
  uniformData[5] = mouseSphere.y;
  uniformData[6] = mouseSphere.z;
  uniformData[7] = mouseSphere.mode;

  uniformData[8] = viewState.scale;
  uniformData[9] = viewState.pan[0];
  uniformData[10] = viewState.pan[1];
  uniformData[11] = PROJECTIONS[viewState.projectionId].shaderType;

  uniformData[12] = optionParams.showOrbitals ? 1 : 0;
  uniformData[13] = showAir ? earthParams.windColors : 0;
  uniformData[14] = spec.scaleMax;
  uniformData[15] = spec.streamMax;

  const rotCols = buildD3RotationMatrixColMajor(viewState.rotate);
  uniformData.set(rotCols, 16);

  uniformData[28] = showOcean ? earthParams.oceanColors : 0;
  uniformData[29] = OCEAN_CURRENTS_SPEC.scaleMax;
  uniformData[30] = earthParams.oceanStrength;
  uniformData[31] = showAir ? earthParams.chemOpacity : 0;

  const lens = tooltipManager.getLensState();
  uniformData[32] = lens.x;
  uniformData[33] = lens.y;
  uniformData[34] = lens.active ? lens.radius : 0;
  uniformData[35] = lens.active ? lens.mag : 1;

  const oceanRMax = 1 / Math.max(oceanOptionParams.r, 0.5);
  uniformData[36] = (oceanRMax / R) * Math.max(oceanOptionParams.beta, 0.25) * 0.42;
  uniformData[37] = showOcean ? earthParams.lifeOpacity : 0;
  uniformData[38] = colourAmt;
  uniformData[39] = oceanColourAmt;

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);
}

function update() {
  requestAnimationFrame(update);
  if (!device || !context || !bgContext) return;

  const start = performance.now();
  const frameDtMs = lastTime > 0 ? start - lastTime : 16.67;
  deltaVs.push(frameDtMs);
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

  syncStreamCanvasOpacity();

  if (oceanCtx && currentOceanGrid) {
    const oceanTint: [number, number, number] =
      showAir && earthParams.windLines > 0.01 ? [115, 235, 255] : [235, 250, 255];
    oceanStreamRenderer.stepAndDraw(
      oceanCtx,
      viewState.width,
      viewState.height,
      d3Proj,
      oceanGridToSampler(currentOceanGrid, oceanTint),
      viewSignature,
      showOcean && earthParams.oceanLines > 0.001,
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
      showAir && earthParams.windLines > 0.001,
    );
  }

  tooltipManager.tick();

  frameCount = (frameCount + 1) % 1000000;
  frameBuf[0] = frameCount;
  if (simBuffer) {
    simData[18] = frameCount;
    device.queue.writeBuffer(simBuffer, 18 * 4, frameBuf);
  }
  if (bioUniformBuffer) {
    bioUniformData[9] = frameCount;
    device.queue.writeBuffer(bioUniformBuffer, 9 * 4, frameBuf);
  }

  writeUniforms();

  const commandEncoder = device.createCommandEncoder();
  tick(commandEncoder);
  updateGpuLens(commandEncoder, frameDtMs);
  render(commandEncoder);

  const commands = commandEncoder.finish();
  device.queue.submit([commands]);

  if (pendingLensReadbackIdx !== null && lensReadbackBuffers) {
    const slot = pendingLensReadbackIdx;
    const buf = lensReadbackBuffers[slot];
    const genSnapshot = prevGpuLensGen;
    lensReadbackInFlight[slot] = true;
    buf
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const f32 = new Float32Array(buf.getMappedRange());
        const cx = f32[0] ?? 0;
        const cy = f32[1] ?? 0;
        const r = f32[2] ?? 0;
        buf.unmap();
        if (tooltipManager.getLensGpuConfig().generation === genSnapshot) {
          tooltipManager.syncGpuLensPos(cx, cy, r > 1.0);
        }
      })
      .catch(() => {
        try {
          buf.unmap();
        } catch {
          // Ignore already unmapped readback buffer
        }
      })
      .finally(() => {
        lensReadbackInFlight[slot] = false;
      });
  }

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

  startWorld();
  saveConfigNow();
});

randomizeBtn.on('click', () => {
  if (!device || !worldRulesBuffer) return;
  applySeed(generateRandomSeed(), true);
  saveConfigNow();
});

const configModal = document.getElementById('configModal') as HTMLElement | null;
const configTextarea = document.getElementById('configTextarea') as HTMLTextAreaElement | null;
const configModalStatus = document.getElementById('configModalStatus') as HTMLElement | null;
const configCloseBtn = document.getElementById('configCloseBtn') as HTMLButtonElement | null;
const configLinkBtn = document.getElementById('configLinkBtn') as HTMLButtonElement | null;
const configCopyBtn = document.getElementById('configCopyBtn') as HTMLButtonElement | null;
const configApplyBtn = document.getElementById('configApplyBtn') as HTMLButtonElement | null;
const configResetBtn = document.getElementById('configResetBtn') as HTMLButtonElement | null;

let copyBtnResetTimer: number | undefined;

function setModalStatus(msg: string, isErr = false) {
  if (!configModalStatus) return;
  configModalStatus.textContent = msg;
  configModalStatus.classList.toggle('err', isErr);
}

function openConfigModal(initialMsg = '') {
  if (!configModal || !configTextarea) return;
  tooltipManager.hide();
  configTextarea.value = JSON.stringify(exportCurrentConfig(), null, 2);
  setModalStatus(initialMsg, false);
  configModal.hidden = false;
  if (window.matchMedia('(pointer: coarse)').matches) {
    configTextarea.scrollTop = 0;
  } else {
    configTextarea.focus();
    configTextarea.select();
  }
}

function closeConfigModal() {
  if (configModal) configModal.hidden = true;
}

async function copyConfigTextWithFallback(text: string, restoreTextarea: boolean = false): Promise<boolean> {
  let syncCopied = false;
  if (configTextarea) {
    const prevVal = configTextarea.value;
    if (prevVal !== text) configTextarea.value = text;
    configTextarea.focus();
    configTextarea.select();
    configTextarea.setSelectionRange(0, configTextarea.value.length);
    try {
      syncCopied = Boolean(document.execCommand?.('copy'));
    } catch {
      syncCopied = false;
    }
    if (restoreTextarea && prevVal !== text) {
      configTextarea.value = prevVal;
    }
  }
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      return syncCopied;
    }
  }
  return syncCopied;
}

copyConfigBtn.on('click', async () => {
  const json = JSON.stringify(exportCurrentConfig(), null, 2);
  let copied = false;
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(json);
      copied = true;
    } catch {
      copied = false;
    }
  }
  if (copied) {
    copyConfigBtn.title = 'Copied ✓';
    if (copyBtnResetTimer !== undefined) window.clearTimeout(copyBtnResetTimer);
    copyBtnResetTimer = window.setTimeout(() => {
      copyConfigBtn.title = 'Copy Config';
    }, 1500);
  } else {
    openConfigModal('Clipboard permission unavailable — copy the selected JSON below.');
  }
});

pasteConfigBtn.on('click', () => {
  openConfigModal();
});

configCloseBtn?.addEventListener('click', closeConfigModal);
configModal?.addEventListener('click', (ev) => {
  if (ev.target === configModal) closeConfigModal();
});
window.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && configModal && !configModal.hidden) {
    closeConfigModal();
  }
});

configLinkBtn?.addEventListener('click', async () => {
  syncUrlStateNow();
  const url = getCurrentShareUrl();
  const copied = await copyConfigTextWithFallback(url, true);
  if (copied) {
    setModalStatus('Copied shareable world URL to clipboard.');
  } else if (configTextarea) {
    configTextarea.value = url;
    configTextarea.select();
    setModalStatus('Tap & hold URL above to Copy.');
  }
});

configCopyBtn?.addEventListener('click', async () => {
  if (!configTextarea) return;
  const text = configTextarea.value.trim() || JSON.stringify(exportCurrentConfig(), null, 2);
  const copied = await copyConfigTextWithFallback(text);
  if (copied) {
    setModalStatus('Copied JSON configuration to clipboard.');
  } else {
    configTextarea.select();
    setModalStatus('Tap & hold selected JSON to Copy.');
  }
});

configApplyBtn?.addEventListener('click', async () => {
  if (!configTextarea) return;
  setModalStatus('Applying configuration…');
  try {
    await importConfiguration(configTextarea.value);
    setModalStatus('Configuration applied & cached cleanly.');
  } catch (err) {
    setModalStatus(err instanceof Error ? err.message : 'Invalid configuration.', true);
  }
});

configResetBtn?.addEventListener('click', async () => {
  setModalStatus('Restoring factory defaults…');
  await resetToFactoryDefaults();
  if (configTextarea) {
    configTextarea.value = JSON.stringify(exportCurrentConfig(), null, 2);
  }
  setModalStatus('Cleared saved cache & restored factory defaults.');
});

seedInput.on('change', () => {
  if (suppressPaneEvents || suppressSeedChange || !device || !worldRulesBuffer) return;
  applySeed(params.seed, false);
  saveConfigNow();
});

newOceanSimBtn.on('click', () => {
  oceanColourAmt = Math.max(1, Math.min(200, Math.floor(oceanOptionParams.colours)));
  startWorld();
  saveConfigNow();
});

randomizeOceanRulesBtn.on('click', () => {
  oceanColourAmt = Math.max(1, Math.min(200, Math.floor(oceanOptionParams.colours)));
  oceanSeed = generateRandomSeed();
  if (!worldRulesBuffer || oceanMatrixData.length !== oceanColourAmt * oceanColourAmt) {
    startWorld();
  } else {
    oceanMatrixData = buildRandomOceanMatrix(oceanColourAmt, oceanSeed);
    writeWorldRuleAndColourBuffers();
    syncBioUniformState();
  }
  saveConfigNow();
});

viewLayerSelect.on('change', (ev) => {
  if (suppressPaneEvents) return;
  setViewLayerChoice(ev.value);
  syncBioUniformState();
});

bioLinkBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

bioRateBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

projectionSelect.on('change', (ev) => {
  if (suppressPaneEvents) return;
  setProjectionChoice(ev.value);
});

windLevelSelect.on('change', (ev) => {
  if (suppressPaneEvents) return;
  loadWindLevel(ev.value);
});

windLinesBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncStreamCanvasOpacity();
  if (earthParams.windLines <= 0.001 && windCtx) {
    windStreamRenderer.clear(windCtx, viewState.width, viewState.height);
  }
});

oceanLinesBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncStreamCanvasOpacity();
  if (earthParams.oceanLines <= 0.001 && oceanCtx) {
    oceanStreamRenderer.clear(oceanCtx, viewState.width, viewState.height);
  }
});

tooltipsBinding.on('change', () => {
  if (suppressPaneEvents) return;
  tooltipManager.syncEnabledState();
});

windColorsBinding.on('change', () => {
  // Picked up immediately by bgRender uniform
});

oceanColorsBinding.on('change', () => {
  // Picked up immediately by bgRender uniform
});

coastlinesBinding.on('change', () => {
  if (suppressPaneEvents) return;
  mapDirty = true;
});

resetViewBtn.on('click', () => {
  earthParams.projection = 'orthographic';
  suppressPaneEvents = true;
  projectionSelect.refresh();
  suppressPaneEvents = false;
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
  if (suppressPaneEvents) return;
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
