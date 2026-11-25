import { CameraFlight, copyView } from './earth/cameraFlight';
import { modeEnabled, saveMode } from './temporal/modes';
import { GlobeTour, coordinates } from './earth/globeTour';
import { TemporalGame } from './temporal/game';
import {
  createSpectrogramSnapshot,
  GlobeCyberSynth,
} from './audio/engine';
import { SynthSpectrogramOverlay } from './audio/spectrogram';
import {
  observeScreenAudioState,
  type ScreenAudioObservation,
  type ScreenEnvObservation,
} from './audio/observer';
import { findColonies, isAlive, readCheckpoint, type Bloom, type Force } from './temporal/world';
import haloShader from './temporal/halo.wgsl?raw';
import gameFieldShader from './temporal/field.wgsl?raw';
import {
  bindGlobeInput,
  zoomViewAt,
  lockGlobeInput,
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
import {
  buildSpeciesMdpPriors,
  MDP_SPECIES_STRIDE,
} from './mdp';
import structsShader from './shaders/structs.wgsl?raw';
import projectionShader from './shaders/projection.wgsl?raw';
import bgRenderShader from './shaders/bgRender.wgsl?raw';
import lensTrackShader from './shaders/lensTrack.wgsl?raw';
import lensOverlayShader from './shaders/lensOverlay.wgsl?raw';
import slimeFieldShader from './shaders/slimeField.wgsl?raw';
import slimeCompositeShader from './shaders/slimeComposite.wgsl?raw';
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
  renderMapOverlayCanvas,
  type EarthTopoFeatures,
} from './earth/mapOverlay';
import {
  airTempColorRgb,
  buildDivergenceFreeWind,
  extendedSinebowColor,
  fetchLiveGfsWind,
  loadFallbackWind,
  sampleWindLonLat,
  sampleWindTempLonLat,
  WIND_LEVELS,
  type WindGrid,
  type WindLevelId,
} from './earth/windData';
import {
  buildGlobalTerrainTexture,
  sampleElevationKmLonLat,
  sampleRidgeMaskLonLat,
} from './earth/terrainData';
import {
  buildDivergenceFreeOcean,
  buildGlobalTempTexture,
  loadOceanCurrents,
  OCEAN_CURRENTS_SPEC,
  sampleOceanLonLat,
  sampleOceanSstLonLat,
  sstColorRgb,
  type OceanGrid,
} from './earth/oceanData';
import {
  buildGlobalUvTexture,
  fetchLiveGfsUvFields,
  loadCpcUvBaseline,
  sampleUvLonLat,
  solarAstronomyAtUtc,
  clearSkyUviGfs,
  cloudTransmission,
  uvCategoryLabel,
  uvIndexColorRgb,
  type LocalUvSample,
  type UvGrid,
} from './earth/uvData';
import {
  buildGlobalHydroTexture,
  createBaselineHydroGrid,
  fetchLiveGfsHydroFields,
  precip3hColorRgb,
  rhColorRgb,
  sampleHydroLonLat,
  type HydroGrid,
  type LocalHydroSample,
} from './earth/hydroData';
import {
  oceanGridToSampler,
  windGridToSampler,
  WindStreamRenderer,
} from './earth/windLines';
import {
  hslToRgb,
  isTimestampActive,
  linkRenderTimestamp,
  readTimestamp,
  requestTimestamps,
  resolveTimestamp,
  setupTimestamp,
} from './utils';
import {
  bindConfigPersistence,
  advanced, minimizeDisplay, particlesBinding,
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
  localTempBinding,
  localUvBinding,
  mineralLinkBinding,
  mineralOpacityBinding,
  mineralRateBinding,
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
  synthVolBinding,
  setEngineDisplay,
  setOceanSim,
  setSim,
  tempColorsBinding,
  tempStatusBinding,
  tempStrengthBinding,
  uvColorsBinding,
  uvStatusBinding,
  uvStrengthBinding,
  rhColorsBinding,
  precipColorsBinding,
  localHydroBinding,
  hydroStrengthBinding,
  hydroStatusBinding,
  tooltipsBinding,
  updatePeriodicLegend,
  viewLayerSelect,
  type ViewLayerMode,
  windColorsBinding,
  windLinesBinding,
  windStatusBinding,
} from './options';
import { setMouseToolMode } from './input';
import {
  buildUrlPartsFromConfig,
  clearCachedConfig,
  FACTORY_DEFAULTS,
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
const mapCanvas = document.getElementById('mapCanvas') as HTMLCanvasElement;
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
const tempHudGroup = document.getElementById('tempHudGroup') as HTMLElement;
const tempLevelBadge = document.getElementById('tempLevelBadge') as HTMLElement;
const tempTimeBadge = document.getElementById('tempTimeBadge') as HTMLElement;
const tempScaleMinLabel = document.getElementById('tempScaleMinLabel') as HTMLElement;
const tempScaleBar = document.getElementById('tempScaleBar') as HTMLElement;
const tempScaleMaxLabel = document.getElementById('tempScaleMaxLabel') as HTMLElement;
const uvHudGroup = document.getElementById('uvHudGroup') as HTMLElement;
const uvLevelBadge = document.getElementById('uvLevelBadge') as HTMLElement;
const uvTimeBadge = document.getElementById('uvTimeBadge') as HTMLElement;
const rhHudGroup = document.getElementById('rhHudGroup') as HTMLElement;
const rhTimeBadge = document.getElementById('rhTimeBadge') as HTMLElement;
const rhScaleBar = document.getElementById('rhScaleBar') as HTMLElement;
const precipHudGroup = document.getElementById('precipHudGroup') as HTMLElement;
const precipTimeBadge = document.getElementById('precipTimeBadge') as HTMLElement;
const precipScaleBar = document.getElementById('precipScaleBar') as HTMLElement;
const scaleMaxLabel = document.getElementById('scaleMaxLabel') as HTMLElement;

const mapCtx = mapCanvas.getContext('2d');
const oceanCtx = oceanCanvas.getContext('2d');
const windCtx = windCanvas.getContext('2d');
const oceanStreamRenderer = new WindStreamRenderer();
const windStreamRenderer = new WindStreamRenderer();
let cachedOceanStreamSampler: { grid: OceanGrid | null; tintKey: string; sampler: ReturnType<typeof oceanGridToSampler> } | null = null;
function getCachedOceanSampler(grid: OceanGrid, tint: [number, number, number]) {
  const key = `${tint[0]},${tint[1]},${tint[2]}`;
  if (cachedOceanStreamSampler && cachedOceanStreamSampler.grid === grid && cachedOceanStreamSampler.tintKey === key) {
    return cachedOceanStreamSampler.sampler;
  }
  const sampler = oceanGridToSampler(grid, tint);
  cachedOceanStreamSampler = { grid, tintKey: key, sampler };
  return sampler;
}
let cachedWindStreamSampler: { grid: WindGrid | null; sampler: ReturnType<typeof windGridToSampler> } | null = null;
function getCachedWindSampler(grid: WindGrid) {
  if (cachedWindStreamSampler && cachedWindStreamSampler.grid === grid) {
    return cachedWindStreamSampler.sampler;
  }
  const sampler = windGridToSampler(grid);
  cachedWindStreamSampler = { grid, sampler };
  return sampler;
}

const uniformsFloats = 56; // 224 bytes
const uniformData = new Float32Array(uniformsFloats);
const simSize = 24;
const simData = new Float32Array(simSize);
const oceanSimSize = 24;
const oceanSimData = new Float32Array(oceanSimSize);
const bioUniformFloats = 20; // 80 bytes
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
  (hasExplicitUrlConfig ? FACTORY_DEFAULTS.oceanSeed : cachedBootCfg?.oceanSeed) ??
  FACTORY_DEFAULTS.oceanSeed;
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
const lifeMdpStride = 48; // 12 words: life(vec4f) + mdpQPack(vec4u) + mdpCtrl(vec4f)
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
let mineralRenderPipeline: GPURenderPipeline | undefined;
let renderPipeline: GPURenderPipeline | undefined;
let slimeNodePipeline: GPURenderPipeline | undefined;
let slimeBridgePipeline: GPURenderPipeline | undefined;
let slimeCompositePipeline: GPURenderPipeline | undefined;
let slimeFieldBindGroupLayout: GPUBindGroupLayout | undefined;
let slimeFieldSampler: GPUSampler | undefined;
let slimeFieldTexture: GPUTexture | undefined;
let membraneFieldTexture: GPUTexture | undefined;
let slimeFieldBindGroup: GPUBindGroup | undefined;
let slimeCompositeBindGroup: GPUBindGroup | undefined;

let worldRulesBuffer: GPUBuffer | undefined;
let worldColoursBuffer: GPUBuffer | undefined;
let valenceBuffer: GPUBuffer | undefined;
let bioForceBuffer: GPUBuffer | undefined;
let particleBuffers: [GPUBuffer, GPUBuffer] | undefined;
let lifeStateBuffers: [GPUBuffer, GPUBuffer] | undefined;
let bioWasActive = false;

let windTexture: GPUTexture | undefined;
let oceanTexture: GPUTexture | undefined;
let tempTexture: GPUTexture | undefined;
let uvTexture: GPUTexture | undefined;
let hydroTexture: GPUTexture | undefined;
let terrainTexture: GPUTexture | undefined;
let watermanFwdTexture: GPUTexture | undefined;
let watermanInvTexture: GPUTexture | undefined;
let bgCacheTexture: GPUTexture | undefined;
let bgCacheKey = '';
let bgDataVersion = 0;
let bgCanCopyDst = true;

let bgBindGroup: GPUBindGroup | undefined;
let mineralRenderBindGroup: GPUBindGroup | undefined;
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
        ? FACTORY_DEFAULTS.view.scale
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
    : earthParams.projection === 'orthographic'
      ? [...FACTORY_DEFAULTS.view.rotate]
      : [...initSpec.defaultRotate];

const initPan: [number, number] =
  initialUrlPan ??
  (useCachedView && cachedBootCfg?.view?.pan
    ? [...cachedBootCfg.view.pan]
    : [...FACTORY_DEFAULTS.view.pan]);

const viewState: GlobeViewState = {
  projectionId: earthParams.projection,
  rotate: initRotate,
  scale: Math.max(140, initScale),
  pan: initPan,
  width: window.innerWidth,
  height: window.innerHeight,
};

const camera = new CameraFlight();
const projectionFade = document.getElementById('projection-fade')!;
let persistDestination = false;
let wheelZooming = false;
let d3Proj = buildConfiguredD3Projection(viewState);
let earthFeatures: EarthTopoFeatures | null = null;
let currentWindGrid: WindGrid = createEmptyWindGrid();
let currentOceanGrid: OceanGrid | null = null;
let currentUvGrid: UvGrid | null = null;
let currentHydroGrid: HydroGrid = createBaselineHydroGrid();
let uvUtcOverrideMs: number | null = null;
let watermanDirty = true;
let mapDirty = true;
let viewSignature = '';
let hashSyncTimer: number | undefined;
let configSaveTimer: number | undefined;
let configPersistenceReady = false;
let suppressPaneEvents = false;

function createEmptyWindGrid(): WindGrid {
  const n = GRID_TEX_W * GRID_TEX_H;
  const climK = new Float32Array(n);
  for (let row = 0; row < GRID_TEX_H; row++) {
    const latDeg = 90 - (row / Math.max(1, GRID_TEX_H - 1)) * 180;
    const cosLat = Math.max(0, Math.cos((latDeg * Math.PI) / 180));
    const tK = 250.15 + 50.0 * Math.pow(cosLat, 1.35);
    for (let col = 0; col < GRID_TEX_W; col++) {
      climK[row * GRID_TEX_W + col] = tK;
    }
  }
  return {
    nx: GRID_TEX_W,
    ny: GRID_TEX_H,
    u: new Float32Array(n),
    v: new Float32Array(n),
    tempK: climK,
    sfcTempK: climK,
    tempLabel: '2m Air',
    level: 'surface',
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

let defaultTempBadgeText = 'GFS + NOAA SST · loading…';
let defaultUvBadgeText = 'CPC + GFS O₃ · loading…';
let pointerHoverTempActive = false;

function sampleLocalUvAtUnit(u: [number, number, number]): LocalUvSample {
  const z = Math.max(-1, Math.min(1, u[2] ?? 0));
  const latDeg = Math.asin(z) * (180 / Math.PI);
  const lonDeg = Math.atan2(u[1] ?? 0, u[0] ?? 1) * (180 / Math.PI);
  return sampleLocalUvAtLonLat(lonDeg, latDeg);
}

function sampleLocalUvAtLonLat(
  lonDeg: number,
  latDeg: number,
  utcMs: number = uvUtcOverrideMs ?? Date.now(),
): LocalUvSample {
  if (currentUvGrid) {
    return sampleUvLonLat(currentUvGrid, lonDeg, latDeg, utcMs);
  }
  const astro = solarAstronomyAtUtc(utcMs);
  const d2r = Math.PI / 180;
  const latRad = latDeg * d2r;
  const cosZNoon = Math.sin(latRad) * astro.sinDecl + Math.cos(latRad) * astro.cosDecl;
  const tstMin =
    (((astro.utcHours * 60 + astro.eqTimeMin + 4 * lonDeg) % 1440) + 1440) % 1440;
  const haRad = ((tstMin / 4 - 180) * d2r);
  const cosZNow =
    Math.sin(latRad) * astro.sinDecl + Math.cos(latRad) * astro.cosDecl * Math.cos(haRad);
  const noonUvi = clearSkyUviGfs(cosZNoon, 300, 0) * cloudTransmission(0.25);
  let uviNow = 0;
  if (cosZNow > 0) {
    uviNow = noonUvi * Math.pow(cosZNow / Math.max(cosZNoon, 0.04), 2.15);
  } else if (cosZNow > -0.055) {
    const twi = (cosZNow + 0.055) / 0.055;
    uviNow = Math.min(0.24, noonUvi * 0.025) * twi * twi;
  }
  return {
    uviNow,
    uviNoon: noonUvi,
    category: uvCategoryLabel(uviNow),
    ozoneDU: 300,
    cloudPct: 25,
  };
}

function sampleLocalHydroAtUnit(u: [number, number, number]): LocalHydroSample {
  const z = Math.max(-1, Math.min(1, u[2] ?? 0));
  const latDeg = Math.asin(z) * (180 / Math.PI);
  const lonDeg = Math.atan2(u[1] ?? 0, u[0] ?? 1) * (180 / Math.PI);
  return sampleHydroLonLat(currentHydroGrid, lonDeg, latDeg);
}

function drawHydroScaleBars() {
  const rhStops: string[] = [];
  const pStops: string[] = [];
  const steps = 12;
  for (let i = 0; i <= steps; i++) {
    const f = i / steps;
    const [rr, rg, rb] = rhColorRgb(f);
    rhStops.push(`rgb(${Math.round(rr)}, ${Math.round(rg)}, ${Math.round(rb)}) ${(f * 100).toFixed(0)}%`);
    const [pr, pg, pb] = precip3hColorRgb(f * 30);
    pStops.push(`rgb(${Math.round(pr)}, ${Math.round(pg)}, ${Math.round(pb)}) ${(f * 100).toFixed(0)}%`);
  }
  rhScaleBar.style.background = `linear-gradient(to right, ${rhStops.join(', ')})`;
  precipScaleBar.style.background = `linear-gradient(to right, ${pStops.join(', ')})`;
}
drawHydroScaleBars();

function refreshHydroHudBadges() {
  const tag = currentHydroGrid.isLive
    ? `${currentHydroGrid.gfsValidTime} · ${currentHydroGrid.source}`
    : currentHydroGrid.source;
  rhTimeBadge.textContent = tag;
  precipTimeBadge.textContent = tag;
  earthParams.hydroStatus = currentHydroGrid.isLive
    ? `Live (${currentHydroGrid.gfsRefTime})`
    : 'Climatology RH+3HPS';
  hydroStatusBinding.refresh();
}

function sampleLocalTempsAtUnit(u: [number, number, number]) {
  const z = Math.max(-1, Math.min(1, u[2] ?? 0));
  const latDeg = Math.asin(z) * (180 / Math.PI);
  const lonDeg = Math.atan2(u[1] ?? 0, u[0] ?? 1) * (180 / Math.PI);
  return sampleLocalTempsAtLonLat(lonDeg, latDeg);
}

function sampleLocalTempsAtLonLat(lonDeg: number, latDeg: number) {
  const [airK, sfcAirK] = sampleWindTempLonLat(currentWindGrid, lonDeg, latDeg);
  let sstK: number | null = null;
  let oceanMask = 0;
  if (currentOceanGrid) {
    const [sK, sM] = sampleOceanSstLonLat(currentOceanGrid, lonDeg, latDeg);
    oceanMask = sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg)[3];
    if (sM >= 0.42 && oceanMask >= 0.35) {
      sstK = sK;
    }
  }
  const airC = airK - 273.15;
  const sfcAirC = sfcAirK - 273.15;
  const sstC = sstK !== null ? sstK - 273.15 : null;
  const surfaceC = sstC !== null ? sstC : sfcAirC;
  return {
    airC,
    sfcAirC,
    sstC,
    surfaceC,
    oceanMask,
    airLabel: currentWindGrid.tempLabel || '2m Air',
  };
}

function drawTempScaleBar(isOceanSst: boolean) {
  const minC = isOceanSst ? -2 : -40;
  const maxC = isOceanSst ? 40 : 45;
  const stops: string[] = [];
  const steps = 16;
  for (let i = 0; i <= steps; i++) {
    const f = i / steps;
    const tK = 273.15 + minC + (maxC - minC) * f;
    const [r, g, b] = isOceanSst ? sstColorRgb(tK) : airTempColorRgb(tK);
    stops.push(`rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}) ${(f * 100).toFixed(0)}%`);
  }
  tempScaleBar.style.background = `linear-gradient(to right, ${stops.join(', ')})`;
  tempScaleMinLabel.textContent = `${minC}°C`;
  tempScaleMaxLabel.textContent = `+${maxC}°C`;
}

function refreshTempHudBadges() {
  const airTag = currentWindGrid.tempLabel || '2m Air';
  if (earthParams.viewLayer === 'ocean') {
    tempLevelBadge.textContent = 'Temp @ Ocean SST';
  } else if (earthParams.viewLayer === 'air') {
    tempLevelBadge.textContent = `Temp @ ${airTag}`;
  } else {
    tempLevelBadge.textContent = `Temp @ ${airTag} / SST`;
  }
  drawTempScaleBar(earthParams.viewLayer === 'ocean');
  const wTime = currentWindGrid.validTime ? currentWindGrid.validTime : currentWindGrid.source;
  const sstTime = currentOceanGrid
    ? currentOceanGrid.sstRefTime.replace('T12:00:00Z', '').replace('T00:00:00Z', '')
    : 'loading SST';
  defaultTempBadgeText = `${wTime} · SST ${sstTime}`;
  if (!pointerHoverTempActive) {
    tempTimeBadge.textContent = defaultTempBadgeText;
  }
  earthParams.tempStatus = `${airTag} (${currentWindGrid.isLive ? 'live' : 'cached'}) · SST ${sstTime}`;
  tempStatusBinding.refresh();
}

function refreshUvHudBadges() {
  const utcDate = new Date(uvUtcOverrideMs ?? Date.now());
  const hh = String(utcDate.getUTCHours()).padStart(2, '0');
  const mm = String(utcDate.getUTCMinutes()).padStart(2, '0');
  const srcTag = currentUvGrid?.isLive ? 'CPC + Live GFS O₃/Cloud' : 'CPC + GFS O₃';
  defaultUvBadgeText = `${hh}:${mm}Z · ${srcTag}`;
  if (!pointerHoverTempActive) {
    uvTimeBadge.textContent = defaultUvBadgeText;
  }
  uvLevelBadge.textContent = 'UVI @ Surface';
  earthParams.uvStatus = currentUvGrid
    ? `${srcTag} (${currentUvGrid.isLive ? currentUvGrid.gfsRefTime : currentUvGrid.cpcRefTime.slice(0, 10)})`
    : 'Solar Zenith Model';
  uvStatusBinding.refresh();
}

function refreshHudVisibility() {
  const showAir = earthParams.viewLayer !== 'ocean';
  const showOcean = earthParams.viewLayer !== 'air';
  windHudGroup.style.display = showAir ? '' : 'none';
  oceanHudGroup.style.display = showOcean ? '' : 'none';
  tempHudGroup.style.display = '';
  uvHudGroup.style.display = '';
  rhHudGroup.style.display = earthParams.rhColors > 0.01 ? '' : 'none';
  precipHudGroup.style.display = earthParams.precipColors > 0.01 ? '' : 'none';
  refreshTempHudBadges();
  refreshUvHudBadges();
  refreshHydroHudBadges();
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
  const view = persistDestination && camera.target ? camera.target : viewState;
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
      projection: view.projectionId,
      vortex: earthParams.vortex,
      windStrength: earthParams.windStrength,
      windLines: earthParams.windLines,
      windColors: earthParams.windColors,
      chemOpacity: earthParams.chemOpacity,
      oceanMix: earthParams.oceanMix,
      oceanStrength: earthParams.oceanStrength,
      oceanLines: earthParams.oceanLines,
      oceanColors: earthParams.oceanColors,
      tempColors: earthParams.tempColors,
      localTemp: earthParams.localTemp,
      tempStrength: earthParams.tempStrength,
      uvColors: earthParams.uvColors,
      localUv: earthParams.localUv,
      uvStrength: earthParams.uvStrength,
      rhColors: earthParams.rhColors,
      precipColors: earthParams.precipColors,
      localHydro: earthParams.localHydro,
      hydroStrength: earthParams.hydroStrength,
      lifeOpacity: earthParams.lifeOpacity,
      mineralOpacity: earthParams.mineralOpacity,
      terrainOpacity: earthParams.terrainOpacity,
      nightShading: earthParams.nightShading,
      coastlines: earthParams.coastlines,
      bioLink: earthParams.bioLink,
      bioRate: earthParams.bioRate,
      mineralLink: earthParams.mineralLink,
      mineralRate: earthParams.mineralRate,
      tooltips: earthParams.tooltips,
      mouseTool: earthParams.mouseTool,
    },
    view: {
      rotate: [...view.rotate],
      pan: [...view.pan],
      scale: view.scale,
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
    const factory = buildFactoryConfig(FACTORY_DEFAULTS.seed, FACTORY_DEFAULTS.oceanSeed);
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
  let nextViewLayer: ViewLayerMode | null = null;
  let projUpdated = false;
  const importedView = copyView(viewState);

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
    if (!incidentCameraLocked && e.projection && e.projection !== earthParams.projection) {
      importedView.projectionId = e.projection;
      projUpdated = true;
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
    if (e.tempColors !== undefined) earthParams.tempColors = e.tempColors;
    if (e.localTemp !== undefined) earthParams.localTemp = e.localTemp;
    if (e.tempStrength !== undefined) earthParams.tempStrength = e.tempStrength;
    if (e.uvColors !== undefined) earthParams.uvColors = e.uvColors;
    if (e.localUv !== undefined) earthParams.localUv = e.localUv;
    if (e.uvStrength !== undefined) earthParams.uvStrength = e.uvStrength;
    if (e.rhColors !== undefined) earthParams.rhColors = e.rhColors;
    if (e.precipColors !== undefined) earthParams.precipColors = e.precipColors;
    if (e.localHydro !== undefined) earthParams.localHydro = e.localHydro;
    if (e.hydroStrength !== undefined) earthParams.hydroStrength = e.hydroStrength;
    if (e.lifeOpacity !== undefined) earthParams.lifeOpacity = e.lifeOpacity;
    if (e.mineralOpacity !== undefined) earthParams.mineralOpacity = e.mineralOpacity;
    if (e.terrainOpacity !== undefined) earthParams.terrainOpacity = e.terrainOpacity;
    if (e.nightShading !== undefined) earthParams.nightShading = e.nightShading;
    if (e.coastlines !== undefined) {
      earthParams.coastlines = e.coastlines;
      mapDirty = true;
    }
    if (e.bioLink !== undefined) earthParams.bioLink = e.bioLink;
    if (e.bioRate !== undefined) earthParams.bioRate = e.bioRate;
    if (e.mineralLink !== undefined) earthParams.mineralLink = e.mineralLink;
    if (e.mineralRate !== undefined) earthParams.mineralRate = e.mineralRate;
    if (e.tooltips !== undefined) earthParams.tooltips = e.tooltips;
    if (e.mouseTool !== undefined) {
      earthParams.mouseTool = e.mouseTool;
      setMouseToolMode(e.mouseTool);
    }
  }

  if (!incidentCameraLocked && cfg.view) {
    if (cfg.view.rotate) {
      importedView.rotate = [...cfg.view.rotate];
      projUpdated = true;
    }
    if (cfg.view.pan) {
      importedView.pan = [...cfg.view.pan];
      projUpdated = true;
    }
    if (cfg.view.scale !== undefined) {
      importedView.scale = Math.max(120, Math.min(30000, cfg.view.scale));
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
    moveCamera(importedView, 1200, true);
  }

  updatePeriodicLegend(colourAmt);
  if (resetWorld) {
    refreshChemistryDefinition();
    startWorld();
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

function updateProjectionState(persist = true) {
  d3Proj = buildConfiguredD3Projection(viewState);
  refreshBoundProjection(d3Proj);
  if (earthParams.projection !== viewState.projectionId) {
    earthParams.projection = viewState.projectionId;
    projectionSelect.refresh();
  }
  mapDirty = true;
  if (viewState.projectionId === 'waterman') {
    watermanDirty = true;
  }
  viewSignature = `${viewState.projectionId}:${viewState.rotate.map((v) => v.toFixed(2)).join(',')}:${viewState.scale.toFixed(1)}:${viewState.pan.map((v) => v.toFixed(1)).join(',')}:${viewState.width}x${viewState.height}`;
  if (persist) { syncUrlStateDebounced(); scheduleSaveConfig(); }
}

bindGlobeInput(viewState, () => d3Proj, () => {
  takeCameraControl();
  camera.cancel();
  updateProjectionState();
}, (factor, x, y) => {
  const destination = copyView(wheelZooming && camera.target ? camera.target : viewState);
  if (!zoomViewAt(destination, factor, x, y)) return;
  moveCamera(destination, 185, true);
  wheelZooming = true;
});

function uploadTerrainToGpu() {
  if (!device || !terrainTexture) return;
  const texData = buildGlobalTerrainTexture(
    currentUvGrid,
    currentOceanGrid,
    GRID_TEX_W,
    GRID_TEX_H,
  );
  device.queue.writeTexture(
    { texture: terrainTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
  bgDataVersion++;
}

function uploadTempToGpu() {
  if (freezeGameWeather && particleBuffers) return;
  if (!device || !tempTexture) return;
  const texData = buildGlobalTempTexture(
    currentWindGrid,
    currentOceanGrid,
    GRID_TEX_W,
    GRID_TEX_H,
  );
  device.queue.writeTexture(
    { texture: tempTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
  bgDataVersion++;
  refreshTempHudBadges();
}

function uploadUvToGpu() {
  if (freezeGameWeather && particleBuffers) return;
  if (!device || !uvTexture) return;
  const texData = buildGlobalUvTexture(
    currentUvGrid,
    uvUtcOverrideMs ?? Date.now(),
    GRID_TEX_W,
    GRID_TEX_H,
  );
  device.queue.writeTexture(
    { texture: uvTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
  bgDataVersion++;
  refreshUvHudBadges();
}

function uploadHydroToGpu() {
  if (freezeGameWeather && particleBuffers) return;
  if (!device || !hydroTexture) return;
  const texData = buildGlobalHydroTexture(currentHydroGrid, GRID_TEX_W, GRID_TEX_H);
  device.queue.writeTexture(
    { texture: hydroTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
  bgDataVersion++;
  refreshHydroHudBadges();
}

function uploadWindToGpu(grid: WindGrid) {
  if (freezeGameWeather && particleBuffers) return;
  if (!device || !windTexture) return;
  const texData = buildDivergenceFreeWind(grid, GRID_TEX_W, GRID_TEX_H);
  device.queue.writeTexture(
    { texture: windTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
  uploadTempToGpu();
}

function uploadOceanToGpu(grid: OceanGrid) {
  if (freezeGameWeather && particleBuffers) return;
  if (!device || !oceanTexture) return;
  const texData = buildDivergenceFreeOcean(grid, GRID_TEX_W, GRID_TEX_H);
  device.queue.writeTexture(
    { texture: oceanTexture },
    texData,
    { bytesPerRow: GRID_TEX_W * 16 },
    { width: GRID_TEX_W, height: GRID_TEX_H, depthOrArrayLayers: 1 },
  );
  uploadTerrainToGpu();
  uploadTempToGpu();
}

let activeWindLoadId = 0;

async function loadSurfaceWind() {
  const reqId = ++activeWindLoadId;
  const spec = WIND_LEVELS.surface;
  earthParams.windStatus = `Fetching ${spec.label}…`;
  windStatusBinding.refresh();

  if (!currentWindGrid.isLive) {
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
    const live = await fetchLiveGfsWind('surface');
    if (reqId === activeWindLoadId) {
      currentWindGrid = live;
      uploadWindToGpu(live);
      refreshHudBadges(live);
      if (currentUvGrid) {
        void fetchLiveGfsUvFields(currentUvGrid)
          .then((liveUv) => {
            currentUvGrid = liveUv;
            uploadUvToGpu();
          })
          .catch(() => {
            // Keep CPC + cached GFS ozone baseline
          });
      }
      void fetchLiveGfsHydroFields()
        .then((liveHydro) => {
          currentHydroGrid = liveHydro;
          uploadHydroToGpu();
        })
        .catch(() => {
          // Keep climatology RH + 3HPS fallback
        });
    }
  } catch {
    if (reqId === activeWindLoadId && !currentWindGrid.isLive) {
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

const GFS_AUTO_REFRESH_MS = 15 * 60 * 1000;
const DAILY_SNAPSHOT_REFRESH_MS = 6 * 60 * 60 * 1000;
let lastGfsCheckAt = Date.now();
let lastSnapshotCheckAt = Date.now();
let bgRefreshInFlight: Promise<boolean> | null = null;

async function backgroundRefreshWeather(opts?: {
  forceGfs?: boolean;
  forceSnapshots?: boolean;
}): Promise<boolean> {
  if (bgRefreshInFlight) return bgRefreshInFlight;
  bgRefreshInFlight = (async () => {
    const now = Date.now();
    const doSnapshots =
      Boolean(opts?.forceSnapshots) || now - lastSnapshotCheckAt >= DAILY_SNAPSHOT_REFRESH_MS;
    const doGfs = Boolean(opts?.forceGfs) || now - lastGfsCheckAt >= GFS_AUTO_REFRESH_MS;
    if (!doSnapshots && !doGfs) return false;

    let updatedAny = false;
    let uvBaselineUpdated = false;

    if (doSnapshots) {
      lastSnapshotCheckAt = now;
      const [freshOcean, freshUv] = await Promise.all([
        loadOceanCurrents(true).catch(() => null),
        loadCpcUvBaseline(true).catch(() => null),
      ]);
      if (
        freshOcean &&
        (!currentOceanGrid ||
          freshOcean.refTime !== currentOceanGrid.refTime ||
          freshOcean.sstRefTime !== currentOceanGrid.sstRefTime)
      ) {
        currentOceanGrid = freshOcean;
        uploadOceanToGpu(freshOcean);
        refreshOceanHud(freshOcean);
        updatedAny = true;
      }
      if (
        freshUv &&
        (!currentUvGrid || freshUv.cpcRefTime !== currentUvGrid.cpcRefTime)
      ) {
        currentUvGrid = freshUv;
        uvBaselineUpdated = true;
        updatedAny = true;
      }
    }

    if (doGfs || uvBaselineUpdated) {
      lastGfsCheckAt = now;
      const reqId = activeWindLoadId;
      try {
        const live = await fetchLiveGfsWind('surface', {
          forceRefresh: Boolean(opts?.forceGfs),
        });
        if (
          reqId === activeWindLoadId &&
          (!currentWindGrid.isLive ||
            live.refTime !== currentWindGrid.refTime ||
            live.validTime !== currentWindGrid.validTime)
        ) {
          currentWindGrid = live;
          uploadWindToGpu(live);
          refreshHudBadges(live);
          updatedAny = true;
        }
        if (
          currentUvGrid &&
          (uvBaselineUpdated ||
            !currentUvGrid.isLive ||
            currentUvGrid.gfsRefTime !== live.refTime ||
            currentUvGrid.gfsValidTime !== live.validTime)
        ) {
          try {
            const liveUv = await fetchLiveGfsUvFields(currentUvGrid, {
              forceRefresh: Boolean(opts?.forceGfs),
            });
            currentUvGrid = liveUv;
            uploadUvToGpu();
            updatedAny = true;
          } catch {
            if (uvBaselineUpdated) uploadUvToGpu();
          }
        }
        if (
          !currentHydroGrid.isLive ||
          currentHydroGrid.gfsRefTime !== live.refTime ||
          currentHydroGrid.gfsValidTime !== live.validTime
        ) {
          try {
            currentHydroGrid = await fetchLiveGfsHydroFields({
              forceRefresh: Boolean(opts?.forceGfs),
            });
            uploadHydroToGpu();
            updatedAny = true;
          } catch {
            // Keep existing hydrology grid
          }
        }
      } catch {
        if (uvBaselineUpdated) uploadUvToGpu();
      }
    }

    return updatedAny;
  })().finally(() => {
    bgRefreshInFlight = null;
  });
  return bgRefreshInFlight;
}

function writeWorldRuleAndColourBuffers() {
  if (!device || !worldRulesBuffer || !worldColoursBuffer) return;
  const { flat: traitsFlat, table } = computeSpeciesTraitsFromMatrix(
    oceanMatrixData,
    oceanColourAmt,
  );
  speciesTraits = table;
  const mdpPriorsFlat = buildSpeciesMdpPriors(table, oceanColourAmt, oceanSeed);
  const rules = new Float32Array(
    chemistryData.length +
      oceanMatrixData.length +
      traitsFlat.length +
      mdpPriorsFlat.length,
  );
  rules.set(chemistryData, 0);
  rules.set(oceanMatrixData, chemistryData.length);
  rules.set(traitsFlat, chemistryData.length + oceanMatrixData.length);
  rules.set(
    mdpPriorsFlat,
    chemistryData.length + oceanMatrixData.length + traitsFlat.length,
  );
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
      getMineralLink: () => boolean;
      setMineralLink: (on: boolean) => void;
      getMineralRate: () => number;
      setMineralRate: (rate: number) => void;
      getTooltips: () => boolean;
      setTooltips: (on: boolean) => void;
      getOpacities: () => {
        windLines: number;
        windColors: number;
        chemOpacity: number;
        oceanLines: number;
        oceanColors: number;
        tempColors: number;
        uvColors: number;
        lifeOpacity: number;
        mineralOpacity: number;
      };
      setChemOpacity: (val: number) => void;
      setLifeOpacity: (val: number) => void;
      setMineralOpacity: (val: number) => void;
      setOceanColors: (val: boolean | number) => void;
      setOceanLines: (val: boolean | number) => void;
      setWindColors: (val: boolean | number) => void;
      setWindLines: (val: boolean | number) => void;
      setTempColors: (val: boolean | number) => void;
      setUvColors: (val: boolean | number) => void;
      getLocalTemp: () => boolean;
      setLocalTemp: (on: boolean) => void;
      getTempStrength: () => number;
      setTempStrength: (val: number) => void;
      getLocalUv: () => boolean;
      setLocalUv: (on: boolean) => void;
      getUvStrength: () => number;
      setUvStrength: (val: number) => void;
      getTempAtLonLat: (lonDeg: number, latDeg: number) => {
        airC: number;
        sfcAirC: number;
        sstC: number | null;
        surfaceC: number;
        oceanMask: number;
        airLabel: string;
      };
      getTempInfo: () => {
        airLevel: WindLevelId;
        airLabel: string;
        airRefTime: string;
        sstRefTime: string;
        sstSource: string;
        isLive: boolean;
      };
      getUvAtLonLat: (lonDeg: number, latDeg: number, utcMs?: number) => LocalUvSample;
      setUvUtcTime: (isoOrMs: string | number | null) => void;
      getUvInfo: () => {
        cpcRefTime: string;
        gfsRefTime: string;
        gfsValidTime: string;
        source: string;
        isLive: boolean;
      };
      refreshWeatherData: (opts?: { forceSnapshots?: boolean }) => Promise<boolean>;
      getWeatherRefreshState: () => {
        lastGfsCheckAt: number;
        lastSnapshotCheckAt: number;
        gfsIntervalMs: number;
        snapshotIntervalMs: number;
      };
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
      setAudioEnabled: (on: boolean) => void;
      getAudioStatus: () => {
        enabled: boolean;
        state: string;
        volume: number;
        lastObservation: ScreenAudioObservation | null;
      };
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
        mineralCount: number;
        mineralOceanCount: number;
        mineralLandCount: number;
        mineralTypeCounts: [number, number, number, number, number];
        mineralTrapCount: number;
        nobleMineralViolations: number;
        avgMineralIntegrity: number;
        mineralElementCounts: Record<string, number>;
        guildCounts: [number, number, number, number];
        abioticBioViolations: number;
        lipidAbioticViolations: number;
        chemElementCounts: Record<string, number>;
        bioElementCounts: Record<string, number>;
        avgBioEnergy: number;
        oceanBioCount: number;
        landBioCount: number;
        terrestrialBlobs: number;
        blobCount: number;
        multiSpeciesBlobs: number;
        complexBlobs: number;
        encapsulatedBlobs: number;
        encapsulatedSimpleViolations: number;
        graduatedBlobs: number;
        graduatedParticles: number;
        prematureGraduatedViolations: number;
        maxGraduatedBlobSize: number;
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
      f32[b + 11],
    ];
  }
  return out;
}

function unpackVec4F32(f32: Float32Array, n: number, floatStride: number = 4): number[][] {
  const out: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const b = i * floatStride;
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
  floatStride: number = 4,
): Promise<number[][]> {
  if (!device || !buf) return [];
  const n = Math.min(count, totalAmt);
  const byteLength = n * floatStride * 4;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = unpackVec4F32(new Float32Array(staging.getMappedRange()), n, floatStride);
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
  const stg: Record<string, { buf: GPUBuffer; count: number; isVec4: boolean; floatStride: number }> = {};
  const queueRead = (
    name: string,
    src: GPUBuffer,
    byteLength: number,
    isVec4: boolean,
    floatStride: number = 4,
  ) => {
    const buf = device!.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    enc.copyBufferToBuffer(src, 0, buf, 0, byteLength);
    stg[name] = { buf, count: n, isVec4, floatStride };
  };

  queueRead('particles', particleBuffers[alternate], n * particleStride, false, 12);
  queueRead('lifeState', lifeStateBuffers[alternate], n * lifeMdpStride, true, 12);
  if (includeExtras) {
    if (valenceBuffer) {
      queueRead('valence', valenceBuffer, n * valenceStride, true, 4);
    }
    if (bioForceBuffer) {
      queueRead('bioForce', bioForceBuffer, n * valenceStride, true, 4);
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
      return entry.isVec4
        ? unpackVec4F32(f32, entry.count, entry.floatStride)
        : unpackParticlesF32(f32, entry.count);
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
  const mineralElementCounts: Record<string, number> = {};
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
      chemGasCount: 0,
      chemLiquidCount: 0,
      chemIceCount: 0,
      bioCount: 0,
      cytoplasmCount: 0,
      lipidCount: 0,
      mineralCount: 0,
      mineralOceanCount: 0,
      mineralLandCount: 0,
      mineralTypeCounts: [0, 0, 0, 0, 0] as [number, number, number, number, number],
      mineralTrapCount: 0,
      nobleMineralViolations: 0,
      avgMineralIntegrity: 0,
      mineralElementCounts,
      guildCounts: [0, 0, 0, 0] as [number, number, number, number],
      abioticBioViolations: 0,
      lipidAbioticViolations: 0,
      chemElementCounts,
      bioElementCounts,
      avgBioEnergy: 0,
      oceanBioCount: 0,
      landBioCount: 0,
      terrestrialBlobs: 0,
      blobCount: 0,
      multiSpeciesBlobs: 0,
      complexBlobs: 0,
      encapsulatedBlobs: 0,
      encapsulatedSimpleViolations: 0,
      graduatedBlobs: 0,
      graduatedParticles: 0,
      prematureGraduatedViolations: 0,
      maxGraduatedBlobSize: 0,
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
  let chemGasCount = 0;
  let chemLiquidCount = 0;
  let chemIceCount = 0;
  let bioCount = 0;
  let cytoplasmCount = 0;
  let lipidCount = 0;
  let mineralCount = 0;
  let mineralOceanCount = 0;
  let mineralLandCount = 0;
  const mineralTypeCounts: [number, number, number, number, number] = [0, 0, 0, 0, 0];
  let mineralTrapCount = 0;
  let nobleMineralViolations = 0;
  let mineralIntegSum = 0;
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
    graduated: boolean;
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
    if (meta.phase === 1 || meta.phase === 2) {
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
    } else if (meta.phase === 3) {
      mineralCount++;
      const mType = Math.min(4, Math.max(0, meta.bioSpecies));
      mineralTypeCounts[mType]++;
      if (meta.graduated) mineralTrapCount++;
      mineralIntegSum += ls[0] ?? 0;
      mineralElementCounts[sym] = (mineralElementCounts[sym] ?? 0) + 1;
      if (el.valence === 0) {
        nobleMineralViolations++;
      }
      if (currentOceanGrid) {
        const z = Math.max(-1, Math.min(1, p[2] ?? 0));
        const latDeg = Math.asin(z) * RAD2DEG;
        const lonDeg = Math.atan2(p[1] ?? 0, p[0] ?? 1) * RAD2DEG;
        const m = sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg)[3];
        if (m >= 0.42) mineralOceanCount++;
        else mineralLandCount++;
      }
    } else {
      chemCount++;
      const comp = ls[1] ?? 0;
      if (comp >= 1.5) chemIceCount++;
      else if (comp >= 0.5) chemLiquidCount++;
      else chemGasCount++;
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
  let terrestrialBlobs = 0;
  let multiSpeciesBlobs = 0;
  let complexBlobs = 0;
  let encapsulatedBlobs = 0;
  let encapsulatedSimpleViolations = 0;
  let graduatedBlobs = 0;
  let graduatedParticles = 0;
  let prematureGraduatedViolations = 0;
  let maxGraduatedBlobSize = 0;
  let nakedBlobs = 0;
  let blobSizeSum = 0;
  let maxBlobSize = 0;
  let blobSpeciesSum = 0;
  let maxBlobDistinctSpecies = 0;
  let encapSpeciesSum = 0;
  let nakedSpeciesSum = 0;
  let encapGeneSum = 0;
  const mdpBlobActions: [number, number, number, number] = [0, 0, 0, 0];

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
    if (summary.graduated) {
      graduatedBlobs++;
      graduatedParticles += summary.graduatedCount;
      if (comp.length > maxGraduatedBlobSize) maxGraduatedBlobSize = comp.length;
      if (comp.length < 48 || summary.lipidCount < 4 || summary.distinctSpecies < 6) {
        prematureGraduatedViolations++;
      }
    } else if (summary.graduatedCount > 0) {
      graduatedParticles += summary.graduatedCount;
    }
    if (summary.encapStrategy === 'early-armor') encapStrategies.earlyArmorBlobs++;
    else if (summary.encapStrategy === 'balanced') encapStrategies.balancedBlobs++;
    else encapStrategies.lateComplexBlobs++;

    if (currentOceanGrid) {
      let landMembers = 0;
      for (const id of comp) {
        const p = data.particles[id];
        const z = Math.max(-1, Math.min(1, p[2] ?? 0));
        const latDeg = Math.asin(z) * RAD2DEG;
        const lonDeg = Math.atan2(p[1] ?? 0, p[0] ?? 1) * RAD2DEG;
        if (sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg)[3] < 0.42) {
          landMembers++;
        }
      }
      if (landMembers > 0 && landMembers >= comp.length * 0.5) {
        terrestrialBlobs++;
      }
    }

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
    mdpBlobActions[summary.mdpAction] = (mdpBlobActions[summary.mdpAction] ?? 0) + 1;
  }

  return {
    total,
    chemCount,
    chemGasCount,
    chemLiquidCount,
    chemIceCount,
    bioCount,
    cytoplasmCount,
    lipidCount,
    mineralCount,
    mineralOceanCount,
    mineralLandCount,
    mineralTypeCounts,
    mineralTrapCount,
    nobleMineralViolations,
    avgMineralIntegrity: mineralCount > 0 ? mineralIntegSum / mineralCount : 0,
    mineralElementCounts,
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
    terrestrialBlobs,
    blobCount,
    multiSpeciesBlobs,
    complexBlobs,
    encapsulatedBlobs,
    encapsulatedSimpleViolations,
    graduatedBlobs,
    graduatedParticles,
    prematureGraduatedViolations,
    maxGraduatedBlobSize,
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
    mdpBlobActions,
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
  if (incidentCameraLocked) return;
  const destination = copyView(viewState);
  destination.projectionId = projId;
  destination.rotate = [...PROJECTIONS[projId].defaultRotate];
  destination.pan = [0, 0];
  destination.scale = fitScaleForProjection(projId, destination.rotate, destination.width, destination.height);
  moveCamera(destination, 1200, true);
}

const clampOpacity = (val: boolean | number) =>
  typeof val === 'number' ? Math.max(0, Math.min(1, val)) : val ? 0.2 : 0;

function syncStreamCanvasOpacity() {
  oceanCanvas.style.opacity = String(earthParams.oceanLines);
  windCanvas.style.opacity = String(earthParams.windLines);
}
syncStreamCanvasOpacity();

let freezeGameWeather = false;
let deviceLost = false;
let gamePipeline: GPUComputePipeline | undefined;
let gameUniform: GPUBuffer | undefined;
let gameTracking: GPUBuffer | undefined;
let gameGroups: GPUBindGroup[] = [];
let haloPipeline: GPURenderPipeline | undefined;
let haloMembers: GPUBuffer | undefined;
let haloGroup: GPUBindGroup | undefined;
let savedCamera: GlobeViewState | undefined;
let incidentCameraLocked = false;
const tour = new GlobeTour();
tour.setEnabled(modeEnabled('tour'));
const rescuedEpisodes = new Map<number, number>();
let incidentEpisodes = new Map<number, number>();
let nextTourSampleMs = 0;
let samplingTour = false;
let tourGeneration = 0;

function syncEpisodeTracking() {
  if (freezeGameWeather || tour.enabled || rescuedEpisodes.size) {
    if (!gameGroups.length) buildGameGroups();
  } else {
    gameGroups = [];
    gameTracking?.destroy();
    gameTracking = undefined;
  }
}

function takeCameraControl() {
  persistDestination = false;
  wheelZooming = false;
  if (tour.enabled || rescuedEpisodes.size || tour.caption) {
    if (tour.enabled) saveMode('tour', false);
    tour.manualMove();
    tourGeneration++;
    rescuedEpisodes.clear();
    refreshHalo();
    syncEpisodeTracking();
    displayCamera();
  }
}

function moveCamera(destination: GlobeViewState, durationMs: number, manual = false) {
  if (manual) takeCameraControl();
  persistDestination = manual;
  wheelZooming = false;
  camera.move(viewState, destination, durationMs);
  tour.pauseClock();
  if (manual) scheduleSaveConfig();
}

function displayCamera() {
  const button = document.getElementById('tour-toggle');
  const caption = document.getElementById('tour-caption');
  if (!button || !caption) return;
  if (button.getAttribute('aria-pressed') !== String(tour.enabled))
    button.setAttribute('aria-pressed', String(tour.enabled));
  const label = [tour.caption, rescuedEpisodes.size ? 'Survivors ringed' : '']
    .filter(Boolean)
    .join(' · ');
  if (label !== caption.textContent) caption.textContent = label;
  caption.hidden = !label;
}

const colonyRadius = () =>
  Math.max(
    0.02,
    1.35 /
      Math.max(0.5, oceanOptionParams.r) /
      Math.max(0.25, optionParams.worldSize),
  );

function buildGameGroups() {
  if (!device || !particleBuffers || !lifeStateBuffers || gameGroups.length)
    return;
  gamePipeline ??= device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: gameFieldShader }),
      entryPoint: 'main',
    },
  });
  gameUniform ??= device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  gameTracking = device.createBuffer({
    size: particleAmt * Uint32Array.BYTES_PER_ELEMENT * 2,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.VERTEX |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  gameGroups = [0, 1].map((index) =>
    device!.createBindGroup({
      layout: gamePipeline!.getBindGroupLayout(0),
      entries: [
        particleBuffers![index],
        lifeStateBuffers![index],
        gameUniform!,
        lifeStateBuffers![1 - index],
        gameTracking!,
      ].map((buffer, binding) => ({ binding, resource: { buffer } })),
    }),
  );
}

function stepWorld(
  force?: Force,
  bloom?: Bloom,
  sharedEncoder?: GPUCommandEncoder,
) {
  if (!device || !particleBuffers) return;
  if (incidentCameraLocked) tooltipManager.invalidateInspectionFrame();
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
  if (freezeGameWeather && uniformBuffer) {
    // All gameplay forces come from the editable, tick-indexed plan.
    device.queue.writeBuffer(uniformBuffer, 7 * 4, new Float32Array([0]));
  }
  const encoder = sharedEncoder ?? device.createCommandEncoder();
  tick(encoder);
  if (gameUniform && gamePipeline && gameGroups[alternate]) {
    const field = new Float32Array(16);
    if (force) {
      field.set(force.point, 0);
      field[3] = force.radius;
      field.set(force.pathStart, 4);
      field.set(force.direction, 8);
      field[11] = force.strength;
    }
    if (bloom) {
      field.set(bloom.point, 12);
      field[15] = bloom.radius;
    }
    device.queue.writeBuffer(gameUniform, 0, field);
    const pass = encoder.beginComputePass();
    pass.setPipeline(gamePipeline);
    pass.setBindGroup(0, gameGroups[alternate]);
    pass.dispatchWorkgroups(Math.ceil(particleAmt / (1 << 6)));
    pass.end();
  }
  if (!sharedEncoder) device.queue.submit([encoder.finish()]);
}

function observeSandbox(nowMs: number) {
  if (
    !tour.enabled ||
    game.mode !== 'off' ||
    samplingTour ||
    nowMs < nextTourSampleMs ||
    !device ||
    !particleBuffers ||
    !lifeStateBuffers ||
    !gameTracking
  )
    return;
  const intervalMs = 1000;
  nextTourSampleMs +=
    (1 + Math.floor((nowMs - nextTourSampleMs) / intervalMs)) * intervalMs;
  const generation = tourGeneration;
  samplingTour = true;
  void readCheckpoint(
    device,
    particleBuffers[alternate],
    lifeStateBuffers[alternate],
    gameTracking,
    0,
    frameCount,
  )
    .then((state) => {
      if (generation === tourGeneration && tour.enabled && game.mode === 'off')
        tour.observe(findColonies(state, colonyRadius()), state.episodes);
    })
    .catch(() => {
      /* Geographic stops remain available during a missed readback. */
    })
    .finally(() => {
      samplingTour = false;
    });
}

function refreshHalo() {
  if (!rescuedEpisodes.size && !incidentEpisodes.size) {
    haloMembers?.destroy();
    haloMembers = undefined;
    haloGroup = undefined;
    return;
  }
  if (
    !device ||
    !uniformBuffer ||
    !watermanFwdTexture ||
    !particleBuffers ||
    !lifeStateBuffers ||
    !gameTracking
  )
    return;
  if (!haloPipeline) {
    const module = device.createShaderModule({
      code: structsShader + '\n' + projectionShader + '\n' + haloShader,
    });
    haloPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vertex',
        buffers: [
          {
            arrayStride: particleStride,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          {
            arrayStride: lifeMdpStride,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 1, offset: 12, format: 'float32' }],
          },
          {
            arrayStride: 4,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 2, offset: 0, format: 'uint32' }],
          },
          {
            arrayStride: 4,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 3, offset: 0, format: 'uint32' }],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: 'fragment',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
    });
  }
  const members = new Uint32Array(particleAmt);
  for (const [index, episode] of rescuedEpisodes) members[index] = episode + 1;
  for (const [index, episode] of incidentEpisodes) members[index] = episode + 1;
  haloMembers?.destroy();
  haloMembers = device.createBuffer({
    label: 'colonyHaloEpisodes',
    size: members.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(haloMembers, 0, members);
  haloGroup = device.createBindGroup({
    layout: haloPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: watermanFwdTexture.createView() },
    ],
  });
}

const game = new TemporalGame({
  ready: () =>
    Boolean(device && particleBuffers && lifeStateBuffers && gameTracking),
  snapshot(tickNumber) {
    return readCheckpoint(
      device!,
      particleBuffers![alternate],
      lifeStateBuffers![alternate],
      gameTracking!,
      tickNumber,
      frameCount,
      {
        valence: valenceBuffer!,
        bioForce: bioForceBuffer!,
      },
    );
  },
  restore(state) {
    if (!device || !particleBuffers || !lifeStateBuffers || !gameTracking)
      return;
    tooltipManager.invalidateInspectionFrame();
    device.queue.writeBuffer(particleBuffers[alternate], 0, state.particles);
    const pCount = Math.floor(state.life.length / 4);
    const combinedLife = new Float32Array(pCount * 12);
    const combinedU32 = new Uint32Array(combinedLife.buffer);
    const mdpWords = state.inspection?.mdp;
    for (let i = 0; i < pCount; i++) {
      const s = i * 4;
      const d = i * 12;
      combinedLife[d + 0] = state.life[s + 0];
      combinedLife[d + 1] = state.life[s + 1];
      combinedLife[d + 2] = state.life[s + 2];
      combinedLife[d + 3] = state.life[s + 3];
      if (mdpWords && mdpWords.length >= (i + 1) * 8) {
        const ms = i * 8;
        for (let k = 0; k < 8; k++) combinedU32[d + 4 + k] = mdpWords[ms + k];
      }
    }
    device.queue.writeBuffer(lifeStateBuffers[alternate], 0, combinedLife);
    device.queue.writeBuffer(gameTracking, 0, state.episodes);
    device.queue.writeBuffer(gameTracking, state.episodes.byteLength, state.deathHistory);
    if (state.inspection && valenceBuffer && bioForceBuffer) {
      device.queue.writeBuffer(valenceBuffer, 0, state.inspection.valence);
      device.queue.writeBuffer(bioForceBuffer, 0, state.inspection.bioForce);
    }
    frameCount = state.frame;
  },
  step: stepWorld,
  forceStarted: () => tooltipManager.cancelInspection(),
  focus(point) {
    savedCamera = copyView(viewState);
    const [longitude, latitude] = coordinates(point);
    minimizeDisplay();
    moveCamera({ ...viewState, projectionId: 'orthographic', rotate: [-longitude, -latitude, 0],
      scale: Math.min(viewState.width, viewState.height) * 2.65, pan: [0, -viewState.height * 0.06] }, 1450);
  },
  focusing: () => camera.active,
  highlight(colony, baseline) {
    incidentEpisodes = new Map(
      colony.members.map((index) => [index, baseline.episodes[index]]),
    );
    refreshHalo();
  },
  continueRescue(colony, baseline, outcome) {
    for (const index of colony.members) {
      if (
        isAlive(outcome, index) &&
        outcome.episodes[index] === baseline.episodes[index]
      )
        rescuedEpisodes.set(index, outcome.episodes[index]);
    }
    savedCamera = undefined;
    tour.rescue();
    moveCamera({ ...viewState, scale: viewState.scale * 0.7, pan: [0, 0] }, 1800);
    displayCamera();
  },
  observeColonies(colonies, state) {
    tour.observe(colonies, state.episodes);
  },
  leaveFocus() {
    if (incidentEpisodes.size) {
      incidentEpisodes.clear();
      refreshHalo();
    }
    if (!savedCamera) return;
    moveCamera(savedCamera, 2100);
    savedCamera = undefined;
  },
  project(point) {
    const location = coordinates(point);
    const screen = d3Proj(location);
    if (!screen || !d3Proj.invert) return null;
    const visible = d3Proj.invert(screen);
    if (
      !visible ||
      Math.abs(((visible[0] - location[0] + 540) % 360) - 180) > 0.1 ||
      Math.abs(visible[1] - location[1]) > 0.1
    )
      return null;
    return { x: screen[0], y: screen[1], scale: viewState.scale };
  },
  pointer(x, y) {
    const location = d3Proj.invert?.([x, y]);
    if (!location) return null;
    const check = d3Proj(location);
    if (!check || Math.hypot(check[0] - x, check[1] - y) > 3) return null;
    const lon = (location[0] * Math.PI) / 180,
      lat = (location[1] * Math.PI) / 180;
    return [
      Math.cos(lat) * Math.cos(lon),
      Math.cos(lat) * Math.sin(lon),
      Math.sin(lat),
    ];
  },
  groupRadius: colonyRadius,
  settingsLocked(locked, incident = false) {
    if (incidentCameraLocked !== incident) {
      incidentCameraLocked = incident;
      lockGlobeInput(incident);
      tour.pauseClock();
    }
    advanced.disabled = incident;
    particlesBinding.disabled = incident;
    if (locked === freezeGameWeather) return;
    freezeGameWeather = locked;
    tourGeneration++;
    syncEpisodeTracking();
    if (!locked) {
      if (currentOceanGrid) uploadOceanToGpu(currentOceanGrid);
      uploadWindToGpu(currentWindGrid);
      uploadUvToGpu();
      uploadHydroToGpu();
    }
  },
});

document.getElementById('tour-toggle')!.onclick = () => {
  if (incidentCameraLocked) return;
  tour.setEnabled(!tour.enabled);
  saveMode('tour', tour.enabled);
  tourGeneration++;
  nextTourSampleMs = 0;
  syncEpisodeTracking();
  displayCamera();
};
displayCamera();

const cyberSynth = new GlobeCyberSynth();
cyberSynth.setVolume(params.synthVol);
cyberSynth.setEnabled(modeEnabled('music'), false);
const synthSpectrogram = new SynthSpectrogramOverlay();
const synthSpecSnapshot = createSpectrogramSnapshot();
const windStreamBands = new Float32Array(8);
const oceanStreamBands = new Float32Array(8);
synthSpectrogram.onStreamlinesToggle((enabled) => {
  cyberSynth.setStreamlinesEnabled(enabled);
});
const musicToggleBtn = document.getElementById('music-toggle') as HTMLButtonElement | null;
let lastAudioObserveMs = 0;
let audioObserveInFlight = false;
let lastAudioObservation: ScreenAudioObservation | null = null;
let audioParticleStage: GPUBuffer | undefined;
let audioLifeStage: GPUBuffer | undefined;
let audioStageCount = 0;

function syncMusicButtonUi(rms = 0) {
  if (!musicToggleBtn) return;
  const enabled = cyberSynth.isEnabled();
  const running = cyberSynth.isRunning();
  musicToggleBtn.setAttribute('aria-pressed', String(enabled));
  musicToggleBtn.dataset.audioState = !enabled ? 'muted' : running ? 'running' : 'locked';
  musicToggleBtn.style.setProperty('--rms', Math.min(1, rms * 3.5).toFixed(3));
}
syncMusicButtonUi(0);

function setCyberSynthEnabled(on: boolean, fromGesture = false) {
  cyberSynth.setEnabled(on, fromGesture);
  saveMode('music', on);
  syncMusicButtonUi(0);
}

if (musicToggleBtn) {
  musicToggleBtn.onclick = () => {
    if (cyberSynth.isEnabled() && !cyberSynth.isRunning()) {
      setCyberSynthEnabled(true, true);
    } else {
      setCyberSynthEnabled(!cyberSynth.isEnabled(), true);
    }
  };
}

for (const evType of ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'touchend', 'click', 'keydown'] as const) {
  window.addEventListener(
    evType,
    (ev) => {
      if (ev.target instanceof Element && ev.target.closest('#music-toggle')) return;
      if (cyberSynth.isEnabled() && cyberSynth.getState() !== 'running') {
        void cyberSynth.unlock().then(() => syncMusicButtonUi(0));
      }
    },
    { capture: true, passive: true },
  );
}

function sampleCenterEnvForAudio(): ScreenEnvObservation {
  const ll = d3Proj.invert?.([viewState.width * 0.5, viewState.height * 0.5]);
  const lonDeg = ll && Number.isFinite(ll[0]) ? ll[0] : 0;
  const latDeg = ll && Number.isFinite(ll[1]) ? ll[1] : 0;

  const [, , windMag] = sampleWindLonLat(currentWindGrid, lonDeg, latDeg);
  const [, airSfcK] = sampleWindTempLonLat(currentWindGrid, lonDeg, latDeg);
  const windSpec = WIND_LEVELS.surface;
  const windSpeedNorm = Math.max(0, Math.min(1, windMag / Math.max(windSpec.streamMax, 8)));

  let oceanFraction = 0.5;
  let oceanSpeedNorm = 0.15;
  let habTempK = airSfcK;
  if (currentOceanGrid) {
    const [, , curMag, curMask] = sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg);
    oceanFraction = Math.max(0, Math.min(1, curMask));
    oceanSpeedNorm = Math.max(0, Math.min(1, curMag / Math.max(OCEAN_CURRENTS_SPEC.streamMax, 0.4)));
    const [sstK, sstMask] = sampleOceanSstLonLat(currentOceanGrid, lonDeg, latDeg);
    if (sstMask > 0.35 && oceanFraction > 0.35) {
      habTempK = habTempK * (1 - oceanFraction * 0.7) + sstK * (oceanFraction * 0.7);
    }
  }
  const tempNorm = Math.max(0, Math.min(1, (habTempK - 245) / 68));

  const uvSample = sampleLocalUvAtLonLat(lonDeg, latDeg, uvUtcOverrideMs ?? Date.now());
  const nightFactor = Math.max(0, Math.min(1, 1 - uvSample.uviNow / 0.85));

  const hydroSample = sampleHydroLonLat(currentHydroGrid, lonDeg, latDeg);
  const rhNorm = Math.max(0, Math.min(1, hydroSample.rhPct / 100));
  const precipNorm = Math.max(0, Math.min(1, hydroSample.precip3hMm / 8));

  const elevKm = sampleElevationKmLonLat(currentUvGrid, lonDeg, latDeg);
  const elevKmNorm = Math.max(0, Math.min(1, Math.max(0, elevKm) / 4.5));
  const ridgeMask = Math.max(0, Math.min(1, sampleRidgeMaskLonLat(lonDeg, latDeg, GRID_TEX_W, GRID_TEX_H)));

  return {
    windSpeedNorm,
    oceanFraction,
    oceanSpeedNorm,
    tempNorm,
    nightFactor,
    rhNorm,
    precipNorm,
    elevKmNorm,
    ridgeMask,
  };
}

function observeGlobeAudioCenter(nowMs: number) {
  if (
    !cyberSynth.isEnabled() ||
    cyberSynth.getState() !== 'running' ||
    nowMs - lastAudioObserveMs < 640
  ) {
    return;
  }
  if (!device || !particleBuffers || !lifeStateBuffers) {
    lastAudioObserveMs = nowMs;
    const envCopy = sampleCenterEnvForAudio();
    const obs = observeScreenAudioState(
      new Float32Array(0),
      new Float32Array(0),
      0,
      elements,
      copyView(viewState),
      d3Proj,
      envCopy,
      computeBioSeedHash(params.seed, oceanSeed),
      document.hidden,
    );
    obs.intensity = Math.max(0.24, Math.min(1, 0.25 + 0.32 * envCopy.windSpeedNorm + 0.22 * envCopy.oceanSpeedNorm + 0.18 * envCopy.ridgeMask));
    lastAudioObservation = obs;
    cyberSynth.setObservation(obs);
    return;
  }
  if (audioObserveInFlight) {
    return;
  }
  lastAudioObserveMs = nowMs;
  const n = particleAmt;
  if (n <= 0) return;

  if (audioStageCount !== n || !audioParticleStage || !audioLifeStage) {
    audioParticleStage?.destroy();
    audioLifeStage?.destroy();
    audioParticleStage = device.createBuffer({
      size: n * particleStride,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'audioParticleStage',
    });
    audioLifeStage = device.createBuffer({
      size: n * lifeMdpStride,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'audioLifeStage',
    });
    audioStageCount = n;
  }

  const pStage = audioParticleStage;
  const lStage = audioLifeStage;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(particleBuffers[alternate], 0, pStage, 0, n * particleStride);
  enc.copyBufferToBuffer(lifeStateBuffers[alternate], 0, lStage, 0, n * lifeMdpStride);
  device.queue.submit([enc.finish()]);

  audioObserveInFlight = true;
  const viewCopy = copyView(viewState);
  const envCopy = sampleCenterEnvForAudio();
  const seedHash = computeBioSeedHash(params.seed, oceanSeed);
  const paused = document.hidden;

  void Promise.all([pStage.mapAsync(GPUMapMode.READ), lStage.mapAsync(GPUMapMode.READ)])
    .then(() => {
      const pF32 = new Float32Array(pStage.getMappedRange());
      const lF32 = new Float32Array(lStage.getMappedRange());
      const obs = observeScreenAudioState(
        pF32,
        lF32,
        n,
        elements,
        viewCopy,
        d3Proj,
        envCopy,
        seedHash,
        paused,
      );
      lastAudioObservation = obs;
      cyberSynth.setObservation(obs);
    })
    .catch(() => {
      // Ignore transient device loss or reset overlaps
    })
    .finally(() => {
      try { pStage.unmap(); } catch { /* ignore */ }
      try { lStage.unmap(); } catch { /* ignore */ }
      audioObserveInFlight = false;
    });
}

document.addEventListener('visibilitychange', () => {
  camera.pauseClock();
  tour.pauseClock();
  cyberSynth.handleVisibilityChange(document.hidden);
});
window.addEventListener('pagehide', () => {
  camera.pauseClock();
  tour.pauseClock();
  cyberSynth.handleVisibilityChange(true);
});
window.addEventListener('pageshow', () => {
  if (!document.hidden) cyberSynth.handleVisibilityChange(false);
});
window.addEventListener('focus', () => {
  if (!document.hidden) cyberSynth.handleVisibilityChange(false);
});

const tooltipManager = new TooltipManager({
  isEnabled: () => earthParams.tooltips && game.canInspect,
  inspectionPaused: () => game.inspectionPaused,
  getProj: () => d3Proj,
  getViewLayer: () => earthParams.viewLayer,
  isBioLinked: () => earthParams.bioLink && earthParams.bioRate > 0,
  getAirRMax: () => 1 / Math.max(optionParams.r, 0.5),
  getOceanRMax: () => 1 / Math.max(oceanOptionParams.r, 0.5),
  getWorldSize: () => Math.max(optionParams.worldSize, 0.25),
  getBeta: () => optionParams.beta,
  getTemperature: () => optionParams.temperature,
  getScale: () => viewState.scale,
  getOceanMask: (u) => {
    if (!currentOceanGrid) return 1;
    const z = Math.max(-1, Math.min(1, u[2] ?? 0));
    const latDeg = Math.asin(z) * (180 / Math.PI);
    const lonDeg = Math.atan2(u[1] ?? 0, u[0] ?? 1) * (180 / Math.PI);
    return sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg)[3];
  },
  getOceanSpeedMs: (u) => {
    if (!currentOceanGrid || !earthParams.oceanMix) return 0;
    const z = Math.max(-1, Math.min(1, u[2] ?? 0));
    const latDeg = Math.asin(z) * (180 / Math.PI);
    const lonDeg = Math.atan2(u[1] ?? 0, u[0] ?? 1) * (180 / Math.PI);
    return (
      sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg)[2] *
      earthParams.oceanStrength
    );
  },
  getWindSpeedMs: (u) => {
    if (!earthParams.vortex) return 0;
    const z = Math.max(-1, Math.min(1, u[2] ?? 0));
    const latDeg = Math.asin(z) * (180 / Math.PI);
    const lonDeg = Math.atan2(u[1] ?? 0, u[0] ?? 1) * (180 / Math.PI);
    return (
      sampleWindLonLat(currentWindGrid, lonDeg, latDeg)[2] *
      earthParams.windStrength
    );
  },
  getLocalTemps: (u) => sampleLocalTempsAtUnit(u),
  getLocalUv: (u) => sampleLocalUvAtUnit(u),
  getLocalHydro: (u) => sampleLocalHydroAtUnit(u),
  getEnvColorRgb: (kind, value) => {
    if (kind === 'air') return airTempColorRgb(273.15 + value);
    if (kind === 'sst') return sstColorRgb(273.15 + value);
    if (kind === 'uv') return uvIndexColorRgb(value);
    if (kind === 'rh') return rhColorRgb(value / 100);
    return precip3hColorRgb(value);
  },
  getMineralOpacity: () => earthParams.mineralOpacity,
  requestSnapshot: captureHoverSnapshot,
});

canvas.addEventListener('pointermove', (ev) => {
  if (!d3Proj.invert || ev.pointerType === 'touch') return;
  const ll = d3Proj.invert([ev.clientX, ev.clientY]);
  if (!ll || !Number.isFinite(ll[0]) || !Number.isFinite(ll[1])) {
    if (pointerHoverTempActive) {
      pointerHoverTempActive = false;
      tempTimeBadge.textContent = defaultTempBadgeText;
      uvTimeBadge.textContent = defaultUvBadgeText;
    }
    return;
  }
  const chk = d3Proj(ll);
  if (!chk || Math.hypot(chk[0] - ev.clientX, chk[1] - ev.clientY) > 12) {
    if (pointerHoverTempActive) {
      pointerHoverTempActive = false;
      tempTimeBadge.textContent = defaultTempBadgeText;
      uvTimeBadge.textContent = defaultUvBadgeText;
    }
    return;
  }
  pointerHoverTempActive = true;
  const t = sampleLocalTempsAtLonLat(ll[0], ll[1]);
  const aSign = t.airC >= 0 ? '+' : '';
  const sstPart =
    t.sstC !== null ? ` · SST ${t.sstC >= 0 ? '+' : ''}${t.sstC.toFixed(1)}°C` : '';
  tempTimeBadge.textContent = `${t.airLabel} ${aSign}${t.airC.toFixed(1)}°C${sstPart}`;
  const uv = sampleLocalUvAtLonLat(ll[0], ll[1]);
  uvTimeBadge.textContent = `UVI ${uv.uviNow.toFixed(1)} (${uv.category}) · Noon ${uv.uviNoon.toFixed(1)} · O₃ ${Math.round(uv.ozoneDU)} DU`;
});

canvas.addEventListener('pointerleave', () => {
  if (pointerHoverTempActive) {
    pointerHoverTempActive = false;
    tempTimeBadge.textContent = defaultTempBadgeText;
    uvTimeBadge.textContent = defaultUvBadgeText;
  }
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
  getMineralLink: () => earthParams.mineralLink,
  setMineralLink: (on: boolean) => {
    earthParams.mineralLink = on;
    mineralLinkBinding.refresh();
    syncBioUniformState();
  },
  getMineralRate: () => earthParams.mineralRate,
  setMineralRate: (rate: number) => {
    earthParams.mineralRate = rate;
    mineralRateBinding.refresh();
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
    tempColors: earthParams.tempColors,
    uvColors: earthParams.uvColors,
    lifeOpacity: earthParams.lifeOpacity,
    mineralOpacity: earthParams.mineralOpacity,
  }),
  setChemOpacity: (val: number) => {
    earthParams.chemOpacity = clampOpacity(val);
    chemOpacityBinding.refresh();
  },
  setLifeOpacity: (val: number) => {
    earthParams.lifeOpacity = clampOpacity(val);
    lifeOpacityBinding.refresh();
  },
  setMineralOpacity: (val: number) => {
    earthParams.mineralOpacity = clampOpacity(val);
    mineralOpacityBinding.refresh();
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
  setTempColors: (val: boolean | number) => {
    earthParams.tempColors = clampOpacity(val);
    tempColorsBinding.refresh();
  },
  setUvColors: (val: boolean | number) => {
    earthParams.uvColors = clampOpacity(val);
    uvColorsBinding.refresh();
  },
  getLocalTemp: () => earthParams.localTemp,
  setLocalTemp: (on: boolean) => {
    earthParams.localTemp = on;
    localTempBinding.refresh();
    syncBioUniformState();
  },
  getTempStrength: () => earthParams.tempStrength,
  setTempStrength: (val: number) => {
    earthParams.tempStrength = Math.max(0, Math.min(2, val));
    tempStrengthBinding.refresh();
    syncBioUniformState();
  },
  getLocalUv: () => earthParams.localUv,
  setLocalUv: (on: boolean) => {
    earthParams.localUv = on;
    localUvBinding.refresh();
    syncBioUniformState();
  },
  getUvStrength: () => earthParams.uvStrength,
  setUvStrength: (val: number) => {
    earthParams.uvStrength = Math.max(0, Math.min(2, val));
    uvStrengthBinding.refresh();
    syncBioUniformState();
  },
  getTempAtLonLat: (lonDeg: number, latDeg: number) =>
    sampleLocalTempsAtLonLat(lonDeg, latDeg),
  getTempInfo: () => ({
    airLevel: currentWindGrid.level,
    airLabel: currentWindGrid.tempLabel || '2m Air',
    airRefTime: currentWindGrid.validTime || currentWindGrid.refTime,
    sstRefTime: currentOceanGrid?.sstRefTime ?? '',
    sstSource: currentOceanGrid?.sstSource ?? 'Climatology SST',
    isLive: currentWindGrid.isLive,
  }),
  getUvAtLonLat: (lonDeg: number, latDeg: number, utcMs?: number) =>
    sampleLocalUvAtLonLat(lonDeg, latDeg, utcMs),
  setUvUtcTime: (isoOrMs: string | number | null) => {
    if (isoOrMs === null) {
      uvUtcOverrideMs = null;
    } else if (typeof isoOrMs === 'number' && Number.isFinite(isoOrMs)) {
      uvUtcOverrideMs = isoOrMs;
    } else if (typeof isoOrMs === 'string') {
      const parsed = Date.parse(isoOrMs);
      uvUtcOverrideMs = Number.isFinite(parsed) ? parsed : null;
    }
    uploadUvToGpu();
  },
  getUvInfo: () => ({
    cpcRefTime: currentUvGrid?.cpcRefTime ?? '',
    gfsRefTime: currentUvGrid?.gfsRefTime ?? '',
    gfsValidTime: currentUvGrid?.gfsValidTime ?? '',
    source: currentUvGrid?.source ?? 'Solar Zenith Model',
    isLive: Boolean(currentUvGrid?.isLive),
  }),
  refreshWeatherData: (opts?: { forceSnapshots?: boolean }) =>
    backgroundRefreshWeather({
      forceGfs: true,
      forceSnapshots: opts?.forceSnapshots ?? true,
    }),
  getWeatherRefreshState: () => ({
    lastGfsCheckAt,
    lastSnapshotCheckAt,
    gfsIntervalMs: GFS_AUTO_REFRESH_MS,
    snapshotIntervalMs: DAILY_SNAPSHOT_REFRESH_MS,
  }),
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
    if (incidentCameraLocked) return;
    const destination = copyView(viewState);
    if (patch.projectionId) destination.projectionId = patch.projectionId;
    if (patch.rotate) destination.rotate = [...patch.rotate];
    if (patch.pan) destination.pan = [...patch.pan];
    if (patch.scale && patch.scale > 0) destination.scale = patch.scale;
    moveCamera(destination, 1000, true);
  },
  setAudioEnabled: (on: boolean) => setCyberSynthEnabled(on, true),
  getAudioStatus: () => ({
    enabled: cyberSynth.isEnabled(),
    state: cyberSynth.getState(),
    volume: cyberSynth.getVolume(),
    lastObservation: lastAudioObservation,
  }),
  readParticles: (count: number = 16) =>
    readParticleBuffer(particleBuffers, alternate, particleAmt, count),
  readOceanParticles: (count: number = 16) =>
    readParticleBuffer(particleBuffers, alternate, particleAmt, count),
  readValence: (count: number = 16) => readVec4Buffer(valenceBuffer, particleAmt, count),
  readOceanState: (count: number = 16) =>
    readVec4Buffer(lifeStateBuffers?.[alternate], particleAmt, count, 12),
  readAirBioReact: (count: number = 16) =>
    readVec4Buffer(bioForceBuffer, particleAmt, count),
  readLifeState: (count: number = 16) =>
    readVec4Buffer(lifeStateBuffers?.[alternate], particleAmt, count, 12),
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
  for (const c of [bgCanvas, mapCanvas, oceanCanvas, windCanvas, canvas]) {
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }
  if (viewState.width !== w || viewState.height !== h) {
    if (incidentCameraLocked) {
      viewState.scale *= Math.min(w, h) / Math.min(viewState.width, viewState.height);
      viewState.pan = [viewState.pan[0] * w / viewState.width, viewState.pan[1] * h / viewState.height];
    }
    viewState.width = w;
    viewState.height = h;
    updateProjectionState(false);
  }
}

const webGpuNoticeEl = document.getElementById('webgpu-notice') as HTMLElement | null;
const webGpuNoticeTextEl = document.getElementById('webgpu-notice-text') as HTMLElement | null;
const webGpuNoticeCloseEl = document.getElementById('webgpu-notice-close') as HTMLButtonElement | null;
webGpuNoticeCloseEl?.addEventListener('click', (ev) => {
  ev.stopPropagation();
  if (webGpuNoticeEl) webGpuNoticeEl.hidden = true;
});

function showWebGpuNotice(reason: string) {
  game.unavailable(reason);
  if (webGpuNoticeTextEl) {
    const isFirefox = /firefox|fxios/i.test(navigator.userAgent);
    const isChromium = /chrome|chromium|edg/i.test(navigator.userAgent);
    const hasGpuApi = typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
    const hint = isFirefox
      ? 'Open in Chrome, or in Firefox Nightly set about:config -> gfx.webgpu.ignore-blocklist = true (and dom.webgpu.enabled = true) then restart.'
      : isChromium && hasGpuApi
        ? 'Chrome blocked the WebGPU GPU adapter (check chrome://gpu, disable Android Advanced Protection for Chrome, or enable chrome://flags/#enable-unsafe-webgpu).'
        : 'Open in Chrome/Edge 121+ with hardware acceleration (or enable WebGPU in browser flags).';
    webGpuNoticeTextEl.textContent = `${reason} ${hint} Showing 2D globe, wind/ocean currents & synth.`;
  }
  if (webGpuNoticeEl) {
    webGpuNoticeEl.hidden = false;
  }
}

const windBootPromise = loadFallbackWind().catch(() => null);
const oceanBootPromise = loadOceanCurrents().catch(() => null);
const uvBootPromise = loadCpcUvBaseline().catch(() => null);

syncCanvasDimensions();
updateProjectionState();
drawScaleBar(WIND_LEVELS.surface.scaleMax);
refreshHudVisibility();
bindConfigPersistence(() => {
  if (!suppressPaneEvents) scheduleSaveConfig();
});
configPersistenceReady = true;
syncUrlStateNow();

loadEarthFeatures()
  .then((feats) => {
    earthFeatures = feats;
    mapDirty = true;
  })
  .catch(() => {
    // Non-fatal if coastline JSON fails to load
  });

void Promise.all([oceanBootPromise, uvBootPromise, windBootPromise]).then(
  ([loadedOcean, loadedUv, loadedWind]) => {
    if (loadedWind) {
      currentWindGrid = loadedWind;
      if (device) uploadWindToGpu(loadedWind);
      refreshHudBadges(loadedWind);
    }
    if (loadedOcean) {
      currentOceanGrid = loadedOcean;
      if (device) uploadOceanToGpu(loadedOcean);
      refreshOceanHud(loadedOcean);
    } else {
      earthParams.oceanStatus = 'Ocean data unavailable';
      oceanStatusBinding.refresh();
    }
    if (loadedUv) {
      currentUvGrid = loadedUv;
      if (device) {
        uploadUvToGpu();
        uploadTerrainToGpu();
      }
    }
  },
);
loadSurfaceWind();
window.setInterval(() => {
  if (uvUtcOverrideMs === null && device) {
    uploadUvToGpu();
  }
}, 45_000);
window.setInterval(() => {
  if (!document.hidden) {
    void backgroundRefreshWeather();
  }
}, GFS_AUTO_REFRESH_MS);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    if (uvUtcOverrideMs === null && device) {
      uploadUvToGpu();
    }
    void backgroundRefreshWeather();
  }
});

(async () => {
  try {
    if (!navigator.gpu) {
      showWebGpuNotice('This browser does not expose WebGPU.');
      return;
    }
    const tryAdapter = async (opts?: GPURequestAdapterOptions) => {
      try {
        return await navigator.gpu.requestAdapter(opts);
      } catch {
        return null;
      }
    };
    let adapter =
      (await tryAdapter({ featureLevel: 'compatibility' })) ??
      (await tryAdapter({ powerPreference: 'high-performance' })) ??
      (await tryAdapter());
    for (let retry = 0; retry < 4 && !adapter; retry++) {
      await new Promise((r) => setTimeout(r, 200));
      adapter =
        (await tryAdapter({ featureLevel: 'compatibility' })) ??
        (await tryAdapter());
    }
    if (!adapter) {
      showWebGpuNotice('WebGPU GPU adapter unavailable on this browser.');
      return;
    }

    device = await requestTimestamps(adapter);
    device.lost.then(() => {
      deviceLost = true;
      showWebGpuNotice('GPU connection lost. Reload to restart simulation.');
    });

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
    if (!bgContext || !context) {
      showWebGpuNotice('Unable to initialize WebGPU canvas contexts.');
      return;
    }

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    try {
      bgContext.configure({
        device,
        format: presentationFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
        alphaMode: 'opaque',
      });
      bgCanCopyDst = true;
    } catch {
      bgContext.configure({
        device,
        format: presentationFormat,
        alphaMode: 'opaque',
      });
      bgCanCopyDst = false;
    }
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
    size: bioUniformFloats * 4,
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
  tempTexture = device.createTexture({
    size: { width: GRID_TEX_W, height: GRID_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'tempTexture',
  });
  uploadTempToGpu();
  uvTexture = device.createTexture({
    size: { width: GRID_TEX_W, height: GRID_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'uvTexture',
  });
  uploadUvToGpu();
  hydroTexture = device.createTexture({
    size: { width: GRID_TEX_W, height: GRID_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'hydroTexture',
  });
  uploadHydroToGpu();
  terrainTexture = device.createTexture({
    size: { width: GRID_TEX_W, height: GRID_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'terrainTexture',
  });
  uploadTerrainToGpu();
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
      { binding: 4, resource: tempTexture.createView() },
      { binding: 5, resource: uvTexture.createView() },
      { binding: 6, resource: hydroTexture.createView() },
      { binding: 7, resource: terrainTexture.createView() },
    ],
  });

  const renderModule = device.createShaderModule({
    code: `${structsShader}\n${projectionShader}\n${renderShader}`,
  });

  const particleVertexBuffers: GPUVertexBufferLayout[] = [
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
      arrayStride: lifeMdpStride,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 5, offset: 0, format: 'float32x4' },
      ],
    },
  ];

  const renderTargets: GPUColorTargetState[] = [
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
  ];

  mineralRenderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vertexMinerals',
      buffers: particleVertexBuffers,
    },
    fragment: {
      module: renderModule,
      entryPoint: 'fragment',
      targets: renderTargets,
    },
    primitive: { topology: 'triangle-strip' },
  });

  renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vertexForeground',
      buffers: particleVertexBuffers,
    },
    fragment: {
      module: renderModule,
      entryPoint: 'fragment',
      targets: renderTargets,
    },
    primitive: { topology: 'triangle-strip' },
  });

  slimeFieldBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  const slimeFieldPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [slimeFieldBindGroupLayout],
  });
  const slimeFieldModule = device.createShaderModule({
    code: `${structsShader}\n${projectionShader}\n${slimeFieldShader}`,
  });
  const slimeFieldTargets: GPUColorTargetState[] = [
    {
      format: 'rgba16float',
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      },
    },
    {
      format: 'rgba16float',
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      },
    },
  ];
  slimeNodePipeline = device.createRenderPipeline({
    layout: slimeFieldPipelineLayout,
    vertex: {
      module: slimeFieldModule,
      entryPoint: 'vertexNode',
      buffers: particleVertexBuffers,
    },
    fragment: {
      module: slimeFieldModule,
      entryPoint: 'fragment',
      targets: slimeFieldTargets,
    },
    primitive: { topology: 'triangle-list' },
  });
  slimeBridgePipeline = device.createRenderPipeline({
    layout: slimeFieldPipelineLayout,
    vertex: {
      module: slimeFieldModule,
      entryPoint: 'vertexBridge',
      buffers: particleVertexBuffers,
    },
    fragment: {
      module: slimeFieldModule,
      entryPoint: 'fragment',
      targets: slimeFieldTargets,
    },
    primitive: { topology: 'triangle-list' },
  });

  const slimeCompositeModule = device.createShaderModule({
    code: `${structsShader}\n${slimeCompositeShader}`,
  });
  slimeCompositePipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: slimeCompositeModule,
      entryPoint: 'vertex',
    },
    fragment: {
      module: slimeCompositeModule,
      entryPoint: 'fragment',
      targets: renderTargets,
    },
    primitive: { topology: 'triangle-list' },
  });
  slimeFieldSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
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

  const [loadedOcean, loadedUv, loadedWind] = await Promise.all([
    oceanBootPromise,
    uvBootPromise,
    windBootPromise,
  ]);
  if (loadedWind) {
    currentWindGrid = loadedWind;
    uploadWindToGpu(loadedWind);
    refreshHudBadges(loadedWind);
  }
  if (loadedOcean) {
    currentOceanGrid = loadedOcean;
    uploadOceanToGpu(loadedOcean);
    refreshOceanHud(loadedOcean);
  }
  if (loadedUv) {
    currentUvGrid = loadedUv;
    uploadUvToGpu();
    uploadTerrainToGpu();
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
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'WebGPU initialization failed.';
    showWebGpuNotice(msg);
  }
})();

let activeWorldRefs: UnifiedEngineRefs | undefined;
const initializedEngines = new Set<string>();
const boundEngines = new Set<string>();
function prepareEngine() {
  if (!device || !activeWorldRefs || boundEngines.has(engine)) return;
  if (!initializedEngines.has(engine)) {
    engines[engine].setup(device);
    initializedEngines.add(engine);
  }
  engines[engine].start(device, activeWorldRefs);
  boundEngines.add(engine);
}

function tick(commandEncoder: GPUCommandEncoder) {
  if (!device || !particleBuffers) return;
  prepareEngine();
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

  const bgTargetTex = bgContext.getCurrentTexture();
  if (!bgCanCopyDst) {
    const bgPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: bgTargetTex.createView(),
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
  } else {
    if (
      !bgCacheTexture ||
      bgCacheTexture.width !== bgTargetTex.width ||
      bgCacheTexture.height !== bgTargetTex.height
    ) {
      bgCacheTexture?.destroy();
      bgCacheTexture = device.createTexture({
        size: { width: bgTargetTex.width, height: bgTargetTex.height },
        format: navigator.gpu.getPreferredCanvasFormat(),
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        label: 'bgCacheTexture',
      });
      bgCacheKey = '';
    }

    const lens = tooltipManager.getLensState();
    const nextBgKey = [
      bgTargetTex.width,
      bgTargetTex.height,
      viewSignature,
      bgDataVersion,
      earthParams.viewLayer,
      earthParams.terrainOpacity.toFixed(3),
      earthParams.nightShading.toFixed(3),
      earthParams.oceanColors.toFixed(3),
      earthParams.windColors.toFixed(3),
      earthParams.tempColors.toFixed(3),
      earthParams.uvColors.toFixed(3),
      earthParams.rhColors.toFixed(3),
      earthParams.precipColors.toFixed(3),
      lens.active ? `${Math.round(lens.x)},${Math.round(lens.y)},${lens.mag.toFixed(2)}` : '0',
    ].join('|');

    if (nextBgKey !== bgCacheKey) {
      const bgPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: bgCacheTexture.createView(),
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
      bgCacheKey = nextBgKey;
    }

    commandEncoder.copyTextureToTexture(
      { texture: bgCacheTexture },
      { texture: bgTargetTex },
      { width: bgTargetTex.width, height: bgTargetTex.height, depthOrArrayLayers: 1 },
    );
  }

  const canvasTargetTex = context.getCurrentTexture();
  const showChem = earthParams.viewLayer !== 'ocean' && earthParams.chemOpacity > 0.001;
  const showLife = earthParams.viewLayer !== 'air' && earthParams.lifeOpacity > 0.001;
  const showMinerals = earthParams.mineralOpacity > 0.001;
  const canRenderSlime =
    showLife &&
    !!slimeNodePipeline &&
    !!slimeBridgePipeline &&
    !!slimeCompositePipeline &&
    !!slimeFieldBindGroup &&
    !!slimeFieldSampler &&
    !!particleBuffers &&
    !!valenceBuffer &&
    !!lifeStateBuffers;

  if (canRenderSlime) {
    const fieldW = Math.max(1, Math.floor(canvasTargetTex.width * 0.5));
    const fieldH = Math.max(1, Math.floor(canvasTargetTex.height * 0.5));
    if (
      !slimeFieldTexture ||
      !membraneFieldTexture ||
      slimeFieldTexture.width !== fieldW ||
      slimeFieldTexture.height !== fieldH ||
      !slimeCompositeBindGroup
    ) {
      slimeFieldTexture?.destroy();
      membraneFieldTexture?.destroy();
      slimeFieldTexture = device.createTexture({
        size: { width: fieldW, height: fieldH },
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        label: 'slimeFieldTexture',
      });
      membraneFieldTexture = device.createTexture({
        size: { width: fieldW, height: fieldH },
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        label: 'membraneFieldTexture',
      });
      slimeCompositeBindGroup = device.createBindGroup({
        layout: slimeCompositePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer! } },
          { binding: 1, resource: slimeFieldTexture.createView() },
          { binding: 2, resource: membraneFieldTexture.createView() },
          { binding: 3, resource: slimeFieldSampler! },
        ],
      });
    }

    const slimePass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: slimeFieldTexture.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: membraneFieldTexture.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    slimePass.setBindGroup(0, slimeFieldBindGroup!);
    slimePass.setVertexBuffer(0, particleBuffers![alternate]);
    slimePass.setVertexBuffer(1, valenceBuffer!);
    slimePass.setVertexBuffer(2, lifeStateBuffers![alternate]);
    slimePass.setPipeline(slimeNodePipeline!);
    slimePass.draw(6, particleAmt, 0, 0);
    slimePass.setPipeline(slimeBridgePipeline!);
    slimePass.draw(12, particleAmt, 0, 0);
    slimePass.end();
  }

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: canvasTargetTex.createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

  const profileRender = (frameCount % 12) === 0;
  const passEncoder = commandEncoder.beginRenderPass(
    profileRender ? linkRenderTimestamp(device, renderPassDescriptor, 'render') : renderPassDescriptor,
  );

  if (particleBuffers && valenceBuffer && lifeStateBuffers) {
    if (showMinerals && mineralRenderPipeline && mineralRenderBindGroup) {
      passEncoder.setPipeline(mineralRenderPipeline);
      passEncoder.setVertexBuffer(0, particleBuffers[alternate]);
      passEncoder.setVertexBuffer(1, valenceBuffer);
      passEncoder.setVertexBuffer(2, lifeStateBuffers[alternate]);
      passEncoder.setBindGroup(0, mineralRenderBindGroup);
      passEncoder.draw(6, particleAmt, 0, 0);
    }
    if (canRenderSlime && slimeCompositePipeline && slimeCompositeBindGroup) {
      passEncoder.setPipeline(slimeCompositePipeline);
      passEncoder.setBindGroup(0, slimeCompositeBindGroup);
      passEncoder.draw(3, 1, 0, 0);
    }
    if ((showChem || showLife) && renderPipeline && renderBindGroup) {
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setVertexBuffer(0, particleBuffers[alternate]);
      passEncoder.setVertexBuffer(1, valenceBuffer);
      passEncoder.setVertexBuffer(2, lifeStateBuffers[alternate]);
      passEncoder.setBindGroup(0, renderBindGroup);
      passEncoder.draw(6, particleAmt, 0, 0);
    }
  }

  if (haloPipeline && haloMembers && haloGroup && gameTracking && particleBuffers && lifeStateBuffers) {
    passEncoder.setPipeline(haloPipeline);
    passEncoder.setBindGroup(0,haloGroup);
    passEncoder.setVertexBuffer(0,particleBuffers[alternate]);
    passEncoder.setVertexBuffer(1,lifeStateBuffers[alternate]);
    passEncoder.setVertexBuffer(2,gameTracking);
    passEncoder.setVertexBuffer(3,haloMembers);
    passEncoder.draw(6,particleAmt);
  }
  if (lensOverlayPipeline && lensOverlayBindGroup) {
    passEncoder.setPipeline(lensOverlayPipeline);
    passEncoder.setBindGroup(0, lensOverlayBindGroup);
    passEncoder.draw(3, 1, 0, 0);
  }

  passEncoder.end();
  if (profileRender) resolveTimestamp(commandEncoder, 'render');
}

function packLifeMetaJs(
  phase: number,
  bioSp: number,
  kinAngle: number,
  encapGene: number = 7,
  graduated: number = 0,
): number {
  if (phase === 0) return 0;
  const twoPi = Math.PI * 2;
  let a = kinAngle % twoPi;
  if (a < 0) a += twoPi;
  const k = Math.floor((a / twoPi) * 256) & 255;
  const g = Math.max(0, Math.min(15, Math.floor(encapGene)));
  const gr = graduated ? 1 : 0;
  return (
    (phase & 3) |
    ((Math.min(255, Math.max(0, bioSp)) & 255) << 2) |
    (k << 10) |
    (g << 19) |
    (gr << 23)
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
    !tempTexture ||
    !uvTexture ||
    !hydroTexture ||
    !terrainTexture ||
    !watermanFwdTexture
  ) {
    return;
  }

  game.reset();
  tooltipManager.invalidateInspectionFrame();
  tourGeneration++;
  tour.resetWorld();
  rescuedEpisodes.clear();
  incidentEpisodes.clear();
  refreshHalo();
  displayCamera();
  gameTracking?.destroy();
  gameTracking = undefined;
  gameGroups = [];
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
      size: particleAmt * lifeMdpStride,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: 'lifeState0',
    }),
    device.createBuffer({
      size: particleAmt * lifeMdpStride,
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
    size: Math.max(
      (chemistryData.length +
        oceanMatrixData.length +
        oceanColourAmt * (4 + MDP_SPECIES_STRIDE)) *
        4,
      16,
    ),
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
        if (sampleOceanLonLat(currentOceanGrid, candLon, candLat)[3] >= 0.50) {
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

  const sampleLandHumidCenter = (): [number, number, number] => {
    let lonDeg = (rng() - 0.5) * 360;
    let latDeg = (rng() - 0.5) * 60;
    for (let t = 0; t < 110; t++) {
      const candLon = (rng() - 0.5) * 360;
      const candLat = Math.asin(rng() * 1.35 - 0.675) * (180 / Math.PI);
      const isLand = !currentOceanGrid || sampleOceanLonLat(currentOceanGrid, candLon, candLat)[3] < 0.38;
      if (!isLand) continue;
      const elev = sampleElevationKmLonLat(currentUvGrid, candLon, candLat);
      if (elev >= 3.2) continue;
      const hydro = currentHydroGrid ? sampleHydroLonLat(currentHydroGrid, candLon, candLat) : { rhPct: 68, precip3hMm: 0.4 };
      const tempK = sampleWindTempLonLat(currentWindGrid, candLon, candLat)[1];
      if ((hydro.precip3hMm >= 0.25 || hydro.rhPct >= 56) && tempK >= 281.0) {
        lonDeg = candLon;
        latDeg = candLat;
        break;
      }
    }
    const lonR = lonDeg * DEG2RAD;
    const latR = latDeg * DEG2RAD;
    const cLat = Math.cos(latR);
    return [cLat * Math.cos(lonR), cLat * Math.sin(lonR), Math.sin(latR)];
  };

  let bioCursor = 0;
  const numLandProducers = Math.round(numBlobs * 0.35);
  if (numBlobs > 0) {
    for (let bi = 0; bi < numBlobs; bi++) {
      const endCursor =
        bi === numBlobs - 1
          ? targetBio
          : Math.min(targetBio, Math.round(((bi + 1) * targetBio) / numBlobs));
      const seedOnLand = bi < numLandProducers;
      const [cx, cy, cz] = seedOnLand ? sampleLandHumidCenter() : sampleOceanCenter();
      const [e1x, e1y, e1z] = unitTangentFromSphere(cx, cy, cz, rng() * Math.PI * 2);
      const e2x = cy * e1z - cz * e1y;
      const e2y = cz * e1x - cx * e1z;
      const e2z = cx * e1y - cy * e1x;
      const kinPhase = rng() * Math.PI * 2;
      const blobSpecies = pickHeterogeneousSpeciesSet(!seedOnLand && bi % 2 === 1);
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

    const el = elements[elemChoice[pi]];
    const lonDeg = Math.atan2(ny, nx) * (180 / Math.PI);
    const latDeg = Math.asin(Math.max(-1, Math.min(1, nz))) * (180 / Math.PI);
    const inSea = currentOceanGrid ? sampleOceanLonLat(currentOceanGrid, lonDeg, latDeg)[3] >= 0.45 : true;
    let initComp = 0;
    let initAlt = 0.08 + 0.48 * rng();
    if (el && el.valence > 0) {
      if (el.atomicNumber === 1) {
        initComp = inSea ? 1 : rng() < 0.25 ? 1 : 0;
      } else if (el.atomicNumber === 6) {
        initComp = inSea && rng() < 0.72 ? 1 : 0;
      } else if (el.atomicNumber === 5) {
        const seaT = currentOceanGrid ? sampleOceanSstLonLat(currentOceanGrid, lonDeg, latDeg)[0] : 288.15;
        initComp = inSea && seaT < 290.0 ? 1 : 0;
      } else if (inSea && el.atomicNumber !== 2 && el.atomicNumber !== 8 && el.atomicNumber !== 14) {
        initComp = 1;
      }
      if (initComp >= 1) initAlt = 0;
    }
    const lBase = pi * 4;
    lifeData[lBase + 0] = 0;
    lifeData[lBase + 1] = initComp;
    lifeData[lBase + 2] = initAlt;
    lifeData[lBase + 3] = 0;
  };

  const freePool = [...biogenicPool.slice(bioCursor), ...abioticPool];
  for (let i = freePool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = freePool[i];
    freePool[i] = freePool[j];
    freePool[j] = tmp;
  }

  const usedInMineral = new Set<number>();
  if (earthParams.mineralLink && particleAmt >= 200) {
    const fitsMinType = (z: number, val: number, mType: number): boolean => {
      if (val <= 0) return false;
      if (mType === 0) return z === 11 || z === 6 || z === 10 || z === 3;
      if (mType === 1) return z === 9 || z === 3 || z === 13 || z === 7 || z === 10;
      if (mType === 2) return z === 5 || z === 6 || z === 10 || z === 4;
      if (mType === 3) return z === 12 || z === 10 || z === 6;
      return z === 11 || z === 6 || z === 1 || z === 10 || z === 5;
    };

    const sampleHabitatCenter = (mode: 'land' | 'coast' | 'ocean' | 'any'): [number, number, number] => {
      let lonDeg = (rng() - 0.5) * 360;
      let latDeg = Math.asin(rng() * 2 - 1) * (180 / Math.PI);
      if (currentOceanGrid && mode !== 'any') {
        for (let t = 0; t < 80; t++) {
          const cLon = (rng() - 0.5) * 360;
          const cLat = Math.asin(rng() * 2 - 1) * (180 / Math.PI);
          const m = sampleOceanLonLat(currentOceanGrid, cLon, cLat)[3];
          const ok =
            (mode === 'land' && m < 0.34) ||
            (mode === 'coast' && m >= 0.36 && m <= 0.72) ||
            (mode === 'ocean' && m >= 0.54);
          if (ok) {
            lonDeg = cLon;
            latDeg = cLat;
            break;
          }
        }
      }
      const lonR = lonDeg * DEG2RAD;
      const latR = latDeg * DEG2RAD;
      const cLat = Math.cos(latR);
      return [cLat * Math.cos(lonR), cLat * Math.sin(lonR), Math.sin(latR)];
    };

    const bins: number[][] = [[], [], [], [], []];
    for (const pi of freePool) {
      const el = elements[elemChoice[pi]];
      if (!el || el.valence <= 0) continue;
      for (let t = 0; t < 5; t++) {
        if (fitsMinType(el.atomicNumber, el.valence, t)) {
          bins[t].push(pi);
        }
      }
    }
    // Interleave signature crustal elements (Si, Na/Cl/F, C/Be/Mg, S/Mg, H) naturally with abundant O
    const sigScore = (mType: number, z: number, pi: number): number => {
      const jitter = ((pi * 2654435761) >>> 0) / 4294967296;
      let bias = 0;
      if (mType === 0) bias = z === 11 ? 0.55 : z === 10 || z === 3 ? 0.45 : 0;
      else if (mType === 1) bias = z === 9 || z === 13 || z === 7 ? 0.60 : z === 3 || z === 10 ? 0.45 : 0;
      else if (mType === 2) bias = z === 5 ? 0.75 : z === 10 ? 0.55 : z === 4 ? 0.35 : 0;
      else if (mType === 3) bias = z === 12 ? 0.90 : z === 10 ? 0.50 : 0;
      else bias = z === 11 ? 0.55 : z === 10 || z === 1 ? 0.40 : 0;
      return bias + jitter * 0.85;
    };
    for (let t = 0; t < 5; t++) {
      bins[t].sort((a, b) => {
        const za = elements[elemChoice[a]]?.atomicNumber ?? 0;
        const zb = elements[elemChoice[b]]?.atomicNumber ?? 0;
        return sigScore(t, zb, b) - sigScore(t, za, a);
      });
    }

    const veinsPerType = Math.max(2, Math.min(10, Math.round((particleAmt / 28000) * 9)));
    const crystalsPerVein = Math.max(4, Math.min(14, Math.round((particleAmt / 28000) * 12)));
    const habitats: ('any' | 'land' | 'coast' | 'ocean')[] = ['any', 'land', 'any', 'ocean', 'coast'];

    for (let mType = 0; mType < 5; mType++) {
      let cursor = 0;
      const pool = bins[mType];
      for (let v = 0; v < veinsPerType; v++) {
        // Place most veins in their stable habitat, plus a few coastal boundary veins actively eroding on Day 0
        const hab: 'any' | 'land' | 'coast' | 'ocean' =
          v === 0 ? 'coast' : habitats[mType];
        const [cx, cy, cz] = sampleHabitatCenter(hab);
        const [e1x, e1y, e1z] = unitTangentFromSphere(cx, cy, cz, rng() * Math.PI * 2);
        const e2x = cy * e1z - cz * e1y;
        const e2y = cz * e1x - cx * e1z;
        const e2z = cx * e1y - cy * e1x;
        const veinAngle = rng() * Math.PI * 2;
        const isBeTrapVein = mType === 2 && v % 3 === 1;
        const isFluorTrapVein = mType === 1 && v % 2 === 0;

        let placed = 0;
        const veinPositions: [number, number, number][] = [];
        const veinIndices: number[] = [];
        const minSep = 0.0075 / worldR;
        const minSep2 = minSep * minSep;
        const symSteps = mType === 1 || mType === 3 ? 4 : 6;
        while (placed < crystalsPerVein && cursor < pool.length) {
          const pi = pool[cursor++];
          if (usedInMineral.has(pi)) continue;
          const el = elements[elemChoice[pi]];
          if (!el) continue;

          // Crystallographic spacing across symmetry branches so outcrops form crisp self-similar clusters
          let nx = cx;
          let ny = cy;
          let nz = cz;
          if (placed > 0) {
            for (let att = 0; att < 10; att++) {
              const angR = (0.007 + Math.sqrt(rng()) * 0.025) / worldR;
              const arm = Math.floor(rng() * symSteps);
              const th = veinAngle + (arm * Math.PI * 2) / symSteps + (rng() - 0.5) * 0.42;
              const ct = Math.cos(th);
              const st = Math.sin(th);
              const ca = Math.cos(angR);
              const sa = Math.sin(angR);
              let cxp = ca * cx + sa * (ct * e1x + st * e2x);
              let cyp = ca * cy + sa * (ct * e1y + st * e2y);
              let czp = ca * cz + sa * (ct * e1z + st * e2z);
              const invL = 1 / (Math.hypot(cxp, cyp, czp) || 1);
              cxp *= invL;
              cyp *= invL;
              czp *= invL;
              nx = cxp;
              ny = cyp;
              nz = czp;
              let tooClose = false;
              for (const vp of veinPositions) {
                const dx = vp[0] - nx;
                const dy = vp[1] - ny;
                const dz = vp[2] - nz;
                if (dx * dx + dy * dy + dz * dz < minSep2) {
                  tooClose = true;
                  break;
                }
              }
              if (!tooClose) break;
            }
          }
          const cAngle = (veinAngle + (Math.floor(rng() * symSteps) * Math.PI * 2) / symSteps) % (Math.PI * 2);
          const [ux, uy, uz] = unitTangentFromSphere(nx, ny, nz, cAngle);
          const b = pi * 12;
          particleData[b + 0] = nx;
          particleData[b + 1] = ny;
          particleData[b + 2] = nz;
          particleData[b + 3] = elemChoice[pi];
          particleData[b + 8] = ux;
          particleData[b + 9] = uy;
          particleData[b + 10] = uz;

          const trap =
            (isBeTrapVein ||
              el.atomicNumber === 4 ||
              (mType === 1 && (isFluorTrapVein || el.atomicNumber === 7 || el.atomicNumber === 13)))
              ? 1
              : 0;
          const nutr =
            mType === 0 ? 0.25 : mType === 1 ? (trap ? 0.1 : 0.3) : mType === 2 ? (trap ? 0.4 : 0.85) : mType === 3 ? 0.9 : 0.95;
          const haz =
            mType === 0 ? 0.0 : mType === 1 ? (trap ? 0.88 : 0.45) : mType === 2 ? (trap ? 0.92 : 0.0) : mType === 3 ? 0.08 : 0.0;
          const lBase = pi * 4;
          lifeData[lBase + 0] = 0.56 + 0.44 * rng();
          lifeData[lBase + 1] = nutr;
          lifeData[lBase + 2] = haz;
          lifeData[lBase + 3] = packLifeMetaJs(3, mType, cAngle, 7, trap);

          veinPositions.push([nx, ny, nz]);
          veinIndices.push(pi);
          usedInMineral.add(pi);
          placed++;
        }
        // Encode live cluster particle count into initial crystal grade (self-similar volume depth)
        const veinGrade = Math.max(0, Math.min(15, Math.round((placed - 1) * 1.35 + (mType >= 2 ? 2 : 1))));
        for (const pi of veinIndices) {
          const lBase = pi * 4;
          const oldPacked = Math.round(lifeData[lBase + 3]);
          const trap = (oldPacked >>> 23) & 1;
          const ang = (((oldPacked >>> 10) & 255) / 256) * Math.PI * 2;
          lifeData[lBase + 3] = packLifeMetaJs(3, mType, ang, veinGrade, trap);
        }
      }
    }
  }

  for (const pi of freePool) {
    if (!usedInMineral.has(pi)) {
      placeFreeAtom(pi);
    }
  }

  device.queue.writeBuffer(particleBuffers[0], 0, particleData.buffer);
  const lifeMdpInit = new Float32Array(particleAmt * 12);
  for (let i = 0; i < particleAmt; i++) {
    const s = i * 4;
    const d = i * 12;
    lifeMdpInit[d + 0] = lifeData[s + 0];
    lifeMdpInit[d + 1] = lifeData[s + 1];
    lifeMdpInit[d + 2] = lifeData[s + 2];
    lifeMdpInit[d + 3] = lifeData[s + 3];
  }
  device.queue.writeBuffer(lifeStateBuffers[0], 0, lifeMdpInit.buffer);
  device.queue.writeBuffer(lifeStateBuffers[1], 0, new Float32Array(particleAmt * 12));
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
    tempTexView: tempTexture.createView(),
    uvTexView: uvTexture.createView(),
    hydroTexView: hydroTexture.createView(),
    terrainTexView: terrainTexture.createView(),
    particleAmt,
    cellAmt: params.cells,
  };
  activeWorldRefs = refs;
  boundEngines.clear();
  prepareEngine();

  mineralRenderBindGroup = mineralRenderPipeline
    ? device.createBindGroup({
        layout: mineralRenderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: watermanFwdTexture.createView() },
          { binding: 2, resource: { buffer: worldColoursBuffer } },
          { binding: 3, resource: { buffer: worldRulesBuffer } },
          { binding: 4, resource: uvTexture.createView() },
        ],
      })
    : undefined;

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: watermanFwdTexture.createView() },
      { binding: 2, resource: { buffer: worldColoursBuffer } },
      { binding: 3, resource: { buffer: worldRulesBuffer } },
      { binding: 4, resource: uvTexture.createView() },
    ],
  });
  slimeFieldBindGroup = slimeFieldBindGroupLayout
    ? device.createBindGroup({
        layout: slimeFieldBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: watermanFwdTexture.createView() },
          { binding: 2, resource: { buffer: worldColoursBuffer } },
        ],
      })
    : undefined;
  syncEpisodeTracking();
  rebuildLensTrackBindGroups();
}

let lastTime = 0;
const deltaVs: number[] = [];

function writeUniforms() {
  if (!device || !uniformBuffer) return;
  const spec = WIND_LEVELS.surface;
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
  uniformData[40] = earthParams.mineralOpacity;
  uniformData[41] = earthParams.tempColors;
  uniformData[42] = earthParams.localTemp ? earthParams.tempStrength : 0;
  uniformData[43] =
    earthParams.viewLayer === 'ocean' ? 1 : earthParams.viewLayer === 'air' ? 2 : 0;
  uniformData[44] = earthParams.uvColors;
  uniformData[45] = earthParams.localUv ? earthParams.uvStrength : 0;
  uniformData[46] = 0;
  uniformData[47] = 0;
  uniformData[48] = earthParams.rhColors;
  uniformData[49] = earthParams.precipColors;
  uniformData[50] = earthParams.localHydro ? earthParams.hydroStrength : 0;
  uniformData[51] = 0;
  uniformData[52] = earthParams.terrainOpacity;
  uniformData[53] = earthParams.nightShading;
  uniformData[54] = 0;
  uniformData[55] = 0;

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);
}

function update() {
  requestAnimationFrame(update);
  const gpuReady = Boolean(device && context && bgContext && particleBuffers && !deviceLost);

  const start = performance.now();
  const frameDtMs = lastTime > 0 ? start - lastTime : 16.67;
  deltaVs.push(frameDtMs);
  if (deltaVs.length > 1000) {
    deltaVs.splice(0, 1);
  }
  lastTime = start;

  syncCanvasDimensions();
  const ambient = game.mode === 'watch' || game.mode === 'off';
  if (ambient && tour.enabled && !camera.active && viewState.projectionId !== 'orthographic')
    moveCamera({ ...viewState, projectionId: 'orthographic' }, 750);
  let cameraMoving = !document.hidden && camera.frame(start, viewState);
  if (!document.hidden && ambient) {
    if (!cameraMoving) cameraMoving = tour.frame(start, viewState);
    observeSandbox(start);
    if (tour.enabled || rescuedEpisodes.size) displayCamera();
  }
  if (cameraMoving) {
    projectionFade.style.opacity = String(camera.opacity);
    updateProjectionState(false);
  }

  if (
    gpuReady &&
    device &&
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
    bgDataVersion++;
  }

  if (mapDirty && mapCtx) {
    mapDirty = false;
    renderMapOverlayCanvas(
      mapCtx,
      viewState.width,
      viewState.height,
      d3Proj,
      earthFeatures,
      earthParams.coastlines,
      viewState.projectionId === 'orthographic',
    );
  }

  const showAir = earthParams.viewLayer !== 'ocean';
  const showOcean = earthParams.viewLayer !== 'air';

  syncStreamCanvasOpacity();

  tooltipManager.tick();
  observeGlobeAudioCenter(start);
  const rmsAudio = cyberSynth.tick(start);
  cyberSynth.readSpectrogram(
    synthSpecSnapshot,
    cyberSynth.isEnabled() && !synthSpectrogram.isMinimized(),
  );
  synthSpectrogram.render(synthSpecSnapshot, start);
  if ((frameCount & 3) === 0) syncMusicButtonUi(rmsAudio);

  const streamModActive =
    cyberSynth.isRunning() &&
    cyberSynth.isStreamlinesEnabled() &&
    !document.hidden;
  const bassCutoffNorm = Math.max(0, Math.min(1, (synthSpecSnapshot.bassCutoffHz - 220) / 1800));

  if (oceanCtx && currentOceanGrid) {
    const oceanTint: [number, number, number] =
      showAir && earthParams.windLines > 0.01 ? [115, 235, 255] : [235, 250, 255];
    oceanStreamRenderer.stepAndDraw(
      oceanCtx,
      viewState.width,
      viewState.height,
      d3Proj,
      getCachedOceanSampler(currentOceanGrid, oceanTint),
      viewSignature,
      showOcean && earthParams.oceanLines > 0.001,
      cameraMoving,
      streamModActive
        ? {
            enabled: true,
            phase: synthSpecSnapshot.lowBassPhase,
            beatPulse: synthSpecSnapshot.downbeatPulse,
            bassEnergy: synthSpecSnapshot.lowBassEnergy,
            cutoffNorm: bassCutoffNorm,
            isOcean: true,
          }
        : undefined,
      oceanStreamBands,
    );
  } else {
    oceanStreamBands.fill(0);
  }

  if (windCtx) {
    windStreamRenderer.stepAndDraw(
      windCtx,
      viewState.width,
      viewState.height,
      d3Proj,
      getCachedWindSampler(currentWindGrid),
      viewSignature,
      showAir && earthParams.windLines > 0.001,
      cameraMoving,
      streamModActive
        ? {
            enabled: true,
            phase: synthSpecSnapshot.lowBassPhase + Math.PI * 0.5,
            beatPulse: synthSpecSnapshot.offbeatPulse,
            bassEnergy: synthSpecSnapshot.lowBassEnergy,
            cutoffNorm: bassCutoffNorm,
            isOcean: false,
          }
        : undefined,
      windStreamBands,
    );
  } else {
    windStreamBands.fill(0);
  }

  cyberSynth.updateStreamlineTelemetry(start, windStreamBands, oceanStreamBands);

  if (!gpuReady || !device) {
    const cpuTime = performance.now() - start;
    globalPerformanceParams.cpu = cpuTime;
    updateTotal();
    fpsc++;
    return;
  }

  const checkValidation = frameCount <= 120 || (frameCount % 60 === 0);
  if (checkValidation) {
    device.pushErrorScope('validation');
  }
  const commandEncoder = device.createCommandEncoder();
  if (game.mode === 'off') stepWorld(undefined,undefined,commandEncoder);
  else { game.frame(start); writeUniforms(); }
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

  if (checkValidation) {
    device.popErrorScope().then((error) => {
      if (error) {
        console.error('WebGPU validation error:', error.message);
        if (isTimestampActive() && !window.location.search.includes('noTimestamp')) {
          const url = new URL(window.location.href);
          url.searchParams.set('noTimestamp', '');
          window.location.replace(url.toString());
        }
      }
    }).catch(() => { /* device.lost reports disconnection */ });
  }

  const cpuTime = performance.now() - start;
  globalPerformanceParams.cpu = cpuTime;
  updateTotal();

  if (frameCount % 12 === 0) {
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
  }

  fpsc++;
}

requestAnimationFrame(update);

function physicsSettings() {
  return JSON.stringify([engine, optionParams, oceanOptionParams,
    earthParams.vortex, earthParams.windStrength, earthParams.oceanMix, earthParams.oceanStrength,
    earthParams.localTemp, earthParams.tempStrength, earthParams.localUv, earthParams.uvStrength,
    earthParams.bioLink, earthParams.bioRate, earthParams.mineralLink, earthParams.mineralRate]);
}
let recordedSettings = physicsSettings();
advanced.on('change', event => {
  if (!event.last || suppressPaneEvents || game.mode !== 'watch') return;
  const settings = physicsSettings();
  if (recordedSettings === settings) return;
  recordedSettings = settings;
  game.reset();
});

particlesBinding.on('change', event => {
  if (event.last && !suppressPaneEvents && particleAmt !== Math.max(1, Math.floor(params.particles))) {
    particleAmt = Math.max(1, Math.floor(params.particles));
    startWorld(); saveConfigNow();
  }
});

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

synthVolBinding.on('change', (ev) => {
  cyberSynth.setVolume(ev.value);
});

mineralLinkBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

mineralRateBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

projectionSelect.on('change', (ev) => {
  // Refreshing the selector after an automated move also emits a change.
  if (suppressPaneEvents || ev.value === viewState.projectionId) return;
  setProjectionChoice(ev.value);
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

tempColorsBinding.on('change', () => {
  // Picked up immediately by bgRender uniform
});

localTempBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

tempStrengthBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

uvColorsBinding.on('change', () => {
  // Picked up immediately by bgRender uniform
});

localUvBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

uvStrengthBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

rhColorsBinding.on('change', () => {
  if (suppressPaneEvents) return;
  refreshHudVisibility();
});

precipColorsBinding.on('change', () => {
  if (suppressPaneEvents) return;
  refreshHudVisibility();
});

localHydroBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

hydroStrengthBinding.on('change', () => {
  if (suppressPaneEvents) return;
  syncBioUniformState();
});

coastlinesBinding.on('change', () => {
  if (suppressPaneEvents) return;
  mapDirty = true;
});

resetViewBtn.on('click', () => {
  if (incidentCameraLocked) return;
  moveCamera({ ...viewState, projectionId: 'orthographic',
    rotate: [...PROJECTIONS.orthographic.defaultRotate], pan: [0, 0],
    scale: Math.max(140, Math.round(423 * Math.min(viewState.width / 1280, viewState.height / 800))) }, 1200, true);
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
