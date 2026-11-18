import { Pane } from 'tweakpane';
import {
  BindingApi,
  type BindingParams,
  type FolderApi,
} from '@tweakpane/core';
import { generateRandomSeed } from './matrix';
import {
  computeAbundanceDistribution,
  getElement2D,
  getElementSymbol,
} from './periodicTable2D';
import {
  parseNullschoolHash,
  PROJECTIONS,
  type ProjectionId,
} from './earth/projections';
import { WIND_LEVELS, type WindLevelId } from './earth/windData';
import { setMouseToolMode } from './input';
import {
  FACTORY_DEFAULTS,
  hasConfigQueryParams,
  loadCachedConfig,
  parsePanParam,
  type ConfigMouseTool,
  type SimConfigV1,
} from './config';

const urlParams = new URLSearchParams(window.location.search);
const parsedHash = parseNullschoolHash(window.location.hash);
const cachedConfig: SimConfigV1 | null = loadCachedConfig();

export function getCachedConfig(): SimConfigV1 | null {
  return cachedConfig;
}

export const hasExplicitUrlHash = window.location.hash.length > 1;
export const hasExplicitUrlConfig = hasConfigQueryParams(window.location.search);
const effectiveCachedConfig: SimConfigV1 | null = hasExplicitUrlConfig ? null : cachedConfig;

function validateURLParam(key: string, defaultValue: number): number;
function validateURLParam(key: string, defaultValue: string): string;
function validateURLParam(
  key: string,
  defaultValue: number | string,
): number | string {
  const val = urlParams.get(key);
  if (val === null) return defaultValue;
  if (typeof defaultValue === 'number') {
    const parsed = Number(val);
    return Number.isFinite(parsed) ? parsed : defaultValue;
  }
  return val;
}

function resolveNumber(
  urlKey: string | null,
  cachedVal: number | undefined,
  factoryVal: number,
  min?: number,
): number {
  let out = typeof cachedVal === 'number' && Number.isFinite(cachedVal) ? cachedVal : factoryVal;
  if (urlKey && urlParams.has(urlKey)) {
    out = validateURLParam(urlKey, out);
  }
  return min !== undefined ? Math.max(min, out) : out;
}

function resolveBool(
  urlKey: string | null,
  cachedVal: boolean | undefined,
  factoryVal: boolean,
): boolean {
  let out = typeof cachedVal === 'boolean' ? cachedVal : factoryVal;
  if (urlKey && urlParams.has(urlKey)) {
    out = urlParams.get(urlKey) !== 'false';
  }
  return out;
}

export const particleAmt = Math.max(
  1,
  Math.floor(resolveNumber('particleCount', effectiveCachedConfig?.particles, FACTORY_DEFAULTS.particles, 1)),
);
export const colourAmt = Math.max(
  1,
  Math.floor(resolveNumber('colours', effectiveCachedConfig?.colours, FACTORY_DEFAULTS.colours, 1)),
);
const cellAmt = Math.max(
  1,
  Math.floor(resolveNumber('cellCount', effectiveCachedConfig?.cells, FACTORY_DEFAULTS.cells, 1)),
);
const defaultEngine = urlParams.has('engine')
  ? validateURLParam('engine', FACTORY_DEFAULTS.engine)
  : (effectiveCachedConfig?.engine ?? FACTORY_DEFAULTS.engine);
const defaultRadius = resolveNumber('radius', effectiveCachedConfig?.air?.r, FACTORY_DEFAULTS.air.r);
const defaultWorldSize = resolveNumber('worldSize', effectiveCachedConfig?.air?.worldSize, FACTORY_DEFAULTS.air.worldSize);
const defaultTorque = resolveNumber('torque', effectiveCachedConfig?.air?.torque, FACTORY_DEFAULTS.air.torque);
const defaultAngularFriction = resolveNumber('angularFriction', effectiveCachedConfig?.air?.angularFriction, FACTORY_DEFAULTS.air.angularFriction);
const defaultValency = resolveNumber('valency', effectiveCachedConfig?.air?.valency, FACTORY_DEFAULTS.air.valency);
const defaultTemperature = resolveNumber('temperature', effectiveCachedConfig?.air?.temperature, FACTORY_DEFAULTS.air.temperature);
const defaultActivation = resolveNumber('activation', effectiveCachedConfig?.air?.activation, FACTORY_DEFAULTS.air.activation);
const defaultOrbitalSharpness = resolveNumber('orbitalSharpness', effectiveCachedConfig?.air?.orbitalSharpness, FACTORY_DEFAULTS.air.orbitalSharpness);
const defaultBondStability = resolveNumber('bondStability', effectiveCachedConfig?.air?.bondStability, FACTORY_DEFAULTS.air.bondStability, 0);
const defaultExchangeDrive = resolveNumber('exchangeDrive', effectiveCachedConfig?.air?.exchangeDrive, FACTORY_DEFAULTS.air.exchangeDrive, 0);
const defaultShearBreak = resolveNumber('shearBreak', effectiveCachedConfig?.air?.shearBreak, FACTORY_DEFAULTS.air.shearBreak, 0);
const defaultShearFloor = resolveNumber('shearFloor', effectiveCachedConfig?.air?.shearFloor, FACTORY_DEFAULTS.air.shearFloor, 0);
const defaultSeed = urlParams.has('seed')
  ? validateURLParam('seed', generateRandomSeed())
  : (effectiveCachedConfig?.seed ?? generateRandomSeed());

const urlProj = validateURLParam('projection', '');
const initialProjection: ProjectionId =
  urlProj in PROJECTIONS
    ? (urlProj as ProjectionId)
    : hasExplicitUrlHash && parsedHash.projection && parsedHash.projection in PROJECTIONS
      ? parsedHash.projection
      : effectiveCachedConfig?.earth?.projection && effectiveCachedConfig.earth.projection in PROJECTIONS
        ? effectiveCachedConfig.earth.projection
        : FACTORY_DEFAULTS.earth.projection;

const urlLevel = validateURLParam('windLevel', '');
const initialLevel: WindLevelId =
  urlLevel in WIND_LEVELS
    ? (urlLevel as WindLevelId)
    : hasExplicitUrlHash && parsedHash.level && parsedHash.level in WIND_LEVELS
      ? (parsedHash.level as WindLevelId)
      : effectiveCachedConfig?.earth?.windLevel && effectiveCachedConfig.earth.windLevel in WIND_LEVELS
        ? effectiveCachedConfig.earth.windLevel
        : FACTORY_DEFAULTS.earth.windLevel;

const urlLayer = validateURLParam('layer', '');
export type ViewLayerMode = 'both' | 'ocean' | 'air';
const initialViewLayer: ViewLayerMode =
  urlLayer === 'ocean' || urlLayer === 'air' || urlLayer === 'both'
    ? urlLayer
    : hasExplicitUrlHash && parsedHash.mode
      ? parsedHash.mode
      : (effectiveCachedConfig?.earth?.viewLayer ?? FACTORY_DEFAULTS.earth.viewLayer);

export const oceanParticleAmt = Math.max(
  1,
  Math.floor(resolveNumber('oceanParticles', effectiveCachedConfig?.ocean?.particles, FACTORY_DEFAULTS.ocean.particles, 1)),
);
export const oceanColourAmt = Math.max(
  1,
  Math.floor(resolveNumber('oceanColours', effectiveCachedConfig?.ocean?.colours, FACTORY_DEFAULTS.ocean.colours, 1)),
);

export const initialOrientation = parsedHash.orientation;
export const initialUrlPan = parsePanParam(urlParams.get('pan'));

export const params = {
  fps: 0,
  engine: defaultEngine,
  particles: particleAmt,
  colours: colourAmt,
  cells: cellAmt,
  seed: defaultSeed,
};

function parseOpacityParam(key: string, cachedVal: number | undefined, defaultValue: number): number {
  const base =
    typeof cachedVal === 'number' && Number.isFinite(cachedVal)
      ? Math.max(0, Math.min(1, cachedVal))
      : defaultValue;
  const val = urlParams.get(key);
  if (val === null) return base;
  if (val === 'false') return 0;
  if (val === 'true') return 1;
  const parsed = Number(val);
  return Number.isFinite(parsed) ? Math.max(0, Math.min(1, parsed)) : base;
}

const urlMouseTool = urlParams.get('mouseTool');
const initialMouseTool: ConfigMouseTool =
  urlMouseTool === 'globe' || urlMouseTool === 'attract' || urlMouseTool === 'repel'
    ? urlMouseTool
    : (effectiveCachedConfig?.earth?.mouseTool ?? FACTORY_DEFAULTS.earth.mouseTool);

const initialWindMix = urlParams.has('windMix')
  ? urlParams.get('windMix') !== 'false'
  : resolveBool('vortex', effectiveCachedConfig?.earth?.vortex, FACTORY_DEFAULTS.earth.vortex);

export const earthParams = {
  viewLayer: initialViewLayer,
  projection: initialProjection,
  windLevel: initialLevel,
  vortex: initialWindMix,
  windStrength: resolveNumber('windStrength', effectiveCachedConfig?.earth?.windStrength, FACTORY_DEFAULTS.earth.windStrength),
  windLines: parseOpacityParam('windLines', effectiveCachedConfig?.earth?.windLines, FACTORY_DEFAULTS.earth.windLines),
  windColors: parseOpacityParam('windColors', effectiveCachedConfig?.earth?.windColors, FACTORY_DEFAULTS.earth.windColors),
  oceanMix: resolveBool('oceanMix', effectiveCachedConfig?.earth?.oceanMix, FACTORY_DEFAULTS.earth.oceanMix),
  oceanStrength: resolveNumber('oceanStrength', effectiveCachedConfig?.earth?.oceanStrength, FACTORY_DEFAULTS.earth.oceanStrength),
  oceanLines: parseOpacityParam('oceanLines', effectiveCachedConfig?.earth?.oceanLines, FACTORY_DEFAULTS.earth.oceanLines),
  oceanColors: parseOpacityParam('oceanColors', effectiveCachedConfig?.earth?.oceanColors, FACTORY_DEFAULTS.earth.oceanColors),
  chemOpacity: parseOpacityParam('chemOpacity', effectiveCachedConfig?.earth?.chemOpacity, FACTORY_DEFAULTS.earth.chemOpacity),
  lifeOpacity: parseOpacityParam('lifeOpacity', effectiveCachedConfig?.earth?.lifeOpacity, FACTORY_DEFAULTS.earth.lifeOpacity),
  coastlines: resolveBool('coastlines', effectiveCachedConfig?.earth?.coastlines, FACTORY_DEFAULTS.earth.coastlines),
  bioLink: resolveBool('bioLink', effectiveCachedConfig?.earth?.bioLink, FACTORY_DEFAULTS.earth.bioLink),
  bioRate: resolveNumber('bioRate', effectiveCachedConfig?.earth?.bioRate, FACTORY_DEFAULTS.earth.bioRate),
  tooltips: resolveBool('tooltips', effectiveCachedConfig?.earth?.tooltips, FACTORY_DEFAULTS.earth.tooltips),
  mouseTool: initialMouseTool,
  windStatus: 'Loading GFS…',
  oceanStatus: 'Loading CoastWatch…',
};
setMouseToolMode(earthParams.mouseTool);

// Classic Particle Life parameters preserved from the original fork (51bc381)
export const oceanOptionParams = {
  particles: oceanParticleAmt,
  colours: oceanColourAmt,
  r: resolveNumber('oceanR', effectiveCachedConfig?.ocean?.r, FACTORY_DEFAULTS.ocean.r),
  force: resolveNumber('oceanForce', effectiveCachedConfig?.ocean?.force, FACTORY_DEFAULTS.ocean.force),
  beta: resolveNumber('oceanBeta', effectiveCachedConfig?.ocean?.beta, FACTORY_DEFAULTS.ocean.beta),
  delta: resolveNumber('oceanDelta', effectiveCachedConfig?.ocean?.delta, FACTORY_DEFAULTS.ocean.delta),
  friction: resolveNumber('oceanFriction', effectiveCachedConfig?.ocean?.friction, FACTORY_DEFAULTS.ocean.friction),
  avoidance: resolveNumber('oceanAvoidance', effectiveCachedConfig?.ocean?.avoidance, FACTORY_DEFAULTS.ocean.avoidance),
  cells: Math.max(1, Math.floor(resolveNumber('oceanCells', effectiveCachedConfig?.ocean?.cells, cellAmt, 1))),
  worldSize: defaultWorldSize,
};

export function setOceanSim(
  device: GPUDevice,
  oceanSimData: Float32Array<ArrayBuffer>,
  oceanSimBuffer: GPUBuffer,
) {
  const dt = Math.max(oceanOptionParams.delta, 1e-4);
  const fric = Math.max(oceanOptionParams.friction, 1e-4);
  const rMax = 1 / Math.max(oceanOptionParams.r, 0.5);

  oceanSimData[0] = oceanOptionParams.colours;
  oceanSimData[1] = oceanOptionParams.beta;
  oceanSimData[2] = rMax;
  oceanSimData[3] = oceanOptionParams.force / rMax;
  oceanSimData[4] = Math.pow(0.5, dt / fric);
  oceanSimData[5] = dt;
  oceanSimData[6] = rMax * 2;
  oceanSimData[7] = oceanOptionParams.cells;
  oceanSimData[8] = oceanOptionParams.avoidance;
  oceanSimData[9] = optionParams.worldSize;
  oceanSimData[10] = earthParams.oceanStrength;
  oceanSimData[11] = earthParams.oceanMix ? 1 : 0;
  device.queue.writeBuffer(oceanSimBuffer, 0, oceanSimData);
}

export function setBioLink(
  device: GPUDevice,
  bioData: Float32Array<ArrayBuffer>,
  bioBuffer: GPUBuffer,
  enabled: boolean,
) {
  const worldSize = Math.max(optionParams.worldSize, 0.25);
  const airRMax = 1 / Math.max(optionParams.r, 0.5);
  const oceanRMax = 1 / Math.max(oceanOptionParams.r, 0.5);
  const catRadius = Math.max(oceanRMax * 1.35, 0.065);
  const dt = Math.min(Math.max(optionParams.delta, 1e-4), Math.max(oceanOptionParams.delta, 1e-4));

  bioData[0] = enabled && earthParams.bioRate > 0 ? 1 : 0;
  bioData[1] = earthParams.bioRate;
  bioData[2] = catRadius;
  bioData[3] = worldSize;
  bioData[4] = airRMax * 2;
  bioData[5] = params.cells;
  bioData[6] = oceanRMax * 2;
  bioData[7] = oceanOptionParams.cells;
  bioData[8] = params.colours;
  bioData[9] = oceanOptionParams.colours;
  bioData[10] = dt;
  bioData[11] = 0;
  device.queue.writeBuffer(bioBuffer, 0, bioData);
}

export const optionParams = {
  colours: colourAmt,
  r: defaultRadius,
  force: resolveNumber('force', effectiveCachedConfig?.air?.force, FACTORY_DEFAULTS.air.force),
  beta: resolveNumber('beta', effectiveCachedConfig?.air?.beta, FACTORY_DEFAULTS.air.beta),
  delta: resolveNumber('delta', effectiveCachedConfig?.air?.delta, FACTORY_DEFAULTS.air.delta),
  friction: resolveNumber('friction', effectiveCachedConfig?.air?.friction, FACTORY_DEFAULTS.air.friction),
  torque: defaultTorque,
  angularFriction: defaultAngularFriction,
  valency: defaultValency,
  temperature: defaultTemperature,
  activation: defaultActivation,
  orbitalSharpness: defaultOrbitalSharpness,
  bondStability: defaultBondStability,
  exchangeDrive: defaultExchangeDrive,
  shearBreak: defaultShearBreak,
  shearFloor: defaultShearFloor,
  showOrbitals: resolveBool('showOrbitals', effectiveCachedConfig?.air?.showOrbitals, FACTORY_DEFAULTS.air.showOrbitals),
  cells: cellAmt,
  avoidance: resolveNumber('avoidance', effectiveCachedConfig?.air?.avoidance, FACTORY_DEFAULTS.air.avoidance),
  worldSize: defaultWorldSize,
};

export function setSim(
  device: GPUDevice,
  simData: Float32Array<ArrayBuffer>,
  simBuffer: GPUBuffer,
  frameCount: number = 0,
) {
  const dt = Math.max(optionParams.delta, 1e-4);
  const linFriction = Math.max(optionParams.friction, 1e-4);
  const angFriction = Math.max(optionParams.angularFriction, 1e-4);

  simData[0] = params.colours;
  simData[1] = optionParams.beta;
  simData[2] = 1 / optionParams.r;
  simData[3] = optionParams.force / (1 / optionParams.r);
  simData[4] = Math.pow(0.5, dt / linFriction);
  simData[5] = dt;
  simData[6] = (1 / optionParams.r) * 2;
  simData[7] = params.cells;
  simData[8] = optionParams.avoidance;
  simData[9] = optionParams.worldSize;
  simData[10] = 0; // border unused on closed sphere
  simData[11] = earthParams.vortex ? 1 : 0;
  simData[12] = optionParams.torque;
  simData[13] = Math.pow(0.5, dt / angFriction);
  simData[14] = optionParams.valency;
  simData[15] = optionParams.temperature;
  simData[16] = optionParams.activation;
  simData[17] = optionParams.orbitalSharpness;
  simData[18] = frameCount;
  simData[19] = earthParams.windStrength;
  simData[20] = optionParams.bondStability;
  simData[21] = optionParams.exchangeDrive;
  simData[22] = optionParams.shearBreak;
  simData[23] = optionParams.shearFloor;
  device.queue.writeBuffer(simBuffer, 0, simData);
}

function withTooltip<T extends { element: HTMLElement }>(
  api: T,
  title: string,
  description: string,
): T {
  const el = api.element;
  const target = el.classList.contains('tp-fldv')
    ? (el.querySelector<HTMLElement>(':scope > .tp-fldv_b') ?? el)
    : el;
  target.dataset.tipTitle = title;
  target.dataset.tipBody = description;
  return api;
}

const opacitySliderParams: BindingParams = {
  min: 0,
  max: 1,
  step: 0.05,
  format: (v: number) => `${Math.round(v * 100)}%`,
};

const pane = new Pane({
  title: 'Complex Life · Earth Wind',
  expanded: window.innerWidth >= 768,
});
{
  const rootHeader = pane.element.querySelector<HTMLElement>(':scope > .tp-rotv_b');
  if (rootHeader) {
    rootHeader.dataset.tipTitle = 'Parameter Controls';
    rootHeader.dataset.tipBody = 'Expand or collapse simulation parameters, globe overlays, and bio-chemical settings.';
  }
}

withTooltip(
  pane.addBinding(params, 'fps', { readonly: true }),
  'FPS (Frames Per Second)',
  'Real-time rendered frames per second across the WebGPU compute and render passes.',
);

export const engineSelect = withTooltip(
  pane.addBinding(params, 'engine', {
    options: {
      'Prefix Sum': 'prefixSum',
      'Counting Sort': 'countingSort',
      'Atomic Linked Lists': 'linkedList',
    },
  }),
  'Spatial Hash Engine',
  'GPU parallel neighbor-search backend: Prefix Sum (subgroups), Counting Sort, or Atomic Linked Lists.',
);

export const particlesBinding = withTooltip(
  pane.addBinding(params, 'particles', { min: 1, step: 1 }),
  'Chemistry Particle Count',
  'Total atmospheric & surface chemistry atoms simulated on the globe (takes effect on New Sim).',
);
export const seedInput = withTooltip(
  pane.addBinding(params, 'seed', { view: 'text' }),
  'Simulation Seed',
  'Deterministic PRNG seed controlling initial chemical positions and element placement.',
);
export const newSimBtn = withTooltip(
  pane.addButton({ title: 'New Sim' }),
  'New Chemistry Sim',
  'Respawn atmospheric chemistry particles using the current particle count and seed.',
);
export const randomizeBtn = withTooltip(
  pane.addButton({ title: 'Randomize' }),
  'Randomize Seed',
  'Pick a new random seed and respawn atmospheric chemistry particles.',
);
export const copyConfigBtn = withTooltip(
  pane.addButton({ title: 'Copy Config' }),
  'Copy Configuration JSON',
  'Copy all current simulation, chemistry, ocean, bioChem, opacity, and camera settings to clipboard for prompts or reuse.',
);
export const pasteConfigBtn = withTooltip(
  pane.addButton({ title: 'Paste / Load Config' }),
  'Paste / Load Configuration',
  'Open config box to paste a saved JSON config or URL from another run, or reset defaults & clear browser cache.',
);

// Earth & Wind / Ocean folder (emulating earth.nullschool.net layers, levels & projections)
const earthFolder = withTooltip(
  pane.addFolder({
    title: 'Earth · Wind & Ocean',
    expanded: true,
  }),
  'Earth · Wind & Ocean',
  'Global atmospheric winds, NOAA ocean currents, map projection, layer opacities, tooltips, and bio-chemical linking.',
);

export const viewLayerSelect = withTooltip(
  earthFolder.addBinding(earthParams, 'viewLayer', {
    label: 'layer',
    options: {
      'Both (Air + Ocean)': 'both',
      'Ocean (Currents)': 'ocean',
      'Air (Wind)': 'air',
    },
  }),
  'Active Layer View',
  'Switch between Both layers together, Ocean Currents + Life only, or Atmospheric Chemistry + Wind only.',
);

const projectionOptions: Record<string, ProjectionId> = {};
for (const key of Object.keys(PROJECTIONS) as ProjectionId[]) {
  projectionOptions[PROJECTIONS[key].label] = key;
}

const windLevelOptions: Record<string, WindLevelId> = {};
for (const key of Object.keys(WIND_LEVELS) as WindLevelId[]) {
  windLevelOptions[WIND_LEVELS[key].label] = key;
}

export const projectionSelect = withTooltip(
  earthFolder.addBinding(earthParams, 'projection', {
    label: 'proj',
    options: projectionOptions,
  }),
  'Map Projection',
  'Cartographic globe projection matching earth.nullschool.net (Orthographic, Atlantis, Waterman Butterfly, etc.).',
);
export const windLevelSelect = withTooltip(
  earthFolder.addBinding(earthParams, 'windLevel', {
    label: 'windLevel',
    options: windLevelOptions,
  }),
  'Wind Pressure Altitude',
  'Atmospheric pressure surface (Surface 10m to 10 hPa jet stream) for GFS/NCEP wind velocities.',
);
export const windMixBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'vortex', {
    label: 'windMix',
  }),
  'Wind Advection Mixing',
  'Enable or disable global atmospheric winds transporting and mixing surface chemistry atoms.',
);
export const windStrengthBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'windStrength', {
    label: 'windForce',
    min: 0,
    max: 4,
    step: 0.05,
  }),
  'Wind Force Scale',
  'Multiplier scaling how strongly wind vectors accelerate atmospheric chemistry atoms.',
);
export const windLinesBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'windLines', {
    label: 'windLines',
    ...opacitySliderParams,
  }),
  'Wind Streamlines Opacity',
  'Opacity (0%–100%) of animated atmospheric wind streamlines overlaid across the globe.',
);
export const windColorsBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'windColors', {
    label: 'windColors',
    ...opacitySliderParams,
  }),
  'Wind Color Map Opacity',
  'Opacity (0%–100%) of the sinebow wind-speed color overlay across Earth.',
);
export const chemOpacityBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'chemOpacity', {
    label: 'chemAlpha',
    ...opacitySliderParams,
  }),
  'Chemistry Layer Opacity',
  'Opacity (0%–100%) of atmospheric chemistry atoms, valence orbitals, and catalytic glow halos.',
);
export const oceanMixBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'oceanMix', {
    label: 'oceanMix',
  }),
  'Ocean Current Mixing',
  'Enable or disable daily NOAA CoastWatch currents advecting marine Particle Life organisms.',
);
export const oceanStrengthBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'oceanStrength', {
    label: 'oceanForce',
    min: 0,
    max: 4,
    step: 0.05,
  }),
  'Ocean Current Force',
  'Multiplier scaling how strongly ocean currents stir marine Particle Life organisms.',
);
export const oceanLinesBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'oceanLines', {
    label: 'oceanLines',
    ...opacitySliderParams,
  }),
  'Ocean Streamlines Opacity',
  'Opacity (0%–100%) of animated ocean current streamlines inside marine basins.',
);
export const oceanColorsBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'oceanColors', {
    label: 'oceanColors',
    ...opacitySliderParams,
  }),
  'Ocean Color Map Opacity',
  'Opacity (0%–100%) of ocean current velocity color shading inside marine basins.',
);
export const lifeOpacityBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'lifeOpacity', {
    label: 'lifeAlpha',
    ...opacitySliderParams,
  }),
  'Ocean Life Opacity',
  'Opacity (0%–100%) of Ocean Life organisms rendered over marine basins.',
);
export const coastlinesBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'coastlines'),
  'Vector Coastlines',
  'Toggle world vector coastlines, inland lakes, and latitude/longitude graticule lines.',
);
export const bioLinkBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'bioLink', {
    label: 'bioChem',
  }),
  'Bio-Chemistry Exchange',
  'Enable bidirectional metabolic & catalytic coupling between immutable Ocean Life and Air Chemistry.',
);
export const bioRateBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'bioRate', {
    label: 'bioRate',
    min: 0,
    max: 3,
    step: 0.05,
  }),
  'Bio-Chem Coupling Rate',
  'Master rate multiplier for enzymatic barrier lowering, fuel harvesting, vitamin support, and toxin transfer.',
);
export const tooltipsBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'tooltips', {
    label: 'tooltips',
  }),
  'Hover Tooltips',
  'Show or hide mouse-over tooltips on ongoing globe chemical reactions and parameter menu controls.',
);
export const mouseToolBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'mouseTool', {
    label: 'tool',
    options: {
      'Rotate / Pan': 'globe',
      'Attract Particles': 'attract',
      'Repel Particles': 'repel',
    },
  }),
  'Pointer Interaction Tool',
  'Single-finger / left-drag mode: Rotate/Pan globe, Attract nearby particles, or Repel particles (2-finger touch always moves camera).',
);
mouseToolBinding.on('change', (ev) => {
  setMouseToolMode(ev.value);
});

export const windStatusBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'windStatus', {
    readonly: true,
    label: 'windSrc',
  }),
  'Wind Data Source',
  'Forecast timestamp and NOAA GFS/NCEP data source driving atmospheric winds.',
);
export const oceanStatusBinding = withTooltip(
  earthFolder.addBinding(earthParams, 'oceanStatus', {
    readonly: true,
    label: 'oceanSrc',
  }),
  'Ocean Data Source',
  'Reference timestamp and NOAA CoastWatch dataset driving surface ocean currents.',
);
export const resetViewBtn = withTooltip(
  earthFolder.addButton({ title: 'Reset Globe View' }),
  'Reset Globe View',
  'Restore default Atlantic Orthographic projection, scale, and orientation.',
);

// Classic Particle Life mechanics from the original fork (51bc381) running on Ocean Currents
const oceanLifeFolder = withTooltip(
  pane.addFolder({
    title: 'Ocean Life (Original Fork)',
    expanded: false,
  }),
  'Ocean Life Parameters',
  'Species attraction matrix and physical Swarm parameters for marine Particle Life on Earth oceans.',
);
export const oceanCountBinding = withTooltip(
  oceanLifeFolder.addBinding(oceanOptionParams, 'particles', {
    label: 'oceanCount',
    min: 500,
    max: 40000,
    step: 500,
  }),
  'Ocean Life Count',
  'Number of marine Particle Life organisms simulated inside ocean basins (takes effect on Respawn).',
);
export const oceanColoursBinding = withTooltip(
  oceanLifeFolder.addBinding(oceanOptionParams, 'colours', {
    label: 'species',
    min: 1,
    max: 200,
    step: 1,
  }),
  'Ocean Species Count',
  'Number of marine species (roles cycle across Autotroph, Consumer, Halophile Detoxifier, Decomposer).',
);
const oceanParamSpecs: Record<string, BindingParams> = {
  r: { min: 2, max: 60, step: 0.5 },
  beta: { min: 0.02, max: 0.95, step: 0.01 },
  force: { min: 0, max: 5, step: 0.05 },
  delta: { label: 'dt', min: 0.002, max: 0.06, step: 0.002 },
  friction: { min: 0.005, max: 0.4, step: 0.005 },
  avoidance: { min: 0, max: 20, step: 0.5 },
  cells: { min: 1, step: 1 },
};
const oceanParamTips: Record<string, [string, string]> = {
  r: ['Ocean Radius Divisor (r)', 'Inverse interaction cutoff for marine Particle Life (cutoff = 1 / r).'],
  beta: ['Ocean Core Repulsion (beta)', 'Inner hard-core repulsion fraction keeping swarms from collapsing onto one point.'],
  force: ['Ocean Force Scale', 'Overall strength of pairwise species attraction and repulsion in marine swarms.'],
  delta: ['Ocean Time Step (dt)', 'Integration timestep size for ocean Particle Life motion and metabolism.'],
  friction: ['Ocean Viscous Friction', 'Velocity half-life damping marine organism motion through seawater.'],
  avoidance: ['Ocean Crowding Avoidance', 'Divisor damping net forces inside high-density marine clusters.'],
  cells: ['Ocean Grid Cells', 'Spatial hash cell count for Ocean Life neighbor lookups (takes effect on Respawn).'],
};
const oceanParamBindings: BindingApi[] = [];
for (const key in oceanParamSpecs) {
  const tip = oceanParamTips[key];
  oceanParamBindings.push(
    withTooltip(
      oceanLifeFolder.addBinding(
        oceanOptionParams,
        key as keyof typeof oceanOptionParams,
        oceanParamSpecs[key],
      ),
      tip[0],
      tip[1],
    ),
  );
}
export const newOceanSimBtn = withTooltip(
  oceanLifeFolder.addButton({ title: 'Respawn Ocean Life' }),
  'Respawn Ocean Life',
  'Re-initialize all marine Particle Life positions inside ocean basins with current count & species.',
);
export const randomizeOceanRulesBtn = withTooltip(
  oceanLifeFolder.addButton({ title: 'Randomize Ocean Matrix' }),
  'Randomize Ocean Rules',
  'Generate a fresh random pairwise attraction/repulsion matrix across ocean species.',
);

const constantOptions = ['colours', 'cells'];
const constants: Record<string, BindingParams> = {
  colours: { min: 1, max: 62, step: 1 },
  cells: { min: 1, step: 1 },
};
const constantTips: Record<string, [string, string]> = {
  colours: ['Chemistry Element Count', 'Active 2D Periodic Table species (Z=1..N); takes effect on New Sim.'],
  cells: ['Chemistry Grid Cells', 'Spatial hash cell count for atmospheric Chemistry neighbor lookups (takes effect on New Sim).'],
};

const constantsFolder = withTooltip(
  pane.addFolder({ title: 'Constants', expanded: false }),
  'Chemistry Constants',
  'Structural Chemistry settings applied when starting a new simulation.',
);
const optionBindings: Record<string, BindingApi> = {};
for (const param in optionParams) {
  if (!constantOptions.includes(param)) continue;
  const tip = constantTips[param];
  const binding = withTooltip(
    constantsFolder.addBinding(
      optionParams,
      param as keyof typeof optionParams,
      param in constants ? constants[param] : {},
    ),
    tip[0],
    tip[1],
  );
  optionBindings[param] = binding;
}

const optionsFolder = withTooltip(
  pane.addFolder({
    title: 'Chemistry Options',
    expanded: false,
  }),
  'Chemistry Options',
  'Real-time valence bonding, Pauli repulsion, temperature noise, and sphere geometry constants.',
);

const options: Record<string, BindingParams> = {
  beta: { min: 0.05, max: 0.7 },
  r: { min: 0.01 },
  force: { min: 0, max: 10 },
  delta: { min: 0.001, max: 0.1 },
  friction: { min: 0.001 },
  torque: { min: 0, max: 100 },
  angularFriction: { label: 'angFriction', min: 0.001 },
  valency: { min: 0, max: 3 },
  temperature: { min: 0, max: 3 },
  activation: { min: 0, max: 1.5 },
  orbitalSharpness: { label: 'lobeFocus', min: 1, max: 12, step: 0.5 },
  bondStability: { label: 'bondStab', min: 0, max: 1, step: 0.02 },
  exchangeDrive: { label: 'reactSwap', min: 0, max: 2, step: 0.05 },
  shearBreak: { label: 'shearBreak', min: 0, max: 3, step: 0.05 },
  shearFloor: { label: 'shearFloor', min: 0, max: 0.8, step: 0.02 },
  showOrbitals: { label: 'showValence' },
  avoidance: { min: 0 },
  worldSize: { label: 'sphereRadius', min: 0.5, max: 5.0 },
};

const optionTips: Record<string, [string, string]> = {
  beta: ['Bond Equilibrium Radius (beta)', 'Fraction of cutoff r where Pauli core repulsion yields to the valence bond well.'],
  r: ['Chemistry Cutoff Divisor (r)', 'Inverse chemical interaction cutoff radius (rMax = 1 / r in sphere units).'],
  force: ['Chemical Force Strength', 'Overall amplitude of covalent/ionic bonding attraction and steric repulsion.'],
  delta: ['Chemistry Time Step (dt)', 'Integration step size for atmospheric atom positions and bond torque.'],
  friction: ['Translational Friction', 'Velocity half-life damping translational momentum across steps.'],
  torque: ['Orbital Alignment Torque', 'Rotational strength aligning atomic valence slots directly toward bonded partners.'],
  angularFriction: ['Angular Friction', 'Spin damping half-life controlling how quickly valence orbital orientations settle.'],
  valency: ['Valency Stiffness', 'Suppression strength blocking extra bonds once an atom fills its outer valence slots.'],
  temperature: ['Thermal Noise (Temperature)', 'Stochastic Brownian kinetic agitation driving free-atom collisions while bonded molecules cool cleanly.'],
  activation: ['Activation Energy Barrier', 'Repulsive barrier outside the bond well lowered by catalysts and high-affinity reaction candidates.'],
  orbitalSharpness: ['Orbital Lobe Focus', 'Angular sharpness exponent concentrating attraction/repulsion onto discrete valence lobes.'],
  bondStability: ['Molecular Bond Stabilization', 'Calms bonded thermal jitter, tightens bond equilibrium so valence orbitals overlap, and shields coherent drift in low winds.'],
  exchangeDrive: ['Reaction Displacement Drive', 'Selectivity allowing strictly higher-affinity reaction candidates to displace weaker bonded partners while blocking equal/weaker collisions.'],
  shearBreak: ['High-Wind Shear Breakup', 'How strongly storm/jet wind gradients exceeding shearFloor weaken and tear apart molecules.'],
  shearFloor: ['Low-Wind Shear Shield', 'Wind-shear threshold below which formed molecules drift coherently on gentle winds without shearing apart.'],
  showOrbitals: ['Show Valence Orbitals', 'Render outer valence electron slots/rings around atoms instead of plain discs.'],
  avoidance: ['Crowding Avoidance', 'Short-range crowding divisor preventing instability in tight multi-atom clusters.'],
  worldSize: ['Sphere Radius', 'Simulated physical sphere radius controlling local interaction range relative to Earth.'],
};

for (const param in optionParams) {
  if (constantOptions.includes(param)) continue;
  const tip = optionTips[param];
  const binding = withTooltip(
    optionsFolder.addBinding(
      optionParams,
      param as keyof typeof optionParams,
      param in options ? options[param] : {},
    ),
    tip?.[0] ?? param,
    tip?.[1] ?? 'Chemistry simulation parameter.',
  );
  optionBindings[param] = binding;
}

export function bindOptions(
  device: GPUDevice,
  simData: Float32Array<ArrayBuffer>,
  simBuffer: GPUBuffer,
  oceanSimData: Float32Array<ArrayBuffer>,
  oceanSimBuffer: GPUBuffer,
  onBioSync?: () => void,
) {
  for (const param in optionBindings) {
    optionBindings[param].on('change', () => {
      setSim(device, simData, simBuffer);
      if (param === 'worldSize') {
        setOceanSim(device, oceanSimData, oceanSimBuffer);
      }
      onBioSync?.();
    });
  }
  windMixBinding.on('change', () => setSim(device, simData, simBuffer));
  windStrengthBinding.on('change', () => setSim(device, simData, simBuffer));

  for (const b of oceanParamBindings) {
    b.on('change', () => {
      setOceanSim(device, oceanSimData, oceanSimBuffer);
      onBioSync?.();
    });
  }
  oceanColoursBinding.on('change', () => {
    setOceanSim(device, oceanSimData, oceanSimBuffer);
    onBioSync?.();
  });
  oceanMixBinding.on('change', () =>
    setOceanSim(device, oceanSimData, oceanSimBuffer),
  );
  oceanStrengthBinding.on('change', () =>
    setOceanSim(device, oceanSimData, oceanSimBuffer),
  );
}

export function refreshAllPane() {
  pane.refresh();
}

export function buildFactoryConfig(seed: string, oceanSeed: string): SimConfigV1 {
  return {
    v: 1,
    engine: FACTORY_DEFAULTS.engine,
    particles: FACTORY_DEFAULTS.particles,
    colours: FACTORY_DEFAULTS.colours,
    cells: FACTORY_DEFAULTS.cells,
    seed,
    oceanSeed,
    air: { ...FACTORY_DEFAULTS.air },
    ocean: { ...FACTORY_DEFAULTS.ocean },
    earth: { ...FACTORY_DEFAULTS.earth },
    view: {
      rotate: [...PROJECTIONS.orthographic.defaultRotate],
      pan: [0, 0],
      scale: 586,
    },
  };
}

export function bindConfigPersistence(onChange: () => void) {
  const tracked: BindingApi[] = [
    engineSelect,
    particlesBinding,
    seedInput,
    viewLayerSelect,
    projectionSelect,
    windLevelSelect,
    windMixBinding,
    windStrengthBinding,
    windLinesBinding,
    windColorsBinding,
    chemOpacityBinding,
    oceanMixBinding,
    oceanStrengthBinding,
    oceanLinesBinding,
    oceanColorsBinding,
    lifeOpacityBinding,
    coastlinesBinding,
    bioLinkBinding,
    bioRateBinding,
    tooltipsBinding,
    mouseToolBinding,
    oceanCountBinding,
    oceanColoursBinding,
    ...oceanParamBindings,
    ...Object.values(optionBindings),
  ];
  for (const b of tracked) {
    b.on('change', onChange);
  }
}

// 2D Periodic Table legend folder
const periodicFolder = withTooltip(
  pane.addFolder({
    title: '2D Periodic Table',
    expanded: false,
  }),
  '2D Periodic Table',
  'Active element distribution in the Chemistry layer (Z=1..N) and biological roles under BioChem linking.',
);
const periodicState: Record<string, string> = {};
const periodicBindings: BindingApi[] = [];

function elementBioSummary(atomicNum: number, valence: number): string {
  if (valence === 0) return 'Inert noble gas';
  if (atomicNum === 7 || atomicNum === 13) return 'Halogen poison (detoxified by halophiles)';
  if (atomicNum === 3 || atomicNum === 4 || atomicNum === 10) return 'Enzyme vitamin cofactor';
  if (atomicNum === 5) return 'Carbon food / biomass backbone';
  if (atomicNum === 1 || atomicNum === 12) return 'Exothermic hydride/chalcogen fuel';
  return 'Oxide / silicate / salt builder';
}

export function updatePeriodicLegend(colourCount: number) {
  for (const b of periodicBindings) {
    periodicFolder.remove(b);
  }
  periodicBindings.length = 0;
  const probs = computeAbundanceDistribution(colourCount);
  const shown = Math.min(colourCount, 24);
  for (let c = 0; c < shown; c++) {
    const el = getElement2D(c);
    const sym = getElementSymbol(el.atomicNumber);
    const key = `Z${el.atomicNumber}`;
    const pct = (probs[c] * 100).toFixed(probs[c] >= 0.01 ? 1 : 2);
    periodicState[key] = `${sym} ${el.name.replace(`Z=${el.atomicNumber} `, '')} (${pct}%)`;
    periodicBindings.push(
      withTooltip(
        periodicFolder.addBinding(periodicState, key, { readonly: true }),
        `${sym} · ${el.subshellLabel} (Z=${el.atomicNumber})`,
        `Earth-surface share: ${pct}% · Valence: ${el.valence} · Bio role: ${elementBioSummary(el.atomicNumber, el.valence)}.`,
      ),
    );
  }
}

updatePeriodicLegend(colourAmt);

export const performanceParams = {
  countingSort: {
    cell: 0,
    prefix: 0,
    sort: 0,
    valence: 0,
    sim: 0,
  },
  prefixSum: {
    cell: 0,
    prefix: 0,
    sort: 0,
    valence: 0,
    sim: 0,
  },
  linkedList: {
    construct: 0,
    valence: 0,
    sim: 0,
  },
};

export const globalPerformanceParams = {
  cpu: 0,
  render: 0,
  total: 0,
  graph: 0,
};

const performanceTimes = withTooltip(
  pane.addFolder({
    title: 'Performance Times',
    expanded: false,
  }),
  'Performance Times',
  'GPU compute/render pass durations (ms) and per-frame CPU overhead.',
);
const engineFolders: Record<string, FolderApi> = {};
const passDescriptions: Record<string, string> = {
  cell: 'Assign particles into spatial hash cells and count per-cell occupancy.',
  prefix: 'Compute prefix sums over spatial cell counts.',
  sort: 'Reorder particles into contiguous cell buckets for coalesced neighbor reads.',
  construct: 'Atomically link particles into per-cell linked lists on GPU.',
  valence: 'Compute live per-atom directional valence slot fill and bond load.',
  sim: 'Accumulate short-range forces, wind/ocean advection, and integrate positions on S².',
  cpu: 'CPU frame bookkeeping and uniform update duration (ms).',
  render: 'WebGPU background field and instanced particle rendering pass duration (ms).',
  total: 'Combined active GPU pass + CPU frame duration (ms).',
  graph: 'Scrolling real-time frame time history chart.',
};

for (const engine in performanceParams) {
  engineFolders[engine] = withTooltip(
    performanceTimes.addFolder({
      title: engine,
      hidden: true,
    }),
    `${engine} Timings`,
    `GPU compute pass timings for the ${engine} spatial engine.`,
  );

  const engineParams =
    performanceParams[engine as keyof typeof performanceParams];

  for (const param in engineParams) {
    withTooltip(
      engineFolders[engine].addBinding(
        engineParams,
        param as keyof typeof engineParams,
        {
          readonly: true,
          interval: 50,
          format: (v: number) => `${v.toFixed(2)}ms`,
        },
      ),
      `GPU Pass: ${param}`,
      passDescriptions[param] ?? 'GPU pass execution duration (ms).',
    );
  }
}

for (const param in globalPerformanceParams) {
  if (param == 'graph') continue;
  withTooltip(
    performanceTimes.addBinding(
      globalPerformanceParams,
      param as keyof typeof globalPerformanceParams,
      {
        readonly: true,
        interval: 50,
        format: (v: number) => `${v.toFixed(2)}ms`,
      },
    ),
    `Timing: ${param}`,
    passDescriptions[param] ?? 'Frame timing metric.',
  );
}

withTooltip(
  performanceTimes.addBinding(globalPerformanceParams, 'graph', {
    label: '',
    readonly: true,
    view: 'graph',
    min: 0,
    max: 1,
    interval: 50,
  }),
  'Frame Time Graph',
  passDescriptions.graph,
);

export function setEngineDisplay(engine: string) {
  for (const engine2 in performanceParams) {
    engineFolders[engine2].hidden = engine2 != engine;
  }
}
