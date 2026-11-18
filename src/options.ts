import { Pane } from 'tweakpane';
import {
  BindingApi,
  type BindingParams,
  type FolderApi,
} from '@tweakpane/core';
import { generateRandomSeed } from './matrix';
import { computeAbundanceDistribution, getElement2D } from './periodicTable2D';
import {
  parseNullschoolHash,
  PROJECTIONS,
  type ProjectionId,
} from './earth/projections';
import { WIND_LEVELS, type WindLevelId } from './earth/windData';
import { setMouseToolMode } from './input';

const urlParams = new URLSearchParams(window.location.search);
const parsedHash = parseNullschoolHash(window.location.hash);

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

export const particleAmt = Math.max(
  1,
  Math.floor(validateURLParam('particleCount', 20000)),
);
export const colourAmt = Math.max(
  1,
  Math.floor(validateURLParam('colours', 14)),
);
const cellAmt = Math.max(
  1,
  Math.floor(validateURLParam('cellCount', 4096)),
);
const defaultEngine = validateURLParam('engine', 'prefixSum');
const defaultRadius = validateURLParam('radius', 12);
const defaultWorldSize = validateURLParam('worldSize', 1.8);
const defaultTorque = validateURLParam('torque', 22);
const defaultAngularFriction = validateURLParam('angularFriction', 0.04);
const defaultValency = validateURLParam('valency', 1.15);
const defaultTemperature = validateURLParam('temperature', 0.12);
const defaultActivation = validateURLParam('activation', 0.15);
const defaultOrbitalSharpness = validateURLParam('orbitalSharpness', 4);
const defaultSeed = validateURLParam('seed', generateRandomSeed());

const urlProj = validateURLParam('projection', '');
const initialProjection: ProjectionId =
  urlProj in PROJECTIONS
    ? (urlProj as ProjectionId)
    : parsedHash.projection && parsedHash.projection in PROJECTIONS
      ? parsedHash.projection
      : 'orthographic';

const urlLevel = validateURLParam('windLevel', '');
const initialLevel: WindLevelId =
  urlLevel in WIND_LEVELS
    ? (urlLevel as WindLevelId)
    : parsedHash.level && parsedHash.level in WIND_LEVELS
      ? (parsedHash.level as WindLevelId)
      : 'surface';

const urlLayer = validateURLParam('layer', '');
export type ViewLayerMode = 'both' | 'ocean' | 'air';
const initialViewLayer: ViewLayerMode =
  urlLayer === 'ocean' || urlLayer === 'air' || urlLayer === 'both'
    ? urlLayer
    : parsedHash.mode ?? 'both';

export const oceanParticleAmt = Math.max(
  1,
  Math.floor(validateURLParam('oceanParticles', 12000)),
);
export const oceanColourAmt = Math.max(
  1,
  Math.floor(validateURLParam('oceanColours', 12)),
);

export const initialOrientation = parsedHash.orientation;

export const params = {
  fps: 0,
  engine: defaultEngine,
  particles: particleAmt,
  colours: colourAmt,
  cells: cellAmt,
  seed: defaultSeed,
};

export const earthParams = {
  viewLayer: initialViewLayer,
  projection: initialProjection,
  windLevel: initialLevel,
  vortex: true,
  windStrength: validateURLParam('windStrength', 1.0),
  windLines: urlParams.get('windLines') !== 'false',
  windColors: urlParams.get('windColors') !== 'false',
  oceanMix: urlParams.get('oceanMix') !== 'false',
  oceanStrength: validateURLParam('oceanStrength', 1.0),
  oceanLines: urlParams.get('oceanLines') !== 'false',
  oceanColors: urlParams.get('oceanColors') !== 'false',
  coastlines: urlParams.get('coastlines') !== 'false',
  mouseTool: 'globe' as 'globe' | 'attract' | 'repel',
  windStatus: 'Loading GFS…',
  oceanStatus: 'Loading CoastWatch…',
};

// Classic Particle Life parameters preserved from the original fork (51bc381)
export const oceanOptionParams = {
  particles: oceanParticleAmt,
  colours: oceanColourAmt,
  r: validateURLParam('oceanR', 20),
  force: 1.0,
  beta: 0.3,
  delta: 0.02,
  friction: 0.04,
  avoidance: 4.0,
  cells: cellAmt,
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

export const optionParams = {
  colours: colourAmt,
  r: defaultRadius,
  force: 2.5,
  beta: 0.4,
  delta: 0.02,
  friction: 0.04,
  torque: defaultTorque,
  angularFriction: defaultAngularFriction,
  valency: defaultValency,
  temperature: defaultTemperature,
  activation: defaultActivation,
  orbitalSharpness: defaultOrbitalSharpness,
  showOrbitals: true,
  cells: cellAmt,
  avoidance: 4,
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
  device.queue.writeBuffer(simBuffer, 0, simData);
}

const pane = new Pane({
  title: 'Complex Life · Earth Wind',
  expanded: window.innerWidth >= 768,
});

pane.addBinding(params, 'fps', { readonly: true });

export const engineSelect = pane.addBinding(params, 'engine', {
  options: {
    'Prefix Sum': 'prefixSum',
    'Counting Sort': 'countingSort',
    'Atomic Linked Lists': 'linkedList',
  },
});

pane.addBinding(params, 'particles', { min: 1, step: 1 });
export const seedInput = pane.addBinding(params, 'seed', { view: 'text' });
export const newSimBtn = pane.addButton({ title: 'New Sim' });
export const randomizeBtn = pane.addButton({ title: 'Randomize' });

// Earth & Wind / Ocean folder (emulating earth.nullschool.net layers, levels & projections)
const earthFolder = pane.addFolder({
  title: 'Earth · Wind & Ocean',
  expanded: true,
});

export const viewLayerSelect = earthFolder.addBinding(earthParams, 'viewLayer', {
  label: 'layer',
  options: {
    'Both (Air + Ocean)': 'both',
    'Ocean (Currents)': 'ocean',
    'Air (Wind)': 'air',
  },
});

const projectionOptions: Record<string, ProjectionId> = {};
for (const key of Object.keys(PROJECTIONS) as ProjectionId[]) {
  projectionOptions[PROJECTIONS[key].label] = key;
}

const windLevelOptions: Record<string, WindLevelId> = {};
for (const key of Object.keys(WIND_LEVELS) as WindLevelId[]) {
  windLevelOptions[WIND_LEVELS[key].label] = key;
}

export const projectionSelect = earthFolder.addBinding(earthParams, 'projection', {
  label: 'proj',
  options: projectionOptions,
});
export const windLevelSelect = earthFolder.addBinding(earthParams, 'windLevel', {
  label: 'windLevel',
  options: windLevelOptions,
});
export const windMixBinding = earthFolder.addBinding(earthParams, 'vortex', {
  label: 'windMix',
});
export const windStrengthBinding = earthFolder.addBinding(
  earthParams,
  'windStrength',
  { label: 'windForce', min: 0, max: 4, step: 0.05 },
);
export const windLinesBinding = earthFolder.addBinding(earthParams, 'windLines');
export const windColorsBinding = earthFolder.addBinding(
  earthParams,
  'windColors',
);
export const oceanMixBinding = earthFolder.addBinding(earthParams, 'oceanMix', {
  label: 'oceanMix',
});
export const oceanStrengthBinding = earthFolder.addBinding(
  earthParams,
  'oceanStrength',
  { label: 'oceanForce', min: 0, max: 4, step: 0.05 },
);
export const oceanLinesBinding = earthFolder.addBinding(
  earthParams,
  'oceanLines',
);
export const oceanColorsBinding = earthFolder.addBinding(
  earthParams,
  'oceanColors',
);
export const coastlinesBinding = earthFolder.addBinding(
  earthParams,
  'coastlines',
);
const mouseToolBinding = earthFolder.addBinding(earthParams, 'mouseTool', {
  label: 'tool',
  options: {
    'Rotate / Pan': 'globe',
    'Attract Particles': 'attract',
    'Repel Particles': 'repel',
  },
});
mouseToolBinding.on('change', (ev) => {
  setMouseToolMode(ev.value);
});

export const windStatusBinding = earthFolder.addBinding(
  earthParams,
  'windStatus',
  { readonly: true, label: 'windSrc' },
);
export const oceanStatusBinding = earthFolder.addBinding(
  earthParams,
  'oceanStatus',
  { readonly: true, label: 'oceanSrc' },
);
export const resetViewBtn = earthFolder.addButton({ title: 'Reset Globe View' });

// Classic Particle Life mechanics from the original fork (51bc381) running on Ocean Currents
const oceanLifeFolder = pane.addFolder({
  title: 'Ocean Life (Original Fork)',
  expanded: false,
});
oceanLifeFolder.addBinding(oceanOptionParams, 'particles', {
  label: 'oceanCount',
  min: 500,
  max: 40000,
  step: 500,
});
export const oceanColoursBinding = oceanLifeFolder.addBinding(
  oceanOptionParams,
  'colours',
  { label: 'species', min: 1, max: 200, step: 1 },
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
const oceanParamBindings: BindingApi[] = [];
for (const key in oceanParamSpecs) {
  oceanParamBindings.push(
    oceanLifeFolder.addBinding(
      oceanOptionParams,
      key as keyof typeof oceanOptionParams,
      oceanParamSpecs[key],
    ),
  );
}
export const newOceanSimBtn = oceanLifeFolder.addButton({
  title: 'Respawn Ocean Life',
});
export const randomizeOceanRulesBtn = oceanLifeFolder.addButton({
  title: 'Randomize Ocean Matrix',
});

const constantOptions = ['colours', 'cells'];
const constants: Record<string, BindingParams> = {
  colours: { min: 1, max: 62, step: 1 },
  cells: { min: 1, step: 1 },
};

const constantsFolder = pane.addFolder({ title: 'Constants', expanded: false });
const optionBindings: Record<string, BindingApi> = {};
for (const param in optionParams) {
  if (!constantOptions.includes(param)) continue;
  const binding = constantsFolder.addBinding(
    optionParams,
    param as keyof typeof optionParams,
    param in constants ? constants[param] : {},
  );
  optionBindings[param] = binding;
}

const optionsFolder = pane.addFolder({
  title: 'Chemistry Options',
  expanded: false,
});

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
  showOrbitals: { label: 'showValence' },
  avoidance: { min: 0 },
  worldSize: { label: 'sphereRadius', min: 0.5, max: 5.0 },
};

for (const param in optionParams) {
  if (constantOptions.includes(param)) continue;
  const binding = optionsFolder.addBinding(
    optionParams,
    param as keyof typeof optionParams,
    param in options ? options[param] : {},
  );
  optionBindings[param] = binding;
}

export function bindOptions(
  device: GPUDevice,
  simData: Float32Array<ArrayBuffer>,
  simBuffer: GPUBuffer,
  oceanSimData: Float32Array<ArrayBuffer>,
  oceanSimBuffer: GPUBuffer,
) {
  for (const param in optionBindings) {
    optionBindings[param].on('change', () => {
      setSim(device, simData, simBuffer);
      if (param === 'worldSize') {
        setOceanSim(device, oceanSimData, oceanSimBuffer);
      }
    });
  }
  windMixBinding.on('change', () => setSim(device, simData, simBuffer));
  windStrengthBinding.on('change', () => setSim(device, simData, simBuffer));

  for (const b of oceanParamBindings) {
    b.on('change', () => setOceanSim(device, oceanSimData, oceanSimBuffer));
  }
  oceanColoursBinding.on('change', () =>
    setOceanSim(device, oceanSimData, oceanSimBuffer),
  );
  oceanMixBinding.on('change', () =>
    setOceanSim(device, oceanSimData, oceanSimBuffer),
  );
  oceanStrengthBinding.on('change', () =>
    setOceanSim(device, oceanSimData, oceanSimBuffer),
  );
}

// 2D Periodic Table legend folder
const periodicFolder = pane.addFolder({
  title: '2D Periodic Table',
  expanded: false,
});
const periodicState: Record<string, string> = {};
const periodicBindings: BindingApi[] = [];

export function updatePeriodicLegend(colourCount: number) {
  for (const b of periodicBindings) {
    periodicFolder.remove(b);
  }
  periodicBindings.length = 0;
  const probs = computeAbundanceDistribution(colourCount);
  const shown = Math.min(colourCount, 24);
  for (let c = 0; c < shown; c++) {
    const el = getElement2D(c);
    const key = `Z${el.atomicNumber}`;
    const pct = (probs[c] * 100).toFixed(probs[c] >= 0.01 ? 1 : 2);
    periodicState[key] = `${el.name.replace(`Z=${el.atomicNumber} `, '')} (${pct}%)`;
    periodicBindings.push(
      periodicFolder.addBinding(periodicState, key, { readonly: true }),
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

const performanceTimes = pane.addFolder({
  title: 'Performance Times',
  expanded: false,
});
const engineFolders: Record<string, FolderApi> = {};

for (const engine in performanceParams) {
  engineFolders[engine] = performanceTimes.addFolder({
    title: engine,
    hidden: true,
  });

  const engineParams =
    performanceParams[engine as keyof typeof performanceParams];

  for (const param in engineParams) {
    engineFolders[engine].addBinding(
      engineParams,
      param as keyof typeof engineParams,
      {
        readonly: true,
        interval: 50,
        format: (v: number) => `${v.toFixed(2)}ms`,
      },
    );
  }
}

for (const param in globalPerformanceParams) {
  if (param == 'graph') continue;
  performanceTimes.addBinding(
    globalPerformanceParams,
    param as keyof typeof globalPerformanceParams,
    {
      readonly: true,
      interval: 50,
      format: (v: number) => `${v.toFixed(2)}ms`,
    },
  );
}

performanceTimes.addBinding(globalPerformanceParams, 'graph', {
  label: '',
  readonly: true,
  view: 'graph',
  min: 0,
  max: 1,
  interval: 50,
});

export function setEngineDisplay(engine: string) {
  for (const engine2 in performanceParams) {
    engineFolders[engine2].hidden = engine2 != engine;
  }
}
