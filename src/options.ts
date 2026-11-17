import { THETA_SAMPLES } from './matrix';

const urlParams = new URLSearchParams(window.location.search);

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
  Math.floor(validateURLParam('particleCount', 50000)),
);
export const colourAmt = Math.max(
  1,
  Math.floor(validateURLParam('colours', 200)),
);
const cellAmt = Math.max(
  1,
  Math.floor(validateURLParam('cellCount', 2000)),
);
const defaultEngine = validateURLParam('engine', 'linkedList');
const defaultRadius = validateURLParam('radius', 15);
const defaultWorldSize = validateURLParam('worldSize', 6);
const defaultTorque = validateURLParam('torque', 12);
const defaultAngularFriction = validateURLParam('angularFriction', 0.05);
const defaultValency = validateURLParam('valency', 0.5);
const defaultTemperature = validateURLParam('temperature', 0);

export const params = {
  fps: 0,
  engine: defaultEngine,
  particles: particleAmt,
  colours: colourAmt,
  cells: cellAmt,
};

export const optionParams = {
  colours: colourAmt,
  r: defaultRadius,
  force: 1,
  beta: 0.3,
  delta: 0.02,
  friction: 0.04,
  torque: defaultTorque,
  angularFriction: defaultAngularFriction,
  valency: defaultValency,
  temperature: defaultTemperature,
  cells: cellAmt,
  avoidance: 4,
  worldSize: defaultWorldSize,
  border: true,
  vortex: false,
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
  simData[10] = optionParams.border ? 1 : 0;
  simData[11] = optionParams.vortex ? 1 : 0;
  simData[12] = optionParams.torque;
  simData[13] = Math.pow(0.5, dt / angFriction);
  simData[14] = optionParams.valency;
  simData[15] = optionParams.temperature;
  simData[16] = THETA_SAMPLES;
  simData[17] = frameCount;
  device.queue.writeBuffer(simBuffer, 0, simData);
}

import { Pane } from 'tweakpane';
import {
  BindingApi,
  type BindingParams,
  type FolderApi,
} from '@tweakpane/core';
import { isSample } from './input';

const pane = new Pane({ title: 'Complex Life' });

pane.addBinding(params, 'fps', { readonly: true });

export const engineSelect = pane.addBinding(params, 'engine', {
  options: {
    'Counting Sort': 'countingSort',
    'Atomic Linked Lists': 'linkedList',
    NSquared: 'nSquared',
    'Prefix Sum': 'prefixSum',
  },
});

pane.addBinding(params, 'particles', { min: 1, step: 1 });

export const newSimBtn = pane.addButton({ title: 'New Sim' });
export const randomizeBtn = pane.addButton({ title: 'Randomize' });

const constantOptions = ['colours', 'cells'];

const constants: Record<string, BindingParams> = {
  colours: { min: 1, step: 1 },
  cells: { min: 1, step: 1 },
};

const constantsFolder = pane.addFolder({ title: 'Constants' });
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

const optionsFolder = pane.addFolder({ title: 'Options' });

const options: Record<string, BindingParams> = {
  beta: { min: 0, max: 1 },
  r: { min: 0.01 },
  delta: { min: 0.001, max: 0.2 },
  friction: { min: 0.001 },
  torque: { min: 0, max: 100 },
  angularFriction: { label: 'angFriction', min: 0.001 },
  valency: { min: 0, max: 10 },
  temperature: { min: 0, max: 5 },
  avoidance: { min: 0 },
  worldSize: { min: 0.01 },
  colours: { min: 1, step: 1 },
  cells: { min: 1, step: 1 },
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
) {
  for (const param in optionBindings)
    optionBindings[param].on('change', () => {
      setSim(device, simData, simBuffer);
    });
}

//

export const performanceParams = {
  countingSort: {
    cell: 0,
    prefix: 0,
    sort: 0,
    sim: 0,
  },
  prefixSum: {
    cell: 0,
    prefix: 0,
    sort: 0,
    sim: 0,
  },
  linkedList: {
    construct: 0,
    sim: 0,
  },
  nSquared: {
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
  expanded: !isSample,
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
