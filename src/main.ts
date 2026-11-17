import { tcamera, mouse } from './input';
import {
  THETA_SAMPLES,
  createSeededRng,
  generateRandomSeed,
  makeRandomMatrix,
  sampleMatrix,
  type MatrixGrid,
  type ThetaFunction,
} from './matrix';
import renderShaders from './render.wgsl?raw';
import {
  hslToRgb,
  lerp5,
  linkRenderTimestamp,
  readTimestamp,
  requestTimestamps,
  resolveTimestamp,
  setupTimestamp,
} from './utils';
import {
  engineSelect,
  globalPerformanceParams,
  newSimBtn,
  optionParams,
  params,
  performanceParams,
  randomizeBtn,
  seedInput,
  setEngineDisplay,
  setSim,
  particleAmt as particleAmtOptn,
  colourAmt as colourAmtOptn,
  bindOptions,
} from './options';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

const uniformsSize = 16;
const uniformData = new Float32Array(uniformsSize);

const simSize = 20;
const simData = new Float32Array(simSize);
const frameBuf = new Float32Array(1);

let colourAmt = colourAmtOptn;
let colours: [number, number, number][] = [];
for (let i = 0; i < colourAmt; i++) {
  colours.push(hslToRgb((i / colourAmt) * 360, 1, 0.5));
}
let matrix: MatrixGrid = makeRandomMatrix(colourAmt, params.seed);

const particleStride = 32;

const multistep = 1;

const camera = { ...tcamera };
const cameraData = new Float32Array([camera.x, camera.y, camera.zoom]);

let particleAmt = particleAmtOptn;

let device: GPUDevice | undefined;
let context: GPUCanvasContext | undefined;
let uniformBuffer: GPUBuffer | undefined;
let simBuffer: GPUBuffer | undefined;
let renderPipeline: GPURenderPipeline | undefined;
let cameraBuffer: GPUBuffer | undefined;

let matrixBuffer: GPUBuffer | undefined;
let colourBuffer: GPUBuffer | undefined;

let particleBuffers: [GPUBuffer, GPUBuffer] | undefined;

let renderBindGroup: GPUBindGroup | undefined;

let alternate = 0;
let fpsc = 0;
let frameCount = 0;

import * as nSquared from './nSquared/main';
import * as linkedList from './linkedList/main';
import * as countingSort from './countingSort/main';
import * as prefixSum from './countingSortOptimized/main';
const engines: Record<
  string,
  typeof nSquared | typeof linkedList | typeof countingSort | typeof prefixSum
> = {
  nSquared,
  linkedList,
  countingSort,
};

let engine: string = params.engine in engines ? params.engine : 'linkedList';
setEngineDisplay(engine);

function writeMatrixBuffer() {
  if (!device || !matrixBuffer) return;
  const matrixData = sampleMatrix(matrix, colourAmt, THETA_SAMPLES);
  device.queue.writeBuffer(matrixBuffer, 0, matrixData.buffer);
}

function setMatrixFunction(c1: number, c2: number, fn: ThetaFunction) {
  if (c1 < 0 || c1 >= colourAmt || c2 < 0 || c2 >= colourAmt) {
    throw new RangeError(`Colour index out of range [0, ${colourAmt})`);
  }
  matrix[c1][c2] = fn;
  if (!device || !matrixBuffer) return;
  const slice = new Float32Array(THETA_SAMPLES);
  const step = (Math.PI * 2) / THETA_SAMPLES;
  for (let k = 0; k < THETA_SAMPLES; k++) {
    slice[k] = fn(k * step);
  }
  const byteOffset =
    (c1 * colourAmt + c2) * THETA_SAMPLES * Float32Array.BYTES_PER_ELEMENT;
  device.queue.writeBuffer(matrixBuffer, byteOffset, slice.buffer);
}

let suppressSeedChange = false;

function applyMatrixSeed(seed: string, updateInput: boolean = true) {
  params.seed = seed;
  if (updateInput) {
    suppressSeedChange = true;
    seedInput.refresh();
    suppressSeedChange = false;
  }
  matrix = makeRandomMatrix(colourAmt, seed);
  writeMatrixBuffer();
}

declare global {
  interface Window {
    complexLife?: {
      getMatrix: () => MatrixGrid;
      setMatrixFunction: (c1: number, c2: number, fn: ThetaFunction) => void;
      randomizeMatrix: (seed?: string) => void;
      getSeed: () => string;
      getThetaSamples: () => number;
      getEngine: () => string;
      getParticleCount: () => number;
      getColourCount: () => number;
      readParticles: (count?: number) => Promise<number[][]>;
    };
  }
}

async function readParticles(count: number = 16): Promise<number[][]> {
  if (!device || !particleBuffers) return [];
  const n = Math.min(count, particleAmt);
  const byteLength = n * particleStride;
  const activeBuf = particleBuffers[alternate];
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(activeBuf, 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const f32 = new Float32Array(staging.getMappedRange());
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    out.push([
      f32[i * 8],
      f32[i * 8 + 1],
      f32[i * 8 + 2],
      f32[i * 8 + 3],
      f32[i * 8 + 4],
      f32[i * 8 + 5],
      f32[i * 8 + 6],
    ]);
  }
  staging.unmap();
  staging.destroy();
  return out;
}

window.complexLife = {
  getMatrix: () => matrix,
  setMatrixFunction,
  randomizeMatrix: (seed?: string) => {
    applyMatrixSeed(seed ?? generateRandomSeed(), true);
  },
  getSeed: () => params.seed,
  getThetaSamples: () => THETA_SAMPLES,
  getEngine: () => engine,
  getParticleCount: () => particleAmt,
  getColourCount: () => colourAmt,
  readParticles,
};

(async () => {
  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'compatibility',
  });

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

  context = canvas.getContext('webgpu') ?? undefined;

  if (!context) return;

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({ device, format: presentationFormat });

  //

  uniformBuffer = device.createBuffer({
    size: uniformsSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'uniformBuffer',
  });

  simBuffer = device.createBuffer({
    size: simSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'simBuffer',
  });

  cameraBuffer = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'cameraBuffer',
  });

  //

  for (const engineName in engines) {
    engines[engineName].setup(device);
  }

  const renderModule = device.createShaderModule({ code: renderShaders });

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
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 8, format: 'float32x2' },
            { shaderLocation: 2, offset: 16, format: 'float32' },
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
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'zero',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  setupTimestamp(device, 'render');

  startParticles();

  bindOptions(device, simData, simBuffer);
})();

function tick(commandEncoder: GPUCommandEncoder) {
  if (!device) return;
  engines[engine].tick(device, commandEncoder, alternate, particleAmt);
  alternate = (alternate + 1) % 2;
}

function render(context: GPUCanvasContext, commandEncoder: GPUCommandEncoder) {
  if (!renderPipeline || !particleBuffers || !device) return;

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
  passEncoder.setVertexBuffer(0, particleBuffers[(alternate + 1) % 2]);
  passEncoder.setBindGroup(0, renderBindGroup);
  passEncoder.draw(6, particleAmt, 0, 0);
  passEncoder.end();

  resolveTimestamp(commandEncoder, 'render');
}

function startParticles() {
  if (
    !device ||
    !uniformBuffer ||
    !renderPipeline ||
    !simBuffer ||
    !cameraBuffer
  )
    return;

  if (particleBuffers) {
    particleBuffers[0].destroy();
    particleBuffers[1].destroy();
  }
  matrixBuffer?.destroy();
  colourBuffer?.destroy();

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

  alternate = 0;
  frameCount = 0;
  const particleRng = createSeededRng(`${params.seed}:particles`);
  const data = new Float32Array(bufferSize / 4);
  let pi = 0;
  while (pi < particleAmt) {
    const spawnAmt = ((particleRng() * (particleAmt - pi)) / colourAmt) * 5;
    const c = Math.floor(particleRng() * colourAmt);

    const a = particleRng() * Math.PI * 2;
    const d = particleRng() * optionParams.worldSize * 0.9;

    const x = Math.cos(a) * d;
    const y = Math.sin(a) * d;
    for (let i = 0; i < spawnAmt && pi < particleAmt; i++) {
      const a2 = particleRng() * Math.PI * 2;
      const d2 = (particleRng() ** 3 / 10) * optionParams.worldSize;
      data[pi * 8] = x + Math.cos(a2) * d2;
      data[pi * 8 + 1] = y + Math.sin(a2) * d2;
      data[pi * 8 + 2] = 0;
      data[pi * 8 + 3] = 0;
      data[pi * 8 + 4] = c;
      data[pi * 8 + 5] = particleRng() * Math.PI * 2;
      data[pi * 8 + 6] = 0;
      data[pi * 8 + 7] = 0;

      pi++;
    }
  }

  device.queue.writeBuffer(particleBuffers[0], 0, data.buffer);

  setSim(device, simData, simBuffer, frameCount);

  matrixBuffer = device.createBuffer({
    size: colourAmt * colourAmt * THETA_SAMPLES * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  writeMatrixBuffer();

  colourBuffer = device.createBuffer({
    size: colourAmt * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const colourData = new Float32Array(colourAmt * 3);
  for (let c = 0; c < colourAmt; c++) {
    colourData[c * 3] = colours[c][0];
    colourData[c * 3 + 1] = colours[c][1];
    colourData[c * 3 + 2] = colours[c][2];
  }
  device.queue.writeBuffer(colourBuffer, 0, colourData.buffer);

  for (const engineName in engines) {
    engines[engineName].start(
      device,
      uniformBuffer,
      simBuffer,
      matrixBuffer,
      particleBuffers,
      particleAmt,
      params.cells,
    );
  }

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: cameraBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: colourBuffer,
        },
      },
    ],
  });
}

let lastTime = 0;
const deltaVs: number[] = [];

function update() {
  requestAnimationFrame(update);
  if (!device || !context) return;

  const start = performance.now();
  const delta = (start - lastTime) / 1000;
  deltaVs.push(start - lastTime);
  if (deltaVs.length > 1000) {
    deltaVs.splice(0, 1);
  }
  lastTime = start;

  camera.x = lerp5(camera.x, tcamera.x, delta * 50);
  camera.y = lerp5(camera.y, tcamera.y, delta * 50);
  camera.zoom = lerp5(camera.zoom, tcamera.zoom, delta * 50);

  frameCount = (frameCount + 1) % 1000000;
  if (simBuffer && optionParams.temperature > 0) {
    frameBuf[0] = frameCount;
    simData[17] = frameCount;
    device.queue.writeBuffer(simBuffer, 17 * 4, frameBuf);
  }

  const commandEncoder = device.createCommandEncoder();

  if (uniformBuffer) {
    uniformData[0] = canvas.width / canvas.height;

    uniformData[4] = mouse.x;
    uniformData[5] = mouse.y;
    uniformData[6] = mouse.down;
    uniformData[7] = mouse.type;

    uniformData[8] = (1 / optionParams.r) * 0.015;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  }

  if (cameraBuffer) {
    cameraData[0] = camera.x;
    cameraData[1] = camera.y;
    cameraData[2] = camera.zoom;

    device.queue.writeBuffer(cameraBuffer, 0, cameraData);
  }

  for (let i = 0; i < multistep; i++) {
    tick(commandEncoder);
  }

  render(context, commandEncoder);

  const commands = commandEncoder.finish();
  device.queue.submit([commands]);

  device.popErrorScope().then((error) => {
    if (error && !window.location.search.includes('noTimestamp')) {
      // some weird bug happened with timestamps, just disable it and restart the simulation
      window.location.href +=
        (window.location.search ? '&' : '?') + 'noTimestamp';
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
  colourAmt = Math.max(1, Math.floor(optionParams.colours));
  params.colours = colourAmt;
  colours = [];
  for (let i = 0; i < colourAmt; i++) {
    colours.push(hslToRgb((i / colourAmt) * 360, 1, 0.5));
  }

  params.cells = Math.max(1, Math.floor(optionParams.cells));
  particleAmt = Math.max(1, Math.floor(params.particles));

  matrix = makeRandomMatrix(colourAmt, params.seed);
  startParticles();
});

randomizeBtn.on('click', () => {
  if (!device || !matrixBuffer) return;
  applyMatrixSeed(generateRandomSeed(), true);
});

seedInput.on('change', () => {
  if (suppressSeedChange || !device || !matrixBuffer) return;
  applyMatrixSeed(params.seed, false);
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
  globalPerformanceParams.graph = total / deltaTotal;
}

window.onresize = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
};
