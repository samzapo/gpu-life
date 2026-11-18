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
import { WindStreamRenderer } from './earth/windLines';
import {
  linkRenderTimestamp,
  readTimestamp,
  requestTimestamps,
  resolveTimestamp,
  setupTimestamp,
} from './utils';
import {
  bindOptions,
  coastlinesBinding,
  colourAmt as colourAmtOptn,
  earthParams,
  engineSelect,
  globalPerformanceParams,
  initialOrientation,
  newSimBtn,
  optionParams,
  params,
  particleAmt as particleAmtOptn,
  performanceParams,
  projectionSelect,
  randomizeBtn,
  resetViewBtn,
  seedInput,
  setEngineDisplay,
  setSim,
  updatePeriodicLegend,
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
const windCanvas = document.getElementById('windCanvas') as HTMLCanvasElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const scaleBar = document.getElementById('scaleBar') as HTMLElement;
const windLevelBadge = document.getElementById('windLevelBadge') as HTMLElement;
const windTimeBadge = document.getElementById('windTimeBadge') as HTMLElement;
const scaleMaxLabel = document.getElementById('scaleMaxLabel') as HTMLElement;

const windCtx = windCanvas.getContext('2d', { willReadFrequently: true });
const windStreamRenderer = new WindStreamRenderer();

const uniformsFloats = 28; // 112 bytes
const uniformData = new Float32Array(uniformsFloats);
const simSize = 20;
const simData = new Float32Array(simSize);
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

const particleStride = 48; // 12 floats: pos(3), colour(1), vel(3), omega(1), orient(3), pad(1)
const valenceStride = 16;
const WIND_TEX_W = 360;
const WIND_TEX_H = 181;
const WATERMAN_INV_W = 640;
const WATERMAN_INV_H = 360;

let particleAmt = particleAmtOptn;

let device: GPUDevice | undefined;
let bgContext: GPUCanvasContext | undefined;
let context: GPUCanvasContext | undefined;
let uniformBuffer: GPUBuffer | undefined;
let simBuffer: GPUBuffer | undefined;
let bgRenderPipeline: GPURenderPipeline | undefined;
let renderPipeline: GPURenderPipeline | undefined;

let matrixBuffer: GPUBuffer | undefined;
let colourBuffer: GPUBuffer | undefined;
let valenceBuffer: GPUBuffer | undefined;
let particleBuffers: [GPUBuffer, GPUBuffer] | undefined;

let windTexture: GPUTexture | undefined;
let watermanFwdTexture: GPUTexture | undefined;
let watermanInvTexture: GPUTexture | undefined;

let bgBindGroup: GPUBindGroup | undefined;
let renderBindGroup: GPUBindGroup | undefined;

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

// Globe projection state (defaults to earth.nullschool.net orthographic=-36.25,34.05,586)
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
  scale: Math.max(220, initScale),
  pan: [0, 0],
  width: window.innerWidth,
  height: window.innerHeight,
};

let d3Proj = buildConfiguredD3Projection(viewState);
let earthFeatures: EarthTopoFeatures | null = null;
let currentWindGrid: WindGrid = createEmptyWindGrid();
let watermanDirty = true;
let mapDirty = true;
let viewSignature = '';
let hashSyncTimer: number | undefined;

function createEmptyWindGrid(): WindGrid {
  return {
    nx: WIND_TEX_W,
    ny: WIND_TEX_H,
    u: new Float32Array(WIND_TEX_W * WIND_TEX_H),
    v: new Float32Array(WIND_TEX_W * WIND_TEX_H),
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
    const levelPart =
      spec.surfaceType === 'surface' ? 'surface/level' : `isobaric/${spec.id}`;
    const nextHash = `#current/wind/${levelPart}/${viewState.projectionId}=${lon},${lat},${sc}`;
    if (window.location.hash !== nextHash) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${nextHash}`);
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
  const texData = buildDivergenceFreeWind(grid, WIND_TEX_W, WIND_TEX_H);

  device.queue.writeTexture(
    { texture: windTexture },
    texData,
    { bytesPerRow: WIND_TEX_W * 16 },
    { width: WIND_TEX_W, height: WIND_TEX_H, depthOrArrayLayers: 1 },
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

declare global {
  interface Window {
    complexLife?: {
      getElements: () => Element2D[];
      randomizeMatrix: (seed?: string) => void;
      getSeed: () => string;
      getEngine: () => string;
      getParticleCount: () => number;
      getColourCount: () => number;
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
      readValence: (count?: number) => Promise<number[][]>;
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

async function readValence(count: number = 16): Promise<number[][]> {
  if (!device || !valenceBuffer) return [];
  const n = Math.min(count, particleAmt);
  const byteLength = n * valenceStride;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(valenceBuffer, 0, staging, 0, byteLength);
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

function applySeed(seed: string, updateInput: boolean = true) {
  params.seed = seed;
  if (updateInput) {
    suppressSeedChange = true;
    seedInput.refresh();
    suppressSeedChange = false;
  }
  startParticles();
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
  getParticleCount: () => particleAmt,
  getColourCount: () => colourAmt,
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
  readParticles,
  readValence,
};

function syncCanvasDimensions() {
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  for (const c of [bgCanvas, windCanvas, canvas]) {
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }
  mapSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  viewState.width = w;
  viewState.height = h;
}

(async () => {
  loadEarthFeatures()
    .then((feats) => {
      earthFeatures = feats;
      mapDirty = true;
    })
    .catch(() => {
      // Non-fatal if coastline JSON fails to load
    });

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

  windTexture = device.createTexture({
    size: { width: WIND_TEX_W, height: WIND_TEX_H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'windTexture',
  });
  watermanFwdTexture = device.createTexture({
    size: { width: WIND_TEX_W, height: WIND_TEX_H },
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
  startParticles();
  bindOptions(device, simData, simBuffer);
  loadWindLevel(earthParams.windLevel);
})();

function tick(commandEncoder: GPUCommandEncoder) {
  if (!device) return;
  engines[engine].tick(device, commandEncoder, alternate, particleAmt);
  alternate = (alternate + 1) % 2;
}

function render(commandEncoder: GPUCommandEncoder) {
  if (
    !device ||
    !bgContext ||
    !context ||
    !bgRenderPipeline ||
    !renderPipeline ||
    !bgBindGroup ||
    !renderBindGroup ||
    !particleBuffers ||
    !valenceBuffer
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
  passEncoder.setVertexBuffer(0, particleBuffers[alternate]);
  passEncoder.setVertexBuffer(1, valenceBuffer);
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

  // Uniform area-preserving spawn on S^2 with isotropic tangent slot-0 orientations
  const data = new Float32Array(bufferSize / 4);
  for (let pi = 0; pi < particleAmt; pi++) {
    const z = rng() * 2 - 1;
    const lon = rng() * Math.PI * 2;
    const rxy = Math.sqrt(Math.max(0, 1 - z * z));
    const nx = rxy * Math.cos(lon);
    const ny = rxy * Math.sin(lon);
    const nz = z;

    const refX = Math.abs(nz) > 0.9 ? 1 : 0;
    const refZ = Math.abs(nz) > 0.9 ? 0 : 1;
    // e1 = normalize(cross(ref, n))
    let e1x = 0 * nz - refZ * ny;
    let e1y = refZ * nx - refX * nz;
    let e1z = refX * ny - 0 * nx;
    const e1Len = Math.hypot(e1x, e1y, e1z) || 1;
    e1x /= e1Len;
    e1y /= e1Len;
    e1z /= e1Len;
    // e2 = cross(n, e1)
    const e2x = ny * e1z - nz * e1y;
    const e2y = nz * e1x - nx * e1z;
    const e2z = nx * e1y - ny * e1x;

    const alpha = rng() * Math.PI * 2;
    const ca = Math.cos(alpha);
    const sa = Math.sin(alpha);
    const ux = ca * e1x + sa * e2x;
    const uy = ca * e1y + sa * e2y;
    const uz = ca * e1z + sa * e2z;

    const b = pi * 12;
    data[b + 0] = nx;
    data[b + 1] = ny;
    data[b + 2] = nz;
    data[b + 3] = pickElement();
    data[b + 4] = 0;
    data[b + 5] = 0;
    data[b + 6] = 0;
    data[b + 7] = 0;
    data[b + 8] = ux;
    data[b + 9] = uy;
    data[b + 10] = uz;
    data[b + 11] = 0;
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
    );
  }

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

let lastTime = 0;
const deltaVs: number[] = [];

function writeUniforms() {
  if (!device || !uniformBuffer) return;
  const spec = WIND_LEVELS[earthParams.windLevel];
  const R = Math.max(optionParams.worldSize, 0.25);

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
  uniformData[13] = earthParams.windColors ? 1 : 0;
  uniformData[14] = spec.scaleMax;
  uniformData[15] = spec.streamMax;

  const rotCols = buildD3RotationMatrixColMajor(viewState.rotate);
  uniformData.set(rotCols, 16);

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);
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

  if (windCtx) {
    windStreamRenderer.stepAndDraw(
      windCtx,
      viewState.width,
      viewState.height,
      d3Proj,
      currentWindGrid,
      viewSignature,
      earthParams.windLines,
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

windColorsBinding.on('change', () => {
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
  viewState.scale = Math.round(
    586 * Math.min(viewState.width / 1280, viewState.height / 800) || 586,
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
  globalPerformanceParams.graph = total / deltaTotal;
}

window.onresize = () => {
  syncCanvasDimensions();
  updateProjectionState();
};
