import {
  parseNullschoolHash,
  PROJECTIONS,
  type ProjectionId,
} from './earth/projections';

export const CONFIG_STORAGE_KEY = 'complexLife:config:v2';

export const FACTORY_DEFAULTS = {
  engine: 'prefixSum',
  particles: 10000,
  colours: 14,
  cells: 4096,
  seed: '19825834',
  oceanSeed: '211716319',
  air: {
    r: 12,
    beta: 0.4,
    force: 3.8,
    delta: 0.016,
    friction: 0.04,
    torque: 42,
    angularFriction: 0.03,
    valency: 1.35,
    temperature: 0.1,
    activation: 0.12,
    orbitalSharpness: 6,
    bondStability: 0.82,
    exchangeDrive: 0.85,
    shearBreak: 1.1,
    shearFloor: 0.2,
    showOrbitals: true,
    avoidance: 4,
    worldSize: 1.8,
  },
  ocean: {
    colours: 12,
    r: 20,
    force: 1.0,
    beta: 0.3,
    delta: 0.02,
    friction: 0.04,
    avoidance: 4.0,
  },
  earth: {
    viewLayer: 'both' as ConfigViewLayerMode,
    projection: 'orthographic' as ProjectionId,
    vortex: true,
    windStrength: 1.0,
    windLines: 1.0,
    windColors: 0.0,
    chemOpacity: 0.6,
    oceanMix: true,
    oceanStrength: 0.85,
    oceanLines: 1.0,
    oceanColors: 0.0,
    tempColors: 0.15,
    localTemp: true,
    tempStrength: 1.0,
    uvColors: 0.0,
    localUv: true,
    uvStrength: 1.0,
    rhColors: 0.2,
    precipColors: 0.25,
    localHydro: true,
    hydroStrength: 1.0,
    lifeOpacity: 1.0,
    mineralOpacity: 1.0,
    terrainOpacity: 0.85,
    nightShading: 0.35,
    coastlines: true,
    bioLink: true,
    bioRate: 1.0,
    mineralLink: true,
    mineralRate: 1.0,
    tooltips: true,
    mouseTool: 'globe' as ConfigMouseTool,
  },
  view: {
    rotate: [
      -146.1304722783222,
      -43.961772363420415,
      -6.251222977677344,
    ] as [number, number, number],
    pan: [0, 0] as [number, number],
    scale: 2971.0524096688187,
  },
};

export const CONFIG_URL_KEYS: ReadonlySet<string> = new Set([
  'engine',
  'seed',
  'oceanSeed',
  'particleCount',
  'colours',
  'cellCount',
  'radius',
  'beta',
  'force',
  'delta',
  'friction',
  'torque',
  'angularFriction',
  'valency',
  'temperature',
  'activation',
  'orbitalSharpness',
  'bondStability',
  'exchangeDrive',
  'shearBreak',
  'shearFloor',
  'showOrbitals',
  'avoidance',
  'worldSize',
  'oceanParticles',
  'oceanColours',
  'oceanR',
  'oceanBeta',
  'oceanForce',
  'oceanDelta',
  'oceanFriction',
  'oceanAvoidance',
  'oceanCells',
  'projection',
  'layer',
  'windMix',
  'vortex',
  'windStrength',
  'windLines',
  'windColors',
  'chemOpacity',
  'oceanMix',
  'oceanStrength',
  'oceanLines',
  'oceanColors',
  'tempColors',
  'localTemp',
  'tempStrength',
  'uvColors',
  'localUv',
  'uvStrength',
  'rhColors',
  'precipColors',
  'localHydro',
  'hydroStrength',
  'lifeOpacity',
  'mineralOpacity',
  'terrainOpacity',
  'nightShading',
  'coastlines',
  'bioLink',
  'bioRate',
  'mineralLink',
  'mineralRate',
  'tooltips',
  'mouseTool',
  'pan',
]);

export function hasConfigQueryParams(search: string): boolean {
  if (!search || search === '?') return false;
  const sp = new URLSearchParams(search);
  for (const k of sp.keys()) {
    if (CONFIG_URL_KEYS.has(k)) return true;
  }
  return false;
}

export function parsePanParam(raw: string | null): [number, number] | undefined {
  if (!raw) return undefined;
  const parts = raw.split(',').map(Number);
  if (parts.length === 2 && parts.every(Number.isFinite)) {
    return [parts[0], parts[1]];
  }
  return undefined;
}

export type ConfigViewLayerMode = 'both' | 'ocean' | 'air';
export type ConfigMouseTool = 'globe' | 'attract' | 'repel';

export interface SimConfigV1 {
  v: 1;
  engine?: string;
  particles?: number;
  colours?: number;
  cells?: number;
  seed?: string;
  oceanSeed?: string;
  air?: {
    r?: number;
    beta?: number;
    force?: number;
    delta?: number;
    friction?: number;
    torque?: number;
    angularFriction?: number;
    valency?: number;
    temperature?: number;
    activation?: number;
    orbitalSharpness?: number;
    showOrbitals?: boolean;
    avoidance?: number;
    worldSize?: number;
    bondStability?: number;
    exchangeDrive?: number;
    shearBreak?: number;
    shearFloor?: number;
  };
  ocean?: {
    particles?: number;
    colours?: number;
    r?: number;
    beta?: number;
    force?: number;
    delta?: number;
    friction?: number;
    avoidance?: number;
    cells?: number;
  };
  earth?: {
    viewLayer?: ConfigViewLayerMode;
    projection?: ProjectionId;
    vortex?: boolean;
    windStrength?: number;
    windLines?: number;
    windColors?: number;
    chemOpacity?: number;
    oceanMix?: boolean;
    oceanStrength?: number;
    oceanLines?: number;
    oceanColors?: number;
    tempColors?: number;
    localTemp?: boolean;
    tempStrength?: number;
    uvColors?: number;
    localUv?: boolean;
    uvStrength?: number;
    rhColors?: number;
    precipColors?: number;
    localHydro?: boolean;
    hydroStrength?: number;
    lifeOpacity?: number;
    mineralOpacity?: number;
    terrainOpacity?: number;
    nightShading?: number;
    coastlines?: boolean;
    bioLink?: boolean;
    bioRate?: number;
    mineralLink?: boolean;
    mineralRate?: number;
    tooltips?: boolean;
    mouseTool?: ConfigMouseTool;
  };
  view?: {
    rotate?: [number, number, number];
    pan?: [number, number];
    scale?: number;
  };
}

export function loadCachedConfig(): SimConfigV1 | null {
  try {
    const raw = window.localStorage.getItem(CONFIG_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object') return null;
    return parsed as SimConfigV1;
  } catch {
    return null;
  }
}

export function saveCachedConfig(cfg: SimConfigV1): void {
  try {
    window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(cfg));
  } catch {
    // Ignore storage quota or private browsing errors
  }
}

export function clearCachedConfig(): void {
  try {
    window.localStorage.removeItem(CONFIG_STORAGE_KEY);
  } catch {
    // Ignore storage errors
  }
}

function finiteNum(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function boolVal(v: unknown): boolean | undefined {
  return typeof v === 'boolean' ? v : undefined;
}

function unitOpacity(v: unknown): number | undefined {
  const n = finiteNum(v);
  return n !== undefined ? Math.max(0, Math.min(1, n)) : undefined;
}

function parseFromUrlString(text: string): SimConfigV1 | null {
  const trimmed = text.trim();
  if (
    !trimmed.startsWith('http://') &&
    !trimmed.startsWith('https://') &&
    !trimmed.startsWith('?') &&
    !trimmed.startsWith('#')
  ) {
    return null;
  }

  let searchStr = '';
  let hashStr = '';
  try {
    const u = trimmed.startsWith('http')
      ? new URL(trimmed)
      : new URL(trimmed, window.location.origin);
    searchStr = u.search;
    hashStr = u.hash;
  } catch {
    return null;
  }

  const sp = new URLSearchParams(searchStr);
  const hasCfgQuery = hasConfigQueryParams(searchStr);
  const hashParsed = parseNullschoolHash(hashStr);
  const cfg: SimConfigV1 = hasCfgQuery
    ? {
        v: 1,
        engine: FACTORY_DEFAULTS.engine,
        particles: FACTORY_DEFAULTS.particles,
        colours: FACTORY_DEFAULTS.colours,
        cells: FACTORY_DEFAULTS.cells,
        air: { ...FACTORY_DEFAULTS.air },
        ocean: { ...FACTORY_DEFAULTS.ocean },
        earth: { ...FACTORY_DEFAULTS.earth },
        view: { pan: [0, 0] },
      }
    : { v: 1, earth: {}, view: {} };

  const numParam = (k: string): number | undefined => {
    const raw = sp.get(k);
    if (raw === null) return undefined;
    const n = Number(raw);
    return Number.isFinite(n) ? n : undefined;
  };
  const opParam = (k: string): number | undefined => {
    const raw = sp.get(k);
    if (raw === null) return undefined;
    if (raw === 'true') return 1;
    if (raw === 'false') return 0;
    const n = Number(raw);
    return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : undefined;
  };

  if (hasCfgQuery) {
    if (sp.has('engine')) cfg.engine = sp.get('engine') ?? FACTORY_DEFAULTS.engine;
    if (sp.has('seed')) cfg.seed = sp.get('seed') ?? undefined;
    if (sp.has('oceanSeed')) cfg.oceanSeed = sp.get('oceanSeed') ?? undefined;
    cfg.particles = numParam('particleCount') ?? FACTORY_DEFAULTS.particles;
    cfg.colours = numParam('colours') ?? FACTORY_DEFAULTS.colours;
    cfg.cells = numParam('cellCount') ?? FACTORY_DEFAULTS.cells;

    cfg.air!.r = numParam('radius') ?? FACTORY_DEFAULTS.air.r;
    cfg.air!.beta = numParam('beta') ?? FACTORY_DEFAULTS.air.beta;
    cfg.air!.force = numParam('force') ?? FACTORY_DEFAULTS.air.force;
    cfg.air!.delta = numParam('delta') ?? FACTORY_DEFAULTS.air.delta;
    cfg.air!.friction = numParam('friction') ?? FACTORY_DEFAULTS.air.friction;
    cfg.air!.torque = numParam('torque') ?? FACTORY_DEFAULTS.air.torque;
    cfg.air!.angularFriction = numParam('angularFriction') ?? FACTORY_DEFAULTS.air.angularFriction;
    cfg.air!.valency = numParam('valency') ?? FACTORY_DEFAULTS.air.valency;
    cfg.air!.temperature = numParam('temperature') ?? FACTORY_DEFAULTS.air.temperature;
    cfg.air!.activation = numParam('activation') ?? FACTORY_DEFAULTS.air.activation;
    cfg.air!.orbitalSharpness = numParam('orbitalSharpness') ?? FACTORY_DEFAULTS.air.orbitalSharpness;
    cfg.air!.bondStability = numParam('bondStability') ?? FACTORY_DEFAULTS.air.bondStability;
    cfg.air!.exchangeDrive = numParam('exchangeDrive') ?? FACTORY_DEFAULTS.air.exchangeDrive;
    cfg.air!.shearBreak = numParam('shearBreak') ?? FACTORY_DEFAULTS.air.shearBreak;
    cfg.air!.shearFloor = numParam('shearFloor') ?? FACTORY_DEFAULTS.air.shearFloor;
    if (sp.has('showOrbitals')) cfg.air!.showOrbitals = sp.get('showOrbitals') !== 'false';
    cfg.air!.avoidance = numParam('avoidance') ?? FACTORY_DEFAULTS.air.avoidance;
    cfg.air!.worldSize = numParam('worldSize') ?? FACTORY_DEFAULTS.air.worldSize;

    cfg.ocean!.colours = numParam('oceanColours') ?? FACTORY_DEFAULTS.ocean.colours;
    cfg.ocean!.r = numParam('oceanR') ?? FACTORY_DEFAULTS.ocean.r;
    cfg.ocean!.beta = numParam('oceanBeta') ?? FACTORY_DEFAULTS.ocean.beta;
    cfg.ocean!.force = numParam('oceanForce') ?? FACTORY_DEFAULTS.ocean.force;
    cfg.ocean!.delta = numParam('oceanDelta') ?? FACTORY_DEFAULTS.ocean.delta;
    cfg.ocean!.friction = numParam('oceanFriction') ?? FACTORY_DEFAULTS.ocean.friction;
    cfg.ocean!.avoidance = numParam('oceanAvoidance') ?? FACTORY_DEFAULTS.ocean.avoidance;

    if (sp.has('windMix')) cfg.earth!.vortex = sp.get('windMix') !== 'false';
    else if (sp.has('vortex')) cfg.earth!.vortex = sp.get('vortex') !== 'false';
    cfg.earth!.windStrength = numParam('windStrength') ?? FACTORY_DEFAULTS.earth.windStrength;
    cfg.earth!.oceanStrength = numParam('oceanStrength') ?? FACTORY_DEFAULTS.earth.oceanStrength;
    cfg.earth!.windLines = opParam('windLines') ?? FACTORY_DEFAULTS.earth.windLines;
    cfg.earth!.windColors = opParam('windColors') ?? FACTORY_DEFAULTS.earth.windColors;
    cfg.earth!.oceanLines = opParam('oceanLines') ?? FACTORY_DEFAULTS.earth.oceanLines;
    cfg.earth!.oceanColors = opParam('oceanColors') ?? FACTORY_DEFAULTS.earth.oceanColors;
    cfg.earth!.tempColors = opParam('tempColors') ?? FACTORY_DEFAULTS.earth.tempColors;
    if (sp.has('localTemp')) cfg.earth!.localTemp = sp.get('localTemp') !== 'false';
    cfg.earth!.tempStrength = numParam('tempStrength') ?? FACTORY_DEFAULTS.earth.tempStrength;
    cfg.earth!.uvColors = opParam('uvColors') ?? FACTORY_DEFAULTS.earth.uvColors;
    if (sp.has('localUv')) cfg.earth!.localUv = sp.get('localUv') !== 'false';
    cfg.earth!.uvStrength = numParam('uvStrength') ?? FACTORY_DEFAULTS.earth.uvStrength;
    cfg.earth!.rhColors = opParam('rhColors') ?? FACTORY_DEFAULTS.earth.rhColors;
    cfg.earth!.precipColors = opParam('precipColors') ?? FACTORY_DEFAULTS.earth.precipColors;
    if (sp.has('localHydro')) cfg.earth!.localHydro = sp.get('localHydro') !== 'false';
    cfg.earth!.hydroStrength = numParam('hydroStrength') ?? FACTORY_DEFAULTS.earth.hydroStrength;
    cfg.earth!.chemOpacity = opParam('chemOpacity') ?? FACTORY_DEFAULTS.earth.chemOpacity;
    cfg.earth!.lifeOpacity = opParam('lifeOpacity') ?? FACTORY_DEFAULTS.earth.lifeOpacity;
    cfg.earth!.mineralOpacity = opParam('mineralOpacity') ?? FACTORY_DEFAULTS.earth.mineralOpacity;
    cfg.earth!.terrainOpacity = opParam('terrainOpacity') ?? FACTORY_DEFAULTS.earth.terrainOpacity;
    cfg.earth!.nightShading = opParam('nightShading') ?? FACTORY_DEFAULTS.earth.nightShading;
    if (sp.has('oceanMix')) cfg.earth!.oceanMix = sp.get('oceanMix') !== 'false';
    if (sp.has('coastlines')) cfg.earth!.coastlines = sp.get('coastlines') !== 'false';
    if (sp.has('bioLink')) cfg.earth!.bioLink = sp.get('bioLink') !== 'false';
    cfg.earth!.bioRate = numParam('bioRate') ?? FACTORY_DEFAULTS.earth.bioRate;
    if (sp.has('mineralLink')) cfg.earth!.mineralLink = sp.get('mineralLink') !== 'false';
    cfg.earth!.mineralRate = numParam('mineralRate') ?? FACTORY_DEFAULTS.earth.mineralRate;
    if (sp.has('tooltips')) cfg.earth!.tooltips = sp.get('tooltips') !== 'false';
    const mt = sp.get('mouseTool');
    if (mt === 'globe' || mt === 'attract' || mt === 'repel') {
      cfg.earth!.mouseTool = mt;
    }

    const parsedPan = parsePanParam(sp.get('pan'));
    if (parsedPan) cfg.view!.pan = parsedPan;
  }

  const urlProj = sp.get('projection');
  if (urlProj && urlProj in PROJECTIONS) {
    cfg.earth!.projection = urlProj as ProjectionId;
  } else if (hashParsed.projection && hashParsed.projection in PROJECTIONS) {
    cfg.earth!.projection = hashParsed.projection;
  }

  const urlLayer = sp.get('layer');
  if (urlLayer === 'both' || urlLayer === 'ocean' || urlLayer === 'air') {
    cfg.earth!.viewLayer = urlLayer;
  } else if (hashParsed.mode) {
    cfg.earth!.viewLayer = hashParsed.mode;
  }

  if (hashParsed.orientation) {
    const roll = PROJECTIONS[cfg.earth?.projection ?? 'orthographic'].defaultRotate[2];
    cfg.view!.rotate = [-hashParsed.orientation[0], -hashParsed.orientation[1], roll];
    cfg.view!.scale = hashParsed.orientation[2];
  }
  return cfg;
}

function fmtNumParam(val: number): string {
  return String(Number(val.toFixed(6)));
}

export function buildUrlPartsFromConfig(
  cfg: SimConfigV1,
  currentSearch: string = '',
): { search: string; hash: string } {
  const sp = new URLSearchParams();
  // Preserve non-config operational query flags such as noTimestamp or sample
  if (currentSearch) {
    const prev = new URLSearchParams(currentSearch);
    for (const [k, v] of prev.entries()) {
      if (!CONFIG_URL_KEYS.has(k)) {
        sp.set(k, v);
      }
    }
  }

  if (cfg.seed) sp.set('seed', cfg.seed);
  if (cfg.oceanSeed) sp.set('oceanSeed', cfg.oceanSeed);

  const setNum = (k: string, val: number | undefined, def: number) => {
    if (typeof val === 'number' && Number.isFinite(val) && Math.abs(val - def) > 1e-6) {
      sp.set(k, fmtNumParam(val));
    }
  };
  const setBool = (k: string, val: boolean | undefined, def: boolean) => {
    if (typeof val === 'boolean' && val !== def) {
      sp.set(k, val ? 'true' : 'false');
    }
  };

  if (cfg.engine && cfg.engine !== FACTORY_DEFAULTS.engine) sp.set('engine', cfg.engine);
  setNum('particleCount', cfg.particles, FACTORY_DEFAULTS.particles);
  setNum('colours', cfg.colours, FACTORY_DEFAULTS.colours);
  setNum('cellCount', cfg.cells, FACTORY_DEFAULTS.cells);

  const a = cfg.air;
  if (a) {
    setNum('radius', a.r, FACTORY_DEFAULTS.air.r);
    setNum('beta', a.beta, FACTORY_DEFAULTS.air.beta);
    setNum('force', a.force, FACTORY_DEFAULTS.air.force);
    setNum('delta', a.delta, FACTORY_DEFAULTS.air.delta);
    setNum('friction', a.friction, FACTORY_DEFAULTS.air.friction);
    setNum('torque', a.torque, FACTORY_DEFAULTS.air.torque);
    setNum('angularFriction', a.angularFriction, FACTORY_DEFAULTS.air.angularFriction);
    setNum('valency', a.valency, FACTORY_DEFAULTS.air.valency);
    setNum('temperature', a.temperature, FACTORY_DEFAULTS.air.temperature);
    setNum('activation', a.activation, FACTORY_DEFAULTS.air.activation);
    setNum('orbitalSharpness', a.orbitalSharpness, FACTORY_DEFAULTS.air.orbitalSharpness);
    setNum('bondStability', a.bondStability, FACTORY_DEFAULTS.air.bondStability);
    setNum('exchangeDrive', a.exchangeDrive, FACTORY_DEFAULTS.air.exchangeDrive);
    setNum('shearBreak', a.shearBreak, FACTORY_DEFAULTS.air.shearBreak);
    setNum('shearFloor', a.shearFloor, FACTORY_DEFAULTS.air.shearFloor);
    setBool('showOrbitals', a.showOrbitals, FACTORY_DEFAULTS.air.showOrbitals);
    setNum('avoidance', a.avoidance, FACTORY_DEFAULTS.air.avoidance);
    setNum('worldSize', a.worldSize, FACTORY_DEFAULTS.air.worldSize);
  }

  const o = cfg.ocean;
  if (o) {
    setNum('oceanColours', o.colours, FACTORY_DEFAULTS.ocean.colours);
    setNum('oceanR', o.r, FACTORY_DEFAULTS.ocean.r);
    setNum('oceanBeta', o.beta, FACTORY_DEFAULTS.ocean.beta);
    setNum('oceanForce', o.force, FACTORY_DEFAULTS.ocean.force);
    setNum('oceanDelta', o.delta, FACTORY_DEFAULTS.ocean.delta);
    setNum('oceanFriction', o.friction, FACTORY_DEFAULTS.ocean.friction);
    setNum('oceanAvoidance', o.avoidance, FACTORY_DEFAULTS.ocean.avoidance);
  }

  const e = cfg.earth;
  if (e) {
    setBool('windMix', e.vortex, FACTORY_DEFAULTS.earth.vortex);
    setNum('windStrength', e.windStrength, FACTORY_DEFAULTS.earth.windStrength);
    setNum('windLines', e.windLines, FACTORY_DEFAULTS.earth.windLines);
    setNum('windColors', e.windColors, FACTORY_DEFAULTS.earth.windColors);
    setNum('chemOpacity', e.chemOpacity, FACTORY_DEFAULTS.earth.chemOpacity);
    setBool('oceanMix', e.oceanMix, FACTORY_DEFAULTS.earth.oceanMix);
    setNum('oceanStrength', e.oceanStrength, FACTORY_DEFAULTS.earth.oceanStrength);
    setNum('oceanLines', e.oceanLines, FACTORY_DEFAULTS.earth.oceanLines);
    setNum('oceanColors', e.oceanColors, FACTORY_DEFAULTS.earth.oceanColors);
    setNum('tempColors', e.tempColors, FACTORY_DEFAULTS.earth.tempColors);
    setBool('localTemp', e.localTemp, FACTORY_DEFAULTS.earth.localTemp);
    setNum('tempStrength', e.tempStrength, FACTORY_DEFAULTS.earth.tempStrength);
    setNum('uvColors', e.uvColors, FACTORY_DEFAULTS.earth.uvColors);
    setBool('localUv', e.localUv, FACTORY_DEFAULTS.earth.localUv);
    setNum('uvStrength', e.uvStrength, FACTORY_DEFAULTS.earth.uvStrength);
    setNum('rhColors', e.rhColors, FACTORY_DEFAULTS.earth.rhColors);
    setNum('precipColors', e.precipColors, FACTORY_DEFAULTS.earth.precipColors);
    setBool('localHydro', e.localHydro, FACTORY_DEFAULTS.earth.localHydro);
    setNum('hydroStrength', e.hydroStrength, FACTORY_DEFAULTS.earth.hydroStrength);
    setNum('lifeOpacity', e.lifeOpacity, FACTORY_DEFAULTS.earth.lifeOpacity);
    setNum('mineralOpacity', e.mineralOpacity, FACTORY_DEFAULTS.earth.mineralOpacity);
    setNum('terrainOpacity', e.terrainOpacity, FACTORY_DEFAULTS.earth.terrainOpacity);
    setNum('nightShading', e.nightShading, FACTORY_DEFAULTS.earth.nightShading);
    setBool('coastlines', e.coastlines, FACTORY_DEFAULTS.earth.coastlines);
    setBool('bioLink', e.bioLink, FACTORY_DEFAULTS.earth.bioLink);
    setNum('bioRate', e.bioRate, FACTORY_DEFAULTS.earth.bioRate);
    setBool('mineralLink', e.mineralLink, FACTORY_DEFAULTS.earth.mineralLink);
    setNum('mineralRate', e.mineralRate, FACTORY_DEFAULTS.earth.mineralRate);
    setBool('tooltips', e.tooltips, FACTORY_DEFAULTS.earth.tooltips);
    if (e.mouseTool && e.mouseTool !== FACTORY_DEFAULTS.earth.mouseTool) {
      sp.set('mouseTool', e.mouseTool);
    }
  }

  const pan = cfg.view?.pan;
  if (pan && (Math.abs(pan[0]) > 1 || Math.abs(pan[1]) > 1)) {
    sp.set('pan', `${Math.round(pan[0])},${Math.round(pan[1])}`);
  }

  const proj = e?.projection ?? FACTORY_DEFAULTS.earth.projection;
  const viewLayer = e?.viewLayer ?? FACTORY_DEFAULTS.earth.viewLayer;
  const r = cfg.view?.rotate ?? PROJECTIONS[proj].defaultRotate;
  const lon = (-r[0]).toFixed(2);
  const lat = (-r[1]).toFixed(2);
  const sc = Math.round(cfg.view?.scale ?? 423);

  const hash =
    viewLayer === 'ocean'
      ? `#current/ocean/surface/currents/${proj}=${lon},${lat},${sc}`
      : viewLayer === 'air'
        ? `#current/air/surface/level/${proj}=${lon},${lat},${sc}`
        : `#current/both/surface/level/${proj}=${lon},${lat},${sc}`;

  const q = sp.toString();
  return { search: q ? `?${q}` : '', hash };
}

function sanitizeParsedObject(rawObj: Record<string, unknown>): SimConfigV1 {
  const airRaw = (rawObj.air && typeof rawObj.air === 'object' ? rawObj.air : {}) as Record<string, unknown>;
  const oceanRaw = (rawObj.ocean && typeof rawObj.ocean === 'object' ? rawObj.ocean : {}) as Record<string, unknown>;
  const earthRaw = (rawObj.earth && typeof rawObj.earth === 'object' ? rawObj.earth : {}) as Record<string, unknown>;
  const viewRaw = (rawObj.view && typeof rawObj.view === 'object' ? rawObj.view : {}) as Record<string, unknown>;

  const out: SimConfigV1 = { v: 1 };
  if (typeof rawObj.engine === 'string') out.engine = rawObj.engine;
  out.particles = finiteNum(rawObj.particles);
  out.colours = finiteNum(rawObj.colours);
  out.cells = finiteNum(rawObj.cells);
  if (typeof rawObj.seed === 'string') out.seed = rawObj.seed;
  if (typeof rawObj.oceanSeed === 'string') out.oceanSeed = rawObj.oceanSeed;

  out.air = {
    r: finiteNum(airRaw.r),
    beta: finiteNum(airRaw.beta),
    force: finiteNum(airRaw.force),
    delta: finiteNum(airRaw.delta),
    friction: finiteNum(airRaw.friction),
    torque: finiteNum(airRaw.torque),
    angularFriction: finiteNum(airRaw.angularFriction),
    valency: finiteNum(airRaw.valency),
    temperature: finiteNum(airRaw.temperature),
    activation: finiteNum(airRaw.activation),
    orbitalSharpness: finiteNum(airRaw.orbitalSharpness),
    showOrbitals: boolVal(airRaw.showOrbitals),
    avoidance: finiteNum(airRaw.avoidance),
    worldSize: finiteNum(airRaw.worldSize),
    bondStability: finiteNum(airRaw.bondStability),
    exchangeDrive: finiteNum(airRaw.exchangeDrive),
    shearBreak: finiteNum(airRaw.shearBreak),
    shearFloor: finiteNum(airRaw.shearFloor),
  };

  out.ocean = {
    particles: finiteNum(oceanRaw.particles),
    colours: finiteNum(oceanRaw.colours),
    r: finiteNum(oceanRaw.r),
    beta: finiteNum(oceanRaw.beta),
    force: finiteNum(oceanRaw.force),
    delta: finiteNum(oceanRaw.delta),
    friction: finiteNum(oceanRaw.friction),
    avoidance: finiteNum(oceanRaw.avoidance),
    cells: finiteNum(oceanRaw.cells),
  };

  const viewLayerRaw = earthRaw.viewLayer;
  const projRaw = earthRaw.projection;
  const toolRaw = earthRaw.mouseTool;

  out.earth = {
    viewLayer:
      viewLayerRaw === 'both' || viewLayerRaw === 'ocean' || viewLayerRaw === 'air'
        ? viewLayerRaw
        : undefined,
    projection:
      typeof projRaw === 'string' && projRaw in PROJECTIONS
        ? (projRaw as ProjectionId)
        : undefined,
    vortex: boolVal(earthRaw.vortex),
    windStrength: finiteNum(earthRaw.windStrength),
    windLines: unitOpacity(earthRaw.windLines),
    windColors: unitOpacity(earthRaw.windColors),
    chemOpacity: unitOpacity(earthRaw.chemOpacity),
    oceanMix: boolVal(earthRaw.oceanMix),
    oceanStrength: finiteNum(earthRaw.oceanStrength),
    oceanLines: unitOpacity(earthRaw.oceanLines),
    oceanColors: unitOpacity(earthRaw.oceanColors),
    tempColors: unitOpacity(earthRaw.tempColors),
    localTemp: boolVal(earthRaw.localTemp),
    tempStrength: finiteNum(earthRaw.tempStrength),
    uvColors: unitOpacity(earthRaw.uvColors),
    localUv: boolVal(earthRaw.localUv),
    uvStrength: finiteNum(earthRaw.uvStrength),
    rhColors: unitOpacity(earthRaw.rhColors),
    precipColors: unitOpacity(earthRaw.precipColors),
    localHydro: boolVal(earthRaw.localHydro),
    hydroStrength: finiteNum(earthRaw.hydroStrength),
    lifeOpacity: unitOpacity(earthRaw.lifeOpacity),
    mineralOpacity: unitOpacity(earthRaw.mineralOpacity),
    terrainOpacity: unitOpacity(earthRaw.terrainOpacity),
    nightShading: unitOpacity(earthRaw.nightShading),
    coastlines: boolVal(earthRaw.coastlines),
    bioLink: boolVal(earthRaw.bioLink),
    bioRate: finiteNum(earthRaw.bioRate),
    mineralLink: boolVal(earthRaw.mineralLink),
    mineralRate: finiteNum(earthRaw.mineralRate),
    tooltips: boolVal(earthRaw.tooltips),
    mouseTool:
      toolRaw === 'globe' || toolRaw === 'attract' || toolRaw === 'repel'
        ? toolRaw
        : undefined,
  };

  const rot = Array.isArray(viewRaw.rotate) && viewRaw.rotate.length === 3
    ? (viewRaw.rotate.every((x) => typeof x === 'number' && Number.isFinite(x))
        ? ([viewRaw.rotate[0], viewRaw.rotate[1], viewRaw.rotate[2]] as [number, number, number])
        : undefined)
    : undefined;
  const pan = Array.isArray(viewRaw.pan) && viewRaw.pan.length === 2
    ? (viewRaw.pan.every((x) => typeof x === 'number' && Number.isFinite(x))
        ? ([viewRaw.pan[0], viewRaw.pan[1]] as [number, number])
        : undefined)
    : undefined;

  out.view = {
    rotate: rot,
    pan,
    scale: finiteNum(viewRaw.scale),
  };

  return out;
}

export function parseConfigInput(rawInput: string | SimConfigV1): SimConfigV1 {
  if (typeof rawInput !== 'string') {
    if (!rawInput || typeof rawInput !== 'object') {
      throw new Error('Configuration must be a JSON object or URL string.');
    }
    return sanitizeParsedObject(rawInput as unknown as Record<string, unknown>);
  }

  const trimmed = rawInput.trim();
  if (!trimmed) {
    throw new Error('Configuration input is empty.');
  }

  const fromUrl = parseFromUrlString(trimmed);
  if (fromUrl) {
    return fromUrl;
  }

  let candidateJson = trimmed;
  // Support pasting JSON copied from markdown prompts (```json ... ``` or prose around {...})
  if (!candidateJson.startsWith('{')) {
    const firstBrace = candidateJson.indexOf('{');
    const lastBrace = candidateJson.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      candidateJson = candidateJson.slice(firstBrace, lastBrace + 1);
    }
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(candidateJson);
  } catch {
    throw new Error('Invalid JSON or URL configuration string.');
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Parsed configuration must be a JSON object.');
  }

  return sanitizeParsedObject(parsed as Record<string, unknown>);
}
