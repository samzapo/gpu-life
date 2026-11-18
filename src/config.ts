import {
  parseNullschoolHash,
  PROJECTIONS,
  type ProjectionId,
} from './earth/projections';
import { WIND_LEVELS, type WindLevelId } from './earth/windData';

export const CONFIG_STORAGE_KEY = 'complexLife:config:v1';

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
    windLevel?: WindLevelId;
    vortex?: boolean;
    windStrength?: number;
    windLines?: number;
    windColors?: number;
    chemOpacity?: number;
    oceanMix?: boolean;
    oceanStrength?: number;
    oceanLines?: number;
    oceanColors?: number;
    lifeOpacity?: number;
    coastlines?: boolean;
    bioLink?: boolean;
    bioRate?: number;
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
  const hashParsed = parseNullschoolHash(hashStr);
  const cfg: SimConfigV1 = { v: 1, earth: {}, ocean: {}, air: {}, view: {} };

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

  if (sp.has('engine')) cfg.engine = sp.get('engine') ?? undefined;
  if (sp.has('seed')) cfg.seed = sp.get('seed') ?? undefined;
  cfg.particles = numParam('particleCount');
  cfg.colours = numParam('colours');
  cfg.cells = numParam('cellCount');

  cfg.air!.r = numParam('radius');
  cfg.air!.worldSize = numParam('worldSize');
  cfg.air!.torque = numParam('torque');
  cfg.air!.angularFriction = numParam('angularFriction');
  cfg.air!.valency = numParam('valency');
  cfg.air!.temperature = numParam('temperature');
  cfg.air!.activation = numParam('activation');
  cfg.air!.orbitalSharpness = numParam('orbitalSharpness');

  cfg.ocean!.particles = numParam('oceanParticles');
  cfg.ocean!.colours = numParam('oceanColours');
  cfg.ocean!.r = numParam('oceanR');

  const urlProj = sp.get('projection');
  if (urlProj && urlProj in PROJECTIONS) {
    cfg.earth!.projection = urlProj as ProjectionId;
  } else if (hashParsed.projection && hashParsed.projection in PROJECTIONS) {
    cfg.earth!.projection = hashParsed.projection;
  }

  const urlLevel = sp.get('windLevel');
  if (urlLevel && urlLevel in WIND_LEVELS) {
    cfg.earth!.windLevel = urlLevel as WindLevelId;
  } else if (hashParsed.level && hashParsed.level in WIND_LEVELS) {
    cfg.earth!.windLevel = hashParsed.level as WindLevelId;
  }

  const urlLayer = sp.get('layer');
  if (urlLayer === 'both' || urlLayer === 'ocean' || urlLayer === 'air') {
    cfg.earth!.viewLayer = urlLayer;
  } else if (hashParsed.mode) {
    cfg.earth!.viewLayer = hashParsed.mode;
  }

  cfg.earth!.windStrength = numParam('windStrength');
  cfg.earth!.oceanStrength = numParam('oceanStrength');
  cfg.earth!.windLines = opParam('windLines');
  cfg.earth!.windColors = opParam('windColors');
  cfg.earth!.oceanLines = opParam('oceanLines');
  cfg.earth!.oceanColors = opParam('oceanColors');
  cfg.earth!.chemOpacity = opParam('chemOpacity');
  cfg.earth!.lifeOpacity = opParam('lifeOpacity');
  if (sp.has('oceanMix')) cfg.earth!.oceanMix = sp.get('oceanMix') !== 'false';
  if (sp.has('coastlines')) cfg.earth!.coastlines = sp.get('coastlines') !== 'false';
  if (sp.has('bioLink')) cfg.earth!.bioLink = sp.get('bioLink') !== 'false';
  cfg.earth!.bioRate = numParam('bioRate');
  if (sp.has('tooltips')) cfg.earth!.tooltips = sp.get('tooltips') !== 'false';

  if (hashParsed.orientation) {
    cfg.view!.rotate = [-hashParsed.orientation[0], -hashParsed.orientation[1], 0];
    cfg.view!.scale = hashParsed.orientation[2];
  }
  return cfg;
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
  const windRaw = earthRaw.windLevel;
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
    windLevel:
      typeof windRaw === 'string' && windRaw in WIND_LEVELS
        ? (windRaw as WindLevelId)
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
    lifeOpacity: unitOpacity(earthRaw.lifeOpacity),
    coastlines: boolVal(earthRaw.coastlines),
    bioLink: boolVal(earthRaw.bioLink),
    bioRate: finiteNum(earthRaw.bioRate),
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
