import { decodeGrib2Field } from './grib2';

export type WindLevelId =
  | 'surface'
  | '1000hPa'
  | '850hPa'
  | '700hPa'
  | '500hPa'
  | '250hPa'
  | '70hPa'
  | '10hPa';

export interface WindLevelSpec {
  id: WindLevelId;
  label: string;
  gribLevel: string;
  surfaceType: 'surface' | 'isobaric';
  /** Upper bound (m/s) of extendedSinebow wind speed color scale */
  scaleMax: number;
  /** Velocity normalization for streamline intensity & advection */
  streamMax: number;
}

export const WIND_LEVELS: Record<WindLevelId, WindLevelSpec> = {
  surface: {
    id: 'surface',
    label: 'Sfc (10m)',
    gribLevel: '10 m above ground',
    surfaceType: 'surface',
    scaleMax: 100,
    streamMax: 17,
  },
  '1000hPa': {
    id: '1000hPa',
    label: '1000 hPa',
    gribLevel: '1000 mb',
    surfaceType: 'isobaric',
    scaleMax: 100,
    streamMax: 18,
  },
  '850hPa': {
    id: '850hPa',
    label: '850 hPa',
    gribLevel: '850 mb',
    surfaceType: 'isobaric',
    scaleMax: 100,
    streamMax: 24,
  },
  '700hPa': {
    id: '700hPa',
    label: '700 hPa',
    gribLevel: '700 mb',
    surfaceType: 'isobaric',
    scaleMax: 100,
    streamMax: 30,
  },
  '500hPa': {
    id: '500hPa',
    label: '500 hPa',
    gribLevel: '500 mb',
    surfaceType: 'isobaric',
    scaleMax: 100,
    streamMax: 40,
  },
  '250hPa': {
    id: '250hPa',
    label: '250 hPa',
    gribLevel: '250 mb',
    surfaceType: 'isobaric',
    scaleMax: 100,
    streamMax: 60,
  },
  '70hPa': {
    id: '70hPa',
    label: '70 hPa',
    gribLevel: '70 mb',
    surfaceType: 'isobaric',
    scaleMax: 100,
    streamMax: 45,
  },
  '10hPa': {
    id: '10hPa',
    label: '10 hPa',
    gribLevel: '10 mb',
    surfaceType: 'isobaric',
    scaleMax: 100,
    streamMax: 65,
  },
};

export interface WindGrid {
  nx: number;
  ny: number;
  u: Float32Array;
  v: Float32Array;
  level: WindLevelId;
  refTime: string;
  validTime: string;
  source: string;
  isLive: boolean;
}

interface IdxEntry {
  offset: number;
  end: number | null;
  varName: string;
  level: string;
}

const GFS_S3_BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com';
const levelCache = new Map<string, WindGrid>();

function pad2(n: number): string {
  return String(n).padStart(2, '0');
}

function pad3(n: number): string {
  return String(n).padStart(3, '0');
}

function parseIdx(text: string): IdxEntry[] {
  const lines = text.trim().split('\n');
  const out: IdxEntry[] = [];
  for (let i = 0; i < lines.length; i++) {
    const parts = lines[i].split(':');
    if (parts.length < 5) continue;
    const offset = Number(parts[1]);
    const nextOffset = i + 1 < lines.length ? Number(lines[i + 1].split(':')[1]) : NaN;
    out.push({
      offset,
      end: Number.isFinite(nextOffset) ? nextOffset - 1 : null,
      varName: parts[3],
      level: parts[4],
    });
  }
  return out;
}

async function resolveLatestGfsCycle(nowMs: number): Promise<{
  gribUrl: string;
  idxEntries: IdxEntry[];
  cycleLabel: string;
  validLabel: string;
}> {
  // GFS 1.0 deg runs at 00/06/12/18 UTC and usually lands on S3 ~3.5-4h after cycle init.
  const latestReadyMs = nowMs - 3.5 * 3600 * 1000;
  const candidateDate = new Date(latestReadyMs);
  const startCycHour = Math.floor(candidateDate.getUTCHours() / 6) * 6;
  const baseUtcMs = Date.UTC(
    candidateDate.getUTCFullYear(),
    candidateDate.getUTCMonth(),
    candidateDate.getUTCDate(),
    startCycHour,
  );

  for (let step = 0; step < 4; step++) {
    const cycMs = baseUtcMs - step * 6 * 3600 * 1000;
    const cycDate = new Date(cycMs);
    const ymd = `${cycDate.getUTCFullYear()}${pad2(cycDate.getUTCMonth() + 1)}${pad2(cycDate.getUTCDate())}`;
    const hh = pad2(cycDate.getUTCHours());
    const diffHours = Math.max(0, (nowMs - cycMs) / (3600 * 1000));
    const fhr = Math.min(24, Math.max(0, Math.round(diffHours / 3) * 3));
    const tryFhrs = fhr > 0 ? [fhr, Math.max(0, fhr - 3), 0] : [0];

    for (const fh of tryFhrs) {
      const fStr = `f${pad3(fh)}`;
      const gribUrl = `${GFS_S3_BASE}/gfs.${ymd}/${hh}/atmos/gfs.t${hh}z.pgrb2.1p00.${fStr}`;
      const idxUrl = `${gribUrl}.idx`;
      try {
        const res = await fetch(idxUrl);
        if (!res.ok) continue;
        const text = await res.text();
        const idxEntries = parseIdx(text);
        if (idxEntries.length === 0) continue;
        const validDate = new Date(cycMs + fh * 3600 * 1000);
        const validLabel = `${validDate.getUTCFullYear()}-${pad2(validDate.getUTCMonth() + 1)}-${pad2(validDate.getUTCDate())} ${pad2(validDate.getUTCHours())}:00Z`;
        return {
          gribUrl,
          idxEntries,
          cycleLabel: `${ymd} ${hh}Z+${pad2(fh)}h`,
          validLabel,
        };
      } catch {
        // Try earlier cycle/forecast
      }
    }
  }
  throw new Error('Unable to resolve live NOAA GFS index from S3');
}

let cachedCyclePromise: ReturnType<typeof resolveLatestGfsCycle> | null = null;
let cachedCycleFetchedAt = 0;

function getLatestCycleInfo() {
  const now = Date.now();
  if (!cachedCyclePromise || now - cachedCycleFetchedAt > 15 * 60 * 1000) {
    cachedCycleFetchedAt = now;
    cachedCyclePromise = resolveLatestGfsCycle(now).catch((err) => {
      cachedCyclePromise = null;
      throw err;
    });
  }
  return cachedCyclePromise;
}

export async function loadFallbackWind(): Promise<WindGrid> {
  const cached = levelCache.get('fallback');
  if (cached) return cached;

  const res = await fetch('./data/wind-surface-fallback.json');
  if (!res.ok) {
    throw new Error(`Fallback wind HTTP ${res.status}`);
  }
  const raw = (await res.json()) as {
    nx: number;
    ny: number;
    scale: number;
    refTime: string;
    u: number[];
    v: number[];
  };
  const n = raw.nx * raw.ny;
  const u = new Float32Array(n);
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    u[i] = raw.u[i] * raw.scale;
    v[i] = raw.v[i] * raw.scale;
  }
  const grid: WindGrid = {
    nx: raw.nx,
    ny: raw.ny,
    u,
    v,
    level: 'surface',
    refTime: raw.refTime,
    validTime: raw.refTime,
    source: 'GFS / NCEP / NWS (cached)',
    isLive: false,
  };
  levelCache.set('fallback', grid);
  return grid;
}

export async function fetchLiveGfsWind(levelId: WindLevelId): Promise<WindGrid> {
  const spec = WIND_LEVELS[levelId];
  const cycle = await getLatestCycleInfo();
  const cacheKey = `${cycle.cycleLabel}:${levelId}`;
  const cached = levelCache.get(cacheKey);
  if (cached) return cached;

  const uEntry = cycle.idxEntries.find(
    (e) => e.varName === 'UGRD' && e.level === spec.gribLevel,
  );
  const vEntry = cycle.idxEntries.find(
    (e) => e.varName === 'VGRD' && e.level === spec.gribLevel,
  );
  if (!uEntry || !vEntry) {
    throw new Error(`GFS level "${spec.gribLevel}" not found in index`);
  }

  const fetchSlice = async (entry: IdxEntry) => {
    const range =
      entry.end !== null ? `bytes=${entry.offset}-${entry.end}` : `bytes=${entry.offset}-`;
    const res = await fetch(cycle.gribUrl, { headers: { Range: range } });
    if (!res.ok) {
      throw new Error(`GFS byte-range HTTP ${res.status}`);
    }
    return decodeGrib2Field(await res.arrayBuffer());
  };

  const [uGrid, vGrid] = await Promise.all([fetchSlice(uEntry), fetchSlice(vEntry)]);
  const grid: WindGrid = {
    nx: uGrid.nx,
    ny: uGrid.ny,
    u: uGrid.values,
    v: vGrid.values,
    level: levelId,
    refTime: cycle.cycleLabel,
    validTime: cycle.validLabel,
    source: 'GFS / NCEP / NWS (live)',
    isLive: true,
  };
  levelCache.set(cacheKey, grid);
  return grid;
}

/**
 * Samples bilinearly interpolated `[u, v, speed]` (m/s) at geographic `[lonDeg, latDeg]`.
 * Grid convention: row `0` = `+90N`, row `ny-1` = `-90S`; col `0` = `0E` wrapping cleanly over `360`.
 */
export function sampleWindLonLat(
  grid: WindGrid,
  lonDeg: number,
  latDeg: number,
): [number, number, number] {
  const { nx, ny, u, v } = grid;
  const wrappedLon = ((lonDeg % 360) + 360) % 360;
  const clampedLat = Math.max(-90, Math.min(90, latDeg));

  const gx = (wrappedLon / 360) * nx;
  const gy = ((90 - clampedLat) / 180) * (ny - 1);

  const i0 = Math.floor(gx) % nx;
  const i1 = (i0 + 1) % nx;
  const fx = gx - Math.floor(gx);

  const j0 = Math.min(ny - 1, Math.max(0, Math.floor(gy)));
  const j1 = Math.min(ny - 1, j0 + 1);
  const fy = gy - j0;

  const k00 = j0 * nx + i0;
  const k10 = j0 * nx + i1;
  const k01 = j1 * nx + i0;
  const k11 = j1 * nx + i1;

  const w00 = (1 - fx) * (1 - fy);
  const w10 = fx * (1 - fy);
  const w01 = (1 - fx) * fy;
  const w11 = fx * fy;

  const uu = u[k00] * w00 + u[k10] * w10 + u[k01] * w01 + u[k11] * w11;
  const vv = v[k00] * w00 + v[k10] * w10 + v[k01] * w01 + v[k11] * w11;
  return [uu, vv, Math.hypot(uu, vv)];
}

/**
 * Exact `cambecc/earth` extendedSinebowColor gradient for normalized wind speed `[0..1]`.
 */
export function extendedSinebowColor(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const BOUNDARY = 0.45;
  const sinebow = (hue: number): [number, number, number] => {
    const rad = hue * Math.PI * 2 * (5 / 6) * 0.75;
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    return [
      Math.max(0, -c) * 255,
      Math.max(s, 0) * 255,
      Math.max(c, 0, -s) * 255,
    ];
  };
  if (v <= BOUNDARY) {
    return sinebow(v / BOUNDARY);
  }
  const c1 = sinebow(1.0);
  const f = (v - BOUNDARY) / (1 - BOUNDARY);
  return [
    c1[0] + (255 - c1[0]) * f,
    c1[1] + (255 - c1[1]) * f,
    c1[2] + (255 - c1[2]) * f,
  ];
}

/**
 * Computes the divergence-free rotational component `(uDf, vDf)` via spherical Helmholtz
 * projection on a coarse synoptic grid (keeps cyclones, Rossby jets, and trade winds intact
 * while preventing 2D surface atoms from jamming into compressible convergence sinks).
 */
export function buildDivergenceFreeWind(
  grid: WindGrid,
  outW: number = 360,
  outH: number = 181,
): Float32Array<ArrayBuffer> {
  const cw = 90;
  const ch = 45;
  const dLam = (2 * Math.PI) / cw;
  const dPhi = Math.PI / (ch - 1);

  const uC = new Float32Array(cw * ch);
  const vC = new Float32Array(cw * ch);
  for (let j = 0; j < ch; j++) {
    const lat = 90 - (j / (ch - 1)) * 180;
    for (let i = 0; i < cw; i++) {
      const lon = (i / cw) * 360;
      const [uu, vv] = sampleWindLonLat(grid, lon, lat);
      uC[j * cw + i] = uu;
      vC[j * cw + i] = vv;
    }
  }

  // Compute metric spherical divergence div = (1/cosPhi) * (du/dLam + d(v*cosPhi)/dPhi)
  const div = new Float32Array(cw * ch);
  const cosPhi = new Float32Array(ch);
  for (let j = 0; j < ch; j++) {
    const phi = ((90 - (j / (ch - 1)) * 180) * Math.PI) / 180;
    cosPhi[j] = Math.max(0.15, Math.cos(phi));
  }
  for (let j = 1; j < ch - 1; j++) {
    const c = cosPhi[j];
    const cp = cosPhi[j - 1]; // j-1 is further North (higher lat -> dPhi > 0 upward)
    const cm = cosPhi[j + 1];
    for (let i = 0; i < cw; i++) {
      const ip = (i + 1) % cw;
      const im = (i - 1 + cw) % cw;
      const duDl = (uC[j * cw + ip] - uC[j * cw + im]) / (2 * dLam);
      // Note j decreases northward!
      const dvCdP = (vC[(j - 1) * cw + i] * cp - vC[(j + 1) * cw + i] * cm) / (2 * dPhi);
      div[j * cw + i] = (duDl + dvCdP) / c;
    }
  }

  // SOR solver for spherical Laplacian(pot) = div
  const pot = new Float32Array(cw * ch);
  const invDP2 = 1 / (dPhi * dPhi);
  const omega = 1.7;
  for (let iter = 0; iter < 140; iter++) {
    for (let j = 1; j < ch - 1; j++) {
      const c = cosPhi[j];
      const invDL2 = 1 / (c * c * dLam * dLam);
      const cpMid = 0.5 * (c + cosPhi[j - 1]) / c;
      const cmMid = 0.5 * (c + cosPhi[j + 1]) / c;
      const denom = 2 * invDL2 + (cpMid + cmMid) * invDP2;
      for (let i = 0; i < cw; i++) {
        const ip = (i + 1) % cw;
        const im = (i - 1 + cw) % cw;
        const rhs =
          (pot[j * cw + ip] + pot[j * cw + im]) * invDL2 +
          (pot[(j - 1) * cw + i] * cpMid + pot[(j + 1) * cw + i] * cmMid) * invDP2 -
          div[j * cw + i];
        const target = rhs / denom;
        pot[j * cw + i] += omega * (target - pot[j * cw + i]);
      }
    }
  }

  // Subtract irrotational gradient grad(pot) from coarse wind
  const uRotC = new Float32Array(cw * ch);
  const vRotC = new Float32Array(cw * ch);
  for (let j = 1; j < ch - 1; j++) {
    const c = cosPhi[j];
    for (let i = 0; i < cw; i++) {
      const ip = (i + 1) % cw;
      const im = (i - 1 + cw) % cw;
      const dpDl = (pot[j * cw + ip] - pot[j * cw + im]) / (2 * dLam * c);
      const dpDp = (pot[(j - 1) * cw + i] - pot[(j + 1) * cw + i]) / (2 * dPhi);
      uRotC[j * cw + i] = uC[j * cw + i] - dpDl;
      vRotC[j * cw + i] = vC[j * cw + i] - dpDp;
    }
  }

  // Pack RGBA32Float: (u_divFree, v_divFree, rawSpeedMag, 1)
  const out = new Float32Array(outW * outH * 4);
  for (let row = 0; row < outH; row++) {
    const latDeg = 90 - (row / (outH - 1)) * 180;
    const gy = ((90 - latDeg) / 180) * (ch - 1);
    const j0 = Math.min(ch - 1, Math.max(0, Math.floor(gy)));
    const j1 = Math.min(ch - 1, j0 + 1);
    const fy = gy - j0;

    for (let col = 0; col < outW; col++) {
      const lonDeg = (col / outW) * 360;
      const gx = (lonDeg / 360) * cw;
      const i0 = Math.floor(gx) % cw;
      const i1 = (i0 + 1) % cw;
      const fx = gx - Math.floor(gx);

      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      const uDf =
        uRotC[j0 * cw + i0] * w00 +
        uRotC[j0 * cw + i1] * w10 +
        uRotC[j1 * cw + i0] * w01 +
        uRotC[j1 * cw + i1] * w11;
      const vDf =
        vRotC[j0 * cw + i0] * w00 +
        vRotC[j0 * cw + i1] * w10 +
        vRotC[j1 * cw + i0] * w01 +
        vRotC[j1 * cw + i1] * w11;

      const [, , rawMag] = sampleWindLonLat(grid, lonDeg, latDeg);
      const o = (row * outW + col) * 4;
      out[o + 0] = uDf;
      out[o + 1] = vDf;
      out[o + 2] = rawMag;
      out[o + 3] = 1;
    }
  }
  return out;
}
