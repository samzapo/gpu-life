import { decodeBase64Bytes } from './oceanData';
import {
  fetchGfsFieldSlice,
  getLatestGfsCycleInfo,
  interpolateColorStops,
} from './windData';

export const UV_INDEX_SPEC = {
  minUvi: 0,
  maxUvi: 18,
} as const;

export interface UvGrid {
  nx: number;
  ny: number;
  cpcNoonUvi: Float32Array;
  calibFactor: Float32Array;
  refOzoneDU: Float32Array;
  refCloudFrac: Float32Array;
  ozoneDU: Float32Array;
  cloudFrac: Float32Array;
  elevKm: Float32Array;
  cpcRefTime: string;
  cpcRefMs: number;
  gfsRefTime: string;
  gfsValidTime: string;
  source: string;
  isLive: boolean;
}

export interface LocalUvSample {
  uviNow: number;
  uviNoon: number;
  category: string;
  ozoneDU: number;
  cloudPct: number;
}

const UV_COLOR_STOPS: readonly [number, [number, number, number]][] = [
  [0.0, [0, 0, 15]],
  [0.2, [32, 32, 32]],
  [0.5, [62, 167, 45]],
  [1.5, [105, 194, 32]],
  [2.5, [174, 224, 17]],
  [3.5, [255, 238, 0]],
  [4.9, [248, 186, 0]],
  [6.25, [241, 139, 0]],
  [7.4, [235, 91, 8]],
  [8.5, [229, 50, 16]],
  [9.75, [207, 58, 104]],
  [11.0, [181, 103, 164]],
  [14.5, [209, 148, 234]],
  [18.0, [231, 225, 255]],
];

/**
 * Exact `earth.nullschool.net` UVI gradient (`[0 .. 18]` UV Index, `0 .. 0.45 W/m²` erythemal irradiance).
 */
export function uvIndexColorRgb(uvi: number): [number, number, number] {
  return interpolateColorStops(uvi, UV_COLOR_STOPS);
}

const UV_CATS = [
  'low',
  'low',
  'low',
  'moderate',
  'moderate',
  'moderate',
  'high',
  'high',
  'very high',
  'very high',
  'very high',
  'extreme',
] as const;

export function uvCategoryLabel(uvi: number): string {
  const idx = Math.max(0, Math.min(11, Math.round(uvi)));
  return UV_CATS[idx];
}

export function solarAstronomyAtUtc(utcMs: number): {
  declRad: number;
  sinDecl: number;
  cosDecl: number;
  eqTimeMin: number;
  utcHours: number;
} {
  const d = new Date(utcMs);
  const yearStart = Date.UTC(d.getUTCFullYear(), 0, 1);
  const doy = Math.floor((utcMs - yearStart) / 86_400_000) + 1;
  const utcHours =
    d.getUTCHours() + d.getUTCMinutes() / 60 + d.getUTCSeconds() / 3600;
  const gamma = ((2 * Math.PI) / 365) * (doy - 1 + (utcHours - 12) / 24);
  const declRad =
    0.006918 -
    0.399912 * Math.cos(gamma) +
    0.070257 * Math.sin(gamma) -
    0.006758 * Math.cos(2 * gamma) +
    0.000907 * Math.sin(2 * gamma) -
    0.002697 * Math.cos(3 * gamma) +
    0.00148 * Math.sin(3 * gamma);
  const eqTimeMin =
    229.18 *
    (0.000075 +
      0.001868 * Math.cos(gamma) -
      0.032077 * Math.sin(gamma) -
      0.014615 * Math.cos(2 * gamma) -
      0.040849 * Math.sin(2 * gamma));
  return {
    declRad,
    sinDecl: Math.sin(declRad),
    cosDecl: Math.cos(declRad),
    eqTimeMin,
    utcHours,
  };
}

export function clearSkyUviGfs(
  cosZ: number,
  ozoneDU: number,
  elevKm: number,
): number {
  if (cosZ <= 0.0005) return 0;
  const du = Math.max(80, ozoneDU);
  const altBoost = 1 + 0.06 * Math.max(0, elevKm);
  return 12.5 * Math.pow(cosZ, 2.42) * Math.pow(300 / du, 1.23) * altBoost;
}

export function cloudTransmission(cloudFrac: number): number {
  const c = Math.max(0, Math.min(1, cloudFrac));
  return 1 - 0.52 * Math.pow(c, 1.55);
}

let baselineUvPromise: Promise<UvGrid> | null = null;
const gfsUvCache = new Map<string, UvGrid>();

export async function loadCpcUvBaseline(forceRefresh = false): Promise<UvGrid> {
  if (forceRefresh) {
    baselineUvPromise = null;
  }
  if (baselineUvPromise) return baselineUvPromise;
  baselineUvPromise = (async () => {
    const res = await fetch('./data/uv-index.json', forceRefresh ? { cache: 'no-store' } : undefined);
    if (!res.ok) {
      throw new Error(`UV index snapshot HTTP ${res.status}`);
    }
    const raw = (await res.json()) as {
      nx: number;
      ny: number;
      cpcNoonScale: number;
      tozneScale: number;
      cpcSource: string;
      cpcRefTime: string;
      gfsRefTime: string;
      cpcNoonB64: string;
      tozneB64: string;
      tcdcB64: string;
      hgtB64: string;
    };
    const nx = raw.nx;
    const ny = raw.ny;
    const n = nx * ny;

    const cpcNoonUvi = new Float32Array(n);
    const ozoneDU = new Float32Array(n);
    const cloudFrac = new Float32Array(n);
    const elevKm = new Float32Array(n);
    const calibFactor = new Float32Array(n);

    const cpcBytes = decodeBase64Bytes(raw.cpcNoonB64);
    const cpcView = new DataView(
      cpcBytes.buffer,
      cpcBytes.byteOffset,
      cpcBytes.byteLength,
    );
    const ozBytes = decodeBase64Bytes(raw.tozneB64);
    const ozView = new DataView(
      ozBytes.buffer,
      ozBytes.byteOffset,
      ozBytes.byteLength,
    );
    const cldBytes = decodeBase64Bytes(raw.tcdcB64);
    const hgtBytes = decodeBase64Bytes(raw.hgtB64);
    const hgtView = new DataView(
      hgtBytes.buffer,
      hgtBytes.byteOffset,
      hgtBytes.byteLength,
    );

    const cScale = raw.cpcNoonScale ?? 0.01;
    const ozScale = raw.tozneScale ?? 0.1;
    const cpcRefMs = Date.parse(raw.cpcRefTime) || Date.now();
    const refAstro = solarAstronomyAtUtc(cpcRefMs);
    const d2r = Math.PI / 180;

    for (let j = 0; j < ny; j++) {
      const latDeg = 90 - (j / Math.max(1, ny - 1)) * 180;
      const latRad = latDeg * d2r;
      const cosZNoonRef =
        Math.sin(latRad) * refAstro.sinDecl +
        Math.cos(latRad) * refAstro.cosDecl;
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const cUvi = cpcView.getUint16(idx * 2, true) * cScale;
        const du = ozView.getUint16(idx * 2, true) * ozScale || 300;
        const cld = Math.max(0, Math.min(1, cldBytes[idx] / 100));
        const km = Math.max(0, hgtView.getInt16(idx * 2, true) / 1000);
        cpcNoonUvi[idx] = cUvi;
        ozoneDU[idx] = du;
        cloudFrac[idx] = cld;
        elevKm[idx] = km;

        const modelRef =
          clearSkyUviGfs(cosZNoonRef, du, km) * cloudTransmission(cld);
        if (cosZNoonRef > 0.08 && cUvi > 0.25 && modelRef > 0.2) {
          calibFactor[idx] = Math.max(
            0.7,
            Math.min(1.68, cUvi / modelRef),
          );
        } else {
          calibFactor[idx] = Math.abs(latDeg) > 65 ? 1.28 : 1.0;
        }
      }
    }

    return {
      nx,
      ny,
      cpcNoonUvi,
      calibFactor,
      refOzoneDU: ozoneDU,
      refCloudFrac: cloudFrac,
      ozoneDU,
      cloudFrac,
      elevKm,
      cpcRefTime: raw.cpcRefTime,
      cpcRefMs,
      gfsRefTime: raw.gfsRefTime,
      gfsValidTime: raw.gfsRefTime,
      source: `${raw.cpcSource} (cached)`,
      isLive: false,
    };
  })().catch((err) => {
    baselineUvPromise = null;
    throw err;
  });

  return baselineUvPromise;
}

export async function fetchLiveGfsUvFields(
  baseGrid: UvGrid,
  opts?: { forceRefresh?: boolean },
): Promise<UvGrid> {
  const cycle = await getLatestGfsCycleInfo(Boolean(opts?.forceRefresh));
  const cacheKey = `${cycle.cycleLabel}:${baseGrid.cpcRefTime}`;
  const cached = gfsUvCache.get(cacheKey);
  if (cached) return cached;

  // Prune older cycles / baselines so open tabs stay bounded in memory
  for (const k of gfsUvCache.keys()) {
    if (!k.startsWith(`${cycle.cycleLabel}:`)) {
      gfsUvCache.delete(k);
    }
  }

  const ozEntry = cycle.idxEntries.find(
    (e) =>
      e.varName === 'TOZNE' && e.level.startsWith('entire atmosphere'),
  );
  const cldEntry = cycle.idxEntries.find(
    (e) => e.varName === 'TCDC' && e.level === 'entire atmosphere',
  );
  if (!ozEntry || !cldEntry) {
    throw new Error('GFS TOZNE / TCDC fields not found in index');
  }

  const [ozGrid, cldGrid] = await Promise.all([
    fetchGfsFieldSlice(cycle, ozEntry),
    fetchGfsFieldSlice(cycle, cldEntry),
  ]);

  const n = baseGrid.nx * baseGrid.ny;
  const ozoneDU =
    ozGrid.values.length === n ? ozGrid.values : baseGrid.ozoneDU;
  let cloudFrac = baseGrid.cloudFrac;
  if (cldGrid.values.length === n) {
    cloudFrac = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const pct = cldGrid.values[i];
      cloudFrac[i] = Number.isFinite(pct)
        ? Math.max(0, Math.min(1, pct / 100))
        : baseGrid.cloudFrac[i];
    }
  }

  const live: UvGrid = {
    ...baseGrid,
    ozoneDU,
    cloudFrac,
    gfsRefTime: cycle.cycleLabel,
    gfsValidTime: cycle.validLabel,
    source: 'NOAA CPC + Live GFS O₃/Clouds',
    isLive: true,
  };
  gfsUvCache.set(cacheKey, live);
  return live;
}

function evalCellNoonAndNow(
  grid: UvGrid,
  idx: number,
  latRad: number,
  lonDeg: number,
  astro: ReturnType<typeof solarAstronomyAtUtc>,
  cpcBlend: number,
): [number, number] {
  const du = grid.ozoneDU[idx];
  const cld = grid.cloudFrac[idx];
  const km = grid.elevKm[idx];
  const cal = grid.calibFactor[idx];

  const cosZNoon = Math.sin(latRad) * astro.sinDecl + Math.cos(latRad) * astro.cosDecl;
  const tstMin =
    (((astro.utcHours * 60 + astro.eqTimeMin + 4 * lonDeg) % 1440) + 1440) %
    1440;
  const haRad = ((tstMin / 4 - 180) * Math.PI) / 180;
  const cosZNow =
    Math.sin(latRad) * astro.sinDecl +
    Math.cos(latRad) * astro.cosDecl * Math.cos(haRad);

  const gfsNoon = clearSkyUviGfs(cosZNoon, du, km) * cloudTransmission(cld) * cal;
  let noonUvi = gfsNoon;
  const cpcBase = grid.cpcNoonUvi[idx];
  if (cpcBlend > 0.01 && cpcBase > 0.01) {
    const refDu = Math.max(80, grid.refOzoneDU[idx]);
    const ozRatio = Math.pow(refDu / Math.max(80, du), 1.23);
    const cldRatio =
      cloudTransmission(cld) / Math.max(0.35, cloudTransmission(grid.refCloudFrac[idx]));
    const cpcAdj = Math.max(0, Math.min(25, cpcBase * ozRatio * cldRatio));
    noonUvi = gfsNoon * (1 - cpcBlend) + cpcAdj * cpcBlend;
  }

  let uviNow = 0;
  if (cosZNow > 0) {
    const ratio = Math.pow(cosZNow / Math.max(cosZNoon, 0.04), 2.15);
    uviNow = noonUvi * ratio;
  } else if (cosZNow > -0.055) {
    const twi = (cosZNow + 0.055) / 0.055;
    uviNow = Math.min(0.24, noonUvi * 0.025) * twi * twi;
  }
  return [uviNow, noonUvi];
}

export function sampleUvLonLat(
  grid: UvGrid,
  lonDeg: number,
  latDeg: number,
  utcMs: number = Date.now(),
): LocalUvSample {
  const { nx, ny, ozoneDU, cloudFrac } = grid;
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

  const w00 = (1 - fx) * (1 - fy);
  const w10 = fx * (1 - fy);
  const w01 = (1 - fx) * fy;
  const w11 = fx * fy;

  const k00 = j0 * nx + i0;
  const k10 = j0 * nx + i1;
  const k01 = j1 * nx + i0;
  const k11 = j1 * nx + i1;

  const astro = solarAstronomyAtUtc(utcMs);
  const ageDays = Math.abs(utcMs - grid.cpcRefMs) / 86_400_000;
  const cpcBlend = Math.max(0, Math.min(1, 1 - Math.max(0, ageDays - 2) / 12));

  const lat0Rad = ((90 - (j0 / Math.max(1, ny - 1)) * 180) * Math.PI) / 180;
  const lat1Rad = ((90 - (j1 / Math.max(1, ny - 1)) * 180) * Math.PI) / 180;
  const lon0 = (i0 / nx) * 360;
  const lon1 = (i1 / nx) * 360;

  const [n00, m00] = evalCellNoonAndNow(grid, k00, lat0Rad, lon0, astro, cpcBlend);
  const [n10, m10] = evalCellNoonAndNow(grid, k10, lat0Rad, lon1, astro, cpcBlend);
  const [n01, m01] = evalCellNoonAndNow(grid, k01, lat1Rad, lon0, astro, cpcBlend);
  const [n11, m11] = evalCellNoonAndNow(grid, k11, lat1Rad, lon1, astro, cpcBlend);

  const uviNow = n00 * w00 + n10 * w10 + n01 * w01 + n11 * w11;
  const uviNoon = m00 * w00 + m10 * w10 + m01 * w01 + m11 * w11;
  const du =
    ozoneDU[k00] * w00 + ozoneDU[k10] * w10 + ozoneDU[k01] * w01 + ozoneDU[k11] * w11;
  const cld =
    cloudFrac[k00] * w00 + cloudFrac[k10] * w10 + cloudFrac[k01] * w01 + cloudFrac[k11] * w11;

  return {
    uviNow,
    uviNoon,
    category: uvCategoryLabel(uviNow),
    ozoneDU: du,
    cloudPct: Math.round(cld * 100),
  };
}

/**
 * Packs `(uviNow, uviNoon, ozoneDU, cloudFrac)` into a `(360 x 181)` `rgba32float` texture
 * using live UTC solar zenith + NOAA CPC noontime baseline + GFS TOZNE/TCDC/HGT.
 */
export function buildGlobalUvTexture(
  grid: UvGrid | null,
  utcMs: number = Date.now(),
  outW: number = 360,
  outH: number = 181,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(outW * outH * 4);
  const astro = solarAstronomyAtUtc(utcMs);
  const d2r = Math.PI / 180;
  const ageDays = grid ? Math.abs(utcMs - grid.cpcRefMs) / 86_400_000 : 999;
  const cpcBlend = grid ? Math.max(0, Math.min(1, 1 - Math.max(0, ageDays - 2) / 12)) : 0;

  // Precompute cos(hourAngle) per longitude column (fast 360-element LUT)
  const cosHa = new Float32Array(outW);
  for (let col = 0; col < outW; col++) {
    const lonDeg = (col / outW) * 360;
    const tstMin =
      (((astro.utcHours * 60 + astro.eqTimeMin + 4 * lonDeg) % 1440) + 1440) % 1440;
    const haRad = (tstMin / 4 - 180) * d2r;
    cosHa[col] = Math.cos(haRad);
  }

  for (let row = 0; row < outH; row++) {
    const latDeg = 90 - (row / Math.max(1, outH - 1)) * 180;
    const latRad = latDeg * d2r;
    const sinLat = Math.sin(latRad);
    const cosLat = Math.cos(latRad);
    const cosZNoon = sinLat * astro.sinDecl + cosLat * astro.cosDecl;
    const noonDenom = Math.max(cosZNoon, 0.04);

    for (let col = 0; col < outW; col++) {
      const o = (row * outW + col) * 4;
      const cosZNow = sinLat * astro.sinDecl + cosLat * astro.cosDecl * cosHa[col];

      let du = 300;
      let cld = 0.25;
      let km = 0;
      let cal = Math.abs(latDeg) > 65 ? 1.28 : 1.0;
      let cpcBase = 0;
      let refDu = 300;
      let refCld = 0.25;

      if (grid && grid.nx === outW && grid.ny === outH) {
        const idx = row * outW + col;
        du = grid.ozoneDU[idx];
        cld = grid.cloudFrac[idx];
        km = grid.elevKm[idx];
        cal = grid.calibFactor[idx];
        cpcBase = grid.cpcNoonUvi[idx];
        refDu = Math.max(80, grid.refOzoneDU[idx]);
        refCld = grid.refCloudFrac[idx];
      }

      const gfsNoon =
        clearSkyUviGfs(cosZNoon, du, km) * cloudTransmission(cld) * cal;
      let noonUvi = gfsNoon;
      if (cpcBlend > 0.01 && cpcBase > 0.01) {
        const ozRatio = Math.pow(refDu / Math.max(80, du), 1.23);
        const cldRatio =
          cloudTransmission(cld) / Math.max(0.35, cloudTransmission(refCld));
        const cpcAdj = Math.max(0, Math.min(25, cpcBase * ozRatio * cldRatio));
        noonUvi = gfsNoon * (1 - cpcBlend) + cpcAdj * cpcBlend;
      }

      let uviNow = 0;
      if (cosZNow > 0) {
        uviNow = noonUvi * Math.pow(cosZNow / noonDenom, 2.15);
      } else if (cosZNow > -0.055) {
        const twi = (cosZNow + 0.055) / 0.055;
        uviNow = Math.min(0.24, noonUvi * 0.025) * twi * twi;
      }

      out[o + 0] = uviNow;
      out[o + 1] = noonUvi;
      out[o + 2] = du;
      out[o + 3] = cld;
    }
  }
  return out;
}
