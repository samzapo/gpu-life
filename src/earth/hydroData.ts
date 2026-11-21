import {
  fetchGfsCycleAtOffset,
  fetchGfsFieldSlice,
  getLatestGfsCycleInfo,
  interpolateColorStops,
} from './windData';

export const HYDRO_SPEC = {
  rhMin: 0,
  rhMax: 100,
  precipMaxMm: 30,
} as const;

export interface HydroGrid {
  nx: number;
  ny: number;
  /** 2m Relative Humidity `[0..1]` */
  rh: Float32Array;
  /** 3-hour precipitation accumulation (`mm` / 3h) */
  precip3hMm: Float32Array;
  gfsRefTime: string;
  gfsValidTime: string;
  source: string;
  isLive: boolean;
}

export interface LocalHydroSample {
  rhPct: number;
  precip3hMm: number;
}

const RH_COLOR_STOPS: readonly [number, [number, number, number]][] = [
  [0.0, [94, 62, 30]],
  [0.2, [168, 134, 68]],
  [0.4, [202, 198, 118]],
  [0.6, [92, 186, 174]],
  [0.8, [34, 146, 212]],
  [1.0, [68, 74, 214]],
];

const PRECIP_COLOR_STOPS: readonly [number, [number, number, number]][] = [
  [0.0, [14, 20, 32]],
  [0.5, [28, 118, 178]],
  [2.5, [34, 196, 172]],
  [7.0, [168, 228, 64]],
  [15.0, [248, 164, 34]],
  [30.0, [238, 46, 128]],
];

export function rhColorRgb(rh01: number): [number, number, number] {
  return interpolateColorStops(Math.max(0, Math.min(1, rh01)), RH_COLOR_STOPS);
}

export function precip3hColorRgb(mm: number): [number, number, number] {
  return interpolateColorStops(
    Math.max(0, Math.min(HYDRO_SPEC.precipMaxMm, mm)),
    PRECIP_COLOR_STOPS,
  );
}

export function createBaselineHydroGrid(
  nx: number = 360,
  ny: number = 181,
): HydroGrid {
  const n = nx * ny;
  const rh = new Float32Array(n);
  const precip3hMm = new Float32Array(n);
  for (let row = 0; row < ny; row++) {
    const latDeg = 90 - (row / Math.max(1, ny - 1)) * 180;
    const absLat = Math.abs(latDeg);
    // ITCZ peak near equator, dry subtropics around ~25 deg, moist midlatitude storm tracks ~50 deg
    const itcz = Math.exp(-((latDeg - 3) * (latDeg - 3)) / 140);
    const stormTrack = Math.exp(-((absLat - 50) * (absLat - 50)) / 190);
    const subtropDry = Math.exp(-((absLat - 25) * (absLat - 25)) / 120);
    const rhVal = Math.max(
      0.24,
      Math.min(0.9, 0.56 + 0.24 * itcz + 0.2 * stormTrack - 0.24 * subtropDry),
    );
    const rainVal = Math.max(
      0,
      2.6 * itcz + 1.4 * stormTrack - 0.8 * subtropDry,
    );
    for (let col = 0; col < nx; col++) {
      const idx = row * nx + col;
      rh[idx] = rhVal;
      precip3hMm[idx] = rainVal;
    }
  }
  return {
    nx,
    ny,
    rh,
    precip3hMm,
    gfsRefTime: '',
    gfsValidTime: '',
    source: 'Climatology RH + 3HPS',
    isLive: false,
  };
}

const hydroCache = new Map<string, HydroGrid>();

export async function fetchLiveGfsHydroFields(
  opts?: { forceRefresh?: boolean },
): Promise<HydroGrid> {
  const cycle = await getLatestGfsCycleInfo(Boolean(opts?.forceRefresh));
  const cached = hydroCache.get(cycle.cycleLabel);
  if (cached) return cached;

  for (const k of hydroCache.keys()) {
    if (k !== cycle.cycleLabel) hydroCache.delete(k);
  }

  const rhEntry = cycle.idxEntries.find(
    (e) => e.varName === 'RH' && e.level === '2 m above ground',
  );
  if (!rhEntry) {
    throw new Error('GFS 2m RH field not found in index');
  }

  // APCP 3-hour accumulation lives in f003+ forecast files (not f000 analysis).
  let accumCycle = cycle;
  let apcpEntries = cycle.idxEntries.filter(
    (e) => e.varName === 'APCP' && e.level === 'surface',
  );
  if (apcpEntries.length === 0 && cycle.fh === 0) {
    const f003 = await fetchGfsCycleAtOffset(cycle.cycMs, 3);
    if (f003) {
      accumCycle = f003;
      apcpEntries = f003.idxEntries.filter(
        (e) => e.varName === 'APCP' && e.level === 'surface',
      );
    }
  }
  const apcpEntry = apcpEntries.at(-1);
  const prateEntry = !apcpEntry
    ? accumCycle.idxEntries.find(
        (e) => e.varName === 'PRATE' && e.level === 'surface',
      )
    : undefined;

  const [rhGrid, precipGrid, prateGrid] = await Promise.all([
    fetchGfsFieldSlice(cycle, rhEntry),
    apcpEntry
      ? fetchGfsFieldSlice(accumCycle, apcpEntry).catch(() => null)
      : Promise.resolve(null),
    !apcpEntry && prateEntry
      ? fetchGfsFieldSlice(accumCycle, prateEntry).catch(() => null)
      : Promise.resolve(null),
  ]);

  const nx = rhGrid.nx;
  const ny = rhGrid.ny;
  const n = nx * ny;
  const rh = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const pct = rhGrid.values[i];
    rh[i] = Number.isFinite(pct) ? Math.max(0, Math.min(1, pct / 100)) : 0.65;
  }

  const precip3hMm = new Float32Array(n);
  if (precipGrid && precipGrid.values.length === n) {
    for (let i = 0; i < n; i++) {
      const val = precipGrid.values[i];
      precip3hMm[i] = Number.isFinite(val) ? Math.max(0, Math.min(120, val)) : 0;
    }
  } else if (prateGrid && prateGrid.values.length === n) {
    for (let i = 0; i < n; i++) {
      const rate = prateGrid.values[i];
      precip3hMm[i] = Number.isFinite(rate)
        ? Math.max(0, Math.min(120, rate * 10800))
        : 0;
    }
  }

  const grid: HydroGrid = {
    nx,
    ny,
    rh,
    precip3hMm,
    gfsRefTime: cycle.cycleLabel,
    gfsValidTime: cycle.validLabel,
    source: 'NOAA GFS RH + 3HPS',
    isLive: true,
  };
  hydroCache.set(cycle.cycleLabel, grid);
  return grid;
}

export function sampleHydroLonLat(
  grid: HydroGrid,
  lonDeg: number,
  latDeg: number,
): LocalHydroSample {
  const { nx, ny, rh, precip3hMm } = grid;
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

  const rhVal =
    rh[k00] * w00 + rh[k10] * w10 + rh[k01] * w01 + rh[k11] * w11;
  const precipVal =
    precip3hMm[k00] * w00 +
    precip3hMm[k10] * w10 +
    precip3hMm[k01] * w01 +
    precip3hMm[k11] * w11;

  return {
    rhPct: Math.round(Math.max(0, Math.min(1, rhVal)) * 100),
    precip3hMm: Math.max(0, precipVal),
  };
}

/**
 * Packs `(rh01, precip3hMm, 0, 0)` into `(360 x 181)` `rgba32float` texture.
 */
export function buildGlobalHydroTexture(
  grid: HydroGrid | null,
  outW: number = 360,
  outH: number = 181,
): Float32Array<ArrayBuffer> {
  const active = grid ?? createBaselineHydroGrid(outW, outH);
  const out = new Float32Array(outW * outH * 4);
  if (active.nx === outW && active.ny === outH) {
    for (let i = 0; i < outW * outH; i++) {
      const o = i * 4;
      out[o + 0] = active.rh[i];
      out[o + 1] = active.precip3hMm[i];
    }
    return out;
  }
  for (let row = 0; row < outH; row++) {
    const latDeg = 90 - (row / Math.max(1, outH - 1)) * 180;
    for (let col = 0; col < outW; col++) {
      const lonDeg = (col / outW) * 360;
      const sample = sampleHydroLonLat(active, lonDeg, latDeg);
      const o = (row * outW + col) * 4;
      out[o + 0] = sample.rhPct / 100;
      out[o + 1] = sample.precip3hMm;
    }
  }
  return out;
}
