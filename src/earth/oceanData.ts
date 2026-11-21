import {
  buildDivergenceFreeSphereTex,
  interpolateColorStops,
  sampleWindTempLonLat,
  type WindGrid,
} from './windData';

export const OCEAN_CURRENTS_SPEC = {
  id: 'currents',
  label: 'Sfc Currents',
  scaleMax: 1.5,
  streamMax: 0.7,
} as const;

export const OCEAN_SST_SPEC = {
  minK: 270.0,
  maxK: 313.15,
  minC: -3.15,
  maxC: 40.0,
} as const;

export interface OceanGrid {
  nx: number;
  ny: number;
  u: Float32Array;
  v: Float32Array;
  mask: Uint8Array;
  sstC: Float32Array;
  sstMask: Uint8Array;
  refTime: string;
  source: string;
  sstRefTime: string;
  sstSource: string;
}

let oceanPromise: Promise<OceanGrid> | null = null;

export function decodeBase64Bytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    out[i] = bin.charCodeAt(i);
  }
  return out;
}

function defaultSstClimatologyC(j: number, ny: number): number {
  const latDeg = 90 - (j / Math.max(1, ny - 1)) * 180;
  const cosLat = Math.max(0, Math.cos((latDeg * Math.PI) / 180));
  return -1.5 + 30.5 * Math.pow(cosLat, 1.35);
}

/**
 * Loads the daily NOAA CoastWatch Blended NRT global ocean surface currents & Geo-Polar Blended SST snapshot (`./data/ocean-currents.json`).
 */
export function loadOceanCurrents(forceRefresh = false): Promise<OceanGrid> {
  if (forceRefresh) {
    oceanPromise = null;
  }
  if (oceanPromise) return oceanPromise;

  oceanPromise = (async () => {
    const res = await fetch('./data/ocean-currents.json', forceRefresh ? { cache: 'no-store' } : undefined);
    if (!res.ok) {
      throw new Error(`Ocean currents HTTP ${res.status}`);
    }
    const raw = (await res.json()) as {
      nx: number;
      ny: number;
      scale: number;
      sentinel: number;
      refTime: string;
      source: string;
      uvB64: string;
      sstScale?: number;
      sstSentinel?: number;
      sstRefTime?: string;
      sstSource?: string;
      sstB64?: string;
    };

    const bytes = decodeBase64Bytes(raw.uvB64);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const n = raw.nx * raw.ny;
    const u = new Float32Array(n);
    const v = new Float32Array(n);
    const mask = new Uint8Array(n);

    for (let idx = 0; idx < n; idx++) {
      const iu = view.getInt16(idx * 4, true);
      const iv = view.getInt16(idx * 4 + 2, true);
      if (iu === raw.sentinel || iv === raw.sentinel) {
        mask[idx] = 0;
        u[idx] = 0;
        v[idx] = 0;
      } else {
        mask[idx] = 1;
        u[idx] = iu * raw.scale;
        v[idx] = iv * raw.scale;
      }
    }

    const sstC = new Float32Array(n);
    const sstMask = new Uint8Array(n);
    if (typeof raw.sstB64 === 'string' && raw.sstB64.length > 0) {
      const sBytes = decodeBase64Bytes(raw.sstB64);
      const sView = new DataView(sBytes.buffer, sBytes.byteOffset, sBytes.byteLength);
      const sScale = raw.sstScale ?? 0.01;
      const sSentinel = raw.sstSentinel ?? -32768;
      for (let idx = 0; idx < n; idx++) {
        const it = sView.getInt16(idx * 2, true);
        if (it !== sSentinel) {
          sstMask[idx] = 1;
          sstC[idx] = it * sScale;
        } else if (mask[idx]) {
          // Polar ice / coast gap fill from smooth latitudinal climatology
          const j = Math.floor(idx / raw.nx);
          sstMask[idx] = 1;
          sstC[idx] = defaultSstClimatologyC(j, raw.ny);
        }
      }
    } else {
      for (let j = 0; j < raw.ny; j++) {
        const cVal = defaultSstClimatologyC(j, raw.ny);
        for (let i = 0; i < raw.nx; i++) {
          const idx = j * raw.nx + i;
          if (mask[idx]) {
            sstMask[idx] = 1;
            sstC[idx] = cVal;
          }
        }
      }
    }

    return {
      nx: raw.nx,
      ny: raw.ny,
      u,
      v,
      mask,
      sstC,
      sstMask,
      refTime: raw.refTime,
      source: raw.source,
      sstRefTime: raw.sstRefTime ?? raw.refTime,
      sstSource: raw.sstSource ?? 'Climatology SST',
    };
  })().catch((err) => {
    oceanPromise = null;
    throw err;
  });

  return oceanPromise;
}

/**
 * Masked bilinear sample `[u, v, mag, mask]` (m/s) at `[lonDeg, latDeg]`.
 * Velocity is normalized by ocean weight so coastal boundary currents stay sharp right to `mask = 0.45`.
 */
export function sampleOceanLonLat(
  grid: OceanGrid,
  lonDeg: number,
  latDeg: number,
): [number, number, number, number] {
  const { nx, ny, u, v, mask } = grid;
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

  const r0 = j0 * nx;
  const r1 = j1 * nx;
  const k00 = r0 + i0;
  const k10 = r0 + i1;
  const k01 = r1 + i0;
  const k11 = r1 + i1;

  let mSum = 0;
  let uSum = 0;
  let vSum = 0;
  if (mask[k00]) { mSum += w00; uSum += u[k00] * w00; vSum += v[k00] * w00; }
  if (mask[k10]) { mSum += w10; uSum += u[k10] * w10; vSum += v[k10] * w10; }
  if (mask[k01]) { mSum += w01; uSum += u[k01] * w01; vSum += v[k01] * w01; }
  if (mask[k11]) { mSum += w11; uSum += u[k11] * w11; vSum += v[k11] * w11; }

  if (mSum <= 1e-4) {
    return [0, 0, 0, 0];
  }
  const uu = uSum / mSum;
  const vv = vSum / mSum;
  return [uu, vv, Math.hypot(uu, vv), mSum];
}

/**
 * Masked bilinear sample `[sstK, sstMask]` at geographic `[lonDeg, latDeg]`.
 */
export function sampleOceanSstLonLat(
  grid: OceanGrid,
  lonDeg: number,
  latDeg: number,
): [number, number] {
  const { nx, ny, sstC, sstMask } = grid;
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

  const ws = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
  const ks = [j0 * nx + i0, j0 * nx + i1, j1 * nx + i0, j1 * nx + i1];

  let mSum = 0;
  let tSum = 0;
  for (let q = 0; q < 4; q++) {
    const k = ks[q];
    if (sstMask[k]) {
      const w = ws[q];
      mSum += w;
      tSum += sstC[k] * w;
    }
  }

  if (mSum <= 1e-4) {
    return [288.15, 0];
  }
  return [273.15 + tSum / mSum, mSum];
}

const SST_TEMP_STOPS: readonly [number, [number, number, number]][] = [
  [270.0, [255, 255, 255]],
  [271.4, [255, 255, 255]],
  [271.5, [15, 4, 168]],
  [273.15, [15, 54, 208]],
  [273.25, [15, 54, 188]],
  [275.65, [15, 4, 168]],
  [281.65, [24, 132, 14]],
  [291.15, [247, 251, 59]],
  [295.0, [235, 167, 0]],
  [299.65, [245, 0, 39]],
  [303.0, [87, 17, 0]],
  [304.65, [210, 22, 210]],
  [313.15, [255, 255, 255]],
];

/**
 * Exact `earth.nullschool.net` Sea Surface Temperature color gradient `[270.0 .. 313.15] K`.
 */
export function sstColorRgb(tempK: number): [number, number, number] {
  return interpolateColorStops(tempK, SST_TEMP_STOPS);
}

/**
 * Masked spherical Helmholtz projection packing RGBA32Float `(uDf, vDf, rawMag, oceanMask)` for GPU.
 */
export function buildDivergenceFreeOcean(
  grid: OceanGrid,
  outW: number = 360,
  outH: number = 181,
): Float32Array<ArrayBuffer> {
  return buildDivergenceFreeSphereTex(
    (lonDeg, latDeg) => sampleOceanLonLat(grid, lonDeg, latDeg),
    outW,
    outH,
    true,
  );
}

/**
 * Packs RGBA32Float `(airTempK, sstK, sstMask, sfcAirTempK)` onto `(outW x outH)` for GPU compute + overlay rendering.
 */
export function buildGlobalTempTexture(
  windGrid: WindGrid,
  oceanGrid: OceanGrid | null,
  outW: number = 360,
  outH: number = 181,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(outW * outH * 4);
  for (let row = 0; row < outH; row++) {
    const latDeg = 90 - (row / Math.max(1, outH - 1)) * 180;
    for (let col = 0; col < outW; col++) {
      const lonDeg = (col / outW) * 360;
      const [airK, sfcAirK] = sampleWindTempLonLat(windGrid, lonDeg, latDeg);
      const [sstK, sstMask] = oceanGrid
        ? sampleOceanSstLonLat(oceanGrid, lonDeg, latDeg)
        : [sfcAirK, 0];
      const o = (row * outW + col) * 4;
      out[o + 0] = airK;
      out[o + 1] = sstMask >= 0.45 ? sstK : sfcAirK;
      out[o + 2] = sstMask;
      out[o + 3] = sfcAirK;
    }
  }
  return out;
}
