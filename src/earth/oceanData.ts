import { buildDivergenceFreeSphereTex } from './windData';

export const OCEAN_CURRENTS_SPEC = {
  id: 'currents',
  label: 'Sfc Currents',
  scaleMax: 1.5,
  streamMax: 0.7,
} as const;

export interface OceanGrid {
  nx: number;
  ny: number;
  u: Float32Array;
  v: Float32Array;
  mask: Uint8Array;
  refTime: string;
  source: string;
}

let oceanPromise: Promise<OceanGrid> | null = null;

function decodeBase64Bytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    out[i] = bin.charCodeAt(i);
  }
  return out;
}

/**
 * Loads the daily NOAA CoastWatch Blended NRT global ocean surface currents snapshot (`./data/ocean-currents.json`).
 */
export function loadOceanCurrents(): Promise<OceanGrid> {
  if (oceanPromise) return oceanPromise;

  oceanPromise = (async () => {
    const res = await fetch('./data/ocean-currents.json');
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

    return {
      nx: raw.nx,
      ny: raw.ny,
      u,
      v,
      mask,
      refTime: raw.refTime,
      source: raw.source,
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

  const ws = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
  const ks = [j0 * nx + i0, j0 * nx + i1, j1 * nx + i0, j1 * nx + i1];

  let mSum = 0;
  let uSum = 0;
  let vSum = 0;
  for (let q = 0; q < 4; q++) {
    const k = ks[q];
    if (mask[k]) {
      const w = ws[q];
      mSum += w;
      uSum += u[k] * w;
      vSum += v[k] * w;
    }
  }

  if (mSum <= 1e-4) {
    return [0, 0, 0, 0];
  }
  const uu = uSum / mSum;
  const vv = vSum / mSum;
  return [uu, vv, Math.hypot(uu, vv), mSum];
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
