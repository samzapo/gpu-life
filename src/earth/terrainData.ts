import type { OceanGrid } from './oceanData';
import type { UvGrid } from './uvData';

const DEG2RAD = Math.PI / 180;

/**
 * Deterministic major tectonic divergent ridges, rift zones, and volcanic subduction arcs
 * used as a static hydrothermal vent / volcanism probability field until live feeds are wired.
 */
const TECTONIC_POLYLINES: readonly (readonly [number, number])[][ ] = [
  // Mid-Atlantic Ridge (Iceland -> South Atlantic / Bouvet)
  [
    [-18, 66], [-26, 62], [-32, 52], [-30, 40], [-38, 30], [-45, 20],
    [-40, 10], [-25, 2], [-14, -2], [-13, -15], [-14, -30], [-14, -45], [-4, -55],
  ],
  // East Pacific Rise & Galapagos Rift
  [
    [-110, 23], [-105, 16], [-104, 8], [-102, 2], [-90, 1], [-83, 2],
    [-105, 0], [-110, -10], [-113, -22], [-115, -35], [-105, -45], [-85, -55],
  ],
  // Juan de Fuca & Gorda Ridges
  [
    [-129, 50], [-130, 45], [-127, 41],
  ],
  // Southwest, Central, & Southeast Indian Ridges
  [
    [-4, -55], [20, -52], [45, -42], [62, -30], [70, -25],
    [66, -15], [60, -2], [52, 12], [43, 12],
  ],
  [
    [70, -25], [82, -36], [105, -46], [135, -52], [158, -58], [175, -62],
  ],
  // East African & Red Sea Rift
  [
    [35, 30], [39, 20], [42, 12], [40, 6], [36, 0], [35, -8], [34, -18],
  ],
  // Pacific Ring of Fire (Aleutians -> Kuril -> Japan -> Marianas -> PNG -> Tonga-Kermadec -> NZ)
  [
    [-152, 59], [-165, 55], [-178, 52], [172, 53], [160, 52], [152, 46],
    [143, 40], [140, 34], [144, 24], [147, 15], [140, 10],
  ],
  [
    [122, 24], [126, 12], [127, 2], [138, -3], [152, -5], [166, -12],
    [174, -18], [-175, -22], [-178, -32], [177, -38],
  ],
  // Sunda / Java Trench & Arc
  [
    [94, 10], [98, 2], [104, -5], [114, -9], [125, -8],
  ],
  // American Cordillera / Andes Subduction Arc
  [
    [-123, 47], [-122, 42], [-104, 19], [-92, 14], [-85, 10], [-78, 4],
    [-78, -2], [-73, -14], [-68, -22], [-70, -34], [-73, -46],
  ],
  // Mediterranean & Aegean Arc
  [
    [14, 40], [15, 38], [24, 36], [28, 36],
  ],
];

function unitFromLonLat(lonDeg: number, latDeg: number): [number, number, number] {
  const lon = lonDeg * DEG2RAD;
  const lat = latDeg * DEG2RAD;
  const c = Math.cos(lat);
  return [c * Math.cos(lon), c * Math.sin(lon), Math.sin(lat)];
}

function pointToSegmentArcDistRad(
  px: number,
  py: number,
  pz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const lenSq = abx * abx + aby * aby + abz * abz;
  let t = 0;
  if (lenSq > 1e-8) {
    t = Math.max(0, Math.min(1, ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / lenSq));
  }
  let qx = ax + abx * t;
  let qy = ay + aby * t;
  let qz = az + abz * t;
  const qLen = Math.hypot(qx, qy, qz) || 1;
  qx /= qLen;
  qy /= qLen;
  qz /= qLen;
  return Math.acos(Math.max(-1, Math.min(1, px * qx + py * qy + pz * qz)));
}

let cachedRidgeMask: { w: number; h: number; data: Float32Array } | null = null;

function getStaticRidgeVentMask(outW: number, outH: number): Float32Array {
  if (cachedRidgeMask && cachedRidgeMask.w === outW && cachedRidgeMask.h === outH) {
    return cachedRidgeMask.data;
  }
  const segments: [number, number, number, number, number, number][] = [];
  for (const poly of TECTONIC_POLYLINES) {
    for (let i = 0; i < poly.length - 1; i++) {
      const a = unitFromLonLat(poly[i][0], poly[i][1]);
      const b = unitFromLonLat(poly[i + 1][0], poly[i + 1][1]);
      segments.push([a[0], a[1], a[2], b[0], b[1], b[2]]);
    }
  }

  const sigmaRad = 4.2 * DEG2RAD;
  const inv2Sig2 = 1 / (2 * sigmaRad * sigmaRad);
  const data = new Float32Array(outW * outH);
  for (let row = 0; row < outH; row++) {
    const latDeg = 90 - (row / Math.max(1, outH - 1)) * 180;
    for (let col = 0; col < outW; col++) {
      const lonDeg = (col / outW) * 360;
      const [px, py, pz] = unitFromLonLat(lonDeg, latDeg);
      let minAng = 1e9;
      for (let s = 0; s < segments.length; s++) {
        const seg = segments[s];
        const d = pointToSegmentArcDistRad(
          px,
          py,
          pz,
          seg[0],
          seg[1],
          seg[2],
          seg[3],
          seg[4],
          seg[5],
        );
        if (d < minAng) minAng = d;
      }
      const w = Math.exp(-minAng * minAng * inv2Sig2);
      data[row * outW + col] = w < 0.01 ? 0 : Math.min(1, w);
    }
  }
  cachedRidgeMask = { w: outW, h: outH, data };
  return data;
}

export function sampleRidgeMaskLonLat(
  lonDeg: number,
  latDeg: number,
  w: number = 360,
  h: number = 181,
): number {
  const data = getStaticRidgeVentMask(w, h);
  const wrappedLon = ((lonDeg % 360) + 360) % 360;
  const clampedLat = Math.max(-90, Math.min(90, latDeg));
  const col = Math.floor((wrappedLon / 360) * w) % w;
  const row = Math.min(h - 1, Math.max(0, Math.round(((90 - clampedLat) / 180) * (h - 1))));
  return data[row * w + col] ?? 0;
}

export function sampleElevationKmLonLat(
  uvGrid: UvGrid | null,
  lonDeg: number,
  latDeg: number,
): number {
  if (!uvGrid || uvGrid.elevKm.length === 0) return 0;
  const { nx, ny, elevKm } = uvGrid;
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
  const v00 = elevKm[j0 * nx + i0] ?? 0;
  const v10 = elevKm[j0 * nx + i1] ?? 0;
  const v01 = elevKm[j1 * nx + i0] ?? 0;
  const v11 = elevKm[j1 * nx + i1] ?? 0;
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
}

/**
 * Packs `(elevKm, dElevEast, dElevNorth, ridgeVentMask)` into `(outW x outH)` `rgba32float` texture.
 * Over ocean (`oceanMask >= 0.45`), `.r` transitions smoothly down to negative shelf/abyssal depth (`-0.8..0.0`)
 * so slope gradients gently pull coastal water & runoff into marine basins cleanly.
 */
export function buildGlobalTerrainTexture(
  uvGrid: UvGrid | null,
  oceanGrid: OceanGrid | null,
  outW: number = 360,
  outH: number = 181,
): Float32Array<ArrayBuffer> {
  const ridgeMask = getStaticRidgeVentMask(outW, outH);
  const elev = new Float32Array(outW * outH);
  const oceanDepth = new Float32Array(outW * outH);

  for (let row = 0; row < outH; row++) {
    const latDeg = 90 - (row / Math.max(1, outH - 1)) * 180;
    for (let col = 0; col < outW; col++) {
      const lonDeg = (col / outW) * 360;
      const idx = row * outW + col;
      let e = sampleElevationKmLonLat(uvGrid, lonDeg, latDeg);
      let isSea = false;
      if (oceanGrid) {
        const ogx = Math.floor((((lonDeg % 360) + 360) % 360) / 360 * oceanGrid.nx) % oceanGrid.nx;
        const ogy = Math.min(oceanGrid.ny - 1, Math.max(0, Math.round(((90 - latDeg) / 180) * (oceanGrid.ny - 1))));
        isSea = (oceanGrid.mask[ogy * oceanGrid.nx + ogx] ?? 0) > 0;
      }
      if (isSea && e < 0.05) {
        e = -0.35;
        oceanDepth[idx] = 1;
      }
      elev[idx] = e;
    }
  }

  // Smooth coastline shelf depth slightly so coastal gradients point seaward reliably
  const smoothElev = new Float32Array(elev);
  for (let pass = 0; pass < 2; pass++) {
    for (let row = 1; row < outH - 1; row++) {
      for (let col = 0; col < outW; col++) {
        const idx = row * outW + col;
        if (oceanDepth[idx] <= 0) continue;
        const cE = (col + 1) % outW;
        const cW = (col - 1 + outW) % outW;
        const avg =
          0.25 * (elev[row * outW + cE] + elev[row * outW + cW] + elev[(row - 1) * outW + col] + elev[(row + 1) * outW + col]);
        smoothElev[idx] = Math.min(0, 0.55 * elev[idx] + 0.45 * avg);
      }
    }
    elev.set(smoothElev);
  }

  const out = new Float32Array(outW * outH * 4);
  for (let row = 0; row < outH; row++) {
    const latDeg = 90 - (row / Math.max(1, outH - 1)) * 180;
    const cosLat = Math.max(0.2, Math.cos(latDeg * DEG2RAD));
    const rN = Math.max(0, row - 1);
    const rS = Math.min(outH - 1, row + 1);
    const dy = (rS - rN) * 1.0;

    for (let col = 0; col < outW; col++) {
      const idx = row * outW + col;
      const cE = (col + 1) % outW;
      const cW = (col - 1 + outW) % outW;
      const dE = (elev[row * outW + cE] - elev[row * outW + cW]) / (2.0 * cosLat);
      const dN = (elev[rN * outW + col] - elev[rS * outW + col]) / Math.max(1.0, dy);

      const o = idx * 4;
      out[o + 0] = elev[idx];
      out[o + 1] = dE;
      out[o + 2] = dN;
      out[o + 3] = ridgeMask[idx];
    }
  }
  return out;
}
