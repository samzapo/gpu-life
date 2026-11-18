import * as d3Geo from 'd3-geo';
import {
  geoMollweide,
  geoPatterson,
  geoWinkel3,
} from 'd3-geo-projection';
import { geoPolyhedralWaterman } from 'd3-geo-polygon';

export type ProjectionId =
  | 'orthographic'
  | 'atlantis'
  | 'conic_equidistant'
  | 'equirectangular'
  | 'equal_earth'
  | 'patterson'
  | 'stereographic'
  | 'waterman'
  | 'winkel3';

export interface ProjectionSpec {
  id: ProjectionId;
  shortCode: string;
  label: string;
  shaderType: number;
  defaultRotate: [number, number, number];
  /** If true, dragging rotates along spherical axes; otherwise rotates lon and shifts pan cleanly. */
  sphericalDrag: boolean;
}

export const PROJECTIONS: Record<ProjectionId, ProjectionSpec> = {
  orthographic: {
    id: 'orthographic',
    shortCode: 'O',
    label: 'Orthographic (O)',
    shaderType: 0,
    defaultRotate: [36.25, -34.05, 0],
    sphericalDrag: true,
  },
  atlantis: {
    id: 'atlantis',
    shortCode: 'A',
    label: 'Atlantis (A)',
    shaderType: 1,
    defaultRotate: [30, -45, 90],
    sphericalDrag: false,
  },
  conic_equidistant: {
    id: 'conic_equidistant',
    shortCode: 'CE',
    label: 'Conic Equidistant (CE)',
    shaderType: 2,
    defaultRotate: [0, 0, 0],
    sphericalDrag: false,
  },
  equirectangular: {
    id: 'equirectangular',
    shortCode: 'E',
    label: 'Equirectangular (E)',
    shaderType: 3,
    defaultRotate: [0, 0, 0],
    sphericalDrag: false,
  },
  equal_earth: {
    id: 'equal_earth',
    shortCode: 'EQ',
    label: 'Equal Earth (EQ)',
    shaderType: 4,
    defaultRotate: [0, 0, 0],
    sphericalDrag: false,
  },
  patterson: {
    id: 'patterson',
    shortCode: 'P',
    label: 'Patterson (P)',
    shaderType: 5,
    defaultRotate: [0, 0, 0],
    sphericalDrag: false,
  },
  stereographic: {
    id: 'stereographic',
    shortCode: 'S',
    label: 'Stereographic (S)',
    shaderType: 6,
    defaultRotate: [-43, -20, 0],
    sphericalDrag: true,
  },
  waterman: {
    id: 'waterman',
    shortCode: 'WB',
    label: 'Waterman Butterfly (WB)',
    shaderType: 7,
    defaultRotate: [20, 0, 0],
    sphericalDrag: false,
  },
  winkel3: {
    id: 'winkel3',
    shortCode: 'W3',
    label: 'Winkel Tripel (W3)',
    shaderType: 8,
    defaultRotate: [0, 0, 0],
    sphericalDrag: false,
  },
};

export interface GlobeViewState {
  projectionId: ProjectionId;
  rotate: [number, number, number];
  scale: number;
  pan: [number, number];
  width: number;
  height: number;
}

const DEG_TO_RAD = Math.PI / 180;

export function lonLatToUnitVec(lonDeg: number, latDeg: number): [number, number, number] {
  const lam = lonDeg * DEG_TO_RAD;
  const phi = latDeg * DEG_TO_RAD;
  const cosphi = Math.cos(phi);
  return [cosphi * Math.cos(lam), cosphi * Math.sin(lam), Math.sin(phi)];
}

export function makeBaseD3Projection(id: ProjectionId): d3Geo.GeoProjection {
  switch (id) {
    case 'orthographic':
      return d3Geo.geoOrthographic().clipAngle(90).precision(0.1);
    case 'atlantis':
      return geoMollweide().precision(0.1);
    case 'conic_equidistant':
      return d3Geo.geoConicEquidistant().precision(0.1);
    case 'equirectangular':
      return d3Geo.geoEquirectangular().precision(0.1);
    case 'equal_earth':
      return d3Geo.geoEqualEarth().precision(0.1);
    case 'patterson':
      return geoPatterson().precision(0.1);
    case 'stereographic':
      return d3Geo.geoStereographic().clipAngle(179).precision(0.5);
    case 'waterman':
      return geoPolyhedralWaterman().precision(0.1);
    case 'winkel3':
      return geoWinkel3().precision(0.1);
  }
}

export function buildConfiguredD3Projection(state: GlobeViewState): d3Geo.GeoProjection {
  const proj = makeBaseD3Projection(state.projectionId);
  proj
    .rotate(state.rotate)
    .scale(state.scale)
    .translate([state.width * 0.5 + state.pan[0], state.height * 0.5 + state.pan[1]]);
  if (state.projectionId === 'stereographic') {
    proj.clipExtent([
      [-state.width * 0.25, -state.height * 0.25],
      [state.width * 1.25, state.height * 1.25],
    ]);
  }
  return proj;
}

export function fitScaleForProjection(
  id: ProjectionId,
  rotate: [number, number, number],
  width: number,
  height: number,
): number {
  const unitProj = makeBaseD3Projection(id).rotate(rotate).scale(1).translate([0, 0]);
  if (id === 'stereographic') {
    return Math.min(width, height) * 0.42;
  }
  const bounds = d3Geo.geoPath(unitProj).bounds({ type: 'Sphere' });
  const w1 = bounds[1][0] - bounds[0][0];
  const h1 = bounds[1][1] - bounds[0][1];
  if (!Number.isFinite(w1) || !Number.isFinite(h1) || w1 <= 0 || h1 <= 0) {
    return Math.min(width, height) * 0.35;
  }
  return Math.min(width / w1, height / h1) * 0.9;
}

/**
 * Exact 3x3 orthogonal matrix (column-major Float32, padded to mat3x3<f32> = 12 floats)
 * matching `d3.geoRotation(rotate)` in Geographic XYZ space.
 */
export function buildD3RotationMatrixColMajor(rotate: [number, number, number]): Float32Array {
  const rot = d3Geo.geoRotation(rotate);
  const c0 = lonLatToUnitVec(...rot([0, 0]));
  const c1 = lonLatToUnitVec(...rot([90, 0]));
  const c2 = lonLatToUnitVec(...rot([0, 90]));
  // WGSL mat3x3<f32> stores 3 vec3f columns aligned to 16 bytes (stride 4 floats per column)
  return new Float32Array([
    c0[0], c0[1], c0[2], 0,
    c1[0], c1[1], c1[2], 0,
    c2[0], c2[1], c2[2], 0,
  ]);
}

/**
 * Computes unit-scale forward and inverse lookup textures for Waterman Butterfly (`WB`).
 * Forward grid: `360 x 181` indexed by geographic `(lonDeg 0..359, 90 - latDeg 0..180)`.
 * Inverse grid: `320 x 180` across normalized view viewport storing `(nx, ny, nz, hit)`.
 */
export function buildWatermanLutData(
  state: GlobeViewState,
  proj: d3Geo.GeoProjection,
): {
  fwdData: Float32Array<ArrayBuffer>;
  fwdW: number;
  fwdH: number;
  invData: Float32Array<ArrayBuffer>;
  invW: number;
  invH: number;
} {
  const fwdW = 360;
  const fwdH = 181;
  const fwdData = new Float32Array(fwdW * fwdH * 4);
  const unitWaterman = geoPolyhedralWaterman()
    .rotate(state.rotate)
    .scale(1)
    .translate([0, 0])
    .precision(0.1);

  for (let row = 0; row < fwdH; row++) {
    const lat = 90 - row;
    for (let col = 0; col < fwdW; col++) {
      const lon = col > 180 ? col - 360 : col;
      const p = unitWaterman([lon, lat]);
      const o = (row * fwdW + col) * 4;
      if (p && Number.isFinite(p[0]) && Number.isFinite(p[1])) {
        // Check local frame rotation relative to East so orbitals preserve local angle
        const eps = 0.05;
        const pE = unitWaterman([lon + eps, lat]);
        let cosA = 1;
        let sinA = 0;
        if (pE && Number.isFinite(pE[0]) && Number.isFinite(pE[1])) {
          const dx = pE[0] - p[0];
          const dy = pE[1] - p[1];
          const len = Math.hypot(dx, dy);
          if (len > 1e-6 && len < 0.05) {
            cosA = dx / len;
            sinA = dy / len;
          }
        }
        fwdData[o + 0] = p[0];
        fwdData[o + 1] = p[1];
        fwdData[o + 2] = cosA;
        fwdData[o + 3] = sinA;
      } else {
        fwdData[o + 0] = 1e6;
        fwdData[o + 1] = 1e6;
        fwdData[o + 2] = 0;
        fwdData[o + 3] = 0;
      }
    }
  }

  const invW = 640;
  const invH = 360;
  const invData = new Float32Array(invW * invH * 4);
  const invertFn = proj.invert;
  for (let y = 0; y < invH; y++) {
    const sy = ((y + 0.5) / invH) * state.height;
    for (let x = 0; x < invW; x++) {
      const sx = ((x + 0.5) / invW) * state.width;
      const o = (y * invW + x) * 4;
      const coord = invertFn ? invertFn([sx, sy]) : null;
      if (coord && Number.isFinite(coord[0]) && Number.isFinite(coord[1])) {
        // Verify roundtrip lies close to pixel (clips outside butterfly polygon cleanly)
        const check = proj(coord);
        if (check && Math.hypot(check[0] - sx, check[1] - sy) < 6) {
          const v = lonLatToUnitVec(coord[0], coord[1]);
          invData[o + 0] = v[0];
          invData[o + 1] = v[1];
          invData[o + 2] = v[2];
          invData[o + 3] = 1;
          continue;
        }
      }
      invData[o + 0] = 0;
      invData[o + 1] = 0;
      invData[o + 2] = 0;
      invData[o + 3] = 0;
    }
  }

  return { fwdData, fwdW, fwdH, invData, invW, invH };
}

export interface ParsedNullschoolHash {
  level?: string;
  projection?: ProjectionId;
  orientation?: [number, number, number];
}

export function parseNullschoolHash(hash: string): ParsedNullschoolHash {
  const clean = hash.replace(/^#/, '').trim();
  if (!clean) return {};
  const result: ParsedNullschoolHash = {};
  const parts = clean.split('/');
  for (const part of parts) {
    if (part === 'surface') {
      result.level = 'surface';
    } else if (/^\d+hPa$/.test(part)) {
      result.level = part;
    }
    const m = /^(\w+)=([\d\-.,]+)$/.exec(part);
    if (m) {
      const candidate = m[1] as ProjectionId;
      if (candidate in PROJECTIONS) {
        result.projection = candidate;
        const nums = m[2].split(',').map(Number);
        if (nums.length >= 2 && nums.every(Number.isFinite)) {
          result.orientation = [nums[0], nums[1], nums[2] ?? 586];
        }
      }
    } else if (part in PROJECTIONS) {
      result.projection = part as ProjectionId;
    }
  }
  return result;
}
