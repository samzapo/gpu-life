import * as d3Geo from 'd3-geo';

interface TopoGeometry {
  type: 'LineString' | 'MultiLineString' | 'Polygon' | 'MultiPolygon';
  arcs: unknown;
}

interface TopoJsonRoot {
  transform?: { scale: [number, number]; translate: [number, number] };
  arcs: [number, number][][];
  objects: Record<string, { type: string; geometries?: TopoGeometry[] }>;
}

export interface EarthTopoFeatures {
  coastlines: d3Geo.GeoPermissibleObjects;
  lakes: d3Geo.GeoPermissibleObjects;
  graticule: d3Geo.GeoPermissibleObjects;
  hemisphere: d3Geo.GeoPermissibleObjects;
}

function decodeArc(root: TopoJsonRoot, idx: number): [number, number][] {
  const reversed = idx < 0;
  const arcIndex = reversed ? ~idx : idx;
  const rawArc = root.arcs[arcIndex];
  const scale = root.transform?.scale ?? [1, 1];
  const translate = root.transform?.translate ?? [0, 0];

  const coords: [number, number][] = [];
  let x = 0;
  let y = 0;
  for (let i = 0; i < rawArc.length; i++) {
    x += rawArc[i][0];
    y += rawArc[i][1];
    coords.push([x * scale[0] + translate[0], y * scale[1] + translate[1]]);
  }
  return reversed ? coords.reverse() : coords;
}

function stitchArcRing(root: TopoJsonRoot, arcIndices: number[]): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < arcIndices.length; i++) {
    const pts = decodeArc(root, arcIndices[i]);
    if (i > 0 && out.length > 0 && pts.length > 0) {
      out.pop();
    }
    out.push(...pts);
  }
  return out;
}

function objectToMultiLine(root: TopoJsonRoot, objectName: string): d3Geo.GeoPermissibleObjects {
  const obj = root.objects[objectName];
  const coordinates: [number, number][][] = [];
  if (!obj?.geometries) {
    return { type: 'MultiLineString', coordinates } as unknown as d3Geo.GeoPermissibleObjects;
  }

  for (const geom of obj.geometries) {
    if (geom.type === 'LineString') {
      coordinates.push(stitchArcRing(root, geom.arcs as number[]));
    } else if (geom.type === 'MultiLineString') {
      for (const ring of geom.arcs as number[][]) {
        coordinates.push(stitchArcRing(root, ring));
      }
    } else if (geom.type === 'Polygon') {
      for (const ring of geom.arcs as number[][]) {
        coordinates.push(stitchArcRing(root, ring));
      }
    } else if (geom.type === 'MultiPolygon') {
      for (const poly of geom.arcs as number[][][]) {
        for (const ring of poly) {
          coordinates.push(stitchArcRing(root, ring));
        }
      }
    }
  }
  return { type: 'MultiLineString', coordinates } as unknown as d3Geo.GeoPermissibleObjects;
}

let cachedFeatures: Promise<EarthTopoFeatures> | null = null;

export function loadEarthFeatures(): Promise<EarthTopoFeatures> {
  if (!cachedFeatures) {
    cachedFeatures = fetch('./data/earth-topo-mobile.json')
      .then((res) => {
        if (!res.ok) throw new Error(`Failed earth-topo HTTP ${res.status}`);
        return res.json() as Promise<TopoJsonRoot>;
      })
      .then((topo) => ({
        coastlines: objectToMultiLine(topo, 'coastline_110m'),
        lakes: objectToMultiLine(topo, 'lakes_110m'),
        graticule: d3Geo.geoGraticule10(),
        hemisphere: d3Geo.geoGraticule().stepMinor([0, 90]).stepMajor([0, 90])(),
      }));
  }
  return cachedFeatures;
}

export function renderMapOverlayCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  proj: d3Geo.GeoProjection,
  features: EarthTopoFeatures | null,
  showCoastlines: boolean,
  isOrthographic: boolean,
): void {
  ctx.clearRect(0, 0, width, height);
  if (!showCoastlines) {
    return;
  }

  const path = d3Geo.geoPath(proj, ctx);
  if (!isOrthographic) {
    ctx.save();
    ctx.beginPath();
    path({ type: 'Sphere' });
    ctx.clip();
  }

  if (features) {
    ctx.beginPath();
    path(features.graticule);
    ctx.strokeStyle = 'rgba(140, 165, 190, 0.25)';
    ctx.lineWidth = 0.8;
    ctx.stroke();

    ctx.beginPath();
    path(features.hemisphere);
    ctx.strokeStyle = 'rgba(165, 190, 215, 0.34)';
    ctx.lineWidth = 1.0;
    ctx.stroke();

    ctx.beginPath();
    path(features.coastlines);
    ctx.strokeStyle = 'rgba(245, 248, 255, 0.82)';
    ctx.lineWidth = 1.15;
    ctx.stroke();

    ctx.beginPath();
    path(features.lakes);
    ctx.strokeStyle = 'rgba(220, 232, 245, 0.62)';
    ctx.lineWidth = 0.95;
    ctx.stroke();
  }

  if (!isOrthographic) {
    ctx.restore();
  }

  ctx.beginPath();
  path({ type: 'Sphere' });
  ctx.strokeStyle = isOrthographic
    ? 'rgba(155, 200, 235, 0.65)'
    : 'rgba(120, 140, 165, 0.50)';
  ctx.lineWidth = isOrthographic ? 1.6 : 1.1;
  ctx.stroke();
}
