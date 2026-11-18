declare module 'd3-geo-projection' {
  import type { GeoProjection } from 'd3-geo';
  export function geoMollweide(): GeoProjection;
  export function geoPatterson(): GeoProjection;
  export function geoWinkel3(): GeoProjection;
}

declare module 'd3-geo-polygon' {
  import type { GeoProjection } from 'd3-geo';
  export function geoPolyhedralWaterman(): GeoProjection;
}
