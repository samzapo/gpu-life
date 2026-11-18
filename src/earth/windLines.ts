import * as d3Geo from 'd3-geo';
import { sampleWindLonLat, WIND_LEVELS, type WindGrid } from './windData';
import { OCEAN_CURRENTS_SPEC, sampleOceanLonLat, type OceanGrid } from './oceanData';

export interface StreamFieldSampler {
  streamMax: number;
  masked: boolean;
  fadeAlpha: number;
  tintRgb: [number, number, number];
  lineWidth: number;
  sample: (lonDeg: number, latDeg: number) => [number, number, number, number];
}

export function windGridToSampler(grid: WindGrid): StreamFieldSampler {
  const spec = WIND_LEVELS[grid.level];
  return {
    streamMax: Math.max(spec.streamMax, 5),
    masked: false,
    fadeAlpha: 0.94,
    tintRgb: [255, 255, 255],
    lineWidth: 1.05,
    sample: (lon, lat) => {
      const [u, v, mag] = sampleWindLonLat(grid, lon, lat);
      return [u, v, mag, 1];
    },
  };
}

export function oceanGridToSampler(
  grid: OceanGrid,
  rgbTint: [number, number, number] = [215, 248, 255],
): StreamFieldSampler {
  return {
    streamMax: OCEAN_CURRENTS_SPEC.streamMax,
    masked: true,
    fadeAlpha: 0.95,
    tintRgb: rgbTint,
    lineWidth: 1.15,
    sample: (lon, lat) => sampleOceanLonLat(grid, lon, lat),
  };
}

interface StreamParticle {
  lon: number;
  lat: number;
  age: number;
}

const MAX_AGE = 85;
const INTENSITY_BUCKETS = 12;

function buildIntensityStyles(rgb: [number, number, number]): string[] {
  const out: string[] = [];
  for (let i = 0; i < INTENSITY_BUCKETS; i++) {
    const scale = (105 + (150 * i) / (INTENSITY_BUCKETS - 1)) / 255;
    const r = Math.round(rgb[0] * scale);
    const g = Math.round(rgb[1] * scale);
    const b = Math.round(rgb[2] * scale);
    out.push(`rgba(${r}, ${g}, ${b}, 0.88)`);
  }
  return out;
}

export class WindStreamRenderer {
  private particles: StreamParticle[] = [];
  private viewKey = '';
  private frameCounter = 0;
  private cachedTintKey = '';
  private cachedStyles: string[] = [];

  private getStyles(tint: [number, number, number]): string[] {
    const key = `${tint[0]},${tint[1]},${tint[2]}`;
    if (key !== this.cachedTintKey) {
      this.cachedTintKey = key;
      this.cachedStyles = buildIntensityStyles(tint);
    }
    return this.cachedStyles;
  }

  clear(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.clearRect(0, 0, width, height);
    this.particles.length = 0;
  }

  private ensureCount(width: number, masked: boolean): void {
    const base = Math.max(1200, Math.min(3400, Math.round(width * (masked ? 2.5 : 2.2))));
    while (this.particles.length < base) {
      this.particles.push({
        lon: (Math.random() - 0.5) * 360,
        lat: Math.asin(Math.random() * 2 - 1) * (180 / Math.PI),
        age: Math.floor(Math.random() * MAX_AGE),
      });
    }
    if (this.particles.length > base) {
      this.particles.length = base;
    }
  }

  private respawnInViewport(
    p: StreamParticle,
    proj: d3Geo.GeoProjection,
    width: number,
    height: number,
    sampler: StreamFieldSampler,
  ): void {
    const inv = proj.invert;
    if (inv) {
      const maxTries = sampler.masked ? 14 : 6;
      for (let tries = 0; tries < maxTries; tries++) {
        const sx = Math.random() * width;
        const sy = Math.random() * height;
        const ll = inv([sx, sy]);
        if (ll && Number.isFinite(ll[0]) && Number.isFinite(ll[1])) {
          const chk = proj(ll);
          if (
            chk &&
            Number.isFinite(chk[0]) &&
            Number.isFinite(chk[1]) &&
            chk[0] >= 0 &&
            chk[0] <= width &&
            chk[1] >= 0 &&
            chk[1] <= height &&
            Math.hypot(chk[0] - sx, chk[1] - sy) < 8
          ) {
            if (!sampler.masked || sampler.sample(ll[0], ll[1])[3] >= 0.45) {
              p.lon = ll[0];
              p.lat = ll[1];
              p.age = Math.floor(Math.random() * 14);
              return;
            }
          }
        }
      }
    }
    if (sampler.masked) {
      for (let t = 0; t < 10; t++) {
        const lon = (Math.random() - 0.5) * 360;
        const lat = Math.asin(Math.random() * 2 - 1) * (180 / Math.PI);
        if (sampler.sample(lon, lat)[3] >= 0.45) {
          p.lon = lon;
          p.lat = lat;
          p.age = Math.floor(Math.random() * 14);
          return;
        }
      }
    }
    p.lon = (Math.random() - 0.5) * 360;
    p.lat = Math.asin(Math.random() * 2 - 1) * (180 / Math.PI);
    p.age = Math.floor(Math.random() * 14);
  }

  stepAndDraw(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    proj: d3Geo.GeoProjection,
    gridOrSampler: WindGrid | StreamFieldSampler,
    viewSignature: string,
    active: boolean,
  ): void {
    if (!active) {
      if (this.particles.length > 0) {
        this.clear(ctx, width, height);
      }
      return;
    }

    const sampler =
      'sample' in gridOrSampler ? gridOrSampler : windGridToSampler(gridOrSampler);
    const styles = this.getStyles(sampler.tintRgb);

    if (viewSignature !== this.viewKey) {
      this.viewKey = viewSignature;
      ctx.clearRect(0, 0, width, height);
      this.ensureCount(width, sampler.masked);
      for (const p of this.particles) {
        this.respawnInViewport(p, proj, width, height, sampler);
        p.age = Math.floor(Math.random() * MAX_AGE);
      }
    } else {
      this.ensureCount(width, sampler.masked);
    }

    // Fade existing trails using destination-in composite (exact cambecc/earth trail look)
    const prevOp = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = 'destination-in';
    ctx.fillStyle = `rgba(0, 0, 0, ${sampler.fadeAlpha})`;
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = prevOp;

    const scale = proj.scale();
    // Target ~4.2 px/frame at streamMax across all zoom levels, keeping fast gales, jets & currents crisp
    const radScale = 4.2 / (Math.max(scale, 80) * Math.max(sampler.streamMax, 0.25));

    const buckets: [number, number, number, number][][] = Array.from(
      { length: INTENSITY_BUCKETS },
      () => [],
    );

    const maxSegJump = 28;
    const DEG2RAD = Math.PI / 180;
    const RAD2DEG = 180 / Math.PI;

    for (const p of this.particles) {
      if (p.age >= MAX_AGE) {
        this.respawnInViewport(p, proj, width, height, sampler);
      }
      p.age++;

      let curLon = p.lon;
      let curLat = p.lat;
      let scr0 = proj([curLon, curLat]);
      const [u0, v0, mag0, mask0] = sampler.sample(curLon, curLat);
      if (sampler.masked && mask0 < 0.42) {
        p.age = MAX_AGE;
        continue;
      }

      const nSub = Math.min(4, Math.max(1, Math.ceil(mag0 / Math.max(sampler.streamMax, 0.25))));
      const subRadScale = radScale / nSub;

      for (let s = 0; s < nSub; s++) {
        const [u, v, mag, mask] =
          s === 0 ? [u0, v0, mag0, mask0] : sampler.sample(curLon, curLat);
        if (sampler.masked && mask < 0.40) {
          p.age = MAX_AGE;
          break;
        }
        const cosLat = Math.max(0.18, Math.cos(curLat * DEG2RAD));
        const dLon = (u / cosLat) * subRadScale * RAD2DEG;
        const dLat = v * subRadScale * RAD2DEG;

        let nextLon = curLon + dLon;
        if (nextLon > 180) nextLon -= 360;
        else if (nextLon < -180) nextLon += 360;
        const nextLat = Math.max(-89.5, Math.min(89.5, curLat + dLat));

        const scr1 = proj([nextLon, nextLat]);
        curLon = nextLon;
        curLat = nextLat;

        if (
          scr0 &&
          scr1 &&
          Number.isFinite(scr0[0]) &&
          Number.isFinite(scr0[1]) &&
          Number.isFinite(scr1[0]) &&
          Number.isFinite(scr1[1])
        ) {
          const dx = scr1[0] - scr0[0];
          const dy = scr1[1] - scr0[1];
          if (dx * dx + dy * dy <= maxSegJump * maxSegJump) {
            const bIdx = Math.min(
              INTENSITY_BUCKETS - 1,
              Math.floor(
                (Math.min(mag, sampler.streamMax) / sampler.streamMax) *
                  (INTENSITY_BUCKETS - 1),
              ),
            );
            buckets[bIdx].push([scr0[0], scr0[1], scr1[0], scr1[1]]);
          }
        }
        scr0 = scr1;
      }

      p.lon = curLon;
      p.lat = curLat;

      if (
        !scr0 ||
        !Number.isFinite(scr0[0]) ||
        !Number.isFinite(scr0[1]) ||
        scr0[0] < -40 ||
        scr0[0] > width + 40 ||
        scr0[1] < -40 ||
        scr0[1] > height + 40
      ) {
        if (scr0 || this.frameCounter % 4 === 0) {
          p.age = MAX_AGE;
        }
      }
    }

    ctx.save();
    ctx.beginPath();
    d3Geo.geoPath(proj, ctx)({ type: 'Sphere' });
    ctx.clip();

    ctx.lineWidth = sampler.lineWidth;
    ctx.lineCap = 'round';
    for (let i = 0; i < INTENSITY_BUCKETS; i++) {
      const segs = buckets[i];
      if (segs.length === 0) continue;
      ctx.beginPath();
      ctx.strokeStyle = styles[i];
      for (let k = 0; k < segs.length; k++) {
        const s = segs[k];
        ctx.moveTo(s[0], s[1]);
        ctx.lineTo(s[2], s[3]);
      }
      ctx.stroke();
    }
    ctx.restore();
    this.frameCounter++;
  }
}
