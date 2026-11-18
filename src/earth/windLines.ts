import * as d3Geo from 'd3-geo';
import { sampleWindLonLat, WIND_LEVELS, type WindGrid } from './windData';

interface StreamParticle {
  lon: number;
  lat: number;
  age: number;
}

const MAX_AGE = 85;
const INTENSITY_BUCKETS = 12;

function buildIntensityStyles(): string[] {
  const out: string[] = [];
  for (let i = 0; i < INTENSITY_BUCKETS; i++) {
    const lum = Math.round(105 + (150 * i) / (INTENSITY_BUCKETS - 1));
    out.push(`rgba(${lum}, ${lum}, ${lum}, 0.88)`);
  }
  return out;
}

const COLOR_STYLES = buildIntensityStyles();

export class WindStreamRenderer {
  private particles: StreamParticle[] = [];
  private viewKey = '';
  private frameCounter = 0;

  clear(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.clearRect(0, 0, width, height);
    this.particles.length = 0;
  }

  private ensureCount(width: number): void {
    const target = Math.max(1200, Math.min(3400, Math.round(width * 2.2)));
    while (this.particles.length < target) {
      this.particles.push({
        lon: (Math.random() - 0.5) * 360,
        lat: Math.asin(Math.random() * 2 - 1) * (180 / Math.PI),
        age: Math.floor(Math.random() * MAX_AGE),
      });
    }
    if (this.particles.length > target) {
      this.particles.length = target;
    }
  }

  private respawnInViewport(
    p: StreamParticle,
    proj: d3Geo.GeoProjection,
    width: number,
    height: number,
  ): void {
    const inv = proj.invert;
    if (inv) {
      for (let tries = 0; tries < 6; tries++) {
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
            p.lon = ll[0];
            p.lat = ll[1];
            p.age = Math.floor(Math.random() * 14);
            return;
          }
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
    grid: WindGrid,
    viewSignature: string,
    active: boolean,
  ): void {
    if (!active) {
      if (this.particles.length > 0) {
        this.clear(ctx, width, height);
      }
      return;
    }

    if (viewSignature !== this.viewKey) {
      this.viewKey = viewSignature;
      ctx.clearRect(0, 0, width, height);
      this.ensureCount(width);
      for (const p of this.particles) {
        this.respawnInViewport(p, proj, width, height);
        p.age = Math.floor(Math.random() * MAX_AGE);
      }
    } else {
      this.ensureCount(width);
    }

    // Fade existing trails using destination-in composite (exact cambecc/earth trail look)
    const prevOp = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = 'destination-in';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.94)';
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = prevOp;

    const spec = WIND_LEVELS[grid.level];
    const scale = proj.scale();
    // Angular advance keeps apparent pixel streak speed consistent from whole-globe zoom to regional zoom
    const radScale = (1.55 / Math.max(scale, 80)) / Math.max(spec.streamMax, 5) * 18;

    const buckets: [number, number, number, number][][] = Array.from(
      { length: INTENSITY_BUCKETS },
      () => [],
    );

    const maxSegJump = 18;
    const DEG2RAD = Math.PI / 180;
    const RAD2DEG = 180 / Math.PI;

    for (const p of this.particles) {
      if (p.age >= MAX_AGE) {
        this.respawnInViewport(p, proj, width, height);
      }
      p.age++;

      const scr0 = proj([p.lon, p.lat]);
      const [u, v, mag] = sampleWindLonLat(grid, p.lon, p.lat);
      const cosLat = Math.max(0.18, Math.cos(p.lat * DEG2RAD));
      const dLon = (u / cosLat) * radScale * RAD2DEG;
      const dLat = v * radScale * RAD2DEG;

      let nextLon = p.lon + dLon;
      if (nextLon > 180) nextLon -= 360;
      else if (nextLon < -180) nextLon += 360;
      const nextLat = Math.max(-89.5, Math.min(89.5, p.lat + dLat));

      const scr1 = proj([nextLon, nextLat]);
      p.lon = nextLon;
      p.lat = nextLat;

      if (
        !scr0 ||
        !scr1 ||
        !Number.isFinite(scr0[0]) ||
        !Number.isFinite(scr0[1]) ||
        !Number.isFinite(scr1[0]) ||
        !Number.isFinite(scr1[1])
      ) {
        if (this.frameCounter % 4 === 0 && !scr1) {
          p.age = MAX_AGE;
        }
        continue;
      }

      if (
        scr0[0] < -40 ||
        scr0[0] > width + 40 ||
        scr0[1] < -40 ||
        scr0[1] > height + 40
      ) {
        p.age = MAX_AGE;
        continue;
      }

      if (Math.hypot(scr1[0] - scr0[0], scr1[1] - scr0[1]) > maxSegJump) {
        continue;
      }

      const bIdx = Math.min(
        INTENSITY_BUCKETS - 1,
        Math.floor((Math.min(mag, spec.streamMax) / spec.streamMax) * (INTENSITY_BUCKETS - 1)),
      );
      buckets[bIdx].push([scr0[0], scr0[1], scr1[0], scr1[1]]);
    }

    ctx.save();
    ctx.beginPath();
    d3Geo.geoPath(proj, ctx)({ type: 'Sphere' });
    ctx.clip();

    ctx.lineWidth = 1.05;
    ctx.lineCap = 'round';
    for (let i = 0; i < INTENSITY_BUCKETS; i++) {
      const segs = buckets[i];
      if (segs.length === 0) continue;
      ctx.beginPath();
      ctx.strokeStyle = COLOR_STYLES[i];
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
