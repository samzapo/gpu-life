import * as d3Geo from 'd3-geo';
import { buildD3RotationMatrixColMajor } from './projections';
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
const MAX_PARTICLES = 3400;
const TRAIL_LEN = 22;
const FADE_BANDS = 4;
const INTENSITY_BUCKETS = 12;
const REPROJECT_BUCKETS = 6;
const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

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

function buildBandIntensityStyles(rgb: [number, number, number]): string[][] {
  const bands: string[][] = [];
  // Band 0 = oldest tail, Band FADE_BANDS-1 = newest head
  const alphas = [0.22, 0.42, 0.66, 0.88];
  for (let band = 0; band < FADE_BANDS; band++) {
    const row: string[] = [];
    for (let i = 0; i < REPROJECT_BUCKETS; i++) {
      const scale = (105 + (150 * i) / (REPROJECT_BUCKETS - 1)) / 255;
      const r = Math.round(rgb[0] * scale);
      const g = Math.round(rgb[1] * scale);
      const b = Math.round(rgb[2] * scale);
      row.push(`rgba(${r}, ${g}, ${b}, ${alphas[band]})`);
    }
    bands.push(row);
  }
  return bands;
}

export class WindStreamRenderer {
  private particles: StreamParticle[] = [];
  private viewKey = '';
  private projId = '';
  private lastWidth = 0;
  private lastHeight = 0;
  private frameCounter = 0;
  private cachedTintKey = '';
  private cachedStyles: string[] = [];
  private cachedBandStyles: string[][] = [];

  // Ring-buffered geographic trail history per particle so camera movement never wipes or resets streamlines.
  private trailX = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailY = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailZ = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailLon = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailLat = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailBucket = new Uint8Array(MAX_PARTICLES * TRAIL_LEN);
  private trailHead = new Uint8Array(MAX_PARTICLES);
  private trailCount = new Uint8Array(MAX_PARTICLES);

  private getStyles(tint: [number, number, number]): string[] {
    const key = `${tint[0]},${tint[1]},${tint[2]}`;
    if (key !== this.cachedTintKey) {
      this.cachedTintKey = key;
      this.cachedStyles = buildIntensityStyles(tint);
      this.cachedBandStyles = buildBandIntensityStyles(tint);
    }
    return this.cachedStyles;
  }

  clear(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.clearRect(0, 0, width, height);
    this.particles.length = 0;
    this.trailHead.fill(0);
    this.trailCount.fill(0);
    this.viewKey = '';
  }

  private pushTrailPoint(
    index: number,
    lonDeg: number,
    latDeg: number,
    bucket: number,
  ): void {
    const head = (this.trailHead[index] + 1) % TRAIL_LEN;
    const slot = index * TRAIL_LEN + head;
    const lonRad = lonDeg * DEG2RAD;
    const latRad = latDeg * DEG2RAD;
    const cosLat = Math.cos(latRad);
    this.trailX[slot] = cosLat * Math.cos(lonRad);
    this.trailY[slot] = cosLat * Math.sin(lonRad);
    this.trailZ[slot] = Math.sin(latRad);
    this.trailLon[slot] = lonDeg;
    this.trailLat[slot] = latDeg;
    this.trailBucket[slot] = bucket;
    this.trailHead[index] = head;
    if (this.trailCount[index] < TRAIL_LEN) {
      this.trailCount[index]++;
    }
  }

  private ensureCount(width: number, masked: boolean): void {
    const base = Math.max(
      1200,
      Math.min(MAX_PARTICLES, Math.round(width * (masked ? 2.5 : 2.2))),
    );
    while (this.particles.length < base) {
      const idx = this.particles.length;
      this.particles.push({
        lon: (Math.random() - 0.5) * 360,
        lat: Math.asin(Math.random() * 2 - 1) * RAD2DEG,
        age: Math.floor(Math.random() * MAX_AGE),
      });
      this.trailHead[idx] = 0;
      this.trailCount[idx] = 0;
    }
    if (this.particles.length > base) {
      this.particles.length = base;
    }
  }

  private respawnInViewport(
    index: number,
    p: StreamParticle,
    proj: d3Geo.GeoProjection,
    width: number,
    height: number,
    sampler: StreamFieldSampler,
  ): void {
    this.trailHead[index] = 0;
    this.trailCount[index] = 0;
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
        const lat = Math.asin(Math.random() * 2 - 1) * RAD2DEG;
        if (sampler.sample(lon, lat)[3] >= 0.45) {
          p.lon = lon;
          p.lat = lat;
          p.age = Math.floor(Math.random() * 14);
          return;
        }
      }
    }
    p.lon = (Math.random() - 0.5) * 360;
    p.lat = Math.asin(Math.random() * 2 - 1) * RAD2DEG;
    p.age = Math.floor(Math.random() * 14);
  }

  private drawStoredTrails(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    proj: d3Geo.GeoProjection,
    isOrthographic: boolean,
    sampler: StreamFieldSampler,
  ): void {
    const scale = proj.scale();
    const [tx, ty] = proj.translate();
    const rot = isOrthographic
      ? buildD3RotationMatrixColMajor(proj.rotate())
      : null;
    const r00 = rot ? rot[0] : 0;
    const r01 = rot ? rot[1] : 0;
    const r02 = rot ? rot[2] : 0;
    const r10 = rot ? rot[4] : 0;
    const r11 = rot ? rot[5] : 0;
    const r12 = rot ? rot[6] : 0;
    const r20 = rot ? rot[8] : 0;
    const r21 = rot ? rot[9] : 0;
    const r22 = rot ? rot[10] : 0;

    const buckets: number[][][] = Array.from({ length: FADE_BANDS }, () =>
      Array.from({ length: REPROJECT_BUCKETS }, () => []),
    );
    const maxSegJumpSq = 56 * 56;
    const margin = 48;

    for (let i = 0; i < this.particles.length; i++) {
      const count = this.trailCount[i];
      if (count < 2) continue;
      const head = this.trailHead[i];
      const base = i * TRAIL_LEN;
      const startAge = count - 1;

      let prevValid = false;
      let prevX = 0;
      let prevY = 0;

      for (let k = startAge; k >= 0; k--) {
        const ringIdx = (head - k + TRAIL_LEN) % TRAIL_LEN;
        const slot = base + ringIdx;
        let sx = 0;
        let sy = 0;
        let valid = false;

        if (isOrthographic) {
          const ux = this.trailX[slot];
          const uy = this.trailY[slot];
          const uz = this.trailZ[slot];
          const vx = r00 * ux + r10 * uy + r20 * uz;
          if (vx >= 0.004) {
            const vy = r01 * ux + r11 * uy + r21 * uz;
            const vz = r02 * ux + r12 * uy + r22 * uz;
            sx = tx + vy * scale;
            sy = ty - vz * scale;
            valid =
              sx >= -margin &&
              sx <= width + margin &&
              sy >= -margin &&
              sy <= height + margin;
          }
        } else {
          const scr = proj([this.trailLon[slot], this.trailLat[slot]]);
          if (scr && Number.isFinite(scr[0]) && Number.isFinite(scr[1])) {
            sx = scr[0];
            sy = scr[1];
            valid =
              sx >= -margin &&
              sx <= width + margin &&
              sy >= -margin &&
              sy <= height + margin;
          }
        }

        if (prevValid && valid) {
          const dx = sx - prevX;
          const dy = sy - prevY;
          if (dx * dx + dy * dy <= maxSegJumpSq) {
            const band = Math.min(
              FADE_BANDS - 1,
              Math.floor(((TRAIL_LEN - 1 - k) / TRAIL_LEN) * FADE_BANDS),
            );
            const rBucket = Math.min(
              REPROJECT_BUCKETS - 1,
              Math.floor(
                (this.trailBucket[slot] / INTENSITY_BUCKETS) * REPROJECT_BUCKETS,
              ),
            );
            buckets[band][rBucket].push(prevX, prevY, sx, sy);
          }
        }
        prevValid = valid;
        prevX = sx;
        prevY = sy;
      }
    }

    ctx.save();
    ctx.beginPath();
    d3Geo.geoPath(proj, ctx)({ type: 'Sphere' });
    ctx.clip();
    ctx.lineWidth = sampler.lineWidth;
    ctx.lineCap = 'round';

    for (let band = 0; band < FADE_BANDS; band++) {
      const bandRow = buckets[band];
      const styles = this.cachedBandStyles[band];
      for (let b = 0; b < REPROJECT_BUCKETS; b++) {
        const segs = bandRow[b];
        if (segs.length === 0) continue;
        ctx.beginPath();
        ctx.strokeStyle = styles[b];
        for (let k = 0; k < segs.length; k += 4) {
          ctx.moveTo(segs[k], segs[k + 1]);
          ctx.lineTo(segs[k + 2], segs[k + 3]);
        }
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  stepAndDraw(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    proj: d3Geo.GeoProjection,
    gridOrSampler: WindGrid | StreamFieldSampler,
    viewSignature: string,
    active: boolean,
    cameraMoving = false,
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
    const colon = viewSignature.indexOf(':');
    const nextProjId = colon > 0 ? viewSignature.slice(0, colon) : '';
    const isOrthographic = nextProjId === 'orthographic';

    this.ensureCount(width, sampler.masked);

    const isInitialOrProjSwitch =
      !this.viewKey ||
      (nextProjId !== '' && this.projId !== '' && nextProjId !== this.projId);
    const viewChanged =
      viewSignature !== this.viewKey ||
      width !== this.lastWidth ||
      height !== this.lastHeight;

    if (isInitialOrProjSwitch) {
      ctx.clearRect(0, 0, width, height);
      for (let i = 0; i < this.particles.length; i++) {
        const particle = this.particles[i];
        this.respawnInViewport(i, particle, proj, width, height, sampler);
        particle.age = Math.floor(Math.random() * MAX_AGE);
      }
    }

    this.viewKey = viewSignature;
    this.projId = nextProjId;
    this.lastWidth = width;
    this.lastHeight = height;

    const fullRedraw = viewChanged && !isInitialOrProjSwitch;
    if (!fullRedraw) {
      // Fade existing trails using destination-in composite when camera is stationary.
      const prevOp = ctx.globalCompositeOperation;
      ctx.globalCompositeOperation = 'destination-in';
      ctx.fillStyle = `rgba(0, 0, 0, ${sampler.fadeAlpha})`;
      ctx.fillRect(0, 0, width, height);
      ctx.globalCompositeOperation = prevOp;
    }

    const scale = proj.scale();
    // Target ~4.2 px/frame at streamMax across all zoom levels, keeping fast gales, jets & currents crisp
    const radScale = 4.2 / (Math.max(scale, 80) * Math.max(sampler.streamMax, 0.25));

    const buckets: [number, number, number, number][][] = fullRedraw
      ? []
      : Array.from({ length: INTENSITY_BUCKETS }, () => []);

    const maxSegJump = 28;
    const center = cameraMoving ? proj.invert?.(proj.translate()) : null;

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      if (center && d3Geo.geoDistance(center, [p.lon, p.lat]) > Math.PI / 2) {
        p.age = MAX_AGE;
      }
      if (p.age >= MAX_AGE) {
        this.respawnInViewport(i, p, proj, width, height, sampler);
      }
      p.age++;

      let curLon = p.lon;
      let curLat = p.lat;
      let scr0 = fullRedraw ? null : proj([curLon, curLat]);
      const [u0, v0, mag0, mask0] = sampler.sample(curLon, curLat);
      if (sampler.masked && mask0 < 0.42) {
        p.age = MAX_AGE;
        continue;
      }

      const bIdx0 = Math.min(
        INTENSITY_BUCKETS - 1,
        Math.floor(
          (Math.min(mag0, sampler.streamMax) / sampler.streamMax) *
            (INTENSITY_BUCKETS - 1),
        ),
      );
      if (this.trailCount[i] === 0) {
        this.pushTrailPoint(i, curLon, curLat, bIdx0);
      }

      const nSub = Math.min(4, Math.max(1, Math.ceil(mag0 / Math.max(sampler.streamMax, 0.25))));
      const subRadScale = radScale / nSub;
      let lastBucket = bIdx0;

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

        curLon = nextLon;
        curLat = nextLat;

        const bIdx = Math.min(
          INTENSITY_BUCKETS - 1,
          Math.floor(
            (Math.min(mag, sampler.streamMax) / sampler.streamMax) *
              (INTENSITY_BUCKETS - 1),
          ),
        );
        lastBucket = bIdx;

        if (!fullRedraw) {
          const scr1 = proj([nextLon, nextLat]);
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
              buckets[bIdx].push([scr0[0], scr0[1], scr1[0], scr1[1]]);
            }
          }
          scr0 = scr1;
        }
      }

      p.lon = curLon;
      p.lat = curLat;
      if (p.age < MAX_AGE) {
        this.pushTrailPoint(i, curLon, curLat, lastBucket);
      }

      if (fullRedraw) {
        scr0 = proj([curLon, curLat]);
      }

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

    if (fullRedraw) {
      ctx.clearRect(0, 0, width, height);
      this.drawStoredTrails(ctx, width, height, proj, isOrthographic, sampler);
    } else {
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
    }
    this.frameCounter++;
  }
}
