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
const MAX_PARTICLES = 2400;
const TRAIL_LEN = 18;
// Subsampled offsets from oldest (17) to head (0) for smooth, low-overhead polyline redraws
const TAIL_OFFSETS = [17, 13, 9, 6] as const;
const HEAD_OFFSETS = [6, 3, 1, 0] as const;
const FADE_BANDS = 2;
const INTENSITY_BUCKETS = 12;
const REPROJECT_BUCKETS = 4;
const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const BREAK_SENTINEL = -1e9;

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
  const alphas = [0.36, 0.86];
  return alphas.map((alpha) => {
    const row: string[] = [];
    for (let i = 0; i < REPROJECT_BUCKETS; i++) {
      const scale = (115 + (140 * i) / (REPROJECT_BUCKETS - 1)) / 255;
      const r = Math.round(rgb[0] * scale);
      const g = Math.round(rgb[1] * scale);
      const b = Math.round(rgb[2] * scale);
      row.push(`rgba(${r}, ${g}, ${b}, ${alpha})`);
    }
    return row;
  });
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

  private trailX = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailY = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailZ = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailLon = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailLat = new Float32Array(MAX_PARTICLES * TRAIL_LEN);
  private trailBucket = new Uint8Array(MAX_PARTICLES * TRAIL_LEN);
  private trailHead = new Uint8Array(MAX_PARTICLES);
  private trailCount = new Uint8Array(MAX_PARTICLES);

  private polyPools: number[][][] = Array.from({ length: FADE_BANDS }, () =>
    Array.from({ length: REPROJECT_BUCKETS }, () => []),
  );
  private segPools: number[][] = Array.from({ length: INTENSITY_BUCKETS }, () => []);

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
      1000,
      Math.min(MAX_PARTICLES, Math.round(width * (masked ? 1.8 : 1.6))),
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
      const maxTries = sampler.masked ? 12 : 5;
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

  private emitBandPolyline(
    pool: number[],
    offsets: readonly number[],
    base: number,
    head: number,
    count: number,
    width: number,
    height: number,
    proj: d3Geo.GeoProjection,
    isOrthographic: boolean,
    scale: number,
    tx: number,
    ty: number,
    r00: number,
    r01: number,
    r02: number,
    r10: number,
    r11: number,
    r12: number,
    r20: number,
    r21: number,
    r22: number,
    maxJumpSq: number,
  ): void {
    const margin = 48;
    let started = false;
    let prevX = 0;
    let prevY = 0;

    for (let o = 0; o < offsets.length; o++) {
      const k = offsets[o];
      if (k >= count) continue;
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

      if (!valid) {
        started = false;
        continue;
      }

      if (!started) {
        pool.push(BREAK_SENTINEL, sx, sy);
        started = true;
      } else {
        const dx = sx - prevX;
        const dy = sy - prevY;
        if (dx * dx + dy * dy <= maxJumpSq) {
          pool.push(sx, sy);
        } else {
          pool.push(BREAK_SENTINEL, sx, sy);
        }
      }
      prevX = sx;
      prevY = sy;
    }
  }

  private drawStoredTrails(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    proj: d3Geo.GeoProjection,
    isOrthographic: boolean,
    sampler: StreamFieldSampler,
    scale: number,
    tx: number,
    ty: number,
    r00: number,
    r01: number,
    r02: number,
    r10: number,
    r11: number,
    r12: number,
    r20: number,
    r21: number,
    r22: number,
  ): void {
    for (let band = 0; band < FADE_BANDS; band++) {
      for (let b = 0; b < REPROJECT_BUCKETS; b++) {
        this.polyPools[band][b].length = 0;
      }
    }
    const maxJumpSq = 96 * 96;

    for (let i = 0; i < this.particles.length; i++) {
      const count = this.trailCount[i];
      if (count < 2) continue;
      const head = this.trailHead[i];
      const base = i * TRAIL_LEN;
      const headBucket = this.trailBucket[base + head];
      const rBucket = Math.min(
        REPROJECT_BUCKETS - 1,
        Math.floor((headBucket / INTENSITY_BUCKETS) * REPROJECT_BUCKETS),
      );

      if (count > 7) {
        this.emitBandPolyline(
          this.polyPools[0][rBucket],
          TAIL_OFFSETS,
          base,
          head,
          count,
          width,
          height,
          proj,
          isOrthographic,
          scale,
          tx,
          ty,
          r00,
          r01,
          r02,
          r10,
          r11,
          r12,
          r20,
          r21,
          r22,
          maxJumpSq,
        );
      }
      this.emitBandPolyline(
        this.polyPools[1][rBucket],
        HEAD_OFFSETS,
        base,
        head,
        count,
        width,
        height,
        proj,
        isOrthographic,
        scale,
        tx,
        ty,
        r00,
        r01,
        r02,
        r10,
        r11,
        r12,
        r20,
        r21,
        r22,
        maxJumpSq,
      );
    }

    ctx.save();
    ctx.beginPath();
    if (isOrthographic) {
      ctx.arc(tx, ty, scale + 1, 0, Math.PI * 2);
    } else {
      d3Geo.geoPath(proj, ctx)({ type: 'Sphere' });
    }
    ctx.clip();
    ctx.lineWidth = sampler.lineWidth;
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'bevel';

    for (let band = 0; band < FADE_BANDS; band++) {
      const styles = this.cachedBandStyles[band];
      const bandRow = this.polyPools[band];
      for (let b = 0; b < REPROJECT_BUCKETS; b++) {
        const pts = bandRow[b];
        if (pts.length === 0) continue;
        ctx.beginPath();
        ctx.strokeStyle = styles[b];
        let k = 0;
        while (k < pts.length) {
          if (pts[k] === BREAK_SENTINEL) {
            ctx.moveTo(pts[k + 1], pts[k + 2]);
            k += 3;
          } else {
            ctx.lineTo(pts[k], pts[k + 1]);
            k += 2;
          }
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
      const prevOp = ctx.globalCompositeOperation;
      ctx.globalCompositeOperation = 'destination-in';
      ctx.fillStyle = `rgba(0, 0, 0, ${sampler.fadeAlpha})`;
      ctx.fillRect(0, 0, width, height);
      ctx.globalCompositeOperation = prevOp;
      for (let b = 0; b < INTENSITY_BUCKETS; b++) {
        this.segPools[b].length = 0;
      }
    }

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

    // Target ~4.2 px/frame at streamMax across all zoom levels, keeping fast gales, jets & currents crisp
    const radScale = 4.2 / (Math.max(scale, 80) * Math.max(sampler.streamMax, 0.25));
    const maxSegJumpSq = 28 * 28;
    const center =
      !isOrthographic && cameraMoving ? proj.invert?.([tx, ty]) : null;

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
      let scr0x = 0;
      let scr0y = 0;
      let scr0Ok = false;

      if (!fullRedraw) {
        if (isOrthographic) {
          const lonRad = curLon * DEG2RAD;
          const latRad = curLat * DEG2RAD;
          const cosLat = Math.cos(latRad);
          const ux = cosLat * Math.cos(lonRad);
          const uy = cosLat * Math.sin(lonRad);
          const uz = Math.sin(latRad);
          const vx = r00 * ux + r10 * uy + r20 * uz;
          if (vx >= 0.004) {
            scr0x = tx + (r01 * ux + r11 * uy + r21 * uz) * scale;
            scr0y = ty - (r02 * ux + r12 * uy + r22 * uz) * scale;
            scr0Ok = true;
          }
        } else {
          const scr0 = proj([curLon, curLat]);
          if (scr0 && Number.isFinite(scr0[0]) && Number.isFinite(scr0[1])) {
            scr0x = scr0[0];
            scr0y = scr0[1];
            scr0Ok = true;
          }
        }
      }

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
          let scr1x = 0;
          let scr1y = 0;
          let scr1Ok = false;
          if (isOrthographic) {
            const lonRad = nextLon * DEG2RAD;
            const latRad = nextLat * DEG2RAD;
            const cLat = Math.cos(latRad);
            const ux = cLat * Math.cos(lonRad);
            const uy = cLat * Math.sin(lonRad);
            const uz = Math.sin(latRad);
            const vx = r00 * ux + r10 * uy + r20 * uz;
            if (vx >= 0.004) {
              scr1x = tx + (r01 * ux + r11 * uy + r21 * uz) * scale;
              scr1y = ty - (r02 * ux + r12 * uy + r22 * uz) * scale;
              scr1Ok = true;
            }
          } else {
            const scr1 = proj([nextLon, nextLat]);
            if (scr1 && Number.isFinite(scr1[0]) && Number.isFinite(scr1[1])) {
              scr1x = scr1[0];
              scr1y = scr1[1];
              scr1Ok = true;
            }
          }
          if (scr0Ok && scr1Ok) {
            const dx = scr1x - scr0x;
            const dy = scr1y - scr0y;
            if (dx * dx + dy * dy <= maxSegJumpSq) {
              this.segPools[bIdx].push(scr0x, scr0y, scr1x, scr1y);
            }
          }
          scr0x = scr1x;
          scr0y = scr1y;
          scr0Ok = scr1Ok;
        }
      }

      p.lon = curLon;
      p.lat = curLat;
      if (p.age < MAX_AGE) {
        this.pushTrailPoint(i, curLon, curLat, lastBucket);
      }

      if (fullRedraw) {
        if (isOrthographic) {
          const slot = i * TRAIL_LEN + this.trailHead[i];
          const ux = this.trailX[slot];
          const uy = this.trailY[slot];
          const uz = this.trailZ[slot];
          const vx = r00 * ux + r10 * uy + r20 * uz;
          if (vx >= 0.004) {
            scr0x = tx + (r01 * ux + r11 * uy + r21 * uz) * scale;
            scr0y = ty - (r02 * ux + r12 * uy + r22 * uz) * scale;
            scr0Ok = true;
          }
        } else {
          const scr = proj([curLon, curLat]);
          if (scr && Number.isFinite(scr[0]) && Number.isFinite(scr[1])) {
            scr0x = scr[0];
            scr0y = scr[1];
            scr0Ok = true;
          }
        }
      }

      if (
        !scr0Ok ||
        scr0x < -40 ||
        scr0x > width + 40 ||
        scr0y < -40 ||
        scr0y > height + 40
      ) {
        if (scr0Ok || this.frameCounter % 4 === 0) {
          p.age = MAX_AGE;
        }
      }
    }

    if (fullRedraw) {
      ctx.clearRect(0, 0, width, height);
      this.drawStoredTrails(
        ctx,
        width,
        height,
        proj,
        isOrthographic,
        sampler,
        scale,
        tx,
        ty,
        r00,
        r01,
        r02,
        r10,
        r11,
        r12,
        r20,
        r21,
        r22,
      );
    } else {
      ctx.save();
      ctx.beginPath();
      if (isOrthographic) {
        ctx.arc(tx, ty, scale + 1, 0, Math.PI * 2);
      } else {
        d3Geo.geoPath(proj, ctx)({ type: 'Sphere' });
      }
      ctx.clip();

      ctx.lineWidth = sampler.lineWidth;
      ctx.lineCap = 'round';
      for (let i = 0; i < INTENSITY_BUCKETS; i++) {
        const segs = this.segPools[i];
        if (segs.length === 0) continue;
        ctx.beginPath();
        ctx.strokeStyle = styles[i];
        for (let k = 0; k < segs.length; k += 4) {
          ctx.moveTo(segs[k], segs[k + 1]);
          ctx.lineTo(segs[k + 2], segs[k + 3]);
        }
        ctx.stroke();
      }
      ctx.restore();
    }
    this.frameCounter++;
  }
}
