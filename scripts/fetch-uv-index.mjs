import fs from 'node:fs';
import path from 'node:path';
import { decodeGrib2Field } from '../src/earth/grib2.ts';

const NOMADS_UV_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/uvi/prod';
const GFS_S3_BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com';
const NX = 360;
const NY = 181;
const CPC_UVI_SCALE = 0.01;
const TOZNE_SCALE = 0.1;

function pad2(n) {
  return String(n).padStart(2, '0');
}

async function fetchWithTimeout(url, init = {}, timeoutMs = 45_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      ...init,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; ComplexLife/1.0)',
        ...(init.headers ?? {}),
      },
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`${url} HTTP ${res.status}`);
    }
    return res;
  } finally {
    clearTimeout(timer);
  }
}

async function fetchLatestCpcNoontime() {
  const now = new Date();
  let lastErr = null;
  for (let dayBack = 0; dayBack <= 4; dayBack++) {
    const d = new Date(now.getTime() - dayBack * 86400 * 1000);
    const ymd = `${d.getUTCFullYear()}${pad2(d.getUTCMonth() + 1)}${pad2(d.getUTCDate())}`;
    const url = `${NOMADS_UV_BASE}/uvi.${ymd}/uv.noontime.t12z.d1.dat`;
    try {
      const res = await fetchWithTimeout(url, {}, 45_000);
      const txt = await res.text();
      const lines = txt.trim().split(/\r?\n/);
      if (lines.length < 1000) {
        throw new Error(`Unexpected small CPC UVI payload from ${url}`);
      }
      const hdrParts = lines[0].trim().split(/\s+/);
      const hdrYmd = hdrParts[0] || ymd;
      const refIso = `${hdrYmd.slice(0, 4)}-${hdrYmd.slice(4, 6)}-${hdrYmd.slice(6, 8)}T12:00:00Z`;
      const raw = new Float32Array(1440 * 721);
      let k = 0;
      for (let li = 1; li < lines.length && k < raw.length; li++) {
        const toks = lines[li].trim().split(/\s+/);
        for (let ti = 0; ti < toks.length && k < raw.length; ti++) {
          const v = parseFloat(toks[ti]);
          raw[k++] = Number.isFinite(v) && v >= 0 ? v : 0;
        }
      }
      if (k !== 1440 * 721) {
        throw new Error(`Unexpected CPC grid count ${k} (expected ${1440 * 721})`);
      }

      // Box-averaged downsample from 0.25 deg (1440x721) onto 1.0 deg (360x181)
      const cpcUvi = new Uint16Array(NX * NY);
      let maxUvi = 0;
      for (let j = 0; j < NY; j++) {
        const rCenter = j * 4;
        for (let i = 0; i < NX; i++) {
          const cCenter = i * 4;
          let sumWm2 = 0;
          let wSum = 0;
          for (let dr = -1; dr <= 1; dr++) {
            const r = Math.max(0, Math.min(720, rCenter + dr));
            for (let dc = -1; dc <= 1; dc++) {
              const c = (cCenter + dc + 1440) % 1440;
              const w = dr === 0 && dc === 0 ? 4 : dr === 0 || dc === 0 ? 2 : 1;
              sumWm2 += raw[r * 1440 + c] * w;
              wSum += w;
            }
          }
          const uvi = (sumWm2 / wSum) / 25.0;
          if (uvi > maxUvi) maxUvi = uvi;
          cpcUvi[j * NX + i] = Math.min(65535, Math.max(0, Math.round(uvi / CPC_UVI_SCALE)));
        }
      }

      return {
        cpcUvi,
        cpcRefTime: refIso,
        maxUvi,
      };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr ?? new Error('Unable to fetch NOAA CPC noontime UVI');
}

async function fetchLatestGfsUvInputs() {
  const nowMs = Date.now();
  const latestReadyMs = nowMs - 4.25 * 3600 * 1000;
  const candidateDate = new Date(latestReadyMs);
  const startHour = Math.floor(candidateDate.getUTCHours() / 6) * 6;
  const baseUtcMs = Date.UTC(
    candidateDate.getUTCFullYear(),
    candidateDate.getUTCMonth(),
    candidateDate.getUTCDate(),
    startHour,
  );

  let gribUrl = null;
  let idxLines = null;
  let cycleLabel = '';

  for (let step = 0; step < 5; step++) {
    const cycMs = baseUtcMs - step * 6 * 3600 * 1000;
    const d = new Date(cycMs);
    const ymd = `${d.getUTCFullYear()}${pad2(d.getUTCMonth() + 1)}${pad2(d.getUTCDate())}`;
    const hh = pad2(d.getUTCHours());
    const url = `${GFS_S3_BASE}/gfs.${ymd}/${hh}/atmos/gfs.t${hh}z.pgrb2.1p00.f000`;
    try {
      const idxRes = await fetchWithTimeout(`${url}.idx`, {}, 20_000);
      const txt = await idxRes.text();
      const lines = txt.trim().split('\n');
      if (lines.length > 50) {
        gribUrl = url;
        idxLines = lines;
        cycleLabel = `${ymd} ${hh}Z+00h`;
        break;
      }
    } catch {
      // Try earlier cycle
    }
  }
  if (!gribUrl || !idxLines) {
    throw new Error('Unable to locate latest GFS f000 index on S3');
  }

  async function fetchSlice(pat) {
    const i = idxLines.findIndex((l) => l.includes(pat));
    if (i < 0) throw new Error(`Missing GFS field ${pat}`);
    const start = idxLines[i].split(':')[1];
    const nextOff = i + 1 < idxLines.length ? Number(idxLines[i + 1].split(':')[1]) - 1 : '';
    const range = nextOff !== '' ? `bytes=${start}-${nextOff}` : `bytes=${start}-`;
    const res = await fetchWithTimeout(gribUrl, { headers: { Range: range } }, 30_000);
    return decodeGrib2Field(await res.arrayBuffer());
  }

  const [tozne, tcdc, hgt] = await Promise.all([
    fetchSlice(':TOZNE:entire atmosphere'),
    fetchSlice(':TCDC:entire atmosphere:'),
    fetchSlice(':HGT:surface:'),
  ]);
  if (tozne.values.length !== NX * NY || tcdc.values.length !== NX * NY || hgt.values.length !== NX * NY) {
    throw new Error(`Unexpected GFS grid dimensions: ${tozne.nx}x${tozne.ny}`);
  }

  const toznePacked = new Uint16Array(NX * NY);
  const tcdcPacked = new Uint8Array(NX * NY);
  const hgtPacked = new Int16Array(NX * NY);
  let minDu = 1e9;
  let maxDu = -1e9;
  for (let i = 0; i < NX * NY; i++) {
    const du = Math.max(50, Math.min(800, tozne.values[i] || 300));
    if (du < minDu) minDu = du;
    if (du > maxDu) maxDu = du;
    toznePacked[i] = Math.round(du / TOZNE_SCALE);
    tcdcPacked[i] = Math.max(0, Math.min(100, Math.round(tcdc.values[i] || 0)));
    hgtPacked[i] = Math.max(-500, Math.min(9000, Math.round(hgt.values[i] || 0)));
  }

  return {
    gfsRefTime: cycleLabel,
    toznePacked,
    tcdcPacked,
    hgtPacked,
    minDu,
    maxDu,
  };
}

function toB64(arr) {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64');
}

async function main() {
  const outPath = path.resolve(process.cwd(), 'public/data/uv-index.json');
  let prev = null;
  try {
    if (fs.existsSync(outPath)) {
      prev = JSON.parse(fs.readFileSync(outPath, 'utf8'));
    }
  } catch {
    prev = null;
  }

  const [cpcRes, gfsRes] = await Promise.allSettled([
    fetchLatestCpcNoontime(),
    fetchLatestGfsUvInputs(),
  ]);

  if (cpcRes.status !== 'fulfilled' && !prev?.cpcNoonB64) {
    throw cpcRes.reason;
  }
  if (gfsRes.status !== 'fulfilled' && !prev?.tozneB64) {
    throw gfsRes.reason;
  }

  const payload = {
    nx: NX,
    ny: NY,
    cpcNoonScale: CPC_UVI_SCALE,
    tozneScale: TOZNE_SCALE,
    cpcSource: 'NOAA CPC / NWS UVI v2',
    gfsSource: 'NOAA GFS Ozone & Clouds',
    cpcRefTime: cpcRes.status === 'fulfilled' ? cpcRes.value.cpcRefTime : prev.cpcRefTime,
    gfsRefTime: gfsRes.status === 'fulfilled' ? gfsRes.value.gfsRefTime : prev.gfsRefTime,
    cpcNoonB64: cpcRes.status === 'fulfilled' ? toB64(cpcRes.value.cpcUvi) : prev.cpcNoonB64,
    tozneB64: gfsRes.status === 'fulfilled' ? toB64(gfsRes.value.toznePacked) : prev.tozneB64,
    tcdcB64: gfsRes.status === 'fulfilled' ? toB64(gfsRes.value.tcdcPacked) : prev.tcdcB64,
    hgtB64: gfsRes.status === 'fulfilled' ? toB64(gfsRes.value.hgtPacked) : prev.hgtB64,
  };

  if (cpcRes.status !== 'fulfilled') {
    console.warn(`CPC UVI fetch warning (kept cached): ${cpcRes.reason?.message ?? cpcRes.reason}`);
  }
  if (gfsRes.status !== 'fulfilled') {
    console.warn(`GFS ozone/cloud fetch warning (kept cached): ${gfsRes.reason?.message ?? gfsRes.reason}`);
  }

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(payload));
  const stats = fs.statSync(outPath);
  const cpcInfo = cpcRes.status === 'fulfilled' ? `maxNoonUVI=${cpcRes.value.maxUvi.toFixed(1)}` : 'cached CPC';
  const gfsInfo =
    gfsRes.status === 'fulfilled'
      ? `O3=${gfsRes.value.minDu.toFixed(0)}..${gfsRes.value.maxDu.toFixed(0)} DU`
      : 'cached GFS';
  console.log(
    `Saved public/data/uv-index.json: cpcRef=${payload.cpcRefTime} (${cpcInfo}), gfsRef=${payload.gfsRefTime} (${gfsInfo}), ${(stats.size / 1024).toFixed(1)} KB`,
  );
}

main().catch((err) => {
  console.error('Failed to update UV index data:', err);
  process.exitCode = 1;
});
