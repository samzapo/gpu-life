import fs from 'node:fs';
import path from 'node:path';

const ERDDAP_URL =
  'https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDNRTcurrentsDaily.csv0' +
  '?u_current%5B(last)%5D%5B(-89.875):2:(89.875)%5D%5B(-179.875):2:(179.875)%5D' +
  ',v_current%5B(last)%5D%5B(-89.875):2:(89.875)%5D%5B(-179.875):2:(179.875)%5D';

const SST_ERDDAP_URL =
  'https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDsstDNDaily.csv0' +
  '?analysed_sst%5B(last)%5D%5B(-89.975):20:(89.975)%5D%5B(-179.975):20:(179.975)%5D';

const NX = 360;
const NY = 181;
const SCALE = 0.0005; // 0.5 mm/s precision; Int16 range [-16.38, +16.38] m/s, -32768 sentinel
const SST_SCALE = 0.01; // 0.01 deg C precision; Int16 range [-327.67, +327.67] C, -32768 sentinel
const SENTINEL = -32768;

async function fetchTextWithTimeout(url, label, timeoutMs = 45_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) {
      throw new Error(`${label} HTTP ${res.status}`);
    }
    return await res.text();
  } finally {
    clearTimeout(timer);
  }
}

function packSstCsv(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 10000) {
    throw new Error(`Unexpected small SST ERDDAP payload: ${lines.length} lines`);
  }
  const sstRefTime = lines[0].split(',')[0].trim();
  const sumT = new Float64Array(NX * NY);
  const cntT = new Uint16Array(NX * NY);

  for (let li = 0; li < lines.length; li++) {
    const parts = lines[li].split(',');
    if (parts.length < 4) continue;
    const tStr = parts[3];
    if (!tStr || tStr === 'NaN') continue;
    const tC = Number(tStr);
    const lat = Number(parts[1]);
    const lon = Number(parts[2]);
    if (!Number.isFinite(tC) || !Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const lon360 = ((lon % 360) + 360) % 360;
    const iBase = Math.round(lon360) % NX;
    const jBase = Math.max(0, Math.min(NY - 1, Math.round(90 - lat)));
    const idx = jBase * NX + iBase;
    sumT[idx] += Math.max(-5, Math.min(45, tC));
    cntT[idx]++;
  }

  const packed = new Int16Array(NX * NY);
  let validCells = 0;
  let minC = 1e9;
  let maxC = -1e9;
  for (let idx = 0; idx < NX * NY; idx++) {
    const c = cntT[idx];
    if (c === 0) {
      packed[idx] = SENTINEL;
      continue;
    }
    const meanC = sumT[idx] / c;
    if (meanC < minC) minC = meanC;
    if (meanC > maxC) maxC = meanC;
    validCells++;
    packed[idx] = Math.max(-32767, Math.min(32767, Math.round(meanC / SST_SCALE)));
  }

  const outBytes = Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength);
  return {
    sstScale: SST_SCALE,
    sstSentinel: SENTINEL,
    sstRefTime,
    sstSource: 'NOAA Geo-Polar Blended SST',
    sstB64: outBytes.toString('base64'),
    validCells,
    minC,
    maxC,
  };
}

async function main() {
  const outPath = path.resolve(process.cwd(), 'public/data/ocean-currents.json');
  let prevJson = null;
  try {
    if (fs.existsSync(outPath)) {
      prevJson = JSON.parse(fs.readFileSync(outPath, 'utf8'));
    }
  } catch {
    prevJson = null;
  }

  const [curResult, sstResult] = await Promise.allSettled([
    fetchTextWithTimeout(ERDDAP_URL, 'CoastWatch Currents ERDDAP'),
    fetchTextWithTimeout(SST_ERDDAP_URL, 'CoastWatch Blended SST ERDDAP'),
  ]);
  if (curResult.status !== 'fulfilled') {
    throw curResult.reason;
  }
  const text = curResult.value;

  const lines = text.trim().split('\n');
  if (lines.length < 10000) {
    throw new Error(`Unexpected small ERDDAP payload: ${lines.length} lines`);
  }

  const refTime = lines[0].split(',')[0].trim();

  // Box-accumulate 0.5-deg CoastWatch points into 360x181 (1-deg lon 0..359E, lat +90..-90)
  const sumU = new Float64Array(NX * NY);
  const sumV = new Float64Array(NX * NY);
  const cnt = new Uint16Array(NX * NY);

  for (let li = 0; li < lines.length; li++) {
    const parts = lines[li].split(',');
    if (parts.length < 5) continue;
    const uStr = parts[3];
    const vStr = parts[4];
    if (!uStr || !vStr || uStr === 'NaN' || vStr === 'NaN') continue;
    const u = Number(uStr);
    const v = Number(vStr);
    const lat = Number(parts[1]);
    const lon = Number(parts[2]);
    if (!Number.isFinite(u) || !Number.isFinite(v) || !Number.isFinite(lat) || !Number.isFinite(lon)) {
      continue;
    }

    const lon360 = ((lon % 360) + 360) % 360;
    const iBase = Math.round(lon360) % NX;
    const jBase = Math.max(0, Math.min(NY - 1, Math.round(90 - lat)));

    // Accumulate directly onto nearest 1-deg cell (1..4 half-degree observations per ocean cell)
    const idx = jBase * NX + iBase;
    sumU[idx] += Math.max(-5, Math.min(5, u));
    sumV[idx] += Math.max(-5, Math.min(5, v));
    cnt[idx]++;
  }

  const packed = new Int16Array(NX * NY * 2);
  let validCells = 0;
  let maxSpd = 0;

  for (let idx = 0; idx < NX * NY; idx++) {
    const c = cnt[idx];
    if (c === 0) {
      packed[idx * 2] = SENTINEL;
      packed[idx * 2 + 1] = SENTINEL;
      continue;
    }
    const uMean = sumU[idx] / c;
    const vMean = sumV[idx] / c;
    const spd = Math.hypot(uMean, vMean);
    if (spd > maxSpd) maxSpd = spd;
    validCells++;
    packed[idx * 2] = Math.max(-32767, Math.min(32767, Math.round(uMean / SCALE)));
    packed[idx * 2 + 1] = Math.max(-32767, Math.min(32767, Math.round(vMean / SCALE)));
  }

  const outBytes = Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength);
  let sstPayload = {};
  let sstSummary = 'preserved previous SST';
  if (sstResult.status === 'fulfilled') {
    const sstInfo = packSstCsv(sstResult.value);
    sstPayload = {
      sstScale: sstInfo.sstScale,
      sstSentinel: sstInfo.sstSentinel,
      sstRefTime: sstInfo.sstRefTime,
      sstSource: sstInfo.sstSource,
      sstB64: sstInfo.sstB64,
    };
    sstSummary = `${sstInfo.validCells}/${NX * NY} SST cells (${sstInfo.minC.toFixed(1)}..${sstInfo.maxC.toFixed(1)} C, ref=${sstInfo.sstRefTime})`;
  } else {
    console.warn('Warning: SST fetch failed, falling back if cached:', sstResult.reason);
    if (prevJson && typeof prevJson.sstB64 === 'string') {
      sstPayload = {
        sstScale: prevJson.sstScale,
        sstSentinel: prevJson.sstSentinel,
        sstRefTime: prevJson.sstRefTime,
        sstSource: prevJson.sstSource,
        sstB64: prevJson.sstB64,
      };
    }
  }

  const payload = {
    nx: NX,
    ny: NY,
    scale: SCALE,
    sentinel: SENTINEL,
    refTime,
    source: 'NOAA CoastWatch NRT',
    uvB64: outBytes.toString('base64'),
    ...sstPayload,
  };

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(payload));
  console.log(
    `Wrote ${outPath}: ${validCells}/${NX * NY} ocean cells, refTime=${refTime}, maxSpd=${maxSpd.toFixed(3)} m/s, ${sstSummary}, size=${(fs.statSync(outPath).size / 1024).toFixed(1)} KB`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
