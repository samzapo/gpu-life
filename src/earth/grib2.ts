/**
 * Minimal WMO FM-92 GRIB2 decoder for NOAA GFS global wind fields (Templates 5.0, 5.2, 5.3).
 */

export interface DecodedGrib2Grid {
  nx: number;
  ny: number;
  values: Float32Array;
}

class BitReader {
  private readonly buf: Uint8Array;
  private bitPos = 0;

  constructor(buf: Uint8Array) {
    this.buf = buf;
  }

  readU(n: number): number {
    if (n <= 0) return 0;
    let val = 0;
    let rem = n;
    while (rem > 0) {
      const byteIdx = this.bitPos >> 3;
      const bitInByte = this.bitPos & 7;
      const avail = 8 - bitInByte;
      const take = rem < avail ? rem : avail;
      const shift = avail - take;
      const mask = (1 << take) - 1;
      const bits = (this.buf[byteIdx] >> shift) & mask;
      val = val * (1 << take) + bits;
      this.bitPos += take;
      rem -= take;
    }
    return val;
  }

  readSignMag(n: number): number {
    if (n <= 0) return 0;
    const sign = this.readU(1);
    const mag = this.readU(n - 1);
    return sign ? -mag : mag;
  }

  align(): void {
    const rem = this.bitPos & 7;
    if (rem !== 0) {
      this.bitPos += 8 - rem;
    }
  }
}

function readGrib2Signed16(view: DataView, offset: number): number {
  const raw = view.getUint16(offset, false);
  const mag = raw & 0x7fff;
  return (raw & 0x8000) !== 0 ? -mag : mag;
}

export function decodeGrib2Field(buffer: ArrayBuffer): DecodedGrib2Grid {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  // Skip 16-byte Indicator Section 0
  let pos = 16;
  const sections = new Map<number, { offset: number; length: number }>();

  while (pos + 5 <= bytes.length) {
    if (
      bytes[pos] === 0x37 &&
      bytes[pos + 1] === 0x37 &&
      bytes[pos + 2] === 0x37 &&
      bytes[pos + 3] === 0x37
    ) {
      break;
    }
    const len = view.getUint32(pos, false);
    const num = bytes[pos + 4];
    if (len < 5 || pos + len > bytes.length) break;
    sections.set(num, { offset: pos, length: len });
    pos += len;
  }

  const s3 = sections.get(3);
  const s5 = sections.get(5);
  const s6 = sections.get(6);
  const s7 = sections.get(7);
  if (!s3 || !s5 || !s7) {
    throw new Error('Incomplete GRIB2 sections');
  }

  const nx = view.getUint32(s3.offset + 30, false);
  const ny = view.getUint32(s3.offset + 34, false);
  const scanMode = bytes[s3.offset + 71];

  const npts = view.getUint32(s5.offset + 5, false);
  const template = view.getUint16(s5.offset + 9, false);
  const refVal = view.getFloat32(s5.offset + 11, false);
  const binScale = readGrib2Signed16(view, s5.offset + 15);
  const decScale = readGrib2Signed16(view, s5.offset + 17);
  const nbits = bytes[s5.offset + 19];

  const bPow = Math.pow(2, binScale);
  const dPow = Math.pow(10, -decScale);

  const payload = bytes.subarray(s7.offset + 5, s7.offset + s7.length);
  const br = new BitReader(payload);
  const rawInts = new Float64Array(npts);

  if (template === 0) {
    for (let i = 0; i < npts; i++) {
      rawInts[i] = br.readU(nbits);
    }
  } else if (template === 2 || template === 3) {
    const ngroups = view.getUint32(s5.offset + 31, false);
    const refGroupWidth = bytes[s5.offset + 35];
    const nbitGroupWidth = bytes[s5.offset + 36];
    const refGroupLen = view.getUint32(s5.offset + 37, false);
    const lenInc = bytes[s5.offset + 41];
    const trueLastLen = view.getUint32(s5.offset + 42, false);
    const nbitGroupLen = bytes[s5.offset + 46];
    const sdOrder = template === 3 ? bytes[s5.offset + 47] : 0;
    const sdOctets = template === 3 ? bytes[s5.offset + 48] : 0;

    let h1 = 0;
    let h2 = 0;
    let gmin = 0;
    if (template === 3 && sdOrder > 0) {
      const sdBits = sdOctets * 8;
      h1 = br.readSignMag(sdBits);
      if (sdOrder === 2) {
        h2 = br.readSignMag(sdBits);
      }
      gmin = br.readSignMag(sdBits);
    }

    const gRefs = new Float64Array(ngroups);
    for (let gi = 0; gi < ngroups; gi++) {
      gRefs[gi] = br.readU(nbits);
    }
    br.align();

    const gWidths = new Uint16Array(ngroups);
    for (let gi = 0; gi < ngroups; gi++) {
      gWidths[gi] = refGroupWidth + br.readU(nbitGroupWidth);
    }
    br.align();

    const gLens = new Uint32Array(ngroups);
    for (let gi = 0; gi < ngroups; gi++) {
      gLens[gi] = refGroupLen + br.readU(nbitGroupLen) * lenInc;
    }
    if (ngroups > 0) {
      gLens[ngroups - 1] = trueLastLen;
    }
    br.align();

    let idx = 0;
    for (let gi = 0; gi < ngroups; gi++) {
      const ref = gRefs[gi];
      const w = gWidths[gi];
      const len = gLens[gi];
      for (let k = 0; k < len && idx < npts; k++) {
        rawInts[idx++] = ref + br.readU(w);
      }
    }

    if (template === 3 && sdOrder === 2 && idx >= 2) {
      rawInts[0] = h1;
      rawInts[1] = h2;
      for (let i = 2; i < idx; i++) {
        rawInts[i] += gmin + 2 * rawInts[i - 1] - rawInts[i - 2];
      }
    } else if (template === 3 && sdOrder === 1 && idx >= 1) {
      rawInts[0] = h1;
      for (let i = 1; i < idx; i++) {
        rawInts[i] += gmin + rawInts[i - 1];
      }
    }
  } else {
    throw new Error(`Unsupported GRIB2 Data Representation Template 5.${template}`);
  }

  const total = nx * ny;
  const values = new Float32Array(total);
  const hasBitmap = s6 !== undefined && bytes[s6.offset + 5] === 0;
  if (hasBitmap && s6) {
    const bm = bytes.subarray(s6.offset + 6, s6.offset + s6.length);
    let valIdx = 0;
    for (let i = 0; i < total; i++) {
      const present = (bm[i >> 3] >> (7 - (i & 7))) & 1;
      values[i] = present
        ? (refVal + rawInts[valIdx++] * bPow) * dPow
        : 0;
    }
  } else {
    for (let i = 0; i < Math.min(total, npts); i++) {
      values[i] = (refVal + rawInts[i] * bPow) * dPow;
    }
  }

  // Normalize so row 0 = +90N .. row ny-1 = -90S, col 0 = 0E .. col nx-1 = 359E
  const southToNorth = (scanMode & 0x40) !== 0;
  const eastToWest = (scanMode & 0x80) !== 0;
  if (southToNorth || eastToWest) {
    const normalized = new Float32Array(total);
    for (let row = 0; row < ny; row++) {
      const srcRow = southToNorth ? ny - 1 - row : row;
      for (let col = 0; col < nx; col++) {
        const srcCol = eastToWest ? nx - 1 - col : col;
        normalized[row * nx + col] = values[srcRow * nx + srcCol];
      }
    }
    return { nx, ny, values: normalized };
  }

  return { nx, ny, values };
}
