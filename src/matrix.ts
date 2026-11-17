export const THETA_SAMPLES = 64;

export type ThetaFunction = (theta: number) => number;
export type MatrixGrid = ThetaFunction[][];

export function generateRandomSeed(): string {
  return Math.floor(Math.random() * 1_000_000_000).toString();
}

export function createSeededRng(seed: string | number): () => number {
  const str = String(seed).trim();
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  let state = (h ^ (h >>> 16)) >>> 0;

  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeRandomSinusoidFunction(
  rng: () => number = Math.random,
): ThetaFunction {
  const a = 0.5 * rng();
  const b = Math.floor(rng() * 2) + 1;
  const c = rng() - 0.5;
  const d = rng() * 2.0 * Math.PI;
  return (theta: number) => a * Math.sin(b * theta + d) + c;
}

export function makeRandomMatrix(
  colourAmt: number,
  seed?: string | number,
): MatrixGrid {
  const rng = seed !== undefined ? createSeededRng(seed) : Math.random;
  const rows: MatrixGrid = [];
  for (let c1 = 0; c1 < colourAmt; c1++) {
    const row: ThetaFunction[] = [];
    for (let c2 = 0; c2 < colourAmt; c2++) {
      row.push(makeRandomSinusoidFunction(rng));
    }
    rows.push(row);
  }
  return rows;
}

export function sampleMatrix(
  matrix: MatrixGrid,
  colourAmt: number,
  samples: number = THETA_SAMPLES,
): Float32Array<ArrayBuffer> {
  const data = new Float32Array(colourAmt * colourAmt * samples);
  const step = (Math.PI * 2) / samples;
  for (let c1 = 0; c1 < colourAmt; c1++) {
    for (let c2 = 0; c2 < colourAmt; c2++) {
      const fn = matrix[c1][c2];
      const base = (c1 * colourAmt + c2) * samples;
      for (let k = 0; k < samples; k++) {
        data[base + k] = fn(k * step);
      }
    }
  }
  return data;
}
