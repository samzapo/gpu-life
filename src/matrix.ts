export const THETA_SAMPLES = 64;

export type ThetaFunction = (theta: number) => number;
export type MatrixGrid = ThetaFunction[][];

export interface SinusoidParams {
  a: number;
  b: number;
  c: number;
  d: number;
}

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

export function makeSinusoidFunction(params: SinusoidParams): ThetaFunction {
  const { a, b, c, d } = params;
  return (theta: number) => a * Math.sin(b * theta + d) + c;
}

export function makeRandomSinusoidFunction(
  rng: () => number = Math.random,
): ThetaFunction {
  const a = 0.5 * rng();
  const b = Math.floor(rng() * 2) + 1;
  const c = rng() - 0.5;
  const d = rng() * 2.0 * Math.PI;
  return makeSinusoidFunction({ a, b, c, d });
}

function makeMatrix2D<T>(rows: number, cols: number, init: () => T): T[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, init),
  );
}

/**
 * Generates an ordered, living ecosystem rule set across `colourAmt` colours.
 * Groups hues into K chemical families with gauge-consistent base phases
 * (avoiding spin-glass frustration), dispersive/ring-forming self-terms,
 * chiral predator-prey cycles (driving sustained motion), and selective
 * symmetric/asymmetric cross-bridges (forming membranes, nuclei, and chains).
 */
export function makeRandomMatrix(
  colourAmt: number,
  seed?: string | number,
): MatrixGrid {
  const rng = seed !== undefined ? createSeededRng(seed) : Math.random;
  const kChoices = [6, 8, 10, 12];
  const K = Math.min(
    kChoices[Math.floor(rng() * kChoices.length)],
    Math.max(1, colourAmt),
  );
  const cycles = rng() < 0.35 ? 2 : 1;
  const chiralDir = rng() < 0.5 ? 1 : -1;
  const lagMag = 0.42 + 0.3 * rng();
  const anchorProb = 0.18 + 0.2 * rng();

  const phi = Array.from({ length: K }, () => rng() * Math.PI * 2);
  const bFam = Array.from({ length: K }, () => Math.floor(rng() * 3) + 1);
  const isAnchor = Array.from(
    { length: K },
    (_, i) => i % 3 === 0 && rng() < anchorProb + 0.25,
  );

  const famA = makeMatrix2D(K, K, () => 0);
  const famB = makeMatrix2D(K, K, () => 1);
  const famC = makeMatrix2D(K, K, () => 0);
  const famD = makeMatrix2D(K, K, () => 0);

  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      famC[i][j] = -0.24 - 0.14 * rng();
      famA[i][j] = 0.35 + 0.25 * rng();
      famB[i][j] = bFam[i];
      famD[i][j] = Math.PI / 2 + (phi[j] - phi[i]);
    }
  }

  for (let i = 0; i < K; i++) {
    // Self-interaction: mostly dispersive or chiral-anchored so clumps expand into lattices/rings
    if (isAnchor[i]) {
      famC[i][i] = -0.1 + 0.12 * rng();
      famA[i][i] = 0.72 + 0.2 * rng();
      famD[i][i] = Math.PI / 2 + chiralDir * lagMag;
    } else {
      famC[i][i] = -0.34 - 0.08 * rng();
      famA[i][i] = 0.32 + 0.16 * rng();
      famD[i][i] = Math.PI / 2 + chiralDir * lagMag * 0.7;
    }

    // Chiral chasing spine (+cycles around the family ring)
    const fwd = (i + cycles) % K;
    if (fwd !== i) {
      famC[i][fwd] = 0.35 + 0.14 * rng();
      famA[i][fwd] = 0.62 + 0.25 * rng();
      famD[i][fwd] = Math.PI / 2 + (phi[fwd] - phi[i]) + chiralDir * lagMag;

      famC[fwd][i] = -0.2 - 0.14 * rng();
      famA[fwd][i] = 0.58 + 0.24 * rng();
      famD[fwd][i] = Math.PI / 2 + (phi[i] - phi[fwd]) - chiralDir * lagMag;
    }

    // Cross-family structural bridge (builds multi-species nuclei and membranes)
    const bridge = (i + Math.floor(K / 2)) % K;
    if (bridge !== i && bridge !== fwd && rng() < 0.75) {
      const cVal = 0.18 + 0.14 * rng();
      const aVal = 0.72 + 0.2 * rng();
      famC[i][bridge] = cVal;
      famC[bridge][i] = cVal * (0.7 + 0.3 * rng());
      famA[i][bridge] = aVal;
      famA[bridge][i] = aVal;
      famD[i][bridge] = Math.PI / 2 + (phi[bridge] - phi[i]) + chiralDir * 0.15;
      famD[bridge][i] = Math.PI / 2 + (phi[i] - phi[bridge]) - chiralDir * 0.15;
    }

    // Next-nearest shell harmonic
    if (K >= 6) {
      const shell = (i + 2) % K;
      if (shell !== i && shell !== fwd && shell !== bridge) {
        famC[i][shell] = 0.12 + 0.16 * rng();
        famA[i][shell] = 0.68 + 0.2 * rng();
        famD[i][shell] =
          Math.PI / 2 + (phi[shell] - phi[i]) + chiralDir * 0.25;
      }
    }
  }

  const rows: MatrixGrid = [];
  for (let c1 = 0; c1 < colourAmt; c1++) {
    const f1 = Math.floor((c1 / colourAmt) * K);
    const row: ThetaFunction[] = [];
    for (let c2 = 0; c2 < colourAmt; c2++) {
      const f2 = Math.floor((c2 / colourAmt) * K);
      const a = Math.max(
        0.05,
        Math.min(1.0, famA[f1][f2] * (0.95 + 0.1 * rng())),
      );
      const b = famB[f1][f2];
      const c = Math.max(
        -0.5,
        Math.min(0.5, famC[f1][f2] + (rng() - 0.5) * 0.04),
      );
      const d = famD[f1][f2] + (rng() - 0.5) * 0.08;
      row.push(makeSinusoidFunction({ a, b, c, d }));
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
