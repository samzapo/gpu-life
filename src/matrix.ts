export const THETA_SAMPLES = 64;

export type ThetaFunction = (theta: number) => number;
export type MatrixGrid = ThetaFunction[][];

export function makeRandomSinusoidFunction(): ThetaFunction {
  // Math.random() is in [0, 1), so 1 - Math.random() is strictly in (0, 1].
  const a = 1 - Math.random();
  const b = Math.floor(Math.random() * 3) + 1;
  return (theta: number) => a * Math.sin(b * theta);
}

export function makeRandomMatrix(colourAmt: number): MatrixGrid {
  const rows: MatrixGrid = [];
  for (let c1 = 0; c1 < colourAmt; c1++) {
    const row: ThetaFunction[] = [];
    for (let c2 = 0; c2 < colourAmt; c2++) {
      row.push(makeRandomSinusoidFunction());
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
