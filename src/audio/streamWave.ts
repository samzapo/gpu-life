export const SYNTH_OCEAN_WAVE_BASE_RAD = 0.0036;
export const SYNTH_WIND_WAVE_BASE_RAD = 0.0030;
export const SYNTH_STREAM_MAX_WAVE_PX = 96;

/**
 * Computes perpendicular screen-space wave amplitude (px) for a streamline segment
 * modulated by synth transients and the separate Display amplitude multipliers.
 */
export function computeSynthStreamWaveAmplitude(
  scale: number,
  speedFrac: number,
  synthBandHit: number,
  synthAmpScale: number,
  isOcean: boolean,
): number {
  if (synthBandHit <= 0.004 || synthAmpScale <= 0.001) return 0;
  const baseRad = isOcean ? SYNTH_OCEAN_WAVE_BASE_RAD : SYNTH_WIND_WAVE_BASE_RAD;
  const surfaceAmpRad = baseRad * synthBandHit * synthAmpScale;
  return Math.min(SYNTH_STREAM_MAX_WAVE_PX, surfaceAmpRad * scale) * (0.28 + 0.72 * speedFrac);
}
