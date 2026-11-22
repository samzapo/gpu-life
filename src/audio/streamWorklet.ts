/**
 * Inline Web Audio AudioWorkletProcessor for spatial Wind & Ocean Streamline synthesis.
 * Runs 16 stereo harmonic voices (8 wind jet bands + 8 ocean current bands) in a single
 * zero-allocation render loop at ~0.15ms per 128-sample quantum.
 */
export const STREAMLINE_WORKLET_NAME = 'streamline-synth-processor';

export const STREAMLINE_WORKLET_SOURCE = `
class StreamlineSynthProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.phases = new Float32Array(16);
    this.detunePhases = new Float32Array(16);
    this.amps = new Float32Array(16);
    this.targetAmps = new Float32Array(16);
    this.freqs = new Float32Array([
      146.83, 220.0, 293.66, 349.23, 440.0, 523.25, 587.33, 659.25,
      73.42, 110.0, 146.83, 174.61, 220.0, 261.63, 293.66, 329.63
    ]);
    this.pans = new Float32Array(16);
    for (let i = 0; i < 8; i++) {
      const p = -0.82 + (i / 7) * 1.64;
      this.pans[i] = p;
      this.pans[8 + i] = p;
    }
    this.drive = 0.2;
    this.masterAmp = 0;
    this.targetMasterAmp = 0;

    this.port.onmessage = (ev) => {
      const d = ev.data;
      if (!d) return;
      if (d.bands && d.bands.length >= 16) {
        for (let i = 0; i < 16; i++) {
          this.targetAmps[i] = Math.max(0, Math.min(1, d.bands[i] || 0));
        }
      }
      if (d.freqs && d.freqs.length >= 16) {
        for (let i = 0; i < 16; i++) {
          const f = d.freqs[i];
          if (f > 28 && f < 2200) this.freqs[i] = f;
        }
      }
      if (typeof d.drive === 'number') {
        this.drive = Math.max(0, Math.min(1, d.drive));
      }
      if (typeof d.masterAmp === 'number') {
        this.targetMasterAmp = Math.max(0, Math.min(1, d.masterAmp));
      }
      if (d.resetState) {
        this.masterAmp = 0;
        this.amps.fill(0);
        this.phases.fill(0);
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const outL = output[0];
    const outR = output[1] || output[0];
    const len = outL.length;
    const invSr = 1 / Math.max(8000, sampleRate);
    const TAU = Math.PI * 2;
    const sat = 1.0 + this.drive * 0.65;

    for (let n = 0; n < len; n++) {
      this.masterAmp += (this.targetMasterAmp - this.masterAmp) * 0.004;
      if (this.masterAmp < 0.0002 && this.targetMasterAmp < 0.0002) {
        outL[n] = 0;
        if (outR !== outL) outR[n] = 0;
        continue;
      }

      let sumL = 0;
      let sumR = 0;
      for (let i = 0; i < 16; i++) {
        this.amps[i] += (this.targetAmps[i] - this.amps[i]) * 0.0035;
        const amp = this.amps[i];
        if (amp < 0.001) continue;

        const detuneRate = 0.0022 + (i & 3) * 0.0008;
        const df = this.freqs[i] * (1.0 + 0.0045 * Math.sin(this.detunePhases[i]));
        this.phases[i] = (this.phases[i] + df * invSr) % 1;
        this.detunePhases[i] = (this.detunePhases[i] + detuneRate * 0.06) % TAU;

        const p = this.phases[i];
        // Smooth warm triangle + fundamental blend (dark ambient cyberpunk filament)
        const tri = 4 * Math.abs(p - 0.5) - 1;
        const fund = Math.sin(TAU * p);
        const harmWeight = i < 8 ? 0.36 + 0.30 * this.drive : 0.20 + 0.16 * this.drive;
        const sig = Math.tanh((fund * (1 - harmWeight) + tri * harmWeight) * sat) * amp;

        const pan = this.pans[i];
        const lGain = 0.5 * (1 - pan * 0.72);
        const rGain = 0.5 * (1 + pan * 0.72);
        sumL += sig * lGain;
        sumR += sig * rGain;
      }

      const norm = 0.042 * this.masterAmp;
      const sampleL = sumL * norm;
      const sampleR = sumR * norm;
      if (outR === outL) {
        outL[n] = (sampleL + sampleR) * 0.5;
      } else {
        outL[n] = sampleL;
        outR[n] = sampleR;
      }
    }
    return true;
  }
}
registerProcessor('${STREAMLINE_WORKLET_NAME}', StreamlineSynthProcessor);
`;
