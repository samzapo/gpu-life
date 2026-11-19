export interface UnifiedEngineRefs {
  uniformBuffer: GPUBuffer;
  simBuffer: GPUBuffer;
  oceanSimBuffer: GPUBuffer;
  bioUniformBuffer: GPUBuffer;
  rulesMatrixBuffer: GPUBuffer;
  particleBuffers: [GPUBuffer, GPUBuffer];
  lifeStateBuffers: [GPUBuffer, GPUBuffer];
  valenceBuffer: GPUBuffer;
  bioForceBuffer: GPUBuffer;
  windTexView: GPUTextureView;
  oceanTexView: GPUTextureView;
  tempTexView: GPUTextureView;
  particleAmt: number;
  cellAmt: number;
}
