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
  windStackTexView: GPUTextureView;
  oceanTexView: GPUTextureView;
  tempTexView: GPUTextureView;
  tempStackTexView: GPUTextureView;
  uvTexView: GPUTextureView;
  hydroTexView: GPUTextureView;
  terrainTexView: GPUTextureView;
  particleAmt: number;
  cellAmt: number;
}
