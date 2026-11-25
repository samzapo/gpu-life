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
  blobPoolBuffer: GPUBuffer;
  blobPoolZeroBuffer: GPUBuffer;
  blobMdpBuffer: GPUBuffer;
  speciesMdpBuffer: GPUBuffer;
  bioColourAmt: number;
  windTexView: GPUTextureView;
  oceanTexView: GPUTextureView;
  tempTexView: GPUTextureView;
  uvTexView: GPUTextureView;
  hydroTexView: GPUTextureView;
  terrainTexView: GPUTextureView;
  particleAmt: number;
  cellAmt: number;
}
