import { parseConfigInput, type SimConfigV1 } from './config.ts';

export const SIM_STATE_KIND = 'complex-life-sim-state' as const;
export const PARTICLE_BYTES = 48;
export const LIFE_MDP_BYTES = 64;
export const VEC4_BYTES = 16;
export const GAME_TRACKING_BYTES = 8;
export const BLOB_MDP_BYTES = 32;
export const SPECIES_MDP_BYTES = 80;

export type SimStateEncoding = 'gzip-base64' | 'raw-base64';

export interface SimStateBufferBlob {
  encoding: SimStateEncoding;
  byteLength: number;
  data: string;
}

export interface SimStateRuntimeMeta {
  engine: string;
  particleCount: number;
  alternate: 0 | 1;
  frameCount: number;
  uvUtcMs?: number;
  userAgent?: string;
  viewport?: [number, number, number];
  audio?: {
    enabled: boolean;
    state: string;
    volume: number;
    lastObservation?: unknown;
  };
}

export interface SimStateExportV1 {
  v: 1;
  kind: typeof SIM_STATE_KIND;
  capturedAt: string;
  config: SimConfigV1;
  runtime: SimStateRuntimeMeta;
  buffers: {
    particles: SimStateBufferBlob;
    lifeMdp: SimStateBufferBlob;
    bioForce?: SimStateBufferBlob;
    valence?: SimStateBufferBlob;
    gameTracking?: SimStateBufferBlob;
    blobMdp?: SimStateBufferBlob;
    speciesMdp?: SimStateBufferBlob;
  };
}

export interface SerializeSimStateInput {
  config: SimConfigV1;
  runtime: SimStateRuntimeMeta;
  particles: Uint8Array;
  lifeMdp: Uint8Array;
  bioForce?: Uint8Array;
  valence?: Uint8Array;
  gameTracking?: Uint8Array;
  blobMdp?: Uint8Array;
  speciesMdp?: Uint8Array;
  capturedAt?: string;
  encoding?: SimStateEncoding;
}

export interface DecodedSimState {
  snapshot: SimStateExportV1;
  config: SimConfigV1;
  runtime: SimStateRuntimeMeta;
  particles: Uint8Array;
  lifeMdp: Uint8Array;
  bioForce?: Uint8Array;
  valence?: Uint8Array;
  gameTracking?: Uint8Array;
  blobMdp?: Uint8Array;
  speciesMdp?: Uint8Array;
}

export function encodeBytesToBase64(bytes: Uint8Array): string {
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const slice = bytes.subarray(i, Math.min(bytes.length, i + chunkSize));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

export function decodeBase64ToBytes(b64: string): Uint8Array {
  const clean = b64.replace(/\s+/g, '');
  const binary = atob(clean);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

async function pipeStreamTransform(
  bytes: Uint8Array,
  transform: CompressionStream | DecompressionStream,
): Promise<Uint8Array> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const stream = new Blob([copy]).stream().pipeThrough(transform as unknown as ReadableWritablePair<Uint8Array, Uint8Array>);
  const ab = await new Response(stream).arrayBuffer();
  return new Uint8Array(ab);
}

export async function compressBufferBlob(
  bytes: Uint8Array,
  preferredEncoding?: SimStateEncoding,
): Promise<SimStateBufferBlob> {
  if (
    preferredEncoding !== 'raw-base64' &&
    typeof CompressionStream !== 'undefined' &&
    typeof Blob !== 'undefined' &&
    typeof Response !== 'undefined'
  ) {
    try {
      const compressed = await pipeStreamTransform(bytes, new CompressionStream('gzip'));
      return {
        encoding: 'gzip-base64',
        byteLength: bytes.byteLength,
        data: encodeBytesToBase64(compressed),
      };
    } catch {
      // Fall back cleanly to uncompressed base64 if gzip stream fails
    }
  }
  return {
    encoding: 'raw-base64',
    byteLength: bytes.byteLength,
    data: encodeBytesToBase64(bytes),
  };
}

export async function decompressBufferBlob(
  blob: unknown,
  label: string,
): Promise<Uint8Array> {
  if (!blob || typeof blob !== 'object') {
    throw new Error(`Sim state buffer "${label}" is missing.`);
  }
  const rec = blob as Record<string, unknown>;
  const encoding = rec.encoding;
  const byteLength = rec.byteLength;
  const data = rec.data;

  if (encoding !== 'gzip-base64' && encoding !== 'raw-base64') {
    throw new Error(`Unsupported encoding for buffer "${label}".`);
  }
  if (
    typeof byteLength !== 'number' ||
    !Number.isInteger(byteLength) ||
    byteLength < 0
  ) {
    throw new Error(`Invalid byteLength for buffer "${label}".`);
  }
  if (typeof data !== 'string' || !data.trim()) {
    throw new Error(`Missing encoded data for buffer "${label}".`);
  }

  const rawBytes = decodeBase64ToBytes(data);
  const decoded =
    encoding === 'gzip-base64'
      ? await (async () => {
          if (typeof DecompressionStream === 'undefined') {
            throw new Error(
              `Browser lacks DecompressionStream('gzip') needed for "${label}".`,
            );
          }
          return pipeStreamTransform(rawBytes, new DecompressionStream('gzip'));
        })()
      : rawBytes;

  if (decoded.byteLength !== byteLength) {
    throw new Error(
      `Decoded "${label}" size mismatch (${decoded.byteLength} !== ${byteLength}).`,
    );
  }
  return decoded;
}

export function extractRawJsonObject(rawInput: unknown): Record<string, unknown> | null {
  if (rawInput && typeof rawInput === 'object' && !Array.isArray(rawInput)) {
    return rawInput as Record<string, unknown>;
  }
  if (typeof rawInput !== 'string') return null;
  let candidate = rawInput.trim();
  if (!candidate) return null;
  if (!candidate.startsWith('{')) {
    const first = candidate.indexOf('{');
    const last = candidate.lastIndexOf('}');
    if (first < 0 || last <= first) return null;
    candidate = candidate.slice(first, last + 1);
  }
  try {
    const parsed = JSON.parse(candidate);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return null;
  }
  return null;
}

export function isSerializedSimStateInput(rawInput: unknown): boolean {
  if (typeof rawInput === 'string' && !rawInput.includes(SIM_STATE_KIND)) {
    return false;
  }
  const obj = extractRawJsonObject(rawInput);
  return Boolean(obj && obj.kind === SIM_STATE_KIND);
}

export async function serializeSimState(
  input: SerializeSimStateInput,
): Promise<SimStateExportV1> {
  const particleCount = Math.max(1, Math.floor(input.runtime.particleCount));
  if (input.particles.byteLength !== particleCount * PARTICLE_BYTES) {
    throw new Error('Particle buffer length does not match particleCount.');
  }
  if (input.lifeMdp.byteLength !== particleCount * LIFE_MDP_BYTES) {
    throw new Error('Life/MDP buffer length does not match particleCount.');
  }

  const [particles, lifeMdp, bioForce, valence, gameTracking, blobMdp, speciesMdp] = await Promise.all([
    compressBufferBlob(input.particles, input.encoding),
    compressBufferBlob(input.lifeMdp, input.encoding),
    input.bioForce ? compressBufferBlob(input.bioForce, input.encoding) : Promise.resolve(undefined),
    input.valence ? compressBufferBlob(input.valence, input.encoding) : Promise.resolve(undefined),
    input.gameTracking
      ? compressBufferBlob(input.gameTracking, input.encoding)
      : Promise.resolve(undefined),
    input.blobMdp ? compressBufferBlob(input.blobMdp, input.encoding) : Promise.resolve(undefined),
    input.speciesMdp
      ? compressBufferBlob(input.speciesMdp, input.encoding)
      : Promise.resolve(undefined),
  ]);

  const buffers: SimStateExportV1['buffers'] = {
    particles,
    lifeMdp,
  };
  if (bioForce) buffers.bioForce = bioForce;
  if (valence) buffers.valence = valence;
  if (gameTracking) buffers.gameTracking = gameTracking;
  if (blobMdp) buffers.blobMdp = blobMdp;
  if (speciesMdp) buffers.speciesMdp = speciesMdp;

  return {
    v: 1,
    kind: SIM_STATE_KIND,
    capturedAt: input.capturedAt ?? new Date().toISOString(),
    config: input.config,
    runtime: {
      ...input.runtime,
      particleCount,
      alternate: input.runtime.alternate === 1 ? 1 : 0,
      frameCount: Math.max(0, Math.floor(input.runtime.frameCount)),
    },
    buffers,
  };
}

export async function parseSimStateInput(
  rawInput: string | SimStateExportV1,
): Promise<DecodedSimState> {
  const obj = extractRawJsonObject(rawInput);
  if (!obj || obj.kind !== SIM_STATE_KIND) {
    throw new Error('Input is not a valid complex-life simulation state snapshot.');
  }
  if (obj.v !== 1) {
    throw new Error(`Unsupported sim state version: ${String(obj.v)}.`);
  }
  if (!obj.config || typeof obj.config !== 'object') {
    throw new Error('Sim state snapshot is missing config.');
  }
  if (!obj.runtime || typeof obj.runtime !== 'object') {
    throw new Error('Sim state snapshot is missing runtime metadata.');
  }
  if (!obj.buffers || typeof obj.buffers !== 'object') {
    throw new Error('Sim state snapshot is missing GPU buffers.');
  }

  const config = parseConfigInput(obj.config as SimConfigV1);
  const rtRaw = obj.runtime as Record<string, unknown>;
  const particleCountRaw = Number(rtRaw.particleCount ?? config.particles);
  if (!Number.isInteger(particleCountRaw) || particleCountRaw <= 0) {
    throw new Error('Invalid particleCount in sim state snapshot.');
  }
  const particleCount = particleCountRaw;
  config.particles = particleCount;

  const altRaw = Number(rtRaw.alternate ?? 0);
  const frameRaw = Number(rtRaw.frameCount ?? 0);
  const runtime: SimStateRuntimeMeta = {
    engine: typeof rtRaw.engine === 'string' ? rtRaw.engine : (config.engine ?? 'cells'),
    particleCount,
    alternate: altRaw === 1 ? 1 : 0,
    frameCount: Number.isFinite(frameRaw) && frameRaw >= 0 ? Math.floor(frameRaw) : 0,
    uvUtcMs:
      typeof rtRaw.uvUtcMs === 'number' && Number.isFinite(rtRaw.uvUtcMs)
        ? rtRaw.uvUtcMs
        : undefined,
    userAgent: typeof rtRaw.userAgent === 'string' ? rtRaw.userAgent : undefined,
    viewport:
      Array.isArray(rtRaw.viewport) && rtRaw.viewport.length === 3
        ? [Number(rtRaw.viewport[0]), Number(rtRaw.viewport[1]), Number(rtRaw.viewport[2])]
        : undefined,
    audio:
      rtRaw.audio && typeof rtRaw.audio === 'object'
        ? (rtRaw.audio as SimStateRuntimeMeta['audio'])
        : undefined,
  };

  const bufsRaw = obj.buffers as Record<string, unknown>;
  const [particles, lifeMdp, bioForce, valence, gameTracking, blobMdp, speciesMdp] = await Promise.all([
    decompressBufferBlob(bufsRaw.particles, 'particles'),
    decompressBufferBlob(bufsRaw.lifeMdp, 'lifeMdp'),
    bufsRaw.bioForce ? decompressBufferBlob(bufsRaw.bioForce, 'bioForce') : Promise.resolve(undefined),
    bufsRaw.valence ? decompressBufferBlob(bufsRaw.valence, 'valence') : Promise.resolve(undefined),
    bufsRaw.gameTracking
      ? decompressBufferBlob(bufsRaw.gameTracking, 'gameTracking')
      : Promise.resolve(undefined),
    bufsRaw.blobMdp ? decompressBufferBlob(bufsRaw.blobMdp, 'blobMdp') : Promise.resolve(undefined),
    bufsRaw.speciesMdp
      ? decompressBufferBlob(bufsRaw.speciesMdp, 'speciesMdp')
      : Promise.resolve(undefined),
  ]);

  if (particles.byteLength !== particleCount * PARTICLE_BYTES) {
    throw new Error(
      `particles byteLength mismatch (${particles.byteLength} !== ${particleCount * PARTICLE_BYTES}).`,
    );
  }
  if (lifeMdp.byteLength !== particleCount * LIFE_MDP_BYTES) {
    throw new Error(
      `lifeMdp byteLength mismatch (${lifeMdp.byteLength} !== ${particleCount * LIFE_MDP_BYTES}).`,
    );
  }
  if (bioForce && bioForce.byteLength !== particleCount * VEC4_BYTES) {
    throw new Error(
      `bioForce byteLength mismatch (${bioForce.byteLength} !== ${particleCount * VEC4_BYTES}).`,
    );
  }
  if (valence && valence.byteLength !== particleCount * VEC4_BYTES) {
    throw new Error(
      `valence byteLength mismatch (${valence.byteLength} !== ${particleCount * VEC4_BYTES}).`,
    );
  }
  if (gameTracking && gameTracking.byteLength !== particleCount * GAME_TRACKING_BYTES) {
    throw new Error(
      `gameTracking byteLength mismatch (${gameTracking.byteLength} !== ${particleCount * GAME_TRACKING_BYTES}).`,
    );
  }
  if (blobMdp && blobMdp.byteLength !== particleCount * BLOB_MDP_BYTES) {
    throw new Error(
      `blobMdp byteLength mismatch (${blobMdp.byteLength} !== ${particleCount * BLOB_MDP_BYTES}).`,
    );
  }
  if (speciesMdp && speciesMdp.byteLength % SPECIES_MDP_BYTES !== 0) {
    throw new Error(
      `speciesMdp byteLength mismatch (${speciesMdp.byteLength} not aligned to ${SPECIES_MDP_BYTES}).`,
    );
  }

  return {
    snapshot: obj as unknown as SimStateExportV1,
    config,
    runtime,
    particles,
    lifeMdp,
    bioForce,
    valence,
    gameTracking,
    blobMdp,
    speciesMdp,
  };
}
