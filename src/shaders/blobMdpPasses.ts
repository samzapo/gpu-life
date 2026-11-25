import structsShader from './structs.wgsl?raw';
import type { UnifiedEngineRefs } from '../engineTypes';

const bioLinkStructWgsl = /* wgsl */ `
struct BioLink {
    enabled: f32,
    rate: f32,
    catRadius: f32,
    worldSize: f32,
    cellSize: f32,
    cellAmt: f32,
    chemColours: f32,
    bioColours: f32,
    dt: f32,
    frame: f32,
    seedHash: f32,
    oceanCurrentScale: f32,
    windScale: f32,
    windStreamMax: f32,
    mineralEnabled: f32,
    mineralRate: f32,
    localTempScale: f32,
    localUvScale: f32,
    localHydroScale: f32,
    pad2: f32,
}

struct BlobPoolSlotAtomic {
    count: atomic<u32>,
    lipidCount: atomic<u32>,
    energySumQ: atomic<i32>,
    poiSumQ: atomic<i32>,
    solarSumQ: atomic<i32>,
    foreignSumQ: atomic<i32>,
    halogenSumQ: atomic<i32>,
    netDeSumQ: atomic<i32>,
}

struct BlobPoolSlotRead {
    count: u32,
    lipidCount: u32,
    energySumQ: i32,
    poiSumQ: i32,
    solarSumQ: i32,
    foreignSumQ: i32,
    halogenSumQ: i32,
    netDeSumQ: i32,
}

struct SpeciesMdpSlotAtomic {
    qFixed: array<atomic<i32>, 16>,
    bestAct: array<u32, 4>,
}

struct SpeciesMdpSlotPlain {
    qFixed: array<i32, 16>,
    bestAct: array<u32, 4>,
}
`;

export function blobScatterShader(workgroupSize: number): string {
  return /* wgsl */ `
${structsShader}
${bioLinkStructWgsl}

@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var<storage, read> lifeStateNext: array<LifeMdpState>;
@group(0) @binding(2) var<storage, read> bioForce: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> blobPool: array<BlobPoolSlotAtomic>;
@group(0) @binding(4) var<storage, read_write> speciesMdp: array<SpeciesMdpSlotAtomic>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (bio.enabled <= 0.5) {
        return;
    }
    let pi = global_id.x;
    let nParticles = arrayLength(&lifeStateNext);
    if (pi >= nParticles) {
        return;
    }
    let entry = lifeStateNext[pi];
    let phase = unpackPhase(entry.life.w);
    if (phase != 1u && phase != 2u) {
        return;
    }

    let poolLen = arrayLength(&blobPool);
    let rawLeader = entry.blob.x;
    let lid = select(pi, rawLeader, rawLeader < poolLen);

    if (phase == 2u) {
        atomicAdd(&blobPool[lid].lipidCount, 1u);
        return;
    }

    let sens = bioForce[pi];
    atomicAdd(&blobPool[lid].count, 1u);
    atomicAdd(&blobPool[lid].energySumQ, i32(round(clamp(entry.life.x, 0.0, 2.5) * 1000.0)));
    atomicAdd(&blobPool[lid].poiSumQ, i32(round(clamp(entry.life.z, 0.0, 1.5) * 1000.0)));
    atomicAdd(&blobPool[lid].solarSumQ, i32(round(clamp(sens.x, 0.0, 2.0) * 1000.0)));
    atomicAdd(&blobPool[lid].foreignSumQ, i32(round(clamp(sens.y, 0.0, 8.0) * 1000.0)));
    atomicAdd(&blobPool[lid].halogenSumQ, i32(round(clamp(sens.z, 0.0, 8.0) * 1000.0)));
    atomicAdd(&blobPool[lid].netDeSumQ, i32(round(clamp(sens.w, -4.0, 4.0) * 1000.0)));

    // Live GPU Species MDP 1-step TD accumulation (~4% stochastic sample to prevent atomic saturation)
    let frameU = u32(bio.frame);
    if ((entry.blob.w & 16u) != 0u && hashFloat(pi, frameU, 79u) < 0.04) {
        let nBio = max(min(u32(bio.bioColours), arrayLength(&speciesMdp)), 1u);
        let sp = min(unpackBioSpecies(entry.life.w), nBio - 1u);
        let prevSpState = entry.blob.w & 3u;
        let prevSpAct = (entry.blob.w >> 2u) & 3u;
        let curSpState = u32(round(entry.ctrl.w)) & 3u;

        let qIdx = prevSpState * 4u + prevSpAct;
        let oldQ = f32(atomicLoad(&speciesMdp[sp].qFixed[qIdx])) * 0.0001;
        var maxNextQ = -1.5;
        let nextBase = curSpState * 4u;
        for (var a = 0u; a < 4u; a++) {
            let qa = f32(atomicLoad(&speciesMdp[sp].qFixed[nextBase + a])) * 0.0001;
            if (qa > maxNextQ) {
                maxNextQ = qa;
            }
        }
        let rSp = clamp(entry.ctrl.z, -1.4, 1.4);
        let tdErr = clamp(rSp + 0.84 * maxNextQ - oldQ, -0.8, 0.8);
        atomicAdd(&speciesMdp[sp].qFixed[qIdx], i32(round(tdErr * 42.0)));
    }
}
`;
}

export function blobPolicyShader(workgroupSize: number): string {
  return /* wgsl */ `
${structsShader}
${bioLinkStructWgsl}

@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var<storage, read> blobPool: array<BlobPoolSlotRead>;
@group(0) @binding(2) var<storage, read_write> blobMdp: array<BlobMdpSlot>;
@group(0) @binding(3) var<storage, read_write> speciesMdp: array<SpeciesMdpSlotPlain>;
@group(0) @binding(4) var<storage, read> lifeStateNext: array<LifeMdpState>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (bio.enabled <= 0.5) {
        return;
    }
    let pi = global_id.x;
    let nBio = max(min(u32(bio.bioColours), arrayLength(&speciesMdp)), 1u);
    let frameU = u32(bio.frame);

    if (pi < nBio) {
        for (var s = 0u; s < 4u; s++) {
            let base = s * 4u;
            var bestA = 0u;
            var bestV: i32 = -100000;
            for (var a = 0u; a < 4u; a++) {
                var qv = clamp(speciesMdp[pi].qFixed[base + a], -14500, 14500);
                if (abs(qv) > 80) {
                    let decay = max(1, abs(qv) / 3500);
                    qv -= select(-decay, decay, qv > 0);
                }
                speciesMdp[pi].qFixed[base + a] = qv;
                if (qv > bestV) {
                    bestV = qv;
                    bestA = a;
                }
            }
            speciesMdp[pi].bestAct[s] = bestA;
        }
    }

    let poolLen = arrayLength(&blobPool);
    if (pi >= poolLen) {
        return;
    }
    let pool = blobPool[pi];
    if (pool.count == 0u) {
        blobMdp[pi].ctrl.w = 0.0;
        return;
    }

    let cnt = f32(pool.count);
    let avgE = f32(pool.energySumQ) / (1000.0 * cnt);
    let avgPoi = f32(pool.poiSumQ) / (1000.0 * cnt);
    let avgSolar = f32(pool.solarSumQ) / (1000.0 * cnt);
    let avgForeign = f32(pool.foreignSumQ) / (1000.0 * cnt);
    let avgHalogen = f32(pool.halogenSumQ) / (1000.0 * cnt);
    let avgNetDE = f32(pool.netDeSumQ) / (1000.0 * cnt);

    var bState: u32 = MDP_STATE_ABUNDANT;
    if (avgPoi > 0.16 || avgHalogen > 0.28 || (avgForeign > 0.65 && avgNetDE < -0.02)) {
        bState = MDP_STATE_HAZARD;
    } else if (avgE < 0.56) {
        bState = MDP_STATE_LOW_ENERGY;
    } else if (avgSolar < 0.16 || pool.count >= 96u) {
        bState = MDP_STATE_DARK_CROWDED;
    }

    let prev = blobMdp[pi];
    let prevBits = u32(round(prev.ctrl.w));
    let isInit = (prevBits & 4096u) != 0u;
    let leaderEntry = lifeStateNext[pi];
    let leaderBits = u32(round(leaderEntry.ctrl.w));

    var qPack = select(leaderEntry.qPack, prev.qPack, isInit);
    let prevState = select(bState, prevBits & 3u, isInit);
    let prevAct = select((leaderBits >> 2u) & 3u, (prevBits >> 2u) & 3u, isInit);
    let prevDwell = select(16u, (prevBits >> 4u) & 255u, isInit);
    let prevAvgE = select(avgE, prev.ctrl.x, isInit);
    let prevCnt = select(cnt, max(prev.ctrl.y, 1.0), isInit);
    var emaR = select(0.0, prev.ctrl.z, isInit);

    let dAvgE = clamp(avgE - prevAvgE, -0.5, 0.5);
    let growth = clamp((cnt - prevCnt) / max(prevCnt, 4.0), -0.45, 0.45);
    let rBlob = clamp(
        avgNetDE * 1.45 + dAvgE * 2.2 + growth * 1.35 - avgPoi * 0.85 - avgHalogen * 0.42 + select(0.0, 0.10, avgE > 0.78),
        -1.45,
        1.45,
    );
    let dtBio = clamp(bio.dt * max(bio.rate, 0.1), 0.002, 0.08);
    emaR = mix(emaR, rBlob, clamp(1.5 * dtBio, 0.02, 0.16));

    let curRow = getQRow(qPack, bState);
    var bestAct = prevAct;
    var bestScore = -1e9;
    var maxNextQ = -1.5;
    for (var aa = 0u; aa < 4u; aa++) {
        let qVal = unpackMdpQ(curRow, aa);
        if (qVal > maxNextQ) {
            maxNextQ = qVal;
        }
        let dwellBonus = select(0.0, 0.14, aa == prevAct && prevDwell > 0u);
        let epsNoise = (hashFloat(pi, frameU, 80u + aa) - 0.5) * 0.10;
        let score = qVal + dwellBonus + epsNoise;
        if (score > bestScore) {
            bestScore = score;
            bestAct = aa;
        }
    }

    let nextDwell = select(select(0u, prevDwell - 1u, prevDwell > 0u), 20u, bestAct != prevAct);

    let prevRow = getQRow(qPack, prevState);
    let oldQ = unpackMdpQ(prevRow, prevAct);
    let alphaQ = clamp(2.2 * dtBio, 0.02, 0.16);
    let updatedQ = clamp(mix(oldQ, rBlob + 0.84 * maxNextQ, alphaQ), -1.45, 1.45);
    qPack = setQRow(qPack, prevState, setMdpQ(prevRow, prevAct, updatedQ));

    blobMdp[pi] = BlobMdpSlot(
        qPack,
        vec4f(avgE, cnt, emaR, packMdpCtrl(bState, bestAct, nextDwell, 1u)),
    );
}
`;
}

export function blobBroadcastShader(workgroupSize: number): string {
  return /* wgsl */ `
${structsShader}
${bioLinkStructWgsl}

@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var<storage, read_write> lifeStateNext: array<LifeMdpState>;
@group(0) @binding(2) var<storage, read> blobMdp: array<BlobMdpSlot>;
@group(0) @binding(3) var<storage, read> speciesMdp: array<SpeciesMdpSlotPlain>;
@group(0) @binding(4) var<storage, read> blobPool: array<BlobPoolSlotRead>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (bio.enabled <= 0.5) {
        return;
    }
    let pi = global_id.x;
    let nParticles = arrayLength(&lifeStateNext);
    if (pi >= nParticles) {
        return;
    }

    var entry = lifeStateNext[pi];
    let phase = unpackPhase(entry.life.w);
    if (phase != 1u && phase != 2u) {
        return;
    }

    let poolLen = arrayLength(&blobPool);
    let rawLeader = entry.blob.x;
    let lid = select(pi, rawLeader, rawLeader < poolLen);

    let pBits = u32(round(entry.ctrl.w));
    let pState = pBits & 3u;
    let pAct = (pBits >> 2u) & 3u;

    let bSlot = blobMdp[lid];
    let bBits = u32(round(bSlot.ctrl.w));
    let bInit = (bBits & 4096u) != 0u;
    let bCount = blobPool[lid].count;
    let hasBlob = select(0u, 1u, bInit && bCount >= 2u);
    let bState = select(pState, bBits & 3u, bInit);
    let bAct = select(pAct, (bBits >> 2u) & 3u, bInit);

    let nBio = max(min(u32(bio.bioColours), arrayLength(&speciesMdp)), 1u);
    let sp = min(unpackBioSpecies(entry.life.w), nBio - 1u);
    var sAct = speciesMdp[sp].bestAct[pState] & 3u;
    let frameU = u32(bio.frame);
    if (hashFloat(pi, frameU, 83u) < 0.08) {
        let offset = 1u + (u32(floor(hashFloat(pi, frameU, 84u) * 3.0)) & 3u);
        sAct = (sAct + offset) & 3u;
    }

    let sizeBucket = min(7u, u32(floor(log2(max(f32(bCount), 1.0)))));
    entry.blob.x = lid;
    entry.blob.z = packBlobPolicyZ(pAct, bAct, sAct, pState, bState, hasBlob, sizeBucket);
    entry.blob.w = (pState & 3u) | ((sAct & 3u) << 2u) | 16u;
    lifeStateNext[pi] = entry;
}
`;
}

export interface BlobMdpBundle {
  scatterPipeline: GPUComputePipeline;
  policyPipeline: GPUComputePipeline;
  broadcastPipeline: GPUComputePipeline;
}

export interface BlobMdpBindGroups {
  scatterGroups: [GPUBindGroup, GPUBindGroup];
  policyGroups: [GPUBindGroup, GPUBindGroup];
  broadcastGroups: [GPUBindGroup, GPUBindGroup];
}

export function createBlobMdpBundle(
  device: GPUDevice,
  workgroupSize: number,
): BlobMdpBundle {
  const scatterMod = device.createShaderModule({
    code: blobScatterShader(workgroupSize),
  });
  const policyMod = device.createShaderModule({
    code: blobPolicyShader(workgroupSize),
  });
  const broadcastMod = device.createShaderModule({
    code: blobBroadcastShader(workgroupSize),
  });
  return {
    scatterPipeline: device.createComputePipeline({
      layout: 'auto',
      compute: { module: scatterMod, entryPoint: 'main' },
    }),
    policyPipeline: device.createComputePipeline({
      layout: 'auto',
      compute: { module: policyMod, entryPoint: 'main' },
    }),
    broadcastPipeline: device.createComputePipeline({
      layout: 'auto',
      compute: { module: broadcastMod, entryPoint: 'main' },
    }),
  };
}

export function buildBlobMdpBindGroups(
  device: GPUDevice,
  bundle: BlobMdpBundle,
  refs: UnifiedEngineRefs,
): BlobMdpBindGroups {
  const scatterGroups: GPUBindGroup[] = [];
  const policyGroups: GPUBindGroup[] = [];
  const broadcastGroups: GPUBindGroup[] = [];

  for (let i = 0; i < 2; i++) {
    const nextLifeBuf = refs.lifeStateBuffers[1 - i];
    scatterGroups.push(
      device.createBindGroup({
        layout: bundle.scatterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.bioUniformBuffer } },
          { binding: 1, resource: { buffer: nextLifeBuf } },
          { binding: 2, resource: { buffer: refs.bioForceBuffer } },
          { binding: 3, resource: { buffer: refs.blobPoolBuffer } },
          { binding: 4, resource: { buffer: refs.speciesMdpBuffer } },
        ],
      }),
    );

    policyGroups.push(
      device.createBindGroup({
        layout: bundle.policyPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.bioUniformBuffer } },
          { binding: 1, resource: { buffer: refs.blobPoolBuffer } },
          { binding: 2, resource: { buffer: refs.blobMdpBuffer } },
          { binding: 3, resource: { buffer: refs.speciesMdpBuffer } },
          { binding: 4, resource: { buffer: nextLifeBuf } },
        ],
      }),
    );

    broadcastGroups.push(
      device.createBindGroup({
        layout: bundle.broadcastPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.bioUniformBuffer } },
          { binding: 1, resource: { buffer: nextLifeBuf } },
          { binding: 2, resource: { buffer: refs.blobMdpBuffer } },
          { binding: 3, resource: { buffer: refs.speciesMdpBuffer } },
          { binding: 4, resource: { buffer: refs.blobPoolBuffer } },
        ],
      }),
    );
  }

  return {
    scatterGroups: [scatterGroups[0], scatterGroups[1]],
    policyGroups: [policyGroups[0], policyGroups[1]],
    broadcastGroups: [broadcastGroups[0], broadcastGroups[1]],
  };
}

export function dispatchBlobMdpPasses(
  commandEncoder: GPUCommandEncoder,
  bundle: BlobMdpBundle,
  bindGroups: BlobMdpBindGroups,
  alternate: number,
  particleAmt: number,
  bioColourAmt: number,
  blobPoolByteLength: number,
  blobPoolZeroBuffer: GPUBuffer,
  blobPoolBuffer: GPUBuffer,
  workgroupSize: number,
) {
  const wgParticles = Math.ceil(particleAmt / workgroupSize);
  const wgPolicy = Math.ceil(Math.max(particleAmt, Math.max(1, bioColourAmt)) / workgroupSize);

  commandEncoder.copyBufferToBuffer(blobPoolZeroBuffer, 0, blobPoolBuffer, 0, blobPoolByteLength);

  const scatterPass = commandEncoder.beginComputePass({});
  scatterPass.setPipeline(bundle.scatterPipeline);
  scatterPass.setBindGroup(0, bindGroups.scatterGroups[alternate]);
  scatterPass.dispatchWorkgroups(wgParticles);
  scatterPass.end();

  const policyPass = commandEncoder.beginComputePass({});
  policyPass.setPipeline(bundle.policyPipeline);
  policyPass.setBindGroup(0, bindGroups.policyGroups[alternate]);
  policyPass.dispatchWorkgroups(wgPolicy);
  policyPass.end();

  const broadcastPass = commandEncoder.beginComputePass({});
  broadcastPass.setPipeline(bundle.broadcastPipeline);
  broadcastPass.setBindGroup(0, bindGroups.broadcastGroups[alternate]);
  broadcastPass.dispatchWorkgroups(wgParticles);
  broadcastPass.end();
}
