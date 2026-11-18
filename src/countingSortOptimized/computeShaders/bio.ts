import structsShader from '../../shaders/structs.wgsl?raw';
import projectionShader from '../../shaders/projection.wgsl?raw';
import bioChemShader from '../../shaders/bioChem.wgsl?raw';

export const bioStepComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}
${projectionShader}
${bioChemShader}

@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var oceanTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read> lifeStateIn: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> lifeStateOut: array<vec4f>;
@group(0) @binding(6) var<storage, read> valence: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> bioForceOut: array<vec4f>;
@group(0) @binding(8) var<storage, read> sorted: array<SortParticle>;
@group(0) @binding(9) var<storage, read> prefixSumIndices: array<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&lifeStateOut)) {
        return;
    }

    let p = input[pi];
    let curLife = lifeStateIn[pi];
    let myPhase = unpackPhase(curLife.w);
    let nChem = max(u32(bio.chemColours), 1u);
    let mySc = min(u32(p.colour), nChem - 1u);
    let mySBase = mySc * SPECIES_STRIDE;
    let myValence = u32(matrix[mySBase + 0u]);
    let myAtomicNum = u32(matrix[mySBase + 7u]);
    let myV = valence[pi];
    let myAtomLoad = select(0.0, myV.x + myV.y + myV.z, myPhase == 0u);
    let oc = sampleOceanTexture(oceanTex, p.pos);

    if (bio.enabled <= 0.5) {
        let resOff = finalizeBioUnified(
            pi, p.pos, myAtomicNum, myValence, myAtomLoad, curLife, oc, initBioUnifiedAccum(), bio,
        );
        lifeStateOut[pi] = resOff.nextLifeState;
        bioForceOut[pi] = resOff.bioForce;
        return;
    }

    var acc = initBioUnifiedAccum();
    let worldP = p.pos * bio.worldSize;
    let cmin = vec3i(floor((worldP - bio.catRadius) / bio.cellSize));
    let cmax = vec3i(floor((worldP + bio.catRadius) / bio.cellSize));
    let numCells = max(u32(bio.cellAmt), 1u);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                let startIdx = prefixSumIndices[c];
                let endIdx = prefixSumIndices[c + 1u];
                for (var i = startIdx; i < endIdx; i++) {
                    let ip = sorted[i];
                    if (ip.idx == pi) {
                        continue;
                    }
                    let nbSc = min(u32(ip.colour), nChem - 1u);
                    let nbSBase = nbSc * SPECIES_STRIDE;
                    let nbValence = u32(matrix[nbSBase + 0u]);
                    let nbAtomicNum = u32(matrix[nbSBase + 7u]);
                    let nbV = valence[ip.idx];
                    let nbAtomLoad = select(0.0, nbV.x + nbV.y + nbV.z, ip.phase == 0u);
                    let nbLife = lifeStateIn[ip.idx];
                    accumulateBioUnifiedNeighbor(
                        p.pos,
                        myPhase,
                        myAtomicNum,
                        myValence,
                        myAtomLoad,
                        curLife,
                        ip.pos,
                        ip.colour,
                        ip.phase,
                        ip.bioSpecies,
                        nbLife,
                        nbAtomLoad,
                        nbAtomicNum,
                        nbValence,
                        bio,
                        &acc,
                    );
                }
            }
        }
    }

    let res = finalizeBioUnified(
        pi,
        p.pos,
        myAtomicNum,
        myValence,
        myAtomLoad,
        curLife,
        oc,
        acc,
        bio,
    );
    lifeStateOut[pi] = res.nextLifeState;
    bioForceOut[pi] = res.bioForce;
}
`;
};
