import structsShader from '../../shaders/structs.wgsl?raw';
import projectionShader from '../../shaders/projection.wgsl?raw';
import bioChemShader from '../../shaders/bioChem.wgsl?raw';

export const bioAirComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}
${projectionShader}
${bioChemShader}

@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var<storage, read> airMatrix: array<f32>;
@group(0) @binding(2) var<storage, read> airInput: array<Particle>;
@group(0) @binding(3) var<storage, read> airValence: array<vec4f>;
@group(0) @binding(4) var<storage, read> oceanSorted: array<SortParticle>;
@group(0) @binding(5) var<storage, read> oceanCounts: array<u32>;
@group(0) @binding(6) var<storage, read> oceanPrefixSum: array<u32>;
@group(0) @binding(7) var<storage, read> oceanState: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> airBioReact: array<vec4f>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&airBioReact)) {
        return;
    }
    if (bio.enabled <= 0.5) {
        airBioReact[pi] = vec4f(0.0);
        return;
    }

    let p = airInput[pi];
    let nCol = max(u32(bio.airColours), 1u);
    let sc = min(u32(p.colour), nCol - 1u);
    let sBase = sc * SPECIES_STRIDE;
    let valence = u32(airMatrix[sBase + 0u]);
    let atomicNum = u32(airMatrix[sBase + 7u]);

    if (bioRoleKind(atomicNum, valence) == 0u) {
        airBioReact[pi] = vec4f(0.0);
        return;
    }

    let vLoad = airValence[pi];
    let atomLoad = vLoad.x + vLoad.y + vLoad.z;
    var acc = BioAirAccum(vec3f(0.0), 1.0, 0.0, 0.0);

    let worldP = p.pos * bio.worldSize;
    let cmin = vec3i(floor((worldP - bio.catRadius) / bio.oceanCellSize));
    let cmax = vec3i(floor((worldP + bio.catRadius) / bio.oceanCellSize));
    let numCells = max(u32(bio.oceanCellAmt), 1u);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                let cnt = oceanCounts[c];
                if (cnt == 0u) {
                    continue;
                }
                let startIdx = oceanPrefixSum[c];
                let endIdx = startIdx + cnt;
                for (var i = startIdx; i < endIdx; i++) {
                    let life = oceanSorted[i];
                    accumulateBioAirNeighbor(
                        p.pos,
                        atomicNum,
                        valence,
                        atomLoad,
                        life.pos,
                        life.colour,
                        oceanState[life.idx],
                        bio,
                        &acc,
                    );
                }
            }
        }
    }

    airBioReact[pi] = finalizeBioAir(acc, bio);
}
`;
};

export const bioOceanComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}
${projectionShader}
${bioChemShader}

@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var<storage, read> airMatrix: array<f32>;
@group(0) @binding(2) var<storage, read> airValence: array<vec4f>;
@group(0) @binding(3) var<storage, read> airSorted: array<SortParticle>;
@group(0) @binding(4) var<storage, read> airCounts: array<u32>;
@group(0) @binding(5) var<storage, read> airPrefixSum: array<u32>;
@group(0) @binding(6) var<storage, read> oceanInput: array<Particle>;
@group(0) @binding(7) var<storage, read_write> oceanState: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> oceanSteer: array<vec4f>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&oceanState)) {
        return;
    }
    if (bio.enabled <= 0.5) {
        oceanState[pi] = vec4f(0.65, 0.0, 0.0, 0.0);
        oceanSteer[pi] = vec4f(0.0);
        return;
    }

    let p = oceanInput[pi];
    let state = oceanState[pi];
    var acc = BioOceanAccum(0.0, 0.0, 0.0, vec3f(0.0), 0.0);

    let worldP = p.pos * bio.worldSize;
    let cmin = vec3i(floor((worldP - bio.catRadius) / bio.airCellSize));
    let cmax = vec3i(floor((worldP + bio.catRadius) / bio.airCellSize));
    let numCells = max(u32(bio.airCellAmt), 1u);
    let nCol = max(u32(bio.airColours), 1u);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                let cnt = airCounts[c];
                if (cnt == 0u) {
                    continue;
                }
                let startIdx = airPrefixSum[c];
                let endIdx = startIdx + cnt;
                for (var i = startIdx; i < endIdx; i++) {
                    let atom = airSorted[i];
                    let sc = min(u32(atom.colour), nCol - 1u);
                    let sBase = sc * SPECIES_STRIDE;
                    let valence = u32(airMatrix[sBase + 0u]);
                    let atomicNum = u32(airMatrix[sBase + 7u]);
                    let vLoad = airValence[atom.idx];
                    let atomLoad = vLoad.x + vLoad.y + vLoad.z;
                    accumulateBioOceanNeighbor(
                        p.pos,
                        p.colour,
                        state,
                        atom.pos,
                        atomicNum,
                        valence,
                        atomLoad,
                        bio,
                        &acc,
                    );
                }
            }
        }
    }

    let nextState = finalizeBioOceanState(state, acc, bio);
    oceanState[pi] = nextState;
    oceanSteer[pi] = finalizeBioOceanSteer(acc, nextState.w, bio);
}
`;
};
