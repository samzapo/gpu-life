import structsShader from '../../shaders/structs.wgsl?raw';
import projectionShader from '../../shaders/projection.wgsl?raw';
import bioChemShader from '../../shaders/bioChem.wgsl?raw';
import valenceSimShader from '../../shaders/valenceSim.wgsl?raw';

export const valenceComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}
${projectionShader}
${bioChemShader}
${valenceSimShader}

@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read_write> valenceOut: array<vec4f>;

@group(0) @binding(5) var<storage, read> sorted: array<SortParticle>;
@group(0) @binding(6) var<storage, read> counts: array<u32>;
@group(0) @binding(7) var<storage, read> prefixSumIndices: array<u32>;

fn getCellValence(
    pi: u32,
    p: Particle,
    sP: SpeciesInfo,
    cell: u32,
    slotAcc: ptr<function, vec4f>,
) {
    let particlesInThisCell = counts[cell];
    if (particlesInThisCell == 0u) {
        return;
    }

    let startIdx = prefixSumIndices[cell];
    let endIdx = startIdx + particlesInThisCell;
    for (var i = startIdx; i < endIdx; i++) {
        let ip = sorted[i];
        if (ip.idx == pi) {
            continue;
        }
        accumulateValenceNeighbor(p, sP, ip.pos, ip.colour, ip.orient, slotAcc);
    }
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&valenceOut)) {
        return;
    }
    let p = input[pi];
    let sP = getSpeciesInfo(u32(p.colour));
    if (u32(sP.valence) == 0u) {
        valenceOut[pi] = vec4f(0.0);
        return;
    }

    var slotAcc = vec4f(0.0);
    let worldP = p.pos * sim.worldSize;
    let cmin = vec3i(floor((worldP - sim.rMax) / sim.cellSize));
    let cmax = vec3i(floor((worldP + sim.rMax) / sim.cellSize));
    let numCells = u32(sim.cellAmt);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                getCellValence(pi, p, sP, c, &slotAcc);
            }
        }
    }

    valenceOut[pi] = slotAcc;
}
`;
};
