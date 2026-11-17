import structsShader from '../../shaders/structs.wgsl?raw';
import covalentSimShader from '../../shaders/covalentSim.wgsl?raw';

export const simComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}
${covalentSimShader}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

// Current Particle Buffer
@group(0) @binding(3) var<storage, read> input: array<Particle>;
// Next Particle Buffer
@group(0) @binding(4) var<storage, read_write> output: array<Particle>;

@group(0) @binding(5) var<storage, read> sorted: array<SortParticle>;
// The number of particles in each cell (counts generated in cell.wgsl)
@group(0) @binding(6) var<storage, read> counts: array<u32>;
// The index at which a cell's particles first appear
// Generated from the exclusive prefix sum of counts
@group(0) @binding(7) var<storage, read> prefixSumIndices: array<u32>;

fn getCellForce(
    pi: u32,
    p: Particle,
    cell: u32,
    acc: ptr<function, ForceAccum>,
) {
    let particlesInThisCell = counts[cell];

    if (particlesInThisCell == 0u) {
        return;
    }

    let particlesThatCameBeforeThisCell = prefixSumIndices[cell];
    let particleEnd = particlesThatCameBeforeThisCell + particlesInThisCell;

    for (
        var i = particlesThatCameBeforeThisCell;
        i < particleEnd;
        i++
    ) {
        let ip = sorted[i];
        if (ip.idx.y == pi) {
            continue;
        }
        accumulateNeighbor(pi, p, ip.pos, ip.colour, ip.theta, acc);
    }
}

fn getForce(pi: u32) -> ForceResult {
    let p = input[pi];
    var acc = ForceAccum();

    let cxmin = i32(floor((p.pos.x - sim.rMax) / sim.cellSize));
    let cymin = i32(floor((p.pos.y - sim.rMax) / sim.cellSize));
    let cxmax = i32(floor((p.pos.x + sim.rMax) / sim.cellSize));
    let cymax = i32(floor((p.pos.y + sim.rMax) / sim.cellSize));

    let numCells = u32(sim.cellAmt);

    for (var cx = cxmin; cx <= cxmax; cx++) {
        for (var cy = cymin; cy <= cymax; cy++) {
            let c = hash3i(vec3i(cx, cy, 0)) % numCells;
            getCellForce(pi, p, c, &acc);
        }
    }

    return finalizeForces(acc);
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&output)) {
        return;
    }
    let p = input[global_id.x];
    let res = getForce(global_id.x);
    output[global_id.x] = integrateParticle(global_id.x, p, res);
}
`;
};
