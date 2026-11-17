@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read_write> output: array<Particle>;

@group(0) @binding(5) var<storage, read> sorted: array<SortParticle>;
@group(0) @binding(6) var<storage, read> indices: array<vec2<u32>>;

fn getCellForce(pi: u32, p: Particle, cell: u32, acc: ptr<function, ForceAccum>) {
    let index = indices[cell];

    if (index.y == 0u) {
        return;
    }

    for (var i = index.x; i < index.x + index.y; i++) {
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

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&output)) {
        return;
    }
    let p = input[global_id.x];
    let res = getForce(global_id.x);
    output[global_id.x] = integrateParticle(global_id.x, p, res);
}
