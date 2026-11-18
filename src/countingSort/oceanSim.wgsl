@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: OceanSim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read_write> output: array<Particle>;

@group(0) @binding(5) var<storage, read> sorted: array<SortParticle>;
@group(0) @binding(6) var<storage, read> indices: array<vec2<u32>>;
@group(0) @binding(7) var oceanTex: texture_2d<f32>;

fn getOceanCellForce(
    pi: u32,
    p: Particle,
    cell: u32,
    acc: ptr<function, OceanForceAccum>,
) {
    let index = indices[cell];
    if (index.y == 0u) {
        return;
    }

    let nCol = max(u32(sim.colours), 1u);
    let row = min(u32(p.colour), nCol - 1u) * nCol;

    for (var i = index.x; i < index.x + index.y; i++) {
        let ip = sorted[i];
        if (ip.idx == pi) {
            continue;
        }
        let colIdx = min(u32(ip.colour), nCol - 1u);
        let a = matrix[row + colIdx];
        accumulateOceanNeighbor(pi, p, ip.pos, a, acc, sim);
    }
}

fn getOceanForce(pi: u32) -> vec3f {
    let p = input[pi];
    var acc = OceanForceAccum();

    let worldP = p.pos * sim.worldSize;
    let cmin = vec3i(floor((worldP - sim.rMax) / sim.cellSize));
    let cmax = vec3i(floor((worldP + sim.rMax) / sim.cellSize));
    let numCells = u32(sim.cellAmt);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                getOceanCellForce(pi, p, c, &acc);
            }
        }
    }

    return finalizeOceanForce(acc, sim);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&output)) {
        return;
    }
    let p = input[pi];
    let force = getOceanForce(pi);
    output[pi] = integrateOceanParticle(
        pi,
        p,
        force,
        sim,
        oceanTex,
        uniforms.mouseSphere,
    );
}
