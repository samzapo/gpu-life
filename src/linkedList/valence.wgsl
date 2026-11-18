struct Heads {
    num: u32,
    data: array<u32>
}

struct LinkedList {
    data: array<ListParticle>
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read_write> valenceOut: array<vec4f>;

@group(0) @binding(5) var<storage, read> heads: Heads;
@group(0) @binding(6) var<storage, read> linkedList: LinkedList;

fn getCellValence(pi: u32, p: Particle, sP: SpeciesInfo, cell: u32, slotAcc: ptr<function, vec4f>) {
    const maxParticles = 5000u;

    var numParticles = 0u;
    var elementIndex = heads.data[cell];

    while (elementIndex != 0xFFFFFFFFu && numParticles < maxParticles) {
        let ip = linkedList.data[elementIndex];
        if (ip.idx != pi) {
            accumulateValenceNeighbor(p, sP, ip.pos, ip.colour, ip.theta, slotAcc);
        }
        numParticles++;
        elementIndex = ip.next;
    }
}

@compute @workgroup_size(64)
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
    let cxmin = i32(floor((p.pos.x - sim.rMax) / sim.cellSize));
    let cymin = i32(floor((p.pos.y - sim.rMax) / sim.cellSize));
    let cxmax = i32(floor((p.pos.x + sim.rMax) / sim.cellSize));
    let cymax = i32(floor((p.pos.y + sim.rMax) / sim.cellSize));
    let headCount = arrayLength(&heads.data);

    for (var cx = cxmin; cx <= cxmax; cx++) {
        for (var cy = cymin; cy <= cymax; cy++) {
            let c = hash3i(vec3i(cx, cy, 0)) % headCount;
            getCellValence(pi, p, sP, c, &slotAcc);
        }
    }

    valenceOut[pi] = slotAcc;
}
