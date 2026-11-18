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
@group(0) @binding(4) var<storage, read_write> output: array<Particle>;

@group(0) @binding(5) var<storage, read> heads: Heads;
@group(0) @binding(6) var<storage, read> linkedList: LinkedList;
@group(0) @binding(7) var<storage, read> valence: array<vec4f>;
@group(0) @binding(8) var windTex: texture_2d<f32>;
@group(0) @binding(9) var<storage, read> airBioReact: array<vec4f>;

fn getCellForce(
    pi: u32,
    p: Particle,
    sP: SpeciesInfo,
    mySlots: vec4f,
    actScale: f32,
    bioCleave: f32,
    shearStress: f32,
    cell: u32,
    acc: ptr<function, ForceAccum>,
) {
    const maxParticles = 5000u;

    var numParticles = 0u;
    var elementIndex = heads.data[cell];

    while (elementIndex != 0xFFFFFFFFu && numParticles < maxParticles) {
        let ip = linkedList.data[elementIndex];
        if (ip.idx != pi) {
            accumulateSimNeighbor(
                pi,
                p,
                sP,
                mySlots,
                ip.pos,
                ip.colour,
                ip.orient,
                valence[ip.idx],
                actScale,
                bioCleave,
                shearStress,
                acc,
            );
        }

        numParticles++;
        elementIndex = ip.next;
    }
}

fn getForce(pi: u32, actScale: f32, bioCleave: f32, shearStress: f32) -> ForceResult {
    let p = input[pi];
    let sP = getSpeciesInfo(u32(p.colour));
    let mySlots = valence[pi];
    var acc = ForceAccum();

    let worldP = p.pos * sim.worldSize;
    let cmin = vec3i(floor((worldP - sim.rMax) / sim.cellSize));
    let cmax = vec3i(floor((worldP + sim.rMax) / sim.cellSize));
    let headCount = arrayLength(&heads.data);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % headCount;
                getCellForce(pi, p, sP, mySlots, actScale, bioCleave, shearStress, c, &acc);
            }
        }
    }

    return finalizeForces(acc);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&output)) {
        return;
    }
    let p = input[pi];
    let mySlots = valence[pi];
    let bioReact = airBioReact[pi];
    let actGlow = unpackActGlow(bioReact.w);
    let actScale = actGlow.x;
    let bioCleave = clamp(-actGlow.y, 0.0, 1.0);
    let windSample = evalWindAndShear(windTex, p.pos, uniforms.flags.w);
    let shearStress = clamp(max(0.0, windSample.shearRel - sim.shearFloor) * 4.5 * max(sim.shearBreak, 0.0), 0.0, 1.0);
    let res = getForce(pi, actScale, bioCleave, shearStress);
    output[pi] = integrateParticle(
        pi,
        p,
        mySlots,
        res,
        windSample,
        shearStress,
        uniforms.mouseSphere,
        bioReact,
    );
}
