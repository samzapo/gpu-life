@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read_write> valenceOut: array<vec4f>;

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
    let l = arrayLength(&input);
    for (var i = 0u; i < l; i++) {
        if (i == pi) {
            continue;
        }
        let ip = input[i];
        accumulateValenceNeighbor(p, sP, ip.pos, ip.colour, ip.theta, &slotAcc);
    }

    valenceOut[pi] = slotAcc;
}
