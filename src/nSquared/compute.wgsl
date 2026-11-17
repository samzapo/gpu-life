@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read_write> output: array<Particle>;

fn getForce(pi: u32) -> ForceResult {
    let p = input[pi];
    var acc = ForceAccum();
    let l = arrayLength(&input);

    for (var i = 0u; i < l; i++) {
        if (i == pi) {
            continue;
        }
        let ip = input[i];
        accumulateNeighbor(pi, p, ip.pos, ip.colour, ip.theta, &acc);
    }

    return finalizeForces(acc);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&output)) {
        return;
    }
    let p = input[global_id.x];
    let res = getForce(global_id.x);
    output[global_id.x] = integrateParticle(global_id.x, p, res);
}
