import structsShader from '../../shaders/structs.wgsl?raw';

export const cellComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}

@group(0) @binding(0) var<uniform> sim: Sim;

// Particle Buffer:
@group(0) @binding(1) var<storage, read> input: array<Particle>;
// Cell Buffer:
@group(0) @binding(2) var<storage, read_write> output: array<SortParticle>;

@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&input)) {
        return;
    }

    let p = input[global_id.x];
    let gridPos = vec2i(floor(p.pos / sim.cellSize));
    let cellHash = hash3i(vec3i(gridPos, 0)) % u32(sim.cellAmt);

    var op = SortParticle();
    op.idx = vec2u(cellHash, global_id.x);
    op.pos = p.pos;
    op.colour = p.colour;
    op.theta = p.theta;

    output[global_id.x] = op;

    atomicAdd(&counts[cellHash], 1u);
}`;
};
