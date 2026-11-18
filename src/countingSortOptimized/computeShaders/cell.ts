import structsShader from '../../shaders/structs.wgsl?raw';

export const cellComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> input: array<Particle>;
@group(0) @binding(2) var<storage, read> lifeState: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> output: array<SortParticle>;
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&input)) {
        return;
    }

    let p = input[pi];
    let ls = lifeState[pi];
    let gridPos = vec3i(floor((p.pos * sim.worldSize) / sim.cellSize));
    let cellHash = hash3i(gridPos) % u32(sim.cellAmt);

    var op = SortParticle();
    op.cellIdx = cellHash;
    op.idx = pi;
    op.pos = p.pos;
    op.colour = p.colour;
    op.orient = p.orient;
    op.phase = unpackPhase(ls.w);
    op.bioSpecies = unpackBioSpecies(ls.w);
    op.bioEnergy = ls.x;

    output[pi] = op;

    atomicAdd(&counts[cellHash], 1u);
}`;
};
