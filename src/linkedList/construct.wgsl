struct Heads {
    num: atomic<u32>,
    data: array<atomic<u32>>
}

struct LinkedList {
    data: array<ListParticle>
}

@group(0) @binding(0) var<uniform> sim: Sim;

@group(0) @binding(1) var<storage, read> input: array<Particle>;
@group(0) @binding(2) var<storage, read> lifeState: array<LifeMdpState>;

@group(0) @binding(3) var<storage, read_write> heads: Heads;
@group(0) @binding(4) var<storage, read_write> linkedList: LinkedList;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&input)) {
        return;
    }
    let p = input[pi];
    let ls = lifeState[pi].life;

    let gridPos = vec3i(floor((p.pos * sim.worldSize) / sim.cellSize));
    let cellHash = hash3i(gridPos) % arrayLength(&heads.data);
    let listIndex = atomicAdd(&heads.num, 1u);

    if (listIndex < arrayLength(&input)) {
        let lastHead = atomicExchange(&heads.data[cellHash], listIndex);
        linkedList.data[listIndex].pos = p.pos;
        linkedList.data[listIndex].colour = p.colour;
        linkedList.data[listIndex].orient = p.orient;
        linkedList.data[listIndex].idx = pi;
        linkedList.data[listIndex].next = lastHead;
        linkedList.data[listIndex].phase = unpackPhase(ls.w);
        linkedList.data[listIndex].bioSpecies = unpackBioSpecies(ls.w);
        linkedList.data[listIndex].bioEnergy = ls.x;
    }
}
