struct Heads {
    num: atomic<u32>,
    data: array<atomic<u32>>
}

struct LinkedList {
    data: array<ListParticle>
}

@group(0) @binding(0) var<uniform> sim: Sim;

@group(0) @binding(1) var<storage, read> input: array<Particle>;

@group(0) @binding(2) var<storage, read_write> heads: Heads;
@group(0) @binding(3) var<storage, read_write> linkedList: LinkedList;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&input)) {
        return;
    }
    let p = input[global_id.x];

    let gridPos = vec2i(floor(p.pos / sim.cellSize));
    let cellHash = hash3i(vec3i(gridPos, 0)) % arrayLength(&heads.data);
    let listIndex = atomicAdd(&heads.num, 1u);

    if (listIndex < arrayLength(&input)) {
        let lastHead = atomicExchange(&heads.data[cellHash], listIndex);
        linkedList.data[listIndex].pos = p.pos;
        linkedList.data[listIndex].colour = p.colour;
        linkedList.data[listIndex].theta = p.theta;
        linkedList.data[listIndex].idx = global_id.x;
        linkedList.data[listIndex].next = lastHead;
    }
}
