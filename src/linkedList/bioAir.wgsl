struct Heads {
    num: u32,
    data: array<u32>
}

struct LinkedList {
    data: array<ListParticle>
}

@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var<storage, read> airMatrix: array<f32>;
@group(0) @binding(2) var<storage, read> airInput: array<Particle>;
@group(0) @binding(3) var<storage, read> airValence: array<vec4f>;
@group(0) @binding(4) var<storage, read> oceanHeads: Heads;
@group(0) @binding(5) var<storage, read> oceanLinkedList: LinkedList;
@group(0) @binding(6) var<storage, read> oceanState: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> airBioReact: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&airBioReact)) {
        return;
    }
    if (bio.enabled <= 0.5) {
        airBioReact[pi] = vec4f(0.0);
        return;
    }

    let p = airInput[pi];
    let nCol = max(u32(bio.airColours), 1u);
    let sc = min(u32(p.colour), nCol - 1u);
    let sBase = sc * SPECIES_STRIDE;
    let valence = u32(airMatrix[sBase + 0u]);
    let atomicNum = u32(airMatrix[sBase + 7u]);

    if (bioRoleKind(atomicNum, valence) == 0u) {
        airBioReact[pi] = vec4f(0.0);
        return;
    }

    let atomLoad = airValence[pi].w;
    var acc = BioAirAccum(vec3f(0.0), 1.0, 0.0, 0.0);

    let worldP = p.pos * bio.worldSize;
    let cmin = vec3i(floor((worldP - bio.catRadius) / bio.oceanCellSize));
    let cmax = vec3i(floor((worldP + bio.catRadius) / bio.oceanCellSize));
    let headCount = arrayLength(&oceanHeads.data);
    const maxParticles = 5000u;

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % headCount;
                var elementIndex = oceanHeads.data[c];
                var nSeen = 0u;
                while (elementIndex != 0xFFFFFFFFu && nSeen < maxParticles) {
                    let life = oceanLinkedList.data[elementIndex];
                    accumulateBioAirNeighbor(
                        p.pos,
                        atomicNum,
                        valence,
                        atomLoad,
                        life.pos,
                        life.colour,
                        oceanState[life.idx],
                        bio,
                        &acc,
                    );
                    nSeen++;
                    elementIndex = life.next;
                }
            }
        }
    }

    airBioReact[pi] = finalizeBioAir(acc, bio);
}
