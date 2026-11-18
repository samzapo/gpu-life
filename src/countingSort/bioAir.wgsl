@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var<storage, read> airMatrix: array<f32>;
@group(0) @binding(2) var<storage, read> airInput: array<Particle>;
@group(0) @binding(3) var<storage, read> airValence: array<vec4f>;
@group(0) @binding(4) var<storage, read> oceanSorted: array<SortParticle>;
@group(0) @binding(5) var<storage, read> oceanIndices: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> oceanState: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> airBioReact: array<vec4f>;

@compute @workgroup_size(128)
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
    let numCells = max(u32(bio.oceanCellAmt), 1u);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                let index = oceanIndices[c];
                if (index.y == 0u) {
                    continue;
                }
                for (var i = index.x; i < index.x + index.y; i++) {
                    let life = oceanSorted[i];
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
                }
            }
        }
    }

    airBioReact[pi] = finalizeBioAir(acc, bio);
}
