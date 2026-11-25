@group(0) @binding(0) var<uniform> bio: BioLink;
@group(0) @binding(1) var oceanTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<Particle>;
@group(0) @binding(4) var<storage, read> lifeStateIn: array<LifeMdpState>;
@group(0) @binding(5) var<storage, read_write> lifeStateOut: array<LifeMdpState>;
@group(0) @binding(6) var<storage, read> valence: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> bioForceOut: array<vec4f>;
@group(0) @binding(8) var<storage, read> sorted: array<SortParticle>;
@group(0) @binding(9) var<storage, read> indices: array<vec2<u32>>;
@group(0) @binding(10) var windTex: texture_2d<f32>;
@group(0) @binding(11) var tempTex: texture_2d<f32>;
@group(0) @binding(12) var uvTex: texture_2d<f32>;
@group(0) @binding(13) var hydroTex: texture_2d<f32>;
@group(0) @binding(14) var terrainTex: texture_2d<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&lifeStateOut)) {
        return;
    }

    let p = input[pi];
    let curEntry = lifeStateIn[pi];
    let curLife = curEntry.life;
    let myPhase = unpackPhase(curLife.w);
    let curMdp = MdpState(curEntry.qPack, curEntry.ctrl);
    let nChem = max(u32(bio.chemColours), 1u);
    let mySc = min(u32(p.colour), nChem - 1u);
    let mySBase = mySc * SPECIES_STRIDE;
    let myValence = u32(matrix[mySBase + 0u]);
    let myAtomicNum = u32(matrix[mySBase + 7u]);
    let myV = valence[pi];
    let myAtomLoad = select(0.0, myV.x + myV.y + myV.z, myPhase == 0u);
    let envDims = textureDimensions(oceanTex, 0);
    let eq = makeEquirectCoord(p.pos, envDims);
    let windRaw = sampleWindTextureCoord(windTex, eq);
    let oc = sampleOceanTextureCoord(oceanTex, eq);
    let tSample = sampleTempTextureCoord(tempTex, eq);
    let uvSample = sampleUvTextureCoord(uvTex, eq);
    let hydroSample = sampleHydroTextureCoord(hydroTex, eq);
    let terrainSample = sampleTerrainTextureCoord(terrainTex, eq);

    if (bio.enabled <= 0.5 && bio.mineralEnabled <= 0.5) {
        let zeroGrads = BioEnvGrads(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        let resOff = finalizeBioUnified(
            pi, p.pos, p.vel, myAtomicNum, myValence, myAtomLoad, curLife, p._pad, curMdp, windRaw, oc, tSample, uvSample, hydroSample, terrainSample, zeroGrads, initBioUnifiedAccum(), bio,
        );
        lifeStateOut[pi] = LifeMdpState(resOff.nextLifeState, resOff.nextMdpState.qPack, resOff.nextMdpState.ctrl);
        bioForceOut[pi] = resOff.bioForce;
        return;
    }

    var acc = initBioUnifiedAccum();
    let worldP = p.pos * bio.worldSize;
    let cmin = vec3i(floor((worldP - bio.catRadius) / bio.cellSize));
    let cmax = vec3i(floor((worldP + bio.catRadius) / bio.cellSize));
    let numCells = max(u32(bio.cellAmt), 1u);
    let maxAng = bio.catRadius / max(bio.worldSize, 0.25);
    let maxChord2 = maxAng * maxAng;

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                let idxRange = indices[c];
                if (idxRange.y == 0u) {
                    continue;
                }
                for (var i = idxRange.x; i < idxRange.x + idxRange.y; i++) {
                    let ip = sorted[i];
                    if (ip.idx == pi) {
                        continue;
                    }
                    let dVec = ip.pos - p.pos;
                    let chord2 = dot(dVec, dVec);
                    if (chord2 <= 1e-12 || chord2 >= maxChord2) {
                        continue;
                    }
                    let nbSc = min(u32(ip.colour), nChem - 1u);
                    let nbSBase = nbSc * SPECIES_STRIDE;
                    let nbValence = u32(matrix[nbSBase + 0u]);
                    let nbAtomicNum = u32(matrix[nbSBase + 7u]);
                    let nbV = valence[ip.idx];
                    let nbAtomLoad = select(0.0, nbV.x + nbV.y + nbV.z, ip.phase == 0u);
                    let nbEntry = lifeStateIn[ip.idx];
                    let nbLife = nbEntry.life;
                    let nbPart = input[ip.idx];
                    var nbMdp = emptyMdpState();
                    if (myPhase == 1u && ip.phase == 1u) {
                        nbMdp = MdpState(nbEntry.qPack, nbEntry.ctrl);
                    }
                    accumulateBioUnifiedNeighbor(
                        p.pos,
                        p.vel,
                        myPhase,
                        myAtomicNum,
                        myValence,
                        myAtomLoad,
                        curLife,
                        ip.pos,
                        nbPart.vel,
                        ip.colour,
                        ip.phase,
                        ip.bioSpecies,
                        nbLife,
                        nbAtomLoad,
                        nbAtomicNum,
                        nbValence,
                        nbPart._pad,
                        nbMdp,
                        oc,
                        bio,
                        &acc,
                    );
                }
            }
        }
    }

    let envGrads = sampleBioEnvGradsCoord(p.pos, eq, envDims, oceanTex, tempTex, uvTex, hydroTex, myPhase);
    let res = finalizeBioUnified(
        pi,
        p.pos,
        p.vel,
        myAtomicNum,
        myValence,
        myAtomLoad,
        curLife,
        p._pad,
        curMdp,
        windRaw,
        oc,
        tSample,
        uvSample,
        hydroSample,
        terrainSample,
        envGrads,
        acc,
        bio,
    );
    lifeStateOut[pi] = LifeMdpState(res.nextLifeState, res.nextMdpState.qPack, res.nextMdpState.ctrl);
    bioForceOut[pi] = res.bioForce;
}
