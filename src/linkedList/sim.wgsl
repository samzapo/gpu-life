struct HeadsRead {
    num: u32,
    data: array<u32>
}

struct LinkedList {
    data: array<ListParticle>
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<uniform> oceanSim: OceanSim;
@group(0) @binding(3) var windTex: texture_2d<f32>;
@group(0) @binding(4) var oceanTex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> matrix: array<f32>;
@group(0) @binding(6) var<storage, read> input: array<Particle>;
@group(0) @binding(7) var<storage, read_write> output: array<Particle>;
@group(0) @binding(8) var<storage, read> lifeStateNext: array<LifeMdpState>;
@group(0) @binding(9) var<storage, read> valence: array<vec4f>;
@group(0) @binding(10) var<storage, read> bioForce: array<vec4f>;
@group(0) @binding(11) var<storage, read> heads: HeadsRead;
@group(0) @binding(12) var<storage, read> linkedList: LinkedList;
@group(0) @binding(13) var tempTex: texture_2d<f32>;
@group(0) @binding(14) var uvTex: texture_2d<f32>;
@group(0) @binding(15) var hydroTex: texture_2d<f32>;
@group(0) @binding(16) var terrainTex: texture_2d<f32>;

fn getChemCellForce(
    pi: u32,
    p: Particle,
    sP: SpeciesInfo,
    mySlots: vec4f,
    actScale: f32,
    bioCleave: f32,
    shearStress: f32,
    tempWellMod: f32,
    maxChord2: f32,
    cell: u32,
    acc: ptr<function, ForceAccum>,
) {
    var elementIndex = heads.data[cell];
    while (elementIndex != 0xffffffffu && elementIndex < arrayLength(&linkedList.data)) {
        let ip = linkedList.data[elementIndex];
        if (ip.idx != pi && ip.phase == 0u) {
            let dVec = ip.pos - p.pos;
            if (dot(dVec, dVec) < maxChord2) {
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
                    tempWellMod,
                    acc,
                );
            }
        }
        elementIndex = ip.next;
    }
}

fn getChemForce(pi: u32, p: Particle, actScale: f32, bioCleave: f32, shearStress: f32, tempWellMod: f32) -> ForceResult {
    let sP = getSpeciesInfo(u32(p.colour));
    let mySlots = valence[pi];
    var acc = ForceAccum();

    let worldP = p.pos * sim.worldSize;
    let cmin = vec3i(floor((worldP - sim.rMax) / sim.cellSize));
    let cmax = vec3i(floor((worldP + sim.rMax) / sim.cellSize));
    let numCells = arrayLength(&heads.data);
    let maxAng = sim.rMax / max(sim.worldSize, 0.25);
    let maxChord2 = maxAng * maxAng;

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                getChemCellForce(pi, p, sP, mySlots, actScale, bioCleave, shearStress, tempWellMod, maxChord2, c, &acc);
            }
        }
    }

    return finalizeForces(acc);
}

fn getBioCellForce(
    pi: u32,
    p: Particle,
    myPhase: u32,
    myKin: f32,
    pol: MdpLeverProfile,
    nChem: u32,
    oceanBase: u32,
    rowOffset: u32,
    nBio: u32,
    cell: u32,
    acc: ptr<function, OceanForceAccum>,
) {
    var elementIndex = heads.data[cell];
    while (elementIndex != 0xffffffffu && elementIndex < arrayLength(&linkedList.data)) {
        let ip = linkedList.data[elementIndex];
        if (ip.idx != pi) {
            let nbLs = lifeStateNext[ip.idx].life;
            let nbPhase = unpackPhase(nbLs.w);
            var skipNb = false;
            var nbChemRole = 0u;
            var nbKin = 0.0;
            var a = 0.0;
            if (nbPhase == 0u) {
                if (myPhase != 1u) {
                    skipNb = true;
                } else {
                    let nbSc = min(u32(ip.colour), nChem - 1u);
                    let nbSBase = nbSc * SPECIES_STRIDE;
                    nbChemRole = bioRoleKind(u32(matrix[nbSBase + 7u]), u32(matrix[nbSBase + 0u]));
                    if (nbChemRole == 0u) {
                        skipNb = true;
                    }
                }
            } else {
                nbKin = unpackKinPhase(nbLs.w);
                if (myPhase == 1u && nbPhase == 1u) {
                    let colIdx = min(unpackBioSpecies(nbLs.w), nBio - 1u);
                    a = matrix[oceanBase + rowOffset + colIdx];
                }
            }
            if (!skipNb) {
                accumulateUnifiedEmergentNeighbor(
                    pi,
                    p,
                    myPhase,
                    myKin,
                    pol,
                    ip.pos,
                    ip.orient,
                    nbPhase,
                    nbChemRole,
                    nbKin,
                    a,
                    acc,
                    oceanSim,
                );
            }
        }
        elementIndex = ip.next;
    }
}

fn getBioForce(pi: u32, p: Particle, myPhase: u32, bioSp: u32, myKin: f32, packedBlobZ: u32) -> ForceResult {
    let nChem = max(u32(sim.colours), 1u);
    let nBio = max(u32(oceanSim.colours), 1u);
    let oceanBase = nChem * SPECIES_STRIDE + nChem * nChem;
    let rowOffset = min(bioSp, nBio - 1u) * nBio;
    let pol = evalCombinedMdpPolicy(packedBlobZ);
    var acc = OceanForceAccum();

    let worldP = p.pos * oceanSim.worldSize;
    let cmin = vec3i(floor((worldP - oceanSim.rMax) / sim.cellSize));
    let cmax = vec3i(floor((worldP + oceanSim.rMax) / sim.cellSize));
    let numCells = arrayLength(&heads.data);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                getBioCellForce(pi, p, myPhase, myKin, pol, nChem, oceanBase, rowOffset, nBio, c, &acc);
            }
        }
    }

    return ForceResult(finalizeOceanForce(acc, oceanSim), acc.torque);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&output)) {
        return;
    }

    let p = input[pi];
    let myEntry = lifeStateNext[pi];
    let ls = myEntry.life;
    let myPhase = unpackPhase(ls.w);
    if (myPhase == 3u) {
        output[pi] = integrateMineralParticle(p, ls, bioForce[pi]);
        return;
    }

    let eq = makeEquirectCoord(p.pos, textureDimensions(oceanTex, 0));
    let tSample = sampleTempTextureCoord(tempTex, eq);
    let oc = sampleOceanTextureCoord(oceanTex, eq);
    let terrainSample = sampleTerrainTextureCoord(terrainTex, eq);
    let localTempScale = uniforms.mineralParams.z;

    if (myPhase == 1u || myPhase == 2u) {
        let bioSp = unpackBioSpecies(ls.w);
        let myKin = unpackKinPhase(ls.w);
        let resBio = getBioForce(pi, p, myPhase, bioSp, myKin, myEntry.blob.z);
        let windRaw = sampleWindTextureCoord(windTex, eq);
        output[pi] = integrateOceanParticle(
            pi,
            p,
            resBio.force,
            resBio.torque,
            myPhase,
            oceanSim,
            oc,
            windRaw,
            sim.windStrength,
            uniforms.flags.w,
            terrainSample,
            tSample,
            localTempScale,
            uniforms.mouseSphere,
            ls,
            bioForce[pi],
            sim.frame,
            myEntry.blob.x,
            myEntry.blob.z,
        );
        return;
    }

    let mySlots = valence[pi];
    let bReact = bioForce[pi];
    let actGlow = unpackActGlow(bReact.w);
    let airTherm = evalAirTempChemFactors(tSample.x, localTempScale);
    let uvS = sampleUvTextureCoord(uvTex, eq);
    let uvChem = evalUvChemFactors(uvS.x, uniforms.uvParams.y);
    let hydroS = sampleHydroTextureCoord(hydroTex, eq);
    let hydroChem = evalHydroChemFactors(hydroS.x, hydroS.y, uniforms.hydroParams.z);
    let actScale = actGlow.x * airTherm.x * uvChem.x * hydroChem.x;
    let bioCleave = clamp(-actGlow.y, 0.0, 1.0);
    let windSample = evalWindAndShear(windTex, p.pos, uniforms.flags.w);
    let shearStress = clamp(max(0.0, windSample.shearRel - sim.shearFloor) * 4.5 * max(sim.shearBreak, 0.0), 0.0, 1.0);
    let res = getChemForce(pi, p, actScale, bioCleave, shearStress, airTherm.y * uvChem.y * hydroChem.y);
    output[pi] = integrateParticle(
        pi,
        p,
        mySlots,
        res,
        windSample,
        oc,
        terrainSample,
        ls,
        oceanSim.currentStrength,
        shearStress,
        uniforms.mouseSphere,
        bReact,
    );
}
