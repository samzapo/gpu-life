import structsShader from '../../shaders/structs.wgsl?raw';
import projectionShader from '../../shaders/projection.wgsl?raw';
import bioChemShader from '../../shaders/bioChem.wgsl?raw';
import valenceSimShader from '../../shaders/valenceSim.wgsl?raw';
import oceanSimShader from '../../shaders/oceanSim.wgsl?raw';

export const simComputeShader = (workgroupSize: number) => {
  return /* wgsl */ `
${structsShader}
${projectionShader}
${bioChemShader}
${valenceSimShader}
${oceanSimShader}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<uniform> oceanSim: OceanSim;
@group(0) @binding(3) var windTex: texture_2d<f32>;
@group(0) @binding(4) var oceanTex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> matrix: array<f32>;
@group(0) @binding(6) var<storage, read> input: array<Particle>;
@group(0) @binding(7) var<storage, read_write> output: array<Particle>;
@group(0) @binding(8) var<storage, read> lifeStateNext: array<vec4f>;
@group(0) @binding(9) var<storage, read> valence: array<vec4f>;
@group(0) @binding(10) var<storage, read> bioForce: array<vec4f>;
@group(0) @binding(11) var<storage, read> sorted: array<SortParticle>;
@group(0) @binding(12) var<storage, read> prefixSumIndices: array<u32>;

fn getChemCellForce(
    pi: u32,
    p: Particle,
    sP: SpeciesInfo,
    mySlots: vec4f,
    actScale: f32,
    bioCleave: f32,
    shearStress: f32,
    cell: u32,
    acc: ptr<function, ForceAccum>,
) {
    let startIdx = prefixSumIndices[cell];
    let particleEnd = prefixSumIndices[cell + 1u];

    for (var i = startIdx; i < particleEnd; i++) {
        let ip = sorted[i];
        if (ip.idx == pi || ip.phase != 0u) {
            continue;
        }
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
            acc,
        );
    }
}

fn getChemForce(pi: u32, p: Particle, actScale: f32, bioCleave: f32, shearStress: f32) -> ForceResult {
    let sP = getSpeciesInfo(u32(p.colour));
    let mySlots = valence[pi];
    var acc = ForceAccum();

    let worldP = p.pos * sim.worldSize;
    let cmin = vec3i(floor((worldP - sim.rMax) / sim.cellSize));
    let cmax = vec3i(floor((worldP + sim.rMax) / sim.cellSize));
    let numCells = u32(sim.cellAmt);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                getChemCellForce(pi, p, sP, mySlots, actScale, bioCleave, shearStress, c, &acc);
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
    oceanBase: u32,
    rowOffset: u32,
    nBio: u32,
    cell: u32,
    acc: ptr<function, OceanForceAccum>,
) {
    let startIdx = prefixSumIndices[cell];
    let particleEnd = prefixSumIndices[cell + 1u];

    for (var i = startIdx; i < particleEnd; i++) {
        let ip = sorted[i];
        if (ip.idx == pi) {
            continue;
        }
        let nbLs = lifeStateNext[ip.idx];
        let nbPhase = unpackPhase(nbLs.w);
        if (nbPhase == 0u) {
            continue;
        }
        let nbKin = unpackKinPhase(nbLs.w);
        let colIdx = min(unpackBioSpecies(nbLs.w), nBio - 1u);
        let a = select(0.0, matrix[oceanBase + rowOffset + colIdx], myPhase == 1u && nbPhase == 1u);
        accumulateOceanOrLipidNeighbor(
            pi,
            p,
            myPhase,
            myKin,
            ip.pos,
            ip.orient,
            nbPhase,
            nbKin,
            a,
            acc,
            oceanSim,
        );
    }
}

fn getBioForce(pi: u32, p: Particle, myPhase: u32, bioSp: u32, myKin: f32) -> ForceResult {
    let nChem = max(u32(sim.colours), 1u);
    let nBio = max(u32(oceanSim.colours), 1u);
    let oceanBase = nChem * SPECIES_STRIDE + nChem * nChem;
    let rowOffset = min(bioSp, nBio - 1u) * nBio;
    var acc = OceanForceAccum();

    let worldP = p.pos * oceanSim.worldSize;
    let cmin = vec3i(floor((worldP - oceanSim.rMax) / sim.cellSize));
    let cmax = vec3i(floor((worldP + oceanSim.rMax) / sim.cellSize));
    let numCells = u32(sim.cellAmt);

    for (var cx = cmin.x; cx <= cmax.x; cx++) {
        for (var cy = cmin.y; cy <= cmax.y; cy++) {
            for (var cz = cmin.z; cz <= cmax.z; cz++) {
                let c = hash3i(vec3i(cx, cy, cz)) % numCells;
                getBioCellForce(pi, p, myPhase, myKin, oceanBase, rowOffset, nBio, c, &acc);
            }
        }
    }

    return ForceResult(finalizeOceanForce(acc, oceanSim), acc.torque);
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pi = global_id.x;
    if (pi >= arrayLength(&output)) {
        return;
    }

    let p = input[pi];
    let ls = lifeStateNext[pi];
    let myPhase = unpackPhase(ls.w);

    if (myPhase == 3u) {
        output[pi] = integrateMineralParticle(p, ls, bioForce[pi]);
        return;
    }

    if (myPhase == 1u || myPhase == 2u) {
        let bioSp = unpackBioSpecies(ls.w);
        let myKin = unpackKinPhase(ls.w);
        let resBio = getBioForce(pi, p, myPhase, bioSp, myKin);
        output[pi] = integrateOceanParticle(
            pi,
            p,
            resBio.force,
            resBio.torque,
            myPhase,
            oceanSim,
            oceanTex,
            uniforms.mouseSphere,
            ls,
            bioForce[pi],
        );
        return;
    }

    let mySlots = valence[pi];
    let bReact = bioForce[pi];
    let actGlow = unpackActGlow(bReact.w);
    let actScale = actGlow.x;
    let bioCleave = clamp(-actGlow.y, 0.0, 1.0);
    let windSample = evalWindAndShear(windTex, p.pos, uniforms.flags.w);
    let shearStress = clamp(max(0.0, windSample.shearRel - sim.shearFloor) * 4.5 * max(sim.shearBreak, 0.0), 0.0, 1.0);
    let res = getChemForce(pi, p, actScale, bioCleave, shearStress);
    output[pi] = integrateParticle(
        pi,
        p,
        mySlots,
        res,
        windSample,
        shearStress,
        uniforms.mouseSphere,
        bReact,
    );
}
`;
};
