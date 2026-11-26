const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const HALF_PI: f32 = 1.57079632679;
const SPECIES_STRIDE: u32 = 8u;

struct Uniforms {
    viewSize: vec2f,
    aspect: f32,
    size: f32,
    mouseSphere: vec4f,
    viewParams: vec4f,
    flags: vec4f,
    rotCol0: vec4f,
    rotCol1: vec4f,
    rotCol2: vec4f,
    earthFlags: vec4f,
    lensParams: vec4f,
    renderParams: vec4f, // x: bioSize, y: lifeOpacity, z: nChemColours, w: nBioColours
    mineralParams: vec4f, // x: mineralOpacity, y: tempColors, z: localTempScale, w: viewLayerCode
    uvParams: vec4f, // x: uvColors, y: localUvScale, z: _pad, w: _pad
    hydroParams: vec4f, // x: rhColors, y: precipColors, z: localHydroScale, w: _pad
    terrainParams: vec4f, // x: terrainOpacity, y: nightShading, z: _pad, w: _pad
}

struct Sim {
    colours: f32,
    beta: f32,
    rMax: f32,
    force: f32,
    friction: f32,
    dt: f32,
    cellSize: f32,
    cellAmt: f32,
    avoidance: f32,
    worldSize: f32,
    border: f32,
    vortex: f32,
    torque: f32,
    angularFriction: f32,
    valency: f32,
    temperature: f32,
    activation: f32,
    orbitalSharpness: f32,
    frame: f32,
    windStrength: f32,
    bondStability: f32,
    exchangeDrive: f32,
    shearBreak: f32,
    shearFloor: f32,
}

// Original-fork (51bc381) Particle Life uniforms for the Ocean Currents layer.
// Offsets 0..9 intentionally align with Sim so cell/construct spatial passes reuse identically.
struct OceanSim {
    colours: f32,
    beta: f32,
    rMax: f32,
    force: f32,
    friction: f32,
    dt: f32,
    cellSize: f32,
    cellAmt: f32,
    avoidance: f32,
    worldSize: f32,
    currentStrength: f32,
    vortex: f32,
}

struct SpeciesInfo {
    valence: f32,
    role: f32,
    outerElectrons: f32,
    outerCapacity: f32,
    slotAngle0: f32,
    slotAngle1: f32,
    slotAngle2: f32,
    atomicNumber: f32,
}

struct Particle {
    pos: vec3f,
    colour: f32,
    vel: vec3f,
    omega: f32,
    orient: vec3f,
    _pad: f32,
}

struct SortParticle {
    pos: vec3f,
    colour: f32,
    orient: vec3f,
    cellIdx: u32,
    idx: u32,
    phase: u32,
    bioSpecies: u32,
    bioEnergy: f32,
}

struct ListParticle {
    pos: vec3f,
    colour: f32,
    orient: vec3f,
    idx: u32,
    next: u32,
    phase: u32,
    bioSpecies: u32,
    bioEnergy: f32,
}

fn unpackPhase(packed: f32) -> u32 {
    if (packed < 0.5) {
        return 0u;
    }
    return u32(round(packed)) & 3u;
}

fn unpackBioSpecies(packed: f32) -> u32 {
    if (packed < 0.5) {
        return 0u;
    }
    return (u32(round(packed)) >> 2u) & 255u;
}

fn unpackKinPhase(packed: f32) -> f32 {
    if (packed < 0.5) {
        return 0.0;
    }
    return (f32((u32(round(packed)) >> 10u) & 255u) / 256.0) * TWO_PI;
}

fn unpackEncapGene(packed: f32) -> u32 {
    if (packed < 0.5) {
        return 7u;
    }
    return (u32(round(packed)) >> 19u) & 15u;
}

fn unpackGraduated(packed: f32) -> u32 {
    if (packed < 0.5) {
        return 0u;
    }
    return (u32(round(packed)) >> 23u) & 1u;
}

fn packLifeMetaGrad(
    phase: u32,
    bioSp: u32,
    kinAngle: f32,
    encapGene: u32,
    graduated: u32,
) -> f32 {
    if (phase == 0u) {
        return 0.0;
    }
    let k = u32(floor((wrapAngle(kinAngle) / TWO_PI) * 256.0)) & 255u;
    let g = min(encapGene, 15u);
    let gr = min(graduated, 1u);
    return f32((phase & 3u) | (min(bioSp, 255u) << 2u) | (k << 10u) | (g << 19u) | (gr << 23u));
}

fn packLifeMeta(phase: u32, bioSp: u32, kinAngle: f32, encapGene: u32) -> f32 {
    return packLifeMetaGrad(phase, bioSp, kinAngle, encapGene, 0u);
}

struct ForceAccum {
    bondForce: vec3f,
    repelForce: vec3f,
    avoidForce: vec3f,
    div: f32,
    torque: f32,
}

struct ForceResult {
    force: vec3f,
    torque: f32,
}

fn hash3i(k: vec3<i32>) -> u32 {
    let offset: u32 = 0x80000000u;
    var x: u32 = (u32(k.x) + offset) * 0x9E3779B1u;
    var y: u32 = (u32(k.y) + offset) * 0x85EBCA6Bu;
    var z: u32 = (u32(k.z) + offset) * 0xC2B2AE35u;

    var h: u32 = x ^ y ^ z;

    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;

    return h;
}

fn hashFloat(id: u32, frame: u32, channel: u32) -> f32 {
    var h: u32 = (id * 0x9E3779B1u) ^ (frame * 0x85EBCA6Bu) ^ (channel * 0xC2B2AE35u);
    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;
    return f32(h & 0x00FFFFFFu) / 16777216.0;
}

fn wrapAngle(angle: f32) -> f32 {
    var a = angle % TWO_PI;
    if (a < 0.0) {
        a += TWO_PI;
    }
    return a;
}

const MDP_SPECIES_STRIDE: u32 = 16u;
const MDP_STATE_LOW_ENERGY: u32 = 0u;
const MDP_STATE_HAZARD: u32 = 1u;
const MDP_STATE_ABUNDANT: u32 = 2u;
const MDP_STATE_DARK_CROWDED: u32 = 3u;

const MDP_ACT_ANCHOR: u32 = 0u;
const MDP_ACT_FORAGE: u32 = 1u;
const MDP_ACT_CHASE: u32 = 2u;
const MDP_ACT_EVADE: u32 = 3u;

const MAX_LEADER_HOPS: u32 = 24u;
const NO_LEADER: u32 = 0xffffffffu;
const MDP_PAD_TAG: u32 = 2097152u;

struct MdpState {
    qPack: vec4u,
    ctrl: vec4f,
}

struct LifeMdpState {
    life: vec4f,
    qPack: vec4u,
    ctrl: vec4f,
    blob: vec4u, // x=leaderId, y=distHops, z=packed3Mdp, w=prevSpeciesCtrl
}

struct BlobMdpSlot {
    qPack: vec4u,
    ctrl: vec4f,
}

struct MdpLeverProfile {
    dSameKin: f32,
    dForeignBio: f32,
    dChemNutr: f32,
    dChemHalogen: f32,
    dLipid: f32,
    dMineral: f32,
    dBeta: f32,
    dDrag: f32,
    chiralTwist: f32,
    swimThrust: f32,
    lipidAlloc: f32,
    fixation: f32,
}

fn emptyMdpState() -> MdpState {
    return MdpState(vec4u(0u), vec4f(0.0));
}

fn quantizeMdpQ(q: f32) -> u32 {
    let norm = clamp((q + 1.5) / 3.0, 0.0, 1.0);
    return u32(round(norm * 255.0)) & 255u;
}

fn dequantizeMdpQ(qByte: u32) -> f32 {
    return (f32(qByte & 255u) / 255.0) * 3.0 - 1.5;
}

fn unpackMdpQ(rowPacked: u32, actionIdx: u32) -> f32 {
    let shift = (actionIdx & 3u) * 8u;
    return dequantizeMdpQ((rowPacked >> shift) & 255u);
}

fn setMdpQ(rowPacked: u32, actionIdx: u32, val: f32) -> u32 {
    let shift = (actionIdx & 3u) * 8u;
    let clearMask = ~(255u << shift);
    return (rowPacked & clearMask) | (quantizeMdpQ(val) << shift);
}

fn packMdpCtrl(prevState: u32, action: u32, dwell: u32, initBit: u32) -> f32 {
    let packed = (prevState & 3u) | ((action & 3u) << 2u) | (min(dwell, 255u) << 4u) | ((initBit & 1u) << 12u);
    return f32(packed);
}

fn getQRow(qPack: vec4u, stateIdx: u32) -> u32 {
    let s = stateIdx & 3u;
    if (s == 0u) { return qPack.x; }
    if (s == 1u) { return qPack.y; }
    if (s == 2u) { return qPack.z; }
    return qPack.w;
}

fn setQRow(qPack: vec4u, stateIdx: u32, rowVal: u32) -> vec4u {
    let s = stateIdx & 3u;
    var out = qPack;
    if (s == 0u) { out.x = rowVal; }
    else if (s == 1u) { out.y = rowVal; }
    else if (s == 2u) { out.z = rowVal; }
    else { out.w = rowVal; }
    return out;
}

fn packBlobPolicyZ(
    pAct: u32,
    bAct: u32,
    sAct: u32,
    pState: u32,
    bState: u32,
    hasBlob: u32,
    sizeBucket: u32,
) -> u32 {
    return (pAct & 3u)
        | ((bAct & 3u) << 2u)
        | ((sAct & 3u) << 4u)
        | ((pState & 3u) << 6u)
        | ((bState & 3u) << 8u)
        | ((hasBlob & 1u) << 10u)
        | ((sizeBucket & 7u) << 11u)
        | 16384u; // bit 14: valid 3-MDP broadcast
}

fn packThreeMdpPadFromZ(packedBlobZ: u32) -> f32 {
    return f32(MDP_PAD_TAG | (packedBlobZ & 16383u));
}

fn actionLeverProfile(act: u32) -> MdpLeverProfile {
    let a = act & 3u;
    if (a == MDP_ACT_ANCHOR) {
        return MdpLeverProfile(0.38, -0.10, 0.62, -0.35, 0.58, 0.76, 0.0, 0.78, 0.0, 0.0, 0.80, 0.92);
    }
    if (a == MDP_ACT_FORAGE) {
        return MdpLeverProfile(0.22, -0.06, 0.96, -0.65, 0.16, 0.30, 0.0, -0.24, 0.34, 0.64, 0.16, 0.14);
    }
    if (a == MDP_ACT_CHASE) {
        return MdpLeverProfile(0.18, 1.30, 0.12, -0.40, -0.22, -0.10, -0.02, -0.34, 0.10, 1.06, 0.05, 0.0);
    }
    return MdpLeverProfile(0.05, -1.18, 0.0, -1.18, -0.28, -0.32, 0.02, -0.32, -0.42, 0.92, 0.02, 0.0);
}

fn evalCombinedMdpPolicy(packedBlobZ: u32) -> MdpLeverProfile {
    let pAct = packedBlobZ & 3u;
    let bAct = (packedBlobZ >> 2u) & 3u;
    let sAct = (packedBlobZ >> 4u) & 3u;
    let hasBlob = ((packedBlobZ >> 10u) & 1u) != 0u;
    let lp = actionLeverProfile(pAct);
    let lb = actionLeverProfile(bAct);
    let ls = actionLeverProfile(sAct);
    let wp = select(0.65, 0.35, hasBlob);
    let wb = select(0.0, 0.40, hasBlob);
    let ws = select(0.35, 0.25, hasBlob);
    return MdpLeverProfile(
        wp * lp.dSameKin + wb * lb.dSameKin + ws * ls.dSameKin,
        wp * lp.dForeignBio + wb * lb.dForeignBio + ws * ls.dForeignBio,
        wp * lp.dChemNutr + wb * lb.dChemNutr + ws * ls.dChemNutr,
        wp * lp.dChemHalogen + wb * lb.dChemHalogen + ws * ls.dChemHalogen,
        wp * lp.dLipid + wb * lb.dLipid + ws * ls.dLipid,
        wp * lp.dMineral + wb * lb.dMineral + ws * ls.dMineral,
        wp * lp.dBeta + wb * lb.dBeta + ws * ls.dBeta,
        wp * lp.dDrag + wb * lb.dDrag + ws * ls.dDrag,
        wp * lp.chiralTwist + wb * lb.chiralTwist + ws * ls.chiralTwist,
        wp * lp.swimThrust + wb * lb.swimThrust + ws * ls.swimThrust,
        wp * lp.lipidAlloc + wb * lb.lipidAlloc + ws * ls.lipidAlloc,
        wp * lp.fixation + wb * lb.fixation + ws * ls.fixation,
    );
}
