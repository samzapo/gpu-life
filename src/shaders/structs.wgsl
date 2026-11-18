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
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

struct ListParticle {
    pos: vec3f,
    colour: f32,
    orient: vec3f,
    idx: u32,
    next: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
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
