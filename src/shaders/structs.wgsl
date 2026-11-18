const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const SPECIES_STRIDE: u32 = 8u;

struct Uniforms {
    aspect: f32,
    mouse: vec4<f32>,
    size: f32,
    showOrbitals: f32,
    _pad0: f32,
    _pad1: f32,
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
    _pad: f32,
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
    pos: vec2f,
    vel: vec2f,
    colour: f32,
    theta: f32,
    omega: f32,
    _pad: f32,
}

struct SortParticle {
    pos: vec2f,
    idx: vec2u,
    colour: f32,
    theta: f32,
}

struct ListParticle {
    pos: vec2f,
    colour: f32,
    theta: f32,
    idx: u32,
    next: u32,
}

struct ForceAccum {
    bondForce: vec2f,
    repelForce: vec2f,
    avoidForce: vec2f,
    div: f32,
    torque: f32,
}

struct ForceResult {
    force: vec2f,
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
