const SPECIES_STRIDE: u32 = 8u;

struct Uniforms {
    aspect: f32,
    mouse: vec4<f32>,
    size: f32,
    showOrbitals: f32,
    _pad0: f32,
    _pad1: f32,
}

struct Camera {
    pos: vec2f,
    zoom: f32,
}

struct Particle {
    @location(0) pos: vec2f,
    @location(1) vel: vec2f,
    @location(2) colour: f32,
    @location(3) theta: f32,
    @location(4) slots: vec4f,
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) colour: f32,
    @location(2) theta: f32,
    @location(3) slots: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> camera: Camera;

@vertex
fn vertex(
    particle: Particle,
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    let a = uniforms.aspect;

    var positions = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );

    let baseSize = max(uniforms.size, 0.005 / camera.zoom);
    let scale = select(1.0, 1.45, uniforms.showOrbitals > 0.5);
    let offset = positions[vertexIndex] * baseSize * scale;
    let worldPos = particle.pos + offset;

    var output: VertexOutput;
    output.pos = vec4f((worldPos.x - camera.pos.x) * camera.zoom / a, (worldPos.y - camera.pos.y) * camera.zoom, 0.0, 1.0);
    output.uv = positions[vertexIndex];
    output.colour = particle.colour;
    output.theta = particle.theta;
    output.slots = particle.slots;
    return output;
}

@group(0) @binding(2) var<storage, read> colours: array<f32>;
@group(0) @binding(3) var<storage, read> matrix: array<f32>;

@fragment
fn fragment(
    @location(0) uv: vec2f,
    @location(1) colour: f32,
    @location(2) theta: f32,
    @location(3) slots: vec4f,
) -> @location(0) vec4f {
    let cIdx = u32(colour);
    let baseCol = vec3f(colours[cIdx * 3u], colours[cIdx * 3u + 1u], colours[cIdx * 3u + 2u]);
    let d = length(uv);
    if (d > 1.0) {
        discard;
    }

    if (uniforms.showOrbitals <= 0.5) {
        let delta = fwidth(d);
        let alpha = 1.0 - smoothstep(1.0 - delta, 1.0, d);
        return vec4f(baseCol, alpha);
    }

    let delta = max(fwidth(d), 0.02);
    let sBase = cIdx * SPECIES_STRIDE;
    let valence = u32(matrix[sBase + 0u]);
    let role = matrix[sBase + 1u];

    var outCol = vec3f(0.0);
    var outAlpha = 0.0;

    // Core atom circle
    let coreR = 0.56;
    if (d < coreR + delta) {
        let coreA = 1.0 - smoothstep(coreR - delta, coreR + delta, d);
        let shade = 0.85 + 0.22 * (1.0 - min(d / coreR, 1.0));
        outCol = min(baseCol * shade, vec3f(1.0));
        outAlpha = coreA;
    }

    if (valence == 0u) {
        // Noble gas: closed complete outer shell ring
        let ringDist = abs(d - 0.80);
        let ringA = (1.0 - smoothstep(0.04, 0.04 + delta, ringDist)) * 0.45;
        if (ringA > outAlpha) {
            outCol = mix(baseCol, vec3f(0.85, 0.9, 1.0), 0.35);
            outAlpha = ringA;
        }
    } else {
        for (var k = 0u; k < valence; k++) {
            let slotAngle = matrix[sBase + 4u + k];
            let ang = theta + slotAngle;
            let dotPos = vec2f(cos(ang), sin(ang)) * 0.80;
            let distDot = length(uv - dotPos);
            let occ = clamp(select(select(slots.z, slots.y, k == 1u), slots.x, k == 0u), 0.0, 1.0);

            if (role < -0.5) {
                // Acceptor: hollow slot ring when unfilled, fills bright as electrons complete it
                let ringA = 1.0 - smoothstep(0.04, 0.04 + delta, abs(distDot - 0.12));
                let fillA = (1.0 - smoothstep(0.11 - delta, 0.11 + delta, distDot)) * occ;
                let slotA = max(ringA * 0.85, fillA);
                if (slotA > outAlpha) {
                    outCol = mix(baseCol, vec3f(1.0, 0.96, 0.82), 0.55 + 0.45 * occ);
                    outAlpha = slotA;
                }
            } else {
                // Donor or half-shell covalent: bright outer electron dot (glows stronger when shared)
                let dotR = 0.12 + 0.03 * occ;
                let dotA = 1.0 - smoothstep(dotR - delta, dotR + delta, distDot);
                if (dotA > outAlpha) {
                    outCol = mix(baseCol, vec3f(1.0, 0.98, 0.88), 0.65 + 0.35 * occ);
                    outAlpha = dotA;
                }
            }
        }
    }

    if (outAlpha <= 0.001) {
        discard;
    }
    return vec4f(outCol, outAlpha);
}
