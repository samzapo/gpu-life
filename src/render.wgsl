struct ParticleVert {
    @location(0) pos: vec3f,
    @location(1) colour: f32,
    @location(2) orient: vec3f,
    @location(3) slots: vec4f,
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) colour: f32,
    @location(2) slots: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var watermanFwdTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> colours: array<f32>;
@group(0) @binding(3) var<storage, read> matrix: array<f32>;

@vertex
fn vertex(
    particle: ParticleVert,
    @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
    var positions = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );

    let n = normalize(particle.pos);
    let p0 = projectUnitPos(uniforms, watermanFwdTex, n);

    var output: VertexOutput;
    if (p0.z < 0.5) {
        output.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        output.uv = vec2f(0.0);
        output.colour = particle.colour;
        output.slots = particle.slots;
        return output;
    }

    var uHat = particle.orient - dot(particle.orient, n) * n;
    let uLen = length(uHat);
    if (uLen > 1e-5) {
        uHat /= uLen;
    } else {
        let refA = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
        uHat = normalize(cross(refA, n));
    }
    let bHat = normalize(cross(n, uHat));

    let eps = 0.004;
    let pu = projectUnitPos(uniforms, watermanFwdTex, normalize(n + eps * uHat));
    let pb = projectUnitPos(uniforms, watermanFwdTex, normalize(n + eps * bHat));

    var dU = vec2f(1.0, 0.0);
    var dB = vec2f(0.0, -1.0);
    if (pu.z > 0.5 && pb.z > 0.5) {
        let cu = (pu.xy - p0.xy) / eps;
        let cb = (pb.xy - p0.xy) / eps;
        let lu = length(cu);
        let lb = length(cb);
        if (lu > 0.04 && lu < 8.0 && lb > 0.04 && lb < 8.0) {
            dU = cu;
            dB = cb;
        }
    }

    let scalePx = max(uniforms.viewParams.x, 1.0);
    let orbScale = select(1.0, 1.45, uniforms.flags.x > 0.5);
    let angR = max(uniforms.size * orbScale, 1.45 / scalePx);
    let cx = uniforms.viewSize.x * 0.5 + uniforms.viewParams.y;
    let cy = uniforms.viewSize.y * 0.5 + uniforms.viewParams.z;

    let corner = positions[vertexIndex];
    let pScreen = p0.xy * scalePx + vec2f(cx, cy) + (corner.x * dU + corner.y * dB) * (angR * scalePx);

    let ndcX = (pScreen.x / max(uniforms.viewSize.x, 1.0)) * 2.0 - 1.0;
    let ndcY = 1.0 - (pScreen.y / max(uniforms.viewSize.y, 1.0)) * 2.0;

    output.pos = vec4f(ndcX, ndcY, 0.0, 1.0);
    output.uv = corner;
    output.colour = particle.colour;
    output.slots = particle.slots;
    return output;
}

@fragment
fn fragment(
    @location(0) uv: vec2f,
    @location(1) colour: f32,
    @location(2) slots: vec4f,
) -> @location(0) vec4f {
    let cIdx = u32(colour);
    let baseCol = vec3f(colours[cIdx * 3u], colours[cIdx * 3u + 1u], colours[cIdx * 3u + 2u]);
    let d = length(uv);
    if (d > 1.0) {
        discard;
    }

    if (uniforms.flags.x <= 0.5) {
        let delta = max(fwidth(d), 0.03);
        let alpha = 1.0 - smoothstep(1.0 - delta, 1.0, d);
        return vec4f(baseCol * alpha, alpha);
    }

    let delta = max(fwidth(d), 0.025);
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
        let ringDist = abs(d - 0.80);
        let ringA = (1.0 - smoothstep(0.04, 0.04 + delta, ringDist)) * 0.45;
        if (ringA > outAlpha) {
            outCol = mix(baseCol, vec3f(0.85, 0.9, 1.0), 0.35);
            outAlpha = ringA;
        }
    } else {
        for (var k = 0u; k < valence; k++) {
            let slotAngle = matrix[sBase + 4u + k];
            let dotPos = vec2f(cos(slotAngle), sin(slotAngle)) * 0.80;
            let distDot = length(uv - dotPos);
            let occ = clamp(select(select(slots.z, slots.y, k == 1u), slots.x, k == 0u), 0.0, 1.0);

            if (role < -0.5) {
                let ringA = 1.0 - smoothstep(0.04, 0.04 + delta, abs(distDot - 0.12));
                let fillA = (1.0 - smoothstep(0.11 - delta, 0.11 + delta, distDot)) * occ;
                let slotA = max(ringA * 0.85, fillA);
                if (slotA > outAlpha) {
                    outCol = mix(baseCol, vec3f(1.0, 0.96, 0.82), 0.55 + 0.45 * occ);
                    outAlpha = slotA;
                }
            } else {
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
    return vec4f(outCol * outAlpha, outAlpha);
}
