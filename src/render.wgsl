struct ParticleVert {
    @location(0) pos: vec3f,
    @location(1) colour: f32,
    @location(2) orient: vec3f,
    @location(3) slots: vec4f,
    @location(4) chemGlow: f32,
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) colour: f32,
    @location(2) slots: vec4f,
    @location(3) chemGlow: f32,
    @location(4) limbFade: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var watermanFwdTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> colours: array<f32>;
@group(0) @binding(3) var<storage, read> matrix: array<f32>;

fn safeOrthoStepDiff(
    u: Uniforms,
    watermanFwdTex: texture_2d<f32>,
    n: vec3f,
    p0: vec2f,
    axis: vec3f,
) -> vec3f {
    let eps = 0.004;
    let pPlus = projectUnitPos(u, watermanFwdTex, normalize(n + eps * axis));
    if (pPlus.z > 0.5) {
        let d = (pPlus.xy - p0) / eps;
        let l = length(d);
        if (l > 0.02 && l < 6.0) {
            return vec3f(d, 1.0);
        }
    }
    let pMinus = projectUnitPos(u, watermanFwdTex, normalize(n - eps * axis));
    if (pMinus.z > 0.5) {
        let d = (p0 - pMinus.xy) / eps;
        let l = length(d);
        if (l > 0.02 && l < 6.0) {
            return vec3f(d, 1.0);
        }
    }
    return vec3f(0.0, 0.0, 0.0);
}

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
        output.chemGlow = particle.chemGlow;
        output.limbFade = 0.0;
        return output;
    }

    let uHat = orthoBasis(n, particle.orient);
    let bHat = normalize(cross(n, uHat));
    let pType = u32(uniforms.viewParams.w + 0.5);
    let vView = rotateToView(uniforms, n);

    var limbFade = 1.0;
    var dU = vec2f(1.0, 0.0);
    var dB = vec2f(0.0, -1.0);

    if (pType == 0u) {
        limbFade = smoothstep(0.005, 0.16, vView.x);
        let uView = rotateToView(uniforms, uHat);
        let bView = rotateToView(uniforms, bHat);
        let rLimb = sqrt(max(0.0, vView.y * vView.y + vView.z * vView.z));
        if (rLimb < 1e-4) {
            dU = vec2f(uView.y, -uView.z);
            dB = vec2f(bView.y, -bView.z);
        } else {
            let invR = 1.0 / rLimb;
            let t3 = vec3f(0.0, -vView.z * invR, vView.y * invR);
            let r3 = vec3f(-rLimb, vView.x * vView.y * invR, vView.x * vView.z * invR);
            let eRad2D = vec2f(vView.y * invR, -vView.z * invR);
            let eTan2D = vec2f(-vView.z * invR, -vView.y * invR);
            let kRad = mix(0.48, 1.0, vView.x);
            dU = dot(uView, r3) * kRad * eRad2D + dot(uView, t3) * eTan2D;
            dB = dot(bView, r3) * kRad * eRad2D + dot(bView, t3) * eTan2D;
        }
    } else {
        if (pType == 6u) {
            limbFade = smoothstep(0.01, 0.12, 1.0 + vView.x);
        }
        let duProbe = safeOrthoStepDiff(uniforms, watermanFwdTex, n, p0.xy, uHat);
        let dbProbe = safeOrthoStepDiff(uniforms, watermanFwdTex, n, p0.xy, bHat);
        if (duProbe.z > 0.5 && dbProbe.z > 0.5) {
            var cu = duProbe.xy;
            var cb = dbProbe.xy;
            let lu = length(cu);
            let lb = length(cb);
            let maxL = max(lu, lb);
            let targetMax = min(maxL, 1.15);
            let targetMin = targetMax * 0.48;
            if (lu > 1e-4) {
                cu *= clamp(lu, targetMin, targetMax) / lu;
            }
            if (lb > 1e-4) {
                cb *= clamp(lb, targetMin, targetMax) / lb;
            }
            dU = cu;
            dB = cb;
        }
    }

    let scalePx = max(uniforms.viewParams.x, 1.0);
    let orbScale = select(1.0, 1.68, uniforms.flags.x > 0.5);
    let angR = max(uniforms.size * orbScale, 1.55 / scalePx);
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
    output.chemGlow = particle.chemGlow;
    output.limbFade = limbFade;
    return output;
}

@fragment
fn fragment(
    @location(0) uv: vec2f,
    @location(1) colour: f32,
    @location(2) slots: vec4f,
    @location(3) chemGlow: f32,
    @location(4) limbFade: f32,
) -> @location(0) vec4f {
    let uvStep = length(fwidth(uv));
    let delta = clamp(uvStep, 0.025, 0.32);
    let deltaDot = clamp(uvStep, 0.045, 0.42);
    let cIdx = u32(colour);
    let baseCol = vec3f(colours[cIdx * 3u], colours[cIdx * 3u + 1u], colours[cIdx * 3u + 2u]);
    let d = length(uv);
    let layerOpacity = clamp(uniforms.earthFlags.w, 0.0, 1.0) * clamp(limbFade, 0.0, 1.0);
    if (d > 1.0 || layerOpacity <= 0.001) {
        discard;
    }

    if (uniforms.flags.x <= 0.5) {
        let alpha = 1.0 - smoothstep(1.0 - delta, 1.0, d);
        let energy = clamp(slots.x, 0.0, 1.5);
        let vit = clamp(slots.y, 0.0, 1.0);
        let poi = clamp(slots.z, 0.0, 1.0);
        let vitalityBright = clamp(0.675 + 0.50 * energy + 0.15 * vit, 0.45, 1.35);
        var lifeCol = min(baseCol * vitalityBright, vec3f(1.0));
        if (poi > 0.15) {
            let rim = smoothstep(0.62, 0.92, d) * clamp((poi - 0.15) * 1.15, 0.0, 0.85);
            lifeCol = mix(lifeCol, vec3f(0.88, 0.25, 0.96), rim);
        }
        if (vit > 0.25) {
            let glowRim = smoothstep(0.70, 0.95, d) * clamp((vit - 0.25) * 0.6, 0.0, 0.45);
            lifeCol = mix(lifeCol, vec3f(0.32, 0.98, 0.78), glowRim);
        }
        let lifeA = alpha * layerOpacity;
        return vec4f(lifeCol * lifeA, lifeA);
    }

    let sBase = cIdx * SPECIES_STRIDE;
    let valence = u32(matrix[sBase + 0u]);
    let role = matrix[sBase + 1u];

    var outCol = vec3f(0.0);
    var outAlpha = 0.0;

    // Core atom circle sized so equilibrium bonded partners leave crisp overlapping valence rings/dots
    let coreR = 0.45;
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
                let ringA = 1.0 - smoothstep(0.045, 0.045 + deltaDot, abs(distDot - 0.13));
                let fillA = (1.0 - smoothstep(0.12 - deltaDot, 0.12 + deltaDot, distDot)) * occ;
                let slotA = max(ringA * 0.88, fillA);
                if (slotA > outAlpha) {
                    outCol = mix(baseCol, vec3f(1.0, 0.96, 0.82), 0.55 + 0.45 * occ);
                    outAlpha = slotA;
                }
            } else {
                let dotR = 0.13 + 0.03 * occ;
                let dotA = 1.0 - smoothstep(dotR - deltaDot, dotR + deltaDot, distDot);
                if (dotA > outAlpha) {
                    outCol = mix(baseCol, vec3f(1.0, 0.98, 0.88), 0.65 + 0.35 * occ);
                    outAlpha = dotA;
                }
            }
        }
    }

    var premulCol = outCol * outAlpha;
    var finalAlpha = outAlpha;
    if (abs(chemGlow) > 0.06) {
        let auraCol = select(vec3f(1.0, 0.50, 0.20), vec3f(0.20, 0.96, 0.78), chemGlow > 0.0);
        let auraBand = smoothstep(0.42, 0.66, d) * (1.0 - smoothstep(0.66, 0.99, d));
        let auraA = auraBand * 0.55 * clamp(abs(chemGlow), 0.0, 1.0);
        premulCol = premulCol + auraCol * auraA * (1.0 - finalAlpha);
        finalAlpha = finalAlpha + auraA * (1.0 - finalAlpha);
    }

    if (finalAlpha <= 0.001) {
        discard;
    }
    return vec4f(premulCol * layerOpacity, finalAlpha * layerOpacity);
}
