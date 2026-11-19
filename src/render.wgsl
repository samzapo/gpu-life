struct ParticleVert {
    @location(0) pos: vec3f,
    @location(1) colour: f32,
    @location(2) orient: vec3f,
    @location(3) slots: vec4f,
    @location(4) chemGlow: f32,
    @location(5) lifeState: vec4f,
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) chemSpecies: f32,
    @location(2) slots: vec4f,
    @location(3) chemGlow: f32,
    @location(4) limbFade: f32,
    @location(5) phaseF: f32,
    @location(6) bioSpeciesF: f32,
    @location(7) lifeState: vec4f,
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

    let phase = unpackPhase(particle.lifeState.w);
    let bioSp = unpackBioSpecies(particle.lifeState.w);
    let graduated = unpackGraduated(particle.lifeState.w);
    let phaseF = f32(phase);
    let bioSpeciesF = f32(bioSp);
    var layerOpacity = uniforms.renderParams.y;
    if (phase == 0u) {
        layerOpacity = uniforms.earthFlags.w;
    } else if (phase == 3u) {
        layerOpacity = uniforms.mineralParams.x;
    }

    let n = normalize(particle.pos);
    let p0 = projectUnitPos(uniforms, watermanFwdTex, n);

    var output: VertexOutput;
    if (p0.z < 0.5 || layerOpacity <= 0.001) {
        output.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        output.uv = vec2f(0.0);
        output.chemSpecies = particle.colour;
        output.slots = particle.slots;
        output.chemGlow = particle.chemGlow;
        output.limbFade = 0.0;
        output.phaseF = phaseF;
        output.bioSpeciesF = bioSpeciesF;
        output.lifeState = particle.lifeState;
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
    let chemAngR = max(uniforms.size * orbScale, 1.55 / scalePx);
    let bioAngR = max(uniforms.renderParams.x, 1.75 / scalePx);
    let lipidAngR = bioAngR * select(1.28, 1.40, graduated == 1u);
    var angR = bioAngR;
    if (phase == 0u) {
        angR = chemAngR;
    } else if (phase == 2u) {
        angR = lipidAngR;
    } else if (phase == 3u) {
        let integ = clamp(particle.lifeState.x, 0.08, 1.15);
        angR = max(bioAngR * 1.18, chemAngR * 1.32) * mix(0.68, 1.14, clamp(integ, 0.0, 1.0));
    }
    let cx = uniforms.viewSize.x * 0.5 + uniforms.viewParams.y;
    let cy = uniforms.viewSize.y * 0.5 + uniforms.viewParams.z;

    let corner = positions[vertexIndex];
    let cScreenRaw = p0.xy * scalePx + vec2f(cx, cy);
    var cScreen = cScreenRaw;
    var localMag = 1.0;
    if (uniforms.lensParams.z > 1.0 && uniforms.lensParams.w > 1.01) {
        let uDist = length(cScreenRaw - uniforms.lensParams.xy) / uniforms.lensParams.z;
        if (uDist < 1.0) {
            cScreen = applyLensForward(uniforms, cScreenRaw);
            localMag = lensParticleScale(uniforms.lensParams.w, uDist);
        }
    }
    let pScreen = cScreen + (corner.x * dU + corner.y * dB) * (angR * scalePx * localMag);

    let ndcX = (pScreen.x / max(uniforms.viewSize.x, 1.0)) * 2.0 - 1.0;
    let ndcY = 1.0 - (pScreen.y / max(uniforms.viewSize.y, 1.0)) * 2.0;

    output.pos = vec4f(ndcX, ndcY, 0.0, 1.0);
    output.uv = corner;
    output.chemSpecies = particle.colour;
    output.slots = particle.slots;
    output.chemGlow = particle.chemGlow;
    output.limbFade = limbFade;
    output.phaseF = phaseF;
    output.bioSpeciesF = bioSpeciesF;
    output.lifeState = particle.lifeState;
    return output;
}

@fragment
fn fragment(
    @location(0) uv: vec2f,
    @location(1) chemSpecies: f32,
    @location(2) slots: vec4f,
    @location(3) chemGlow: f32,
    @location(4) limbFade: f32,
    @location(5) phaseF: f32,
    @location(6) bioSpeciesF: f32,
    @location(7) lifeState: vec4f,
) -> @location(0) vec4f {
    let uvStep = length(fwidth(uv));
    let delta = clamp(uvStep, 0.025, 0.32);
    let deltaDot = clamp(uvStep, 0.045, 0.42);
    let nChem = max(u32(uniforms.renderParams.z + 0.5), 1u);
    let nBio = max(u32(uniforms.renderParams.w + 0.5), 1u);
    let cIdx = min(u32(max(chemSpecies, 0.0)), nChem - 1u);
    let chemCol = vec3f(colours[cIdx * 3u], colours[cIdx * 3u + 1u], colours[cIdx * 3u + 2u]);
    let d = length(uv);
    let isMineral = phaseF > 2.5;
    let isLipid = !isMineral && phaseF > 1.5;
    let isBio = !isMineral && !isLipid && phaseF > 0.5;
    var layerOpacity = uniforms.renderParams.y;
    if (phaseF < 0.5) {
        layerOpacity = uniforms.earthFlags.w;
    } else if (isMineral) {
        layerOpacity = uniforms.mineralParams.x;
    }
    layerOpacity = clamp(layerOpacity, 0.0, 1.0) * clamp(limbFade, 0.0, 1.0);
    if ((!isLipid && !isMineral && d > 1.0) || ((isLipid || isMineral) && max(abs(uv.x), abs(uv.y)) > 1.0) || layerOpacity <= 0.001) {
        discard;
    }

    if (isMineral) {
        let minType = u32(max(bioSpeciesF, 0.0) + 0.5);
        let integ = clamp(lifeState.x, 0.05, 1.25);
        let nutr = clamp(lifeState.y, 0.0, 1.0);
        let haz = clamp(lifeState.z, 0.0, 1.0);
        let isTrap = unpackGraduated(lifeState.w) == 1u;

        var sdf = 1.0;
        var baseCol = vec3f(0.65);
        var facetShade = 1.0;
        var coreCol = vec3f(0.8);
        var coreMask = 0.0;
        var rimCol = vec3f(0.95);

        if (minType == 0u) {
            // 0: Silicate Bedrock (elongated hexagonal quartz/olivine prism)
            let p = abs(vec2f(uv.x * 1.12, uv.y * 0.94));
            sdf = max(p.x * 0.8660254 + p.y * 0.5, p.y) - 0.84;
            let divDiag = select(-0.18, 0.18, uv.x * 0.7 + uv.y * 0.4 > 0.0);
            let divSide = select(-0.10, 0.12, uv.y > 0.0);
            facetShade = clamp(0.88 + divDiag + divSide, 0.62, 1.22);
            baseCol = mix(vec3f(0.46, 0.55, 0.64), chemCol, 0.28);
            let ridge = 1.0 - smoothstep(0.0, 0.12, min(abs(uv.x), abs(abs(uv.y) * 0.5 - abs(uv.x) * 0.866)));
            coreMask = ridge * 0.45;
            coreCol = vec3f(0.86, 0.94, 1.0);
            rimCol = vec3f(0.85, 0.93, 0.98);
        } else if (minType == 1u) {
            // 1: Evaporite Salt (stepped cubic Halite / Fluorite hopper crystal)
            let dBox = max(abs(uv.x), abs(uv.y));
            sdf = dBox - 0.84;
            let terrace = max(
                1.0 - smoothstep(0.0, 0.055, abs(dBox - 0.56)),
                1.0 - smoothstep(0.0, 0.050, abs(dBox - 0.30)),
            );
            facetShade = clamp(0.86 + 0.16 * select(-1.0, 1.0, uv.x > -uv.y) + 0.18 * terrace, 0.68, 1.26);
            baseCol = mix(vec3f(0.85, 0.94, 0.99), chemCol, 0.18);
            coreCol = mix(vec3f(0.68, 0.92, 1.0), vec3f(0.42, 1.0, 0.46), select(haz * 0.4, 0.95, isTrap));
            coreMask = (1.0 - smoothstep(0.22, 0.52, dBox)) * select(0.35, 0.85, isTrap || haz > 0.45);
            rimCol = select(vec3f(0.95, 0.98, 1.0), vec3f(0.55, 1.0, 0.58), isTrap);
        } else if (minType == 2u) {
            // 2: Carbonate Vein (Dolomite rhomb — vivid emerald Beryl core if Be-trapped)
            let dRhomb = abs(uv.x) * 0.90 + abs(uv.y) * 1.14;
            sdf = dRhomb - 0.86;
            let cleave = select(-0.16, 0.18, (uv.y - 0.35 * uv.x) > 0.0);
            facetShade = clamp(0.88 + cleave, 0.66, 1.20);
            baseCol = mix(vec3f(0.92, 0.80, 0.54), chemCol, 0.25);
            let berylZone = 1.0 - smoothstep(0.18, 0.46, dRhomb);
            coreCol = select(vec3f(1.0, 0.92, 0.68), vec3f(0.14, 0.96, 0.52), isTrap);
            coreMask = select(berylZone * 0.38, berylZone * 0.92, isTrap);
            rimCol = select(vec3f(1.0, 0.92, 0.72), vec3f(0.95, 0.32, 0.78), isTrap && haz > 0.55);
        } else if (minType == 3u) {
            // 3: Sulfide Chimney (intergrown 8-point pyrite octahedra & glowing vent core)
            let sA = max(abs(uv.x), abs(uv.y));
            let sB = max(abs(uv.x + uv.y), abs(uv.x - uv.y)) * 0.70710678;
            sdf = min(sA, sB) - 0.82;
            let pyFacet = select(-0.14, 0.24, (uv.x * uv.y) > 0.0);
            facetShade = clamp(0.86 + pyFacet, 0.65, 1.28);
            baseCol = mix(vec3f(0.28, 0.25, 0.30), vec3f(0.92, 0.76, 0.28), 0.58);
            let ventGlow = 1.0 - smoothstep(0.12, 0.46, d);
            coreCol = select(vec3f(1.0, 0.48, 0.14), vec3f(0.86, 0.30, 0.98), isTrap);
            coreMask = ventGlow * clamp(0.45 + 0.5 * max(nutr, haz), 0.45, 0.95);
            rimCol = select(vec3f(1.0, 0.84, 0.38), vec3f(0.94, 0.44, 1.0), isTrap);
        } else {
            // 4: Hydrated Clay (terraced phyllosilicate plates with turquoise catalytic bands)
            let octD = max(max(abs(uv.x) * 0.94, abs(uv.y) * 1.12), (abs(uv.x) + abs(uv.y)) * 0.76);
            sdf = octD - 0.84;
            let layerBand = 0.5 + 0.5 * cos(uv.y * 18.0);
            facetShade = clamp(0.82 + 0.20 * layerBand + 0.10 * sign(uv.x), 0.66, 1.18);
            baseCol = mix(vec3f(0.76, 0.50, 0.36), chemCol, 0.20);
            coreCol = vec3f(0.22, 0.96, 0.88);
            coreMask = clamp(layerBand * 0.48 + (1.0 - smoothstep(0.20, 0.55, octD)) * 0.35, 0.0, 0.85);
            rimCol = vec3f(0.40, 0.96, 0.90);
        }

        // Erosion chipping & rim sparkle as integrity degrades
        let erodeWear = clamp((0.86 - integ) / 0.76, 0.0, 1.0);
        let cellId = floor((uv + 1.0) * 4.0);
        let chipHash = fract(sin(dot(cellId, vec2f(12.9898, 78.233)) + f32(cIdx) * 3.7) * 43758.5453);
        let chipD = sdf + erodeWear * 0.30 * chipHash;
        let crystAlpha = 1.0 - smoothstep(-delta, delta, chipD);
        if (crystAlpha <= 0.002) {
            discard;
        }
        var col = min(baseCol * facetShade, vec3f(1.0));
        col = mix(col, coreCol, coreMask);
        let edgeHighlight = smoothstep(-0.14 - delta * 1.5, -0.02, chipD) * 0.52;
        col = min(mix(col, rimCol, edgeHighlight), vec3f(1.0));
        let outA = crystAlpha * layerOpacity;
        return vec4f(col * outA, outA);
    }

    if (isLipid) {
        let bIdx = nChem + min(u32(max(bioSpeciesF, 0.0) + 0.5), nBio - 1u);
        let bioCol = vec3f(colours[bIdx * 3u], colours[bIdx * 3u + 1u], colours[bIdx * 3u + 2u]);
        let energy = clamp(lifeState.x, 0.0, 1.45);
        let poi = clamp(lifeState.z, 0.0, 1.0);
        let energyTone = clamp(0.72 + 0.35 * energy, 0.55, 1.25);

        let wave = 0.045 * sin(uv.x * 11.0);
        let tailOffset = 0.14 + wave;
        let dTail = min(abs(uv.y - tailOffset), abs(uv.y + tailOffset));
        let tailMaskX = smoothstep(-0.92, -0.84, uv.x) * (1.0 - smoothstep(0.12, 0.22, uv.x));
        let tailAlpha = (1.0 - smoothstep(0.085 - delta * 0.5, 0.085 + delta * 0.5, dTail)) * tailMaskX;
        let tailCol = min(vec3f(0.96, 0.74, 0.30) * energyTone, vec3f(1.0));

        let dHead = length(uv - vec2f(0.36, 0.0));
        let headAlpha = 1.0 - smoothstep(0.36 - delta, 0.36 + delta * 0.6, dHead);
        let specCore = (1.0 - smoothstep(0.0, 0.24, dHead)) * 0.25;
        var headCol = mix(vec3f(0.76, 0.96, 1.0), bioCol, 0.22) * energyTone + vec3f(specCore);
        if (poi > 0.20) {
            let rim = smoothstep(0.18, 0.36, dHead) * clamp((poi - 0.20) * 1.15, 0.0, 0.85);
            headCol = mix(headCol, vec3f(0.92, 0.28, 0.98), rim);
        }
        headCol = min(headCol, vec3f(1.0));

        let premulTail = tailCol * tailAlpha;
        let premulHead = headCol * headAlpha;
        let premulLip = premulHead + premulTail * (1.0 - headAlpha);
        let lipAlpha = headAlpha + tailAlpha * (1.0 - headAlpha);
        if (lipAlpha <= 0.002) {
            discard;
        }
        return vec4f(premulLip * layerOpacity, lipAlpha * layerOpacity);
    }

    if (isBio) {
        let bIdx = nChem + min(u32(max(bioSpeciesF, 0.0) + 0.5), nBio - 1u);
        let bioCol = vec3f(colours[bIdx * 3u], colours[bIdx * 3u + 1u], colours[bIdx * 3u + 2u]);
        let alpha = 1.0 - smoothstep(1.0 - delta, 1.0, d);
        let energy = clamp(lifeState.x, 0.0, 1.55);
        let vit = clamp(lifeState.y, 0.0, 1.0);
        let poi = clamp(lifeState.z, 0.0, 1.0);
        let graduated = unpackGraduated(lifeState.w);
        let vitalityBright = clamp(0.675 + 0.48 * energy + 0.14 * vit + select(0.0, 0.10, graduated == 1u), 0.45, 1.42);
        var lifeCol = min(bioCol * vitalityBright, vec3f(1.0));
        if (graduated == 1u) {
            let nucBand = smoothstep(0.38, 0.50, d) * (1.0 - smoothstep(0.50, 0.62, d));
            lifeCol = mix(lifeCol, vec3f(0.98, 0.88, 0.48), nucBand * 0.42);
        }
        // Subtle inner tracer dot revealing the conserved chemical element carried inside this organism
        if (d < 0.28 + delta) {
            let coreMix = (1.0 - smoothstep(0.18, 0.28 + delta, d)) * 0.42;
            lifeCol = mix(lifeCol, chemCol, coreMix);
        }
        if (poi > 0.15) {
            let rim = smoothstep(0.62, 0.92, d) * clamp((poi - 0.15) * 1.15, 0.0, 0.85);
            lifeCol = mix(lifeCol, vec3f(0.88, 0.25, 0.96), rim);
        }
        if (vit > 0.25 || graduated == 1u) {
            let glowStrength = max(clamp((vit - 0.25) * 0.6, 0.0, 0.45), select(0.0, 0.34, graduated == 1u));
            let glowRim = smoothstep(0.68, 0.95, d) * glowStrength;
            let glowCol = mix(vec3f(0.32, 0.98, 0.78), vec3f(1.0, 0.86, 0.42), select(0.0, 0.65, graduated == 1u));
            lifeCol = mix(lifeCol, glowCol, glowRim);
        }
        let lifeA = alpha * layerOpacity;
        return vec4f(lifeCol * lifeA, lifeA);
    }

    let sBase = cIdx * SPECIES_STRIDE;
    let valence = u32(matrix[sBase + 0u]);
    let role = matrix[sBase + 1u];

    var outCol = vec3f(0.0);
    var outAlpha = 0.0;

    let coreR = select(0.92, 0.45, uniforms.flags.x > 0.5);
    if (d < coreR + delta) {
        let coreA = 1.0 - smoothstep(coreR - delta, coreR + delta, d);
        let shade = 0.85 + 0.22 * (1.0 - min(d / coreR, 1.0));
        outCol = min(chemCol * shade, vec3f(1.0));
        outAlpha = coreA;
    }

    if (valence == 0u && uniforms.flags.x > 0.5) {
        let ringDist = abs(d - 0.80);
        let ringA = (1.0 - smoothstep(0.04, 0.04 + delta, ringDist)) * 0.45;
        if (ringA > outAlpha) {
            outCol = mix(chemCol, vec3f(0.85, 0.9, 1.0), 0.35);
            outAlpha = ringA;
        }
    } else if (uniforms.flags.x > 0.5) {
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
                    outCol = mix(chemCol, vec3f(1.0, 0.96, 0.82), 0.55 + 0.45 * occ);
                    outAlpha = slotA;
                }
            } else {
                let dotR = 0.13 + 0.03 * occ;
                let dotA = 1.0 - smoothstep(dotR - deltaDot, dotR + deltaDot, distDot);
                if (dotA > outAlpha) {
                    outCol = mix(chemCol, vec3f(1.0, 0.98, 0.88), 0.65 + 0.35 * occ);
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
