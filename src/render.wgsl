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
    @location(8) nightFactor: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var watermanFwdTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> colours: array<f32>;
@group(0) @binding(3) var<storage, read> matrix: array<f32>;
@group(0) @binding(4) var uvTex: texture_2d<f32>;

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

fn sdHex(p: vec2f, r: f32) -> f32 {
    let q = abs(p);
    return max(q.x * 0.8660254 + q.y * 0.5, q.y) - r;
}

fn sdBox(p: vec2f, b: vec2f) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdRhomb(p: vec2f, ax: f32, ay: f32, r: f32) -> f32 {
    let q = abs(p);
    return (q.x * ax + q.y * ay) - r;
}

fn rot2D(p: vec2f, c: f32, s: f32) -> vec2f {
    return vec2f(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn foldD6(pIn: vec2f) -> vec2f {
    var q = abs(pIn);
    let n60 = vec2f(-0.8660254, 0.5);
    q -= 2.0 * max(0.0, dot(q, n60)) * n60;
    q.y = abs(q.y);
    let n30 = vec2f(-0.5, 0.8660254);
    q -= 2.0 * max(0.0, dot(q, n30)) * n30;
    q.y = abs(q.y);
    return q;
}

fn foldD4(pIn: vec2f) -> vec2f {
    let a = abs(pIn);
    return select(a, vec2f(a.y, a.x), a.y > a.x);
}

fn vertexCommon(
    particle: ParticleVert,
    vertexIndex: u32,
    filterMinerals: bool,
    matchMinerals: bool,
    shadowPass: bool,
) -> VertexOutput {
    var positions = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );

    let phase = unpackPhase(particle.lifeState.w);
    let bioSp = unpackBioSpecies(particle.lifeState.w);
    let flightMode = unpackFlightMode(particle.lifeState.w);
    let encapGene = unpackEncapGene(particle.lifeState.w);
    let graduated = unpackGraduated(particle.lifeState.w);
    let phaseF = f32(phase);
    let bioSpeciesF = f32(bioSp);
    var layerOpacity = uniforms.renderParams.y;
    if (phase == 0u) {
        layerOpacity = uniforms.earthFlags.w;
    } else if (phase == 3u) {
        layerOpacity = uniforms.mineralParams.x;
    }

    var output: VertexOutput;
    let isShadowBird = (phase == 1u && flightMode == 1u && clamp(particle.chemGlow, 0.0, 1.0) > 0.06);
    let passMismatch = (shadowPass && !isShadowBird) || (!shadowPass && filterMinerals && ((phase == 3u) != matchMinerals));
    if (passMismatch || layerOpacity <= 0.001) {
        output.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        output.uv = vec2f(0.0);
        output.chemSpecies = particle.colour;
        output.slots = particle.slots;
        output.chemGlow = particle.chemGlow;
        output.limbFade = 0.0;
        output.phaseF = phaseF;
        output.bioSpeciesF = bioSpeciesF;
        output.lifeState = particle.lifeState;
        output.nightFactor = 0.0;
        return output;
    }

    let n = normalize(particle.pos);
    let pType = u32(uniforms.viewParams.w + 0.5);
    let vView = rotateToView(uniforms, n);
    let altScale = clamp(uniforms.terrainParams.z, 0.0, 0.25);
    var altExag = 0.0;
    if (!shadowPass && altScale > 0.0005) {
        if (phase == 1u && flightMode == 1u) {
            altExag = clamp(particle.chemGlow, 0.0, 1.0) * altScale;
        } else if (phase == 0u && particle.lifeState.y < 0.5 && particle.lifeState.z > 0.04) {
            altExag = clamp(particle.lifeState.z, 0.0, 1.0) * altScale * 0.45;
        }
    }

    var p0 = vec3f(0.0);
    if (pType == 0u && altExag > 0.0005) {
        let rScale = 1.0 + altExag;
        let elevR2D = length(vec2f(vView.y, vView.z)) * rScale;
        let visible = (vView.x >= 0.0) || (elevR2D >= 1.0);
        p0 = vec3f(vView.y * rScale, -vView.z * rScale, select(0.0, 1.0, visible));
    } else {
        p0 = projectUnitPos(uniforms, watermanFwdTex, n);
        if (pType != 0u && altExag > 0.0005 && p0.z > 0.5) {
            p0.y -= altExag * 0.35;
        }
    }

    if (p0.z < 0.5) {
        output.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        output.uv = vec2f(0.0);
        output.chemSpecies = particle.colour;
        output.slots = particle.slots;
        output.chemGlow = particle.chemGlow;
        output.limbFade = 0.0;
        output.phaseF = phaseF;
        output.bioSpeciesF = bioSpeciesF;
        output.lifeState = particle.lifeState;
        output.nightFactor = 0.0;
        return output;
    }

    let uHat = orthoBasis(n, particle.orient);
    let bHat = normalize(cross(n, uHat));

    var limbFade = 1.0;
    var dU = vec2f(1.0, 0.0);
    var dB = vec2f(0.0, -1.0);

    if (pType == 0u) {
        limbFade = select(smoothstep(0.005, 0.16, vView.x), smoothstep(-0.18, 0.12, vView.x), altExag > 0.0005);
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
            let kRad = mix(0.48, 1.0, max(vView.x, 0.0));
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
        let volFrac = f32(encapGene) / 15.0;
        let integ = clamp(particle.lifeState.x, 0.08, 1.15);
        angR = max(bioAngR * 0.96, chemAngR * 1.12) * mix(0.76, 1.28, clamp(volFrac * 0.65 + integ * 0.35, 0.0, 1.0));
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

    let nightFactor = select(0.0, smoothstep(0.85, 0.0, sampleUvTexture(uvTex, n).x) * clamp(uniforms.terrainParams.y, 0.0, 1.0), phase == 1u && !shadowPass);
    output.pos = vec4f(ndcX, ndcY, 0.0, 1.0);
    output.uv = corner;
    output.chemSpecies = particle.colour;
    output.slots = particle.slots;
    output.chemGlow = particle.chemGlow;
    output.limbFade = limbFade;
    output.phaseF = phaseF;
    output.bioSpeciesF = bioSpeciesF;
    output.lifeState = particle.lifeState;
    output.nightFactor = nightFactor;
    return output;
}

@vertex
fn vertexMinerals(
    particle: ParticleVert,
    @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
    return vertexCommon(particle, vertexIndex, true, true, false);
}

@vertex
fn vertexForeground(
    particle: ParticleVert,
    @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
    return vertexCommon(particle, vertexIndex, true, false, false);
}

@vertex
fn vertexBirdShadows(
    particle: ParticleVert,
    @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
    return vertexCommon(particle, vertexIndex, false, false, true);
}

@vertex
fn vertex(
    particle: ParticleVert,
    @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
    return vertexCommon(particle, vertexIndex, false, false, false);
}

@fragment
fn fragmentShadow(
    @location(0) uv: vec2f,
    @location(1) chemSpecies: f32,
    @location(2) slots: vec4f,
    @location(3) chemGlow: f32,
    @location(4) limbFade: f32,
    @location(5) phaseF: f32,
    @location(6) bioSpeciesF: f32,
    @location(7) lifeState: vec4f,
    @location(8) nightFactor: f32,
) -> @location(0) vec4f {
    let d = length(uv);
    let op = clamp(uniforms.renderParams.y, 0.0, 1.0) * clamp(limbFade, 0.0, 1.0);
    if (d > 1.0 || op <= 0.001) {
        discard;
    }
    let a = (1.0 - smoothstep(0.15, 0.98, d)) * 0.28 * op;
    return vec4f(0.0, 0.0, 0.0, a);
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
    @location(8) nightFactor: f32,
) -> @location(0) vec4f {
    let uvStep = length(fwidth(uv));
    let px = clamp(uvStep * 0.70710678, 0.0005, 0.22);
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
        let volFrac = f32(unpackEncapGene(lifeState.w)) / 15.0;
        let isTrap = unpackGraduated(lifeState.w) == 1u;
        let grow = clamp((0.28 + 0.72 * volFrac) * (0.52 + 0.48 * min(integ, 1.0)), 0.16, 1.0);

        let edgeW = max(px * 0.90, 1e-4);
        let lineW = max(px * 1.45, 1e-4);

        var sdf = 1.0;
        var baseCol = vec3f(0.65);
        var facetShade = 1.0;
        var accentCol = vec3f(0.95);
        var accentMask = 0.0;
        var innerLine = 99.0;
        var rimCol = vec3f(0.95);

        if (minType == 0u) {
            // 0: Silicate Quartz — Self-similar D6 Koch / hexagonal star-dendrite fractal
            let q0 = foldD6(uv);
            var df = sdHex(q0, 0.42);
            if (grow > 0.20) {
                let c1 = vec2f(0.42 * grow, 0.0);
                let q1 = foldD6(q0 - c1);
                df = min(df, sdHex(q1, 0.21 * grow));
                if (grow > 0.46) {
                    let c2 = vec2f(0.22 * grow, 0.0);
                    let q2 = foldD6(q1 - c2);
                    df = min(df, sdHex(q2, 0.105 * grow));
                    if (grow > 0.74) {
                        df = min(df, sdHex(q2 - vec2f(0.115 * grow, 0.0), 0.055 * grow));
                    }
                }
            }
            sdf = df;
            let sectorSign = select(-0.16, 0.18, uv.x * 0.65 + uv.y * 0.75 > 0.0);
            let facetPlane = select(-0.10, 0.12, q0.y > q0.x * 0.28);
            facetShade = clamp(0.90 + sectorSign + facetPlane, 0.64, 1.24);
            baseCol = mix(vec3f(0.46, 0.56, 0.66), chemCol, 0.26);
            innerLine = min(q0.y, abs(df + 0.14));
            accentCol = vec3f(0.86, 0.95, 1.0);
            rimCol = vec3f(0.90, 0.97, 1.0);
        } else if (minType == 1u) {
            // 1: Evaporite Halite / Fluorite — Self-similar D4 corner-cube & stepped hopper fractal
            let q0 = foldD4(uv);
            let boxCore = sdBox(q0, vec2f(0.42, 0.42));
            var df = boxCore;
            if (grow > 0.22) {
                df = min(df, sdBox(q0 - vec2f(0.36 * grow, 0.36 * grow), vec2f(0.19 * grow, 0.19 * grow)));
                df = min(df, sdBox(q0 - vec2f(0.44 * grow, 0.0), vec2f(0.16 * grow, 0.10 * grow)));
                if (grow > 0.50) {
                    df = min(df, sdBox(q0 - vec2f(0.55 * grow, 0.55 * grow), vec2f(0.10 * grow, 0.10 * grow)));
                    if (grow > 0.76) {
                        df = min(df, sdBox(q0 - vec2f(0.65 * grow, 0.65 * grow), vec2f(0.055 * grow, 0.055 * grow)));
                    }
                }
            }
            sdf = df;
            facetShade = clamp(0.86 + select(-0.14, 0.18, q0.x > q0.y * 1.35) + select(-0.08, 0.10, uv.x > -uv.y), 0.66, 1.22);
            baseCol = mix(vec3f(0.84, 0.93, 0.99), chemCol, 0.16);
            innerLine = min(abs(boxCore + 0.16), abs(boxCore + 0.30));
            let trapCore = abs(uv.x) + abs(uv.y) - 0.24;
            accentMask = select(0.0, 1.0 - smoothstep(-edgeW, edgeW, trapCore), isTrap || haz > 0.55);
            accentCol = vec3f(0.40, 1.0, 0.50);
            rimCol = select(vec3f(0.95, 0.99, 1.0), vec3f(0.58, 1.0, 0.62), isTrap);
        } else if (minType == 2u) {
            // 2: Carbonate Dolomite / Beryl Trap — Self-similar dogtooth rhombic twin fractal
            let q0 = abs(uv);
            let r0 = sdRhomb(q0, 1.15, 0.85, 0.50);
            var df = r0;
            if (grow > 0.22) {
                let b1 = sdRhomb(q0 - vec2f(0.28 * grow, 0.0), 1.20, 0.92, 0.24 * grow);
                let b2 = sdRhomb(q0 - vec2f(0.0, 0.34 * grow), 1.20, 0.92, 0.20 * grow);
                df = min(df, min(b1, b2));
                if (grow > 0.52) {
                    df = min(df, sdRhomb(q0 - vec2f(0.44 * grow, 0.0), 1.25, 0.95, 0.12 * grow));
                }
            }
            sdf = df;
            facetShade = clamp(0.90 + select(-0.16, 0.18, (uv.y - 0.38 * uv.x) > 0.0), 0.66, 1.20);
            baseCol = mix(vec3f(0.92, 0.80, 0.52), chemCol, 0.22);
            innerLine = min(abs(r0 + 0.18), abs(uv.y - 0.38 * uv.x) * 0.75);
            let berylHex = sdHex(uv, 0.22);
            accentMask = select(0.0, 1.0 - smoothstep(-edgeW, edgeW, berylHex), isTrap);
            accentCol = vec3f(0.14, 0.96, 0.54);
            rimCol = select(vec3f(1.0, 0.94, 0.74), vec3f(0.95, 0.36, 0.78), isTrap && haz > 0.55);
        } else if (minType == 3u) {
            // 3: Sulfide Chimney / Pyrite — Self-similar twin-cube octahedron & pyrite needle fractal
            let qRot = rot2D(uv, 0.70710678, 0.70710678);
            var df = min(sdBox(uv, vec2f(0.40, 0.40)), sdBox(qRot, vec2f(0.40, 0.40)));
            let q4 = foldD4(uv);
            if (grow > 0.24) {
                let twigA = q4 - vec2f(0.42 * grow, 0.0);
                let twigB = rot2D(q4 - vec2f(0.36 * grow, 0.36 * grow), 0.70710678, 0.70710678);
                df = min(df, min(sdBox(twigA, vec2f(0.16 * grow, 0.16 * grow)), sdBox(twigB, vec2f(0.15 * grow, 0.15 * grow))));
                if (grow > 0.56) {
                    df = min(df, sdBox(q4 - vec2f(0.58 * grow, 0.0), vec2f(0.09 * grow, 0.09 * grow)));
                }
            }
            sdf = df;
            let octLight = select(-0.18, 0.24, (uv.x * uv.y) > 0.0);
            facetShade = clamp(0.88 + octLight, 0.62, 1.28);
            baseCol = select(vec3f(0.24, 0.22, 0.27), vec3f(0.94, 0.78, 0.28), (q4.x - q4.y) > 0.12);
            innerLine = abs(sdBox(uv, vec2f(0.40, 0.40)) - sdBox(qRot, vec2f(0.40, 0.40))) * 0.65;
            let ventHole = (abs(uv.x) + abs(uv.y)) - 0.18;
            accentMask = (1.0 - smoothstep(-edgeW, edgeW, ventHole)) * clamp(0.55 + 0.45 * max(nutr, haz), 0.55, 1.0);
            accentCol = select(vec3f(1.0, 0.48, 0.12), vec3f(0.86, 0.32, 0.98), isTrap);
            rimCol = select(vec3f(1.0, 0.86, 0.42), vec3f(0.95, 0.46, 1.0), isTrap);
        } else {
            // 4: Hydrated Clay / Zeolite — Self-similar D6 stepped rosette & catalytic turquoise cleavage
            let r1 = rot2D(uv, 0.9396926, 0.3420201);
            let r2 = rot2D(uv, 0.9396926, -0.3420201);
            var df = sdHex(uv, 0.42);
            if (grow > 0.20) {
                df = min(df, min(sdHex(r1 * 1.15 - vec2f(0.18 * grow, 0.0), 0.36 * grow), sdHex(r2 * 1.15 - vec2f(0.0, 0.18 * grow), 0.34 * grow)));
                if (grow > 0.48) {
                    let q1 = foldD6(uv) - vec2f(0.46 * grow, 0.0);
                    df = min(df, sdHex(q1, 0.18 * grow));
                }
            }
            sdf = df;
            let bandVal = 0.5 + 0.5 * cos(uv.y * 22.0);
            facetShade = clamp(0.84 + 0.18 * bandVal + 0.10 * sign(uv.x), 0.66, 1.18);
            baseCol = mix(vec3f(0.76, 0.50, 0.36), chemCol, 0.18);
            innerLine = abs(fract(uv.y * 3.5 + 0.5) - 0.5) * 0.28;
            accentCol = vec3f(0.22, 0.96, 0.88);
            accentMask = (1.0 - smoothstep(lineW * 0.4, lineW * 1.5, innerLine)) * 0.72;
            rimCol = vec3f(0.38, 0.98, 0.90);
        }

        let crystAlpha = 1.0 - smoothstep(-edgeW, edgeW, sdf);
        if (crystAlpha <= 0.002) {
            discard;
        }
        var col = min(baseCol * facetShade, vec3f(1.0));
        col = mix(col, accentCol, accentMask);
        let seamMask = (1.0 - smoothstep(lineW * 0.35, lineW * 1.35, innerLine)) * 0.65;
        let rimMask = (1.0 - smoothstep(lineW * 0.35, lineW * 1.45, abs(sdf))) * 0.70;
        col = min(mix(col, rimCol, max(seamMask, rimMask)), vec3f(1.0));
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
        if (vit > 0.25 || graduated == 1u || nightFactor > 0.15) {
            let bioLum = nightFactor * 0.38 * clamp(energy, 0.3, 1.2);
            let glowStrength = max(max(clamp((vit - 0.25) * 0.6, 0.0, 0.45), select(0.0, 0.34, graduated == 1u)), bioLum);
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
