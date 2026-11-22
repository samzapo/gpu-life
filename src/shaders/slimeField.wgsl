@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var watermanFwdTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> colours: array<f32>;

struct ParticleVert {
    @location(0) pos: vec3f,
    @location(1) colour: f32,
    @location(2) orient: vec3f,
    @location(3) slots: vec4f,
    @location(4) chemGlow: f32,
    @location(5) lifeState: vec4f,
}

struct FieldVertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) bioInfo: vec4f, // x: bioSpeciesF, y: energy, z: vit, w: poi
    @location(2) params: vec4f,  // x: phaseF, y: limbFade, z: gradF, w: gapRatioOrZero
    @location(3) bridgeMode: f32,
    @location(4) linkFade: f32,
}

struct FieldFragOut {
    @location(0) slime: vec4f,
    @location(1) membrane: vec4f,
}

fn deadFieldVert() -> FieldVertexOut {
    var out: FieldVertexOut;
    out.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2f(0.0);
    out.bioInfo = vec4f(0.0);
    out.params = vec4f(0.0);
    out.bridgeMode = 0.0;
    out.linkFade = 0.0;
    return out;
}

fn safeOrthoStepField(n: vec3f, p0: vec2f, axis: vec3f) -> vec3f {
    let eps = 0.004;
    let pPlus = projectUnitPos(uniforms, watermanFwdTex, normalize(n + eps * axis));
    if (pPlus.z > 0.5) {
        let d = (pPlus.xy - p0) / eps;
        let l = length(d);
        if (l > 0.02 && l < 6.0) {
            return vec3f(d, 1.0);
        }
    }
    let pMinus = projectUnitPos(uniforms, watermanFwdTex, normalize(n - eps * axis));
    if (pMinus.z > 0.5) {
        let d = (p0 - pMinus.xy) / eps;
        let l = length(d);
        if (l > 0.02 && l < 6.0) {
            return vec3f(d, 1.0);
        }
    }
    return vec3f(0.0);
}

fn bioColorForParticle(bioSp: u32, energy: f32, vit: f32, poi: f32, gradF: f32) -> vec3f {
    let nChem = max(u32(uniforms.renderParams.z + 0.5), 1u);
    let nBio = max(u32(uniforms.renderParams.w + 0.5), 1u);
    let bIdx = nChem + min(bioSp, nBio - 1u);
    let bioCol = vec3f(colours[bIdx * 3u], colours[bIdx * 3u + 1u], colours[bIdx * 3u + 2u]);
    let bright = clamp(0.72 + 0.44 * energy + 0.14 * vit + 0.10 * gradF, 0.48, 1.38);
    var col = min(bioCol * bright, vec3f(1.0));
    if (poi > 0.18) {
        col = mix(col, vec3f(0.88, 0.25, 0.96), clamp((poi - 0.18) * 0.75, 0.0, 0.55));
    }
    return col;
}

@vertex
fn vertexNode(
    particle: ParticleVert,
    @builtin(vertex_index) vertexIndex: u32,
) -> FieldVertexOut {
    let phase = unpackPhase(particle.lifeState.w);
    if ((phase != 1u && phase != 2u) || uniforms.renderParams.y <= 0.001) {
        return deadFieldVert();
    }

    let n = normalize(particle.pos);
    let p0 = projectUnitPos(uniforms, watermanFwdTex, n);
    if (p0.z < 0.5) {
        return deadFieldVert();
    }

    let pType = u32(uniforms.viewParams.w + 0.5);
    let vView = rotateToView(uniforms, n);
    let uHat = orthoBasis(n, particle.orient);
    let bHat = normalize(cross(n, uHat));

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
            let kRad = mix(0.48, 1.0, max(vView.x, 0.0));
            dU = dot(uView, r3) * kRad * eRad2D + dot(uView, t3) * eTan2D;
            dB = dot(bView, r3) * kRad * eRad2D + dot(bView, t3) * eTan2D;
        }
    } else {
        if (pType == 6u) {
            limbFade = smoothstep(0.01, 0.12, 1.0 + vView.x);
        }
        let duProbe = safeOrthoStepField(n, p0.xy, uHat);
        let dbProbe = safeOrthoStepField(n, p0.xy, bHat);
        if (duProbe.z > 0.5 && dbProbe.z > 0.5) {
            var cu = duProbe.xy;
            var cb = dbProbe.xy;
            let lu = length(cu);
            let lb = length(cb);
            let targetMax = min(max(lu, lb), 1.15);
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
    if (limbFade <= 0.002) {
        return deadFieldVert();
    }

    let scalePx = max(uniforms.viewParams.x, 1.0);
    let graduated = unpackGraduated(particle.lifeState.w);
    let bioAngR = max(uniforms.renderParams.x, 1.75 / scalePx);
    let lipidAngR = bioAngR * select(1.16, 1.28, graduated == 1u);
    let angR = select(
        max(lipidAngR * 0.96, 2.2 / scalePx),
        max(bioAngR * 1.60, 2.5 / scalePx),
        phase == 1u,
    );

    let cx = uniforms.viewSize.x * 0.5 + uniforms.viewParams.y;
    let cy = uniforms.viewSize.y * 0.5 + uniforms.viewParams.z;
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

    var quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    let corner = quad[vertexIndex];
    let pScreen = cScreen + (corner.x * dU + corner.y * dB) * (angR * scalePx * localMag);
    let ndcX = (pScreen.x / max(uniforms.viewSize.x, 1.0)) * 2.0 - 1.0;
    let ndcY = 1.0 - (pScreen.y / max(uniforms.viewSize.y, 1.0)) * 2.0;

    let bioSp = unpackBioSpecies(particle.lifeState.w);
    let energy = clamp(particle.lifeState.x, 0.0, 1.55);
    let vit = clamp(particle.lifeState.y, 0.0, 1.0);
    let poi = clamp(particle.lifeState.z, 0.0, 1.0);
    let gradF = select(0.0, 1.0, graduated == 1u);

    var out: FieldVertexOut;
    out.pos = vec4f(ndcX, ndcY, 0.0, 1.0);
    out.uv = corner;
    out.bioInfo = vec4f(f32(bioSp), energy, vit, poi);
    out.params = vec4f(f32(phase), limbFade, gradF, 0.0);
    out.bridgeMode = 0.0;
    out.linkFade = 1.0;
    return out;
}

@vertex
fn vertexBridge(
    particle: ParticleVert,
    @builtin(vertex_index) vertexIndex: u32,
) -> FieldVertexOut {
    let phase = unpackPhase(particle.lifeState.w);
    if ((phase != 1u && phase != 2u) || uniforms.renderParams.y <= 0.001) {
        return deadFieldVert();
    }

    let slotIdx = vertexIndex / 6u;
    let cornerIdx = vertexIndex % 6u;
    let rawSlot = select(particle.slots.xy, particle.slots.zw, slotIdx == 1u);
    if (rawSlot.x < 1024.25) {
        return deadFieldVert();
    }
    let slotMeta = u32(max(floor(rawSlot.x), 0.0));
    if ((slotMeta & 1024u) == 0u) {
        return deadFieldVert();
    }
    let fadeU = slotMeta & 15u;
    if (fadeU == 0u) {
        return deadFieldVert();
    }
    let linkFade = smoothstep(0.0, 1.0, f32(fadeU) / 8.0);
    let link2D = vec2f(rawSlot.x - f32(slotMeta) - 0.5, rawSlot.y);
    let sin2 = dot(link2D, link2D);
    if (sin2 <= 1e-11 || sin2 >= 0.025) {
        return deadFieldVert();
    }

    let n = normalize(particle.pos);
    let eEast = sphereEastBasis(n);
    let eNorth = normalize(cross(n, eEast));
    let tang = link2D.x * eEast + link2D.y * eNorth;
    let tLen2 = dot(tang, tang);
    if (tLen2 <= 1e-11 || tLen2 >= 0.96) {
        return deadFieldVert();
    }
    let nbN = normalize(sqrt(max(0.0, 1.0 - tLen2)) * n + tang);

    let p0 = projectUnitPos(uniforms, watermanFwdTex, n);
    let p1 = projectUnitPos(uniforms, watermanFwdTex, nbN);
    if (p0.z < 0.5 || p1.z < 0.5) {
        return deadFieldVert();
    }

    let angEst = sqrt(tLen2);
    if (length(p1.xy - p0.xy) > max(angEst * 4.5, 0.08)) {
        return deadFieldVert();
    }

    let pType = u32(uniforms.viewParams.w + 0.5);
    var limbFade = 1.0;
    if (pType == 0u) {
        let vx0 = rotateToView(uniforms, n).x;
        let vx1 = rotateToView(uniforms, nbN).x;
        limbFade = min(smoothstep(0.005, 0.16, vx0), smoothstep(0.005, 0.16, vx1));
    } else if (pType == 6u) {
        let vx0 = rotateToView(uniforms, n).x;
        let vx1 = rotateToView(uniforms, nbN).x;
        limbFade = min(smoothstep(0.01, 0.12, 1.0 + vx0), smoothstep(0.01, 0.12, 1.0 + vx1));
    }
    if (limbFade <= 0.002) {
        return deadFieldVert();
    }

    let scalePx = max(uniforms.viewParams.x, 1.0);
    let cx = uniforms.viewSize.x * 0.5 + uniforms.viewParams.y;
    let cy = uniforms.viewSize.y * 0.5 + uniforms.viewParams.z;
    let s0Raw = p0.xy * scalePx + vec2f(cx, cy);
    let s1Raw = p1.xy * scalePx + vec2f(cx, cy);

    var s0 = s0Raw;
    var s1 = s1Raw;
    var localMag = 1.0;
    if (uniforms.lensParams.z > 1.0 && uniforms.lensParams.w > 1.01) {
        let d0 = length(s0Raw - uniforms.lensParams.xy) / uniforms.lensParams.z;
        let d1 = length(s1Raw - uniforms.lensParams.xy) / uniforms.lensParams.z;
        if (d0 < 1.0) {
            s0 = applyLensForward(uniforms, s0Raw);
        }
        if (d1 < 1.0) {
            s1 = applyLensForward(uniforms, s1Raw);
        }
        let dMid = 0.5 * (d0 + d1);
        if (dMid < 1.0) {
            localMag = lensParticleScale(uniforms.lensParams.w, dMid);
        }
    }

    let seg = s1 - s0;
    let L = length(seg);
    if (L < 0.75) {
        return deadFieldVert();
    }
    let dirT = seg / L;
    let dirN = vec2f(-dirT.y, dirT.x);

    let graduated = unpackGraduated(particle.lifeState.w);
    let bioAngR = max(uniforms.renderParams.x, 1.75 / scalePx);
    let rNodePx = bioAngR * scalePx * localMag;
    let gradF = select(0.0, 1.0, graduated == 1u);

    let maxHalfW = select(
        max(rNodePx * mix(0.54, 0.68, gradF), 2.40),
        max(rNodePx * 1.24, 3.00),
        phase == 1u,
    );

    var corners = array<vec2f, 6>(
        vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0, 1.0),
        vec2f(0.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    let c = corners[cornerIdx];
    let pScreen = s0 + dirT * (c.x * L) + dirN * (c.y * maxHalfW);
    let ndcX = (pScreen.x / max(uniforms.viewSize.x, 1.0)) * 2.0 - 1.0;
    let ndcY = 1.0 - (pScreen.y / max(uniforms.viewSize.y, 1.0)) * 2.0;

    let bioSp = unpackBioSpecies(particle.lifeState.w);
    let gapRatio = L / max(rNodePx, 1.0);
    let energy = clamp(particle.lifeState.x, 0.0, 1.55);
    let vit = clamp(particle.lifeState.y, 0.0, 1.0);
    let poi = clamp(particle.lifeState.z, 0.0, 1.0);

    var out: FieldVertexOut;
    out.pos = vec4f(ndcX, ndcY, 0.0, 1.0);
    // For bridges: uv.x = t in [0,1], uv.y = normalized lateral offset
    out.uv = vec2f(c.x, select(c.y, c.y * (maxHalfW / max(rNodePx, 1.0)), phase == 1u));
    out.bioInfo = vec4f(f32(bioSp), energy, vit, poi);
    // For slime bridges, encode min visible normalized waist floor in extra param via gapRatio clamp
    let pxWaistFloor = clamp(2.2 / max(rNodePx, 1.0), 0.0, 0.85);
    out.params = vec4f(f32(phase), limbFade, gradF, select(poi, gapRatio, phase == 1u));
    out.bridgeMode = 1.0 + pxWaistFloor;
    out.linkFade = linkFade;
    return out;
}

@fragment
fn fragment(in: FieldVertexOut) -> FieldFragOut {
    var out: FieldFragOut;
    out.slime = vec4f(0.0);
    out.membrane = vec4f(0.0);

    let isLipid = in.params.x > 1.5;
    let limbFade = in.params.y;
    let gradF = in.params.z;
    let bioSp = u32(max(in.bioInfo.x, 0.0) + 0.5);
    let energy = clamp(in.bioInfo.y, 0.35, 1.55);
    let vit = clamp(in.bioInfo.z, 0.0, 1.0);
    let poi = clamp(in.bioInfo.w, 0.0, 1.0);
    let isBridge = in.bridgeMode > 0.5;
    let pxWaistFloor = max(in.bridgeMode - 1.0, 0.0);
    let fadeScale = in.linkFade;

    if (!isLipid) {
        var w = 0.0;
        if (!isBridge) {
            let r2 = dot(in.uv, in.uv);
            if (r2 >= 1.0) {
                discard;
            }
            let q = 1.0 - r2;
            w = q * q * (0.94 + 0.14 * energy) * limbFade;
        } else {
            // World-of-Goo concave pinched viscous bridge (iso-edge at u ~ 0.55)
            let t = clamp(in.uv.x, 0.0, 1.0);
            let gapRatio = in.params.w;
            let farT = smoothstep(1.8, 3.6, gapRatio);
            let targetWaist = mix(1.06, clamp(0.74 - 0.048 * max(gapRatio - 2.6, 0.0), 0.38, 0.74), farT);
            let waistR = max(targetWaist, pxWaistFloor);
            let flareR = max(1.22, pxWaistFloor);
            let pinchBell = pow(sin(t * PI), 0.82);
            let localR = mix(flareR, waistR, pinchBell);
            let u = abs(in.uv.y) / max(localR, 0.18);
            if (u >= 1.0) {
                discard;
            }
            let q = 1.0 - u * u;
            let bridgeAmp = mix(0.76, 0.72, farT);
            w = q * q * bridgeAmp * limbFade * fadeScale;
        }
        let col = bioColorForParticle(bioSp, in.bioInfo.y, vit, poi, gradF);
        out.slime = vec4f(col * w, w);
        return out;
    }

    // Lipid bilayer membrane node & perimeter wall segment
    var wm = 0.0;
    if (!isBridge) {
        let r2 = dot(in.uv, in.uv);
        if (r2 >= 1.0) {
            discard;
        }
        let q = 1.0 - r2;
        wm = q * q * 0.72 * limbFade;
    } else {
        let u = abs(in.uv.y);
        if (u >= 1.0) {
            discard;
        }
        let q = 1.0 - u * u;
        let gradBoost = mix(0.62, 0.78, gradF);
        wm = q * q * gradBoost * limbFade * fadeScale;
    }

    let hasBothFlag = mix(0.28, 1.0, gradF);
    out.membrane = vec4f(wm, wm * hasBothFlag, wm * poi, wm * energy);
    return out;
}
