struct LensOverlayTrackUniform {
    targetSpec: vec4f,
    ctrl: vec4f,
    ringColor: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> track: LensOverlayTrackUniform;

struct LensOverlayOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> LensOverlayOut {
    var verts = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var out: LensOverlayOut;
    out.pos = vec4f(verts[vid], 0.0, 1.0);
    return out;
}

@fragment
fn fragment(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let rLens = uniforms.lensParams.z;
    let mag = uniforms.lensParams.w;
    if (rLens <= 1.0 || mag <= 1.01) {
        discard;
    }

    let dVec = fragCoord.xy - uniforms.lensParams.xy;
    let dist = length(dVec);
    if (dist > rLens + 18.0) {
        discard;
    }

    var rgb = vec3f(0.0);
    var alpha = 0.0;

    if (dist <= rLens) {
        // Subtle optical dome highlight + theme inner glow without blurring interior particles
        let hlOff = dVec - vec2f(-rLens * 0.34, -rLens * 0.38);
        let hl = exp(-dot(hlOff, hlOff) / (rLens * rLens * 0.38)) * 0.085;
        let innerGlow = smoothstep(rLens * 0.72, rLens, dist) * 0.14;
        rgb = vec3f(1.0) * hl + track.ringColor.rgb * innerGlow;
        alpha = hl + innerGlow * 0.65;

        // Delicate dashed reticle marking the uniform flat-field zone (0.44 * rLens)
        let retR = rLens * 0.44;
        let retStroke = 1.0 - smoothstep(0.0, 1.05, abs(dist - retR));
        if (retStroke > 0.01) {
            let ang = atan2(dVec.y, dVec.x);
            let dash = smoothstep(-0.15, 0.35, sin(ang * 16.0));
            let aRet = retStroke * dash * 0.24;
            rgb = rgb * (1.0 - aRet) + vec3f(1.0) * aRet;
            alpha = alpha + aRet * (1.0 - alpha);
        }
    } else {
        // Soft exterior drop shadow immediately outside the glass bezel
        let outD = dist - rLens;
        alpha = exp(-(outD * outD) / 36.0) * 0.52;
    }

    // Crisp outer dark border + theme-tinted glowing lens rim rendered lockstep with uniforms.lensParams
    let darkEdge = 1.0 - smoothstep(0.95, 2.45, abs(dist - (rLens + 1.15)));
    if (darkEdge > 0.005) {
        let aDark = darkEdge * 0.82;
        rgb = rgb * (1.0 - aDark) + vec3f(0.02, 0.04, 0.07) * aDark;
        alpha = alpha + aDark * (1.0 - alpha);
    }

    let rimStroke = 1.0 - smoothstep(0.55, 1.85, abs(dist - rLens));
    if (rimStroke > 0.005) {
        let aRim = rimStroke * 0.94;
        rgb = rgb * (1.0 - aRim) + track.ringColor.rgb * aRim;
        alpha = alpha + aRim * (1.0 - alpha);
    }

    if (alpha <= 0.002) {
        discard;
    }
    return vec4f(rgb, alpha);
}
