@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var windTex: texture_2d<f32>;
@group(0) @binding(2) var watermanInvTex: texture_2d<f32>;

struct BgOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> BgOut {
    var verts = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var out: BgOut;
    out.pos = vec4f(verts[vid], 0.0, 1.0);
    return out;
}

@fragment
fn fragment(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let spaceCol = vec3f(0.0, 0.0, 0.0196); // #000005
    let sx = fragCoord.x;
    let sy = fragCoord.y;
    let scale = max(uniforms.viewParams.x, 1.0);
    let cx = uniforms.viewSize.x * 0.5 + uniforms.viewParams.y;
    let cy = uniforms.viewSize.y * 0.5 + uniforms.viewParams.z;
    let xUnit = (sx - cx) / scale;
    let yUnit = (sy - cy) / scale;
    let screenUv = vec2f(sx / max(uniforms.viewSize.x, 1.0), sy / max(uniforms.viewSize.y, 1.0));

    let inv = invertUnitPos(uniforms, watermanInvTex, screenUv, xUnit, yUnit);
    if (inv.w < 0.5) {
        return vec4f(spaceCol, 1.0);
    }

    var baseCol = vec3f(0.15, 0.15, 0.165);
    let pType = u32(uniforms.viewParams.w + 0.5);
    if (pType == 0u) {
        // Orthographic radial limb gradient matching globes.js (#303030 -> #202020 -> #000005)
        let r = clamp(length(vec2f(xUnit, yUnit)), 0.0, 1.0);
        if (r < 0.69) {
            baseCol = vec3f(0.188, 0.188, 0.195);
        } else if (r < 0.91) {
            let f = (r - 0.69) / (0.91 - 0.69);
            baseCol = mix(vec3f(0.188, 0.188, 0.195), vec3f(0.125, 0.125, 0.133), f);
        } else {
            let f = clamp((r - 0.91) / (0.97 - 0.91), 0.0, 1.0);
            baseCol = mix(vec3f(0.125, 0.125, 0.133), spaceCol, f);
        }
    }

    if (uniforms.flags.y > 0.5) {
        let w = sampleWindTexture(windTex, inv.xyz);
        let t = clamp(w.z / max(uniforms.flags.z, 1.0), 0.0, 1.0);
        let speedCol = extendedSinebowColor(t);
        baseCol = mix(baseCol, speedCol, 0.44);
    }

    return vec4f(baseCol, 1.0);
}
