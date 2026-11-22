@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var slimeFieldTex: texture_2d<f32>;
@group(0) @binding(2) var membraneFieldTex: texture_2d<f32>;
@group(0) @binding(3) var fieldSampler: sampler;

struct CompositeVertOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vertex(@builtin(vertex_index) vertexIndex: u32) -> CompositeVertOut {
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    let p = pos[vertexIndex];
    var out: CompositeVertOut;
    out.pos = vec4f(p, 0.0, 1.0);
    out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fragment(in: CompositeVertOut) -> @location(0) vec4f {
    let layerOpacity = clamp(uniforms.renderParams.y, 0.0, 1.0);
    if (layerOpacity <= 0.001) {
        discard;
    }

    let dims = vec2f(max(textureDimensions(slimeFieldTex, 0), vec2u(1u)));
    let texel = 1.0 / dims;

    let s = textureSample(slimeFieldTex, fieldSampler, in.uv);
    let sR = textureSample(slimeFieldTex, fieldSampler, in.uv + vec2f(texel.x, 0.0));
    let sL = textureSample(slimeFieldTex, fieldSampler, in.uv - vec2f(texel.x, 0.0));
    let sU = textureSample(slimeFieldTex, fieldSampler, in.uv - vec2f(0.0, texel.y));
    let sD = textureSample(slimeFieldTex, fieldSampler, in.uv + vec2f(0.0, texel.y));

    let m = textureSample(membraneFieldTex, fieldSampler, in.uv);
    let mR = textureSample(membraneFieldTex, fieldSampler, in.uv + vec2f(texel.x, 0.0));
    let mL = textureSample(membraneFieldTex, fieldSampler, in.uv - vec2f(texel.x, 0.0));
    let mU = textureSample(membraneFieldTex, fieldSampler, in.uv - vec2f(0.0, texel.y));
    let mD = textureSample(membraneFieldTex, fieldSampler, in.uv + vec2f(0.0, texel.y));

    let dSlime = s.a;
    let gradSlime = vec2f((sR.a - sL.a) * 0.5, (sU.a - sD.a) * 0.5);
    let dMem = m.r;
    let gradMem = vec2f((mR.r - mL.r) * 0.5, (mU.r - mD.r) * 0.5);

    var outPremul = vec3f(0.0);
    var outAlpha = 0.0;

    // 1. Organic liquid slime & World-of-Goo iso-surface
    let slimeThresh = 0.35;
    let aaSlime = clamp(max(fwidth(dSlime), length(gradSlime) * 0.35), 0.032, 0.14);
    let slimeMask = smoothstep(slimeThresh - aaSlime, slimeThresh + aaSlime * 0.65, dSlime);
    if (slimeMask > 0.002) {
        let baseCol = clamp(s.rgb / max(dSlime, 1e-4), vec3f(0.0), vec3f(1.25));
        let normal2 = vec3f(-gradSlime.x * 2.8, -gradSlime.y * 2.8, 0.38);
        let N = normalize(normal2);
        let lightDir = normalize(vec3f(-0.46, -0.58, 0.67));
        let spec = pow(max(dot(N, lightDir), 0.0), 18.0) * 0.36;

        let depth = smoothstep(slimeThresh, 1.15, dSlime);
        let meniscus = 1.0 - smoothstep(slimeThresh, slimeThresh + 0.18, dSlime);
        var slimeCol = mix(baseCol * 1.06, baseCol * 0.62, meniscus * 0.65);
        slimeCol += vec3f(1.0, 0.99, 0.96) * spec * smoothstep(slimeThresh + 0.02, slimeThresh + 0.28, dSlime);
        slimeCol = min(slimeCol, vec3f(1.0));

        let bodyAlpha = slimeMask * mix(0.76, 0.88, depth);
        outPremul = slimeCol * bodyAlpha;
        outAlpha = bodyAlpha;
    }

    // 2. Closed multicellular lipid membrane envelope
    let memThresh = 0.25;
    let aaMem = clamp(max(fwidth(dMem), length(gradMem) * 0.35), 0.03, 0.12);
    let memMask = smoothstep(memThresh - aaMem, memThresh + aaMem * 0.65, dMem);
    if (memMask > 0.002) {
        let gradFrac = clamp(m.g / max(dMem, 1e-4), 0.0, 1.0);
        let poiFrac = clamp(m.b / max(dMem, 1e-4), 0.0, 1.0);
        let energyFrac = clamp(m.a / max(dMem, 1e-4), 0.45, 1.30);

        let coolSheath = vec3f(0.32, 0.84, 0.94);
        let gradEnvelope = vec3f(0.96, 0.86, 0.46);
        var memCol = mix(coolSheath, gradEnvelope, smoothstep(0.28, 0.85, gradFrac) * 0.72) * clamp(0.76 + 0.28 * energyFrac, 0.62, 1.16);

        let ridge = smoothstep(memThresh + 0.10, memThresh + 0.36, dMem);
        memCol = mix(memCol * 0.82, min(memCol * 1.16, vec3f(1.0)), ridge);
        if (poiFrac > 0.20) {
            memCol = mix(memCol, vec3f(0.92, 0.30, 0.98), clamp((poiFrac - 0.20) * 1.1, 0.0, 0.75));
        }

        let memAlpha = memMask * mix(0.62, 0.82, smoothstep(0.25, 0.85, gradFrac));
        outPremul = memCol * memAlpha + outPremul * (1.0 - memAlpha);
        outAlpha = memAlpha + outAlpha * (1.0 - memAlpha);
    }

    if (outAlpha <= 0.002) {
        discard;
    }
    return vec4f(outPremul * layerOpacity, outAlpha * layerOpacity);
}
