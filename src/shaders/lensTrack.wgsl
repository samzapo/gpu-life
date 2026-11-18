struct LensTrackUniform {
    targetSpec: vec4f, // x: active (0=off, 1=on), y: particleIdx, z: radiusPx, w: mag
    ctrl: vec4f,       // x: resetFlag (>0.5 snap), y: smoothAlpha, z: pad, w: pad
    ringColor: vec4f,  // rgba
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var watermanFwdTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> track: LensTrackUniform;
@group(0) @binding(3) var<storage, read> particles: array<Particle>;
@group(0) @binding(4) var<storage, read_write> lensOut: array<vec4f>;

@compute @workgroup_size(1)
fn main() {
    let isActive = track.targetSpec.x;
    let radius = track.targetSpec.z;
    let mag = track.targetSpec.w;
    if (isActive < 0.5 || radius <= 1.0) {
        lensOut[0] = vec4f(0.0, 0.0, 0.0, 1.0);
        lensOut[1] = track.ringColor;
        return;
    }

    let nTotal = arrayLength(&particles);
    let rawIdx = u32(max(track.targetSpec.y, 0.0) + 0.5);
    let worldPos = particles[min(rawIdx, max(nTotal, 1u) - 1u)].pos;

    let pLen = length(worldPos);
    if (pLen < 1e-6) {
        lensOut[0] = vec4f(0.0, 0.0, 0.0, 1.0);
        lensOut[1] = track.ringColor;
        return;
    }

    let n = worldPos / pLen;
    let p0 = projectUnitPos(uniforms, watermanFwdTex, n);
    if (p0.z < 0.5) {
        lensOut[0] = vec4f(0.0, 0.0, 0.0, 1.0);
        lensOut[1] = track.ringColor;
        return;
    }

    let scalePx = max(uniforms.viewParams.x, 1.0);
    let cx = uniforms.viewSize.x * 0.5 + uniforms.viewParams.y;
    let cy = uniforms.viewSize.y * 0.5 + uniforms.viewParams.z;
    let rawScreen = p0.xy * scalePx + vec2f(cx, cy);

    let prev = lensOut[0];
    var outXY = rawScreen;
    if (track.ctrl.x < 0.5 && prev.z > 1.0) {
        let alpha = clamp(track.ctrl.y, 0.05, 1.0);
        outXY = mix(prev.xy, rawScreen, alpha);
    }

    lensOut[0] = vec4f(outXY.x, outXY.y, radius, mag);
    lensOut[1] = track.ringColor;
}
