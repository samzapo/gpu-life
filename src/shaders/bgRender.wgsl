@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var windTex: texture_2d<f32>;
@group(0) @binding(2) var watermanInvTex: texture_2d<f32>;
@group(0) @binding(3) var oceanTex: texture_2d<f32>;
@group(0) @binding(4) var tempTex: texture_2d<f32>;
@group(0) @binding(5) var uvTex: texture_2d<f32>;
@group(0) @binding(6) var hydroTex: texture_2d<f32>;
@group(0) @binding(7) var terrainTex: texture_2d<f32>;

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

fn evalTerrainBaseColor(
    elevKm: f32,
    dEast: f32,
    dNorth: f32,
    ridgeMask: f32,
    oceanMask: f32,
    rh01: f32,
    precipMm: f32,
    tS: vec4f,
) -> vec3f {
    let seaBlend = smoothstep(0.36, 0.62, oceanMask);
    // Ocean shelf bathymetry + deep basins + subtle tectonic ridge mineral relief
    let shelfT = smoothstep(-0.32, 0.0, elevKm);
    let deepSeaCol = vec3f(0.032, 0.068, 0.128);
    let shelfCol = vec3f(0.068, 0.165, 0.245);
    var seaCol = mix(deepSeaCol, shelfCol, shelfT);
    let ridgeTint = vec3f(0.145, 0.115, 0.095);
    seaCol = mix(seaCol, ridgeTint, clamp(ridgeMask * 0.22, 0.0, 0.22));
    // Polar & freezing ocean pack ice
    let seaIce = smoothstep(273.0, 270.0, tS.y) * smoothstep(0.30, 0.60, tS.z);
    seaCol = mix(seaCol, vec3f(0.76, 0.85, 0.92), seaIce * 0.78);

    // Continental hypsometric relief + climate vegetation tint + northwest cartographic hillshading
    let moist = clamp(0.55 * smoothstep(0.28, 0.88, rh01) + 0.45 * smoothstep(0.15, 3.5, precipMm), 0.0, 1.0);
    let lowArid = vec3f(0.34, 0.29, 0.22);
    let lowLush = vec3f(0.11, 0.24, 0.12);
    let lowlandCol = mix(lowArid, lowLush, moist);
    let uplandCol = mix(vec3f(0.32, 0.26, 0.19), vec3f(0.18, 0.27, 0.15), moist * 0.55);
    let alpineRock = vec3f(0.39, 0.36, 0.34);
    let snowPeak = vec3f(0.84, 0.88, 0.93);

    var landCol = mix(lowlandCol, uplandCol, smoothstep(0.15, 1.35, elevKm));
    landCol = mix(landCol, alpineRock, smoothstep(1.35, 3.4, elevKm));
    let snowTrigger = smoothstep(2.4, 4.8, elevKm) + smoothstep(274.5, 258.0, tS.w) * 0.85;
    landCol = mix(landCol, snowPeak, clamp(snowTrigger, 0.0, 0.92));

    let reliefGain = 2.6;
    let nSurf = normalize(vec3f(-dEast * reliefGain, dNorth * reliefGain, 1.0));
    let sunLight = normalize(vec3f(-0.55, 0.62, 0.56));
    let hillShade = clamp(0.70 + 0.56 * dot(nSurf, sunLight), 0.64, 1.24);
    landCol *= hillShade;

    return mix(landCol, seaCol, seaBlend);
}

@fragment
fn fragment(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let spaceCol = vec3f(0.0, 0.0, 0.0196); // #000005
    let srcCoord = applyLensInverse(uniforms, fragCoord.xy);
    let sx = srcCoord.x;
    let sy = srcCoord.y;
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

    let gridDims = textureDimensions(oceanTex, 0);
    let eq = makeEquirectCoord(inv.xyz, gridDims);

    let terrainOpacity = clamp(uniforms.terrainParams.x, 0.0, 1.0);
    let nightShading = clamp(uniforms.terrainParams.y, 0.0, 1.0);
    let oceanColOpacity = clamp(uniforms.earthFlags.x, 0.0, 1.0);
    let tempColOpacity = clamp(uniforms.mineralParams.y, 0.0, 1.0);
    let windColOpacity = clamp(uniforms.flags.y, 0.0, 1.0);
    let uvColOpacity = clamp(uniforms.uvParams.x, 0.0, 1.0);
    let rhColOpacity = clamp(uniforms.hydroParams.x, 0.0, 1.0);
    let precipColOpacity = clamp(uniforms.hydroParams.y, 0.0, 1.0);

    var oc = vec4f(0.0);
    if (terrainOpacity > 0.001 || oceanColOpacity > 0.001 || tempColOpacity > 0.001) {
        oc = sampleOceanTextureCoord(oceanTex, eq);
    }

    var tS = vec4f(288.15, 288.15, 0.0, 288.15);
    if (terrainOpacity > 0.001 || tempColOpacity > 0.001) {
        let c00 = textureLoad(tempTex, vec2u(eq.i0, eq.j0), 0);
        let c10 = textureLoad(tempTex, vec2u(eq.i1, eq.j0), 0);
        let c01 = textureLoad(tempTex, vec2u(eq.i0, eq.j1), 0);
        let c11 = textureLoad(tempTex, vec2u(eq.i1, eq.j1), 0);
        let airK = c00.x * eq.w00 + c10.x * eq.w10 + c01.x * eq.w01 + c11.x * eq.w11;
        let sfcK = c00.w * eq.w00 + c10.w * eq.w10 + c01.w * eq.w01 + c11.w * eq.w11;
        let k00 = eq.w00 * select(0.0, 1.0, c00.z > 0.45);
        let k10 = eq.w10 * select(0.0, 1.0, c10.z > 0.45);
        let k01 = eq.w01 * select(0.0, 1.0, c01.z > 0.45);
        let k11 = eq.w11 * select(0.0, 1.0, c11.z > 0.45);
        let maskSum = k00 + k10 + k01 + k11;
        let sstK = select(
            sfcK,
            (c00.y * k00 + c10.y * k10 + c01.y * k01 + c11.y * k11) / max(maskSum, 1e-4),
            maskSum > 1e-4,
        );
        tS = vec4f(airK, sstK, maskSum, sfcK);
    }

    var hS = vec2f(0.65, 0.0);
    if (terrainOpacity > 0.001 || rhColOpacity > 0.001 || precipColOpacity > 0.001) {
        hS = sampleHydroTextureCoord(hydroTex, eq);
    }

    if (terrainOpacity > 0.001) {
        let terr = sampleTerrainTextureCoord(terrainTex, eq);
        let terrCol = evalTerrainBaseColor(
            terr.x,
            terr.y,
            terr.z,
            terr.w,
            oc.w,
            hS.x,
            hS.y,
            tS,
        );
        baseCol = mix(baseCol, terrCol, terrainOpacity);
    }

    if (oceanColOpacity > 0.001) {
        if (oc.w >= 0.42) {
            let ocCol = oceanCurrentsColor(oc.z);
            let ocAlpha = smoothstep(0.42, 0.72, oc.w) * oceanColOpacity;
            baseCol = mix(baseCol, ocCol, ocAlpha);
        } else if (terrainOpacity <= 0.001) {
            baseCol *= mix(1.0, 0.82, clamp(0.35 + 0.65 * oceanColOpacity, 0.0, 1.0));
        }
    }

    if (windColOpacity > 0.001) {
        let c00 = textureLoad(windTex, vec2u(eq.i0, eq.j0), 0).xyz;
        let c10 = textureLoad(windTex, vec2u(eq.i1, eq.j0), 0).xyz;
        let c01 = textureLoad(windTex, vec2u(eq.i0, eq.j1), 0).xyz;
        let c11 = textureLoad(windTex, vec2u(eq.i1, eq.j1), 0).xyz;
        let w = c00 * eq.w00 + c10 * eq.w10 + c01 * eq.w01 + c11 * eq.w11;
        let t = clamp(w.z / max(uniforms.flags.z, 1.0), 0.0, 1.0);
        baseCol = mix(baseCol, extendedSinebowColor(t), windColOpacity);
    }

    if (tempColOpacity > 0.001) {
        let layerMode = u32(uniforms.mineralParams.w + 0.5);
        if (layerMode == 2u) {
            baseCol = mix(baseCol, airTempColor(tS.x), tempColOpacity);
        } else if (layerMode == 1u) {
            let seaAlpha = smoothstep(0.42, 0.72, max(oc.w, tS.z));
            if (seaAlpha > 0.001) {
                baseCol = mix(baseCol, sstTempColor(tS.y), seaAlpha * tempColOpacity);
            }
        } else {
            let seaW = smoothstep(0.42, 0.72, min(oc.w, tS.z));
            let tCol = mix(airTempColor(tS.x), sstTempColor(tS.y), seaW);
            baseCol = mix(baseCol, tCol, tempColOpacity);
        }
    }

    var uvS = vec4f(4.0, 7.0, 300.0, 0.25);
    if (uvColOpacity > 0.001 || nightShading > 0.001) {
        uvS = sampleUvTextureCoord(uvTex, eq);
        if (uvColOpacity > 0.001) {
            baseCol = mix(baseCol, uvIndexColor(uvS.x), uvColOpacity);
        }
    }

    if (rhColOpacity > 0.001) {
        baseCol = mix(baseCol, rhColor(hS.x), rhColOpacity);
    }
    if (precipColOpacity > 0.001) {
        let rainAlpha = smoothstep(0.05, 0.75, hS.y) * precipColOpacity;
        if (rainAlpha > 0.001) {
            baseCol = mix(baseCol, precip3hColor(hS.y), rainAlpha);
        }
    }

    if (nightShading > 0.001) {
        let sunlit = smoothstep(0.0, 1.15, uvS.x);
        let nightDim = mix(1.0 - 0.58 * nightShading, 1.0, sunlit);
        baseCol *= nightDim;
    }

    return vec4f(baseCol, 1.0);
}
