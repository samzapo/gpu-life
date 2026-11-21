// Shared geographic coordinate transforms and forward/inverse map projections
// matching D3 (`earth.nullschool.net`):
//   0: orthographic, 1: atlantis, 2: conic_equidistant, 3: equirectangular,
//   4: equal_earth, 5: patterson, 6: stereographic, 7: waterman, 8: winkel3

fn rotateToView(u: Uniforms, n: vec3f) -> vec3f {
    return n.x * u.rotCol0.xyz + n.y * u.rotCol1.xyz + n.z * u.rotCol2.xyz;
}

fn rotateFromView(u: Uniforms, v: vec3f) -> vec3f {
    return vec3f(
        dot(u.rotCol0.xyz, v),
        dot(u.rotCol1.xyz, v),
        dot(u.rotCol2.xyz, v),
    );
}

fn safeAsin(x: f32) -> f32 {
    return asin(clamp(x, -1.0, 1.0));
}

fn safeAcos(x: f32) -> f32 {
    return acos(clamp(x, -1.0, 1.0));
}

struct EquirectCoord {
    i0: u32,
    i1: u32,
    j0: u32,
    j1: u32,
    w00: f32,
    w10: f32,
    w01: f32,
    w11: f32,
}

// Computes equirectangular grid indices & bilinear weights ONCE for a `(nx x ny)` globe texture.
fn makeEquirectCoord(n: vec3f, dims: vec2u) -> EquirectCoord {
    let nx = f32(max(dims.x, 2u));
    let ny = f32(max(dims.y, 2u));
    var lonDeg = atan2(n.y, n.x) * (180.0 / PI);
    if (lonDeg < 0.0) {
        lonDeg += 360.0;
    }
    let latDeg = clamp(safeAsin(n.z) * (180.0 / PI), -90.0, 90.0);
    let gx = (lonDeg / 360.0) * nx;
    let gy = ((90.0 - latDeg) / 180.0) * (ny - 1.0);
    let i0 = u32(floor(gx)) % max(dims.x, 1u);
    let i1 = (i0 + 1u) % max(dims.x, 1u);
    let fx = gx - floor(gx);
    let j0 = min(max(dims.y, 1u) - 1u, u32(max(0.0, floor(gy))));
    let j1 = min(max(dims.y, 1u) - 1u, j0 + 1u);
    let fy = clamp(gy - f32(j0), 0.0, 1.0);
    return EquirectCoord(
        i0,
        i1,
        j0,
        j1,
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    );
}

fn sampleWindTextureCoord(windTex: texture_2d<f32>, eq: EquirectCoord) -> vec3f {
    let c00 = textureLoad(windTex, vec2u(eq.i0, eq.j0), 0).xyz;
    let c10 = textureLoad(windTex, vec2u(eq.i1, eq.j0), 0).xyz;
    let c01 = textureLoad(windTex, vec2u(eq.i0, eq.j1), 0).xyz;
    let c11 = textureLoad(windTex, vec2u(eq.i1, eq.j1), 0).xyz;
    return c00 * eq.w00 + c10 * eq.w10 + c01 * eq.w01 + c11 * eq.w11;
}

// Samples wind `(uEast, vNorth, speed)` in m/s at unit sphere normal `n`
fn sampleWindTexture(windTex: texture_2d<f32>, n: vec3f) -> vec3f {
    let dims = textureDimensions(windTex, 0);
    if (dims.x < 2u || dims.y < 2u) {
        return vec3f(0.0);
    }
    return sampleWindTextureCoord(windTex, makeEquirectCoord(n, dims));
}

fn sampleOceanTextureCoord(oceanTex: texture_2d<f32>, eq: EquirectCoord) -> vec4f {
    let c00 = textureLoad(oceanTex, vec2u(eq.i0, eq.j0), 0);
    let c10 = textureLoad(oceanTex, vec2u(eq.i1, eq.j0), 0);
    let c01 = textureLoad(oceanTex, vec2u(eq.i0, eq.j1), 0);
    let c11 = textureLoad(oceanTex, vec2u(eq.i1, eq.j1), 0);

    let k00 = eq.w00 * select(0.0, 1.0, c00.w > 0.45);
    let k10 = eq.w10 * select(0.0, 1.0, c10.w > 0.45);
    let k01 = eq.w01 * select(0.0, 1.0, c01.w > 0.45);
    let k11 = eq.w11 * select(0.0, 1.0, c11.w > 0.45);

    let maskSum = k00 + k10 + k01 + k11;
    if (maskSum <= 1e-4) {
        return vec4f(0.0);
    }
    let uvMag = (c00.xyz * k00 + c10.xyz * k10 + c01.xyz * k01 + c11.xyz * k11) / maskSum;
    return vec4f(uvMag, maskSum);
}

// Samples ocean currents `(uEastDf, vNorthDf, rawSpeed, oceanMask)` in m/s at unit normal `n`.
fn sampleOceanTexture(oceanTex: texture_2d<f32>, n: vec3f) -> vec4f {
    let dims = textureDimensions(oceanTex, 0);
    if (dims.x < 2u || dims.y < 2u) {
        return vec4f(0.0);
    }
    return sampleOceanTextureCoord(oceanTex, makeEquirectCoord(n, dims));
}

fn sampleTempTextureCoord(tempTex: texture_2d<f32>, eq: EquirectCoord) -> vec4f {
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

    return vec4f(airK, sstK, maskSum, sfcK);
}

// Samples live global temperatures: `(airTempK, sstK, sstMask, sfcAirTempK)` at sphere normal `n`.
fn sampleTempTexture(tempTex: texture_2d<f32>, n: vec3f) -> vec4f {
    let dims = textureDimensions(tempTex, 0);
    if (dims.x < 2u || dims.y < 2u) {
        return vec4f(288.15, 288.15, 0.0, 288.15);
    }
    return sampleTempTextureCoord(tempTex, makeEquirectCoord(n, dims));
}

fn effectiveSurfaceTempK(tSample: vec4f, oceanMask: f32) -> f32 {
    let seaBlend = smoothstep(0.25, 0.55, min(oceanMask, tSample.z));
    return mix(tSample.w, tSample.y, seaBlend);
}

fn sampleUvTextureCoord(uvTex: texture_2d<f32>, eq: EquirectCoord) -> vec4f {
    let c00 = textureLoad(uvTex, vec2u(eq.i0, eq.j0), 0);
    let c10 = textureLoad(uvTex, vec2u(eq.i1, eq.j0), 0);
    let c01 = textureLoad(uvTex, vec2u(eq.i0, eq.j1), 0);
    let c11 = textureLoad(uvTex, vec2u(eq.i1, eq.j1), 0);
    return c00 * eq.w00 + c10 * eq.w10 + c01 * eq.w01 + c11 * eq.w11;
}

// Samples live global UV index fields: `(uviNow, uviNoon, ozoneDU, cloudFrac)` at unit sphere normal `n`.
fn sampleUvTexture(uvTex: texture_2d<f32>, n: vec3f) -> vec4f {
    let dims = textureDimensions(uvTex, 0);
    if (dims.x < 2u || dims.y < 2u) {
        return vec4f(4.0, 7.0, 300.0, 0.25);
    }
    return sampleUvTextureCoord(uvTex, makeEquirectCoord(n, dims));
}

fn sampleHydroTextureCoord(hydroTex: texture_2d<f32>, eq: EquirectCoord) -> vec2f {
    let c00 = textureLoad(hydroTex, vec2u(eq.i0, eq.j0), 0).xy;
    let c10 = textureLoad(hydroTex, vec2u(eq.i1, eq.j0), 0).xy;
    let c01 = textureLoad(hydroTex, vec2u(eq.i0, eq.j1), 0).xy;
    let c11 = textureLoad(hydroTex, vec2u(eq.i1, eq.j1), 0).xy;
    let h = c00 * eq.w00 + c10 * eq.w10 + c01 * eq.w01 + c11 * eq.w11;
    return vec2f(clamp(h.x, 0.0, 1.0), max(h.y, 0.0));
}

// Samples live global surface hydrology `(rh01, precip3hMm)` at unit sphere normal `n`.
fn sampleHydroTexture(hydroTex: texture_2d<f32>, n: vec3f) -> vec2f {
    let dims = textureDimensions(hydroTex, 0);
    if (dims.x < 2u || dims.y < 2u) {
        return vec2f(0.65, 0.0);
    }
    return sampleHydroTextureCoord(hydroTex, makeEquirectCoord(n, dims));
}

// Samples terrain `(elevKm, dElevEast, dElevNorth, ridgeVentMask)` via shared equirectangular coords.
fn sampleTerrainTextureCoord(terrainTex: texture_2d<f32>, eq: EquirectCoord) -> vec4f {
    let c00 = textureLoad(terrainTex, vec2u(eq.i0, eq.j0), 0);
    let c10 = textureLoad(terrainTex, vec2u(eq.i1, eq.j0), 0);
    let c01 = textureLoad(terrainTex, vec2u(eq.i0, eq.j1), 0);
    let c11 = textureLoad(terrainTex, vec2u(eq.i1, eq.j1), 0);
    return c00 * eq.w00 + c10 * eq.w10 + c01 * eq.w01 + c11 * eq.w11;
}

fn sampleTerrainTexture(terrainTex: texture_2d<f32>, n: vec3f) -> vec4f {
    let dims = textureDimensions(terrainTex, 0);
    if (dims.x < 2u || dims.y < 2u) {
        return vec4f(0.0);
    }
    return sampleTerrainTextureCoord(terrainTex, makeEquirectCoord(n, dims));
}

// Magnus-Tetens saturation vapor pressure (kPa) from temperature in Kelvin.
fn saturationVaporKpa(tempK: f32) -> f32 {
    let tc = clamp(tempK - 273.15, -60.0, 60.0);
    return 0.6108 * exp((17.27 * tc) / (tc + 237.3));
}

// Returns `(actScaleMod, chemWellMod)` for atmospheric humidity catalysis & rain scavenging.
fn evalHydroChemFactors(rh01: f32, precipMm: f32, localHydroScale: f32) -> vec2f {
    let hBlend = clamp(localHydroScale, 0.0, 2.0);
    if (hBlend <= 0.001) {
        return vec2f(1.0, 1.0);
    }
    let humEase = clamp(1.0 - 0.16 * smoothstep(0.35, 0.95, rh01), 0.78, 1.08);
    let rainScavenge = 1.0 + 0.14 * smoothstep(0.4, 8.0, precipMm);
    return vec2f(
        mix(1.0, humEase, min(hBlend, 1.2)),
        mix(1.0, rainScavenge, min(hBlend, 1.2)),
    );
}

fn rhColor(rh01: f32) -> vec3f {
    let r = clamp(rh01, 0.0, 1.0);
    let c0 = vec3f(94.0, 62.0, 30.0) / 255.0;
    let c1 = vec3f(168.0, 134.0, 68.0) / 255.0;
    let c2 = vec3f(202.0, 198.0, 118.0) / 255.0;
    let c3 = vec3f(92.0, 186.0, 174.0) / 255.0;
    let c4 = vec3f(34.0, 146.0, 212.0) / 255.0;
    let c5 = vec3f(68.0, 74.0, 214.0) / 255.0;
    if (r <= 0.2) { return mix(c0, c1, r / 0.2); }
    if (r <= 0.4) { return mix(c1, c2, (r - 0.2) / 0.2); }
    if (r <= 0.6) { return mix(c2, c3, (r - 0.4) / 0.2); }
    if (r <= 0.8) { return mix(c3, c4, (r - 0.6) / 0.2); }
    return mix(c4, c5, (r - 0.8) / 0.2);
}

fn precip3hColor(mm: f32) -> vec3f {
    let p = clamp(mm, 0.0, 30.0);
    let c0 = vec3f(14.0, 20.0, 32.0) / 255.0;
    let c1 = vec3f(28.0, 118.0, 178.0) / 255.0;
    let c2 = vec3f(34.0, 196.0, 172.0) / 255.0;
    let c3 = vec3f(168.0, 228.0, 64.0) / 255.0;
    let c4 = vec3f(248.0, 164.0, 34.0) / 255.0;
    let c5 = vec3f(238.0, 46.0, 128.0) / 255.0;
    if (p <= 0.5) { return mix(c0, c1, p / 0.5); }
    if (p <= 2.5) { return mix(c1, c2, (p - 0.5) / 2.0); }
    if (p <= 7.0) { return mix(c2, c3, (p - 2.5) / 4.5); }
    if (p <= 15.0) { return mix(c3, c4, (p - 7.0) / 8.0); }
    return mix(c4, c5, (p - 15.0) / 15.0);
}

// Exact `earth.nullschool.net` CPC Ultraviolet Index color gradient `[0 .. 18]` (0 .. 0.45 W/m²).
fn uvIndexColor(uvi: f32) -> vec3f {
    let u = clamp(uvi, 0.0, 18.0);
    let c0  = vec3f(0.0, 0.0, 15.0) / 255.0;     // 0.0
    let c1  = vec3f(32.0, 32.0, 32.0) / 255.0;   // 0.2
    let c2  = vec3f(62.0, 167.0, 45.0) / 255.0;  // 0.5
    let c3  = vec3f(105.0, 194.0, 32.0) / 255.0; // 1.5
    let c4  = vec3f(174.0, 224.0, 17.0) / 255.0; // 2.5
    let c5  = vec3f(255.0, 238.0, 0.0) / 255.0;  // 3.5
    let c6  = vec3f(248.0, 186.0, 0.0) / 255.0;  // 4.9
    let c7  = vec3f(241.0, 139.0, 0.0) / 255.0;  // 6.25
    let c8  = vec3f(235.0, 91.0, 8.0) / 255.0;   // 7.4
    let c9  = vec3f(229.0, 50.0, 16.0) / 255.0;  // 8.5
    let c10 = vec3f(207.0, 58.0, 104.0) / 255.0; // 9.75
    let c11 = vec3f(181.0, 103.0, 164.0) / 255.0;// 11.0
    let c12 = vec3f(209.0, 148.0, 234.0) / 255.0;// 14.5
    let c13 = vec3f(231.0, 225.0, 255.0) / 255.0;// 18.0

    if (u <= 0.2) { return mix(c0, c1, u / 0.2); }
    if (u <= 0.5) { return mix(c1, c2, (u - 0.2) / 0.3); }
    if (u <= 1.5) { return mix(c2, c3, u - 0.5); }
    if (u <= 2.5) { return mix(c3, c4, u - 1.5); }
    if (u <= 3.5) { return mix(c4, c5, u - 2.5); }
    if (u <= 4.9) { return mix(c5, c6, (u - 3.5) / 1.4); }
    if (u <= 6.25) { return mix(c6, c7, (u - 4.9) / 1.35); }
    if (u <= 7.4) { return mix(c7, c8, (u - 6.25) / 1.15); }
    if (u <= 8.5) { return mix(c8, c9, (u - 7.4) / 1.1); }
    if (u <= 9.75) { return mix(c9, c10, (u - 8.5) / 1.25); }
    if (u <= 11.0) { return mix(c10, c11, (u - 9.75) / 1.25); }
    if (u <= 14.5) { return mix(c11, c12, (u - 11.0) / 3.5); }
    return mix(c12, c13, (u - 14.5) / 3.5);
}

// Exact `earth.nullschool.net` GFS Air Temperature color gradient `[193.15 .. 328.15] K`.
fn airTempColor(tempK: f32) -> vec3f {
    let t = clamp(tempK, 193.15, 328.15);
    let c0  = vec3f(37.0, 4.0, 42.0) / 255.0;     // 193.15 K (-80 C)
    let c1  = vec3f(41.0, 10.0, 130.0) / 255.0;   // 206.00 K
    let c2  = vec3f(81.0, 40.0, 40.0) / 255.0;    // 219.00 K
    let c3  = vec3f(192.0, 37.0, 149.0) / 255.0;  // 233.15 K (-40 C)
    let c4  = vec3f(70.0, 215.0, 215.0) / 255.0;  // 255.372 K (0 F)
    let c5  = vec3f(21.0, 84.0, 187.0) / 255.0;   // 273.15 K (0 C)
    let c6  = vec3f(24.0, 132.0, 14.0) / 255.0;   // 275.15 K (+2 C)
    let c7  = vec3f(247.0, 251.0, 59.0) / 255.0;  // 291.00 K
    let c8  = vec3f(235.0, 167.0, 21.0) / 255.0;  // 298.00 K
    let c9  = vec3f(230.0, 71.0, 39.0) / 255.0;   // 311.00 K
    let c10 = vec3f(88.0, 27.0, 67.0) / 255.0;    // 328.15 K (+55 C)

    if (t <= 206.0) { return mix(c0, c1, (t - 193.15) / (206.0 - 193.15)); }
    if (t <= 219.0) { return mix(c1, c2, (t - 206.0) / (219.0 - 206.0)); }
    if (t <= 233.15) { return mix(c2, c3, (t - 219.0) / (233.15 - 219.0)); }
    if (t <= 255.372) { return mix(c3, c4, (t - 233.15) / (255.372 - 233.15)); }
    if (t <= 273.15) { return mix(c4, c5, (t - 255.372) / (273.15 - 255.372)); }
    if (t <= 275.15) { return mix(c5, c6, (t - 273.15) / (275.15 - 273.15)); }
    if (t <= 291.0) { return mix(c6, c7, (t - 275.15) / (291.0 - 275.15)); }
    if (t <= 298.0) { return mix(c7, c8, (t - 291.0) / (298.0 - 291.0)); }
    if (t <= 311.0) { return mix(c8, c9, (t - 298.0) / (311.0 - 298.0)); }
    return mix(c9, c10, (t - 311.0) / (328.15 - 311.0));
}

// Exact `earth.nullschool.net` Sea Surface Temperature color gradient `[270.0 .. 313.15] K`.
fn sstTempColor(tempK: f32) -> vec3f {
    let t = clamp(tempK, 270.0, 313.15);
    let c0  = vec3f(255.0, 255.0, 255.0) / 255.0; // 270.0
    let c1  = vec3f(255.0, 255.0, 255.0) / 255.0; // 271.4
    let c2  = vec3f(15.0, 4.0, 168.0) / 255.0;    // 271.5
    let c3  = vec3f(15.0, 54.0, 208.0) / 255.0;   // 273.15
    let c4  = vec3f(15.0, 54.0, 188.0) / 255.0;   // 273.25
    let c5  = vec3f(15.0, 4.0, 168.0) / 255.0;    // 275.65
    let c6  = vec3f(24.0, 132.0, 14.0) / 255.0;   // 281.65
    let c7  = vec3f(247.0, 251.0, 59.0) / 255.0;  // 291.15
    let c8  = vec3f(235.0, 167.0, 0.0) / 255.0;   // 295.0
    let c9  = vec3f(245.0, 0.0, 39.0) / 255.0;    // 299.65
    let c10 = vec3f(87.0, 17.0, 0.0) / 255.0;     // 303.0
    let c11 = vec3f(210.0, 22.0, 210.0) / 255.0;  // 304.65
    let c12 = vec3f(255.0, 255.0, 255.0) / 255.0; // 313.15

    if (t <= 271.4) { return mix(c0, c1, (t - 270.0) / 1.4); }
    if (t <= 271.5) { return mix(c1, c2, (t - 271.4) / 0.1); }
    if (t <= 273.15) { return mix(c2, c3, (t - 271.5) / 1.65); }
    if (t <= 273.25) { return mix(c3, c4, (t - 273.15) / 0.1); }
    if (t <= 275.65) { return mix(c4, c5, (t - 273.25) / 2.4); }
    if (t <= 281.65) { return mix(c5, c6, (t - 275.65) / 6.0); }
    if (t <= 291.15) { return mix(c6, c7, (t - 281.65) / 9.5); }
    if (t <= 295.0) { return mix(c7, c8, (t - 291.15) / 3.85); }
    if (t <= 299.65) { return mix(c8, c9, (t - 295.0) / 4.65); }
    if (t <= 303.0) { return mix(c9, c10, (t - 299.65) / 3.35); }
    if (t <= 304.65) { return mix(c10, c11, (t - 303.0) / 1.65); }
    return mix(c11, c12, (t - 304.65) / 8.5);
}

// Exact `cambecc/earth` segmentedColorScale for ocean surface currents `[0 .. 1.5] m/s`.
fn oceanCurrentsColor(speedMs: f32) -> vec3f {
    let s = clamp(speedMs, 0.0, 1.5);
    let c0 = vec3f(10.0, 25.0, 68.0) / 255.0;
    let c1 = vec3f(10.0, 25.0, 250.0) / 255.0;
    let c2 = vec3f(24.0, 255.0, 93.0) / 255.0;
    let c3 = vec3f(255.0, 233.0, 102.0) / 255.0;
    let c4 = vec3f(255.0, 233.0, 15.0) / 255.0;
    let c5 = vec3f(255.0, 15.0, 15.0) / 255.0;

    if (s <= 0.15) {
        return mix(c0, c1, s / 0.15);
    } else if (s <= 0.40) {
        return mix(c1, c2, (s - 0.15) / 0.25);
    } else if (s <= 0.65) {
        return mix(c2, c3, (s - 0.40) / 0.25);
    } else if (s <= 1.00) {
        return mix(c3, c4, (s - 0.65) / 0.35);
    }
    return mix(c4, c5, (s - 1.00) / 0.50);
}

fn orthoBasis(n: vec3f, uRaw: vec3f) -> vec3f {
    let u = uRaw - dot(uRaw, n) * n;
    let l = length(u);
    if (l > 1e-6) {
        return u / l;
    }
    let refV = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
    return normalize(cross(refV, n));
}

fn sphereEastBasis(n: vec3f) -> vec3f {
    let horiz = length(n.xy);
    if (horiz > 1e-4) {
        return vec3f(-n.y / horiz, n.x / horiz, 0.0);
    }
    return normalize(cross(vec3f(1.0, 0.0, 0.0), n));
}

fn mouseSphereImpulse(pos: vec3f, rMax: f32, worldR: f32, mouseSphere: vec4f) -> vec3f {
    if (mouseSphere.w == 0.0) {
        return vec3f(0.0);
    }
    let mPos = normalize(mouseSphere.xyz);
    let mChord = length(mPos - pos);
    let mDist = worldR * 2.0 * asin(clamp(0.5 * mChord, 0.0, 1.0));
    if (mDist > 1e-5 && mDist < rMax * 2.5) {
        let mTan = mPos - dot(mPos, pos) * pos;
        let mLen = length(mTan);
        if (mLen > 1e-5) {
            let sgn = select(-1.0, 1.0, mouseSphere.w > 0.0);
            return (mTan / mLen) * (mDist * 3.0 * sgn);
        }
    }
    return vec3f(0.0);
}

struct GeodesicStep {
    pos: vec3f,
    vel: vec3f,
    orient: vec3f,
}

fn stepGeodesicOnSphere(
    posIn: vec3f,
    velIn: vec3f,
    omega: f32,
    uBody: vec3f,
    dt: f32,
    worldR: f32,
    cellSize: f32,
) -> GeodesicStep {
    var vel = velIn - dot(velIn, posIn) * posIn;
    let maxStepSpeed = min(cellSize / max(dt, 1e-4) * 0.45, worldR * 0.5 / max(dt, 1e-4));
    let speed = length(vel);
    if (speed > maxStepSpeed && speed > 1e-6) {
        vel *= maxStepSpeed / speed;
    }

    let clampedSpeed = length(vel);
    let phi = (clampedSpeed * dt) / worldR;
    var nNew = posIn;
    var vNew = vel;
    var uNew = uBody;
    if (phi > 1e-7) {
        let kHat = vel / clampedSpeed;
        let cPhi = cos(phi);
        let sPhi = sin(phi);
        nNew = normalize(cPhi * posIn + sPhi * kHat);
        vNew = clampedSpeed * (cPhi * kHat - sPhi * posIn);
        vNew -= dot(vNew, nNew) * nNew;
        let uk = dot(uBody, kHat);
        uNew = uBody - uk * ((1.0 - cPhi) * kHat + sPhi * posIn);
    }

    let dTheta = omega * dt;
    if (abs(dTheta) > 1e-7) {
        let cT = cos(dTheta);
        let sT = sin(dTheta);
        uNew = cT * uNew + sT * cross(nNew, uNew);
    }
    return GeodesicStep(nNew, vNew, orthoBasis(nNew, uNew));
}

fn sinebowColor(hue: f32) -> vec3f {
    let rad = hue * TWO_PI * (5.0 / 6.0) * 0.75;
    let s = sin(rad);
    let c = cos(rad);
    return vec3f(max(0.0, -c), max(s, 0.0), max(max(c, 0.0), -s));
}

fn extendedSinebowColor(t: f32) -> vec3f {
    let i = clamp(t, 0.0, 1.0);
    let boundary = 0.45;
    if (i <= boundary) {
        return sinebowColor(i / boundary);
    }
    let c1 = sinebowColor(1.0);
    let f = (i - boundary) / (1.0 - boundary);
    return mix(c1, vec3f(1.0, 1.0, 1.0), f);
}

fn forwardWaterman(watermanFwdTex: texture_2d<f32>, n: vec3f) -> vec4f {
    let dims = textureDimensions(watermanFwdTex, 0);
    if (dims.x < 10u || dims.y < 10u) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    var lonDeg = atan2(n.y, n.x) * (180.0 / PI);
    if (lonDeg < 0.0) {
        lonDeg += 360.0;
    }
    let latDeg = clamp(safeAsin(n.z) * (180.0 / PI), -90.0, 90.0);
    let gx = (lonDeg / 360.0) * f32(dims.x);
    let gy = clamp((90.0 - latDeg) / 180.0 * f32(dims.y - 1u), 0.0, f32(dims.y - 1u));

    let i0 = u32(floor(gx)) % dims.x;
    let i1 = (i0 + 1u) % dims.x;
    let j0 = min(dims.y - 1u, u32(floor(gy)));
    let j1 = min(dims.y - 1u, j0 + 1u);
    let fx = gx - floor(gx);
    let fy = gy - f32(j0);

    let t00 = textureLoad(watermanFwdTex, vec2u(i0, j0), 0);
    let t10 = textureLoad(watermanFwdTex, vec2u(i1, j0), 0);
    let t01 = textureLoad(watermanFwdTex, vec2u(i0, j1), 0);
    let t11 = textureLoad(watermanFwdTex, vec2u(i1, j1), 0);

    let allValid = t00.x < 1e4 && t10.x < 1e4 && t01.x < 1e4 && t11.x < 1e4;
    if (allValid) {
        let jMax = max(
            max(length(t10.xy - t00.xy), length(t01.xy - t00.xy)),
            max(length(t11.xy - t10.xy), length(t11.xy - t01.xy)),
        );
        if (jMax < 0.08) {
            let top = mix(t00, t10, fx);
            let bot = mix(t01, t11, fx);
            let res = mix(top, bot, fy);
            return vec4f(res.xy, res.z, res.w);
        }
    }

    let ni = u32(round(gx)) % dims.x;
    let nj = min(dims.y - 1u, u32(round(gy)));
    let near = textureLoad(watermanFwdTex, vec2u(ni, nj), 0);
    if (near.x < 1e4) {
        return vec4f(near.xy, near.z, near.w);
    }
    return vec4f(0.0, 0.0, 1e6, 0.0);
}

// Forward projects unit vector `n` to unit-scale D3 projection coords `(px, py)` and `zValid` (`1.0` if visible).
fn projectUnitPos(
    u: Uniforms,
    watermanFwdTex: texture_2d<f32>,
    n: vec3f,
) -> vec3f {
    let pType = u32(u.viewParams.w + 0.5);
    if (pType == 7u) {
        let w = forwardWaterman(watermanFwdTex, n);
        if (abs(w.z) > 100.0) {
            return vec3f(0.0, 0.0, 0.0);
        }
        return vec3f(w.x, w.y, 1.0);
    }

    let v = rotateToView(u, n);
    if (pType == 0u) {
        if (v.x < 0.005) {
            return vec3f(0.0, 0.0, 0.0);
        }
        return vec3f(v.y, -v.z, 1.0);
    }
    if (pType == 6u) {
        let denom = 1.0 + v.x;
        if (denom < 0.008) {
            return vec3f(0.0, 0.0, 0.0);
        }
        return vec3f(v.y / denom, -v.z / denom, 1.0);
    }

    let lam = atan2(v.y, v.x);
    let phi = safeAsin(v.z);

    if (pType == 3u) {
        return vec3f(lam, -phi, 1.0);
    }
    if (pType == 5u) {
        let c1 = 1.0148;
        let c2 = 0.23185;
        let c3 = -0.14499;
        let c4 = 0.02406;
        let p2 = phi * phi;
        let p4 = p2 * p2;
        return vec3f(lam, -phi * (c1 + p4 * (c2 + p2 * (c3 + c4 * p2))), 1.0);
    }
    if (pType == 4u) {
        let A1 = 1.340264;
        let A2 = -0.081106;
        let A3 = 0.000893;
        let A4 = 0.003796;
        let th = safeAsin(0.86602540378 * sin(phi));
        let t2 = th * th;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let px = (1.15470053838 * lam * cos(th)) / (A1 + 3.0 * A2 * t2 + 7.0 * A3 * t6 + 9.0 * A4 * t4 * t4);
        let py = -th * (A1 + A2 * t2 + A3 * t6 + A4 * t4 * t4);
        return vec3f(px, py, 1.0);
    }
    if (pType == 1u) {
        let sq2 = 1.41421356237;
        var th = phi;
        if (abs(phi) < HALF_PI - 1e-4) {
            let rhs = PI * sin(phi);
            for (var it = 0u; it < 5u; it++) {
                let dth = (th * 2.0 + sin(2.0 * th) - rhs) / max(2.0 + 2.0 * cos(2.0 * th), 1e-4);
                th -= dth;
            }
        } else {
            th = select(-HALF_PI, HALF_PI, phi >= 0.0);
        }
        return vec3f((2.0 * sq2 / PI) * lam * cos(th), -sq2 * sin(th), 1.0);
    }
    if (pType == 2u) {
        let y0 = 0.0;
        let y1 = PI / 3.0;
        let nn = (cos(y0) - cos(y1)) / (y1 - y0);
        let G = cos(y0) / nn + y0;
        let yCenter = G - 0.24328;
        let rho = G - phi;
        return vec3f(rho * sin(nn * lam), -(yCenter - rho * cos(nn * lam)), 1.0);
    }

    // pType == 8u: winkel3
    let cosPhi1 = 2.0 / PI;
    let cosPhi = cos(phi);
    let halfLam = 0.5 * lam;
    let alpha = safeAcos(cosPhi * cos(halfLam));
    let sincA = select(sin(alpha) / alpha, 1.0, alpha < 1e-4);
    let px = 0.5 * (lam * cosPhi1 + (2.0 * cosPhi * sin(halfLam)) / sincA);
    let py = 0.5 * (phi + sin(phi) / sincA);
    return vec3f(px, -py, 1.0);
}

fn evalWinkelF(lam: f32, phi: f32, goalPt: vec2f) -> vec2f {
    let cosPhi1 = 2.0 / PI;
    let cosPhi = cos(phi);
    let halfLam = 0.5 * lam;
    let alpha = safeAcos(cosPhi * cos(halfLam));
    let sincA = select(sin(alpha) / alpha, 1.0, alpha < 1e-4);
    let px = 0.5 * (lam * cosPhi1 + (2.0 * cosPhi * sin(halfLam)) / sincA);
    let py = 0.5 * (phi + sin(phi) / sincA);
    return vec2f(px, py) - goalPt;
}

// Inverse projects unit-scale D3 coords `(x, yScreen)` to `(earthX, earthY, earthZ, hit)`.
fn invertUnitPos(
    u: Uniforms,
    watermanInvTex: texture_2d<f32>,
    screenUv: vec2f,
    x: f32,
    yScreen: f32,
) -> vec4f {
    let pType = u32(u.viewParams.w + 0.5);
    if (pType == 7u) {
        let dims = textureDimensions(watermanInvTex, 0);
        if (dims.x < 10u || dims.y < 10u) {
            return vec4f(0.0);
        }
        let ix = min(dims.x - 1u, u32(clamp(screenUv.x, 0.0, 0.9999) * f32(dims.x)));
        let iy = min(dims.y - 1u, u32(clamp(screenUv.y, 0.0, 0.9999) * f32(dims.y)));
        return textureLoad(watermanInvTex, vec2u(ix, iy), 0);
    }

    let y = -yScreen;
    if (pType == 0u) {
        let r2 = x * x + y * y;
        if (r2 > 1.0) {
            return vec4f(0.0);
        }
        let v = vec3f(sqrt(max(0.0, 1.0 - r2)), x, y);
        return vec4f(normalize(rotateFromView(u, v)), 1.0);
    }
    if (pType == 6u) {
        let r2 = x * x + y * y;
        let invD = 1.0 / (1.0 + r2);
        let v = vec3f((1.0 - r2) * invD, (2.0 * x) * invD, (2.0 * y) * invD);
        return vec4f(normalize(rotateFromView(u, v)), 1.0);
    }

    var lam = 0.0;
    var phi = 0.0;

    if (pType == 3u) {
        if (abs(x) > PI || abs(y) > HALF_PI) {
            return vec4f(0.0);
        }
        lam = x;
        phi = y;
    } else if (pType == 5u) {
        let c1 = 1.0148;
        let c2 = 0.23185;
        let c3 = -0.14499;
        let c4 = 0.02406;
        let hp2 = HALF_PI * HALF_PI;
        let maxY = HALF_PI * (c1 + hp2 * hp2 * (c2 + hp2 * (c3 + c4 * hp2)));
        if (abs(x) > PI || abs(y) > maxY) {
            return vec4f(0.0);
        }
        lam = x;
        phi = y;
        for (var it = 0u; it < 6u; it++) {
            let p2 = phi * phi;
            let p4 = p2 * p2;
            let f = phi * (c1 + p4 * (c2 + p2 * (c3 + c4 * p2))) - y;
            let df = c1 + p4 * (5.0 * c2 + p2 * (7.0 * c3 + 9.0 * c4 * p2));
            phi -= f / max(df, 1e-4);
        }
    } else if (pType == 4u) {
        let A1 = 1.340264;
        let A2 = -0.081106;
        let A3 = 0.000893;
        let A4 = 0.003796;
        var th = y / A1;
        for (var it = 0u; it < 6u; it++) {
            let t2 = th * th;
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            let f = th * (A1 + A2 * t2 + A3 * t6 + A4 * t4 * t4) - y;
            let df = A1 + 3.0 * A2 * t2 + 7.0 * A3 * t6 + 9.0 * A4 * t4 * t4;
            th -= f / max(df, 1e-4);
        }
        let sPhi = sin(th) / 0.86602540378;
        if (abs(sPhi) > 1.0001) {
            return vec4f(0.0);
        }
        let t2 = th * th;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let df = A1 + 3.0 * A2 * t2 + 7.0 * A3 * t6 + 9.0 * A4 * t4 * t4;
        lam = (3.0 * x * df) / max(2.0 * 1.73205080757 * cos(th), 1e-5);
        if (abs(lam) > PI) {
            return vec4f(0.0);
        }
        phi = safeAsin(sPhi);
    } else if (pType == 1u) {
        let sq2 = 1.41421356237;
        if (abs(y) > sq2) {
            return vec4f(0.0);
        }
        let th = safeAsin(y / sq2);
        let cTh = cos(th);
        if (cTh < 1e-5) {
            return vec4f(0.0);
        }
        lam = (PI * x) / (2.0 * sq2 * cTh);
        if (abs(lam) > PI) {
            return vec4f(0.0);
        }
        phi = safeAsin((2.0 * th + sin(2.0 * th)) / PI);
    } else if (pType == 2u) {
        let y0 = 0.0;
        let y1 = PI / 3.0;
        let nn = (cos(y0) - cos(y1)) / (y1 - y0);
        let G = cos(y0) / nn + y0;
        let yCenter = G - 0.24328;
        let dy = yCenter - y;
        let rho = sqrt(x * x + dy * dy);
        phi = G - rho;
        lam = atan2(x, dy) / nn;
        if (abs(lam) > PI || abs(phi) > HALF_PI) {
            return vec4f(0.0);
        }
    } else {
        // winkel3
        if (abs(y) > HALF_PI || abs(x) > PI) {
            return vec4f(0.0);
        }
        let goalPt = vec2f(x, y);
        lam = x;
        phi = y;
        let eps = 0.002;
        for (var it = 0u; it < 7u; it++) {
            let e0 = evalWinkelF(lam, phi, goalPt);
            if (dot(e0, e0) < 1e-7) {
                break;
            }
            let eL = (evalWinkelF(lam + eps, phi, goalPt) - e0) / eps;
            let eP = (evalWinkelF(lam, phi + eps, goalPt) - e0) / eps;
            let det = eL.x * eP.y - eP.x * eL.y;
            if (abs(det) < 1e-7) {
                break;
            }
            lam -= (eP.y * e0.x - eP.x * e0.y) / det;
            phi -= (-eL.y * e0.x + eL.x * e0.y) / det;
        }
        let err = evalWinkelF(lam, phi, goalPt);
        if (dot(err, err) > 0.0005 || abs(lam) > PI * 1.001 || abs(phi) > HALF_PI * 1.001) {
            return vec4f(0.0);
        }
    }

    let cP = cos(phi);
    let v = vec3f(cP * cos(lam), cP * sin(lam), sin(phi));
    return vec4f(normalize(rotateFromView(u, v)), 1.0);
}

// Strictly monotonic C^1 flat-field optical magnifying lens:
// Constant `mag` inside `u <= 0.22`, Hermite transition on `[0.22, 1.0]` with `dr_out/du > +0.25` everywhere (no fold).
fn lensRadialMap(mag: f32, u: f32) -> vec2f {
    let m = clamp(mag, 1.0, 2.15);
    if (u <= 0.0) {
        return vec2f(0.0, m);
    }
    if (u >= 1.0) {
        return vec2f(1.0, 1.0);
    }
    let uc = 0.22;
    if (u <= uc) {
        return vec2f(m * u, m);
    }
    let du = 1.0 - uc;
    let rStart = m * uc;
    let dr = 1.0 - rStart;
    let t = (u - uc) / du;
    let t2 = t * t;
    let t3 = t2 * t;
    let m0 = (m * du) / dr;
    let m1 = du / dr;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let rOut = rStart + dr * (h10 * m0 + h01 + h11 * m1);
    let dh = (3.0 * t2 - 4.0 * t + 1.0) * m0 + (6.0 * t - 6.0 * t2) + (3.0 * t2 - 2.0 * t) * m1;
    return vec2f(rOut, (dr / du) * dh);
}

fn lensParticleScale(mag: f32, u: f32) -> f32 {
    if (u >= 1.0) {
        return 1.0;
    }
    let uc = max(u, 0.0);
    let rm = lensRadialMap(mag, uc);
    if (uc <= 0.22) {
        return clamp(mag, 1.0, 2.15);
    }
    let tanScale = rm.x / max(uc, 1e-4);
    return clamp(mix(tanScale, rm.y, 0.45), 0.72, clamp(mag, 1.0, 2.15));
}

fn applyLensForward(u: Uniforms, pScreen: vec2f) -> vec2f {
    let rLens = u.lensParams.z;
    let mag = max(u.lensParams.w, 1.0);
    if (rLens <= 1.0 || mag <= 1.01) {
        return pScreen;
    }
    let center = u.lensParams.xy;
    let d = pScreen - center;
    let dist = length(d);
    if (dist <= 1e-4 || dist >= rLens) {
        return pScreen;
    }
    let rm = lensRadialMap(mag, dist / rLens);
    return center + d * (rm.x * (rLens / dist));
}

fn applyLensInverse(u: Uniforms, pOut: vec2f) -> vec2f {
    let rLens = u.lensParams.z;
    let mag = max(u.lensParams.w, 1.0);
    if (rLens <= 1.0 || mag <= 1.01) {
        return pOut;
    }
    let center = u.lensParams.xy;
    let d = pOut - center;
    let dist = length(d);
    if (dist <= 1e-4 || dist >= rLens) {
        return pOut;
    }
    let uOut = dist / rLens;
    let m = clamp(mag, 1.0, 2.15);
    var uIn = clamp(uOut / m, 0.0, 1.0);
    for (var it = 0u; it < 5u; it++) {
        let rm = lensRadialMap(m, uIn);
        let fVal = rm.x - uOut;
        uIn = clamp(uIn - fVal / max(rm.y, 0.2), 0.0, 1.0);
    }
    return center + d * (uIn / uOut);
}
