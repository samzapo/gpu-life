// Shared sphere-surface (S^2) 2D Periodic Table orbital chemistry helpers.
// Particle positions `pos`, tangent velocities `vel`, and body slot-0 directions `orient`
// live on the unit sphere S^2 with physical surface radius R = sim.worldSize.

fn getSpeciesInfo(c: u32) -> SpeciesInfo {
    let nColours = max(u32(sim.colours), 1u);
    let sc = min(c, nColours - 1u);
    let base = sc * SPECIES_STRIDE;
    return SpeciesInfo(
        matrix[base + 0u],
        matrix[base + 1u],
        matrix[base + 2u],
        matrix[base + 3u],
        matrix[base + 4u],
        matrix[base + 5u],
        matrix[base + 6u],
        matrix[base + 7u],
    );
}

fn getPairAffinity(c1: u32, c2: u32) -> f32 {
    let nColours = max(u32(sim.colours), 1u);
    let sc1 = min(c1, nColours - 1u);
    let sc2 = min(c2, nColours - 1u);
    let header = nColours * SPECIES_STRIDE;
    return matrix[header + sc1 * nColours + sc2];
}

fn encodeSlotAffs(a0: f32, a1: f32, a2: f32) -> f32 {
    let q0 = u32(clamp(round(a0 * 100.0), 0.0, 250.0));
    let q1 = u32(clamp(round(a1 * 100.0), 0.0, 250.0));
    let q2 = u32(clamp(round(a2 * 100.0), 0.0, 250.0));
    return f32(q0 + q1 * 256u + q2 * 65536u);
}

fn decodeSlotAff(packed: f32, slotIdx: u32) -> f32 {
    let u = u32(max(packed, 0.0) + 0.5);
    let q = (u >> (slotIdx * 8u)) & 255u;
    return f32(q) * 0.01;
}

struct WindSample {
    pointWind: vec3f,
    avgWind: vec3f,
    shearDir: vec3f,
    shearRel: f32,
    refSpeed: f32,
}

fn sampleWind3D(windTex: texture_2d<f32>, dir: vec3f) -> vec3f {
    let n = normalize(dir);
    let uv = sampleWindTexture(windTex, n);
    let eEast = sphereEastBasis(n);
    let eNorth = normalize(cross(n, eEast));
    return uv.x * eEast + uv.y * eNorth;
}

fn evalWindAndShear(
    windTex: texture_2d<f32>,
    pos: vec3f,
    windStreamMax: f32,
) -> WindSample {
    if (sim.vortex <= 0.0 || sim.windStrength <= 0.0) {
        return WindSample(vec3f(0.0), vec3f(0.0), vec3f(0.0), 0.0, 10.0);
    }
    let R = max(sim.worldSize, 0.25);
    let eEast = sphereEastBasis(pos);
    let eNorth = normalize(cross(pos, eEast));
    let eps = clamp((sim.rMax / R) * 0.55, 0.008, 0.045);

    let w0 = sampleWind3D(windTex, pos);
    var wE = sampleWind3D(windTex, pos + eps * eEast);
    var wW = sampleWind3D(windTex, pos - eps * eEast);
    var wN = sampleWind3D(windTex, pos + eps * eNorth);
    var wS = sampleWind3D(windTex, pos - eps * eNorth);

    wE -= dot(wE, pos) * pos;
    wW -= dot(wW, pos) * pos;
    wN -= dot(wN, pos) * pos;
    wS -= dot(wS, pos) * pos;

    let avgWind = (w0 + 0.5 * (wE + wW + wN + wS)) / 3.0;
    let dE = wE - wW;
    let dN = wN - wS;
    let shearMs = 0.5 * sqrt(dot(dE, dE) + dot(dN, dN));
    let refSpeed = max(windStreamMax, 10.0);
    // Physical shear rate scaled against a fixed 22 m/s storm shear scale so higher-altitude jet shear genuinely exceeds gentle surface breeze shear
    let shearRel = (shearMs / 22.0) * sim.windStrength;
    var shearVec = (dE * 0.5 + dN * 0.5);
    shearVec -= dot(shearVec, pos) * pos;
    return WindSample(w0, avgWind, shearVec, shearRel, refSpeed);
}

// Evaluates directional orbital lobe L(delta) = max(0, cos(delta))^p and dL/d(delta).
fn evalLobe(delta: f32, p: f32) -> vec2f {
    let c = cos(delta);
    if (c <= 0.0) {
        return vec2f(0.0, 0.0);
    }
    let l = pow(c, p);
    let dl = -p * pow(c, max(p - 1.0, 0.0)) * sin(delta);
    return vec2f(l, dl);
}

// Finds best-aligned active valence slot on species `s` for body-frame bearing `alpha`.
fn bestSlotLobe(s: SpeciesInfo, alpha: f32, p: f32) -> vec3f {
    let v = u32(s.valence);
    if (v == 0u) {
        return vec3f(0.0, 0.0, 0.0);
    }
    var bestIdx = 0u;
    var bestRes = evalLobe(alpha - s.slotAngle0, p);
    if (v >= 2u) {
        let r1 = evalLobe(alpha - s.slotAngle1, p);
        if (r1.x > bestRes.x) {
            bestRes = r1;
            bestIdx = 1u;
        }
    }
    if (v >= 3u) {
        let r2 = evalLobe(alpha - s.slotAngle2, p);
        if (r2.x > bestRes.x) {
            bestRes = r2;
            bestIdx = 2u;
        }
    }
    return vec3f(f32(bestIdx), bestRes.x, bestRes.y);
}

fn radPresence(rNorm: f32) -> f32 {
    let wellFull = min(sim.beta + 0.16, 0.62);
    let wellEnd = 0.82;
    if (rNorm >= wellEnd) {
        return 0.0;
    }
    if (rNorm <= wellFull) {
        return 1.0;
    }
    let t = clamp((rNorm - wellFull) / max(wellEnd - wellFull, 1e-4), 0.0, 1.0);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

fn accumulateValenceNeighbor(
    p: Particle,
    sP: SpeciesInfo,
    ipPos: vec3f,
    ipColour: f32,
    ipOrient: vec3f,
    slotAcc: ptr<function, vec4f>,
) {
    let R = max(sim.worldSize, 0.25);
    let chordVec = ipPos - p.pos;
    let chord2 = dot(chordVec, chordVec);
    let maxAng = sim.rMax / R;
    if (chord2 <= 1e-12 || chord2 >= maxAng * maxAng) {
        return;
    }

    let cIp = u32(ipColour);
    let aff = getPairAffinity(u32(p.colour), cIp);
    if (aff <= 0.0) {
        return;
    }

    let sIp = getSpeciesInfo(cIp);
    if (u32(sIp.valence) == 0u) {
        return;
    }

    let theta = 2.0 * asin(clamp(0.5 * sqrt(chord2), 0.0, 1.0));
    let rNorm = (R * theta) / sim.rMax;
    let pres = radPresence(rNorm);
    if (pres <= 0.0) {
        return;
    }

    let tIraw = ipPos - dot(ipPos, p.pos) * p.pos;
    let tJraw = p.pos - dot(p.pos, ipPos) * ipPos;
    let lenI = length(tIraw);
    let lenJ = length(tJraw);
    if (lenI <= 1e-6 || lenJ <= 1e-6) {
        return;
    }
    let dI = tIraw / lenI;
    let dJ = tJraw / lenJ;

    let uI = orthoBasis(p.pos, p.orient);
    let bI = cross(p.pos, uI);
    let alphaI = atan2(dot(dI, bI), dot(dI, uI));

    let uJ = orthoBasis(ipPos, ipOrient);
    let bJ = cross(ipPos, uJ);
    let alphaJ = atan2(dot(dJ, bJ), dot(dJ, uJ));

    let sharp = max(sim.orbitalSharpness, 1.0);
    let lobeI = bestSlotLobe(sP, alphaI, sharp);
    let lobeJ = bestSlotLobe(sIp, alphaJ, sharp);
    let w = pres * lobeI.y * lobeJ.y;
    if (w <= 0.0) {
        return;
    }

    let k = u32(lobeI.x);
    if (k == 0u) {
        (*slotAcc).x += w;
    } else if (k == 1u) {
        (*slotAcc).y += w;
    } else {
        (*slotAcc).z += w;
    }
    if (w >= 0.16) {
        var a0 = decodeSlotAff((*slotAcc).w, 0u);
        var a1 = decodeSlotAff((*slotAcc).w, 1u);
        var a2 = decodeSlotAff((*slotAcc).w, 2u);
        if (k == 0u) {
            a0 = max(a0, aff);
        } else if (k == 1u) {
            a1 = max(a1, aff);
        } else {
            a2 = max(a2, aff);
        }
        (*slotAcc).w = encodeSlotAffs(a0, a1, a2);
    }
}

fn getSlotValue(slots: vec4f, k: u32) -> f32 {
    if (k == 0u) { return slots.x; }
    if (k == 1u) { return slots.y; }
    return slots.z;
}

fn accumulateSimNeighbor(
    pi: u32,
    p: Particle,
    sP: SpeciesInfo,
    mySlots: vec4f,
    ipPos: vec3f,
    ipColour: f32,
    ipOrient: vec3f,
    nbSlots: vec4f,
    actScale: f32,
    bioCleave: f32,
    shearStress: f32,
    tempWellMod: f32,
    acc: ptr<function, ForceAccum>,
) {
    let R = max(sim.worldSize, 0.25);
    let chordVec = ipPos - p.pos;
    let chord2 = dot(chordVec, chordVec);
    let maxAng = sim.rMax / R;

    if (chord2 <= 1e-12) {
        let uFallback = orthoBasis(p.pos, p.orient);
        let rnd = (f32(pi) * 20293.302 % 4932.329) / 100000.0;
        (*acc).avoidForce += uFallback * rnd;
        return;
    }
    if (chord2 >= maxAng * maxAng) {
        return;
    }

    let theta = 2.0 * asin(clamp(0.5 * sqrt(chord2), 0.0, 1.0));
    let r = R * theta;
    let rNorm = r / sim.rMax;
    if (rNorm >= 1.0) {
        return;
    }

    let tIraw = ipPos - dot(ipPos, p.pos) * p.pos;
    let tJraw = p.pos - dot(p.pos, ipPos) * ipPos;
    let lenI = length(tIraw);
    let lenJ = length(tJraw);
    if (lenI <= 1e-6 || lenJ <= 1e-6) {
        return;
    }
    let dI = tIraw / lenI;
    let dJ = tJraw / lenJ;

    // 1. Hard Pauli core repulsion below equilibrium bond radius beta
    let beta = clamp(sim.beta, 0.05, 0.7);
    if (rNorm < beta) {
        let fCore = (rNorm / beta - 1.0) * 1.6;
        (*acc).avoidForce += dI * fCore;
        (*acc).div += max(0.0, -fCore) / 10.0 * sim.avoidance;
    }

    let cP = u32(p.colour);
    let cIp = u32(ipColour);
    let aff = getPairAffinity(cP, cIp);
    let sIp = getSpeciesInfo(cIp);

    if (aff < 0.0) {
        let fRep = aff * (1.0 - rNorm) * 0.65;
        (*acc).repelForce += dI * fRep;
        return;
    }
    if (aff == 0.0 || u32(sP.valence) == 0u || u32(sIp.valence) == 0u) {
        return;
    }

    // 2. Outer valence slot alignment & completion availability
    let wellEnd = 0.78;
    if (rNorm >= wellEnd) {
        if (sim.activation > 0.0) {
            let u = (rNorm - wellEnd) / max(1.0 - wellEnd, 1e-4);
            let evansPolanyi = max(0.15, 1.0 - max(0.0, aff - 1.04) * 2.4 * max(sim.exchangeDrive, 0.0));
            let fAct = -(sim.activation * clamp(actScale, 0.05, 2.5) * evansPolanyi) * sin(PI * u);
            (*acc).repelForce += dI * fAct;
        }
        return;
    }

    let uI = orthoBasis(p.pos, p.orient);
    let bI = cross(p.pos, uI);
    let alphaI = atan2(dot(dI, bI), dot(dI, uI));

    let uJ = orthoBasis(ipPos, ipOrient);
    let bJ = cross(ipPos, uJ);
    let alphaJ = atan2(dot(dJ, bJ), dot(dJ, uJ));

    let sharp = max(sim.orbitalSharpness, 1.0);
    let lobeI = bestSlotLobe(sP, alphaI, sharp);
    let lobeJ = bestSlotLobe(sIp, alphaJ, sharp);
    let lI = lobeI.y;
    let dlI = lobeI.z;
    let lJ = lobeJ.y;
    let dlJ = lobeJ.z;

    let pres = radPresence(rNorm);
    let w = pres * lI * lJ;

    let k = u32(lobeI.x);
    let m = u32(lobeJ.x);
    let otherI = max(0.0, getSlotValue(mySlots, k) - w);
    let otherJ = max(0.0, getSlotValue(nbSlots, m) - w);

    // Higher-affinity candidates can displace weaker occupants in filled slots
    let exch = max(sim.exchangeDrive, 0.0);
    let incI = decodeSlotAff(mySlots.w, k);
    let incJ = decodeSlotAff(nbSlots.w, m);
    let advI = max(0.0, aff - incI - 0.04);
    let advJ = max(0.0, aff - incJ - 0.04);
    let relaxI = clamp(advI * 3.2 * exch, 0.0, 0.92);
    let relaxJ = clamp(advJ * 3.2 * exch, 0.0, 0.92);
    let effOtherI = otherI * (1.0 - relaxI);
    let effOtherJ = otherJ * (1.0 - relaxJ);
    var avail = max(0.0, 1.0 - sim.valency * effOtherI) * max(0.0, 1.0 - sim.valency * effOtherJ);
    let disfavored = max(0.0, max(incI, incJ) - aff - 0.06);
    if (disfavored > 0.0 && exch > 0.0) {
        avail *= max(0.0, 1.0 - disfavored * 3.0 * exch);
    }

    if (avail <= 0.05) {
        if (rNorm > beta) {
            let fSteric = -0.35 * (1.0 - rNorm / wellEnd);
            (*acc).repelForce += dI * fSteric;
        }
        return;
    }

    let stab = clamp(sim.bondStability, 0.0, 1.0);
    let wellMod = clamp((1.0 - 0.70 * bioCleave) * (1.0 - 0.75 * shearStress) * clamp(tempWellMod, 0.45, 1.18), 0.15, 1.18);

    if (rNorm >= beta) {
        let rPeak = mix(0.5 * (beta + wellEnd), beta + 0.34 * (wellEnd - beta), stab);
        var fWell = 0.0;
        if (rNorm <= rPeak) {
            fWell = (rNorm - beta) / max(rPeak - beta, 1e-4);
        } else {
            fWell = 1.0 - (rNorm - rPeak) / max(wellEnd - rPeak, 1e-4);
        }
        fWell = max(0.0, fWell);
        let cohBoost = 1.0 + 0.35 * stab;
        let fPull = aff * avail * fWell * lI * lJ * cohBoost * wellMod;
        (*acc).bondForce += dI * fPull;
    }

    let bondStrength = aff * avail * pres * mix(1.0, wellMod, 0.65);
    if (bondStrength > 0.0) {
        let tHat = cross(p.pos, dI);
        let invRClamped = 1.0 / max(r, 0.35 * sim.rMax);
        let fTan = -bondStrength * (dlI * lJ + lI * dlJ) * sim.rMax * invRClamped * 0.45;
        (*acc).bondForce += tHat * fTan;
        (*acc).torque += -bondStrength * dlI * lJ;
    }
}

// Arrhenius barrier scaling (x) and bond-well thermal equilibrium modulation (y) from live GFS Air Temp (K).
fn evalAirTempChemFactors(tAirK: f32, localTempScale: f32) -> vec2f {
    let sc = clamp(localTempScale, 0.0, 2.0);
    if (sc <= 0.001) {
        return vec2f(1.0, 1.0);
    }
    let dT = clamp(tAirK - 288.15, -65.0, 45.0);
    // Warm air lowers activation hump; frigid polar/upper-air raises barrier (Arrhenius kinetics)
    let arrhAct = clamp(exp(-0.020 * dT), 0.36, 2.25);
    let coolStab = clamp(1.0 + (286.15 - tAirK) * 0.0032, 0.88, 1.14);
    let hotDissoc = smoothstep(298.15, 326.15, tAirK) * 0.42;
    let wellTherm = clamp(coolStab - hotDissoc, 0.48, 1.14);
    let blend = min(sc, 1.5);
    return vec2f(mix(1.0, arrhAct, blend), mix(1.0, wellTherm, blend));
}

// Solar UV photolysis activation barrier lowering (x) and radical bond cleavage (y) from live CPC+GFS UVI.
fn evalUvChemFactors(uviNow: f32, localUvScale: f32) -> vec2f {
    let sc = clamp(localUvScale, 0.0, 2.0);
    if (sc <= 0.001) {
        return vec2f(1.0, 1.0);
    }
    let u = clamp(uviNow, 0.0, 22.0);
    let actDrop = 1.0 - 0.18 * smoothstep(2.0, 14.0, u);
    let wellLoosen = 1.0 - 0.10 * smoothstep(5.5, 16.5, u);
    let blend = min(sc, 1.5);
    return vec2f(mix(1.0, actDrop, blend), mix(1.0, wellLoosen, blend));
}

fn finalizeForces(acc: ForceAccum) -> ForceResult {
    let denom = 1.0 + acc.div;
    let netForce = (acc.bondForce + acc.repelForce) * (sim.rMax * sim.force / denom) + acc.avoidForce;
    let netTorque = sim.torque * acc.torque / denom;
    return ForceResult(netForce, netTorque);
}

fn integrateParticle(
    pi: u32,
    pIn: Particle,
    mySlots: vec4f,
    res: ForceResult,
    windSample: WindSample,
    shearStress: f32,
    mouseSphere: vec4f,
    bioReact: vec4f,
) -> Particle {
    var p = pIn;
    let R = max(sim.worldSize, 0.25);

    p.vel -= dot(p.vel, p.pos) * p.pos;
    p.vel = p.vel * sim.friction + res.force * sim.dt;
    p.vel += bioReact.xyz * sim.dt;

    p.omega = p.omega * sim.angularFriction + res.torque * sim.dt;

    let uBody = orthoBasis(p.pos, p.orient);
    let bBody = cross(p.pos, uBody);

    let sP = getSpeciesInfo(u32(p.colour));
    let bondLoad = mySlots.x + mySlots.y + mySlots.z;
    let bondFrac = select(0.0, clamp(bondLoad / max(sP.valence, 1.0), 0.0, 1.0), u32(sP.valence) > 0u);
    let stab = clamp(sim.bondStability, 0.0, 1.0);

    let actGlow = unpackActGlow(bioReact.w);
    let bioCleave = clamp(-actGlow.y, 0.0, 1.0);
    let catAgitation = clamp(bioCleave * 0.85 + length(bioReact.xyz) * 0.12 + shearStress, 0.0, 1.0);
    let tempScale = mix(0.28, 1.0, actGlow.x);
    let calmFactor = mix(1.0, max(0.04, 1.0 - 0.92 * stab), bondFrac * (1.0 - catAgitation));

    if (sim.temperature > 0.0) {
        let frame = u32(sim.frame);
        let h1 = hashFloat(pi, frame, 0u) * 2.0 - 1.0;
        let h2 = hashFloat(pi, frame, 1u) * 2.0 - 1.0;
        let h3 = hashFloat(pi, frame, 2u) * 2.0 - 1.0;
        let sqrtDt = sqrt(sim.dt);
        p.vel += (uBody * h1 + bBody * h2) * (sim.temperature * tempScale * calmFactor * sim.rMax * sqrtDt * 2.2);
        p.omega += sim.temperature * tempScale * calmFactor * h3 * sqrtDt * 12.0;
    }

    // Live global atmospheric wind force field with coherent low-wind drift & high-shear tearing
    if (sim.vortex > 0.0 && sim.windStrength > 0.0) {
        let calmWind = bondFrac * stab * (1.0 - shearStress);
        let windEff = mix(windSample.pointWind, windSample.avgWind, calmWind);
        var wTarget = (windEff / windSample.refSpeed) * (0.45 * sim.windStrength);
        if (shearStress > 0.0) {
            let frame = u32(sim.frame);
            let sh1 = hashFloat(pi, frame, 3u) * 2.0 - 1.0;
            let sh2 = hashFloat(pi, frame, 4u) * 2.0 - 1.0;
            let eddy = (windSample.shearDir / windSample.refSpeed) * (sh1 * 0.65)
                     + (uBody * sh1 + bBody * sh2) * (0.18 * windSample.shearRel);
            wTarget += eddy * (shearStress * sim.windStrength);
        }
        let wLen = length(wTarget);
        if (wLen > 1.35) {
            wTarget *= 1.35 / wLen;
        }
        p.vel += (wTarget - p.vel) * clamp(14.0 * sim.dt, 0.0, 0.45);
    }

    p.vel += mouseSphereImpulse(p.pos, sim.rMax, R, mouseSphere);

    let stepRes = stepGeodesicOnSphere(
        p.pos,
        p.vel,
        p.omega,
        uBody,
        sim.dt,
        R,
        sim.cellSize,
    );
    p.pos = stepRes.pos;
    p.vel = stepRes.vel;
    p.orient = stepRes.orient;
    p._pad = actGlow.y;

    return p;
}
