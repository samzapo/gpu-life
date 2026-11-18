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

fn orthoBasis(n: vec3f, uRaw: vec3f) -> vec3f {
    var u = uRaw - dot(uRaw, n) * n;
    let l = length(u);
    if (l > 1e-6) {
        return u / l;
    }
    let refV = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
    return normalize(cross(refV, n));
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
    (*slotAcc).w += w;
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
            let fAct = -sim.activation * sin(PI * u);
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
    let avail = max(0.0, 1.0 - sim.valency * otherI) * max(0.0, 1.0 - sim.valency * otherJ);

    if (avail <= 0.05) {
        if (rNorm > beta) {
            let fSteric = -0.35 * (1.0 - rNorm / wellEnd);
            (*acc).repelForce += dI * fSteric;
        }
        return;
    }

    if (rNorm >= beta) {
        let mid = 0.5 * (beta + wellEnd);
        let halfW = max(0.5 * (wellEnd - beta), 1e-4);
        let fWell = max(0.0, 1.0 - abs(rNorm - mid) / halfW);
        let fPull = aff * avail * fWell * lI * lJ;
        (*acc).bondForce += dI * fPull;
    }

    let bondStrength = aff * avail * pres;
    if (bondStrength > 0.0) {
        let tHat = cross(p.pos, dI);
        let invRClamped = 1.0 / max(r, 0.35 * sim.rMax);
        let fTan = -bondStrength * (dlI * lJ + lI * dlJ) * sim.rMax * invRClamped * 0.45;
        (*acc).bondForce += tHat * fTan;
        (*acc).torque += -bondStrength * dlI * lJ;
    }
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
    res: ForceResult,
    windUv: vec3f,
    windStreamMax: f32,
    mouseSphere: vec4f,
) -> Particle {
    var p = pIn;
    let R = max(sim.worldSize, 0.25);

    p.vel -= dot(p.vel, p.pos) * p.pos;
    p.vel = p.vel * sim.friction + res.force * sim.dt;

    p.omega = p.omega * sim.angularFriction + res.torque * sim.dt;

    let uBody = orthoBasis(p.pos, p.orient);
    let bBody = cross(p.pos, uBody);

    if (sim.temperature > 0.0) {
        let frame = u32(sim.frame);
        let h1 = hashFloat(pi, frame, 0u) * 2.0 - 1.0;
        let h2 = hashFloat(pi, frame, 1u) * 2.0 - 1.0;
        let h3 = hashFloat(pi, frame, 2u) * 2.0 - 1.0;
        let sqrtDt = sqrt(sim.dt);
        p.vel += (uBody * h1 + bBody * h2) * (sim.temperature * sim.rMax * sqrtDt * 2.2);
        p.omega += sim.temperature * h3 * sqrtDt * 12.0;
    }

    // Live global atmospheric wind force field ("vortex" mixing across the sphere)
    if (sim.vortex > 0.0 && sim.windStrength > 0.0) {
        let horiz = length(p.pos.xy);
        var eEast = vec3f(0.0, 1.0, 0.0);
        if (horiz > 1e-4) {
            eEast = vec3f(-p.pos.y / horiz, p.pos.x / horiz, 0.0);
        } else {
            eEast = normalize(cross(vec3f(1.0, 0.0, 0.0), p.pos));
        }
        let eNorth = normalize(cross(p.pos, eEast));
        let wind3D = windUv.x * eEast + windUv.y * eNorth;
        let refSpeed = max(windStreamMax, 10.0);
        var wTarget = (wind3D / refSpeed) * (0.45 * sim.windStrength);
        let wLen = length(wTarget);
        if (wLen > 1.35) {
            wTarget *= 1.35 / wLen;
        }
        p.vel += (wTarget - p.vel) * clamp(14.0 * sim.dt, 0.0, 0.45);
    }

    // Mouse attraction / repulsion projected onto local sphere tangent plane
    if (mouseSphere.w != 0.0) {
        let mPos = normalize(mouseSphere.xyz);
        let mChord = length(mPos - p.pos);
        let mDist = R * 2.0 * asin(clamp(0.5 * mChord, 0.0, 1.0));
        if (mDist > 1e-5 && mDist < sim.rMax * 2.5) {
            let mTan = mPos - dot(mPos, p.pos) * p.pos;
            let mLen = length(mTan);
            if (mLen > 1e-5) {
                let sign = select(-1.0, 1.0, mouseSphere.w > 0.0);
                p.vel += (mTan / mLen) * (mDist * 3.0 * sign);
            }
        }
    }

    p.vel -= dot(p.vel, p.pos) * p.pos;
    let maxStepSpeed = min(sim.cellSize / max(sim.dt, 1e-4) * 0.45, R * 0.5 / max(sim.dt, 1e-4));
    let speed = length(p.vel);
    if (speed > maxStepSpeed && speed > 1e-6) {
        p.vel *= maxStepSpeed / speed;
    }

    // Exact great-circle geodesic advance & parallel transport (zero pole gimbal lock)
    let clampedSpeed = length(p.vel);
    let phi = (clampedSpeed * sim.dt) / R;
    var nNew = p.pos;
    var vNew = p.vel;
    var uNew = uBody;
    if (phi > 1e-7) {
        let kHat = p.vel / clampedSpeed;
        let cPhi = cos(phi);
        let sPhi = sin(phi);
        nNew = normalize(cPhi * p.pos + sPhi * kHat);
        vNew = clampedSpeed * (cPhi * kHat - sPhi * p.pos);
        vNew -= dot(vNew, nNew) * nNew;
        let uk = dot(uBody, kHat);
        uNew = uBody - uk * ((1.0 - cPhi) * kHat + sPhi * p.pos);
    }

    let dTheta = p.omega * sim.dt;
    let cT = cos(dTheta);
    let sT = sin(dTheta);
    uNew = cT * uNew + sT * cross(nNew, uNew);
    p.pos = nNew;
    p.vel = vNew;
    p.orient = orthoBasis(nNew, uNew);

    return p;
}
