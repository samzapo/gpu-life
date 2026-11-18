// Shared chemistry helpers reading packed `matrix` buffer:
// [0 .. colours*8 - 1] = SpeciesInfo per species
// [colours*8 ..]       = pairwise affinity E(c1, c2)

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
// Returns vec3f(slotIdxF, L, dL_dDelta).
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
    ipPos: vec2f,
    ipColour: f32,
    ipTheta: f32,
    slotAcc: ptr<function, vec4f>,
) {
    let rx = ipPos.x - p.pos.x;
    let ry = ipPos.y - p.pos.y;
    let rs = rx * rx + ry * ry;
    let mrs = sim.rMax * sim.rMax;

    if (rs <= 0.0 || rs >= mrs) {
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

    let r = sqrt(rs);
    let rNorm = r / sim.rMax;
    let pres = radPresence(rNorm);
    if (pres <= 0.0) {
        return;
    }

    let phi = atan2(ry, rx);
    let alphaI = phi - p.theta;
    let alphaJ = phi + PI - ipTheta;
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
    ipPos: vec2f,
    ipColour: f32,
    ipTheta: f32,
    nbSlots: vec4f,
    acc: ptr<function, ForceAccum>,
) {
    let rx = ipPos.x - p.pos.x;
    let ry = ipPos.y - p.pos.y;
    let rs = rx * rx + ry * ry;
    let mrs = sim.rMax * sim.rMax;

    if (rs == 0.0) {
        (*acc).avoidForce.x += (f32(pi) * 20293.302 % 4932.329) / 100000.0;
        return;
    }
    if (rs >= mrs) {
        return;
    }

    let r = sqrt(rs);
    let rNorm = r / sim.rMax;
    let invR = 1.0 / r;
    let dx = rx * invR;
    let dy = ry * invR;

    // 1. Hard Pauli core repulsion below equilibrium bond radius beta
    let beta = clamp(sim.beta, 0.05, 0.7);
    if (rNorm < beta) {
        let fCore = (rNorm / beta - 1.0) * 1.6;
        (*acc).avoidForce.x += dx * fCore;
        (*acc).avoidForce.y += dy * fCore;
        (*acc).div += max(0.0, -fCore) / 10.0 * sim.avoidance;
    }

    let cP = u32(p.colour);
    let cIp = u32(ipColour);
    let aff = getPairAffinity(cP, cIp);
    let sIp = getSpeciesInfo(cIp);

    // Noble gases or same-polarity incompatible pairs
    if (aff < 0.0) {
        let fRep = aff * (1.0 - rNorm) * 0.65;
        (*acc).repelForce.x += dx * fRep;
        (*acc).repelForce.y += dy * fRep;
        return;
    }
    if (aff == 0.0 || u32(sP.valence) == 0u || u32(sIp.valence) == 0u) {
        return;
    }

    // 2. Outer valence slot alignment & completion availability
    let wellEnd = 0.78;
    if (rNorm >= wellEnd) {
        // Repulsive activation-energy shoulder outside the covalent bond well
        if (sim.activation > 0.0) {
            let u = (rNorm - wellEnd) / max(1.0 - wellEnd, 1e-4);
            let fAct = -sim.activation * sin(PI * u);
            (*acc).repelForce.x += dx * fAct;
            (*acc).repelForce.y += dy * fAct;
        }
        return;
    }

    let phi = atan2(ry, rx);
    let alphaI = phi - p.theta;
    let alphaJ = phi + PI - ipTheta;
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

    // Occupied-slot steric shell repulsion: prevents completed molecules from merging into blobs
    if (avail <= 0.05) {
        if (rNorm > beta) {
            let fSteric = -0.35 * (1.0 - rNorm / wellEnd);
            (*acc).repelForce.x += dx * fSteric;
            (*acc).repelForce.y += dy * fSteric;
        }
        return;
    }

    // Attractive radial well in [beta, wellEnd]
    if (rNorm >= beta) {
        let mid = 0.5 * (beta + wellEnd);
        let halfW = max(0.5 * (wellEnd - beta), 1e-4);
        let fWell = max(0.0, 1.0 - abs(rNorm - mid) / halfW);
        let fPull = aff * avail * fWell * lI * lJ;
        (*acc).bondForce.x += dx * fPull;
        (*acc).bondForce.y += dy * fPull;
    }

    // Tangential bond-angle restoring force + orbital alignment torque
    let bondStrength = aff * avail * pres;
    if (bondStrength > 0.0) {
        let tx = -dy;
        let ty = dx;
        let invRClamped = 1.0 / max(r, 0.35 * sim.rMax);
        let fTan = -bondStrength * (dlI * lJ + lI * dlJ) * sim.rMax * invRClamped * 0.45;
        (*acc).bondForce.x += tx * fTan;
        (*acc).bondForce.y += ty * fTan;

        (*acc).torque += -bondStrength * dlI * lJ;
    }
}

fn finalizeForces(acc: ForceAccum) -> ForceResult {
    let denom = 1.0 + acc.div;
    let netForce = (acc.bondForce + acc.repelForce) * (sim.rMax * sim.force / denom) + acc.avoidForce;
    let netTorque = sim.torque * acc.torque / denom;
    return ForceResult(netForce, netTorque);
}

fn integrateParticle(pi: u32, pIn: Particle, res: ForceResult) -> Particle {
    var p = pIn;

    p.vel *= sim.friction;
    p.vel += res.force * sim.dt;

    p.omega *= sim.angularFriction;
    p.omega += res.torque * sim.dt;

    if (sim.temperature > 0.0) {
        let frame = u32(sim.frame);
        let h1 = hashFloat(pi, frame, 0u) * 2.0 - 1.0;
        let h2 = hashFloat(pi, frame, 1u) * 2.0 - 1.0;
        let h3 = hashFloat(pi, frame, 2u) * 2.0 - 1.0;
        let sqrtDt = sqrt(sim.dt);
        p.vel.x += sim.temperature * sim.rMax * h1 * sqrtDt * 2.2;
        p.vel.y += sim.temperature * sim.rMax * h2 * sqrtDt * 2.2;
        p.omega += sim.temperature * h3 * sqrtDt * 12.0;
    }

    let d = sqrt(p.pos.x * p.pos.x + p.pos.y * p.pos.y) / sim.worldSize;

    if (sim.vortex > 0.0) {
        let s = getSpeciesInfo(u32(p.colour));
        let tag = max(s.atomicNumber * 0.17, 0.25);
        p.vel.x -= p.pos.x * sim.dt * 4.0 * cos(d * 20.0 * p.pos.x * p.pos.y / tag) * sin(p.pos.x * 8.0 * tag);
        p.vel.y -= p.pos.y * sim.dt * 4.0 * cos(d * 20.0 * p.pos.y * p.pos.x / tag) * cos(p.pos.y * 8.0 * tag);

        p.vel.x -= p.pos.y * sim.dt * 4.0 * pow(d / 0.5, 2.0) * sin(d * 10.0 + tag);
        p.vel.y += p.pos.x * sim.dt * 4.0 * pow(d / 0.5, 2.0) * sin(d * 10.0 + tag);
    }

    if (uniforms.mouse.z != 0.0) {
        let dx = uniforms.mouse.x - p.pos.x;
        let dy = uniforms.mouse.y - p.pos.y;
        let md = sqrt(dx * dx + dy * dy);
        if (md < sim.rMax * 2.0) {
            let sign = select(-1.0, 1.0, uniforms.mouse.z == 1.0);
            p.vel.x += dx * 3.0 * sign;
            p.vel.y += dy * 3.0 * sign;
        }
    }

    if (sim.border > 0.0 && d > 0.9) {
        let f = (d - 0.9) * 10.0;
        p.vel.x -= p.pos.x * f;
        p.vel.y -= p.pos.y * f;
    }

    p.pos += p.vel * sim.dt;
    p.theta = wrapAngle(p.theta + p.omega * sim.dt);

    return p;
}
