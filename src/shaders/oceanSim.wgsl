// Shared sphere-surface (S^2) Original Fork (51bc381) Particle Life + Biogenic 2-Bead Lipid Bilayer + Ocean Currents helpers.
// Keeps zero interaction with the Atmospheric Wind Chemistry layer.

struct OceanForceAccum {
    total: vec3f,
    avoid: vec3f,
    div: f32,
    torque: f32,
}

fn oceanForceCurve(r: f32, a: f32, beta: f32) -> vec2f {
    if (r < beta) {
        return vec2f(0.0, r / max(beta, 1e-4) - 1.0);
    } else if (r > beta && r < 1.0) {
        return vec2f(a * (1.0 - abs(2.0 * r - 1.0 - beta) / max(1.0 - beta, 1e-4)), 0.0);
    }
    return vec2f(0.0);
}

fn siteForce(
    disp: vec3f,
    dFallback: vec3f,
    rCore: f32,
    rCut: f32,
    repelK: f32,
    attractK: f32,
) -> vec3f {
    let d = length(disp);
    let dir = select(dFallback, disp / d, d > 1e-5);
    if (d < rCore) {
        return -dir * (repelK * (1.0 - d / max(rCore, 1e-4)));
    }
    if (attractK > 0.0 && d < rCut) {
        let span = max(rCut - rCore, 1e-4);
        let u = (d - rCore) / span;
        let bell = 1.0 - abs(2.0 * u - 1.0);
        return dir * (attractK * bell);
    }
    return vec3f(0.0);
}

fn accumulateOceanNeighbor(
    pi: u32,
    p: Particle,
    ipPos: vec3f,
    a: f32,
    acc: ptr<function, OceanForceAccum>,
    sim: OceanSim,
) {
    let R = max(sim.worldSize, 0.25);
    let dVec = ipPos - p.pos;
    let chord2 = dot(dVec, dVec);
    let maxAng = sim.rMax / R;

    if (chord2 <= 1e-12) {
        let eEast = sphereEastBasis(p.pos);
        (*acc).avoid += eEast * (((f32(pi) * 20293.302) % 4932.329) / 100000.0);
        return;
    }
    if (chord2 >= maxAng * maxAng) {
        return;
    }

    let chord = sqrt(chord2);
    let r = R * 2.0 * asin(clamp(0.5 * chord, 0.0, 1.0));
    if (r <= 1e-6 || r >= sim.rMax) {
        return;
    }

    let dI = orthoBasis(p.pos, ipPos - dot(ipPos, p.pos) * p.pos);
    let f = oceanForceCurve(r / sim.rMax, a, sim.beta);
    (*acc).total += dI * f.x;
    (*acc).avoid += dI * f.y;
    (*acc).div += max(0.0, -f.y) / 10.0 * sim.avoidance;
}

fn accumulateOceanOrLipidNeighbor(
    pi: u32,
    p: Particle,
    myPhase: u32,
    myKin: f32,
    ipPos: vec3f,
    ipOrient: vec3f,
    nbPhase: u32,
    nbKin: f32,
    a: f32,
    acc: ptr<function, OceanForceAccum>,
    sim: OceanSim,
) {
    if (myPhase == 1u && nbPhase == 1u) {
        accumulateOceanNeighbor(pi, p, ipPos, a, acc, sim);
        return;
    }

    let R = max(sim.worldSize, 0.25);
    let dVec = ipPos - p.pos;
    let chord2 = dot(dVec, dVec);
    let maxAng = sim.rMax / R;

    if (chord2 <= 1e-12) {
        let eEast = sphereEastBasis(p.pos);
        (*acc).avoid += eEast * (((f32(pi) * 20293.302) % 4932.329) / 100000.0);
        return;
    }
    if (chord2 >= maxAng * maxAng) {
        return;
    }

    let r = R * 2.0 * asin(clamp(0.5 * sqrt(chord2), 0.0, 1.0));
    if (r <= 1e-6 || r >= sim.rMax) {
        return;
    }

    let dI = orthoBasis(p.pos, ipPos - dot(ipPos, p.pos) * p.pos);
    let dIJ = dI * r;
    let dArm = 0.11 * sim.rMax;
    let sameKin = kinAngularDist(myKin, nbKin) < 0.30;

    if (myPhase == 2u && nbPhase == 2u) {
        let uI = orthoBasis(p.pos, p.orient);
        let bI = cross(p.pos, uI);
        let uJ = orthoBasis(p.pos, ipOrient);

        let fCore = siteForce(dIJ, dI, 0.40 * sim.rMax, sim.rMax, 2.2, 0.0);
        let dispHH = dIJ + dArm * (uJ - uI);
        let fHH = siteForce(dispHH, dI, 0.50 * sim.rMax, sim.rMax, 1.8, 0.0);
        (*acc).avoid += fCore + fHH * 0.55;
        (*acc).div += 0.65;

        let align = dot(uI, uJ);
        let radialAlign = abs(dot(uI, dI));
        let perpFactor = 1.0 - radialAlign;
        let tailSide = dot(-uI, dI);

        if (align > 0.15) {
            let fLat = siteForce(dIJ, dI, 0.42 * sim.rMax, 0.92 * sim.rMax, 0.8, 2.2 * align * perpFactor);
            var fStack = vec3f(0.0);
            if (radialAlign > 0.55) {
                fStack = siteForce(dIJ, dI, 0.72 * sim.rMax, sim.rMax, 1.4 * radialAlign, 0.0);
            }
            (*acc).total += fLat + fStack;
            (*acc).torque += dot(bI, uJ) * 3.6 * perpFactor + dot(bI, fHH) * 2.2;
        } else if (align < -0.15) {
            if (tailSide > 0.20) {
                let pairW = (-align) * tailSide;
                (*acc).total += siteForce(dIJ, dI, 0.42 * sim.rMax, 0.82 * sim.rMax, 1.4, 2.0 * pairW);
                (*acc).torque += (dot(bI, -uJ) * 3.5 + dot(bI, -dI) * 2.8) * (-align);
            } else {
                (*acc).avoid += siteForce(dIJ, dI, 0.65 * sim.rMax, sim.rMax, 1.8, 0.0);
            }
        } else {
            (*acc).total += siteForce(dIJ, dI, 0.52 * sim.rMax, sim.rMax, 1.0, 0.0);
            (*acc).torque += dot(bI, uJ) * select(-2.5, 2.5, tailSide < 0.25);
        }
        return;
    }

    if (myPhase == 2u && nbPhase == 1u) {
        if (sameKin) {
            (*acc).avoid += siteForce(dIJ, dI, 0.56 * sim.rMax, sim.rMax, 2.6, 0.0);
            (*acc).div += 0.55;
            (*acc).total += siteForce(dIJ, dI, 0.56 * sim.rMax, 0.98 * sim.rMax, 0.0, 1.35);

            let uI = orthoBasis(p.pos, p.orient);
            let bI = cross(p.pos, uI);
            if (dot(uI, dI) >= -0.15) {
                (*acc).torque += dot(bI, dI) * 4.5;
            } else {
                (*acc).torque += dot(bI, -dI) * 3.5;
            }
        } else {
            let fShield = siteForce(dIJ, dI, 0.80 * sim.rMax, sim.rMax, 1.8, 0.0);
            (*acc).avoid += fShield;
        }
        return;
    }

    // myPhase == 1u && nbPhase == 2u
    if (sameKin) {
        let fBound = siteForce(dIJ, dI, 0.52 * sim.rMax, 0.92 * sim.rMax, 1.45, 0.20);
        (*acc).total += fBound * 0.60;
    } else {
        let fShield = siteForce(dIJ, dI, 0.80 * sim.rMax, sim.rMax, 1.9, 0.0);
        (*acc).avoid += fShield;
    }
}

fn finalizeOceanForce(acc: OceanForceAccum, sim: OceanSim) -> vec3f {
    return acc.total * (sim.rMax * sim.force / (1.0 + acc.div)) + acc.avoid;
}

fn integrateOceanParticle(
    pi: u32,
    pIn: Particle,
    netForce: vec3f,
    accTorque: f32,
    myPhase: u32,
    sim: OceanSim,
    oceanTex: texture_2d<f32>,
    mouseSphere: vec4f,
    oceanState: vec4f,
    bioSteer: vec4f,
) -> Particle {
    var p = pIn;
    let R = max(sim.worldSize, 0.25);

    let eEast = sphereEastBasis(p.pos);
    let eNorth = normalize(cross(p.pos, eEast));
    let oc = sampleOceanTexture(oceanTex, p.pos);

    var cTarget = vec3f(0.0);
    if (sim.vortex > 0.0 && sim.currentStrength > 0.0 && oc.w >= 0.35) {
        let cur3D = oc.x * eEast + oc.y * eNorth;
        cTarget = (cur3D / 0.7) * (0.30 * sim.currentStrength) * smoothstep(0.35, 0.55, oc.w);
        let cLen = length(cTarget);
        if (cLen > 1.15) {
            cTarget *= 1.15 / cLen;
        }
    }

    let linBioSteer = select(bioSteer.xyz, vec3f(0.0), myPhase == 2u && bioSteer.w > 1.5);
    let vitality = select(
        clamp(0.65 + 0.50 * oceanState.x - oceanState.z * 0.35, 0.35, 1.30),
        clamp(0.75 + 0.35 * oceanState.x, 0.45, 1.15),
        myPhase == 2u,
    );
    let toxicDrag = max(0.68, 1.0 - 0.20 * oceanState.z);
    let vTang = p.vel - dot(p.vel, p.pos) * p.pos;
    let vRel = vTang - cTarget;
    p.vel = cTarget + vRel * (sim.friction * toxicDrag) + (netForce * vitality + linBioSteer) * sim.dt;

    // Continental coastline containment using smoothed ocean mask gradient
    if (oc.w < 0.48) {
        let dAng = 0.025;
        let mE = sampleOceanTexture(oceanTex, normalize(p.pos + dAng * eEast)).w;
        let mW = sampleOceanTexture(oceanTex, normalize(p.pos - dAng * eEast)).w;
        let mN = sampleOceanTexture(oceanTex, normalize(p.pos + dAng * eNorth)).w;
        let mS = sampleOceanTexture(oceanTex, normalize(p.pos - dAng * eNorth)).w;
        let gradM = (mE - mW) * eEast + (mN - mS) * eNorth;
        let gLen = length(gradM);
        if (gLen > 1e-4) {
            let inward = gradM / gLen;
            let vn = dot(p.vel, inward);
            if (vn < 0.0) {
                p.vel -= 1.55 * vn * inward;
            }
            p.vel += inward * ((0.52 - oc.w) * 4.5 + 0.14);
        } else {
            p.vel *= 0.85;
        }
    }

    p.vel += mouseSphereImpulse(p.pos, sim.rMax, R, mouseSphere);

    var uBody = orthoBasis(p.pos, p.orient);
    if (myPhase == 2u) {
        if (bioSteer.w > 1.5 && length(bioSteer.xyz) > 1e-4) {
            uBody = orthoBasis(p.pos, bioSteer.xyz);
            p.omega = 0.0;
        } else {
            let angFric = pow(0.5, sim.dt / 0.025);
            p.omega = clamp(p.omega * angFric + accTorque * sim.dt, -22.0, 22.0);
        }
    } else {
        p.omega = 0.0;
    }

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
    p._pad = clamp(bioSteer.w, -1.0, 1.0);
    return p;
}
