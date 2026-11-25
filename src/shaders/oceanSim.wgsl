// Emergent sphere-surface (S^2) Particle Life + amphiphilic bilayers + 3-MDP physical levers.
// All attraction/repulsion emerges from the unified combined attractor matrix modulated by
// Particle, Blob, and Species MDPs (no hard-coded steering vectors or role overrides).

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

fn accumulateUnifiedEmergentNeighbor(
    pi: u32,
    p: Particle,
    myPhase: u32,
    myKin: f32,
    pol: MdpLeverProfile,
    ipPos: vec3f,
    ipOrient: vec3f,
    nbPhase: u32,
    nbChemRole: u32,
    nbKin: f32,
    aBioBio: f32,
    acc: ptr<function, OceanForceAccum>,
    sim: OceanSim,
) {
    if (myPhase != 1u && nbPhase == 0u) {
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
    let rNorm = r / sim.rMax;
    let betaEff = clamp(sim.beta + pol.dBeta, 0.16, 0.68);

    // Living cell (`phase == 1u`) interacting with Chemistry atom (`nbPhase == 0u`) via Combined Attractor Matrix
    if (nbPhase == 0u) {
        if (nbChemRole == 0u) {
            return;
        }
        let aChem = select(
            clamp(pol.dChemNutr, -1.2, 1.2) * 0.52,
            clamp(pol.dChemHalogen, -1.2, 1.2) * 0.65,
            nbChemRole == 4u,
        );
        if (rNorm < betaEff) {
            if (aChem < 0.0) {
                (*acc).total += dI * (aChem * 0.55 * (1.0 - rNorm / max(betaEff, 1e-4)));
            }
        } else {
            let fChem = oceanForceCurve(rNorm, aChem, betaEff);
            (*acc).total += dI * fChem.x;
        }
        return;
    }

    // Living cytoplasm (`myPhase == 1u && nbPhase == 1u`) via Combined Attractor Matrix + Chiral Pair-Force Twist
    if (myPhase == 1u && nbPhase == 1u) {
        let sameKin = kinAngularDist(myKin, nbKin) < 0.30;
        let aEff = clamp(aBioBio + select(pol.dForeignBio, pol.dSameKin, sameKin), -1.5, 1.5);
        let f = oceanForceCurve(rNorm, aEff, betaEff);
        let tPerp = cross(p.pos, dI);
        let twist = pol.chiralTwist * clamp(abs(aEff), 0.20, 1.0);
        let dTwist = dI * cos(twist) + tPerp * sin(twist);
        (*acc).total += dTwist * f.x;
        (*acc).avoid += dI * f.y;
        (*acc).div += max(0.0, -f.y) / 10.0 * sim.avoidance;
        return;
    }

    let dIJ = dI * r;
    // Crustal mineral (`nbPhase == 3u`): physical reef steric core + MDP-learned attraction/avoidance
    if (nbPhase == 3u) {
        let aMin = clamp(pol.dMineral, -1.0, 1.0) * 0.55;
        let fMin = oceanForceCurve(rNorm, aMin, 0.44);
        (*acc).total += dI * fMin.x;
        (*acc).avoid += dI * (fMin.y * 1.6);
        (*acc).div += 0.20;
        return;
    }

    let dArm = 0.11 * sim.rMax;
    let sameKin = kinAngularDist(myKin, nbKin) < 0.30;

    // Amphiphilic bilayer self-assembly (`2u <-> 2u`)
    if (myPhase == 2u && nbPhase == 2u) {
        let uI = orthoBasis(p.pos, p.orient);
        let bI = cross(p.pos, uI);
        let uJ = orthoBasis(p.pos, ipOrient);

        let fCore = siteForce(dIJ, dI, 0.40 * sim.rMax, sim.rMax, 2.2, 0.0);
        let dispHH = dIJ + dArm * (uJ - uI);
        let fHH = siteForce(dispHH, dI, 0.50 * sim.rMax, sim.rMax, 1.8, 0.0);
        (*acc).avoid += fCore + fHH * 0.55;
        (*acc).div += 0.45;

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

    let lipGain = clamp(0.75 + 0.45 * pol.dLipid, 0.35, 1.35);
    let aLipForeign = clamp(pol.dForeignBio * 0.65 + pol.dLipid * 0.45, -1.2, 1.0);

    // Lipid (`2u`) interacting with Cytoplasm (`1u`)
    if (myPhase == 2u && nbPhase == 1u) {
        if (sameKin) {
            (*acc).avoid += siteForce(dIJ, dI, 0.52 * sim.rMax, sim.rMax, 2.2, 0.0);
            (*acc).div += 0.35;
            (*acc).total += siteForce(dIJ, dI, 0.52 * sim.rMax, 0.98 * sim.rMax, 0.0, 1.35 * lipGain);

            let uI = orthoBasis(p.pos, p.orient);
            let bI = cross(p.pos, uI);
            if (dot(uI, dI) >= -0.15) {
                (*acc).torque += dot(bI, dI) * 4.5;
            } else {
                (*acc).torque += dot(bI, -dI) * 3.5;
            }
        } else if (aLipForeign < 0.0) {
            (*acc).avoid += siteForce(dIJ, dI, 0.62 * sim.rMax, sim.rMax, -aLipForeign * 1.15, 0.0);
        } else {
            (*acc).total += siteForce(dIJ, dI, 0.48 * sim.rMax, 0.96 * sim.rMax, 0.8, aLipForeign * 0.95);
        }
        return;
    }

    // Cytoplasm (`1u`) interacting with Lipid (`2u`)
    if (sameKin) {
        let fBound = siteForce(dIJ, dI, 0.50 * sim.rMax, 0.92 * sim.rMax, 1.35, 0.25 * lipGain);
        (*acc).total += fBound * 0.60;
    } else if (aLipForeign < 0.0) {
        (*acc).avoid += siteForce(dIJ, dI, 0.62 * sim.rMax, sim.rMax, -aLipForeign * 1.15, 0.0);
    } else {
        (*acc).total += siteForce(dIJ, dI, 0.48 * sim.rMax, 0.96 * sim.rMax, 0.8, aLipForeign * 0.95);
    }
}

fn finalizeOceanForce(acc: OceanForceAccum, sim: OceanSim) -> vec3f {
    return acc.total * (sim.rMax * sim.force / (1.0 + min(acc.div, 6.0))) + acc.avoid;
}

fn integrateOceanParticle(
    pi: u32,
    pIn: Particle,
    netForce: vec3f,
    accTorque: f32,
    myPhase: u32,
    sim: OceanSim,
    oc: vec4f,
    windRaw: vec3f,
    windStrength: f32,
    windRefSpeed: f32,
    terrainSample: vec4f,
    tSample: vec4f,
    localTempScale: f32,
    mouseSphere: vec4f,
    oceanState: vec4f,
    bioSteer: vec4f,
    frameF: f32,
    leaderId: u32,
    packedBlobZ: u32,
) -> Particle {
    _ = windRaw;
    _ = windStrength;
    _ = windRefSpeed;
    var p = pIn;
    let R = max(sim.worldSize, 0.25);

    let eEast = sphereEastBasis(p.pos);
    let eNorth = normalize(cross(p.pos, eEast));
    let tHabK = effectiveSurfaceTempK(tSample, oc.w);
    let tBlend = clamp(localTempScale, 0.0, 1.5);
    let elevKm = max(0.0, terrainSample.x);

    let pol = evalCombinedMdpPolicy(packedBlobZ);
    let landFrac = clamp(1.0 - smoothstep(0.18, 0.48, oc.w), 0.0, 1.0);
    let substrateAvail = clamp(
        max(landFrac, smoothstep(0.30, 0.58, oc.w) * (1.0 - 0.55 * smoothstep(0.60, 0.95, oc.w))),
        0.15,
        1.0,
    );
    let grip = select(0.15, clamp(0.28 + pol.dDrag, 0.02, 0.88) * substrateAvail, myPhase == 1u);

    var cTarget = vec3f(0.0);
    if (sim.vortex > 0.0 && sim.currentStrength > 0.0 && oc.w >= 0.35) {
        let cur3D = oc.x * eEast + oc.y * eNorth;
        cTarget = (cur3D / 0.7) * (0.30 * sim.currentStrength) * smoothstep(0.35, 0.55, oc.w);
        cTarget *= (1.0 - 0.82 * grip);
        let cLen = length(cTarget);
        if (cLen > 1.15) {
            cTarget *= 1.15 / cLen;
        }
    }

    let baseVitality = select(
        clamp(0.65 + 0.50 * oceanState.x - oceanState.z * 0.35, 0.35, 1.30),
        clamp(0.75 + 0.35 * oceanState.x, 0.45, 1.15),
        myPhase == 2u,
    );
    let thermFluid = mix(
        1.0,
        clamp(0.72 + 0.46 * smoothstep(271.5, 304.5, tHabK), 0.68, 1.24),
        tBlend,
    );
    let vitality = baseVitality * thermFluid;
    let frostDrag = mix(1.0, 0.90, smoothstep(271.5, 250.0, tHabK) * landFrac * tBlend);
    let toxicDrag = max(0.68, 1.0 - 0.20 * oceanState.z);
    let velDamp = clamp(sim.friction * (1.0 - 0.36 * grip) * toxicDrag * frostDrag, 0.25, 0.99);
    let slopeResist = select(
        vec3f(0.0),
        -(terrainSample.y * eEast + terrainSample.z * eNorth) * (0.05 * smoothstep(0.4, 3.0, elevKm)) * (1.0 - 0.85 * grip),
        landFrac > 0.40,
    );

    // Klinokinesis (run-and-tumble agitation coherent at leader scale) — zero gradient-steering!
    var kForce = vec3f(0.0);
    if (myPhase == 1u && pol.agitation > 0.005) {
        let eEpoch = u32(floor(max(frameF, 0.0) / 14.0));
        let lAng = hashFloat(leaderId, eEpoch, 91u) * TWO_PI;
        let pAng = hashFloat(pi, eEpoch, 92u) * TWO_PI;
        let dLeader = eEast * cos(lAng) + eNorth * sin(lAng);
        let dSelf = eEast * cos(pAng) + eNorth * sin(pAng);
        let hasSharedLeader = ((packedBlobZ >> 10u) & 1u) != 0u;
        let tMix = select(dSelf, dLeader * 0.78 + dSelf * 0.22, hasSharedLeader);
        let tLen = length(tMix);
        let tumbleDir = select(eEast, tMix / tLen, tLen > 1e-5);
        kForce = tumbleDir * (pol.agitation * 0.42 * vitality * (1.0 - 0.55 * grip));
    }

    let vTang = p.vel - dot(p.vel, p.pos) * p.pos;
    let vRel = vTang - cTarget;
    p.vel = cTarget + vRel * velDamp + (netForce * vitality + kForce + slopeResist) * sim.dt;

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
    p._pad = select(0.0, packThreeMdpPadFromZ(packedBlobZ), myPhase == 1u);
    return p;
}

// Crust-pinned inorganic mineral (`phase == 3u`): resists wind & current advection cleanly until eroded.
fn integrateMineralParticle(
    pIn: Particle,
    mineralState: vec4f,
    bioReact: vec4f,
) -> Particle {
    var p = pIn;
    p.pos = normalize(p.pos);
    p.vel = vec3f(0.0);
    p.omega = 0.0;
    if (bioReact.w > 1.5 && length(bioReact.xyz) > 1e-4) {
        p.orient = orthoBasis(p.pos, bioReact.xyz);
    } else {
        p.orient = orthoBasis(p.pos, p.orient);
    }
    let integrity = clamp(mineralState.x, 0.0, 1.25);
    p._pad = integrity - 0.5;
    return p;
}
