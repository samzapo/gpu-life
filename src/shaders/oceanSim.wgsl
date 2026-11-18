// Shared sphere-surface (S^2) Original Fork (51bc381) Particle Life + Ocean Currents helpers.
// Keeps zero interaction with the Atmospheric Wind Chemistry layer.

struct OceanForceAccum {
    total: vec3f,
    avoid: vec3f,
    div: f32,
}

fn oceanForceCurve(r: f32, a: f32, beta: f32) -> vec2f {
    if (r < beta) {
        return vec2f(0.0, r / max(beta, 1e-4) - 1.0);
    } else if (r > beta && r < 1.0) {
        return vec2f(a * (1.0 - abs(2.0 * r - 1.0 - beta) / max(1.0 - beta, 1e-4)), 0.0);
    }
    return vec2f(0.0);
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

fn finalizeOceanForce(acc: OceanForceAccum, sim: OceanSim) -> vec3f {
    return acc.total * (sim.rMax * sim.force / (1.0 + acc.div)) + acc.avoid;
}

fn integrateOceanParticle(
    pi: u32,
    pIn: Particle,
    netForce: vec3f,
    sim: OceanSim,
    oceanTex: texture_2d<f32>,
    mouseSphere: vec4f,
    oceanState: vec4f,
    bioSteer: vec4f,
) -> Particle {
    var p = pIn;
    let R = max(sim.worldSize, 0.25);

    let vitality = clamp(0.45 + 0.65 * oceanState.x - oceanState.z * 0.40, 0.25, 1.35);
    let toxicDrag = max(0.65, 1.0 - 0.22 * oceanState.z);
    p.vel -= dot(p.vel, p.pos) * p.pos;
    p.vel = p.vel * (sim.friction * toxicDrag) + (netForce * vitality + bioSteer.xyz) * sim.dt;

    let eEast = sphereEastBasis(p.pos);
    let eNorth = normalize(cross(p.pos, eEast));
    let oc = sampleOceanTexture(oceanTex, p.pos);

    if (sim.vortex > 0.0 && sim.currentStrength > 0.0 && oc.w >= 0.35) {
        let cur3D = oc.x * eEast + oc.y * eNorth;
        var cTarget = (cur3D / 0.7) * (0.36 * sim.currentStrength);
        let cLen = length(cTarget);
        if (cLen > 1.25) {
            cTarget *= 1.25 / cLen;
        }
        p.vel += (cTarget - p.vel) * clamp(12.0 * sim.dt, 0.0, 0.42);
    }

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

    let stepRes = stepGeodesicOnSphere(
        p.pos,
        p.vel,
        0.0,
        eEast,
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
