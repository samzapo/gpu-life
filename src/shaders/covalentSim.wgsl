fn force(r: f32, a: f32) -> vec2f {
    let beta = max(sim.beta, 1e-4);
    let oneMinusBeta = max(1.0 - sim.beta, 1e-4);
    if (r < sim.beta) {
        return vec2f(0.0, r / beta - 1.0);
    } else if (sim.beta < r && r < 1.0) {
        return vec2f(a * (1.0 - abs(2.0 * r - 1.0 - sim.beta) / oneMinusBeta), 0.0);
    } else {
        return vec2f(0.0, 0.0);
    }
}

// Evaluates M[c1, c2](theta) and dM/dtheta(theta) using periodic Catmull-Rom interpolation.
fn evalMatrix(c1: u32, c2: u32, theta: f32) -> vec2f {
    let samples = u32(sim.thetaSamples);
    let samplesF = sim.thetaSamples;
    let wrapped = wrapAngle(theta);
    let u = (wrapped / TWO_PI) * samplesF;
    let floorU = floor(u);
    let k1 = u32(floorU) % samples;
    let t = u - floorU;

    let k0 = (k1 + samples - 1u) % samples;
    let k2 = (k1 + 1u) % samples;
    let k3 = (k1 + 2u) % samples;

    let nColours = max(u32(sim.colours), 1u);
    let sc1 = min(c1, nColours - 1u);
    let sc2 = min(c2, nColours - 1u);
    let base = (sc1 * nColours + sc2) * samples;

    let p0 = matrix[base + k0];
    let p1 = matrix[base + k1];
    let p2 = matrix[base + k2];
    let p3 = matrix[base + k3];

    let a0 = p1;
    let a1 = 0.5 * (p2 - p0);
    let a2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let a3 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;

    let val = a0 + t * (a1 + t * (a2 + t * a3));
    let dValDTheta = (a1 + t * (2.0 * a2 + 3.0 * a3 * t)) * (samplesF / TWO_PI);

    return vec2f(val, dValDTheta);
}

fn accumulateNeighbor(
    pi: u32,
    p: Particle,
    ipPos: vec2f,
    ipColour: f32,
    ipTheta: f32,
    acc: ptr<function, ForceAccum>,
) {
    let rx = ipPos.x - p.pos.x;
    let ry = ipPos.y - p.pos.y;
    let rs = rx * rx + ry * ry;
    let mrs = sim.rMax * sim.rMax;

    if (rs > 0.0 && rs < mrs) {
        let r = sqrt(rs);
        let rNorm = r / sim.rMax;
        let dTheta = wrapAngle(ipTheta - p.theta);
        let m = evalMatrix(u32(p.colour), u32(ipColour), dTheta);
        let f = force(rNorm, m.x);

        let invR = 1.0 / r;
        let dx = rx * invR;
        let dy = ry * invR;

        if (f.x >= 0.0) {
            (*acc).attractForce.x += dx * f.x;
            (*acc).attractForce.y += dy * f.x;
        } else {
            (*acc).repelForce.x += dx * f.x;
            (*acc).repelForce.y += dy * f.x;
        }

        (*acc).avoidForce.x += dx * f.y;
        (*acc).avoidForce.y += dy * f.y;

        (*acc).div += max(0.0, -f.y) / 10.0 * sim.avoidance;

        let overlap = 1.0 - rNorm;
        (*acc).bondLoad += max(0.0, m.x) * overlap;
        (*acc).torque += -m.y * overlap;
    } else if (rs == 0.0) {
        (*acc).avoidForce.x += (f32(pi) * 20293.302 % 4932.329) / 100000.0;
    }
}

fn finalizeForces(acc: ForceAccum) -> ForceResult {
    let repDenom = 1.0 + acc.div;
    let attDenom = 1.0 + acc.div + sim.valency * acc.bondLoad;
    let netForce = (acc.attractForce / attDenom + acc.repelForce / repDenom) * (sim.rMax * sim.force) + acc.avoidForce;
    let netTorque = sim.torque * acc.torque / repDenom;
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
        p.vel.x += sim.temperature * sim.rMax * h1 * sqrtDt * 2.0;
        p.vel.y += sim.temperature * sim.rMax * h2 * sqrtDt * 2.0;
        p.omega += sim.temperature * h3 * sqrtDt * 15.0;
    }

    let d = sqrt(p.pos.x * p.pos.x + p.pos.y * p.pos.y) / sim.worldSize;

    if (sim.vortex > 0.0) {
        let samples = u32(sim.thetaSamples);
        let k = min(samples / 8u, samples - 1u);
        let nColours = max(u32(sim.colours), 1u);
        let cIdx = min(u32(p.colour), nColours - 1u);
        let m0 = matrix[(cIdx * nColours + cIdx) * samples + k];
        let m1 = matrix[(cIdx * nColours + ((cIdx + 1u) % nColours)) * samples + k];
        let m0Safe = select(m0, 1.0, abs(m0) < 1e-4);
        p.vel.x -= p.pos.x * sim.dt * 5.0 * cos(d * 30.0 * p.pos.x * p.pos.y / m0Safe) * sin(p.pos.x * 10.0 * m0);
        p.vel.y -= p.pos.y * sim.dt * 5.0 * cos(d * 30.0 * p.pos.y * p.pos.x / m0Safe) * cos(p.pos.y * 10.0 * m0);

        p.vel.x -= p.pos.y * sim.dt * 5.0 * pow(d / 0.5, 2.0) * sin(d * 10.0 + m1);
        p.vel.y += p.pos.x * sim.dt * 5.0 * pow(d / 0.5, 2.0) * sin(d * 10.0 + m1);
    }

    if (uniforms.mouse.z != 0.0) {
        let dx = uniforms.mouse.x - p.pos.x;
        let dy = uniforms.mouse.y - p.pos.y;
        let md = sqrt(dx * dx + dy * dy);
        if (md < sim.rMax) {
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
