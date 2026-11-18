// Bidirectional energy & catalytic exchange between immutable Ocean Life (species)
// and immutable Chemistry atoms (atomicNumber + live valence occupancy).
// No atomic numbers or particle counts are ever mutated across layers.

struct BioLink {
    enabled: f32,
    rate: f32,
    catRadius: f32,
    worldSize: f32,
    airCellSize: f32,
    airCellAmt: f32,
    oceanCellSize: f32,
    oceanCellAmt: f32,
    airColours: f32,
    oceanColours: f32,
    dt: f32,
    _p0: f32,
}

fn bioRoleKind(atomicNum: u32, valence: u32) -> u32 {
    if (valence == 0u) {
        return 0u; // Inert noble gas (He, Ne, Ar)
    }
    // Poison / reactive halogens: F(7), Cl(13), higher halogens
    if (atomicNum == 7u || atomicNum == 13u) {
        return 4u;
    }
    // Trace vitamins / metalloenzyme cofactors: Li(3), Be(4), Mg(10)
    if (atomicNum == 3u || atomicNum == 4u || atomicNum == 10u) {
        return 3u;
    }
    // Primary carbon food backbone: C(5)
    if (atomicNum == 5u) {
        return 2u;
    }
    // Energy fuels & hydrides/chalcogens: H(1), S(12)
    if (atomicNum == 1u || atomicNum == 12u) {
        return 1u;
    }
    // Reactive oxides & silicate mineral builders: O(6), Si(11), Na(9), etc.
    return 5u;
}

// Computes `(dEnergy, dVitamin, dPoison, signedSteerMag)` imparted to one Ocean Life particle
// by one nearby Chemistry atom from frozen time-t state.
fn evalLifeFromAtomPair(
    lifeColour: f32,
    lifeState: vec4f,
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    r: f32,
    bio: BioLink,
) -> vec4f {
    let kind = bioRoleKind(atomicNum, valence);
    if (kind == 0u || r >= bio.catRadius) {
        return vec4f(0.0);
    }

    let w = (1.0 - r / max(bio.catRadius, 1e-4)) * bio.rate;
    let role = u32(max(lifeColour, 0.0)) % 4u;
    let energy = clamp(lifeState.x, 0.0, 1.5);
    let vit = clamp(lifeState.y, 0.0, 1.0);

    var dE = 0.0;
    var dVit = 0.0;
    var dPoi = 0.0;
    var steer = 0.0;

    if (kind == 3u) {
        // Vitamin cofactor (Mg, Be, Li): stabilizes metabolism & detox without consuming atom
        dVit += 0.22 * w;
        steer += 0.25 * w;
    } else if (kind == 4u) {
        // Halogen poison (Cl, F)
        if (role == 2u) {
            // Halophile / detoxifier converts Cl + Na toward benign salts at modest energy gain
            dE += 0.08 * w;
            steer += 0.30 * w;
        } else {
            dPoi += 0.42 * w * (1.0 - 0.50 * vit);
            dE -= 0.16 * w * (1.0 - 0.35 * vit);
            steer -= 0.85 * w;
        }
    } else if (kind == 1u) {
        // Fuel (H, S), especially bonded H2 / HS / HO fuel carriers
        let fuelReady = clamp(atomLoad + 0.35, 0.35, 1.5);
        if (energy < 1.40) {
            let capGain = (0.18 + 0.10 * vit) * fuelReady * w;
            dE += select(capGain * 0.65, capGain, role == 1u || role == 3u);
        }
        steer += 0.35 * w;
    } else if (kind == 2u) {
        // Carbon Food (C): autotrophs spend energy to synthesize; heterotrophs harvest bonded C
        if (role == 0u) {
            if (energy > 0.28) {
                dE -= 0.07 * w;
            }
            steer += 0.30 * w;
        } else {
            let foodRich = clamp(atomLoad, 0.2, 2.2);
            dE += 0.26 * foodRich * w * (1.0 + 0.25 * vit);
            steer += 0.70 * w;
        }
    } else {
        // O(6), Si(11), Na(9): radical O (HO•) offers modest oxidant fuel; saturated oxides cost autotroph energy to split
        if (atomicNum == 6u) {
            if (atomLoad > 0.05 && atomLoad < 1.35) {
                dE += 0.10 * w;
            } else if (role == 0u && energy > 0.40 && atomLoad >= 1.35) {
                dE -= 0.09 * w;
            }
        }
    }

    return vec4f(dE, dVit, dPoi, steer);
}

// Computes `(radialKickScalar, tangentialSpinScalar, actScale, glow)` imparted to one Chemistry atom
// by one nearby Ocean Life particle (`dToLife` points atom -> life on tangent plane).
fn evalAtomFromLifePair(
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    lifeColour: f32,
    lifeState: vec4f,
    r: f32,
    bio: BioLink,
) -> vec4f {
    let kind = bioRoleKind(atomicNum, valence);
    if (kind == 0u || r >= bio.catRadius) {
        return vec4f(0.0, 0.0, 1.0, 0.0);
    }

    let w = (1.0 - r / max(bio.catRadius, 1e-4)) * bio.rate;
    let role = u32(max(lifeColour, 0.0)) % 4u;
    let energy = clamp(lifeState.x, 0.0, 1.5);
    let vit = clamp(lifeState.y, 0.0, 1.0);
    let catPower = clamp(0.25 + energy * 0.75 + vit * 0.35, 0.0, 1.5);

    var kickToLife = 0.0;
    var spinSwirl = 0.0;
    var actScale = 1.0;
    var glow = 0.0;

    if (kind == 3u) {
        // Vitamin cofactor stabilizes nearby bonds by calming thermal noise / activation
        actScale = 0.45;
        glow = 0.35 * w;
    } else if (kind == 4u && role == 2u && energy > 0.25) {
        // Bioremediator catalyzes Cl/F + Na crystallization
        kickToLife = 0.22 * catPower * w;
        actScale = 0.18;
        glow = -0.65 * w;
    } else if (kind == 1u) {
        // H / S fuel
        if (role == 0u && energy > 0.28) {
            // Autotroph focuses H into organic synthesis pockets
            kickToLife = 0.32 * catPower * w;
            actScale = 0.14;
            glow = 0.75 * w;
        } else if (energy < 1.40) {
            // Consumer cleaves weak fuel bonds via thermal-kinetic kick so H seeks O / H2O
            kickToLife = -0.42 * w;
            spinSwirl = 0.55 * w;
            actScale = 0.35;
            glow = -0.75 * w;
        }
    } else if (kind == 2u) {
        // Carbon substrate
        if (role == 0u && energy > 0.28) {
            // Anabolism: drop barrier and pull C atoms together into longer chains
            kickToLife = 0.40 * catPower * w;
            actScale = 0.10;
            glow = 1.0 * w;
        } else if (atomLoad > 0.35) {
            // Catabolism: cleave food chain bonds and release reaction glow
            kickToLife = -0.35 * w;
            spinSwirl = 0.65 * w;
            glow = -0.95 * w;
        }
    } else if (atomicNum == 6u) {
        // Oxygen dynamics: autotrophs cleave excess bound oxygen from saturated C/Si oxides;
        // detoxifiers/synthesists stabilize hydroxyl HO• toward H2O
        if (role == 0u && energy > 0.40 && atomLoad >= 1.35) {
            kickToLife = -0.65 * catPower * w;
            spinSwirl = 0.45 * w;
            glow = -0.75 * w;
        } else if (atomLoad > 0.05 && atomLoad < 1.35) {
            actScale = 0.25;
            kickToLife = 0.18 * w;
            glow = 0.45 * w;
        }
    }

    return vec4f(kickToLife, spinSwirl, actScale, glow);
}

fn packDeltaActGlow(actScale: f32, glow: f32) -> f32 {
    let d = clamp(1.0 - actScale, 0.0, 0.95);
    let g = clamp(glow, -1.0, 1.0);
    let dQ = round(d * 1000.0);
    let gQ = round((g + 1.0) * 1000.0);
    if (dQ <= 0.5 && abs(g) < 1e-4) {
        return 0.0;
    }
    return 1.0 + dQ * 2048.0 + gQ;
}

fn unpackActGlow(packed: f32) -> vec2f {
    if (packed <= 0.5) {
        return vec2f(1.0, 0.0);
    }
    let v = max(packed - 1.0, 0.0);
    let dQ = floor(v / 2048.0);
    let gQ = v - dQ * 2048.0;
    let actScale = clamp(1.0 - dQ / 1000.0, 0.05, 1.0);
    let glow = clamp(gQ / 1000.0 - 1.0, -1.0, 1.0);
    return vec2f(actScale, glow);
}

struct BioAirAccum {
    kick: vec3f,
    actScale: f32,
    glowSum: f32,
    count: f32,
}

fn accumulateBioAirNeighbor(
    airPos: vec3f,
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    lifePos: vec3f,
    lifeColour: f32,
    lifeState: vec4f,
    bio: BioLink,
    acc: ptr<function, BioAirAccum>,
) {
    let R = max(bio.worldSize, 0.25);
    let dVec = lifePos - airPos;
    let chord2 = dot(dVec, dVec);
    let maxAng = bio.catRadius / R;
    if (chord2 <= 1e-12 || chord2 >= maxAng * maxAng) {
        return;
    }
    let r = R * 2.0 * asin(clamp(0.5 * sqrt(chord2), 0.0, 1.0));
    if (r <= 1e-6 || r >= bio.catRadius) {
        return;
    }
    let tRaw = lifePos - dot(lifePos, airPos) * airPos;
    let tLen = length(tRaw);
    if (tLen <= 1e-6) {
        return;
    }
    let dToLife = tRaw / tLen;
    let tSwirl = cross(airPos, dToLife);
    let eff = evalAtomFromLifePair(atomicNum, valence, atomLoad, lifeColour, lifeState, r, bio);
    (*acc).kick += dToLife * eff.x + tSwirl * eff.y;
    (*acc).actScale = min((*acc).actScale, eff.z);
    (*acc).glowSum += eff.w;
    (*acc).count += 1.0;
}

fn finalizeBioAir(acc: BioAirAccum, bio: BioLink) -> vec4f {
    if (acc.count <= 0.0) {
        return vec4f(0.0);
    }
    let norm = 1.0 / (1.0 + 0.25 * max(acc.count - 1.0, 0.0));
    var k = acc.kick * (norm * 5.5);
    let kLen = length(k);
    let maxKick = 9.0 * max(bio.rate, 0.1);
    if (kLen > maxKick) {
        k *= maxKick / kLen;
    }
    let glow = clamp(acc.glowSum * norm, -1.0, 1.0);
    let actScale = mix(1.0, clamp(acc.actScale, 0.05, 1.0), clamp(bio.rate, 0.0, 1.5));
    return vec4f(k, packDeltaActGlow(actScale, glow));
}

struct BioOceanAccum {
    dE: f32,
    dVit: f32,
    dPoi: f32,
    steer: vec3f,
    count: f32,
}

fn accumulateBioOceanNeighbor(
    lifePos: vec3f,
    lifeColour: f32,
    lifeState: vec4f,
    airPos: vec3f,
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    bio: BioLink,
    acc: ptr<function, BioOceanAccum>,
) {
    let R = max(bio.worldSize, 0.25);
    let dVec = airPos - lifePos;
    let chord2 = dot(dVec, dVec);
    let maxAng = bio.catRadius / R;
    if (chord2 <= 1e-12 || chord2 >= maxAng * maxAng) {
        return;
    }
    let r = R * 2.0 * asin(clamp(0.5 * sqrt(chord2), 0.0, 1.0));
    if (r <= 1e-6 || r >= bio.catRadius) {
        return;
    }
    let tRaw = airPos - dot(airPos, lifePos) * lifePos;
    let tLen = length(tRaw);
    if (tLen <= 1e-6) {
        return;
    }
    let dToAtom = tRaw / tLen;
    let eff = evalLifeFromAtomPair(lifeColour, lifeState, atomicNum, valence, atomLoad, r, bio);
    (*acc).dE += eff.x;
    (*acc).dVit += eff.y;
    (*acc).dPoi += eff.z;
    (*acc).steer += dToAtom * eff.w;
    (*acc).count += 1.0;
}

fn finalizeBioOceanState(state: vec4f, acc: BioOceanAccum, bio: BioLink) -> vec4f {
    let norm = select(0.0, 1.0 / (1.0 + 0.35 * max(acc.count - 1.0, 0.0)), acc.count > 0.0);
    let dtBio = bio.dt * 2.4;
    let vit = clamp(state.y + (acc.dVit * norm - 0.18 * bio.rate * state.y) * dtBio, 0.0, 1.0);
    let detoxRate = 0.22 + 0.45 * vit;
    let poi = clamp(state.z + (acc.dPoi * norm - detoxRate * state.z) * dtBio, 0.0, 1.0);
    let basalBurn = 0.08 * bio.rate + 0.12 * poi;
    let netDE = acc.dE * norm - basalBurn;
    let energy = clamp(state.x + netDE * dtBio, 0.05, 1.50);
    let flux = clamp(netDE, -1.0, 1.0);
    return vec4f(energy, vit, poi, flux);
}

fn finalizeBioOceanSteer(acc: BioOceanAccum, flux: f32, bio: BioLink) -> vec4f {
    if (acc.count <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, flux);
    }
    let norm = 1.0 / (1.0 + 0.35 * max(acc.count - 1.0, 0.0));
    var st = acc.steer * (norm * 2.8);
    let stLen = length(st);
    let maxSt = 5.0 * max(bio.rate, 0.1);
    if (stLen > maxSt) {
        st *= maxSt / stLen;
    }
    return vec4f(st, flux);
}
