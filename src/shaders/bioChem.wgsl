// Unified conserved-mass bio-chemistry & trophic ecology helpers.
// Every particle carries an immutable `chemSpecies` (`p.colour`), live `phase`
// (0 = Chemistry, 1 = Biomass), and `bioSpecies` (0 = Producer, 1 = Consumer,
// 2 = Halophile, 3 = Decomposer) packed into `lifeState.w`.

struct BioLink {
    enabled: f32,
    rate: f32,
    catRadius: f32,
    worldSize: f32,
    cellSize: f32,
    cellAmt: f32,
    chemColours: f32,
    bioColours: f32,
    dt: f32,
    frame: f32,
    seedHash: f32,
    _p0: f32,
}

struct BioStepResult {
    nextLifeState: vec4f,
    bioForce: vec4f,
}

fn isBiogenicElement(atomicNum: u32, valence: u32) -> bool {
    if (valence == 0u) {
        return false;
    }
    // Biogenic pool: H(1), C(5), O(6), S(12). Si(11), halogens, metals, noble gases remain strictly abiotic.
    return atomicNum == 1u || atomicNum == 5u || atomicNum == 6u || atomicNum == 12u;
}

fn bioRoleKind(atomicNum: u32, valence: u32) -> u32 {
    if (valence == 0u) {
        return 0u; // Inert noble gas (He, Ne, Ar)
    }
    if (atomicNum == 7u || atomicNum == 13u) {
        return 4u; // Halogen poison (F, Cl)
    }
    if (atomicNum == 3u || atomicNum == 4u || atomicNum == 10u) {
        return 3u; // Vitamin cofactor (Li, Be, Mg)
    }
    if (atomicNum == 5u) {
        return 2u; // Primary carbon food backbone (C)
    }
    if (atomicNum == 1u || atomicNum == 12u) {
        return 1u; // Energy fuel / hydride & sulfur (H, S)
    }
    return 5u; // Oxide / silicate / alkali (O, Si, Na, etc.)
}

fn evalSolarInsolation(pos: vec3f) -> f32 {
    let cosLat = sqrt(max(0.0, 1.0 - pos.z * pos.z));
    return 0.12 + 0.88 * pow(cosLat, 1.25);
}

fn evalCoastalUpwelling(oc: vec4f) -> f32 {
    let shelf = smoothstep(0.38, 0.58, oc.w) * (1.0 - 0.65 * smoothstep(0.76, 0.99, oc.w));
    return clamp(shelf + clamp(oc.z * 0.45, 0.0, 0.35), 0.0, 1.35);
}

fn pickStrainForGuild(guild: u32, nBio: u32, seedRand: f32) -> u32 {
    if (nBio <= 4u) {
        return min(guild, max(nBio, 1u) - 1u);
    }
    let strainCount = (nBio - guild + 3u) / 4u;
    let pick = u32(clamp(floor(seedRand * f32(strainCount)), 0.0, f32(max(strainCount, 1u) - 1u)));
    return min(pick * 4u + guild, nBio - 1u);
}

// Computes `(dEnergy, dVitamin, dPoison, signedSteerMag)` imparted to one living Biomass particle
// by one nearby Chemistry atom from frozen time-t state.
fn evalLifeFromAtomPair(
    bioSpecies: u32,
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
    let role = bioSpecies % 4u;
    let energy = clamp(lifeState.x, 0.0, 1.5);
    let vit = clamp(lifeState.y, 0.0, 1.0);

    var dE = 0.0;
    var dVit = 0.0;
    var dPoi = 0.0;
    var steer = 0.0;

    if (kind == 3u) {
        dVit += 0.24 * w;
        steer += 0.25 * w;
    } else if (kind == 4u) {
        if (role == 2u) {
            dE += 0.14 * w;
            steer += 0.45 * w;
        } else {
            let roleResist = select(1.0, 0.55, role == 3u);
            dPoi += 0.38 * w * (1.0 - 0.50 * vit) * roleResist;
            dE -= 0.14 * w * (1.0 - 0.35 * vit) * roleResist;
            steer -= 0.85 * w * roleResist;
        }
    } else if (kind == 1u) {
        let fuelReady = clamp(atomLoad + 0.35, 0.35, 1.5);
        if (energy < 1.40) {
            let capGain = (0.18 + 0.10 * vit) * fuelReady * w;
            dE += select(capGain * 0.65, capGain, role == 1u || role == 3u);
        }
        steer += 0.35 * w;
    } else if (kind == 2u) {
        if (role == 0u) {
            steer += 0.32 * w;
        } else {
            let foodRich = clamp(atomLoad + 0.25, 0.25, 2.2);
            dE += 0.24 * foodRich * w * (1.0 + 0.25 * vit);
            steer += 0.65 * w;
        }
    } else if (atomicNum == 6u) {
        if (atomLoad > 0.05 && atomLoad < 1.35) {
            dE += 0.09 * w;
        } else if (role == 0u && energy > 0.40 && atomLoad >= 1.35) {
            dE -= 0.05 * w;
        }
    }

    return vec4f(dE, dVit, dPoi, steer);
}

// Computes `(radialKickScalar, tangentialSpinScalar, actScale, glow)` imparted to one Chemistry atom
// by one nearby living Biomass particle (`dToLife` points atom -> life on tangent plane).
fn evalAtomFromLifePair(
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    bioSpecies: u32,
    lifeState: vec4f,
    r: f32,
    bio: BioLink,
) -> vec4f {
    let kind = bioRoleKind(atomicNum, valence);
    if (kind == 0u || r >= bio.catRadius) {
        return vec4f(0.0, 0.0, 1.0, 0.0);
    }

    let w = (1.0 - r / max(bio.catRadius, 1e-4)) * bio.rate;
    let role = bioSpecies % 4u;
    let energy = clamp(lifeState.x, 0.0, 1.5);
    let vit = clamp(lifeState.y, 0.0, 1.0);
    let catPower = clamp(0.25 + energy * 0.75 + vit * 0.35, 0.0, 1.5);

    var kickToLife = 0.0;
    var spinSwirl = 0.0;
    var actScale = 1.0;
    var glow = 0.0;

    if (kind == 3u) {
        actScale = 0.45;
        glow = 0.35 * w;
    } else if (kind == 4u && role == 2u && energy > 0.20) {
        kickToLife = 0.22 * catPower * w;
        actScale = 0.18;
        glow = -0.65 * w;
    } else if (kind == 1u) {
        if (role == 0u && energy > 0.28) {
            kickToLife = 0.32 * catPower * w;
            actScale = 0.14;
            glow = 0.75 * w;
        } else if (energy < 1.40) {
            kickToLife = -0.42 * w;
            spinSwirl = 0.55 * w;
            actScale = 0.35;
            glow = -0.75 * w;
        }
    } else if (kind == 2u) {
        if (role == 0u && energy > 0.28) {
            kickToLife = 0.40 * catPower * w;
            actScale = 0.10;
            glow = 1.0 * w;
        } else if (atomLoad > 0.35) {
            kickToLife = -0.35 * w;
            spinSwirl = 0.65 * w;
            glow = -0.95 * w;
        }
    } else if (atomicNum == 6u) {
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

struct BioUnifiedAccum {
    kick: vec3f,
    actScale: f32,
    glowSum: f32,
    chemBioCount: f32,
    dE: f32,
    dVit: f32,
    dPoi: f32,
    steer: vec3f,
    bioChemCount: f32,
    bioSteer: vec3f,
    nearHalogen: f32,
    nearProd: f32,
    nearEnergeticProd: f32,
    bestProdW: f32,
    bestProdSp: u32,
    nearCons: f32,
    nearActiveCons: f32,
    bestConsW: f32,
    bestConsSp: u32,
    consRepelDir: vec3f,
    nearHalo: f32,
    nearEnergeticHalo: f32,
    bestHaloW: f32,
    bestHaloSp: u32,
    nearDecomp: f32,
    bestDecompW: f32,
    bestDecompSp: u32,
    nearTotalBio: f32,
    nearStressedBio: f32,
}

fn initBioUnifiedAccum() -> BioUnifiedAccum {
    return BioUnifiedAccum(
        vec3f(0.0), 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, vec3f(0.0), 0.0,
        vec3f(0.0),
        0.0,
        0.0, 0.0, 0.0, 0u,
        0.0, 0.0, 0.0, 1u, vec3f(0.0),
        0.0, 0.0, 0.0, 2u,
        0.0, 0.0, 3u,
        0.0, 0.0,
    );
}

fn accumulateBioUnifiedNeighbor(
    myPos: vec3f,
    myPhase: u32,
    myAtomicNum: u32,
    myValence: u32,
    myAtomLoad: f32,
    myLifeState: vec4f,
    nbPos: vec3f,
    nbChemSpecies: f32,
    nbPhase: u32,
    nbBioSpecies: u32,
    nbLifeState: vec4f,
    nbAtomLoad: f32,
    nbAtomicNum: u32,
    nbValence: u32,
    bio: BioLink,
    acc: ptr<function, BioUnifiedAccum>,
) {
    let R = max(bio.worldSize, 0.25);
    let dVec = nbPos - myPos;
    let chord2 = dot(dVec, dVec);
    let maxAng = bio.catRadius / R;
    if (chord2 <= 1e-12 || chord2 >= maxAng * maxAng) {
        return;
    }
    let r = R * 2.0 * asin(clamp(0.5 * sqrt(chord2), 0.0, 1.0));
    if (r <= 1e-6 || r >= bio.catRadius) {
        return;
    }
    let tRaw = nbPos - dot(nbPos, myPos) * myPos;
    let tLen = length(tRaw);
    if (tLen <= 1e-6) {
        return;
    }
    let dToNb = tRaw / tLen;
    let wDist = 1.0 - r / max(bio.catRadius, 1e-4);

    if (nbPhase == 0u) {
        let nbKind = bioRoleKind(nbAtomicNum, nbValence);
        if (nbKind == 4u) {
            (*acc).nearHalogen += wDist;
            if (myPhase == 1u && (unpackBioSpecies(myLifeState.w) % 4u) == 2u) {
                (*acc).bioSteer += dToNb * (wDist * 0.55);
            }
        } else if (myPhase == 1u && isBiogenicElement(nbAtomicNum, nbValence) && (unpackBioSpecies(myLifeState.w) % 4u) == 0u) {
            (*acc).bioSteer += dToNb * (wDist * 0.20);
        }

        if (myPhase == 1u) {
            let myBioSp = unpackBioSpecies(myLifeState.w);
            let eff = evalLifeFromAtomPair(myBioSp, myLifeState, nbAtomicNum, nbValence, nbAtomLoad, r, bio);
            (*acc).dE += eff.x;
            (*acc).dVit += eff.y;
            (*acc).dPoi += eff.z;
            (*acc).steer += dToNb * eff.w;
            (*acc).bioChemCount += 1.0;
        }
        return;
    }

    // Neighbor is living Biomass (`nbPhase == 1u`)
    let nbGuild = nbBioSpecies % 4u;
    let nbEnergy = clamp(nbLifeState.x, 0.0, 1.6);
    let nbPoi = clamp(nbLifeState.z, 0.0, 1.0);

    (*acc).nearTotalBio += 1.0;
    if (nbEnergy < 0.44 || nbPoi > 0.24) {
        (*acc).nearStressedBio += wDist * 1.15;
    }

    if (nbGuild == 0u) {
        (*acc).nearProd += wDist;
        if (nbEnergy > 0.50) {
            (*acc).nearEnergeticProd += wDist;
        }
        if (wDist > (*acc).bestProdW) {
            (*acc).bestProdW = wDist;
            (*acc).bestProdSp = nbBioSpecies;
        }
        if (myPhase == 1u && (unpackBioSpecies(myLifeState.w) % 4u) == 1u) {
            (*acc).bioSteer += dToNb * (wDist * 0.65);
        }
    } else if (nbGuild == 1u) {
        (*acc).nearCons += wDist;
        if (nbEnergy > 0.24) {
            (*acc).nearActiveCons += wDist;
            (*acc).consRepelDir += dToNb * wDist;
        }
        if (wDist > (*acc).bestConsW) {
            (*acc).bestConsW = wDist;
            (*acc).bestConsSp = nbBioSpecies;
        }
    } else if (nbGuild == 2u) {
        (*acc).nearHalo += wDist;
        if (nbEnergy > 0.52) {
            (*acc).nearEnergeticHalo += wDist;
        }
        if (wDist > (*acc).bestHaloW) {
            (*acc).bestHaloW = wDist;
            (*acc).bestHaloSp = nbBioSpecies;
        }
    } else {
        (*acc).nearDecomp += wDist;
        if (wDist > (*acc).bestDecompW) {
            (*acc).bestDecompW = wDist;
            (*acc).bestDecompSp = nbBioSpecies;
        }
        if (myPhase == 1u && (unpackBioSpecies(myLifeState.w) % 4u) == 1u) {
            (*acc).bioSteer += dToNb * (wDist * 0.45);
        }
    }

    if (myPhase == 1u && (unpackBioSpecies(myLifeState.w) % 4u) == 3u && (nbEnergy < 0.44 || nbPoi > 0.24)) {
        (*acc).bioSteer += dToNb * (wDist * 0.60);
    }

    if (myPhase == 0u) {
        let tSwirl = cross(myPos, dToNb);
        let eff = evalAtomFromLifePair(myAtomicNum, myValence, myAtomLoad, nbBioSpecies, nbLifeState, r, bio);
        (*acc).kick += dToNb * eff.x + tSwirl * eff.y;
        (*acc).actScale = min((*acc).actScale, eff.z);
        (*acc).glowSum += eff.w;
        (*acc).chemBioCount += 1.0;
    }
}

fn finalizeChemBioForce(acc: BioUnifiedAccum, bio: BioLink) -> vec4f {
    if (acc.chemBioCount <= 0.0) {
        return vec4f(0.0);
    }
    let norm = 1.0 / (1.0 + 0.25 * max(acc.chemBioCount - 1.0, 0.0));
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

fn finalizeBioUnified(
    pi: u32,
    pos: vec3f,
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    curLifeState: vec4f,
    oc: vec4f,
    acc: BioUnifiedAccum,
    bio: BioLink,
) -> BioStepResult {
    let myPhase = unpackPhase(curLifeState.w);
    if (bio.enabled <= 0.5) {
        if (myPhase == 1u) {
            return BioStepResult(vec4f(0.65, 0.0, 0.0, curLifeState.w), vec4f(0.0));
        }
        return BioStepResult(vec4f(0.0), vec4f(0.0));
    }

    let rate = max(bio.rate, 0.0);
    let dtBio = bio.dt * rate;
    let solar = evalSolarInsolation(pos);
    let upwell = evalCoastalUpwelling(oc);
    let inOcean = oc.w >= 0.42;
    let frameU = u32(max(bio.frame, 0.0)) + u32(max(bio.seedHash, 0.0));
    let nBio = max(u32(bio.bioColours), 1u);

    if (myPhase == 0u) {
        let chemForce = finalizeChemBioForce(acc, bio);
        if (!isBiogenicElement(atomicNum, valence) || !inOcean) {
            return BioStepResult(vec4f(0.0), chemForce);
        }

        let crowdHeadroom = clamp(1.0 - acc.nearTotalBio / 7.5, 0.0, 1.0);
        let bondEase = 1.0 / (1.0 + 0.55 * max(atomLoad, 0.0));

        // 1. Producer autotrophic solar/upwelling carbon & nutrient fixation (phase 0 -> 1)
        if (acc.nearEnergeticProd > 0.0 && crowdHeadroom > 0.01) {
            let pFix = clamp(
                0.28 * solar * (0.55 + 0.85 * upwell) * min(acc.nearEnergeticProd, 3.0) * crowdHeadroom * bondEase * dtBio,
                0.0,
                0.065,
            );
            if (hashFloat(pi, frameU, 10u) < pFix) {
                let targetSp = min(acc.bestProdSp, nBio - 1u);
                return BioStepResult(
                    vec4f(0.62, 0.08, 0.0, packPhaseBio(1u, targetSp)),
                    vec4f(0.0, 0.0, 0.0, 0.9),
                );
            }
        }

        // 2. Halophile biomass fixation in halogen refuges
        if (acc.nearEnergeticHalo > 0.0 && acc.nearHalogen > 0.25 && crowdHeadroom > 0.05) {
            let pHaloFix = clamp(
                0.18 * min(acc.nearEnergeticHalo, 2.5) * min(acc.nearHalogen, 3.0) * crowdHeadroom * bondEase * dtBio,
                0.0,
                0.05,
            );
            if (hashFloat(pi, frameU, 11u) < pHaloFix) {
                let targetSp = min(acc.bestHaloSp, nBio - 1u);
                return BioStepResult(
                    vec4f(0.60, 0.05, 0.0, packPhaseBio(1u, targetSp)),
                    vec4f(0.0, 0.0, 0.0, 0.8),
                );
            }
        }

        // 3. Low-rate abiogenesis / spore floor so no trophic guild permanently goes extinct
        if (acc.nearProd < 0.05 && acc.nearTotalBio < 2.0 && oc.w >= 0.46) {
            let pSporeP = clamp(0.0024 * solar * (0.45 + upwell) * dtBio, 0.0, 0.004);
            if (hashFloat(pi, frameU, 12u) < pSporeP) {
                let sp = pickStrainForGuild(0u, nBio, hashFloat(pi, frameU, 16u));
                return BioStepResult(vec4f(0.62, 0.05, 0.0, packPhaseBio(1u, sp)), vec4f(0.0, 0.0, 0.0, 0.6));
            }
        }
        if (acc.nearHalo < 0.05 && acc.nearHalogen >= 0.45 && oc.w >= 0.44) {
            let pSporeH = clamp(0.0045 * dtBio, 0.0, 0.006);
            if (hashFloat(pi, frameU, 13u) < pSporeH) {
                let sp = pickStrainForGuild(2u, nBio, hashFloat(pi, frameU, 17u));
                return BioStepResult(vec4f(0.62, 0.05, 0.0, packPhaseBio(1u, sp)), vec4f(0.0, 0.0, 0.0, 0.6));
            }
        }
        if (acc.nearCons < 0.05 && (acc.nearProd + acc.nearDecomp) >= 1.8 && oc.w >= 0.46) {
            let pSporeC = clamp(0.0040 * dtBio, 0.0, 0.006);
            if (hashFloat(pi, frameU, 14u) < pSporeC) {
                let sp = pickStrainForGuild(1u, nBio, hashFloat(pi, frameU, 18u));
                return BioStepResult(vec4f(0.64, 0.04, 0.0, packPhaseBio(1u, sp)), vec4f(0.0, 0.0, 0.0, 0.6));
            }
        }
        if (acc.nearDecomp < 0.05 && (acc.nearStressedBio >= 0.5 || acc.nearTotalBio >= 4.0) && oc.w >= 0.44) {
            let pSporeD = clamp(0.0055 * dtBio, 0.0, 0.008);
            if (hashFloat(pi, frameU, 15u) < pSporeD) {
                let sp = pickStrainForGuild(3u, nBio, hashFloat(pi, frameU, 19u));
                return BioStepResult(vec4f(0.60, 0.04, 0.0, packPhaseBio(1u, sp)), vec4f(0.0, 0.0, 0.0, 0.6));
            }
        }

        return BioStepResult(vec4f(0.0), chemForce);
    }

    // Current particle is living Biomass (`myPhase == 1u`)
    let myBioSp = min(unpackBioSpecies(curLifeState.w), nBio - 1u);
    let guild = myBioSp % 4u;
    let normChem = select(0.0, 1.0 / (1.0 + 0.35 * max(acc.bioChemCount - 1.0, 0.0)), acc.bioChemCount > 0.0);

    let dtVit = dtBio * 1.8;
    let vit = clamp(curLifeState.y + (acc.dVit * normChem - 0.16 * rate * curLifeState.y) * dtVit, 0.0, 1.0);
    let detoxRate = select(0.24 + 0.45 * vit, 0.55 + 0.50 * vit, guild == 2u);
    let poiIn = select(acc.dPoi * normChem, 0.0, guild == 2u);
    let poi = clamp(curLifeState.z + (poiIn - detoxRate * curLifeState.z) * dtVit, 0.0, 1.0);

    let oceanFactor = select(0.0, 1.0, inOcean);
    var netDE = 0.0;
    if (guild == 0u) {
        // Producer: gains energy from latitude solar radiation & coastal upwelling
        let growth = 0.36 * solar * (0.55 + 0.85 * upwell) * oceanFactor;
        let chemBonus = 0.05 * acc.dE * normChem;
        let crowd = 0.055 * max(0.0, acc.nearTotalBio - 5.5);
        let landPenalty = select(0.35, 0.0, oc.w >= 0.38);
        netDE = growth + 0.08 * vit + chemBonus - 0.16 - crowd - 0.25 * poi - landPenalty;
    } else if (guild == 1u) {
        // Consumer: thrives on nearby Producer & Decomposer prey + local nutrient fields
        let grazeEnv = 0.34 * min(acc.nearProd + 0.80 * acc.nearDecomp, 2.5);
        let chemBonus = 0.12 * acc.dE * normChem;
        let crowd = 0.040 * max(0.0, acc.nearCons - 4.0);
        let landPenalty = select(0.35, 0.0, oc.w >= 0.38);
        netDE = grazeEnv + chemBonus - 0.24 - crowd - 0.30 * poi - landPenalty;
    } else if (guild == 2u) {
        // Halophile: thrives inside halogen (F/Cl) refuges & upwelling margins
        let haloEnv = 0.38 * min(acc.nearHalogen, 2.5) + 0.14 * upwell * oceanFactor;
        let chemBonus = 0.10 * acc.dE * normChem;
        let crowd = 0.040 * max(0.0, acc.nearHalo - 5.0);
        let landPenalty = select(0.35, 0.0, oc.w >= 0.38);
        netDE = haloEnv + chemBonus - 0.21 - crowd - landPenalty;
    } else {
        // Decomposer: harvests energy from stressed / dense biomass & organic chemistry
        let detEnv = 0.36 * min(acc.nearStressedBio + 0.35 * max(0.0, acc.nearTotalBio - 4.0), 2.5);
        let chemBonus = 0.10 * acc.dE * normChem;
        let crowd = 0.050 * max(0.0, acc.nearDecomp - 4.5);
        let landPenalty = select(0.35, 0.0, oc.w >= 0.38);
        netDE = detEnv + chemBonus - 0.19 - crowd - 0.14 * poi - landPenalty;
    }

    let nextEnergy = clamp(curLifeState.x + netDE * (dtBio * 1.8), 0.04, 1.55);
    let flux = clamp(netDE, -1.0, 1.0);

    // Consumer grazing on Producers & Decomposers (~65% assimilated into Consumer, ~35% respired back to Chemistry)
    if ((guild == 0u || guild == 3u) && acc.nearActiveCons > 0.0) {
        let haloShield = 1.0 / (1.0 + 1.4 * acc.nearHalogen);
        let pGraze = clamp(
            0.42 * min(acc.nearActiveCons, 3.0) * haloShield * (1.0 - 0.30 * vit) * dtBio,
            0.0,
            0.085,
        );
        if (hashFloat(pi, frameU, 20u) < pGraze) {
            let outcome = hashFloat(pi, frameU, 21u);
            if (outcome < 0.65) {
                let consSp = min(acc.bestConsSp, nBio - 1u);
                return BioStepResult(
                    vec4f(0.74, vit * 0.8, poi * 0.6, packPhaseBio(1u, consSp)),
                    vec4f(0.0, 0.0, 0.0, 0.75),
                );
            } else {
                let repelLen = length(acc.consRepelDir);
                let expel = select(vec3f(0.0), -(acc.consRepelDir / repelLen) * 1.4, repelLen > 1e-4);
                return BioStepResult(
                    vec4f(0.0),
                    vec4f(expel, packDeltaActGlow(0.8, -0.9)),
                );
            }
        }
    }

    // Decomposer mineralization (~80% recycled to Chemistry restoring original chemSpecies, ~20% converted to Decomposer)
    let stress = clamp(
        smoothstep(0.48, 0.14, nextEnergy) * 0.95 +
        smoothstep(0.22, 0.75, poi) * 0.80 +
        smoothstep(5.0, 12.0, acc.nearTotalBio) * 0.55 +
        smoothstep(0.44, 0.28, oc.w) * 1.0,
        0.0,
        1.5,
    );
    if (acc.nearDecomp > 0.0 && stress > 0.12 && (guild != 3u || stress > 0.72)) {
        let pDecomp = clamp(0.55 * stress * min(acc.nearDecomp, 3.0) * dtBio, 0.0, 0.105);
        if (hashFloat(pi, frameU, 22u) < pDecomp) {
            let decRoll = hashFloat(pi, frameU, 23u);
            if (decRoll < 0.80 || guild == 3u || !inOcean) {
                return BioStepResult(
                    vec4f(0.0),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.85, -0.85)),
                );
            } else {
                let decSp = min(acc.bestDecompSp, nBio - 1u);
                return BioStepResult(
                    vec4f(0.64, vit * 0.6, poi * 0.5, packPhaseBio(1u, decSp)),
                    vec4f(0.0, 0.0, 0.0, 0.55),
                );
            }
        }
    }

    // Natural senescence/stranding turnover back to Chemistry (phase 1 -> 0) even if isolated
    if (nextEnergy <= 0.075 || poi >= 0.92 || oc.w < 0.30) {
        let pDecay = clamp(0.45 * dtBio, 0.0, 0.14);
        if (hashFloat(pi, frameU, 24u) < pDecay) {
            return BioStepResult(
                vec4f(0.0),
                vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.9, -0.7)),
            );
        }
    }

    // Stay in Biomass phase and compute live chemotaxis/foraging steer vector
    var st = acc.steer * (normChem * 2.4) + acc.bioSteer * max(rate, 0.2);
    let stLen = length(st);
    let maxSt = 5.0 * max(rate, 0.1);
    if (stLen > maxSt) {
        st *= maxSt / stLen;
    }

    return BioStepResult(
        vec4f(nextEnergy, vit, poi, packPhaseBio(1u, myBioSp)),
        vec4f(st, flux),
    );
}
