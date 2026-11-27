// Unified conserved-mass emergent ecology + amphiphilic lipid bilayers + inorganic mineral crust.
// Every particle carries an immutable `chemSpecies` (`p.colour`), live `phase`
// (0 = Fluid Chemistry, 1 = Living Cytoplasm, 2 = Biogenic Lipid Bilayer, 3 = Crust-Pinned Mineral),
// `bioSpecies` (0..M-1) / `mineralType` (0..4), circular `kinPhase` / `crystalAngle`, and 4-bit `gene` / `grade` in `lifeState.w`.

const MINERAL_SILICATE: u32 = 0u;
const MINERAL_EVAPORITE: u32 = 1u;
const MINERAL_CARBONATE: u32 = 2u;
const MINERAL_SULFIDE: u32 = 3u;
const MINERAL_CLAY: u32 = 4u;

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
    oceanCurrentScale: f32,
    windScale: f32,
    windStreamMax: f32,
    mineralEnabled: f32,
    mineralRate: f32,
    localTempScale: f32,
    localUvScale: f32,
    localHydroScale: f32,
    pad2: f32,
}

struct BioStepResult {
    nextLifeState: vec4f,
    bioForce: vec4f,
    nextMdpState: MdpState,
    nextBlob: vec4u,
}

fn bioStepOnly(nextLifeState: vec4f, bioForce: vec4f) -> BioStepResult {
    return BioStepResult(nextLifeState, bioForce, emptyMdpState(), vec4u(NO_LEADER, 99u, 0u, 0u));
}

fn isBiogenicElement(atomicNum: u32, valence: u32) -> bool {
    if (valence == 0u) {
        return false;
    }
    return atomicNum == 1u || atomicNum == 5u || atomicNum == 6u || atomicNum == 12u;
}

fn isLipidElement(atomicNum: u32, valence: u32) -> bool {
    if (valence == 0u) {
        return false;
    }
    return atomicNum == 1u || atomicNum == 5u || atomicNum == 6u;
}

fn isMineralCandidate(atomicNum: u32, valence: u32) -> bool {
    return valence > 0u && atomicNum >= 1u;
}

fn elementFitsMineralType(atomicNum: u32, valence: u32, mType: u32) -> bool {
    if (valence == 0u) {
        return false;
    }
    if (mType == MINERAL_SILICATE) {
        return atomicNum == 11u || atomicNum == 6u || atomicNum == 10u || atomicNum == 3u;
    }
    if (mType == MINERAL_EVAPORITE) {
        return atomicNum == 9u || atomicNum == 3u || atomicNum == 13u || atomicNum == 7u || atomicNum == 10u;
    }
    if (mType == MINERAL_CARBONATE) {
        return atomicNum == 5u || atomicNum == 6u || atomicNum == 10u || atomicNum == 4u;
    }
    if (mType == MINERAL_SULFIDE) {
        return atomicNum == 12u || atomicNum == 10u || atomicNum == 6u || atomicNum == 4u || atomicNum == 7u;
    }
    return atomicNum == 11u || atomicNum == 6u || atomicNum == 1u || atomicNum == 10u || atomicNum == 5u;
}

fn evalWindVel3D(pos: vec3f, windRaw: vec3f, bio: BioLink) -> vec3f {
    if (bio.windScale <= 0.0) {
        return vec3f(0.0);
    }
    let eEast = sphereEastBasis(pos);
    let eNorth = normalize(cross(pos, eEast));
    let w3 = windRaw.x * eEast + windRaw.y * eNorth;
    return (w3 / max(bio.windStreamMax, 10.0)) * (0.45 * bio.windScale);
}

fn mutateGene(baseGene: u32, pi: u32, frameU: u32, salt: u32, pMut: f32) -> u32 {
    let g = min(baseGene, 15u);
    let r = hashFloat(pi, frameU, salt);
    if (r >= pMut) {
        return g;
    }
    if (r < pMut * 0.5) {
        return select(0u, g - 1u, g > 0u);
    }
    return min(g + 1u, 15u);
}

fn bioRoleKind(atomicNum: u32, valence: u32) -> u32 {
    if (valence == 0u) {
        return 0u; // Inert noble gas (He, Ne, Ar)
    }
    if (atomicNum == 7u || atomicNum == 13u) {
        return 4u; // Halogen toxin (F, Cl)
    }
    if (atomicNum == 3u || atomicNum == 4u || atomicNum == 10u) {
        return 3u; // Cofactor (Li, Be, Mg)
    }
    if (atomicNum == 5u) {
        return 2u; // Carbon food backbone (C)
    }
    if (atomicNum == 1u || atomicNum == 12u) {
        return 1u; // Fuel hydride / sulfur (H, S)
    }
    return 5u; // Oxide / silicate / alkali
}

fn elemSolarWeight(atomicNum: u32) -> f32 {
    if (atomicNum == 5u) { return 1.0; }
    if (atomicNum == 6u) { return 0.80; }
    if (atomicNum == 1u) { return 0.25; }
    if (atomicNum == 12u) { return 0.15; }
    return 0.0;
}

fn elemRedoxWeight(atomicNum: u32) -> f32 {
    if (atomicNum == 12u) { return 1.0; }
    if (atomicNum == 1u) { return 0.85; }
    if (atomicNum == 5u) { return 0.25; }
    if (atomicNum == 6u) { return 0.10; }
    return 0.0;
}

fn evalUvSolarAndStress(uvSample: vec4f, localUvScale: f32) -> vec2f {
    let sc = clamp(localUvScale, 0.0, 2.0);
    if (sc <= 0.001) {
        return vec2f(0.68, 0.0);
    }
    let uviNow = clamp(uvSample.x, 0.0, 24.0);
    let blend = min(sc, 1.5);
    let normU = uviNow / 3.8;
    let solarRaw = select(0.0, clamp((normU / (normU + 0.55)) * 1.18, 0.0, 1.25), uviNow > 0.001);
    let solar = solarRaw * blend;
    let uvStress = smoothstep(5.5, 15.5, uviNow) * blend;
    return vec2f(solar, uvStress);
}

fn evalCoastalUpwelling(oc: vec4f) -> f32 {
    let shelf = smoothstep(0.38, 0.58, oc.w) * (1.0 - 0.65 * smoothstep(0.76, 0.99, oc.w));
    return clamp(shelf + clamp(oc.z * 0.45, 0.0, 0.35), 0.0, 1.35);
}

fn evalOceanCurrentVel(pos: vec3f, oc: vec4f, currentScale: f32) -> vec3f {
    if (currentScale <= 0.0 || oc.w < 0.35) {
        return vec3f(0.0);
    }
    let eEast = sphereEastBasis(pos);
    let eNorth = normalize(cross(pos, eEast));
    let cur3D = oc.x * eEast + oc.y * eNorth;
    var cTarget = (cur3D / 0.7) * (0.30 * currentScale) * smoothstep(0.35, 0.55, oc.w);
    let cLen = length(cTarget);
    if (cLen > 1.15) {
        cTarget *= 1.15 / cLen;
    }
    return cTarget;
}

fn kinAngularDist(a: f32, b: f32) -> f32 {
    return 0.5 * (1.0 - cos(a - b));
}

fn getBioMatrixA(nChem: u32, nBio: u32, rowSp: u32, colSp: u32) -> f32 {
    let oceanBase = nChem * SPECIES_STRIDE + nChem * nChem;
    let rSp = min(rowSp, max(nBio, 1u) - 1u);
    let cSp = min(colSp, max(nBio, 1u) - 1u);
    return matrix[oceanBase + rSp * max(nBio, 1u) + cSp];
}

fn getBioSpeciesTraits(nChem: u32, nBio: u32, sp: u32) -> vec4f {
    let oceanBase = nChem * SPECIES_STRIDE + nChem * nChem;
    let traitBase = oceanBase + max(nBio, 1u) * max(nBio, 1u);
    let idx = traitBase + min(sp, max(nBio, 1u) - 1u) * 4u;
    return vec4f(matrix[idx + 0u], matrix[idx + 1u], matrix[idx + 2u], matrix[idx + 3u]);
}

fn getSpeciesMdpPriorPack(nChem: u32, nBio: u32, sp: u32) -> vec4u {
    let oceanBase = nChem * SPECIES_STRIDE + nChem * nChem;
    let bioN = max(nBio, 1u);
    let priorBase = oceanBase + bioN * bioN + bioN * 4u + min(sp, bioN - 1u) * MDP_SPECIES_STRIDE;
    var qPack = vec4u(0u);
    for (var s = 0u; s < 4u; s++) {
        let rIdx = priorBase + s * 4u;
        let row = quantizeMdpQ(matrix[rIdx + 0u])
            | (quantizeMdpQ(matrix[rIdx + 1u]) << 8u)
            | (quantizeMdpQ(matrix[rIdx + 2u]) << 16u)
            | (quantizeMdpQ(matrix[rIdx + 3u]) << 24u);
        qPack = setQRow(qPack, s, row);
    }
    return qPack;
}

fn pickCompatibleNovelSpecies(
    nChem: u32,
    nBio: u32,
    hostSp: u32,
    spMask: u32,
    pi: u32,
    frameU: u32,
    salt: u32,
) -> u32 {
    if (nBio <= 1u) {
        return hostSp;
    }
    var fallbackSp = hostSp;
    var hasFallback = false;
    for (var attempt: u32 = 0u; attempt < 4u; attempt++) {
        let candSp = u32(floor(hashFloat(pi, frameU, salt + attempt) * f32(nBio))) % nBio;
        if (candSp == hostSp) {
            continue;
        }
        let aHC = getBioMatrixA(nChem, nBio, hostSp, candSp);
        let aCH = getBioMatrixA(nChem, nBio, candSp, hostSp);
        let compat = (aHC > 0.08) || (abs(aHC - aCH) > 0.35 && (aHC + aCH) > -0.25);
        if (!compat) {
            continue;
        }
        let isNovel = (spMask & (1u << (candSp % 24u))) == 0u;
        if (isNovel) {
            return candSp;
        }
        if (!hasFallback) {
            fallbackSp = candSp;
            hasFallback = true;
        }
    }
    return fallbackSp;
}

// Pure contact biochemistry: `(dEnergy, dVitamin, dPoison)` from close-contact Chemistry atom (`r < 0.62 * catRadius`).
// Zero explicit steering vectors: motion comes solely from the combined attractor matrix and physical levers.
fn evalLifeFromAtomPair(
    lifeState: vec4f,
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    r: f32,
    bio: BioLink,
) -> vec3f {
    let contactR = bio.catRadius * 0.62;
    let kind = bioRoleKind(atomicNum, valence);
    if (kind == 0u || r >= contactR) {
        return vec3f(0.0);
    }

    let w = (1.0 - r / max(contactR, 1e-4)) * bio.rate;
    let energy = clamp(lifeState.x, 0.0, 1.5);
    let vit = clamp(lifeState.y, 0.0, 1.0);

    var dE = 0.0;
    var dVit = 0.0;
    var dPoi = 0.0;

    if (kind == 3u) {
        dVit += 0.28 * w;
    } else if (kind == 4u) {
        dPoi += 0.34 * w * (1.0 - 0.55 * vit);
        dE -= 0.10 * w * (1.0 - 0.35 * vit);
    } else if (kind == 1u) {
        let fuelReady = clamp(atomLoad + 0.35, 0.35, 1.5);
        if (energy < 1.45) {
            dE += (0.20 + 0.10 * vit) * fuelReady * w;
            dVit += 0.08 * w;
        }
    } else if (kind == 2u) {
        let foodRich = clamp(atomLoad + 0.25, 0.25, 2.2);
        dE += (0.14 + 0.10 * vit) * foodRich * w;
        dVit += 0.06 * w;
    } else if (atomicNum == 6u) {
        if (atomLoad > 0.05 && atomLoad < 1.35) {
            dE += 0.09 * w;
        } else if (energy > 0.40 && atomLoad >= 1.35) {
            dE -= 0.04 * w;
        }
    }

    return vec3f(dE, dVit, dPoi);
}

// Local catalysis modulation `(actScale, glow)` on nearby Chemistry atoms without scripted propulsion vectors.
fn evalAtomFromLifePair(
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    nbLifeState: vec4f,
    r: f32,
    bio: BioLink,
) -> vec2f {
    let contactR = bio.catRadius * 0.72;
    let kind = bioRoleKind(atomicNum, valence);
    if (kind == 0u || r >= contactR) {
        return vec2f(1.0, 0.0);
    }

    let w = (1.0 - r / max(contactR, 1e-4)) * bio.rate;
    let energy = clamp(nbLifeState.x, 0.0, 1.5);
    var actScale = 1.0;
    var glow = 0.0;

    if (kind == 3u) {
        actScale = 0.45;
        glow = 0.35 * w;
    } else if (kind == 4u && energy > 0.28) {
        actScale = 0.25;
        glow = -0.50 * w;
    } else if (kind == 1u || kind == 2u) {
        if (energy > 0.30) {
            actScale = 0.18;
            glow = 0.65 * w;
        }
    } else if (atomicNum == 6u && atomLoad > 0.05 && atomLoad < 1.35) {
        actScale = 0.28;
        glow = 0.42 * w;
    }

    return vec2f(actScale, glow);
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
    bioChemCount: f32,
    nearHalogen: f32,
    nearTotalBio: f32,
    nearForeignBio: f32,
    nearStressedBio: f32,
    // Tight contact assimilation & lipid synthesis (phase == 0u)
    contactCount: f32,
    contactEnergySum: f32,
    contactLipidAllocSum: f32,
    contactSpeciesMask: u32,
    contactBestW: f32,
    contactBestSp: u32,
    contactBestKin: f32,
    contactBestGene: u32,
    contactBestGrad: u32,
    contactKinCos: f32,
    contactKinSin: f32,
    contactKinW: f32,
    blobCentroid: vec3f,
    blobCentW: f32,
    lipidCentroid: vec3f,
    lipidCentW: f32,
    nearLipidCount: f32,
    nearLipidEnergySum: f32,
    bestLipidW: f32,
    bestLipidSp: u32,
    bestLipidKin: f32,
    bestLipidGene: u32,
    // Same-blob organism anatomy, heterogeneity, lipids, & gene sharing
    blobCount: f32,
    blobWeightSum: f32,
    blobEnergySum: f32,
    blobVitSum: f32,
    blobKinCos: f32,
    blobKinSin: f32,
    blobKinWeight: f32,
    blobSpeciesMask: u32,
    blobLipidCount: f32,
    blobLipidGradCount: f32,
    blobGradCount: f32,
    blobGeneSum: f32,
    // Pure physical close-contact engulfment & predation harvest (`r < 0.44 * catRadius`)
    contactPreyHarvest: f32,
    contactEngulfPress: f32,
    contactBestEngulfW: f32,
    contactBestEngulfSp: u32,
    contactBestEngulfKin: f32,
    contactBestEngulfGene: u32,
    contactBestEngulfGrad: u32,
    contactBestEngulfE: f32,
    contactBestEngulfAlloc: f32,
    // Inorganic mineral crust accumulation & exchange
    nearMineralCount: f32,
    nearMinDist: f32,
    nearMineralBioMine: f32,
    nearSeedW: f32,
    nearSeedType: u32,
    nearSeedTrap: u32,
    nearSeedAngle: f32,
    nearClayCat: f32,
    nearPreSilica: f32,
    nearPreHalide: f32,
    nearPreCarb: f32,
    nearPreSulf: f32,
    nearBe: f32,
    nearClayH: f32,
    // Water-cycle & O2/CO2 exchange
    nearWaterFluid: f32,
    nearOxygenGas: f32,
    nearCarbonFeed: f32,
    // Option-2 bounded Bellman-Ford leader election & Particle MDP lateral Q-gossip
    bestLeaderId: u32,
    bestLeaderHops: u32,
    mdpVotes: vec4f,
    mdpVoteWeight: f32,
    bestMdpScore: f32,
    bestMdpQPack: vec4u,
}

fn initBioUnifiedAccum() -> BioUnifiedAccum {
    var acc = BioUnifiedAccum();
    acc.actScale = 1.0;
    acc.bestMdpScore = -1.0;
    acc.nearMinDist = 99.0;
    acc.contactBestGene = 7u;
    acc.bestLipidGene = 7u;
    acc.contactBestEngulfGene = 7u;
    acc.bestLeaderId = NO_LEADER;
    acc.bestLeaderHops = 99u;
    return acc;
}

fn considerLeaderCandidate(
    acc: ptr<function, BioUnifiedAccum>,
    nbIdx: u32,
    nbBlob: vec4u,
) {
    let rawId = select(nbIdx, nbBlob.x, nbBlob.x != NO_LEADER);
    let rawHops = select(0u, nbBlob.y, nbBlob.x != NO_LEADER);
    if (rawHops >= MAX_LEADER_HOPS) {
        return;
    }
    let candHops = rawHops + 1u;
    if (rawId < (*acc).bestLeaderId || (rawId == (*acc).bestLeaderId && candHops < (*acc).bestLeaderHops)) {
        (*acc).bestLeaderId = rawId;
        (*acc).bestLeaderHops = candHops;
    }
}

fn accumulateBioUnifiedNeighbor(
    myPos: vec3f,
    myVel: vec3f,
    myPhase: u32,
    myAtomicNum: u32,
    myValence: u32,
    myAtomLoad: f32,
    myLifeState: vec4f,
    myBlobZ: u32,
    nbIdx: u32,
    nbPos: vec3f,
    nbVel: vec3f,
    nbChemSpecies: f32,
    nbPhase: u32,
    nbBioSpecies: u32,
    nbLifeState: vec4f,
    nbAtomLoad: f32,
    nbAtomicNum: u32,
    nbValence: u32,
    nbBlob: vec4u,
    nbMdp: MdpState,
    oc: vec4f,
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

    let nChem = max(u32(bio.chemColours), 1u);
    let nBio = max(u32(bio.bioColours), 1u);
    let myBioSp = min(unpackBioSpecies(myLifeState.w), nBio - 1u);
    let nbSp = min(nbBioSpecies, nBio - 1u);

    if (nbPhase == 0u) {
        let nbKind = bioRoleKind(nbAtomicNum, nbValence);
        if (nbKind == 4u) {
            (*acc).nearHalogen += wDist;
        }
        if (nbAtomicNum == 1u || nbAtomicNum == 6u) {
            (*acc).nearWaterFluid += wDist * select(0.35, 1.0, nbLifeState.y >= 0.5);
        }
        if (nbAtomicNum == 6u && nbLifeState.y < 0.5) {
            (*acc).nearOxygenGas += wDist;
        }
        if (nbAtomicNum == 5u) {
            (*acc).nearCarbonFeed += wDist;
        }
        if (myPhase == 0u) {
            if (nbAtomicNum == 11u || nbAtomicNum == 6u) {
                (*acc).nearPreSilica += wDist * select(0.55, 1.35, nbAtomicNum == 11u);
            }
            if (nbAtomicNum == 9u || nbAtomicNum == 3u || nbAtomicNum == 13u || nbAtomicNum == 7u) {
                (*acc).nearPreHalide += wDist * select(0.8, 1.25, nbAtomicNum == 13u || nbAtomicNum == 7u);
            }
            if (nbAtomicNum == 5u || nbAtomicNum == 10u || nbAtomicNum == 4u) {
                (*acc).nearPreCarb += wDist * select(0.85, 1.25, nbAtomicNum == 5u || nbAtomicNum == 10u);
                if (nbAtomicNum == 4u) {
                    (*acc).nearBe += wDist;
                }
            }
            if (nbAtomicNum == 12u || nbAtomicNum == 10u) {
                (*acc).nearPreSulf += wDist * select(0.35, 1.5, nbAtomicNum == 12u);
            }
            if (nbAtomicNum == 1u) {
                (*acc).nearClayH += wDist;
            }
        } else if (myPhase == 1u) {
            let eff = evalLifeFromAtomPair(myLifeState, nbAtomicNum, nbValence, nbAtomLoad, r, bio);
            (*acc).dE += eff.x;
            (*acc).dVit += eff.y;
            (*acc).dPoi += eff.z;
            (*acc).bioChemCount += 1.0;
        } else if (myPhase == 2u) {
            if (nbKind == 4u && r < bio.catRadius * 0.62) {
                (*acc).dPoi += 0.28 * wDist * bio.rate;
            }
        }
        return;
    }

    // Neighbor is anchored crustal mineral (`nbPhase == 3u`)
    if (nbPhase == 3u) {
        let nbMinType = unpackBioSpecies(nbLifeState.w);
        let nbMinAngle = unpackKinPhase(nbLifeState.w);
        let nbMinTrap = unpackGraduated(nbLifeState.w);
        let nbInteg = clamp(nbLifeState.x, 0.05, 1.25);
        let nbNutr = clamp(nbLifeState.y, 0.0, 1.0);
        let nbHaz = clamp(nbLifeState.z, 0.0, 1.0);

        if (myPhase == 3u) {
            (*acc).nearMineralCount += 1.0;
            return;
        }
        if (myPhase == 0u) {
            (*acc).nearMineralCount += 1.0;
            (*acc).nearMinDist = min((*acc).nearMinDist, r);
            if (wDist > (*acc).nearSeedW && elementFitsMineralType(myAtomicNum, myValence, nbMinType)) {
                (*acc).nearSeedW = wDist;
                (*acc).nearSeedType = nbMinType;
                (*acc).nearSeedAngle = nbMinAngle;
                (*acc).nearSeedTrap = nbMinTrap;
            }
            if (nbMinType == MINERAL_CLAY) {
                (*acc).nearClayCat += wDist * clamp(nbInteg, 0.25, 1.2);
                (*acc).glowSum += 0.35 * wDist;
            } else if (nbMinType == MINERAL_CARBONATE && nbMinTrap == 0u) {
                (*acc).nearClayCat += 0.45 * wDist * clamp(nbInteg, 0.25, 1.2);
            }
            let minLatticeR = bio.catRadius * 0.34;
            if (elementFitsMineralType(myAtomicNum, myValence, nbMinType) && nbInteg > 0.25) {
                if (r < minLatticeR) {
                    (*acc).kick -= dToNb * (0.28 * (1.0 - r / max(minLatticeR, 1e-4)));
                } else {
                    (*acc).kick += dToNb * (wDist * 0.14);
                }
                (*acc).chemBioCount += 1.0;
            }
            return;
        }

        // Physical surface catalysis/weathering strictly on contact (`r < 0.68 * catRadius`, NO steering vectors!)
        if (r >= bio.catRadius * 0.68) {
            return;
        }
        let minScale = max(bio.mineralRate, 0.20) * clamp(0.35 + 0.85 * nbInteg, 0.25, 1.2);
        let mw = (1.0 - r / (bio.catRadius * 0.68)) * minScale;
        (*acc).bioChemCount += 1.0;

        if (myPhase == 2u) {
            if (nbMinTrap == 1u || nbMinType == MINERAL_EVAPORITE) {
                (*acc).dPoi += (0.16 + 0.22 * nbHaz) * mw;
            }
            return;
        }

        let vit = clamp(myLifeState.y, 0.0, 1.0);
        let myRedox = elemRedoxWeight(myAtomicNum);
        if (nbMinType == MINERAL_SILICATE) {
            (*acc).dPoi -= 0.09 * mw;
            (*acc).dVit += 0.05 * mw;
        } else if (nbMinType == MINERAL_EVAPORITE) {
            let toxEase = 1.0 - 0.55 * vit;
            (*acc).dPoi += (0.30 + 0.40 * max(nbHaz, select(0.25, 0.70, nbMinTrap == 1u))) * toxEase * mw;
            (*acc).dE -= 0.18 * toxEase * mw;
        } else if (nbMinType == MINERAL_CARBONATE) {
            if (nbMinTrap == 0u) {
                (*acc).dE += (0.28 + 0.22 * nbNutr) * mw;
                (*acc).dVit += 0.22 * mw;
            } else {
                (*acc).dPoi += (0.50 + 0.45 * nbHaz) * mw;
                (*acc).dE -= (0.20 + 0.24 * nbHaz) * mw;
            }
        } else if (nbMinType == MINERAL_SULFIDE) {
            let overWater = smoothstep(0.32, 0.54, oc.w);
            if (nbMinTrap == 0u && overWater > 0.35) {
                (*acc).dE += (0.34 + 0.28 * nbNutr) * (0.55 + 0.75 * myRedox) * mw;
                (*acc).dVit += 0.24 * mw;
            } else {
                (*acc).dPoi += (0.36 + 0.34 * nbHaz) * mw;
            }
        } else {
            (*acc).dE += 0.15 * mw;
            (*acc).dVit += (0.34 + 0.26 * nbNutr) * mw;
            (*acc).dPoi -= 0.14 * mw;
            (*acc).nearClayCat += mw;
        }
        return;
    }

    let nbEnergy = clamp(nbLifeState.x, 0.0, 1.6);
    let nbVit = clamp(nbLifeState.y, 0.0, 1.0);
    let nbKin = unpackKinPhase(nbLifeState.w);
    let nbGene = unpackEncapGene(nbLifeState.w);
    let nbGrad = unpackGraduated(nbLifeState.w);

    if (myPhase == 3u) {
        let redoxBoost = select(1.0, 1.55, nbPhase == 1u && elemRedoxWeight(nbAtomicNum) > 0.65);
        let encapBoost = select(1.0, 1.45, nbPhase == 2u || nbGrad == 1u);
        (*acc).nearMineralBioMine += wDist * clamp(nbEnergy, 0.25, 1.4) * encapBoost * redoxBoost;
        return;
    }

    // Neighbor is amphiphilic lipid bilayer (`nbPhase == 2u`)
    if (nbPhase == 2u) {
        if (myPhase == 0u) {
            (*acc).nearLipidCount += 1.0;
            (*acc).nearLipidEnergySum += nbEnergy;
            if (nbGrad == 1u) {
                (*acc).contactBestGrad = 1u;
            }
            (*acc).lipidCentroid += wDist * nbPos;
            (*acc).lipidCentW += wDist;
            let lipScore = wDist * clamp(nbEnergy, 0.25, 1.5);
            if (lipScore > (*acc).bestLipidW) {
                (*acc).bestLipidW = lipScore;
                (*acc).bestLipidSp = nbSp;
                (*acc).bestLipidKin = nbKin;
                (*acc).bestLipidGene = nbGene;
            }
        } else if (myPhase == 1u) {
            let myKin = unpackKinPhase(myLifeState.w);
            if (kinAngularDist(myKin, nbKin) < 0.30) {
                (*acc).blobLipidCount += 1.0;
                if (nbGrad == 1u) {
                    (*acc).blobLipidGradCount += 1.0;
                    (*acc).blobGradCount += 0.85;
                }
                let lw = max(wDist, 0.15) * 0.35;
                (*acc).blobKinCos += lw * cos(nbKin);
                (*acc).blobKinSin += lw * sin(nbKin);
                (*acc).blobKinWeight += lw;
                considerLeaderCandidate(acc, nbIdx, nbBlob);
            }
        } else {
            let myKin = unpackKinPhase(myLifeState.w);
            if (kinAngularDist(myKin, nbKin) < 0.32) {
                let bw = max(wDist, 0.15);
                (*acc).blobLipidCount += 1.0;
                if (nbGrad == 1u) {
                    (*acc).blobLipidGradCount += 1.0;
                    (*acc).blobGradCount += 0.85;
                }
                (*acc).blobWeightSum += 0.85;
                (*acc).blobEnergySum += nbEnergy * 0.85;
                (*acc).blobGeneSum += f32(nbGene) * 0.85;
                (*acc).blobKinCos += (bw * 0.85) * cos(nbKin);
                (*acc).blobKinSin += (bw * 0.85) * sin(nbKin);
                (*acc).blobKinWeight += bw * 0.85;
                considerLeaderCandidate(acc, nbIdx, nbBlob);
            }
        }
        return;
    }

    // Neighbor is living Particle Life cytoplasm (`nbPhase == 1u`)
    (*acc).nearTotalBio += 1.0;
    if (nbEnergy < 0.44 || nbLifeState.z > 0.24) {
        (*acc).nearStressedBio += wDist * 1.15;
    }

    if (myPhase == 0u) {
        let nbPol = evalCombinedMdpPolicy(nbBlob.z);
        if (r < bio.catRadius * 0.62) {
            (*acc).contactSpeciesMask |= (1u << (nbSp % 24u));
            (*acc).contactCount += 1.0;
            (*acc).contactEnergySum += nbEnergy;
            (*acc).contactLipidAllocSum += nbPol.lipidAlloc;
            if (nbGrad == 1u) {
                (*acc).contactBestGrad = 1u;
            }
            (*acc).contactKinCos += wDist * cos(nbKin);
            (*acc).contactKinSin += wDist * sin(nbKin);
            (*acc).contactKinW += wDist;
            let cScore = wDist * max(nbEnergy, 0.25);
            if (cScore > (*acc).contactBestW) {
                (*acc).contactBestW = cScore;
                (*acc).contactBestSp = nbSp;
                (*acc).contactBestKin = nbKin;
                (*acc).contactBestGene = nbGene;
            }
        }
        (*acc).blobCentroid += wDist * nbPos;
        (*acc).blobCentW += wDist;

        let effCat = evalAtomFromLifePair(myAtomicNum, myValence, myAtomLoad, nbLifeState, r, bio);
        (*acc).actScale = min((*acc).actScale, effCat.x);
        (*acc).glowSum += effCat.y;
        (*acc).chemBioCount += 1.0;
        return;
    }

    if (myPhase == 2u) {
        let myKin = unpackKinPhase(myLifeState.w);
        if (kinAngularDist(myKin, nbKin) < 0.32) {
            let bw = max(wDist, 0.15);
            (*acc).blobCount += 1.0;
            if (nbGrad == 1u) {
                (*acc).blobGradCount += 1.0;
            }
            (*acc).blobSpeciesMask |= (1u << (nbSp % 24u));
            (*acc).blobWeightSum += 1.0;
            (*acc).blobEnergySum += nbEnergy;
            (*acc).blobGeneSum += f32(nbGene);
            (*acc).blobKinCos += (bw * 1.4) * cos(nbKin);
            (*acc).blobKinSin += (bw * 1.4) * sin(nbKin);
            (*acc).blobKinWeight += bw * 1.4;
            considerLeaderCandidate(acc, nbIdx, nbBlob);
        } else if (r < bio.catRadius * 0.44 && nbEnergy > 0.35) {
            let aNbMy = getBioMatrixA(nChem, nBio, nbSp, myBioSp);
            if (aNbMy > 0.25) {
                (*acc).contactEngulfPress += (1.0 - r / (bio.catRadius * 0.44)) * aNbMy;
            }
        }
        return;
    }

    // Both particles are living cytoplasm (`myPhase == 1u && nbPhase == 1u`)
    let myKin = unpackKinPhase(myLifeState.w);
    let kinDist = kinAngularDist(myKin, nbKin);
    let aMyNb = getBioMatrixA(nChem, nBio, myBioSp, nbSp);
    let aNbMy = getBioMatrixA(nChem, nBio, nbSp, myBioSp);

    // Compatible mutual attractors blend seamlessly on contact into a unified colony
    let sameBlob = (r < bio.catRadius * 0.92) &&
        ((kinDist < 0.28) || (r < bio.catRadius * 0.68 && aMyNb > 0.05 && aNbMy > 0.05));

    if (sameBlob) {
        let bw = max(wDist, 0.15);
        (*acc).blobCount += 1.0;
        (*acc).blobWeightSum += bw;
        (*acc).blobEnergySum += nbEnergy;
        (*acc).blobVitSum += nbVit;
        if (nbGrad == 1u) {
            (*acc).blobGradCount += 1.0;
        }
        (*acc).blobKinCos += bw * cos(nbKin);
        (*acc).blobKinSin += bw * sin(nbKin);
        (*acc).blobKinWeight += bw;
        (*acc).blobSpeciesMask |= (1u << (nbSp % 24u));
        (*acc).blobGeneSum += f32(nbGene);
        considerLeaderCandidate(acc, nbIdx, nbBlob);

        let nbCtrl = u32(round(nbMdp.ctrl.w));
        if ((nbCtrl & 4096u) != 0u) {
            let nbAct = (nbCtrl >> 2u) & 3u;
            let cw = bw * (0.65 + 0.55 * clamp(nbMdp.ctrl.z + 0.5, 0.1, 1.5));
            if (nbAct == 0u) { (*acc).mdpVotes.x += cw; }
            else if (nbAct == 1u) { (*acc).mdpVotes.y += cw; }
            else if (nbAct == 2u) { (*acc).mdpVotes.z += cw; }
            else { (*acc).mdpVotes.w += cw; }
            (*acc).mdpVoteWeight += cw;
            let mScore = nbEnergy + 0.45 * nbMdp.ctrl.z;
            if (mScore > (*acc).bestMdpScore) {
                (*acc).bestMdpScore = mScore;
                (*acc).bestMdpQPack = nbMdp.qPack;
            }
        }
        return;
    }

    (*acc).nearForeignBio += wDist;

    // Pure physical close-contact competition/engulfment (`r < 0.54 * catRadius`):
    // Driven purely by whether the combined attractor matrix physically compresses foreign cells into overlap.
    let engulfR = bio.catRadius * 0.54;
    if (r < engulfR) {
        let myPol = evalCombinedMdpPolicy(myBlobZ);
        let nbPol = evalCombinedMdpPolicy(nbBlob.z);
        let wClose = 1.0 - r / max(engulfR, 1e-4);
        let effPushMeIntoNb = aMyNb + myPol.dForeignBio * 0.45;
        let effPushNbIntoMe = aNbMy + nbPol.dForeignBio * 0.45;
        if (effPushMeIntoNb > effPushNbIntoMe + 0.08 && nbEnergy > 0.15) {
            (*acc).contactPreyHarvest += wClose * clamp(nbEnergy, 0.2, 1.5) * clamp(effPushMeIntoNb - effPushNbIntoMe, 0.05, 1.4);
        }
        if (nbEnergy > 0.22 && effPushNbIntoMe > effPushMeIntoNb + 0.06) {
            let press = wClose * clamp(effPushNbIntoMe - effPushMeIntoNb, 0.05, 1.5) * clamp(nbEnergy, 0.3, 1.5);
            (*acc).contactEngulfPress += press;
            if (press > (*acc).contactBestEngulfW) {
                (*acc).contactBestEngulfW = press;
                (*acc).contactBestEngulfSp = nbSp;
                (*acc).contactBestEngulfKin = nbKin;
                (*acc).contactBestEngulfGene = nbGene;
                (*acc).contactBestEngulfGrad = nbGrad;
                (*acc).contactBestEngulfE = nbEnergy;
                (*acc).contactBestEngulfAlloc = nbPol.lipidAlloc;
            }
        }
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

fn attemptMineralCrystallize(
    pi: u32,
    pos: vec3f,
    atomicNum: u32,
    valence: u32,
    bondEase: f32,
    solar: f32,
    upwell: f32,
    oc: vec4f,
    tSurfK: f32,
    tAirSfcK: f32,
    hydroSample: vec2f,
    landFrac: f32,
    windNorm: f32,
    curNorm: f32,
    frameU: u32,
    acc: BioUnifiedAccum,
    bio: BioLink,
) -> BioStepResult {
    if (bio.mineralEnabled <= 0.5 || !isMineralCandidate(atomicNum, valence) || acc.nearMineralCount >= 11.0) {
        return bioStepOnly(vec4f(0.0), vec4f(0.0));
    }
    let mRate = max(bio.mineralRate, 0.0);
    if (mRate <= 0.001) {
        return bioStepOnly(vec4f(0.0), vec4f(0.0));
    }

    let dtMin = bio.dt * mRate * 0.24;
    let tBlend = clamp(bio.localTempScale, 0.0, 1.5);
    let hBlend = clamp(bio.localHydroScale, 0.0, 1.5);
    let rh = hydroSample.x;
    let precipMm = hydroSample.y;
    let eAir = rh * saturationVaporKpa(tAirSfcK);
    let eSea = saturationVaporKpa(tSurfK);
    let vpdLand = max(0.0, saturationVaporKpa(tAirSfcK) - eAir);
    let rainWash = smoothstep(0.3, 6.0, precipMm);
    let netOceanEvap = max(0.0, (eSea - eAir) * (0.45 + 0.75 * windNorm) - precipMm * 0.25);
    let evapHydro = mix(1.0, mix(clamp(0.45 + 0.45 * netOceanEvap, 0.30, 1.65), clamp(0.40 + 0.45 * vpdLand, 0.25, 1.75) * (1.0 - 0.72 * rainWash), landFrac), hBlend);
    let clayHydro = mix(1.0, clamp(0.65 + 0.50 * rh + 0.40 * rainWash - 0.30 * smoothstep(1.4, 4.0, vpdLand), 0.35, 1.35), hBlend);
    let evapTemp = mix(1.0, clamp(0.28 + 1.15 * smoothstep(275.0, 314.0, tSurfK), 0.25, 1.45), tBlend) * evapHydro;
    let carbTemp = mix(1.0, clamp(0.55 + 0.85 * smoothstep(278.0, 303.0, tSurfK), 0.50, 1.40), tBlend);
    let sulfTemp = mix(1.0, clamp(1.28 - 0.50 * smoothstep(274.0, 302.0, tSurfK), 0.72, 1.32), tBlend);
    let clayTemp = mix(1.0, clamp(1.15 - 0.55 * smoothstep(296.0, 318.0, tSurfK), 0.58, 1.18), tBlend) * clayHydro;
    let mineralHeadroom = clamp(1.0 - acc.nearMineralCount / 11.0, 0.0, 1.0);
    let calmWind = 1.0 / (1.0 + 2.6 * windNorm * windNorm);
    let calmCur = 1.0 / (1.0 + 2.4 * curNorm * curNorm);
    let calmEnv = mix(calmCur, calmWind, landFrac);

    var chooseType = 99u;
    var pCryst = 0.0;
    var crystAngle = hashFloat(pi, frameU, 60u) * TWO_PI;
    var trapFlag = 0u;
    var nutr = 0.25;
    var haz = 0.0;

    if (acc.nearSeedW > 0.0 && acc.nearMinDist >= bio.catRadius * 0.25 && elementFitsMineralType(atomicNum, valence, acc.nearSeedType)) {
        let mT = acc.nearSeedType;
        var habFit = 1.0;
        if (mT == MINERAL_EVAPORITE) {
            habFit = mix(select(0.25, 0.05, curNorm > 0.22), 0.75 + 0.55 * solar, landFrac) * evapTemp;
        } else if (mT == MINERAL_CARBONATE) {
            habFit = carbTemp;
        } else if (mT == MINERAL_SULFIDE) {
            habFit = smoothstep(0.32, 0.52, oc.w) * (0.65 + 0.7 * upwell) * sulfTemp;
        } else if (mT == MINERAL_CLAY) {
            habFit = mix(1.0, 0.15, smoothstep(0.35, 0.78, landFrac)) * clayTemp;
        }
        if (habFit > 0.18) {
            chooseType = mT;
            pCryst = clamp(0.14 * mRate * habFit * calmEnv * mineralHeadroom * bondEase * dtMin, 0.0, 0.014);
            let symSteps = select(6.0, 4.0, mT == MINERAL_EVAPORITE || mT == MINERAL_SULFIDE);
            let stepIdx = floor(hashFloat(pi, frameU, 61u) * symSteps);
            crystAngle = wrapAngle(acc.nearSeedAngle + stepIdx * (TWO_PI / symSteps));
            let toxicAtom = (atomicNum == 4u || atomicNum == 7u || atomicNum == 13u);
            trapFlag = select(acc.nearSeedTrap, 1u, toxicAtom || (mT == MINERAL_CARBONATE && acc.nearBe > 0.25));
        }
    } else if (acc.nearMineralCount < 1.0) {
        let sSil = select(0.0, acc.nearPreSilica * (0.65 + 0.55 * landFrac), elementFitsMineralType(atomicNum, valence, MINERAL_SILICATE));
        let sEvap = select(0.0, acc.nearPreHalide * (0.20 + 0.95 * landFrac) * (0.45 + 0.75 * solar) * evapTemp, elementFitsMineralType(atomicNum, valence, MINERAL_EVAPORITE));
        let sCarb = select(0.0, acc.nearPreCarb * (0.60 + 0.45 * (1.0 - windNorm)) * carbTemp, elementFitsMineralType(atomicNum, valence, MINERAL_CARBONATE));
        let sSulf = select(0.0, acc.nearPreSulf * smoothstep(0.35, 0.55, oc.w) * (0.55 + upwell) * sulfTemp, elementFitsMineralType(atomicNum, valence, MINERAL_SULFIDE));
        let sClay = select(0.0, (acc.nearClayH + acc.nearPreSilica * 0.45) * smoothstep(0.32, 0.55, oc.w) * clayTemp, elementFitsMineralType(atomicNum, valence, MINERAL_CLAY));

        var bestS = 1.35;
        if (sSil > bestS) { bestS = sSil; chooseType = MINERAL_SILICATE; }
        if (sEvap > bestS) { bestS = sEvap; chooseType = MINERAL_EVAPORITE; }
        if (sCarb > bestS) { bestS = sCarb; chooseType = MINERAL_CARBONATE; }
        if (sSulf > bestS) { bestS = sSulf; chooseType = MINERAL_SULFIDE; }
        if (sClay > bestS) { bestS = sClay; chooseType = MINERAL_CLAY; }

        if (chooseType <= MINERAL_CLAY) {
            pCryst = clamp(0.011 * mRate * min(bestS, 3.5) * calmEnv * bondEase * dtMin, 0.0, 0.0022);
            let toxicAtom = (atomicNum == 4u || atomicNum == 7u || atomicNum == 13u);
            if (chooseType == MINERAL_EVAPORITE) {
                trapFlag = select(0u, 1u, toxicAtom);
            } else if (chooseType == MINERAL_CARBONATE) {
                trapFlag = select(0u, 1u, atomicNum == 4u || acc.nearBe > 0.18);
            } else if (chooseType == MINERAL_SULFIDE) {
                trapFlag = select(0u, 1u, landFrac > 0.48 || toxicAtom);
            }
        }
    }

    if (chooseType <= MINERAL_CLAY && pCryst > 0.0 && hashFloat(pi, frameU, 62u) < pCryst) {
        if (chooseType == MINERAL_SILICATE) {
            nutr = 0.25; haz = 0.0;
        } else if (chooseType == MINERAL_EVAPORITE) {
            nutr = select(0.30, 0.10, trapFlag == 1u);
            haz = select(0.45, 0.88, trapFlag == 1u);
        } else if (chooseType == MINERAL_CARBONATE) {
            nutr = select(0.85, 0.40, trapFlag == 1u);
            haz = select(0.0, 0.92, trapFlag == 1u);
        } else if (chooseType == MINERAL_SULFIDE) {
            nutr = 0.90;
            haz = select(0.08, 0.72, trapFlag == 1u);
        } else {
            nutr = 0.95; haz = 0.0;
        }
        let grade = u32(clamp(round(nutr * 12.0 + clamp(acc.nearMineralCount, 0.0, 3.0)), 0.0, 15.0));
        let eBase = sphereEastBasis(pos);
        let eNorth = normalize(cross(pos, eBase));
        let crystDir = cos(crystAngle) * eBase + sin(crystAngle) * eNorth;
        let initInteg = clamp(0.84 + 0.18 * calmEnv, 0.78, 1.05);
        return bioStepOnly(
            vec4f(initInteg, nutr, haz, packLifeMetaGrad(3u, chooseType, crystAngle, grade, trapFlag)),
            vec4f(crystDir, 2.0),
        );
    }

    return bioStepOnly(vec4f(0.0), vec4f(0.0));
}

const DEATH_STARVATION = 1u;
const DEATH_CROWDING = 2u;
const DEATH_PREDATION = 3u;
const DEATH_DECOMPOSITION = 4u;
const DEATH_POISON = 5u;
const DEATH_MEMBRANE = 6u;
const DEATH_TEMPERATURE = 7u;
const DEATH_UV = 8u;
const DEATH_DESICCATION = 9u;

fn deadCell(cause: u32) -> vec4f {
    return vec4f(0.0, 0.0, 0.0, -4.0 * f32(cause));
}

fn energyDeath(rawEnergy: f32, energyBeforeStep: f32, threshold: f32, dt: f32, crowd: f32, stress: vec4f) -> u32 {
    let cutoff = select(threshold, energyBeforeStep, rawEnergy <= energyBeforeStep && energyBeforeStep < threshold);
    let largest = max(max(stress.x, stress.y), max(stress.z, stress.w));
    let crowded = crowd > 0.0 && rawEnergy + crowd * dt > cutoff;
    if (crowded && crowd >= largest) { return DEATH_CROWDING; }
    if (largest > 0.0 && rawEnergy + dot(stress, vec4f(1.0)) * dt > cutoff) {
        if (largest == stress.x) { return DEATH_POISON; }
        if (largest == stress.y) { return DEATH_TEMPERATURE; }
        if (largest == stress.z) { return DEATH_UV; }
        return DEATH_DESICCATION;
    }
    if (crowded) { return DEATH_CROWDING; }
    return DEATH_STARVATION;
}

fn updatePhase0FluidState(
    pi: u32,
    atomicNum: u32,
    valence: u32,
    curLifeState: vec4f,
    oc: vec4f,
    tHabK: f32,
    tSample: vec4f,
    solar: f32,
    rh: f32,
    precipMm: f32,
    seaEvapFlux: f32,
    vpdLand: f32,
    oroLift: f32,
    upwell: f32,
    elevKm: f32,
    dtBio: f32,
    frameU: u32,
) -> vec4f {
    let dtHyd = max(dtBio, 0.004) * 1.4;
    let isWaterAtom = (atomicNum == 1u || atomicNum == 6u);
    let isCarbonAtom = (atomicNum == 5u);
    let isSolubleIon = (valence > 0u) && (atomicNum == 3u || atomicNum == 4u || atomicNum == 7u || atomicNum == 9u || atomicNum == 10u || atomicNum == 12u || atomicNum == 13u);
    let soilHumus = clamp(curLifeState.x * (1.0 - 0.06 * dtHyd), 0.0, 1.0);
    var comp = clamp(round(curLifeState.y), 0.0, 2.0);
    var alt = clamp(curLifeState.z, 0.0, 1.0);

    if (curLifeState.y <= 0.001 && curLifeState.z <= 0.001 && oc.w >= 0.48 && (isWaterAtom || isSolubleIon)) {
        if (atomicNum == 1u || isSolubleIon || hashFloat(pi, frameU, 71u) < 0.72) {
            comp = 1.0;
        }
    }

    if (valence == 0u) {
        let convectTarget = clamp(0.18 + 0.55 * solar + 0.15 * oroLift, 0.08, 0.92);
        alt = clamp(mix(alt, convectTarget, clamp(0.45 * dtHyd, 0.0, 0.15)), 0.0, 1.0);
        return vec4f(soilHumus, 0.0, alt, 0.0);
    }

    let inSea = oc.w >= 0.44;
    if (isWaterAtom) {
        if (inSea) {
            if (tHabK < 271.35) {
                comp = 2.0; alt = 0.0;
            } else if (comp > 1.5 && tHabK >= 272.2) {
                comp = 1.0; alt = 0.0;
            }
            if (comp >= 0.5 && comp < 1.5) {
                let pEvap = clamp(0.08 * seaEvapFlux * (0.65 + 0.65 * solar) * dtHyd, 0.0, 0.09);
                let waterBias = select(0.58, 1.0, atomicNum == 1u);
                if (hashFloat(pi, frameU, 72u) < pEvap * waterBias) {
                    comp = 0.0;
                    alt = clamp(max(alt, 0.22 + 0.25 * solar), 0.15, 0.75);
                }
            } else if (comp < 0.5) {
                let condenseDrive = smoothstep(0.68, 0.96, rh) * 0.65 + smoothstep(0.3, 4.5, precipMm) * 1.15;
                let pCond = clamp(0.16 * condenseDrive * dtHyd, 0.0, 0.16);
                if (hashFloat(pi, frameU, 73u) < pCond) {
                    comp = 1.0; alt = 0.0;
                }
            }
        } else {
            if (comp > 1.5) { comp = 1.0; }
            if (comp < 0.5) {
                let oroTrig = smoothstep(0.08, 0.85, oroLift) * (0.45 + 0.55 * rh);
                let rainTrig = smoothstep(0.25, 4.0, precipMm) + smoothstep(0.74, 0.98, rh) * 0.65;
                let pRain = clamp(0.18 * (oroTrig + rainTrig) * dtHyd, 0.0, 0.18);
                if (hashFloat(pi, frameU, 74u) < pRain) {
                    comp = 1.0; alt = 0.0;
                }
            } else {
                let pEvapLand = clamp(0.055 * (vpdLand * 0.45 + 0.45 * solar) * (1.0 - 0.65 * smoothstep(0.3, 4.0, precipMm)) * dtHyd, 0.0, 0.075);
                if (hashFloat(pi, frameU, 75u) < pEvapLand) {
                    comp = 0.0; alt = 0.24;
                }
            }
        }
    } else if (isCarbonAtom) {
        if (inSea) {
            let coldDissolve = clamp((1.15 - smoothstep(276.0, 298.0, tHabK)) + 0.45 * upwell, 0.0, 1.5);
            if (comp < 0.5 && hashFloat(pi, frameU, 76u) < clamp(0.12 * coldDissolve * dtHyd, 0.0, 0.12)) {
                comp = 1.0; alt = 0.0;
            } else if (comp >= 0.5 && tHabK > 292.0 && hashFloat(pi, frameU, 77u) < clamp(0.09 * smoothstep(292.0, 305.0, tHabK) * dtHyd, 0.0, 0.10)) {
                comp = 0.0; alt = 0.18;
            }
        } else {
            comp = 0.0;
        }
    } else if (isSolubleIon) {
        comp = select(0.0, 1.0, inSea || soilHumus > 0.25);
        if (comp >= 0.5) { alt = 0.0; }
    } else {
        comp = 0.0;
    }

    if (comp < 0.5) {
        let thermalLift = max(0.0, tSample.y - tSample.w) * 0.04 + solar * 0.42 + oroLift * 0.35 + clamp(elevKm * 0.08, 0.0, 0.30);
        let altGoal = clamp(0.14 + thermalLift - 0.18 * smoothstep(0.4, 5.0, precipMm), 0.04, 0.95);
        alt = clamp(mix(alt, altGoal, clamp(0.65 * dtHyd, 0.0, 0.20)), 0.0, 1.0);
    } else {
        alt = 0.0;
    }

    return vec4f(soilHumus, comp, alt, 0.0);
}

fn finalizeBioUnified(
    pi: u32,
    pos: vec3f,
    vel: vec3f,
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    curLifeState: vec4f,
    curBlob: vec4u,
    curMdp: MdpState,
    windRaw: vec3f,
    oc: vec4f,
    tSample: vec4f,
    uvSample: vec4f,
    hydroSample: vec2f,
    terrainSample: vec4f,
    acc: BioUnifiedAccum,
    bio: BioLink,
) -> BioStepResult {
    let myPhase = unpackPhase(curLifeState.w);
    let landFrac = clamp(1.0 - smoothstep(0.18, 0.48, oc.w), 0.0, 1.0);
    let tBlend = clamp(bio.localTempScale, 0.0, 1.5);
    let hBlend = clamp(bio.localHydroScale, 0.0, 1.5);
    let elevKm = max(0.0, terrainSample.x);
    let ridgeVentMask = clamp(terrainSample.w, 0.0, 1.0);
    let oroLift = max(0.0, windRaw.x * terrainSample.y + windRaw.y * terrainSample.z) * 0.06;
    let tHabK = effectiveSurfaceTempK(tSample, oc.w) - clamp(elevKm, 0.0, 6.0) * 3.8 * landFrac * tBlend;
    let uvRes = evalUvSolarAndStress(uvSample, bio.localUvScale);
    let solar = uvRes.x;
    let windSim = evalWindVel3D(pos, windRaw, bio);
    let windNorm = length(windSim);
    let curSim = evalOceanCurrentVel(pos, oc, bio.oceanCurrentScale);
    let curNorm = length(curSim);
    let rh = hydroSample.x;
    let precipMm = hydroSample.y + oroLift * 1.4;
    let eAir = rh * saturationVaporKpa(tSample.w);
    let eSea = saturationVaporKpa(tSample.y);
    let seaEvapFlux = max(0.0, eSea - eAir) * (0.45 + 0.75 * windNorm);
    let seaFogCondense = smoothstep(0.0, 0.35, eAir - eSea);
    let rainWash = smoothstep(0.3, 6.0, precipMm);
    let vpdLand = max(0.0, saturationVaporKpa(tSample.w) - eAir);
    let uvStress = uvRes.y * mix(1.0, max(0.38, 1.0 - 0.50 * seaFogCondense - 0.32 * rainWash), hBlend);
    let frameU = u32(max(bio.frame, 0.0)) + u32(max(bio.seedHash, 0.0));

    // Inorganic Crustal Mineral (`myPhase == 3u`)
    if (myPhase == 3u) {
        if (bio.mineralEnabled <= 0.5) {
            return bioStepOnly(vec4f(clamp(curLifeState.x, 0.1, 1.25), curLifeState.y, curLifeState.z, curLifeState.w), vec4f(0.0));
        }
        let mType = min(unpackBioSpecies(curLifeState.w), MINERAL_CLAY);
        let mAngle = unpackKinPhase(curLifeState.w);
        var mTrap = unpackGraduated(curLifeState.w);
        let integ = clamp(curLifeState.x, 0.02, 1.25);
        let nutr = clamp(curLifeState.y, 0.0, 1.0);
        var haz = clamp(curLifeState.z, 0.0, 1.0);
        let dtMin = bio.dt * max(bio.mineralRate, 0.0) * 0.24;
        let liveGrade = u32(clamp(round(acc.nearMineralCount * 1.35 + nutr * 2.5), 0.0, 15.0));

        let latticeShield = 1.0 / (1.0 + 0.16 * max(0.0, acc.nearMineralCount - 1.0));
        let bioWeather = min(acc.nearMineralBioMine, 3.5) * 0.16;
        let inWater = smoothstep(0.36, 0.54, oc.w);
        let freezeThaw = smoothstep(273.15, 252.0, tHabK) * landFrac * tBlend * 0.24;

        var waterErode = 0.0;
        var airErode = 0.0;
        var mechErode = 0.0;

        if (mType == MINERAL_SILICATE) {
            mechErode = 0.06 * (windNorm + curNorm) + 0.20 * (windNorm * windNorm + curNorm * curNorm) + freezeThaw * 0.5;
        } else if (mType == MINERAL_EVAPORITE) {
            let evapDissolveT = mix(1.0, clamp(0.65 + 0.70 * smoothstep(273.0, 305.0, tHabK), 0.60, 1.40), tBlend);
            waterErode = (inWater * (0.65 + 3.0 * curNorm) + landFrac * rainWash * 2.2 * hBlend) * evapDissolveT;
            airErode = (1.0 - inWater) * (0.07 * windNorm + 0.18 * windNorm * windNorm);
        } else if (mType == MINERAL_CARBONATE) {
            let coldCarbDissolve = mix(1.0, clamp(1.38 - 0.65 * smoothstep(278.0, 303.0, tHabK), 0.70, 1.55), tBlend);
            waterErode = (inWater * (0.14 + 1.05 * curNorm) + landFrac * rainWash * 0.45 * hBlend) * coldCarbDissolve;
            airErode = (1.0 - inWater) * (0.09 + 0.65 * windNorm) + freezeThaw;
        } else if (mType == MINERAL_SULFIDE) {
            waterErode = inWater * (0.04 + 0.22 * curNorm);
            let airExposed = smoothstep(0.24, 0.68, landFrac);
            let hotOx = mix(1.0, clamp(0.70 + 0.65 * smoothstep(275.0, 310.0, tHabK), 0.65, 1.40), tBlend) * (1.0 + 0.35 * uvStress) * mix(1.0, 1.0 + 0.55 * rainWash + 0.25 * smoothstep(0.5, 0.95, rh), hBlend);
            airErode = airExposed * (0.58 + 2.6 * windNorm) * hotOx;
            if (airExposed > 0.40) {
                mTrap = 1u;
                haz = clamp(haz + 0.48 * hotOx * dtMin, 0.0, 1.0);
            }
        } else {
            waterErode = inWater * (0.04 + 0.20 * curNorm);
            let clayHeat = mix(1.0, clamp(0.75 + 0.75 * smoothstep(294.0, 316.0, tHabK), 0.75, 1.55), tBlend) * (1.0 + 0.30 * uvStress);
            let clayDry = mix(1.0, clamp(0.75 + 0.45 * smoothstep(1.2, 4.0, vpdLand) - 0.35 * smoothstep(0.4, 0.95, rh) - 0.45 * rainWash, 0.35, 1.50), hBlend);
            airErode = smoothstep(0.26, 0.72, landFrac) * (0.40 + 2.0 * windNorm) * clayHeat * clayDry + freezeThaw * 0.6;
        }

        let erodeRate = (waterErode + airErode + mechErode + bioWeather) * latticeShield;
        var nextInteg = integ - erodeRate * dtMin;
        if (erodeRate < 0.015 && acc.nearMineralCount >= 5.0) {
            nextInteg = clamp(integ + 0.014 * dtMin, 0.02, 1.05);
        }
        nextInteg = clamp(nextInteg, 0.0, 1.25);

        let pSpall = clamp(erodeRate * (1.20 - 0.72 * clamp(nextInteg, 0.0, 1.0)) * dtMin * 1.15, 0.0, 0.045);
        if (nextInteg <= 0.10 || (erodeRate > 0.04 && hashFloat(pi, frameU, 65u) < pSpall)) {
            let flowPlume = select(windSim, curSim, inWater >= 0.45) * 6.5;
            let eEast = sphereEastBasis(pos);
            let eNorth = normalize(cross(pos, eEast));
            let s1 = hashFloat(pi, frameU, 63u) * 2.0 - 1.0;
            let s2 = hashFloat(pi, frameU, 64u) * 2.0 - 1.0;
            let ejectVel = flowPlume + (eEast * s1 + eNorth * s2) * 0.85;
            let ejectGlow = select(0.75, -0.85, mTrap == 1u || haz > 0.45);
            return bioStepOnly(
                vec4f(0.0),
                vec4f(ejectVel, packDeltaActGlow(0.9, ejectGlow)),
            );
        }

        return bioStepOnly(
            vec4f(nextInteg, nutr, haz, packLifeMetaGrad(3u, mType, mAngle, liveGrade, mTrap)),
            vec4f(0.0),
        );
    }

    if (bio.enabled <= 0.5 && bio.mineralEnabled <= 0.5) {
        if (myPhase == 0u) {
            let nextFluid = updatePhase0FluidState(
                pi, atomicNum, valence, curLifeState, oc, tHabK, tSample, solar, rh, precipMm,
                seaEvapFlux, vpdLand, oroLift, evalCoastalUpwelling(oc) + 0.75 * ridgeVentMask, elevKm, bio.dt, frameU,
            );
            return bioStepOnly(nextFluid, vec4f(0.0));
        }
        return bioStepOnly(curLifeState, vec4f(0.0));
    }

    if (bio.enabled <= 0.5) {
        if (myPhase != 0u) {
            return bioStepOnly(vec4f(max(curLifeState.x, 0.55), curLifeState.y, curLifeState.z, curLifeState.w), vec4f(0.0));
        }
        let chemUpwell = evalCoastalUpwelling(oc) + 0.75 * ridgeVentMask + 0.25 * smoothstep(1.0, 9.0, tSample.y - tSample.w) * smoothstep(0.36, 0.54, oc.w) * hBlend;
        let crystRes = attemptMineralCrystallize(
            pi, pos, atomicNum, valence, 1.0 / (1.0 + 0.55 * max(atomLoad, 0.0)),
            solar, chemUpwell, oc, tHabK, tSample.w, hydroSample, landFrac, windNorm, curNorm, frameU, acc, bio,
        );
        if (unpackPhase(crystRes.nextLifeState.w) == 3u) {
            return crystRes;
        }
        let nextFluidOff = updatePhase0FluidState(
            pi, atomicNum, valence, curLifeState, oc, tHabK, tSample, solar, rh, precipMm,
            seaEvapFlux, vpdLand, oroLift, chemUpwell, elevKm, bio.dt, frameU,
        );
        return bioStepOnly(nextFluidOff, vec4f(0.0));
    }

    let rate = max(bio.rate, 0.0);
    let dtBio = bio.dt * rate;
    let q10Bio = mix(1.0, clamp(pow(1.38, clamp((tHabK - 288.15) / 10.0, -2.5, 2.2)), 0.42, 1.55), tBlend);
    let convectiveMix = smoothstep(1.0, 9.0, tSample.y - tSample.w) * smoothstep(0.36, 0.54, oc.w);
    let upwell = evalCoastalUpwelling(oc) + 0.75 * ridgeVentMask + (0.28 * convectiveMix + 0.16 * rainWash) * hBlend * smoothstep(0.36, 0.54, oc.w);
    let landHydration = clamp(0.42 * smoothstep(0.32, 0.92, rh) + 0.68 * rainWash + 0.32 * seaFogCondense, 0.0, 1.25);
    let aridStress = smoothstep(1.1, 4.0, vpdLand) * (1.0 - smoothstep(0.4, 3.0, precipMm));
    let oceanBleachBoost = smoothstep(301.0, 307.0, tHabK) * smoothstep(0.36, 0.54, oc.w) * uvStress * (1.0 - rainWash) * hBlend;
    let nChem = max(u32(bio.chemColours), 1u);
    let nBio = max(u32(bio.bioColours), 1u);

    // Phase 0: Fluid Chemistry (emergent mineral nucleation + strictly close-contact assimilation driven by MDP allocation)
    if (myPhase == 0u) {
        let chemForce = finalizeChemBioForce(acc, bio);
        let bondEase = 1.0 / (1.0 + 0.55 * max(atomLoad, 0.0));
        var nextFluid = updatePhase0FluidState(
            pi, atomicNum, valence, curLifeState, oc, tHabK, tSample, solar, rh, precipMm,
            seaEvapFlux, vpdLand, oroLift, upwell, elevKm, dtBio, frameU,
        );
        if (landFrac > 0.40 && acc.nearTotalBio > 0.5 && isBiogenicElement(atomicNum, valence)) {
            nextFluid.x = clamp(nextFluid.x + 0.08 * dtBio * min(acc.nearTotalBio, 3.0), 0.0, 1.0);
        }
        if (!isBiogenicElement(atomicNum, valence)) {
            let crystRes = attemptMineralCrystallize(
                pi, pos, atomicNum, valence, bondEase,
                solar, upwell, oc, tHabK, tSample.w, hydroSample, landFrac, windNorm, curNorm, frameU, acc, bio,
            );
            if (unpackPhase(crystRes.nextLifeState.w) == 3u) {
                return crystRes;
            }
            return bioStepOnly(nextFluid, chemForce);
        }

        let contactCount = acc.contactCount;
        let contactAvgE = acc.contactEnergySum / max(contactCount, 1.0);
        let avgLipidAlloc = acc.contactLipidAllocSum / max(contactCount, 1.0);
        let contactGene = select(
            acc.bestLipidGene,
            u32(clamp(round(f32(acc.contactBestGene)), 0.0, 15.0)),
            contactCount > 0.5,
        );
        let avgLipidE = acc.nearLipidEnergySum / max(acc.nearLipidCount, 1.0);
        let eBase = sphereEastBasis(pos);
        var outwardDir = eBase;
        if (acc.blobCentW > 1e-4) {
            let cLen = length(acc.blobCentroid);
            if (cLen > 1e-5) {
                let blobCenter = acc.blobCentroid / cLen;
                outwardDir = orthoBasis(pos, pos - blobCenter);
            }
        }
        let clayCatBoost = 1.0 + 0.85 * clamp(acc.nearClayCat, 0.0, 1.5);
        let landEnvFertile = clamp(0.36 + 0.68 * landHydration + 0.35 * curLifeState.x + 0.22 * clamp(acc.nearLipidCount * 0.25, 0.0, 1.0), 0.28, 1.35);
        let envMedia = mix(0.55 + 0.85 * upwell, landEnvFertile, landFrac) * clayCatBoost;
        let hostGrad = acc.contactBestGrad == 1u;

        // 0A. Emergent Lipid Bilayer Synthesis (`0 -> 2`): driven strictly by host MDP `lipidAlloc` lever on close contact
        if (isLipidElement(atomicNum, valence) && contactCount >= 1.0 && avgLipidAlloc > 0.24 && max(contactAvgE, avgLipidE) > 0.48) {
            let maxLocalLipids = select(7.5, 11.5, hostGrad);
            let lipidHeadroom = clamp(1.0 - acc.nearLipidCount / maxLocalLipids, 0.0, 1.0);
            if (lipidHeadroom > 0.05) {
                let pLipid = clamp(
                    0.75 * avgLipidAlloc * q10Bio * max(solar, 0.12) * envMedia * clamp(max(contactAvgE, avgLipidE), 0.5, 1.4) * lipidHeadroom * bondEase * dtBio,
                    0.0,
                    0.18,
                );
                if (hashFloat(pi, frameU, 30u) < pLipid) {
                    let hostSp = min(select(acc.bestLipidSp, acc.contactBestSp, acc.contactBestW > 0.0), nBio - 1u);
                    let hostKin = select(acc.bestLipidKin, acc.contactBestKin, acc.contactBestW > 0.0);
                    var initOrient = -outwardDir;
                    if (acc.nearLipidCount >= 1.5 && acc.lipidCentW > 1e-4 && acc.blobCentW > 1e-4) {
                        let bLen = length(acc.blobCentroid);
                        let lLen = length(acc.lipidCentroid);
                        if (bLen > 1e-5 && lLen > 1e-5) {
                            let blobCenter = acc.blobCentroid / bLen;
                            let lipidCenter = acc.lipidCentroid / lLen;
                            let dPos = pos - blobCenter;
                            let dLip = lipidCenter - blobCenter;
                            if (dot(dPos, dPos) > dot(dLip, dLip)) {
                                initOrient = outwardDir;
                            }
                        }
                    }
                    let lipidGrad = select(0u, 1u, hostGrad);
                    return BioStepResult(
                        vec4f(0.76, 0.06, 0.0, packLifeMetaGrad(2u, hostSp, hostKin, contactGene, lipidGrad)),
                        vec4f(initOrient, 2.0),
                        emptyMdpState(),
                        vec4u(pi, 0u, 0u, 0u),
                    );
                }
            }
        }

        // 0B. Close-Contact Cytoplasm Assimilation (`0 -> 1`): no hard-coded fixer presets, strictly physical contact growth
        let stericHeadroom = clamp(1.0 - acc.nearTotalBio / 220.0, 0.0, 1.0);
        if (contactCount >= 1.0 && stericHeadroom > 0.02 && contactAvgE > 0.48) {
            let growthAlloc = clamp(1.0 - 0.55 * avgLipidAlloc, 0.25, 1.0);
            let elemAffinity = 0.85 + 0.35 * elemSolarWeight(atomicNum);
            let pAssimilate = clamp(
                0.38 * growthAlloc * q10Bio * max(solar, 0.08) * envMedia * clamp(contactCount * 0.5, 0.5, 2.8) * elemAffinity * stericHeadroom * bondEase * dtBio,
                0.0,
                0.10,
            );
            if (hashFloat(pi, frameU, 10u) < pAssimilate) {
                var nextSp = min(acc.contactBestSp, nBio - 1u);
                let pMut = clamp(0.032 * clayCatBoost * (1.0 + 0.75 * uvStress) * growthAlloc, 0.01, 0.14);
                if (nBio > 1u && hashFloat(pi, frameU, 11u) < pMut) {
                    nextSp = pickCompatibleNovelSpecies(nChem, nBio, nextSp, acc.contactSpeciesMask, pi, frameU, 40u);
                }
                let childGene = mutateGene(acc.contactBestGene, pi, frameU, 17u, 0.05);
                let childGrad = select(0u, 1u, hostGrad && acc.nearLipidCount >= 1.0);
                let childInitAct = u32(floor(hashFloat(pi, frameU, 18u) * 2.0)) & 1u;
                let childMdp = MdpState(getSpeciesMdpPriorPack(nChem, nBio, nextSp), vec4f(0.68, 0.0, 0.0, packMdpCtrl(MDP_STATE_ABUNDANT, childInitAct, 6u, 1u)));
                return BioStepResult(
                    vec4f(0.68, 0.06, 0.0, packLifeMetaGrad(1u, nextSp, acc.contactBestKin, childGene, childGrad)),
                    vec4f(0.0),
                    childMdp,
                    vec4u(pi, 0u, 0u, 0u),
                );
            }
        }

        // 0C. Neutral Abiogenesis / Spore Nucleation (`0 -> 1`): uniform random species + tabula-rasa priors
        let wetLandSpore = (oc.w < 0.46) && (rainWash > 0.12 || landHydration > 0.45) && (tHabK > 274.5) && (elevKm < 3.6);
        if (acc.nearTotalBio < 1.2 && acc.nearLipidCount < 0.5 && (oc.w >= 0.46 || wetLandSpore)) {
            let siteFertility = select(0.50 + upwell, clamp(0.45 + 0.85 * landHydration, 0.25, 1.45), wetLandSpore);
            let pSpore = clamp(0.0034 * q10Bio * max(solar, 0.04) * siteFertility * dtBio, 0.0, 0.0065);
            if (hashFloat(pi, frameU, 13u) < pSpore) {
                let sp = u32(floor(hashFloat(pi, frameU, 14u) * f32(nBio))) % nBio;
                let freshKin = hashFloat(pi, frameU, 15u) * TWO_PI;
                let freshGene = u32(floor(hashFloat(pi, frameU, 16u) * 16.0)) & 15u;
                let initAct = u32(floor(hashFloat(pi, frameU, 18u) * 2.0)) & 1u;
                let sporeMdp = MdpState(getSpeciesMdpPriorPack(nChem, nBio, sp), vec4f(0.68, 0.0, 0.0, packMdpCtrl(MDP_STATE_ABUNDANT, initAct, 8u, 1u)));
                return BioStepResult(
                    vec4f(0.68, 0.06, 0.0, packLifeMeta(1u, sp, freshKin, freshGene)),
                    vec4f(0.0),
                    sporeMdp,
                    vec4u(pi, 0u, 0u, 0u),
                );
            }
        }

        return bioStepOnly(nextFluid, chemForce);
    }

    // Leader election step (for all living `phase == 1u || phase == 2u`)
    var nextLeaderId = pi;
    var nextLeaderHops = 0u;
    if (acc.bestLeaderId < pi && acc.bestLeaderHops <= MAX_LEADER_HOPS) {
        nextLeaderId = acc.bestLeaderId;
        nextLeaderHops = acc.bestLeaderHops;
    }
    let outBlob = vec4u(nextLeaderId, nextLeaderHops, curBlob.z, curBlob.w);

    let myBioSp = min(unpackBioSpecies(curLifeState.w), nBio - 1u);
    let myKin = unpackKinPhase(curLifeState.w);
    let myGene = unpackEncapGene(curLifeState.w);
    let myGrad = unpackGraduated(curLifeState.w);
    let dtVit = dtBio * 1.8;

    // Phase 2: Amphiphilic Lipid Bilayer (`myPhase == 2u`)
    if (myPhase == 2u) {
        let totalWeight = 1.0 + acc.blobWeightSum;
        let avgE = (curLifeState.x + acc.blobEnergySum) / totalWeight;
        let diffAlpha = select(0.0, clamp(4.2 * dtBio, 0.0, 0.38), acc.blobWeightSum > 0.1);
        let eDiffused = mix(curLifeState.x, avgE, diffAlpha);

        var nextKin = myKin;
        if (acc.blobKinWeight > 0.01) {
            let goalKin = wrapAngle(atan2(acc.blobKinSin, acc.blobKinCos));
            var dAngle = goalKin - myKin;
            if (dAngle > PI) { dAngle -= TWO_PI; }
            else if (dAngle < -PI) { dAngle += TWO_PI; }
            nextKin = wrapAngle(myKin + dAngle * clamp(3.5 * dtBio, 0.0, 0.30));
        }

        var nextGene = myGene;
        if (acc.blobWeightSum > 0.1) {
            let avgG = (f32(myGene) + acc.blobGeneSum) / totalWeight;
            nextGene = u32(clamp(round(mix(f32(myGene), avgG, clamp(2.5 * dtBio, 0.0, 0.25))), 0.0, 15.0));
        }

        let vit = clamp(curLifeState.y - 0.12 * rate * curLifeState.y * dtVit, 0.0, 1.0);
        let poi = clamp(curLifeState.z + (min(acc.dPoi, 1.2) - 0.20 * curLifeState.z) * dtVit, 0.0, 1.0);

        let lipHealthy = (acc.blobCount >= 1.0 || acc.blobLipidCount >= 2.0);
        let gradNow = (myGrad == 1u && lipHealthy && (acc.blobGradCount >= 1.5 || acc.blobCount >= 8.0)) ||
            (acc.blobGradCount >= 3.0 && acc.blobCount >= 16.0);
        let heatLipidStress = (smoothstep(307.0, 323.0, tHabK) * tBlend + 0.65 * oceanBleachBoost) * 0.09;
        let coldLipidStab = mix(1.0, 0.78, smoothstep(278.0, 266.0, tHabK) * tBlend);
        let upkeep = select(0.45, select(0.055, 0.036, gradNow), lipHealthy);
        let poisonPenalty = 0.24 * poi;
        let landPenalty = 0.045 * landFrac * mix(1.0, clamp(0.28 + 1.35 * aridStress - 0.30 * landHydration, 0.16, 2.1), hBlend);
        let nextEnergy = clamp(eDiffused - (upkeep + poisonPenalty + landPenalty + heatLipidStress) * dtVit, 0.02, 1.45);
        let hostIsGrad = gradNow && (nextEnergy >= 0.36) && (poi < 0.72);

        let orphanStress = (acc.blobCount < 0.5 && nextEnergy < 0.45);
        if (nextEnergy <= 0.14 || poi >= 0.85 || orphanStress) {
            if (hashFloat(pi, frameU, 31u) < clamp(0.65 * coldLipidStab * dtBio, 0.0, 0.18)) {
                var cause = DEATH_MEMBRANE;
                if (poi >= 0.85) { cause = DEATH_POISON; }
                else if (nextEnergy <= 0.14) {
                    let stress = vec4f(poisonPenalty, heatLipidStress, 0.0, landPenalty);
                    cause = energyDeath(eDiffused - (upkeep + dot(stress, vec4f(1.0))) * dtVit, eDiffused, 0.14, dtVit, 0.0, stress);
                }
                return bioStepOnly(
                    deadCell(cause),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.9, -0.7)),
                );
            }
        }
        if (acc.contactEngulfPress > 0.35) {
            let pBreach = clamp(0.24 * acc.contactEngulfPress * select(1.0, 0.45, hostIsGrad) * dtBio, 0.0, 0.06);
            if (hashFloat(pi, frameU, 32u) < pBreach) {
                return bioStepOnly(
                    deadCell(DEATH_PREDATION),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.85, -0.8)),
                );
            }
        }

        return BioStepResult(
            vec4f(nextEnergy, vit, poi, packLifeMetaGrad(2u, myBioSp, nextKin, nextGene, select(0u, 1u, hostIsGrad))),
            vec4f(0.0),
            emptyMdpState(),
            outBlob,
        );
    }

    // Phase 1: Living Cytoplasm (`myPhase == 1u`)
    let curVel = evalOceanCurrentVel(pos, oc, bio.oceanCurrentScale);
    let windFlow = select(windSim * (0.31 * landFrac), windSim * (0.21 * smoothstep(0.35, 0.55, oc.w)), oc.w >= 0.35);
    let mySelfSpeed = length(vel - (curVel + windFlow));
    let motility = clamp(mySelfSpeed / 0.10, 0.0, 1.5);
    let curPol = evalCombinedMdpPolicy(curBlob.z);

    let nMembers = 1.0 + acc.blobCount;
    let avgBlobE = (curLifeState.x + acc.blobEnergySum) / nMembers;
    let avgBlobVit = (curLifeState.y + acc.blobVitSum) / nMembers;
    let mySpeciesMask = acc.blobSpeciesMask | (1u << (myBioSp % 24u));
    let myDistinctColors = countOneBits(mySpeciesMask);
    let membraneShield = clamp(acc.blobLipidCount / 6.0, 0.0, 1.0);
    let lipPresent = (acc.blobLipidCount >= 0.8) || (acc.blobLipidGradCount >= 0.5) || (acc.blobGradCount >= 4.0);
    let isGradMaintain = (myGrad == 1u) && (acc.blobCount >= 14.0) && (curLifeState.x >= 0.38) && (curLifeState.z < 0.65) && lipPresent;
    let isGradSpread = (acc.blobGradCount >= 2.5) && (acc.blobCount >= 18.0) && (avgBlobE >= 0.60) && (curLifeState.z < 0.40) && ((acc.blobLipidCount >= 1.0) || (acc.blobGradCount >= 5.0));
    let isGradEmergent = (acc.blobLipidCount >= 4.0) && (acc.blobCount >= 16.0) && (myDistinctColors >= 5u) && (avgBlobE >= 0.72) && (curLifeState.z < 0.25);
    let isGradNow = isGradMaintain || isGradSpread || isGradEmergent;
    let nextGene = select(
        myGene,
        u32(clamp(round(mix(f32(myGene), (f32(myGene) + acc.blobGeneSum) / nMembers, clamp(2.5 * dtBio, 0.0, 0.25))), 0.0, 15.0)),
        acc.blobCount > 0.0,
    );

    let normChem = select(0.0, 1.0 / (1.0 + 0.35 * max(acc.bioChemCount - 1.0, 0.0)), acc.bioChemCount > 0.0);
    let vitRaw = curLifeState.y + (acc.dVit * normChem + 0.03 * q10Bio * solar - 0.12 * rate * curLifeState.y) * dtVit;
    let vitDiffAlpha = select(0.0, clamp(2.4 * dtBio, 0.0, 0.26), acc.blobCount > 0.0);
    let vit = clamp(mix(vitRaw, avgBlobVit, vitDiffAlpha), 0.0, 1.0);
    let detoxRate = 0.22 + 0.48 * vit;
    let poiIn = acc.dPoi * normChem * (1.0 - 0.72 * membraneShield);
    let poi = clamp(curLifeState.z + (poiIn - detoxRate * curLifeState.z) * dtVit, 0.0, 1.0);

    // Intra-blob energy & kin phase synchronization
    let diffAlpha = select(0.0, clamp(2.8 * dtBio, 0.0, 0.28), acc.blobCount > 0.0);
    let eDiffused = mix(curLifeState.x, avgBlobE, diffAlpha);

    var nextKin = myKin;
    if (acc.blobKinWeight > 0.01) {
        let goalKin = wrapAngle(atan2(acc.blobKinSin, acc.blobKinCos));
        var dAngle = goalKin - myKin;
        if (dAngle > PI) { dAngle -= TWO_PI; }
        else if (dAngle < -PI) { dAngle += TWO_PI; }
        nextKin = wrapAngle(myKin + dAngle * clamp(3.0 * dtBio, 0.0, 0.25));
    } else {
        let jitter = (hashFloat(pi, frameU, 19u) * 2.0 - 1.0) * 0.004;
        nextKin = wrapAngle(myKin + jitter);
    }

    // Pure physical environmental energetics (no hard-coded producer vs consumer traits)
    let rawLandMoisture = clamp(
        0.10 + 0.62 * smoothstep(0.22, 0.90, rh) + 0.55 * rainWash + 0.28 * seaFogCondense - 0.28 * aridStress + 0.16 * membraneShield,
        0.06,
        1.25,
    );
    let moistureFactor = mix(1.0, mix(1.0, rawLandMoisture, hBlend), landFrac);
    let carbWaterBoost = clamp(0.84 + 0.14 * min(acc.nearCarbonFeed + acc.nearWaterFluid, 2.5), 0.80, 1.25);
    let o2RespireBoost = clamp(0.76 + 0.16 * min(acc.nearOxygenGas + ridgeVentMask * 1.4, 2.5), 0.72, 1.25);
    let coastalMoisture = smoothstep(0.08, 0.45, oc.w) * 0.42;
    let landFertilityScale = clamp(mix(1.0, rawLandMoisture + coastalMoisture, hBlend), 0.08, 1.32) * (1.0 - 0.34 * smoothstep(2.2, 5.2, elevKm));
    let calmFactor = 1.0 - clamp(motility / 0.45, 0.0, 1.0);
    let calmFixBoost = 1.0 + 0.48 * curPol.fixation * calmFactor;
    let lightShade = 1.0 / (1.0 + 0.068 * max(acc.nearTotalBio - 8.0, 0.0));
    let chemReceptor = clamp(curPol.dChemNutr, 0.05, 1.0);
    let huntSweep = clamp(curPol.dForeignBio * 0.5, 0.0, 0.65);
    let swimSweep = 1.0 + 0.82 * clamp(motility, 0.0, 1.4) * max(chemReceptor, huntSweep);

    let solarElem = elemSolarWeight(atomicNum);
    let redoxElem = elemRedoxWeight(atomicNum);
    let solarGain = 0.52 * q10Bio * solar * lightShade * (0.20 + 0.85 * solarElem) * mix(0.55 + 0.75 * upwell, landFertilityScale, landFrac) * moistureFactor * carbWaterBoost * calmFixBoost;
    let engulfGain = 1.55 * min(acc.contactPreyHarvest, 2.5) * o2RespireBoost;
    let detritusGain = 0.28 * min(acc.nearStressedBio, 2.5) * (0.25 + 0.75 * redoxElem) * o2RespireBoost * swimSweep;
    let chemBonus = 0.34 * acc.dE * normChem * q10Bio * swimSweep;

    let coldTorpor = mix(1.0, clamp(0.58 + 0.42 * smoothstep(270.0, 297.0, tHabK), 0.48, 1.15), tBlend);
    let feedDrive = clamp(max(curPol.dChemNutr, curPol.dForeignBio), 0.0, 1.0);
    let panicCost = 0.068 * curPol.swimThrust * curPol.swimThrust * (1.0 - feedDrive);
    let basalBurn = (0.11 + 0.012 * motility + 0.016 * curPol.swimThrust * curPol.swimThrust + panicCost + 0.016 * curPol.lipidAlloc)
        * coldTorpor * (1.0 - 0.16 * membraneShield - 0.18 * select(0.0, 1.0, isGradNow));

    // Pure physical steric crowding (no arbitrary own-blob cap or foreign bonus tricks)
    let crowdPenalty = 0.022 * max(0.0, acc.nearTotalBio - 26.0) * (1.0 - 0.45 * membraneShield);
    let poisonPenalty = 0.26 * poi;
    let heatShock = smoothstep(305.0, 321.0, tHabK) * tBlend + 0.55 * oceanBleachBoost;
    let frostShock = smoothstep(271.0, 249.0, tHabK) * tBlend;
    let thermStressPenalty = (heatShock * 0.20 + frostShock * 0.14) * (1.0 - 0.62 * membraneShield);
    let uvStressPenalty = 0.14 * uvStress * (1.0 - 0.72 * membraneShield);
    let oceanEvapOsmotic = max(0.0, seaEvapFlux - precipMm * 0.25) * smoothstep(0.36, 0.54, oc.w) * 0.018 * hBlend;
    let landPenalty = (0.055 * landFrac * mix(1.0, clamp(0.24 + 1.35 * aridStress - 0.35 * landHydration, 0.10, 2.0), hBlend) + oceanEvapOsmotic)
        * (1.0 - 0.68 * membraneShield) * (1.0 - 0.28 * curPol.fixation);

    let netDE = solarGain + engulfGain + detritusGain + 0.06 * vit + chemBonus
        - basalBurn - crowdPenalty - poisonPenalty - landPenalty - thermStressPenalty - uvStressPenalty;
    let nextEnergy = clamp(eDiffused + netDE * dtVit, 0.04, 1.85);

    // -------------------------------------------------------------------------
    // Particle-Level MDP (Fast Local Reflex in {0..3} -> Action in {0..3})
    // -------------------------------------------------------------------------
    let priorQPack = getSpeciesMdpPriorPack(nChem, nBio, myBioSp);
    let mdpBits = u32(round(curMdp.ctrl.w));
    let mdpInit = (mdpBits & 4096u) != 0u;
    var qPack = select(priorQPack, curMdp.qPack, mdpInit);
    let prevState = select(MDP_STATE_ABUNDANT, mdpBits & 3u, mdpInit);
    let prevAction = select(u32(floor(hashFloat(pi, frameU, 45u) * 2.0)) & 1u, (mdpBits >> 2u) & 3u, mdpInit);
    var prevDwell = select(10u, (mdpBits >> 4u) & 255u, mdpInit);
    let prevMdpE = select(curLifeState.x, curMdp.ctrl.x, mdpInit);
    let prevMdpPoi = select(curLifeState.z, curMdp.ctrl.y, mdpInit);

    var curState: u32 = MDP_STATE_ABUNDANT;
    if (poi > 0.18 || acc.nearHalogen > 0.30 || acc.contactEngulfPress > 0.18) {
        curState = MDP_STATE_HAZARD;
    } else if (avgBlobE < 0.56 || eDiffused < 0.50) {
        curState = MDP_STATE_LOW_ENERGY;
    } else if (solar < 0.16 || acc.nearTotalBio > 42.0) {
        curState = MDP_STATE_DARK_CROWDED;
    }

    // Lateral Q-gossip across connected same-blob peers (pure online diffusion without resetting to priors)
    let hasGossip = acc.bestMdpScore > -0.5;
    let gossipAlpha = clamp(2.2 * dtBio, 0.0, 0.18);
    let activeMyRow = getQRow(qPack, curState);
    let activePeerRow = select(activeMyRow, getQRow(acc.bestMdpQPack, curState), hasGossip);
    var blendedActiveRow = 0u;
    for (var aa = 0u; aa < 4u; aa++) {
        let qSelf = unpackMdpQ(activeMyRow, aa);
        let qPeer = unpackMdpQ(activePeerRow, aa);
        let qStep = clamp(mix(qSelf, qPeer, select(0.0, gossipAlpha, hasGossip)), -1.45, 1.45);
        blendedActiveRow |= quantizeMdpQ(qStep) << (aa * 8u);
    }
    qPack = setQRow(qPack, curState, blendedActiveRow);

    let curStateRow = getQRow(qPack, curState);
    let voteDenom = max(acc.mdpVoteWeight, 1e-4);
    let votesNorm = acc.mdpVotes / voteDenom;
    let voteStrength = clamp(acc.mdpVoteWeight / 2.2, 0.0, 1.0) * 0.32;

    var bestAct: u32 = prevAction;
    var bestScore: f32 = -1e9;
    var maxNextQ: f32 = -1.5;
    for (var aa = 0u; aa < 4u; aa++) {
        let qVal = unpackMdpQ(curStateRow, aa);
        if (qVal > maxNextQ) {
            maxNextQ = qVal;
        }
        var voteFrac = votesNorm.x;
        if (aa == 1u) { voteFrac = votesNorm.y; }
        else if (aa == 2u) { voteFrac = votesNorm.z; }
        else if (aa == 3u) { voteFrac = votesNorm.w; }

        let dwellBonus = select(0.0, 0.14, aa == prevAction && prevDwell > 0u);
        let epsNoise = (hashFloat(pi, frameU, 41u + aa) - 0.5) * 0.12;
        let candScore = qVal + voteStrength * voteFrac + dwellBonus + epsNoise;
        if (candScore > bestScore) {
            bestScore = candScore;
            bestAct = aa;
        }
    }

    let nextAction = bestAct;
    let nextDwell = select(select(0u, prevDwell - 1u, prevDwell > 0u), 18u, nextAction != prevAction);

    // 1-step Bellman TD update + real scalar reward passed to live Species MDP
    let dERate = (nextEnergy - prevMdpE) / max(dtVit, 0.015);
    let dPoiRate = (poi - prevMdpPoi) / max(dtVit, 0.015);
    let hazardPen = poi * 0.55 + acc.nearHalogen * 0.30 + acc.contactEngulfPress * 0.85;
    let rewardMdp = clamp(
        netDE * 1.45 + dERate * 0.16 - dPoiRate * 0.28 - hazardPen + select(0.0, 0.12, nextEnergy > 0.78),
        -1.4,
        1.4,
    );
    let alphaQ = clamp(2.4 * dtBio, 0.02, 0.16);
    let prevRow = getQRow(qPack, prevState);
    let oldQ = unpackMdpQ(prevRow, prevAction);
    let updatedQ = clamp(mix(oldQ, rewardMdp + 0.84 * maxNextQ, alphaQ), -1.45, 1.45);
    qPack = setQRow(qPack, prevState, setMdpQ(prevRow, prevAction, updatedQ));

    // Store immediate scalar reward in `.z` for `blobScatterPass` -> Species MDP TD update
    let nextMdp = MdpState(qPack, vec4f(nextEnergy, poi, rewardMdp, packMdpCtrl(curState, nextAction, nextDwell, 1u)));

    // Telemetry packed in `bioForceOut[pi]` (ZERO directional steering vectors!)
    let sensTelemetry = vec4f(solar, min(acc.nearForeignBio, 8.0), min(acc.nearHalogen, 8.0), clamp(netDE, -4.0, 4.0));

    // Pure Physical Close-Contact Engulfment (`r < 0.44 * catRadius`)
    if (acc.contactEngulfPress > 0.32) {
        let haloShield = 1.0 / (1.0 + 1.2 * acc.nearHalogen);
        let pAbsorb = clamp(
            0.44 * min(acc.contactEngulfPress, 3.0) * haloShield * (1.0 - 0.22 * vit) * (1.0 - 0.52 * membraneShield) * select(1.0, 0.58, isGradNow) * dtBio,
            0.0,
            0.10,
        );
        if (hashFloat(pi, frameU, 20u) < pAbsorb) {
            let outcome = hashFloat(pi, frameU, 21u);
            if (outcome < 1.0 - 0.40 * acc.contactBestEngulfAlloc) {
                let engSp = min(acc.contactBestEngulfSp, nBio - 1u);
                let fedE = clamp(max(0.92, max(curLifeState.x, acc.contactBestEngulfE) * 0.50 + 0.54), 0.92, 1.42);
                let fedVit = clamp(max(vit, 0.25) + 0.40, 0.0, 1.0);
                let preyGrad = select(0u, 1u, acc.contactBestEngulfGrad == 1u);
                let inheritedMdp = MdpState(qPack, vec4f(fedE, poi * 0.6, rewardMdp, packMdpCtrl(curState, nextAction, nextDwell, 1u)));
                return BioStepResult(
                    vec4f(fedE, fedVit, poi * 0.6, packLifeMetaGrad(1u, engSp, acc.contactBestEngulfKin, acc.contactBestEngulfGene, preyGrad)),
                    sensTelemetry,
                    inheritedMdp,
                    outBlob,
                );
            } else {
                return bioStepOnly(
                    deadCell(DEATH_PREDATION),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.8, -0.9)),
                );
            }
        }
    }

    // Natural decomposition / lysis of dying cells (`poi >= 0.90` or starved below decayThreshold)
    let decayThreshold = select(0.075, 0.042, isGradNow || curPol.fixation > 0.35);
    if (nextEnergy <= decayThreshold || poi >= 0.90) {
        let pDecay = clamp(select(0.42, 0.14, isGradNow || curPol.fixation > 0.35) * dtBio, 0.0, 0.14);
        if (hashFloat(pi, frameU, 24u) < pDecay) {
            let depletion = energyDeath(eDiffused + netDE * dtVit, eDiffused, decayThreshold, dtVit, crowdPenalty,
                vec4f(poisonPenalty, thermStressPenalty, uvStressPenalty, landPenalty));
            return bioStepOnly(
                deadCell(select(depletion, DEATH_POISON, poi >= 0.90)),
                vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.9, -0.7)),
            );
        }
    }

    let outGrad = select(0u, 1u, isGradNow);
    var outSp = myBioSp;
    if ((acc.blobLipidCount >= 1.5 || acc.nearClayCat >= 0.65) && nextEnergy > 0.78 && myDistinctColors < 12u && acc.blobCount >= 5.0) {
        let clayInnov = 1.0 + 1.2 * clamp(acc.nearClayCat, 0.0, 1.5);
        let uvInnov = 1.0 + 0.8 * uvStress * (1.0 - 0.55 * membraneShield);
        let pInnov = clamp(0.012 * clayInnov * uvInnov * dtBio, 0.0, 0.012);
        if (hashFloat(pi, frameU, 26u) < pInnov) {
            outSp = pickCompatibleNovelSpecies(nChem, nBio, myBioSp, mySpeciesMask, pi, frameU, 50u);
        }
    }

    return BioStepResult(
        vec4f(nextEnergy, vit, poi, packLifeMetaGrad(1u, outSp, nextKin, nextGene, outGrad)),
        sensTelemetry,
        nextMdp,
        outBlob,
    );
}

struct BioLinkAccum {
    n: vec3f,
    eEast: vec3f,
    eNorth: vec3f,
    b2: vec2f,
    myPhase: u32,
    mySp: u32,
    myKin: f32,
    rConnectAng: f32,
    rSearchAng: f32,

    hasPrev0: bool,
    prevFade0: u32,
    prevCooldown0: u32,
    prevTang0: vec2f,
    prevDir0: vec2f,
    prevLen0: f32,

    hasPrev1: bool,
    prevFade1: u32,
    prevCooldown1: u32,
    prevTang1: vec2f,
    prevDir1: vec2f,
    prevLen1: f32,

    trackTang0: vec2f,
    trackScore0: f32,
    trackTang1: vec2f,
    trackScore1: f32,

    c0Tang: vec2f,
    c0Dir: vec2f,
    c0Score: f32,
    c1Tang: vec2f,
    c1Dir: vec2f,
    c1Score: f32,
}

fn initBioLinkAccum(
    pos: vec3f,
    orient: vec3f,
    myLs: vec4f,
    prevVal: vec4f,
    simRMax: f32,
    cellSize: f32,
    worldSize: f32,
) -> BioLinkAccum {
    let n = normalize(pos);
    let eEast = sphereEastBasis(n);
    let eNorth = normalize(cross(n, eEast));
    let uI = orthoBasis(n, orient);
    let bI = cross(n, uI);
    let b2Raw = vec2f(dot(bI, eEast), dot(bI, eNorth));
    let b2 = select(vec2f(1.0, 0.0), normalize(b2Raw), dot(b2Raw, b2Raw) > 1e-8);
    let R = max(worldSize, 0.25);
    let phase = unpackPhase(myLs.w);
    let capFrac = select(0.19, 0.28, phase == 2u);
    let surfCap = select(0.030, 0.046, phase == 2u);
    let rConnect = min(max(cellSize * capFrac, simRMax * 0.34), surfCap);
    let rDisconnect = rConnect * 1.24;

    var acc: BioLinkAccum;
    acc.n = n;
    acc.eEast = eEast;
    acc.eNorth = eNorth;
    acc.b2 = b2;
    acc.myPhase = phase;
    acc.mySp = unpackBioSpecies(myLs.w);
    acc.myKin = unpackKinPhase(myLs.w);
    acc.rConnectAng = rConnect / R;
    acc.rSearchAng = rDisconnect / R;

    let m0 = u32(max(floor(prevVal.x), 0.0));
    let isBioSlot0 = (m0 & 1024u) != 0u && prevVal.x >= 1024.25;
    acc.prevFade0 = select(0u, min(m0 & 15u, 8u), isBioSlot0);
    acc.prevCooldown0 = select(0u, min((m0 >> 4u) & 31u, 18u), isBioSlot0);
    let candTang0 = select(vec2f(0.0), vec2f(prevVal.x - f32(m0) - 0.5, prevVal.y), isBioSlot0 && acc.prevFade0 > 0u);
    let candLen0 = length(candTang0);
    let validPrev0 = isBioSlot0 && (acc.prevFade0 > 0u) && (candLen0 <= acc.rSearchAng * 1.15);
    acc.hasPrev0 = validPrev0;
    if (!validPrev0) { acc.prevFade0 = 0u; }
    acc.prevTang0 = select(vec2f(0.0), candTang0, validPrev0);
    acc.prevLen0 = select(0.0, candLen0, validPrev0);
    acc.prevDir0 = select(vec2f(0.0), candTang0 / candLen0, validPrev0 && candLen0 > 1e-6);

    let m1 = u32(max(floor(prevVal.z), 0.0));
    let isBioSlot1 = (m1 & 1024u) != 0u && prevVal.z >= 1024.25;
    acc.prevFade1 = select(0u, min(m1 & 15u, 8u), isBioSlot1);
    acc.prevCooldown1 = select(0u, min((m1 >> 4u) & 31u, 18u), isBioSlot1);
    let candTang1 = select(vec2f(0.0), vec2f(prevVal.z - f32(m1) - 0.5, prevVal.w), isBioSlot1 && acc.prevFade1 > 0u);
    let candLen1 = length(candTang1);
    let validPrev1 = isBioSlot1 && (acc.prevFade1 > 0u) && (candLen1 <= acc.rSearchAng * 1.15);
    acc.hasPrev1 = validPrev1;
    if (!validPrev1) { acc.prevFade1 = 0u; }
    acc.prevTang1 = select(vec2f(0.0), candTang1, validPrev1);
    acc.prevLen1 = select(0.0, candLen1, validPrev1);
    acc.prevDir1 = select(vec2f(0.0), candTang1 / candLen1, validPrev1 && candLen1 > 1e-6);

    acc.trackTang0 = vec2f(0.0);
    acc.trackScore0 = 99.0;
    acc.trackTang1 = vec2f(0.0);
    acc.trackScore1 = 99.0;
    acc.c0Tang = vec2f(0.0);
    acc.c0Dir = vec2f(0.0);
    acc.c0Score = 99.0;
    acc.c1Tang = vec2f(0.0);
    acc.c1Dir = vec2f(0.0);
    acc.c1Score = 99.0;
    return acc;
}

fn insertSlimeSecondCandidate(
    acc: ptr<function, BioLinkAccum>,
    tang: vec2f,
    dir2: vec2f,
    sinA: f32,
) {
    if (sinA >= (*acc).rConnectAng) {
        return;
    }
    if ((*acc).c0Score < 90.0 && dot(dir2, (*acc).c0Dir) < 0.40) {
        if (sinA < (*acc).c1Score) {
            (*acc).c1Score = sinA;
            (*acc).c1Tang = tang;
            (*acc).c1Dir = dir2;
        }
    }
}

fn accumulateBioLinkNeighbor(
    ipPos: vec3f,
    nbPhase: u32,
    nbLs: vec4f,
    acc: ptr<function, BioLinkAccum>,
) {
    if (nbPhase != (*acc).myPhase) {
        return;
    }
    let dVec = ipPos - (*acc).n;
    let chord2 = dot(dVec, dVec);
    let maxChord2 = (*acc).rSearchAng * (*acc).rSearchAng;
    if (chord2 <= 1e-11 || chord2 >= maxChord2) {
        return;
    }

    let nbKin = unpackKinPhase(nbLs.w);
    let kinDist = kinAngularDist((*acc).myKin, nbKin);
    let maxKin = select(0.11, 0.14, (*acc).myPhase == 2u);
    if (kinDist >= maxKin) {
        return;
    }

    let tang = vec2f(dot(ipPos, (*acc).eEast), dot(ipPos, (*acc).eNorth));
    let sin2 = dot(tang, tang);
    if (sin2 <= 1e-11) {
        return;
    }
    let sinA = sqrt(sin2);
    if (sinA >= (*acc).rSearchAng) {
        return;
    }
    let dir2 = tang / sinA;

    if ((*acc).hasPrev0 && (*acc).prevLen0 > 1e-6) {
        let dirMatch0 = dot(dir2, (*acc).prevDir0);
        let lenDiff0 = abs(sinA - (*acc).prevLen0);
        if (dirMatch0 > 0.72 && lenDiff0 < (*acc).rConnectAng * 0.65) {
            let tScore0 = lenDiff0 + (1.0 - dirMatch0) * (*acc).rConnectAng;
            if (tScore0 < (*acc).trackScore0) {
                (*acc).trackScore0 = tScore0;
                (*acc).trackTang0 = tang;
            }
        }
    }
    if ((*acc).hasPrev1 && (*acc).prevLen1 > 1e-6) {
        let dirMatch1 = dot(dir2, (*acc).prevDir1);
        let lenDiff1 = abs(sinA - (*acc).prevLen1);
        if (dirMatch1 > 0.72 && lenDiff1 < (*acc).rConnectAng * 0.65) {
            let tScore1 = lenDiff1 + (1.0 - dirMatch1) * (*acc).rConnectAng;
            if (tScore1 < (*acc).trackScore1) {
                (*acc).trackScore1 = tScore1;
                (*acc).trackTang1 = tang;
            }
        }
    }

    if (sinA >= (*acc).rConnectAng) {
        return;
    }

    if ((*acc).myPhase == 2u) {
        let side = dot(dir2, (*acc).b2);
        let absSide = abs(side);
        if (absSide < 0.20 && sinA > (*acc).rConnectAng * 0.50) {
            return;
        }
        let tangBias = 1.85 - 0.85 * absSide;
        let score = sinA * tangBias;
        if (side <= 0.0) {
            if (score < (*acc).c0Score) {
                (*acc).c0Score = score;
                (*acc).c0Tang = tang;
                (*acc).c0Dir = dir2;
            }
        } else {
            if (score < (*acc).c1Score) {
                (*acc).c1Score = score;
                (*acc).c1Tang = tang;
                (*acc).c1Dir = dir2;
            }
        }
        return;
    }

    if (sinA < (*acc).c0Score) {
        let prevTang = (*acc).c0Tang;
        let prevDir = (*acc).c0Dir;
        let prevSin = (*acc).c0Score;
        (*acc).c0Score = sinA;
        (*acc).c0Tang = tang;
        (*acc).c0Dir = dir2;
        if (prevSin < 90.0) {
            insertSlimeSecondCandidate(acc, prevTang, prevDir, prevSin);
        }
    } else {
        insertSlimeSecondCandidate(acc, tang, dir2, sinA);
    }
}

fn stepBioLinkSlot(
    hasPrev: bool,
    prevFade: u32,
    prevCooldown: u32,
    prevTang: vec2f,
    trackValid: bool,
    trackTang: vec2f,
    bestValid: bool,
    bestTang: vec2f,
) -> vec2f {
    var fade = 0u;
    var cooldown = prevCooldown;
    var outTang = vec2f(0.0);

    if (hasPrev) {
        if (trackValid) {
            fade = min(prevFade + 1u, 8u);
            cooldown = select(0u, prevCooldown - 1u, prevCooldown > 0u);
            outTang = trackTang;
        } else if (prevCooldown > 0u && bestValid) {
            fade = min(prevFade + 1u, 8u);
            cooldown = prevCooldown - 1u;
            outTang = bestTang;
        } else {
            fade = select(0u, prevFade - 1u, prevFade > 0u);
            if (fade == 0u) {
                cooldown = 18u;
                outTang = vec2f(0.0);
            } else {
                cooldown = select(0u, prevCooldown - 1u, prevCooldown > 0u);
                outTang = prevTang;
            }
        }
    } else {
        if (prevCooldown > 0u) {
            cooldown = prevCooldown - 1u;
            fade = 0u;
        } else if (bestValid) {
            fade = 1u;
            cooldown = 18u;
            outTang = bestTang;
        } else {
            cooldown = 0u;
            fade = 0u;
        }
    }

    let clampedCd = min(cooldown, 18u);
    let clampedFade = min(fade, 8u);
    let metaU = 1024u | (clampedCd << 4u) | clampedFade;
    let encX = select(f32(metaU) + 0.5, f32(metaU) + clamp(outTang.x, -0.08, 0.08) + 0.5, clampedFade > 0u);
    let encY = select(0.0, outTang.y, clampedFade > 0u);
    return vec2f(encX, encY);
}

fn finalizeBioLink(acc: BioLinkAccum) -> vec4f {
    let slot0 = stepBioLinkSlot(
        acc.hasPrev0,
        acc.prevFade0,
        acc.prevCooldown0,
        acc.prevTang0,
        acc.trackScore0 < 90.0,
        acc.trackTang0,
        acc.c0Score < 90.0,
        acc.c0Tang,
    );
    let slot1 = stepBioLinkSlot(
        acc.hasPrev1,
        acc.prevFade1,
        acc.prevCooldown1,
        acc.prevTang1,
        acc.trackScore1 < 90.0,
        acc.trackTang1,
        acc.c1Score < 90.0,
        acc.c1Tang,
    );
    return vec4f(slot0, slot1);
}
