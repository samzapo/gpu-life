// Unified conserved-mass emergent blob ecology + biogenic lipid bilayers + inorganic mineral crust.
// Every particle carries an immutable `chemSpecies` (`p.colour`), live `phase`
// (0 = Fluid Chemistry, 1 = Organism Cytoplasm, 2 = Biogenic Lipid Bilayer, 3 = Crust-Pinned Mineral),
// `bioSpecies` (0..M-1) / `mineralType` (0..4), circular `kinPhase` / `crystalAngle`, and 4-bit `encapGene` / `grade` in `lifeState.w`.

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
}

fn isBiogenicElement(atomicNum: u32, valence: u32) -> bool {
    if (valence == 0u) {
        return false;
    }
    // Biogenic pool: H(1), C(5), O(6), S(12). Si(11), halogens, metals, noble gases remain strictly abiotic.
    return atomicNum == 1u || atomicNum == 5u || atomicNum == 6u || atomicNum == 12u;
}

fn isLipidElement(atomicNum: u32, valence: u32) -> bool {
    if (valence == 0u) {
        return false;
    }
    // Membrane lipids assemble strictly from conserved H(1), C(5), O(6).
    return atomicNum == 1u || atomicNum == 5u || atomicNum == 6u;
}

fn isMineralCandidate(atomicNum: u32, valence: u32) -> bool {
    // Noble gases (valence == 0: He, Ne, Ar) never crystallize into crust minerals.
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
    // MINERAL_CLAY (Hydrated phyllosilicates / zeolites): traps Si, O, H, Mg, C
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

fn encapReqColors(gene: u32) -> u32 {
    if (gene <= 2u) {
        return 3u;
    }
    if (gene <= 7u) {
        return 4u;
    }
    if (gene <= 12u) {
        return 5u;
    }
    return 6u;
}

fn encapReqSize(gene: u32) -> f32 {
    if (gene <= 4u) {
        return 6.0 + 0.5 * f32(gene);
    }
    if (gene <= 10u) {
        return 8.5 + 0.65 * f32(gene - 4u);
    }
    return 12.5 + 0.9 * f32(gene - 10u);
}

fn encapNoveltyDrive(gene: u32, currentColors: u32) -> f32 {
    let g = f32(min(gene, 15u)) / 15.0;
    let goalColors = 5.0 + g * 4.0;
    return 0.10 + 0.24 * clamp((goalColors - f32(currentColors)) / 4.0, 0.0, 1.0);
}

fn mutateEncapGene(baseGene: u32, pi: u32, frameU: u32, salt: u32, pMut: f32) -> u32 {
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

fn elemSolarWeight(atomicNum: u32) -> f32 {
    if (atomicNum == 5u) {
        return 1.0; // C
    }
    if (atomicNum == 6u) {
        return 0.80; // O
    }
    if (atomicNum == 1u) {
        return 0.25; // H
    }
    if (atomicNum == 12u) {
        return 0.15; // S
    }
    return 0.0;
}

fn elemRedoxWeight(atomicNum: u32) -> f32 {
    if (atomicNum == 12u) {
        return 1.0; // S
    }
    if (atomicNum == 1u) {
        return 0.85; // H
    }
    if (atomicNum == 5u) {
        return 0.25; // C
    }
    if (atomicNum == 6u) {
        return 0.10; // O
    }
    return 0.0;
}

// Computes photosynthetic/evaporative UV irradiance (.x) and live UV-B/A mutagenesis & photodamage stress (.y)
// from NOAA CPC noontime UVI + live GFS Ozone (TOZNE) + Cloud Cover (TCDC) + UTC Solar Zenith `(uviNow, uviNoon, ozoneDU, cloudFrac)`.
fn evalUvSolarAndStress(uvSample: vec4f, localUvScale: f32) -> vec2f {
    let sc = clamp(localUvScale, 0.0, 2.0);
    if (sc <= 0.001) {
        return vec2f(0.68, 0.0);
    }
    let uviNow = clamp(uvSample.x, 0.0, 24.0);
    let uviNoon = clamp(uvSample.y, 0.0, 24.0);
    let effUvi = 0.42 * uviNoon + 0.58 * uviNow;
    let normU = effUvi / 10.0;
    // Michaelis-Menten P-I saturation: polar winter (~0 UVI) stays dormant; cloudy high-lats stay modest; sunlit tropics saturate ~1.14
    let solarRaw = clamp(0.05 + 1.56 * (normU / (normU + 0.48)), 0.05, 1.14);
    let blend = min(sc, 1.5);
    let solar = mix(0.68, solarRaw, clamp(blend, 0.0, 1.0));
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

// Returns vec4f(coh, chase, flee, asym) derived dynamically from random matrix `A`.
fn getBioSpeciesTraits(nChem: u32, nBio: u32, sp: u32) -> vec4f {
    let oceanBase = nChem * SPECIES_STRIDE + nChem * nChem;
    let traitBase = oceanBase + max(nBio, 1u) * max(nBio, 1u);
    let idx = traitBase + min(sp, max(nBio, 1u) - 1u) * 4u;
    return vec4f(matrix[idx + 0u], matrix[idx + 1u], matrix[idx + 2u], matrix[idx + 3u]);
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
        let compat = (aHC > 0.12) || (abs(aHC - aCH) > 0.42 && (aHC + aCH) > -0.22);
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

// Computes `(dEnergy, dVitamin, dPoison, signedSteerMag)` imparted to one living Biomass particle
// by one nearby Chemistry atom from frozen time-t state using emergent traits.
fn evalLifeFromAtomPair(
    traits: vec4f,
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
    let coh = clamp(traits.x, 0.0, 1.0);
    let asym = clamp(traits.w, 0.0, 1.0);
    let energy = clamp(lifeState.x, 0.0, 1.5);
    let vit = clamp(lifeState.y, 0.0, 1.0);
    let haloAffinity = clamp(0.20 + 0.35 * coh + 0.35 * asym + 0.25 * vit, 0.0, 1.0);

    var dE = 0.0;
    var dVit = 0.0;
    var dPoi = 0.0;
    var steer = 0.0;

    if (kind == 3u) {
        dVit += 0.24 * w;
        steer += 0.20 * w;
    } else if (kind == 4u) {
        if (haloAffinity > 0.62) {
            dE += 0.10 * haloAffinity * w;
            steer += 0.28 * w;
        } else {
            let resist = 0.45 + 0.55 * (1.0 - haloAffinity);
            dPoi += 0.36 * w * (1.0 - 0.50 * vit) * resist;
            dE -= 0.12 * w * (1.0 - 0.35 * vit) * resist;
            steer -= 0.70 * w * resist;
        }
    } else if (kind == 1u) {
        let fuelReady = clamp(atomLoad + 0.35, 0.35, 1.5);
        if (energy < 1.40) {
            let capGain = (0.18 + 0.10 * vit) * fuelReady * (0.65 + 0.35 * asym) * w;
            dE += capGain;
        }
        steer += 0.28 * w;
    } else if (kind == 2u) {
        let foodRich = clamp(atomLoad + 0.25, 0.25, 2.2);
        dE += (0.12 + 0.14 * asym) * foodRich * w * (1.0 + 0.25 * vit);
        steer += (0.25 + 0.30 * coh) * w;
    } else if (atomicNum == 6u) {
        if (atomLoad > 0.05 && atomLoad < 1.35) {
            dE += 0.09 * w;
        } else if (coh > 0.45 && energy > 0.40 && atomLoad >= 1.35) {
            dE -= 0.04 * w;
        }
    }

    return vec4f(dE, dVit, dPoi, steer);
}

// Computes `(radialKickScalar, tangentialSpinScalar, actScale, glow)` imparted to one Chemistry atom
// by one nearby living Biomass particle (`dToNb` points atom -> life on tangent plane).
fn evalAtomFromLifePair(
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    nbTraits: vec4f,
    nbLifeState: vec4f,
    r: f32,
    bio: BioLink,
) -> vec4f {
    let kind = bioRoleKind(atomicNum, valence);
    if (kind == 0u || r >= bio.catRadius) {
        return vec4f(0.0, 0.0, 1.0, 0.0);
    }

    let w = (1.0 - r / max(bio.catRadius, 1e-4)) * bio.rate;
    let coh = clamp(nbTraits.x, 0.0, 1.0);
    let asym = clamp(nbTraits.w, 0.0, 1.0);
    let energy = clamp(nbLifeState.x, 0.0, 1.5);
    let vit = clamp(nbLifeState.y, 0.0, 1.0);
    let catPower = clamp(0.25 + energy * 0.75 + vit * 0.35, 0.0, 1.5);
    let isAnabolic = coh >= asym;

    var kickToLife = 0.0;
    var spinSwirl = 0.0;
    var actScale = 1.0;
    var glow = 0.0;

    if (kind == 3u) {
        actScale = 0.45;
        glow = 0.35 * w;
    } else if (kind == 4u && (coh + vit) > 0.70 && energy > 0.24) {
        kickToLife = 0.20 * catPower * w;
        actScale = 0.22;
        glow = -0.60 * w;
    } else if (kind == 1u) {
        if (isAnabolic && energy > 0.28) {
            kickToLife = 0.30 * catPower * w;
            actScale = 0.16;
            glow = 0.72 * w;
        } else if (energy < 1.40) {
            kickToLife = -0.40 * w;
            spinSwirl = 0.52 * w;
            actScale = 0.35;
            glow = -0.72 * w;
        }
    } else if (kind == 2u) {
        if (isAnabolic && energy > 0.28) {
            kickToLife = 0.38 * catPower * w;
            actScale = 0.12;
            glow = 0.96 * w;
        } else if (atomLoad > 0.35) {
            kickToLife = -0.35 * w;
            spinSwirl = 0.62 * w;
            glow = -0.92 * w;
        }
    } else if (atomicNum == 6u) {
        if (isAnabolic && energy > 0.40 && atomLoad >= 1.35) {
            kickToLife = -0.60 * catPower * w;
            spinSwirl = 0.42 * w;
            glow = -0.70 * w;
        } else if (atomLoad > 0.05 && atomLoad < 1.35) {
            actScale = 0.28;
            kickToLife = 0.16 * w;
            glow = 0.42 * w;
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
    nearTotalBio: f32,
    nearStressedBio: f32,
    // Chemistry-side emergent autotrophic fixer + lipid bilayer accumulation
    nearFixer: f32,
    bestFixW: f32,
    bestFixSp: u32,
    bestFixKin: f32,
    bestFixGene: u32,
    bestAdjW: f32,
    bestAdjSp: u32,
    bestAdjKin: f32,
    adjSpeciesMask: u32,
    adjBlobCount: f32,
    adjEnergySum: f32,
    adjGradCount: f32,
    adjGeneSum: f32,
    adjKinCos: f32,
    adjKinSin: f32,
    adjKinW: f32,
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
    // Same-blob organism anatomy, heterogeneity mask, lipids, & gene sharing
    blobCount: f32,
    blobWeightSum: f32,
    blobEnergySum: f32,
    blobVitSum: f32,
    blobCohSum: f32,
    blobAsymSum: f32,
    blobSolarAtomSum: f32,
    blobRedoxAtomSum: f32,
    blobSpeedSum: f32,
    blobKinCos: f32,
    blobKinSin: f32,
    blobKinWeight: f32,
    blobSpeciesMask: u32,
    blobLipidCount: f32,
    blobLipidGradCount: f32,
    blobGradCount: f32,
    blobGeneSum: f32,
    // Foreign-blob grazing & predation
    nearForeignPrey: f32,
    nearPredThreat: f32,
    bestPredW: f32,
    bestPredSp: u32,
    bestPredKin: f32,
    bestPredGene: u32,
    bestPredGrad: u32,
    bestPredE: f32,
    predRepelDir: vec3f,
    // Emergent saprotrophic decomposition
    nearDecompPower: f32,
    bestDecompW: f32,
    bestDecompSp: u32,
    bestDecompKin: f32,
    bestDecompGene: u32,
    // Inorganic mineral crust accumulation & bio-geochemical exchange
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
}

fn initBioUnifiedAccum() -> BioUnifiedAccum {
    var acc = BioUnifiedAccum();
    acc.actScale = 1.0;
    acc.nearMinDist = 99.0;
    acc.bestFixGene = 7u;
    acc.bestLipidGene = 7u;
    acc.bestPredGene = 7u;
    acc.bestDecompGene = 7u;
    return acc;
}

fn accumulateBioUnifiedNeighbor(
    myPos: vec3f,
    myVel: vec3f,
    myPhase: u32,
    myAtomicNum: u32,
    myValence: u32,
    myAtomLoad: f32,
    myLifeState: vec4f,
    nbPos: vec3f,
    nbVel: vec3f,
    nbChemSpecies: f32,
    nbPhase: u32,
    nbBioSpecies: u32,
    nbLifeState: vec4f,
    nbAtomLoad: f32,
    nbAtomicNum: u32,
    nbValence: u32,
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
    let myTraits = getBioSpeciesTraits(nChem, nBio, myBioSp);
    let nbTraits = getBioSpeciesTraits(nChem, nBio, nbSp);

    if (nbPhase == 0u) {
        let nbKind = bioRoleKind(nbAtomicNum, nbValence);
        if (nbKind == 4u) {
            (*acc).nearHalogen += wDist;
        }
        if (myPhase == 0u) {
            // Geochemical stoichiometric precursor clustering for primary mineral nucleation
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
            let eff = evalLifeFromAtomPair(myTraits, myLifeState, nbAtomicNum, nbValence, nbAtomLoad, r, bio);
            (*acc).dE += eff.x;
            (*acc).dVit += eff.y;
            (*acc).dPoi += eff.z;
            (*acc).steer += dToNb * eff.w;
            (*acc).bioChemCount += 1.0;

            // Gentle photosynthetic scavenging when cohesive autotroph meets free biogenic atoms
            if (isBiogenicElement(nbAtomicNum, nbValence) && myTraits.x > 0.45) {
                (*acc).bioSteer += dToNb * (wDist * 0.14);
            }
        } else if (myPhase == 2u) {
            // Amphiphilic bilayer intercepts dissolving halogens on behalf of sheltered cytoplasm
            if (nbKind == 4u) {
                (*acc).dPoi += 0.30 * wDist * bio.rate;
            }
        }
        return;
    }

    // Neighbor is an anchored crustal mineral (`nbPhase == 3u`)
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
            // Crystal spacing guide: repel away from core overlap so epitaxial buds attach cleanly onto outer branches
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

        // Living cytoplasm (`1u`) or bilayer (`2u`) encountering crustal mineral (`nbPhase == 3u`): Windfall vs Death Sentence
        let minScale = max(bio.mineralRate, 0.20) * clamp(0.35 + 0.85 * nbInteg, 0.25, 1.2);
        let mw = wDist * minScale;
        (*acc).bioChemCount += 1.0;

        if (myPhase == 2u) {
            // Lipid bilayer shields internals from brine/Be/acid mineral leaching at modest shell stress
            if (nbMinTrap == 1u || nbMinType == MINERAL_EVAPORITE) {
                (*acc).dPoi += (0.18 + 0.24 * nbHaz) * mw;
            }
            return;
        }

        let coh = clamp(myTraits.x, 0.0, 1.0);
        let asym = clamp(myTraits.w, 0.0, 1.0);
        let vit = clamp(myLifeState.y, 0.0, 1.0);
        let haloAff = clamp(0.20 + 0.35 * coh + 0.35 * asym + 0.25 * vit, 0.0, 1.0);
        let myRedox = elemRedoxWeight(myAtomicNum);

        if (nbMinType == MINERAL_SILICATE) {
            // Stable reef substrate: mild halogen detox & shelter from fast motion abrasion
            (*acc).dPoi -= 0.09 * mw;
            (*acc).dVit += 0.05 * mw;
            (*acc).dE -= 0.04 * clamp(length(myVel) * 2.0, 0.0, 1.0) * mw;
        } else if (nbMinType == MINERAL_EVAPORITE) {
            // Water-soluble salt / fluorite: lethal halogen & osmotic shock unless halotolerant
            if (haloAff > 0.56) {
                (*acc).dE += (0.24 + 0.16 * nbNutr) * haloAff * mw;
                (*acc).steer += dToNb * (0.28 * mw);
            } else {
                let shock = (0.42 + 0.48 * max(nbHaz, select(0.25, 0.70, nbMinTrap == 1u))) * mw;
                (*acc).dPoi += shock;
                (*acc).dE -= shock * 0.58;
                (*acc).steer -= dToNb * (0.52 * mw);
            }
        } else if (nbMinType == MINERAL_CARBONATE) {
            if (nbMinTrap == 0u) {
                // Pure Dolomite/Carbonate: rich C/O energy & cofactor windfall
                (*acc).dE += (0.28 + 0.24 * nbNutr) * mw;
                (*acc).dVit += 0.22 * mw;
                (*acc).steer += dToNb * (0.32 * mw);
            } else {
                // Be-doped Beryl Carbonate Trap: attracts cells briefly then unleashes toxic beryllium shock
                (*acc).dPoi += (0.56 + 0.48 * nbHaz) * mw;
                (*acc).dE -= (0.24 + 0.26 * nbHaz) * mw;
                (*acc).steer -= dToNb * (0.35 * mw);
            }
        } else if (nbMinType == MINERAL_SULFIDE) {
            let overWater = smoothstep(0.32, 0.54, oc.w);
            if (nbMinTrap == 0u && overWater > 0.35) {
                // Submerged hydrothermal chimney: massive chemosynthetic ATP windfall (works on night side too!)
                (*acc).dE += (0.34 + 0.30 * nbNutr) * (0.55 + 0.75 * myRedox) * mw;
                (*acc).dVit += 0.24 * mw;
                (*acc).steer += dToNb * (0.36 * mw);
            } else {
                // Air-oxidizing sulfuric weathering burst on windy land
                (*acc).dPoi += (0.40 + 0.36 * nbHaz) * mw;
                (*acc).dE += (0.12 * myRedox - 0.18) * mw;
            }
        } else {
            // Hydrated Clay (MINERAL_CLAY): evolutionary cofactor windfall & toxin scavenger
            (*acc).dE += 0.15 * mw;
            (*acc).dVit += (0.34 + 0.26 * nbNutr) * mw;
            (*acc).dPoi -= 0.14 * mw;
            (*acc).nearClayCat += mw;
            (*acc).steer += dToNb * (0.26 * mw);
        }
        return;
    }

    let nbEnergy = clamp(nbLifeState.x, 0.0, 1.6);
    let nbVit = clamp(nbLifeState.y, 0.0, 1.0);
    let nbPoi = clamp(nbLifeState.z, 0.0, 1.0);
    let nbKin = unpackKinPhase(nbLifeState.w);
    let nbGene = unpackEncapGene(nbLifeState.w);
    let nbGrad = unpackGraduated(nbLifeState.w);

    if (myPhase == 3u) {
        // Living cells and bilayers contacting an intact mineral actively bio-weather / mine it
        let redoxBoost = select(1.0, 1.55, nbPhase == 1u && elemRedoxWeight(nbAtomicNum) > 0.65);
        let encapBoost = select(1.0, 1.45, nbPhase == 2u || nbGrad == 1u);
        (*acc).nearMineralBioMine += wDist * clamp(nbEnergy, 0.25, 1.4) * encapBoost * redoxBoost;
        return;
    }

    // Neighbor is an oriented amphiphilic lipid bilayer particle (`nbPhase == 2u`)
    if (nbPhase == 2u) {
        if (myPhase == 0u) {
            (*acc).nearLipidCount += 1.0;
            (*acc).nearLipidEnergySum += nbEnergy;
            if (nbGrad == 1u) {
                (*acc).adjGradCount += 1.0;
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
            let myKind = bioRoleKind(myAtomicNum, myValence);
            if (myKind == 4u) {
                (*acc).kick -= dToNb * (wDist * 1.55 * bio.rate);
                (*acc).chemBioCount += 1.0;
            } else if (isBiogenicElement(myAtomicNum, myValence)) {
                (*acc).kick += dToNb * (wDist * 0.45 * bio.rate);
                (*acc).glowSum += 0.40 * wDist;
                (*acc).chemBioCount += 1.0;
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
            } else {
                (*acc).predRepelDir += dToNb * wDist;
            }
        } else {
            // Lateral energy, kin, and gene diffusion along own bilayer leaflets
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
            }
        }
        return;
    }

    // Neighbor is living Particle Life cytoplasm (`nbPhase == 1u`)
    let curVel = evalOceanCurrentVel(nbPos, oc, bio.oceanCurrentScale);
    let nbSelfVel = nbVel - curVel;
    let nbSelfSpeed = length(nbSelfVel);
    let nbMotility = clamp(nbSelfSpeed / 0.45, 0.0, 1.5);
    let nbSolarElem = elemSolarWeight(nbAtomicNum);
    let nbRedoxElem = elemRedoxWeight(nbAtomicNum);

    (*acc).nearTotalBio += 1.0;
    if (nbEnergy < 0.44 || nbPoi > 0.24) {
        (*acc).nearStressedBio += wDist * 1.15;
    }

    if (myPhase == 0u) {
        // Order-independent host telemetry for heterogeneous encapsulation gating
        (*acc).adjSpeciesMask |= (1u << (nbSp % 24u));
        (*acc).adjBlobCount += 1.0;
        (*acc).adjEnergySum += nbEnergy;
        if (nbGrad == 1u) {
            (*acc).adjGradCount += 1.0;
        }
        (*acc).adjGeneSum += f32(nbGene);
        (*acc).adjKinCos += wDist * cos(nbKin);
        (*acc).adjKinSin += wDist * sin(nbKin);
        (*acc).adjKinW += wDist;
        (*acc).blobCentroid += wDist * nbPos;
        (*acc).blobCentW += wDist;
        let adjScore = wDist * max(nbEnergy, 0.25);
        if (adjScore > (*acc).bestAdjW) {
            (*acc).bestAdjW = adjScore;
            (*acc).bestAdjSp = nbSp;
            (*acc).bestAdjKin = nbKin;
        }

        // Check if nearby living blob particle can fix free biogenic C/H/O/S atom
        let nbAutoInst = clamp(
            (0.35 + 0.85 * nbTraits.x) * (0.45 + 0.85 * nbSolarElem) / (1.0 + 0.65 * nbMotility),
            0.05,
            1.4,
        );
        if (nbEnergy > 0.52 && nbAutoInst > 0.34) {
            let fixScore = wDist * nbAutoInst * clamp(nbEnergy, 0.4, 1.4);
            (*acc).nearFixer += fixScore;
            if (fixScore > (*acc).bestFixW) {
                (*acc).bestFixW = fixScore;
                (*acc).bestFixSp = nbSp;
                (*acc).bestFixKin = nbKin;
                (*acc).bestFixGene = nbGene;
            }
        }

        let tSwirl = cross(myPos, dToNb);
        let eff = evalAtomFromLifePair(myAtomicNum, myValence, myAtomLoad, nbTraits, nbLifeState, r, bio);
        (*acc).kick += dToNb * eff.x + tSwirl * eff.y;
        (*acc).actScale = min((*acc).actScale, eff.z);
        (*acc).glowSum += eff.w;
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
        } else {
            let nbPredInst = clamp(
                (0.20 + 0.95 * nbTraits.w) * (0.35 + 0.85 * nbMotility) * (0.55 + 0.65 * nbRedoxElem),
                0.05,
                1.60,
            );
            if (nbEnergy > 0.24 && nbPredInst > 0.32) {
                (*acc).nearPredThreat += wDist * nbPredInst;
            }
        }
        return;
    }

    // Both particles are living cytoplasm (`myPhase == 1u && nbPhase == 1u`)
    let myKin = unpackKinPhase(myLifeState.w);
    let kinDist = kinAngularDist(myKin, nbKin);
    let aMyNb = getBioMatrixA(nChem, nBio, myBioSp, nbSp);
    let aNbMy = getBioMatrixA(nChem, nBio, nbSp, myBioSp);
    let mutualCoh = aMyNb + aNbMy;

    let sameBlob = (r < bio.catRadius * 0.92) &&
        ((kinDist < 0.24) || (r < bio.catRadius * 0.62 && (mutualCoh > 0.08 || (abs(aMyNb - aNbMy) > 0.45 && mutualCoh > -0.30) || length(myVel - nbVel) < 0.32)));

    if (sameBlob) {
        let bw = max(wDist, 0.15);
        (*acc).blobCount += 1.0;
        (*acc).blobWeightSum += bw;
        (*acc).blobEnergySum += nbEnergy;
        (*acc).blobVitSum += nbVit;
        if (nbGrad == 1u) {
            (*acc).blobGradCount += 1.0;
        }
        (*acc).blobCohSum += nbTraits.x;
        (*acc).blobAsymSum += nbTraits.w;
        (*acc).blobSolarAtomSum += nbSolarElem;
        (*acc).blobRedoxAtomSum += nbRedoxElem;
        (*acc).blobSpeedSum += nbSelfSpeed;
        (*acc).blobKinCos += bw * cos(nbKin);
        (*acc).blobKinSin += bw * sin(nbKin);
        (*acc).blobKinWeight += bw;
        (*acc).blobSpeciesMask |= (1u << (nbSp % 24u));
        (*acc).blobGeneSum += f32(nbGene);
        (*acc).blobCentroid += bw * nbPos;
        (*acc).blobCentW += bw;
        return;
    }

    // Foreign blob interaction: compare emergent predatory vs. prey/detrital roles
    let mySelfSpeed = length(myVel - evalOceanCurrentVel(myPos, oc, bio.oceanCurrentScale));
    let myMotility = clamp(mySelfSpeed / 0.45, 0.0, 1.5);
    let myRedoxElem = elemRedoxWeight(myAtomicNum);

    let myPredInst = clamp(
        (0.20 + 0.95 * myTraits.w) * (0.35 + 0.85 * myMotility) * (0.55 + 0.65 * myRedoxElem),
        0.05,
        1.60,
    );
    let nbPredInst = clamp(
        (0.20 + 0.95 * nbTraits.w) * (0.35 + 0.85 * nbMotility) * (0.55 + 0.65 * nbRedoxElem),
        0.05,
        1.60,
    );
    let nbDecompInst = clamp(
        (0.30 + 0.90 * nbRedoxElem) * (0.55 + 0.55 * (1.0 - min(nbMotility, 1.0))),
        0.10,
        1.40,
    );

    if (mutualCoh < 0.35) {
        if (myPredInst > nbPredInst * 0.85 && myPredInst > 0.32) {
            (*acc).nearForeignPrey += wDist;
            (*acc).bioSteer += dToNb * (wDist * 0.28);
        }

        let closingSpeed = max(0.0, dot(nbVel - myVel, -dToNb));
        if (nbEnergy > 0.24 && nbPredInst > myPredInst * 0.82 && nbPredInst > 0.28) {
            let threatScore = wDist * nbPredInst * (0.45 + max(0.0, aNbMy - aMyNb) + 0.65 * closingSpeed);
            (*acc).nearPredThreat += threatScore;
            (*acc).predRepelDir += dToNb * wDist;
            if (threatScore > (*acc).bestPredW) {
                (*acc).bestPredW = threatScore;
                (*acc).bestPredSp = nbSp;
                (*acc).bestPredKin = nbKin;
                (*acc).bestPredGene = nbGene;
                (*acc).bestPredGrad = nbGrad;
                (*acc).bestPredE = nbEnergy;
            }
        }
    }

    if (nbEnergy > 0.22 && nbDecompInst > 0.38) {
        let decScore = wDist * nbDecompInst;
        (*acc).nearDecompPower += decScore;
        if (decScore > (*acc).bestDecompW) {
            (*acc).bestDecompW = decScore;
            (*acc).bestDecompSp = nbSp;
            (*acc).bestDecompKin = nbKin;
            (*acc).bestDecompGene = nbGene;
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
        return BioStepResult(vec4f(0.0), vec4f(0.0));
    }
    let mRate = max(bio.mineralRate, 0.0);
    if (mRate <= 0.001) {
        return BioStepResult(vec4f(0.0), vec4f(0.0));
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
            // Slow geological accretion along crystal symmetry arms
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
            nutr = 0.25;
            haz = 0.0;
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
            nutr = 0.95;
            haz = 0.0;
        }
        let grade = u32(clamp(round(nutr * 12.0 + clamp(acc.nearMineralCount, 0.0, 3.0)), 0.0, 15.0));
        let eBase = sphereEastBasis(pos);
        let eNorth = normalize(cross(pos, eBase));
        let crystDir = cos(crystAngle) * eBase + sin(crystAngle) * eNorth;
        let initInteg = clamp(0.84 + 0.18 * calmEnv, 0.78, 1.05);
        return BioStepResult(
            vec4f(initInteg, nutr, haz, packLifeMetaGrad(3u, chooseType, crystAngle, grade, trapFlag)),
            vec4f(crystDir, 2.0),
        );
    }

    return BioStepResult(vec4f(0.0), vec4f(0.0));
}

// Dead material remains in the chemistry phase. The temporal pass consumes the
// one-tick marker. Negative multiples of four decode as phase zero everywhere.
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

// For energy-triggered decay, ask whether removing the environmental penalties
// would have cleared the lethal threshold on this tick. Report the largest
// contributor in that case. Test crowding next, then ordinary energy depletion.
fn energyDeath(rawEnergy: f32, energyBeforeStep: f32, threshold: f32, dt: f32, crowd: f32, stress: vec4f) -> u32 {
    // A cell can persist below the threshold for several ticks before it dies.
    // In that case also recognize a pressure that kept it from recovering.
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

fn finalizeBioUnified(
    pi: u32,
    pos: vec3f,
    vel: vec3f,
    atomicNum: u32,
    valence: u32,
    atomLoad: f32,
    curLifeState: vec4f,
    windRaw: vec3f,
    oc: vec4f,
    tSample: vec4f,
    uvSample: vec4f,
    hydroSample: vec2f,
    acc: BioUnifiedAccum,
    bio: BioLink,
) -> BioStepResult {
    let myPhase = unpackPhase(curLifeState.w);
    let landFrac = clamp(1.0 - smoothstep(0.18, 0.48, oc.w), 0.0, 1.0);
    let tBlend = clamp(bio.localTempScale, 0.0, 1.5);
    let hBlend = clamp(bio.localHydroScale, 0.0, 1.5);
    let tHabK = effectiveSurfaceTempK(tSample, oc.w);
    let uvRes = evalUvSolarAndStress(uvSample, bio.localUvScale);
    let solar = uvRes.x;
    let windSim = evalWindVel3D(pos, windRaw, bio);
    let windNorm = length(windSim);
    let curSim = evalOceanCurrentVel(pos, oc, bio.oceanCurrentScale);
    let curNorm = length(curSim);
    let rh = hydroSample.x;
    let precipMm = hydroSample.y;
    let eAir = rh * saturationVaporKpa(tSample.w);
    let eSea = saturationVaporKpa(tSample.y);
    let seaEvapFlux = max(0.0, eSea - eAir) * (0.45 + 0.75 * windNorm);
    let seaFogCondense = smoothstep(0.0, 0.35, eAir - eSea);
    let rainWash = smoothstep(0.3, 6.0, precipMm);
    let vpdLand = max(0.0, saturationVaporKpa(tSample.w) - eAir);
    let uvStress = uvRes.y * mix(1.0, max(0.38, 1.0 - 0.50 * seaFogCondense - 0.32 * rainWash), hBlend);
    let frameU = u32(max(bio.frame, 0.0)) + u32(max(bio.seedHash, 0.0));

    // Inorganic Crustal Mineral (`myPhase == 3u`): stability, dual-layer wind/current erosion & thermal geochemistry
    if (myPhase == 3u) {
        if (bio.mineralEnabled <= 0.5) {
            return BioStepResult(vec4f(clamp(curLifeState.x, 0.1, 1.25), curLifeState.y, curLifeState.z, curLifeState.w), vec4f(0.0));
        }
        let mType = min(unpackBioSpecies(curLifeState.w), MINERAL_CLAY);
        let mAngle = unpackKinPhase(curLifeState.w);
        var mTrap = unpackGraduated(curLifeState.w);
        let integ = clamp(curLifeState.x, 0.02, 1.25);
        let nutr = clamp(curLifeState.y, 0.0, 1.0);
        var haz = clamp(curLifeState.z, 0.0, 1.0);
        // Geological weathering is comparatively slow (~4x slower than metabolic chemistry & life)
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
            // Carbonate exhibits retrograde solubility: cold polar seas dissolve faster than warm tropical waters
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
            // Hydrated Clay: stable underwater/moist coasts, desiccates faster under hot continental heat, UV, high VPD, or frost
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

        // Slow weathered spalling / dissolution (`3 -> 0`), kicking freed elements down active wind/current plumes
        let pSpall = clamp(erodeRate * (1.20 - 0.72 * clamp(nextInteg, 0.0, 1.0)) * dtMin * 1.15, 0.0, 0.045);
        if (nextInteg <= 0.10 || (erodeRate > 0.04 && hashFloat(pi, frameU, 65u) < pSpall)) {
            let flowPlume = select(windSim, curSim, inWater >= 0.45) * 6.5;
            let eEast = sphereEastBasis(pos);
            let eNorth = normalize(cross(pos, eEast));
            let s1 = hashFloat(pi, frameU, 63u) * 2.0 - 1.0;
            let s2 = hashFloat(pi, frameU, 64u) * 2.0 - 1.0;
            let ejectVel = flowPlume + (eEast * s1 + eNorth * s2) * 0.85;
            let ejectGlow = select(0.75, -0.85, mTrap == 1u || haz > 0.45);
            return BioStepResult(
                vec4f(0.0),
                vec4f(ejectVel, packDeltaActGlow(0.9, ejectGlow)),
            );
        }

        return BioStepResult(
            vec4f(nextInteg, nutr, haz, packLifeMetaGrad(3u, mType, mAngle, liveGrade, mTrap)),
            vec4f(0.0),
        );
    }

    if (bio.enabled <= 0.5) {
        if (myPhase != 0u) {
            return BioStepResult(vec4f(max(curLifeState.x, 0.55), curLifeState.y, curLifeState.z, curLifeState.w), vec4f(0.0));
        }
        let chemUpwell = evalCoastalUpwelling(oc) + 0.25 * smoothstep(1.0, 9.0, tSample.y - tSample.w) * smoothstep(0.36, 0.54, oc.w) * hBlend;
        let crystRes = attemptMineralCrystallize(
            pi, pos, atomicNum, valence, 1.0 / (1.0 + 0.55 * max(atomLoad, 0.0)),
            solar, chemUpwell, oc, tHabK, tSample.w, hydroSample, landFrac, windNorm, curNorm, frameU, acc, bio,
        );
        return crystRes;
    }

    let rate = max(bio.rate, 0.0);
    let dtBio = bio.dt * rate;
    let q10Bio = mix(1.0, clamp(pow(1.38, clamp((tHabK - 288.15) / 10.0, -2.5, 2.2)), 0.42, 1.55), tBlend);
    let convectiveMix = smoothstep(1.0, 9.0, tSample.y - tSample.w) * smoothstep(0.36, 0.54, oc.w);
    let upwell = evalCoastalUpwelling(oc) + (0.28 * convectiveMix + 0.16 * rainWash) * hBlend * smoothstep(0.36, 0.54, oc.w);
    let landHydration = clamp(0.42 * smoothstep(0.32, 0.92, rh) + 0.68 * rainWash + 0.32 * seaFogCondense, 0.0, 1.25);
    let aridStress = smoothstep(1.1, 4.0, vpdLand) * (1.0 - smoothstep(0.4, 3.0, precipMm));
    let oceanBleachBoost = smoothstep(301.0, 307.0, tHabK) * smoothstep(0.36, 0.54, oc.w) * uvStress * (1.0 - rainWash) * hBlend;
    let nChem = max(u32(bio.chemColours), 1u);
    let nBio = max(u32(bio.bioColours), 1u);

    if (myPhase == 0u) {
        let chemForce = finalizeChemBioForce(acc, bio);
        let bondEase = 1.0 / (1.0 + 0.55 * max(atomLoad, 0.0));
        if (!isBiogenicElement(atomicNum, valence)) {
            let crystRes = attemptMineralCrystallize(
                pi, pos, atomicNum, valence, bondEase,
                solar, upwell, oc, tHabK, tSample.w, hydroSample, landFrac, windNorm, curNorm, frameU, acc, bio,
            );
            if (unpackPhase(crystRes.nextLifeState.w) == 3u) {
                return crystRes;
            }
            return BioStepResult(vec4f(0.0), chemForce);
        }

        let adjColors = countOneBits(acc.adjSpeciesMask);
        let adjCount = acc.adjBlobCount;
        let adjAvgE = acc.adjEnergySum / max(adjCount, 1.0);
        let hasAdj = adjCount > 0.5;
        let adjGene = select(
            acc.bestLipidGene,
            u32(clamp(round(acc.adjGeneSum / max(adjCount, 1.0)), 0.0, 15.0)),
            hasAdj,
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
        let envMedia = mix(0.55 + 0.85 * upwell, 0.44 + 0.22 * clamp(acc.nearLipidCount * 0.25, 0.0, 1.0), landFrac) * clayCatBoost;
        let hostGrad = acc.adjGradCount >= 1.5;

        // 0A. Deferred biogenic lipid bilayer synthesis (`0 -> 2`, strictly C/H/O only; works across ocean and land)
        let kinCoherence = length(vec2f(acc.adjKinCos, acc.adjKinSin)) / max(acc.adjKinW, 1e-4);
        if (isLipidElement(atomicNum, valence) && adjCount >= 3.0 && kinCoherence > 0.80) {
            let matureDeNovo = (adjColors >= encapReqColors(adjGene)) && (adjCount >= encapReqSize(adjGene)) && (adjAvgE > 0.54);
            let expandExisting = (acc.nearLipidCount >= 1.0) && (adjColors >= 3u) && (adjCount >= 3.0) && (max(adjAvgE, avgLipidE) > 0.48);
            let maxLocalLipids = select(7.0, 12.0, hostGrad);
            let lipidHeadroom = clamp(1.0 - acc.nearLipidCount / maxLocalLipids, 0.0, 1.0);
            if ((matureDeNovo || expandExisting) && lipidHeadroom > 0.05) {
                let zipBoost = select(1.0, 1.65, expandExisting && acc.nearLipidCount < 5.0) * select(1.0, 1.50, hostGrad);
                let pLipid = clamp(
                    0.85 * zipBoost * q10Bio * solar * envMedia * clamp(max(adjAvgE, avgLipidE), 0.5, 1.4) * lipidHeadroom * bondEase * dtBio,
                    0.0,
                    0.20,
                );
                if (hashFloat(pi, frameU, 30u) < pLipid) {
                    let hostSp = min(select(select(acc.bestLipidSp, acc.bestAdjSp, acc.bestAdjW > 0.0), acc.bestFixSp, acc.bestFixW > 0.0), nBio - 1u);
                    let hostKin = select(select(acc.bestLipidKin, acc.bestAdjKin, acc.bestAdjW > 0.0), acc.bestFixKin, acc.bestFixW > 0.0);
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
                        vec4f(0.78, 0.06, 0.0, packLifeMetaGrad(2u, hostSp, hostKin, adjGene, lipidGrad)),
                        vec4f(initOrient, 2.0),
                    );
                }
            }
        }

        // 0B. Transmembrane nutrient import into encapsulated protocells OR naked autotrophic fixation (`0 -> 1`)
        if (acc.nearLipidCount >= 1.0 && adjCount >= 1.0) {
            let cap = (22.0 + 3.2 * acc.nearLipidCount + 1.2 * f32(adjGene)) * select(1.0, 1.95, hostGrad);
            let encapHeadroom = clamp(1.0 - adjCount / cap, 0.0, 1.0);
            if (encapHeadroom > 0.02 && max(adjAvgE, avgLipidE) > 0.52) {
                let fixPow = clamp(max(acc.nearFixer, acc.nearLipidCount * 0.6), 0.4, 3.0);
                let pImport = clamp(
                    0.32 * select(1.0, 1.25, hostGrad) * q10Bio * solar * envMedia * fixPow * encapHeadroom * bondEase * dtBio,
                    0.0,
                    0.10,
                );
                if (hashFloat(pi, frameU, 10u) < pImport) {
                    let hostSp = min(select(acc.bestAdjSp, acc.bestFixSp, acc.bestFixW > 0.0), nBio - 1u);
                    let hostKin = select(acc.bestAdjKin, acc.bestFixKin, acc.bestFixW > 0.0);
                    var nextSp = hostSp;
                    if (nBio > 1u && hashFloat(pi, frameU, 11u) < encapNoveltyDrive(adjGene, adjColors)) {
                        nextSp = pickCompatibleNovelSpecies(nChem, nBio, nextSp, acc.adjSpeciesMask, pi, frameU, 40u);
                    }
                    let childGene = mutateEncapGene(adjGene, pi, frameU, 17u, 0.06);
                    let childGrad = select(0u, 1u, hostGrad && (acc.nearLipidCount >= 1.0 || acc.adjGradCount >= 3.0));
                    return BioStepResult(
                        vec4f(0.68, 0.06, 0.0, packLifeMetaGrad(1u, nextSp, hostKin, childGene, childGrad)),
                        vec4f(-outwardDir * 2.2, 0.9),
                    );
                }
            }
        } else if (acc.nearFixer > 0.0) {
            let crowdHeadroom = clamp(1.0 - acc.nearTotalBio / 34.0, 0.0, 1.0);
            if (crowdHeadroom > 0.01) {
                let elemAffinity = 0.85 + 0.35 * elemSolarWeight(atomicNum);
                let pFix = clamp(
                    0.36 * q10Bio * solar * envMedia * min(acc.nearFixer, 3.0) * elemAffinity * crowdHeadroom * bondEase * dtBio,
                    0.0,
                    0.10,
                );
                if (hashFloat(pi, frameU, 10u) < pFix) {
                    var nextSp = min(acc.bestFixSp, nBio - 1u);
                    let pDiff = select(0.18, 0.07, adjColors >= 6u);
                    if (nBio > 1u && hashFloat(pi, frameU, 11u) < pDiff) {
                        nextSp = pickCompatibleNovelSpecies(nChem, nBio, nextSp, acc.adjSpeciesMask, pi, frameU, 40u);
                    }
                    let childGene = mutateEncapGene(acc.bestFixGene, pi, frameU, 17u, 0.05);
                    return BioStepResult(
                        vec4f(0.66, 0.06, 0.0, packLifeMeta(1u, nextSp, acc.bestFixKin, childGene)),
                        vec4f(0.0, 0.0, 0.0, 0.9),
                    );
                }
            }
        }

        // 0C. Low-rate abiogenesis / spore floor in sparse sunlit water so regions never become permanently barren
        if (acc.nearTotalBio < 1.5 && acc.nearLipidCount < 0.5 && oc.w >= 0.46) {
            let pSpore = clamp(0.0028 * q10Bio * solar * (0.50 + upwell) * dtBio, 0.0, 0.0052);
            if (hashFloat(pi, frameU, 13u) < pSpore) {
                let sp = u32(floor(hashFloat(pi, frameU, 14u) * f32(nBio))) % nBio;
                let freshKin = hashFloat(pi, frameU, 15u) * TWO_PI;
                let freshGene = u32(floor(hashFloat(pi, frameU, 16u) * 16.0)) & 15u;
                return BioStepResult(
                    vec4f(0.66, 0.05, 0.0, packLifeMeta(1u, sp, freshKin, freshGene)),
                    vec4f(0.0, 0.0, 0.0, 0.6),
                );
            }
        }

        // 0D. Geochemical nucleation into crustal minerals (`0 -> 3`) when unassimilated
        let crystBioRes = attemptMineralCrystallize(
            pi, pos, atomicNum, valence, bondEase,
            solar, upwell, oc, tHabK, tSample.w, hydroSample, landFrac, windNorm, curNorm, frameU, acc, bio,
        );
        if (unpackPhase(crystBioRes.nextLifeState.w) == 3u) {
            return crystBioRes;
        }

        return BioStepResult(vec4f(0.0), chemForce);
    }

    let myBioSp = min(unpackBioSpecies(curLifeState.w), nBio - 1u);
    let myKin = unpackKinPhase(curLifeState.w);
    let myGene = unpackEncapGene(curLifeState.w);
    let myGrad = unpackGraduated(curLifeState.w);
    let dtVit = dtBio * 1.8;

    // Amphiphilic Lipid Bilayer (`myPhase == 2u`): sustained by host cytoplasm + lateral diffusion, dissolves back to C/H/O on rupture
    if (myPhase == 2u) {
        let totalWeight = 1.0 + acc.blobWeightSum;
        let avgE = (curLifeState.x + acc.blobEnergySum) / totalWeight;
        let diffAlpha = select(0.0, clamp(4.2 * dtBio, 0.0, 0.38), acc.blobWeightSum > 0.1);
        let eDiffused = mix(curLifeState.x, avgE, diffAlpha);

        var nextKin = myKin;
        if (acc.blobKinWeight > 0.01) {
            let goalKin = wrapAngle(atan2(acc.blobKinSin, acc.blobKinCos));
            var dAngle = goalKin - myKin;
            if (dAngle > PI) {
                dAngle -= TWO_PI;
            } else if (dAngle < -PI) {
                dAngle += TWO_PI;
            }
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
            (acc.blobGradCount >= 3.0 && acc.blobCount >= 20.0);
        let heatLipidStress = (smoothstep(307.0, 323.0, tHabK) * tBlend + 0.65 * oceanBleachBoost) * 0.09;
        let coldLipidStab = mix(1.0, 0.78, smoothstep(278.0, 266.0, tHabK) * tBlend);
        let upkeep = select(0.55, select(0.055, 0.036, gradNow), lipHealthy);
        let poisonPenalty = 0.24 * poi;
        let landPenalty = 0.045 * landFrac * mix(1.0, clamp(0.28 + 1.35 * aridStress - 0.30 * landHydration, 0.16, 2.1), hBlend);
        let nextEnergy = clamp(eDiffused - (upkeep + poisonPenalty + landPenalty + heatLipidStress) * dtVit, 0.02, 1.45);
        let hostIsGrad = gradNow && (nextEnergy >= 0.36) && (poi < 0.72);

        let hostColors = countOneBits(acc.blobSpeciesMask);
        let simpleHostStress = (acc.blobCount >= 1.0 && hostColors <= 2u && acc.blobLipidCount >= 2.0);
        let orphanStress = (acc.blobCount < 0.5 && nextEnergy < 0.52);
        if (nextEnergy <= 0.14 || poi >= 0.85 || orphanStress || simpleHostStress) {
            let pDissolve = select(clamp(0.65 * coldLipidStab * dtBio, 0.0, 0.18), clamp(2.4 * dtBio, 0.0, 0.35), simpleHostStress);
            if (hashFloat(pi, frameU, 31u) < pDissolve) {
                var cause = DEATH_MEMBRANE;
                if (poi >= 0.85) { cause = DEATH_POISON; }
                else if (nextEnergy <= 0.14) {
                    let stress = vec4f(poisonPenalty, heatLipidStress, 0.0, landPenalty);
                    cause = energyDeath(eDiffused - (upkeep + dot(stress, vec4f(1.0))) * dtVit, eDiffused, 0.14, dtVit, 0.0, stress);
                }
                return BioStepResult(
                    deadCell(cause),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.9, -0.7)),
                );
            }
        }
        if (acc.nearPredThreat > 1.15) {
            let pBreach = clamp(0.20 * (acc.nearPredThreat - 1.0) * select(1.0, 0.42, hostIsGrad) * dtBio, 0.0, 0.05);
            if (hashFloat(pi, frameU, 32u) < pBreach) {
                return BioStepResult(
                    deadCell(DEATH_PREDATION),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.85, -0.8)),
                );
            }
        }

        return BioStepResult(
            vec4f(nextEnergy, vit, poi, packLifeMetaGrad(2u, myBioSp, nextKin, nextGene, select(0u, 1u, hostIsGrad))),
            vec4f(0.0),
        );
    }

    // Current particle is living Particle Life cytoplasm (`myPhase == 1u`)
    let myTraits = getBioSpeciesTraits(nChem, nBio, myBioSp);
    let curVel = evalOceanCurrentVel(pos, oc, bio.oceanCurrentScale);
    let mySelfSpeed = length(vel - curVel);

    // Average local blob traits & stoichiometry over self + cohesive same-blob neighbors
    let nMembers = 1.0 + acc.blobCount;
    let avgBlobE = (curLifeState.x + acc.blobEnergySum) / nMembers;
    let avgBlobVit = (curLifeState.y + acc.blobVitSum) / nMembers;
    let avgCoh = (myTraits.x + acc.blobCohSum) / nMembers;
    let avgAsym = (myTraits.w + acc.blobAsymSum) / nMembers;
    let avgSolarFrac = (elemSolarWeight(atomicNum) + acc.blobSolarAtomSum) / nMembers;
    let avgRedoxFrac = (elemRedoxWeight(atomicNum) + acc.blobRedoxAtomSum) / nMembers;
    let avgSelfSpeed = (mySelfSpeed + acc.blobSpeedSum) / nMembers;

    let mySpeciesMask = acc.blobSpeciesMask | (1u << (myBioSp % 24u));
    let myDistinctColors = countOneBits(mySpeciesMask);
    let membraneShield = clamp(acc.blobLipidCount / 6.0, 0.0, 1.0);
    let lipPresent = (acc.blobLipidCount >= 0.8) || (acc.blobLipidGradCount >= 0.5) || (acc.blobGradCount >= 4.0);
    let isGradMaintain = (myGrad == 1u) && (acc.blobCount >= 18.0) && (curLifeState.x >= 0.38) && (curLifeState.z < 0.65) && lipPresent;
    let isGradSpread = (acc.blobGradCount >= 2.5) && (acc.blobCount >= 24.0) && (avgBlobE >= 0.62) && (curLifeState.z < 0.40) && ((acc.blobLipidCount >= 1.0) || (acc.blobGradCount >= 5.0));
    let isGradNow = isGradMaintain || isGradSpread;
    let isGradNowF = select(0.0, 1.0, isGradNow);
    let nextGene = select(
        myGene,
        u32(clamp(round(mix(f32(myGene), (f32(myGene) + acc.blobGeneSum) / nMembers, clamp(2.5 * dtBio, 0.0, 0.25))), 0.0, 15.0)),
        acc.blobCount > 0.0,
    );

    let motility = clamp(avgSelfSpeed / 0.45, 0.0, 1.5);
    let lightExposure = clamp(1.12 - 0.015 * acc.nearTotalBio, 0.45, 1.0);

    let autoCap = clamp(
        (0.35 + 0.85 * avgCoh) * (0.45 + 0.85 * avgSolarFrac) * lightExposure / (1.0 + 0.65 * motility),
        0.08,
        1.35,
    );
    let predCap = clamp(
        (0.20 + 0.95 * avgAsym) * (0.35 + 0.85 * motility) * (0.55 + 0.65 * avgRedoxFrac),
        0.05,
        1.60,
    );
    let decompCap = clamp(
        (0.30 + 0.90 * avgRedoxFrac) * (0.55 + 0.55 * (1.0 - min(motility, 1.0))),
        0.10,
        1.40,
    );
    let haloTol = clamp(0.25 + 0.65 * avgRedoxFrac + 0.25 * avgCoh, 0.0, 1.0);

    let normChem = select(0.0, 1.0 / (1.0 + 0.35 * max(acc.bioChemCount - 1.0, 0.0)), acc.bioChemCount > 0.0);
    let grazeVitBoost = select(0.0, 0.24 * min(acc.nearForeignPrey, 2.2) * predCap, acc.nearForeignPrey > 0.12 && predCap > 0.34);
    let vitRaw = curLifeState.y + (acc.dVit * normChem + grazeVitBoost - 0.13 * rate * curLifeState.y) * dtVit;
    let vitDiffAlpha = select(0.0, clamp(2.4 * dtBio, 0.0, 0.26), acc.blobCount > 0.0);
    let vit = clamp(mix(vitRaw, avgBlobVit, vitDiffAlpha), 0.0, 1.0);
    let detoxRate = 0.24 + 0.45 * vit + 0.35 * max(0.0, haloTol - 0.40);
    let poiIn = acc.dPoi * normChem * (1.0 - 0.65 * max(0.0, haloTol - 0.50)) * (1.0 - 0.72 * membraneShield);
    let poi = clamp(curLifeState.z + (poiIn - detoxRate * curLifeState.z) * dtVit, 0.0, 1.0);

    // Intra-blob energy sharing (sunlit skin feeds interior; hunter head shares prey energy with tail)
    let diffAlpha = select(0.0, clamp(2.8 * dtBio, 0.0, 0.28), acc.blobCount > 0.0);
    let eDiffused = mix(curLifeState.x, avgBlobE, diffAlpha);

    // Intra-blob kin signature consensus + slow stochastic drift when isolated
    var nextKin = myKin;
    if (acc.blobKinWeight > 0.01) {
        let goalKin = wrapAngle(atan2(acc.blobKinSin, acc.blobKinCos));
        var dAngle = goalKin - myKin;
        if (dAngle > PI) {
            dAngle -= TWO_PI;
        } else if (dAngle < -PI) {
            dAngle += TWO_PI;
        }
        nextKin = wrapAngle(myKin + dAngle * clamp(3.0 * dtBio, 0.0, 0.25));
    } else {
        let jitter = (hashFloat(pi, frameU, 19u) * 2.0 - 1.0) * 0.004;
        nextKin = wrapAngle(myKin + jitter);
    }

    let moistureFactor = mix(
        1.0,
        mix(
            0.74 + 0.16 * membraneShield,
            clamp(0.52 + 0.44 * landHydration - 0.24 * aridStress + 0.16 * membraneShield, 0.35, 1.12),
            hBlend,
        ),
        landFrac,
    );
    let solarGain = 0.56 * mix(1.0, q10Bio, 0.75) * solar * mix(0.55 + 0.85 * upwell, 0.55, landFrac) * autoCap * moistureFactor;
    let haloGain = select(0.0, 0.24 * min(acc.nearHalogen, 2.5) * haloTol, haloTol > 0.52);
    let detritusGain = 0.26 * min(acc.nearStressedBio, 2.5) * decompCap;
    let grazeEnvGain = 0.34 * min(acc.nearForeignPrey, 2.5) * predCap;
    let chemBonus = 0.24 * acc.dE * normChem * q10Bio * (1.0 + 0.35 * landFrac);
    let coldTorpor = mix(1.0, clamp(0.66 + 0.38 * smoothstep(271.0, 296.0, tHabK), 0.62, 1.18), tBlend);
    let basalBurn = (0.13 + 0.11 * motility * motility + 0.04 * avgAsym) * coldTorpor * (1.0 - 0.14 * membraneShield - 0.26 * isGradNowF);
    let foreignBio = max(0.0, acc.nearTotalBio - acc.blobCount);
    let ownCrowdStart = select(56.0, 145.0, isGradNow);
    let ownCrowdSlope = select(0.020, 0.008, isGradNow);
    let foreignCrowdScale = mix(1.0, 0.35, isGradNowF * membraneShield);
    let crowdPenalty = 0.032 * max(0.0, foreignBio - 12.0) * foreignCrowdScale + ownCrowdSlope * max(0.0, acc.blobCount - ownCrowdStart);
    let poisonPenalty = 0.28 * poi * (1.0 - 0.65 * clamp(haloTol - 0.35, 0.0, 1.0));
    let desiccationShield = 0.55 * membraneShield;
    let heatShock = smoothstep(305.0, 321.0, tHabK) * tBlend + 0.55 * oceanBleachBoost;
    let frostShock = smoothstep(271.0, 249.0, tHabK) * tBlend;
    let thermStressPenalty = (heatShock * 0.20 + frostShock * 0.14) * (1.0 - 0.62 * membraneShield);
    let uvStressPenalty = 0.16 * uvStress * (1.0 - 0.72 * membraneShield);
    let oceanEvapOsmotic = max(0.0, seaEvapFlux - precipMm * 0.25) * smoothstep(0.36, 0.54, oc.w) * (1.0 - clamp(haloTol, 0.0, 1.0)) * 0.022 * hBlend;
    let landPenalty = (0.075 * landFrac * mix(1.0, clamp(0.28 + 1.35 * aridStress - 0.30 * landHydration, 0.16, 2.1), hBlend) + oceanEvapOsmotic) * (1.0 - desiccationShield);

    let netDE = solarGain + haloGain + detritusGain + grazeEnvGain + 0.06 * vit + chemBonus
        - basalBurn - crowdPenalty - poisonPenalty - landPenalty - thermStressPenalty - uvStressPenalty;
    let nextEnergy = clamp(eDiffused + netDE * dtVit, 0.04, 1.55);

    // Inter-blob Grazing: foreign hunter eats vulnerable prey (~65% assimilated with endosymbiosis, ~35% respired)
    if (acc.nearPredThreat > 0.0) {
        let haloShield = 1.0 / (1.0 + 1.4 * acc.nearHalogen);
        let pGraze = clamp(
            0.42 * min(acc.nearPredThreat, 3.0) * haloShield * (1.0 - 0.30 * vit) * (1.0 - 0.76 * membraneShield) * select(1.0, 0.45, isGradNow) * dtBio,
            0.0,
            0.085,
        );
        if (hashFloat(pi, frameU, 20u) < pGraze) {
            let outcome = hashFloat(pi, frameU, 21u);
            if (outcome < 0.65) {
                let predSp = min(acc.bestPredSp, nBio - 1u);
                var nextSpAssimilated = predSp;
                if (outcome >= 0.43 && myBioSp != predSp && getBioMatrixA(nChem, nBio, predSp, myBioSp) > -0.15) {
                    nextSpAssimilated = myBioSp;
                }
                let fedE = clamp(max(0.94, max(curLifeState.x, acc.bestPredE) * 0.50 + 0.56), 0.94, 1.38);
                let fedVit = clamp(max(vit, 0.28) + 0.44, 0.0, 1.0);
                let preyGrad = select(0u, 1u, acc.bestPredGrad == 1u && acc.bestPredE >= 0.55);
                return BioStepResult(
                    vec4f(fedE, fedVit, poi * 0.6, packLifeMetaGrad(1u, nextSpAssimilated, acc.bestPredKin, acc.bestPredGene, preyGrad)),
                    vec4f(0.0, 0.0, 0.0, 0.75),
                );
            } else {
                let repelLen = length(acc.predRepelDir);
                let expel = select(vec3f(0.0), -(acc.predRepelDir / repelLen) * 1.4, repelLen > 1e-4);
                return BioStepResult(
                    deadCell(DEATH_PREDATION),
                    vec4f(expel, packDeltaActGlow(0.8, -0.9)),
                );
            }
        }
    }

    // Emergent Decomposition: ~80% returned to Chemistry (restoring immutable chemSpecies), ~20% assimilated
    let stress = clamp(
        smoothstep(0.45, 0.14, nextEnergy) * 0.95 +
        smoothstep(0.24, 0.78, poi) * 0.80 +
        smoothstep(14.0, 28.0, foreignBio) * select(0.55, 0.18, isGradNow && membraneShield > 0.3) +
        landFrac * 0.12,
        0.0,
        1.5,
    );
    if (acc.nearDecompPower > 0.0 && stress > 0.12) {
        let gradDecompScale = select(1.0, mix(0.28, 0.08, membraneShield), isGradNow && poi < 0.50 && nextEnergy > 0.35);
        let pDecomp = clamp(0.55 * stress * min(acc.nearDecompPower, 3.0) * gradDecompScale * dtBio, 0.0, 0.105);
        if (hashFloat(pi, frameU, 22u) < pDecomp) {
            let decRoll = hashFloat(pi, frameU, 23u);
            if (decRoll < 0.80) {
                return BioStepResult(
                    deadCell(DEATH_DECOMPOSITION),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.85, -0.85)),
                );
            } else {
                let decSp = min(acc.bestDecompSp, nBio - 1u);
                return BioStepResult(
                    vec4f(0.64, vit * 0.6, poi * 0.5, packLifeMeta(1u, decSp, acc.bestDecompKin, acc.bestDecompGene)),
                    vec4f(0.0, 0.0, 0.0, 0.55),
                );
            }
        }
    }

    // Natural senescence / turnover back to Chemistry (phase 1 -> 0)
    let decayThreshold = select(0.08, 0.045, isGradNow);
    if (nextEnergy <= decayThreshold || poi >= 0.90) {
        let pDecay = clamp(select(0.45, 0.14, isGradNow) * dtBio, 0.0, 0.14);
        if (hashFloat(pi, frameU, 24u) < pDecay) {
            let depletion = energyDeath(eDiffused + netDE * dtVit, eDiffused, decayThreshold, dtVit, crowdPenalty,
                vec4f(poisonPenalty, thermStressPenalty, uvStressPenalty, landPenalty));
            return BioStepResult(
                deadCell(select(depletion, DEATH_POISON, poi >= 0.90)),
                vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.9, -0.7)),
            );
        }
    }

    // Strong Evolutionary Filter for Complex Organism Graduation:
    // Requires ALL of: semi-formed lipid bilayer + large post-engulfment size + endosymbiotic diversity + post-meal surge
    var outGrad = 0u;
    if (isGradMaintain) {
        outGrad = 1u;
    } else if (isGradSpread) {
        if (hashFloat(pi, frameU, 25u) < clamp(4.5 * dtBio, 0.0, 0.68)) {
            outGrad = 1u;
        }
    } else {
        let hasSemiFormedShell = acc.blobLipidCount >= 4.0;
        let hasLargeMass = acc.blobCount >= 32.0;
        let hasEndosymbiosis = myDistinctColors >= 8u;
        let recentlyConsumed = (acc.nearForeignPrey >= 0.18 && predCap >= 0.34) || (vit >= 0.32 && avgBlobVit >= 0.30);
        let metabolicSurplus = nextEnergy >= 0.88 && avgBlobE >= 0.85 && poi < 0.22;
        if (hasSemiFormedShell && hasLargeMass && hasEndosymbiosis && recentlyConsumed && metabolicSurplus) {
            if (hashFloat(pi, frameU, 27u) < clamp(1.8 * dtBio, 0.0, 0.32)) {
                outGrad = 1u;
            }
        }
    }

    // Controlled post-encapsulation somatic innovation & mega-cell budding
    var outSp = myBioSp;
    var outKin = nextKin;
    var outGene = nextGene;
    let maxNovelColors = select(8u, 12u, outGrad == 1u);
    let clayInnovBoost = 1.0 + 1.25 * clamp(acc.nearClayCat, 0.0, 1.5);
    let uvMutBoost = 1.0 + 0.85 * uvStress * (1.0 - 0.55 * membraneShield);
    if ((acc.blobLipidCount >= 2.0 || acc.nearClayCat >= 0.65) && nextEnergy > 0.80 && myDistinctColors < maxNovelColors && acc.blobCount >= 7.0) {
        let pInnov = clamp(0.018 * clayInnovBoost * uvMutBoost * encapNoveltyDrive(nextGene, myDistinctColors) * dtBio, 0.0, 0.015);
        if (hashFloat(pi, frameU, 26u) < pInnov) {
            outSp = pickCompatibleNovelSpecies(nChem, nBio, myBioSp, mySpeciesMask, pi, frameU, 50u);
        }
    }
    let budThreshold = select(72.0, 120.0, outGrad == 1u);
    if (acc.blobCount > budThreshold && acc.blobLipidCount < 1.5 && nextEnergy > 1.04 && hashFloat(pi, frameU, 28u) < clamp(0.010 * dtBio, 0.0, 0.004)) {
        outKin = wrapAngle(nextKin + 0.55);
        outGene = mutateEncapGene(nextGene, pi, frameU, 29u, 0.5);
        outGrad = 0u;
    }

    // Gentle chemotaxis / foraging steer clamped so native asymmetric Particle Life blobs stay crisp
    var st = acc.steer * (normChem * 1.4) + acc.bioSteer * max(rate, 0.2);
    let stLen = length(st);
    let maxSt = 1.6 * max(rate, 0.1);
    if (stLen > maxSt) {
        st *= maxSt / stLen;
    }
    let trophicCue = clamp((predCap - autoCap) * 0.85, -1.0, 1.0);

    return BioStepResult(
        vec4f(nextEnergy, vit, poi, packLifeMetaGrad(1u, outSp, outKin, outGene, outGrad)),
        vec4f(st, trophicCue),
    );
}
