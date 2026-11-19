// Unified conserved-mass emergent blob ecology + biogenic lipid bilayers + bio-chemistry helpers.
// Every particle carries an immutable `chemSpecies` (`p.colour`), live `phase`
// (0 = Chemistry, 1 = Particle Life Organism Cytoplasm, 2 = Biogenic Amphiphilic Lipid Bilayer),
// `bioSpecies` (0..M-1), circular `kinPhase`, and heritable 4-bit `encapGene` (0..15) packed in `lifeState.w`.
// Producer / Consumer / Mixotroph / Decomposer / Halophile identities emerge
// directly from the asymmetric Particle Life matrix traits, self-propulsion speed,
// and conserved elemental makeup (C, H, O, S) of each blob's constituent particles.

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

fn evalSolarInsolation(pos: vec3f) -> f32 {
    let cosLat = sqrt(max(0.0, 1.0 - pos.z * pos.z));
    return 0.12 + 0.88 * pow(cosLat, 1.25);
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
}

fn initBioUnifiedAccum() -> BioUnifiedAccum {
    var acc = BioUnifiedAccum();
    acc.actScale = 1.0;
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
        if (myPhase == 1u) {
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

    let nbEnergy = clamp(nbLifeState.x, 0.0, 1.6);
    let nbVit = clamp(nbLifeState.y, 0.0, 1.0);
    let nbPoi = clamp(nbLifeState.z, 0.0, 1.0);
    let nbKin = unpackKinPhase(nbLifeState.w);
    let nbGene = unpackEncapGene(nbLifeState.w);
    let nbGrad = unpackGraduated(nbLifeState.w);

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

fn finalizeBioUnified(
    pi: u32,
    pos: vec3f,
    vel: vec3f,
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
        if (myPhase != 0u) {
            return BioStepResult(vec4f(max(curLifeState.x, 0.55), curLifeState.y, curLifeState.z, curLifeState.w), vec4f(0.0));
        }
        return BioStepResult(vec4f(0.0), vec4f(0.0));
    }

    let rate = max(bio.rate, 0.0);
    let dtBio = bio.dt * rate;
    let solar = evalSolarInsolation(pos);
    let upwell = evalCoastalUpwelling(oc);
    let landFrac = clamp(1.0 - smoothstep(0.18, 0.48, oc.w), 0.0, 1.0);
    let frameU = u32(max(bio.frame, 0.0)) + u32(max(bio.seedHash, 0.0));
    let nChem = max(u32(bio.chemColours), 1u);
    let nBio = max(u32(bio.bioColours), 1u);

    if (myPhase == 0u) {
        let chemForce = finalizeChemBioForce(acc, bio);
        if (!isBiogenicElement(atomicNum, valence)) {
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
        let bondEase = 1.0 / (1.0 + 0.55 * max(atomLoad, 0.0));
        let envMedia = mix(0.55 + 0.85 * upwell, 0.44 + 0.22 * clamp(acc.nearLipidCount * 0.25, 0.0, 1.0), landFrac);
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
                    0.85 * zipBoost * solar * envMedia * clamp(max(adjAvgE, avgLipidE), 0.5, 1.4) * lipidHeadroom * bondEase * dtBio,
                    0.0,
                    0.18,
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
                    0.32 * select(1.0, 1.25, hostGrad) * solar * envMedia * fixPow * encapHeadroom * bondEase * dtBio,
                    0.0,
                    0.085,
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
                    0.36 * solar * envMedia * min(acc.nearFixer, 3.0) * elemAffinity * crowdHeadroom * bondEase * dtBio,
                    0.0,
                    0.085,
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
            let pSpore = clamp(0.0028 * solar * (0.50 + upwell) * dtBio, 0.0, 0.0045);
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
        let upkeep = select(0.55, select(0.055, 0.036, gradNow), lipHealthy);
        let poisonPenalty = 0.24 * poi;
        let landPenalty = 0.045 * landFrac;
        let nextEnergy = clamp(eDiffused - (upkeep + poisonPenalty + landPenalty) * dtVit, 0.02, 1.45);
        let hostIsGrad = gradNow && (nextEnergy >= 0.36) && (poi < 0.72);

        let hostColors = countOneBits(acc.blobSpeciesMask);
        let simpleHostStress = (acc.blobCount >= 1.0 && hostColors <= 2u && acc.blobLipidCount >= 2.0);
        let orphanStress = (acc.blobCount < 0.5 && nextEnergy < 0.52);
        if (nextEnergy <= 0.14 || poi >= 0.85 || orphanStress || simpleHostStress) {
            let pDissolve = select(clamp(0.65 * dtBio, 0.0, 0.18), clamp(2.4 * dtBio, 0.0, 0.35), simpleHostStress);
            if (hashFloat(pi, frameU, 31u) < pDissolve) {
                return BioStepResult(
                    vec4f(0.0),
                    vec4f(0.0, 0.0, 0.0, packDeltaActGlow(0.9, -0.7)),
                );
            }
        }
        if (acc.nearPredThreat > 1.15) {
            let pBreach = clamp(0.20 * (acc.nearPredThreat - 1.0) * select(1.0, 0.42, hostIsGrad) * dtBio, 0.0, 0.05);
            if (hashFloat(pi, frameU, 32u) < pBreach) {
                return BioStepResult(
                    vec4f(0.0),
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

    let moistureFactor = mix(1.0, 0.74 + 0.16 * membraneShield, landFrac);
    let solarGain = 0.56 * solar * mix(0.55 + 0.85 * upwell, 0.55, landFrac) * autoCap * moistureFactor;
    let haloGain = select(0.0, 0.24 * min(acc.nearHalogen, 2.5) * haloTol, haloTol > 0.52);
    let detritusGain = 0.26 * min(acc.nearStressedBio, 2.5) * decompCap;
    let grazeEnvGain = 0.34 * min(acc.nearForeignPrey, 2.5) * predCap;
    let chemBonus = 0.24 * acc.dE * normChem * (1.0 + 0.35 * landFrac);
    let basalBurn = (0.13 + 0.11 * motility * motility + 0.04 * avgAsym) * (1.0 - 0.14 * membraneShield - 0.26 * isGradNowF);
    let foreignBio = max(0.0, acc.nearTotalBio - acc.blobCount);
    let ownCrowdStart = select(56.0, 145.0, isGradNow);
    let ownCrowdSlope = select(0.020, 0.008, isGradNow);
    let foreignCrowdScale = mix(1.0, 0.35, isGradNowF * membraneShield);
    let crowdPenalty = 0.032 * max(0.0, foreignBio - 12.0) * foreignCrowdScale + ownCrowdSlope * max(0.0, acc.blobCount - ownCrowdStart);
    let poisonPenalty = 0.28 * poi * (1.0 - 0.65 * clamp(haloTol - 0.35, 0.0, 1.0));
    let desiccationShield = 0.55 * membraneShield;
    let landPenalty = 0.075 * landFrac * (1.0 - desiccationShield);

    let netDE = solarGain + haloGain + detritusGain + grazeEnvGain + 0.06 * vit + chemBonus
        - basalBurn - crowdPenalty - poisonPenalty - landPenalty;
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
                    vec4f(0.0),
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
                    vec4f(0.0),
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
            return BioStepResult(
                vec4f(0.0),
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
    if (acc.blobLipidCount >= 2.0 && nextEnergy > 0.86 && myDistinctColors < maxNovelColors && acc.blobCount >= 7.0) {
        let pInnov = clamp(0.018 * encapNoveltyDrive(nextGene, myDistinctColors) * dtBio, 0.0, 0.008);
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
