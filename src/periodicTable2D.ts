export interface Element2D {
  atomicNumber: number;
  subshellLabel: string;
  name: string;
  period: number;
  block: 's' | 'p' | 'd' | 'f';
  outerCapacity: number;
  outerElectrons: number;
  /** Number of active outer valence slots (0 = noble gas, 1..3 = bonding slots). */
  valence: number;
  /** +1 = sheds electrons (donor), -1 = gains electrons (acceptor), 0 = half-shell covalent or noble gas. */
  role: -1 | 0 | 1;
  /** Body-frame slot angles (up to 3 slots) in radians. */
  slotAngles: [number, number, number];
  /** Visual RGB colour [0..1]. */
  color: [number, number, number];
}

interface SubshellSpec {
  period: number;
  block: 's' | 'p' | 'd' | 'f';
  capacity: number;
  outerCapacity: number;
}

// 2D Periodic Table subshell filling order (s=2, p=4, d=4, f=4;
// period lengths: 2, 6, 6, 10, 10, 14, 14).
const SUBSHELLS_2D: SubshellSpec[] = [
  { period: 1, block: 's', capacity: 2, outerCapacity: 2 },
  { period: 2, block: 's', capacity: 2, outerCapacity: 6 },
  { period: 2, block: 'p', capacity: 4, outerCapacity: 6 },
  { period: 3, block: 's', capacity: 2, outerCapacity: 6 },
  { period: 3, block: 'p', capacity: 4, outerCapacity: 6 },
  { period: 4, block: 's', capacity: 2, outerCapacity: 6 },
  { period: 4, block: 'd', capacity: 4, outerCapacity: 6 },
  { period: 4, block: 'p', capacity: 4, outerCapacity: 6 },
  { period: 5, block: 's', capacity: 2, outerCapacity: 6 },
  { period: 5, block: 'd', capacity: 4, outerCapacity: 6 },
  { period: 5, block: 'p', capacity: 4, outerCapacity: 6 },
  { period: 6, block: 's', capacity: 2, outerCapacity: 6 },
  { period: 6, block: 'f', capacity: 4, outerCapacity: 6 },
  { period: 6, block: 'd', capacity: 4, outerCapacity: 6 },
  { period: 6, block: 'p', capacity: 4, outerCapacity: 6 },
  { period: 7, block: 's', capacity: 2, outerCapacity: 6 },
  { period: 7, block: 'f', capacity: 4, outerCapacity: 6 },
  { period: 7, block: 'd', capacity: 4, outerCapacity: 6 },
  { period: 7, block: 'p', capacity: 4, outerCapacity: 6 },
];

function computeElementColor(
  period: number,
  valence: number,
  role: -1 | 0 | 1,
  z: number,
): [number, number, number] {
  const dim =
    period === 1 ? 1.0 : period === 2 ? 0.96 : Math.max(0.68, 1.0 - (period - 2) * 0.08);
  if (valence === 0) {
    // Noble gases (subtle cool indigo-slate)
    return [0.32 * dim, 0.38 * dim, 0.58 * dim];
  }
  if (z === 1) {
    // 2D Hydrogen (crisp white)
    return [0.92, 0.94, 0.98];
  }
  if (role === 0 && valence === 3) {
    // 2D Carbon / Silicon backbone (warm amber-gold in P2, teal in P3+)
    return period === 2 ? [1.0, 0.78, 0.22] : [0.22, 0.88, 0.78];
  }
  if (role === -1 && valence === 2) {
    // 2D Oxygen / Chalcogen analog (crimson / coral)
    return period === 2 ? [0.98, 0.26, 0.34] : [0.96, 0.52, 0.2];
  }
  if (role === -1 && valence === 1) {
    // 2D Halogen analog (bright green)
    return period === 2 ? [0.26, 0.95, 0.44] : [0.58, 0.95, 0.25];
  }
  if (role === 1 && valence === 1) {
    // 2D Alkali donor (magenta-violet)
    return period === 2 ? [0.88, 0.32, 0.95] : [0.7, 0.42, 1.0];
  }
  // Valence-2 donors / transition metals (sky blue)
  return [0.26 * dim, 0.68 * dim, 1.0 * dim];
}

export function getElement2D(speciesIndex: number): Element2D {
  const z = (Math.max(0, Math.floor(speciesIndex)) % 62) + 1;
  let remaining = z;
  let period = 1;
  let block: 's' | 'p' | 'd' | 'f' = 's';
  let outerCapacity = 2;
  let outerElectrons = 1;
  let subshellFill = 1;

  for (const sub of SUBSHELLS_2D) {
    if (remaining <= sub.capacity) {
      period = sub.period;
      block = sub.block;
      outerCapacity = sub.outerCapacity;
      subshellFill = remaining;
      if (sub.block === 's') {
        outerElectrons = remaining;
      } else if (sub.block === 'p') {
        outerElectrons = 2 + remaining;
      } else {
        outerElectrons = 2;
      }
      break;
    }
    remaining -= sub.capacity;
  }

  const toComplete = outerCapacity - outerElectrons;
  const toShed = outerElectrons;

  let valence = 0;
  let role: -1 | 0 | 1 = 0;

  if (toComplete === 0) {
    valence = 0;
    role = 0;
  } else if (toShed < toComplete) {
    valence = toShed;
    role = 1;
  } else if (toComplete < toShed) {
    valence = toComplete;
    role = -1;
  } else {
    valence = toShed;
    role = 0;
  }

  let slotAngles: [number, number, number] = [0, 0, 0];
  if (valence === 1) {
    slotAngles = [0, 0, 0];
  } else if (valence === 2) {
    // Bent 120° orbitals for p-acceptors, linear 180° for s-donors
    slotAngles = role === -1 ? [0, (2 * Math.PI) / 3, 0] : [0, Math.PI, 0];
  } else if (valence === 3) {
    slotAngles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
  }

  const subshellLabel = `${period}${block}${subshellFill}`;
  const roleLabel =
    valence === 0
      ? 'Noble'
      : role === 0
        ? `Covalent(${valence})`
        : role === 1
          ? `Donor(+${valence})`
          : `Acceptor(-${valence})`;

  return {
    atomicNumber: z,
    subshellLabel,
    name: `Z=${z} ${subshellLabel} [${outerElectrons}/${outerCapacity} ${roleLabel}]`,
    period,
    block,
    outerCapacity,
    outerElectrons,
    valence,
    role,
    slotAngles,
    color: computeElementColor(period, valence, role, z),
  };
}

/**
 * Pairwise electronic affinity based on 2D periodic table outer-shell completion/shedding.
 */
export function computePairAffinity(e1: Element2D, e2: Element2D): number {
  if (e1.valence === 0 || e2.valence === 0) {
    return 0;
  }
  if (e1.role !== 0 && e2.role !== 0 && e1.role !== e2.role) {
    return 1.35; // Donor sheds to complete Acceptor
  }
  if (e1.role === 0 && e2.role === 0) {
    // Half-shell covalent sharing (2D C-C, C-H, H-H)
    return e1.valence === 3 && e2.valence === 3 ? 1.25 : 1.05;
  }
  if (e1.role === 0 || e2.role === 0) {
    // Half-shell covalent sharing with donor or acceptor
    return 1.0;
  }
  // Same-polarity (Donor+Donor or Acceptor+Acceptor) repel
  return -0.55;
}

export const SPECIES_STRIDE_FLOATS = 8;

/**
 * Packs species metadata (`colourAmt * 8` floats) followed by pairwise affinity matrix
 * (`colourAmt * colourAmt` floats) into a single Float32Array for GPU storage binding.
 */
export function buildPackedChemistryBuffer(
  colourAmt: number,
  affinityMatrix?: number[][],
): {
  elements: Element2D[];
  packedData: Float32Array<ArrayBuffer>;
} {
  const elements: Element2D[] = [];
  const headerFloats = colourAmt * SPECIES_STRIDE_FLOATS;
  const packedData = new Float32Array(headerFloats + colourAmt * colourAmt);

  for (let c = 0; c < colourAmt; c++) {
    const el = getElement2D(c);
    elements.push(el);
    const base = c * SPECIES_STRIDE_FLOATS;
    packedData[base + 0] = el.valence;
    packedData[base + 1] = el.role;
    packedData[base + 2] = el.outerElectrons;
    packedData[base + 3] = el.outerCapacity;
    packedData[base + 4] = el.slotAngles[0];
    packedData[base + 5] = el.slotAngles[1];
    packedData[base + 6] = el.slotAngles[2];
    packedData[base + 7] = el.atomicNumber;
  }

  for (let c1 = 0; c1 < colourAmt; c1++) {
    for (let c2 = 0; c2 < colourAmt; c2++) {
      packedData[headerFloats + c1 * colourAmt + c2] = affinityMatrix
        ? affinityMatrix[c1][c2]
        : computePairAffinity(elements[c1], elements[c2]);
    }
  }

  return { elements, packedData };
}
