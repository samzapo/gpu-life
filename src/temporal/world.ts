export type Point = [number, number, number];
export type Force = {
  point: Point;
  direction: Point;
  strength: number;
  radius: number;
};
export type Bloom = {
  point: Point;
  startTick: number;
  endTick: number;
  radius: number;
};
export interface Snapshot {
  tick: number;
  frame: number;
  particles: Float32Array<ArrayBuffer>;
  life: Float32Array<ArrayBuffer>;
  episodes: Uint32Array<ArrayBuffer>;
  /** Eight four-bit cause codes per particle, indexed by birth episode. */
  deathHistory: Uint32Array<ArrayBuffer>;
  inspection?: {
    valence: Float32Array<ArrayBuffer>;
    bioForce: Float32Array<ArrayBuffer>;
  };
}
export interface Colony {
  members: number[];
  center: Point;
  radius?: number;
}

export const ticksPerSecond = 20;

export function isAlive(snapshot: Snapshot, index: number) {
  const phase = Math.round(snapshot.life[index * 4 + 3]) & 3;
  return phase === 1 || phase === 2;
}

export function countSurvivors(
  snapshot: Snapshot,
  members: number[],
  baseline: Snapshot,
) {
  return members.reduce(
    (count, index) =>
      count +
      Number(
        isAlive(snapshot, index) &&
          snapshot.episodes[index] === baseline.episodes[index],
      ),
    0,
  );
}

// Match by material membership across samples. All surviving fragments count,
// even when a current carries them outside the original neighborhood.
export function findColonies(snapshot: Snapshot, radius: number): Colony[] {
  const buckets = new Map<number, number[]>();
  const cellsPerAxis = Math.ceil(2 / radius) + 3;
  const bucketKey = (x: number, y: number, z: number) =>
    (x * cellsPerAxis + y) * cellsPerAxis + z;
  const cell = (value: number) => Math.floor((value + 1) / radius) + 1;
  const particles = snapshot.particles;
  const particleCount = snapshot.life.length / 4;
  const visited = new Uint8Array(particleCount);
  const living: number[] = [];
  for (let index = 0; index < particleCount; index++) {
    if (!isAlive(snapshot, index)) continue;
    living.push(index);
    const offset = index * 12;
    const key = bucketKey(
      cell(particles[offset]),
      cell(particles[offset + 1]),
      cell(particles[offset + 2]),
    );
    const bucket = buckets.get(key);
    if (bucket) bucket.push(index);
    else buckets.set(key, [index]);
  }
  const colonies: Colony[] = [];
  for (const seed of living) {
    if (visited[seed]) continue;
    visited[seed] = 1;
    const members = [seed];
    const center: Point = [0, 0, 0];
    for (let cursor = 0; cursor < members.length; cursor++) {
      const index = members[cursor];
      const offset = index * 12;
      const x = particles[offset],
        y = particles[offset + 1],
        z = particles[offset + 2];
      center[0] += x;
      center[1] += y;
      center[2] += z;
      const gx = cell(x),
        gy = cell(y),
        gz = cell(z);
      const kin = (Math.round(snapshot.life[index * 4 + 3]) >>> 10) & 511;
      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            const bucket = buckets.get(bucketKey(gx + dx, gy + dy, gz + dz));
            if (!bucket) continue;
            for (const other of bucket) {
              if (visited[other]) continue;
              const next = other * 12;
              const distance2 =
                (particles[next] - x) ** 2 +
                (particles[next + 1] - y) ** 2 +
                (particles[next + 2] - z) ** 2;
              if (distance2 > radius * radius) continue;
              const delta = Math.abs(
                kin - ((Math.round(snapshot.life[other * 4 + 3]) >>> 10) & 511),
              );
              if (
                Math.min(delta, 512 - delta) > 25 &&
                distance2 > radius * radius * 0.3
              )
                continue;
              visited[other] = 1;
              members.push(other);
            }
          }
    }
    if (members.length < 5) continue;
    const length = Math.hypot(...center) || 1;
    const unitCenter: Point = [
      center[0] / length,
      center[1] / length,
      center[2] / length,
    ];
    const distances = members
      .map((memberIndex) => {
        const offset = memberIndex * 12;
        return Math.hypot(
          particles[offset] - unitCenter[0],
          particles[offset + 1] - unitCenter[1],
          particles[offset + 2] - unitCenter[2],
        );
      })
      .sort((a, b) => a - b);
    const colonyRadius = distances[Math.floor(distances.length * 0.85)] ?? 0.06;
    colonies.push({
      members,
      center: unitCenter,
      radius: colonyRadius,
    });
  }
  return colonies.sort(
    (left, right) => right.members.length - left.members.length,
  );
}

export async function readCheckpoint(
  device: GPUDevice,
  particles: GPUBuffer,
  life: GPUBuffer,
  tracking: GPUBuffer,
  tick: number,
  frame: number,
  inspection?: { valence: GPUBuffer; bioForce: GPUBuffer },
): Promise<Snapshot> {
  const valenceOffset = particles.size + life.size + tracking.size;
  const bioForceOffset = valenceOffset + (inspection?.valence.size ?? 0);
  const staging = device.createBuffer({
    size: bioForceOffset + (inspection?.bioForce.size ?? 0),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(particles, 0, staging, 0, particles.size);
    encoder.copyBufferToBuffer(life, 0, staging, particles.size, life.size);
    encoder.copyBufferToBuffer(
      tracking,
      0,
      staging,
      particles.size + life.size,
      tracking.size,
    );
    if (inspection) {
      encoder.copyBufferToBuffer(
        inspection.valence,
        0,
        staging,
        valenceOffset,
        inspection.valence.size,
      );
      encoder.copyBufferToBuffer(
        inspection.bioForce,
        0,
        staging,
        bioForceOffset,
        inspection.bioForce.size,
      );
    }
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const bytes = staging.getMappedRange().slice(0);
    const particleCount = life.size / (4 * Float32Array.BYTES_PER_ELEMENT);
    return {
      tick,
      frame,
      inspection: inspection && {
        valence: new Float32Array(
          bytes,
          valenceOffset,
          inspection.valence.size / 4,
        ),
        bioForce: new Float32Array(
          bytes,
          bioForceOffset,
          inspection.bioForce.size / 4,
        ),
      },
      particles: new Float32Array(bytes, 0, particles.size / 4),
      life: new Float32Array(bytes, particles.size, life.size / 4),
      episodes: new Uint32Array(
        bytes,
        particles.size + life.size,
        particleCount,
      ),
      deathHistory: new Uint32Array(
        bytes,
        particles.size +
          life.size +
          particleCount * Uint32Array.BYTES_PER_ELEMENT,
        particleCount,
      ),
    };
  } finally {
    staging.destroy();
  }
}
