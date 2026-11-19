import { isAlive, type Snapshot } from './world.ts';

// Codes are written by the lethal branches in bioChem.wgsl and the bloom in field.wgsl.
const labels = [
  'Unattributed',
  'Starvation',
  'Overcrowding',
  'Predation',
  'Decomposition',
  'Poisoning',
  'Membrane failure',
  'Temperature stress',
  'UV exposure',
  'Desiccation',
  'Toxic bloom',
];

export function colonyLosses(
  members: number[],
  baseline: Snapshot,
  outcome: Snapshot,
  history: Snapshot[],
) {
  const counts = Array<number>(labels.length).fill(0);
  let unformedCount = 0;
  for (const index of members) {
    const episode = baseline.episodes[index];
    if (outcome.episodes[index] === episode && isAlive(outcome, index))
      continue;
    if (outcome.episodes[index] < episode) {
      // An earlier intervention can prevent a member of the original cohort
      // from forming. There is no death to attribute in the edited timeline.
      unformedCount++;
      continue;
    }
    // A four-bit slot records the death of this particular lifetime. If the slot
    // wrapped, recover it from an earlier checkpoint on this same timeline.
    const distance = (outcome.episodes[index] - episode) >>> 0;
    const source =
      distance < 8 || (distance === 8 && isAlive(outcome, index))
        ? outcome
        : history.find((state) => {
            if (state.tick > outcome.tick) return false;
            const delta = (state.episodes[index] - episode) >>> 0;
            return (
              (delta < 8 || (delta === 8 && isAlive(state, index))) &&
              (delta > 0 || !isAlive(state, index)) &&
              state.tick >= baseline.tick
            );
          });
    const cause = source
      ? (source.deathHistory[index] >>> ((episode & 7) * 4)) & 15
      : 0;
    counts[cause < labels.length ? cause : 0]++;
  }
  const ranked = counts
    .map((count, code) => ({ count, code }))
    .filter((row) => row.count > 0)
    .sort((a, b) => b.count - a.count);
  const entries = ranked.map(({ count, code }) => labels[code] + ' ' + count);
  if (unformedCount) entries.push('Did not form ' + unformedCount);
  return {
    qualifies: ranked.some(({ code }) => code !== 1 && code !== 2),
    summary: entries.join(' · ') || 'No cells lost',
  };
}
