struct Cell {
  pos: vec3f, colour: f32, vel: vec3f, omega: f32, orient: vec3f, padding: f32,
}
struct Field { source: vec4f, bloom: vec4f }
@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<storage, read_write> life: array<vec4f>;
@group(0) @binding(2) var<uniform> field: Field;

// Apply the intervention or toxic bloom in geographic coordinates, after the
// regular globe solver. Matter remains in the world when living tissue dies.
@group(0) @binding(3) var<storage, read> previousLife: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> episodes: array<u32>;

// First half stores birth episodes for the halo. Second half stores the last
// eight death codes per particle, indexed by the episode that actually died.
fn recordDeath(index: u32, cause: u32) {
  let address = arrayLength(&cells) + index;
  let shift = (episodes[index] & 7u) * 4u;
  episodes[address] = (episodes[address] & ~(15u << shift)) | ((cause & 15u) << shift);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= arrayLength(&cells)) { return; }
  let packedState = life[index].w;
  let phase = u32(max(0.0, round(packedState))) & 3u;
  let prior = u32(max(0.0, round(previousLife[index].w))) & 3u;
  if (phase != 1u && phase != 2u) {
    if (prior == 1u || prior == 2u) {
      recordDeath(index, u32(max(0.0, -packedState * 0.25)));
      if (packedState < 0.0) { life[index].w = 0.0; }
    }
    return;
  }
  if (prior != 1u && prior != 2u) { episodes[index] += 1u; }
  var position = cells[index].pos;
  let distance = length(field.source.xyz - position);
  if (field.source.w != 0.0 && distance < 0.22 && distance > 0.0001) {
    let tangent = field.source.xyz - position * dot(position, field.source.xyz);
    position = normalize(position + normalize(tangent) * field.source.w * 0.007 * (1.0 - distance / 0.22));
    cells[index].pos = position;
  }
  let radius = field.bloom.w;
  if (radius > 0.0 && length(position - field.bloom.xyz) < radius) {
    let exposure = 1.0 - smoothstep(radius * 0.65, radius, length(position - field.bloom.xyz));
    let energy = life[index].x - 0.12 * exposure;
    life[index].x = max(0.0, energy);
    life[index].z = min(1.0, life[index].z + 0.065 * exposure);
    if (energy <= 0.035 || life[index].z >= 0.96) {
      recordDeath(index, 10u); // Toxic bloom
      life[index] = vec4f(0.0);
    }
  }
}
