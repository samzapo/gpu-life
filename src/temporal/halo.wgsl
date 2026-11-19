struct Halo { @builtin(position) position: vec4f, @location(0) corner: vec2f }
@group(0) @binding(0) var<uniform> view: Uniforms;
@group(0) @binding(1) var projectionMap: texture_2d<f32>;
@vertex fn vertex(@location(0) point: vec3f, @location(1) lifeMeta: f32,
    @location(2) episode: u32, @location(3) expected: u32, @builtin(vertex_index) vertexId: u32) -> Halo {
  let phase = unpackPhase(lifeMeta);
  if (expected == 0u || episode + 1u != expected || (phase != 1u && phase != 2u)) {
    return Halo(vec4f(2.0,2.0,2.0,1.0),vec2f(0.0));
  }
  let projected = projectUnitPos(view, projectionMap, normalize(point));
  if (projected.z < 0.5) {
    return Halo(vec4f(2.0,2.0,2.0,1.0),vec2f(0.0));
  }
  let corners = array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
  let corner = corners[vertexId];
  let radius = clamp(view.viewParams.x * 0.0055,4.2,10.0);
  let pixel = projected.xy * view.viewParams.x + view.viewSize*0.5 + view.viewParams.yz + corner*radius;
  let ndc = pixel / view.viewSize * 2.0 - 1.0;
  return Halo(vec4f(ndc.x,-ndc.y,0,1),corner);
}
@fragment fn fragment(input: Halo) -> @location(0) vec4f {
  let r = length(input.corner);
  if (r > 1.0 || r < 0.7) { discard; }
  return vec4f(0.60,1.0,0.89,1.0);
}
