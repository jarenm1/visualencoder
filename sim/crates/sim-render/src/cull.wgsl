// SPDX-License-Identifier: MIT OR Apache-2.0
struct BodyState {
    position: vec4<f32>, velocity: vec4<f32>, rotation: vec4<f32>, angular_velocity: vec4<f32>,
}
struct Instance {
    position: vec4<f32>, rotation: vec4<f32>, scale: vec4<f32>, color: vec4<f32>,
    surface: vec4<f32>, metadata: vec4<u32>, pbr: vec4<f32>,
}
struct Camera { view: mat4x4<f32>, view_projection: mat4x4<f32>, eye: vec4<f32> }
struct DrawInfo { bounds_min: vec4<f32>, bounds_max: vec4<f32>, command: vec4<u32> }
struct DrawIndexed {
    index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32,
}
@group(0) @binding(0) var<storage, read> bodies: array<BodyState>;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> cameras: array<Camera>;
@group(0) @binding(3) var<storage, read> draws: array<DrawInfo>;
@group(0) @binding(4) var<storage, read_write> commands: array<DrawIndexed>;
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn outside(plane: vec4<f32>, center: vec3<f32>, radius: f32) -> bool {
    return dot(plane, vec4<f32>(center, 1.0)) < -radius * length(plane.xyz);
}
@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let draw_count = arrayLength(&draws);
    if id.x >= draw_count || id.y >= arrayLength(&cameras) { return; }
    let draw = draws[id.x];
    let instance = instances[draw.command.w];
    let local_center = (draw.bounds_min.xyz + draw.bounds_max.xyz) * 0.5;
    let half_size = (draw.bounds_max.xyz - draw.bounds_min.xyz) * 0.5;
    var center = rotate(instance.rotation, local_center * instance.scale.xyz) + instance.position.xyz;
    let radius = length(half_size * instance.scale.xyz);
    if instance.metadata.x != 0xffffffffu {
        let body = bodies[instance.metadata.x];
        center = rotate(body.rotation, center) + body.position.xyz;
    }
    let m = transpose(cameras[id.y].view_projection);
    // WebGPU clip volume is -w <= x,y <= w and 0 <= z <= w.
    let visible = !(outside(m[3] + m[0], center, radius)
        || outside(m[3] - m[0], center, radius)
        || outside(m[3] + m[1], center, radius)
        || outside(m[3] - m[1], center, radius)
        || outside(m[2], center, radius)
        || outside(m[3] - m[2], center, radius));
    commands[id.y * draw_count + id.x] = DrawIndexed(draw.command.x,
        select(0u, 1u, visible), draw.command.y, bitcast<i32>(draw.command.z), draw.command.w);
}
