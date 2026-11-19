// SPDX-License-Identifier: MIT OR Apache-2.0
// One workgroup per camera. Each thread culls a contiguous range of draws; a workgroup prefix sum
// gives every thread its output offset, so visible commands are written compacted and in their
// original order (deterministic depth ties), followed by the camera's command count.
struct BodyState {
    position: vec4<f32>, velocity: vec4<f32>, rotation: vec4<f32>, angular_velocity: vec4<f32>,
}
struct Instance {
    position: vec4<f32>, rotation: vec4<f32>, scale: vec4<f32>, color: vec4<f32>,
    surface: vec4<f32>, metadata: vec4<u32>, pbr: vec4<f32>,
}
struct Camera { view: mat4x4<f32>, view_projection: mat4x4<f32>, eye: vec4<f32> }
// Levels of detail 1..4: first index, index count (0 = absent), error in mesh units (renderer.rs).
struct DrawInfo {
    bounds_min: vec4<f32>, bounds_max: vec4<f32>, command: vec4<u32>,
    lod_first: vec4<u32>, lod_count: vec4<u32>, lod_error: vec4<f32>,
}
struct DrawIndexed {
    index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32,
}
// tile_height in pixels; lod_pixels as in RenderConfig (0 always draws the full mesh).
struct Params { tile_height: f32, lod_pixels: f32, pad: vec2<f32> }
@group(0) @binding(0) var<storage, read> bodies: array<BodyState>;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> cameras: array<Camera>;
@group(0) @binding(3) var<storage, read> draws: array<DrawInfo>;
@group(0) @binding(4) var<storage, read_write> commands: array<DrawIndexed>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> counts: array<u32>;

const THREADS: u32 = 256u;
var<workgroup> offsets: array<u32, 256>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn outside(plane: vec4<f32>, center: vec3<f32>, radius: f32) -> bool {
    return dot(plane, vec4<f32>(center, 1.0)) < -radius * length(plane.xyz);
}

// The command this camera draws for `index`; instance_count 0 when culled.
fn select_command(index: u32, camera_index: u32) -> DrawIndexed {
    let draw = draws[index];
    let instance = instances[draw.command.w];
    let local_center = (draw.bounds_min.xyz + draw.bounds_max.xyz) * 0.5;
    let half_size = (draw.bounds_max.xyz - draw.bounds_min.xyz) * 0.5;
    var center = rotate(instance.rotation, local_center * instance.scale.xyz) + instance.position.xyz;
    let radius = length(half_size * instance.scale.xyz);
    if instance.metadata.x != 0xffffffffu {
        let body = bodies[instance.metadata.x];
        center = rotate(body.rotation, center) + body.position.xyz;
    }
    // Camera filter from metadata.z: 0 draws in every camera, otherwise only camera z - 1.
    let owned = instance.metadata.z == 0u || instance.metadata.z == camera_index + 1u;
    let camera = cameras[camera_index];
    let m = transpose(camera.view_projection);
    // WebGPU clip volume is -w <= x,y <= w and 0 <= z <= w.
    let visible = owned && !(outside(m[3] + m[0], center, radius)
        || outside(m[3] - m[0], center, radius)
        || outside(m[3] + m[1], center, radius)
        || outside(m[3] - m[1], center, radius)
        || outside(m[2], center, radius)
        || outside(m[3] - m[2], center, radius));
    // Pixels per metre at distance d is focal / d, focal = P[1][1] * tile_height / 2. Row 1 of the
    // view-projection is P[1][1] times the (unit) camera up axis of the rigid view.
    let focal = length(m[1].xyz) * params.tile_height * 0.5;
    let to_center = length(center - camera.eye.xyz);
    var index_count = draw.command.x;
    var first_index = draw.command.y;
    if visible && params.lod_pixels > 0.0 {
        // Error is judged at the nearest point of the bounding sphere (conservative).
        let scale = max(instance.scale.x, max(instance.scale.y, instance.scale.z));
        let nearest = max(to_center - radius, 0.05);
        for (var level = 0u; level < 4u; level++) {
            if draw.lod_count[level] == 0u || draw.lod_error[level] * scale * focal > params.lod_pixels * nearest {
                break;
            }
            index_count = draw.lod_count[level];
            first_index = draw.lod_first[level];
        }
    }
    return DrawIndexed(index_count, select(0u, 1u, visible), first_index, bitcast<i32>(draw.command.z),
        draw.command.w);
}

@compute @workgroup_size(256)
fn cull(@builtin(local_invocation_index) thread: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let camera_index = group.y;
    let draw_count = arrayLength(&draws);
    let per_thread = (draw_count + THREADS - 1u) / THREADS;
    let first = min(thread * per_thread, draw_count);
    let last = min(first + per_thread, draw_count);
    var visible = 0u;
    for (var index = first; index < last; index++) {
        visible += select_command(index, camera_index).instance_count;
    }
    // Inclusive Hillis-Steele scan of the per-thread visible counts.
    offsets[thread] = visible;
    workgroupBarrier();
    for (var stride = 1u; stride < THREADS; stride *= 2u) {
        var add = 0u;
        if thread >= stride { add = offsets[thread - stride]; }
        workgroupBarrier();
        offsets[thread] += add;
        workgroupBarrier();
    }
    var slot = camera_index * draw_count + offsets[thread] - visible;
    for (var index = first; index < last; index++) {
        let command = select_command(index, camera_index);
        if command.instance_count != 0u {
            commands[slot] = command;
            slot += 1u;
        }
    }
    if thread == THREADS - 1u {
        counts[camera_index] = offsets[thread];
    }
}
