// SPDX-License-Identifier: MIT OR Apache-2.0
// Directional/Blinn-Phong/fill lighting adapted from Triage
// crates/sim-graphics/src/shader.wgsl (MIT OR Apache-2.0).
struct BodyState {
    position: vec4<f32>,
    velocity: vec4<f32>,
    rotation: vec4<f32>,
    angular_velocity: vec4<f32>,
}
struct Instance {
    position: vec4<f32>,
    rotation: vec4<f32>,
    scale: vec4<f32>,
    color: vec4<f32>,
    surface: vec4<f32>,
    metadata: vec4<u32>,
    pbr: vec4<f32>,
}
struct Camera {
    view: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    eye: vec4<f32>,
}
struct Globals {
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    ambient: vec4<f32>,
    camera_index: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> bodies: array<BodyState>;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> cameras: array<Camera>;
@group(0) @binding(3) var<uniform> globals: Globals;

struct TextureInfo { data: vec4<u32> }
@group(0) @binding(4) var texture_pixels: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> texture_info: array<TextureInfo>;
struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) view_depth: f32,
    @location(3) @interpolate(flat) instance_index: u32,
    @location(4) uv: vec2<f32>,
}
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let instance = instances[instance_index];
    var world = rotate(instance.rotation, position * instance.scale.xyz) + instance.position.xyz;
    var world_normal = rotate(instance.rotation, normal / instance.scale.xyz);
    if instance.metadata.x != 0xffffffffu {
        let body = bodies[instance.metadata.x];
        world = rotate(body.rotation, world) + body.position.xyz;
        world_normal = rotate(body.rotation, world_normal);
    }
    let camera = cameras[globals.camera_index.x];
    // Construct stage outputs directly: function-local structs must not carry
    // the SPIR-V Offset decorations Naga emits for these interface types.
    return VertexOutput(
        camera.view_projection * vec4<f32>(world, 1.0),
        normalize(world_normal),
        world,
        -(camera.view * vec4<f32>(world, 1.0)).z,
        instance_index,
        uv,
    );
}
fn linear_to_srgb(rgb: vec3<f32>) -> vec3<f32> {
    let value = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(value, vec3<f32>(1.0 / 2.4)) - 0.055,
                  12.92 * value, value <= vec3<f32>(0.0031308));
}
fn wrap_texel(index: i32, size: i32, mode: u32) -> i32 {
    if mode == 2u { return clamp(index, 0, size - 1); }
    if mode == 1u {
        let period = size * 2;
        let i = ((index % period) + period) % period;
        return select(i, period - 1 - i, i >= size);
    }
    return ((index % size) + size) % size;
}
fn texel(info: vec4<u32>, xy: vec2<i32>) -> vec4<f32> {
    let x = wrap_texel(xy.x, i32(info.y), info.w & 3u);
    let y = wrap_texel(xy.y, i32(info.z), (info.w >> 2u) & 3u);
    return textureLoad(texture_pixels, vec2<i32>(x, y + i32(info.x)), 0);
}
fn base_color(texture_index: u32, uv: vec2<f32>) -> vec4<f32> {
    let info = texture_info[texture_index].data;
    let pixel = uv * vec2<f32>(f32(info.y), f32(info.z));
    if (info.w & 16u) == 0u { return texel(info, vec2<i32>(floor(pixel))); }
    let shifted = pixel - 0.5;
    let lo = vec2<i32>(floor(shifted));
    let t = fract(shifted);
    return mix(mix(texel(info, lo), texel(info, lo + vec2<i32>(1, 0)), t.x),
        mix(texel(info, lo + vec2<i32>(0, 1)), texel(info, lo + vec2<i32>(1, 1)), t.x), t.y);
}
struct SensorOutput {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
    @location(2) object_id: u32,
}
@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front: bool) -> SensorOutput {
    let instance = instances[input.instance_index];
    let camera = cameras[globals.camera_index.x];
    if !front && instance.metadata.w == 0u { discard; }
    let n = normalize(input.world_normal) * select(-1.0, 1.0, front);
    let albedo = instance.color * base_color(instance.metadata.z, input.uv);
    let l = normalize(globals.light_direction.xyz);
    let v = normalize(camera.eye.xyz - input.world_position);
    let h = normalize(l + v);
    let ndotl = max(dot(n, l), 0.0);
    let diffuse = globals.light_color.rgb * ndotl;
    let imported = instance.pbr.z > 0.0;
    let metallic = instance.pbr.x;
    let roughness = max(instance.pbr.y, 0.04);
    let shininess = select(instance.surface.x, clamp(2.0 / pow(roughness, 4.0) - 2.0, 1.0, 2048.0), imported);
    let reflectance = select(vec3<f32>(instance.surface.y), mix(vec3<f32>(0.04), albedo.rgb, metallic), imported);
    let specular = globals.light_color.rgb * reflectance
        * pow(max(dot(n, h), 0.0), max(shininess, 1.0))
        * select(0.0, 1.0, ndotl > 0.0);
    let fill_direction = normalize(vec3<f32>(-l.x * 0.7, 0.3, -l.z * 0.7));
    let fill = max(dot(n, fill_direction), 0.0) * vec3<f32>(0.28, 0.32, 0.38);
    var pattern = 1.0;
    if instance.surface.z > 0.0 {
        let cell = floor(input.world_position.xz * instance.surface.z);
        let parity = (i32(cell.x) + i32(cell.y)) & 1;
        pattern = mix(1.0, select(0.55, 1.0, parity == 0), clamp(instance.surface.w, 0.0, 1.0));
    }
    let diffuse_color = albedo.rgb * select(1.0, 1.0 - metallic, imported);
    let rgb = pattern * (albedo.rgb * globals.ambient.rgb + diffuse_color * (diffuse + fill)) + specular;
    return SensorOutput(
        vec4<f32>(linear_to_srgb(rgb), albedo.a),
        input.view_depth,
        instance.metadata.y,
    );
}
