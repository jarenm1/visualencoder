// SPDX-License-Identifier: MIT OR Apache-2.0
@group(0) @binding(0) var colors: texture_2d_array<f32>;
@group(0) @binding(1) var depths: texture_2d_array<f32>;
@group(0) @binding(2) var object_ids: texture_2d_array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let size = textureDimensions(colors);
    let layers = textureNumLayers(colors);
    if pixel.x >= size.x || pixel.y >= size.y || pixel.z >= layers {
        return;
    }
    let index = (pixel.z * size.y + pixel.y) * size.x + pixel.x;
    let region_words = size.x * size.y * layers;
    let coordinate = vec2<i32>(pixel.xy);
    let layer = i32(pixel.z);
    output[index] = pack4x8unorm(textureLoad(colors, coordinate, layer, 0));
    output[region_words + index] = bitcast<u32>(textureLoad(depths, coordinate, layer, 0).x);
    output[2u * region_words + index] = textureLoad(object_ids, coordinate, layer, 0).x;
}
