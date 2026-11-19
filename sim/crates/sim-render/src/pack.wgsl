// SPDX-License-Identifier: MIT OR Apache-2.0
// Camera c's image is the atlas tile at column c % columns, row c / columns (renderer.rs `Atlas`).
struct Grid { tile_width: u32, tile_height: u32, columns: u32, cameras: u32 }
@group(0) @binding(0) var colors: texture_2d<f32>;
@group(0) @binding(1) var depths: texture_2d<f32>;
@group(0) @binding(2) var object_ids: texture_2d<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> grid: Grid;

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) pixel: vec3<u32>) {
    if pixel.x >= grid.tile_width || pixel.y >= grid.tile_height || pixel.z >= grid.cameras {
        return;
    }
    let index = (pixel.z * grid.tile_height + pixel.y) * grid.tile_width + pixel.x;
    let region_words = grid.tile_width * grid.tile_height * grid.cameras;
    let tile = vec2<u32>(pixel.z % grid.columns, pixel.z / grid.columns);
    let coordinate = vec2<i32>(tile * vec2<u32>(grid.tile_width, grid.tile_height) + pixel.xy);
    output[index] = pack4x8unorm(textureLoad(colors, coordinate, 0));
    output[region_words + index] = bitcast<u32>(textureLoad(depths, coordinate, 0).x);
    output[2u * region_words + index] = textureLoad(object_ids, coordinate, 0).x;
}
