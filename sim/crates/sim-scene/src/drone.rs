//! Visual-goal task catalogue: target designs, split districts and seeded episode candidates.
//!
//! Host-side and deterministic. Each candidate draws from its own stream keyed by
//! (seed, style, index), so results are independent of the visibility thread count.

use anyhow::{Result, bail, ensure};
use glam::Vec3;
use serde_json::json;
use sim_render::{Material, MeshData};
use sim_state::collision::LANDABLE;
use sim_state::drone::{self, BUCKETS, EpisodeSpec};

use crate::CityScene;

const SHAPES: [&str; 6] = ["sphere", "cube", "cylinder", "cone", "octahedron", "torus"];
const VARIANTS: usize = 8;
const SPLIT_SIZES: [usize; 3] = [32, 8, 8];
const DESIGN_SEED: u64 = 0x7A11_D351_6E00_0001;
const PALETTE: [[f32; 3]; VARIANTS] = [
    [0.85, 0.15, 0.12],
    [0.10, 0.35, 0.85],
    [0.95, 0.80, 0.10],
    [0.15, 0.70, 0.25],
    [0.95, 0.95, 0.95],
    [0.75, 0.20, 0.75],
    [0.95, 0.50, 0.10],
    [0.10, 0.75, 0.80],
];
/// Edge length of the unit design in metres: at the 20 m maximum start distance a 1.5 m
/// target spans about one 16 px encoder patch at 320x240 with the 75-degree camera.
const SCALE_RANGE: [f32; 2] = [1.5, 2.5];
const _: () = assert!(SPLIT_SIZES[0] + SPLIT_SIZES[1] + SPLIT_SIZES[2] == SHAPES.len() * VARIANTS);

const SURFACE_SAMPLES: usize = 64;
/// Visible fraction thresholds: bucket 0 at or above the first, bucket 1 at or above the second.
const BUCKET_EDGES: [f32; 2] = [0.5, 0.05];
const MIN_BUCKET: usize = 512;
const MIN_CANDIDATES: usize = 4096;
const MAX_ATTEMPTS: u32 = 256;
/// Validation and test districts are 96 m squares, so their half size is 48 m.
const DISTRICT_HALF: f32 = 48.0;
const EXCLUSION_MARGIN: f32 = 16.0;
const DISTRICT_MARGIN: f32 = 2.0;
const TARGET_HEIGHT: [f32; 2] = [2.0, 7.0];
const TARGET_CLEARANCE: f32 = 1.0;
/// Bottom of the target's enclosing sphere stays this far above the landable plane.
const TARGET_GROUND_CLEARANCE: f32 = 0.5;
const FLYING_HEIGHT: [f32; 2] = [1.5, 12.0];
const FLYING_CLEARANCE: f32 = 1.0;
/// 15 degrees.
const START_TILT_RAD: f32 = core::f32::consts::PI / 12.0;
const START_SPEED: f32 = 2.0;
const START_RATE: f32 = 0.5;
const START_TARGET_DISTANCE: [f32; 2] = [4.0, 20.0];
/// Aspect ratio used by the visibility frustum. Catalogue construction takes no image
/// size, so this is fixed here; renders must use the same aspect for the buckets to match.
const CATALOG_ASPECT: f32 = 4.0 / 3.0;

/// One target design. `index` is split-local; `scale` is the object's size in metres.
pub struct TargetDesign {
    pub name: String,
    pub mesh: MeshData,
    pub material: Material,
    pub scale: f32,
    pub split: u32,
    pub index: u32,
}

/// Returns the 48 designs ordered by split, then split-local index. The assignment is a
/// fixed seeded shuffle of the 6 shapes x 8 variants, cut into 32 train, 8 validation, 8 test.
pub fn target_designs() -> Vec<TargetDesign> {
    let mut order: Vec<usize> = (0..SHAPES.len() * VARIANTS).collect();
    let mut rng = Rng::new(DESIGN_SEED);
    for i in (1..order.len()).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let mut designs = Vec::with_capacity(order.len());
    let mut offset = 0;
    for (split, &count) in SPLIT_SIZES.iter().enumerate() {
        for (index, &id) in order[offset..offset + count].iter().enumerate() {
            designs.push(design(id, split as u32, index as u32));
        }
        offset += count;
    }
    designs
}

fn design(id: usize, split: u32, index: u32) -> TargetDesign {
    let shape = id / VARIANTS;
    let variant = id % VARIANTS;
    let mesh = match shape {
        0 => MeshData::sphere(24, 16),
        1 => MeshData::cube(),
        2 => MeshData::cylinder(24),
        3 => MeshData::cone(24),
        4 => MeshData::octahedron(),
        _ => MeshData::torus(24, 12, 0.35),
    };
    let [r, g, b] = PALETTE[variant];
    let material = Material {
        color: [r, g, b, 1.0],
        shininess: 32.0,
        checker_scale: if variant % 2 == 1 { 3.0 } else { 0.0 },
        checker_contrast: 0.5,
        ..Material::default()
    };
    let unit = Rng::new(DESIGN_SEED ^ id as u64).unit();
    let scale = SCALE_RANGE[0] + (SCALE_RANGE[1] - SCALE_RANGE[0]) * unit;
    TargetDesign { name: format!("{}-{}", SHAPES[shape], variant), mesh, material, scale, split, index }
}

/// Horizontal (x, z) region for one split. Start and target points must lie inside
/// `[min, max]` shrunk by a margin and outside every exclusion rectangle.
#[derive(Clone, Debug, PartialEq)]
pub struct District {
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub exclusions: Vec<([f32; 2], [f32; 2])>,
}

impl District {
    pub fn contains(&self, p: [f32; 2], margin: f32) -> bool {
        let inside = |lo: [f32; 2], hi: [f32; 2]| p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1];
        let lo = [self.min[0] + margin, self.min[1] + margin];
        let hi = [self.max[0] - margin, self.max[1] - margin];
        inside(lo, hi) && !self.exclusions.iter().any(|&(elo, ehi)| inside(elo, ehi))
    }
}

/// District for a split. `style` 0 is train, 1 validation, 2 test.
pub fn district(style: u32, half_extent: f32) -> District {
    let h = half_extent;
    let validation = [-0.4 * h, 0.4 * h];
    let test = [0.4 * h, 0.4 * h];
    let square = |c: [f32; 2], half: f32| ([c[0] - half, c[1] - half], [c[0] + half, c[1] + half]);
    match style {
        0 => District {
            min: [-h, -h],
            max: [h, h],
            exclusions: vec![
                square(validation, DISTRICT_HALF + EXCLUSION_MARGIN),
                square(test, DISTRICT_HALF + EXCLUSION_MARGIN),
            ],
        },
        1 => {
            let (min, max) = square(validation, DISTRICT_HALF);
            District { min, max, exclusions: Vec::new() }
        }
        2 => {
            let (min, max) = square(test, DISTRICT_HALF);
            District { min, max, exclusions: Vec::new() }
        }
        _ => panic!("style must be 0, 1 or 2, got {style}"),
    }
}

/// Candidate episodes for one split. `episodes` is sorted by bucket; `bucket_ranges` holds
/// `[first, count]` per bucket; `rules` records the generation rules and bucket counts.
pub struct Catalog {
    pub episodes: Vec<EpisodeSpec>,
    pub bucket_ranges: [[u32; 2]; BUCKETS],
    pub rules: serde_json::Value,
}

struct Candidate {
    bucket: usize,
    episode: EpisodeSpec,
}

/// Per-design data the candidate generator needs. Samples are in unit mesh space.
struct Slot {
    index: u32,
    scale: f32,
    radius: f32,
    samples: Vec<Vec3>,
}

impl Slot {
    fn new(design: &TargetDesign) -> Self {
        let radius = design.mesh.vertices.iter()
            .map(|v| Vec3::from_array(v.position).length())
            .fold(0.0, f32::max);
        Self { index: design.index, scale: design.scale, radius, samples: surface_samples(&design.mesh) }
    }
}

struct Ctx<'a> {
    scene: &'a CityScene,
    slots: Vec<Slot>,
    district: District,
    seed: u64,
    style: u32,
}

/// Builds the seeded catalogue of `size` candidates for one split. Errors if any bucket
/// holds fewer than 512 episodes or a candidate cannot be placed.
pub fn build_catalog(scene: &CityScene, style: u32, size: usize, seed: u64) -> Result<Catalog> {
    ensure!(style < 3, "style must be 0, 1 or 2, got {style}");
    ensure!(size >= MIN_CANDIDATES, "catalog needs at least {MIN_CANDIDATES} candidates, got {size}");
    let slots: Vec<Slot> = target_designs().iter().filter(|d| d.split == style).map(Slot::new).collect();
    let ctx = Ctx { scene, slots, district: district(style, scene.half_extent), seed, style };
    let candidates = generate(&ctx, size)?;
    let (episodes, bucket_ranges, counts) = finalize(candidates, MIN_BUCKET)?;
    let rules = rules(scene, &ctx, size, counts);
    Ok(Catalog { episodes, bucket_ranges, rules })
}

fn generate(ctx: &Ctx, size: usize) -> Result<Vec<Candidate>> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(size.max(1));
    let chunk = size.div_ceil(threads);
    let parts = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads).map(|t| {
            let lo = (t * chunk).min(size);
            let hi = ((t + 1) * chunk).min(size);
            scope.spawn(move || (lo..hi).map(|i| candidate(ctx, i)).collect::<Result<Vec<_>>>())
        }).collect();
        handles.into_iter().map(|h| h.join().expect("catalog worker panicked")).collect::<Vec<_>>()
    });
    let mut out = Vec::with_capacity(size);
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

fn candidate(ctx: &Ctx, index: usize) -> Result<Candidate> {
    let mut rng = Rng::stream(ctx.seed, ctx.style, index as u64);
    let slot = &ctx.slots[rng.below(ctx.slots.len())];
    for _ in 0..MAX_ATTEMPTS {
        if let Some(candidate) = attempt(ctx, slot, &mut rng) {
            return Ok(candidate);
        }
    }
    bail!("candidate {index} found no valid start and target in {MAX_ATTEMPTS} attempts")
}

fn attempt(ctx: &Ctx, slot: &Slot, rng: &mut Rng) -> Option<Candidate> {
    let scene = ctx.scene;
    let plane = scene.landable.height;
    let district = &ctx.district;

    // Target: hovering over the district, clear of non-landable geometry by 1 m beyond
    // its enclosing sphere.
    let target_xz = [rng.range(district.min[0], district.max[0]), rng.range(district.min[1], district.max[1])];
    if !district.contains(target_xz, DISTRICT_MARGIN + slot.scale) {
        return None;
    }
    let target_y = plane + rng.range(TARGET_HEIGHT[0], TARGET_HEIGHT[1]);
    let target = Vec3::new(target_xz[0], target_y, target_xz[1]);
    let enclosing = slot.scale * slot.radius;
    if target_y - enclosing < plane + TARGET_GROUND_CLEARANCE
        || scene.sweep_hits(target.to_array(), target.to_array(), TARGET_CLEARANCE + enclosing, LANDABLE) {
        return None;
    }

    // Start: landed on the plane or flying, at a 3-D distance inside the goal range.
    let landed = rng.unit() < 0.5;
    let start_y = if landed { plane + drone::RADIUS } else { plane + rng.range(FLYING_HEIGHT[0], FLYING_HEIGHT[1]) };
    let distance = rng.range(START_TARGET_DISTANCE[0], START_TARGET_DISTANCE[1]);
    let dy = start_y - target_y;
    if dy.abs() >= distance || distance <= enclosing + drone::SUCCESS_MARGIN + drone::RADIUS {
        return None;
    }
    let horizontal = (distance * distance - dy * dy).sqrt();
    let bearing = rng.range(0.0, core::f32::consts::TAU);
    let start_xz = [target_xz[0] + horizontal * bearing.cos(), target_xz[1] + horizontal * bearing.sin()];
    if !district.contains(start_xz, DISTRICT_MARGIN) {
        return None;
    }
    let start = Vec3::new(start_xz[0], start_y, start_xz[1]);
    let clearance = if landed { drone::RADIUS } else { FLYING_CLEARANCE };
    if scene.sweep_hits(start.to_array(), start.to_array(), clearance, LANDABLE) {
        return None;
    }

    // Attitude: yaw about world up, then a tilt of at most 15 degrees about a horizontal axis.
    let yaw = quat_axis_angle(Vec3::Y, rng.range(0.0, core::f32::consts::TAU));
    let rotation = if landed {
        yaw
    } else {
        let azimuth = rng.range(0.0, core::f32::consts::TAU);
        let axis = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
        quat_mul(quat_axis_angle(axis, rng.range(0.0, START_TILT_RAD)), yaw)
    };
    let (velocity, rates) = if landed {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (rng.in_ball(START_SPEED), rng.in_ball(START_RATE))
    };

    // Visibility from the start camera: each surface sample must be inside the frustum and
    // reachable from the eye along a radius-0 segment free of non-landable geometry.
    let (eye_offset, forward, up) = drone::camera_basis(rotation);
    let eye = start + Vec3::from_array(eye_offset);
    let forward = Vec3::from_array(forward);
    let up = Vec3::from_array(up);
    let right = forward.cross(up);
    let half_tan = (drone::CAMERA_VFOV_RAD * 0.5).tan();
    let visible = slot.samples.iter().copied()
        .filter(|s| visible_from(scene, eye, forward, up, right, half_tan, target + *s * slot.scale))
        .count();
    let fraction = visible as f32 / slot.samples.len() as f32;
    let bucket = if fraction >= BUCKET_EDGES[0] {
        0
    } else if fraction >= BUCKET_EDGES[1] {
        1
    } else {
        2
    };

    Some(Candidate {
        bucket,
        episode: EpisodeSpec {
            start_position: start.extend(0.0).to_array(),
            start_rotation: rotation,
            start_velocity: velocity.extend(0.0).to_array(),
            start_rates: rates.extend(0.0).to_array(),
            target: target.extend(0.0).to_array(),
            meta: [slot.index, bucket as u32, landed as u32, 0],
            stats: [fraction, start.distance(target), enclosing + drone::SUCCESS_MARGIN, 0.0],
        },
    })
}

fn visible_from(scene: &CityScene, eye: Vec3, forward: Vec3, up: Vec3, right: Vec3, half_tan: f32, point: Vec3) -> bool {
    let d = point - eye;
    let depth = d.dot(forward);
    if !(drone::CAMERA_NEAR..=drone::CAMERA_FAR).contains(&depth) {
        return false;
    }
    let half_h = depth * half_tan;
    if d.dot(up).abs() > half_h || d.dot(right).abs() > half_h * CATALOG_ASPECT {
        return false;
    }
    !scene.sweep_hits(eye.to_array(), point.to_array(), 0.0, LANDABLE)
}

/// Sorts candidates by bucket, stamps catalogue indices, and returns per-bucket
/// `[first, count]` ranges. Errors when any bucket holds fewer than `min_bucket` episodes.
fn finalize(mut candidates: Vec<Candidate>, min_bucket: usize)
    -> Result<(Vec<EpisodeSpec>, [[u32; 2]; BUCKETS], [usize; BUCKETS])> {
    let mut counts = [0usize; BUCKETS];
    for c in &candidates {
        counts[c.bucket] += 1;
    }
    for (bucket, &count) in counts.iter().enumerate() {
        ensure!(count >= min_bucket, "bucket {bucket} holds {count} episodes; at least {min_bucket} required");
    }
    candidates.sort_by_key(|c| c.bucket);
    let mut ranges = [[0u32; 2]; BUCKETS];
    let mut first = 0usize;
    for bucket in 0..BUCKETS {
        ranges[bucket] = [first as u32, counts[bucket] as u32];
        first += counts[bucket];
    }
    let episodes = candidates.into_iter().enumerate().map(|(index, c)| {
        let mut episode = c.episode;
        episode.meta[3] = index as u32;
        episode
    }).collect();
    Ok((episodes, ranges, counts))
}

fn rules(scene: &CityScene, ctx: &Ctx, size: usize, counts: [usize; BUCKETS]) -> serde_json::Value {
    json!({
        "style": ctx.style,
        "seed": ctx.seed,
        "candidates": size,
        "landable": { "height_m": scene.landable.height, "triangles": scene.landable.triangles,
            "non_level_triangles": scene.landable.non_level },
        "district": { "min": ctx.district.min, "max": ctx.district.max, "exclusions": ctx.district.exclusions },
        "out_of_bounds": { "margin_m": DISTRICT_MARGIN, "ceiling_y_m": drone::CEILING },
        "start": {
            "landed_fraction": 0.5,
            "landed_height_m": drone::RADIUS,
            "landed_clearance_m": drone::RADIUS,
            "flying_height_m": FLYING_HEIGHT,
            "flying_clearance_m": FLYING_CLEARANCE,
            "tilt_max_rad": START_TILT_RAD,
            "speed_max_mps": START_SPEED,
            "rate_max_rps": START_RATE,
            "start_target_distance_m": START_TARGET_DISTANCE,
            "flying_thrust": "hover",
        },
        "target": {
            "height_above_landable_m": TARGET_HEIGHT,
            "clearance_m": TARGET_CLEARANCE,
            "ground_clearance_m": TARGET_GROUND_CLEARANCE,
            "size_m": SCALE_RANGE,
            "success_radius": "enclosing sphere radius of the scaled design + 1.0 m, per episode (EpisodeSpec.stats[2])",
            "clearance_radius": "enclosing sphere of scaled design",
            "district_margin_m": DISTRICT_MARGIN,
        },
        "camera": {
            "aspect": CATALOG_ASPECT,
            "vfov_rad": drone::CAMERA_VFOV_RAD,
            "near_m": drone::CAMERA_NEAR,
            "far_m": drone::CAMERA_FAR,
        },
        "visibility": {
            "surface_samples": SURFACE_SAMPLES,
            "occlusion": "radius-0 segment eye to sample over non-landable triangles",
            "visible_min_fraction": BUCKET_EDGES[0],
            "partial_min_fraction": BUCKET_EDGES[1],
        },
        "min_bucket": MIN_BUCKET,
        "bucket_counts": counts,
    })
}

/// Area-weighted, stratified sample points on the design's mesh surface (unit space).
fn surface_samples(mesh: &MeshData) -> Vec<Vec3> {
    let triangles: Vec<[Vec3; 3]> = mesh.indices.chunks_exact(3).map(|t| {
        [t[0], t[1], t[2]].map(|i| Vec3::from_array(mesh.vertices[i as usize].position))
    }).collect();
    let areas: Vec<f32> = triangles.iter()
        .map(|t| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length())
        .collect();
    let total: f32 = areas.iter().sum();
    (0..SURFACE_SAMPLES).map(|k| {
        let target = (k as f32 + 0.5) / SURFACE_SAMPLES as f32 * total;
        let mut acc = 0.0;
        let mut pick = triangles.len() - 1;
        for (i, area) in areas.iter().enumerate() {
            acc += *area;
            if acc >= target {
                pick = i;
                break;
            }
        }
        let t = triangles[pick];
        let s = radical_inverse(k as u64, 2).sqrt();
        let v = radical_inverse(k as u64, 3);
        let (b0, b1) = (1.0 - s, v * s);
        t[0] * b0 + t[1] * b1 + t[2] * (1.0 - b0 - b1)
    }).collect()
}

fn radical_inverse(mut n: u64, base: u64) -> f32 {
    let inv = 1.0 / base as f32;
    let mut scale = inv;
    let mut value = 0.0;
    while n > 0 {
        value += (n % base) as f32 * scale;
        n /= base;
        scale *= inv;
    }
    value
}

/// Hamilton product for xyzw quaternions: returns a * b.
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_axis_angle(axis: Vec3, angle: f32) -> [f32; 4] {
    let (s, c) = ((0.5 * angle).sin(), (0.5 * angle).cos());
    let v = axis * s;
    [v.x, v.y, v.z, c]
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// SplitMix64 stream. Host-only; never used on device.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn stream(seed: u64, style: u32, index: u64) -> Self {
        Self(mix(mix(seed ^ mix(u64::from(style) + 0x51)) ^ index))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    /// Uniform in [0, 1) from the high 24 bits.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn in_ball(&mut self, radius: f32) -> Vec3 {
        loop {
            let v = Vec3::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            if v.length_squared() <= 1.0 {
                return v * radius;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn designs_have_fixed_split_counts_and_unique_names() {
        let designs = target_designs();
        assert_eq!(designs.len(), 48);
        for split in 0..3u32 {
            let members: Vec<_> = designs.iter().filter(|d| d.split == split).collect();
            assert_eq!(members.len(), SPLIT_SIZES[split as usize]);
            for (i, d) in members.iter().enumerate() {
                assert_eq!(d.index as usize, i);
                assert!((SCALE_RANGE[0]..=SCALE_RANGE[1]).contains(&d.scale));
            }
        }
        let mut names: Vec<_> = designs.iter().map(|d| d.name.clone()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 48);
    }

    #[test]
    fn districts_match_contract_geometry() {
        let h = 245.0;
        let validation = district(1, h);
        assert!((validation.min[0] + 146.0).abs() < 1e-3 && (validation.max[0] + 50.0).abs() < 1e-3);
        assert!((validation.min[1] - 50.0).abs() < 1e-3 && (validation.max[1] - 146.0).abs() < 1e-3);
        let test = district(2, h);
        assert!((test.min[0] - 50.0).abs() < 1e-3 && (test.max[0] - 146.0).abs() < 1e-3);
        let train = district(0, h);
        assert_eq!(train.exclusions.len(), 2);
        assert!(train.contains([0.0, 0.0], DISTRICT_MARGIN));
        assert!(!train.contains([-98.0, 98.0], DISTRICT_MARGIN));
        assert!(!train.contains([-h + 1.0, 0.0], DISTRICT_MARGIN));
        assert!(validation.contains([-98.0, 98.0], DISTRICT_MARGIN));
    }

    #[test]
    fn cube_surface_samples_lie_on_faces() {
        let samples = surface_samples(&MeshData::cube());
        assert_eq!(samples.len(), SURFACE_SAMPLES);
        for p in samples {
            let m = p.x.abs().max(p.y.abs()).max(p.z.abs());
            assert!((m - 0.5).abs() < 1e-5, "sample {p:?} is off the cube surface");
        }
    }

    #[test]
    fn finalize_orders_buckets_and_stamps_indices() {
        let candidate = |bucket: usize| Candidate {
            bucket,
            episode: EpisodeSpec { start_position: [0.0; 4], start_rotation: [0.0, 0.0, 0.0, 1.0],
                start_velocity: [0.0; 4], start_rates: [0.0; 4], target: [0.0; 4],
                meta: [0, bucket as u32, 0, 0], stats: [0.0; 4] },
        };
        let mut input = Vec::new();
        for _ in 0..3 { input.push(candidate(2)); }
        for _ in 0..2 { input.push(candidate(0)); }
        for _ in 0..4 { input.push(candidate(1)); }
        let (episodes, ranges, counts) = finalize(input, 2).unwrap();
        assert_eq!(counts, [2, 4, 3]);
        assert_eq!(ranges, [[0, 2], [2, 4], [6, 3]]);
        for (i, e) in episodes.iter().enumerate() {
            assert_eq!(e.meta[3] as usize, i);
        }
        assert_eq!(episodes[0].meta[1], 0);
        assert_eq!(episodes[8].meta[1], 2);
        assert!(finalize(vec![candidate(0), candidate(1), candidate(2)], 2).is_err());
    }

    #[test]
    fn quaternion_product_matches_glam() {
        let a = quat_axis_angle(Vec3::X, 0.2);
        let b = quat_axis_angle(Vec3::Y, 0.7);
        let ours = quat_mul(a, b);
        let reference = (glam::Quat::from_array(a) * glam::Quat::from_array(b)).to_array();
        for (x, y) in ours.iter().zip(reference) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "reads the real city; run with --ignored"]
    fn real_city_catalogs_fill_every_bucket_deterministically() {
        let scene = CityScene::load(std::path::Path::new("../../../assets/city")).unwrap();
        let size = 16 * MIN_CANDIDATES;
        for style in 0..3 {
            let started = std::time::Instant::now();
            let a = build_catalog(&scene, style, size, 7).unwrap();
            let elapsed = started.elapsed();
            let b = build_catalog(&scene, style, size, 7).unwrap();
            assert_eq!(a.episodes, b.episodes);
            assert_eq!(a.bucket_ranges, b.bucket_ranges);
            for range in a.bucket_ranges {
                assert!(range[1] as usize >= MIN_BUCKET);
            }
            eprintln!("style {style}: {size} candidates, bucket counts {}, build {elapsed:?}", a.rules["bucket_counts"]);
        }
    }
}
