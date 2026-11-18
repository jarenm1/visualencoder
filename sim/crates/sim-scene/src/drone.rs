//! Visual-goal task catalogue: city-object targets, split districts and seeded episodes.
//!
//! Host-side and deterministic. Each candidate draws from its own stream keyed by
//! (seed, style, index), so results are independent of the visibility thread count.
//!
//! A target is one placed asset of the city layout (building, vehicle, vegetation or
//! prop) lying wholly inside the split's district. Every episode also fixes a goal
//! reference camera that sees the object and the crop of that view the policy receives
//! as its goal: either the start view itself (the operator taps the object in the first
//! frame) or an independent nearby viewpoint.

use anyhow::{Result, bail, ensure};
use glam::Vec3;
use serde_json::json;
use sim_state::collision::LANDABLE;
use sim_state::drone::{self, BUCKETS, EpisodeSpec, box_distance, segment_box_entry};

use crate::{CityObject, CityScene};

/// Layout classes that can be targets: the discrete, solid objects of the city.
pub const TARGET_CLASSES: [&str; 4] = ["building", "vehicle", "vegetation", "prop"];
/// Box-surface sample points per visibility estimate.
const BOX_SAMPLES: usize = 64;
/// Visible fraction thresholds: bucket 0 at or above the first, bucket 1 at or above the second.
const BUCKET_EDGES: [f32; 2] = [0.5, 0.05];
const MIN_BUCKET: usize = 512;
const MIN_CANDIDATES: usize = 4096;
/// Placement attempts per drawn object, and object draws per candidate.
const MAX_ATTEMPTS: u32 = 32;
const MAX_OBJECTS: u32 = 64;
/// Validation and test districts are 96 m squares, so their half size is 48 m.
const DISTRICT_HALF: f32 = 48.0;
const EXCLUSION_MARGIN: f32 = 16.0;
const DISTRICT_MARGIN: f32 = 2.0;
/// Targets need at least this extent (metres) along some axis.
const MIN_TARGET_EXTENT: f32 = 0.5;
/// Classes whose members look alike and stand in clusters (forest trees, rocks, lamps) must
/// be at least `ISOLATION_GAP` metres (horizontal box gap) from any object of their class,
/// so a goal crop identifies one object. Buildings differ by design; vehicles stand apart.
const ISOLATED_CLASSES: [&str; 2] = ["vegetation", "prop"];
const ISOLATION_GAP: f32 = 2.0;
const FLYING_HEIGHT: [f32; 2] = [1.5, 12.0];
const FLYING_CLEARANCE: f32 = 1.0;
/// 15 degrees.
const START_TILT_RAD: f32 = core::f32::consts::PI / 12.0;
const START_SPEED: f32 = 2.0;
const START_RATE: f32 = 0.5;
/// Start distance from the target's box surface; always beyond the standoff band.
const START_BOX_DISTANCE: [f32; 2] = [8.0, 30.0];
/// Goal reference camera: distance from the box surface and height above the landable plane,
/// in the range the drone itself flies (bird's-eye references made unrecognizable goals).
const REFERENCE_BOX_DISTANCE: [f32; 2] = [5.0, 25.0];
const REFERENCE_HEIGHT: [f32; 2] = [1.5, 12.0];
/// Yaw and pitch jitter of the reference camera away from the box centre (~11 degrees).
const REFERENCE_JITTER_RAD: f32 = 0.2;
const REFERENCE_CLEARANCE: f32 = 0.5;
const REFERENCE_MIN_VISIBLE: f32 = 0.5;
const REFERENCE_ATTEMPTS: u32 = 64;
/// Visible starts reuse the start view as the goal reference with this probability.
const START_REFERENCE_PROBABILITY: f32 = 0.5;
/// Goal crop: projected box bounds grown by this fraction per side, squared in normalized
/// units (4:3 in pixels). The object must project to at least `MIN_PROJECTED` of the frame and
/// the crop may cover at most `CROP_MAX`, so every goal shows one recognizable object.
const CROP_MARGIN: f32 = 0.15;
const MIN_PROJECTED: f32 = 0.08;
const CROP_MAX: f32 = 0.8;
/// Aspect ratio used by the visibility frustum and goal crops. Catalogue construction
/// takes no image size, so this is fixed here; renders must use the same aspect.
const CATALOG_ASPECT: f32 = 4.0 / 3.0;

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

/// Target-eligible object indices of one split, grouped by `TARGET_CLASSES` order.
pub fn target_pools(scene: &CityScene, district: &District) -> Result<[Vec<u32>; TARGET_CLASSES.len()]> {
    let mut classes = [0u32; TARGET_CLASSES.len()];
    for (slot, name) in TARGET_CLASSES.iter().enumerate() {
        classes[slot] = scene.classes.iter().position(|c| c == name)
            .ok_or_else(|| anyhow::anyhow!("layout lacks target class {name}"))? as u32;
    }
    let isolated: Vec<bool> = TARGET_CLASSES.iter().map(|name| ISOLATED_CLASSES.contains(name)).collect();
    let mut pools: [Vec<u32>; TARGET_CLASSES.len()] = Default::default();
    for (index, object) in scene.objects.iter().enumerate() {
        let Some(slot) = classes.iter().position(|&c| c == object.class) else { continue };
        let (c, h) = (object.center, object.half);
        let corners = [[c.x - h.x, c.z - h.z], [c.x + h.x, c.z - h.z], [c.x - h.x, c.z + h.z], [c.x + h.x, c.z + h.z]];
        if 2.0 * h.max_element() < MIN_TARGET_EXTENT || !corners.iter().all(|&p| district.contains(p, DISTRICT_MARGIN)) {
            continue;
        }
        let crowded = isolated[slot] && scene.objects.iter().enumerate().any(|(other, neighbour)| {
            other != index && neighbour.class == object.class && horizontal_gap(object, neighbour) < ISOLATION_GAP
        });
        if !crowded {
            pools[slot].push(index as u32);
        }
    }
    ensure!(pools.iter().any(|pool| !pool.is_empty()), "district holds no target objects");
    Ok(pools)
}

/// Horizontal (x, z) gap between two objects' boxes; 0 when they overlap.
fn horizontal_gap(a: &CityObject, b: &CityObject) -> f32 {
    let gap = |ca: f32, ha: f32, cb: f32, hb: f32| ((ca - cb).abs() - ha - hb).max(0.0);
    let (x, z) = (gap(a.center.x, a.half.x, b.center.x, b.half.x), gap(a.center.z, a.half.z, b.center.z, b.half.z));
    (x * x + z * z).sqrt()
}

/// Candidate episodes for one split. `episodes` is sorted by bucket; `bucket_ranges` holds
/// `[first, count]` per bucket; `rules` records the generation rules and counts.
pub struct Catalog {
    pub episodes: Vec<EpisodeSpec>,
    pub bucket_ranges: [[u32; 2]; BUCKETS],
    pub rules: serde_json::Value,
}

struct Candidate {
    bucket: usize,
    episode: EpisodeSpec,
}

struct Ctx<'a> {
    scene: &'a CityScene,
    pools: [Vec<u32>; TARGET_CLASSES.len()],
    district: District,
    seed: u64,
    style: u32,
    start_distance: [f32; 2],
}

/// A roll-free or body camera: eye and orthonormal forward/up/right.
#[derive(Clone, Copy)]
struct View {
    eye: Vec3,
    forward: Vec3,
    up: Vec3,
    right: Vec3,
}

impl View {
    /// Roll-free camera looking along `forward` (world up stays up in the image).
    fn level(eye: Vec3, forward: Vec3) -> Self {
        let forward = forward.normalize();
        let right = forward.cross(Vec3::Y).normalize();
        Self { eye, forward, up: right.cross(forward), right }
    }
}

/// Builds the seeded catalogue of `size` candidates for one split. Errors if any bucket
/// holds fewer than 512 episodes or a candidate cannot be placed.
pub fn build_catalog(scene: &CityScene, style: u32, size: usize, seed: u64) -> Result<Catalog> {
    build_catalog_with_start_distance(scene, style, size, seed, START_BOX_DISTANCE)
}

/// Same seeded task with a configurable start-distance cohort; reference views are unchanged.
pub fn build_catalog_with_start_distance(scene: &CityScene, style: u32, size: usize, seed: u64,
    start_distance: [f32; 2]) -> Result<Catalog> {
    ensure!(start_distance[0].is_finite() && start_distance[1].is_finite()
        && start_distance[0] > drone::STANDOFF_MAX && start_distance[1] > start_distance[0],
        "start distance must be finite, increasing, and outside the standoff band");
    ensure!(style < 3, "style must be 0, 1 or 2, got {style}");
    ensure!(size >= MIN_CANDIDATES, "catalog needs at least {MIN_CANDIDATES} candidates, got {size}");
    let district = district(style, scene.half_extent);
    let pools = target_pools(scene, &district)?;
    let ctx = Ctx { scene, pools, district, seed, style, start_distance };
    let candidates = generate(&ctx, size)?;
    let (episodes, bucket_ranges, counts) = finalize(candidates, MIN_BUCKET)?;
    let rules = rules(scene, &ctx, size, counts, &episodes);
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
    // Classes are drawn uniformly so the thousands of forest trees do not dominate. Objects
    // without a clear start or goal view (e.g. trees deep in the forest) are re-drawn.
    let classes: Vec<&Vec<u32>> = ctx.pools.iter().filter(|pool| !pool.is_empty()).collect();
    for _ in 0..MAX_OBJECTS {
        let pool = classes[rng.below(classes.len())];
        let object = pool[rng.below(pool.len())];
        for _ in 0..MAX_ATTEMPTS {
            if let Some(candidate) = attempt(ctx, object, &mut rng) {
                return Ok(candidate);
            }
        }
    }
    bail!("candidate {index} found no placeable object in {MAX_OBJECTS} draws of {MAX_ATTEMPTS} attempts")
}

/// Probe length (metres) of the open-air test for starts.
const ENCLOSURE_PROBE_M: f32 = 25.0;

/// A hollow building's interior passes the clearance sweep, but no flight leaves it (the solvability
/// verifier found 17% of test starts inside suburban houses). A start is enclosed when the rays
/// straight up and in eight horizontal directions all hit geometry within ENCLOSURE_PROBE_M.
fn enclosed(scene: &CityScene, start: Vec3) -> bool {
    let blocked = |direction: Vec3| {
        scene.sweep_hits(start.to_array(), (start + direction * ENCLOSURE_PROBE_M).to_array(), 0.0, LANDABLE)
    };
    blocked(Vec3::Y) && (0..8).all(|k| {
        let angle = k as f32 * core::f32::consts::FRAC_PI_4;
        blocked(Vec3::new(angle.cos(), 0.0, angle.sin()))
    })
}

fn attempt(ctx: &Ctx, object: u32, rng: &mut Rng) -> Option<Candidate> {
    let scene = ctx.scene;
    let plane = scene.landable.height;
    let target = &scene.objects[object as usize];
    let (center, half) = (target.center, target.half);
    let radius = (half.x * half.x + half.z * half.z).sqrt();

    // Start: landed on the plane or flying, at a box distance inside the start range.
    let landed = rng.unit() < 0.5;
    let start_y = if landed { plane + drone::RADIUS } else { plane + rng.range(FLYING_HEIGHT[0], FLYING_HEIGHT[1]) };
    let bearing = rng.range(0.0, core::f32::consts::TAU);
    let reach = rng.range(radius + ctx.start_distance[0], radius + ctx.start_distance[1]);
    let start = Vec3::new(center.x + reach * bearing.cos(), start_y, center.z + reach * bearing.sin());
    let distance = box_distance(start.to_array(), center.to_array(), half.to_array());
    if !(ctx.start_distance[0]..=ctx.start_distance[1]).contains(&distance)
        || !ctx.district.contains([start.x, start.z], DISTRICT_MARGIN) {
        return None;
    }
    let clearance = if landed { drone::RADIUS } else { FLYING_CLEARANCE };
    if scene.sweep_hits(start.to_array(), start.to_array(), clearance, LANDABLE) || enclosed(scene, start) {
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

    let (eye_offset, forward, up) = drone::camera_basis(rotation);
    let forward = Vec3::from_array(forward);
    let up = Vec3::from_array(up);
    let start_view = View { eye: start + Vec3::from_array(eye_offset), forward, up, right: forward.cross(up) };
    let fraction = visible_fraction(scene, &start_view, center, half);
    let bucket = if fraction >= BUCKET_EDGES[0] {
        0
    } else if fraction >= BUCKET_EDGES[1] {
        1
    } else {
        2
    };

    // Goal reference: the start view (level copy) or an independent viewpoint of the object,
    // whichever gives a valid crop (the start view of a close building may not).
    let start_reference = (bucket == 0 && rng.unit() < START_REFERENCE_PROBABILITY).then(|| {
        let level = View::level(start_view.eye, start_view.forward);
        (level, visible_fraction(scene, &level, center, half))
    }).filter(|(view, visible)| *visible >= REFERENCE_MIN_VISIBLE && crop_rect(view, center, half).is_some());
    let (reference, reference_fraction) = match start_reference {
        Some(reference) => reference,
        None => reference_view(ctx, center, half, radius, rng)?,
    };
    let crop = crop_rect(&reference, center, half)?;

    Some(Candidate {
        bucket,
        episode: EpisodeSpec {
            start_position: start.extend(0.0).to_array(),
            start_rotation: rotation,
            start_velocity: velocity.extend(0.0).to_array(),
            start_rates: rates.extend(0.0).to_array(),
            target: center.extend(0.0).to_array(),
            target_half: half.extend(0.0).to_array(),
            reference_eye: reference.eye.extend(0.0).to_array(),
            reference_forward: reference.forward.extend(0.0).to_array(),
            crop,
            meta: [object, bucket as u32, landed as u32, 0],
            stats: [fraction, distance, target.class as f32, reference_fraction],
        },
    })
}

/// A level camera near the object that sees at least half of its box samples and frames it
/// in a valid goal crop.
fn reference_view(ctx: &Ctx, center: Vec3, half: Vec3, radius: f32, rng: &mut Rng) -> Option<(View, f32)> {
    let scene = ctx.scene;
    let plane = scene.landable.height;
    for _ in 0..REFERENCE_ATTEMPTS {
        let bearing = rng.range(0.0, core::f32::consts::TAU);
        let reach = rng.range(radius + REFERENCE_BOX_DISTANCE[0], radius + REFERENCE_BOX_DISTANCE[1]);
        let height = plane + rng.range(REFERENCE_HEIGHT[0], REFERENCE_HEIGHT[1]);
        let eye = Vec3::new(center.x + reach * bearing.cos(), height, center.z + reach * bearing.sin());
        let distance = box_distance(eye.to_array(), center.to_array(), half.to_array());
        if !(REFERENCE_BOX_DISTANCE[0]..=REFERENCE_BOX_DISTANCE[1]).contains(&distance)
            || !ctx.district.contains([eye.x, eye.z], DISTRICT_MARGIN)
            || scene.sweep_hits(eye.to_array(), eye.to_array(), REFERENCE_CLEARANCE, LANDABLE) {
            continue;
        }
        let toward = (center - eye).normalize();
        let yaw = rng.range(-REFERENCE_JITTER_RAD, REFERENCE_JITTER_RAD);
        let pitch = (toward.y.asin() + rng.range(-REFERENCE_JITTER_RAD, REFERENCE_JITTER_RAD)).clamp(-1.3, 1.3);
        let heading = toward.z.atan2(toward.x) + yaw;
        let forward = Vec3::new(pitch.cos() * heading.cos(), pitch.sin(), pitch.cos() * heading.sin());
        let view = View::level(eye, forward);
        let fraction = visible_fraction(scene, &view, center, half);
        if fraction >= REFERENCE_MIN_VISIBLE && crop_rect(&view, center, half).is_some() {
            return Some((view, fraction));
        }
    }
    None
}

/// Fraction of box-surface samples inside the frustum whose sight line reaches the box
/// unobstructed by non-landable geometry. The line stops 2 cm before the box, so the
/// object's own triangles never count as occluders.
fn visible_fraction(scene: &CityScene, view: &View, center: Vec3, half: Vec3) -> f32 {
    let half_tan = (drone::CAMERA_VFOV_RAD * 0.5).tan();
    let visible = box_samples(half).into_iter().filter(|&s| {
        let point = center + s;
        let d = point - view.eye;
        let depth = d.dot(view.forward);
        if !(drone::CAMERA_NEAR..=drone::CAMERA_FAR).contains(&depth) {
            return false;
        }
        let half_h = depth * half_tan;
        if d.dot(view.up).abs() > half_h || d.dot(view.right).abs() > half_h * CATALOG_ASPECT {
            return false;
        }
        let entry = segment_box_entry(view.eye.to_array(), point.to_array(), center.to_array(), half.to_array());
        let stop = (entry.max(0.0) - 0.02 / d.length()).max(0.0);
        stop == 0.0 || !scene.sweep_hits(view.eye.to_array(), (view.eye + d * stop).to_array(), 0.0, LANDABLE)
    }).count();
    visible as f32 / BOX_SAMPLES as f32
}

/// Area-stratified sample points on the faces of a box with half extents `half`, relative
/// to its centre.
fn box_samples(half: Vec3) -> Vec<Vec3> {
    // Faces: +-x (area y*z), +-y (x*z), +-z (x*y).
    let areas = [half.y * half.z, half.y * half.z, half.x * half.z, half.x * half.z, half.x * half.y, half.x * half.y];
    let total: f32 = areas.iter().sum();
    (0..BOX_SAMPLES).map(|k| {
        let goal = (k as f32 + 0.5) / BOX_SAMPLES as f32 * total;
        let mut acc = 0.0;
        let mut face = areas.len() - 1;
        for (i, area) in areas.iter().enumerate() {
            acc += area;
            if acc >= goal {
                face = i;
                break;
            }
        }
        let u = 2.0 * radical_inverse(k as u64, 2) - 1.0;
        let v = 2.0 * radical_inverse(k as u64, 3) - 1.0;
        let sign = if face % 2 == 0 { 1.0 } else { -1.0 };
        match face / 2 {
            0 => Vec3::new(sign * half.x, u * half.y, v * half.z),
            1 => Vec3::new(u * half.x, sign * half.y, v * half.z),
            _ => Vec3::new(u * half.x, v * half.y, sign * half.z),
        }
    }).collect()
}

/// Normalized goal crop `[x0, y0, x1, y1]` (top-left origin) around the box's projection,
/// grown by `CROP_MARGIN`, squared, and shifted inside the frame. None when a box corner lies
/// behind the near plane, the object projects smaller than `MIN_PROJECTED`, or the crop would
/// exceed `CROP_MAX` of the frame.
fn crop_rect(view: &View, center: Vec3, half: Vec3) -> Option<[f32; 4]> {
    let half_tan = (drone::CAMERA_VFOV_RAD * 0.5).tan();
    let (mut u0, mut v0, mut u1, mut v1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for corner in 0..8 {
        let sign = |bit: usize| if corner & bit == 0 { -1.0 } else { 1.0 };
        let point = center + Vec3::new(sign(1) * half.x, sign(2) * half.y, sign(4) * half.z);
        let d = point - view.eye;
        let depth = d.dot(view.forward);
        if depth <= drone::CAMERA_NEAR {
            return None;
        }
        let x = d.dot(view.right) / (depth * half_tan * CATALOG_ASPECT);
        let y = d.dot(view.up) / (depth * half_tan);
        let (u, v) = (0.5 * (x + 1.0), 0.5 * (1.0 - y));
        u0 = u0.min(u);
        u1 = u1.max(u);
        v0 = v0.min(v);
        v1 = v1.max(v);
    }
    let (u0, u1, v0, v1) = (u0.clamp(0.0, 1.0), u1.clamp(0.0, 1.0), v0.clamp(0.0, 1.0), v1.clamp(0.0, 1.0));
    let extent = (u1 - u0).max(v1 - v0);
    let side = extent * (1.0 + 2.0 * CROP_MARGIN);
    if extent < MIN_PROJECTED || side > CROP_MAX {
        return None;
    }
    let x0 = (0.5 * (u0 + u1) - 0.5 * side).clamp(0.0, 1.0 - side);
    let y0 = (0.5 * (v0 + v1) - 0.5 * side).clamp(0.0, 1.0 - side);
    Some([x0, y0, x0 + side, y0 + side])
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

fn rules(scene: &CityScene, ctx: &Ctx, size: usize, counts: [usize; BUCKETS], episodes: &[EpisodeSpec]) -> serde_json::Value {
    let pool_sizes: serde_json::Map<_, _> = TARGET_CLASSES.iter().zip(&ctx.pools)
        .map(|(name, pool)| (name.to_string(), json!(pool.len()))).collect();
    let mut class_episodes = vec![0usize; scene.classes.len()];
    for episode in episodes {
        class_episodes[episode.stats[2] as usize] += 1;
    }
    let class_episodes: serde_json::Map<_, _> = scene.classes.iter().zip(class_episodes)
        .filter(|(_, count)| *count > 0).map(|(name, count)| (name.clone(), json!(count))).collect();
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
            "open_air": format!("rejected when rays up and in 8 horizontal directions all hit geometry within {ENCLOSURE_PROBE_M} m (inside a hollow building)"),
            "tilt_max_rad": START_TILT_RAD,
            "speed_max_mps": START_SPEED,
            "rate_max_rps": START_RATE,
            "box_distance_m": ctx.start_distance,
            "flying_thrust": "hover",
        },
        "target": {
            "classes": TARGET_CLASSES,
            "class_choice": "uniform over target classes present in the district, then uniform object",
            "eligible_objects": pool_sizes,
            "episodes_by_class": class_episodes,
            "min_extent_m": MIN_TARGET_EXTENT,
            "placement": "object AABB wholly inside the district minus the margin",
            "isolation": { "classes": ISOLATED_CLASSES, "min_same_class_gap_m": ISOLATION_GAP },
            "success": {
                "standoff_max_m": drone::STANDOFF_MAX,
                "hold_speed_max_mps": drone::HOLD_SPEED,
                "hold_steps": drone::HOLD_STEPS,
                "facing": "box anchor (box point nearest the camera ray) inside the frustum and unoccluded",
            },
        },
        "goal_reference": {
            "start_view_probability_if_visible": START_REFERENCE_PROBABILITY,
            "box_distance_m": REFERENCE_BOX_DISTANCE,
            "height_above_landable_m": REFERENCE_HEIGHT,
            "jitter_rad": REFERENCE_JITTER_RAD,
            "clearance_m": REFERENCE_CLEARANCE,
            "min_visible_fraction": REFERENCE_MIN_VISIBLE,
            "camera": "level (roll-free), world up",
            "crop": { "margin_per_side": CROP_MARGIN, "min_projected_extent": MIN_PROJECTED, "max_side": CROP_MAX,
                "shape": "square in normalized units (4:3 in pixels), shifted inside the frame" },
        },
        "camera": {
            "aspect": CATALOG_ASPECT,
            "vfov_rad": drone::CAMERA_VFOV_RAD,
            "near_m": drone::CAMERA_NEAR,
            "far_m": drone::CAMERA_FAR,
        },
        "visibility": {
            "box_samples": BOX_SAMPLES,
            "occlusion": "radius-0 segment eye to 2 cm before the target box, over non-landable triangles",
            "visible_min_fraction": BUCKET_EDGES[0],
            "partial_min_fraction": BUCKET_EDGES[1],
        },
        "min_bucket": MIN_BUCKET,
        "bucket_counts": counts,
    })
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
    use bytemuck::Zeroable;

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
    fn box_samples_lie_on_faces() {
        let half = Vec3::new(1.0, 3.0, 0.5);
        let samples = box_samples(half);
        assert_eq!(samples.len(), BOX_SAMPLES);
        for p in samples {
            let on_face = (p.x.abs() - half.x).abs() < 1e-5 || (p.y.abs() - half.y).abs() < 1e-5
                || (p.z.abs() - half.z).abs() < 1e-5;
            let inside = p.abs().cmple(half + 1e-5).all();
            assert!(on_face && inside, "sample {p:?} is off the box surface");
        }
    }

    #[test]
    fn crop_frames_a_centred_box_as_a_square() {
        let view = View::level(Vec3::new(0.0, 1.0, 10.0), Vec3::new(0.0, 0.0, -1.0));
        let crop = crop_rect(&view, Vec3::new(0.0, 1.0, 0.0), Vec3::splat(1.0)).unwrap();
        let (w, h) = (crop[2] - crop[0], crop[3] - crop[1]);
        assert!((w - h).abs() < 1e-6 && w < 0.5);
        assert!(((crop[0] + crop[2]) * 0.5 - 0.5).abs() < 1e-5 && ((crop[1] + crop[3]) * 0.5 - 0.5).abs() < 1e-5);
        // No goal from a box enclosing the camera, one too far to recognize, or one filling the frame.
        assert_eq!(crop_rect(&view, Vec3::new(0.0, 1.0, 10.0), Vec3::splat(2.0)), None);
        assert_eq!(crop_rect(&view, Vec3::new(0.0, 1.0, -200.0), Vec3::splat(0.5)), None);
        assert_eq!(crop_rect(&view, Vec3::new(0.0, 1.0, 5.0), Vec3::splat(3.0)), None);
    }

    #[test]
    fn finalize_orders_buckets_and_stamps_indices() {
        let candidate = |bucket: usize| {
            let mut episode = EpisodeSpec::zeroed();
            episode.meta[1] = bucket as u32;
            Candidate { bucket, episode }
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
            for episode in &a.episodes {
                let crop = episode.crop;
                assert!(crop[0] >= 0.0 && crop[1] >= 0.0 && crop[2] <= 1.0 + 1e-6 && crop[3] <= 1.0 + 1e-6);
                assert!(episode.stats[1] >= START_BOX_DISTANCE[0] && episode.stats[3] >= REFERENCE_MIN_VISIBLE);
            }
            eprintln!("style {style}: {size} candidates, bucket counts {}, objects {}, by class {}, build {elapsed:?}",
                a.rules["bucket_counts"], a.rules["target"]["eligible_objects"], a.rules["target"]["episodes_by_class"]);
        }
    }
}
