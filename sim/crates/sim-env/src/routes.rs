//! Initialization-only shortest-path preprocessing over the actual city geometry.
use std::{cmp::{Ordering, Reverse}, collections::{BinaryHeap, BTreeSet}};
use anyhow::{Result, ensure};
use sim_scene::CityScene;
use crate::task::*;

pub(crate) struct Catalogs {
    pub routes: Vec<Route>,
    pub waypoints: Vec<Waypoint>,
    pub centers: [[f32; 2]; 3],
    pub lengths: [[f32; 2]; 3],
}

#[derive(Clone, Copy)]
struct QueueEntry { distance: f32, node: usize }
impl PartialEq for QueueEntry { fn eq(&self, other: &Self) -> bool { self.distance == other.distance && self.node == other.node } }
impl Eq for QueueEntry {}
impl PartialOrd for QueueEntry { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance).then(self.node.cmp(&other.node))
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

impl Catalogs {
    pub fn build(scene: &CityScene) -> Result<Self> {
        ensure!(scene.half_extent.is_finite() && scene.half_extent > 80.0,
            "city extent {} cannot support disjoint 64m navigation districts", scene.half_extent);
        let h = scene.half_extent * 0.4;
        let mut result = Self { routes: Vec::new(), waypoints: Vec::new(),
            centers: [[-h, -h], [h, -h], [h, h]], lengths: [[f32::INFINITY, 0.0]; 3] };
        for split in 0..3 { result.build_split(scene, split)?; }
        Ok(result)
    }

    fn build_split(&mut self, scene: &CityScene, split: usize) -> Result<()> {
        const SIDE: usize = 33;
        const COUNT: usize = SIDE * SIDE;
        let center = self.centers[split];
        let positions: Vec<_> = (0..COUNT).map(|i| [
            center[0] - DISTRICT_HALF_SIZE + (i % SIDE) as f32 * GRID_SPACING,
            ALTITUDE,
            center[1] - DISTRICT_HALF_SIZE + (i / SIDE) as f32 * GRID_SPACING,
        ]).collect();
        // A vertical clearance sweep also excludes samples buried inside closed
        // building meshes; a stationary surface-distance test alone cannot.
        let free: Vec<_> = positions.iter().map(|&p| {
            !scene.sweep_hits(p, p, PLANNING_RADIUS)
                && !scene.sweep_hits(p, [p[0], 256.0, p[2]], PLANNING_RADIUS)
        }).collect();
        let available: Vec<_> = (0..COUNT).filter(|&i| free[i]).collect();
        ensure!(available.len() >= 32,
            "city split {split} at {center:?} has only {} sky-clear navigation samples; need >=32", available.len());
        let mut edges = vec![Vec::<(usize, f32)>::new(); COUNT];
        for i in 0..COUNT {
            if !free[i] { continue; }
            let x = (i % SIDE) as isize;
            let z = (i / SIDE) as isize;
            for (dx, dz) in [(1, 0), (0, 1), (1, 1), (-1, 1)] {
                let nx = x + dx;
                let nz = z + dz;
                if nx < 0 || nz < 0 || nx >= SIDE as isize || nz >= SIDE as isize { continue; }
                let j = nz as usize * SIDE + nx as usize;
                if free[j] && !scene.sweep_hits(positions[i], positions[j], PLANNING_RADIUS) {
                    let length = distance(positions[i], positions[j]);
                    edges[i].push((j, length));
                    edges[j].push((i, length));
                }
            }
        }
        // The independently seeded permutation controls both start and target
        // order. Every reachable target is considered; no lucky-attempt cutoff.
        let mut starts = available.clone();
        starts.sort_unstable_by_key(|&i| mix(CATALOG_SEEDS[split] ^ i as u64));
        let mut targets = available;
        targets.sort_unstable_by_key(|&i| mix(CATALOG_SEEDS[split] ^ 0xa83d_4f71 ^ i as u64));
        let mut pairs = BTreeSet::new();
        let initial_count = self.routes.len();
        for start in starts {
            let mut distances = vec![f32::INFINITY; COUNT];
            let mut parents = vec![usize::MAX; COUNT];
            distances[start] = 0.0;
            let mut queue = BinaryHeap::new();
            queue.push(Reverse(QueueEntry { distance: 0.0, node: start }));
            while let Some(Reverse(entry)) = queue.pop() {
                if entry.distance > distances[entry.node] { continue; }
                if entry.distance > MAX_ROUTE_LENGTH + 8.0 { break; }
                for &(next, length) in &edges[entry.node] {
                    let candidate = entry.distance + length;
                    if candidate < distances[next] {
                        distances[next] = candidate;
                        parents[next] = entry.node;
                        queue.push(Reverse(QueueEntry { distance: candidate, node: next }));
                    }
                }
            }
            // Cap each start at four destinations to avoid a catalog dominated
            // by one spawn; the complete start permutation is still searched.
            let mut per_start = 0;
            for &goal in &targets {
                if !distances[goal].is_finite() || distances[goal] < MIN_ROUTE_LENGTH
                    || distances[goal] > MAX_ROUTE_LENGTH + 8.0 { continue; }
                let direct = distance(positions[start], positions[goal]);
                if direct < 8.0 || !scene.sweep_hits(positions[start], positions[goal], DRONE_RADIUS) { continue; }
                let pair = (start.min(goal), start.max(goal));
                if pairs.contains(&pair) { continue; }
                let mut reversed = vec![goal];
                let mut cursor = goal;
                while cursor != start {
                    cursor = parents[cursor];
                    ensure!(cursor != usize::MAX, "broken shortest-path predecessor in city split {split}");
                    reversed.push(cursor);
                }
                reversed.reverse();
                // Visibility shortcut the shortest eight-connected grid path,
                // retaining the larger planning clearance on every chord.
                let mut path = vec![positions[start]];
                let mut at = 0;
                while at + 1 < reversed.len() {
                    let mut next = reversed.len() - 1;
                    while next > at + 1 && scene.sweep_hits(positions[reversed[at]], positions[reversed[next]], PLANNING_RADIUS) {
                        next -= 1;
                    }
                    path.push(positions[reversed[next]]);
                    at = next;
                }
                let length: f32 = path.windows(2).map(|p| distance(p[0], p[1])).sum();
                if !(MIN_ROUTE_LENGTH..=MAX_ROUTE_LENGTH).contains(&length)
                    || length < direct + 2.0 || length < direct * 1.08 { continue; }
                let first = self.waypoints.len() as u32;
                let mut remaining = 0.0;
                let mut packed = Vec::with_capacity(path.len());
                for i in (0..path.len()).rev() {
                    if i + 1 < path.len() { remaining += distance(path[i], path[i + 1]); }
                    packed.push(Waypoint { position: path[i], remaining });
                }
                packed.reverse();
                self.waypoints.extend(packed);
                self.routes.push(Route { first, count: path.len() as u32, length, reserved: 0 });
                self.lengths[split][0] = self.lengths[split][0].min(length);
                self.lengths[split][1] = self.lengths[split][1].max(length);
                pairs.insert(pair);
                per_start += 1;
                if self.routes.len() - initial_count == ROUTES_PER_SPLIT { return Ok(()); }
                if per_start == 4 { break; }
            }
        }
        anyhow::bail!("actual city split {split} at {center:?} produced only {} of {ROUTES_PER_SPLIT} blocked-line, 12..38m collision-free routes (planning radius {PLANNING_RADIUS}m); no toy/easy fallback", self.routes.len() - initial_count)
    }
}
