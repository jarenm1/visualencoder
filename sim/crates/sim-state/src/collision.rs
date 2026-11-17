//! Immutable triangle/BVH layouts shared by asset loading and Rust CUDA queries.
use bytemuck::{Pod, Zeroable};

/// World-space triangle. The fourth lane of `a` is the class index (0 sky, 1 ground,
/// 2 road, 3 building, 4 vegetation, 5 vehicle, 6 prop); fourth lanes of `b` and `c` are
/// reserved zero. Coordinates are Y-up metres.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Triangle {
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub c: [f32; 4],
}

/// Stackless, depth-first BVH node. Bounds include all descendant triangles.
/// `meta = [first_triangle, triangle_count, escape_index, 0]`.
/// Internal nodes have count zero and their first child is the next node.
/// A missed node jumps to escape_index; the final escape equals node_count.
/// Leaf triangles occupy one contiguous range in the reordered triangle array.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BvhNode {
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub meta: [u32; 4],
}

#[inline(always)]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]]
}

#[inline(always)]
fn clamp01(x: f32) -> f32 { if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x } }

fn point_segment_sq(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = sub(b, a);
    let ap = sub(p, a);
    let denom = dot(ab, ab);
    let t = if denom > 0.0 { clamp01(dot(ap, ab) / denom) } else { 0.0 };
    let d = [ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]];
    dot(d, d)
}

fn segment_segment_sq(p: [f32; 3], q: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let u = sub(q, p);
    let v = sub(b, a);
    let w = sub(p, a);
    let uu = dot(u, u);
    let vv = dot(v, v);
    if uu == 0.0 { return point_segment_sq(p, a, b); }
    if vv == 0.0 { return point_segment_sq(a, p, q); }
    let uv = dot(u, v);
    let uw = dot(u, w);
    let vw = dot(v, w);
    let denom = uu * vv - uv * uv;
    let mut s = if denom > 0.0 { clamp01((uv * vw - uw * vv) / denom) } else { 0.0 };
    let t = (uv * s + vw) / vv;
    let t = if t < 0.0 {
        s = clamp01(-uw / uu);
        0.0
    } else if t > 1.0 {
        s = clamp01((uv - uw) / uu);
        1.0
    } else { t };
    let d = [w[0] + s * u[0] - t * v[0], w[1] + s * u[1] - t * v[1],
        w[2] + s * u[2] - t * v[2]];
    dot(d, d)
}

fn inside_face(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3], n: [f32; 3]) -> bool {
    dot(cross(sub(b, a), sub(p, a)), n) >= 0.0
        && dot(cross(sub(c, b), sub(p, b)), n) >= 0.0
        && dot(cross(sub(a, c), sub(p, c)), n) >= 0.0
}

fn point_triangle_sq(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3],
    n: [f32; 3], nn: f32) -> f32 {
    // The oriented edge tests are unchanged by projection along the face normal.
    if nn > 0.0 && inside_face(p, a, b, c, n) {
        let d = dot(sub(p, a), n);
        return d * d / nn;
    }
    point_segment_sq(p, a, b).min(point_segment_sq(p, b, c)).min(point_segment_sq(p, c, a))
}

fn triangle_hits(triangle: Triangle, p: [f32; 3], q: [f32; 3], radius_sq: f32) -> bool {
    let a = [triangle.a[0], triangle.a[1], triangle.a[2]];
    let b = [triangle.b[0], triangle.b[1], triangle.b[2]];
    let c = [triangle.c[0], triangle.c[1], triangle.c[2]];
    let n = cross(sub(b, a), sub(c, a));
    let nn = dot(n, n);
    if nn > 0.0 {
        let dp = dot(sub(p, a), n);
        let dq = dot(sub(q, a), n);
        if (dp <= 0.0 && dq >= 0.0) || (dp >= 0.0 && dq <= 0.0) {
            let denom = dp - dq;
            if denom != 0.0 {
                let t = dp / denom;
                let hit = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]),
                    p[2] + t * (q[2] - p[2])];
                if inside_face(hit, a, b, c, n) { return true; }
            }
        }
    }
    point_triangle_sq(p, a, b, c, n, nn) <= radius_sq
        || point_triangle_sq(q, a, b, c, n, nn) <= radius_sq
        || segment_segment_sq(p, q, a, b) <= radius_sq
        || segment_segment_sq(p, q, b, c) <= radius_sq
        || segment_segment_sq(p, q, c, a) <= radius_sq
}

fn bounds_hit(node: BvhNode, from: [f32; 3], to: [f32; 3], radius: f32) -> bool {
    let mut lo: f32 = 0.0;
    let mut hi: f32 = 1.0;
    let mut axis = 0;
    while axis < 3 {
        let min = node.bounds_min[axis] - radius;
        let max = node.bounds_max[axis] + radius;
        let direction = to[axis] - from[axis];
        if direction == 0.0 {
            if from[axis] < min || from[axis] > max { return false; }
        } else {
            let t0 = (min - from[axis]) / direction;
            let t1 = (max - from[axis]) / direction;
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
            if lo > hi { return false; }
        }
        axis += 1;
    }
    true
}

/// Classes whose surfaces are landable (ground and road). Pass as the `ignore` mask to
/// query only non-landable geometry.
pub const LANDABLE: u32 = 0b110;

#[inline(always)]
fn class_bit(triangle: &Triangle) -> u32 {
    let class = triangle.a[3] as u32;
    if class < 32 { 1 << class } else { 0 }
}

/// Exact swept-sphere query using a stackless, radius-expanded BVH traversal.
/// Face crossing, endpoint/face and swept edge/corner distances are all tested.
/// Tangency counts as collision. Degenerate triangles reduce to their edges.
/// Triangles whose class bit (bit k = class k) is set in `ignore` are skipped; BVH
/// bounds still cover them, so traversal stays conservative.
///
/// # Safety
/// `nodes` must reference `node_count` valid preorder nodes with forward escape
/// indices. Leaf ranges must reference valid `triangles`. Inputs must be finite
/// and radius nonnegative. Both arrays remain immutable during the query.
pub unsafe fn sweep_hits(nodes: *const BvhNode, node_count: u32, triangles: *const Triangle,
    from: [f32; 3], to: [f32; 3], radius: f32, ignore: u32) -> bool {
    let mut index = 0;
    while index < node_count {
        let node = unsafe { *nodes.add(index as usize) };
        if !bounds_hit(node, from, to, radius) {
            index = node.meta[2];
            continue;
        }
        let count = node.meta[1];
        let mut triangle = node.meta[0];
        let end = triangle + count;
        while triangle < end {
            let tri = unsafe { *triangles.add(triangle as usize) };
            if class_bit(&tri) & ignore == 0 && triangle_hits(tri, from, to, radius * radius) {
                return true;
            }
            triangle += 1;
        }
        index = if count > 0 { node.meta[2] } else { index + 1 };
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(from: [f32; 3], to: [f32; 3], radius: f32) -> bool {
        let triangle = Triangle { a: [0.0, 0.0, 0.0, 0.0], b: [2.0, 0.0, 0.0, 0.0],
            c: [0.0, 2.0, 0.0, 0.0] };
        let node = BvhNode { bounds_min: [0.0; 4], bounds_max: [2.0, 2.0, 0.0, 0.0],
            meta: [0, 1, 1, 0] };
        unsafe { sweep_hits(&node, 1, &triangle, from, to, radius, 0) }
    }

    #[test]
    fn thin_wall_stops_high_speed_tunneling_both_directions() {
        assert!(hit([0.5, 0.5, -100.0], [0.5, 0.5, 100.0], 0.35));
        assert!(hit([0.5, 0.5, 100.0], [0.5, 0.5, -100.0], 0.35));
    }

    #[test]
    fn swept_edges_corners_and_parallel_face() {
        assert!(hit([1.0, -0.2, -2.0], [1.0, -0.2, 2.0], 0.35));
        assert!(hit([-0.2, -0.2, -2.0], [-0.2, -0.2, 2.0], 0.35));
        assert!(!hit([-0.3, -0.3, -2.0], [-0.3, -0.3, 2.0], 0.35));
        assert!(hit([-1.0, 0.5, 0.2], [3.0, 0.5, 0.2], 0.35));
    }

    #[test]
    fn stationary_overlap_tangency_and_miss() {
        assert!(hit([0.5, 0.5, 0.1], [0.5, 0.5, 0.1], 0.35));
        assert!(hit([0.5, 0.5, 0.0], [0.5, 0.5, 0.0], 0.0));
        assert!(hit([1.0, -0.25, 0.0], [1.0, -0.25, 0.0], 0.25));
        assert!(!hit([0.5, 0.5, 0.4], [0.5, 0.5, 0.4], 0.35));
        assert!(!hit([4.0, 4.0, -2.0], [4.0, 4.0, 2.0], 0.35));
    }

    #[test]
    fn ignore_mask_skips_landable_classes_only() {
        let mut tri = Triangle { a: [0.0, 0.0, 0.0, 1.0], b: [2.0, 0.0, 0.0, 0.0],
            c: [0.0, 2.0, 0.0, 0.0] };
        let node = BvhNode { bounds_min: [0.0; 4], bounds_max: [2.0, 2.0, 0.0, 0.0],
            meta: [0, 1, 1, 0] };
        let from = [0.5, 0.5, -100.0];
        let to = [0.5, 0.5, 100.0];
        let query = |t: &Triangle, ignore: u32| unsafe {
            sweep_hits(&node, 1, t, from, to, 0.35, ignore)
        };
        assert!(query(&tri, 0));
        assert!(!query(&tri, LANDABLE));
        tri.a[3] = 2.0;
        assert!(!query(&tri, LANDABLE));
        tri.a[3] = 3.0;
        assert!(query(&tri, LANDABLE));
    }

    #[test]
    fn traversal_skips_subtrees_and_reaches_later_leaf() {
        let triangle = Triangle { a: [0.0, 0.0, 0.0, 0.0], b: [2.0, 0.0, 0.0, 0.0],
            c: [0.0, 2.0, 0.0, 0.0] };
        let nodes = [
            BvhNode { bounds_min: [-10.0; 4], bounds_max: [10.0; 4], meta: [0, 0, 4, 0] },
            BvhNode { bounds_min: [-10.0; 4], bounds_max: [-9.0; 4], meta: [0, 0, 3, 0] },
            BvhNode { bounds_min: [-10.0; 4], bounds_max: [-9.0; 4], meta: [0, 1, 3, 0] },
            BvhNode { bounds_min: [0.0; 4], bounds_max: [2.0, 2.0, 0.0, 0.0], meta: [0, 1, 4, 0] },
        ];
        assert!(unsafe { sweep_hits(nodes.as_ptr(), 4, &triangle,
            [0.5, 0.5, -1.0], [0.5, 0.5, 1.0], 0.35, 0) });
    }
}
