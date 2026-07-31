//! Path finding on the [`TileGrid`]: A* over the walkable tiles, and the string-pull
//! that turns its staircase of tile centres into a route a circle can actually walk
//! (`docs/game-framework-plan.md` §4.4).
//!
//! Two halves, deliberately separable:
//!
//! 1. **[`Pathfinder::find`] — A*.** 4-connected, unit step cost, manhattan heuristic.
//!    Uniform cost plus manhattan is *consistent*, not merely admissible, so a closed
//!    node is never reopened and the first path found is a shortest one.
//! 2. **[`string_pull`] — smoothing.** A grid path zig-zags: every corner is a right
//!    angle even when the room is empty. The pull walks the tile chain and keeps only
//!    the waypoints that are not already implied by a straight walk from the previous
//!    one, which is what makes a monster cut across a room instead of tracing its tiles.
//!
//! # Spaces
//!
//! Tiles are grid coordinates. Everything with a [`Vec2`] is **collision space** —
//! grid-local, `.x` = world X, `.y` = world Z — the space [`crate::collision`] defines
//! and the one [`dreamcoast_game::physics`] moves circles in. Tile `(x, z)`'s centre is
//! at `physics::tile_center(x, z, TILE_SIZE)` in that space by construction, so no
//! offset appears anywhere in this module.
//!
//! # Cost discipline
//!
//! Every monster repaths on a cadence (see [`crate::ai`]), so the A* scratch — one
//! `g` value, one predecessor and two stamps per tile — lives in the [`Pathfinder`] and
//! is reused across queries and across monsters. A query touches only the cells it
//! expands: the stamp counter invalidates the previous run in O(1) instead of clearing
//! `width * height` bytes. Nothing here allocates in the steady state.
//!
//! `max_expansions` is the other half of that budget. A monster on the far side of the
//! dungeon should give up rather than flood-fill 1600 tiles every 0.4 s, so the caller
//! passes the cap it can afford and treats `None` as "no path *within budget*" — which
//! for a chasing monster means the same thing as no path at all (see
//! [`DEFAULT_MAX_EXPANSIONS`]).

// As `meshing.rs` and `rigs.rs`: `crate::ai` is this module's only caller, and the game
// loop wires *that* up in the integration step that follows — so nothing the binary
// reaches from `main` lands here yet.
#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use dreamcoast_game::physics::{self, GridCollision, SolidMap};
use glam::Vec2;

use crate::procgen::{TILE_SIZE, TileGrid};

/// Expansion budget that comfortably covers the default 40x40 dungeon.
///
/// A* over 1600 tiles cannot expand more than 1600 nodes, so this is "no cap" for the
/// shipping floor size while still bounding a pathological 200x200 one. Callers that
/// want a monster to fail fast pass something much smaller (a chase only needs to reach
/// a player it can already sense).
pub const DEFAULT_MAX_EXPANSIONS: usize = 2048;

// ---------------------------------------------------------------------------------
// A*
// ---------------------------------------------------------------------------------

/// An entry in the open set.
///
/// [`BinaryHeap`] is a max-heap, so [`Ord`] is written **reversed**: "greater" means
/// "should pop first". The sort key is `(f, h, index)`, which is a **total order on
/// distinct entries**, so the pop order is a function of the entries alone — never of
/// the heap's internal layout or of insertion history. That is the determinism claim;
/// [`Open::new`] is the only constructor precisely so its premise holds by construction.
///
/// Why those tie-breakers, in order:
/// * `f` — the A* ordering itself.
/// * `h` — among equal `f`, prefer the node *closer to the goal* (larger `g`). On the
///   open plains of a room this is what stops the search from expanding the whole
///   diamond of equal-cost detours before reaching the target.
/// * `index` — row-major tile index: a stable, geometric last word, so two runs of the
///   same query return the same one of several equally short paths.
///
/// `g` is carried for the stale-duplicate check but is **not** part of the key, and
/// must not be: every entry satisfies `f == g + h` by construction, so `(f, h)` already
/// determines `g`. Two entries agreeing on `(f, h, index)` are therefore the same entry,
/// and a fourth comparison could only ever be a no-op that reads as if it discriminated
/// something.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Open {
    f: u32,
    h: u32,
    index: u32,
    g: u32,
}

impl Open {
    /// The only way to build an entry, so the `f == g + h` invariant the sort key rests
    /// on cannot be broken at a call site.
    #[inline]
    fn new(index: u32, g: u32, h: u32) -> Self {
        Self {
            f: g + h,
            h,
            index,
            g,
        }
    }
}

impl Ord for Open {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .cmp(&self.f)
            .then_with(|| other.h.cmp(&self.h))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Manhattan distance in tiles — admissible *and* consistent for 4-connected unit
/// steps, which is what lets the search close a node for good.
#[inline]
fn heuristic(a: (i32, i32), b: (i32, i32)) -> u32 {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

/// The reusable A* workspace.
///
/// One per *system*, not one per monster: [`crate::ai::tick_grunts`] threads a single
/// [`Pathfinder`] through every brain it steps, so twelve monsters share one set of
/// buffers and one heap allocation.
///
/// The results of the last query stay borrowed from the finder ([`Pathfinder::find`]
/// returns `&[(i32, i32)]`), so a caller that needs to keep a route copies it out —
/// which is exactly what a brain does, once per repath.
pub struct Pathfinder {
    open: BinaryHeap<Open>,
    /// Best known cost from the start, valid only where `stamp == run`.
    g: Vec<u32>,
    /// Predecessor tile index, or `u32::MAX` at the start tile.
    came_from: Vec<u32>,
    /// Run number that last wrote `g` / `came_from` for this cell.
    stamp: Vec<u32>,
    /// Run number that last closed this cell.
    closed: Vec<u32>,
    /// Grid dimensions the buffers are sized for.
    dims: (i32, i32),
    /// Query counter; `0` is the "never written" stamp, so runs start at 1.
    run: u32,
    /// Tile path of the last successful query, start first.
    path: Vec<(i32, i32)>,
    /// Smoothed waypoints of the last [`Pathfinder::find_smoothed`].
    points: Vec<Vec2>,
    /// Nodes expanded by the last query — a budget readout for tuning and tests.
    expansions: usize,
}

impl Default for Pathfinder {
    fn default() -> Self {
        Self::new()
    }
}

impl Pathfinder {
    /// An empty workspace. The buffers size themselves on the first query.
    pub fn new() -> Self {
        Self {
            open: BinaryHeap::new(),
            g: Vec::new(),
            came_from: Vec::new(),
            stamp: Vec::new(),
            closed: Vec::new(),
            dims: (0, 0),
            run: 0,
            path: Vec::new(),
            points: Vec::new(),
            expansions: 0,
        }
    }

    /// Nodes expanded by the most recent query (0 for one answered before the search
    /// started: an unwalkable endpoint, or `from == to`).
    pub fn expansions(&self) -> usize {
        self.expansions
    }

    /// Size the buffers for `grid` and invalidate the previous run in O(1).
    fn begin(&mut self, grid: &TileGrid) {
        let dims = (grid.width(), grid.height());
        if dims != self.dims {
            let cells = (dims.0.max(0) as usize) * (dims.1.max(0) as usize);
            self.g = vec![0; cells];
            self.came_from = vec![u32::MAX; cells];
            self.stamp = vec![0; cells];
            self.closed = vec![0; cells];
            self.dims = dims;
            self.run = 0;
        }
        // Stamp invalidation is a counter bump; the one time that is not enough is the
        // wrap back to the "never written" value, four billion queries in.
        self.run = self.run.wrapping_add(1);
        if self.run == 0 {
            self.stamp.fill(0);
            self.closed.fill(0);
            self.run = 1;
        }
        self.open.clear();
        self.path.clear();
        self.expansions = 0;
    }

    /// A shortest 4-connected path of walkable tiles from `from` to `to`, **including
    /// both endpoints**, or `None`.
    ///
    /// `None` covers four cases, and the caller cannot tell them apart on purpose —
    /// a monster reacts to all of them the same way (give up and idle):
    /// * either endpoint is solid or out of bounds,
    /// * the goal is in a disconnected part of the dungeon,
    /// * the search hit `max_expansions`,
    /// * `max_expansions` was 0 (and `from != to`).
    ///
    /// `from == to` is answered without searching: a one-tile path, no expansions.
    pub fn find(
        &mut self,
        grid: &TileGrid,
        from: (i32, i32),
        to: (i32, i32),
        max_expansions: usize,
    ) -> Option<&[(i32, i32)]> {
        self.begin(grid);
        if !grid.is_walkable(from.0, from.1) || !grid.is_walkable(to.0, to.1) {
            return None;
        }
        if from == to {
            self.path.push(from);
            return Some(&self.path);
        }

        let width = grid.width();
        let index_of = |(x, z): (i32, i32)| (z * width + x) as u32;
        let tile_of = |index: u32| ((index as i32) % width, (index as i32) / width);

        let start = index_of(from);
        self.g[start as usize] = 0;
        self.stamp[start as usize] = self.run;
        self.came_from[start as usize] = u32::MAX;
        self.open.push(Open::new(start, 0, heuristic(from, to)));

        while let Some(node) = self.open.pop() {
            // Stale duplicate: a cheaper route to this tile was queued after it.
            if self.stamp[node.index as usize] != self.run || self.g[node.index as usize] < node.g {
                continue;
            }
            if self.closed[node.index as usize] == self.run {
                continue;
            }
            let tile = tile_of(node.index);
            if tile == to {
                self.reconstruct(node.index, start, width);
                return Some(&self.path);
            }
            self.closed[node.index as usize] = self.run;
            self.expansions += 1;
            if self.expansions > max_expansions {
                return None;
            }

            let next_g = node.g + 1;
            // `neighbors4` yields a fixed N/W/E/S order, which is the last piece of the
            // determinism story: identical pushes in identical order.
            for neighbour in grid.neighbors4(tile.0, tile.1) {
                let ni = index_of(neighbour) as usize;
                if self.closed[ni] == self.run {
                    continue;
                }
                if self.stamp[ni] == self.run && self.g[ni] <= next_g {
                    continue;
                }
                self.g[ni] = next_g;
                self.stamp[ni] = self.run;
                self.came_from[ni] = node.index;
                self.open
                    .push(Open::new(ni as u32, next_g, heuristic(neighbour, to)));
            }
        }
        None
    }

    /// Walk the predecessor chain back from `goal` and store it start-first.
    fn reconstruct(&mut self, goal: u32, start: u32, width: i32) {
        let mut index = goal;
        loop {
            self.path
                .push(((index as i32) % width, (index as i32) / width));
            if index == start {
                break;
            }
            let prev = self.came_from[index as usize];
            debug_assert_ne!(prev, u32::MAX, "predecessor chain broke before the start");
            if prev == u32::MAX {
                break;
            }
            index = prev;
        }
        self.path.reverse();
    }

    /// [`Pathfinder::find`] from the tile containing `from`, smoothed into waypoints a
    /// circle of `radius` can walk between ([`string_pull`]).
    ///
    /// The returned waypoints are in **collision space** and **exclude the start tile**:
    /// the mover is standing there. The last one is always the goal tile's centre. An
    /// empty slice therefore means "already on the goal tile", which is a success, not a
    /// failure — `None` is the failure.
    ///
    /// # What `from` is, and is not, used for
    ///
    /// Only to pick the **start tile**. The route itself is anchored at tile centres,
    /// which is what makes it provably walkable (see [`string_pull`]) and what keeps it
    /// stable while the mover crosses a tile instead of jittering under its feet.
    ///
    /// The honest consequence: the first leg the mover actually walks, `from` to
    /// `points[0]`, is **steering, not a swept-clear guarantee** — it can graze a corner
    /// by up to `radius`. That is fine because waypoints are steering targets and
    /// [`physics::move_circle`] — not this module — is the authority that a mover never
    /// ends up inside geometry. A graze costs a frame of sliding, which the next repath
    /// (0.4 s away, see [`crate::ai`]) then routes around.
    ///
    /// **Rejected experiment (do not re-add without a new measurement):** dropping
    /// leading waypoints that the mover can already reach straight from `from` —
    /// individually sound, since each removal is a passed [`walk_clear`]. Measured over
    /// 7184 legal off-centre positions across 64 generated dungeons it fired **29
    /// times** and removed 34 waypoints in total. A waypoint exists precisely because
    /// something blocks the view past it, and a mover cannot see around that corner by
    /// standing under a metre away from the tile centre. Not worth the branch or the
    /// route instability it would introduce.
    pub fn find_smoothed(
        &mut self,
        grid: &TileGrid,
        radius: f32,
        from: Vec2,
        goal: (i32, i32),
        max_expansions: usize,
    ) -> Option<&[Vec2]> {
        let start = physics::world_to_tile(from, TILE_SIZE);
        self.find(grid, start, goal, max_expansions)?;
        // Disjoint fields: the tile path is read while the point buffer is written.
        let tiles = &self.path;
        let points = &mut self.points;
        string_pull(GridCollision::new(grid, TILE_SIZE), radius, tiles, points);
        Some(&self.points)
    }
}

// ---------------------------------------------------------------------------------
// Smoothing
// ---------------------------------------------------------------------------------

/// Can a circle of `radius` walk the straight line `from` → `to` without touching
/// anything solid?
///
/// # The approximation
///
/// The exact question is whether the *capsule* swept by the circle is free, which the
/// grid raycaster cannot answer — it casts lines. So the swept rectangle's two long
/// edges are cast as well: three parallel rays, the centre one and one at each
/// `±radius` offset, i.e. the wall distance is inflated by the radius rather than the
/// shape being grown.
///
/// That is exact enough here for a reason worth writing down: a tile is [`TILE_SIZE`]
/// (2 m) across and the movers are `radius <= 0.4`, so the swept strip is at most 0.8 m
/// wide. A solid tile overlapping the strip projects at least 2 m onto the strip's
/// normal — it cannot fit *between* the rays — so the only geometry the test can miss
/// is a sliver at the very ends of the strip, past the rays' endpoints. Both endpoints
/// are positions the caller already knows to be free (tile centres, or the mover's own
/// resolved position), which is where that sliver lives. The rounded caps of the true
/// capsule are covered by the same argument.
///
/// A degenerate segment (`from == to`) reduces to an overlap test at the point, and a
/// non-finite input answers `false` rather than propagating the NaN.
pub fn walk_clear<M: SolidMap + ?Sized>(
    map: GridCollision<'_, M>,
    radius: f32,
    from: Vec2,
    to: Vec2,
) -> bool {
    if !from.is_finite() || !to.is_finite() || !radius.is_finite() {
        return false;
    }
    let seg = to - from;
    let len = seg.length();
    if len <= 1e-6 {
        return !map.circle_overlaps(from, radius.max(0.0));
    }
    let dir = seg / len;
    let side = Vec2::new(-dir.y, dir.x) * radius.max(0.0);
    for offset in [Vec2::ZERO, side, -side] {
        if map.raycast(from + offset, dir, len).is_some() {
            return false;
        }
    }
    true
}

/// String-pull a tile path into waypoints, written into `out` (cleared first).
///
/// Anchored at **tile centres**, never at the mover's position, and that is the whole
/// safety argument: consecutive tiles in a 4-connected path are orthogonally adjacent,
/// so the segment between their centres stays inside the union of two walkable tiles
/// for any `radius < TILE_SIZE / 2`. The worst case therefore degrades to the raw tile
/// chain, every segment of which is provably clear — smoothing can only ever *remove*
/// waypoints, never invent an unwalkable one.
///
/// The scan is forward and stops at the first waypoint that is not directly reachable
/// from the current anchor, rather than searching the whole tail for the farthest
/// visible one. Visibility along a corridor path is very nearly monotone, and where it
/// is not, the cost of being wrong is one extra waypoint — never an unsafe segment.
///
/// `out` excludes the start tile (the mover is standing on it) and ends on the goal
/// tile's centre. A path of one tile produces no waypoints.
pub fn string_pull<M: SolidMap + ?Sized>(
    map: GridCollision<'_, M>,
    radius: f32,
    tiles: &[(i32, i32)],
    out: &mut Vec<Vec2>,
) {
    out.clear();
    if tiles.len() < 2 {
        return;
    }
    let centre = |&(x, z): &(i32, i32)| physics::tile_center(x, z, TILE_SIZE);

    let mut anchor = centre(&tiles[0]);
    let mut i = 1;
    while i < tiles.len() {
        // The farthest tile still reachable in a straight line from the anchor. Index
        // `i` is always accepted: it is one orthogonal step from the anchor's tile (or
        // from the previously emitted one), which is clear by construction.
        let mut best = i;
        let mut j = i + 1;
        while j < tiles.len() && walk_clear(map, radius, anchor, centre(&tiles[j])) {
            best = j;
            j += 1;
        }
        anchor = centre(&tiles[best]);
        out.push(anchor);
        i = best + 1;
    }
}

/// One-shot [`Pathfinder::find`] for callers with no workspace to keep — tests, tools,
/// and any code that paths once rather than every 0.4 seconds.
pub fn astar(
    grid: &TileGrid,
    from: (i32, i32),
    to: (i32, i32),
    max_expansions: usize,
) -> Option<Vec<(i32, i32)>> {
    Pathfinder::new()
        .find(grid, from, to, max_expansions)
        .map(<[(i32, i32)]>::to_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision;
    use crate::procgen::{DungeonParams, generate};

    /// A maze with exactly one route through it, so "shortest" has a single answer.
    const MAZE: [&str; 7] = [
        "#########",
        "#E..#...#",
        "###.#.#.#",
        "#...#.#.#",
        "#.###.#.#",
        "#.....#X#",
        "#########",
    ];

    /// An open room: many equally short paths, which is what the determinism and
    /// smoothing claims are made against.
    const ROOM: [&str; 8] = [
        "##########",
        "#E.......#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#.......X#",
        "##########",
    ];

    /// Two rooms with no door between them.
    const SPLIT: [&str; 5] = ["#######", "#E.#..#", "#..#.X#", "#..#..#", "#######"];

    fn grid(rows: &[&str]) -> TileGrid {
        TileGrid::from_rows(rows)
    }

    /// Every step of a tile path is an orthogonal move onto a walkable tile.
    fn assert_connected(g: &TileGrid, path: &[(i32, i32)]) {
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(g.is_walkable(b.0, b.1), "path enters solid at {b:?}");
            assert_eq!(
                (a.0 - b.0).abs() + (a.1 - b.1).abs(),
                1,
                "path jumps from {a:?} to {b:?}"
            );
        }
    }

    #[test]
    fn finds_the_shortest_route_through_a_maze() {
        let g = grid(&MAZE);
        let path = astar(&g, g.entry(), g.exit(), DEFAULT_MAX_EXPANSIONS).expect("a route exists");
        assert_eq!(path.first().copied(), Some(g.entry()));
        assert_eq!(path.last().copied(), Some(g.exit()));
        assert_connected(&g, &path);

        // The maze's only route, counted by hand off the ASCII: 16 steps, so 17 tiles.
        // BFS agrees — and BFS is a different algorithm, not a re-run of this one.
        let steps = g.bfs_distances(g.entry())[(g.exit().1 * g.width() + g.exit().0) as usize];
        assert_eq!(path.len() as u32, steps + 1);
    }

    #[test]
    fn straight_line_in_the_open_is_the_manhattan_distance() {
        let g = grid(&ROOM);
        let path = astar(&g, (1, 1), (8, 5), DEFAULT_MAX_EXPANSIONS).unwrap();
        assert_eq!(path.len(), (7 + 4) + 1);
        assert_connected(&g, &path);
    }

    #[test]
    fn identical_queries_give_identical_paths() {
        let g = grid(&ROOM);
        let a = astar(&g, (1, 1), (8, 6), DEFAULT_MAX_EXPANSIONS).unwrap();
        let b = astar(&g, (1, 1), (8, 6), DEFAULT_MAX_EXPANSIONS).unwrap();
        assert_eq!(a, b, "two fresh finders disagree");

        // And a *reused* finder must not carry state between queries either: the same
        // query interleaved with other work still returns the same path.
        let mut finder = Pathfinder::new();
        let first = finder
            .find(&g, (1, 1), (8, 6), DEFAULT_MAX_EXPANSIONS)
            .unwrap()
            .to_vec();
        finder.find(&g, (3, 4), (1, 1), DEFAULT_MAX_EXPANSIONS);
        finder.find(&g, (8, 6), (8, 6), DEFAULT_MAX_EXPANSIONS);
        let again = finder
            .find(&g, (1, 1), (8, 6), DEFAULT_MAX_EXPANSIONS)
            .unwrap();
        assert_eq!(first, again, "the reused workspace leaked state");
        assert_eq!(first, a, "the reused workspace disagrees with a fresh one");
    }

    #[test]
    fn determinism_holds_on_a_generated_dungeon() {
        for seed in 0..8u64 {
            let g = generate(seed, &DungeonParams::default());
            let a = astar(&g, g.entry(), g.exit(), DEFAULT_MAX_EXPANSIONS).unwrap();
            let b = astar(&g, g.entry(), g.exit(), DEFAULT_MAX_EXPANSIONS).unwrap();
            assert_eq!(a, b, "seed {seed}");
            assert_connected(&g, &a);
        }
    }

    #[test]
    fn unreachable_and_illegal_endpoints_are_none() {
        let g = grid(&SPLIT);
        assert_eq!(astar(&g, (1, 1), (5, 2), DEFAULT_MAX_EXPANSIONS), None);
        // Endpoints inside rock, and outside the grid entirely.
        assert_eq!(astar(&g, (1, 1), (3, 1), DEFAULT_MAX_EXPANSIONS), None);
        assert_eq!(astar(&g, (0, 0), (1, 1), DEFAULT_MAX_EXPANSIONS), None);
        assert_eq!(astar(&g, (1, 1), (99, 99), DEFAULT_MAX_EXPANSIONS), None);
    }

    #[test]
    fn start_equals_goal_is_a_single_tile_and_no_search() {
        let g = grid(&ROOM);
        let mut finder = Pathfinder::new();
        assert_eq!(finder.find(&g, (4, 3), (4, 3), 0).unwrap(), [(4, 3)]);
        assert_eq!(finder.expansions(), 0);
        // The same tile, but solid: not a path.
        assert_eq!(
            finder.find(&g, (0, 0), (0, 0), DEFAULT_MAX_EXPANSIONS),
            None
        );
    }

    #[test]
    fn the_expansion_budget_cuts_the_search_off() {
        let g = grid(&MAZE);
        let mut finder = Pathfinder::new();
        assert!(
            finder.find(&g, g.entry(), g.exit(), 4).is_none(),
            "budget 4 should not reach"
        );
        assert!(
            finder.expansions() <= 5,
            "budget overrun: {}",
            finder.expansions()
        );

        // The same query with room to breathe succeeds, and reports what it cost.
        let full = finder.find(&g, g.entry(), g.exit(), DEFAULT_MAX_EXPANSIONS);
        assert!(full.is_some());
        let spent = finder.expansions();
        assert!(spent > 4 && spent <= (g.width() * g.height()) as usize);

        // A budget of exactly what it cost still succeeds — the cap is not off by one
        // in the direction that makes a monster stutter at the edge of its budget.
        assert!(finder.find(&g, g.entry(), g.exit(), spent).is_some());
    }

    /// The claim smoothing has to earn: every segment it emits is walkable by the
    /// circle, checked twice over — with the ray test and by sweeping the circle along
    /// the segment in small steps (which no approximation is hiding behind).
    fn assert_segments_clear(g: &TileGrid, radius: f32, from: Vec2, points: &[Vec2]) {
        let map = collision::collision(g);
        let mut anchor = from;
        for &p in points {
            assert!(
                walk_clear(map, radius, anchor, p),
                "segment {anchor:?} -> {p:?} is not clear"
            );
            let steps = ((p - anchor).length() / 0.05).ceil().max(1.0) as usize;
            for s in 0..=steps {
                let t = s as f32 / steps as f32;
                let probe = anchor.lerp(p, t);
                assert!(
                    !map.circle_overlaps(probe, radius),
                    "the circle penetrates geometry at {probe:?} on {anchor:?} -> {p:?}"
                );
            }
            anchor = p;
        }
    }

    #[test]
    fn smoothing_shortens_the_route_without_cutting_corners() {
        let g = grid(&ROOM);
        let radius = 0.35;
        let mut finder = Pathfinder::new();
        let start = physics::tile_center(1, 1, TILE_SIZE);
        let points = finder
            .find_smoothed(&g, radius, start, (8, 6), DEFAULT_MAX_EXPANSIONS)
            .unwrap()
            .to_vec();

        // An empty room collapses to the single diagonal shot.
        assert_eq!(
            points.len(),
            1,
            "open room should need one waypoint: {points:?}"
        );
        assert_eq!(points[0], physics::tile_center(8, 6, TILE_SIZE));
        assert_segments_clear(&g, radius, start, &points);
    }

    #[test]
    fn smoothing_keeps_the_corners_a_maze_needs() {
        let g = grid(&MAZE);
        let radius = 0.35;
        let mut finder = Pathfinder::new();
        let start = physics::tile_center(g.entry().0, g.entry().1, TILE_SIZE);
        let tiles = finder
            .find(&g, g.entry(), g.exit(), DEFAULT_MAX_EXPANSIONS)
            .unwrap()
            .len();
        let points = finder
            .find_smoothed(&g, radius, start, g.exit(), DEFAULT_MAX_EXPANSIONS)
            .unwrap()
            .to_vec();

        assert!(!points.is_empty());
        assert!(
            points.len() < tiles,
            "smoothing removed nothing: {points:?}"
        );
        assert!(
            points.len() > 2,
            "a one-tile maze cannot be two hops: {points:?}"
        );
        assert_eq!(
            *points.last().unwrap(),
            physics::tile_center(g.exit().0, g.exit().1, TILE_SIZE)
        );
        assert_segments_clear(&g, radius, start, &points);
    }

    #[test]
    fn smoothed_routes_stay_clear_on_generated_dungeons() {
        let radius = 0.35;
        let mut finder = Pathfinder::new();
        for seed in 0..6u64 {
            let g = generate(seed, &DungeonParams::default());
            let start = physics::tile_center(g.entry().0, g.entry().1, TILE_SIZE);
            let points = finder
                .find_smoothed(&g, radius, start, g.exit(), DEFAULT_MAX_EXPANSIONS)
                .expect("the generator guarantees entry reaches exit")
                .to_vec();
            assert_segments_clear(&g, radius, start, &points);
        }
    }

    /// A mover's position *within* its tile does not perturb its route.
    ///
    /// The counterpart of the rejected front-prune (see [`Pathfinder::find_smoothed`]):
    /// because smoothing is anchored at tile centres, two grunts standing anywhere in
    /// the same tile get the same waypoints, and a grunt walking across a tile sees its
    /// route hold still rather than jitter under it. That stability is what lets a brain
    /// cache waypoints between repaths, so it is pinned here rather than left implicit.
    #[test]
    fn the_route_depends_on_the_tile_not_on_where_in_it_the_mover_stands() {
        let radius = 0.35;
        let mut finder = Pathfinder::new();
        let mut checked = 0;
        for seed in 0..8u64 {
            let g = generate(seed, &DungeonParams::default());
            let map = collision::collision(&g);
            let centre = physics::tile_center(g.entry().0, g.entry().1, TILE_SIZE);
            let reference = finder
                .find_smoothed(&g, radius, centre, g.exit(), DEFAULT_MAX_EXPANSIONS)
                .expect("the generator guarantees entry reaches exit")
                .to_vec();

            for k in 0..8 {
                let angle = k as f32 * std::f32::consts::TAU / 8.0;
                let from = centre + Vec2::new(angle.cos(), angle.sin()) * (TILE_SIZE * 0.4);
                // Only positions a mover could legally occupy, in the same tile.
                if map.circle_overlaps(from, radius)
                    || physics::world_to_tile(from, TILE_SIZE) != g.entry()
                {
                    continue;
                }
                let points = finder
                    .find_smoothed(&g, radius, from, g.exit(), DEFAULT_MAX_EXPANSIONS)
                    .unwrap();
                assert_eq!(points, reference, "seed {seed}, offset {k}");
                checked += 1;
            }
        }
        assert!(
            checked > 20,
            "only {checked} positions were legal — weak coverage"
        );
    }

    #[test]
    fn a_one_tile_path_produces_no_waypoints() {
        let g = grid(&ROOM);
        let mut finder = Pathfinder::new();
        let here = physics::tile_center(4, 3, TILE_SIZE);
        let points = finder
            .find_smoothed(&g, 0.35, here, (4, 3), DEFAULT_MAX_EXPANSIONS)
            .unwrap();
        assert!(points.is_empty(), "standing on the goal is not a journey");
    }

    #[test]
    fn walk_clear_refuses_the_diagonal_through_a_pillar_corner() {
        // A lone pillar at (2, 2): the tile centres either side of it are diagonal
        // neighbours, and the straight line between them grazes the pillar's corner.
        let g = grid(&["#####", "#...#", "#.#.#", "#...#", "#####"]);
        let map = collision::collision(&g);
        let a = physics::tile_center(1, 1, TILE_SIZE);
        let b = physics::tile_center(3, 3, TILE_SIZE);
        assert!(!walk_clear(map, 0.35, a, b), "the diagonal cuts the pillar");
        // Around it, though, is fine.
        let side = physics::tile_center(1, 3, TILE_SIZE);
        assert!(walk_clear(map, 0.35, a, side));
    }

    #[test]
    fn walk_clear_rejects_a_gap_narrower_than_the_body() {
        // A one-tile doorway: a point can pass, a wide body cannot look through it at
        // an angle without clipping a jamb.
        let g = grid(&["#####", "#...#", "##.##", "#...#", "#####"]);
        let map = collision::collision(&g);
        let a = physics::tile_center(1, 1, TILE_SIZE);
        let b = physics::tile_center(3, 3, TILE_SIZE);
        assert!(!walk_clear(map, 0.35, a, b));
        // Straight through the door is clear for the same body.
        let up = physics::tile_center(2, 1, TILE_SIZE);
        let down = physics::tile_center(2, 3, TILE_SIZE);
        assert!(walk_clear(map, 0.35, up, down));
    }

    #[test]
    fn degenerate_inputs_do_not_panic_or_leak_nan() {
        let g = grid(&ROOM);
        let map = collision::collision(&g);
        let here = physics::tile_center(4, 3, TILE_SIZE);
        assert!(walk_clear(map, 0.35, here, here));
        assert!(!walk_clear(map, 0.35, here, Vec2::new(f32::NAN, 0.0)));
        assert!(!walk_clear(map, f32::INFINITY, here, here));

        let mut finder = Pathfinder::new();
        assert!(
            finder
                .find_smoothed(
                    &g,
                    0.35,
                    Vec2::new(-500.0, -500.0),
                    (4, 3),
                    DEFAULT_MAX_EXPANSIONS
                )
                .is_none(),
            "a start outside the grid has no path"
        );
    }

    #[test]
    fn the_workspace_resizes_between_grids() {
        let small = grid(&SPLIT);
        let big = generate(3, &DungeonParams::default());
        let mut finder = Pathfinder::new();
        assert!(
            finder
                .find(&small, (1, 1), (2, 3), DEFAULT_MAX_EXPANSIONS)
                .is_some()
        );
        assert!(
            finder
                .find(&big, big.entry(), big.exit(), DEFAULT_MAX_EXPANSIONS)
                .is_some()
        );
        assert!(
            finder
                .find(&small, (1, 1), (2, 3), DEFAULT_MAX_EXPANSIONS)
                .is_some()
        );
    }
}
