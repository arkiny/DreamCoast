//! Seeded dungeon generation — rooms, corridors and the tile grid they live in
//! (`docs/game-framework-plan.md` §3.1).
//!
//! The [`TileGrid`] this module produces is the **single source of truth** for three
//! consumers, and its API is shaped for all three:
//!
//! * **collision** — [`TileGrid::is_solid`] / [`TileGrid::is_solid_at_world`] answer the
//!   circle-vs-tile slide and the DDA raycast the plan's §3.4 calls for;
//! * **pathfinding** — [`TileGrid::neighbors4`] and [`TileGrid::bfs_distances`] are the
//!   grid the monster A* will run on;
//! * **geometry** — [`crate::meshing`] reads the same tiles and the same world mapping,
//!   so a wall you can see is exactly a wall you cannot walk through.
//!
//! **Determinism** is a hard requirement: the cook cache keys chunk SDF bakes on
//! (generator version, seed), so the same seed must yield a byte-identical grid on every
//! machine. That rules out `HashMap` iteration order and any RNG whose algorithm can
//! change under a dependency bump — hence the self-contained xoshiro256\*\* below rather
//! than a new crate dependency.

// These modules are the M1 foundation; the game loop wires them up in the integration
// step that follows. Until then the binary has no caller for the public surface.
#![allow(dead_code)]

use std::collections::VecDeque;

use glam::Vec3;

/// Edge length of one tile in metres, on the XZ plane (Y is up).
///
/// The whole game agrees on this number: the mesher places geometry with it, the
/// character controller converts its world position with it, and the camera framing is
/// derived from it. 2 m gives a corridor a human-scale width against a ~0.5 m character
/// radius while keeping a 40x40 dungeon inside an 80 m square — small enough that the
/// engine's global distance field still resolves it.
pub const TILE_SIZE: f32 = 2.0;

/// Sentinel for "this tile belongs to no room" (a corridor, a door or solid rock).
pub const ROOM_NONE: u16 = u16::MAX;

// ---------------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------------

/// What occupies one grid cell.
///
/// Everything except [`Tile::Wall`] is walkable; the distinction between the walkable
/// variants is gameplay meaning (where the player enters, where the stairs down are),
/// not collision. Keeping the semantics *in* the grid rather than in a side table means
/// a level round-trip cannot desynchronise them.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Tile {
    /// Solid rock. Blocks movement and carries wall geometry on its walkable faces.
    Wall,
    /// Open floor, inside a room or in a corridor.
    Floor,
    /// A doorway: the one-tile gap where a corridor pierces a room's wall ring.
    /// Walkable, and rendered as open floor in v1 (the plan defers door meshes).
    Door,
    /// Where the player spawns. Exactly one per dungeon.
    Entry,
    /// The way to the next floor. Exactly one per dungeon, in the room that is
    /// *furthest from [`Tile::Entry`] by grid distance* — not by straight line.
    Exit,
}

impl Tile {
    /// Can a character stand here?
    pub const fn is_walkable(self) -> bool {
        !matches!(self, Tile::Wall)
    }

    /// Does this tile block movement (and carry wall geometry)?
    pub const fn is_solid(self) -> bool {
        matches!(self, Tile::Wall)
    }

    /// The debug/serialisation character for this tile (see [`TileGrid::from_rows`]).
    pub const fn to_char(self) -> char {
        match self {
            Tile::Wall => '#',
            Tile::Floor => '.',
            Tile::Door => '+',
            Tile::Entry => 'E',
            Tile::Exit => 'X',
        }
    }

    /// Inverse of [`Tile::to_char`]; unknown characters read as [`Tile::Wall`].
    pub const fn from_char(c: char) -> Tile {
        match c {
            '.' => Tile::Floor,
            '+' => Tile::Door,
            'E' => Tile::Entry,
            'X' => Tile::Exit,
            _ => Tile::Wall,
        }
    }
}

// ---------------------------------------------------------------------------------
// Rooms
// ---------------------------------------------------------------------------------

/// An axis-aligned rectangle of floor. `(x, z)` is the minimum corner in tiles and the
/// room covers `x .. x + w` by `z .. z + h`, all of it [`Tile::Floor`] at carve time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Room {
    /// Index into [`TileGrid::rooms`], and the value stored in the room-id plane.
    pub id: u16,
    pub x: i32,
    pub z: i32,
    pub w: i32,
    pub h: i32,
}

impl Room {
    /// The tile the generator treats as the room's anchor: corridors are routed to it
    /// and entry/exit are placed on it. Always inside the room.
    pub const fn center(&self) -> (i32, i32) {
        (self.x + self.w / 2, self.z + self.h / 2)
    }

    /// Is this tile inside the room rectangle?
    pub const fn contains(&self, x: i32, z: i32) -> bool {
        x >= self.x && x < self.x + self.w && z >= self.z && z < self.z + self.h
    }

    /// Do the two rooms come within `margin` tiles of each other? A margin of 1 is the
    /// minimum that guarantees a wall ring survives between them, which is what makes
    /// the door rule ([`carve_doors`]) well defined.
    pub const fn too_close(&self, other: &Room, margin: i32) -> bool {
        self.x - margin < other.x + other.w
            && other.x - margin < self.x + self.w
            && self.z - margin < other.z + other.h
            && other.z - margin < self.z + self.h
    }
}

// ---------------------------------------------------------------------------------
// RNG — self-contained, so a dependency bump can never move the dungeon
// ---------------------------------------------------------------------------------

/// xoshiro256\*\* seeded through splitmix64: ~10 lines, no dependency, and a fixed
/// algorithm. Both are public-domain reference constructions.
///
/// The alternative — pulling in `rand` + `rand_chacha` — buys nothing here (we need no
/// distributions and no cryptographic quality) and costs the one property that actually
/// matters: an algorithm that is frozen for the lifetime of the cook cache.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seed the state. Every `u64` is a valid seed, including 0 (splitmix64 has no
    /// all-zero fixed point, so the xoshiro state is never degenerate).
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    /// The raw generator step.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `0 .. n`, rejection-sampled so the low values are not favoured.
    ///
    /// # Panics
    /// If `n == 0`.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below needs a non-empty range");
        // The largest multiple of `n` that fits in a u64: drawing above it and taking
        // the remainder would over-represent the first `u64::MAX % n` values.
        let limit = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < limit {
                return x % n;
            }
        }
    }

    /// Uniform in `lo ..= hi`.
    ///
    /// # Panics
    /// If `hi < lo`.
    pub fn range(&mut self, lo: i32, hi: i32) -> i32 {
        assert!(hi >= lo, "Rng::range needs lo <= hi");
        let span = (i64::from(hi) - i64::from(lo) + 1) as u64;
        lo + self.below(span) as i32
    }

    /// A fair coin, taken from the high bit (xoshiro's low bits are the weak ones).
    pub fn flip(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }

    /// Uniform in `[0, 1)`.
    pub fn unit_f32(&mut self) -> f32 {
        // 24 bits is the f32 mantissa; more would just round.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Fisher-Yates, so callers get a deterministic permutation without reaching for a
    /// hash-ordered container.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------

/// Knobs for [`generate`]. [`DungeonParams::default`] is the M1 shipping shape: a 40x40
/// grid (80 m square) with 6-10 rooms.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DungeonParams {
    /// Grid size in tiles. The outermost ring is always left solid so the dungeon is
    /// closed — you can never see or walk off the edge of the world.
    pub width: i32,
    pub height: i32,
    /// Inclusive bounds on the room count actually placed.
    pub min_rooms: usize,
    pub max_rooms: usize,
    /// Inclusive bounds on a room's side length, in tiles.
    pub room_min: i32,
    pub room_max: i32,
    /// Minimum gap between two rooms, in tiles. Must be >= 1 for the wall ring (and
    /// therefore the door rule) to exist; [`DungeonParams::sanitized`] enforces that.
    pub margin: i32,
    /// How many placement draws to spend before settling for the rooms we have.
    pub room_attempts: u32,
    /// Extra corridors added on top of the spanning tree, to break the "every route is
    /// a dead end" feel of a pure tree. Purely cosmetic: connectivity is already total.
    pub loop_corridors: usize,
}

impl Default for DungeonParams {
    fn default() -> Self {
        DungeonParams {
            width: 40,
            height: 40,
            min_rooms: 6,
            max_rooms: 10,
            room_min: 5,
            room_max: 9,
            margin: 1,
            room_attempts: 400,
            loop_corridors: 2,
        }
    }
}

impl DungeonParams {
    /// Clamp the parameters into a range the generator can actually satisfy, so a
    /// caller (or a future data file) cannot produce a panic or an empty dungeon.
    pub fn sanitized(&self) -> DungeonParams {
        let mut p = *self;
        // 7 is the smallest grid that fits a 3x3 room inside a wall ring with slack.
        p.width = p.width.max(7);
        p.height = p.height.max(7);
        // A room needs a wall ring on both sides plus one tile of border: side <= min-4.
        let side_cap = (p.width.min(p.height) - 4).max(3);
        p.room_max = p.room_max.clamp(3, side_cap);
        p.room_min = p.room_min.clamp(3, p.room_max);
        p.max_rooms = p.max_rooms.max(1);
        p.min_rooms = p.min_rooms.clamp(1, p.max_rooms);
        p.margin = p.margin.max(1);
        p.room_attempts = p.room_attempts.max(16);
        p
    }
}

// ---------------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------------

/// The dungeon, as tiles. Row-major, `z` outer and `x` inner.
///
/// World mapping: the grid is centred on the world origin, one tile is [`TILE_SIZE`]
/// metres on XZ, and floors sit at `y = 0`. Centring keeps the dungeon inside the
/// engine's distance-field extent regardless of grid size and keeps the numbers small
/// enough that f32 tile-edge arithmetic stays exact.
#[derive(Clone, Debug)]
pub struct TileGrid {
    width: i32,
    height: i32,
    tiles: Vec<Tile>,
    /// Parallel plane: the room each tile belongs to, or [`ROOM_NONE`]. Corridors and
    /// doors are [`ROOM_NONE`] — "which room am I in" is a room-scoped query (spawn
    /// tables, encounter triggers), and corridors are deliberately not rooms.
    room_of: Vec<u16>,
    rooms: Vec<Room>,
    entry: (i32, i32),
    exit: (i32, i32),
    seed: u64,
}

impl TileGrid {
    /// A grid of solid rock, with no rooms. Mostly useful as a mesher/test fixture.
    pub fn solid(width: i32, height: i32) -> TileGrid {
        let width = width.max(1);
        let height = height.max(1);
        let n = (width * height) as usize;
        TileGrid {
            width,
            height,
            tiles: vec![Tile::Wall; n],
            room_of: vec![ROOM_NONE; n],
            rooms: Vec::new(),
            entry: (0, 0),
            exit: (0, 0),
            seed: 0,
        }
    }

    /// Build a grid from ASCII rows (see [`Tile::from_char`]) — the hand-authored
    /// fixture form used by the tests, and a readable way to pin a bug's repro.
    ///
    /// Rows are `z` increasing downward, characters are `x` increasing rightward. Short
    /// rows are padded with [`Tile::Wall`]. Entry/exit are taken from the `E`/`X` tiles
    /// if present. No rooms are recorded.
    pub fn from_rows(rows: &[&str]) -> TileGrid {
        let height = rows.len() as i32;
        let width = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0) as i32;
        let mut grid = TileGrid::solid(width.max(1), height.max(1));
        for (z, row) in rows.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let t = Tile::from_char(c);
                grid.tiles[z * grid.width as usize + x] = t;
                match t {
                    Tile::Entry => grid.entry = (x as i32, z as i32),
                    Tile::Exit => grid.exit = (x as i32, z as i32),
                    _ => {}
                }
            }
        }
        grid
    }

    /// Render the grid back to ASCII (rows joined by `\n`) — debug output, and a cheap
    /// stable key for the determinism tests.
    pub fn to_ascii(&self) -> String {
        let mut s = String::with_capacity(((self.width + 1) * self.height) as usize);
        for z in 0..self.height {
            if z > 0 {
                s.push('\n');
            }
            for x in 0..self.width {
                s.push(self.get(x, z).to_char());
            }
        }
        s
    }

    pub const fn width(&self) -> i32 {
        self.width
    }

    pub const fn height(&self) -> i32 {
        self.height
    }

    /// The seed this grid was generated from (0 for hand-authored grids). Part of the
    /// cook-cache key alongside the generator version.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub const fn in_bounds(&self, x: i32, z: i32) -> bool {
        x >= 0 && x < self.width && z >= 0 && z < self.height
    }

    /// The tile at `(x, z)`. **Out of bounds reads as [`Tile::Wall`]**, which is what
    /// makes every consumer's edge handling fall out for free: collision stops at the
    /// border, the mesher caps the dungeon with walls, flood fills cannot escape.
    pub fn get(&self, x: i32, z: i32) -> Tile {
        if self.in_bounds(x, z) {
            self.tiles[(z * self.width + x) as usize]
        } else {
            Tile::Wall
        }
    }

    /// Does this tile block movement? See [`TileGrid::get`] for the out-of-bounds rule.
    pub fn is_solid(&self, x: i32, z: i32) -> bool {
        self.get(x, z).is_solid()
    }

    /// Can a character stand on this tile?
    pub fn is_walkable(&self, x: i32, z: i32) -> bool {
        self.get(x, z).is_walkable()
    }

    /// The room owning this tile, if any. Corridors and doors belong to no room.
    pub fn room_at(&self, x: i32, z: i32) -> Option<&Room> {
        let id = self.room_id_at(x, z);
        (id != ROOM_NONE).then(|| &self.rooms[id as usize])
    }

    /// The raw room id plane, or [`ROOM_NONE`].
    pub fn room_id_at(&self, x: i32, z: i32) -> u16 {
        if self.in_bounds(x, z) {
            self.room_of[(z * self.width + x) as usize]
        } else {
            ROOM_NONE
        }
    }

    /// Every room, in placement order. `rooms()[i].id == i`.
    pub fn rooms(&self) -> &[Room] {
        &self.rooms
    }

    /// The player's spawn tile.
    pub const fn entry(&self) -> (i32, i32) {
        self.entry
    }

    /// The stairs-down tile.
    pub const fn exit(&self) -> (i32, i32) {
        self.exit
    }

    // -- world mapping ------------------------------------------------------------

    /// World X of the tile grid's `x = 0` edge. The grid is centred on the origin.
    pub fn world_min_x(&self) -> f32 {
        -(self.width as f32) * TILE_SIZE * 0.5
    }

    /// World Z of the tile grid's `z = 0` edge.
    pub fn world_min_z(&self) -> f32 {
        -(self.height as f32) * TILE_SIZE * 0.5
    }

    /// World X of the boundary *before* tile column `x` (so column `x` spans
    /// `tile_edge_x(x) ..= tile_edge_x(x + 1)`). The mesher places geometry on these.
    pub fn tile_edge_x(&self, x: i32) -> f32 {
        self.world_min_x() + x as f32 * TILE_SIZE
    }

    /// World Z of the boundary before tile row `z`.
    pub fn tile_edge_z(&self, z: i32) -> f32 {
        self.world_min_z() + z as f32 * TILE_SIZE
    }

    /// World position of a tile's centre, on the floor plane (`y = 0`). This is where
    /// spawns go and what pathfinding hands the steering code as a waypoint.
    pub fn tile_center(&self, x: i32, z: i32) -> Vec3 {
        Vec3::new(
            self.tile_edge_x(x) + TILE_SIZE * 0.5,
            0.0,
            self.tile_edge_z(z) + TILE_SIZE * 0.5,
        )
    }

    /// Which tile contains this world position (Y ignored). May be out of bounds — pair
    /// it with [`TileGrid::get`], which reads out-of-bounds as solid.
    pub fn world_to_tile(&self, p: Vec3) -> (i32, i32) {
        (
            ((p.x - self.world_min_x()) / TILE_SIZE).floor() as i32,
            ((p.z - self.world_min_z()) / TILE_SIZE).floor() as i32,
        )
    }

    /// The tile under a world position.
    pub fn tile_at_world(&self, p: Vec3) -> Tile {
        let (x, z) = self.world_to_tile(p);
        self.get(x, z)
    }

    /// Collision query in world space: is this point inside rock?
    pub fn is_solid_at_world(&self, p: Vec3) -> bool {
        self.tile_at_world(p).is_solid()
    }

    /// World position of the player spawn.
    pub fn entry_world(&self) -> Vec3 {
        let (x, z) = self.entry;
        self.tile_center(x, z)
    }

    /// World position of the exit.
    pub fn exit_world(&self) -> Vec3 {
        let (x, z) = self.exit;
        self.tile_center(x, z)
    }

    // -- graph queries ------------------------------------------------------------

    /// The four orthogonal neighbours that are walkable. Fixed N/S/E/W order, because
    /// A*'s tie-breaking (and therefore the path an enemy takes) must be reproducible.
    pub fn neighbors4(&self, x: i32, z: i32) -> impl Iterator<Item = (i32, i32)> + '_ {
        const DIRS: [(i32, i32); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];
        DIRS.iter()
            .map(move |(dx, dz)| (x + dx, z + dz))
            .filter(|&(nx, nz)| self.is_walkable(nx, nz))
    }

    /// Breadth-first step distance from `from` to every tile, in the grid's own
    /// row-major indexing. Unreachable (and solid) tiles hold `u32::MAX`.
    ///
    /// This is the distance metric the exit placement uses: "furthest room" means
    /// furthest *to walk to*, which for a dungeon with a long corridor spine is a very
    /// different room than the euclidean answer.
    pub fn bfs_distances(&self, from: (i32, i32)) -> Vec<u32> {
        let mut dist = vec![u32::MAX; self.tiles.len()];
        if !self.is_walkable(from.0, from.1) {
            return dist;
        }
        let mut queue = VecDeque::new();
        dist[(from.1 * self.width + from.0) as usize] = 0;
        queue.push_back(from);
        while let Some((x, z)) = queue.pop_front() {
            let d = dist[(z * self.width + x) as usize];
            for (nx, nz) in self.neighbors4(x, z).collect::<Vec<_>>() {
                let ni = (nz * self.width + nx) as usize;
                if dist[ni] == u32::MAX {
                    dist[ni] = d + 1;
                    queue.push_back((nx, nz));
                }
            }
        }
        dist
    }

    /// Number of walkable tiles.
    pub fn walkable_count(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_walkable()).count()
    }

    /// Is every walkable tile reachable from [`TileGrid::entry`]? The generator
    /// guarantees this by construction; the tests assert it, and gameplay code can
    /// assert it after any runtime edit (a collapsed corridor, a sealed door).
    pub fn all_walkable_reachable(&self) -> bool {
        let dist = self.bfs_distances(self.entry);
        self.tiles
            .iter()
            .zip(dist.iter())
            .all(|(t, d)| !t.is_walkable() || *d != u32::MAX)
    }

    // -- mutation (generator-internal, but useful for runtime edits) ---------------

    /// Overwrite a tile. Out-of-bounds writes are ignored.
    pub fn set(&mut self, x: i32, z: i32, t: Tile) {
        if self.in_bounds(x, z) {
            self.tiles[(z * self.width + x) as usize] = t;
        }
    }
}

// ---------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------

/// Bumped whenever the generator's output changes for a fixed seed. Belongs in the cook
/// cache key next to the seed, so a generator change invalidates baked chunk SDFs.
pub const GENERATOR_VERSION: u32 = 1;

/// Generate a dungeon from a seed.
///
/// The pipeline, in order (each step is deterministic given the RNG stream):
///
/// 1. **rooms** — rejection-sample axis-aligned rectangles until we have the target
///    count or the attempt budget runs out. Rejection sampling (rather than BSP or
///    grid partitioning) keeps room sizes and spacing irregular, which reads as
///    hand-placed rather than tiled.
/// 2. **corridors** — a minimum spanning tree over the room centres (Prim's, euclidean
///    edge weight) guarantees total connectivity with the fewest corridors, then
///    `loop_corridors` extra edges break the tree feel. Each edge is carved as an
///    L: one axis then the other, with the order coin-flipped.
/// 3. **doors** — a corridor tile that a room lies *through* (room on one side, open
///    floor on the opposite side) is the doorway. Defining it by the crossing rather
///    than by adjacency means a corridor running alongside a room wall does not turn
///    into a row of doors.
/// 4. **entry/exit** — entry on the first room's centre; exit on the centre of the room
///    whose centre is furthest from entry by BFS.
pub fn generate(seed: u64, params: &DungeonParams) -> TileGrid {
    let p = params.sanitized();
    let mut rng = Rng::new(seed);
    let mut grid = TileGrid::solid(p.width, p.height);
    grid.seed = seed;

    place_rooms(&mut grid, &mut rng, &p);
    carve_corridors(&mut grid, &mut rng, &p);
    carve_doors(&mut grid);
    place_entry_exit(&mut grid);

    grid
}

/// Rejection-sample rooms, then fall back to a deterministic scan if the draws were
/// unlucky enough (or the grid small enough) to leave us with fewer than two rooms.
fn place_rooms(grid: &mut TileGrid, rng: &mut Rng, p: &DungeonParams) {
    let target = rng.below((p.max_rooms - p.min_rooms + 1) as u64) as usize + p.min_rooms;

    for _ in 0..p.room_attempts {
        if grid.rooms.len() >= target {
            break;
        }
        let w = rng.range(p.room_min, p.room_max);
        let h = rng.range(p.room_min, p.room_max);
        // Leave the outermost ring solid: x in 1 ..= width - w - 1.
        if grid.width - w - 1 < 1 || grid.height - h - 1 < 1 {
            continue;
        }
        let x = rng.range(1, grid.width - w - 1);
        let z = rng.range(1, grid.height - h - 1);
        try_place(grid, Room { id: 0, x, z, w, h }, p.margin);
    }

    // Fallback: a dungeon needs at least an entry and an exit room. Walk the grid on a
    // fixed lattice and drop minimum-size rooms wherever they fit. Deterministic, so it
    // does not break the "same seed, same grid" contract.
    if grid.rooms.len() < 2 {
        let side = p.room_min;
        let step = side + p.margin;
        let mut z = 1;
        while z + side < grid.height && grid.rooms.len() < 2 {
            let mut x = 1;
            while x + side < grid.width && grid.rooms.len() < 2 {
                try_place(
                    grid,
                    Room {
                        id: 0,
                        x,
                        z,
                        w: side,
                        h: side,
                    },
                    p.margin,
                );
                x += step;
            }
            z += step;
        }
    }
}

/// Accept `room` if it clears every placed room by `margin`, carving its floor and
/// stamping its id. Returns whether it was accepted.
fn try_place(grid: &mut TileGrid, mut room: Room, margin: i32) -> bool {
    if grid.rooms.len() >= ROOM_NONE as usize {
        return false;
    }
    if grid.rooms.iter().any(|r| room.too_close(r, margin)) {
        return false;
    }
    room.id = grid.rooms.len() as u16;
    for z in room.z..room.z + room.h {
        for x in room.x..room.x + room.w {
            let i = (z * grid.width + x) as usize;
            grid.tiles[i] = Tile::Floor;
            grid.room_of[i] = room.id;
        }
    }
    grid.rooms.push(room);
    true
}

/// Connect every room: Prim's MST over the centres, plus `loop_corridors` extra edges.
fn carve_corridors(grid: &mut TileGrid, rng: &mut Rng, p: &DungeonParams) {
    let centers: Vec<(i32, i32)> = grid.rooms.iter().map(|r| r.center()).collect();
    if centers.len() < 2 {
        return;
    }

    // Prim's: grow a tree from room 0, each step taking the cheapest edge out of it.
    // O(n^2) over <= 10 rooms — the clarity is worth more than the asymptotics here.
    let mut in_tree = vec![false; centers.len()];
    in_tree[0] = true;
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(centers.len() - 1);
    for _ in 1..centers.len() {
        let mut best: Option<(i64, usize, usize)> = None;
        for (a, ca) in centers.iter().enumerate() {
            if !in_tree[a] {
                continue;
            }
            for (b, cb) in centers.iter().enumerate() {
                if in_tree[b] {
                    continue;
                }
                let dx = i64::from(ca.0 - cb.0);
                let dz = i64::from(ca.1 - cb.1);
                let cost = dx * dx + dz * dz;
                // Strictly-less keeps the first-found (lowest index) winner on ties, so
                // the tree does not depend on iteration incidentals.
                if best.is_none_or(|(c, _, _)| cost < c) {
                    best = Some((cost, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else { break };
        in_tree[b] = true;
        edges.push((a, b));
    }

    // Extra edges: random pairs that the tree did not already join directly.
    for _ in 0..p.loop_corridors {
        let a = rng.below(centers.len() as u64) as usize;
        let b = rng.below(centers.len() as u64) as usize;
        if a == b || edges.contains(&(a, b)) || edges.contains(&(b, a)) {
            continue;
        }
        edges.push((a, b));
    }

    for (a, b) in edges {
        let horizontal_first = rng.flip();
        carve_l(grid, centers[a], centers[b], horizontal_first);
    }
}

/// Carve an L-shaped corridor between two tiles.
fn carve_l(grid: &mut TileGrid, from: (i32, i32), to: (i32, i32), horizontal_first: bool) {
    if horizontal_first {
        carve_row(grid, from.1, from.0, to.0);
        carve_col(grid, to.0, from.1, to.1);
    } else {
        carve_col(grid, from.0, from.1, to.1);
        carve_row(grid, to.1, from.0, to.0);
    }
}

/// Carve `z`'s row between two x values (inclusive, either order).
fn carve_row(grid: &mut TileGrid, z: i32, x0: i32, x1: i32) {
    for x in x0.min(x1)..=x0.max(x1) {
        carve(grid, x, z);
    }
}

/// Carve `x`'s column between two z values (inclusive, either order).
fn carve_col(grid: &mut TileGrid, x: i32, z0: i32, z1: i32) {
    for z in z0.min(z1)..=z0.max(z1) {
        carve(grid, x, z);
    }
}

/// Turn rock into corridor floor. Room floor is left alone so the room-id plane and the
/// door rule stay meaningful.
fn carve(grid: &mut TileGrid, x: i32, z: i32) {
    if grid.in_bounds(x, z) {
        let i = (z * grid.width + x) as usize;
        if grid.tiles[i] == Tile::Wall {
            grid.tiles[i] = Tile::Floor;
        }
    }
}

/// Promote the corridor tiles that pierce a room's wall ring to [`Tile::Door`].
///
/// The test is a *crossing*, not adjacency: tile `t` is a door when for some axis, one
/// side of `t` is room floor and the opposite side is walkable. A corridor that merely
/// runs alongside a room wall fails it (the far side is rock), so it stays corridor.
fn carve_doors(grid: &mut TileGrid) {
    const AXES: [(i32, i32); 2] = [(1, 0), (0, 1)];
    let mut doors = Vec::new();
    for z in 0..grid.height {
        for x in 0..grid.width {
            // Only a corridor tile can become a door; room floor never does.
            if grid.get(x, z) != Tile::Floor || grid.room_id_at(x, z) != ROOM_NONE {
                continue;
            }
            let crosses = AXES.iter().any(|&(dx, dz)| {
                let a_room = grid.room_id_at(x + dx, z + dz) != ROOM_NONE;
                let b_room = grid.room_id_at(x - dx, z - dz) != ROOM_NONE;
                let a_open = grid.is_walkable(x + dx, z + dz);
                let b_open = grid.is_walkable(x - dx, z - dz);
                (a_room && b_open) || (b_room && a_open)
            });
            if crosses {
                doors.push((x, z));
            }
        }
    }
    for (x, z) in doors {
        grid.set(x, z, Tile::Door);
    }
}

/// Entry on room 0's centre; exit on the centre of the room furthest from it by BFS.
fn place_entry_exit(grid: &mut TileGrid) {
    let Some(first) = grid.rooms.first().copied() else {
        return;
    };
    let entry = first.center();
    grid.entry = entry;
    grid.set(entry.0, entry.1, Tile::Entry);

    let dist = grid.bfs_distances(entry);
    let mut best: Option<(u32, (i32, i32))> = None;
    for room in &grid.rooms {
        if room.id == first.id {
            continue;
        }
        let c = room.center();
        let d = dist[(c.1 * grid.width + c.0) as usize];
        if d == u32::MAX {
            continue;
        }
        // Strictly-greater keeps the lowest room id on ties.
        if best.is_none_or(|(bd, _)| d > bd) {
            best = Some((d, c));
        }
    }
    // With a single room (a tiny grid) entry and exit coincide; documented, not a panic.
    let exit = best.map_or(entry, |(_, c)| c);
    grid.exit = exit;
    if exit != entry {
        grid.set(exit.0, exit.1, Tile::Exit);
    }
}

// ---------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every seed we sweep in the property tests. Fixed list, so a failure is a repro.
    const SEEDS: [u64; 12] = [
        0,
        1,
        2,
        7,
        42,
        99,
        1234,
        65535,
        1 << 20,
        1 << 40,
        u64::MAX,
        0xDEAD_BEEF,
    ];

    // -- RNG ----------------------------------------------------------------------

    #[test]
    fn rng_is_deterministic_and_seed_sensitive() {
        let a: Vec<u64> = (0..8).map(|_| Rng::new(42).next_u64()).collect();
        assert!(a.iter().all(|v| *v == a[0]), "same seed, same first draw");

        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        let mut r3 = Rng::new(43);
        let s1: Vec<u64> = (0..64).map(|_| r1.next_u64()).collect();
        let s2: Vec<u64> = (0..64).map(|_| r2.next_u64()).collect();
        let s3: Vec<u64> = (0..64).map(|_| r3.next_u64()).collect();
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn rng_seed_zero_is_not_degenerate() {
        let mut r = Rng::new(0);
        let draws: Vec<u64> = (0..8).map(|_| r.next_u64()).collect();
        assert!(draws.iter().any(|d| *d != 0));
        assert!(draws.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn rng_range_stays_in_bounds_and_covers_it() {
        let mut r = Rng::new(7);
        let mut seen = [false; 5];
        for _ in 0..2000 {
            let v = r.range(-2, 2);
            assert!((-2..=2).contains(&v));
            seen[(v + 2) as usize] = true;
        }
        assert!(seen.iter().all(|s| *s), "every value in the range is drawn");
        assert_eq!(r.range(3, 3), 3, "degenerate range is allowed");
    }

    #[test]
    fn rng_shuffle_is_a_permutation() {
        let mut r = Rng::new(11);
        let mut items: Vec<u32> = (0..32).collect();
        r.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
        assert_ne!(
            items, sorted,
            "32 items should not shuffle back to identity"
        );
    }

    // -- determinism ---------------------------------------------------------------

    #[test]
    fn same_seed_same_grid() {
        let p = DungeonParams::default();
        for seed in SEEDS {
            let a = generate(seed, &p);
            let b = generate(seed, &p);
            assert_eq!(
                a.to_ascii(),
                b.to_ascii(),
                "seed {seed} is not reproducible"
            );
            assert_eq!(a.entry(), b.entry());
            assert_eq!(a.exit(), b.exit());
            assert_eq!(a.rooms(), b.rooms());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let p = DungeonParams::default();
        let mut layouts: Vec<String> = SEEDS.iter().map(|s| generate(*s, &p).to_ascii()).collect();
        let total = layouts.len();
        layouts.sort();
        layouts.dedup();
        assert_eq!(
            layouts.len(),
            total,
            "distinct seeds produced a duplicate layout"
        );
    }

    // -- structure -----------------------------------------------------------------

    #[test]
    fn room_count_is_in_range_and_rooms_do_not_overlap() {
        let p = DungeonParams::default();
        for seed in SEEDS {
            let g = generate(seed, &p);
            assert!(
                (p.min_rooms..=p.max_rooms).contains(&g.rooms().len()),
                "seed {seed}: {} rooms outside {}..={}",
                g.rooms().len(),
                p.min_rooms,
                p.max_rooms
            );
            for (i, a) in g.rooms().iter().enumerate() {
                assert_eq!(a.id as usize, i, "room ids must index the room list");
                assert!(a.w >= p.room_min && a.w <= p.room_max);
                assert!(a.h >= p.room_min && a.h <= p.room_max);
                for b in &g.rooms()[i + 1..] {
                    assert!(
                        !a.too_close(b, p.margin),
                        "seed {seed}: rooms {:?} and {:?} are closer than the margin",
                        a,
                        b
                    );
                }
            }
        }
    }

    #[test]
    fn rooms_stay_inside_the_solid_border() {
        let p = DungeonParams::default();
        for seed in SEEDS {
            let g = generate(seed, &p);
            for r in g.rooms() {
                assert!(r.x >= 1 && r.z >= 1);
                assert!(r.x + r.w < g.width());
                assert!(r.z + r.h < g.height());
            }
            // The whole outer ring is rock: the dungeon is closed.
            for x in 0..g.width() {
                assert!(g.is_solid(x, 0) && g.is_solid(x, g.height() - 1));
            }
            for z in 0..g.height() {
                assert!(g.is_solid(0, z) && g.is_solid(g.width() - 1, z));
            }
        }
    }

    #[test]
    fn room_id_plane_agrees_with_the_room_list() {
        let g = generate(42, &DungeonParams::default());
        for z in 0..g.height() {
            for x in 0..g.width() {
                match g.room_at(x, z) {
                    Some(r) => {
                        assert!(r.contains(x, z));
                        assert!(g.is_walkable(x, z));
                    }
                    None => assert!(g.rooms().iter().all(|r| !r.contains(x, z))),
                }
            }
        }
    }

    #[test]
    fn every_walkable_tile_is_reachable_from_entry() {
        let p = DungeonParams::default();
        for seed in SEEDS {
            let g = generate(seed, &p);
            assert!(
                g.all_walkable_reachable(),
                "seed {seed}: dungeon has an unreachable pocket\n{}",
                g.to_ascii()
            );
        }
    }

    #[test]
    fn entry_and_exit_are_placed_and_unique() {
        let p = DungeonParams::default();
        for seed in SEEDS {
            let g = generate(seed, &p);
            let entries = (0..g.height())
                .flat_map(|z| (0..g.width()).map(move |x| (x, z)))
                .filter(|&(x, z)| g.get(x, z) == Tile::Entry)
                .count();
            let exits = (0..g.height())
                .flat_map(|z| (0..g.width()).map(move |x| (x, z)))
                .filter(|&(x, z)| g.get(x, z) == Tile::Exit)
                .count();
            assert_eq!(entries, 1, "seed {seed}");
            assert_eq!(exits, 1, "seed {seed}");
            assert_eq!(g.get(g.entry().0, g.entry().1), Tile::Entry);
            assert_eq!(g.get(g.exit().0, g.exit().1), Tile::Exit);
            // Entry is room 0's centre; exit is in a different room.
            assert_eq!(g.entry(), g.rooms()[0].center());
            let exit_room = g
                .room_at(g.exit().0, g.exit().1)
                .expect("exit is in a room");
            assert_ne!(
                exit_room.id, 0,
                "seed {seed}: exit landed in the entry room"
            );
        }
    }

    #[test]
    fn exit_is_the_bfs_farthest_room_not_the_euclidean_one() {
        let p = DungeonParams::default();
        for seed in SEEDS {
            let g = generate(seed, &p);
            let dist = g.bfs_distances(g.entry());
            let exit_d = dist[(g.exit().1 * g.width() + g.exit().0) as usize];
            assert_ne!(exit_d, u32::MAX);
            for r in g.rooms().iter().skip(1) {
                let c = r.center();
                let d = dist[(c.1 * g.width() + c.0) as usize];
                assert!(
                    d <= exit_d,
                    "seed {seed}: room {} is farther than the exit",
                    r.id
                );
            }
        }
    }

    #[test]
    fn doors_are_crossings_never_room_tiles() {
        let p = DungeonParams::default();
        for seed in SEEDS {
            let g = generate(seed, &p);
            for z in 0..g.height() {
                for x in 0..g.width() {
                    if g.get(x, z) != Tile::Door {
                        continue;
                    }
                    assert_eq!(g.room_id_at(x, z), ROOM_NONE, "a door is never room floor");
                    let crossing = [(1, 0), (0, 1)].iter().any(|&(dx, dz)| {
                        let a_room = g.room_id_at(x + dx, z + dz) != ROOM_NONE;
                        let b_room = g.room_id_at(x - dx, z - dz) != ROOM_NONE;
                        (a_room && g.is_walkable(x - dx, z - dz))
                            || (b_room && g.is_walkable(x + dx, z + dz))
                    });
                    assert!(crossing, "seed {seed}: door at ({x},{z}) is not a crossing");
                }
            }
            // Every room must be enterable: at least one door on the whole map, and each
            // room touched by at least one walkable non-room neighbour.
            for r in g.rooms() {
                let ring_open = (r.x - 1..=r.x + r.w)
                    .flat_map(|x| (r.z - 1..=r.z + r.h).map(move |z| (x, z)))
                    .any(|(x, z)| !r.contains(x, z) && g.is_walkable(x, z));
                assert!(ring_open, "seed {seed}: room {} is sealed", r.id);
            }
        }
    }

    // -- world mapping -------------------------------------------------------------

    #[test]
    fn world_mapping_round_trips() {
        let g = generate(42, &DungeonParams::default());
        // The grid is centred on the origin.
        assert_eq!(g.tile_edge_x(0), -40.0);
        assert_eq!(g.tile_edge_x(g.width()), 40.0);
        assert_eq!(g.tile_edge_z(0), -40.0);
        for z in 0..g.height() {
            for x in 0..g.width() {
                assert_eq!(g.world_to_tile(g.tile_center(x, z)), (x, z));
            }
        }
        // Tile edges belong to the tile they open (floor of the exact boundary).
        assert_eq!(
            g.world_to_tile(Vec3::new(g.tile_edge_x(5), 0.0, g.tile_edge_z(9))),
            (5, 9)
        );
        // Collision agrees with the tile query.
        assert!(
            g.is_solid_at_world(Vec3::new(-39.0, 0.0, -39.0)),
            "the border is rock"
        );
        assert!(!g.is_solid_at_world(g.entry_world()));
        assert!(!g.is_solid_at_world(g.exit_world()));
        // Well outside the grid still reads solid, so collision needs no bounds check.
        assert!(g.is_solid_at_world(Vec3::new(1.0e4, 0.0, 0.0)));
    }

    #[test]
    fn tile_size_is_the_only_scale() {
        let g = TileGrid::solid(4, 6);
        assert_eq!(g.tile_edge_x(1) - g.tile_edge_x(0), TILE_SIZE);
        assert_eq!(g.tile_edge_z(1) - g.tile_edge_z(0), TILE_SIZE);
        assert_eq!(g.tile_center(0, 0), Vec3::new(-3.0, 0.0, -5.0));
    }

    // -- hand-authored fixtures ------------------------------------------------------

    #[test]
    fn ascii_round_trips() {
        let rows = ["#####", "#E.X#", "#.+.#", "#####"];
        let g = TileGrid::from_rows(&rows);
        assert_eq!(g.width(), 5);
        assert_eq!(g.height(), 4);
        assert_eq!(g.to_ascii(), rows.join("\n"));
        assert_eq!(g.entry(), (1, 1));
        assert_eq!(g.exit(), (3, 1));
        assert_eq!(g.get(2, 2), Tile::Door);
        assert!(g.all_walkable_reachable());
    }

    #[test]
    fn bfs_measures_walking_distance() {
        // Two cells that are euclidean-adjacent but a long walk apart.
        let g = TileGrid::from_rows(&["#######", "#E#...#", "#.#.#.#", "#...#X#", "#######"]);
        let dist = g.bfs_distances(g.entry());
        let at = |x: i32, z: i32| dist[(z * g.width() + x) as usize];
        assert_eq!(at(1, 1), 0);
        assert_eq!(at(1, 2), 1);
        // (1,1)->(1,3)->(3,3)->(3,1)->(5,1)->(5,3): the walk doubles back twice.
        assert_eq!(at(3, 1), 6);
        assert_eq!(
            at(5, 3),
            10,
            "the exit is ten steps away, though only four tiles across"
        );
        assert_eq!(at(0, 0), u32::MAX, "rock is unreachable");
        assert!(g.all_walkable_reachable());
    }

    #[test]
    fn an_unreachable_pocket_is_detected() {
        let g = TileGrid::from_rows(&["#####", "#E..#", "#####", "#..X#", "#####"]);
        assert!(!g.all_walkable_reachable());
    }

    #[test]
    fn neighbors4_is_ordered_and_walkable_only() {
        let g = TileGrid::from_rows(&["###", "#.#", "...", "#.#"]);
        let n: Vec<_> = g.neighbors4(1, 2).collect();
        // Fixed N, W, E, S order.
        assert_eq!(n, vec![(1, 1), (0, 2), (2, 2), (1, 3)]);
        assert_eq!(g.neighbors4(0, 0).count(), 0);
    }

    // -- parameter robustness --------------------------------------------------------

    #[test]
    fn tiny_and_absurd_params_do_not_panic() {
        let cases = [
            DungeonParams {
                width: 1,
                height: 1,
                ..Default::default()
            },
            DungeonParams {
                width: 12,
                height: 12,
                room_min: 20,
                room_max: 30,
                ..Default::default()
            },
            DungeonParams {
                width: 40,
                height: 40,
                min_rooms: 0,
                max_rooms: 0,
                margin: 0,
                room_attempts: 0,
                loop_corridors: 0,
                ..Default::default()
            },
            DungeonParams {
                width: 64,
                height: 24,
                min_rooms: 12,
                max_rooms: 12,
                room_min: 3,
                room_max: 4,
                ..Default::default()
            },
        ];
        for (i, c) in cases.iter().enumerate() {
            let g = generate(1234, c);
            assert!(
                g.width() >= 7 && g.height() >= 7,
                "case {i} was not sanitized"
            );
            assert!(!g.rooms().is_empty(), "case {i} produced no rooms");
            assert!(
                g.all_walkable_reachable(),
                "case {i} is disconnected\n{}",
                g.to_ascii()
            );
            assert!(g.walkable_count() > 0);
        }
    }

    #[test]
    fn a_long_seed_sweep_stays_connected() {
        // Broader than SEEDS: cheap insurance that the connectivity guarantee is not a
        // property of the twelve seeds that happen to be pinned above.
        let p = DungeonParams::default();
        for seed in 0..200u64 {
            let g = generate(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), &p);
            assert!(
                g.all_walkable_reachable(),
                "seed index {seed} is disconnected"
            );
            assert!(g.rooms().len() >= 2);
        }
    }
}
