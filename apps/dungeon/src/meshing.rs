//! Tile grid → merged chunk meshes (`docs/game-framework-plan.md` §3.2).
//!
//! The deferred path issues **one draw per object** — there is no instancing — so a
//! dungeon meshed tile-by-tile would cost thousands of draws for a floor you could
//! describe with a handful of rectangles. This module therefore does two things at once:
//!
//! * **chunking** — the grid is cut into `chunk_tiles` squares and each chunk becomes one
//!   mesh, so the draw count is `O(chunks)` (nine for the default 40x40 dungeon) rather
//!   than `O(tiles)`. Chunking rather than one giant mesh keeps the per-chunk SDF bake
//!   (the plan's R2 risk) bounded and gives culling something to work with;
//! * **greedy merging** — inside a chunk, coplanar same-facing tile quads are merged into
//!   maximal rectangles, which collapses an open room's floor to a single quad and a
//!   straight wall to a single long quad.
//!
//! The geometry is a pure function of the [`TileGrid`], and the iteration order is fixed
//! at every level (chunks, then faces, then tiles), so the same grid always produces
//! byte-identical vertex and index buffers. That is load-bearing: the cook cache keys
//! baked chunk SDFs on the generator seed, and the ray-tracing acceleration structure is
//! built once at load — both assume the geometry does not wobble between runs.

// See `procgen.rs`: the game loop wires these up in the integration step that follows.
#![allow(dead_code)]

use dreamcoast_asset::MeshVertex;

use crate::procgen::TileGrid;

/// Floor-to-ceiling height in metres. Four metres over a 2 m tile reads as a corridor
/// you could not vault, and keeps the wall texture's world-planar V span a clean 2 tiles.
pub const WALL_HEIGHT: f32 = 4.0;

/// Tiles per chunk edge. 16 tiles = 32 m: large enough that a 40x40 dungeon is nine
/// draws, small enough that one chunk's distance-field bake stays cheap.
pub const CHUNK_TILES: i32 = 16;

/// Metres per UV unit. The projection is world-planar (see [`MeshParams::uv_scale`]), so
/// this is literally "one texture tile every N metres" and is uniform no matter how big
/// a merged rectangle grew.
pub const UV_SCALE: f32 = 2.0;

/// The four orthogonal directions, as *offsets toward the neighbour a wall face would
/// separate us from*. Fixed order: +X, -X, +Z, -Z. Determinism depends on it.
const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Knobs for [`mesh_chunks`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshParams {
    /// Chunk edge in tiles. One merged mesh (one draw) is emitted per non-empty chunk.
    pub chunk_tiles: i32,
    /// Floor-to-ceiling height in metres; wall quads span `0 ..= wall_height`.
    pub wall_height: f32,
    /// Metres per UV unit for the world-planar projection. Floors and ceilings project
    /// on XZ, walls on (horizontal run, Y) — so texel density is identical on every
    /// surface and continuous across the seam between two merged rectangles.
    pub uv_scale: f32,
    /// Emit the ceiling plane. **Off by default:** the top-down camera looks straight
    /// through where the ceiling would be, and drawing it would occlude the whole game.
    /// It exists because an interior needs a ceiling the moment the camera drops to eye
    /// level or the global illumination needs a closed box to bounce inside.
    pub ceiling: bool,
}

impl Default for MeshParams {
    fn default() -> Self {
        MeshParams {
            chunk_tiles: CHUNK_TILES,
            wall_height: WALL_HEIGHT,
            uv_scale: UV_SCALE,
            ceiling: false,
        }
    }
}

impl MeshParams {
    /// Clamp into a range the mesher can satisfy (a zero chunk size or UV scale would
    /// divide by zero / loop forever).
    pub fn sanitized(&self) -> MeshParams {
        MeshParams {
            chunk_tiles: self.chunk_tiles.max(1),
            wall_height: if self.wall_height > 0.0 {
                self.wall_height
            } else {
                WALL_HEIGHT
            },
            uv_scale: if self.uv_scale.abs() > f32::EPSILON {
                self.uv_scale
            } else {
                UV_SCALE
            },
            ceiling: self.ceiling,
        }
    }
}

/// One chunk's merged geometry, ready for the engine's runtime upload path
/// (`upload_geometry(&vertices, &indices)`); [`MeshVertex`] is the engine's 32-byte
/// {pos, normal, uv} layout, so no conversion happens on the way to the GPU.
#[derive(Clone, Debug, Default)]
pub struct ChunkMesh {
    /// Chunk index on the tile grid: tiles `chunk_coord.0 * chunk_tiles ..` on X and
    /// `chunk_coord.1 * chunk_tiles ..` on Z. Part of the cook-cache key for this
    /// chunk's baked distance field.
    pub chunk_coord: (i32, i32),
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    /// Triangles in this chunk.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Quads in this chunk (every quad is 4 vertices and 6 indices).
    pub fn quad_count(&self) -> usize {
        self.vertices.len() / 4
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Aggregate counts across a dungeon's chunks — what the load-time log and the perf
/// budget check report.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MeshStats {
    pub chunks: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub max_chunk_vertices: usize,
    pub max_chunk_triangles: usize,
}

/// Summarise a mesh set.
pub fn mesh_stats(chunks: &[ChunkMesh]) -> MeshStats {
    MeshStats {
        chunks: chunks.len(),
        vertices: chunks.iter().map(|c| c.vertices.len()).sum(),
        triangles: chunks.iter().map(ChunkMesh::triangle_count).sum(),
        max_chunk_vertices: chunks.iter().map(|c| c.vertices.len()).max().unwrap_or(0),
        max_chunk_triangles: chunks
            .iter()
            .map(ChunkMesh::triangle_count)
            .max()
            .unwrap_or(0),
    }
}

/// Build the merged chunk meshes for a dungeon.
///
/// Per chunk, in this fixed order:
///
/// 1. **floors** — the walkable tiles are greedy-merged into maximal rectangles (grow
///    right as far as the row allows, then grow down while the whole run stays walkable)
///    and emitted at `y = 0` facing up;
/// 2. **ceilings** — the *same* rectangles at `y = wall_height`, facing down, when
///    [`MeshParams::ceiling`] is set;
/// 3. **walls** — for each of the four directions, every walkable tile whose neighbour in
///    that direction is solid contributes a floor-to-ceiling face on the shared boundary.
///    Because the height is uniform, the 2D greedy collapses to merging contiguous runs
///    along the axis perpendicular to the face normal.
///
/// A face is owned by its **walkable** tile, so a boundary that straddles two chunks
/// produces exactly one quad, in the chunk of the tile you can stand on. Empty chunks
/// (no geometry at all) are skipped rather than emitted as degenerate meshes.
///
/// 4. **caps** — the SOLID tiles (the negative of the floor set) are greedy-merged the
///    same way and emitted at `y = wall_height` facing up: the rock between rooms is a
///    VOLUME, not a paper boundary, and its top is what makes walls read as thick
///    carved-slab blocks from the game's camera (and what closes the box the moment
///    GI or a sky term looks at the level from above);
/// 5. **rim** — the grid's outer boundary gets outward side faces on its solid tiles,
///    so the slab is closed from every angle the camera can reach (the underside stays
///    open: nothing looks at a dungeon from below).
pub fn mesh_chunks(grid: &TileGrid, params: &MeshParams) -> Vec<ChunkMesh> {
    let p = params.sanitized();
    let chunks_x =
        grid.width().div_euclid(p.chunk_tiles) + i32::from(grid.width() % p.chunk_tiles != 0);
    let chunks_z =
        grid.height().div_euclid(p.chunk_tiles) + i32::from(grid.height() % p.chunk_tiles != 0);

    let mut out = Vec::new();
    for cz in 0..chunks_z {
        for cx in 0..chunks_x {
            let x0 = cx * p.chunk_tiles;
            let z0 = cz * p.chunk_tiles;
            let x1 = (x0 + p.chunk_tiles).min(grid.width());
            let z1 = (z0 + p.chunk_tiles).min(grid.height());
            let mut mesh = ChunkMesh {
                chunk_coord: (cx, cz),
                ..Default::default()
            };
            let rects = greedy_floor_rects(grid, x0, z0, x1, z1);
            emit_floors(&mut mesh, grid, &rects, &p);
            if p.ceiling {
                emit_ceilings(&mut mesh, grid, &rects, &p);
            }
            let caps = greedy_solid_rects(grid, x0, z0, x1, z1);
            emit_caps(&mut mesh, grid, &caps, &p);
            emit_rim(&mut mesh, grid, x0, z0, x1, z1, &p);
            emit_walls(&mut mesh, grid, x0, z0, x1, z1, &p);
            if !mesh.is_empty() {
                out.push(mesh);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------
// Greedy rectangles
// ---------------------------------------------------------------------------------

/// A maximal run of walkable tiles: `x .. x + w` by `z .. z + h`, in grid tiles.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Rect {
    x: i32,
    z: i32,
    w: i32,
    h: i32,
}

/// The classic 2D greedy pass, restricted to one chunk: scan row-major for an unclaimed
/// walkable tile, extend it right as far as the row allows, then extend the whole run
/// downward while every column of the next row is still free. Marks what it takes, so
/// each tile lands in exactly one rectangle.
fn greedy_floor_rects(grid: &TileGrid, x0: i32, z0: i32, x1: i32, z1: i32) -> Vec<Rect> {
    let (cw, ch) = ((x1 - x0).max(0), (z1 - z0).max(0));
    if cw == 0 || ch == 0 {
        return Vec::new();
    }
    let mut taken = vec![false; (cw * ch) as usize];
    let free = |taken: &[bool], lx: i32, lz: i32| {
        !taken[(lz * cw + lx) as usize] && grid.is_walkable(x0 + lx, z0 + lz)
    };

    let mut rects = Vec::new();
    for lz in 0..ch {
        for lx in 0..cw {
            if !free(&taken, lx, lz) {
                continue;
            }
            let mut w = 1;
            while lx + w < cw && free(&taken, lx + w, lz) {
                w += 1;
            }
            let mut h = 1;
            while lz + h < ch && (0..w).all(|k| free(&taken, lx + k, lz + h)) {
                h += 1;
            }
            for dz in 0..h {
                for dx in 0..w {
                    taken[((lz + dz) * cw + lx + dx) as usize] = true;
                }
            }
            rects.push(Rect {
                x: x0 + lx,
                z: z0 + lz,
                w,
                h,
            });
        }
    }
    rects
}

/// The same greedy pass over the SOLID tiles — the rock volume whose top becomes the
/// wall caps. Shares [`greedy_floor_rects`]'s scan order for the same determinism.
fn greedy_solid_rects(grid: &TileGrid, x0: i32, z0: i32, x1: i32, z1: i32) -> Vec<Rect> {
    let (cw, ch) = ((x1 - x0).max(0), (z1 - z0).max(0));
    if cw == 0 || ch == 0 {
        return Vec::new();
    }
    let mut taken = vec![false; (cw * ch) as usize];
    let free = |taken: &[bool], lx: i32, lz: i32| {
        !taken[(lz * cw + lx) as usize] && grid.is_solid(x0 + lx, z0 + lz)
    };

    let mut rects = Vec::new();
    for lz in 0..ch {
        for lx in 0..cw {
            if !free(&taken, lx, lz) {
                continue;
            }
            let mut w = 1;
            while lx + w < cw && free(&taken, lx + w, lz) {
                w += 1;
            }
            let mut h = 1;
            while lz + h < ch && (0..w).all(|k| free(&taken, lx + k, lz + h)) {
                h += 1;
            }
            for dz in 0..h {
                for dx in 0..w {
                    taken[((lz + dz) * cw + lx + dx) as usize] = true;
                }
            }
            rects.push(Rect {
                x: x0 + lx,
                z: z0 + lz,
                w,
                h,
            });
        }
    }
    rects
}

// ---------------------------------------------------------------------------------
// Face emission
// ---------------------------------------------------------------------------------

fn emit_floors(mesh: &mut ChunkMesh, grid: &TileGrid, rects: &[Rect], p: &MeshParams) {
    for r in rects {
        let (x0, x1) = (grid.tile_edge_x(r.x), grid.tile_edge_x(r.x + r.w));
        let (z0, z1) = (grid.tile_edge_z(r.z), grid.tile_edge_z(r.z + r.h));
        push_quad(
            mesh,
            [[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]],
            [0.0, 1.0, 0.0],
            p.uv_scale,
        );
    }
}

fn emit_ceilings(mesh: &mut ChunkMesh, grid: &TileGrid, rects: &[Rect], p: &MeshParams) {
    let y = p.wall_height;
    for r in rects {
        let (x0, x1) = (grid.tile_edge_x(r.x), grid.tile_edge_x(r.x + r.w));
        let (z0, z1) = (grid.tile_edge_z(r.z), grid.tile_edge_z(r.z + r.h));
        // Reversed corner order against the floor: the ceiling faces down, into the room.
        push_quad(
            mesh,
            [[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]],
            [0.0, -1.0, 0.0],
            p.uv_scale,
        );
    }
}

/// The rock volume's top: the solid rectangles at `y = wall_height`, facing up. Same
/// corner order as a floor — an up-facing quad is an up-facing quad.
fn emit_caps(mesh: &mut ChunkMesh, grid: &TileGrid, rects: &[Rect], p: &MeshParams) {
    let y = p.wall_height;
    for r in rects {
        let (x0, x1) = (grid.tile_edge_x(r.x), grid.tile_edge_x(r.x + r.w));
        let (z0, z1) = (grid.tile_edge_z(r.z), grid.tile_edge_z(r.z + r.h));
        push_quad(
            mesh,
            [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]],
            [0.0, 1.0, 0.0],
            p.uv_scale,
        );
    }
}

/// The slab's outer skirt: outward faces on the GRID boundary's solid tiles, merged
/// into runs like the interior walls. Normals point off the grid — these faces exist
/// for oblique camera angles at the map edge, not for gameplay space.
fn emit_rim(
    mesh: &mut ChunkMesh,
    grid: &TileGrid,
    x0: i32,
    z0: i32,
    x1: i32,
    z1: i32,
    p: &MeshParams,
) {
    let y = p.wall_height;
    let run_emit =
        |solid: &dyn Fn(i32) -> bool, a0: i32, a1: i32, emit: &mut dyn FnMut(i32, i32)| {
            let mut a = a0;
            while a < a1 {
                if !solid(a) {
                    a += 1;
                    continue;
                }
                let mut run = 1;
                while a + run < a1 && solid(a + run) {
                    run += 1;
                }
                emit(a, run);
                a += run;
            }
        };
    // West rim (grid x = 0): faces -X, runs along Z.
    if x0 == 0 {
        let x = grid.tile_edge_x(0);
        run_emit(&|z| grid.is_solid(0, z), z0, z1, &mut |z, run| {
            let (za, zb) = (grid.tile_edge_z(z), grid.tile_edge_z(z + run));
            push_quad(
                mesh,
                [[x, 0.0, zb], [x, 0.0, za], [x, y, za], [x, y, zb]],
                [-1.0, 0.0, 0.0],
                p.uv_scale,
            );
        });
    }
    // East rim (grid x = width - 1): faces +X.
    if x1 == grid.width() {
        let tx = grid.width() - 1;
        let x = grid.tile_edge_x(grid.width());
        run_emit(&|z| grid.is_solid(tx, z), z0, z1, &mut |z, run| {
            let (za, zb) = (grid.tile_edge_z(z), grid.tile_edge_z(z + run));
            push_quad(
                mesh,
                [[x, 0.0, za], [x, 0.0, zb], [x, y, zb], [x, y, za]],
                [1.0, 0.0, 0.0],
                p.uv_scale,
            );
        });
    }
    // North rim (grid z = 0): faces -Z, runs along X.
    if z0 == 0 {
        let z = grid.tile_edge_z(0);
        run_emit(&|x| grid.is_solid(x, 0), x0, x1, &mut |x, run| {
            let (xa, xb) = (grid.tile_edge_x(x), grid.tile_edge_x(x + run));
            push_quad(
                mesh,
                [[xa, 0.0, z], [xb, 0.0, z], [xb, y, z], [xa, y, z]],
                [0.0, 0.0, -1.0],
                p.uv_scale,
            );
        });
    }
    // South rim (grid z = height - 1): faces +Z.
    if z1 == grid.height() {
        let tz = grid.height() - 1;
        let z = grid.tile_edge_z(grid.height());
        run_emit(&|x| grid.is_solid(x, tz), x0, x1, &mut |x, run| {
            let (xa, xb) = (grid.tile_edge_x(x), grid.tile_edge_x(x + run));
            push_quad(
                mesh,
                [[xb, 0.0, z], [xa, 0.0, z], [xa, y, z], [xb, y, z]],
                [0.0, 0.0, 1.0],
                p.uv_scale,
            );
        });
    }
}

/// Walls, direction by direction. `dir` points at the *solid* neighbour, so the face
/// normal is `-dir`: a wall always faces the tile you can stand on.
fn emit_walls(
    mesh: &mut ChunkMesh,
    grid: &TileGrid,
    x0: i32,
    z0: i32,
    x1: i32,
    z1: i32,
    p: &MeshParams,
) {
    for (dx, dz) in DIRS {
        let face = |x: i32, z: i32| grid.is_walkable(x, z) && grid.is_solid(x + dx, z + dz);
        if dx != 0 {
            // Faces lie on YZ planes at a constant x: merge contiguous runs along Z.
            for x in x0..x1 {
                let mut z = z0;
                while z < z1 {
                    if !face(x, z) {
                        z += 1;
                        continue;
                    }
                    let mut run = 1;
                    while z + run < z1 && face(x, z + run) {
                        run += 1;
                    }
                    push_wall_x(mesh, grid, x, z, run, dx, p);
                    z += run;
                }
            }
        } else {
            // Faces lie on XY planes at a constant z: merge contiguous runs along X.
            for z in z0..z1 {
                let mut x = x0;
                while x < x1 {
                    if !face(x, z) {
                        x += 1;
                        continue;
                    }
                    let mut run = 1;
                    while x + run < x1 && face(x + run, z) {
                        run += 1;
                    }
                    push_wall_z(mesh, grid, x, z, run, dz, p);
                    x += run;
                }
            }
        }
    }
}

/// A wall on a YZ plane. `dx` is +1 when the rock is to the east (so the face looks west)
/// and -1 when it is to the west.
fn push_wall_x(
    mesh: &mut ChunkMesh,
    grid: &TileGrid,
    tx: i32,
    tz: i32,
    run: i32,
    dx: i32,
    p: &MeshParams,
) {
    let y = p.wall_height;
    let (za, zb) = (grid.tile_edge_z(tz), grid.tile_edge_z(tz + run));
    if dx > 0 {
        let x = grid.tile_edge_x(tx + 1);
        push_quad(
            mesh,
            [[x, 0.0, zb], [x, 0.0, za], [x, y, za], [x, y, zb]],
            [-1.0, 0.0, 0.0],
            p.uv_scale,
        );
    } else {
        let x = grid.tile_edge_x(tx);
        push_quad(
            mesh,
            [[x, 0.0, za], [x, 0.0, zb], [x, y, zb], [x, y, za]],
            [1.0, 0.0, 0.0],
            p.uv_scale,
        );
    }
}

/// A wall on an XY plane. `dz` is +1 when the rock is to the south, -1 when to the north.
fn push_wall_z(
    mesh: &mut ChunkMesh,
    grid: &TileGrid,
    tx: i32,
    tz: i32,
    run: i32,
    dz: i32,
    p: &MeshParams,
) {
    let y = p.wall_height;
    let (xa, xb) = (grid.tile_edge_x(tx), grid.tile_edge_x(tx + run));
    if dz > 0 {
        let z = grid.tile_edge_z(tz + 1);
        push_quad(
            mesh,
            [[xa, 0.0, z], [xb, 0.0, z], [xb, y, z], [xa, y, z]],
            [0.0, 0.0, -1.0],
            p.uv_scale,
        );
    } else {
        let z = grid.tile_edge_z(tz);
        push_quad(
            mesh,
            [[xb, 0.0, z], [xa, 0.0, z], [xa, y, z], [xb, y, z]],
            [0.0, 0.0, 1.0],
            p.uv_scale,
        );
    }
}

/// World-planar UV for a vertex, chosen by the dominant axis of its face normal.
///
/// Projecting from world position (rather than from the rectangle's own extent) is what
/// makes texel density uniform: a 1x1 tile and a 16x4 merged rectangle get exactly the
/// same texture scale, and two rectangles that meet share UVs along the seam.
fn planar_uv(pos: [f32; 3], normal: [f32; 3], scale: f32) -> [f32; 2] {
    if normal[1].abs() > 0.5 {
        // Floor / ceiling: project on XZ.
        [pos[0] / scale, pos[2] / scale]
    } else if normal[0].abs() > 0.5 {
        // East/west wall: the horizontal run is Z.
        [pos[2] / scale, pos[1] / scale]
    } else {
        // North/south wall: the horizontal run is X.
        [pos[0] / scale, pos[1] / scale]
    }
}

/// Append one quad as two triangles, **wound counter-clockwise about its own normal**.
///
/// Both triangles satisfy `cross(p1 - p0, p2 - p0) · normal > 0`, i.e. the glTF
/// front-face convention, which is what every consumer that cares about facing assumes:
///
/// * the **virtual-geometry G-buffer producer** (`P14_VGEO`, *default on* for content
///   scenes) backface-culls single-sided materials per triangle, with `area > 0` in
///   screen-pixel space meaning "back". A quad wound the other way is culled — it
///   vanishes from the G-buffer while still being lit, shadowed and in the distance
///   field, which reads as a hole in the floor rather than as a winding bug;
/// * a ray-tracing hit shader deriving a geometric normal, and any future pass that
///   turns culling on, agree with the shading normal instead of opposing it.
///
/// The corner order the callers pass is the *outward* order for the face (`+X`, `-X`,
/// … as named at each call site), so the flip lives here, once: the emitted triangles
/// are `[0,2,1]` and `[0,3,2]`.
///
/// **Not** the order `crates/asset/src/primitives.rs` `unit_cube` uses — it indexes its
/// (identically ordered) corners `[0,1,2, 2,3,0]`, which winds *opposite* its shading
/// normals. That is invisible in the fixed-function path (raster culling is off on every
/// backend) and it is why this module originally copied it, but it is not invisible to
/// the vgeo producer. See the M1 report: the engine's own `cube`/`ground` procedural
/// assets have the same problem, and fixing them is an engine-track change.
fn push_quad(mesh: &mut ChunkMesh, corners: [[f32; 3]; 4], normal: [f32; 3], uv_scale: f32) {
    let base = mesh.vertices.len() as u32;
    for pos in corners {
        mesh.vertices.push(MeshVertex {
            pos,
            normal,
            uv: planar_uv(pos, normal, uv_scale),
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
}

// ---------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procgen::{DungeonParams, TILE_SIZE, Tile, generate};
    use std::collections::BTreeSet;

    /// A 10x6 grid whose interior is one open 8x4 room. Small enough to state the whole
    /// expected mesh by hand, big enough that greedy merging has something to do.
    const ROOM_ROWS: [&str; 6] = [
        "##########",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "##########",
    ];

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Round a world coordinate back to the tile boundary index it sits on.
    fn edge_index(world: f32, world_min: f32) -> i32 {
        ((world - world_min) / TILE_SIZE).round() as i32
    }

    /// Reconstruct, from the emitted geometry alone, which tile-to-tile boundaries carry
    /// a wall face. Returns `(tile_x, tile_z, dir_index)` where `dir_index` indexes
    /// [`DIRS`] and the tile is the *walkable* side. Duplicates are returned too, so a
    /// double-covered edge is detectable.
    fn covered_wall_edges(grid: &TileGrid, chunks: &[ChunkMesh]) -> Vec<(i32, i32, usize)> {
        let mut edges = Vec::new();
        for c in chunks {
            for quad in c.vertices.chunks_exact(4) {
                let n = quad[0].normal;
                if n[1].abs() > 0.5 {
                    continue; // floor or ceiling
                }
                let (xs, zs): (Vec<f32>, Vec<f32>) = (
                    quad.iter().map(|v| v.pos[0]).collect(),
                    quad.iter().map(|v| v.pos[2]).collect(),
                );
                let fmin = |v: &[f32]| v.iter().copied().fold(f32::INFINITY, f32::min);
                let fmax = |v: &[f32]| v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if n[0].abs() > 0.5 {
                    // Plane at constant x; the run is along Z.
                    let ex = edge_index(xs[0], grid.world_min_x());
                    // Rim faces sit on the grid's outer boundary planes, facing off the
                    // grid — they are the slab's skirt, not wall coverage.
                    if (n[0] < 0.0 && ex == 0) || (n[0] > 0.0 && ex == grid.width()) {
                        continue;
                    }
                    // normal -X => rock is east, so the walkable tile is the one before
                    // the plane; normal +X => rock is west, walkable tile starts here.
                    let (tx, dir) = if n[0] < 0.0 { (ex - 1, 0) } else { (ex, 1) };
                    let z0 = edge_index(fmin(&zs), grid.world_min_z());
                    let z1 = edge_index(fmax(&zs), grid.world_min_z());
                    for tz in z0..z1 {
                        edges.push((tx, tz, dir));
                    }
                } else {
                    let ez = edge_index(zs[0], grid.world_min_z());
                    if (n[2] < 0.0 && ez == 0) || (n[2] > 0.0 && ez == grid.height()) {
                        continue;
                    }
                    let (tz, dir) = if n[2] < 0.0 { (ez - 1, 2) } else { (ez, 3) };
                    let x0 = edge_index(fmin(&xs), grid.world_min_x());
                    let x1 = edge_index(fmax(&xs), grid.world_min_x());
                    for tx in x0..x1 {
                        edges.push((tx, tz, dir));
                    }
                }
            }
        }
        edges
    }

    /// Every boundary the grid says should carry a wall.
    fn expected_wall_edges(grid: &TileGrid) -> BTreeSet<(i32, i32, usize)> {
        let mut set = BTreeSet::new();
        for z in 0..grid.height() {
            for x in 0..grid.width() {
                if !grid.is_walkable(x, z) {
                    continue;
                }
                for (i, (dx, dz)) in DIRS.iter().enumerate() {
                    if grid.is_solid(x + dx, z + dz) {
                        set.insert((x, z, i));
                    }
                }
            }
        }
        set
    }

    // -- the hand-authored fixture ---------------------------------------------------

    #[test]
    fn one_open_room_meshes_to_exactly_five_quads() {
        let grid = TileGrid::from_rows(&ROOM_ROWS);
        let chunks = mesh_chunks(&grid, &MeshParams::default());

        assert_eq!(chunks.len(), 1, "10x6 tiles fits in one 16-tile chunk");
        let m = &chunks[0];
        assert_eq!(m.chunk_coord, (0, 0));
        // 1 merged floor + 4 solid caps (top row / left column / right column / bottom
        // row of the rock ring) + 4 rim faces (one per grid side) + 4 merged walls.
        // Naive would be 32 floors + 24 walls + 28 caps + 28 rim faces = 112.
        assert_eq!(m.quad_count(), 13);
        assert_eq!(m.vertices.len(), 52);
        assert_eq!(m.indices.len(), 78);
        assert_eq!(m.triangle_count(), 26);

        // The floor is emitted first and spans the whole 8x4 interior in one rectangle.
        let floor: Vec<[f32; 3]> = m.vertices[..4].iter().map(|v| v.pos).collect();
        assert_eq!(
            floor,
            vec![
                [-8.0, 0.0, -4.0],
                [8.0, 0.0, -4.0],
                [8.0, 0.0, 4.0],
                [-8.0, 0.0, 4.0],
            ]
        );
        assert!(m.vertices[..4].iter().all(|v| v.normal == [0.0, 1.0, 0.0]));
        let floor_uv: Vec<[f32; 2]> = m.vertices[..4].iter().map(|v| v.uv).collect();
        assert_eq!(
            floor_uv,
            vec![[-4.0, -2.0], [4.0, -2.0], [4.0, 2.0], [-4.0, 2.0]],
            "world-planar XZ at 2 m per UV unit"
        );

        // Caps + rim sit between the floor and the walls (fixed emission order); the
        // walls follow in DIRS order: +X rock (west-facing), -X rock (east-facing),
        // +Z rock (north-facing), -Z rock (south-facing).
        let wall_normals: Vec<[f32; 3]> = m.vertices[36..]
            .chunks_exact(4)
            .map(|q| q[0].normal)
            .collect();
        assert_eq!(
            wall_normals,
            vec![
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        );

        // The east-facing wall (rock to the west, at tile column 1's left edge).
        let west = &m.vertices[40..44];
        assert_eq!(
            west.iter().map(|v| v.pos).collect::<Vec<_>>(),
            vec![
                [-8.0, 0.0, -4.0],
                [-8.0, 0.0, 4.0],
                [-8.0, 4.0, 4.0],
                [-8.0, 4.0, -4.0],
            ]
        );
        assert_eq!(
            west.iter().map(|v| v.uv).collect::<Vec<_>>(),
            vec![[-2.0, 0.0], [2.0, 0.0], [2.0, 2.0], [-2.0, 2.0]],
            "walls project (horizontal run, Y); 4 m of height is 2 UV units"
        );
    }

    #[test]
    fn ceiling_is_opt_in_and_mirrors_the_floor() {
        let grid = TileGrid::from_rows(&ROOM_ROWS);
        let off = mesh_chunks(&grid, &MeshParams::default());
        let on = mesh_chunks(
            &grid,
            &MeshParams {
                ceiling: true,
                ..Default::default()
            },
        );
        assert_eq!(off[0].quad_count() + 1, on[0].quad_count());

        let ceil = &on[0].vertices[4..8];
        assert!(
            ceil.iter().all(|v| v.normal == [0.0, -1.0, 0.0]),
            "faces down"
        );
        assert!(ceil.iter().all(|v| v.pos[1] == WALL_HEIGHT));
        // Same footprint as the floor rectangle, opposite winding.
        let xs: Vec<f32> = ceil.iter().map(|v| v.pos[0]).collect();
        let zs: Vec<f32> = ceil.iter().map(|v| v.pos[2]).collect();
        assert_eq!(xs.iter().cloned().fold(f32::INFINITY, f32::min), -8.0);
        assert_eq!(zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max), 4.0);
    }

    // -- greedy merging actually merges -----------------------------------------------

    #[test]
    fn greedy_merging_beats_naive_by_a_wide_margin() {
        let grid = TileGrid::from_rows(&ROOM_ROWS);
        let chunks = mesh_chunks(&grid, &MeshParams::default());

        // Naive: one quad per walkable tile, one per walkable/solid boundary, one per
        // solid tile (cap), one per solid border tile (rim).
        let solid = (grid.width() * grid.height()) as usize - grid.walkable_count();
        let rim = 2 * (grid.width() + grid.height()) as usize - 4;
        let naive = grid.walkable_count() + expected_wall_edges(&grid).len() + solid + rim;
        assert_eq!(naive, 32 + 24 + 28 + 28);
        let merged: usize = chunks.iter().map(ChunkMesh::quad_count).sum();
        assert_eq!(merged, 13);
        assert!(
            merged * 8 < naive,
            "greedy must be a large win, not a rounding one"
        );
    }

    #[test]
    fn greedy_rectangles_tile_the_walkable_set_exactly_once() {
        let grid = generate(42, &DungeonParams::default());
        let p = MeshParams::default().sanitized();
        let mut covered = vec![0u32; (grid.width() * grid.height()) as usize];
        for cz in 0..(grid.height() + p.chunk_tiles - 1) / p.chunk_tiles {
            for cx in 0..(grid.width() + p.chunk_tiles - 1) / p.chunk_tiles {
                let (x0, z0) = (cx * p.chunk_tiles, cz * p.chunk_tiles);
                let x1 = (x0 + p.chunk_tiles).min(grid.width());
                let z1 = (z0 + p.chunk_tiles).min(grid.height());
                for r in greedy_floor_rects(&grid, x0, z0, x1, z1) {
                    assert!(r.w > 0 && r.h > 0);
                    for z in r.z..r.z + r.h {
                        for x in r.x..r.x + r.w {
                            assert!(grid.is_walkable(x, z), "a rectangle covered rock");
                            covered[(z * grid.width() + x) as usize] += 1;
                        }
                    }
                }
            }
        }
        for z in 0..grid.height() {
            for x in 0..grid.width() {
                let c = covered[(z * grid.width() + x) as usize];
                assert_eq!(
                    c,
                    u32::from(grid.is_walkable(x, z)),
                    "tile ({x},{z}) covered {c}x"
                );
            }
        }
    }

    #[test]
    fn a_long_corridor_merges_into_one_quad_per_side() {
        // 1-tile corridor, 12 tiles long: floor merges to one rectangle, each side wall
        // to one long quad, plus the two end caps.
        let rows = ["##############", "#............#", "##############"];
        let grid = TileGrid::from_rows(&rows);
        let chunks = mesh_chunks(&grid, &MeshParams::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].quad_count(),
            13,
            "1 floor + 2 long sides + 2 end walls + 4 solid caps + 4 rim faces"
        );
    }

    // -- watertightness ----------------------------------------------------------------

    #[test]
    fn every_boundary_carries_exactly_one_wall_quad() {
        for seed in [0u64, 1, 42, 4711, u64::MAX] {
            let grid = generate(seed, &DungeonParams::default());
            let chunks = mesh_chunks(&grid, &MeshParams::default());
            let covered = covered_wall_edges(&grid, &chunks);
            let unique: BTreeSet<_> = covered.iter().copied().collect();
            assert_eq!(
                covered.len(),
                unique.len(),
                "seed {seed}: a boundary is covered by more than one wall quad"
            );
            assert_eq!(
                unique,
                expected_wall_edges(&grid),
                "seed {seed}: wall coverage does not match the grid"
            );
        }
    }

    #[test]
    fn no_wall_sits_inside_rock_or_between_two_floors() {
        let grid = generate(7, &DungeonParams::default());
        let chunks = mesh_chunks(&grid, &MeshParams::default());
        for (x, z, dir) in covered_wall_edges(&grid, &chunks) {
            let (dx, dz) = DIRS[dir];
            assert!(grid.is_walkable(x, z), "wall owner ({x},{z}) is rock");
            assert!(
                grid.is_solid(x + dx, z + dz),
                "wall at ({x},{z}) dir {dir} separates two walkable tiles"
            );
        }
    }

    #[test]
    fn a_boundary_that_straddles_two_chunks_is_meshed_once() {
        // A 20-wide corridor crosses the chunk seam at x = 16 with an 8-tile chunk size.
        let rows = [
            "######################",
            "#....................#",
            "######################",
        ];
        let grid = TileGrid::from_rows(&rows);
        let p = MeshParams {
            chunk_tiles: 8,
            ..Default::default()
        };
        let chunks = mesh_chunks(&grid, &p);
        assert!(chunks.len() > 1, "the fixture must actually span chunks");
        let covered = covered_wall_edges(&grid, &chunks);
        let unique: BTreeSet<_> = covered.iter().copied().collect();
        assert_eq!(covered.len(), unique.len());
        assert_eq!(unique, expected_wall_edges(&grid));
    }

    // -- orientation --------------------------------------------------------------------

    #[test]
    fn wall_normals_point_into_walkable_space() {
        let grid = generate(42, &DungeonParams::default());
        let chunks = mesh_chunks(&grid, &MeshParams::default());
        for c in &chunks {
            for quad in c.vertices.chunks_exact(4) {
                let n = quad[0].normal;
                assert!(
                    quad.iter().all(|v| v.normal == n),
                    "a quad must be flat-shaded"
                );
                if n[1].abs() > 0.5 {
                    continue;
                }
                // Rim faces sit on the grid's outer boundary planes and face off the
                // grid; they have no walkable side by design.
                let on_boundary = |v: f32, min: f32, edges: i32| {
                    let e = edge_index(v, min);
                    e == 0 || e == edges
                };
                if (n[0].abs() > 0.5
                    && on_boundary(quad[0].pos[0], grid.world_min_x(), grid.width()))
                    || (n[2].abs() > 0.5
                        && on_boundary(quad[0].pos[2], grid.world_min_z(), grid.height()))
                {
                    continue;
                }
                // Step from the face's own centre along the normal: we must land on a
                // walkable tile, and stepping the other way must land on rock.
                let centre = [
                    quad.iter().map(|v| v.pos[0]).sum::<f32>() / 4.0,
                    0.0,
                    quad.iter().map(|v| v.pos[2]).sum::<f32>() / 4.0,
                ];
                let step = TILE_SIZE * 0.5;
                let inside = glam::Vec3::new(centre[0] + n[0] * step, 0.0, centre[2] + n[2] * step);
                let outside =
                    glam::Vec3::new(centre[0] - n[0] * step, 0.0, centre[2] - n[2] * step);
                let (ix, iz) = grid.world_to_tile(inside);
                assert!(grid.is_walkable(ix, iz), "normal {n:?} points into rock");
                assert!(
                    grid.is_solid_at_world(outside),
                    "the far side of a wall must be rock"
                );
                // And the normal genuinely faces the open tile.
                let to_open = sub(grid.tile_center(ix, iz).to_array(), centre);
                assert!(dot(to_open, n) > 0.0);
            }
        }
    }

    /// Floors lie on the floor plane facing up, ceilings on the ceiling plane facing
    /// down — and **every emitted triangle is wound counter-clockwise about its own
    /// normal** (the glTF front-face convention).
    ///
    /// The winding half is the load-bearing one: the virtual-geometry producer
    /// backface-culls single-sided materials per triangle, so a quad wound the other way
    /// is simply absent from the G-buffer. This asserts on the *emitted indices*, not on
    /// the corner order, because the indices are what the rasterizer sees.
    #[test]
    fn floors_face_up_and_every_triangle_is_front_facing() {
        let grid = generate(42, &DungeonParams::default());
        let chunks = mesh_chunks(
            &grid,
            &MeshParams {
                ceiling: true,
                ..Default::default()
            },
        );
        let mut floors = 0;
        let mut ceilings = 0;
        for c in &chunks {
            for quad in c.vertices.chunks_exact(4) {
                let n = quad[0].normal;
                match n {
                    [0.0, 1.0, 0.0] => {
                        // Up-facing quads are floors (y = 0) or solid caps (y = height).
                        let y = quad[0].pos[1];
                        assert!(y == 0.0 || y == WALL_HEIGHT, "up-facing quad at y = {y}");
                        assert!(quad.iter().all(|v| v.pos[1] == y));
                        if y == 0.0 {
                            floors += 1;
                        }
                    }
                    [0.0, -1.0, 0.0] => {
                        ceilings += 1;
                        assert!(quad.iter().all(|v| v.pos[1] == WALL_HEIGHT));
                    }
                    _ => {}
                }
            }
            for tri in c.indices.chunks_exact(3) {
                let p = |i: usize| c.vertices[tri[i] as usize].pos;
                let n = c.vertices[tri[0] as usize].normal;
                let g = cross(sub(p(1), p(0)), sub(p(2), p(0)));
                assert!(
                    dot(g, n) > 0.0,
                    "triangle {tri:?} is wound backwards for normal {n:?}"
                );
            }
        }
        assert!(floors > 0 && floors == ceilings);
    }

    // -- index validity and determinism -------------------------------------------------

    #[test]
    fn indices_are_in_range_and_form_whole_triangles() {
        let grid = generate(42, &DungeonParams::default());
        for ceiling in [false, true] {
            let chunks = mesh_chunks(
                &grid,
                &MeshParams {
                    ceiling,
                    ..Default::default()
                },
            );
            for c in &chunks {
                assert!(!c.vertices.is_empty(), "empty chunks must not be emitted");
                assert_eq!(c.indices.len() % 3, 0);
                assert_eq!(c.vertices.len() % 4, 0);
                assert_eq!(c.indices.len(), c.quad_count() * 6);
                let n = c.vertices.len() as u32;
                assert!(c.indices.iter().all(|i| *i < n), "index out of range");
                // Every vertex is referenced (no orphans bloating the buffer).
                let used: BTreeSet<u32> = c.indices.iter().copied().collect();
                assert_eq!(used.len(), c.vertices.len());
                // No degenerate triangles.
                for t in c.indices.chunks_exact(3) {
                    assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
                    let a = c.vertices[t[0] as usize].pos;
                    let b = c.vertices[t[1] as usize].pos;
                    let d = c.vertices[t[2] as usize].pos;
                    let g = cross(sub(b, a), sub(d, a));
                    assert!(dot(g, g) > 1.0e-6, "degenerate triangle");
                }
            }
        }
    }

    /// Byte view of a mesh set — exactly what the upload path hands the GPU, so hashing
    /// it is the strictest determinism check available on the CPU side.
    fn mesh_bytes(chunks: &[ChunkMesh]) -> Vec<u8> {
        let mut out = Vec::new();
        for c in chunks {
            out.extend_from_slice(&c.chunk_coord.0.to_le_bytes());
            out.extend_from_slice(&c.chunk_coord.1.to_le_bytes());
            for v in &c.vertices {
                for f in v.pos.iter().chain(v.normal.iter()).chain(v.uv.iter()) {
                    out.extend_from_slice(&f.to_bits().to_le_bytes());
                }
            }
            for i in &c.indices {
                out.extend_from_slice(&i.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn the_same_grid_meshes_byte_identically() {
        let p = MeshParams::default();
        for seed in [0u64, 42, 999] {
            let grid = generate(seed, &DungeonParams::default());
            let a = mesh_bytes(&mesh_chunks(&grid, &p));
            let b = mesh_bytes(&mesh_chunks(&grid, &p));
            assert_eq!(a, b, "seed {seed}: mesh output is not reproducible");
            // And re-generating the grid from the seed reproduces it too, which is the
            // property the cook cache key (generator version + seed) actually relies on.
            let regen = generate(seed, &DungeonParams::default());
            assert_eq!(mesh_bytes(&mesh_chunks(&regen, &p)), a);
        }
    }

    #[test]
    fn chunk_order_is_row_major_and_coords_are_consistent() {
        let grid = generate(42, &DungeonParams::default());
        let p = MeshParams::default();
        let chunks = mesh_chunks(&grid, &p);
        let coords: Vec<(i32, i32)> = chunks.iter().map(|c| c.chunk_coord).collect();
        let mut sorted = coords.clone();
        sorted.sort_by_key(|c| (c.1, c.0));
        assert_eq!(
            coords, sorted,
            "chunks must come out in a fixed row-major order"
        );
        // Every vertex of a chunk lies inside that chunk's world extent (walls sit on
        // the far boundary of their owning tile, so the extent is inclusive).
        for c in &chunks {
            let x0 = grid.tile_edge_x(c.chunk_coord.0 * p.chunk_tiles);
            let x1 = grid.tile_edge_x(((c.chunk_coord.0 + 1) * p.chunk_tiles).min(grid.width()));
            let z0 = grid.tile_edge_z(c.chunk_coord.1 * p.chunk_tiles);
            let z1 = grid.tile_edge_z(((c.chunk_coord.1 + 1) * p.chunk_tiles).min(grid.height()));
            for v in &c.vertices {
                assert!(
                    v.pos[0] >= x0 && v.pos[0] <= x1,
                    "vertex escaped its chunk on X"
                );
                assert!(
                    v.pos[2] >= z0 && v.pos[2] <= z1,
                    "vertex escaped its chunk on Z"
                );
            }
        }
    }

    // -- UVs -------------------------------------------------------------------------

    #[test]
    fn uvs_are_continuous_across_a_merged_rectangle_seam() {
        // Two chunks of floor side by side: the merged rectangles meet at x = 0 and must
        // agree on U there, or the texture would visibly jump at the chunk seam.
        let grid = TileGrid::from_rows(&[
            "##################",
            "#................#",
            "#................#",
            "##################",
        ]);
        let p = MeshParams {
            chunk_tiles: 8,
            ..Default::default()
        };
        let chunks = mesh_chunks(&grid, &p);
        assert!(chunks.len() > 1);
        for c in &chunks {
            for quad in c.vertices.chunks_exact(4) {
                for v in quad {
                    let expect = planar_uv(v.pos, v.normal, p.uv_scale);
                    assert_eq!(v.uv, expect, "UV must be a pure function of world position");
                }
            }
        }
        // Concretely: two vertices at the same world position have the same UV, whichever
        // chunk or rectangle they came from.
        let mut seen: Vec<([u32; 3], [f32; 2], [f32; 3])> = Vec::new();
        for c in &chunks {
            for v in &c.vertices {
                let key = [v.pos[0].to_bits(), v.pos[1].to_bits(), v.pos[2].to_bits()];
                if let Some((_, uv, n)) = seen.iter().find(|(k, _, n)| *k == key && *n == v.normal)
                {
                    assert_eq!(*uv, v.uv);
                    assert_eq!(*n, v.normal);
                } else {
                    seen.push((key, v.uv, v.normal));
                }
            }
        }
    }

    #[test]
    fn uv_density_does_not_depend_on_rectangle_size() {
        let grid = TileGrid::from_rows(&ROOM_ROWS);
        let chunks = mesh_chunks(&grid, &MeshParams::default());
        let floor = &chunks[0].vertices[..4];
        let du = floor[1].uv[0] - floor[0].uv[0];
        let dx = floor[1].pos[0] - floor[0].pos[0];
        assert_eq!(dx / du, UV_SCALE, "one UV unit is exactly UV_SCALE metres");
    }

    // -- degenerate inputs ----------------------------------------------------------

    #[test]
    fn an_all_rock_grid_is_a_closed_slab() {
        // No rooms: nothing walkable, so no floors and no interior walls — but the rock
        // volume itself still has a top and an outer skirt.
        let grid = TileGrid::solid(32, 32);
        let chunks = mesh_chunks(&grid, &MeshParams::default());
        assert!(!chunks.is_empty());
        for c in &chunks {
            for quad in c.vertices.chunks_exact(4) {
                let n = quad[0].normal;
                if n[1] > 0.5 {
                    assert!(quad.iter().all(|v| v.pos[1] == WALL_HEIGHT), "cap only");
                } else {
                    // Side faces may only be the rim: on the grid's outer boundary.
                    let on_x = quad[0].pos[0] == grid.tile_edge_x(0)
                        || quad[0].pos[0] == grid.tile_edge_x(grid.width());
                    let on_z = quad[0].pos[2] == grid.tile_edge_z(0)
                        || quad[0].pos[2] == grid.tile_edge_z(grid.height());
                    assert!(on_x || on_z, "an interior wall appeared with no rooms");
                }
            }
        }
    }

    #[test]
    fn absurd_mesh_params_do_not_panic() {
        let grid = generate(3, &DungeonParams::default());
        for p in [
            MeshParams {
                chunk_tiles: 0,
                ..Default::default()
            },
            MeshParams {
                chunk_tiles: 1,
                ..Default::default()
            },
            MeshParams {
                chunk_tiles: 4096,
                ..Default::default()
            },
            MeshParams {
                wall_height: 0.0,
                uv_scale: 0.0,
                ..Default::default()
            },
        ] {
            let chunks = mesh_chunks(&grid, &p);
            assert!(!chunks.is_empty());
            let covered = covered_wall_edges(&grid, &chunks);
            assert_eq!(
                covered.iter().copied().collect::<BTreeSet<_>>(),
                expected_wall_edges(&grid),
                "chunking must not change what gets meshed"
            );
        }
    }

    #[test]
    fn chunk_size_changes_the_split_but_not_the_surface_area() {
        let grid = generate(42, &DungeonParams::default());
        let area = |p: &MeshParams| -> f32 {
            mesh_chunks(&grid, p)
                .iter()
                .flat_map(|c| c.vertices.chunks_exact(4))
                .map(|q| {
                    let g = cross(sub(q[1].pos, q[0].pos), sub(q[2].pos, q[0].pos));
                    dot(g, g).sqrt()
                })
                .sum()
        };
        let a = area(&MeshParams::default());
        let b = area(&MeshParams {
            chunk_tiles: 4,
            ..Default::default()
        });
        assert!(
            (a - b).abs() / a < 1.0e-4,
            "merged area must be chunk-size invariant"
        );
    }

    // -- doors are open floor in v1 ---------------------------------------------------

    #[test]
    fn door_tiles_mesh_as_open_floor() {
        let grid = generate(42, &DungeonParams::default());
        let doors: Vec<(i32, i32)> = (0..grid.height())
            .flat_map(|z| (0..grid.width()).map(move |x| (x, z)))
            .filter(|&(x, z)| grid.get(x, z) == Tile::Door)
            .collect();
        assert!(!doors.is_empty(), "the fixture seed should produce doors");
        let chunks = mesh_chunks(&grid, &MeshParams::default());
        // A door contributes floor like any walkable tile, and no wall blocks the
        // crossing it opens.
        for (x, z) in doors {
            assert!(grid.is_walkable(x, z));
            for (i, (dx, dz)) in DIRS.iter().enumerate() {
                if grid.is_walkable(x + dx, z + dz) {
                    assert!(
                        !covered_wall_edges(&grid, &chunks).contains(&(x, z, i)),
                        "a wall was meshed across an open doorway"
                    );
                }
            }
        }
    }

    // -- the shipping numbers ---------------------------------------------------------

    #[test]
    fn the_default_dungeon_stays_inside_its_draw_and_triangle_budget() {
        let grid = generate(42, &DungeonParams::default());
        let stats = mesh_stats(&mesh_chunks(&grid, &MeshParams::default()));
        // 40x40 tiles at 16 per chunk is a 3x3 chunk grid; rock-only chunks are dropped.
        assert!(stats.chunks <= 9, "{stats:?}");
        assert!(stats.chunks >= 4, "{stats:?}");
        // A merged dungeon is a few thousand triangles, not a few hundred thousand. The
        // bound is loose on purpose: it catches a merging regression, not a layout tweak.
        assert!(
            stats.triangles < 5_000,
            "triangle count regressed: {stats:?}"
        );
        assert!(stats.vertices < 10_000, "vertex count regressed: {stats:?}");
        assert_eq!(stats.vertices % 4, 0);
    }
}
