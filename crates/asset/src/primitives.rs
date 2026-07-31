//! Procedural mesh builders — fallbacks and demo geometry (cube, quad, box, Cornell
//! box, UV sphere). RHI-agnostic CPU `MeshData`, like the glTF path.

use crate::{Material, MeshData, MeshVertex};

/// A unit cube centered at the origin with per-face normals and UVs. Fallback
/// when no glTF file is available.
pub fn unit_cube() -> MeshData {
    // The 4 corner positions of each face, listed clockwise when viewed along the
    // outward normal; the index emission below flips them to CCW triangles so the
    // geometric (winding) normal agrees with the shading normal. Single-sided
    // consumers (the vgeo producer's per-triangle backface cull) depend on that
    // agreement — the fixed-function raster path never notices (culling is NONE).
    type Quad = ([f32; 3], [f32; 3], [f32; 3], [f32; 3]);
    const FACES: [Quad; 6] = [
        // +X
        (
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ),
        // -X
        (
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ),
        // +Y
        (
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ),
        // -Y
        (
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ),
        // +Z
        (
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ),
        // -Z
        (
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ),
    ];
    const NORMALS: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    const UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (f, face) in FACES.iter().enumerate() {
        let base = vertices.len() as u32;
        let corners = [face.0, face.1, face.2, face.3];
        for (c, pos) in corners.iter().enumerate() {
            vertices.push(MeshVertex {
                pos: *pos,
                normal: NORMALS[f],
                uv: UVS[c],
            });
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    MeshData {
        vertices,
        indices,
        material: Material::default(),
    }
}

/// A single quad (two CCW triangles) from four corner positions and a normal.
fn quad(p: [[f32; 3]; 4], normal: [f32; 3]) -> MeshData {
    let v = |pos: [f32; 3], uv: [f32; 2]| MeshVertex { pos, normal, uv };
    MeshData {
        vertices: vec![
            v(p[0], [0.0, 0.0]),
            v(p[1], [1.0, 0.0]),
            v(p[2], [1.0, 1.0]),
            v(p[3], [0.0, 1.0]),
        ],
        indices: vec![0, 2, 1, 0, 3, 2],
        material: Material::default(),
    }
}

/// An axis-aligned box from `min` to `max` (inward-agnostic; the path tracer is
/// two-sided), as one mesh with 6 quad faces.
fn axis_box(min: [f32; 3], max: [f32; 3]) -> MeshData {
    let faces = [
        // +X / -X
        (
            [
                [max[0], min[1], min[2]],
                [max[0], min[1], max[2]],
                [max[0], max[1], max[2]],
                [max[0], max[1], min[2]],
            ],
            [1.0, 0.0, 0.0],
        ),
        (
            [
                [min[0], min[1], max[2]],
                [min[0], min[1], min[2]],
                [min[0], max[1], min[2]],
                [min[0], max[1], max[2]],
            ],
            [-1.0, 0.0, 0.0],
        ),
        // +Y / -Y
        (
            [
                [min[0], max[1], min[2]],
                [max[0], max[1], min[2]],
                [max[0], max[1], max[2]],
                [min[0], max[1], max[2]],
            ],
            [0.0, 1.0, 0.0],
        ),
        (
            [
                [min[0], min[1], max[2]],
                [max[0], min[1], max[2]],
                [max[0], min[1], min[2]],
                [min[0], min[1], min[2]],
            ],
            [0.0, -1.0, 0.0],
        ),
        // +Z / -Z
        (
            [
                [max[0], min[1], max[2]],
                [min[0], min[1], max[2]],
                [min[0], max[1], max[2]],
                [max[0], max[1], max[2]],
            ],
            [0.0, 0.0, 1.0],
        ),
        (
            [
                [min[0], min[1], min[2]],
                [max[0], min[1], min[2]],
                [max[0], max[1], min[2]],
                [min[0], max[1], min[2]],
            ],
            [0.0, 0.0, -1.0],
        ),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (corners, n) in faces {
        let base = vertices.len() as u32;
        for (i, c) in corners.iter().enumerate() {
            vertices.push(MeshVertex {
                pos: *c,
                normal: n,
                uv: if i == 1 || i == 2 {
                    [1.0, 0.0]
                } else {
                    [0.0, 0.0]
                },
            });
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    MeshData {
        vertices,
        indices,
        material: Material::default(),
    }
}

/// A Cornell box for the path-tracer GI demo: a white floor/ceiling/back, a red
/// left wall, a green right wall, a large emissive ceiling light, and two white
/// boxes inside. The interior spans x,z in [-1, 1] and y in [0, 2]; geometry is
/// in world space (instances use the identity transform). Each entry pairs a mesh
/// with `[albedo_r, albedo_g, albedo_b, emissive_scale]` for the instance table.
pub fn cornell_box() -> Vec<(MeshData, [f32; 4])> {
    let white = [0.73, 0.73, 0.73, 0.0];
    let red = [0.65, 0.05, 0.05, 0.0];
    let green = [0.12, 0.45, 0.15, 0.0];
    // Large area light: bright emissive white covering most of the ceiling (keeps
    // variance low without next-event estimation).
    let light = [1.0, 0.95, 0.85, 12.0];
    vec![
        // Floor (y=0) and ceiling (y=2).
        (
            quad(
                [
                    [-1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                [0.0, 1.0, 0.0],
            ),
            white,
        ),
        (
            quad(
                [
                    [-1.0, 2.0, -1.0],
                    [-1.0, 2.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 2.0, -1.0],
                ],
                [0.0, -1.0, 0.0],
            ),
            white,
        ),
        // Back wall (z=-1).
        (
            quad(
                [
                    [-1.0, 0.0, -1.0],
                    [-1.0, 2.0, -1.0],
                    [1.0, 2.0, -1.0],
                    [1.0, 0.0, -1.0],
                ],
                [0.0, 0.0, 1.0],
            ),
            white,
        ),
        // Left wall red (x=-1), right wall green (x=1).
        (
            quad(
                [
                    [-1.0, 0.0, 1.0],
                    [-1.0, 2.0, 1.0],
                    [-1.0, 2.0, -1.0],
                    [-1.0, 0.0, -1.0],
                ],
                [1.0, 0.0, 0.0],
            ),
            red,
        ),
        (
            quad(
                [
                    [1.0, 0.0, -1.0],
                    [1.0, 2.0, -1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0],
                ],
                [-1.0, 0.0, 0.0],
            ),
            green,
        ),
        // Emissive ceiling light (just below the ceiling).
        (
            quad(
                [
                    [-0.5, 1.98, -0.5],
                    [-0.5, 1.98, 0.5],
                    [0.5, 1.98, 0.5],
                    [0.5, 1.98, -0.5],
                ],
                [0.0, -1.0, 0.0],
            ),
            light,
        ),
        // Tall box (back-left) and short box (front-right).
        (axis_box([-0.55, 0.0, -0.55], [-0.05, 1.2, -0.05]), white),
        (axis_box([0.1, 0.0, 0.1], [0.6, 0.6, 0.6]), white),
    ]
}

/// A unit UV sphere centered at the origin (radius 1) with smooth outward
/// normals. Good for showing off PBR / image-based reflections.
pub fn uv_sphere(segments: u32, rings: u32) -> MeshData {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let mut vertices = Vec::with_capacity(((segments + 1) * (rings + 1)) as usize);
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI; // 0 (top) .. PI (bottom)
        let (sin_phi, cos_phi) = phi.sin_cos();
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            // Radius 1, so the position doubles as the outward normal.
            let pos = [sin_phi * cos_theta, cos_phi, sin_phi * sin_theta];
            vertices.push(MeshVertex {
                pos,
                normal: pos,
                uv: [u, v],
            });
        }
    }
    let stride = segments + 1;
    let mut indices = Vec::with_capacity((segments * rings * 6) as usize);
    for r in 0..rings {
        for s in 0..segments {
            let a = r * stride + s;
            let b = a + stride;
            // CCW about the outward (radial) normal: with theta advancing +X->+Z and
            // phi advancing top->bottom, (east x south) points outward.
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    MeshData {
        vertices,
        indices,
        material: Material::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every triangle's geometric (winding) normal must agree with its vertices'
    /// shading normal. The fixed-function raster path never notices a violation
    /// (culling is NONE on all backends), but single-sided consumers — the vgeo
    /// producer's per-triangle backface cull — drop miswound triangles entirely.
    fn assert_winding_agrees(mesh: &MeshData, what: &str) {
        for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let p = |i: usize| mesh.vertices[i].pos;
            let (pa, pb, pc) = (p(a), p(b), p(c));
            let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            // Average the three shading normals (curved meshes vary per vertex).
            let n = [
                mesh.vertices[a].normal[0]
                    + mesh.vertices[b].normal[0]
                    + mesh.vertices[c].normal[0],
                mesh.vertices[a].normal[1]
                    + mesh.vertices[b].normal[1]
                    + mesh.vertices[c].normal[1],
                mesh.vertices[a].normal[2]
                    + mesh.vertices[b].normal[2]
                    + mesh.vertices[c].normal[2],
            ];
            // Degenerate triangles (the collapsed pole rings of `uv_sphere`) have no
            // winding to check.
            let area_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
            if area_sq <= 1e-12 {
                continue;
            }
            let dot = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
            assert!(
                dot > 0.0,
                "{what}: triangle {t} winds against its shading normal (dot {dot})"
            );
        }
    }

    #[test]
    fn unit_cube_winding_agrees_with_normals() {
        assert_winding_agrees(&unit_cube(), "unit_cube");
    }

    #[test]
    fn uv_sphere_winding_agrees_with_normals() {
        assert_winding_agrees(&uv_sphere(24, 16), "uv_sphere");
    }

    /// Covers `quad` and `axis_box` through every Cornell call site.
    #[test]
    fn cornell_box_winding_agrees_with_normals() {
        for (i, (mesh, _)) in cornell_box().iter().enumerate() {
            assert_winding_agrees(mesh, &format!("cornell_box[{i}]"));
        }
    }
}
