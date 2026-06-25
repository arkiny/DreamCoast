//! Pure push-constant packers + small camera/matrix helpers extracted from `main.rs`.
//!
//! Each `*_push` function lays out a shader's push-constant block byte-for-byte
//! (little-endian) for the corresponding pipeline; the rest are leaf math helpers.
//! All are pure (no GPU/RHI state), so they live apart from the render loop.

use dreamcoast_core::glam::{Mat4, Vec3, Vec4};

use crate::normalize3;

/// The 6 cube-face view-projections from `eye` (90° FOV, aspect 1), matching the
/// `TextureCube` face convention. The Vulkan clip-space Y flip keeps the captured
/// faces oriented the same as the procedural sky on both backends.
pub(crate) fn cube_face_view_proj(eye: Vec3, vulkan: bool) -> [Mat4; 6] {
    let dirs = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let ups = [-Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z, -Vec3::Y, -Vec3::Y];
    let mut proj = Mat4::perspective_rh(90f32.to_radians(), 1.0, 0.05, 100.0);
    if vulkan {
        proj.y_axis.y *= -1.0;
    }
    let mut out = [Mat4::IDENTITY; 6];
    for i in 0..6 {
        let view = Mat4::look_at_rh(eye, eye + dirs[i], ups[i]);
        out[i] = proj * view;
    }
    out
}

/// Pack the capture push block (208 bytes). Layout: mvp(64), model(64),
/// base_color(16), sun(16 — xyz dir, w intensity), misc(16 — x ambient,
/// y roughness, z metallic, w prefilter max LOD), eye(16 — xyz), ibl(16 — int4
/// irradiance/prefilter/BRDF indices, -1 = no previous environment).
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_push(
    mvp: [f32; 16],
    model: [f32; 16],
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    ambient: f32,
    eye: Vec3,
    prefilter_max_lod: f32,
    ibl: [i32; 3],
) -> [u8; 208] {
    let mut pc = [0u8; 208];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in model.iter().enumerate() {
        let o = 64 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in base_color.iter().enumerate() {
        let o = 128 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    let n = normalize3(sun_dir);
    for (i, f) in n.iter().take(3).enumerate() {
        let o = 144 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[156..160].copy_from_slice(&sun_intensity.to_le_bytes());
    // misc: x ambient, y roughness, z metallic, w prefilter max LOD.
    pc[160..164].copy_from_slice(&ambient.to_le_bytes());
    pc[164..168].copy_from_slice(&roughness.to_le_bytes());
    pc[168..172].copy_from_slice(&metallic.to_le_bytes());
    pc[172..176].copy_from_slice(&prefilter_max_lod.to_le_bytes());
    // eye: xyz capture/camera position.
    pc[176..180].copy_from_slice(&eye.x.to_le_bytes());
    pc[180..184].copy_from_slice(&eye.y.to_le_bytes());
    pc[184..188].copy_from_slice(&eye.z.to_le_bytes());
    // ibl: int4 previous-frame irradiance / prefilter / BRDF indices.
    pc[192..196].copy_from_slice(&ibl[0].to_le_bytes());
    pc[196..200].copy_from_slice(&ibl[1].to_le_bytes());
    pc[200..204].copy_from_slice(&ibl[2].to_le_bytes());
    pc
}

/// Pack the sky push block: sun float4 (xyz dir, w intensity) + face + flip_y +
/// pad (32 bytes).
pub(crate) fn sky_push(sun_dir: [f32; 3], intensity: f32, face: u32, flip_y: u32) -> [u8; 32] {
    let n = normalize3(sun_dir);
    let mut pc = [0u8; 32];
    for (i, v) in n.iter().take(3).enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[12..16].copy_from_slice(&intensity.to_le_bytes());
    pc[16..20].copy_from_slice(&face.to_le_bytes());
    pc[20..24].copy_from_slice(&flip_y.to_le_bytes());
    pc
}

/// Pack the irradiance push block: face + flip_y + env_index + pad (16 bytes).
pub(crate) fn cube_gen_push(face: u32, flip_y: u32, env_index: u32, roughness: f32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&face.to_le_bytes());
    pc[4..8].copy_from_slice(&flip_y.to_le_bytes());
    pc[8..12].copy_from_slice(&env_index.to_le_bytes());
    pc[12..16].copy_from_slice(&roughness.to_le_bytes());
    pc
}

/// Pack the prefilter push block: face + flip_y + env_index + roughness +
/// env_mips (20 bytes — env_mips drives the mip-based importance sampling).
pub(crate) fn prefilter_push(
    face: u32,
    flip_y: u32,
    env_index: u32,
    roughness: f32,
    env_mips: u32,
) -> [u8; 20] {
    let mut pc = [0u8; 20];
    pc[0..4].copy_from_slice(&face.to_le_bytes());
    pc[4..8].copy_from_slice(&flip_y.to_le_bytes());
    pc[8..12].copy_from_slice(&env_index.to_le_bytes());
    pc[12..16].copy_from_slice(&roughness.to_le_bytes());
    pc[16..20].copy_from_slice(&env_mips.to_le_bytes());
    pc
}

/// Pack the tonemap push block: hdr_index + mode + flip_y + pad (16 bytes).
pub(crate) fn post_push(hdr_index: u32, mode: u32, flip_y: u32, exposure: f32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&mode.to_le_bytes());
    pc[8..12].copy_from_slice(&flip_y.to_le_bytes());
    pc[12..16].copy_from_slice(&exposure.to_le_bytes());
    pc
}

/// Pack the particle-sim push block: buffer_index + count + dt + time + init.
pub(crate) fn particle_sim_push(
    read_index: u32,
    write_index: u32,
    count: u32,
    dt: f32,
    time: f32,
    init: u32,
) -> [u8; 24] {
    let mut pc = [0u8; 24];
    pc[0..4].copy_from_slice(&read_index.to_le_bytes());
    pc[4..8].copy_from_slice(&write_index.to_le_bytes());
    pc[8..12].copy_from_slice(&count.to_le_bytes());
    pc[12..16].copy_from_slice(&dt.to_le_bytes());
    pc[16..20].copy_from_slice(&time.to_le_bytes());
    pc[20..24].copy_from_slice(&init.to_le_bytes());
    pc
}

/// Pack the particle-draw push block: view_proj(64) + cam_right(16) + cam_up(16)
/// + buffer_index + count + size + pad (16) = 112 bytes.
pub(crate) fn particle_draw_push(
    view_proj: &[f32; 16],
    cam_right: Vec3,
    cam_up: Vec3,
    buffer_index: u32,
    count: u32,
    size: f32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, v) in view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_right.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_right.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_right.z.to_le_bytes());
    pc[80..84].copy_from_slice(&cam_up.x.to_le_bytes());
    pc[84..88].copy_from_slice(&cam_up.y.to_le_bytes());
    pc[88..92].copy_from_slice(&cam_up.z.to_le_bytes());
    pc[96..100].copy_from_slice(&buffer_index.to_le_bytes());
    pc[100..104].copy_from_slice(&count.to_le_bytes());
    pc[104..108].copy_from_slice(&size.to_le_bytes());
    pc
}

/// Extract the six normalized, inward-facing frustum planes from a view-proj
/// matrix (Gribb-Hartmann; near plane uses row2 for [0,1] clip depth). Use a
/// Y-flip-free matrix so culling is identical on both backends.
pub(crate) fn frustum_planes(vp: Mat4) -> [[f32; 4]; 6] {
    let r0 = Vec4::new(vp.x_axis.x, vp.y_axis.x, vp.z_axis.x, vp.w_axis.x);
    let r1 = Vec4::new(vp.x_axis.y, vp.y_axis.y, vp.z_axis.y, vp.w_axis.y);
    let r2 = Vec4::new(vp.x_axis.z, vp.y_axis.z, vp.z_axis.z, vp.w_axis.z);
    let r3 = Vec4::new(vp.x_axis.w, vp.y_axis.w, vp.z_axis.w, vp.w_axis.w);
    let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
    let mut out = [[0.0f32; 4]; 6];
    for (i, p) in raw.iter().enumerate() {
        let len = p.truncate().length().max(1e-6);
        let n = *p / len;
        out[i] = [n.x, n.y, n.z, n.w];
    }
    out
}

/// Pack the cull push block (128 bytes): 6 planes + buffer indices + grid params.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cull_push(
    planes: &[[f32; 4]; 6],
    args_index: u32,
    visible_index: u32,
    count: u32,
    grid_dim: u32,
    spacing: f32,
    cube_radius: f32,
    y_height: f32,
    index_count: u32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    for (i, pl) in planes.iter().enumerate() {
        for (j, v) in pl.iter().enumerate() {
            let o = i * 16 + j * 4;
            pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    pc[96..100].copy_from_slice(&args_index.to_le_bytes());
    pc[100..104].copy_from_slice(&visible_index.to_le_bytes());
    pc[104..108].copy_from_slice(&count.to_le_bytes());
    pc[108..112].copy_from_slice(&grid_dim.to_le_bytes());
    pc[112..116].copy_from_slice(&spacing.to_le_bytes());
    pc[116..120].copy_from_slice(&cube_radius.to_le_bytes());
    pc[120..124].copy_from_slice(&y_height.to_le_bytes());
    pc[124..128].copy_from_slice(&index_count.to_le_bytes());
    pc
}

/// Pack the cull-draw push block (112 bytes): view_proj + sun_dir + grid params.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cull_draw_push(
    view_proj: &[f32; 16],
    sun_dir: [f32; 3],
    visible_index: u32,
    grid_dim: u32,
    spacing: f32,
    cube_scale: f32,
    y_height: f32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, v) in view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&sun_dir[0].to_le_bytes());
    pc[68..72].copy_from_slice(&sun_dir[1].to_le_bytes());
    pc[72..76].copy_from_slice(&sun_dir[2].to_le_bytes());
    pc[80..84].copy_from_slice(&visible_index.to_le_bytes());
    pc[84..88].copy_from_slice(&grid_dim.to_le_bytes());
    pc[88..92].copy_from_slice(&spacing.to_le_bytes());
    pc[92..96].copy_from_slice(&cube_scale.to_le_bytes());
    pc[96..100].copy_from_slice(&y_height.to_le_bytes());
    pc
}

/// Pack the inline ray-query trace push block (Phase 8 M3): inv_view_proj (64) +
/// cam_pos (16, xyz) + sun_dir (16, xyz) + out_index/width/height/flip_y (16).
pub(crate) fn rt_trace_push(
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    sun_dir: [f32; 3],
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_pos.z.to_le_bytes());
    let sun = normalize3(sun_dir);
    pc[80..84].copy_from_slice(&sun[0].to_le_bytes());
    pc[84..88].copy_from_slice(&sun[1].to_le_bytes());
    pc[88..92].copy_from_slice(&sun[2].to_le_bytes());
    pc[96..100].copy_from_slice(&out_index.to_le_bytes());
    pc[100..104].copy_from_slice(&width.to_le_bytes());
    pc[104..108].copy_from_slice(&height.to_le_bytes());
    pc[108..112].copy_from_slice(&flip_y.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage A SDF-trace push block (112 bytes): inv_view_proj (64) +
/// cam_pos (16) + sun dir+intensity (16) + (out_index, width, height, flip_y) (16).
/// Same layout as `rt_trace_push` but `sun.w` carries the sun intensity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdf_trace_push(
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_pos.z.to_le_bytes());
    let sun = normalize3(sun_dir);
    pc[80..84].copy_from_slice(&sun[0].to_le_bytes());
    pc[84..88].copy_from_slice(&sun[1].to_le_bytes());
    pc[88..92].copy_from_slice(&sun[2].to_le_bytes());
    pc[92..96].copy_from_slice(&sun_intensity.to_le_bytes());
    pc[96..100].copy_from_slice(&out_index.to_le_bytes());
    pc[100..104].copy_from_slice(&width.to_le_bytes());
    pc[104..108].copy_from_slice(&height.to_le_bytes());
    pc[108..112].copy_from_slice(&flip_y.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage B volume-test push block (32 bytes): vol_storage,
/// vol_sampled, dim, out_index, width, height, slice (f32), pad.
#[allow(clippy::too_many_arguments)]
pub(crate) fn volume_push(
    vol_storage: u32,
    vol_sampled: u32,
    dim: u32,
    out_index: u32,
    width: u32,
    height: u32,
    slice: f32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    pc[0..4].copy_from_slice(&vol_storage.to_le_bytes());
    pc[4..8].copy_from_slice(&vol_sampled.to_le_bytes());
    pc[8..12].copy_from_slice(&dim.to_le_bytes());
    pc[12..16].copy_from_slice(&out_index.to_le_bytes());
    pc[16..20].copy_from_slice(&width.to_le_bytes());
    pc[20..24].copy_from_slice(&height.to_le_bytes());
    pc[24..28].copy_from_slice(&slice.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage B2 SDF-bake push block (64 bytes): vol_storage, dim,
/// tri_count, vtx_index, idx_index, pad0, then float4 aabb_min / aabb_max (16-byte
/// aligned, so 8 bytes of padding precede them). The volume's AABB is the unit cube
/// — matching B1's analytic fill so the baked sphere is pixel-comparable.
pub(crate) fn sdf_bake_push(
    vol_storage: u32,
    dim: u32,
    tri_count: u32,
    vtx_index: u32,
    idx_index: u32,
) -> [u8; 64] {
    let mut pc = [0u8; 64];
    pc[0..4].copy_from_slice(&vol_storage.to_le_bytes());
    pc[4..8].copy_from_slice(&dim.to_le_bytes());
    pc[8..12].copy_from_slice(&tri_count.to_le_bytes());
    pc[12..16].copy_from_slice(&vtx_index.to_le_bytes());
    pc[16..20].copy_from_slice(&idx_index.to_le_bytes());
    // pc[20..32]: pad0 + alignment padding to the float4 boundary.
    // aabb_min = (0,0,0,0), aabb_max = (1,1,1,0): the unit-cube volume extent.
    pc[32..36].copy_from_slice(&0.0f32.to_le_bytes());
    pc[36..40].copy_from_slice(&0.0f32.to_le_bytes());
    pc[40..44].copy_from_slice(&0.0f32.to_le_bytes());
    pc[48..52].copy_from_slice(&1.0f32.to_le_bytes());
    pc[52..56].copy_from_slice(&1.0f32.to_le_bytes());
    pc[56..60].copy_from_slice(&1.0f32.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage B3 GDF-merge push block (48 bytes): gdf_storage, dim,
/// inst_table, inst_count, then float4 aabb_min / aabb_max (the GDF world extent;
/// the unit cube here, matching the per-mesh bake box so a whole-cube single
/// instance reproduces the B2 bake exactly).
pub(crate) fn gdf_merge_push(
    gdf_storage: u32,
    dim: u32,
    inst_table: u32,
    inst_count: u32,
) -> [u8; 48] {
    let mut pc = [0u8; 48];
    pc[0..4].copy_from_slice(&gdf_storage.to_le_bytes());
    pc[4..8].copy_from_slice(&dim.to_le_bytes());
    pc[8..12].copy_from_slice(&inst_table.to_le_bytes());
    pc[12..16].copy_from_slice(&inst_count.to_le_bytes());
    // aabb_min = (0,0,0,0): the float4 is 16-byte aligned at offset 16.
    // aabb_max = (1,1,1,0): the unit-cube GDF extent.
    pc[32..36].copy_from_slice(&1.0f32.to_le_bytes());
    pc[36..40].copy_from_slice(&1.0f32.to_le_bytes());
    pc[40..44].copy_from_slice(&1.0f32.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage B4 GDF-trace push block (128 bytes): inv_view_proj (64) +
/// cam_pos (16) + sun dir+intensity (16) + (out, width, height, flip_y) (16) +
/// (gdf_sampled, mode, pad, pad) (16). `mode` bit0 swaps the GDF sample for the
/// analytic reference field. Same head layout as `sdf_trace_push`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_trace_push(
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    gdf_sampled: u32,
    mode: u32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_pos.z.to_le_bytes());
    let sun = normalize3(sun_dir);
    pc[80..84].copy_from_slice(&sun[0].to_le_bytes());
    pc[84..88].copy_from_slice(&sun[1].to_le_bytes());
    pc[88..92].copy_from_slice(&sun[2].to_le_bytes());
    pc[92..96].copy_from_slice(&sun_intensity.to_le_bytes());
    pc[96..100].copy_from_slice(&out_index.to_le_bytes());
    pc[100..104].copy_from_slice(&width.to_le_bytes());
    pc[104..108].copy_from_slice(&height.to_le_bytes());
    pc[108..112].copy_from_slice(&flip_y.to_le_bytes());
    pc[112..116].copy_from_slice(&gdf_sampled.to_le_bytes());
    pc[116..120].copy_from_slice(&mode.to_le_bytes());
    pc
}

/// Pack the path-tracer push block (Phase 8 M4, 128 bytes): inv_view_proj (64) +
/// cam_pos (16) + sun dir+intensity (16) + (out, accum, inst, frame) (16) +
/// (width, height, flip_y, spp) (16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rt_path_push(
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    out_index: u32,
    accum_index: u32,
    inst_index: u32,
    inst_count: u32,
    frame: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    spp: u32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_pos.z.to_le_bytes());
    // cam_pos.w carries the instance count for the closest-hit bounds check.
    pc[76..80].copy_from_slice(&(inst_count as f32).to_le_bytes());
    let sun = normalize3(sun_dir);
    pc[80..84].copy_from_slice(&sun[0].to_le_bytes());
    pc[84..88].copy_from_slice(&sun[1].to_le_bytes());
    pc[88..92].copy_from_slice(&sun[2].to_le_bytes());
    pc[92..96].copy_from_slice(&sun_intensity.to_le_bytes());
    pc[96..100].copy_from_slice(&out_index.to_le_bytes());
    pc[100..104].copy_from_slice(&accum_index.to_le_bytes());
    pc[104..108].copy_from_slice(&inst_index.to_le_bytes());
    pc[108..112].copy_from_slice(&frame.to_le_bytes());
    pc[112..116].copy_from_slice(&width.to_le_bytes());
    pc[116..120].copy_from_slice(&height.to_le_bytes());
    pc[120..124].copy_from_slice(&flip_y.to_le_bytes());
    pc[124..128].copy_from_slice(&spp.to_le_bytes());
    pc
}

/// Pack the compute-post push block: hdr_index + out_index + width + height.
pub(crate) fn post_compute_push(
    hdr_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&out_index.to_le_bytes());
    pc[8..12].copy_from_slice(&width.to_le_bytes());
    pc[12..16].copy_from_slice(&height.to_le_bytes());
    pc
}

/// Convert a column-major glam [`Mat4`] object-to-world transform into the
/// row-major 3x4 (12-float) form acceleration-structure instances expect (Phase 8).
pub(crate) fn mat4_to_3x4(m: Mat4) -> [f32; 12] {
    let c = m.to_cols_array(); // column-major: [c0(0..4), c1(4..8), c2(8..12), c3(12..16)]
    [
        c[0], c[4], c[8], c[12], // row 0
        c[1], c[5], c[9], c[13], // row 1
        c[2], c[6], c[10], c[14], // row 2
    ]
}
