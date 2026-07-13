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
/// sky_gain + pad, then the sky white-balance float4 (xyz gain, w unused) — 48 bytes.
/// `wb = [1, 1, 1]` is a neutral no-op (the shader's `col *= wb` is exact ×1).
pub(crate) fn sky_push(
    sun_dir: [f32; 3],
    intensity: f32,
    face: u32,
    flip_y: u32,
    sky_gain: f32,
    wb: [f32; 3],
) -> [u8; 48] {
    let n = normalize3(sun_dir);
    let mut pc = [0u8; 48];
    for (i, v) in n.iter().take(3).enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[12..16].copy_from_slice(&intensity.to_le_bytes());
    pc[16..20].copy_from_slice(&face.to_le_bytes());
    pc[20..24].copy_from_slice(&flip_y.to_le_bytes());
    pc[24..28].copy_from_slice(&sky_gain.to_le_bytes());
    // wb at offset 32 (float4, 16-byte aligned to match the HLSL cbuffer layout).
    for (i, v) in wb.iter().take(3).enumerate() {
        pc[32 + i * 4..32 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the screen-space AO (gtao.slang) push block (144 bytes): inv_view_proj + camera_pos +
/// the sampled/storage indices + dims + the two param vectors. `dir_index`/`in_index` are only
/// read by the blur entry; the AO entry ignores them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gtao_push(
    inv_view_proj: &[f32; 16],
    camera_pos: [f32; 3],
    depth_index: u32,
    normal_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    dir_index: u32,
    in_index: u32,
    radius: f32,
    intensity: f32,
    bias: f32,
    proj_scale: f32,
    aspect: f32,
    power: f32,
    blur_sigma: f32,
) -> [u8; 144] {
    let mut pc = [0u8; 144];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&camera_pos[0].to_le_bytes());
    pc[68..72].copy_from_slice(&camera_pos[1].to_le_bytes());
    pc[72..76].copy_from_slice(&camera_pos[2].to_le_bytes());
    pc[80..84].copy_from_slice(&depth_index.to_le_bytes());
    pc[84..88].copy_from_slice(&normal_index.to_le_bytes());
    pc[88..92].copy_from_slice(&out_index.to_le_bytes());
    pc[92..96].copy_from_slice(&width.to_le_bytes());
    pc[96..100].copy_from_slice(&height.to_le_bytes());
    pc[100..104].copy_from_slice(&flip_y.to_le_bytes());
    pc[104..108].copy_from_slice(&dir_index.to_le_bytes());
    pc[108..112].copy_from_slice(&in_index.to_le_bytes());
    pc[112..116].copy_from_slice(&radius.to_le_bytes());
    pc[116..120].copy_from_slice(&intensity.to_le_bytes());
    pc[120..124].copy_from_slice(&bias.to_le_bytes());
    pc[124..128].copy_from_slice(&proj_scale.to_le_bytes());
    pc[128..132].copy_from_slice(&aspect.to_le_bytes());
    pc[132..136].copy_from_slice(&power.to_le_bytes());
    pc[136..140].copy_from_slice(&blur_sigma.to_le_bytes());
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

/// Neutral (identity) ASC-CDL color grade: slope 1, offset 0, power 1. Passing this
/// with `grade_on = 0` (or these exact values) is a byte-identical no-op — the anchor.
pub(crate) const CDL_NEUTRAL: ([f32; 3], [f32; 3], [f32; 3]) =
    ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

/// Pack the tonemap push block (112 bytes): hdr_index + mode + flip_y + exposure (16) +
/// sharpen + inv_w + inv_h + bloom_index (16) + bloom_intensity + grade_on + exposure_buf +
/// pad (16) + cdl_slope float4 (16) + cdl_offset float4 (16) + cdl_power float4 (16) +
/// lut_index + lut_size + pad + pad (16). `exposure_buf == u32::MAX` uses the constant
/// `exposure` (byte-identical anchor); otherwise the adapted auto-exposure is read from it.
///
/// PR-5 added the bloom composite slot + the ASC-CDL color-grading hook. `bloom_index ==
/// u32::MAX` skips the bloom add; `grade_on == 0` skips grading. Each CDL vector is a
/// full float4 row (never float3 + trailing scalar) to match the HLSL/SPIR-V vs. MSL
/// push-constant packing. Neutral params (`CDL_NEUTRAL`) + `grade_on = 0` + no bloom =
/// the byte-identical anchor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn post_push(
    hdr_index: u32,
    mode: u32,
    flip_y: u32,
    exposure: f32,
    sharpen: f32,
    inv_w: f32,
    inv_h: f32,
    bloom_index: u32,
    bloom_intensity: f32,
    grade_on: u32,
    exposure_buf: u32,
    cdl_slope: [f32; 3],
    cdl_offset: [f32; 3],
    cdl_power: [f32; 3],
    lut_index: u32,
    lut_size: u32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&mode.to_le_bytes());
    pc[8..12].copy_from_slice(&flip_y.to_le_bytes());
    pc[12..16].copy_from_slice(&exposure.to_le_bytes());
    pc[16..20].copy_from_slice(&sharpen.to_le_bytes());
    pc[20..24].copy_from_slice(&inv_w.to_le_bytes());
    pc[24..28].copy_from_slice(&inv_h.to_le_bytes());
    pc[28..32].copy_from_slice(&bloom_index.to_le_bytes());
    pc[32..36].copy_from_slice(&bloom_intensity.to_le_bytes());
    pc[36..40].copy_from_slice(&grade_on.to_le_bytes());
    // pc[40..44] = exposure_buf (auto-exposure buffer bindless index; MAX = use constant);
    // pc[44..48] = pad1 to the float4 boundary.
    pc[40..44].copy_from_slice(&exposure_buf.to_le_bytes());
    for (i, v) in cdl_slope.iter().enumerate() {
        pc[48 + i * 4..52 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in cdl_offset.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in cdl_power.iter().enumerate() {
        pc[80 + i * 4..84 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // Baked tonemap-LUT row: index (u32::MAX = off -> the legacy per-pixel curve, the
    // byte-identical anchor) + LUT resolution N as a float (the shader's texel math).
    pc[96..100].copy_from_slice(&lut_index.to_le_bytes());
    pc[100..104].copy_from_slice(&(lut_size as f32).to_le_bytes());
    pc
}

/// Pack the tonemap-LUT bake push block (64 bytes): (out_index, size, grade_on, pad) uints +
/// the three ASC-CDL float4 rows (slope/offset/power — full rows, never float3+scalar, per the
/// HLSL/SPIR-V vs MSL packing note on `post_push`). See `tonemap_lut.slang`.
pub(crate) fn tonemap_lut_push(
    out_index: u32,
    size: u32,
    grade_on: u32,
    cdl_slope: [f32; 3],
    cdl_offset: [f32; 3],
    cdl_power: [f32; 3],
) -> [u8; 64] {
    let mut pc = [0u8; 64];
    pc[0..4].copy_from_slice(&out_index.to_le_bytes());
    pc[4..8].copy_from_slice(&size.to_le_bytes());
    pc[8..12].copy_from_slice(&grade_on.to_le_bytes());
    for (i, v) in cdl_slope.iter().enumerate() {
        pc[16 + i * 4..20 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in cdl_offset.iter().enumerate() {
        pc[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in cdl_power.iter().enumerate() {
        pc[48 + i * 4..52 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the PR-4 atmosphere/height-fog push block (80 bytes): 4 uints (hdr_index,
/// position_index, out_index [unused by the graphics entry], flip_y [unused]) + three
/// float4 rows (camera_pos.xyz + density.w, sun_dir.xyz + sun_intensity.w, sky_wb.xyz +
/// inscatter_gain.w) + a final float4 (height_falloff.x + exposure.y + unused zw). Every
/// row is a full float4 (never a bare float3 followed by a scalar) to dodge the HLSL/
/// SPIR-V vs. MSL push-constant packing divergence documented on `gdf_gi_push`'s
/// `ground_albedo`. `exposure` is the same scalar `record_lighting` bakes into `hdr`
/// (`globals.ambient.a` / the auto-exposure buffer) — `procedural_sky` returns raw
/// unexposed radiance (like the sky-background miss path in `pbr.slang`), so the
/// inscatter color must be exposed the same way before blending, or a physically-scaled
/// sun (tens of thousands of lux) blows the composite out to white.
#[allow(clippy::too_many_arguments)]
pub(crate) fn atmosphere_push(
    hdr_index: u32,
    position_index: u32,
    camera_pos: [f32; 3],
    density: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    sky_wb: [f32; 3],
    inscatter_gain: f32,
    height_falloff: f32,
    exposure: f32,
    flip_y: u32,
) -> [u8; 80] {
    let mut pc = [0u8; 80];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&position_index.to_le_bytes());
    // pc[8..12] = out_index (unused by the graphics entry). pc[12..16] = flip_y drives the
    // full-screen VS clip-space Y orientation (1 = Vulkan) — same as `tonemap_push`; without it
    // the Vulkan fog composite renders vertically flipped (a DX≡VK parity break).
    pc[12..16].copy_from_slice(&flip_y.to_le_bytes());
    for (i, v) in camera_pos.iter().enumerate() {
        pc[16 + i * 4..20 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[28..32].copy_from_slice(&density.to_le_bytes());
    let sun = normalize3(sun_dir);
    for (i, v) in sun.iter().take(3).enumerate() {
        pc[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[44..48].copy_from_slice(&sun_intensity.to_le_bytes());
    for (i, v) in sky_wb.iter().enumerate() {
        pc[48 + i * 4..52 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[60..64].copy_from_slice(&inscatter_gain.to_le_bytes());
    pc[64..68].copy_from_slice(&height_falloff.to_le_bytes());
    pc[68..72].copy_from_slice(&exposure.to_le_bytes());
    pc
}

/// Pack the forward translucency push block (176 bytes, PR-3). Layout: mvp(64), model(64),
/// base_color(16), material(16 = metallic, roughness, base_color tex-index bits, flip_y
/// bits), misc(16 = shadow_index then 3 reserved). `base_tex`/`flip_y`/`shadow_index` are
/// `u32` values stored in the float slots (`material.zw` / `misc.x`) that the shader reads
/// with `asuint` — the reinterpret keeps the block a single 16-byte-aligned float4 grid
/// across all three backends' cbuffer packing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn translucent_push(
    mvp: &[f32; 16],
    model: &[f32; 16],
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    base_tex: u32,
    flip_y: u32,
    shadow_index: u32,
) -> [u8; 176] {
    let mut pc = [0u8; 176];
    for (i, v) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in model.iter().enumerate() {
        pc[64 + i * 4..64 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in base_color.iter().enumerate() {
        pc[128 + i * 4..128 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[144..148].copy_from_slice(&metallic.to_le_bytes());
    pc[148..152].copy_from_slice(&roughness.to_le_bytes());
    pc[152..156].copy_from_slice(&base_tex.to_le_bytes());
    pc[156..160].copy_from_slice(&flip_y.to_le_bytes());
    pc[160..164].copy_from_slice(&shadow_index.to_le_bytes());
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

/// Pack the HZB build push block (32 bytes): source/dest bindless indices + level
/// dims + reduction tap counts (see `hzb_build.slang`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn hzb_build_push(
    src_index: u32,
    dst_index: u32,
    dst_w: u32,
    dst_h: u32,
    src_w: u32,
    src_h: u32,
    tap_x: u32,
    tap_y: u32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    for (i, v) in [
        src_index, dst_index, dst_w, dst_h, src_w, src_h, tap_x, tap_y,
    ]
    .iter()
    .enumerate()
    {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the HZB-aware cull push block (224 bytes): the identical 128-byte frustum
/// block (see `cull_push`) followed by the occlusion block — unjittered no-Y-flip
/// view_proj + HZB metadata (see `csCullHzb` in `cull.slang`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cull_hzb_push(
    planes: &[[f32; 4]; 6],
    args_index: u32,
    visible_index: u32,
    count: u32,
    grid_dim: u32,
    spacing: f32,
    cube_radius: f32,
    y_height: f32,
    index_count: u32,
    view_proj: &[f32; 16],
    hzb_base: u32,
    hzb_levels: u32,
    hzb_w: u32,
    hzb_h: u32,
    enabled: bool,
    stats_index: u32,
) -> [u8; 224] {
    let mut pc = [0u8; 224];
    // Bytes 0..128: identical frustum block.
    let head = cull_push(
        planes,
        args_index,
        visible_index,
        count,
        grid_dim,
        spacing,
        cube_radius,
        y_height,
        index_count,
    );
    pc[..128].copy_from_slice(&head);
    // Bytes 128..192: view_proj (column-major, matches vp[4] float4 columns).
    for (i, v) in view_proj.iter().enumerate() {
        let o = 128 + i * 4;
        pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    // Bytes 192..: HZB metadata.
    pc[192..196].copy_from_slice(&hzb_base.to_le_bytes());
    pc[196..200].copy_from_slice(&hzb_levels.to_le_bytes());
    pc[200..204].copy_from_slice(&hzb_w.to_le_bytes());
    pc[204..208].copy_from_slice(&hzb_h.to_le_bytes());
    pc[208..212].copy_from_slice(&(enabled as u32).to_le_bytes());
    pc[212..216].copy_from_slice(&stats_index.to_le_bytes());
    pc
}

/// Pack the cull-draw push block (112 bytes): view_proj + sun_dir + grid params +
/// the scene-depth manual test (index + display→render pixel scale; see cull_draw.slang).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cull_draw_push(
    view_proj: &[f32; 16],
    sun_dir: [f32; 3],
    visible_index: u32,
    grid_dim: u32,
    spacing: f32,
    cube_scale: f32,
    y_height: f32,
    depth_index: u32,
    depth_scale: [f32; 2],
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
    pc[100..104].copy_from_slice(&depth_index.to_le_bytes());
    pc[104..108].copy_from_slice(&depth_scale[0].to_le_bytes());
    pc[108..112].copy_from_slice(&depth_scale[1].to_le_bytes());
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

/// Pack the Phase 11 SDF-bake push block (64 bytes): vol_storage, dim, tri_count,
/// vtx_index, idx_index, pad0, then float4 aabb_min / aabb_max (16-byte aligned, so 8
/// bytes of padding precede them) — the world-space extent the volume's voxel grid maps
/// to. B2 passes the unit cube (baked sphere pixel-comparable to B1's analytic fill);
/// C1 passes the world scene AABB for the fused scene bake.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdf_bake_push(
    vol_storage: u32,
    dim: u32,
    tri_count: u32,
    vtx_index: u32,
    idx_index: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> [u8; 64] {
    let mut pc = [0u8; 64];
    pc[0..4].copy_from_slice(&vol_storage.to_le_bytes());
    pc[4..8].copy_from_slice(&dim.to_le_bytes());
    pc[8..12].copy_from_slice(&tri_count.to_le_bytes());
    pc[12..16].copy_from_slice(&vtx_index.to_le_bytes());
    pc[16..20].copy_from_slice(&idx_index.to_le_bytes());
    // pc[20..32]: pad0 + alignment padding to the float4 boundary.
    for (i, v) in aabb_min.iter().enumerate() {
        pc[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in aabb_max.iter().enumerate() {
        pc[48 + i * 4..52 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the Phase 11 Stage C8a albedo-bake push block (64 bytes): three storage-volume
/// indices (R/G/B), dim, tri_count, vtx_index, idx_index, per-triangle albedo buffer index,
/// then float4 aabb_min / aabb_max (16-byte aligned). Same scene AABB + voxel grid as the
/// distance bake so the color volumes register identically to the scene GDF.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdf_albedo_bake_push(
    vol_storage_r: u32,
    vol_storage_g: u32,
    vol_storage_b: u32,
    dim: u32,
    tri_count: u32,
    vtx_index: u32,
    idx_index: u32,
    albedo_index: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> [u8; 64] {
    let mut pc = [0u8; 64];
    let u = [
        vol_storage_r,
        vol_storage_g,
        vol_storage_b,
        dim,
        tri_count,
        vtx_index,
        idx_index,
        albedo_index,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in aabb_min.iter().enumerate() {
        pc[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in aabb_max.iter().enumerate() {
        pc[48 + i * 4..52 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the Phase 11 Stage C8b1 surface-cache capture push block (80 bytes): cards buffer,
/// the two output cache buffers (pos / albedo), the GDF sampled index, num_cards / tile /
/// num_texels, the 3 C8a albedo channel indices, then float4 aabb_min / aabb_max.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cache_capture_push(
    cards_index: u32,
    cache_pos_index: u32,
    cache_alb_index: u32,
    gdf_sampled: u32,
    num_cards: u32,
    tile: u32,
    num_texels: u32,
    albedo_rgb: [u32; 3],
    clip_desc: u32,
    clip_count: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    dist_clamp: f32,
    card_albedo_index: u32,
    // C1 mesh-triangle capture: (vtx, idx, table, card_inst) bindless indices; all u32::MAX = off.
    mesh: [u32; 4],
    // F1 Stage 3 streaming re-capture flag buffer (0xFFFFFFFF = off ⇒ trace every card).
    slot_dirty: u32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    let u = [
        cards_index,
        cache_pos_index,
        cache_alb_index,
        gdf_sampled,
        num_cards,
        tile,
        num_texels,
        clip_desc, // Stage B: former pad0
        albedo_rgb[0],
        albedo_rgb[1],
        albedo_rgb[2],
        clip_count, // Stage B: former pad1
    ];
    for (i, v) in u.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in aabb_min.iter().enumerate() {
        pc[48 + i * 4..52 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // C: pack the per-card source-albedo buffer index into the unused `aabb_min.w` slot
    // (bytes 60..64) — no layout/size change (DX≡VK-safe). 0xFFFFFFFF ⇒ legacy volume path.
    pc[60..64].copy_from_slice(&card_albedo_index.to_le_bytes());
    for (i, v) in aabb_max.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[76..80].copy_from_slice(&dist_clamp.to_le_bytes());
    // C1 mesh-triangle capture row (all-sentinel = off ⇒ legacy stamped/volume albedo).
    for (i, v) in mesh.iter().enumerate() {
        pc[80 + i * 4..84 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // F1 Stage 3 streaming re-capture flag (byte 96; the initial capture passes 0xFFFFFFFF).
    pc[96..100].copy_from_slice(&slot_dirty.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C8b1 surface-cache atlas-viz push block (32 bytes). `cache_src`
/// is the buffer shown — the captured albedo (C8b1) or the lit radiance (C8b2).
pub(crate) fn cache_view_push(
    cache_pos_index: u32,
    cache_src_index: u32,
    out_index: u32,
    num_cards: u32,
    tile: u32,
    width: u32,
    height: u32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    let u = [
        cache_pos_index,
        cache_src_index,
        out_index,
        num_cards,
        tile,
        width,
        height,
        0,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the reflection cone-LOD MIP-generation push (48 bytes = 12 u32). One dispatch per level
/// downsamples the surface-cache radiance atlas 2×2 into the MIP pyramid. `src_rad` is mip0 (the
/// freshly-lit slot) at level 1 and the pyramid buffer for higher levels; `src_pos` supplies mip0
/// validity (level 1 only). `parent_stride`/`off_prev` locate the parent within its buffer,
/// `dst_stride`/`off_cur` the destination within the pyramid. `res_cur`/`res_prev` are the level
/// edges. Layout mirrors `MipGenPush` in `sdf_cache_mipgen.slang`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdf_cache_mipgen_push(
    src_rad: u32,
    src_pos: u32,
    dst: u32,
    num_cards: u32,
    level: u32,
    res_cur: u32,
    res_prev: u32,
    parent_stride: u32,
    off_prev: u32,
    dst_stride: u32,
    off_cur: u32,
    // C2a: adaptive layout index + 1 (0 = uniform legacy).
    layout1: u32,
) -> [u8; 48] {
    let mut pc = [0u8; 48];
    let fields = [
        src_rad,
        src_pos,
        dst,
        num_cards,
        level,
        res_cur,
        res_prev,
        parent_stride,
        off_prev,
        dst_stride,
        off_cur,
        layout1,
    ];
    for (i, v) in fields.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the Phase 11 Stage C8b2 surface-cache lighting push block (112 bytes): card +
/// cache buffer indices, GDF sampled, atlas dims, spp/frame/reset, then float4 sun /
/// aabb_min(+ground) / aabb_max(+clamp) / params(sky_fill, temporal alpha, bias, ray max).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cache_light_push(
    cards_index: u32,
    cache_pos_index: u32,
    cache_alb_index: u32,
    cache_rad_read: u32,
    cache_rad_write: u32,
    gdf_sampled: u32,
    num_cards: u32,
    tile: u32,
    num_texels: u32,
    spp: u32,
    frame: u32,
    reset: u32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    aabb_min: [f32; 3],
    ground_y: f32,
    aabb_max: [f32; 3],
    dist_clamp: f32,
    skylight_floor: f32,
    alpha: f32,
    bias: f32,
    ray_max: f32,
    clip_desc: u32,
    clip_count: u32,
    relight_period: u32,
    card_vis_index: u32,
    cone_k: f32,
    conv_buf: u32,
    irradiance_index: u32,
    gather_firefly: f32,
    sky_gain: f32,
    sky_wb: [f32; 3],
    skyvis_index: u32,
    skyvis_tint: f32,
    skyvis_min_occ: f32,
    flags: u32,
    ao_params: (f32, f32, f32, f32),
) -> [u8; 192] {
    let mut pc = [0u8; 192];
    let u = [
        cards_index,
        cache_pos_index,
        cache_alb_index,
        cache_rad_read,
        cache_rad_write,
        gdf_sampled,
        num_cards,
        tile,
        num_texels,
        spp,
        frame,
        reset,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let sun = normalize3(sun_dir);
    pc[48..52].copy_from_slice(&sun[0].to_le_bytes());
    pc[52..56].copy_from_slice(&sun[1].to_le_bytes());
    pc[56..60].copy_from_slice(&sun[2].to_le_bytes());
    pc[60..64].copy_from_slice(&sun_intensity.to_le_bytes());
    for (i, v) in aabb_min.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[76..80].copy_from_slice(&ground_y.to_le_bytes());
    for (i, v) in aabb_max.iter().enumerate() {
        pc[80 + i * 4..84 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[92..96].copy_from_slice(&dist_clamp.to_le_bytes());
    pc[96..100].copy_from_slice(&skylight_floor.to_le_bytes());
    pc[100..104].copy_from_slice(&alpha.to_le_bytes());
    pc[104..108].copy_from_slice(&bias.to_le_bytes());
    pc[108..112].copy_from_slice(&ray_max.to_le_bytes());
    // Stage B clipmap descriptor (uint4 clip at offset 112): x = index, y = level count.
    pc[112..116].copy_from_slice(&clip_desc.to_le_bytes());
    pc[116..120].copy_from_slice(&clip_count.to_le_bytes());
    // Stage D2: clip.z carries the amortized-relight period (round-robin card budget; 1 = legacy
    // every-frame). Stage D2b: clip.w carries the per-card visibility buffer index (0xFFFFFFFF =
    // no feedback => uniform period). See sdf_cache_light.slang.
    pc[120..124].copy_from_slice(&relight_period.to_le_bytes());
    pc[124..128].copy_from_slice(&card_vis_index.to_le_bytes());
    // P3: cone-trace LOD slope on its own 16-byte-aligned row (offset 128). 0 = legacy linear march.
    pc[128..132].copy_from_slice(&cone_k.to_le_bytes());
    // A2-fix: host-visible convergence buffer index (former pad0 slot @132). 0xFFFFFFFF = disabled.
    pc[132..136].copy_from_slice(&conv_buf.to_le_bytes());
    // Skylight-floor IBL irradiance cube index (former pad1 slot @136). 0xFFFFFFFF = absent (floor off).
    pc[136..140].copy_from_slice(&irradiance_index.to_le_bytes());
    // Per-sample firefly clamp for the indirect gather (former pad2 slot @140). 0 = off.
    pc[140..144].copy_from_slice(&gather_firefly.to_le_bytes());
    // Sky (float4 at offset 144, after cone_k + 3 pad floats): x = gain, yzw = white balance —
    // the same procedural-sky params the path tracer uses, for the relight's sky-on-miss.
    pc[144..148].copy_from_slice(&sky_gain.to_le_bytes());
    for (i, v) in sky_wb.iter().enumerate() {
        pc[148 + i * 4..152 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // Deferred-parity skylight (offset 160): sky-visibility SH volume base + the SAME
    // min-occlusion / tint values the lighting pass applies. 0xFFFFFFFF = legacy sky-on-miss.
    pc[160..164].copy_from_slice(&skyvis_index.to_le_bytes());
    pc[164..168].copy_from_slice(&skyvis_tint.to_le_bytes());
    pc[168..172].copy_from_slice(&skyvis_min_occ.to_le_bytes());
    // bit0: TLAS gather (the HWRT-shadow permutation's indirect rays trace exact triangles).
    pc[172..176].copy_from_slice(&flags.to_le_bytes());
    // GDF-AO params (offset 176): (reach, strength, bias, floor) — the SAME `GiSystem::ao_params`
    // the screen-space AO pass uses, so the parity skylight is occluded by the identical formula.
    pc[176..180].copy_from_slice(&ao_params.0.to_le_bytes());
    pc[180..184].copy_from_slice(&ao_params.1.to_le_bytes());
    pc[184..188].copy_from_slice(&ao_params.2.to_le_bytes());
    pc[188..192].copy_from_slice(&ao_params.3.to_le_bytes());
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

/// Pack the Phase 11 GDF-trace push block (160 bytes): inv_view_proj (64) + cam_pos
/// (16) + sun dir+intensity (16) + (out, width, height, flip_y) (16) + (gdf_sampled,
/// mode, pad, pad) (16) + aabb_min.xyz/ground_y (16) + aabb_max.xyz/dist_clamp (16).
/// `mode` bit0 swaps the GDF sample for the analytic reference field. The GDF world
/// extent + ground height + sample clamp move with the field (B4 unit cube vs. C1
/// scene). Same head layout as `sdf_trace_push`.
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
    clip_desc: u32,
    clip_count: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    ground_y: f32,
    dist_clamp: f32,
) -> [u8; 160] {
    let mut pc = [0u8; 160];
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
    // Stage B clipmap descriptor (former pad0/pad1 slots).
    pc[120..124].copy_from_slice(&clip_desc.to_le_bytes());
    pc[124..128].copy_from_slice(&clip_count.to_le_bytes());
    // aabb_min.xyz + ground_y in .w
    for (i, v) in aabb_min.iter().enumerate() {
        pc[128 + i * 4..132 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[140..144].copy_from_slice(&ground_y.to_le_bytes());
    // aabb_max.xyz + sample clamp in .w
    for (i, v) in aabb_max.iter().enumerate() {
        pc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[156..160].copy_from_slice(&dist_clamp.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C2 GDF-AO push block (144 bytes): inv_view_proj (64) +
/// (depth_index, normal_index, gdf_sampled, out_index) (16) + (width, height, flip_y,
/// pad) (16) + aabb_min.xyz/ground_y (16) + aabb_max.xyz/dist_clamp (16) +
/// (reach, strength, bias, pad) (16). World position is reconstructed from the depth
/// G-buffer (the position MRT is object-space), so only `inv_view_proj` is needed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_ao_push(
    inv_view_proj: &[f32; 16],
    depth_index: u32,
    normal_index: u32,
    gdf_sampled: u32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    ground_y: f32,
    dist_clamp: f32,
    reach: f32,
    strength: f32,
    bias: f32,
    floor: f32,
    clip_desc: u32,
    clip_count: u32,
) -> [u8; 160] {
    let mut pc = [0u8; 160];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&depth_index.to_le_bytes());
    pc[68..72].copy_from_slice(&normal_index.to_le_bytes());
    pc[72..76].copy_from_slice(&gdf_sampled.to_le_bytes());
    pc[76..80].copy_from_slice(&out_index.to_le_bytes());
    pc[80..84].copy_from_slice(&width.to_le_bytes());
    pc[84..88].copy_from_slice(&height.to_le_bytes());
    pc[88..92].copy_from_slice(&flip_y.to_le_bytes());
    // pc[92..96]: pad to the float4 boundary.
    for (i, v) in aabb_min.iter().enumerate() {
        pc[96 + i * 4..100 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[108..112].copy_from_slice(&ground_y.to_le_bytes());
    for (i, v) in aabb_max.iter().enumerate() {
        pc[112 + i * 4..116 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[124..128].copy_from_slice(&dist_clamp.to_le_bytes());
    pc[128..132].copy_from_slice(&reach.to_le_bytes());
    pc[132..136].copy_from_slice(&strength.to_le_bytes());
    pc[136..140].copy_from_slice(&bias.to_le_bytes());
    pc[140..144].copy_from_slice(&floor.to_le_bytes()); // params.w = AO floor (min AO value)
    // Stage B clipmap descriptor (uint4 clip at offset 144): x = index, y = level count.
    pc[144..148].copy_from_slice(&clip_desc.to_le_bytes());
    pc[148..152].copy_from_slice(&clip_count.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C3 GDF-GI push block (256 bytes): inv_view_proj (64) +
/// sun dir+intensity (16) + (depth, normal, gdf_sampled, out) (16) + (width, height,
/// flip_y, spp) (16) + (frame, albedo_rgb) (16) + aabb_min.xyz/ground_y (16) +
/// aabb_max.xyz/dist_clamp (16) + (ray_max_dist, bias, sky_term, hit_albedo) (16) +
/// (cache uint4 + tile, clamp_max, clip_desc, clip_count) (16) + ground_albedo.xyz/cone_k (16) +
/// (max_steps, vol_rgb) (16) + F3 (hwrt, pad×3) (16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_gi_push(
    inv_view_proj: &[f32; 16],
    sun_dir: [f32; 3],
    sun_intensity: f32,
    depth_index: u32,
    normal_index: u32,
    gdf_sampled: u32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    spp: u32,
    frame: u32,
    albedo_rgb: [u32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    ground_y: f32,
    dist_clamp: f32,
    ray_max_dist: f32,
    bias: f32,
    sky_term: f32,
    hit_albedo: f32,
    cache: [u32; 5],
    clamp_max: f32,
    clip_desc: u32,
    clip_count: u32,
    ground_albedo: [f32; 3],
    max_steps: u32,
    cone_k: f32,
    vol_rgb: [u32; 3],
    // F3 (HW-RT high-fidelity path): 0 = off (SW sphere-march, default & byte-identical anchor);
    // 1 = hardware-traced visibility gather against the scene TLAS. Opt-in High tier.
    hwrt: u32,
    // F4 (importance-sampled final gather): fraction of the gather rays drawn from the sun-steered
    // irradiance lobe (MIS with the cosine lobe). 0.0 = legacy cosine gather (byte-identical anchor).
    gi_importance: f32,
    // F4 (hierarchical radiance cache): storage-buffer index of the camera-anchored fine-level
    // AABB (2×float4: min.xyz+0, max.xyz+0) the volume-sampling branch reads. 0xFFFFFFFF = single
    // level (the legacy volume branch, an untouched instruction stream).
    fine_buf: u32,
) -> [u8; 256] {
    let mut pc = [0u8; 256];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let sun = normalize3(sun_dir);
    pc[64..68].copy_from_slice(&sun[0].to_le_bytes());
    pc[68..72].copy_from_slice(&sun[1].to_le_bytes());
    pc[72..76].copy_from_slice(&sun[2].to_le_bytes());
    pc[76..80].copy_from_slice(&sun_intensity.to_le_bytes());
    pc[80..84].copy_from_slice(&depth_index.to_le_bytes());
    pc[84..88].copy_from_slice(&normal_index.to_le_bytes());
    pc[88..92].copy_from_slice(&gdf_sampled.to_le_bytes());
    pc[92..96].copy_from_slice(&out_index.to_le_bytes());
    pc[96..100].copy_from_slice(&width.to_le_bytes());
    pc[100..104].copy_from_slice(&height.to_le_bytes());
    pc[104..108].copy_from_slice(&flip_y.to_le_bytes());
    pc[108..112].copy_from_slice(&spp.to_le_bytes());
    pc[112..116].copy_from_slice(&frame.to_le_bytes());
    // C8a albedo channel indices (former pad slots): 0xFFFFFFFF = constant fallback.
    pc[116..120].copy_from_slice(&albedo_rgb[0].to_le_bytes());
    pc[120..124].copy_from_slice(&albedo_rgb[1].to_le_bytes());
    pc[124..128].copy_from_slice(&albedo_rgb[2].to_le_bytes());
    for (i, v) in aabb_min.iter().enumerate() {
        pc[128 + i * 4..132 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[140..144].copy_from_slice(&ground_y.to_le_bytes());
    for (i, v) in aabb_max.iter().enumerate() {
        pc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[156..160].copy_from_slice(&dist_clamp.to_le_bytes());
    pc[160..164].copy_from_slice(&ray_max_dist.to_le_bytes());
    pc[164..168].copy_from_slice(&bias.to_le_bytes());
    pc[168..172].copy_from_slice(&sky_term.to_le_bytes());
    pc[172..176].copy_from_slice(&hit_albedo.to_le_bytes());
    // C8b3 surface-cache lookup indices (uint4 cache + tile): cards = 0xFFFFFFFF -> off.
    for (i, v) in cache.iter().enumerate() {
        pc[176 + i * 4..180 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // Firefly clamp (pad_c0 slot after cache_tile): 1e30 = off.
    pc[196..200].copy_from_slice(&clamp_max.to_le_bytes());
    // Stage B clipmap descriptor (former pad_c1/pad_c2 slots): storage-buffer index +
    // level count (1 = single volume = legacy single-level field).
    pc[200..204].copy_from_slice(&clip_desc.to_le_bytes());
    pc[204..208].copy_from_slice(&clip_count.to_le_bytes());
    // Analytic-ground albedo as a float4 at offset 208 (xyz = albedo, w = cone_k): floor bounce
    // hits re-light with xyz instead of albedo_at(). Packed float4 (not float3 + trailing scalars)
    // so the Metal/MSL push layout matches HLSL/SPIR-V — a scalar after a float3 packs at +12 on
    // HLSL/SPIR-V but pads to +16 on MSL, which previously mis-aligned max_steps/cone_k on Metal.
    for (i, v) in ground_albedo.iter().enumerate() {
        pc[208 + i * 4..212 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // P3 cone-trace LOD slope = ground_albedo.w (offset 220). 0 = legacy linear march.
    pc[220..224].copy_from_slice(&cone_k.to_le_bytes());
    // Stage D3: bounce-ray march step cap on its own 16-byte row (offset 224). Content lowers it;
    // the gallery passes the legacy 64.
    pc[224..228].copy_from_slice(&max_steps.to_le_bytes());
    // GI irradiance-volume sampled indices (R/G/B) — 3 contiguous uints after max_steps (offset
    // 228..240), Metal-safe (all scalars). 0xFFFFFFFF = off (trace rays instead of sampling).
    pc[228..232].copy_from_slice(&vol_rgb[0].to_le_bytes());
    pc[232..236].copy_from_slice(&vol_rgb[1].to_le_bytes());
    pc[236..240].copy_from_slice(&vol_rgb[2].to_le_bytes());
    // F3 + F4 share the last 16-byte row: the HW-RT gather toggle at 240..244 (0 = SW march =
    // gallery byte-identical) and the F4 importance-sampling mix at 244..248 (0.0 = legacy
    // cosine gather = byte-identical); +248 = F4 fine-AABB storage-buffer index (0xFFFFFFFF =
    // off = the legacy single-level volume branch); 252..256 stays padding.
    pc[240..244].copy_from_slice(&hwrt.to_le_bytes());
    pc[244..248].copy_from_slice(&gi_importance.to_le_bytes());
    pc[248..252].copy_from_slice(&fine_buf.to_le_bytes());
    pc
}

/// Pack the GI irradiance-volume (DDGI-lite) update push block (192 bytes): aabb_min(+ground_y),
/// aabb_max(+dist_clamp), sun(+intensity), dims(+frame), read_rgb(+reset), write_rgb, albedo_rgb,
/// clip(desc,count), params(spp,ray_max,sky_fill,alpha), ground(xyz albedo, w bias), then the F4
/// fine-level rows: fine_min(xyz, w = fine_active — 0 = single level = legacy) + fine_max(xyz).
/// See gi_volume.slang.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gi_volume_push(
    aabb_min: [f32; 3],
    ground_y: f32,
    aabb_max: [f32; 3],
    dist_clamp: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    dims: [u32; 3],
    frame: u32,
    read_rgb: [u32; 3],
    reset: u32,
    write_rgb: [u32; 3],
    albedo_rgb: [u32; 3],
    clip_desc: u32,
    clip_count: u32,
    spp: f32,
    ray_max: f32,
    sky_fill: f32,
    alpha: f32,
    ground_albedo: [f32; 3],
    bias: f32,
    fine_min: [f32; 3],
    fine_active: f32,
    fine_max: [f32; 3],
) -> [u8; 192] {
    let mut pc = [0u8; 192];
    let put3 = |pc: &mut [u8], o: usize, v: [f32; 3]| {
        for (i, x) in v.iter().enumerate() {
            pc[o + i * 4..o + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    let put3u = |pc: &mut [u8], o: usize, v: [u32; 3]| {
        for (i, x) in v.iter().enumerate() {
            pc[o + i * 4..o + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    let sun = normalize3(sun_dir);
    put3(&mut pc, 0, aabb_min);
    pc[12..16].copy_from_slice(&ground_y.to_le_bytes());
    put3(&mut pc, 16, aabb_max);
    pc[28..32].copy_from_slice(&dist_clamp.to_le_bytes());
    put3(&mut pc, 32, [sun[0], sun[1], sun[2]]);
    pc[44..48].copy_from_slice(&sun_intensity.to_le_bytes());
    put3u(&mut pc, 48, dims);
    pc[60..64].copy_from_slice(&frame.to_le_bytes());
    put3u(&mut pc, 64, read_rgb);
    pc[76..80].copy_from_slice(&reset.to_le_bytes());
    put3u(&mut pc, 80, write_rgb);
    put3u(&mut pc, 96, albedo_rgb);
    pc[112..116].copy_from_slice(&clip_desc.to_le_bytes());
    pc[116..120].copy_from_slice(&clip_count.to_le_bytes());
    pc[128..132].copy_from_slice(&spp.to_le_bytes());
    pc[132..136].copy_from_slice(&ray_max.to_le_bytes());
    pc[136..140].copy_from_slice(&sky_fill.to_le_bytes());
    pc[140..144].copy_from_slice(&alpha.to_le_bytes());
    put3(&mut pc, 144, ground_albedo);
    pc[156..160].copy_from_slice(&bias.to_le_bytes());
    // F4 camera-anchored fine level: world AABB + active flag (fine_min.w). 0 = single level —
    // the shader's expressions then reduce to the legacy values exactly. 188..192 stays zero.
    put3(&mut pc, 160, fine_min);
    pc[172..176].copy_from_slice(&fine_active.to_le_bytes());
    put3(&mut pc, 176, fine_max);
    pc
}

/// Pack the Phase 11 Stage C4 GI-temporal push block (192 bytes): inv_view_proj (64) +
/// prev_view_proj (64) + (gi_raw, depth, normal, out) (16) + (hist_r, hist_w, pos_r,
/// pos_w) (16) + (width, height, flip_y, reset) (16) + (reject_dist, max_hist, min_alpha,
/// pad) (16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_temporal_push(
    inv_view_proj: &[f32; 16],
    prev_view_proj: &[f32; 16],
    gi_raw_index: u32,
    depth_index: u32,
    normal_index: u32,
    out_index: u32,
    hist_read: u32,
    hist_write: u32,
    pos_read: u32,
    pos_write: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    reset: u32,
    reject_dist: f32,
    max_hist: f32,
    min_alpha: f32,
    neighborhood: f32,
) -> [u8; 192] {
    let mut pc = [0u8; 192];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in prev_view_proj.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    let u = [
        gi_raw_index,
        depth_index,
        normal_index,
        out_index,
        hist_read,
        hist_write,
        pos_read,
        pos_write,
        width,
        height,
        flip_y,
        reset,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[128 + i * 4..132 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[176..180].copy_from_slice(&reject_dist.to_le_bytes());
    pc[180..184].copy_from_slice(&max_hist.to_le_bytes());
    pc[184..188].copy_from_slice(&min_alpha.to_le_bytes());
    pc[188..192].copy_from_slice(&neighborhood.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C4 GI-à-trous push block (112 bytes): inv_view_proj (64) +
/// (in, depth, normal, out) (16) + (width, height, step, flip_y) (16) + (pos_sigma,
/// normal_power, pad, pad) (16). The `float4 params` aligns to offset 96, so the block
/// is 112 bytes, not 96.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_atrous_push(
    inv_view_proj: &[f32; 16],
    in_index: u32,
    depth_index: u32,
    normal_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
    step: u32,
    flip_y: u32,
    pos_sigma: f32,
    normal_power: f32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let u = [
        in_index,
        depth_index,
        normal_index,
        out_index,
        width,
        height,
        step,
        flip_y,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[96..100].copy_from_slice(&pos_sigma.to_le_bytes());
    pc[100..104].copy_from_slice(&normal_power.to_le_bytes());
    pc
}

/// Pack the Stage D2b surface-cache visibility push block (112 bytes): 6 frustum planes
/// (96, xyz inward normal + w) + (cards_index, out_index, num_cards, pad) uints (96..112).
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cache_vis_push(
    planes: &[[f32; 4]; 6],
    cards_index: u32,
    out_index: u32,
    num_cards: u32,
    marks_index: u32,
    // Lit-calibration probe (None = off/sentinel): prev view-proj, prev eye, probe seed, then
    // the (corr, lit_hist, cache_pos, cache_rad) indices, the uniform tile edge, the lit-history
    // pixel dims, and the Y-flip word. See sdf_cache_visibility.slang's CacheVisPush.
    calib: Option<(
        [f32; 16],
        Vec3,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        (u32, u32),
        u32,
    )>,
    // F1 Stage 2 — page-pool LRU clock (0xFFFFFFFF = off) + the current frame timestamp.
    touched_index: u32,
    frame: u32,
) -> [u8; 232] {
    let mut pc = [0u8; 232];
    for (i, p) in planes.iter().enumerate() {
        for (j, v) in p.iter().enumerate() {
            let o = i * 16 + j * 4;
            pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    pc[96..100].copy_from_slice(&cards_index.to_le_bytes());
    pc[100..104].copy_from_slice(&out_index.to_le_bytes());
    pc[104..108].copy_from_slice(&num_cards.to_le_bytes());
    // Mirror-feedback flags (u32::MAX = off): merged into the visibility priority + cleared.
    pc[108..112].copy_from_slice(&marks_index.to_le_bytes());
    // Calibration block (offset 112): corr_index = u32::MAX disables the probe entirely.
    let (pvp, eye, seed, corr, lit, cpos, rad, tile, (w, h), flip) = calib.unwrap_or((
        [0.0; 16],
        Vec3::ZERO,
        0,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        0,
        (0, 0),
        0,
    ));
    for (i, v) in pvp.iter().enumerate() {
        pc[112 + i * 4..116 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[176..180].copy_from_slice(&eye.x.to_le_bytes());
    pc[180..184].copy_from_slice(&eye.y.to_le_bytes());
    pc[184..188].copy_from_slice(&eye.z.to_le_bytes());
    // prev_eye.w carries the probe stratification seed (frame counter).
    pc[188..192].copy_from_slice(&seed.to_le_bytes());
    pc[192..196].copy_from_slice(&corr.to_le_bytes());
    pc[196..200].copy_from_slice(&lit.to_le_bytes());
    pc[200..204].copy_from_slice(&cpos.to_le_bytes());
    pc[204..208].copy_from_slice(&rad.to_le_bytes());
    pc[208..212].copy_from_slice(&tile.to_le_bytes());
    pc[212..216].copy_from_slice(&w.to_le_bytes());
    pc[216..220].copy_from_slice(&h.to_le_bytes());
    pc[220..224].copy_from_slice(&flip.to_le_bytes());
    // F1 Stage 2 — page-pool LRU clock + frame timestamp.
    pc[224..228].copy_from_slice(&touched_index.to_le_bytes());
    pc[228..232].copy_from_slice(&frame.to_le_bytes());
    pc
}

/// Pack the QHD/UHD FXAA push block (16 bytes): in_index, out_index, width, height.
pub(crate) fn fxaa_push(in_index: u32, out_index: u32, width: u32, height: u32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&in_index.to_le_bytes());
    pc[4..8].copy_from_slice(&out_index.to_le_bytes());
    pc[8..12].copy_from_slice(&width.to_le_bytes());
    pc[12..16].copy_from_slice(&height.to_le_bytes());
    pc
}

/// Pack the QHD/UHD TAAU push block (208 bytes): inv_view_proj (64) + prev_view_proj (64) +
/// 13 uints (hdr, depth, out, hist_r/w, pos_r/w, out_w/h, in_w/h, flip, reset) at 128..180 +
/// params float4 (reject_dist, max_hist, min_alpha) at the next 16-byte row (192).
#[allow(clippy::too_many_arguments)]
pub(crate) fn taau_push(
    inv_view_proj: &[f32; 16],
    prev_view_proj: &[f32; 16],
    hdr_index: u32,
    depth_index: u32,
    out_index: u32,
    hist_read: u32,
    hist_write: u32,
    pos_read: u32,
    pos_write: u32,
    out_width: u32,
    out_height: u32,
    in_width: u32,
    in_height: u32,
    flip_y: u32,
    reset: u32,
    reject_dist: f32,
    max_hist: f32,
    variance_gamma: f32,
    clamp_expand: f32,
    jitter_uv: [f32; 2],
    velocity_index: u32,
    packed_hist: u32,
) -> [u8; 240] {
    let mut pc = [0u8; 240];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in prev_view_proj.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    let u = [
        hdr_index,
        depth_index,
        out_index,
        hist_read,
        hist_write,
        pos_read,
        pos_write,
        out_width,
        out_height,
        in_width,
        in_height,
        flip_y,
        reset,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[128 + i * 4..132 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[192..196].copy_from_slice(&reject_dist.to_le_bytes());
    pc[196..200].copy_from_slice(&max_hist.to_le_bytes());
    pc[200..204].copy_from_slice(&variance_gamma.to_le_bytes());
    // params.w: TSR-style clamp-box expansion factor (0 = tight box = byte-identical anchor).
    pc[204..208].copy_from_slice(&clamp_expand.to_le_bytes());
    // float4 jitter (xy = current jitter in UV) at the next 16-byte row.
    pc[208..212].copy_from_slice(&jitter_uv[0].to_le_bytes());
    pc[212..216].copy_from_slice(&jitter_uv[1].to_le_bytes());
    // velocity target index (PR-2) at the next 16-byte row; 0xFFFFFFFF = absent (camera-only
    // reprojection, byte-identical to the pre-velocity path).
    pc[224..228].copy_from_slice(&velocity_index.to_le_bytes());
    // fp16-packed history flag (0 = legacy 16B hist + 16B pos layout, the byte-identical anchor).
    pc[228..232].copy_from_slice(&packed_hist.to_le_bytes());
    pc
}

/// Pack the Stage D1 half-res GI upsample push block (128 bytes): inv_view_proj (64) +
/// (gi_half, depth, normal, out, width, height, half_width, half_height, flip_y) uints
/// (64..100) + params float4 (pos_sigma, normal_power) at the next 16-byte row (112).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_gi_upsample_push(
    inv_view_proj: &[f32; 16],
    gi_half_index: u32,
    depth_index: u32,
    normal_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
    half_width: u32,
    half_height: u32,
    flip_y: u32,
    pos_sigma: f32,
    normal_power: f32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let u = [
        gi_half_index,
        depth_index,
        normal_index,
        out_index,
        width,
        height,
        half_width,
        half_height,
        flip_y,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // float4 params at offset 112 (the next 16-byte boundary after the 9 uints).
    pc[112..116].copy_from_slice(&pos_sigma.to_le_bytes());
    pc[116..120].copy_from_slice(&normal_power.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C5 SSR push block (192 bytes): view_proj (64) +
/// inv_view_proj (64) + cam_pos (16) + (depth, normal, material, color, out) +
/// (width, height, flip_y) (32 across two rows) + (max_dist, thickness, steps,
/// edge_fade) (16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssr_push(
    view_proj: &[f32; 16],
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    depth_index: u32,
    normal_index: u32,
    material_index: u32,
    hist_index: u32,
    color_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    frame: u32,
    full_width: u32,
    full_height: u32,
    max_dist: f32,
    thickness: f32,
    steps: f32,
    edge_fade: f32,
    out_b_index: u32,
) -> [u8; 224] {
    let mut pc = [0u8; 224];
    for (i, v) in view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[128..132].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[132..136].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[136..140].copy_from_slice(&cam_pos.z.to_le_bytes());
    let u = [
        depth_index,
        normal_index,
        material_index,
        hist_index,
        color_index,
        out_index,
        width,
        height,
        flip_y,
        frame,
        full_width,
        full_height,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[192..196].copy_from_slice(&max_dist.to_le_bytes());
    pc[196..200].copy_from_slice(&thickness.to_le_bytes());
    pc[200..204].copy_from_slice(&steps.to_le_bytes());
    pc[204..208].copy_from_slice(&edge_fade.to_le_bytes());
    pc[208..212].copy_from_slice(&out_b_index.to_le_bytes());
    pc
}

/// Pack the stochastic-SSR ratio-estimator resolve push block (224 bytes): inv_view_proj
/// (64) + prev_view_proj (64) + cam_pos (16) + 14 uints (ssr_a, ssr_b, depth, normal,
/// material, out, accum_r/w, pos_r/w, width, height, flip_y, reset) + params (reject_dist,
/// alpha, clamp_max, kernel_radius).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssr_resolve_push(
    inv_view_proj: &[f32; 16],
    prev_view_proj: &[f32; 16],
    cam_pos: Vec3,
    ssr_a_index: u32,
    ssr_b_index: u32,
    depth_index: u32,
    normal_index: u32,
    material_index: u32,
    out_index: u32,
    accum_read: u32,
    accum_write: u32,
    pos_read: u32,
    pos_write: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    reset: u32,
    reject_dist: f32,
    alpha: f32,
    clamp_max: f32,
    kernel_radius: f32,
    clamp_mode: u32,
    clamp_gamma: f32,
) -> [u8; 240] {
    let mut pc = [0u8; 240];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in prev_view_proj.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[128..132].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[132..136].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[136..140].copy_from_slice(&cam_pos.z.to_le_bytes());
    let u = [
        ssr_a_index,
        ssr_b_index,
        depth_index,
        normal_index,
        material_index,
        out_index,
        accum_read,
        accum_write,
        pos_read,
        pos_write,
        width,
        height,
        flip_y,
        reset,
    ];
    for (i, v) in u.iter().enumerate() {
        pc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[208..212].copy_from_slice(&reject_dist.to_le_bytes());
    pc[212..216].copy_from_slice(&alpha.to_le_bytes());
    pc[216..220].copy_from_slice(&clamp_max.to_le_bytes());
    pc[220..224].copy_from_slice(&kernel_radius.to_le_bytes());
    // History neighbourhood clamp (mode 0 = off = byte-identical). float4 `params` occupies 208..224,
    // so the two scalars land in the next 16-byte register (224..232; 232..240 is tail padding).
    pc[224..228].copy_from_slice(&clamp_mode.to_le_bytes());
    pc[228..232].copy_from_slice(&clamp_gamma.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C6 GDF-reflection push block (240 bytes). Layout: inv_view_proj
/// 64, cam_pos 16, sun dir+intensity 16, then four uints depth/normal/gdf_sampled/out 16,
/// then width/height/flip_y/pad 16, then aabb_min.xyz+ground_y 16, aabb_max.xyz+clamp 16,
/// ray_max_dist/hit_albedo/sky_fill/bias 16, the C8a albedo R/G/B volume indices + frame 16,
/// cache uint4+tile +pad×3 16, then ground_albedo.xyz on its own 16-aligned row 16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_reflect_push(
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    depth_index: u32,
    normal_index: u32,
    gdf_sampled: u32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    // GI irradiance-volume base index (radiance cache) sampled at reflection hits for the indirect
    // term; u32::MAX = off (gallery) -> legacy analytic sky fill, byte-identical. Packed into
    // flip_y's spare bits below (the 240-byte block is full; D3D12 root budget forbids growing it).
    gi_vol_base: u32,
    material_index: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    ground_y: f32,
    dist_clamp: f32,
    ray_max_dist: f32,
    hit_albedo: f32,
    sky_fill: f32,
    bias: f32,
    albedo_rgb: [u32; 3],
    frame: u32,
    cache: [u32; 5],
    clip_desc: u32,
    clip_count: u32,
    ground_albedo: [f32; 3],
    max_steps: u32,
    cone_k: f32,
    // Adaptive temporal skip (docs/lossless-opt-ledger.md A3): (skip_read, skip_write, k_stagger,
    // real_frame). skip_read/write = 0xFFFFFFFF sentinel disables reuse/write (gallery + viz paths).
    // Byte-packed into the unused `gdf_sampled` slot to keep the block at 240 B — growing it to 256 B
    // would push it over the D3D12 64-DWORD root budget (CBV spill = ~4ms on the reflect pass).
    skip: [u32; 4],
) -> [u8; 240] {
    let mut pc = [0u8; 240];
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
    pc[96..100].copy_from_slice(&depth_index.to_le_bytes());
    pc[100..104].copy_from_slice(&normal_index.to_le_bytes());
    // A3 skip, byte-packed (reuses the unused `gdf_sampled` slot; `gdf_sampled` is ignored by the
    // shader). Bindless indices are < 64 (STORAGE_BUFFER_COUNT), so read/write fit in a byte with
    // 0xFF as the "disabled" sentinel; K < 256; frame is taken mod 256 (the stagger rotates 1/K).
    let _ = gdf_sampled;
    let pack = |v: u32| if v == u32::MAX { 0xFFu32 } else { v & 0xFF };
    let skip_packed =
        pack(skip[0]) | (pack(skip[1]) << 8) | ((skip[2] & 0xFF) << 16) | ((skip[3] & 0xFF) << 24);
    pc[104..108].copy_from_slice(&skip_packed.to_le_bytes());
    pc[108..112].copy_from_slice(&out_index.to_le_bytes());
    pc[112..116].copy_from_slice(&width.to_le_bytes());
    pc[116..120].copy_from_slice(&height.to_le_bytes());
    // Pack the GI-volume base into flip_y's upper bits (shader reads bit0 for the Y-flip and
    // `>> 1` for the volume). Encode (base+1) so 0 = off; base indices are small (< bindless
    // volume count) so this never collides with bit0.
    let flip_packed = (flip_y & 1)
        | if gi_vol_base == u32::MAX {
            0
        } else {
            (gi_vol_base + 1) << 1
        };
    pc[120..124].copy_from_slice(&flip_packed.to_le_bytes());
    pc[124..128].copy_from_slice(&material_index.to_le_bytes());
    for (i, v) in aabb_min.iter().enumerate() {
        pc[128 + i * 4..132 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[140..144].copy_from_slice(&ground_y.to_le_bytes());
    for (i, v) in aabb_max.iter().enumerate() {
        pc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[156..160].copy_from_slice(&dist_clamp.to_le_bytes());
    pc[160..164].copy_from_slice(&ray_max_dist.to_le_bytes());
    pc[164..168].copy_from_slice(&hit_albedo.to_le_bytes());
    pc[168..172].copy_from_slice(&sky_fill.to_le_bytes());
    pc[172..176].copy_from_slice(&bias.to_le_bytes());
    // C8a albedo channel indices (uint4 row): 0xFFFFFFFF = constant fallback.
    pc[176..180].copy_from_slice(&albedo_rgb[0].to_le_bytes());
    pc[180..184].copy_from_slice(&albedo_rgb[1].to_le_bytes());
    pc[184..188].copy_from_slice(&albedo_rgb[2].to_le_bytes());
    pc[188..192].copy_from_slice(&frame.to_le_bytes()); // C8j GGX-jitter RNG decorrelation
    // C8b3 surface-cache lookup indices (uint4 cache + tile): cards = 0xFFFFFFFF -> off.
    for (i, v) in cache.iter().enumerate() {
        pc[192 + i * 4..196 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // Stage B clipmap descriptor (former pad_c0/pad_c1 slots).
    pc[212..216].copy_from_slice(&clip_desc.to_le_bytes());
    pc[216..220].copy_from_slice(&clip_count.to_le_bytes());
    // Stage D3: reflection-ray march step cap (former pad_c2). Content lowers it; gallery = 96.
    pc[220..224].copy_from_slice(&max_steps.to_le_bytes());
    // Analytic-ground albedo (float3 on its own 16-byte-aligned row, offset 224): floor hits
    // re-light with this instead of albedo_at() (no ground data -> nearest object's colour).
    // 16-aligned so SPIR-V (vec3 align 16) and DXIL agree on the offset.
    for (i, v) in ground_albedo.iter().enumerate() {
        pc[224 + i * 4..228 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // P3: cone-trace LOD slope reuses the float3 ground_albedo's .w padding (offset 236; the block
    // is already 240 bytes). 0 = legacy linear march = byte-identical.
    pc[236..240].copy_from_slice(&cone_k.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C7 hybrid-composite push block (32 bytes): SSR + GDF image
/// indices, the output storage index, width/height, and `gdf_scale` (the exposure applied
/// to the raw GDF radiance so it shares the SSR's post-exposure viz space).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reflect_composite_push(
    ssr_index: u32,
    gdf_index: u32, // GDF reflection (GGX-resolved, already roughness-blurred)
    out_index: u32,
    width: u32,
    height: u32,
    gdf_scale: f32,
    clamp_max: f32,
    material_index: u32,
    max_roughness: f32,
    skip_mirror_ssr: bool,
    rough_blur: f32,
    ssr_cut: bool,
    // B2 mirror compaction: (refine target sampled index, refine grid w, h). `None` = off
    // (0xFFFFFFFF sentinel) — the legacy 48-byte prefix is bit-identical either way.
    refine: Option<(u32, u32, u32)>,
    refine_thresh: f32,
) -> [u8; 64] {
    let mut pc = [0u8; 64];
    pc[0..4].copy_from_slice(&ssr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&gdf_index.to_le_bytes());
    pc[8..12].copy_from_slice(&out_index.to_le_bytes());
    pc[12..16].copy_from_slice(&width.to_le_bytes());
    pc[16..20].copy_from_slice(&height.to_le_bytes());
    pc[20..24].copy_from_slice(&gdf_scale.to_le_bytes());
    pc[24..28].copy_from_slice(&clamp_max.to_le_bytes());
    pc[28..32].copy_from_slice(&material_index.to_le_bytes());
    pc[32..36].copy_from_slice(&max_roughness.to_le_bytes());
    pc[36..40].copy_from_slice(&u32::from(skip_mirror_ssr).to_le_bytes()); // pad0: content near-mirror SSR skip
    pc[40..44].copy_from_slice(&rough_blur.to_le_bytes()); // pad1: roughness-blur radius (0 = off/anchor)
    pc[44..48].copy_from_slice(&u32::from(ssr_cut).to_le_bytes()); // pad2: B1-lite SSR hard cut
    // B2 mirror-compaction row: refine target + grid, roughness gate.
    let (ri, rw, rh) = refine.unwrap_or((u32::MAX, 0, 0));
    pc[48..52].copy_from_slice(&ri.to_le_bytes());
    pc[52..56].copy_from_slice(&rw.to_le_bytes());
    pc[56..60].copy_from_slice(&rh.to_le_bytes());
    pc[60..64].copy_from_slice(&refine_thresh.to_le_bytes());
    pc
}

/// Pack the B2 mirror-compaction classify/reset/args push block (32 bytes): depth + material +
/// list + args + refine-target indices, the refine grid extent, and the near-mirror roughness
/// threshold (lockstep with `gdf_reflect.slang`'s content `mirror_thresh`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reflect_compact_push(
    depth_index: u32,
    material_index: u32,
    list_index: u32,
    args_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
    mirror_thresh: f32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    pc[0..4].copy_from_slice(&depth_index.to_le_bytes());
    pc[4..8].copy_from_slice(&material_index.to_le_bytes());
    pc[8..12].copy_from_slice(&list_index.to_le_bytes());
    pc[12..16].copy_from_slice(&args_index.to_le_bytes());
    pc[16..20].copy_from_slice(&out_index.to_le_bytes());
    pc[20..24].copy_from_slice(&width.to_le_bytes());
    pc[24..28].copy_from_slice(&height.to_le_bytes());
    pc[28..32].copy_from_slice(&mirror_thresh.to_le_bytes());
    pc
}

/// Pack the C8j stochastic-GDF-reflection temporal-resolve push block (208 bytes):
/// inv_view_proj (64) + prev_view_proj (64) + cam_pos (16) + image/buffer indices (32) +
/// (width, height, flip_y, reset) (16) + `float4 params` (reject dist, max history len,
/// firefly clamp, tonemap range) aligned at offset 192.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reflect_temporal_push(
    inv_view_proj: &[f32; 16],
    prev_view_proj: &[f32; 16],
    cam_pos: Vec3,
    refl_index: u32,
    depth_index: u32,
    out_index: u32,
    accum_read: u32,
    accum_write: u32,
    pos_read: u32,
    pos_write: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    reset: u32,
    material_index: u32,
    reject_dist: f32,
    max_len: f32,
    firefly_clamp: f32,
    tonemap_range: f32,
    clamp_mode: u32,
    clamp_gamma: f32,
    spatial_off: u32,
    // A4a: per-pixel 2nd-moment (variance) accumulation buffers + enable. Sentinel/0 = off (no M2).
    moment_read: u32,
    moment_write: u32,
    denoise: u32,
) -> [u8; 240] {
    let mut pc = [0u8; 240];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in prev_view_proj.iter().enumerate() {
        pc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[128..132].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[132..136].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[136..140].copy_from_slice(&cam_pos.z.to_le_bytes());
    pc[144..148].copy_from_slice(&refl_index.to_le_bytes());
    pc[148..152].copy_from_slice(&depth_index.to_le_bytes());
    pc[152..156].copy_from_slice(&out_index.to_le_bytes());
    pc[156..160].copy_from_slice(&accum_read.to_le_bytes());
    pc[160..164].copy_from_slice(&accum_write.to_le_bytes());
    pc[164..168].copy_from_slice(&pos_read.to_le_bytes());
    pc[168..172].copy_from_slice(&pos_write.to_le_bytes());
    pc[172..176].copy_from_slice(&width.to_le_bytes());
    pc[176..180].copy_from_slice(&height.to_le_bytes());
    pc[180..184].copy_from_slice(&flip_y.to_le_bytes());
    pc[184..188].copy_from_slice(&reset.to_le_bytes());
    pc[188..192].copy_from_slice(&material_index.to_le_bytes());
    pc[192..196].copy_from_slice(&reject_dist.to_le_bytes());
    pc[196..200].copy_from_slice(&max_len.to_le_bytes());
    pc[200..204].copy_from_slice(&firefly_clamp.to_le_bytes());
    pc[204..208].copy_from_slice(&tonemap_range.to_le_bytes());
    // History neighbourhood-clamp permutation (own 16-byte row at 208): mode (0 off / 1 hard /
    // 2 variance) + variance gamma. mode 0 => the shader skips the clamp = byte-identical legacy.
    pc[208..212].copy_from_slice(&clamp_mode.to_le_bytes());
    pc[212..216].copy_from_slice(&clamp_gamma.to_le_bytes());
    // A1: skip the spatial box average when the ratio-estimator resolve already ran (0 = keep it,
    // byte-identical legacy).
    pc[216..220].copy_from_slice(&spatial_off.to_le_bytes());
    // A4a: 2nd-moment ping-pong indices + enable (own 16-byte row at 220; denoise 0 => byte-identical).
    pc[220..224].copy_from_slice(&moment_read.to_le_bytes());
    pc[224..228].copy_from_slice(&moment_write.to_le_bytes());
    pc[228..232].copy_from_slice(&denoise.to_le_bytes());
    pc
}

/// Pack the Track A1 reflection spatial-resolve push block (128 bytes): inv_view_proj (64) +
/// cam_pos (16) + the four sampled indices (16) + (out, width, height, flip_y) (16) +
/// (frame, mirror_thresh, kernel_radius, pad) (16). Stateless — no history buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reflect_resolve_push(
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    refl_index: u32,
    depth_index: u32,
    normal_index: u32,
    material_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    frame: u32,
    mirror_thresh: f32,
    kernel_radius: f32,
    stochastic: u32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_pos.z.to_le_bytes());
    pc[80..84].copy_from_slice(&refl_index.to_le_bytes());
    pc[84..88].copy_from_slice(&depth_index.to_le_bytes());
    pc[88..92].copy_from_slice(&normal_index.to_le_bytes());
    pc[92..96].copy_from_slice(&material_index.to_le_bytes());
    pc[96..100].copy_from_slice(&out_index.to_le_bytes());
    pc[100..104].copy_from_slice(&width.to_le_bytes());
    pc[104..108].copy_from_slice(&height.to_le_bytes());
    pc[108..112].copy_from_slice(&flip_y.to_le_bytes());
    pc[112..116].copy_from_slice(&frame.to_le_bytes());
    pc[116..120].copy_from_slice(&mirror_thresh.to_le_bytes());
    pc[120..124].copy_from_slice(&kernel_radius.to_le_bytes());
    pc[124..128].copy_from_slice(&stochastic.to_le_bytes());
    pc
}

/// Pack the Track A4b reflection spatial-denoiser push block (128 bytes): inv_view_proj (64) +
/// cam_pos (16) + sampled indices (16) + (out, width, height, flip_y) (16) + (kernel_radius,
/// tonemap_range, pad, pad) (16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reflect_spatial_push(
    inv_view_proj: &[f32; 16],
    cam_pos: Vec3,
    refl_index: u32,
    depth_index: u32,
    normal_index: u32,
    material_index: u32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    kernel_radius: f32,
    tonemap_range: f32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_pos.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_pos.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_pos.z.to_le_bytes());
    pc[80..84].copy_from_slice(&refl_index.to_le_bytes());
    pc[84..88].copy_from_slice(&depth_index.to_le_bytes());
    pc[88..92].copy_from_slice(&normal_index.to_le_bytes());
    pc[92..96].copy_from_slice(&material_index.to_le_bytes());
    pc[96..100].copy_from_slice(&out_index.to_le_bytes());
    pc[100..104].copy_from_slice(&width.to_le_bytes());
    pc[104..108].copy_from_slice(&height.to_le_bytes());
    pc[108..112].copy_from_slice(&flip_y.to_le_bytes());
    pc[112..116].copy_from_slice(&kernel_radius.to_le_bytes());
    pc[116..120].copy_from_slice(&tonemap_range.to_le_bytes());
    pc
}

/// Pack the Phase 11 Stage C7b lit-history push block (32 bytes): the lit-HDR sampled
/// index, the history storage-buffer index, width/height, and `inv_exposure` (recovers
/// raw radiance from the exposure-baked HDR).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lit_history_push(
    hdr_index: u32,
    out_buffer: u32,
    width: u32,
    height: u32,
    inv_exposure: f32,
    clamp_max: f32,
    exposure_buf: u32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&out_buffer.to_le_bytes());
    pc[8..12].copy_from_slice(&width.to_le_bytes());
    pc[12..16].copy_from_slice(&height.to_le_bytes());
    pc[16..20].copy_from_slice(&inv_exposure.to_le_bytes());
    pc[20..24].copy_from_slice(&clamp_max.to_le_bytes());
    pc[24..28].copy_from_slice(&exposure_buf.to_le_bytes());
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
    sky_gain: f32,
    sky_wb: [f32; 3],
) -> [u8; 144] {
    let mut pc = [0u8; 144];
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
    // float4 sky: x = sky_gain (sun:sky ratio), yzw = sky white balance. Threaded from the
    // host so the path tracer's miss shader matches the env-capture's procedural sky exactly.
    pc[128..132].copy_from_slice(&sky_gain.to_le_bytes());
    pc[132..136].copy_from_slice(&sky_wb[0].to_le_bytes());
    pc[136..140].copy_from_slice(&sky_wb[1].to_le_bytes());
    pc[140..144].copy_from_slice(&sky_wb[2].to_le_bytes());
    pc
}

/// Pack the screen-probe TRACE push block (240 bytes): inv_view_proj (64) + sun (16) +
/// aabb_min/+ground_y (16) + aabb_max/+sample_clamp (16) + params (16) + ground_albedo/+cone_k
/// (16) + cache uint4 (16) + 14 scalar indices/dims (4 rows of 16) + clamp_max. All scalars are
/// grouped after the vec4 block (Metal-safe: no float3+trailing-scalar mis-packing).
/// See `screen_probe_trace.slang`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn screen_probe_trace_push(
    inv_view_proj: &[f32; 16],
    sun_dir: [f32; 3],
    sun_intensity: f32,
    aabb_min: [f32; 3],
    ground_y: f32,
    aabb_max: [f32; 3],
    sample_clamp: f32,
    ray_max: f32,
    bias: f32,
    sky_term: f32,
    albedo_fallback: f32,
    ground_albedo: [f32; 3],
    cone_k: f32,
    cache: [u32; 4],
    depth_index: u32,
    normal_index: u32,
    atlas_index: u32,
    cache_tile: u32,
    screen_w: u32,
    screen_h: u32,
    probes_x: u32,
    probes_y: u32,
    downsample: u32,
    oct_res: u32,
    flip_y: u32,
    frame: u32,
    max_steps: u32,
    clip_desc: u32,
    clip_count: u32,
    clamp_max: f32,
    wrc_atlas: u32,
    wrc_grid: u32,
    wrc_oct: u32,
) -> [u8; 240] {
    let mut pc = [0u8; 240];
    let put3 = |pc: &mut [u8], o: usize, v: [f32; 3]| {
        for (i, x) in v.iter().enumerate() {
            pc[o + i * 4..o + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    let putu = |pc: &mut [u8], o: usize, v: u32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let putf = |pc: &mut [u8], o: usize, v: f32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let sun = normalize3(sun_dir);
    put3(&mut pc, 64, [sun[0], sun[1], sun[2]]);
    putf(&mut pc, 76, sun_intensity);
    put3(&mut pc, 80, aabb_min);
    putf(&mut pc, 92, ground_y);
    put3(&mut pc, 96, aabb_max);
    putf(&mut pc, 108, sample_clamp);
    putf(&mut pc, 112, ray_max);
    putf(&mut pc, 116, bias);
    putf(&mut pc, 120, sky_term);
    putf(&mut pc, 124, albedo_fallback);
    put3(&mut pc, 128, ground_albedo);
    putf(&mut pc, 140, cone_k);
    for (i, v) in cache.iter().enumerate() {
        putu(&mut pc, 144 + i * 4, *v);
    }
    putu(&mut pc, 160, depth_index);
    putu(&mut pc, 164, normal_index);
    putu(&mut pc, 168, atlas_index);
    putu(&mut pc, 172, cache_tile);
    putu(&mut pc, 176, screen_w);
    putu(&mut pc, 180, screen_h);
    putu(&mut pc, 184, probes_x);
    putu(&mut pc, 188, probes_y);
    putu(&mut pc, 192, downsample);
    putu(&mut pc, 196, oct_res);
    putu(&mut pc, 200, flip_y);
    putu(&mut pc, 204, frame);
    putu(&mut pc, 208, max_steps);
    putu(&mut pc, 212, clip_desc);
    putu(&mut pc, 216, clip_count);
    putf(&mut pc, 220, clamp_max);
    // World radiance cache fallback (0xFFFFFFFF atlas = unbound). grid/oct describe the atlas.
    putu(&mut pc, 224, wrc_atlas);
    putu(&mut pc, 228, wrc_grid);
    putu(&mut pc, 232, wrc_oct);
    putu(&mut pc, 236, 0);
    pc
}

/// Pack the world radiance cache UPDATE push block (128 bytes): sun (16) + params (16) +
/// ground_albedo/+cone_k (16) + cache uint4 (16) + scalars (clip_desc, clip_count, grid, oct,
/// atlas_write, atlas_prev, cache_tile, max_steps, frame, reset, alpha, sample_clamp, ground_y +
/// pad). See `wrc_update.slang`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wrc_update_push(
    sun_dir: [f32; 3],
    sun_intensity: f32,
    ray_max: f32,
    bias: f32,
    sky_term: f32,
    albedo_fallback: f32,
    ground_albedo: [f32; 3],
    cone_k: f32,
    cache: [u32; 4],
    clip_desc: u32,
    clip_count: u32,
    grid: u32,
    oct: u32,
    atlas_write: u32,
    atlas_prev: u32,
    cache_tile: u32,
    max_steps: u32,
    frame: u32,
    reset: u32,
    alpha: f32,
    sample_clamp: f32,
    ground_y: f32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    let put3 = |pc: &mut [u8], o: usize, v: [f32; 3]| {
        for (i, x) in v.iter().enumerate() {
            pc[o + i * 4..o + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    let putu = |pc: &mut [u8], o: usize, v: u32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let putf = |pc: &mut [u8], o: usize, v: f32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let sun = normalize3(sun_dir);
    put3(&mut pc, 0, [sun[0], sun[1], sun[2]]);
    putf(&mut pc, 12, sun_intensity);
    putf(&mut pc, 16, ray_max);
    putf(&mut pc, 20, bias);
    putf(&mut pc, 24, sky_term);
    putf(&mut pc, 28, albedo_fallback);
    put3(&mut pc, 32, ground_albedo);
    putf(&mut pc, 44, cone_k);
    for (i, v) in cache.iter().enumerate() {
        putu(&mut pc, 48 + i * 4, *v);
    }
    putu(&mut pc, 64, clip_desc);
    putu(&mut pc, 68, clip_count);
    putu(&mut pc, 72, grid);
    putu(&mut pc, 76, oct);
    putu(&mut pc, 80, atlas_write);
    putu(&mut pc, 84, atlas_prev);
    putu(&mut pc, 88, cache_tile);
    putu(&mut pc, 92, max_steps);
    putu(&mut pc, 96, frame);
    putu(&mut pc, 100, reset);
    putf(&mut pc, 104, alpha);
    putf(&mut pc, 108, sample_clamp);
    putf(&mut pc, 112, ground_y);
    pc
}

/// Pack the screen-probe INTEGRATE push block (128 bytes): inv_view_proj (64) + 12 scalar
/// indices/dims (3 rows of 16) + (pos_sigma, normal_power, pad, pad) (16). See
/// `screen_probe_integrate.slang`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn screen_probe_integrate_push(
    inv_view_proj: &[f32; 16],
    depth_index: u32,
    normal_index: u32,
    atlas_index: u32,
    out_index: u32,
    screen_w: u32,
    screen_h: u32,
    probes_x: u32,
    probes_y: u32,
    downsample: u32,
    oct_res: u32,
    flip_y: u32,
    skyvis_index: u32,
    pos_sigma: f32,
    normal_power: f32,
    mode: u32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    let putu = |pc: &mut [u8], o: usize, v: u32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let putf = |pc: &mut [u8], o: usize, v: f32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    putu(&mut pc, 64, depth_index);
    putu(&mut pc, 68, normal_index);
    putu(&mut pc, 72, atlas_index);
    putu(&mut pc, 76, out_index);
    putu(&mut pc, 80, screen_w);
    putu(&mut pc, 84, screen_h);
    putu(&mut pc, 88, probes_x);
    putu(&mut pc, 92, probes_y);
    putu(&mut pc, 96, downsample);
    putu(&mut pc, 100, oct_res);
    putu(&mut pc, 104, flip_y);
    putu(&mut pc, 108, skyvis_index);
    putf(&mut pc, 112, pos_sigma);
    putf(&mut pc, 116, normal_power);
    putu(&mut pc, 120, mode);
    pc
}

/// Pack the screen-probe IRRADIANCE pre-integration push block (32 bytes): atlas_in, atlas_out,
/// probes_x, probes_y, oct + pad. See `screen_probe_irradiance.slang`.
pub(crate) fn screen_probe_irradiance_push(
    atlas_in: u32,
    atlas_out: u32,
    probes_x: u32,
    probes_y: u32,
    oct: u32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    let putu = |pc: &mut [u8], o: usize, v: u32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    putu(&mut pc, 0, atlas_in);
    putu(&mut pc, 4, atlas_out);
    putu(&mut pc, 8, probes_x);
    putu(&mut pc, 12, probes_y);
    putu(&mut pc, 16, oct);
    pc
}

/// Pack the screen-probe FILTER push block (128 bytes): inv_view_proj (64) + 12 scalar
/// indices/dims (3 rows of 16) + (pos_sigma, normal_power, pad, pad) (16). See
/// `screen_probe_filter.slang`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn screen_probe_filter_push(
    inv_view_proj: &[f32; 16],
    depth_index: u32,
    normal_index: u32,
    atlas_in_index: u32,
    atlas_out_index: u32,
    screen_w: u32,
    screen_h: u32,
    probes_x: u32,
    probes_y: u32,
    downsample: u32,
    oct_res: u32,
    flip_y: u32,
    half_kernel: u32,
    pos_sigma: f32,
    normal_power: f32,
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    let putu = |pc: &mut [u8], o: usize, v: u32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let putf = |pc: &mut [u8], o: usize, v: f32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    putu(&mut pc, 64, depth_index);
    putu(&mut pc, 68, normal_index);
    putu(&mut pc, 72, atlas_in_index);
    putu(&mut pc, 76, atlas_out_index);
    putu(&mut pc, 80, screen_w);
    putu(&mut pc, 84, screen_h);
    putu(&mut pc, 88, probes_x);
    putu(&mut pc, 92, probes_y);
    putu(&mut pc, 96, downsample);
    putu(&mut pc, 100, oct_res);
    putu(&mut pc, 104, flip_y);
    putu(&mut pc, 108, half_kernel);
    putf(&mut pc, 112, pos_sigma);
    putf(&mut pc, 116, normal_power);
    pc
}

/// Pack the GI-on-distance-field VIEW push block (192 bytes): inv_view_proj (64), cam_pos (16),
/// aabb_min/ground_y (16), aabb_max/sample_clamp (16), clay/gain (16), then scalars. See
/// `wrc_view.slang`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wrc_view_push(
    inv_view_proj: &[f32; 16],
    cam_pos: [f32; 3],
    aabb_min: [f32; 3],
    ground_y: f32,
    aabb_max: [f32; 3],
    sample_clamp: f32,
    clay: [f32; 3],
    gain: f32,
    out_index: u32,
    width: u32,
    height: u32,
    flip_y: u32,
    clip_desc: u32,
    clip_count: u32,
    wrc_atlas: u32,
    wrc_grid: u32,
    wrc_oct: u32,
    mode: u32,
    source: u32,
    sc: [u32; 5], // surface cache: cards, cache_pos, cache_rad, num_cards, tile (0xFFFFFFFF = off)
) -> [u8; 192] {
    let mut pc = [0u8; 192];
    let put3 = |pc: &mut [u8], o: usize, v: [f32; 3]| {
        for (i, x) in v.iter().enumerate() {
            pc[o + i * 4..o + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    let putu = |pc: &mut [u8], o: usize, v: u32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let putf = |pc: &mut [u8], o: usize, v: f32| pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    for (i, v) in inv_view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    put3(&mut pc, 64, cam_pos);
    put3(&mut pc, 80, aabb_min);
    putf(&mut pc, 92, ground_y);
    put3(&mut pc, 96, aabb_max);
    putf(&mut pc, 108, sample_clamp);
    put3(&mut pc, 112, clay);
    putf(&mut pc, 124, gain);
    putu(&mut pc, 128, out_index);
    putu(&mut pc, 132, width);
    putu(&mut pc, 136, height);
    putu(&mut pc, 140, flip_y);
    putu(&mut pc, 144, clip_desc);
    putu(&mut pc, 148, clip_count);
    putu(&mut pc, 152, wrc_atlas);
    putu(&mut pc, 156, wrc_grid);
    putu(&mut pc, 160, wrc_oct);
    putu(&mut pc, 164, mode);
    putu(&mut pc, 168, source);
    putu(&mut pc, 172, sc[0]); // cards
    putu(&mut pc, 176, sc[1]); // cache_pos
    putu(&mut pc, 180, sc[2]); // cache_rad
    putu(&mut pc, 184, sc[3]); // num_cards
    putu(&mut pc, 188, sc[4]); // tile
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
