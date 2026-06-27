//! The classic DXT / BCn family: BC1 (RGB), BC3 (RGBA = BC1 + BC4 alpha), BC4
//! (single channel), BC5 (two channels = two BC4). All share the 565 colour-block
//! and BC4 single-channel-block primitives.

use super::{blocks, texel};

// --- BC1 --------------------------------------------------------------------

#[inline]
fn rgb565(r: u8, g: u8, b: u8) -> u16 {
    ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3)
}

/// Expand an RGB565 endpoint to RGB888 the way the hardware does (bit replication),
/// so the encoder's index search matches what the GPU will reconstruct.
#[inline]
fn expand565(c: u16) -> [i32; 3] {
    let r = ((c >> 11) & 0x1f) as i32;
    let g = ((c >> 5) & 0x3f) as i32;
    let b = (c & 0x1f) as i32;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// Encode one 4×4 RGB block to BC1 (8 bytes). Endpoints are the per-channel min/max
/// (colour-bounding-box), which is the standard fast encoder; the 4-colour mode is
/// forced by ordering `c0 > c1`.
fn encode_bc1_block(rgba: &[u8], width: u32, height: u32, bx: u32, by: u32) -> [u8; 8] {
    let (mut lo, mut hi) = ([255i32; 3], [0i32; 3]);
    for j in 0..4 {
        for i in 0..4 {
            let p = texel(rgba, width, height, bx * 4 + i, by * 4 + j);
            for c in 0..3 {
                lo[c] = lo[c].min(p[c] as i32);
                hi[c] = hi[c].max(p[c] as i32);
            }
        }
    }
    let mut c0 = rgb565(hi[0] as u8, hi[1] as u8, hi[2] as u8);
    let mut c1 = rgb565(lo[0] as u8, lo[1] as u8, lo[2] as u8);
    // 4-colour mode needs c0 > c1; if equal (flat block) nudge so indices are valid.
    if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    if c0 == c1 {
        // Flat block: every pixel maps to endpoint 0; keep c0 >= c1.
        if c0 > 0 {
            c1 = c0 - 1;
        } else {
            c0 = 1;
        }
    }

    // The four palette entries: c0, c1, 2/3·c0+1/3·c1, 1/3·c0+2/3·c1.
    let e0 = expand565(c0);
    let e1 = expand565(c1);
    let palette = [
        e0,
        e1,
        [
            (2 * e0[0] + e1[0]) / 3,
            (2 * e0[1] + e1[1]) / 3,
            (2 * e0[2] + e1[2]) / 3,
        ],
        [
            (e0[0] + 2 * e1[0]) / 3,
            (e0[1] + 2 * e1[1]) / 3,
            (e0[2] + 2 * e1[2]) / 3,
        ],
    ];

    let mut indices = 0u32;
    for j in 0..4 {
        for i in 0..4 {
            let p = texel(rgba, width, height, bx * 4 + i, by * 4 + j);
            let mut best = 0u32;
            let mut best_d = i32::MAX;
            for (k, pe) in palette.iter().enumerate() {
                let dr = p[0] as i32 - pe[0];
                let dg = p[1] as i32 - pe[1];
                let db = p[2] as i32 - pe[2];
                let d = dr * dr + dg * dg + db * db;
                if d < best_d {
                    best_d = d;
                    best = k as u32;
                }
            }
            indices |= best << (2 * (j * 4 + i));
        }
    }

    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&indices.to_le_bytes());
    out
}

/// Compress an RGBA8 image to BC1 (RGB; alpha ignored). Output is
/// `blocks(w)*blocks(h)*8` bytes, row-major by block.
pub fn encode_bc1(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (bw, bh) = (blocks(width), blocks(height));
    let mut out = Vec::with_capacity((bw * bh) as usize * 8);
    for by in 0..bh {
        for bx in 0..bw {
            out.extend_from_slice(&encode_bc1_block(rgba, width, height, bx, by));
        }
    }
    out
}

// --- BC4 / BC5 / BC3 --------------------------------------------------------

/// Encode one 4×4 single-channel block to BC4 (8 bytes): two 8-bit endpoints + a
/// 3-bit index per texel. `ch` selects the source channel (0=R, 1=G, 3=A).
fn encode_bc4_block(rgba: &[u8], width: u32, height: u32, bx: u32, by: u32, ch: usize) -> [u8; 8] {
    let (mut lo, mut hi) = (255i32, 0i32);
    for j in 0..4 {
        for i in 0..4 {
            let v = texel(rgba, width, height, bx * 4 + i, by * 4 + j)[ch] as i32;
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    // 8-value mode needs r0 > r1; r0 = hi, r1 = lo.
    let (r0, r1) = if hi > lo {
        (hi, lo)
    } else {
        // Flat block: keep r0 > r1 with a valid 1-step span.
        ((lo + 1).min(255), lo.max(1) - 1)
    };
    // Palette: r0, r1, then 6 evenly-spaced values between (8-value mode).
    let mut pal = [0i32; 8];
    pal[0] = r0;
    pal[1] = r1;
    for k in 1..7 {
        pal[k + 1] = ((7 - k as i32) * r0 + k as i32 * r1) / 7;
    }

    let mut bits = 0u64;
    for j in 0..4 {
        for i in 0..4 {
            let v = texel(rgba, width, height, bx * 4 + i, by * 4 + j)[ch] as i32;
            let mut best = 0u64;
            let mut best_d = i32::MAX;
            for (k, &pv) in pal.iter().enumerate() {
                let d = (v - pv).abs();
                if d < best_d {
                    best_d = d;
                    best = k as u64;
                }
            }
            bits |= best << (3 * (j * 4 + i));
        }
    }

    let mut out = [0u8; 8];
    out[0] = r0 as u8;
    out[1] = r1 as u8;
    // 48 bits of indices into bytes 2..8.
    out[2..8].copy_from_slice(&bits.to_le_bytes()[0..6]);
    out
}

/// Compress an RGBA8 image to BC5 (R,G channels). Output is
/// `blocks(w)*blocks(h)*16` bytes (BC4-R block followed by BC4-G block).
pub fn encode_bc5(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (bw, bh) = (blocks(width), blocks(height));
    let mut out = Vec::with_capacity((bw * bh) as usize * 16);
    for by in 0..bh {
        for bx in 0..bw {
            out.extend_from_slice(&encode_bc4_block(rgba, width, height, bx, by, 0));
            out.extend_from_slice(&encode_bc4_block(rgba, width, height, bx, by, 1));
        }
    }
    out
}

/// Compress an RGBA8 image to BC3 / DXT5 (RGBA): a BC4 alpha block followed by a
/// BC1 colour block, 16 bytes / 4×4 block. Unlike BC1 this keeps real alpha.
pub fn encode_bc3(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (bw, bh) = (blocks(width), blocks(height));
    let mut out = Vec::with_capacity((bw * bh) as usize * 16);
    for by in 0..bh {
        for bx in 0..bw {
            out.extend_from_slice(&encode_bc4_block(rgba, width, height, bx, by, 3)); // alpha
            out.extend_from_slice(&encode_bc1_block(rgba, width, height, bx, by)); // colour
        }
    }
    out
}

/// Compress the R channel of an RGBA8 image to BC4 (single channel), 8 bytes /
/// 4×4 block. For grayscale masks / single-channel data.
pub fn encode_bc4(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (bw, bh) = (blocks(width), blocks(height));
    let mut out = Vec::with_capacity((bw * bh) as usize * 8);
    for by in 0..bh {
        for bx in 0..bw {
            out.extend_from_slice(&encode_bc4_block(rgba, width, height, bx, by, 0));
        }
    }
    out
}

// --- decode (used by the parent's `decode_block_rgba8`) ---------------------

/// Decode a BC1 colour block to its 16 RGB texels (shared by BC1 / BC3 decode).
pub(super) fn decode_bc1_colors(block: &[u8]) -> [[u8; 3]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let idx = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let e0 = expand565(c0);
    let e1 = expand565(c1);
    let pal = [
        e0,
        e1,
        [
            (2 * e0[0] + e1[0]) / 3,
            (2 * e0[1] + e1[1]) / 3,
            (2 * e0[2] + e1[2]) / 3,
        ],
        [
            (e0[0] + 2 * e1[0]) / 3,
            (e0[1] + 2 * e1[1]) / 3,
            (e0[2] + 2 * e1[2]) / 3,
        ],
    ];
    let mut out = [[0u8; 3]; 16];
    for (p, o) in out.iter_mut().enumerate() {
        let k = ((idx >> (2 * p)) & 3) as usize;
        *o = [pal[k][0] as u8, pal[k][1] as u8, pal[k][2] as u8];
    }
    out
}

/// Decode one BC4 block to its 16 single-channel values.
pub(super) fn decode_bc4_block(block: &[u8]) -> [u8; 16] {
    let r0 = block[0] as i32;
    let r1 = block[1] as i32;
    let mut bits = 0u64;
    for (k, &b) in block[2..8].iter().enumerate() {
        bits |= (b as u64) << (8 * k);
    }
    let mut pal = [0i32; 8];
    pal[0] = r0;
    pal[1] = r1;
    for k in 1..7 {
        pal[k + 1] = ((7 - k as i32) * r0 + k as i32 * r1) / 7;
    }
    let mut out = [0u8; 16];
    for (p, o) in out.iter_mut().enumerate() {
        let idx = ((bits >> (3 * p)) & 7) as usize;
        *o = pal[idx] as u8;
    }
    out
}
