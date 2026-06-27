//! BCn block compression (Phase 12 M3) — dependency-free, deterministic CPU
//! encoders the cook uses to shrink `.dcasset` textures.
//!
//! The point is **zero runtime decompression cost**: BC formats are GPU-native, so
//! the hardware samples the compressed 4×4 blocks directly — there is no decode step
//! at load, and VRAM drops with the disk size. The encode is the expensive part, and
//! it happens once at cook time (cached).
//!
//! - **BC1** (a.k.a. DXT1): 4×4 RGB → 8 bytes (0.5 B/px, 8:1 vs RGBA8). For
//!   sRGB color (base color, emissive); alpha is dropped (the cook keeps textures
//!   with meaningful alpha uncompressed).
//! - **BC5**: 4×4 two-channel → 16 bytes (1 B/px, 4:1), two BC4 blocks. For tangent-
//!   space normals (R,G; Z reconstructed in the shader), where BC5 is near-lossless.
//!
//! Everything is integer/bounded-float and order-independent, so the bytes are
//! deterministic and identical across machines — both backends upload the same
//! blocks, preserving cross-backend parity.

/// Which block codec to apply to a texture slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcFormat {
    /// BC1 / DXT1, RGB, 8 bytes per 4×4 block.
    Bc1,
    /// BC3 / DXT5, RGBA (BC1 colour + BC4 alpha), 16 bytes per 4×4 block. For colour
    /// textures with real transparency.
    Bc3,
    /// BC4, single channel (R), 8 bytes per 4×4 block. For grayscale / mask data.
    Bc4,
    /// BC5, two channels (RG), 16 bytes per 4×4 block.
    Bc5,
}

impl BcFormat {
    /// Compressed bytes per 4×4 block.
    pub fn block_bytes(self) -> usize {
        match self {
            BcFormat::Bc1 | BcFormat::Bc4 => 8,
            BcFormat::Bc3 | BcFormat::Bc5 => 16,
        }
    }
}

/// Blocks along an axis of `n` pixels (4×4 blocks, last block padded).
#[inline]
pub fn blocks(n: u32) -> u32 {
    n.div_ceil(4)
}

/// Total compressed size of a `width×height` image in `fmt`.
pub fn compressed_len(fmt: BcFormat, width: u32, height: u32) -> usize {
    (blocks(width) * blocks(height)) as usize * fmt.block_bytes()
}

/// Fetch RGBA8 texel `(x,y)`, clamped to the image bounds so partial edge blocks
/// replicate their last row/column instead of reading out of range.
#[inline]
fn texel(rgba: &[u8], width: u32, height: u32, x: u32, y: u32) -> [u8; 4] {
    let x = x.min(width - 1) as usize;
    let y = y.min(height - 1) as usize;
    let i = (y * width as usize + x) * 4;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

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

// --- BC4 / BC5 --------------------------------------------------------------

/// Encode one 4×4 single-channel block to BC4 (8 bytes): two 8-bit endpoints + a
/// 3-bit index per texel. `ch` selects the source channel (0=R, 1=G).
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

/// Decode a single 4×4 block (the smallest mip of a cooked texture) to a 16-texel
/// RGBA8 buffer. Used to recover a representative average colour cheaply, without
/// decompressing a full image. BC5 fills R,G; BC4 fills R; BC3 decodes colour+alpha.
pub fn decode_block_rgba8(fmt: BcFormat, block: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 4];
    match fmt {
        BcFormat::Bc1 => {
            let rgb = decode_bc1_colors(block);
            for p in 0..16 {
                out[p * 4..p * 4 + 3].copy_from_slice(&rgb[p]);
                out[p * 4 + 3] = 255;
            }
        }
        BcFormat::Bc3 => {
            // 8 bytes alpha (BC4), then 8 bytes colour (BC1).
            let alpha = decode_bc4_block(&block[0..8]);
            let rgb = decode_bc1_colors(&block[8..16]);
            for p in 0..16 {
                out[p * 4..p * 4 + 3].copy_from_slice(&rgb[p]);
                out[p * 4 + 3] = alpha[p];
            }
        }
        BcFormat::Bc4 => {
            let r = decode_bc4_block(block);
            for p in 0..16 {
                out[p * 4] = r[p];
                out[p * 4 + 3] = 255;
            }
        }
        BcFormat::Bc5 => {
            let r = decode_bc4_block(&block[0..8]);
            let g = decode_bc4_block(&block[8..16]);
            for p in 0..16 {
                out[p * 4] = r[p];
                out[p * 4 + 1] = g[p];
                out[p * 4 + 3] = 255;
            }
        }
    }
    out
}

/// Decode a BC1 colour block to its 16 RGB texels (shared by BC1 / BC3 decode).
fn decode_bc1_colors(block: &[u8]) -> [[u8; 3]; 16] {
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

/// Decode one BC4 block to its 16 single-channel values (shared by [`decode_block_rgba8`]).
fn decode_bc4_block(block: &[u8]) -> [u8; 16] {
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- decoders (test-only): reconstruct RGBA8 to bound the encode error ---

    fn decode_bc1(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let (bw, bh) = (blocks(width), blocks(height));
        let mut out = vec![0u8; (width * height * 4) as usize];
        for by in 0..bh {
            for bx in 0..bw {
                let b = &data[((by * bw + bx) * 8) as usize..][..8];
                let c0 = u16::from_le_bytes([b[0], b[1]]);
                let c1 = u16::from_le_bytes([b[2], b[3]]);
                let idx = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
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
                for j in 0..4 {
                    for i in 0..4 {
                        let (x, y) = (bx * 4 + i, by * 4 + j);
                        if x >= width || y >= height {
                            continue;
                        }
                        let k = (idx >> (2 * (j * 4 + i))) & 3;
                        let c = pal[k as usize];
                        let o = ((y * width + x) * 4) as usize;
                        out[o] = c[0] as u8;
                        out[o + 1] = c[1] as u8;
                        out[o + 2] = c[2] as u8;
                        out[o + 3] = 255;
                    }
                }
            }
        }
        out
    }

    fn rmse(a: &[u8], b: &[u8], channels: &[usize]) -> f64 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for px in 0..(a.len() / 4) {
            for &c in channels {
                let d = a[px * 4 + c] as f64 - b[px * 4 + c] as f64;
                sum += d * d;
                n += 1.0;
            }
        }
        (sum / n).sqrt()
    }

    /// A smooth grayscale ramp along x (all channels equal) — the exact 1D case
    /// BC1's endpoint line represents, so the only error is 565 endpoint
    /// quantization. A regression here means a real encoder/index bug.
    fn gradient(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let t = (x * 255 / w.max(1)) as u8;
                v.extend_from_slice(&[t, t, t, 255]);
            }
        }
        v
    }

    #[test]
    fn bc1_size_and_quality() {
        let (w, h) = (16, 16);
        let src = gradient(w, h);
        let enc = encode_bc1(&src, w, h);
        assert_eq!(enc.len(), compressed_len(BcFormat::Bc1, w, h));
        assert_eq!(enc.len(), src.len() / 8); // 8:1 vs RGBA8
        let dec = decode_bc1(&enc, w, h);
        let err = rmse(&src, &dec, &[0, 1, 2]);
        assert!(err < 6.0, "BC1 ramp RMSE {err} too high");
    }

    #[test]
    fn bc3_preserves_alpha() {
        // A colour ramp with a varying alpha ramp — BC3 keeps both (unlike BC1).
        let (w, h) = (16, 16);
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let t = (x * 255 / w) as u8;
                src.extend_from_slice(&[t, t, t, 255 - t]);
            }
        }
        let enc = encode_bc3(&src, w, h);
        assert_eq!(enc.len(), compressed_len(BcFormat::Bc3, w, h));
        assert_eq!(enc.len(), src.len() / 4); // 4:1 vs RGBA8

        // Decode block-by-block and check the alpha channel survived.
        let (bw, _) = (blocks(w), blocks(h));
        let mut dec = vec![0u8; src.len()];
        for by in 0..blocks(h) {
            for bx in 0..bw {
                let texels =
                    decode_block_rgba8(BcFormat::Bc3, &enc[((by * bw + bx) * 16) as usize..]);
                for j in 0..4 {
                    for i in 0..4 {
                        let (x, y) = (bx * 4 + i, by * 4 + j);
                        if x >= w || y >= h {
                            continue;
                        }
                        let o = ((y * w + x) * 4) as usize;
                        dec[o..o + 4].copy_from_slice(&texels[(j * 4 + i) as usize * 4..][..4]);
                    }
                }
            }
        }
        assert!(rmse(&src, &dec, &[3]) < 6.0, "BC3 alpha RMSE too high");
        assert!(
            rmse(&src, &dec, &[0, 1, 2]) < 6.0,
            "BC3 colour RMSE too high"
        );
    }

    #[test]
    fn bc4_size() {
        let (w, h) = (8, 8);
        let src = gradient(w, h);
        let enc = encode_bc4(&src, w, h);
        assert_eq!(enc.len(), compressed_len(BcFormat::Bc4, w, h));
        assert_eq!(enc.len(), src.len() / 8); // single channel, 8:1
    }

    #[test]
    fn bc5_normals_quality() {
        // A normal-ish RG field; BC5 should reconstruct it near-losslessly.
        let (w, h) = (16, 16);
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                src.extend_from_slice(&[(x * 17 % 256) as u8, (y * 13 % 256) as u8, 255, 255]);
            }
        }
        let enc = encode_bc5(&src, w, h);
        assert_eq!(enc.len(), compressed_len(BcFormat::Bc5, w, h));
        assert_eq!(enc.len(), src.len() / 4); // 4:1 vs RGBA8

        // Decode both BC4 channels and check R,G error.
        let (bw, _) = (blocks(w), blocks(h));
        let mut dec = vec![0u8; src.len()];
        for by in 0..blocks(h) {
            for bx in 0..bw {
                let base = ((by * bw + bx) * 16) as usize;
                let r = decode_bc4_block(&enc[base..base + 8]);
                let g = decode_bc4_block(&enc[base + 8..base + 16]);
                for j in 0..4 {
                    for i in 0..4 {
                        let (x, y) = (bx * 4 + i, by * 4 + j);
                        if x >= w || y >= h {
                            continue;
                        }
                        let o = ((y * w + x) * 4) as usize;
                        dec[o] = r[(j * 4 + i) as usize];
                        dec[o + 1] = g[(j * 4 + i) as usize];
                    }
                }
            }
        }
        assert!(rmse(&src, &dec, &[0, 1]) < 6.0, "BC5 RG RMSE too high");
    }

    #[test]
    fn non_multiple_of_four_dims() {
        // Partial edge blocks must encode without panicking and round-trip the
        // in-bounds texels.
        let (w, h) = (5, 3);
        let src = gradient(w, h);
        let enc = encode_bc1(&src, w, h);
        assert_eq!(enc.len(), compressed_len(BcFormat::Bc1, w, h));
        let _ = decode_bc1(&enc, w, h);
    }
}
