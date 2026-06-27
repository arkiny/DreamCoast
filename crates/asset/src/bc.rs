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
    /// BC7 (mode 6), high-quality RGBA, 16 bytes per 4×4 block.
    Bc7,
}

impl BcFormat {
    /// Compressed bytes per 4×4 block.
    pub fn block_bytes(self) -> usize {
        match self {
            BcFormat::Bc1 | BcFormat::Bc4 => 8,
            BcFormat::Bc3 | BcFormat::Bc5 | BcFormat::Bc7 => 16,
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

// --- BC7 (mode 6 only) ------------------------------------------------------
//
// BC7 is the high-quality 4:1 RGBA codec. Full BC7 has 8 modes/partitions; this
// encoder emits **mode 6 only** — a single subset, 2 RGBA endpoints (7 bits + a
// per-endpoint p-bit = 8-bit precision), and a 4-bit (16-weight) index per texel.
// Mode 6 alone is markedly better than BC1 on photographic colour and keeps alpha,
// at 16 bytes/block (vs BC1's 8). More modes are a future quality refinement.

/// BC7 4-bit interpolation weights (the `aWeight4` table).
const BC7_W4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

#[inline]
fn bc7_interp(a: i32, b: i32, w: i32) -> i32 {
    (a * (64 - w) + b * w + 32) >> 6
}

/// Quantize an 8-bit RGBA endpoint to 7 bits + one shared p-bit (mode 6 precision).
/// Returns the 7-bit channels and the p-bit minimizing reconstruction error.
fn bc7_quantize_endpoint(v: [i32; 4]) -> ([i32; 4], i32) {
    let mut best = ([0i32; 4], 0i32);
    let mut best_err = i32::MAX;
    for p in 0..2 {
        let mut q = [0i32; 4];
        let mut err = 0;
        for c in 0..4 {
            let qq = (((v[c] - p).max(0) + 1) >> 1).clamp(0, 127);
            let recon = (qq << 1) | p;
            err += (recon - v[c]) * (recon - v[c]);
            q[c] = qq;
        }
        if err < best_err {
            best_err = err;
            best = (q, p);
        }
    }
    best
}

/// Encode one 4×4 RGBA block to BC7 mode 6 (16 bytes).
fn encode_bc7_block(rgba: &[u8], width: u32, height: u32, bx: u32, by: u32) -> [u8; 16] {
    let mut px = [[0i32; 4]; 16];
    let mut lo = [255i32; 4];
    let mut hi = [0i32; 4];
    for j in 0..4 {
        for i in 0..4 {
            let t = texel(rgba, width, height, bx * 4 + i, by * 4 + j);
            let p = [t[0] as i32, t[1] as i32, t[2] as i32, t[3] as i32];
            px[(j * 4 + i) as usize] = p;
            for c in 0..4 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
    }

    let (mut q0, mut p0) = bc7_quantize_endpoint(lo);
    let (mut q1, mut p1) = bc7_quantize_endpoint(hi);
    let recon = |q: [i32; 4], p: i32| {
        [
            (q[0] << 1) | p,
            (q[1] << 1) | p,
            (q[2] << 1) | p,
            (q[3] << 1) | p,
        ]
    };
    let mut e0 = recon(q0, p0);
    let mut e1 = recon(q1, p1);

    let pick = |e0: [i32; 4], e1: [i32; 4], p: [i32; 4]| -> usize {
        let mut best = 0;
        let mut best_d = i32::MAX;
        for (w, &weight) in BC7_W4.iter().enumerate() {
            let mut d = 0;
            for c in 0..4 {
                let s = bc7_interp(e0[c], e1[c], weight) - p[c];
                d += s * s;
            }
            if d < best_d {
                best_d = d;
                best = w;
            }
        }
        best
    };
    let mut idx = [0usize; 16];
    for (n, p) in px.iter().enumerate() {
        idx[n] = pick(e0, e1, *p);
    }

    // Anchor: pixel 0's index high bit must be 0. If not, swap endpoints + invert.
    if idx[0] >= 8 {
        std::mem::swap(&mut q0, &mut q1);
        std::mem::swap(&mut p0, &mut p1);
        std::mem::swap(&mut e0, &mut e1);
        for v in &mut idx {
            *v = 15 - *v;
        }
    }

    // Pack 128 bits LSB-first: mode(7) | R0 R1 G0 G1 B0 B1 A0 A1 (7 each) | p0 p1 |
    // index[0] (3 bits, anchor) | index[1..16] (4 bits).
    let mut bits = 0u128;
    let mut pos = 0u32;
    let mut put = |val: u128, n: u32| {
        bits |= (val & ((1u128 << n) - 1)) << pos;
        pos += n;
    };
    put(1 << 6, 7); // mode 6 marker
    for c in 0..4 {
        put(q0[c] as u128, 7);
        put(q1[c] as u128, 7);
    }
    put(p0 as u128, 1);
    put(p1 as u128, 1);
    put(idx[0] as u128, 3);
    for &i in &idx[1..] {
        put(i as u128, 4);
    }
    bits.to_le_bytes()
}

/// Compress an RGBA8 image to BC7 (mode 6), 16 bytes / 4×4 block. High-quality RGBA
/// (keeps alpha); the GPU samples it natively.
pub fn encode_bc7(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (bw, bh) = (blocks(width), blocks(height));
    let mut out = Vec::with_capacity((bw * bh) as usize * 16);
    for by in 0..bh {
        for bx in 0..bw {
            out.extend_from_slice(&encode_bc7_block(rgba, width, height, bx, by));
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
        BcFormat::Bc7 => return decode_bc7_block(block),
    }
    out
}

/// Decode a BC7 **mode 6** block to 16 RGBA8 texels. Mirrors [`encode_bc7_block`];
/// other modes are not produced by this encoder (and would need their own decode).
fn decode_bc7_block(block: &[u8]) -> Vec<u8> {
    let bits = u128::from_le_bytes(block[..16].try_into().unwrap());
    let mut pos = 0u32;
    let mut get = |n: u32| -> u128 {
        let v = (bits >> pos) & ((1u128 << n) - 1);
        pos += n;
        v
    };
    let _mode = get(7); // mode 6 marker (assumed)
    let mut q0 = [0i32; 4];
    let mut q1 = [0i32; 4];
    for c in 0..4 {
        q0[c] = get(7) as i32;
        q1[c] = get(7) as i32;
    }
    let p0 = get(1) as i32;
    let p1 = get(1) as i32;
    let e0 = [
        (q0[0] << 1) | p0,
        (q0[1] << 1) | p0,
        (q0[2] << 1) | p0,
        (q0[3] << 1) | p0,
    ];
    let e1 = [
        (q1[0] << 1) | p1,
        (q1[1] << 1) | p1,
        (q1[2] << 1) | p1,
        (q1[3] << 1) | p1,
    ];
    let mut idx = [0usize; 16];
    idx[0] = get(3) as usize;
    for i in idx.iter_mut().skip(1) {
        *i = get(4) as usize;
    }
    let mut out = vec![0u8; 16 * 4];
    for (p, &i) in idx.iter().enumerate() {
        let w = BC7_W4[i];
        for c in 0..4 {
            out[p * 4 + c] = bc7_interp(e0[c], e1[c], w) as u8;
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

    fn decode_full(fmt: BcFormat, enc: &[u8], w: u32, h: u32) -> Vec<u8> {
        let bb = fmt.block_bytes();
        let bw = blocks(w);
        let mut dec = vec![0u8; (w * h * 4) as usize];
        for by in 0..blocks(h) {
            for bx in 0..bw {
                let texels = decode_block_rgba8(fmt, &enc[((by * bw + bx) as usize) * bb..]);
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
        dec
    }

    /// A 2D RGB gradient (R varies with x, G with y) — the hard case for a single
    /// endpoint line, where BC7 mode 6's 8-bit endpoints + 4-bit indices clearly
    /// beat BC1's 565 + 2-bit.
    fn rgb_field(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&[
                    (x * 255 / w.max(1)) as u8,
                    (y * 255 / h.max(1)) as u8,
                    128,
                    255,
                ]);
            }
        }
        v
    }

    #[test]
    fn bc7_beats_bc1_on_colour() {
        // A 2D RGB gradient: BC7 mode 6's 8-bit endpoints + 4-bit indices reconstruct
        // it more accurately than BC1's 565 + 2-bit.
        let (w, h) = (16, 16);
        let src = rgb_field(w, h);
        let bc7 = encode_bc7(&src, w, h);
        assert_eq!(bc7.len(), compressed_len(BcFormat::Bc7, w, h));

        let d7 = decode_full(BcFormat::Bc7, &bc7, w, h);
        let d1 = decode_full(BcFormat::Bc1, &encode_bc1(&src, w, h), w, h);
        let e7 = rmse(&src, &d7, &[0, 1, 2]);
        let e1 = rmse(&src, &d1, &[0, 1, 2]);
        assert!(e7 < e1, "BC7 RMSE {e7} should beat BC1 {e1}");
    }

    #[test]
    fn bc7_carries_alpha() {
        // Colour and alpha varying together along x (the case mode 6's single shared
        // index represents well) — confirms alpha is preserved, not dropped like BC1.
        let (w, h) = (16, 16);
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let t = (x * 255 / w) as u8;
                src.extend_from_slice(&[t, t, t, t]);
            }
        }
        let d7 = decode_full(BcFormat::Bc7, &encode_bc7(&src, w, h), w, h);
        let ea = rmse(&src, &d7, &[3]);
        assert!(ea < 4.0, "BC7 alpha RMSE {ea} too high");
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
