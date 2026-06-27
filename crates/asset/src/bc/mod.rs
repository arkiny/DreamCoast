//! BCn block compression (Phase 12 M3) — dependency-free, deterministic CPU
//! encoders the cook uses to shrink `.dcasset` textures.
//!
//! The point is **zero runtime decompression cost**: BC formats are GPU-native, so
//! the hardware samples the compressed 4×4 blocks directly — there is no decode step
//! at load, and VRAM drops with the disk size. The encode is the expensive part, and
//! it happens once at cook time (cached).
//!
//! Everything is integer/bounded-float and order-independent, so the bytes are
//! deterministic and identical across machines — both backends upload the same
//! blocks, preserving cross-backend parity.
//!
//! Submodules: [`dxt`] (the classic BC1/BC3/BC4/BC5 family), [`bc7`] (high-quality
//! BC7 mode 6).

mod bc7;
mod dxt;

pub use bc7::encode_bc7;
pub use dxt::{encode_bc1, encode_bc3, encode_bc4, encode_bc5};

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
pub(crate) fn texel(rgba: &[u8], width: u32, height: u32, x: u32, y: u32) -> [u8; 4] {
    let x = x.min(width - 1) as usize;
    let y = y.min(height - 1) as usize;
    let i = (y * width as usize + x) * 4;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

/// Decode a single 4×4 block (the smallest mip of a cooked texture) to a 16-texel
/// RGBA8 buffer. Used to recover a representative average colour cheaply, without
/// decompressing a full image. BC5 fills R,G; BC4 fills R; BC3 decodes colour+alpha.
pub fn decode_block_rgba8(fmt: BcFormat, block: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 4];
    match fmt {
        BcFormat::Bc1 => {
            let rgb = dxt::decode_bc1_colors(block);
            for p in 0..16 {
                out[p * 4..p * 4 + 3].copy_from_slice(&rgb[p]);
                out[p * 4 + 3] = 255;
            }
        }
        BcFormat::Bc3 => {
            // 8 bytes alpha (BC4), then 8 bytes colour (BC1).
            let alpha = dxt::decode_bc4_block(&block[0..8]);
            let rgb = dxt::decode_bc1_colors(&block[8..16]);
            for p in 0..16 {
                out[p * 4..p * 4 + 3].copy_from_slice(&rgb[p]);
                out[p * 4 + 3] = alpha[p];
            }
        }
        BcFormat::Bc4 => {
            let r = dxt::decode_bc4_block(block);
            for p in 0..16 {
                out[p * 4] = r[p];
                out[p * 4 + 3] = 255;
            }
        }
        BcFormat::Bc5 => {
            let r = dxt::decode_bc4_block(&block[0..8]);
            let g = dxt::decode_bc4_block(&block[8..16]);
            for p in 0..16 {
                out[p * 4] = r[p];
                out[p * 4 + 1] = g[p];
                out[p * 4 + 3] = 255;
            }
        }
        BcFormat::Bc7 => return bc7::decode_bc7_block(block),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Decode a whole BCn image (block by block, via the public dispatch).
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

    /// A 2D RGB gradient (R varies with x, G with y) — the hard case for a single
    /// endpoint line, where BC7 mode 6 clearly beats BC1.
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
    fn bc1_size_and_quality() {
        let (w, h) = (16, 16);
        let src = gradient(w, h);
        let enc = encode_bc1(&src, w, h);
        assert_eq!(enc.len(), compressed_len(BcFormat::Bc1, w, h));
        assert_eq!(enc.len(), src.len() / 8); // 8:1 vs RGBA8
        let dec = decode_full(BcFormat::Bc1, &enc, w, h);
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

        let dec = decode_full(BcFormat::Bc3, &enc, w, h);
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
        let dec = decode_full(BcFormat::Bc5, &enc, w, h);
        assert!(rmse(&src, &dec, &[0, 1]) < 6.0, "BC5 RG RMSE too high");
    }

    #[test]
    fn bc7_beats_bc1_on_colour() {
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
    fn non_multiple_of_four_dims() {
        // Partial edge blocks must encode without panicking and round-trip.
        let (w, h) = (5, 3);
        let src = gradient(w, h);
        let enc = encode_bc1(&src, w, h);
        assert_eq!(enc.len(), compressed_len(BcFormat::Bc1, w, h));
        let _ = decode_full(BcFormat::Bc1, &enc, w, h);
    }
}
