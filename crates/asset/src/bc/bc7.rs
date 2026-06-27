//! BC7 (mode 6 only).
//!
//! BC7 is the high-quality 4:1 RGBA codec. Full BC7 has 8 modes/partitions; this
//! encoder emits **mode 6 only** — a single subset, 2 RGBA endpoints (7 bits + a
//! per-endpoint p-bit = 8-bit precision), and a 4-bit (16-weight) index per texel.
//! Mode 6 alone is markedly better than BC1 on photographic colour and keeps alpha,
//! at 16 bytes/block (vs BC1's 8). More modes are a future quality refinement.

use super::{blocks, texel};

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

/// Decode a BC7 **mode 6** block to 16 RGBA8 texels. Mirrors [`encode_bc7_block`];
/// other modes are not produced by this encoder (and would need their own decode).
pub(super) fn decode_bc7_block(block: &[u8]) -> Vec<u8> {
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
