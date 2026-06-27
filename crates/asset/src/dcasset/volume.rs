//! SDF + per-voxel albedo volume chunk codecs (Phase 12 M2). Each is a single-chunk
//! `.dcasset` (no mesh), framed by the shared [`super::write_single_chunk`] /
//! [`super::open_chunk`] helpers.

use dreamcoast_core::EngineError;

use super::{CHUNK_ALBEDO, CHUNK_SDF, Header, Reader, Writer, open_chunk, write_single_chunk};
use crate::sdf::{AlbedoVolumes, SdfVolume};

/// Serialize an SDF volume into a `.dcasset`. `src_hash` is the invalidation key the
/// loader compares against the live source (here, the fused-geometry + grid hash).
pub fn write_sdf(vol: &SdfVolume, src_hash: u64) -> Vec<u8> {
    write_single_chunk(CHUNK_SDF, &encode_sdf(vol), src_hash)
}

/// Decode a `.dcasset` buffer's SDF chunk into its [`Header`] and [`SdfVolume`].
pub fn read_sdf(bytes: &[u8]) -> Result<(Header, SdfVolume), EngineError> {
    let (header, mut r) = open_chunk(bytes, CHUNK_SDF, "sdf")?;
    Ok((header, decode_sdf(&mut r)?))
}

/// Serialize the per-voxel albedo volumes into a `.dcasset`. `src_hash` is the
/// invalidation key (the fused-geometry + per-triangle albedo + grid hash).
pub fn write_albedo(vol: &AlbedoVolumes, src_hash: u64) -> Vec<u8> {
    write_single_chunk(CHUNK_ALBEDO, &encode_albedo(vol), src_hash)
}

/// Decode a `.dcasset` buffer's albedo chunk into its [`Header`] and [`AlbedoVolumes`].
pub fn read_albedo(bytes: &[u8]) -> Result<(Header, AlbedoVolumes), EngineError> {
    let (header, mut r) = open_chunk(bytes, CHUNK_ALBEDO, "albedo")?;
    Ok((header, decode_albedo(&mut r)?))
}

/// Encode the SDF chunk payload: `dim`, `aabb_min`, `aabb_max`, then the voxels.
fn encode_sdf(vol: &SdfVolume) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(vol.dim);
    for c in vol.aabb_min {
        w.f32(c);
    }
    for c in vol.aabb_max {
        w.f32(c);
    }
    for &v in &vol.voxels {
        w.f32(v);
    }
    w.buf
}

/// Decode an SDF chunk payload, validating the voxel count against `dim³`.
fn decode_sdf(r: &mut Reader) -> Result<SdfVolume, EngineError> {
    let dim = r.u32()?;
    let aabb_min = [r.f32()?, r.f32()?, r.f32()?];
    let aabb_max = [r.f32()?, r.f32()?, r.f32()?];
    let count = (dim as usize)
        .checked_pow(3)
        .ok_or_else(|| EngineError::Asset("dcasset: sdf dim overflow".into()))?;
    let mut voxels = Vec::with_capacity(count);
    for _ in 0..count {
        voxels.push(r.f32()?);
    }
    Ok(SdfVolume {
        dim,
        aabb_min,
        aabb_max,
        voxels,
    })
}

/// Encode the albedo chunk payload: `dim`, then the R, G, B channels in order.
fn encode_albedo(vol: &AlbedoVolumes) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(vol.dim);
    for ch in &vol.channels {
        for &v in ch {
            w.f32(v);
        }
    }
    w.buf
}

/// Decode an albedo chunk payload (`dim` + three `dim³` channels).
fn decode_albedo(r: &mut Reader) -> Result<AlbedoVolumes, EngineError> {
    let dim = r.u32()?;
    let count = (dim as usize)
        .checked_pow(3)
        .ok_or_else(|| EngineError::Asset("dcasset: albedo dim overflow".into()))?;
    let mut channels = [
        Vec::with_capacity(count),
        Vec::with_capacity(count),
        Vec::with_capacity(count),
    ];
    for ch in &mut channels {
        for _ in 0..count {
            ch.push(r.f32()?);
        }
    }
    Ok(AlbedoVolumes { dim, channels })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdf_chunk_roundtrip() {
        let vol = SdfVolume {
            dim: 2,
            aabb_min: [-1.0, 0.0, 0.5],
            aabb_max: [1.0, 2.0, 1.5],
            voxels: vec![-0.3, 0.1, 0.0, 0.7, -0.5, 0.2, 0.9, -0.1],
        };
        let bytes = write_sdf(&vol, 0xabc);
        let (header, decoded) = read_sdf(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0xabc);
        assert_eq!(decoded.dim, 2);
        assert_eq!(decoded.aabb_min, vol.aabb_min);
        assert_eq!(decoded.aabb_max, vol.aabb_max);
        assert_eq!(decoded.voxels, vol.voxels);
    }

    #[test]
    fn albedo_chunk_roundtrip() {
        let vol = AlbedoVolumes {
            dim: 2,
            channels: [
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                vec![0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            ],
        };
        let bytes = write_albedo(&vol, 0x5151);
        let (header, decoded) = read_albedo(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0x5151);
        assert_eq!(decoded.dim, 2);
        assert_eq!(decoded.channels, vol.channels);
    }
}
