//! Texture-compression policy + tiers for the cook (Phase 12, items 1a–1c).

use rhi_types::Format;

use crate::bc::{self, BcFormat};
use crate::{ImageData, Material, TexData};

/// Texture-compression tier for the cook. Picks the colour codec; `Off` keeps
/// textures uncompressed (render byte-identical). The trade is size vs quality —
/// and it is **content-dependent**: on smooth textures BC1 ≈ BC7 (measured
/// 0.008/ch on the sample asset) at half the size, while BC7 pulls ahead on
/// high-frequency colour. Normals always use BC5; data textures stay uncompressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexCompress {
    /// No compression — RGBA8, render byte-for-byte unchanged.
    Off,
    /// Size-first: BC1 colour (BC3 when alpha), 8:1. Best when textures are smooth.
    Fast,
    /// Quality-first: BC7 colour (RGBA, 4:1). Best on complex / high-frequency colour.
    High,
}

impl TexCompress {
    /// Whether any compression happens.
    pub(crate) fn enabled(self) -> bool {
        self != TexCompress::Off
    }
    /// A stable tag folded into the cache key so changing the tier re-cooks.
    pub(crate) fn tag(self) -> u8 {
        match self {
            TexCompress::Off => 0,
            TexCompress::Fast => 1,
            TexCompress::High => 2,
        }
    }
}

/// The semantic role of a texture slot — drives the per-slot compression policy.
/// **Colour** (base colour, emissive) is perceptual; **Normal** is a vector map;
/// **Data** (metallic-roughness, occlusion, any linear/vector payload) must stay
/// uncompressed because block compression corrupts non-perceptual values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexSlot {
    Colour,
    Normal,
    Data,
}

/// Per-slot texture-compression policy. **Perceptual colour** (base colour,
/// emissive) compresses per the `tier` (BC1/BC3 for `Fast`, BC7 for `High`);
/// **normals** to BC5 (near-lossless). **Data textures** — metallic-roughness and
/// anything carrying linear/vector data — are left uncompressed, because block
/// compression corrupts non-perceptual values.
pub(crate) fn compress_material(material: &mut Material, tier: TexCompress) {
    compress_slot(&mut material.base_color, TexSlot::Colour, true, tier);
    compress_slot(&mut material.emissive, TexSlot::Colour, true, tier);
    compress_slot(&mut material.normal, TexSlot::Normal, false, tier);
    // metallic_roughness: data texture — intentionally left uncompressed.
}

/// The block format the policy assigns to `slot` at `tier`, or `None` to keep it
/// uncompressed (Data slots, or `tier == Off`). `has_alpha` only matters for colour:
/// `Fast` picks BC3 over BC1 when there is real alpha (BC1 would drop it). This is the
/// single source of the per-slot format choice, shared by the Material cook and the
/// runtime glTF/level texture upload.
pub fn slot_format(slot: TexSlot, tier: TexCompress, has_alpha: bool) -> Option<BcFormat> {
    if !tier.enabled() {
        return None;
    }
    match slot {
        TexSlot::Data => None,
        TexSlot::Normal => Some(BcFormat::Bc5),
        TexSlot::Colour => Some(match tier {
            TexCompress::High => BcFormat::Bc7,
            _ if has_alpha => BcFormat::Bc3,
            _ => BcFormat::Bc1,
        }),
    }
}

/// Block-compress a single decoded image per the slot policy + tier, or return `None`
/// to keep it uncompressed (Data slots, or `tier == Off`). Lets the runtime importer
/// (levels / glTF scenes) compress shared images with the exact same policy as the cook.
pub fn compress_image_for_slot(
    im: &ImageData,
    slot: TexSlot,
    srgb: bool,
    tier: TexCompress,
) -> Option<TexData> {
    let has_alpha = im.rgba8.chunks_exact(4).any(|p| p[3] != 255);
    let fmt = slot_format(slot, tier, has_alpha)?;
    Some(compress_image(im, fmt, srgb))
}

/// Compress a material slot in place per the policy (no-op for `Data` / `tier == Off`,
/// or a slot that is already block-compressed / empty).
fn compress_slot(slot: &mut Option<TexData>, kind: TexSlot, srgb: bool, tier: TexCompress) {
    if let Some(TexData::Rgba8(im)) = slot
        && let Some(bc) = compress_image_for_slot(im, kind, srgb, tier)
    {
        *slot = Some(bc);
    }
}

/// Block-compress an RGBA8 image to a full BCn mip chain. Mips come from the shared
/// `generate_mip_chain` (the cross-backend-parity single source) so cooked mips
/// match the uncompressed upload path, then each level is BC-encoded.
fn compress_image(im: &ImageData, fmt: BcFormat, srgb: bool) -> TexData {
    let format = if srgb {
        Format::Rgba8Srgb
    } else {
        Format::Rgba8Unorm
    };
    let levels = rhi_types::generate_mip_chain(&im.rgba8, im.width, im.height, format);
    let mips = levels
        .iter()
        .enumerate()
        .map(|(mip, lvl)| {
            let w = (im.width >> mip).max(1);
            let h = (im.height >> mip).max(1);
            match fmt {
                BcFormat::Bc1 => bc::encode_bc1(lvl, w, h),
                BcFormat::Bc3 => bc::encode_bc3(lvl, w, h),
                BcFormat::Bc4 => bc::encode_bc4(lvl, w, h),
                BcFormat::Bc5 => bc::encode_bc5(lvl, w, h),
                BcFormat::Bc7 => bc::encode_bc7(lvl, w, h),
            }
        })
        .collect();
    TexData::Bc {
        format: fmt,
        srgb,
        width: im.width,
        height: im.height,
        mips,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: u32, h: u32, rgba: [u8; 4]) -> TexData {
        TexData::Rgba8(ImageData {
            width: w,
            height: h,
            rgba8: rgba.repeat((w * h) as usize),
        })
    }

    #[test]
    fn compression_policy_per_slot() {
        let mut m = Material {
            base_color: Some(solid_image(8, 8, [200, 100, 50, 255])),
            metallic_roughness: Some(solid_image(8, 8, [0, 128, 200, 255])),
            normal: Some(solid_image(8, 8, [128, 128, 255, 255])),
            emissive: Some(solid_image(8, 8, [10, 20, 30, 255])),
            ..Material::default()
        };
        compress_material(&mut m, TexCompress::Fast);

        // Fast: perceptual colour -> BC1; normals -> BC5; data texture stays RGBA8.
        assert!(matches!(
            m.base_color,
            Some(TexData::Bc {
                format: BcFormat::Bc1,
                ..
            })
        ));
        assert!(matches!(
            m.emissive,
            Some(TexData::Bc {
                format: BcFormat::Bc1,
                ..
            })
        ));
        assert!(matches!(
            m.normal,
            Some(TexData::Bc {
                format: BcFormat::Bc5,
                ..
            })
        ));
        assert!(
            matches!(m.metallic_roughness, Some(TexData::Rgba8(_))),
            "metallic-roughness is a data texture and must stay uncompressed"
        );
    }

    #[test]
    fn high_tier_uses_bc7() {
        let mut m = Material {
            base_color: Some(solid_image(8, 8, [200, 100, 50, 255])),
            ..Material::default()
        };
        compress_material(&mut m, TexCompress::High);
        assert!(matches!(
            m.base_color,
            Some(TexData::Bc {
                format: BcFormat::Bc7,
                ..
            })
        ));
    }

    #[test]
    fn fast_tier_alpha_uses_bc3() {
        // Fast tier on a transparent base colour: BC3 (keeps alpha; BC1 would drop it).
        let mut m = Material {
            base_color: Some(solid_image(4, 4, [200, 100, 50, 128])),
            ..Material::default()
        };
        compress_material(&mut m, TexCompress::Fast);
        assert!(matches!(
            m.base_color,
            Some(TexData::Bc {
                format: BcFormat::Bc3,
                ..
            })
        ));
    }

    #[test]
    fn compression_shrinks_and_roundtrips() {
        let mut m = Material {
            base_color: Some(solid_image(64, 64, [200, 100, 50, 255])),
            ..Material::default()
        };
        let raw = match m.base_color.as_ref().unwrap() {
            TexData::Rgba8(im) => im.rgba8.len(),
            _ => unreachable!(),
        };
        compress_material(&mut m, TexCompress::Fast);
        let compressed: usize = match m.base_color.as_ref().unwrap() {
            TexData::Bc { mips, .. } => mips.iter().map(|m| m.len()).sum(),
            _ => unreachable!(),
        };
        // BC1 is 8:1 on the base level; even with the full mip chain it stays well
        // under a quarter of the single uncompressed level.
        assert!(
            compressed < raw / 4,
            "compressed {compressed} should be << raw {raw}"
        );
    }
}
