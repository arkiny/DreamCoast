//! Native Alembic (`.abc`) reader — **Ogawa container** layer (from-scratch, no deps).
//!
//! Alembic is Intel New Sponza's animated-knight deliverable (a baked vertex cache); no
//! Rust crate exists, so we parse it directly (RHI/FBX-style from-scratch). This module
//! is the *container* layer — the Ogawa binary tree of **groups** and **data** nodes.
//! The Alembic *schema* layer (object hierarchy → PolyMesh `P`/`.faceIndices` + Xform)
//! sits on top and is decoded in [`schema`] (see docs/alembic-usd-import.md).
//!
//! ## Ogawa format (reverse-engineered + verified against the knight `.abc`)
//! - Header (16 B): `b"Ogawa"` + `0xff` frozen flag + `u16` version + **`u64` root-group
//!   offset** (near EOF — the tree is written back-to-front).
//! - **Group** @ off: `u64 num_children`, then `num_children × u64` child refs. A child
//!   ref's **MSB (bit 63) set ⇒ data node, clear ⇒ group**; the low 63 bits are the file
//!   offset. `0` / `u64::MAX` ⇒ empty (null child).
//! - **Data** @ off: `u64 size`, then `size` payload bytes.

use std::path::Path;

use dreamcoast_core::EngineError;

/// The MSB of a child reference: set ⇒ the child is a data node, clear ⇒ a group.
const DATA_BIT: u64 = 1 << 63;
/// The low 63 bits of a child reference hold the file offset.
const OFFSET_MASK: u64 = !DATA_BIT;

fn err(msg: impl std::fmt::Display) -> EngineError {
    EngineError::Asset(format!("alembic: {msg}"))
}

/// A parsed Ogawa archive: the whole file in memory + the root group offset. Offsets
/// index into `bytes`; groups/data are read on demand (the tree is not eagerly walked).
pub struct Ogawa {
    bytes: Vec<u8>,
    root_offset: u64,
}

/// A child reference within a group: either a nested group or a leaf data node, each at
/// a file offset. `Null` is an empty slot (`0` / `u64::MAX`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Child {
    Group(u64),
    Data(u64),
    Null,
}

impl Ogawa {
    /// Parse the Ogawa header of an `.abc` file (reads the whole file into memory).
    pub fn open(path: impl AsRef<Path>) -> Result<Ogawa, EngineError> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| err(format!("read {}: {e}", path.as_ref().display())))?;
        Self::from_bytes(bytes)
    }

    /// Parse an in-memory `.abc` image.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Ogawa, EngineError> {
        if bytes.len() < 16 || &bytes[..5] != b"Ogawa" {
            return Err(err("not an Ogawa archive (bad magic)"));
        }
        // bytes[5] = frozen flag (0xff), [6..8] = version, [8..16] = root group offset.
        let root_offset = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let o = Ogawa { bytes, root_offset };
        // Validate the root is a readable group.
        if o.group_children(o.root_offset).is_none() {
            return Err(err("root group offset is invalid"));
        }
        Ok(o)
    }

    /// The root group's offset.
    pub fn root(&self) -> u64 {
        self.root_offset
    }

    fn u64_at(&self, off: u64) -> Option<u64> {
        let o = off as usize;
        let end = o.checked_add(8)?;
        if end > self.bytes.len() {
            return None;
        }
        Some(u64::from_le_bytes(self.bytes[o..end].try_into().unwrap()))
    }

    /// The children of the group at `off` (`None` if the offset doesn't hold a plausible
    /// group — used both to read and to validate). Bounds-checked against the file size.
    pub fn group_children(&self, off: u64) -> Option<Vec<Child>> {
        let n = self.u64_at(off)?;
        // A group of > file-size/8 children can't fit — reject (guards misparsed offsets).
        let max = (self.bytes.len() as u64) / 8;
        if n > max {
            return None;
        }
        let base = off.checked_add(8)?;
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let raw = self.u64_at(base.checked_add(i.checked_mul(8)?)?)?;
            out.push(if raw == 0 || raw == u64::MAX {
                Child::Null
            } else if raw & DATA_BIT != 0 {
                Child::Data(raw & OFFSET_MASK)
            } else {
                Child::Group(raw & OFFSET_MASK)
            });
        }
        Some(out)
    }

    /// The payload bytes of the data node at `off` (`None` if out of bounds).
    pub fn data(&self, off: u64) -> Option<&[u8]> {
        let size = self.u64_at(off)? as usize;
        let start = off as usize + 8;
        let end = start.checked_add(size)?;
        self.bytes.get(start..end)
    }

    /// The size (in bytes) of the data node at `off`, without borrowing the payload.
    pub fn data_size(&self, off: u64) -> Option<u64> {
        self.u64_at(off)
    }

    /// The child at index `i` of the group at `off` (bounds/type-checked convenience).
    fn child(&self, off: u64, i: usize) -> Option<Child> {
        self.group_children(off)?.get(i).copied()
    }

    /// Walk the whole group tree, returning `(num_groups, num_data_nodes)`. Dedups by
    /// offset (the tree is a DAG — samples can be shared). Diagnostic / validation helper.
    pub fn node_counts(&self) -> (usize, usize) {
        use std::collections::HashSet;
        let mut seen: HashSet<u64> = HashSet::new();
        let mut groups = 0usize;
        let mut data = 0usize;
        let mut stack = vec![Child::Group(self.root_offset)];
        while let Some(c) = stack.pop() {
            match c {
                Child::Group(off) => {
                    if !seen.insert(off) {
                        continue;
                    }
                    let Some(children) = self.group_children(off) else {
                        continue;
                    };
                    groups += 1;
                    stack.extend(children);
                }
                Child::Data(off) => {
                    if seen.insert(off | DATA_BIT) {
                        data += 1;
                    }
                }
                Child::Null => {}
            }
        }
        (groups, data)
    }
}

// ============================================================================
// Alembic schema layer → vertex cache (docs/alembic-usd-import.md, A2/A3).
//
// On top of the Ogawa container sits Alembic's object/property tree. This decodes the
// minimal subset for a baked **vertex cache**: every `AbcGeom_PolyMesh_v1` object's
// per-frame `P` positions + constant `.faceIndices`/`.faceCounts` topology. Verified vs a
// Python reference against the Intel Sponza knight `.abc` (138 mesh parts, 300 frames, all
// pre-assembled in one space so object `Xform`s need not be applied). Format ported from
// Alembic `AbcCoreOgawa/ReadUtil.cpp` (ReadPropertyHeaders / ReadObjectHeaders).
// ============================================================================

/// One mesh of a [`VertexCache`]: a constant triangle-index list plus a position array
/// per frame (all frames share topology). Positions are in the engine's metres/Y-up.
pub struct VcMesh {
    pub name: String,
    pub indices: Vec<u32>,
    /// `frames[f]` = per-vertex positions for frame `f` (all the same length).
    pub frames: Vec<Vec<[f32; 3]>>,
}

/// A baked vertex-animation cache decoded from an Alembic `.abc`: a set of meshes each
/// carrying every frame's deformed positions. Playback = pick `frames[frame]`.
pub struct VertexCache {
    pub meshes: Vec<VcMesh>,
    pub num_frames: usize,
    pub fps: f32,
}

/// Source-unit → engine-metre scale. Alembic here is authored in centimetres (Maya).
const ABC_TO_M: f32 = 0.01;

/// Read a `u8`/`u16`/`u32` per Alembic's `sizeHint` (0/1/2), advancing `pos`.
fn get_hint(buf: &[u8], hint: u32, pos: &mut usize) -> Option<u32> {
    let v = match hint {
        0 => {
            let b = *buf.get(*pos)?;
            *pos += 1;
            b as u32
        }
        1 => {
            let b = buf.get(*pos..*pos + 2)?;
            *pos += 2;
            u16::from_le_bytes(b.try_into().unwrap()) as u32
        }
        _ => {
            let b = buf.get(*pos..*pos + 4)?;
            *pos += 4;
            u32::from_le_bytes(b.try_into().unwrap())
        }
    };
    Some(v)
}

/// A parsed property header (subset): name, kind, and — for non-compound — the POD /
/// extent / sample count needed to locate its data.
struct PropHeader {
    name: String,
    is_compound: bool,
    pod: u32,
    extent: u32,
}

/// The `schema=` value from an Alembic metadata string (`key=value;` pairs).
fn schema_of(meta: &str) -> &str {
    for kv in meta.split(';') {
        if let Some(s) = kv.strip_prefix("schema=") {
            return s;
        }
    }
    ""
}

/// Byte size of an Alembic POD type (subset: the numeric types a mesh uses).
fn pod_bytes(pod: u32) -> u64 {
    match pod {
        0..=2 => 1,      // bool / u8 / i8
        3 | 4 | 9 => 2,  // u16 / i16 / f16
        5 | 6 | 10 => 4, // u32 / i32 / f32
        7 | 8 | 11 => 8, // u64 / i64 / f64
        _ => 4,
    }
}

impl Ogawa {
    /// The archive's indexed-metadata pool (root child[5]); index 0 is the empty default.
    fn indexed_metadata(&self) -> Vec<String> {
        let mut pool = vec![String::new()];
        let Some(Child::Data(off)) = self.child(self.root_offset, 5) else {
            return pool;
        };
        let Some(buf) = self.data(off) else {
            return pool;
        };
        let mut pos = 0;
        while pos < buf.len() {
            let sz = buf[pos] as usize;
            pos += 1;
            if pos + sz > buf.len() {
                break;
            }
            pool.push(String::from_utf8_lossy(&buf[pos..pos + sz]).into_owned());
            pos += sz;
        }
        pool
    }

    /// Parse a compound property's header blob into per-property headers, in child order
    /// (property `i` ↔ compound child `i`). Each entry: header + its metadata string.
    fn property_headers(&self, blob: &[u8], meta_pool: &[String]) -> Vec<(PropHeader, String)> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos + 4 <= blob.len() {
            let info = u32::from_le_bytes(blob[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let ptype = info & 0x3;
            let hint = (info >> 2) & 0x3;
            let is_compound = ptype == 0;
            let (mut pod, mut extent) = (0u32, 0u32);
            if !is_compound {
                pod = (info >> 4) & 0xf;
                extent = (info >> 12) & 0xff;
                if get_hint(blob, hint, &mut pos).is_none() {
                    break;
                }
                if info & 0x200 != 0 {
                    get_hint(blob, hint, &mut pos);
                    get_hint(blob, hint, &mut pos);
                }
                if info & 0x100 != 0 {
                    get_hint(blob, hint, &mut pos);
                }
            }
            let Some(name_sz) = get_hint(blob, hint, &mut pos) else {
                break;
            };
            let name_sz = name_sz as usize;
            if pos + name_sz > blob.len() {
                break;
            }
            let name = String::from_utf8_lossy(&blob[pos..pos + name_sz]).into_owned();
            pos += name_sz;
            let md_idx = (info >> 20) & 0xff;
            let meta = if md_idx == 0xff {
                let Some(msz) = get_hint(blob, hint, &mut pos) else {
                    break;
                };
                let msz = msz as usize;
                if pos + msz > blob.len() {
                    break;
                }
                let m = String::from_utf8_lossy(&blob[pos..pos + msz]).into_owned();
                pos += msz;
                m
            } else {
                meta_pool.get(md_idx as usize).cloned().unwrap_or_default()
            };
            out.push((
                PropHeader {
                    name,
                    is_compound,
                    pod,
                    extent,
                },
                meta,
            ));
        }
        out
    }

    /// The child-object headers (name + metadata) of the object group at `off` — its last
    /// data child, minus a trailing 32-byte hash block. Child object `i` is group child `i+1`.
    fn object_headers(&self, off: u64, meta_pool: &[String]) -> Vec<(String, String)> {
        let Some(children) = self.group_children(off) else {
            return Vec::new();
        };
        let Some(Child::Data(hoff)) = children.last().copied() else {
            return Vec::new();
        };
        let Some(full) = self.data(hoff) else {
            return Vec::new();
        };
        if full.len() <= 32 {
            return Vec::new();
        }
        let buf = &full[..full.len() - 32];
        let mut out = Vec::new();
        let mut pos = 0;
        while pos + 4 <= buf.len() {
            let name_sz = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if name_sz == 0 || pos + name_sz + 1 > buf.len() {
                break;
            }
            let name = String::from_utf8_lossy(&buf[pos..pos + name_sz]).into_owned();
            pos += name_sz;
            let md_idx = buf[pos];
            pos += 1;
            let meta = if md_idx == 0xff {
                if pos + 4 > buf.len() {
                    break;
                }
                let msz = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                if pos + msz > buf.len() {
                    break;
                }
                let m = String::from_utf8_lossy(&buf[pos..pos + msz]).into_owned();
                pos += msz;
                m
            } else {
                meta_pool.get(md_idx as usize).cloned().unwrap_or_default()
            };
            out.push((name, meta));
        }
        out
    }

    /// The data-node offsets of an array property's samples, filtered to the expected byte
    /// size (`16-byte key + count × elem_bytes`) so the interleaved dims nodes are skipped.
    /// Returned in child (frame) order.
    fn array_sample_data(&self, prop_off: u64, elem_bytes: u64) -> Vec<u64> {
        let Some(children) = self.group_children(prop_off) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for c in children {
            if let Child::Data(off) = c
                && let Some(sz) = self.data_size(off)
                && sz > 16
                && (sz - 16) % elem_bytes == 0
            {
                out.push(off);
            }
        }
        out
    }

    /// Read a constant `int32` array sample (`[16-byte key][i32…]`) at `off`.
    fn read_i32_array(&self, off: u64) -> Option<Vec<i32>> {
        let bytes = &self.data(off)?[16..];
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    /// Find a named compound sub-property of the compound-property group at `prop_c`,
    /// returning its property headers + child list.
    #[allow(clippy::type_complexity)]
    fn find_compound(
        &self,
        prop_c: u64,
        name: &str,
        meta_pool: &[String],
    ) -> Option<(Vec<(PropHeader, String)>, Vec<Child>)> {
        let children = self.group_children(prop_c)?;
        let Child::Data(hoff) = children.last().copied()? else {
            return None;
        };
        let headers = self.property_headers(self.data(hoff)?, meta_pool);
        for (i, (h, _)) in headers.iter().enumerate() {
            if h.name == name && h.is_compound {
                let Some(Child::Group(g)) = children.get(i).copied() else {
                    return None;
                };
                let gch = self.group_children(g)?;
                let Child::Data(gh) = gch.last().copied()? else {
                    return None;
                };
                let gheaders = self.property_headers(self.data(gh)?, meta_pool);
                return Some((gheaders, gch));
            }
        }
        None
    }

    /// Decode one `AbcGeom_PolyMesh_v1` object into a [`VcMesh`] (topology + all frames).
    fn read_polymesh(&self, obj: u64, name: &str, meta_pool: &[String]) -> Option<VcMesh> {
        // Object child[0] = the property compound; its `.geom` sub-compound holds geometry.
        let Child::Group(prop_c) = self.child(obj, 0)? else {
            return None;
        };
        let (headers, geom_children) = self.find_compound(prop_c, ".geom", meta_pool)?;

        // Locate P / .faceIndices / .faceCounts by name → compound child index.
        let mut p_samples: Vec<u64> = Vec::new();
        let mut p_elem = 0u64;
        let mut nverts = 0u64;
        let mut fi_off = None;
        let mut fc_off = None;
        for (i, (h, _)) in headers.iter().enumerate() {
            if h.is_compound {
                continue;
            }
            let Some(Child::Group(g)) = geom_children.get(i).copied() else {
                continue;
            };
            match h.name.as_str() {
                "P" => {
                    p_elem = pod_bytes(h.pod) * h.extent.max(1) as u64;
                    p_samples = self.array_sample_data(g, p_elem);
                    if let Some(&first) = p_samples.first() {
                        nverts = (self.data_size(first)? - 16) / p_elem;
                    }
                }
                ".faceIndices" => {
                    fi_off = self.array_sample_data(g, pod_bytes(h.pod)).first().copied();
                }
                ".faceCounts" => {
                    fc_off = self.array_sample_data(g, pod_bytes(h.pod)).first().copied();
                }
                _ => {}
            }
        }
        if nverts == 0 || p_samples.is_empty() {
            return None;
        }
        let fi = self.read_i32_array(fi_off?)?;
        let fc = self.read_i32_array(fc_off?)?;

        // Triangulate the polygon list (fan) into a constant index buffer.
        let mut indices = Vec::new();
        let mut k = 0usize;
        for &count in &fc {
            let c = count as usize;
            if c >= 3 && k + c <= fi.len() {
                for t in 1..c - 1 {
                    indices.push(fi[k] as u32);
                    indices.push(fi[k + t] as u32);
                    indices.push(fi[k + t + 1] as u32);
                }
            }
            k += c;
        }

        // Per-frame positions (cm → m).
        let want = 16 + nverts * p_elem;
        let mut frames = Vec::with_capacity(p_samples.len());
        for off in p_samples {
            if self.data_size(off)? != want {
                continue;
            }
            let bytes = &self.data(off)?[16..];
            let mut verts = Vec::with_capacity(nverts as usize);
            for v in bytes.chunks_exact(12) {
                verts.push([
                    f32::from_le_bytes(v[0..4].try_into().unwrap()) * ABC_TO_M,
                    f32::from_le_bytes(v[4..8].try_into().unwrap()) * ABC_TO_M,
                    f32::from_le_bytes(v[8..12].try_into().unwrap()) * ABC_TO_M,
                ]);
            }
            frames.push(verts);
        }
        Some(VcMesh {
            name: name.to_owned(),
            indices,
            frames,
        })
    }
}

/// Decode an Alembic `.abc` into a [`VertexCache`]: every PolyMesh's per-frame positions
/// and constant topology, converted to the engine's metres. Object `Xform`s are not
/// applied (this cache's parts are authored pre-assembled in one space; a general importer
/// would compose the parent Xform chain — a documented follow-up).
pub fn read_vertex_cache(path: impl AsRef<Path>) -> Result<VertexCache, EngineError> {
    let abc = Ogawa::open(path)?;
    let meta_pool = abc.indexed_metadata();
    let Some(Child::Group(top)) = abc.child(abc.root(), 2) else {
        return Err(err("archive has no top object"));
    };

    let mut meshes = Vec::new();
    let mut num_frames = 0usize;
    // DFS the object tree; child-object `i`'s group is the object group's child `i+1`.
    let mut stack = vec![top];
    while let Some(obj) = stack.pop() {
        let headers = abc.object_headers(obj, &meta_pool);
        let Some(children) = abc.group_children(obj) else {
            continue;
        };
        for (i, (name, meta)) in headers.iter().enumerate() {
            let Some(Child::Group(cg)) = children.get(i + 1).copied() else {
                continue;
            };
            if schema_of(meta) == "AbcGeom_PolyMesh_v1"
                && let Some(mesh) = abc.read_polymesh(cg, name, &meta_pool)
            {
                num_frames = num_frames.max(mesh.frames.len());
                meshes.push(mesh);
            }
            stack.push(cg);
        }
    }
    if meshes.is_empty() {
        return Err(err("no PolyMesh objects found"));
    }
    Ok(VertexCache {
        meshes,
        num_frames,
        fps: 24.0, // TODO: read from the archive's TimeSampling (root child[4])
    })
}
