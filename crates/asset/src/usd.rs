//! Native **ASCII USD (`.usda`) point-cache reader** — from-scratch, no deps.
//!
//! Intel New Sponza ships the animated knight as a USD *point cache* (not a rig): the
//! `knight_USD_PREVIEW_SURFACE_ANIM` layer is `#usda 1.0` ASCII with 135 `def Mesh`
//! prims, each carrying `point3f[] points.timeSamples` (per-frame deformed positions) +
//! constant `faceVertexIndices` / `faceVertexCounts` topology. There is **no** UsdSkel
//! content (verified: zero `SkelRoot`/`Skeleton`/`primvars:skel`), so the deliverable is
//! the same baked deformation the Alembic carries — this decodes it to the identical
//! neutral [`VertexCache`], and the cook serializes either the same way.
//!
//! ## Scope (minimal USD subset — this is not a general USD reader)
//! The parser walks the prim tree (`def`/`over`/`class` + `{ … }` bodies), reads typed
//! attributes and `.timeSamples` dicts, and **materializes only** `points` /
//! `faceVertexIndices` / `faceVertexCounts` on `Mesh` prims. Every other value (extent,
//! `primvars:*`, `xformOp:*`, prim/attr metadata) is skipped by fast brace-matching, so
//! only the point floats are number-parsed. Transforms are ignored: this asset's
//! `xformOpOrder` is `[translate:pivot, !invert!translate:pivot]` (a net identity) with
//! `!resetXformStack!` at the root, so the points are already in one assembled metre-space
//! (same as the Alembic parts — see [`crate::alembic`]). A general importer would compose
//! the parent Xform chain; that's a documented follow-up.

use std::path::Path;

use dreamcoast_core::EngineError;

use crate::vcache::{VcMesh, VertexCache};

fn err(msg: impl std::fmt::Display) -> EngineError {
    EngineError::Asset(format!("usd: {msg}"))
}

/// Decode an ASCII USD `.usda` into a [`VertexCache`]: every `Mesh` prim's per-frame
/// `points` (scaled by the layer's `metersPerUnit`) + its constant triangulated topology.
pub fn read_vertex_cache(path: impl AsRef<Path>) -> Result<VertexCache, EngineError> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| err(format!("read {}: {e}", path.as_ref().display())))?;
    from_bytes(&bytes)
}

/// Parse an in-memory `.usda` image into a [`VertexCache`].
pub fn from_bytes(bytes: &[u8]) -> Result<VertexCache, EngineError> {
    if !bytes.starts_with(b"#usda") {
        return Err(err("not an ASCII USD file (missing '#usda' magic)"));
    }
    let mut p = Parser::new(bytes);
    p.ws();

    // Layer metadata block `( … )` — grab framesPerSecond + metersPerUnit, skip the rest.
    let (mut fps, mut meters) = (24.0f32, 0.01f32);
    if p.peek() == b'(' {
        let (f, m) = p.parse_header_meta();
        fps = f;
        meters = m;
    }

    // Top-level prims.
    let mut meshes = Vec::new();
    loop {
        p.ws();
        if p.at_end() {
            break;
        }
        if p.looking_at_specifier() {
            p.parse_prim(meters, &mut meshes)?;
        } else {
            // Not a prim (stray token / unsupported top-level construct) — skip a value to
            // stay in sync rather than spin.
            p.skip_value();
        }
    }

    if meshes.is_empty() {
        return Err(err("no Mesh prims with point caches found"));
    }
    let num_frames = meshes.iter().map(|m| m.frames.len()).max().unwrap_or(0);
    Ok(VertexCache {
        meshes,
        num_frames,
        fps,
    })
}

/// Whether a byte can appear in a bare word token (attribute type / name / prim
/// specifier). Includes `[` `]` (`token[]`), `:` `.` (`primvars:st.timeSamples`).
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b':' | b'.' | b'[' | b']')
}

/// Whether a byte is part of a numeric token (`-1.5e+03`).
fn is_num(b: u8) -> bool {
    b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E')
}

/// A byte cursor over a `.usda` image. Structural constructs are parsed recursively; the
/// heavy point arrays are scanned inline (no per-number allocation).
struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    #[inline]
    fn at_end(&self) -> bool {
        self.i >= self.b.len()
    }

    #[inline]
    fn peek(&self) -> u8 {
        self.b.get(self.i).copied().unwrap_or(0)
    }

    #[inline]
    fn bump(&mut self) -> u8 {
        let c = self.peek();
        self.i += 1;
        c
    }

    #[inline]
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == c {
            self.i += 1;
            true
        } else {
            false
        }
    }

    /// Skip whitespace and `#` line comments.
    fn ws(&mut self) {
        while !self.at_end() {
            match self.peek() {
                b' ' | b'\t' | b'\r' | b'\n' => self.i += 1,
                b'#' => {
                    while !self.at_end() && self.peek() != b'\n' {
                        self.i += 1;
                    }
                }
                _ => break,
            }
        }
    }

    /// Skip only inline whitespace (spaces/tabs) — used between the words of one attribute
    /// statement header so a declaration ends cleanly at its newline.
    fn ws_inline(&mut self) {
        while matches!(self.peek(), b' ' | b'\t') {
            self.i += 1;
        }
    }

    /// Read a bare word token (may be empty if not positioned on one). Borrows the source.
    fn word(&mut self) -> &'a str {
        let start = self.i;
        while !self.at_end() && is_word(self.peek()) {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i]).unwrap_or("")
    }

    /// Peek the next bare word without consuming.
    fn peek_word(&self) -> &'a str {
        let mut j = self.i;
        while j < self.b.len() && is_word(self.b[j]) {
            j += 1;
        }
        std::str::from_utf8(&self.b[self.i..j]).unwrap_or("")
    }

    fn looking_at_specifier(&self) -> bool {
        matches!(self.peek_word(), "def" | "over" | "class")
    }

    /// Consume a numeric token and parse it as `f32` (0.0 on garbage).
    fn read_f32(&mut self) -> f32 {
        let start = self.i;
        while !self.at_end() && is_num(self.peek()) {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    }

    /// Consume a numeric token and parse it as `i64` (0 on garbage).
    fn read_i64(&mut self) -> i64 {
        let start = self.i;
        while !self.at_end() && is_num(self.peek()) {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// Skip a numeric token (a timecode key) without parsing.
    fn skip_num(&mut self) {
        while !self.at_end() && is_num(self.peek()) {
            self.i += 1;
        }
    }

    /// Skip a `"…"` string (handles `\` escapes). Assumes `peek() == '"'`.
    fn skip_string(&mut self) {
        self.bump(); // opening quote
        while !self.at_end() {
            match self.bump() {
                b'\\' => {
                    self.i += 1;
                }
                b'"' => break,
                _ => {}
            }
        }
    }

    /// Skip a balanced bracketed group starting at the current `[ ( {`. Treats all bracket
    /// kinds as one depth counter (well-formed USD nests them properly) and skips strings +
    /// line comments so their contents never affect the depth.
    fn skip_bracketed(&mut self) {
        let mut depth = 0i32;
        loop {
            if self.at_end() {
                break;
            }
            match self.bump() {
                b'[' | b'(' | b'{' => depth += 1,
                b']' | b')' | b'}' => {
                    depth -= 1;
                    if depth <= 0 {
                        break;
                    }
                }
                b'"' => {
                    while !self.at_end() {
                        match self.bump() {
                            b'\\' => self.i += 1,
                            b'"' => break,
                            _ => {}
                        }
                    }
                }
                b'#' => {
                    while !self.at_end() && self.peek() != b'\n' {
                        self.i += 1;
                    }
                }
                _ => {}
            }
        }
    }

    /// Skip one attribute/metadata **value** of any shape (array, tuple, dict, string,
    /// asset ref `@…@`, path `<…>`, or a bare word/number).
    fn skip_value(&mut self) {
        self.ws();
        match self.peek() {
            b'[' | b'(' | b'{' => self.skip_bracketed(),
            b'"' => self.skip_string(),
            b'<' => while !self.at_end() && self.bump() != b'>' {},
            b'@' => {
                self.bump();
                while !self.at_end() && self.bump() != b'@' {}
            }
            _ => {
                // A bare word or number token: run to the next separator.
                while !self.at_end() {
                    match self.peek() {
                        b' ' | b'\t' | b'\r' | b'\n' | b')' | b']' | b'}' | b',' => break,
                        _ => self.i += 1,
                    }
                }
            }
        }
    }

    /// Parse the layer's `( … )` metadata block, returning `(framesPerSecond, metersPerUnit)`.
    fn parse_header_meta(&mut self) -> (f32, f32) {
        let (mut fps, mut meters) = (24.0f32, 0.01f32);
        self.bump(); // '('
        loop {
            self.ws();
            if self.at_end() || self.eat(b')') {
                break;
            }
            let key = self.word();
            if key.is_empty() {
                // Not a key=value pair (defensive) — skip a token and continue.
                self.skip_value();
                continue;
            }
            self.ws();
            if !self.eat(b'=') {
                continue;
            }
            self.ws();
            match key {
                "framesPerSecond" | "timeCodesPerSecond" => fps = self.read_f32(),
                "metersPerUnit" => meters = self.read_f32(),
                _ => self.skip_value(),
            }
        }
        (fps, meters)
    }

    /// Parse one prim (positioned at its `def`/`over`/`class` specifier), recursing into
    /// nested prims and pushing a [`VcMesh`] for each `Mesh` that has a point cache.
    fn parse_prim(&mut self, meters: f32, meshes: &mut Vec<VcMesh>) -> Result<(), EngineError> {
        self.word(); // specifier (def/over/class)
        self.ws();
        // Optional type name, then the required prim-name string.
        let mut prim_type = "";
        if self.peek() != b'"' {
            prim_type = self.word();
            self.ws();
        }
        let name = self.parse_quoted()?;
        self.ws();
        // Optional prim metadata `( … )`.
        if self.peek() == b'(' {
            self.skip_bracketed();
            self.ws();
        }
        if !self.eat(b'{') {
            return Err(err(format!("prim '{name}': expected '{{' body")));
        }

        let is_mesh = prim_type == "Mesh";
        let mut frames: Vec<Vec<[f32; 3]>> = Vec::new();
        let mut fv_counts: Vec<i64> = Vec::new();
        let mut fv_indices: Vec<i64> = Vec::new();

        loop {
            self.ws();
            if self.eat(b'}') {
                break;
            }
            if self.at_end() {
                return Err(err(format!("prim '{name}': unterminated body")));
            }
            if self.looking_at_specifier() {
                self.parse_prim(meters, meshes)?;
            } else if self.peek_word() == "variantSet" {
                // `variantSet "name" = { … }` — unsupported; skip the whole construct.
                self.word();
                self.ws();
                if self.peek() == b'"' {
                    self.skip_string();
                }
                self.ws();
                self.eat(b'=');
                self.ws();
                if self.peek() == b'{' {
                    self.skip_bracketed();
                }
            } else {
                self.parse_attribute(
                    is_mesh,
                    meters,
                    &mut frames,
                    &mut fv_counts,
                    &mut fv_indices,
                );
            }
        }

        if is_mesh && !frames.is_empty() && !fv_indices.is_empty() {
            meshes.push(VcMesh {
                name: name.to_owned(),
                indices: triangulate(&fv_counts, &fv_indices),
                frames,
            });
        }
        Ok(())
    }

    /// Read a `"…"` string's contents (no escapes expected in prim names).
    fn parse_quoted(&mut self) -> Result<&'a str, EngineError> {
        if !self.eat(b'"') {
            return Err(err("expected '\"'"));
        }
        let start = self.i;
        while !self.at_end() && self.peek() != b'"' {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).unwrap_or("");
        self.eat(b'"');
        Ok(s)
    }

    /// Parse one attribute (or relationship) statement. Materializes `points` /
    /// `faceVertexIndices` / `faceVertexCounts` on a `Mesh`; skips every other value.
    fn parse_attribute(
        &mut self,
        is_mesh: bool,
        meters: f32,
        frames: &mut Vec<Vec<[f32; 3]>>,
        fv_counts: &mut Vec<i64>,
        fv_indices: &mut Vec<i64>,
    ) {
        // The statement header is on one line: `{qualifier}* TYPE NAME` (or `rel NAME`).
        // The last word before `=` / `(` / end-of-line is the attribute name.
        let mut name = "";
        loop {
            self.ws_inline();
            match self.peek() {
                b'=' | b'(' | b'\n' | b'\r' | b'}' | 0 => break,
                _ => {}
            }
            let w = self.word();
            if w.is_empty() {
                // Unexpected char — consume it so we can't spin.
                self.i += 1;
                continue;
            }
            name = w;
        }

        self.ws_inline();
        match self.peek() {
            b'=' => {
                self.bump();
                self.ws();
                // `name.timeSamples = { tc: value, … }` vs a plain `name = value`.
                let base = name.strip_suffix(".timeSamples").unwrap_or(name);
                if !is_mesh {
                    self.skip_value();
                    return;
                }
                let timesampled = name.ends_with(".timeSamples");
                match base {
                    "points" => {
                        if timesampled {
                            self.read_points_timesamples(meters, frames);
                        } else if self.peek() == b'[' {
                            frames.push(self.read_point_array(meters));
                        } else {
                            self.skip_value();
                        }
                    }
                    "faceVertexIndices" => self.read_topology(timesampled, fv_indices),
                    "faceVertexCounts" => self.read_topology(timesampled, fv_counts),
                    _ => self.skip_value(),
                }
            }
            // Declaration-only attribute with metadata (`primvars:st ( … )`) — no value.
            b'(' => self.skip_bracketed(),
            // Declaration-only, no value (rare) — nothing to consume.
            _ => {}
        }
    }

    /// Read a topology attribute's ints into `out`. For a `.timeSamples` dict, uses the
    /// first sample (topology is constant here — stored only at timecode 1).
    fn read_topology(&mut self, timesampled: bool, out: &mut Vec<i64>) {
        if !timesampled {
            if self.peek() == b'[' {
                *out = self.read_int_array();
            } else {
                self.skip_value();
            }
            return;
        }
        // Dict `{ tc: [ints], … }` — keep the first array, skip any others.
        if !self.eat(b'{') {
            self.skip_value();
            return;
        }
        loop {
            self.ws();
            if self.at_end() || self.eat(b'}') {
                break;
            }
            self.skip_num(); // timecode key
            self.ws();
            self.eat(b':');
            self.ws();
            if self.peek() == b'[' {
                if out.is_empty() {
                    *out = self.read_int_array();
                } else {
                    self.skip_bracketed();
                }
            } else {
                self.skip_value();
            }
            self.ws();
            self.eat(b',');
        }
    }

    /// Read a `points.timeSamples` dict `{ tc: [(x,y,z)…], … }` into `frames` (file order =
    /// ascending timecode = frame order), scaling positions by `meters` (cm→m).
    fn read_points_timesamples(&mut self, meters: f32, frames: &mut Vec<Vec<[f32; 3]>>) {
        if !self.eat(b'{') {
            self.skip_value();
            return;
        }
        loop {
            self.ws();
            if self.at_end() || self.eat(b'}') {
                break;
            }
            self.skip_num(); // timecode key
            self.ws();
            self.eat(b':');
            self.ws();
            if self.peek() == b'[' {
                frames.push(self.read_point_array(meters));
            } else {
                self.skip_value(); // e.g. `None` block value
            }
            self.ws();
            self.eat(b',');
        }
    }

    /// Read a `point3f[]` array `[ (x,y,z), … ]` into scaled `[f32;3]`s. Ignores the tuple
    /// grouping and chunks every 3 floats (each point is exactly 3) — the hot path, so the
    /// separator skip is inlined (no comments appear inside an array).
    fn read_point_array(&mut self, meters: f32) -> Vec<[f32; 3]> {
        self.bump(); // '['
        let mut out = Vec::new();
        let mut cur = [0f32; 3];
        let mut k = 0usize;
        loop {
            while matches!(
                self.peek(),
                b' ' | b'\t' | b'\r' | b'\n' | b',' | b'(' | b')'
            ) {
                self.i += 1;
            }
            match self.peek() {
                b']' => {
                    self.i += 1;
                    break;
                }
                0 => break,
                _ => {
                    cur[k] = self.read_f32() * meters;
                    k += 1;
                    if k == 3 {
                        out.push(cur);
                        k = 0;
                    }
                }
            }
        }
        out
    }

    /// Read an `int[]` array `[ i, j, … ]`.
    fn read_int_array(&mut self) -> Vec<i64> {
        self.bump(); // '['
        let mut out = Vec::new();
        loop {
            while matches!(self.peek(), b' ' | b'\t' | b'\r' | b'\n' | b',') {
                self.i += 1;
            }
            match self.peek() {
                b']' => {
                    self.i += 1;
                    break;
                }
                0 => break,
                _ => out.push(self.read_i64()),
            }
        }
        out
    }
}

/// Triangulate a polygon list (per-face vertex `counts` + flat `indices`) into a triangle
/// index buffer by fan triangulation — identical to the Alembic path so both caches share
/// topology handling.
fn triangulate(counts: &[i64], indices: &[i64]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut k = 0usize;
    for &count in counts {
        let c = count.max(0) as usize;
        if c >= 3 && k + c <= indices.len() {
            for t in 1..c - 1 {
                out.push(indices[k] as u32);
                out.push(indices[k + t] as u32);
                out.push(indices[k + t + 1] as u32);
            }
        }
        k += c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal 2-frame quad (one face, 4 verts) nested under an Xform, in cm.
    const SAMPLE: &str = r#"#usda 1.0
(
    endTimeCode = 2
    framesPerSecond = 30
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "root"
{
    uniform token[] xformOpOrder = ["!resetXformStack!"]

    def Mesh "quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        uniform bool doubleSided = 1
        float3[] extent.timeSamples = {
            1: [(0, 0, 0), (100, 100, 0)],
        }
        int[] faceVertexCounts.timeSamples = {
            1: [4],
        }
        int[] faceVertexIndices.timeSamples = {
            1: [0, 1, 2, 3],
        }
        texCoord2f[] primvars:st (
            interpolation = "faceVarying"
        )
        point3f[] points.timeSamples = {
            1: [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)],
            2: [(0, 0, 0), (200, 0, 0), (200, 100, 0), (0, 100, 0)],
        }
    }
}
"#;

    #[test]
    fn parses_point_cache() {
        let vc = from_bytes(SAMPLE.as_bytes()).expect("parse");
        assert_eq!(vc.fps, 30.0);
        assert_eq!(vc.num_frames, 2);
        assert_eq!(vc.meshes.len(), 1);
        let m = &vc.meshes[0];
        assert_eq!(m.name, "quad");
        assert_eq!(m.frames.len(), 2);
        assert_eq!(m.frames[0].len(), 4);
        // metersPerUnit = 0.01 → 100 cm becomes 1.0 m.
        assert_eq!(m.frames[0][1], [1.0, 0.0, 0.0]);
        assert_eq!(m.frames[1][1], [2.0, 0.0, 0.0]);
        // Quad (count 4) fan-triangulates to 2 tris = 6 indices: [0,1,2, 0,2,3].
        assert_eq!(m.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn rejects_non_usda() {
        assert!(from_bytes(b"Ogawa\xff...").is_err());
    }
}
