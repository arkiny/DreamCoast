//! Baked **vertex-animation cache** — the neutral, importer-agnostic type.
//!
//! A vertex cache is a set of meshes, each with constant topology (a triangle-index
//! list) and one deformed position array **per frame**. It is the natural target for a
//! baked deformation deliverable that carries no rig — the Intel Sponza knight ships as
//! exactly this, in two formats that decode to this same type:
//! [`crate::alembic`] (Ogawa `.abc`) and [`crate::usd`] (ASCII `.usda` point cache).
//!
//! Keeping the type here (not in either importer) is the single source of truth the
//! separate-asset cook ([`crate::dcasset::write_vcache`] / [`crate::cook::load_or_cook_vcache`])
//! serializes, so both importers feed one cooked-`.dcasset` path and the runtime plays
//! either without a live 665 MB / 1.4 GB decode.

/// One mesh of a [`VertexCache`]: a constant triangle-index list plus a position array
/// per frame (all frames share topology). Positions are in the engine's metres/Y-up.
pub struct VcMesh {
    pub name: String,
    pub indices: Vec<u32>,
    /// `frames[f]` = per-vertex positions for frame `f` (all the same length).
    pub frames: Vec<Vec<[f32; 3]>>,
}

/// A baked vertex-animation cache: a set of meshes each carrying every frame's deformed
/// positions. Playback = pick `frames[frame]`. Decoded from an Alembic `.abc`
/// ([`crate::alembic::read_vertex_cache`]) or an ASCII USD `.usda`
/// ([`crate::usd::read_vertex_cache`]); cooked to / loaded from a `.dcasset`.
pub struct VertexCache {
    pub meshes: Vec<VcMesh>,
    pub num_frames: usize,
    pub fps: f32,
}

impl VertexCache {
    /// Reduce the resident frame set to at most `max_frames` by evenly subsampling (both endpoints
    /// kept), scaling `fps` so the playback **duration is unchanged** (fewer frames over the same
    /// wall-clock). A no-op when `max_frames == 0` (unbudgeted) or already `<= max_frames`.
    ///
    /// This is the memory-budget lever for a large baked cache: the knight `.abc` is ~1.26 GB and
    /// the `.usda` ~223 MB fully resident (30–135 parts × 300 frames × up to ~350 K positions).
    /// Applied at cook time so the budget bounds BOTH the on-disk `.dcasset` and the resident set,
    /// and is deterministic (headless captures reproduce, DX≡VK). Folded into the cook cache key so
    /// changing the budget re-cooks. Streaming a frame window from disk (finer than a fixed cap) is
    /// a documented follow-up; this is the coarse, always-correct budget.
    pub fn decimate(&mut self, max_frames: usize) {
        if max_frames == 0 || self.num_frames <= max_frames {
            return;
        }
        let (n, m) = (self.num_frames, max_frames);
        // Evenly-spaced kept source indices in `[0, n)`: `i*n/m` for `i in 0..m` (monotone, index 0
        // included; the last is `(m-1)*n/m < n`). Same indices for every part (they share `n`).
        let kept: Vec<usize> = (0..m).map(|i| i * n / m).collect();
        for mesh in &mut self.meshes {
            if mesh.frames.len() == n {
                mesh.frames = kept.iter().map(|&k| mesh.frames[k].clone()).collect();
            }
        }
        // Preserve total duration `n/fps`: new_fps = fps * m/n so `m/new_fps == n/fps`.
        self.fps = self.fps * m as f32 / n as f32;
        self.num_frames = m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(n: usize) -> VertexCache {
        // Two parts sharing `n` frames; each frame tags its index into x so we can check which
        // source frames survived decimation.
        let mk = |name: &str| VcMesh {
            name: name.to_owned(),
            indices: vec![0, 1, 2],
            frames: (0..n).map(|f| vec![[f as f32, 0.0, 0.0]; 3]).collect(),
        };
        VertexCache {
            meshes: vec![mk("a"), mk("b")],
            num_frames: n,
            fps: 24.0,
        }
    }

    #[test]
    fn decimate_caps_frames_and_preserves_duration() {
        let mut c = cache(300);
        let before = 300.0 / c.fps; // total seconds
        c.decimate(100);
        assert_eq!(c.num_frames, 100);
        for m in &c.meshes {
            assert_eq!(m.frames.len(), 100, "every part decimated to the cap");
        }
        // Duration unchanged (fps scaled): m/new_fps == n/old_fps.
        let after = c.num_frames as f32 / c.fps;
        assert!((before - after).abs() < 1e-4, "duration preserved");
        // Kept source indices are monotone increasing and start at 0 (i*n/m).
        let kept: Vec<f32> = c.meshes[0].frames.iter().map(|f| f[0][0]).collect();
        assert_eq!(kept[0], 0.0);
        assert!(kept.windows(2).all(|w| w[1] > w[0]), "monotone");
        assert!(*kept.last().unwrap() < 300.0, "last kept index < n");
    }

    #[test]
    fn decimate_is_noop_when_unbudgeted_or_already_small() {
        let mut c = cache(50);
        c.decimate(0); // unbudgeted
        assert_eq!(c.num_frames, 50);
        c.decimate(200); // cap above frame count
        assert_eq!(c.num_frames, 50);
        assert_eq!(c.fps, 24.0);
    }
}
