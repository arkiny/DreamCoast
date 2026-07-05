//! Probe the native ASCII USD (`.usda`) point-cache reader against a `.usd`/`.usda` file.
//! Usage: `cargo run -p dreamcoast-asset --example usd_probe -- <file.usda>`

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: usd_probe <file.usda>");
    let t = std::time::Instant::now();
    let vc = dreamcoast_asset::usd::read_vertex_cache(&path).expect("vertex cache");
    let secs = t.elapsed().as_secs_f32();

    let tris: usize = vc.meshes.iter().map(|m| m.indices.len() / 3).sum();
    let verts: usize = vc
        .meshes
        .iter()
        .map(|m| m.frames.first().map_or(0, Vec::len))
        .sum();
    println!(
        "vertex cache: {} meshes, {} frames @ {} fps  (decoded in {secs:.1}s)",
        vc.meshes.len(),
        vc.num_frames,
        vc.fps
    );
    println!("  total tris={tris} total verts={verts}");

    if let Some(m) = vc
        .meshes
        .iter()
        .max_by_key(|m| m.frames.first().map_or(0, Vec::len))
    {
        let f0 = &m.frames[0];
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in f0 {
            for i in 0..3 {
                lo[i] = lo[i].min(v[i]);
                hi[i] = hi[i].max(v[i]);
            }
        }
        println!(
            "  biggest '{}': {} verts, {} frames",
            m.name,
            f0.len(),
            m.frames.len()
        );
        println!(
            "  bbox(m) min=[{:.2},{:.2},{:.2}] max=[{:.2},{:.2},{:.2}]",
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
        );
        let mid = m.frames.len() / 2;
        println!(
            "  vert0 frame0={:?} frame{}={:?} (moved = animated)",
            m.frames[0][0], mid, m.frames[mid][0]
        );
    }
}
