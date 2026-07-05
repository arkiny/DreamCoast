//! Probe the native Alembic reader against a `.abc` file.
//! Usage: `cargo run -p dreamcoast-asset --example abc_probe -- <file.abc>`

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: abc_probe <file.abc>");
    let abc = dreamcoast_asset::alembic::Ogawa::open(&path).expect("open abc");
    let (groups, data) = abc.node_counts();
    println!(
        "container: root={} groups={groups} data_nodes={data}",
        abc.root()
    );

    let vc = dreamcoast_asset::alembic::read_vertex_cache(&path).expect("vertex cache");
    let tris: usize = vc.meshes.iter().map(|m| m.indices.len() / 3).sum();
    let verts: usize = vc
        .meshes
        .iter()
        .map(|m| m.frames.first().map_or(0, Vec::len))
        .sum();
    println!(
        "vertex cache: {} meshes, {} frames @ {} fps",
        vc.meshes.len(),
        vc.num_frames,
        vc.fps
    );
    println!("  total tris={tris} total verts={verts}");
    // Biggest mesh: bbox of frame 0 (metres) + motion frame0->mid.
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
        let d = m.frames[mid][0];
        println!(
            "  vert0 frame0={:?} frame{}={:?} (moved = animated)",
            m.frames[0][0], mid, d
        );
    }
}
