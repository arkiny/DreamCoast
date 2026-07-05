//! Throwaway probe for the ufbx FBX importer (Phase 13 Stage E).
//!
//! Usage: `cargo run -p dreamcoast-asset --features fbx --example fbx_probe -- <mesh.fbx> [anim.fbx]`
//! Prints node/mesh/skin/clip counts, the AABB, and a few sample joints so the import
//! can be sanity-checked (units/orientation/skeleton) before wiring it into the sandbox.

#[cfg(feature = "fbx")]
fn main() {
    let mut args = std::env::args().skip(1);
    let mesh = args.next().expect("usage: fbx_probe <mesh.fbx> [anim.fbx]");
    let anim = args.next();
    let scene = dreamcoast_asset::load_fbx_scene(&mesh, anim.as_ref()).expect("fbx load");

    // AABB over every primitive vertex.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut tris = 0usize;
    let mut skinned_prims = 0usize;
    for prims in &scene.meshes {
        for p in prims {
            tris += p.indices.len() / 3;
            if p.joints.is_some() {
                skinned_prims += 1;
            }
            for v in &p.vertices {
                for i in 0..3 {
                    lo[i] = lo[i].min(v.pos[i]);
                    hi[i] = hi[i].max(v.pos[i]);
                }
            }
        }
    }
    println!("--- FBX probe ---");
    println!(
        "nodes={} meshes={} materials={} images={} skins={}",
        scene.nodes.len(),
        scene.meshes.len(),
        scene.materials.len(),
        scene.images.len(),
        scene.skins.len()
    );
    println!("triangles={tris} skinned_prims={skinned_prims}");
    println!(
        "aabb min=[{:.3},{:.3},{:.3}] max=[{:.3},{:.3},{:.3}]  size=[{:.3},{:.3},{:.3}]",
        lo[0],
        lo[1],
        lo[2],
        hi[0],
        hi[1],
        hi[2],
        hi[0] - lo[0],
        hi[1] - lo[1],
        hi[2] - lo[2]
    );
    for (i, s) in scene.skins.iter().enumerate() {
        println!(
            "skin[{i}]: {} joints (first few node idx: {:?})",
            s.joints.len(),
            &s.joints[..s.joints.len().min(6)]
        );
    }
    for (i, a) in scene.animations.iter().enumerate() {
        println!(
            "clip[{i}]: '{}' {:.2}s, {} channels",
            a.name.as_deref().unwrap_or("?"),
            a.duration,
            a.channels.len()
        );
    }
    // A couple of node names to eyeball the skeleton.
    for n in scene.nodes.iter().take(8) {
        println!(
            "node: '{}' t={:?}",
            n.name.as_deref().unwrap_or("?"),
            n.translation
        );
    }
}

#[cfg(not(feature = "fbx"))]
fn main() {
    eprintln!("build with --features fbx");
}
