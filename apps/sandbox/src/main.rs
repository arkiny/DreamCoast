//! The `sandbox` binary: the stock technique playground.
//!
//! Everything lives in the crate's library half (`lib.rs`) so a game application can
//! link it and inject gameplay through `sandbox::GameHooks` instead of forking the
//! render path. This bin installs no hooks — `GameConfig::default()` is the historic
//! behavior, byte-for-byte.

fn main() -> anyhow::Result<()> {
    sandbox::main_entry(sandbox::GameConfig::default())
}
