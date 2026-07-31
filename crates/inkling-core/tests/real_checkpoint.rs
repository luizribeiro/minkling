//! Assertions against a real Inkling-Small checkpoint, which is far too large
//! to commit. Set `INKLINGRS_CHECKPOINT` to a checkpoint directory to run them;
//! unset, each test reports a skip and passes.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use inkling_core::{Checkpoint, Dtype};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os(CHECKPOINT_VAR).map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: {CHECKPOINT_VAR} is unset");
    }
    dir
}

fn resident_bytes() -> u64 {
    let pid = std::process::id().to_string();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .expect("ps runs");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .expect("ps reports rss in KiB")
        * 1024
}

#[test]
fn mxfp4_checkpoint_spans_thirty_shards() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");

    assert_eq!(ckpt.num_shards(), 30);
    assert_eq!(ckpt.tensor_names().count(), 1508);
}

#[test]
fn routed_expert_weights_are_packed_eight_nibbles_per_u32() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");

    // 256 experts x 2048 intermediate x 4096 hidden, packed 8 per u32.
    let moe = ckpt
        .tensor("language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight")
        .expect("layer 2 is MoE");
    assert_eq!(moe.dtype(), Dtype::U32);
    assert_eq!(moe.shape(), [256, 2048, 512]);
    assert_eq!(moe.data().len(), 256 * 2048 * 512 * 4);

    // Layers 0 and 1 are dense, at the 16384-wide `dense_intermediate_size`.
    let dense = ckpt
        .tensor("language_model.model.layers.0.mlp.gate_proj.weight")
        .expect("layer 0 is dense");
    assert_eq!(dense.dtype(), Dtype::U32);
    assert_eq!(dense.shape(), [16384, 512]);
}

#[test]
fn opening_does_not_fault_in_the_weights() {
    let Some(dir) = checkpoint_dir() else { return };

    let before = resident_bytes();
    let started = Instant::now();
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let elapsed = started.elapsed();
    let after = resident_bytes();
    assert_eq!(ckpt.num_shards(), 30);

    let grew = after.saturating_sub(before);
    eprintln!("open took {elapsed:?}, RSS {before} -> {after} (+{grew} bytes)");

    // The mapped bytes are ~131 GiB. Reading 30 headers should cost single-digit
    // MiB; a gibibyte is a loose bound that still catches an eager read.
    assert!(
        grew < (1 << 30),
        "open grew RSS by {grew} bytes ({before} -> {after})"
    );
}
