//! Assertions against a real Inkling-Small checkpoint, which is far too large
//! to commit. Set `INKLINGRS_CHECKPOINT` to a checkpoint directory to run them;
//! unset, each test reports a skip and passes.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use inkling_core::attention::{Attention, AttentionConfig, AttentionWeights};
use inkling_core::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, deviation};
use inkling_core::quant::dequantize;
use inkling_core::{Checkpoint, Config, Dtype, TextConfig};

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

/// What the synthetic attention fixture cannot settle: that the tensor names,
/// the checkpoint's layouts and the config fields a layer reads are the ones
/// this port assumes. Its weights are `[4096, 4096]` MXFP4 — 67 MB each once
/// decoded — so they are far too large to commit, and only a real checkpoint
/// carries them.
///
/// The reference multiplies its 4-bit weights without decoding them, in
/// bfloat16, through `mx.quantized_matmul`; this decodes them and multiplies in
/// float32. Every intermediate of the recorded pass was rounded to bfloat16 and
/// none of this port's are, over a chain of five projections, two convolutions,
/// two norms and a softmax — so the gap is a dtype's rather than an arithmetic
/// one, and 6e-3 is the same bound, for the same reason, as the recorded
/// attention step and the trained masks. Worst observed when this landed:
/// 2.6e-3 on layer 0, against a mask flattened to no learned bias at 1.4e-1.
///
/// What this settles is the wiring; the synthetic cases settle the arithmetic.
/// It cannot settle everything: both captured layers are sliding ones, and at
/// eight tokens their 512-token window caps nothing, so a port that read a
/// sliding layer's config fields as a global layer's would agree here.
const ATTENTION_TOLERANCE: f32 = 6e-3;

fn text_config(dir: &Path) -> TextConfig {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).expect("checkpoint carries a config.json");
    serde_json::from_str::<Config>(&text)
        .expect("config.json parses")
        .text_config
}

/// One layer's attention tensors, decoded out of the checkpoint.
///
/// The projections are MXFP4 — packed codes plus a scale byte per group of 32 —
/// and everything else is bfloat16, so the two are read different ways.
struct Weights {
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    r_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_sconv: Vec<f32>,
    v_sconv: Vec<f32>,
    rel_proj: Vec<f32>,
}

impl Weights {
    fn load(ckpt: &Checkpoint, layer: usize) -> Self {
        let of = |name: &str| {
            let name = format!("language_model.model.layers.{layer}.self_attn.{name}");
            ckpt.tensor(&name)
                .unwrap_or_else(|err| panic!("checkpoint holds {name}: {err}"))
        };
        let quantized = |name: &str| {
            dequantize(
                &of(&format!("{name}.weight")),
                &of(&format!("{name}.scales")),
            )
            .unwrap_or_else(|err| panic!("layer {layer} {name} decodes: {err}"))
            .values
        };
        let widened = |name: &str| of(name).to_f32().expect("a bfloat16 tensor");

        Self {
            q_proj: quantized("q_proj"),
            k_proj: quantized("k_proj"),
            v_proj: quantized("v_proj"),
            r_proj: quantized("r_proj"),
            o_proj: quantized("o_proj"),
            q_norm: widened("q_norm.weight"),
            k_norm: widened("k_norm.weight"),
            k_sconv: widened("k_sconv.conv.weight"),
            v_sconv: widened("v_sconv.conv.weight"),
            rel_proj: widened("rel_proj"),
        }
    }

    fn view(&self) -> AttentionWeights<'_> {
        AttentionWeights {
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            r_proj: &self.r_proj,
            o_proj: &self.o_proj,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            k_sconv: &self.k_sconv,
            v_sconv: &self.v_sconv,
            rel_proj: &self.rel_proj,
        }
    }
}

#[test]
fn attention_reproduces_the_reference_layers_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);

    let mut worst = 0.0f32;
    for layer in CAPTURED_LAYERS {
        let recorded =
            |name: &str| fixture::f32s(&fixture::layer_tensor(&activations, layer, name));
        let (x, want) = (recorded("input_layernorm_out"), recorded("o_proj_out"));
        let weights = Weights::load(&ckpt, layer);
        let layer_config = AttentionConfig::for_layer(&config, layer);

        let attention = Attention::new(layer_config, weights.view());
        let against_reference = deviation(&attention.forward(&mut attention.cache(), &x), &want);
        assert!(
            against_reference <= ATTENTION_TOLERANCE,
            "layer {layer}: deviation {against_reference:e}"
        );
        worst = worst.max(against_reference);

        // The learned relative-position bias is most of what makes this an
        // Inkling layer rather than an ordinary causal one, and at eight tokens
        // it is the only part of the mask that is not plain causality. Without
        // it the layer still attends, over a flat band.
        let flat = vec![0.0; weights.rel_proj.len()];
        let mut unbiased = weights.view();
        unbiased.rel_proj = &flat;
        let unbiased = Attention::new(layer_config, unbiased);
        let flattened = deviation(&unbiased.forward(&mut unbiased.cache(), &x), &want);
        assert!(
            flattened > ATTENTION_TOLERANCE,
            "layer {layer}: a flat band deviates by only {flattened:e}"
        );
    }
    assert!(
        worst > 0.0,
        "a run that matched exactly would mean the reference's bfloat16 vanished"
    );
}
