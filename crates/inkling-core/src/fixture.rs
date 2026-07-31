//! Test-only access to the committed reference fixtures, behind the
//! `test-support` feature: the paths here are relative to this crate's source
//! tree, so nothing outside a test can use them.
//!
//! Each fixture is a safetensors bundle under `reference/fixtures`, written by
//! a `just dump-*` recipe and read back through [`Checkpoint`]'s single-file
//! layout.

use std::path::PathBuf;

use crate::checkpoint::{Checkpoint, Dtype, TensorView};
use crate::moe::ExpertBank;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/fixtures");

/// The forward pass `just dump-activations` recorded: eight tokens through the
/// whole model, keeping every intermediate of the layers it captured.
pub const ACTIVATIONS: &str = "layer_activations.safetensors";

/// The decoder layers that pass kept, and so the layers every trained case is
/// cut from. Which three comes from the checkpoint — the dump script refuses a
/// set that does not cover both a dense and a MoE MLP, and both a sliding and a
/// global attention.
pub const CAPTURED_LAYERS: [usize; 3] = [0, 2, 5];

pub fn open(file: &str) -> Checkpoint {
    let path = PathBuf::from(DIR).join(file);
    Checkpoint::open(&path).unwrap_or_else(|err| panic!("{file} opens: {err}"))
}

pub fn tensor<'a>(ckpt: &'a Checkpoint, name: &str) -> TensorView<'a> {
    ckpt.tensor(name)
        .unwrap_or_else(|err| panic!("fixture holds {name}: {err}"))
}

/// A tensor a dump recorded per decoder layer, which every bundle names
/// `layer{layer}.{name}`.
pub fn layer_tensor<'a>(ckpt: &'a Checkpoint, layer: usize, name: &str) -> TensorView<'a> {
    tensor(ckpt, &format!("layer{layer}.{name}"))
}

/// A fixture tensor's values. Every dump casts to float32 before saving, so a
/// comparison never has to reason about the reference's dtype choices, and
/// anything else in a fixture is a dump that stopped doing that.
pub fn f32s(view: &TensorView<'_>) -> Vec<f32> {
    assert_eq!(view.dtype(), Dtype::F32);
    view.to_f32().expect("float32 widens")
}

/// An index tensor's values. Every dump casts integers to int32 before saving,
/// for the same reason it casts floats to float32.
pub fn indices(view: &TensorView<'_>) -> Vec<usize> {
    assert_eq!(view.dtype(), Dtype::I32);
    view.data()
        .chunks_exact(size_of::<i32>())
        .map(|b| i32::from_le_bytes(b.try_into().expect("chunked into ints")) as usize)
        .collect()
}

/// One `SwitchGLU`'s three projections, owned so the borrowed [`ExpertBank`]
/// can be handed out repeatedly.
///
/// Every bundle names them `{prefix}.{gate,up,down}_proj.weight`. A synthetic
/// bank is held whole, which the checkpoint's 25 GB of routed experts cannot be
/// — those are decoded per expert per call instead.
pub struct Bank {
    experts: usize,
    dim: usize,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl Bank {
    pub fn load(ckpt: &Checkpoint, prefix: &str, experts: usize, dim: usize) -> Self {
        let of = |name: &str| f32s(&tensor(ckpt, &format!("{prefix}.{name}.weight")));
        Self {
            experts,
            dim,
            gate_proj: of("gate_proj"),
            up_proj: of("up_proj"),
            down_proj: of("down_proj"),
        }
    }

    /// `[rows, dim]` through one of its experts.
    pub fn expert(&self, index: usize, rows: &[f32]) -> Vec<f32> {
        ExpertBank::new(
            self.experts,
            self.dim,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        )
        .expert(index)
        .forward(rows)
    }
}

/// The worst absolute disagreement with a reference tensor, as a fraction of
/// that tensor's largest value.
///
/// Scaled by the tensor rather than element by element: an output that lands
/// near zero by cancellation has no meaningful relative error of its own, and
/// every op here reduces over an axis where cancellation is ordinary.
pub fn deviation(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(0.0f32, |worst, w| worst.max(w.abs()));
    assert!(scale > 0.0, "reference tensor is all zeros");
    got.iter()
        .zip(want)
        .fold(0.0f32, |worst, (got, want)| worst.max((got - want).abs()))
        / scale
}
