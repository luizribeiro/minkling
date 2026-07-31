//! Test-only access to the committed reference fixtures.
//!
//! Each is a safetensors bundle under `reference/fixtures`, written by a
//! `just dump-*` recipe and read back through [`Checkpoint`]'s single-file
//! layout.

use std::path::PathBuf;

use crate::checkpoint::{Checkpoint, Dtype, TensorView};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/fixtures");

pub fn open(file: &str) -> Checkpoint {
    let path = PathBuf::from(DIR).join(file);
    Checkpoint::open(&path).unwrap_or_else(|err| panic!("{file} opens: {err}"))
}

pub fn tensor<'a>(ckpt: &'a Checkpoint, name: &str) -> TensorView<'a> {
    ckpt.tensor(name)
        .unwrap_or_else(|err| panic!("fixture holds {name}: {err}"))
}

/// A fixture tensor's values. Every dump casts to float32 before saving, so a
/// comparison never has to reason about the reference's dtype choices.
pub fn f32s(view: &TensorView<'_>) -> Vec<f32> {
    assert_eq!(view.dtype(), Dtype::F32);
    view.data()
        .chunks_exact(size_of::<f32>())
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunked into floats")))
        .collect()
}
