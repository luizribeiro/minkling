//! The multi-token prediction heads' multiplies, on the device.
//!
//! [`inkling_core::mtp`] is the authority on what a head computes; this is
//! where its eight weights are multiplied. They are the one part of the model
//! the quantisers did not touch — bfloat16, 532 MiB a head — so every one of
//! them goes through [`crate::dense`], the kernel the routers' gates already
//! use, rather than through the packed matmul the rest of the model does.
//!
//! **A head's block is a decoder layer and is wrapped as one.** What the
//! quantiser left it in is the only thing that differs: every weight of the
//! model's own forty-two layers is MXFP4 and every weight of these eight is
//! bfloat16, so the five projections and the feed-forward network go through
//! [`crate::dense`] where a layer's go through the packed matmul — and
//! [`Multiply`](crate::Multiply) is the seam that says that is the whole of it.
//! Everything around them is the layer's own: its norms, its four convolutions,
//! its head norms and the attention step, which
//! [`LayerDevice`](crate::LayerDevice) already holds.
//!
//! **So a head's state is where a layer's is.** The span it attends over and
//! the four convolution windows behind it are the device's, held between rounds
//! and rewound by [`ModelHeads::rewind`] rather than copied over on every call
//! — which is what a head has to have before its dispatches can share one
//! command buffer, and what `slack` at wrap time is for.
//!
//! **The fused gate and up are two weights over one tensor.** `w13_dn` holds
//! them interleaved row by row, and
//! [`DenseWeight::wrap_rows`](crate::DenseWeight::wrap_rows) reads every other
//! row of it — so both projections are the checkpoint's own bytes where the
//! mapping put them, and nothing is de-interleaved into memory.

use inkling_core::attention::Projections;
use inkling_core::head::Tail;
use inkling_core::layer::{DecoderCache, Passed};
use inkling_core::mtp::{Guessed, HeadBackend, HeadDevice, HeadPacked, HeadStep};
use inkling_core::ops::{MlpProjections, Projection};
use inkling_core::profile::{self, Op};

use crate::dense::{DenseMatmul, DenseWeight};
use crate::device::Device;
use crate::matmul::{MatmulError, Multiply};
use crate::projections::{
    Block, DenseFfn, LayerDevice, LayerKernels, LayerMlpDevice, ProjectionError, Wrapping,
};
use crate::swiglu::SwiGlu;
use crate::tail::{Landed, ModelTail};

/// One head on the device: the projection that reads the pair it is handed, and
/// the decoder layer behind it.
///
/// Not public, where [`crate::LayerDevice`] is: a caller reaches a layer through
/// the stack that holds it, and a head only ever through [`ModelHeads`].
#[derive(Debug)]
struct WrappedHead<'a> {
    input_proj: DenseWeight<'a>,
    block: LayerDevice<'a>,
}

/// Every head's weights, wrapped where the checkpoint mapped them.
#[derive(Debug)]
pub struct ModelHeads<'a> {
    heads: Vec<WrappedHead<'a>>,
    /// The model's own final norm, muP divide and `lm_head`, where this holds
    /// them — which is the second half of a head's command buffer and half of
    /// a chain's submissions.
    ///
    /// A head's rows have to *be* a token before the head after it can embed
    /// one, so the guess is what closes the buffer either way; what the tail
    /// decides is whether reading it costs another. `None` leaves it where it
    /// was, one submission behind each of the eight.
    tail: Option<ModelTail<'a>>,
}

impl<'a> ModelHeads<'a> {
    /// Wrap every weight `heads` names, holding none of them: 4.2 GiB of
    /// bfloat16 across the eight, handed to the GPU where the checkpoint mapped
    /// it and costing no resident set of its own — the same bargain
    /// [`ModelLayers::wrap`](crate::ModelLayers::wrap) makes for the stack.
    ///
    /// `slack` is how many timesteps a head has to be able to give back, which
    /// for a chain is the frontier row every round runs and takes back again —
    /// see [`FRONTIER`](inkling_core::mtp::FRONTIER). A head wrapped with none
    /// holds its state as firmly as a layer of a run that never speculates.
    pub fn wrap(
        device: &'a Device,
        kernels: &'a LayerKernels,
        matmul: &'a DenseMatmul,
        swiglu: &'a SwiGlu,
        heads: &[HeadPacked<'a>],
        tail: Option<ModelTail<'a>>,
        slack: usize,
    ) -> Result<Self, ProjectionError> {
        let wrapped = heads
            .iter()
            .map(|head| {
                let whole = |weight| -> Result<Box<dyn Multiply + 'a>, MatmulError> {
                    Ok(Box::new(DenseWeight::wrap(device, matmul, weight)?))
                };
                // The gate is the fused tensor's even rows and the up its odd
                // ones — see the module documentation.
                let interleaved = |first| -> Result<Box<dyn Multiply + 'a>, MatmulError> {
                    Ok(Box::new(DenseWeight::wrap_rows(
                        device, matmul, &head.w13, first, 2,
                    )?))
                };
                Ok(WrappedHead {
                    input_proj: DenseWeight::wrap(device, matmul, &head.input_proj)?,
                    block: LayerDevice::wrapping(
                        device,
                        kernels,
                        Wrapping {
                            config: head.config,
                            q_proj: whole(&head.attention.q_proj)?,
                            k_proj: whole(&head.attention.k_proj)?,
                            v_proj: whole(&head.attention.v_proj)?,
                            r_proj: whole(&head.attention.r_proj)?,
                            o_proj: whole(&head.attention.o_proj)?,
                            input_layernorm: &head.input_layernorm,
                            q_norm: &head.q_norm,
                            k_norm: &head.k_norm,
                            k_sconv: &head.k_sconv,
                            v_sconv: &head.v_sconv,
                            rel_proj: &head.rel_proj,
                        },
                        Block {
                            dim: head.config.hidden,
                            attn_sconv: &head.attn_sconv,
                            post_attention_layernorm: &head.post_attention_layernorm,
                            mlp: Some(LayerMlpDevice::Dense(Box::new(DenseFfn::over(
                                interleaved(0)?,
                                interleaved(1)?,
                                whole(&head.w2)?,
                                swiglu,
                            )))),
                            mlp_sconv: &head.mlp_sconv,
                        },
                        slack,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, ProjectionError>>()?;
        Ok(Self {
            heads: wrapped,
            tail,
        })
    }

    pub fn heads(&self) -> usize {
        self.heads.len()
    }
}

impl HeadBackend for ModelHeads<'_> {
    fn input_proj(&self, head: usize) -> Option<&dyn Projection> {
        Some(&self.heads.get(head)?.input_proj as &dyn Projection)
    }

    fn attention(&self, head: usize) -> Option<&dyn Projections> {
        Some(self.heads.get(head)?.block.attention() as &dyn Projections)
    }

    fn mlp(&self, head: usize) -> Option<&dyn MlpProjections> {
        Some(self.heads.get(head)?.block.dense_mlp()? as &dyn MlpProjections)
    }

    /// A head whose projection *and* whose block are both here, which is the
    /// condition for one command buffer — and `None` for one this does not
    /// hold, which still runs, a submission a piece.
    fn device(&self, head: usize) -> Option<&dyn HeadDevice> {
        self.heads.get(head)?;
        Some(self as &dyn HeadDevice)
    }

    /// Take back the last `rows` timesteps of everything head `head` holds for
    /// the sequence in flight: its keys and values, and the four convolution
    /// windows around them.
    ///
    /// **A head is rewound every round and a layer only after a rejection**,
    /// which is the one way the two differ: the row a chain guessed from is a
    /// row the model has not been asked about yet, so the round after it runs
    /// that position again against the token the model produced.
    fn rewind(&self, head: usize, rows: usize) {
        if let Some(held) = self.heads.get(head) {
            held.block.rewind(rows);
        }
    }
}

/// **One command buffer a head**, where the same eight multiplies asked for a
/// piece at a time were six submissions and four CPU rows between them.
///
/// Where a chain has to end one is the guess: a head's rows have to be a token
/// before the head after it can embed it, and turning a hidden state into a
/// token is `lm_head` and an argmax over 201024 logits. So a head waits once,
/// and what it waits for is everything between the pair of normed rows it was
/// handed and the `[rows, hidden]` it answers with.
impl HeadDevice for ModelHeads<'_> {
    fn run(&self, head: usize, cache: &mut DecoderCache, step: HeadStep<'_>) -> Option<Guessed> {
        let held = self.heads.get(head)?;
        Some(
            held.run(cache, step, self.tail.as_ref())
                .unwrap_or_else(|err| panic!("the head did not run: {err}")),
        )
    }
}

impl WrappedHead<'_> {
    /// The projection and the block encoded into one command buffer, submitted
    /// and waited for.
    ///
    /// Nothing between the `[rows, 2 * hidden]` this is handed and the `[rows,
    /// hidden]` it answers with is a value this process forms: what
    /// `input_proj` produces is what the block's input layernorm reads, and the
    /// twenty dispatches behind that are a layer's own.
    fn run(
        &self,
        cache: &mut DecoderCache,
        step: HeadStep<'_>,
        tail: Option<&ModelTail<'_>>,
    ) -> Result<Guessed, ProjectionError> {
        let device = self.input_proj.device();
        let mut batch = device.batch()?;
        let mut input = device.buffer(step.input)?;
        let mut x = self
            .input_proj
            .encode_over(&mut batch, &mut input)?
            .buffer();
        let mut rows = self
            .block
            .encode_into(&mut batch, cache, step.block, &mut x)?;
        // The last row alone, and undivided rows nobody wants: what a head is
        // chained from is its own state, which is these rows before any norm,
        // and what the guess needs is the model's tail over the last of them.
        let landed = tail
            .map(|tail| {
                tail.encode_into(
                    &mut batch,
                    &mut rows,
                    Tail {
                        block: 1,
                        chained: false,
                    },
                )
            })
            .transpose()?;
        batch.wait()?;
        Ok(Guessed {
            hidden: Passed::Rows(profile::timed(Op::Readback, || rows.to_vec())),
            logits: landed.as_ref().map(Landed::logits),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::fixture::deviation;
    use inkling_core::weights::Bf16;

    use crate::dense::testing::{narrowed, widened};
    use crate::testing::device;

    /// How far a dispatch may land from the CPU's answer over the same bytes.
    ///
    /// Both sides widen the same bfloat16 to the same float32 and reduce the
    /// same products; what separates them is the order of the reduction, which
    /// the kernel splits across a run of simdgroups. Worst observed when this
    /// landed: 1.3e-7, a few ulps of the tensor's peak.
    const TOLERANCE: f32 = 1e-6;

    const IN_DIM: usize = 64;
    const ROWS: usize = 12;

    /// A weight whose rows are all different, so a row read from the wrong
    /// place is a different answer rather than the same one.
    fn weight(rows: usize, salt: usize) -> Vec<f32> {
        (0..rows * IN_DIM)
            .map(|i| ((i * 7 + salt * 13) % 23) as f32 / 16.0 - 0.7)
            .collect()
    }

    fn rows(count: usize) -> Vec<f32> {
        (0..count * IN_DIM)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 9.0)
            .collect()
    }

    /// The CPU's own multiply over the same weight, which is the oracle every
    /// kernel in this tree is checked against.
    fn on_the_cpu(weight: &[f32], x: &[f32]) -> Vec<f32> {
        inkling_core::linear(x, weight, IN_DIM)
    }

    /// **The interleave, on the device.** `w13_dn` holds a head's gate and its
    /// up in one tensor, even rows and odd, and the two weights this wraps over
    /// it read the checkpoint's own bytes in place. A stride read the other way
    /// round — or not read at all — is a projection of the right shape drawn
    /// from the wrong rows, which still multiplies.
    #[test]
    fn the_two_halves_of_a_fused_weight_are_its_even_rows_and_its_odd_ones() {
        let Some(device) = device() else { return };
        let matmul = DenseMatmul::new(&device).expect("the kernel compiles");

        let fused = weight(2 * ROWS, 1);
        let bytes = narrowed(&fused);
        let tensor = Bf16::over(&bytes, 2 * ROWS, IN_DIM);
        let gate = DenseWeight::wrap_rows(&device, &matmul, &tensor, 0, 2).expect("the gate wraps");
        let up = DenseWeight::wrap_rows(&device, &matmul, &tensor, 1, 2).expect("the up wraps");
        assert_eq!((gate.out_dim(), up.out_dim()), (ROWS, ROWS));

        // The same de-interleave, spelled out on this side.
        let widened = widened(&bytes);
        let half = |first: usize| -> Vec<f32> {
            widened
                .chunks_exact(IN_DIM)
                .skip(first)
                .step_by(2)
                .flatten()
                .copied()
                .collect()
        };
        let x = rows(2);
        for (what, got, want) in [
            ("the gate", gate.forward(&x), on_the_cpu(&half(0), &x)),
            ("the up", up.forward(&x), on_the_cpu(&half(1), &x)),
        ] {
            let agreed = deviation(&got, &want);
            assert!(agreed <= TOLERANCE, "{what}: deviation {agreed:e}");
        }
        assert!(
            deviation(&gate.forward(&x), &on_the_cpu(&half(1), &x)) > TOLERANCE,
            "the two halves agree, so nothing here says which is which"
        );
    }

    /// A weight that is a whole tensor is the weight this kernel always
    /// multiplied — a stride of one over row zero, which is what
    /// [`DenseWeight::wrap`] is.
    #[test]
    fn a_weight_of_every_row_is_the_tensor_itself() {
        let Some(device) = device() else { return };
        let matmul = DenseMatmul::new(&device).expect("the kernel compiles");

        let values = weight(ROWS, 2);
        let bytes = narrowed(&values);
        let tensor = Bf16::over(&bytes, ROWS, IN_DIM);
        let whole = DenseWeight::wrap(&device, &matmul, &tensor).expect("the weight wraps");
        let strided =
            DenseWeight::wrap_rows(&device, &matmul, &tensor, 0, 1).expect("the weight wraps");

        let x = rows(3);
        assert_eq!(whole.forward(&x), strided.forward(&x));
        let agreed = deviation(&whole.forward(&x), &on_the_cpu(&widened(&bytes), &x));
        assert!(agreed <= TOLERANCE, "deviation {agreed:e}");
    }
}
