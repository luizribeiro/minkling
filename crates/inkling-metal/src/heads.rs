//! The multi-token prediction heads' multiplies, on the device.
//!
//! [`inkling_core::mtp`] is the authority on what a head computes; this is
//! where its eight weights are multiplied. They are the one part of the model
//! the quantisers did not touch — bfloat16, 532 MiB a head — so every one of
//! them goes through [`crate::dense`], the kernel the routers' gates already
//! use, rather than through the packed matmul the rest of the model does.
//!
//! **The seam is one step out from a layer's.** A decoder layer hands the
//! device everything between one hidden state and the next; a head hands it the
//! eight multiplies and keeps the rest here — the two norms in front of it, the
//! concatenation between them, its own attention step, its convolutions and its
//! activation. That is the shape this engine's layers had before they were
//! merged.
//!
//! **What it costs is measured now and it is the larger of the two numbers.**
//! `which_kernels_own_a_chain_of_heads` prices a chain of eight at 88
//! dispatches in 48 submissions — six a head, against a whole decode step's
//! fifteen — and about 390 microseconds of round trip on each, which is about
//! half of a 4.5 ms guess against the 2.2 ms the device executes for. What made
//! this seam defensible was the study's reading that a
//! block's extra token is a third of a decode step where a head's whole chain is
//! a tenth; a chain of eight is 1.71 decode steps. Merging a head into one
//! command buffer means generalising every kernel a layer uses over a second
//! weight format, and the README's speculation section is where that trade now
//! stands.
//!
//! **The fused gate and up are two weights over one tensor.** `w13_dn` holds
//! them interleaved row by row, and
//! [`DenseWeight::wrap_rows`](crate::DenseWeight::wrap_rows) reads every other
//! row of it — so both projections are the checkpoint's own bytes where the
//! mapping put them, and nothing is de-interleaved into memory.

use inkling_core::attention::{AttentionStep, Projections, Qkvr};
use inkling_core::mtp::{HeadBackend, HeadPacked};
use inkling_core::ops::{MlpProjections, Projection};

use crate::attention::{FusedAttention, LayerAttention, Step};
use crate::dense::{DenseMatmul, DenseWeight};
use crate::device::Device;
use crate::matmul::together;
use crate::projections::ProjectionError;

/// One head's eight weights on the device.
#[derive(Debug)]
pub struct HeadDevice<'a> {
    input_proj: DenseWeight<'a>,
    attention: HeadAttention<'a>,
    mlp: HeadFfn<'a>,
}

/// A head's five attention projections and the step between them, which are a
/// layer's over a weight format the quantiser left alone.
#[derive(Debug)]
pub struct HeadAttention<'a> {
    /// The attention step, resident for the reason
    /// [`LayerProjections`](crate::LayerProjections)'s is: what it holds is the
    /// head's own band, and what running it here buys is the `[heads, queries,
    /// keys]` scores and the mask beside them never being built at all. On the
    /// CPU those are `queries * keys * heads` multiply-adds a head a round,
    /// which is where a chain's cost stops being its weights and starts being
    /// the context.
    step: LayerAttention<'a>,
    q_proj: DenseWeight<'a>,
    k_proj: DenseWeight<'a>,
    v_proj: DenseWeight<'a>,
    r_proj: DenseWeight<'a>,
    o_proj: DenseWeight<'a>,
}

/// A head's feed-forward network: the two halves of `w13_dn`, and `w2_md`.
#[derive(Debug)]
pub struct HeadFfn<'a> {
    gate_proj: DenseWeight<'a>,
    up_proj: DenseWeight<'a>,
    down_proj: DenseWeight<'a>,
}

/// Every head's weights, wrapped where the checkpoint mapped them.
#[derive(Debug)]
pub struct ModelHeads<'a> {
    heads: Vec<HeadDevice<'a>>,
}

impl<'a> ModelHeads<'a> {
    /// Wrap every weight `heads` names, holding none of them: 4.2 GiB of
    /// bfloat16 across the eight, handed to the GPU where the checkpoint mapped
    /// it and costing no resident set of its own — the same bargain
    /// [`ModelLayers::wrap`](crate::ModelLayers::wrap) makes for the stack.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a DenseMatmul,
        attention: &'a FusedAttention,
        heads: &[HeadPacked<'a>],
    ) -> Result<Self, ProjectionError> {
        let wrapped = heads
            .iter()
            .map(|head| {
                let whole = |weight| DenseWeight::wrap(device, matmul, weight);
                Ok(HeadDevice {
                    input_proj: whole(&head.input_proj)?,
                    attention: HeadAttention {
                        step: LayerAttention::new(device, attention, head.config, &head.rel_proj)?,
                        q_proj: whole(&head.attention.q_proj)?,
                        k_proj: whole(&head.attention.k_proj)?,
                        v_proj: whole(&head.attention.v_proj)?,
                        r_proj: whole(&head.attention.r_proj)?,
                        o_proj: whole(&head.attention.o_proj)?,
                    },
                    mlp: HeadFfn {
                        gate_proj: DenseWeight::wrap_rows(device, matmul, &head.w13, 0, 2)?,
                        up_proj: DenseWeight::wrap_rows(device, matmul, &head.w13, 1, 2)?,
                        down_proj: whole(&head.w2)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, ProjectionError>>()?;
        Ok(Self { heads: wrapped })
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
        Some(&self.heads.get(head)?.attention as &dyn Projections)
    }

    fn mlp(&self, head: usize) -> Option<&dyn MlpProjections> {
        Some(&self.heads.get(head)?.mlp as &dyn MlpProjections)
    }
}

impl Projections for HeadAttention<'_> {
    /// The four that read one input, in one command buffer.
    ///
    /// The default runs them as four submissions and answers the same numbers;
    /// what this saves is three round trips of about 250 microseconds each,
    /// against four multiplies that are 3 MB of bfloat16 between them. It is
    /// the same trade [`LayerProjections::qkvr`](crate::LayerProjections) makes
    /// one step further in, where the norm in front of them is on the device
    /// too.
    fn qkvr(&self, x: &[f32]) -> Qkvr {
        let [q, k, v, r] = together(self.q_proj.device(), |batch| {
            let mut input = self.q_proj.device().buffer(x)?;
            Ok([
                self.q_proj.encode_over(batch, &mut input)?,
                self.k_proj.encode_over(batch, &mut input)?,
                self.v_proj.encode_over(batch, &mut input)?,
                self.r_proj.encode_over(batch, &mut input)?,
            ])
        })
        .unwrap_or_else(|err| panic!("a head's projections did not run: {err}"));
        Qkvr { q, k, v, r }
    }

    /// The attention step and `o_proj`, in one command buffer.
    ///
    /// The mirror of [`LayerProjections::attend`](crate::LayerProjections), and
    /// the same argument: the step multiplies activations against activations,
    /// so what a backend does differently with it is decline to *build* a
    /// tensor — the `[heads, queries, keys]` mask the reference materialises is
    /// derived per element inside the kernel instead.
    ///
    /// The span is handed over as a slice on every call, where a layer's is
    /// held on the device between them. That is the copy a head still pays and
    /// a layer does not: the heads' caches are the CPU's, because a head is
    /// rewound a row every round and what rewinds it is the proposer.
    fn attend(&self, step: AttentionStep<'_>) -> Vec<f32> {
        let [out] = together(self.q_proj.device(), |batch| {
            let mut attended = self.step.encode(
                batch,
                Step {
                    q: step.q,
                    k: step.k,
                    v: step.v,
                    rel: step.rel,
                    taus: step.taus,
                    q_offset: step.q_offset,
                },
            )?;
            Ok([self.o_proj.encode_over(batch, &mut attended)?])
        })
        .unwrap_or_else(|err| panic!("a head's attention step did not run: {err}"));
        out
    }

    fn q_proj(&self) -> &dyn Projection {
        &self.q_proj
    }

    fn k_proj(&self) -> &dyn Projection {
        &self.k_proj
    }

    fn v_proj(&self) -> &dyn Projection {
        &self.v_proj
    }

    fn r_proj(&self) -> &dyn Projection {
        &self.r_proj
    }

    fn o_proj(&self) -> &dyn Projection {
        &self.o_proj
    }
}

impl MlpProjections for HeadFfn<'_> {
    /// The two halves of the fused weight, in one command buffer — which is
    /// what the interleave costs and does not cost: two dispatches over one
    /// mapping, where a layer's separate gate and up are two dispatches over
    /// two.
    fn gate_up(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let [gate, up] = together(self.gate_proj.device(), |batch| {
            let mut input = self.gate_proj.device().buffer(x)?;
            Ok([
                self.gate_proj.encode_over(batch, &mut input)?,
                self.up_proj.encode_over(batch, &mut input)?,
            ])
        })
        .unwrap_or_else(|err| panic!("a head's feed-forward network did not run: {err}"));
        (gate, up)
    }

    fn gate_proj(&self) -> &dyn Projection {
        &self.gate_proj
    }

    fn up_proj(&self) -> &dyn Projection {
        &self.up_proj
    }

    fn down_proj(&self) -> &dyn Projection {
        &self.down_proj
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
