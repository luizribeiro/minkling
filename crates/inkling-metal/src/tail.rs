//! The back of the model, on the device: the final norm, the muP divide,
//! `lm_head` behind them, and the argmax behind that.
//!
//! Everything else in this crate is an operation of a layer. This is the three
//! lines `LanguageModel._logits_from_norm` is, put where the rows they read
//! already are — so that the last thing a decode step does is not a round trip,
//! and so that an MTP head's guess comes back out of the head's own command
//! buffer rather than out of a second one behind it.
//!
//! **And the fourth line is here too.** What a caller wants of a row of logits
//! is which id it names, and taking that on this side meant a row of 200058
//! floats crossing back for a pass that produces four bytes. [`crate::argmax`]
//! is that pass as two dispatches in the same command buffer, and what it had
//! to buy first is the tie rule the engine's token identity rests on — see its
//! own module documentation, where the exactness argument is made.
//!
//! # Why the divide is in the weight
//!
//! The reference divides the normed hidden state by
//! `logits_mup_width_multiplier` and only then projects, and the divide has to
//! happen between the two: `[`inkling_core::head`]` says what dropping it costs
//! and what doing it after the projection would cost. A dispatch of its own for
//! one multiply is a dispatch, and the norm's own per-row scale is a *multiply*
//! where the reference divides — so what is here is neither. **The multiplier
//! is divided into a copy of the norm's weight**, once, at wrap time.
//!
//! That is exact rather than close, and only because of what it is exact
//! against. `w/m` is exact when `m` is a power of two, and scaling by a power of
//! two commutes with rounding — so `a * (w/m)` and `(a * w)/m` are the same
//! float, and the normed row this produces is the normed row the CPU produces
//! divided by the multiplier, to the bit. [`ModelTail::wrap`] checks that rather
//! than assuming it, over every value of the weight, and answers `None` for a
//! checkpoint whose multiplier does not divide cleanly — which leaves the tail
//! where it always ran rather than moving a logit.
//!
//! What is *not* exact is the norm itself, and nothing here pretends otherwise:
//! [`crate::norm`] reduces a row across simdgroups where the CPU accumulates it
//! in f64, so the two land a few ulps apart. That is the same difference every
//! other norm in this engine already has, and what says it does not reach a
//! token is a comparison of the two tails over real logits rather than an
//! argument.
//!
//! # Two norms, and when the second is dispatched
//!
//! The undivided norm is what an MTP head is chained from and the divided one is
//! what the projection reads, so a call that wants both dispatches both. A round
//! that speculates nothing wants only the second and encodes only the second;
//! and where the block is the whole call — which every decode step is — the two
//! would be the same rows, so the chain's copy is the one that reads them all
//! and the projection's is the one over the block.

use inkling_core::head::{Tail, Tailed};
use inkling_core::profile::{self, Op};
use inkling_core::weights::Packed;

use crate::argmax::{GreedyArgmax, Vocabulary};
use crate::buffer::Buffer;
use crate::device::Device;
use crate::kernel::Batch;
use crate::matmul::{PackedMatmul, PackedProjection};
use crate::norm::{LayerNorm, RmsNorm};
use crate::projections::ProjectionError;

/// What the back of the model is made of, as the checkpoint holds it.
///
/// The norm's weight travels widened and `lm_head` travels packed, which is the
/// same split every layer's handover already makes: a norm has no packed form to
/// be left in and 16 KB to copy, and the head is 411 MB of codes the device
/// reads where the checkpoint mapped them.
#[derive(Debug, Clone)]
pub struct TailWeights<'a> {
    /// `[hidden]`, the weight of the model's final norm.
    ///
    /// Owned where the head is borrowed, because the two are not the same kind
    /// of tensor: 16 KB widened once out of bfloat16, against 411 MB of codes
    /// the device reads where the checkpoint mapped them.
    pub norm: Vec<f32>,
    /// The `rms_norm_eps` that norm shares with every other norm in the model.
    pub eps: f32,
    /// `logits_mup_width_multiplier`, which the normed state is divided by.
    pub mup: f32,
    /// `lm_head` itself, still packed.
    pub head: Packed<'a>,
    /// How many of its rows are vocabulary, which is where the projection is
    /// cut — see [`PackedProjection::wrap_packed`].
    pub vocab: usize,
}

/// The model's final norm and `lm_head`, on the device, with the muP divide
/// between them.
///
/// Held by whatever ran the rows it reads — [`crate::ModelLayers`] for the
/// stack's last layer and [`crate::ModelHeads`] for a head's — because what
/// makes this worth having is that the rows are already there. Each holds its
/// own: a wrap is a binding over pages the checkpoint owns and a 16 KB copy of a
/// norm, so two of them cost two bindings rather than two of anything.
#[derive(Debug)]
pub struct ModelTail<'a> {
    /// The final norm as the model states it, which is what an MTP head is
    /// chained from.
    norm: LayerNorm<'a>,
    /// The same norm with the muP divide folded into its weight — see the
    /// module documentation for why that is the same value and not a near one.
    divided: LayerNorm<'a>,
    head: PackedProjection<'a>,
    /// The argmax over what the projection produced, which is the last thing
    /// past the model — see [`crate::argmax`] for the tie rule it has to
    /// reproduce before it may be here at all.
    argmax: &'a GreedyArgmax,
    /// The ids a row of this head's output holds, which is where the projection
    /// was already cut and so where the argmax stops.
    vocab: Vocabulary,
}

impl<'a> ModelTail<'a> {
    /// The tail wrapped where the checkpoint mapped it, and `None` where the
    /// muP multiplier does not divide the norm's weight exactly.
    ///
    /// **A refusal rather than a rounding**, because of what the alternative
    /// would be: a tail whose divide lands an ulp from the reference's is a
    /// tail that can move a logit, and a moved logit at the top of the
    /// distribution is a different token. A checkpoint this answers `None` for
    /// runs exactly as it did before — the norm and the divide on the CPU, and
    /// `lm_head` in a submission of its own.
    pub fn wrap(
        device: &'a Device,
        rms: &'a RmsNorm,
        matmul: &'a PackedMatmul,
        argmax: &'a GreedyArgmax,
        weights: &TailWeights<'a>,
    ) -> Result<Option<Self>, ProjectionError> {
        let Some(divided) = divided(&weights.norm, weights.mup) else {
            return Ok(None);
        };
        Ok(Some(Self {
            norm: LayerNorm::new(device, rms, &weights.norm, weights.eps)?,
            divided: LayerNorm::new(device, rms, &divided, weights.eps)?,
            head: PackedProjection::wrap_packed(device, matmul, &weights.head, weights.vocab)?,
            argmax,
            vocab: Vocabulary::of(weights.vocab),
        }))
    }

    /// The width a row of the hidden state is.
    pub fn hidden(&self) -> usize {
        self.norm.width()
    }

    /// The tail encoded into `batch` over the rows `x` holds, and what has to be
    /// read once it has run.
    ///
    /// Nothing here waits. The caller owns the command buffer — it is the run of
    /// layers or the head this is behind — and what closes it is the same thing
    /// that closed it before this existed.
    pub(crate) fn encode_into(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        want: Tail,
    ) -> Result<Landed, ProjectionError> {
        let rows = x.len() / self.hidden();
        assert!(want.block <= rows, "a block longer than the call");

        let chained = match want.chained {
            true => Some(self.norm.encode(batch, x)?),
            false => None,
        };
        let mut block = self.divided.encode_last(batch, x, want.block)?;
        let mut logits = self.head.encode_over(batch, &mut block)?.buffer();
        let picks = self.argmax.encode(batch, &mut logits, self.vocab)?;
        Ok(Landed {
            chained,
            logits: want.logits.then_some(logits),
            picks,
        })
    }
}

/// What a tail leaves on the device for the call that submitted it to read.
///
/// Two buffers rather than a value, for [`crate::LayerNorm::encode`]'s reason:
/// what is in them is not this process's until somebody waits, and who waits is
/// the caller.
pub(crate) struct Landed {
    /// The undivided normed rows, where [`Tail::chained`] asked for them.
    chained: Option<Buffer<f32>>,
    /// The block's own logits, where [`Tail::logits`] asked for them — which a
    /// generation never does, the ids being what it takes from a row.
    ///
    /// The buffer is dropped rather than never allocated, because the argmax is
    /// what reads it: the flag decides whether it crosses back and not whether
    /// it exists.
    logits: Option<Buffer<f32>>,
    /// The id each row of the block names, taken where the row was written.
    picks: Buffer<u32>,
}

impl Landed {
    /// Both halves read back, once the command buffer they were encoded into
    /// has completed.
    pub(crate) fn read(&self) -> Tailed {
        profile::timed(Op::Readback, || Tailed {
            normed: self
                .chained
                .as_ref()
                .map(Buffer::to_vec)
                .unwrap_or_default(),
            logits: self.logits.as_ref().map(Buffer::to_vec).unwrap_or_default(),
            picks: widened(&self.picks),
        })
    }

    /// The id the block's last row names, for the caller that wants nothing
    /// else of the tail — a head is chained from its own state and not from the
    /// model's normed one, and nothing in a chain ever reads a head's logits.
    ///
    /// **Four bytes where [`Landed::read`] crosses a row of 800 KB**, which is
    /// what the argmax being on the device is worth to a chain: eight heads are
    /// eight rows of the vocabulary that stay where the projection wrote them.
    pub(crate) fn guess(&self) -> usize {
        profile::timed(Op::Readback, || {
            *self.picks.as_slice().last().expect("a block names an id") as usize
        })
    }
}

/// The ids a dispatch named, as the indices this side counts in.
fn widened(picks: &Buffer<u32>) -> Vec<usize> {
    picks.as_slice().iter().map(|id| *id as usize).collect()
}

/// The norm's weight with the muP multiplier divided into it, and `None` where
/// folding the divide in would move a bit.
///
/// **Two conditions, and neither is the division undone.** A weight that
/// multiplies back to what it was proves nothing — `0.3 / 12.0 * 12.0` is
/// `0.3` — because rounding twice can land where rounding never happened. What
/// is actually needed is that the divide *commutes* with the multiply the
/// kernel then does: `a * (w/m)` and `(a * w)/m` are the same float exactly
/// when scaling by `m` moves an exponent and nothing else.
///
/// So the multiplier has to be a positive power of two, which is its mantissa
/// being empty; and every divided weight has to still be normal, since a value
/// that fell into the subnormals lost the bits at the bottom rather than
/// shifting them.
fn divided(norm: &[f32], mup: f32) -> Option<Vec<f32>> {
    assert!(mup != 0.0, "the muP width multiplier divides");
    if !mup.is_normal() || mup.is_sign_negative() || mup.to_bits() & MANTISSA != 0 {
        return None;
    }
    let divided: Vec<f32> = norm.iter().map(|w| w / mup).collect();
    divided
        .iter()
        .all(|w| *w == 0.0 || w.is_normal())
        .then_some(divided)
}

/// The bits of an `f32` below its exponent, which a power of two has none of.
const MANTISSA: u32 = (1 << f32::MANTISSA_DIGITS.saturating_sub(1)) - 1;

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's own multiplier, which is what the fold rests on being.
    const MUP: f32 = 16.0;

    /// **The whole of why the divide can live in the weight.** A power of two
    /// shifts an exponent and touches no mantissa bit, so the folded weight is
    /// the weight divided and the product is the product divided.
    #[test]
    fn a_power_of_two_divides_a_norms_weight_exactly() {
        let norm: Vec<f32> = (0..64).map(|i| 0.3 + (i % 13) as f32 / 7.0).collect();
        let divided = divided(&norm, MUP).expect("a power of two divides");

        for (was, divided) in norm.iter().zip(&divided) {
            assert_eq!(*divided, was / MUP);
            for row in [-3.25e3, -1.0, 1e-7, 0.5, 7.75, 4.1e6] {
                assert_eq!(
                    row * divided,
                    (row * was) / MUP,
                    "the fold moved a product of {row} and {was}"
                );
            }
        }
    }

    /// A multiplier that is not a power of two leaves the tail where it was
    /// rather than an ulp from where the reference puts it.
    ///
    /// **The weight that would have passed a round trip is the case worth
    /// naming.** `0.3 / 12.0 * 12.0` is `0.3` and `0.3 / 12.0` is not exact, so
    /// a check written as the division undone would have folded this multiplier
    /// in and moved a product.
    #[test]
    fn a_multiplier_that_does_not_divide_exactly_is_declined() {
        assert_eq!(0.3f32 / 12.0 * 12.0, 0.3, "the round trip says nothing");
        assert_ne!(
            0.109_375_f32 * (0.3f32 / 12.0),
            (0.109_375_f32 * 0.3) / 12.0,
            "and the product it says nothing about is this one"
        );

        let norm: Vec<f32> = (0..64).map(|i| 0.3 + (i % 13) as f32 / 7.0).collect();
        assert!(divided(&norm, 12.0).is_none());
        assert!(
            divided(&norm, 1.0).is_some(),
            "dividing by one loses nothing"
        );
    }

    /// A weight small enough that the division falls into the subnormals loses
    /// bits at the bottom rather than shifting them, which is the one way a
    /// power of two can still be inexact.
    #[test]
    fn a_weight_the_division_would_flush_is_declined() {
        assert!(divided(&[1.0, f32::MIN_POSITIVE * 1.5], MUP).is_none());
        assert!(divided(&[1.0, 0.0], MUP).is_some(), "zero divides exactly");
    }
}
