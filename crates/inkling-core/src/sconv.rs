//! The short causal convolution Inkling uses in place of RoPE.
//!
//! Every layer runs four of them — on the key and value projections, on the
//! attention output and on the MLP output — over a kernel of four timesteps.
//! Together with the banded relative-logit bias they are the only thing that
//! tells the model where a token sits, so an implementation that is subtly
//! wrong still produces fluent text.
//!
//! Pinned to mlx-vlm by `reference/fixtures/sconv.safetensors`, whose synthetic
//! cases are float32 throughout, and by the trained weights in that bundle
//! against the activation pairs in `layer_activations.safetensors`.

use crate::profile::{self, Op};

/// A depthwise causal convolution with a residual add: one kernel per channel,
/// no bias, no mixing across channels.
#[derive(Debug, Clone, Copy)]
pub struct ShortConv<'a> {
    channels: usize,
    kernel_size: usize,
    weight: &'a [f32],
}

/// The last `kernel_size - 1` inputs of one sequence, `[kernel_size - 1,
/// channels]` row-major.
///
/// This is the whole of a short convolution's history. It cannot be trimmed to
/// a shorter window the way a KV cache can, so a rejected speculative token
/// needs the state restored and replayed rather than truncated.
#[derive(Debug, Clone)]
pub struct ConvState {
    channels: usize,
    history: Vec<f32>,
}

impl ConvState {
    /// The state a sequence starts from: `kernel_size - 1` zeroed timesteps,
    /// which is what makes the first output causal.
    ///
    /// Built from the two shapes rather than from a [`ShortConv`], because a
    /// stack of forty-two layers allocates every cache it will need before it
    /// has decoded a single weight.
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        assert!(kernel_size > 0, "a kernel needs at least one tap");
        Self {
            channels,
            history: vec![0.0; (kernel_size - 1) * channels],
        }
    }

    /// The `kernel_size - 1` timesteps preceding the next input, oldest first.
    pub fn history(&self) -> &[f32] {
        &self.history
    }
}

impl<'a> ShortConv<'a> {
    /// `weight` is the checkpoint's own `sconv` tensor: `channels` contiguous
    /// runs of `kernel_size` taps, tap `k` multiplying the input `kernel_size -
    /// 1 - k` timesteps back.
    ///
    /// The two published checkpoints disagree on the tensor's declared shape and
    /// not on its bytes. `thinkingmachines/Inkling-Small` stores `[channels, 1,
    /// kernel]`; `Model.sanitize` transposes it through `(0, 2, 1)` into the
    /// `[channels, kernel, 1]` that `nn.Conv1d` wants, which the mlx-community
    /// quantisations then store directly. That transpose only moves a length-1
    /// axis, so it leaves every element where it was and both layouts flatten to
    /// the same run-of-taps-per-channel this expects.
    pub fn new(channels: usize, weight: &'a [f32]) -> Self {
        assert_eq!(
            weight.len() % channels,
            0,
            "{} taps are not whole kernels of {channels} channels",
            weight.len()
        );
        let kernel_size = weight.len() / channels;
        assert!(kernel_size > 0, "a kernel needs at least one tap");

        Self {
            channels,
            kernel_size,
            weight,
        }
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// The state a sequence starts from, for this convolution's own shape.
    pub fn state(&self) -> ConvState {
        ConvState::new(self.channels, self.kernel_size)
    }

    /// `[rows, channels]` in and out, continuing from `state` and leaving the
    /// last `kernel_size - 1` timesteps of `history ++ x` behind in it.
    ///
    /// A `mask` of one boolean per row zeroes what the convolution reads, and
    /// what the state carries forward, but not the residual: a masked timestep
    /// still passes its own value through. That asymmetry is the reference's,
    /// and it is what lets a padded batch position stay inert without the
    /// padding leaking into the convolution's window.
    pub fn forward(&self, state: &mut ConvState, x: &[f32], mask: Option<&[bool]>) -> Vec<f32> {
        let _timed = profile::scope(Op::Sconv);
        assert_eq!(state.channels, self.channels, "state is for another conv");
        assert_eq!(
            x.len() % self.channels,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.channels
        );
        let rows = x.len() / self.channels;
        if let Some(mask) = mask {
            assert_eq!(mask.len(), rows, "one mask entry per row");
        }

        let padded = self.pad(state, x, mask);
        let mut out = self.convolve(&padded, rows);
        for (out, residual) in out.iter_mut().zip(x) {
            *out += residual;
        }

        let tail = padded.len() - state.history.len();
        state.history.copy_from_slice(&padded[tail..]);
        out
    }

    /// `history ++ masked(x)`, the window every output row is cut from.
    fn pad(&self, state: &ConvState, x: &[f32], mask: Option<&[bool]>) -> Vec<f32> {
        let mut padded = Vec::with_capacity(state.history.len() + x.len());
        padded.extend_from_slice(&state.history);
        for (t, row) in x.chunks_exact(self.channels).enumerate() {
            if mask.is_none_or(|mask| mask[t]) {
                padded.extend_from_slice(row);
            } else {
                padded.extend(std::iter::repeat_n(0.0, self.channels));
            }
        }
        padded
    }

    /// The convolution alone, without the residual. Taps run outermost so each
    /// pass over a row is contiguous in both the window and the output.
    fn convolve(&self, padded: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * self.channels];
        for (t, out) in out.chunks_exact_mut(self.channels).enumerate() {
            for k in 0..self.kernel_size {
                let window = &padded[(t + k) * self.channels..];
                for (c, out) in out.iter_mut().enumerate() {
                    *out += self.weight[c * self.kernel_size + k] * window[c];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, deviation};

    /// Synthetic float32 cases and the trained kernels, from
    /// `just dump-sconv-fixture`.
    const FIXTURE: &str = "sconv.safetensors";

    /// Each conv's kernel in `FIXTURE`, and the activation pair in
    /// `ACTIVATIONS` it maps between.
    const TRAINED: [(&str, &str, &str); 4] = [
        ("k_sconv", "k_proj_out", "k_sconv_out"),
        ("v_sconv", "v_proj_out", "v_sconv_out"),
        ("attn_sconv", "o_proj_out", "attn_sconv_out"),
        ("mlp_sconv", "mlp_out", "mlp_sconv_out"),
    ];

    /// The synthetic cases are float32 end to end, so only summation order
    /// separates this from MLX and 1e-6 is a few tens of ulps — the same bound,
    /// for the same reason, as the RMSNorm and MLP cases. Worst observed when
    /// this landed: 9.0e-8, an order of magnitude in hand.
    const TOLERANCE: f32 = 1e-6;

    /// The trained pairs cannot be held anywhere near that. The model runs in
    /// bfloat16 and `InklingShortConvolution` casts its padded input to the
    /// weight's dtype, so the recorded output was rounded to bfloat16 once after
    /// the convolution and again after the residual add — and the convolution
    /// alone can be several times larger than the sum it feeds, so that first
    /// rounding lands with the larger tensor's quantum.
    ///
    /// Worst observed when this landed: 1.0e-2, on `layer0.v_sconv`, where the
    /// convolution peaks at four times the output. The weakest mutation these
    /// tests rely on catching — dropping the residual on `layer0.mlp_sconv` —
    /// moves the answer by 5.5e-2, so 2e-2 sits about a factor of two from
    /// either side. What this bound settles is the weight layout and the
    /// trained kernel, not the last bits; the synthetic cases settle those.
    const TRAINED_TOLERANCE: f32 = 2e-2;

    /// A `[batch, rows, channels]` fixture tensor and the shape to cut it by.
    struct Synthetic {
        ckpt: Checkpoint,
        batch: usize,
        channels: usize,
        kernel_size: usize,
    }

    impl Synthetic {
        fn load() -> Self {
            let ckpt = fixture::open(FIXTURE);
            let shape = fixture::tensor(&ckpt, "synthetic.input").shape();
            let (batch, channels) = (shape[0], shape[2]);
            let kernel_size = fixture::f32s(&fixture::tensor(&ckpt, "kernel_size"))[0] as usize;
            Self {
                batch,
                channels,
                kernel_size,
                ckpt,
            }
        }

        fn tensor(&self, name: &str) -> Vec<f32> {
            fixture::f32s(&fixture::tensor(&self.ckpt, &format!("synthetic.{name}")))
        }

        /// One sequence out of a `[batch, ..., channels]` tensor.
        fn sequence<'t>(&self, tensor: &'t [f32], b: usize) -> &'t [f32] {
            let stride = tensor.len() / self.batch;
            &tensor[b * stride..(b + 1) * stride]
        }

        fn mask(&self, mask: &[f32], b: usize) -> Vec<bool> {
            self.sequence(mask, b).iter().map(|k| *k != 0.0).collect()
        }
    }

    /// One conv's trained kernel, the activation it consumed and the one
    /// mlx-vlm produced from it.
    struct Trained {
        name: String,
        channels: usize,
        weight: Vec<f32>,
        input: Vec<f32>,
        output: Vec<f32>,
    }

    impl Trained {
        fn load_all() -> Vec<Self> {
            let fixture = fixture::open(FIXTURE);
            let activations = fixture::open(ACTIVATIONS);
            CAPTURED_LAYERS
                .iter()
                .flat_map(|layer| TRAINED.map(|conv| (*layer, conv)))
                .map(|(layer, (conv, input, output))| {
                    let kernel = fixture::layer_tensor(&fixture, layer, &format!("{conv}.weight"));
                    let of = |name: &str| {
                        fixture::f32s(&fixture::layer_tensor(&activations, layer, name))
                    };
                    Self {
                        name: format!("layer{layer}.{conv}"),
                        channels: kernel.shape()[0],
                        weight: fixture::f32s(&kernel),
                        input: of(input),
                        output: of(output),
                    }
                })
                .collect()
        }

        fn conv(&self) -> ShortConv<'_> {
            ShortConv::new(self.channels, &self.weight)
        }

        fn forward(&self, weight: &[f32]) -> Vec<f32> {
            let conv = ShortConv::new(self.channels, weight);
            conv.forward(&mut conv.state(), &self.input, None)
        }
    }

    /// Each channel's taps in reverse, which is the same convolution walked
    /// backwards in time.
    fn reversed(weight: &[f32], kernel_size: usize) -> Vec<f32> {
        weight
            .chunks_exact(kernel_size)
            .flat_map(|taps| taps.iter().rev().copied())
            .collect()
    }

    #[test]
    fn the_synthetic_case_reproduces_mlx() {
        let fx = Synthetic::load();
        let (weight, input, want) = (fx.tensor("weight"), fx.tensor("input"), fx.tensor("whole"));
        let conv = ShortConv::new(fx.channels, &weight);
        assert_eq!(conv.kernel_size(), fx.kernel_size);

        for b in 0..fx.batch {
            let got = conv.forward(&mut conv.state(), fx.sequence(&input, b), None);
            let deviation = deviation(&got, fx.sequence(&want, b));
            assert!(
                deviation <= TOLERANCE,
                "sequence {b}: deviation {deviation:e}"
            );
        }
    }

    /// The property decode and continuous batching rest on: a sequence split
    /// anywhere and carried across the split by the state is the same sequence.
    ///
    /// Exact equality rather than a tolerance, because it has to be. Both paths
    /// multiply the same taps by the same values in the same order; the only
    /// thing a split changes is where the numbers come from. A split that moved
    /// even the last bit would compound over a long generation.
    ///
    /// The fixture's `streamed` is mlx-vlm's own one-timestep-at-a-time answer,
    /// so the split is checked against the reference's cache path and not only
    /// against this port's whole-sequence one.
    #[test]
    fn streaming_a_sequence_matches_feeding_it_whole() {
        let fx = Synthetic::load();
        let (weight, input) = (fx.tensor("weight"), fx.tensor("input"));
        let want = fx.tensor("streamed");
        let conv = ShortConv::new(fx.channels, &weight);
        let rows = input.len() / (fx.batch * fx.channels);

        for b in 0..fx.batch {
            let sequence = fx.sequence(&input, b);
            let whole = conv.forward(&mut conv.state(), sequence, None);

            // Chunks that straddle the kernel in both directions: shorter than
            // the state, exactly one timestep, and longer than the state.
            for chunks in [vec![1; rows], vec![2, 1, rows - 3], vec![rows - 1, 1]] {
                let mut state = conv.state();
                let mut streamed = Vec::new();
                let mut at = 0;
                for chunk in &chunks {
                    let end = at + chunk * fx.channels;
                    streamed.extend(conv.forward(&mut state, &sequence[at..end], None));
                    at = end;
                }
                assert_eq!(streamed, whole, "sequence {b} split {chunks:?}");
            }

            let deviation = deviation(&whole, fx.sequence(&want, b));
            assert!(
                deviation <= TOLERANCE,
                "sequence {b}: deviation {deviation:e}"
            );
        }
    }

    /// A cache holds the last `kernel_size - 1` inputs and nothing else, which
    /// is what makes the state a fixed cost per sequence.
    #[test]
    fn the_state_left_behind_is_the_last_timesteps_of_the_input() {
        let fx = Synthetic::load();
        let (weight, input) = (fx.tensor("weight"), fx.tensor("input"));
        let want = fx.tensor("streamed_state");
        let conv = ShortConv::new(fx.channels, &weight);

        for b in 0..fx.batch {
            let sequence = fx.sequence(&input, b);
            let mut state = conv.state();
            conv.forward(&mut state, sequence, None);

            let kept = (fx.kernel_size - 1) * fx.channels;
            assert_eq!(state.history(), &sequence[sequence.len() - kept..]);
            assert_eq!(state.history(), fx.sequence(&want, b));
        }
    }

    /// A chunk shorter than the state cannot fill it, so the reference keeps the
    /// last `kernel_size - 1` of the *padded* sequence — part of what was
    /// already there. Decoding one token at a time is entirely this case.
    #[test]
    fn a_short_chunk_keeps_the_tail_of_the_previous_state() {
        let fx = Synthetic::load();
        let (weight, input) = (fx.tensor("weight"), fx.tensor("short_input"));
        let (before, want) = (fx.tensor("primed_state"), fx.tensor("primed_output"));
        let after = fx.tensor("primed_final_state");
        let conv = ShortConv::new(fx.channels, &weight);
        let rows = input.len() / (fx.batch * fx.channels);
        assert!(
            rows < fx.kernel_size - 1,
            "{rows} rows would fill the state"
        );

        for b in 0..fx.batch {
            let mut state = conv.state();
            state.history.copy_from_slice(fx.sequence(&before, b));

            let got = conv.forward(&mut state, fx.sequence(&input, b), None);
            let deviation = deviation(&got, fx.sequence(&want, b));
            assert!(
                deviation <= TOLERANCE,
                "sequence {b}: deviation {deviation:e}"
            );
            assert_eq!(state.history(), fx.sequence(&after, b));
        }
    }

    /// An empty state is the zero-left-padding the reference's no-cache path
    /// applies, so a prefill that allocates a cache and one that does not are
    /// the same prefill — which is why this port needs only one of them.
    ///
    /// Reconstructed by prepending the zero rows to the input instead of to the
    /// state: the padded run's trailing outputs are cut from the same windows,
    /// so they have to be the same numbers.
    #[test]
    fn an_empty_state_is_the_zero_left_padding() {
        let fx = Synthetic::load();
        let (weight, input) = (fx.tensor("weight"), fx.tensor("input"));
        let conv = ShortConv::new(fx.channels, &weight);
        let lead = (fx.kernel_size - 1) * fx.channels;

        for b in 0..fx.batch {
            let sequence = fx.sequence(&input, b);
            let mut padded = vec![0.0; lead];
            padded.extend_from_slice(sequence);

            let from_padding = conv.forward(&mut conv.state(), &padded, None);
            assert_eq!(
                conv.forward(&mut conv.state(), sequence, None),
                from_padding[lead..],
                "sequence {b}"
            );
        }
    }

    #[test]
    fn the_mask_zeroes_what_the_convolution_reads() {
        let fx = Synthetic::load();
        let (weight, input) = (fx.tensor("weight"), fx.tensor("input"));
        let (mask, want) = (fx.tensor("mask"), fx.tensor("masked_output"));
        let state_want = fx.tensor("masked_state");
        let conv = ShortConv::new(fx.channels, &weight);

        for b in 0..fx.batch {
            let mut state = conv.state();
            let mask = fx.mask(&mask, b);
            let got = conv.forward(&mut state, fx.sequence(&input, b), Some(&mask));

            let deviation = deviation(&got, fx.sequence(&want, b));
            assert!(
                deviation <= TOLERANCE,
                "sequence {b}: deviation {deviation:e}"
            );
            assert_eq!(state.history(), fx.sequence(&state_want, b), "sequence {b}");
        }
    }

    /// The residual is added from the *unmasked* input. A fully masked sequence
    /// therefore comes back unchanged: the convolution reads nothing but every
    /// timestep still passes its own value through.
    ///
    /// This is the test that fails if the mask is applied to the residual too —
    /// the answer would be zero — and if it is not applied at all.
    #[test]
    fn a_masked_timestep_still_carries_its_own_residual() {
        let fx = Synthetic::load();
        let (weight, input) = (fx.tensor("weight"), fx.tensor("input"));
        let mask = fx.tensor("mask");
        let conv = ShortConv::new(fx.channels, &weight);

        let dropped = fx
            .mask(&mask, 1)
            .iter()
            .all(|keep| !keep)
            .then(|| fx.sequence(&input, 1))
            .expect("the fixture masks out the whole of sequence 1");

        let got = conv.forward(
            &mut conv.state(),
            dropped,
            Some(&vec![false; dropped.len() / fx.channels]),
        );
        assert_eq!(got, dropped);
        assert_eq!(
            deviation(&got, fx.sequence(&fx.tensor("masked_output"), 1)),
            0.0
        );
    }

    #[test]
    fn the_trained_kernels_reproduce_the_reference_activations() {
        let mut worst = 0.0f32;
        for case in Trained::load_all() {
            let deviation = deviation(&case.forward(&case.weight), &case.output);
            assert!(
                deviation <= TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
            worst = worst.max(deviation);
        }
        assert!(
            worst > 0.0,
            "a run that matched exactly would mean bfloat16 rounding vanished"
        );
    }

    /// Dropping the residual leaves a convolution that is still smooth, still
    /// causal and still plausible. Only the numbers say otherwise.
    #[test]
    fn dropping_the_residual_changes_the_answer() {
        for case in Trained::load_all() {
            let convolution_only: Vec<f32> = case
                .forward(&case.weight)
                .iter()
                .zip(&case.input)
                .map(|(out, residual)| out - residual)
                .collect();
            let deviation = deviation(&convolution_only, &case.output);
            assert!(
                deviation > TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
        }
    }

    /// Reading the kernel backwards keeps the convolution causal and keeps
    /// every tap, so it produces numbers of the right magnitude and the wrong
    /// position. Tap `kernel_size - 1` is the one that meets the current
    /// timestep.
    #[test]
    fn reversing_the_kernel_changes_the_answer() {
        let fx = Synthetic::load();
        let (weight, input) = (fx.tensor("weight"), fx.tensor("input"));
        let want = fx.tensor("whole");
        let backwards = reversed(&weight, fx.kernel_size);
        let conv = ShortConv::new(fx.channels, &backwards);

        for b in 0..fx.batch {
            let got = conv.forward(&mut conv.state(), fx.sequence(&input, b), None);
            let deviation = deviation(&got, fx.sequence(&want, b));
            assert!(
                deviation > TOLERANCE,
                "sequence {b}: deviation {deviation:e}"
            );
        }

        for case in Trained::load_all() {
            let kernel_size = case.conv().kernel_size();
            let deviation = deviation(
                &case.forward(&reversed(&case.weight, kernel_size)),
                &case.output,
            );
            assert!(
                deviation > TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
        }
    }
}
