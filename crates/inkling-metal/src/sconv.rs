//! The depthwise causal short convolution, on the device.
//!
//! [`inkling_core::sconv`] is the authority on what this computes and every fact
//! it states is one this kernel has to hold: the convolution is a
//! cross-correlation, so tap `K-1` is the one that meets the current timestep;
//! there is a residual add, and the residual is the input rather than anything
//! the convolution produced; and the window a call leaves behind is the last
//! `K-1` timesteps of `history ++ x`, which for a call shorter than the window is
//! partly what was already there.
//!
//! # The streaming property is the invariant, and it is why this is one thread an
//! element
//!
//! Feeding one timestep at a time through the window has to equal feeding the
//! whole sequence, bit for bit — the CPU path holds that across three chunkings
//! and a generation compounds any drift. What buys it here is that a thread
//! computes one output element by walking the taps in one order, from a window
//! whose values are the same floats whichever call put them there. There is no
//! reduction across threads to associate differently and no accumulation across
//! calls, so a split cannot move a bit.
//!
//! # Two windows, because a dispatch cannot read one and write it
//!
//! The window a call leaves behind overlaps the window it reads: at `K = 4` a
//! decode step's new window is two timesteps of the old one and the token just
//! seen. Threads of one dispatch are not ordered against each other, so a kernel
//! that wrote the new window over the old would be racing whichever threads had
//! not read it yet. Each convolution therefore holds two and alternates, which
//! costs `2 * (K-1) * channels` floats a layer — 24 KB across the stack for both
//! convolutions — and no synchronisation at all.
//!
//! # It runs in float32, where the reference runs in bfloat16
//!
//! `InklingShortConvolution` casts its padded input to the weight's dtype, so
//! mlx-vlm rounds once after the convolution and again after the residual add.
//! The CPU path here models neither and is what every kernel in this tree is
//! checked against — see [`crate::norm`], which makes the same choice for the
//! same reason. A kernel that reproduced the reference's rounding would be the
//! one operation in the model whose backend changed the answer.

use std::cell::{Cell, RefCell};

use inkling_core::profile::{self, Op};

use crate::buffer::{Buffer, Landing};
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "short_conv";

/// Threads one threadgroup of a dispatch holds.
///
/// A thread here is one channel of one timestep and reads `K` values that are
/// `channels` floats apart, so consecutive threads read consecutive floats and
/// the width is the ordinary elementwise one. There is nothing to reduce and no
/// barrier to pay for, which is what separates this from [`crate::norm`]'s
/// threadgroup-per-row.
const THREADS_PER_GROUP: usize = 256;

/// The compiled kernel, which every short convolution on a device shares.
///
/// Per source string rather than per weight, like [`crate::RmsNorm`]: the source
/// names no shape, so one of these serves both convolutions of all forty-two
/// layers.
#[derive(Debug)]
pub struct ShortConvolution {
    kernel: Kernel,
}

impl ShortConvolution {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, BODY)
    }

    /// [`ShortConvolution::new`] out of a source string of the caller's own,
    /// which is how a test puts a deliberately wrong kernel through the same
    /// plumbing as the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
        })
    }
}

/// One short convolution's kernel on the device, and the window it carries
/// between calls.
///
/// The weight is `[channels, taps]` — one contiguous run of taps per channel,
/// which is what both published checkpoints flatten to — and is copied once at
/// wrap time for the reason [`crate::LayerNorm`]'s weight is: it is bfloat16 in
/// the checkpoint and the kernel wants float32, so there is nothing here to hand
/// over in place.
#[derive(Debug)]
pub struct LayerConv<'a> {
    device: &'a Device,
    conv: &'a ShortConvolution,
    /// Behind a cell for the reason [`crate::LayerNorm`]'s weight is: binding a
    /// buffer to a dispatch borrows it exclusively, and the weight belongs to
    /// the layer rather than to the call.
    weight: RefCell<Buffer<f32>>,
    /// The two windows a call reads one of and writes the other — see the
    /// module documentation.
    windows: RefCell<[Buffer<f32>; 2]>,
    /// Which of the two the next call reads.
    reading: Cell<usize>,
    channels: usize,
    taps: usize,
}

impl<'a> LayerConv<'a> {
    /// `weight` is the checkpoint's own `sconv` tensor over `channels` channels:
    /// `channels` contiguous runs of `taps`, tap `k` multiplying the input
    /// `taps - 1 - k` timesteps back.
    pub fn new(
        device: &'a Device,
        conv: &'a ShortConvolution,
        channels: usize,
        weight: &[f32],
    ) -> Result<Self, MetalError> {
        assert!(channels > 0, "a convolution has channels");
        assert_eq!(
            weight.len() % channels,
            0,
            "{} taps are not whole kernels of {channels} channels",
            weight.len()
        );
        let taps = weight.len() / channels;
        assert!(
            taps > 1,
            "a window of {} timesteps carries nothing",
            taps - 1
        );

        let window = || device.zeroed::<f32>((taps - 1) * channels);
        Ok(Self {
            weight: RefCell::new(device.buffer(weight)?),
            windows: RefCell::new([window()?, window()?]),
            reading: Cell::new(0),
            device,
            conv,
            channels,
            taps,
        })
    }

    pub fn taps(&self) -> usize {
        self.taps
    }

    /// The window a sequence starts from, which is `taps - 1` zeroed timesteps —
    /// and is what makes the first output causal.
    ///
    /// Only the window the next call will read is cleared. The other is written
    /// before it is read, so what a previous sequence left in it is memory
    /// nobody indexes.
    pub fn restart(&self) {
        self.windows.borrow_mut()[self.reading.get()]
            .as_mut_slice()
            .fill(0.0);
    }

    /// The `taps - 1` timesteps preceding the next input, oldest first — the
    /// window as [`ConvState::history`](inkling_core::ConvState::history) hands
    /// it out.
    pub fn window(&self) -> Vec<f32> {
        self.windows.borrow()[self.reading.get()].to_vec()
    }

    /// `[rows, channels]` in and out, submitted on its own.
    ///
    /// What a caller with nothing to batch it against wants, and what the cases
    /// here drive. A layer reaches for [`LayerConv::encode`], because what
    /// produced this call's input and what consumes its output are dispatches
    /// that could have been in the same command buffer.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, MetalError> {
        let mut batch = self.device.batch()?;
        let mut input = self.device.buffer(x)?;
        let out = self.encode(&mut batch, &mut input)?;
        batch.wait()?;
        Ok(profile::timed(Op::Readback, || out.to_vec()))
    }

    /// The same convolution over rows a dispatch already left on the device,
    /// encoded into `batch` and leaving its own rows there in turn.
    ///
    /// **The window advances here rather than when the batch completes.** Which
    /// of the two windows a call reads is decided as it is encoded, and the
    /// dispatch that writes the other is in the command buffer by the time this
    /// returns — so the next call reads what this one wrote whether or not
    /// anyone has waited in between. What a caller must not do is encode two
    /// calls of one sequence into one command buffer expecting the second to be
    /// a second timestep; the dispatches are ordered, but a sequence's own
    /// convolution is asked for once a call.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Buffer<f32>, MetalError> {
        let rows = self.rows(x.len());
        let mut out = self.device.zeroed::<f32>(x.len())?;
        self.encode_over(
            batch,
            x,
            Landing {
                out: &mut out,
                groups: 1,
                stride: rows,
                base: 0,
            },
        )?;
        Ok(out)
    }

    /// The same convolution with its rows scattered into `landing` rather than
    /// left in a buffer of their own.
    ///
    /// **This is where the value's convolution ends.** Nothing between it and
    /// the attention step touches what it produced — the value is convolved and
    /// never normed — so its rows are keys of the span the layer is keeping, and
    /// the split into heads and the append are the indexing of the write. The
    /// key's convolution has a head norm behind it and lands in a buffer of its
    /// own.
    pub fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        landing: Landing<'_>,
    ) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Encode);
        let rows = self.rows(x.len());
        assert!(landing.groups > 0, "a row has groups");
        assert_eq!(
            self.channels % landing.groups,
            0,
            "{} channels are not {} groups",
            self.channels,
            landing.groups
        );
        landing.fits(rows, self.channels / landing.groups);

        let fields = [
            extent(rows, "the rows of a call"),
            extent(self.channels, "the channels of a convolution"),
            extent(self.taps, "the taps of a kernel"),
            extent(landing.groups, "the groups of a row"),
            extent(landing.stride, "the rows a group has room for"),
            extent(landing.base, "where a call's rows start"),
        ];
        let mut shape = self.device.inline(&fields)?;
        let mut weight = self.weight.borrow_mut();
        let mut windows = self.windows.borrow_mut();
        let [first, second] = &mut *windows;
        let (window, kept) = match self.reading.get() {
            0 => (first, second),
            _ => (second, first),
        };

        // A thread to each channel of each timestep, and one more timestep's
        // worth for the window left behind — which reads the same padded
        // sequence the outputs are cut from and writes somewhere no output
        // thread touches.
        let threads = (rows + self.taps - 1) * self.channels;
        batch.add(
            &self.conv.kernel,
            &[
                shape.arg(),
                x.arg(),
                weight.arg(),
                window.arg(),
                landing.out.arg(),
                kept.arg(),
            ],
            Grid::new(threads, THREADS_PER_GROUP),
        )?;
        self.reading.set(1 - self.reading.get());
        Ok(())
    }

    /// How many rows of this convolution's width `values` is.
    fn rows(&self, values: usize) -> usize {
        assert_eq!(
            values % self.channels,
            0,
            "{values} values are not whole rows of {}",
            self.channels
        );
        values / self.channels
    }
}

/// The kernel. No constant of this crate's decides anything here — the taps, the
/// channels and the rows are all a call's — so the source is the whole of it.
const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint rows;
    uint channels;
    uint taps;
    uint groups;
    uint stride;
    uint base;
};

/// One channel of one timestep of `window ++ x`, which is the padded sequence
/// every output row is cut from and the window left behind is the tail of.
inline float padded(
    device const float *x,
    device const float *window,
    constant Shape &shape,
    uint at,
    uint c
) {
    const uint lead = shape.taps - 1;
    if (at < lead) {
        return window[at * shape.channels + c];
    }
    return x[(at - lead) * shape.channels + c];
}

/// A depthwise causal convolution with a residual add, one thread to a channel
/// of a timestep, and the window the next call reads left behind.
///
/// **It is a cross-correlation.** Tap `k` multiplies the input `taps - 1 - k`
/// timesteps back, so the *last* tap is the one that meets the current timestep
/// and the loop below walks the window forwards. Reading the kernel the textbook
/// way round keeps the convolution causal and keeps every tap, so it produces
/// numbers of the right magnitude at the wrong positions — fluent text and a
/// wrong model.
///
/// **The residual is the input**, not the convolution's own output and not
/// anything scaled: `out = conv(x) + x`. Dropping it leaves a convolution that is
/// still smooth, still causal and still plausible.
///
/// The taps are walked from zero and accumulated in that order, which is the
/// order `inkling_core::sconv` accumulates them in — and which is what makes a
/// sequence split anywhere the same sequence, since the only thing a split
/// changes is which call put a value in the window.
kernel void short_conv(
    constant Shape &shape [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *window [[buffer(3)]],
    device float *out [[buffer(4)]],
    device float *kept [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    const uint lead = shape.taps - 1;
    if (id >= (shape.rows + lead) * shape.channels) {
        return;
    }
    const uint t = id / shape.channels;
    const uint c = id % shape.channels;

    // The last `taps - 1` timesteps of the padded sequence, which is what the
    // next call reads. A call shorter than the window cannot fill it, so part of
    // what it keeps is part of what it was given — the reference takes the tail
    // of the padding either way, and decoding one token at a time is entirely
    // that case.
    if (t >= shape.rows) {
        kept[(t - shape.rows) * shape.channels + c] = padded(x, window, shape, t, c);
        return;
    }

    device const float *taps = weight + (ulong)c * shape.taps;
    float acc = 0.0f;
    for (uint k = 0; k < shape.taps; ++k) {
        acc += taps[k] * padded(x, window, shape, t + k, c);
    }

    // Where the row lands, which for the value's convolution is the span the
    // layer keeps — see `Landing`. With one group and a stride of `rows` this is
    // `out[t * channels + c]`, the row where it was computed.
    const uint width = shape.channels / shape.groups;
    const uint group = c / width;
    device float *result =
        out + ((ulong)group * shape.stride + shape.base + t) * width + (c % width);
    *result = acc + x[t * shape.channels + c];
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::ShortConv;
    use inkling_core::fixture::{self, deviation};

    use crate::testing::device;

    /// The synthetic float32 cases and the trained kernels
    /// [`inkling_core::sconv`] is pinned to, from `just dump-sconv-fixture`.
    const FIXTURE: &str = "sconv.safetensors";

    /// How far a dispatch may land from the CPU's answer.
    ///
    /// Both sides multiply the same taps by the same values and add them in the
    /// same order, so there is no summation order left to differ about — what is
    /// left is that Metal compiles `acc += w * v` with fast math on and may
    /// contract it to an FMA, which rounds once where the CPU rounds twice. That
    /// is a bound of an ulp or so per tap, and it is a bound on the *oracle*:
    /// the contracted form is the more accurate of the two.
    ///
    /// Worst observed when this landed: 9.0e-8, which is under two f32 ulps of
    /// the tensor's peak. The weakest mutation these tests rely on catching is
    /// the kernel read backwards, at 4.7e-1 — seven decades above.
    const TOLERANCE: f32 = 1e-6;

    /// A `[batch, rows, channels]` fixture tensor and the shape to cut it by —
    /// the same cases `inkling_core::sconv::tests` drives, so what this says is
    /// that both backends answer the same questions.
    struct Synthetic {
        batch: usize,
        channels: usize,
        kernel_size: usize,
        weight: Vec<f32>,
        input: Vec<f32>,
    }

    impl Synthetic {
        fn load() -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of = |name: &str| fixture::f32s(&fixture::tensor(&ckpt, name));
            let shape = fixture::tensor(&ckpt, "synthetic.input").shape();
            Self {
                batch: shape[0],
                channels: shape[2],
                kernel_size: of("kernel_size")[0] as usize,
                weight: of("synthetic.weight"),
                input: of("synthetic.input"),
            }
        }

        fn tensor(&self, name: &str) -> Vec<f32> {
            fixture::f32s(&fixture::tensor(
                &fixture::open(FIXTURE),
                &format!("synthetic.{name}"),
            ))
        }

        /// One sequence out of a `[batch, ..., channels]` tensor.
        fn sequence<'t>(&self, tensor: &'t [f32], b: usize) -> &'t [f32] {
            let stride = tensor.len() / self.batch;
            &tensor[b * stride..(b + 1) * stride]
        }

        fn rows(&self) -> usize {
            self.input.len() / (self.batch * self.channels)
        }

        fn wrapped<'d>(
            &self,
            device: &'d Device,
            conv: &'d ShortConvolution,
            weight: &[f32],
        ) -> LayerConv<'d> {
            LayerConv::new(device, conv, self.channels, weight).expect("the kernel uploads")
        }

        fn on_the_cpu<'w>(&self, weight: &'w [f32]) -> ShortConv<'w> {
            ShortConv::new(self.channels, weight)
        }
    }

    /// Every synthetic case dispatched, against `inkling_core`'s own answer for
    /// the same case.
    ///
    /// The cases are that module's rather than this one's, and deliberately:
    /// they are the ones the CPU path is pinned to mlx-vlm by, so what this says
    /// is that both backends answer the same questions and not that each answers
    /// its own.
    #[test]
    fn the_kernel_reproduces_the_cpu_for_every_synthetic_case() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let cpu = fx.on_the_cpu(&fx.weight);
        assert_eq!(layer.taps(), fx.kernel_size);
        let mut worst = 0.0f32;

        for b in 0..fx.batch {
            let sequence = fx.sequence(&fx.input, b);
            layer.restart();
            let got = layer.forward(sequence).expect("the dispatch completes");
            let want = cpu.forward(&mut cpu.state(), sequence, None);

            assert_eq!(got.len(), want.len());
            let deviation = deviation(&got, &want);
            assert!(deviation <= TOLERANCE, "sequence {b}: {deviation:e}");
            worst = worst.max(deviation);
        }
        eprintln!(
            "worst deviation from the CPU over {} cases: {worst:e}",
            fx.batch
        );
    }

    /// **The property decode and continuous batching rest on**, on the device:
    /// a sequence split anywhere and carried across the split by the window is
    /// the same sequence.
    ///
    /// Exact equality rather than a tolerance, for the reason
    /// `inkling_core::sconv`'s own split test demands it: both paths multiply
    /// the same taps by the same values in the same order, and the only thing a
    /// split changes is which call put a value in the window. A split that moved
    /// even the last bit would compound over a long generation.
    ///
    /// The three chunkings straddle the window in both directions — shorter than
    /// it, exactly one timestep, and longer — which are the same three that
    /// module drives.
    #[test]
    fn streaming_a_sequence_matches_feeding_it_whole_on_the_device() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let rows = fx.rows();

        for b in 0..fx.batch {
            let sequence = fx.sequence(&fx.input, b);
            layer.restart();
            let whole = layer.forward(sequence).expect("the dispatch completes");

            for chunks in [vec![1; rows], vec![2, 1, rows - 3], vec![rows - 1, 1]] {
                layer.restart();
                let mut streamed = Vec::new();
                let mut at = 0;
                for chunk in &chunks {
                    let end = at + chunk * fx.channels;
                    streamed.extend(
                        layer
                            .forward(&sequence[at..end])
                            .expect("the dispatch completes"),
                    );
                    at = end;
                }
                assert_eq!(streamed, whole, "sequence {b} split {chunks:?}");
            }
        }
    }

    /// The window a call leaves behind is the last `taps - 1` timesteps of what
    /// it was given, which is what makes the state a fixed cost per sequence.
    ///
    /// And the same window after a chunk *shorter* than it, which cannot fill it
    /// — the reference keeps the tail of the padded sequence, so part of what is
    /// kept is part of what was already there. Decoding one token at a time is
    /// entirely that case, which is why it is the one asserted against the
    /// reference's own recorded state.
    #[test]
    fn the_window_left_behind_is_the_tail_of_the_padded_sequence() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let kept = (fx.kernel_size - 1) * fx.channels;
        let want = fx.tensor("streamed_state");

        for b in 0..fx.batch {
            let sequence = fx.sequence(&fx.input, b);
            layer.restart();
            layer.forward(sequence).expect("the dispatch completes");

            assert_eq!(layer.window(), sequence[sequence.len() - kept..]);
            assert_eq!(layer.window(), fx.sequence(&want, b));
        }

        // A chunk of one timestep out of an empty window, which fills a third of
        // it: what is kept is two zeroed rows and the row just seen.
        layer.restart();
        let one = &fx.sequence(&fx.input, 0)[..fx.channels];
        layer.forward(one).expect("the dispatch completes");
        let mut carried = vec![0.0; kept - fx.channels];
        carried.extend_from_slice(one);
        assert_eq!(layer.window(), carried);
    }

    /// A sequence that has seen nothing starts from a zeroed window, which is
    /// the zero left-padding the reference's no-cache path applies — so the
    /// second sequence through one layer is not the first's continuation.
    #[test]
    fn restarting_a_convolution_is_the_zero_left_padding() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let sequence = fx.sequence(&fx.input, 0);

        layer.restart();
        assert_eq!(
            layer.window(),
            vec![0.0; (fx.kernel_size - 1) * fx.channels]
        );
        let first = layer.forward(sequence).expect("the dispatch completes");

        let carried = layer.forward(sequence).expect("the dispatch completes");
        assert_ne!(carried, first, "a window that carried nothing forward");

        layer.restart();
        assert_eq!(
            layer.forward(sequence).expect("the dispatch completes"),
            first
        );
    }

    /// **Tap `taps - 1` is the one that meets the current timestep.** Reading
    /// each channel's taps in reverse is the same convolution walked backwards
    /// in time: still causal, every tap kept, numbers of the right magnitude at
    /// the wrong positions.
    ///
    /// The mutation is the weight rather than the kernel, because what it has to
    /// name is a reading of the checkpoint's own bytes.
    #[test]
    fn reversing_the_kernel_changes_the_answer() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let backwards: Vec<f32> = fx
            .weight
            .chunks_exact(fx.kernel_size)
            .flat_map(|taps| taps.iter().rev().copied())
            .collect();
        assert_ne!(backwards, fx.weight, "a palindromic kernel proves nothing");

        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let mutant = fx.wrapped(&device, &conv, &backwards);
        let sequence = fx.sequence(&fx.input, 0);
        layer.restart();
        mutant.restart();

        let deviation = deviation(
            &mutant.forward(sequence).expect("the dispatch completes"),
            &layer.forward(sequence).expect("the dispatch completes"),
        );
        eprintln!("the kernel read backwards: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// **The residual is added, and it is the input.** A convolution without it
    /// is still smooth, still causal and still plausible; only the numbers say
    /// otherwise.
    #[test]
    fn dropping_the_residual_changes_the_answer() {
        let Some(device) = device() else { return };
        let fx = Synthetic::load();
        let sequence = fx.sequence(&fx.input, 0);

        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        layer.restart();
        let want = layer.forward(sequence).expect("the dispatch completes");

        let without = BODY.replace(
            "*result = acc + x[t * shape.channels + c];",
            "*result = acc;",
        );
        assert_ne!(without, BODY, "the mutation changed nothing");
        let mutant = ShortConvolution::from_source(&device, &without).expect("the mutant compiles");
        let dropped = fx.wrapped(&device, &mutant, &fx.weight);
        dropped.restart();

        let deviation = deviation(
            &dropped.forward(sequence).expect("the dispatch completes"),
            &want,
        );
        eprintln!("the residual dropped: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// **Where the value's convolution ends.** Its rows go straight into the
    /// span the layer keeps — split into heads and placed past the keys already
    /// there — because nothing between it and the attention step touches them.
    ///
    /// Checked against the same convolution left where it was computed and
    /// scattered here, which is the copy the landing replaces. Exact rather than
    /// bounded: the arithmetic is the same dispatch either way and the only
    /// thing that differs is the index it writes to.
    ///
    /// Two calls at different offsets, and the slots after them checked to be
    /// untouched, because a landing that ignored its base would agree on the
    /// first call and overwrite it on the second.
    #[test]
    fn a_landing_places_a_convolutions_rows_where_the_step_reads_them() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");

        // The attention convolutions' shape rather than the fixture's, whose
        // channel count does not divide into heads: `kv_heads` groups of
        // `head_dim`, with the span given room for more keys than these fill.
        let (groups, width, taps, stride) = (2, 5, 4, 8);
        let channels = groups * width;
        let of = |len, salt: usize| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 23 + salt) % 37) as f32 / 8.0 - 2.0)
                .collect()
        };
        let (weight, sequence) = (of(channels * taps, 1), of(6 * channels, 2));
        let chunks = [1, 5];

        let layer = LayerConv::new(&device, &conv, channels, &weight).expect("the kernel uploads");
        let mut span = device
            .zeroed::<f32>(groups * stride * width)
            .expect("the span allocates");
        let mut at = 0;
        layer.restart();
        for rows in chunks {
            let call = &sequence[at * channels..][..rows * channels];
            let mut input = device.buffer(call).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            layer
                .encode_over(
                    &mut batch,
                    &mut input,
                    Landing {
                        out: &mut span,
                        groups,
                        stride,
                        base: at,
                    },
                )
                .expect("the convolution encodes");
            batch.wait().expect("the batch completes");
            at += rows;
        }

        // The same sequence in the same two chunks, left flat and scattered
        // here — which is what the landing is instead of.
        layer.restart();
        let mut flat = Vec::new();
        at = 0;
        for rows in chunks {
            let call = &sequence[at * channels..][..rows * channels];
            flat.extend(layer.forward(call).expect("the dispatch completes"));
            at += rows;
        }
        let mut want = vec![0.0; span.len()];
        for (t, row) in flat.chunks_exact(channels).enumerate() {
            for group in 0..groups {
                want[(group * stride + t) * width..][..width]
                    .copy_from_slice(&row[group * width..][..width]);
            }
        }
        assert_eq!(span.to_vec(), want);
        assert!(at < stride, "the span had no room left over to check");
    }

    /// A kernel of one tap carries no window and would leave the two buffers
    /// this alternates between empty, which the device refuses to allocate — so
    /// it is refused here, where the shape is known, rather than there.
    #[test]
    #[should_panic(expected = "a window of 0 timesteps carries nothing")]
    fn a_kernel_with_no_window_is_refused() {
        let Some(device) = device() else {
            panic!("a window of 0 timesteps carries nothing")
        };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        LayerConv::new(&device, &conv, 4, &[1.0, 2.0, 3.0, 4.0]).ok();
    }
}
