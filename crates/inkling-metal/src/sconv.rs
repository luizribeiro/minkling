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
use inkling_core::sconv::{ConvMark, Held};

use crate::buffer::{Buffer, Landing};
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "short_conv";

/// The entry that runs two of those calls as one dispatch — see [`encode_pair`].
const PAIRED_ENTRY: &str = "short_conv_pair";

/// How many `uint`s the kernel's `Shape` struct declares.
const FIELDS: usize = 9;

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
    /// The same source's paired entry, compiled beside it because a model
    /// wanting one wants the other: a layer's key and value convolutions pair
    /// and the two on its residual path are a block apart.
    paired: Kernel,
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
            paired: device.compile(source, PAIRED_ENTRY)?,
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
    /// What each of them holds, which is the `taps - 1` the convolution reads
    /// and the timesteps behind them a rejected speculative token is taken back
    /// out of. The arithmetic is
    /// [`ConvState`](inkling_core::ConvState)'s and so is the argument for it;
    /// what differs here is only that the rows are on a device.
    held: Cell<Held>,
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
        Self::with_slack(device, conv, channels, weight, 0)
    }

    /// The same, holding `slack` timesteps further back than the convolution
    /// reads so that a speculative round can be rewound rather than replayed.
    ///
    /// What it costs a step is what the kernel writes: a window is written once
    /// per call and read once, so a slack of eight takes the two windows of a
    /// `[1, 4096]` convolution from 98 KB of traffic to 360 KB — against the
    /// 5.9 GB a decode step reads, and against a replay that is a whole forward
    /// pass. A layer whose sequence never speculates asks for none.
    pub fn with_slack(
        device: &'a Device,
        conv: &'a ShortConvolution,
        channels: usize,
        weight: &[f32],
        slack: usize,
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

        let held = Held::new(channels, taps, slack);
        let window = || device.zeroed::<f32>(held.floats());
        Ok(Self {
            weight: RefCell::new(device.buffer(weight)?),
            windows: RefCell::new([window()?, window()?]),
            reading: Cell::new(0),
            held: Cell::new(held),
            device,
            conv,
            channels,
            taps,
        })
    }

    /// How many timesteps may still be taken back.
    pub fn rewindable(&self) -> usize {
        self.held.get().rewindable()
    }

    /// Take back the last `rows` timesteps of the window the next call will
    /// read, leaving the window this convolution would have had without them.
    ///
    /// **The rows have to be there to be taken back**, which on a device means
    /// the command buffer that wrote them has completed. Every caller of this
    /// has read something back from the pass it is rewinding — a speculative
    /// round decides what to take back by reading the logits of what it fed —
    /// so the wait has already happened where this is reached.
    ///
    /// The same shift [`ConvState::rewind`](inkling_core::ConvState::rewind)
    /// makes, on the buffer the device holds: unified memory is what lets a
    /// window be moved along without a dispatch or a copy across a bus.
    pub fn rewind(&self, rows: usize) {
        let mut held = self.held.get();
        let mut windows = self.windows.borrow_mut();
        held.rewind(rows, windows[self.reading.get()].as_mut_slice());
        self.held.set(held);
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
        let mut held = self.held.get();
        held.restarted();
        self.held.set(held);
    }

    /// The `taps - 1` timesteps preceding the next input, oldest first — the
    /// window as [`ConvState::history`](inkling_core::ConvState::history) hands
    /// it out.
    pub fn window(&self) -> Vec<f32> {
        self.windows.borrow()[self.reading.get()].as_slice()[self.held.get().reading()..].to_vec()
    }

    /// The rows this holds now, as something that can put them back later — the
    /// device's half of [`ConvState::mark`](inkling_core::ConvState::mark).
    ///
    /// **Between runs, and for the reason [`LayerConv::rewind`] says it**: what
    /// this reads is a window a dispatch wrote, so the command buffer that wrote
    /// it has to have completed.
    pub fn mark(&self) -> ConvMark {
        let held = self.held.get();
        ConvMark::new(
            held,
            self.windows.borrow()[self.reading.get()]
                .as_slice()
                .to_vec(),
        )
    }

    /// The window this had when `mark` was taken, whatever has gone through it
    /// since.
    ///
    /// The window the *next* call reads and not the other, which is the one
    /// [`LayerConv::restart`] clears and [`LayerConv::rewind`] shifts: the other
    /// is written before it is read, so what is in it is memory nobody indexes.
    pub fn resume(&self, mark: &ConvMark) {
        let mut windows = self.windows.borrow_mut();
        let window = windows[self.reading.get()].as_mut_slice();
        assert_eq!(
            (self.channels, window.len()),
            (mark.held().channels(), mark.timesteps().len()),
            "a mark of another convolution's width"
        );
        window.copy_from_slice(mark.timesteps());
        self.held.set(mark.held());
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
        let out = self.encode(&mut batch, &mut input, None, 1.0)?;
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
        carried: Option<&mut Buffer<f32>>,
        scale: f32,
    ) -> Result<Buffer<f32>, MetalError> {
        let rows = self.rows(x.len());
        let mut out = self.device.zeroed::<f32>(x.len())?;
        self.encode_over(
            batch,
            x,
            carried,
            scale,
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
    ///
    /// `carried` is the layer's *own* residual — the value before the norm the
    /// block this convolution ends began with — added to every row on the way
    /// out. It is a second addend and not a second convolution: `out = conv(x) +
    /// x + carried`, where `+ x` is the convolution's own residual and belongs
    /// to it wherever it runs. The two inside attention have no block around
    /// them and pass `None`; the two on a layer's residual path are the whole
    /// reason this argument exists, since the add is otherwise the one operation
    /// that would force the command buffer closed between `o_proj` and the
    /// second norm.
    ///
    /// `scale` multiplies the rows where they are read, which is the same
    /// convolution over a scaled input. It is here for one caller: a dense
    /// layer's `mlp_sconv` reads what `InklingDenseMLP` produced, and that
    /// network's trailing `global_scale` — see
    /// [`DenseMlp::scale`](inkling_core::ops::DenseMlp::scale) — is arithmetic
    /// its three dispatches leave over. Applying it where the rows are read
    /// costs a multiply and no dispatch; every other convolution in the model
    /// passes 1, a routed layer's included, because a router's two scales are
    /// already in the weights it applied.
    pub fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        carried: Option<&mut Buffer<f32>>,
        scale: f32,
        landing: Landing<'_>,
    ) -> Result<(), MetalError> {
        self.encode_rows(
            batch,
            Reading::whole(self.rows(x.len())),
            x,
            carried,
            scale,
            landing,
        )
    }

    /// The same convolution over `reading`'s rows of what a dispatch left on
    /// the device rather than over all of them.
    ///
    /// **A range because a batch's rows are one buffer.** N sequences advance
    /// through one set of projections, so what a convolution carrying one
    /// sequence's window has to read is that sequence's rows out of what they
    /// all produced — and the residual beside them is the same rows of the same
    /// call. A sequence advancing alone reads the whole of what it was handed,
    /// which is [`Reading::whole`] and is the indexing this always did.
    pub fn encode_rows(
        &self,
        batch: &mut Batch<'_>,
        reading: Reading,
        x: &mut Buffer<f32>,
        carried: Option<&mut Buffer<f32>>,
        scale: f32,
        landing: Landing<'_>,
    ) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Encode);
        let call = self.described(reading, x.len(), carried.as_deref(), &landing);
        let mut shape = self.device.inline(&call.fields)?;
        let scaled_by = [scale];
        let mut scaling = self.device.inline(&scaled_by)?;
        let mut weight = self.weight.borrow_mut();
        let mut windows = self.windows.borrow_mut();
        let (window, kept) = self.reading(&mut windows);

        // A slot the kernel is told to ignore still has to be filled, and one
        // float in the command buffer is what filling it costs — see
        // `Device::inline`, which allocates nothing for a value this small.
        let mut absent = self.device.inline(&[0.0f32])?;
        let carried = match carried {
            Some(carried) => carried.arg(),
            None => absent.arg(),
        };

        batch.add(
            &self.conv.kernel,
            &[
                shape.arg(),
                scaling.arg(),
                x.arg(),
                weight.arg(),
                window.arg(),
                landing.out.arg(),
                kept.arg(),
                carried,
            ],
            Grid::new(call.threads, THREADS_PER_GROUP),
            call.moves,
        )?;
        self.advanced(call.rows);
        Ok(())
    }

    /// Everything about a call that does not depend on whether it has a
    /// dispatch to itself: the eight fields the kernel reads, the threads it
    /// needs, and the bytes it moves.
    ///
    /// Here rather than inside [`LayerConv::encode_over`] because a paired
    /// dispatch asks the same of each half, and a second spelling of the shape
    /// struct is one that could drift from the kernel's own.
    fn described(
        &self,
        reading: Reading,
        values: usize,
        carried: Option<&Buffer<f32>>,
        landing: &Landing<'_>,
    ) -> Call {
        let Reading { from, rows } = reading;
        assert!(landing.groups > 0, "a row has groups");
        assert_eq!(
            self.channels % landing.groups,
            0,
            "{} channels are not {} groups",
            self.channels,
            landing.groups
        );
        landing.fits(rows, self.channels / landing.groups);
        assert!(
            from + rows <= self.rows(values),
            "{rows} rows at {from} of a call holding {}",
            self.rows(values)
        );

        if let Some(carried) = carried {
            assert_eq!(
                carried.len(),
                values,
                "a residual against what it is added to"
            );
        }
        Call {
            fields: [
                extent(rows, "the rows of a call"),
                extent(self.channels, "the channels of a convolution"),
                extent(self.taps, "the taps of a kernel"),
                extent(self.held.get().rows(), "the timesteps a window holds"),
                extent(landing.groups, "the groups of a row"),
                extent(landing.stride, "the rows a group has room for"),
                extent(landing.base, "where a call's rows start"),
                carried.is_some() as u32,
                extent(from, "where a call's rows are read from"),
            ],
            rows,
            // A thread to each channel of each timestep, and one more timestep's
            // worth for the window left behind — which reads the same padded
            // sequence the outputs are cut from and writes somewhere no output
            // thread touches.
            threads: (rows + self.held.get().rows()) * self.channels,
            // The sequence in and out, the kernel every timestep reads, the
            // window the call before this one left, the window this one leaves —
            // and the residual, where there is one to add. The scale and the
            // shape are in the command buffer rather than in memory, so they are
            // not traffic.
            moves: size_of::<f32>()
                * (2 * rows * self.channels
                    + self.channels * self.taps
                    + 2 * self.held.get().floats()
                    + carried.map_or(0, |_| rows * self.channels)),
        }
    }

    /// The window a call reads and the one it leaves behind, of the two this
    /// convolution alternates between.
    fn reading<'w>(
        &self,
        windows: &'w mut [Buffer<f32>; 2],
    ) -> (&'w mut Buffer<f32>, &'w mut Buffer<f32>) {
        let [first, second] = windows;
        match self.reading.get() {
            0 => (first, second),
            _ => (second, first),
        }
    }

    /// The two windows swapped and the sequence moved on by `rows`, which is
    /// what a call that has been encoded leaves behind it.
    fn advanced(&self, rows: usize) {
        self.reading.set(1 - self.reading.get());
        let mut held = self.held.get();
        held.advanced(rows);
        self.held.set(held);
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

/// One convolution's call as a paired dispatch describes it.
struct Call {
    fields: [u32; FIELDS],
    rows: usize,
    threads: usize,
    moves: usize,
}

/// One half of a paired convolution: the rows it reads, what it scales and adds,
/// and where its own rows land.
///
/// A struct rather than five more arguments because [`encode_pair`] takes two of
/// everything, and ten positional arguments of which five repeat is a call
/// nobody can read.
pub struct Convolving<'a> {
    pub conv: &'a LayerConv<'a>,
    pub x: &'a mut Buffer<f32>,
    /// Which rows of `x` and of the residual beside it this half reads — see
    /// [`LayerConv::encode_rows`].
    pub reading: Reading,
    pub carried: Option<&'a mut Buffer<f32>>,
    pub scale: f32,
    pub landing: Landing<'a>,
}

/// Which rows of what a dispatch left on the device a convolution reads.
///
/// **A range rather than a buffer because a batch's rows are one buffer.** The
/// projections of a batched step produce every sequence's rows together, and a
/// convolution carries one sequence's window — so what it reads is a run of
/// those rows and not all of them. A sequence advancing alone reads the whole
/// call, which is [`Reading::whole`].
#[derive(Debug, Clone, Copy)]
pub struct Reading {
    /// The row of `x` this call's first row is.
    pub from: usize,
    /// How many rows it takes from there.
    pub rows: usize,
}

impl Reading {
    /// Every row of what the call was handed, which is what a sequence
    /// advancing alone reads.
    pub fn whole(rows: usize) -> Self {
        Self { from: 0, rows }
    }
}

/// **Two convolutions as one dispatch.**
///
/// A layer's key and value convolutions are the pair this exists for: they read
/// different rows against different taps into different landings, each leaves
/// its own window behind, and neither reads what the other writes. A decode step
/// encoded 42 of each.
///
/// **Nothing has to agree between the halves**, which is where this parts
/// company with [`norm::encode_pair`](crate::norm::encode_pair): this kernel
/// declares no threadgroup memory and takes no barrier, so a thread is a channel
/// of a timestep whichever side it is on and there is no threadgroup shape for
/// two calls to disagree about. There is no fallback here because there is
/// nothing to fall back from.
///
/// The answer is the same bits, which
/// `a_paired_convolution_answers_what_the_two_it_replaces_answer` holds exactly:
/// the same taps walked from zero in the same order over the same padded
/// sequence, and the same window left behind.
pub fn encode_pair(
    batch: &mut Batch<'_>,
    first: Convolving<'_>,
    second: Convolving<'_>,
) -> Result<(), MetalError> {
    let _timed = profile::scope(Op::Encode);
    let Convolving {
        conv: one,
        x: first_x,
        reading: first_reading,
        carried: first_carried,
        scale: first_scale,
        landing: first_landing,
    } = first;
    let Convolving {
        conv: other,
        x: second_x,
        reading: second_reading,
        carried: second_carried,
        scale: second_scale,
        landing: second_landing,
    } = second;

    let first_call = one.described(
        first_reading,
        first_x.len(),
        first_carried.as_deref(),
        &first_landing,
    );
    let second_call = other.described(
        second_reading,
        second_x.len(),
        second_carried.as_deref(),
        &second_landing,
    );
    let device = one.device;
    let mut first_shape = device.inline(&first_call.fields)?;
    let mut second_shape = device.inline(&second_call.fields)?;
    let (first_scaled, second_scaled) = ([first_scale], [second_scale]);
    let mut first_scaling = device.inline(&first_scaled)?;
    let mut second_scaling = device.inline(&second_scaled)?;
    let mut first_weight = one.weight.borrow_mut();
    let mut second_weight = other.weight.borrow_mut();
    let mut first_windows = one.windows.borrow_mut();
    let mut second_windows = other.windows.borrow_mut();
    let (first_window, first_kept) = one.reading(&mut first_windows);
    let (second_window, second_kept) = other.reading(&mut second_windows);

    // One a side, for the reason `encode_over` gives — and two rather than one
    // because binding an inline value borrows it exclusively.
    let mut first_absent = device.inline(&[0.0f32])?;
    let mut second_absent = device.inline(&[0.0f32])?;
    let first_carried = match first_carried {
        Some(carried) => carried.arg(),
        None => first_absent.arg(),
    };
    let second_carried = match second_carried {
        Some(carried) => carried.arg(),
        None => second_absent.arg(),
    };

    batch.add(
        &one.conv.paired,
        &[
            first_shape.arg(),
            first_scaling.arg(),
            first_x.arg(),
            first_weight.arg(),
            first_window.arg(),
            first_landing.out.arg(),
            first_kept.arg(),
            first_carried,
            second_shape.arg(),
            second_scaling.arg(),
            second_x.arg(),
            second_weight.arg(),
            second_window.arg(),
            second_landing.out.arg(),
            second_kept.arg(),
            second_carried,
        ],
        Grid::new(first_call.threads + second_call.threads, THREADS_PER_GROUP),
        first_call.moves + second_call.moves,
    )?;
    one.advanced(first_call.rows);
    other.advanced(second_call.rows);
    Ok(())
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
    uint held;
    uint groups;
    uint stride;
    uint base;
    uint carried;
    uint from;
};

/// One channel of one timestep of `window ++ scale * x`, which is the padded
/// sequence every output row is cut from and the window left behind is the tail
/// of.
///
/// **The window may hold more than the convolution reads**, which is what lets
/// a speculative round be taken back — so the sequence starts `held` timesteps
/// before the first row rather than `taps - 1`, and an output row reaches past
/// the slack to find its own. `held` is `taps - 1` where nothing speculates,
/// and then this is the sequence it always was.
///
/// **The scale is on `x` alone.** What the window holds is what a previous call
/// was given, and a previous call was given rows already scaled — it is scaled
/// where a value *enters* the sequence, once, so that the window this leaves
/// behind is the same window `ConvState` would hold on the other side.
///
/// **`from` is where this call's rows start in `x`**, which is what lets a
/// sequence of a batch be convolved out of the buffer every sequence's rows are
/// in. It is zero for a call that was handed its own rows, and then this is the
/// indexing it always was.
inline float padded(
    device const float *x,
    device const float *window,
    constant Shape &shape,
    constant float &scale,
    uint at,
    uint c
) {
    if (at < shape.held) {
        return window[at * shape.channels + c];
    }
    return scale * x[(shape.from + at - shape.held) * shape.channels + c];
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
/// **`carried` is a second residual and belongs to the layer, not here.** A
/// convolution on a residual path has its rows added to the value the block
/// began with, and that add is what would otherwise force a command buffer
/// closed between the block and the norm after it. One addend more costs a read
/// and nothing else; the two convolutions inside attention have no block around
/// them and clear the flag.
///
/// **`scale` belongs to the layer too**, and to one kind of layer: it is what a
/// dense layer's MLP still owes on the rows this reads. Every other call passes
/// 1, so the multiply is exact and the answer is the one the CPU computes — see
/// `padded`, which is where a scaled row enters the sequence.
///
/// The taps are walked from zero and accumulated in that order, which is the
/// order `inkling_core::sconv` accumulates them in — and which is what makes a
/// sequence split anywhere the same sequence, since the only thing a split
/// changes is which call put a value in the window.
///
/// Here rather than inside an entry point because two entry points run it,
/// and what a paired dispatch changes is which of two shapes a thread reads
/// and nothing else.
static void convolve(
    constant Shape &shape,
    constant float &scale,
    device const float *x,
    device const float *weight,
    device const float *window,
    device float *out,
    device float *kept,
    device const float *carried,
    uint id
) {
    const uint t = id / shape.channels;
    const uint c = id % shape.channels;
    // What the window holds beyond what the convolution reads, which an output
    // row skips to reach its own timesteps.
    const uint slack = shape.held - (shape.taps - 1);

    // The last `held` timesteps of the padded sequence, which is what the next
    // call reads and what a rewind reaches back into. A call shorter than the
    // window cannot fill it, so part of what it keeps is part of what it was
    // given — the reference takes the tail of the padding either way, and
    // decoding one token at a time is entirely that case.
    if (t >= shape.rows) {
        kept[(t - shape.rows) * shape.channels + c] = padded(x, window, shape, scale, t, c);
        return;
    }

    device const float *taps = weight + (ulong)c * shape.taps;
    float acc = 0.0f;
    for (uint k = 0; k < shape.taps; ++k) {
        acc += taps[k] * padded(x, window, shape, scale, t + slack + k, c);
    }

    acc += scale * x[(shape.from + t) * shape.channels + c];
    if (shape.carried) {
        acc += carried[(shape.from + t) * shape.channels + c];
    }

    // Where the row lands, which for the value's convolution is the span the
    // layer keeps — see `Landing`. With one group and a stride of `rows` this is
    // `out[t * channels + c]`, the row where it was computed.
    const uint width = shape.channels / shape.groups;
    const uint group = c / width;
    device float *result =
        out + ((ulong)group * shape.stride + shape.base + t) * width + (c % width);
    *result = acc;
}

/// One call of it, over the threads a call's own rows and window need.
kernel void short_conv(
    constant Shape &shape [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const float *weight [[buffer(3)]],
    device const float *window [[buffer(4)]],
    device float *out [[buffer(5)]],
    device float *kept [[buffer(6)]],
    device const float *carried [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= (shape.rows + shape.held) * shape.channels) {
        return;
    }
    convolve(shape, scale, x, weight, window, out, kept, carried, id);
}

/// Two convolutions as one dispatch: a grid of both calls' threads, the first
/// call's at the front of it.
///
/// **A layer's key and value convolutions were two dispatches because they are
/// two sequences and for no other reason.** They read different rows against
/// different taps, they leave their own windows behind, and neither reads
/// anything the other writes.
///
/// This kernel declares no threadgroup memory and takes no barrier, so unlike
/// the paired norm there is nothing about a threadgroup that has to agree
/// between the halves: a thread is a channel of a timestep either way, and the
/// only thing the merge changes is which of two shapes a thread reads. The
/// answer is the same bits — the same taps walked from zero in the same order
/// over the same padded sequence.
kernel void short_conv_pair(
    constant Shape &first [[buffer(0)]],
    constant float &first_scale [[buffer(1)]],
    device const float *first_x [[buffer(2)]],
    device const float *first_weight [[buffer(3)]],
    device const float *first_window [[buffer(4)]],
    device float *first_out [[buffer(5)]],
    device float *first_kept [[buffer(6)]],
    device const float *first_carried [[buffer(7)]],
    constant Shape &second [[buffer(8)]],
    constant float &second_scale [[buffer(9)]],
    device const float *second_x [[buffer(10)]],
    device const float *second_weight [[buffer(11)]],
    device const float *second_window [[buffer(12)]],
    device float *second_out [[buffer(13)]],
    device float *second_kept [[buffer(14)]],
    device const float *second_carried [[buffer(15)]],
    uint id [[thread_position_in_grid]]
) {
    const uint firsts = (first.rows + first.held) * first.channels;
    if (id < firsts) {
        convolve(
            first, first_scale, first_x, first_weight, first_window,
            first_out, first_kept, first_carried, id
        );
    } else if (id - firsts < (second.rows + second.held) * second.channels) {
        convolve(
            second, second_scale, second_x, second_weight, second_window,
            second_out, second_kept, second_carried, id - firsts
        );
    }
}

"#;

#[cfg(test)]
mod tests {
    use std::time::Duration;

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

    /// The same rewind [`inkling_core::sconv`] states, on the device: rows fed,
    /// taken back and replaced are the same sequence as rows that were never
    /// fed — and the same sequence the CPU's own rewind produces.
    ///
    /// Exact against the device's own clean run, because both are the same
    /// dispatch over the same floats and the only thing a rewind changes is
    /// which call put a value in the window. Against the CPU it is the ordinary
    /// tolerance, because the contraction that separates the two backends is
    /// still there.
    #[test]
    fn rewinding_the_rows_a_dispatch_fed_leaves_the_window_it_had_before_them() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let sequence = fx.sequence(&fx.input, 0).to_vec();
        let wrong: Vec<f32> = sequence.iter().map(|value| -3.0 * value).collect();
        let cpu = fx.on_the_cpu(&fx.weight);

        for split in 1..fx.rows() {
            let taken = fx.rows() - split;
            let (before, after) = sequence.split_at(split * fx.channels);
            let layer = LayerConv::with_slack(&device, &conv, fx.channels, &fx.weight, taken)
                .expect("the kernel uploads");
            layer.forward(before).expect("the dispatch completes");
            layer
                .forward(&wrong[split * fx.channels..])
                .expect("the dispatch completes");
            layer.rewind(taken);
            let got = layer.forward(after).expect("the dispatch completes");

            let clean = LayerConv::with_slack(&device, &conv, fx.channels, &fx.weight, taken)
                .expect("the kernel uploads");
            clean.forward(before).expect("the dispatch completes");
            let want = clean.forward(after).expect("the dispatch completes");
            assert_eq!(got, want, "{taken} rows taken back at {split}");
            assert_eq!(layer.window(), clean.window(), "the window at {split}");

            let mut state = cpu.state();
            cpu.forward(&mut state, before, None);
            let deviation = deviation(&got, &cpu.forward(&mut state, after, None));
            assert!(deviation <= TOLERANCE, "at {split}: {deviation:e}");
        }
    }

    /// A window that kept no slack is the window this kernel always had, and
    /// asking it to give a timestep back is refused rather than answered out of
    /// rows that are nobody's.
    #[test]
    fn a_window_without_slack_holds_what_the_convolution_reads_and_nothing_else() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);

        assert_eq!(layer.window().len(), (fx.kernel_size - 1) * fx.channels);
        assert_eq!(layer.rewindable(), 0);
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
    fn a_dispatch_declares_the_window_it_carries_beside_the_rows_it_convolves() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let sequence = fx.sequence(&fx.input, 0);
        let mut x = device.buffer(sequence).expect("the rows upload");

        let plain = crate::testing::moved(&device, |batch| {
            layer
                .encode(batch, &mut x, None, 1.0)
                .expect("the dispatch encodes");
        });
        layer.restart();
        let mut x = device.buffer(sequence).expect("the rows upload");
        let mut residual = device.buffer(sequence).expect("the residual uploads");
        let carrying = crate::testing::moved(&device, |batch| {
            layer
                .encode(batch, &mut x, Some(&mut residual), 1.0)
                .expect("the dispatch encodes");
        });

        let window = (fx.kernel_size - 1) * fx.channels;
        assert_eq!(
            plain as usize,
            size_of::<f32>() * (2 * sequence.len() + fx.channels * fx.kernel_size + 2 * window),
            "the rows in and out, the taps, and the window either side of the call"
        );
        assert_eq!(
            carrying as usize - plain as usize,
            size_of_val(sequence),
            "a residual is one more pass over the rows"
        );
    }

    /// What the bandwidth column divides by, against what the kernel reads.
    ///
    /// **The window either side is the part a bytes-bound derived from the
    /// call's shape would miss**: this kernel reads the `K-1` inputs the call
    /// before it left and writes the `K-1` this one leaves, which is state no
    /// argument of the call names. The residual is the other term, and it is
    /// there on a layer's two convolutions and absent on attention's two.
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

    /// **A call over a run of a buffer's rows is the call over those rows
    /// handed alone**, bit for bit, and leaves the same window behind.
    ///
    /// This is what a batched step needs of the convolution: one set of
    /// projections produces every sequence's rows into one buffer, and the
    /// window a sequence carries is convolved out of the run that is its own.
    /// A range that read the wrong rows would still convolve, still be causal
    /// and still leave a window — so the rows in front of the run and behind it
    /// are made different values rather than padding, and the answer is
    /// compared exactly.
    ///
    /// The residual is carried, because it is indexed by the same row and a
    /// range applied to one and not the other is a layer adding another
    /// sequence's state to its own.
    #[test]
    fn a_convolution_over_a_run_of_rows_answers_what_that_run_alone_answers() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let rows = fx.rows();
        let ahead = fx.sequence(&fx.input, 0);
        let run = fx.sequence(&fx.input, 1);
        let behind = fx.sequence(&fx.input, 2 % fx.batch);
        assert_ne!(ahead, run, "rows a wrong range could be read from");

        let alone = |x: &[f32], residual: &[f32], from: usize| {
            layer.restart();
            let mut x = device.buffer(x).expect("the rows upload");
            let mut carried = device.buffer(residual).expect("the residual uploads");
            let mut out = device
                .zeroed::<f32>(fx.rows() * fx.channels)
                .expect("the landing allocates");
            let mut batch = device.batch().expect("a command buffer opens");
            layer
                .encode_rows(
                    &mut batch,
                    Reading { from, rows },
                    &mut x,
                    Some(&mut carried),
                    1.0,
                    Landing {
                        out: &mut out,
                        groups: 1,
                        stride: rows,
                        base: 0,
                    },
                )
                .expect("the dispatch encodes");
            batch.wait().expect("the batch completes");
            (out.to_vec(), layer.window())
        };

        let batched: Vec<f32> = [ahead, run, behind].concat();
        let residual: Vec<f32> = [behind, ahead, run].concat();
        assert_eq!(
            alone(&batched, &residual, rows),
            alone(run, ahead, 0),
            "the middle run of three against the same run alone"
        );
    }

    /// **A paired dispatch answers what the two dispatches it replaces answer,
    /// exactly, and leaves the same two windows behind** — which is the whole of
    /// what a merge is allowed to be, and here the window is half of it: a
    /// convolution that answered the same rows and kept the wrong state would
    /// pass on this step and be wrong on every one after it.
    ///
    /// The two halves are made as unalike as a layer's key and value
    /// convolutions are: different sequences, different taps, different scales,
    /// one with a carried residual and one without, and landings of different
    /// strides. Then several chunks of a sequence in a row, because a window that
    /// is right once and swapped wrongly is a state machine that fails on the
    /// second call rather than the first.
    #[test]
    fn a_paired_convolution_answers_what_the_two_it_replaces_answer() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        assert!(
            conv.paired.max_threads_per_group() >= THREADS_PER_GROUP,
            "the paired entry cannot be dispatched in the threadgroup this kernel uses"
        );
        let fx = Synthetic::load();
        let other_weight: Vec<f32> = fx.weight.iter().map(|w| 0.5 - w).collect();

        // Three calls of one sequence each way round, so the alternation of the
        // two windows is exercised rather than only their first use.
        let chunks = [2, 1, 3];
        let paired = {
            let one = fx.wrapped(&device, &conv, &fx.weight);
            let other = fx.wrapped(&device, &conv, &other_weight);
            let mut at = 0;
            chunks
                .iter()
                .map(|rows| {
                    let cut = at..at + rows * fx.channels;
                    at = cut.end;
                    convolved_pair(&device, &one, &other, &fx.input[cut], fx.channels)
                })
                .collect::<Vec<_>>()
        };
        let apart = {
            let one = fx.wrapped(&device, &conv, &fx.weight);
            let other = fx.wrapped(&device, &conv, &other_weight);
            let mut at = 0;
            chunks
                .iter()
                .map(|rows| {
                    let cut = at..at + rows * fx.channels;
                    at = cut.end;
                    convolved_apart(&device, &one, &other, &fx.input[cut], fx.channels)
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(paired, apart);
        assert!(
            paired
                .iter()
                .flat_map(|(a, b)| [a, b])
                .any(|rows| rows.iter().any(|value| *value != 0.0)),
            "two calls that wrote nothing would compare equal to two others that did"
        );
    }

    /// The two calls the case above pairs, as one dispatch: the first over `x`
    /// with a carried residual and a scale, the second over the same rows
    /// reversed with neither, into a landing of a different stride.
    fn convolved_pair(
        device: &Device,
        one: &LayerConv<'_>,
        other: &LayerConv<'_>,
        x: &[f32],
        channels: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let rows = x.len() / channels;
        let reversed: Vec<f32> = x.iter().rev().copied().collect();
        let mut first_x = device.buffer(x).expect("the rows upload");
        let mut second_x = device.buffer(&reversed).expect("the rows upload");
        let mut residual = device.buffer(&reversed).expect("the residual uploads");
        let mut first_out = device
            .zeroed::<f32>(x.len())
            .expect("the landing allocates");
        let mut second_out = device
            .zeroed::<f32>(2 * x.len())
            .expect("the landing allocates");
        let mut batch = device.batch().expect("a command buffer opens");
        super::encode_pair(
            &mut batch,
            Convolving {
                conv: one,
                x: &mut first_x,
                reading: Reading::whole(rows),
                carried: Some(&mut residual),
                scale: 1.5,
                landing: Landing {
                    out: &mut first_out,
                    groups: 1,
                    stride: rows,
                    base: 0,
                },
            },
            Convolving {
                conv: other,
                x: &mut second_x,
                reading: Reading::whole(rows),
                carried: None,
                scale: 1.0,
                landing: Landing {
                    out: &mut second_out,
                    groups: 1,
                    stride: 2 * rows,
                    base: 0,
                },
            },
        )
        .expect("the pair encodes");
        batch.wait().expect("the batch completes");
        (first_out.to_vec(), second_out.to_vec())
    }

    /// The same two, as the two dispatches they were.
    fn convolved_apart(
        device: &Device,
        one: &LayerConv<'_>,
        other: &LayerConv<'_>,
        x: &[f32],
        channels: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let rows = x.len() / channels;
        let reversed: Vec<f32> = x.iter().rev().copied().collect();
        let mut first_x = device.buffer(x).expect("the rows upload");
        let mut second_x = device.buffer(&reversed).expect("the rows upload");
        let mut residual = device.buffer(&reversed).expect("the residual uploads");
        let mut first_out = device
            .zeroed::<f32>(x.len())
            .expect("the landing allocates");
        let mut second_out = device
            .zeroed::<f32>(2 * x.len())
            .expect("the landing allocates");
        let mut batch = device.batch().expect("a command buffer opens");
        one.encode_over(
            &mut batch,
            &mut first_x,
            Some(&mut residual),
            1.5,
            Landing {
                out: &mut first_out,
                groups: 1,
                stride: rows,
                base: 0,
            },
        )
        .expect("the first encodes");
        other
            .encode_over(
                &mut batch,
                &mut second_x,
                None,
                1.0,
                Landing {
                    out: &mut second_out,
                    groups: 1,
                    stride: 2 * rows,
                    base: 0,
                },
            )
            .expect("the second encodes");
        batch.wait().expect("the batch completes");
        (first_out.to_vec(), second_out.to_vec())
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
            "acc += scale * x[(shape.from + t) * shape.channels + c];",
            "",
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

    /// **The layer's own residual is a second addend, not a second operation.**
    /// A convolution on a residual path has its rows added to the value the
    /// block began with, and that add is the whole of what would otherwise close
    /// the command buffer between `o_proj` and the layer's second norm.
    ///
    /// Exact rather than bounded, and that is the claim: the taps are summed in
    /// the same order either way and the carried value is added last, so what a
    /// dispatch carrying it produces is what the same dispatch produced plus
    /// that value element for element. A kernel that folded it into the
    /// accumulation instead would be within a tolerance and outside this.
    ///
    /// The carried rows are not the input, which is the mistake worth catching:
    /// against `carried == x` a kernel that added the input twice would agree.
    #[test]
    fn the_carried_residual_is_added_to_what_the_convolution_returns() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let sequence = fx.sequence(&fx.input, 0);
        let carried: Vec<f32> = (0..sequence.len())
            .map(|i| ((i * 29 % 53) as f32 - 26.0) / 4.0)
            .collect();
        assert_ne!(carried, sequence, "a residual equal to the input");

        layer.restart();
        let alone = layer.forward(sequence).expect("the dispatch completes");

        layer.restart();
        let mut input = device.buffer(sequence).expect("the rows upload");
        let mut residual = device.buffer(&carried).expect("the residual uploads");
        let mut batch = device.batch().expect("a command buffer opens");
        let out = layer
            .encode(&mut batch, &mut input, Some(&mut residual), 1.0)
            .expect("the convolution encodes");
        batch.wait().expect("the batch completes");

        let want: Vec<f32> = alone.iter().zip(&carried).map(|(a, b)| a + b).collect();
        assert_eq!(out.to_vec(), want);
        assert_eq!(
            layer.window(),
            sequence[sequence.len() - (fx.kernel_size - 1) * fx.channels..],
            "the window is the input's, whatever was carried"
        );
    }

    /// **A scaled call is the same convolution over scaled rows**, which is what
    /// lets a dense layer's trailing `global_scale` be a multiply where these
    /// rows are read rather than a dispatch of its own.
    ///
    /// Both halves of the claim are asserted, and only one of them is exact.
    /// The window a call leaves behind is `s * x` and nothing else, one rounding
    /// wherever that multiply happens, so it is the tail of the scaled sequence
    /// bit for bit — which is what keeps the *next* call the same too. The rows
    /// are within a couple of ulps rather than equal, because `taps[k] * (s * x)`
    /// is two multiplies Metal may contract where the CPU's pre-scaling rounds
    /// between them. Worst observed when this landed: 6.4e-9, two decades inside
    /// the bound.
    ///
    /// A scale of 1.75 rather than 2, so that the multiply is not exact in the
    /// exponent alone and a path that dropped it is decades away rather than
    /// a factor of two.
    #[test]
    fn a_scaled_call_is_the_convolution_over_rows_already_scaled() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let fx = Synthetic::load();
        let layer = fx.wrapped(&device, &conv, &fx.weight);
        let sequence = fx.sequence(&fx.input, 0);
        let scale = 1.75;

        let scaled: Vec<f32> = sequence.iter().map(|x| x * scale).collect();
        layer.restart();
        let want = layer.forward(&scaled).expect("the dispatch completes");
        let kept = layer.window();

        layer.restart();
        let mut input = device.buffer(sequence).expect("the rows upload");
        let mut batch = device.batch().expect("a command buffer opens");
        let out = layer
            .encode(&mut batch, &mut input, None, scale)
            .expect("the convolution encodes");
        batch.wait().expect("the batch completes");

        let agreed = deviation(&out.to_vec(), &want);
        eprintln!("a scaled call against rows already scaled: deviation {agreed:e}");
        assert!(agreed <= TOLERANCE, "the rows: deviation {agreed:e}");
        assert_eq!(layer.window(), kept, "the window it left behind");

        layer.restart();
        let unscaled = layer.forward(sequence).expect("the dispatch completes");
        assert!(
            deviation(&want, &unscaled) > TOLERANCE,
            "a scale a call could drop and still agree"
        );
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
                    None,
                    1.0,
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

    /// **What each convolution a layer runs costs the device, and how much of it
    /// is the dispatch rather than the convolution.**
    ///
    /// The per-kernel table says `short_conv` is 5.8% of a decode step's device
    /// time for 2% of its bytes, which is a ranking and not a diagnosis. K1 ruled
    /// out the one the norm had — these grids are 16 and 64 threadgroups where a
    /// norm's group was one — so the question left is which of the two things a
    /// small grid can be short of this is: the work, or the launch.
    ///
    /// The empty rows answer it. They dispatch a kernel that returns on its first
    /// instruction over the same grid, so what separates an empty row from the
    /// convolution beside it is everything the convolution does and nothing else.
    /// The row counts then say whether the arithmetic is what a decode-shaped call
    /// is paying for: a call over 97 rows does 97 times the convolutions of a call
    /// over one, over a grid 97 times as wide.
    ///
    /// Read off the device's own clock over a command buffer of `CALLS`
    /// dispatches rather than off the wall clock around one, for the reason
    /// [`crate::norm`]'s own diagnosis is: a submission is 225 microseconds and
    /// every figure here is under twenty.
    ///
    /// **A shape's convolution and a shape's floor are two measurements and are
    /// interleaved as two**, because this machine's own state moves a figure by
    /// more than the differences here are made of — and a floor always taken
    /// straight after the heavier dispatch it is subtracted from would carry
    /// whatever that dispatch left the machine in, in every round, in the same
    /// direction.
    ///
    /// Nothing asserts a duration; the numbers go to stderr for the commit
    /// message to quote.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_decode_steps_convolutions_cost_the_device() {
        let Some(device) = device() else { return };
        let conv = ShortConvolution::new(&device).expect("the kernel compiles");
        let mut empty = crate::testing::EmptyDispatch::new(&device);
        const CALLS: usize = 256;
        const ROUNDS: usize = 5;
        const TAPS: usize = 4;
        // The checkpoint's own shapes: the key and value convolutions inside
        // attention are over the KV heads, and the two on a layer's residual
        // paths are over the hidden width.
        const INSIDE: usize = 8 * 128;
        const HIDDEN: usize = 4096;

        // A thread to each channel of each timestep and one window's worth more,
        // which is the grid `encode_over` dispatches with no slack held.
        let grid = |rows: usize, channels: usize| {
            Grid::new((rows + TAPS - 1) * channels, THREADS_PER_GROUP)
        };

        let cost = |rows: usize, channels: usize, carried: bool| -> Duration {
            let weight: Vec<f32> = (0..channels * TAPS)
                .map(|i| (i % 13) as f32 / 16.0 - 0.4)
                .collect();
            let layer =
                LayerConv::new(&device, &conv, channels, &weight).expect("the kernel uploads");
            let x: Vec<f32> = (0..rows * channels)
                .map(|i| (i % 37) as f32 - 18.0)
                .collect();
            let mut input = device.buffer(&x).expect("the rows upload");
            let mut residual = device.buffer(&x).expect("the residual uploads");
            let mut out = device
                .zeroed::<f32>(x.len())
                .expect("the landing allocates");

            crate::testing::device_time(&device, CALLS, |batch| {
                layer
                    .encode_over(
                        batch,
                        &mut input,
                        carried.then_some(&mut residual),
                        1.0,
                        Landing {
                            out: &mut out,
                            groups: 1,
                            stride: rows,
                            base: 0,
                        },
                    )
                    .expect("the convolution encodes");
            })
        };

        let shapes = [
            ("key or value, decode", 1, INSIDE, false),
            ("residual path, decode", 1, HIDDEN, true),
            ("residual path, 97", 97, HIDDEN, true),
            ("residual path, 385", 385, HIDDEN, true),
            ("residual path, 769", 769, HIDDEN, true),
        ];
        // Warm: the first dispatch of a fresh pipeline pays for the driver's
        // first look at these buffers, which a decode loop pays once.
        let mut taken = shapes.map(|(_, rows, channels, carried)| {
            cost(rows, channels, carried);
            (Vec::with_capacity(ROUNDS), Vec::with_capacity(ROUNDS))
        });
        for (_, rows, channels, _) in shapes {
            empty.cost(&device, CALLS, grid(rows, channels));
        }
        for _ in 0..ROUNDS {
            for (each, (_, rows, channels, carried)) in taken.iter_mut().zip(shapes) {
                each.0.push(cost(rows, channels, carried));
            }
            for (each, (_, rows, channels, _)) in taken.iter_mut().zip(shapes) {
                each.1
                    .push(empty.cost(&device, CALLS, grid(rows, channels)));
            }
        }

        for ((name, rows, channels, _), (spent, launch)) in shapes.iter().zip(&taken) {
            let mean = |each: &Vec<Duration>| each.iter().sum::<Duration>() / each.len() as u32;
            let (spent, launch) = (mean(spent), mean(launch));
            let groups = grid(*rows, *channels).threads().div_ceil(THREADS_PER_GROUP);
            eprintln!(
                "{name:<22} [{rows}, {channels}] over {groups:>5} threadgroups: {spent:>8.2?} a \
                 dispatch, {launch:>8.2?} of it the launch — {:>5.0}% the convolution",
                100.0 * (spent.saturating_sub(launch)).as_secs_f64() / spent.as_secs_f64(),
            );
        }
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
