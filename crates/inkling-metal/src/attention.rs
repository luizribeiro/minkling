//! The attention step, on the device, with the relative-position bias computed
//! per element rather than read from a tensor.
//!
//! Every other kernel here consumes a weight. This one consumes five
//! activations and a `[d_rel, rel_extent]` projection, and what it produces is
//! the one value in a layer that no multiply against a weight can make: a
//! softmax over the keys, weighted into the values.
//!
//! # The mask is never a tensor
//!
//! [`inkling_core::mask`] builds the additive `[B, H, LQ, S]` tensor attention
//! adds to its logits, and that tensor is the reason long prefill is out of
//! reach: `reference/results/prefill.md` measures it at 23% of the peak over
//! resident at 16384 tokens, and the score buffers an explicit additive mask
//! forces MLX to materialise alongside it at another 34% — 57% together, against
//! a 32768-token prefill refused at a projected 406 GiB. It is quadratic in the
//! sequence and it is read exactly once.
//!
//! So it is not built. The kernel holds one query's `d_rel` relative features in
//! threadgroup memory and derives each key's entry from the backward distance
//! `(i + q_offset) - j` as it scores that key, which is the same four branches
//! [`BandedMask`](inkling_core::BandedMask) states, in the same order, evaluated
//! where the score is. What it costs is `d_rel` multiplies an element — 16 here
//! — against a tensor read that was going to happen anyway; what it buys is that
//! the largest allocation in a prefill is one nobody makes.
//!
//! # A magnitude rather than an infinity, all the way through
//!
//! A masked entry is [`MASKED`], the same `-1e30` the reference writes, and it
//! is *added* to the score rather than replacing it. That is what keeps a row
//! with no key it may attend to finite: the running peak of a row that is masked
//! end to end is `-1e30`, every exponent is `exp(0)`, and what comes back is the
//! mean of the values rather than a NaN. The band cannot produce such a row —
//! key `i + q_offset` is always visible to query `i` — but
//! [`inkling_core::Sdpa`] is pinned to that behaviour and a kernel that
//! substituted `-INFINITY` would part from it in the one place the two are
//! compared.
//!
//! # It is a streaming softmax, because the scores are not kept either
//!
//! The keys are walked in tiles, carrying a running peak, a running total and
//! the weighted values so far, and each tile rescales what came before it by
//! `exp(peak_before - peak_now)`. The CPU path shifts by the row's largest entry
//! in one pass over a row it has already written down; this one cannot, and does
//! not have to — the tensor of scores is as quadratic as the mask, and neither
//! is formed.

use std::borrow::Cow;
use std::cell::{Cell, RefCell, RefMut};

use inkling_core::attention::AttentionConfig;
use inkling_core::mask::MASKED;
use inkling_core::profile::{self, Op};

use crate::buffer::{Buffer, Landing};
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "fused_attention";

/// The second dispatch, which folds one query's splits back into one row.
const COMBINE: &str = "attention_combine";

/// What a profile calls the entry on each kind of layer — see
/// [`Kernel::under`], which is where one pipeline becomes two rows.
const WINDOWED: &str = "windowed attention";
const GLOBAL: &str = "global attention";

/// Threads one threadgroup of a dispatch holds, all of them on one query of one
/// head.
///
/// The two phases of a tile want different widths and this is the compromise
/// between them. Scoring gives a simdgroup to each key, so 8 simdgroups score 8
/// keys at once with the 32 lanes of each reading 32 consecutive channels of one
/// key; weighting gives a *thread* to each channel, so 128 of the 256 are busy
/// at Inkling's `head_dim`. Halving this would balance the second phase and
/// double the number of tiles the first pays a barrier for.
const THREADS_PER_GROUP: usize = 256;

/// Keys one simdgroup scores before the threadgroup reduces the tile.
///
/// A tile is `simdgroups * KEYS_PER_SIMD` keys and each tile costs four
/// barriers, so this is what keeps the barriers off a long row: at Inkling's
/// shape a tile is 32 keys, and a decode over a 1024-token context is 32 tiles
/// rather than 128.
const KEYS_PER_SIMD: usize = 4;

/// Entries the kernel's threadgroup arrays hold, which have to be constants
/// where the shapes are not. 1024 threads is the widest threadgroup any Apple
/// GPU allows and 32 the narrowest simdgroup any reports, so 32 simdgroups is
/// the most a threadgroup can hold.
const MOST_SIMDGROUPS: usize = 32;

/// Channels one head may have, which bounds the staged query row and the
/// weighted sum beside it. Inkling's `head_dim` is 128.
const MOST_CHANNELS: usize = 256;

/// Floats of a tile's values one pass of the weighting walks, which is what
/// bounds the tile: a tile is the smaller of what the simdgroups can score and
/// what this allows, so a head wider than 128 channels walks fewer keys a tile
/// rather than more floats.
///
/// **It is a bound on the tiling and not on any array**, which is what lets the
/// two entries below declare different memory and still walk the same keys in
/// the same tiles. Both are compiled from this one number, so a tile is the 32
/// keys eight simdgroups score whichever entry a call lands on, and the claim
/// that the two agree bit for bit rests on that.
const TILED_VALUES: usize = 4096;

/// Floats of a tile's values a threadgroup stages before it weights them, on the
/// entry a split call runs.
///
/// **This is what makes the weighting affordable at a decode step's shape.** A
/// thread that carries one channel across a tile reads that channel of each
/// value in turn, and those reads are `head_dim` floats apart and each waits for
/// the last: a dependent chain as long as the tile, at a memory latency apiece.
/// Staged first, the same values arrive as one cooperative copy that every
/// thread has several independent loads in — 726 µs of device time for one query
/// over 1200 keys before the staging and 458 after, on the same dispatch.
const STAGED_BY_A_SPLIT_CALL: usize = TILED_VALUES;

/// **A threadgroup that stages copies a whole tile whatever it declared**, so an
/// entry that stages has to be able to hold one: the copy is bounded by the keys
/// the tile holds and not by the array, and a smaller array would be a walk
/// writing past its own memory rather than a smaller staging.
const _: () = assert!(STAGED_BY_A_SPLIT_CALL >= TILED_VALUES);

/// What the other entry stages, which is nothing: one float, so that the array
/// is legal to declare and every use of it folds away.
///
/// A threadgroup that stages a tile declares 19 KiB and a core holds four of
/// them, which is two steps past this walk's own occupancy turn. What the entry
/// an unsplit call runs declares instead is [`RESIDENCY`].
const STAGED_BY_AN_UNSPLIT_CALL: usize = 1;

/// Floats of threadgroup memory the entry that stages nothing declares and never
/// reads for anything, which is what puts an unsplit call on the fast side of
/// its occupancy turn.
///
/// **This is a residency control and not a buffer, and it is load-bearing.** A
/// threadgroup's four live arrays are 3 KiB; a core holds as many threadgroups
/// as their declared memory divides into its own 80 KiB, and
/// [`how_many_threadgroups_of_a_prefills_attention_a_core_holds`] measures this
/// walk fastest at six of them — 71.11 ms against the staged entry's 92.63 on
/// the same 2048-token dispatch, and 61% worse again at seven. So the memory
/// nothing reads is what keeps the seventh threadgroup off the core.
///
/// **It has to be memory nobody reads rather than a smaller staging, and that
/// is measured rather than assumed.** A threadgroup that stages *any* of a tile
/// is 116 ms at this declaration where one that stages none is 71 — two keys of
/// the tile costs as much as nineteen — while at the five-threadgroup
/// declaration above it the two agree to 2%. So what the left arm of the curve
/// punishes is the staging and not the memory, and an entry that wants six
/// threadgroups a core cannot bring a value in early at all. Why is not settled
/// here; that it is so is four rows of the sweep.
///
/// **Sized to the middle of its plateau rather than to its edge.** Six
/// threadgroups a core is every declaration in (11.43, 13.33] KiB, measured to
/// the quarter-kibibyte: 11.25 KiB is seven of them at 114.33 ms where 11.5 KiB
/// is six at 71.11, and 13.25 KiB is six at 71.21 where 13.5 KiB is five at
/// 76.73. This is 12.5 KiB declared, which leaves about a kibibyte of margin on
/// each side.
const RESIDENCY: usize = 2432;

/// What the entry that stages a tile declares of it instead, which is nothing:
/// the staging is already past the turn and a second array would only be more
/// of what put it there.
const NO_RESIDENCY: usize = 1;

/// Relative features one query may carry, which bounds the staged `r_proj` row.
/// Inkling's `d_rel` is 16.
const MOST_FEATURES: usize = 128;

/// The narrowest simdgroup any Apple GPU reports, which is what the tile a
/// dispatch will walk in has to be worked out from on this side.
///
/// The kernel takes its own width from `threads_per_simdgroup` and this side
/// cannot ask before dispatching, so the two agree by this being a floor — and
/// the direction is worth working out rather than asserting. A threadgroup is a
/// fixed 256 threads, so a *wider* simdgroup is *fewer* of them and a *smaller*
/// tile, which is *more* tiles in a span than this side counted. A split count
/// bounded by the count from 32 is therefore at or under the tiles that exist,
/// which is what it has to be: the partition itself is the kernel's own
/// arithmetic over the kernel's own tile, and all this bound decides is that no
/// split is asked for that has no tile to hold it.
const NARROWEST_SIMD: usize = 32;

/// Splits a call may cut its key span into, which bounds the combine's own
/// threadgroup array.
///
/// Above where [`WANTED_GROUPS`] turns, so that the constant below is fitted at
/// its own sweep's turn rather than pinned against this one.
const MOST_SPLITS: usize = 128;

/// Threadgroups a call aims to give the machine, which is what decides how many
/// splits it cuts.
///
/// **A decode step is 32 threadgroups on an 80-core machine and that is the
/// whole of what this fixes.** One threadgroup to each query head is 32 of them
/// at one query, so 48 cores have no work at all and the 32 that do have one
/// apiece — nothing to interleave against, on a kernel whose every tile is four
/// barriers and a dependent read. Measured, the step reaches 9 to 16 GB/s
/// against a machine that does 819, and it holds that rate from 97 keys to
/// 65536: the kernel is not waiting on memory, it is waiting on itself.
///
/// So the span is cut instead, which is the lever the argmax took for the same
/// reason — see "Sampling on the device": one threadgroup over a row of the
/// vocabulary was this process's own argmax to within its spread, and the whole
/// of what a device argmax was worth turned out to be the cut. The norm's
/// arithmetic went the other way and is the reason this is a number rather than
/// a rule: a split costs a dispatch, so it has to buy more than one.
///
/// Splits below which a call takes none, unless the span had no more tiles to
/// give — see [`splits_for`], where the sweep that puts it here is quoted.
const LEAST_SPLIT: usize = 16;

/// **Swept rather than reasoned about**, by
/// `what_the_split_over_the_key_span_is_worth` — and the sweep says the turn is
/// not in the same place at two contexts. Summed over the stack's 35 windowed
/// layers and 7 global ones, a 769-key step is best cut 16 ways and an 8192-key
/// one 64 ways, and each is half again as expensive at the other's figure.
///
/// So this is deliberately past the shorter turn and the *tile clamp* is what
/// brings a short context back down: a span has only so many tiles to give, and
/// 97, 385 and 769 keys have 4, 13 and 25 of them where 2048 and up have 64 or
/// more. One number and a clamp lands within a few percent of the swept best at
/// every length, which two numbers and a length threshold would have to earn.
const WANTED_GROUPS: usize = 2048;

/// What wrapping one layer's step can fail with.
///
/// Wrapping and not calling: everything here is a shape, settled once against
/// the layer the checkpoint describes, so a *call* can only fail the way any
/// dispatch fails and answers with a [`MetalError`].
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error(transparent)]
    Metal(#[from] MetalError),

    #[error("{got} projection coefficients are not whole rows of {d_rel}")]
    PartialBand { d_rel: usize, got: usize },

    #[error("{heads} query heads do not divide into {kv_heads} groups")]
    UngroupedHeads { heads: usize, kv_heads: usize },

    #[error("a head of {head_dim} channels is wider than the {MOST_CHANNELS} a threadgroup stages")]
    TooManyChannels { head_dim: usize },

    #[error("{d_rel} relative features are more than the {MOST_FEATURES} a threadgroup stages")]
    TooManyFeatures { d_rel: usize },
}

/// The compiled kernel, which every attention layer on a device shares.
///
/// Per source string rather than per layer, like [`crate::RmsNorm`]: the source
/// names no shape, so one of these serves all forty-two.
#[derive(Debug)]
pub struct FusedAttention {
    /// The entry a global layer's call is charged to.
    global: Kernel,
    /// The same pipeline under the other name, for the 35 layers whose queries
    /// stop at their window. **Two rows and one kernel**: a prefill's two terms
    /// are shaped differently — linear in the prompt at a windowed layer and
    /// quadratic at a global one — and a single row is a sum that says neither.
    windowed: Kernel,
    /// The same two, compiled to stage a whole tile of values rather than the
    /// part of one an unsplit call's residency leaves room for.
    ///
    /// **Two pipelines and one kernel**, the way the two above are two names for
    /// one: the source is the same string with one constant in it, the walk is
    /// the same walk over the same tiles, and the answers are the same bits —
    /// which `a_split_call_stages_a_whole_tile_and_answers_the_same_bits` is
    /// what says. They are charged to the same two rows for the same reason.
    split: [Kernel; 2],
    /// The fold behind a split call, out of the same source string: the two
    /// entries share [`Shape`] and a mutation to one is a source a test has to
    /// be able to compile both out of.
    combine: Kernel,
}

impl FusedAttention {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_sources(
            device,
            &source(STAGED_BY_AN_UNSPLIT_CALL, RESIDENCY),
            &source(STAGED_BY_A_SPLIT_CALL, NO_RESIDENCY),
        )
    }

    /// [`FusedAttention::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    ///
    /// **One string drives both regimes**, so that an arm holding a term out of
    /// the kernel holds it out wherever the call lands rather than only where
    /// the predicate happens to send it.
    #[cfg(test)]
    fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Self::from_sources(device, source, source)
    }

    fn from_sources(device: &Device, unsplit: &str, split: &str) -> Result<Self, MetalError> {
        let whole = device.compile(unsplit, ENTRY)?;
        let cut = device.compile(split, ENTRY)?;
        Ok(Self {
            windowed: whole.under(WINDOWED),
            global: whole.under(GLOBAL),
            split: [cut.under(GLOBAL), cut.under(WINDOWED)],
            combine: device.compile(split, COMBINE)?,
        })
    }

    /// The entry as a layer of this window is charged for it, out of the pair
    /// this many splits picks.
    ///
    /// **The predicate is the one this kernel already draws**, which is whether
    /// the call was cut and not whether it is a prefill. [`splits_for`] leaves a
    /// call whole where the grid already fills the machine, where the span has
    /// too few tiles to cut, and where a cut would be too narrow to spread the
    /// live keys — so an uncut call is every prompt worth the name, a
    /// speculative block at a long context, and a decode step with 32 keys or
    /// fewer behind it, and those take the residency the occupancy turn wants.
    /// A cut call is a decode step at a context somebody has, and keeps the
    /// kernel it has with its tile staged whole.
    ///
    /// **The two answer the same bits**, so what this decides is a rate and
    /// never an answer — which is what lets the line be drawn on a number that
    /// was chosen for something else.
    fn on(&self, sliding: usize, splits: usize) -> &Kernel {
        match (splits, sliding) {
            (1, 0) => &self.global,
            (1, _) => &self.windowed,
            (_, 0) => &self.split[0],
            (_, _) => &self.split[1],
        }
    }
}

/// Keys one tile of the walk holds, as this side has to work it out.
///
/// The kernel takes the same minimum of what its simdgroups can score and what
/// [`TILED_VALUES`] allows; here the first half is a floor rather than the
/// device's own width, for the reason [`NARROWEST_SIMD`] gives.
fn tile_keys(head_dim: usize) -> usize {
    (THREADS_PER_GROUP / NARROWEST_SIMD * KEYS_PER_SIMD).min(TILED_VALUES / head_dim)
}

/// Keys the query at `position` walks, which is the kernel's own `[reach, last)`
/// — the causal bound, the window behind it, and the tile the window is rounded
/// down to so that the bounded loop stays the unbounded one bit for bit.
///
/// A tile wider than the kernel's own would round `reach` further down and count
/// keys the walk skips. It cannot be narrower: `tile_keys` takes the widest a
/// simdgroup floor allows, for the reason [`NARROWEST_SIMD`] gives — so this is
/// exact where a simdgroup is 32 lanes and an over-count of under a tile a row
/// anywhere else.
fn keys_walked(position: usize, keys: usize, sliding: usize, tile: usize) -> usize {
    let last = keys.min(position + 1);
    let reach = match sliding {
        0 => 0,
        window if position >= window => (position - (window - 1)) / tile * tile,
        _ => 0,
    };
    last.saturating_sub(reach)
}

/// The same summed over a call's query rows, which is what separates a decode
/// step's declared bytes from a prefill's.
///
/// **A span is read once a query row and not once a call**, which is the whole
/// of the difference between the two regimes: a decode step's one row walks a
/// span and a prefill of `n` rows through a global layer walks `n²/2` keys.
///
/// Summed rather than closed-form, because the bound above is the kernel's and a
/// second expression for it is a second thing that can drift from the source.
fn keys_a_call_walks(
    queries: usize,
    q_offset: usize,
    keys: usize,
    sliding: usize,
    tile: usize,
) -> usize {
    (0..queries)
        .map(|i| keys_walked(q_offset + i, keys, sliding, tile))
        .sum()
}

/// How many splits a call of `pairs` query-head threadgroups over `keys` keys
/// cuts its span into.
///
/// **One where the grid already fills the machine**, which is every prefill and
/// every speculative round wide enough — 769 queries over 32 heads is 24608
/// threadgroups and nothing about them is short of work. A split there would buy
/// no parallelism and cost a dispatch, a buffer and a fold, which is what
/// `RUNS_A_GROUPING` cost the matmul for existing.
///
/// And never more splits than there are tiles to give them: a split with no tile
/// in it is a threadgroup that returns on its first instruction, and past that
/// point the fold grows while the walk does not.
fn splits_for(pairs: usize, keys: usize, head_dim: usize) -> usize {
    let tiles = keys.div_ceil(tile_keys(head_dim)).max(1);
    let splits = WANTED_GROUPS
        .div_ceil(pairs.max(1))
        .clamp(1, tiles.min(MOST_SPLITS));
    // **Either a cut that spreads the live keys or no cut at all**, which is
    // what the sweep found on a nine-row block: at 8192 keys the predicate's own
    // eight splits are 565 µs on a windowed layer against 326 unsplit, because
    // a windowed layer's live 512 keys fall inside one or two of eight splits
    // while the fold is charged all eight of every one of the block's 288 rows.
    // The fold grows with `pairs * splits` and the walk only shrinks where a
    // split has live keys in it, so a small cut over a wide grid pays for the
    // first and gets none of the second.
    //
    // A span with fewer tiles than that is a different case and keeps its cut:
    // there the clamp is what chose the number, every split has a tile in it,
    // and 97 and 385 keys are 4 and 13 tiles.
    if splits < LEAST_SPLIT && splits < tiles {
        return 1;
    }
    splits
}

/// What one call of the attention step reads, in the layouts the pieces around
/// it already produce.
///
/// `q` arrives with log scaling's `tau` already multiplied through it and the
/// biases do not — see [`LayerAttention::encode`] — which is the one asymmetry
/// here and is the seam's rather than the kernel's.
#[derive(Debug, Clone, Copy)]
pub struct Step<'a> {
    /// `[heads, queries, head_dim]`, the layout
    /// [`split_heads`](inkling_core::split_heads) produces.
    pub q: &'a [f32],
    /// `[kv_heads, keys, head_dim]`, the whole cached span.
    pub k: &'a [f32],
    pub v: &'a [f32],
    /// `[queries, heads, d_rel]` — query-major and head-minor, which is what
    /// `r_proj` produces and is the opposite of everything else here.
    pub rel: &'a [f32],
    /// One `tau` per query, which the biases of that query are multiplied by,
    /// or `None` on a layer with no log scaling — which is what
    /// [`AttentionStep`](inkling_core::AttentionStep) hands over and is a row of
    /// ones by the time the kernel reads it.
    pub taus: Option<&'a [f32]>,
    /// Where this call's queries sit: query `i` is at absolute position
    /// `i + q_offset`.
    pub q_offset: usize,
}

/// Key slots a layer's span starts with, and the least it is ever grown to.
///
/// A span is `[kv_heads, capacity, head_dim]` float32, which at Inkling's shape
/// is 4 KB a slot a layer for each of the two tensors — so 64 slots is 21 MB
/// across the stack and buys a decode loop 64 steps between reallocations.
const LEAST_KEYS: usize = 64;

/// One attention layer's keys and values, kept on the device between steps.
///
/// **What this replaces is a copy of the whole span, per layer, per step.** The
/// keys a decode step attends over are the keys the step before it attended
/// over and one more; handed to [`LayerAttention::encode`] as a slice, all of
/// them are allocated and copied onto the device again — 53 µs a layer at 16
/// keys, and a cost that grows with the context where nothing else in a decode
/// step does. Held here, a step copies the one key it made.
///
/// The span is `[kv_heads, capacity, head_dim]` and grows by powers of two, so
/// what a call appends is `kv_heads` writes of `head_dim` floats at a stride
/// rather than an append to a contiguous tail. That stride is why the kernel
/// takes `key_stride` beside `keys`, and it is what [`KeyValues::landings`] hands
/// a dispatch that means to write the keys itself.
///
/// It is **one sequence's**, and nothing here makes it more than that: the layer
/// holds one span, so two sequences interleaved through the same layer would
/// overwrite each other's keys. [`LayerAttention::hold`] is where that is
/// refused rather than discovered — a sequence's own
/// [`AttentionCache`](inkling_core::AttentionCache) carries how many keys it has
/// seen, and a span holding a different number is not its.
///
/// Not public beyond this crate, and not only for want of a caller: `landings`
/// and `appended` are two halves of one encoding, and a caller that took the
/// first and skipped the second — or took the second and then abandoned the
/// command buffer — would leave the span claiming keys no dispatch wrote.
#[derive(Debug)]
pub(crate) struct KeyValues {
    keys: Buffer<f32>,
    values: Buffer<f32>,
    kv_heads: usize,
    head_dim: usize,
    /// Key slots each head has room for, which is the stride between heads.
    capacity: usize,
    /// How many of them the sequence in flight has filled.
    held: usize,
}

impl KeyValues {
    fn new(device: &Device, kv_heads: usize, head_dim: usize) -> Result<Self, MetalError> {
        Ok(Self {
            keys: device.zeroed(kv_heads * LEAST_KEYS * head_dim)?,
            values: device.zeroed(kv_heads * LEAST_KEYS * head_dim)?,
            capacity: LEAST_KEYS,
            held: 0,
            kv_heads,
            head_dim,
        })
    }

    /// The span a sequence starts from, which is no keys.
    ///
    /// The slots are not cleared. Nothing reads past `held` — the kernel's loop
    /// bound is `keys` — so what a previous sequence left in them is memory
    /// nobody indexes rather than values that could leak into an answer.
    fn restart(&mut self) {
        self.held = 0;
    }

    /// The span with the last `rows` keys unwanted.
    ///
    /// **This is the easy half of taking a speculative token back**, and it is
    /// worth saying why: a key is addressed by its position, and the loop bound
    /// is how many the sequence has, so a key nobody indexes is a key that is
    /// not there. What is left over is a slot the next call writes. The
    /// convolution windows either side of this are the half that had to be
    /// designed for — see [`LayerConv::rewind`](crate::LayerConv::rewind).
    fn rewind(&mut self, rows: usize) {
        assert!(
            rows <= self.held,
            "a rewind of {rows} against a span holding {}",
            self.held
        );
        self.held -= rows;
    }

    /// Room for `keys` keys a head, growing the span if there is not.
    ///
    /// Powers of two, so a decode loop reallocates a logarithmic number of times
    /// over a generation and copies each key a constant number of times. What is
    /// copied is the prefix each head has filled, which is what makes the growth
    /// invisible to the sequence.
    fn reserve(&mut self, device: &Device, keys: usize) -> Result<(), MetalError> {
        if keys <= self.capacity {
            return Ok(());
        }
        let capacity = Self::capacity_for(keys);
        let mut grown = [
            device.zeroed::<f32>(self.kv_heads * capacity * self.head_dim)?,
            device.zeroed::<f32>(self.kv_heads * capacity * self.head_dim)?,
        ];
        let filled = self.held * self.head_dim;
        for (grown, held) in grown.iter_mut().zip([&self.keys, &self.values]) {
            let (grown, held) = (grown.as_mut_slice(), held.as_slice());
            for kv in 0..self.kv_heads {
                let (to, from) = (kv * capacity, kv * self.capacity);
                grown[to * self.head_dim..][..filled]
                    .copy_from_slice(&held[from * self.head_dim..][..filled]);
            }
        }
        let [keys, values] = grown;
        (self.keys, self.values, self.capacity) = (keys, values, capacity);
        Ok(())
    }

    /// Slots a span holding `keys` keys is allocated: powers of two from
    /// [`LEAST_KEYS`], so a decode loop reallocates a logarithmic number of
    /// times over a generation.
    ///
    /// Named rather than inlined into [`KeyValues::reserve`] because
    /// `what_a_context_costs_in_keys_and_values` asks what a span *would* cost
    /// against a key count no span was allocated for, and a second copy of this
    /// rule is one that can go stale against the first while the table it feeds
    /// goes on reading like a measurement.
    pub(crate) fn capacity_for(keys: usize) -> usize {
        keys.next_power_of_two().max(LEAST_KEYS)
    }

    /// What the two spans occupy on the device.
    ///
    /// **The one part of a sequence's footprint that grows with it**, and the
    /// figure the architecture's KV arithmetic is a claim about: the README puts
    /// a 1M-token context under 30 GiB on the grounds that only the 7 global
    /// layers grow, where 35 cap at a 512-token window. This is what a layer
    /// actually holds, so that the claim is measured rather than repeated.
    ///
    /// The capacity and not the keys: a span is allocated in powers of two and
    /// what it costs is what it reserved.
    pub(crate) fn bytes(&self) -> u64 {
        2 * (self.kv_heads * self.capacity * self.head_dim) as u64 * size_of::<f32>() as u64
    }

    /// Where a dispatch writing this call's keys should put them, and where one
    /// writing its values should.
    ///
    /// Both at once because a call writes both and they are separate buffers, so
    /// the borrow of one has to be able to outlive the other's dispatch.
    ///
    /// Neither advances `held`: the rows are not there until the batch has run,
    /// and [`KeyValues::appended`] is what says they are.
    pub(crate) fn landings(&mut self) -> (Landing<'_>, Landing<'_>) {
        let (groups, stride, base) = (self.kv_heads, self.capacity, self.held);
        (
            Landing {
                out: &mut self.keys,
                groups,
                stride,
                base,
            },
            Landing {
                out: &mut self.values,
                groups,
                stride,
                base,
            },
        )
    }

    /// Record `rows` keys and values a dispatch wrote into the span.
    ///
    /// **The caller has to see that command buffer through.** This is called
    /// while the batch is being encoded, because the step that reads the keys is
    /// in it too — so a caller that then abandoned the buffer rather than
    /// submitting it would leave the span counting keys the device never wrote.
    /// [`LayerProjections::layer`](crate::LayerProjections) is the only caller
    /// and treats a batch that does not run as a panic, which is the same thing
    /// every dispatch in this crate does with one.
    pub(crate) fn appended(&mut self, rows: usize) {
        assert!(
            self.held + rows <= self.capacity,
            "{rows} keys past a span reserved for {}",
            self.capacity
        );
        self.held += rows;
    }

    /// Append `[rows, kv_heads * head_dim]` keys and values held in this
    /// process's memory, split into heads on the way in.
    ///
    /// What a caller with the keys here does, which is what
    /// [`LayerAttention::forward`] is for a caller with the whole step here.
    /// A layer reaches for [`KeyValues::landings`] instead, because a key it
    /// wrote out and read back is a key that crossed the seam twice.
    ///
    /// The split is the write's own indexing rather than a pass over a tensor:
    /// what the projections produce is head-major within a row and what the
    /// kernel reads is key-major within a head, and a copy that walks the first
    /// can address the second.
    fn append(&mut self, k: &[f32], v: &[f32]) {
        let stride = self.kv_heads * self.head_dim;
        assert_eq!(k.len(), v.len(), "values against keys");
        assert_eq!(k.len() % stride, 0, "{} values are not keys", k.len());
        let rows = k.len() / stride;
        assert!(
            self.held + rows <= self.capacity,
            "{rows} keys past a span reserved for {}",
            self.capacity
        );

        for (span, rows_of) in [(&mut self.keys, k), (&mut self.values, v)] {
            let span = span.as_mut_slice();
            for (t, row) in rows_of.chunks_exact(stride).enumerate() {
                for kv in 0..self.kv_heads {
                    let at = (kv * self.capacity + self.held + t) * self.head_dim;
                    span[at..][..self.head_dim]
                        .copy_from_slice(&row[kv * self.head_dim..][..self.head_dim]);
                }
            }
        }
        self.held += rows;
    }
}

/// One attention layer's mask projection on the device, and the step through it.
///
/// The projection is `[d_rel, rel_extent]` — 64 KB on a global layer — and is
/// copied once at wrap time rather than per call, for the reason
/// [`crate::LayerNorm`]'s weight is: it is float32 in the checkpoint's own
/// widening, and the CPU path reads it out of that widening on every layer of
/// every step.
///
/// **It is copied transposed, and that is what makes the bias affordable.** The
/// checkpoint stores a row per relative feature, so one distance's `d_rel`
/// coefficients sit `rel_extent` floats apart — 4 KB apart on a global layer,
/// which is sixteen scattered reads for every element of a mask that is
/// computed rather than read. Stored `[rel_extent, d_rel]` they are sixteen
/// consecutive floats: one cache line an element, and the same sixteen products
/// in the same order.
#[derive(Debug)]
pub struct LayerAttention<'a> {
    device: &'a Device,
    attention: &'a FusedAttention,
    /// Behind a cell for the reason [`crate::LayerNorm`]'s weight is: binding a
    /// buffer to a dispatch borrows it exclusively, and the projection belongs
    /// to the layer rather than to the call.
    proj: RefCell<Buffer<f32>>,
    /// The keys and values this layer has attended over, kept across steps.
    ///
    /// Behind a cell for the reason the projection is, and holding the same
    /// relation to the layer: it belongs to the layer rather than to the call,
    /// and a call binds it.
    span: RefCell<KeyValues>,
    config: AttentionConfig,
    rel_extent: usize,
    /// Splits a call cuts its span into, where a caller has pinned it rather
    /// than left it to [`splits_for`].
    ///
    /// A seam for the same reason [`FusedAttention::from_source`] is one: what
    /// fits [`WANTED_GROUPS`] is a sweep across split counts on one shape, and
    /// the alternative to pinning it here is a sweep that re-derives the
    /// predicate it is meant to be fitting. Nothing outside this crate can set
    /// it and nothing inside it does but
    /// `what_the_split_over_the_key_span_is_worth`.
    pinned: Cell<Option<usize>>,
}

impl<'a> LayerAttention<'a> {
    /// `proj` is the layer's own `rel_proj`, `[d_rel, rel_extent]` row-major.
    ///
    /// Its extent is read off its length rather than passed, which is where
    /// [`BandedMask::new`](inkling_core::BandedMask::new) reads it too: a
    /// sliding layer's band is its window and a global layer's is `rel_extent`,
    /// and the checkpoint stores the tensor at whichever width the layer uses.
    pub fn new(
        device: &'a Device,
        attention: &'a FusedAttention,
        config: AttentionConfig,
        proj: &[f32],
    ) -> Result<Self, AttentionError> {
        if config.d_rel == 0 || proj.len() % config.d_rel != 0 {
            return Err(AttentionError::PartialBand {
                d_rel: config.d_rel,
                got: proj.len(),
            });
        }
        // What `Sdpa::new` asserts, for the reason a kernel has to have it
        // asserted: the grouping is a divide in the kernel, so a `kv_heads` of
        // zero divides by zero there and one that does not divide the query
        // heads sends the last block of them past the end of the keys — an
        // address a GPU read answers with whatever is there rather than with a
        // fault.
        if config.kv_heads == 0 || config.heads % config.kv_heads != 0 {
            return Err(AttentionError::UngroupedHeads {
                heads: config.heads,
                kv_heads: config.kv_heads,
            });
        }
        if config.head_dim > MOST_CHANNELS {
            return Err(AttentionError::TooManyChannels {
                head_dim: config.head_dim,
            });
        }
        if config.d_rel > MOST_FEATURES {
            return Err(AttentionError::TooManyFeatures {
                d_rel: config.d_rel,
            });
        }

        let rel_extent = proj.len() / config.d_rel;
        Ok(Self {
            proj: RefCell::new(device.buffer(&by_distance(proj, config.d_rel, rel_extent))?),
            span: RefCell::new(KeyValues::new(device, config.kv_heads, config.head_dim)?),
            rel_extent,
            pinned: Cell::new(None),
            device,
            attention,
            config,
        })
    }

    /// The shape the layer was wrapped for.
    pub fn config(&self) -> AttentionConfig {
        self.config
    }

    /// How far back the learned band reaches, which is the projection's own
    /// width.
    pub fn rel_extent(&self) -> usize {
        self.rel_extent
    }

    /// The step submitted on its own, `[queries, heads * head_dim]` out — the
    /// layout `o_proj` reads.
    ///
    /// What a caller with nothing to batch it against wants, and what the cases
    /// here drive. The layer reaches for [`LayerAttention::encode`], because the
    /// projection that consumes this is a dispatch that could have been in the
    /// same command buffer.
    pub fn forward(&self, step: Step<'_>) -> Result<Vec<f32>, MetalError> {
        let mut batch = self.device.batch()?;
        let out = self.encode(&mut batch, step)?;
        batch.wait()?;
        Ok(profile::timed(Op::Readback, || out.to_vec()))
    }

    /// The same step encoded into `batch`, with the result left on the device
    /// for `o_proj` to read.
    ///
    /// **Merged on the way out.** The kernel writes `[queries, heads *
    /// head_dim]` rather than the `[heads, queries, head_dim]` it reads, which
    /// is [`merge_heads`](inkling_core::merge_heads) done by choosing an output
    /// index — so the transpose between the attention step and `o_proj` is not
    /// an operation anything performs.
    ///
    /// The keys and values are copied over for the call, which is what a caller
    /// holding the whole span in its own memory has to do. A layer that let this
    /// one keep them encodes the step against the span in place instead, which
    /// is the path a decode step takes.
    pub fn encode(&self, batch: &mut Batch<'_>, step: Step<'_>) -> Result<Buffer<f32>, MetalError> {
        let _timed = profile::scope(Op::Encode);
        let span = self.config.kv_channels();
        assert_eq!(
            step.k.len() % span,
            0,
            "{} key values are not whole keys of {span}",
            step.k.len()
        );
        assert_eq!(step.v.len(), step.k.len(), "values against keys");
        let keys = step.k.len() / span;

        let mut q = self.device.buffer(step.q)?;
        let mut k = self.device.buffer(step.k)?;
        let mut v = self.device.buffer(step.v)?;
        let mut rel = self.device.buffer(step.rel)?;
        self.encoding(
            batch,
            Encoding {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                rel: &mut rel,
                keys,
                key_stride: keys,
                taus: step.taus,
                q_offset: step.q_offset,
            },
        )
    }

    /// The step over a query and a relative-feature row a dispatch already left
    /// on the device, against the keys and values this layer is holding.
    ///
    /// **The other half of the residency**, and the half that grows with the
    /// context: [`LayerAttention::encode`] above copies the whole cached span
    /// over on every call, which at 16 keys is 53 µs a layer and at 1024 keys is
    /// two megabytes. Here the span is where it was left, and what the call
    /// carries is where in it the sequence's keys stop.
    ///
    /// The span is handed in rather than taken from the cell, because the
    /// dispatches that *wrote* this call's keys are in the same command buffer
    /// and hold it too — see [`KeyValues::landings`]. One borrow for the layer's
    /// whole encoding is what lets a key be written and read without leaving the
    /// device.
    pub(crate) fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        span: &mut KeyValues,
        q: &mut Buffer<f32>,
        rel: &mut Buffer<f32>,
        taus: Option<&[f32]>,
        q_offset: usize,
    ) -> Result<Buffer<f32>, MetalError> {
        let _timed = profile::scope(Op::Encode);
        let (keys, key_stride) = (span.held, span.capacity);
        self.encoding(
            batch,
            Encoding {
                q,
                k: &mut span.keys,
                v: &mut span.values,
                rel,
                keys,
                key_stride,
                taus,
                q_offset,
            },
        )
    }

    /// The span this layer is holding, borrowed for the whole of a call.
    pub(crate) fn span(&self) -> RefMut<'_, KeyValues> {
        self.span.borrow_mut()
    }

    /// The dispatch itself, which is the same whichever of the two put the
    /// buffers where they are.
    fn encoding(
        &self,
        batch: &mut Batch<'_>,
        step: Encoding<'_>,
    ) -> Result<Buffer<f32>, MetalError> {
        let (heads, kv_heads) = (self.config.heads, self.config.kv_heads);
        let head_dim = self.config.head_dim;

        let stride = heads * head_dim;
        assert_eq!(
            step.q.len() % stride,
            0,
            "{} query values are not whole calls of {stride}",
            step.q.len()
        );
        let queries = step.q.len() / stride;
        let taus = match step.taus {
            Some(taus) => {
                assert_eq!(taus.len(), queries, "a tau a query");
                Cow::Borrowed(taus)
            }
            None => Cow::Owned(vec![1.0; queries]),
        };
        assert_eq!(
            step.rel.len(),
            queries * heads * self.config.d_rel,
            "{} relative features are not {queries} rows of {heads} heads",
            step.rel.len()
        );
        // **Every query's own key is one of the keys**, which the module
        // documentation states and which nothing checked while the kernel scored
        // the whole span: a query sitting past the last key used to attend over
        // all of them, which is wrong quietly. The kernel bounds its loop by that
        // position now, so the same call would attend over the tail of the span
        // or over nothing at all — and a row of zeroes is wrong more quietly
        // still. Refused here, where the shape is known.
        assert!(
            step.q_offset + queries <= step.keys,
            "{queries} queries from {} sit past the {} keys of the call",
            step.q_offset,
            step.keys
        );

        let pairs = heads * queries;
        let splits = self
            .pinned
            .get()
            .unwrap_or_else(|| splits_for(pairs, step.keys, head_dim));
        let fields = [
            extent(heads, "the heads of a layer"),
            extent(kv_heads, "the KV heads of a layer"),
            extent(head_dim, "the channels of a head"),
            extent(queries, "the queries of a call"),
            extent(step.keys, "the keys of a call"),
            extent(step.key_stride, "the keys a span has room for"),
            extent(step.q_offset, "the offset of a call"),
            extent(self.config.d_rel, "the relative features of a layer"),
            extent(self.rel_extent, "the band of a layer"),
            extent(self.config.sliding, "the window of a layer"),
            extent(splits, "the splits of a call"),
            (1.0 / head_dim as f32).to_bits(),
        ];
        let mut shape = self.device.inline(&fields)?;
        let mut scaled = self.device.inline(&taus)?;
        let mut proj = self.proj.borrow_mut();
        let mut out = self.device.zeroed::<f32>(queries * heads * head_dim)?;
        // What each split leaves for the fold: its weighted sum, its peak and
        // its total. An unsplit call writes its row straight out and this is the
        // smallest allocation the device will make rather than the buffer it
        // does not need — so a prefill allocates what it always did to within a
        // float.
        let partial = head_dim + 2;
        let mut partials = self.device.zeroed::<f32>(match splits {
            1 => 1,
            splits => pairs * splits * partial,
        })?;

        // A threadgroup to each split of each query of each head, which is what
        // makes the pair the threadgroup's own position and so what makes the
        // barriers below uniform: a threadgroup either runs a split or returns
        // from one, and never splits over the question.
        let grid = Grid::new(pairs * splits * THREADS_PER_GROUP, THREADS_PER_GROUP);
        // **The spans are bound whole and walked a query row at a time**, which
        // is the difference between what this dispatch binds and what it moves:
        // a layer's keys and values have room for a thousand, an eight-token
        // sequence has eight, and a call of `n` rows reads the ones each of them
        // reaches. So the keys are charged per *query head* rather than per KV
        // head — four query heads share a span here and each threadgroup walks
        // it for itself, and a figure that charged the span once would be the
        // distinct bytes rather than the reads. The band is the same distinction
        // again — the projection is `[rel_extent, d_rel]` and a call reaches the
        // distances its keys span. Beside them the queries in and the same shape
        // out, the relative features, and a scale for each query.
        //
        // A split call writes partials where an unsplit one writes the row, and
        // the fold below is charged reading them — so the pair of dispatches is
        // charged the row once between them however many splits made it. The
        // walk itself is charged the same whichever way it is cut: the splits
        // partition `[0, keys)`, so between them they hold each live key once.
        let row = out.len();
        let staged = pairs * splits * partial;
        let walked = keys_a_call_walks(
            queries,
            step.q_offset,
            step.keys,
            self.config.sliding,
            tile_keys(head_dim),
        );
        let moves = size_of::<f32>()
            * (step.q.len()
                + if splits == 1 { row } else { staged }
                + 2 * heads * walked * head_dim
                + step.rel.len()
                + step.keys.min(self.rel_extent) * self.config.d_rel
                + queries);
        batch.add(
            self.attention.on(self.config.sliding, splits),
            &[
                shape.arg(),
                step.q.arg(),
                step.k.arg(),
                step.v.arg(),
                step.rel.arg(),
                proj.arg(),
                scaled.arg(),
                out.arg(),
                partials.arg(),
            ],
            grid,
            moves,
        )?;
        if splits > 1 {
            // A thread to a channel, which is the width the fold's own loop
            // wants and is what the weighting phase above gives each thread.
            let width = head_dim.next_multiple_of(NARROWEST_SIMD);
            batch.add(
                &self.attention.combine,
                &[shape.arg(), partials.arg(), out.arg()],
                Grid::new(pairs * width, width),
                size_of::<f32>() * (staged + row),
            )?;
        }
        Ok(out)
    }

    /// How many keys this layer is holding, which says which sequence's they
    /// are.
    pub fn held(&self) -> usize {
        self.span.borrow().held
    }

    /// Cut every call's span into `splits`, or `None` to leave it to
    /// [`splits_for`] — see [`LayerAttention::pinned`].
    ///
    /// Only the sweep sets it, so only the sweep's build has it: what a release
    /// build reads is the `None` the constructor put there, and a caller that
    /// could pin a split count is a caller that could take the predicate off the
    /// one path this measures.
    #[cfg(test)]
    pub(crate) fn split_into(&self, splits: Option<usize>) {
        self.pinned.set(splits);
    }

    /// What this layer's keys and values occupy on the device.
    ///
    /// **A windowed layer is charged the same as a global one here and that is
    /// the finding rather than the interface.** [`KeyValues`] is allocated
    /// against the keys a sequence has seen and nothing in it consults
    /// [`AttentionConfig::sliding`], so a layer that can only ever attend over
    /// its last 512 keys retains all of them — see
    /// `what_a_context_costs_in_keys_and_values`, which is where that is
    /// weighed against the window.
    pub fn span_bytes(&self) -> u64 {
        self.span.borrow().bytes()
    }

    /// Take the span for a sequence that has seen `keys` keys, with room for
    /// `queries` more.
    ///
    /// **This is where one span serving two sequences is refused.** A sequence
    /// that has seen nothing is one starting, and the span is emptied for it; a
    /// sequence that has seen keys the span is not holding is a second sequence
    /// interleaved through the same layer, which would otherwise read the
    /// first's keys under its own offset and answer plausibly.
    pub fn hold(&self, keys: usize, queries: usize) -> Result<(), MetalError> {
        let mut span = self.span.borrow_mut();
        if keys == 0 {
            span.restart();
        }
        assert_eq!(
            span.held, keys,
            "a sequence at {keys} keys against a span holding {}",
            span.held
        );
        span.reserve(self.device, keys + queries)
    }

    /// Append `[rows, kv_heads * head_dim]` keys and values to the span.
    pub fn append(&self, k: &[f32], v: &[f32]) {
        self.span.borrow_mut().append(k, v);
    }

    /// Unwant the last `rows` keys and values of the span this layer holds.
    ///
    /// The convolution windows those keys came through are
    /// [`LayerProjections`](crate::LayerProjections)'s to take back, and the
    /// two have to move together: see [`crate::LayerConv::rewind`].
    pub fn rewind(&self, rows: usize) {
        self.span.borrow_mut().rewind(rows);
    }
}

/// What the dispatch reads, once both callers have put it where it is.
struct Encoding<'a> {
    q: &'a mut Buffer<f32>,
    k: &'a mut Buffer<f32>,
    v: &'a mut Buffer<f32>,
    rel: &'a mut Buffer<f32>,
    /// Keys the sequence has, which is the kernel's loop bound.
    keys: usize,
    /// Key slots a head has room for, which is the stride between two heads.
    key_stride: usize,
    taus: Option<&'a [f32]>,
    q_offset: usize,
}

/// The checkpoint's `[d_rel, rel_extent]` projection as `[rel_extent, d_rel]`,
/// which is a distance's coefficients gathered where the kernel reads them.
fn by_distance(proj: &[f32], d_rel: usize, rel_extent: usize) -> Vec<f32> {
    (0..rel_extent)
        .flat_map(|back| (0..d_rel).map(move |c| proj[c * rel_extent + back]))
        .collect()
}

/// The kernel, with the constants its threadgroup arrays and its masked entry
/// rest on written into the prelude rather than spelled twice.
///
/// [`MASKED`] is [`inkling_core::mask`]'s, because the constant is a fact about
/// the reference rather than about this kernel — the CPU path emits it and the
/// committed masks hold the bfloat16 rounding of it, and a second spelling here
/// is one that can drift from the module that owns it.
fn source(staged: usize, residency: usize) -> String {
    format!(
        "constant uint MOST_SIMDGROUPS = {MOST_SIMDGROUPS};\n\
         constant uint MOST_CHANNELS = {MOST_CHANNELS};\n\
         constant uint MOST_FEATURES = {MOST_FEATURES};\n\
         constant uint MOST_SPLITS = {MOST_SPLITS};\n\
         constant uint KEYS_PER_SIMD = {KEYS_PER_SIMD};\n\
         constant uint TILED_VALUES = {TILED_VALUES};\n\
         constant uint STAGED_VALUES = {staged};\n\
         constant uint RESIDENCY = {residency};\n\
         constant float MASKED = {MASKED:e}f;\n{BODY}"
    )
}

/// Everything of the kernel that those constants do not decide.
///
/// The logit scale arrives as the bits of a float in a `uint` field rather than
/// as a float, because the ten scalars are one buffer and a struct mixing the
/// two types is a layout the Rust side and the source would each have to get
/// right independently.
const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint heads;
    uint kv_heads;
    uint head_dim;
    uint queries;
    uint keys;
    uint key_stride;
    uint q_offset;
    uint d_rel;
    uint rel_extent;
    uint sliding;
    uint splits;
    uint scale_bits;
};

/// One query-key pair's entry of the banded relative-position mask, from the
/// backward distance alone.
///
/// **The four cases are ordered and the order is the whole of it.** A key ahead
/// of the query is masked before anything indexes the projection with a negative
/// distance; a key past the window is masked whether or not the band still
/// covers it; what is left inside the band is the learned bias, and what is left
/// outside it is exactly zero — neither masked nor biased, which is a case only
/// a global layer reaches, its window being zero.
///
/// `tau` multiplies the bias and not the mask. Scaling `-1e30` would overflow,
/// and it rules a key out at any magnitude — so log scaling reaches the entries
/// that carry a number and leaves the ones that carry a decision.
///
/// `proj` is `[rel_extent, d_rel]` — the checkpoint's rows gathered by distance
/// — so the `d_rel` coefficients this reads are consecutive, and the products
/// are summed in the order `inkling_core::mask` sums them, by one lane rather
/// than across a simdgroup. Sixteen multiplies over one cache line is not work
/// worth a reduction, and matching the CPU's association costs nothing to
/// arrange.
inline float banded_entry(
    device const float *proj,
    threadgroup const float *features,
    constant Shape &shape,
    int dist,
    float tau
) {
    if (dist < 0) {
        return MASKED;
    }
    const uint back = (uint)dist;
    if (shape.sliding > 0 && back >= shape.sliding) {
        return MASKED;
    }
    if (back >= shape.rel_extent) {
        return 0.0f;
    }

    device const float *coefficients = proj + (ulong)back * shape.d_rel;
    float bias = 0.0f;
    for (uint c = 0; c < shape.d_rel; ++c) {
        bias += features[c] * coefficients[c];
    }
    return bias * tau;
}

/// The attention step for one query of one head: score every key, bias it, mask
/// it, softmax the row and weight the values by it.
///
/// One threadgroup to a query of a head. The keys are walked in tiles of
/// `simdgroups * KEYS_PER_SIMD`, and each tile is two phases with the barriers
/// between them:
///
/// - **Score.** A simdgroup to a key: lane `l` walks the key's channels from `l`
///   in strides of the simdgroup width and the group sums what the lanes held,
///   so the 32 lanes of one reduction read 32 consecutive channels. Lane 0 adds
///   the band's entry, which it derives rather than reads.
/// - **Weight.** A thread to a channel: the tile's exponents are formed once in
///   threadgroup memory, and each thread carries its own channel's weighted sum
///   across every tile. As many of the tile's values as `STAGED_VALUES` allows
///   are brought in first, by the whole threadgroup and in the order they lie,
///   so that what a thread then walks is threadgroup memory rather than a chain
///   of `head_dim`-spaced device reads; the rest are read where they lie.
///
/// The running peak and total are every thread's rather than one thread's and
/// broadcast: each reduces the same tile out of the same threadgroup memory, and
/// what that costs at 32 entries is less than the barrier a broadcast needs.
kernel void fused_attention(
    constant Shape &shape [[buffer(0)]],
    device const float *q [[buffer(1)]],
    device const float *k [[buffer(2)]],
    device const float *v [[buffer(3)]],
    device const float *rel [[buffer(4)]],
    device const float *proj [[buffer(5)]],
    device const float *taus [[buffer(6)]],
    device float *out [[buffer(7)]],
    device float *partials [[buffer(8)]],
    uint slot [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]]
) {
    threadgroup float query[MOST_CHANNELS];
    threadgroup float weighted[MOST_CHANNELS];
    threadgroup float features[MOST_FEATURES];
    threadgroup float scores[MOST_SIMDGROUPS * KEYS_PER_SIMD];

    // **One of these two is a tile of values and the other is a declaration,
    // and which is which is the whole of what separates the two entries.** Both
    // sizes are compile-time constants, so the entry that stages nothing folds
    // every line below that mentions `staged` away and the entry that stages a
    // tile folds away every line that mentions `residency`.
    threadgroup float staged[STAGED_VALUES];
    threadgroup float residency[RESIDENCY];
    const bool stages = STAGED_VALUES > 1u;

    // A threadgroup is one split of one query of one head, split-minor — so the
    // splits of a pair are consecutive threadgroups, which is the order they are
    // written in and the order the combine reads them in.
    const uint pair = slot / shape.splits;
    const uint split = slot % shape.splits;

    // Unreachable under the grid this is dispatched over, which gives exactly
    // one threadgroup to each split of each query of each head. It is here for
    // what it would have to be if that ever stopped being true: the pair is the
    // threadgroup's own position, so this turns away a whole group and never
    // splits one — and a bounds check on `local` instead would leave some
    // threads at the barriers below and others past them, which is undefined
    // rather than slow.
    if (pair >= shape.heads * shape.queries) {
        return;
    }

    const uint head = pair / shape.queries;
    const uint i = pair % shape.queries;
    // Each KV head serves a contiguous block of query heads: with 32 query heads
    // over 8 KV heads, query heads 0..4 all read KV head 0. Striding instead —
    // `head % kv_heads` — pairs every query head with keys of the right shape.
    const uint kv = head / (shape.heads / shape.kv_heads);
    const float scale = as_type<float>(shape.scale_bits);
    const float tau = taus[i];

    // `key_stride` rather than `keys`: a span the layer keeps between steps has
    // room for more keys than the sequence has put in it, so what separates one
    // KV head's keys from the next is the slots allocated and not the slots
    // filled. A call handed the span whole passes the same number twice.
    device const float *q_row = q + (ulong)pair * shape.head_dim;
    device const float *keys_of = k + (ulong)kv * shape.key_stride * shape.head_dim;
    device const float *values_of = v + (ulong)kv * shape.key_stride * shape.head_dim;
    device const float *rel_row = rel + ((ulong)i * shape.heads + head) * shape.d_rel;

    // The residency is read for the zero it was filled with, and by the thread
    // that filled it, so nothing here needs a barrier. **What says the array
    // survived the compiler is not this loop** — a store of a constant followed
    // by a load of it is what a forwarding pass exists to remove — but
    // `an_unsplit_call_gets_the_entry_its_occupancy_turn_rests_on`, which reads
    // the bytes the pipeline reports.
    if (!stages) {
        for (uint at = local; at < MOST_CHANNELS; at += threads) {
            residency[at] = 0.0f;
        }
    }
    for (uint d = local; d < shape.head_dim; d += threads) {
        query[d] = q_row[d];
        weighted[d] = stages ? 0.0f : residency[d];
    }
    for (uint c = local; c < shape.d_rel; c += threads) {
        features[c] = rel_row[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int position = (int)(i + shape.q_offset);
    // **The tile does not move between the entries.** TILED_VALUES bounds it in
    // both, so the two walk the same keys in the same tiles and differ in where
    // a value is read from and in nothing else.
    const uint tile = min(simds * KEYS_PER_SIMD, TILED_VALUES / shape.head_dim);
    float peak = -INFINITY;
    float total = 0.0f;

    // BOUND: the keys this query can reach, which is where two of
    // `banded_entry`'s four branches are answered by not walking rather than by
    // walking and discarding. Nothing at or after the query's own position is
    // causal, and on a windowed layer nothing further back than the window is in
    // it — the same two comparisons the entry makes, made once for the row
    // instead of once for every key of it.
    //
    // **The start is rounded down to a tile so that this is the same arithmetic
    // and not merely the same answer.** A tile's softmax takes a maximum over
    // what the tile holds and rescales what came before by it, so tiles cut in
    // different places accumulate in a different order and land a few ulps
    // apart. Aligned, every tile this walks holds the keys it held when the loop
    // began at zero, and the tiles it skips contributed exactly zero to that
    // accumulation: the ones before rescale by `exp(-1e30 - peak)` and the ones
    // after add `exp(-1e30 - peak)`, which underflow to a zero that is exact.
    // So the bounded kernel is the unbounded one bit for bit, which is what
    // `the_bounded_loop_is_the_unbounded_one_bit_for_bit` asserts and what makes
    // it worth asserting. It costs at most one tile of keys already inside the
    // window.
    //
    // The branches stay, and the tests that pin them drive the loop unbounded —
    // which this is measured against, so they pin what it is measured against.
    const uint last = min(shape.keys, (uint)position + 1u);
    const uint reach = shape.sliding > 0 && (uint)position >= shape.sliding
        ? ((uint)position - (shape.sliding - 1u)) / tile * tile
        : 0u;

    // SPLIT: this threadgroup's share of the walk, in whole tiles.
    //
    // **Cut out of the whole span rather than out of the live range, and that
    // is what keeps the bound above bit-exact.** A split of `[reach, last)`
    // would put its boundaries somewhere else the moment the bound moved them,
    // so a kernel with the bound taken off would accumulate over different
    // tiles and the claim that walking a masked key is the same as not walking
    // it could no longer be made on the bits. Cut out of `[0, keys)` the
    // boundaries are the same either way: every split walks the tiles it would
    // have walked, and a split with no live key in it contributes a peak of
    // `-1e30` or `-INFINITY` that the combine rescales by an exact zero.
    //
    // A split is at least one tile, so `per_split * tile` never cuts a tile in
    // half and `from` stays on the alignment `reach` already has.
    const uint spans = (shape.keys + tile - 1u) / tile;
    const uint per_split = (spans + shape.splits - 1u) / shape.splits;
    const uint from = max(reach, split * per_split * tile);
    const uint to = min(last, min(shape.keys, (split + 1u) * per_split * tile));

    for (uint first = from; first < to; first += tile) {
        const uint held = min(tile, to - first);

        device const float *tiled = values_of + (ulong)first * shape.head_dim;

        // The tile's values, brought in by the whole threadgroup in the order
        // they lie rather than by each thread down its own channel.
        if (stages) {
            for (uint at = local; at < held * shape.head_dim; at += threads) {
                staged[at] = tiled[at];
            }
        }

        for (uint n = 0; n < KEYS_PER_SIMD; ++n) {
            const uint s = n * simds + simd;
            if (s >= held) {
                continue;
            }
            const uint j = first + s;

            float dot = 0.0f;
            device const float *key = keys_of + (ulong)j * shape.head_dim;
            for (uint d = lane; d < shape.head_dim; d += width) {
                dot += query[d] * key[d];
            }
            dot = simd_sum(dot);
            if (lane == 0) {
                const float entry =
                    banded_entry(proj, features, shape, position - (int)j, tau);
                scores[s] = dot * scale + entry;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float top = peak;
        for (uint s = 0; s < held; ++s) {
            top = fmax(top, scores[s]);
        }
        // A tile whose largest score is below the running peak rescales by one
        // and a first tile rescales what is not there yet by `exp(-INFINITY)`,
        // which is zero. Neither is a case worth branching on.
        //
        // `precise` for the reason the RMSNorm kernel's `rsqrt` is: the default
        // is a hardware approximation, and every weight this row hands a value
        // comes out of one of these.
        const float rescale = precise::exp(peak - top);
        peak = top;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = local; s < held; s += threads) {
            scores[s] = precise::exp(scores[s] - top);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sum = 0.0f;
        for (uint s = 0; s < held; ++s) {
            sum += scores[s];
        }
        total = total * rescale + sum;

        // **The same keys in the same order out of two memories**, which is what
        // keeps the two entries bit-identical: a value is the same float
        // whichever one it was read from, and each thread meets its tile's in
        // the order it always met them. The walk is written twice rather than
        // once around a choice because a threadgroup pointer and a device
        // pointer are different types and nothing can name both.
        for (uint d = local; d < shape.head_dim; d += threads) {
            float acc = weighted[d] * rescale;
            if (stages) {
                for (uint s = 0; s < held; ++s) {
                    acc += scores[s] * staged[s * shape.head_dim + d];
                }
            } else {
                for (uint s = 0; s < held; ++s) {
                    acc += scores[s] * tiled[s * shape.head_dim + d];
                }
            }
            weighted[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // **One split is the whole walk, and it answers where it always did.** The
    // partials and the dispatch that folds them are what a split call needs and
    // what an unsplit one would only pay for, so the branch is here rather than
    // in a second kernel: at one split this is the same arithmetic over the same
    // tiles writing the same row, which is what lets a prefill go on being the
    // dispatch it was.
    //
    // `[queries, heads * head_dim]` rather than the `[heads, queries, head_dim]`
    // read above: the merge `o_proj` needs is an output index here rather than a
    // pass over a tensor.
    //
    // A call over no keys leaves a total of zero, which is a forward pass over
    // no tokens rather than a row to divide by it.
    if (shape.splits == 1u) {
        device float *result = out + ((ulong)i * shape.heads + head) * shape.head_dim;
        const float norm = total > 0.0f ? 1.0f / total : 0.0f;
        for (uint d = local; d < shape.head_dim; d += threads) {
            result[d] = weighted[d] * norm;
        }
        return;
    }

    // What this split reached, unnormalised and beside the peak it is relative
    // to — which is what the combine needs and what a normalised row could not
    // give it, since two splits' totals are not comparable until they have been
    // shifted onto one peak.
    device float *part = partials + (ulong)slot * (shape.head_dim + 2u);
    for (uint d = local; d < shape.head_dim; d += threads) {
        part[d] = weighted[d];
    }
    if (local == 0) {
        part[shape.head_dim] = peak;
        part[shape.head_dim + 1u] = total;
    }
}

/// One query's splits folded back into the row the unsplit kernel would have
/// written.
///
/// A threadgroup to a query of a head, reading the `splits` partials that pair
/// left. The fold is the same streaming softmax the tile loop above runs, at the
/// grain of a split rather than of a tile: take the largest peak, rescale every
/// split's total and weighted sum onto it, and normalise by what they come to.
///
/// **A split with no live key in it is rescaled to nothing rather than branched
/// around.** Its peak is `-INFINITY` where the loop above never entered, or
/// `-1e30` where it walked masked tiles only, and `exp` of either against a real
/// peak underflows to a zero that is exact — which is what lets the bound above
/// stay bit-exact against a kernel that walks keys this one skips.
kernel void attention_combine(
    constant Shape &shape [[buffer(0)]],
    device const float *partials [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint pair [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float rescale[MOST_SPLITS];

    if (pair >= shape.heads * shape.queries) {
        return;
    }
    const uint head = pair / shape.queries;
    const uint i = pair % shape.queries;
    const uint stride = shape.head_dim + 2u;
    device const float *base = partials + (ulong)pair * shape.splits * stride;
    device float *result = out + ((ulong)i * shape.heads + head) * shape.head_dim;

    // Every thread reduces the same handful of entries rather than one reducing
    // and broadcasting, which is what the tile loop above does with its scores
    // and for the same reason: at this many entries that costs less than the
    // barrier a broadcast needs.
    float peak = -INFINITY;
    for (uint s = 0; s < shape.splits; ++s) {
        peak = fmax(peak, base[s * stride + shape.head_dim]);
    }
    // A row no split reached at all, which the grid cannot produce — every query
    // can see its own key — and which would otherwise be `exp(-inf - -inf)`.
    if (!(peak > -INFINITY)) {
        for (uint d = local; d < shape.head_dim; d += threads) {
            result[d] = 0.0f;
        }
        return;
    }

    for (uint s = local; s < shape.splits; s += threads) {
        rescale[s] = precise::exp(base[s * stride + shape.head_dim] - peak);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint s = 0; s < shape.splits; ++s) {
        total += base[s * stride + shape.head_dim + 1u] * rescale[s];
    }
    const float norm = total > 0.0f ? 1.0f / total : 0.0f;

    for (uint d = local; d < shape.head_dim; d += threads) {
        float acc = 0.0f;
        for (uint s = 0; s < shape.splits; ++s) {
            acc += base[s * stride + d] * rescale[s];
        }
        result[d] = acc * norm;
    }
}
"#;

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use inkling_core::attention::LogScaling;
    use inkling_core::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, LONG_ACTIVATIONS, deviation};
    use inkling_core::{
        AttentionStep, BandedMask, Checkpoint, MASKED, Sdpa, merge_heads, split_heads,
    };

    use super::*;
    use crate::testing::device;

    /// The synthetic cases and the trained projections `inkling_core::mask` is
    /// pinned to, which is what puts the branches this kernel derives beside the
    /// tensor mlx-vlm wrote for them.
    const MASK_FIXTURE: &str = "mask.safetensors";

    /// The kernel a call that fills the machine runs, which is the arm every
    /// mutation here is cut from and the one both sweeps are taken over.
    ///
    /// The other entry is the same string with one constant in it and is
    /// reached through [`staging_the_tiles_values`], which is what
    /// `a_split_call_stages_a_whole_tile_and_answers_the_same_bits` holds
    /// against this.
    fn source() -> String {
        super::source(STAGED_BY_AN_UNSPLIT_CALL, RESIDENCY)
    }

    /// The kernel with its loop bound taken off, which scores every key of the
    /// span and lets the softmax discard the ones the band masks.
    ///
    /// **This is the kernel this one is measured against, and the reason three
    /// cases below drive it rather than the shipped source.** Bounding the loop
    /// makes a masked key one the dispatch never visits, so a mutation to the
    /// branch that would have masked it stops changing any answer — and a case
    /// asserting that it does would go on passing while proving nothing. What
    /// those cases pin is `banded_entry`, which is still the authority on which
    /// keys are live; what pins the bound to it is
    /// `the_bounded_loop_is_the_unbounded_one_bit_for_bit`.
    fn unbounded() -> String {
        let source = source();
        let (head, tail) = source
            .split_once("    // BOUND:")
            .expect("the bound is marked");
        // **The cut ends where the split begins, and the split is kept.** What
        // this mutant exists to be is the same walk over every key of the span,
        // which is the *bound* removed and not the parallelism — and the split
        // partitions `[0, keys)` on tile boundaries either way, so a mutant that
        // dropped it would compare two different accumulations and the bit-for-
        // bit claim below would be about the wrong thing.
        let (_, tail) = tail
            .split_once("\n    // SPLIT:")
            .expect("the bound ends where the split begins");
        let whole = format!(
            "{head}    const uint last = shape.keys;\n    const uint reach = 0u;\n\n    // SPLIT:{tail}"
        );
        // What is asserted is that the bound is *gone*, not merely that the
        // string moved: cutting between two markers is only as good as the
        // markers, and a comment that came to hold the loop's own header would
        // shift the cut somewhere a length comparison would not notice.
        assert!(
            !whole.contains("shape.sliding - 1u") && !whole.contains("(uint)position + 1u"),
            "the bound survived the cut"
        );
        assert!(
            whole.contains("const uint reach = 0u;")
                && whole.contains("const uint last = shape.keys;"),
            "the loop was not put back over the whole span"
        );
        whole
    }

    /// **A call cuts its span only where the grid is short of the machine**,
    /// which is the whole of the predicate and the thing a table of timings
    /// cannot say. A decode step is 32 threadgroups on 80 cores and splits; a
    /// prefill is tens of thousands and does not, so it goes on being the one
    /// dispatch it was and pays for no fold.
    ///
    /// Here rather than left to the timings because a predicate that quietly
    /// stopped splitting a decode step would read as a regression in a number
    /// nobody was measuring that week, and one that started splitting a prefill
    /// would cost a dispatch and a buffer a layer at every length.
    #[test]
    fn a_call_splits_its_span_only_where_the_grid_is_short_of_the_machine() {
        let (heads, head_dim) = (32, 128);
        // A decode step: one query, 32 heads, and a context somebody has. Each
        // either reaches the threadgroups it wanted or ran out of tiles to cut,
        // which is the whole of the predicate — and the second case is not a
        // shortfall but where a short context is meant to land, see
        // `WANTED_GROUPS`.
        for keys in [97, 385, 769, 4096, 65536] {
            let splits = splits_for(heads, keys, head_dim);
            let tiles = keys.div_ceil(tile_keys(head_dim));
            assert!(
                splits * heads >= WANTED_GROUPS || splits == tiles.min(MOST_SPLITS),
                "{keys} keys split {splits} ways is {} threadgroups against {tiles} tiles",
                splits * heads
            );
            assert!(
                splits <= tiles,
                "{keys} keys split {splits} ways is more splits than the {tiles} tiles it has"
            );
        }
        // And a span with too few tiles to cut takes the tiles it has rather
        // than threadgroups that return on their first instruction.
        assert_eq!(splits_for(heads, 8, head_dim), 1);
        assert_eq!(splits_for(heads, 3 * tile_keys(head_dim), head_dim), 3);

        // A prefill takes no cut at any length, which is what keeps it the one
        // dispatch it was.
        for queries in [97, 385, 769] {
            assert_eq!(
                splits_for(heads * queries, 8192, head_dim),
                1,
                "{queries} queries split a span the grid already fills"
            );
        }
        // And the widest block a round can propose takes none either — not
        // because its grid fills the machine but because the cut it would get is
        // too small to spread a windowed layer's live keys, which is the floor
        // `splits_for` states and the sweep measured.
        assert_eq!(splits_for(heads * 9, 8192, head_dim), 1);
        assert!(splits_for(heads * 3, 8192, head_dim) >= LEAST_SPLIT);
    }

    /// The shape the cases below are driven at does reach the split, so that
    /// what they assert is asserted of a call that cuts its span.
    ///
    /// **Without this the coverage is accidental.** Every equivalence claim in
    /// this module — against mlx-vlm's own mask, against the CPU over the same
    /// band, and the bit-for-bit one against the unbounded loop — is made over
    /// cases whose split count is decided by a predicate none of them mentions,
    /// and a fixture retuned until they stopped splitting would leave all three
    /// passing about the unsplit kernel alone.
    #[test]
    fn the_cases_that_pin_the_answer_are_driven_through_a_split_span() {
        let decode = synthetic("decode");
        let splits = splits_for(
            decode.config.heads * decode.queries,
            decode.keys,
            decode.config.head_dim,
        );
        assert!(
            splits > 1,
            "the decode case is one split, so nothing here pins the fold"
        );
        // And the long capture is the other side of the predicate, which is what
        // says both paths are exercised rather than one of them twice.
        for (case, _) in Case::all(LONG_ACTIVATIONS).unwrap_or_default() {
            assert_eq!(
                splits_for(
                    case.config.heads * case.queries,
                    case.keys,
                    case.config.head_dim
                ),
                1,
                "{}: a 1280-query capture split a grid that already fills the machine",
                case.name
            );
        }
    }

    /// The synthetic cases, and the branches each was placed to reach — named
    /// here as `inkling_core::mask` names them, so a fixture retuned until a
    /// case stops covering its branch fails here too.
    const SYNTHETIC: [(&str, &[u8]); 5] = [
        ("sliding_window", &[1, 2, 3]),
        ("global_band", &[1, 3, 4]),
        ("decode", &[2, 3]),
        ("prefill", &[1, 3]),
        ("narrow_window", &[1, 2, 3]),
    ];

    /// The synthetic cases are float32 end to end and so are the queries, keys
    /// and values invented for them, so what separates this kernel from the CPU
    /// is summation order in two places: a `head_dim`-long dot product reduced
    /// across a simdgroup where the CPU sums it serially, and a softmax
    /// accumulated tile by tile where the CPU shifts a written-down row by its
    /// peak. The same bound, for the same reason, as `matmul`'s.
    ///
    /// Worst observed when this landed: 1.6e-6 over the five cases, a factor of
    /// twelve in hand, against a weakest mutation — the band consulted before
    /// the window — of 1.3e-1, five decades above.
    const TOLERANCE: f32 = 2e-5;

    /// The same bound over the trained captures, where what it is measured
    /// against is the CPU path running the *same* band rather than mlx-vlm.
    ///
    /// This is the number that says the kernel is right. It is looser than
    /// [`TOLERANCE`] for one reason: the long capture reduces a softmax over
    /// 1280 keys where a synthetic case reduces over 1200 of which 512 are live,
    /// and a streaming softmax rescales once a tile rather than shifting a
    /// written-down row by its peak. Worst observed when this landed: 4.2e-6, on
    /// the long capture's global layer.
    const CPU_TOLERANCE: f32 = 6e-5;

    /// And the bound against mlx-vlm's own `sdpa_out`, which is four times
    /// looser than the 6e-3 `inkling_core::attention` holds the same tensors to.
    ///
    /// **The reference's step read a mask that had been rounded to bfloat16, and
    /// this one reads no mask at all.** `inkling_core::mask` measures the
    /// computed band against the recorded one at up to 2.9e-3 of the band's
    /// largest entry, which is bfloat16's quantum and nothing else — but that
    /// error lands in a logit, and a logit's error is exponentiated before it is
    /// normalised. Handed mlx-vlm's own rounded mask the CPU reproduces
    /// `sdpa_out` to 2.8e-3; handed the band computed in float32, both paths
    /// land 7.9e-3 away, together and in the same direction.
    ///
    /// So this bound is not the kernel's accuracy. It is the reference's mask
    /// dtype, and the assertion beside it — that the CPU path over the same band
    /// is the same distance out — is what says so.
    const TRAINED_TOLERANCE: f32 = 1.2e-2;

    /// A head narrow enough to keep the synthetic cases quick and wide enough
    /// that it is not a simdgroup: 40 channels over 32 lanes leaves the first
    /// eight with two apiece and the rest with one, which is the ragged stride
    /// the dot product has to get right.
    const HEAD_DIM: usize = 40;

    /// Values spread over both signs so that a reduction cancels the way a
    /// trained one does, from an index rather than from a generator — the two
    /// paths have to be handed the same numbers and a seed is one more thing to
    /// keep in step.
    fn values(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + salt * 11) % 101) as f32 - 50.0) / 64.0)
            .collect()
    }

    /// One configuration of the step: the shapes, the layer's band and window,
    /// and everything both paths are handed.
    struct Case {
        name: String,
        config: AttentionConfig,
        proj: Vec<f32>,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        rel: Vec<f32>,
        /// `None` on the cases whose layer has no log scaling, which is every
        /// one but the case placed to reach it — the floor is 128000 tokens.
        taus: Option<Vec<f32>>,
        q_offset: usize,
        queries: usize,
        keys: usize,
        /// The mask mlx-vlm wrote for this case, where there is one. The
        /// synthetic cases carry it; the trained captures carry their own.
        mask: Option<Vec<f32>>,
    }

    impl Case {
        fn sdpa(&self) -> Sdpa {
            Sdpa::new(
                self.config.heads,
                self.config.kv_heads,
                self.config.head_dim,
            )
        }

        fn step(&self) -> Step<'_> {
            Step {
                q: &self.q,
                k: &self.k,
                v: &self.v,
                rel: &self.rel,
                taus: self.taus.as_deref(),
                q_offset: self.q_offset,
            }
        }

        /// The same call as the seam `inkling_core` states, so that what the
        /// kernel is checked against is that module's own step and not a second
        /// spelling of it living here.
        fn on_the_cpu(&self) -> Vec<f32> {
            AttentionStep {
                sdpa: self.sdpa(),
                mask: BandedMask::new(self.config.d_rel, &self.proj, self.config.sliding),
                q: &self.q,
                k: &self.k,
                v: &self.v,
                rel: &self.rel,
                taus: self.taus.as_deref(),
                q_offset: self.q_offset,
            }
            .on_the_cpu()
        }

        /// The step over a mask somebody else built, which is how mlx-vlm's own
        /// recorded tensor is put beside a band this kernel derives.
        fn through(&self, mask: &[f32]) -> Vec<f32> {
            let out = self.sdpa().forward(&self.q, &self.k, &self.v, mask);
            merge_heads(&out, self.config.heads, self.config.head_dim)
        }

        fn on_the_device(&self, device: &Device, attention: &FusedAttention) -> Vec<f32> {
            self.wrapped(device, attention)
                .forward(self.step())
                .expect("the dispatch completes")
        }

        /// The same step with the number of splits pinned rather than left to
        /// [`splits_for`], which is what a case comparing two kernels needs:
        /// unsplit is what a prefill runs and a fold is what a decode step runs,
        /// and holding the number fixed leaves the kernel as the only thing that
        /// differs between two arms.
        fn cut(&self, device: &Device, attention: &FusedAttention, splits: usize) -> Vec<f32> {
            let layer = self.wrapped(device, attention);
            layer.split_into(Some(splits));
            layer.forward(self.step()).expect("the dispatch completes")
        }

        /// What the bandwidth column divides by, against what the kernel reads:
        /// the queries in and the same shape out, the keys and values each query
        /// row of this call walks, the relative features, the band's
        /// coefficients for the distances those keys span, and a scale a query.
        ///
        /// **Both dispatches where the span is cut**, because that is what a
        /// caller encodes and what the profile sums: the split call writes
        /// partials where an unsplit one writes the row and the fold reads them
        /// back and writes the row itself, so a cut adds the partials twice. How
        /// many splits is [`splits_for`]'s and is not what this is checking.
        fn moves(&self) -> usize {
            let rel_extent = self.proj.len() / self.config.d_rel;
            let pairs = self.config.heads * self.queries;
            let splits = splits_for(pairs, self.keys, self.config.head_dim);
            let folded = match splits {
                1 => 0,
                splits => 2 * pairs * splits * (self.config.head_dim + 2),
            };
            size_of::<f32>()
                * (2 * self.q.len()
                    + 2 * self.config.heads * self.walked() * self.config.head_dim
                    + self.rel.len()
                    + self.keys.min(rel_extent) * self.config.d_rel
                    + self.queries
                    + folded)
        }

        /// The keys this call's threadgroups walk, counted one key at a time
        /// against the loop bound written out rather than differenced the way
        /// [`keys_a_call_walks`] differences it — so the two agree only where
        /// both are right, which is what `branch` is to the band.
        fn walked(&self) -> usize {
            let tile = tile_keys(self.config.head_dim);
            (0..self.queries)
                .map(|i| i + self.q_offset)
                .map(|position| {
                    (0..self.keys)
                        .filter(|&key| key <= position)
                        .filter(|&key| match self.config.sliding {
                            window if window > 0 && position >= window => {
                                key >= (position - (window - 1)) / tile * tile
                            }
                            _ => true,
                        })
                        .count()
                })
                .sum()
        }

        fn wrapped<'d>(
            &self,
            device: &'d Device,
            attention: &'d FusedAttention,
        ) -> LayerAttention<'d> {
            LayerAttention::new(device, attention, self.config, &self.proj)
                .expect("the projection uploads")
        }

        /// The synthetic cases of the mask fixture, given queries, keys and
        /// values of their own.
        ///
        /// The `prefill` case is two sequences; everything here takes one at a
        /// time, so its second is the one loaded — the batch axis is the
        /// scheduler's and a batch of sequences is a loop over these.
        fn synthetic() -> Vec<Self> {
            let ckpt = fixture::open(MASK_FIXTURE);
            SYNTHETIC
                .iter()
                .map(|(name, _)| {
                    let of = |field| fixture::tensor(&ckpt, &format!("{name}.{field}"));
                    let recorded = fixture::f32s(&of("config"));
                    let &[q_offset, keys, sliding, rel_extent] = recorded.as_slice() else {
                        panic!("{name}: config is [q_offset, keys, sliding, rel_extent]")
                    };
                    let rel = of("rel");
                    let &[batch, queries, heads, d_rel] = rel.shape() else {
                        panic!("{name}: rel is [batch, queries, heads, d_rel]")
                    };
                    let proj = fixture::f32s(&fixture::tensor(
                        &ckpt,
                        &format!("proj{}", rel_extent as usize),
                    ));

                    // The last of the case's sequences, and its own slice of
                    // both tensors that carry a batch axis.
                    let last = batch - 1;
                    let (keys, queries) = (keys as usize, queries);
                    let rel = fixture::f32s(&rel)[last * queries * heads * d_rel..].to_vec();
                    let mask = fixture::f32s(&of("mask"))[last * heads * queries * keys..].to_vec();

                    let config = AttentionConfig {
                        hidden: heads * HEAD_DIM,
                        heads,
                        // Two query heads over one KV head, so that every case
                        // runs a KV head shared by a block of them rather than
                        // one apiece.
                        kv_heads: 1,
                        head_dim: HEAD_DIM,
                        d_rel,
                        sliding: sliding as usize,
                        rms_norm_eps: 1e-6,
                        log_scaling: None,
                    };
                    let span = config.kv_heads * keys * HEAD_DIM;
                    Self {
                        name: name.to_string(),
                        q: values(heads * queries * HEAD_DIM, 1),
                        k: values(span, 2),
                        v: values(span, 3),
                        taus: None,
                        config,
                        proj,
                        rel,
                        queries,
                        keys,
                        q_offset: q_offset as usize,
                        mask: Some(mask),
                    }
                })
                .collect()
        }

        /// One captured layer's step: everything mlx-vlm handed its kernel and
        /// the band the checkpoint's own `rel_proj` builds, with no weights at
        /// all.
        fn captured(masks: &Checkpoint, activations: &Checkpoint, layer: usize) -> Self {
            let of = |name: &str| fixture::layer_tensor(activations, layer, name);
            let q = of("q_norm_out");
            let k = of("k_norm_out");
            let &[_, heads, queries, head_dim] = q.shape() else {
                panic!("q_norm_out is [batch, heads, queries, head_dim]")
            };
            let &[_, kv_heads, keys, _] = k.shape() else {
                panic!("k_norm_out is [batch, kv_heads, keys, head_dim]")
            };

            let recorded = fixture::f32s(&fixture::layer_tensor(masks, layer, "config"));
            let proj = fixture::f32s(&fixture::layer_tensor(masks, layer, "rel_proj"));
            let rel = of("r_proj_out");
            let &[_, _, _, d_rel] = rel.shape() else {
                panic!("r_proj_out is [batch, queries, heads, d_rel]")
            };

            Self {
                name: format!("layer{layer}"),
                config: AttentionConfig {
                    hidden: heads * head_dim,
                    heads,
                    kv_heads,
                    head_dim,
                    d_rel,
                    sliding: recorded[2] as usize,
                    rms_norm_eps: 1e-6,
                    log_scaling: None,
                },
                proj,
                // `v_sconv_out` is the one input the capture holds in the
                // projection's own layout: the two norms were taken as
                // attention passed them to the kernel, already split into
                // heads, and this one was taken a step earlier.
                v: split_heads(&fixture::f32s(&of("v_sconv_out")), kv_heads, head_dim),
                q: fixture::f32s(&q),
                k: fixture::f32s(&k),
                rel: fixture::f32s(&rel),
                // The floor is 128000 tokens, so no capture reaches a `tau`
                // that is not exactly 1.
                taus: None,
                queries,
                keys,
                q_offset: recorded[0] as usize,
                mask: Some(fixture::f32s(&of("mask"))),
            }
        }

        /// What mlx-vlm's own kernel produced for a captured layer, in the
        /// layout this kernel writes.
        fn recorded(activations: &Checkpoint, layer: usize, case: &Case) -> Vec<f32> {
            merge_heads(
                &fixture::f32s(&fixture::layer_tensor(activations, layer, "sdpa_out")),
                case.config.heads,
                case.config.head_dim,
            )
        }

        /// The captured layers of a bundle, or nothing where it has not been
        /// generated.
        fn all(bundle: &str) -> Option<Vec<(Self, Vec<f32>)>> {
            let masks = fixture::open(MASK_FIXTURE);
            let activations = fixture::try_open(bundle)?;
            Some(
                CAPTURED_LAYERS
                    .iter()
                    .filter(|&&layer| fixture::holds_layer(&activations, layer))
                    .map(|&layer| {
                        let case = Self::captured(&masks, &activations, layer);
                        let want = Self::recorded(&activations, layer, &case);
                        (case, want)
                    })
                    .collect(),
            )
        }
    }

    fn synthetic(name: &str) -> Case {
        Case::synthetic()
            .into_iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("no {name} case"))
    }

    /// **The claim the whole kernel rests on.** Every entry of the additive mask
    /// mlx-vlm wrote is reproduced by a kernel that never writes one, measured
    /// where it is used rather than where it is built: the same queries, keys
    /// and values through the reference's materialised mask on the CPU, and
    /// through the four branches derived per element here.
    ///
    /// The five cases between them reach all four branches, including a
    /// decode-shaped one — one query, 1200 keys, an offset of 1024 — and a
    /// global one where a key in context sits outside the band.
    #[test]
    fn the_synthetic_cases_reproduce_the_step_through_mlxs_own_mask() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let mut worst = 0.0f32;

        for case in Case::synthetic() {
            let want = case.through(case.mask.as_ref().expect("a recorded mask"));
            let deviation = deviation(&case.on_the_device(&device, &attention), &want);
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
            worst = worst.max(deviation);
        }
        eprintln!(
            "worst deviation from the step through mlx's own mask over {} cases: {worst:e}",
            SYNTHETIC.len()
        );
    }

    /// The cases reach the branches they were placed to reach, asserted on the
    /// reference's own masks. Without it every claim above holds of whichever
    /// branches happen to be live, and the mutations below would be measuring an
    /// empty set.
    #[test]
    fn a_dispatch_declares_the_keys_each_of_its_query_rows_walks() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");

        let declared = |case: &Case| -> u64 {
            let layer = case.wrapped(&device, &attention);
            layer.hold(0, case.keys).expect("the span reserves");
            let moved = crate::testing::moved(&device, |batch| {
                layer.encode(batch, case.step()).expect("the step encodes");
            });
            assert_eq!(moved as usize, case.moves(), "{}", case.name);
            moved
        };

        // The span the layer holds has room for 64 keys and this call reaches
        // sixteen. A figure taken off what was bound would be charging the step
        // for keys no sequence has yet.
        let case = decode_shaped(16);
        let over_one_row = declared(&case);
        assert!(
            (over_one_row as usize)
                < size_of::<f32>() * 2 * LEAST_KEYS * case.config.heads * case.config.head_dim,
            "{over_one_row} bytes is the whole reserved span rather than the keys a call reaches"
        );

        // **A block of rows is charged a span a row**, which is the half a
        // single-row call cannot tell apart: nine rows over sixteen keys walk 108
        // of them between them, where the causal bound leaves the first row eight.
        let over_nine_rows = declared(&blocked(16, 0, 9));
        assert!(
            over_nine_rows > 4 * over_one_row,
            "{over_nine_rows} bytes over nine rows against {over_one_row} over one"
        );

        // **And a windowed row is charged its window rather than its span**,
        // which is the other bound in the loop and the one no global case
        // reaches. Both shapes a prefill runs it at, since the window and the
        // causal bound are live together only over a block.
        let over_a_window = declared(&windowed(600, SLIDING_WINDOW));
        let over_the_span = declared(&windowed(600, 0));
        assert!(
            over_a_window < over_the_span,
            "a window of {SLIDING_WINDOW} was charged {over_a_window} bytes against the \
             {over_the_span} of the whole 600-key span"
        );
        declared(&blocked(600, SLIDING_WINDOW, 9));
    }

    /// The keys a query row walks, at the four bounds that decide it: a global
    /// row walks everything up to itself, a windowed row walks its window
    /// rounded out to a tile, a row inside its first window walks everything,
    /// and a call of `n` rows from nothing walks `n(n + 1)/2`.
    ///
    /// **Worked out by hand rather than read off the code**, because the whole
    /// use the figure is put to is dividing a bandwidth column and a formula
    /// that agreed with itself would divide it wrongly and say nothing.
    #[test]
    fn a_query_row_walks_the_keys_its_window_and_its_position_leave_it() {
        let tile = tile_keys(128);
        assert_eq!(tile, 32);

        assert_eq!(keys_walked(768, 769, 0, tile), 769);
        // 768 - 511 is 257, which rounds down to the tile at 256 — so a windowed
        // row walks its 512 keys and the part-tile the alignment leaves in.
        assert_eq!(keys_walked(768, 769, 512, tile), 769 - 256);
        assert_eq!(keys_walked(400, 769, 512, tile), 401);

        assert_eq!(keys_a_call_walks(4, 0, 4, 0, tile), 1 + 2 + 3 + 4);
        assert_eq!(
            keys_a_call_walks(2048, 0, 2048, 0, tile),
            2048 * 2049 / 2,
            "a global prefill walks the square"
        );
        // The same prompt through a window is linear in it instead, at the
        // window plus what the tile alignment leaves.
        assert!(keys_a_call_walks(2048, 0, 2048, 512, tile) < 2048 * 544);
    }

    /// What the bandwidth column divides by, against what the kernel reads.
    ///
    /// **A span is bound whole and read to the keys there are**, which is the
    /// distinction a declared figure exists to make: a layer keeps room for at
    /// least 64 keys from its first step, and an eight-token sequence has eight.
    #[test]
    fn the_cases_reach_the_branches_they_were_placed_to_reach() {
        let mut reached = Vec::new();
        for (case, want) in Case::synthetic().iter().zip(SYNTHETIC.map(|(_, b)| b)) {
            let mut branches: Vec<u8> = (0..case.queries)
                .flat_map(|i| {
                    (0..case.keys).map(move |j| (i + case.q_offset) as isize - j as isize)
                })
                .map(|dist| branch(case, dist))
                .collect();
            branches.sort_unstable();
            branches.dedup();
            assert_eq!(branches, want, "{}", case.name);
            reached.extend(branches);
        }
        reached.sort_unstable();
        reached.dedup();
        assert_eq!(reached, [1, 2, 3, 4]);
    }

    /// Which of the four cases a backward distance falls in, written out of the
    /// distance alone so it agrees with the kernel only where the kernel is
    /// right.
    fn branch(case: &Case, dist: isize) -> u8 {
        let (sliding, extent) = (case.config.sliding, case.proj.len() / case.config.d_rel);
        if dist < 0 {
            1
        } else if sliding > 0 && dist >= sliding as isize {
            2
        } else if dist < extent as isize {
            3
        } else {
            4
        }
    }

    /// The trained band, over the eight tokens the committed capture holds, from
    /// the reference's own `q_norm_out`, `k_norm_out`, `v_sconv_out` and
    /// `r_proj_out` — the whole attention step against `sdpa_out`, with no
    /// weight but `rel_proj` and no mask at all.
    #[test]
    fn the_captured_layers_reproduce_the_reference_step_without_a_mask() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let cases = Case::all(ACTIVATIONS).expect("the committed capture");
        assert_eq!(cases.len(), CAPTURED_LAYERS.len());

        let mut worst = 0.0f32;
        for (case, want) in &cases {
            worst = worst.max(agrees(case, &case.on_the_device(&device, &attention), want));
        }
        eprintln!("worst deviation from the recorded step: {worst:e}");
        assert!(
            worst > 0.0,
            "a run that matched exactly would mean bfloat16 rounding vanished"
        );
    }

    /// That `got` is the CPU's answer over the same band, and that both land the
    /// same distance from the step mlx-vlm recorded — which is the pair of
    /// claims [`TRAINED_TOLERANCE`] exists to keep apart. Returns the distance
    /// from mlx-vlm.
    fn agrees(case: &Case, got: &[f32], want: &[f32]) -> f32 {
        let ours = case.on_the_cpu();
        let (from_mlx, cpu_from_mlx) = (deviation(got, want), deviation(&ours, want));
        eprintln!(
            "{}: {from_mlx:e} from mlx, {cpu_from_mlx:e} for the CPU over the same band, {:e} \
             between the two",
            case.name,
            deviation(got, &ours),
        );

        assert!(
            deviation(got, &ours) <= CPU_TOLERANCE,
            "{}: the kernel is not the CPU's answer",
            case.name
        );
        for (what, deviation) in [("the kernel", from_mlx), ("the CPU", cpu_from_mlx)] {
            assert!(
                deviation <= TRAINED_TOLERANCE,
                "{}: {what} deviates from mlx by {deviation:e}",
                case.name
            );
        }
        from_mlx
    }

    /// The same, over the long capture, which is the only place the window cap
    /// and the far side of the band are live on trained numbers — and which is
    /// prefill shape rather than decode: 1280 queries over 1280 keys, `q_offset`
    /// zero, every query attending over its own prefix.
    ///
    /// Whole rather than cut to a tail. `inkling_core::attention` runs 64 of the
    /// 1280 queries because a scalar float32 step over all of them is 13 GMAC;
    /// this is the shape the kernel was written for, and running it whole is
    /// also what says the watchdog is nowhere near a command buffer whose loop
    /// length is the sequence.
    #[test]
    fn the_long_capture_reproduces_the_reference_step_past_the_band() {
        let Some(device) = device() else { return };
        let Some(cases) = Case::all(LONG_ACTIVATIONS) else {
            return;
        };
        assert!(
            !cases.is_empty(),
            "the long capture holds none of the captured layers"
        );
        let attention = FusedAttention::new(&device).expect("the kernel compiles");

        let mut worst = 0.0f32;
        for (case, want) in &cases {
            assert!(
                case.keys > case.proj.len() / case.config.d_rel,
                "{}: {} keys do not outrun the band",
                case.name,
                case.keys
            );
            let started = Instant::now();
            let got = case.on_the_device(&device, &attention);
            eprintln!(
                "{}: {} queries over {} keys in {:.2?}",
                case.name,
                case.queries,
                case.keys,
                started.elapsed()
            );
            worst = worst.max(agrees(case, &got, want));
        }
        assert!(
            worst > 0.0,
            "a run that matched exactly would mean bfloat16 rounding vanished"
        );
    }

    /// **The decode case, which is what the profile is about.** One query over
    /// 1200 keys at an offset of 1024, which is the only shape where the cache's
    /// offset is load-bearing — and the kernel is handed the offset rather than
    /// a mask indexed with it.
    ///
    /// Both halves are asserted, because either alone would leave the wrong
    /// impression of what dropping the offset costs. A query at position
    /// `i + offset` indexes the band at backward distances `i + offset - j`;
    /// pinned to zero, every key but the very first sits at a negative distance,
    /// which the band reads as a position that has not happened yet and rules
    /// out — so the row attends over almost nothing rather than drifting.
    #[test]
    fn a_decode_shaped_step_needs_the_caches_offset() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let mut case = synthetic("decode");
        assert_eq!((case.queries, case.keys, case.q_offset), (1, 1200, 1199));

        let want = case.through(case.mask.as_ref().expect("a recorded mask"));
        let agreed = deviation(&case.on_the_device(&device, &attention), &want);
        assert!(agreed <= TOLERANCE, "deviation {agreed:e}");

        case.q_offset = 0;
        let dropped = deviation(&case.on_the_device(&device, &attention), &want);
        assert!(dropped > TOLERANCE, "deviation {dropped:e}");
    }

    /// **The loop bound skips exactly the keys the band would have masked**, and
    /// the two kernels are not merely close: they are the same floats.
    ///
    /// The claim the bound makes is that walking a key and discarding it is the
    /// same as not walking it. Every case here has an answer already pinned to
    /// mlx-vlm's own mask elsewhere in this module, so what this adds is the
    /// stronger form — `assert_eq!` on the bits — which is available because the
    /// bound starts on a tile boundary. Anything weaker would leave "the same to
    /// six decimals" as the standard for a change whose whole argument is that it
    /// changes nothing.
    ///
    /// The trained captures are here beside the synthetic cases because they are
    /// the shapes the model runs: eight tokens on the committed capture, and 1280
    /// queries over 1280 keys on the long one, which is the only case in this
    /// crate where a sliding layer's window is actually narrower than its span
    /// and so the only one where the window half of the bound skips anything at
    /// all.
    #[test]
    fn the_bounded_loop_is_the_unbounded_one_bit_for_bit() {
        let Some(device) = device() else { return };
        let bounded = FusedAttention::new(&device).expect("the kernel compiles");
        let whole = unbounded();
        assert_ne!(whole, source(), "the bound was not taken off");
        let walking = FusedAttention::from_source(&device, &whole).expect("the mutant compiles");

        let mut cases = Case::synthetic();
        cases.extend(
            Case::all(ACTIVATIONS)
                .expect("the committed capture")
                .into_iter()
                .map(|(case, _)| case),
        );
        cases.extend(
            Case::all(LONG_ACTIVATIONS)
                .unwrap_or_default()
                .into_iter()
                .map(|(case, _)| case),
        );
        let mut skipped = 0;

        for case in &cases {
            assert_eq!(
                case.on_the_device(&device, &bounded),
                case.on_the_device(&device, &walking),
                "{}: the bound changed the answer",
                case.name
            );
            skipped += (0..case.queries)
                .flat_map(|i| {
                    (0..case.keys).map(move |j| (i + case.q_offset) as isize - j as isize)
                })
                .filter(|dist| matches!(branch(case, *dist), 1 | 2))
                .count();
        }
        eprintln!(
            "{} cases agree bit for bit, over {skipped} query-key pairs the band masks",
            cases.len()
        );
        assert!(skipped > 0, "no case here has a key the bound could skip");
    }

    /// **A value weighted where it lies is the value weighted out of threadgroup
    /// memory, bit for bit**, which is what lets the two entries be two rates
    /// and one answer.
    ///
    /// The claim is that a copy is a copy: a tile's values were brought into
    /// threadgroup memory and read back out of it, and what the weighting
    /// multiplies is the same float either way, met in the same order by the
    /// same thread — one ascending run of keys, across two loops because a
    /// threadgroup pointer and a device pointer are different types. Nothing
    /// about the tiling moves with it: [`TILED_VALUES`] bounds the tile in both,
    /// so a head of any width walks the keys a tile it walked.
    ///
    /// So this is `assert_eq!` on the bits rather than a tolerance, which is the
    /// standard a change whose whole argument is that it changes nothing has to
    /// be held to. `-0.0` and `0.0` compare equal as floats and are two
    /// different answers.
    ///
    /// **Driven unsplit and through a fold**, because the two are different
    /// paths out of the same walk and because the predicate sends each entry
    /// down one of them: unsplit writes the row and a cut writes partials that a
    /// second dispatch rescales onto one peak. Each arm is driven down both, so
    /// what is compared is the staging rather than the path.
    #[test]
    fn a_value_weighted_where_it_lies_is_a_staged_one_bit_for_bit() {
        let Some(device) = device() else { return };
        let part = FusedAttention::from_source(&device, &source()).expect("the kernel compiles");
        let whole = staging_the_tiles_values();
        assert_ne!(whole, source(), "both entries stage the same values");
        let tile = FusedAttention::from_source(&device, &whole).expect("the staged entry compiles");
        assert!(
            tile.global.threadgroup_memory() > part.global.threadgroup_memory(),
            "staging a whole tile declares no more memory than staging part of one"
        );

        let mut cases = Case::synthetic();
        cases.extend([
            blocked(2048, 0, 2048),
            blocked(2048, SLIDING_WINDOW, 2048),
            blocked(600, SLIDING_WINDOW, 13),
            blocked(97, 0, 97),
            blocked(1200, 0, 1),
            blocked(1200, SLIDING_WINDOW, 1),
        ]);
        cases.extend(
            Case::all(ACTIVATIONS)
                .expect("the committed capture")
                .into_iter()
                .map(|(case, _)| case),
        );
        cases.extend(
            Case::all(LONG_ACTIVATIONS)
                .unwrap_or_default()
                .into_iter()
                .map(|(case, _)| case),
        );

        let mut elements = 0;
        for case in &cases {
            for splits in [1, 8] {
                let want = case.cut(&device, &tile, splits);
                let got = case.cut(&device, &part, splits);
                let apart = want
                    .iter()
                    .zip(&got)
                    .position(|(want, got)| want.to_bits() != got.to_bits());
                assert_eq!(
                    apart,
                    None,
                    "{} in {splits}: reading a value where it lies answered {:?} where staging \
                     it answered {:?}",
                    case.name,
                    apart.map(|at| got[at]),
                    apart.map(|at| want[at]),
                );
                elements += want.len();
            }
        }
        eprintln!(
            "{} cases agree bit for bit over {elements} elements",
            cases.len()
        );
    }

    /// **The memory the turn rests on is memory a call gets**, which is the two
    /// things about it this side can check without a clock: that an unsplit call
    /// declares what leaves a core six threadgroups, and that the predicate
    /// hands it the entry that does.
    ///
    /// The window is the sweep's own. Six threadgroups a core is every
    /// declaration a core's 80 KiB divides into six times and not seven,
    /// measured at 11.25 KiB against 11.5 and at 13.25 against 13.5 — so a
    /// declaration inside it is on the plateau and one outside it is a step down
    /// either side. A shipped figure at either edge would be a figure a
    /// compiler's own rounding could move off the plateau.
    #[test]
    fn an_unsplit_call_gets_the_entry_its_occupancy_turn_rests_on() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let beside = size_of::<f32>()
            * (2 * MOST_CHANNELS + MOST_FEATURES + MOST_SIMDGROUPS * KEYS_PER_SIMD);
        // The one-float array each entry does not use is folded away, so what a
        // pipeline reports is the four live arrays and whichever of the two the
        // entry was compiled to hold.
        for (splits, declared) in [
            (1, RESIDENCY),
            (8, STAGED_BY_A_SPLIT_CALL),
            (64, STAGED_BY_A_SPLIT_CALL),
        ] {
            for sliding in [0, SLIDING_WINDOW] {
                assert_eq!(
                    attention.on(sliding, splits).threadgroup_memory(),
                    beside + size_of::<f32>() * declared,
                    "a call in {splits} at a window of {sliding} took the other entry"
                );
            }
        }

        let held = attention.on(0, 1).threadgroup_memory();
        let (least, most) = (11 * 1024 + 512, 13 * 1024 + 256);
        assert!(
            (least..=most).contains(&held),
            "{held} bytes is off the plateau six threadgroups a core sit on, {least}..={most}"
        );
    }

    /// **And it is tight at both ends**, which the case above cannot say: a bound
    /// that skipped nothing would pass it, and so would a bound one tile too
    /// generous.
    ///
    /// Both mutations are one key too few — the query's own key at the causal
    /// end, which is the one every row is guaranteed to have, and one key past
    /// the far edge of the window, which the `decode` case has 512 live keys
    /// behind. A bound that is right by an accident of alignment is not right
    /// here: `reach` is rounded down to a tile, so the window end is mutated by
    /// the `- 1u` that decides *which* tile rather than by a key inside one.
    #[test]
    fn a_bound_one_key_tighter_at_either_end_changes_the_answer() {
        let Some(device) = device() else { return };
        let case = synthetic("decode");
        assert!(case.config.sliding > 0, "a windowed case");
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let want = case.through(case.mask.as_ref().expect("a recorded mask"));
        assert!(
            deviation(&case.on_the_device(&device, &attention), &want) <= TOLERANCE,
            "the unmutated bound does not agree"
        );

        for (end, from, to) in [
            (
                "the causal end",
                "min(shape.keys, (uint)position + 1u)",
                "min(shape.keys, (uint)position)",
            ),
            (
                "the window end",
                "((uint)position - (shape.sliding - 1u)) / tile * tile",
                "((uint)position - (shape.sliding - 1u) + tile) / tile * tile",
            ),
        ] {
            let tighter = source().replace(from, to);
            assert_ne!(tighter, source(), "{end}: the mutation changed nothing");
            let mutant =
                FusedAttention::from_source(&device, &tighter).expect("the mutant compiles");
            let deviation = deviation(&case.on_the_device(&device, &mutant), &want);
            eprintln!("{end} of the bound one tile tighter: deviation {deviation:e}");
            assert!(deviation > TOLERANCE, "{end}: deviation {deviation:e}");
        }
    }

    /// **Branch 2 before branch 3.** The two overlap only where the window is
    /// narrower than the band, which no Inkling layer configures and which the
    /// `narrow_window` case exists to arrange: every distance from the window
    /// edge to the band edge is masked, and would be a learned bias if the
    /// kernel consulted the band first.
    ///
    /// Driven as a mutation rather than as a claim about the entries, because
    /// the entries are not a tensor anyone can look at — the two orderings are
    /// two kernels through the same plumbing, and what separates them is the
    /// answer.
    ///
    /// **Through [`unbounded`], because the shipped loop does not walk the keys
    /// this is about.** A key past the window is one the bound skips, so the
    /// ordering of the two branches that would have masked it decides nothing a
    /// dispatch computes. What it decides is the answer the bound is measured
    /// against — and this case's own `narrow_window` is one of the ten
    /// `the_bounded_loop_is_the_unbounded_one_bit_for_bit` holds the two to, so
    /// what is pinned here reaches the shipped kernel through that.
    #[test]
    fn the_window_cap_outranks_the_band_in_the_kernel() {
        let Some(device) = device() else { return };
        let case = synthetic("narrow_window");
        assert!(
            case.config.sliding > 0 && case.proj.len() / case.config.d_rel > case.config.sliding,
            "the case's window is not narrower than its band"
        );

        let source = unbounded();
        let want = case.through(case.mask.as_ref().expect("a recorded mask"));
        let attention = FusedAttention::from_source(&device, &source).expect("the kernel compiles");
        let agreed = deviation(&case.on_the_device(&device, &attention), &want);
        assert!(agreed <= TOLERANCE, "deviation {agreed:e}");

        let banded_first = source.replace(
            "    if (shape.sliding > 0 && back >= shape.sliding) {\n        return MASKED;\n    }\n\
             \x20   if (back >= shape.rel_extent) {\n        return 0.0f;\n    }\n",
            "    if (back >= shape.rel_extent) {\n        return 0.0f;\n    }\n\
             \x20   if (shape.sliding > 0 && back >= shape.sliding) {\n        return MASKED;\n    }\n",
        );
        assert_ne!(banded_first, source, "the mutation changed nothing");
        let mutant =
            FusedAttention::from_source(&device, &banded_first).expect("the mutant compiles");
        let deviation = deviation(&case.on_the_device(&device, &mutant), &want);
        eprintln!("the band consulted before the window: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// **Branch 1 before everything.** A key ahead of the query is masked
    /// whatever the band would have said about it, and it is masked *before* the
    /// band is consulted — which is what keeps a negative distance from indexing
    /// the projection.
    ///
    /// The mutation is the causal check dropped. A negative distance read as an
    /// unsigned one is enormous, so on a sliding layer it lands past the window
    /// and is masked anyway; on a *global* layer, whose window is zero, it lands
    /// outside the band and comes back as a plain zero — an unmasked key ahead
    /// of the query, which the `global_band` case is where that shows.
    ///
    /// **Through [`unbounded`]**, for the reason the case above is: a key ahead
    /// of the query is one the shipped loop stops before reaching, so what the
    /// causal branch decides is the answer the bound is measured against rather
    /// than the answer a dispatch produces — and this case's own `global_band` is
    /// one of the ten `the_bounded_loop_is_the_unbounded_one_bit_for_bit` holds
    /// the two to, so what is pinned here reaches the shipped kernel through
    /// that.
    #[test]
    fn a_key_ahead_of_the_query_is_masked_before_the_band_is_indexed() {
        let Some(device) = device() else { return };
        let case = synthetic("global_band");
        assert_eq!(
            case.config.sliding, 0,
            "a global case, whose window is zero"
        );

        let source = unbounded();
        let want = case.through(case.mask.as_ref().expect("a recorded mask"));
        let attention = FusedAttention::from_source(&device, &source).expect("the kernel compiles");
        let agreed = deviation(&case.on_the_device(&device, &attention), &want);
        assert!(agreed <= TOLERANCE, "deviation {agreed:e}");

        let uncaused = source.replace(
            "    if (dist < 0) {\n        return MASKED;\n    }\n",
            "    if (dist < -1000000000) {\n        return MASKED;\n    }\n",
        );
        assert_ne!(uncaused, source, "the mutation changed nothing");
        let mutant = FusedAttention::from_source(&device, &uncaused).expect("the mutant compiles");
        let deviation = deviation(&case.on_the_device(&device, &mutant), &want);
        eprintln!("the causal branch dropped: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// Log scaling multiplies the biases and leaves the masked entries alone,
    /// which is the one thing about `tau` this kernel decides — the queries are
    /// scaled before they are handed over.
    ///
    /// The floor is 128000 tokens, so nothing a capture can reach makes `tau`
    /// anything but 1; the case is driven with a floor low enough to fire, and
    /// against the CPU under the same one. Scaling the masked entries too would
    /// overflow rather than disagree, which is why it is not the mutation here:
    /// what is asserted is that the branch is live at all.
    #[test]
    fn log_scaling_multiplies_the_biases_and_not_the_masked_entries() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let mut case = synthetic("global_band");

        let log = LogScaling::new(4.0, 0.5);
        let taus: Vec<f32> = (0..case.queries)
            .map(|i| log.tau(i + case.q_offset))
            .collect();
        assert!(
            taus.iter().any(|tau| *tau > 1.0),
            "no query clears the floor"
        );

        let inert = case.on_the_device(&device, &attention);
        case.taus = Some(taus);
        let scaled = case.on_the_device(&device, &attention);
        assert!(
            deviation(&scaled, &inert) > TOLERANCE,
            "a tau above one changed nothing"
        );

        let deviation = deviation(&scaled, &case.on_the_cpu());
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }

    /// **Why a masked entry is a magnitude and not an infinity**, which a path
    /// that materialises the mask can be indifferent about and a streaming
    /// softmax cannot.
    ///
    /// The peak this kernel shifts by is a *tile's*, and a whole tile of masked
    /// keys is ordinary rather than exotic: the decode case is one query at
    /// position 1199 over 1200 keys through a 512-token window, so 688 keys —
    /// five tiles of the widest a threadgroup can hold, or 21 of the 32 it
    /// holds at this `head_dim` — are masked end to end before a live key
    /// appears. At `-1e30` those tiles shift by `-1e30` and weigh their values
    /// by one, which the live tiles then rescale to nothing. At `-INFINITY` they
    /// shift by `-INFINITY`, and `exp(-inf - -inf)` is a NaN that no later tile
    /// can rescale away.
    ///
    /// The mutation is that substitution, and the assertion is that the row
    /// stops being a number at all.
    ///
    /// **The bound is what takes this off the shipped kernel, and it is the
    /// second thing the bound is worth.** A tile the loop walks now always holds
    /// a live key, because the loop starts at the tile the window opens in — so
    /// no tile shifts by a masked peak and the choice of magnitude stops being
    /// load-bearing on this path. It is still load-bearing on the path this one
    /// is measured against, which is why the case stays and why it is driven
    /// through [`unbounded`]: the two kernels agree bit for bit only while the
    /// masked entries the longer one walks weigh their values by one and get
    /// rescaled to nothing, and `-INFINITY` is what turns that into a NaN. This
    /// case's own `decode` is one of the ten
    /// `the_bounded_loop_is_the_unbounded_one_bit_for_bit` holds the two to.
    #[test]
    fn a_tile_of_masked_keys_leaves_the_row_a_number() {
        let Some(device) = device() else { return };
        let case = synthetic("decode");
        let masked = (0..case.keys)
            .filter(|j| branch(&case, case.q_offset as isize - *j as isize) == 2)
            .count();
        assert!(
            masked > 5 * MOST_SIMDGROUPS * KEYS_PER_SIMD,
            "{masked} masked keys do not fill five of the widest tile a threadgroup can hold"
        );

        let infinitely = |source: String| {
            let infinite = source.replace(
                &format!("constant float MASKED = {MASKED:e}f;"),
                "constant float MASKED = -INFINITY;",
            );
            assert_ne!(infinite, source, "the mutation changed nothing");
            FusedAttention::from_source(&device, &infinite).expect("the mutant compiles")
        };

        let walking =
            FusedAttention::from_source(&device, &unbounded()).expect("the kernel compiles");
        assert!(
            case.on_the_device(&device, &walking)
                .iter()
                .all(|value| value.is_finite())
        );
        let poisoned = case.on_the_device(&device, &infinitely(unbounded()));
        assert!(
            poisoned.iter().any(|value| !value.is_finite()),
            "an infinite mask left the row finite, so this proves nothing"
        );

        // And the shipped kernel is indifferent to the same substitution, which
        // is the finding rather than a reason to stop asserting the above.
        let bounded = FusedAttention::new(&device).expect("the kernel compiles");
        assert_eq!(
            case.on_the_device(&device, &bounded),
            case.on_the_device(&device, &infinitely(source())),
            "the bounded loop walked a masked key"
        );
    }

    /// A key the band rules out contributes nothing to its query, whatever the
    /// value behind it holds — the test `inkling_core::attention` makes against
    /// a mask it was handed, made here against one the kernel derived.
    ///
    /// This is what fails if the entry is added to the softmax's output rather
    /// than to its input: post-softmax a masked key carries a weight of about
    /// `-1e30` instead of one of about zero, and the value behind it dominates
    /// the answer rather than vanishing from it.
    #[test]
    fn a_key_the_band_rules_out_cannot_reach_its_query() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let mut case = synthetic("decode");
        let head_dim = case.config.head_dim;

        let want = case.on_the_device(&device, &attention);
        let live: Vec<usize> = (0..case.keys)
            .filter(|j| branch(&case, case.q_offset as isize - *j as isize) == 3)
            .collect();
        let (first, last) = (
            *live.first().expect("a key inside the window"),
            *live.last().expect("a key inside the window"),
        );
        assert!(first > 0, "every key is inside the window");

        // Every key the window cap rules out, made enormous. The keys it lets
        // through are untouched, so an answer that moved read one it should not
        // have.
        for j in 0..first {
            case.v[j * head_dim..][..head_dim].fill(1e6);
        }
        assert_eq!(case.on_the_device(&device, &attention), want);

        // And the same value at the newest key, which the window does let
        // through — so the check above is not measuring a step that ignores its
        // values.
        case.v[last * head_dim..][..head_dim].fill(1e6);
        assert_ne!(case.on_the_device(&device, &attention), want);
    }

    /// Each KV head serves a contiguous block of query heads, and the captured
    /// layers are 32 query heads over 8 KV heads — so this is the one mistake
    /// the shapes cannot catch, stated against the same step under the striding
    /// rule with the keys and values gathered to match.
    #[test]
    fn each_kv_head_serves_a_contiguous_block_of_query_heads() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let cases = Case::all(ACTIVATIONS).expect("the committed capture");

        for (case, want) in &cases {
            let (heads, kv_heads) = (case.config.heads, case.config.kv_heads);
            let head_dim = case.config.head_dim;
            assert_eq!((heads, kv_heads), (32, 8));

            // The same keys and values with one KV head per query head, which
            // has no grouping left to get wrong — so a gather under the striding
            // rule is a step over the wrong keys and nothing else.
            let span = case.keys * head_dim;
            let gather = |kv: &[f32]| -> Vec<f32> {
                (0..heads)
                    .flat_map(|head| kv[(head % kv_heads) * span..][..span].to_vec())
                    .collect()
            };
            let strided = Case {
                name: format!("{}.strided", case.name),
                config: AttentionConfig {
                    kv_heads: heads,
                    ..case.config
                },
                k: gather(&case.k),
                v: gather(&case.v),
                proj: case.proj.clone(),
                q: case.q.clone(),
                rel: case.rel.clone(),
                taus: case.taus.clone(),
                queries: case.queries,
                keys: case.keys,
                q_offset: case.q_offset,
                mask: None,
            };

            let deviation = deviation(&strided.on_the_device(&device, &attention), want);
            assert!(
                deviation > TRAINED_TOLERANCE,
                "{}: striding deviates by {deviation:e}",
                case.name
            );
        }
    }

    /// One decode-shaped step at the checkpoint's own shape: one query over a
    /// cached span, 32 query heads over 8 KV heads of 128 channels, a 16-feature
    /// band a thousand distances wide.
    fn decode_shaped(keys: usize) -> Case {
        windowed(keys, 0)
    }

    /// The same, on a layer whose window is `sliding` — which is 35 of the
    /// checkpoint's 42 at 512, against 7 global ones this passes 0 for.
    fn windowed(keys: usize, sliding: usize) -> Case {
        blocked(keys, sliding, 1)
    }

    /// The same over a block of `queries` rows, which is what a speculative
    /// round of depth `k` verifies `k + 1` of in one pass — and is the other
    /// shape a decode-time call comes in.
    fn blocked(keys: usize, sliding: usize, queries: usize) -> Case {
        let (heads, kv_heads, head_dim, d_rel, extent) = (32, 8, 128, 16, 1024);
        Case {
            name: format!("{queries} queries over {keys} keys through a window of {sliding}"),
            config: AttentionConfig {
                hidden: heads * head_dim,
                heads,
                kv_heads,
                head_dim,
                d_rel,
                sliding,
                rms_norm_eps: 1e-6,
                log_scaling: None,
            },
            proj: values(d_rel * extent, 4),
            q: values(queries * heads * head_dim, 1),
            k: values(kv_heads * keys * head_dim, 2),
            v: values(kv_heads * keys * head_dim, 3),
            rel: values(queries * heads * d_rel, 5),
            taus: None,
            queries,
            keys,
            q_offset: keys - queries,
            mask: None,
        }
    }

    /// **A windowed layer's call and a global one's land in rows of their
    /// own**, which is what makes a prefill's two terms readable at all: 35 of
    /// the checkpoint's layers stop at a 512-key window and 7 reach every key
    /// there is, and summed into one row the linear term and the quadratic one
    /// are a number about neither.
    ///
    /// **The same pipeline behind both**, which the entry points say: two rows
    /// that appeared because the kernel had been compiled twice would be the
    /// same table and a slower engine. What [`Kernel::under`] gives is a name.
    #[test]
    fn a_windowed_layers_attention_is_charged_apart_from_a_global_ones() {
        let Some(device) = device() else { return };
        if !device.times_a_pass() {
            eprintln!("skipping: this device does not sample at a stage boundary");
            return;
        }
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        assert_eq!(
            attention.windowed.entry(),
            attention.global.entry(),
            "the two rows are two compiles rather than two names"
        );

        device
            .time_each_dispatch(true)
            .expect("the device times a dispatch");
        profile::take();
        for sliding in [SLIDING_WINDOW, 0] {
            let case = windowed(64, sliding);
            let layer = case.wrapped(&device, &attention);
            layer.forward(case.step()).expect("the dispatch completes");
        }
        device.time_each_dispatch(false).expect("sampling stops");

        let charged = profile::take();
        let rows: Vec<(&str, u64)> = charged
            .kernels()
            .iter()
            .map(|(kernel, each)| (*kernel, each.calls))
            .collect();
        assert_eq!(
            rows.iter().find(|(kernel, _)| *kernel == WINDOWED),
            Some(&(WINDOWED, 1)),
            "{rows:?}"
        );
        assert_eq!(
            rows.iter().find(|(kernel, _)| *kernel == GLOBAL),
            Some(&(GLOBAL, 1)),
            "{rows:?}"
        );
    }

    /// What the attention step costs at the shape a decode step runs it at,
    /// against what the same step costs the CPU with the mask materialised —
    /// which is the figure that says whether moving it here is worth a dispatch.
    ///
    /// **It is not worth one at the context a prompt of a few tokens leaves**,
    /// and the numbers say why: at 16 keys the whole dispatch is the submission
    /// around it, where the CPU does the same arithmetic in tens of
    /// microseconds. The kernel overtakes as the span grows, both because the
    /// CPU's own cost grows with it and because the mask the CPU builds beside
    /// it does. Nothing asserts a ratio; the numbers go to stderr for the commit
    /// message to quote.
    ///
    /// What is asserted is that no span here takes a command buffer anywhere
    /// near the watchdog, which is the constraint a kernel whose loop length is
    /// the sequence has and no kernel before it did.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn a_decode_shaped_step_costs_what_it_costs() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");

        for keys in [16, 256, 1024, 4096] {
            let case = decode_shaped(keys);
            let layer = case.wrapped(&device, &attention);

            // Warm: the first dispatch of a fresh pipeline pays for the driver's
            // first look at these buffers, which a decode loop pays once.
            for _ in 0..2 {
                layer.forward(case.step()).expect("the dispatch completes");
            }

            const CALLS: u32 = 32;
            profile::take();
            let started = Instant::now();
            for _ in 0..CALLS {
                layer.forward(case.step()).expect("the dispatch completes");
            }
            let each = started.elapsed() / CALLS;
            let spent = profile::take();

            let started = Instant::now();
            for _ in 0..CALLS {
                case.on_the_cpu();
            }
            let on_the_cpu = started.elapsed() / CALLS;

            eprintln!(
                "one query over {keys:>5} keys: {each:>9.2?} submitted on its own — {:>9.2?} the \
                 device executing, {:>9.2?} encoding the buffers — against {on_the_cpu:>9.2?} on \
                 the CPU with the mask materialised",
                spent.gpu() / CALLS,
                spent.elapsed(Op::Encode) / CALLS,
            );
            assert!(each < Duration::from_millis(20), "{keys} keys: {each:?}");
        }
    }

    /// **What the attention step costs the device at a decode step's shape, and
    /// how much of it is the dispatch rather than the step.**
    ///
    /// The case above times a submission around one call, which at these sizes is
    /// mostly the submission; the per-kernel table times it inside a step but
    /// under sampling, which charges every pass a boundary. This is the third
    /// question and the one K1 left open: the grid is 32 threadgroups — one to
    /// each query head — so what it is short of is not the norm's one core, and
    /// the empty rows say whether it is the work or the launch.
    ///
    /// The empty rows dispatch a kernel that returns on its first instruction
    /// over the same 32 threadgroups, so what separates them from the step beside
    /// them is everything the step does. The key counts then say what the loop
    /// over the span is worth: eight keys is the context every other measurement
    /// in this crate is taken at, 512 is what a sliding layer caps at, and 4096 is
    /// past anything a decode step here reaches.
    ///
    /// **The windowed rows are the shape 35 of the 42 layers run**, and they are
    /// where the loop bound is worth anything at all: a span shorter than the
    /// window has no key outside it, which is why the rows above and the decode
    /// step this engine measures elsewhere are untouched by the bound. What they
    /// are is flat — a windowed layer stops paying for context past its window,
    /// where before it paid for all of it.
    ///
    /// **A span's step and the floor under it are two measurements and are
    /// interleaved as two**, for the reason [`crate::sconv`]'s own diagnosis
    /// states: a floor always taken straight after the heavier dispatch it is
    /// subtracted from would carry whatever that dispatch left the machine in, in
    /// every round, in the same direction.
    ///
    /// Nothing asserts a duration; the numbers go to stderr for the commit
    /// message to quote.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_decode_steps_attention_step_costs_the_device() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let mut empty = crate::testing::EmptyDispatch::new(&device);
        const CALLS: usize = 128;
        const ROUNDS: usize = 5;

        // One threadgroup to each query of each head, which is the grid
        // `encoding` dispatches over — taken from the case rather than written
        // down, so that a floor cannot go on being subtracted from a step whose
        // grid has moved out from under it.
        // The checkpoint's own two kinds of layer: seven global and thirty-five
        // capped at 512.
        // **Every context the stack is priced at is here at both window
        // settings**, because that is the whole of the split a decode step's
        // `fused_attention` row has to be read as: 35 of the 42 layers are the
        // windowed row and 7 are the global one, and one row per key count could
        // not say which of the two grows. The other four are the shape of each
        // curve either side of the window.
        let spans: Vec<(usize, usize)> = [8, 512, 4096, 16384]
            .into_iter()
            .flat_map(|keys| [(keys, 0), (keys, SLIDING_WINDOW)])
            .chain(
                CONTEXTS
                    .into_iter()
                    .flat_map(|keys| [(keys, 0), (keys, SLIDING_WINDOW)]),
            )
            .collect();
        let grid = |keys: usize, sliding: usize| {
            let case = windowed(keys, sliding);
            Grid::new(
                case.config.heads * case.queries * THREADS_PER_GROUP,
                THREADS_PER_GROUP,
            )
        };
        let groups = grid(spans[0].0, spans[0].1).threads() / THREADS_PER_GROUP;

        // The kernel this one replaced, which walks every key of the span — kept
        // as a column rather than as a number in a commit message, because what
        // the bound is worth is the difference between two rows of one table and
        // a reader should not have to take that on trust.
        let walking =
            FusedAttention::from_source(&device, &unbounded()).expect("the mutant compiles");

        let cost = |kernel: &FusedAttention, keys: usize, sliding: usize| -> Duration {
            let case = windowed(keys, sliding);
            let layer = case.wrapped(&device, kernel);
            // The dispatch's own output, held until the command buffer that
            // writes it has completed.
            let mut held = Vec::with_capacity(CALLS);
            crate::testing::device_time(&device, CALLS, |batch| {
                held.push(layer.encode(batch, case.step()).expect("the step encodes"));
            })
        };

        // Warm: the first dispatch of a fresh pipeline pays for the driver's
        // first look at these buffers, which a decode loop pays once.
        let mut taken: Vec<[Vec<Duration>; 3]> = spans
            .iter()
            .map(|&(keys, sliding)| {
                cost(&attention, keys, sliding);
                cost(&walking, keys, sliding);
                [const { Vec::new() }; 3]
            })
            .collect();
        empty.cost(&device, CALLS, grid(spans[0].0, spans[0].1));
        for _ in 0..ROUNDS {
            for (each, &(keys, sliding)) in taken.iter_mut().zip(&spans) {
                each[0].push(cost(&attention, keys, sliding));
            }
            for (each, &(keys, sliding)) in taken.iter_mut().zip(&spans) {
                each[1].push(cost(&walking, keys, sliding));
            }
            for (each, &(keys, sliding)) in taken.iter_mut().zip(&spans) {
                each[2].push(empty.cost(&device, CALLS, grid(keys, sliding)));
            }
        }

        let mean = |of: &Vec<Duration>| of.iter().sum::<Duration>() / of.len() as u32;
        for ((keys, sliding), each) in spans.iter().zip(&taken) {
            let (bounded, walked, launch) = (mean(&each[0]), mean(&each[1]), mean(&each[2]));
            let window = match sliding {
                0 => "global".to_string(),
                _ => format!("window {sliding}"),
            };
            eprintln!(
                "one query over {keys:>5} keys, {window:>11}, {groups} threadgroups: \
                 {bounded:>8.2?} a dispatch against {walked:>8.2?} walking the span whole \
                 — ×{:.2}, over a {launch:.2?} launch",
                walked.as_secs_f64() / bounded.as_secs_f64(),
            );
        }

        // **The stack's own 42 dispatches, from the two rows they are made of.**
        // A decode step's `fused_attention` row is 35 windowed layers and 7
        // global ones, and the per-kernel table sums them into one figure — so
        // what says which half grows is this sum beside that row, at the same
        // three contexts.
        let at = |keys: usize, sliding: usize| {
            let found = spans
                .iter()
                .position(|span| *span == (keys, sliding))
                .expect("the span is measured");
            mean(&taken[found][0])
        };
        for keys in CONTEXTS {
            let (window, global) = (at(keys, SLIDING_WINDOW), at(keys, 0));
            let (sliding_layers, global_layers) = (
                SLIDING_LAYERS as u32 * window,
                GLOBAL_LAYERS as u32 * global,
            );
            eprintln!(
                "over {keys:>5} keys the stack's 42 dispatches are {:>8.2?} — \
                 {SLIDING_LAYERS} windowed at {window:.2?} making {sliding_layers:.2?}, \
                 {GLOBAL_LAYERS} global at {global:.2?} making {global_layers:.2?}",
                sliding_layers + global_layers,
            );
        }
    }

    /// The contexts a decode step is priced at, which are the cross-engine
    /// table's own prompt lengths.
    const CONTEXTS: [usize; 3] = [97, 385, 769];

    /// The checkpoint's own split: layers 5, 11, 17, 23, 29, 35 and 41 are full
    /// attention and the other 35 cap at a 512-token window.
    const SLIDING_LAYERS: usize = 35;
    const GLOBAL_LAYERS: usize = 7;
    const SLIDING_WINDOW: usize = 512;

    /// **The contexts a coding session actually has**, which is the question
    /// [`CONTEXTS`] cannot answer: every decode figure in this repo tops out at
    /// 769 tokens and a coding turn opens at thousands and grows all session.
    /// Whether a step is linear in the context or plateaus is what decides
    /// whether this engine is usable there at all, and one cannot be told from
    /// the other over an eightfold range.
    const LONG_CONTEXTS: [usize; 9] = [97, 385, 769, 2048, 4096, 8192, 16384, 32768, 65536];

    /// Dispatches one reading of a span puts in a command buffer.
    ///
    /// **Scaled by the span so every reading costs about the same wall time.**
    /// A fixed count is what the sweeps above use over shapes within a factor of
    /// two of each other; here the spans differ by 675, so a count that suits
    /// the short one puts the long one at a minute a reading. Bounded below so a
    /// long span is still an average of several dispatches rather than of one.
    fn readings_of(keys: usize) -> usize {
        ((1 << 20) / keys).clamp(8, 128)
    }

    /// **What a stack of 42 spans holds as the context grows, against what the
    /// window says it needs to.**
    ///
    /// The architecture note has claimed since this repo began that KV costs
    /// "28 KiB/token plus a fixed 70 MiB per sequence" because "only the 7
    /// global layers grow with sequence length", and that "a 1M-token context
    /// fits in under 30 GiB". **That is a claim about a design and this engine
    /// does not implement it.** [`KeyValues::reserve`] allocates against the
    /// keys a sequence has seen and consults nothing else, so all 42 layers
    /// grow — including the 35 that can never read past their own last 512
    /// keys.
    ///
    /// So the table has three columns and the gap between two of them is the
    /// finding: what the spans hold, what the 35 windowed layers hold of that,
    /// and what they would hold capped at a window they cannot see past. Note
    /// what the third column is not — it is not a measurement of anything that
    /// exists, it is the arithmetic of the cap this engine does not make.
    ///
    /// **The bytes are exact and the resident set is the cross-check.** A span
    /// is `[kv_heads, capacity, head_dim]` float32 twice over and capacity is a
    /// power of two, so what a context costs is arithmetic rather than an
    /// estimate; what `ps` reports beside it is whether those pages are this
    /// process's, which for a buffer nothing has written to yet they need not
    /// be.
    ///
    /// One stack grown through the contexts rather than one per context, which
    /// is what a session does: a span only ever grows, so the last row is the
    /// footprint and the ones above it are what it passed through.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_context_costs_in_keys_and_values() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let stack: Vec<LayerAttention<'_>> = (0..SLIDING_LAYERS + GLOBAL_LAYERS)
            .map(|layer| {
                let case = windowed(1, window_of(layer));
                case.wrapped(&device, &attention)
            })
            .collect();

        let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
        eprintln!(
            "{:>8}{:>12}{:>12}{:>14}{:>10}{:>12}",
            "context", "the spans", "windowed", "if capped", "saved", "resident"
        );
        for context in LONG_CONTEXTS {
            for layer in &stack {
                layer.hold(0, context).expect("the span reserves");
            }
            let (held, of_window, capped) = stack
                .iter()
                .map(|layer| {
                    let bytes = layer.span_bytes();
                    let config = layer.config();
                    // What the layer would have reserved if the window were
                    // what it reserved against, which is the keys it may read.
                    let reach = match config.sliding {
                        0 => context,
                        window => window.min(context),
                    };
                    // **A global layer's two columns have to be the same
                    // number**, because a layer with no window reaches every
                    // key it has and the cap is the allocation. That is what
                    // says the third column is this engine's own arithmetic
                    // applied to a different reach rather than a second guess
                    // at what a span costs.
                    if config.sliding == 0 {
                        assert_eq!(span_bytes_for(config, reach), bytes);
                    }
                    (
                        bytes,
                        u64::from(config.sliding > 0) * bytes,
                        span_bytes_for(config, reach),
                    )
                })
                .fold((0, 0, 0), |(a, b, c), (x, y, z)| (a + x, b + y, c + z));
            eprintln!(
                "{context:>8}{:>12}{:>12}{:>14}{:>10}{:>12}",
                format!("{:.0} MiB", mib(held)),
                format!("{:.0} MiB", mib(of_window)),
                format!("{:.0} MiB", mib(capped)),
                format!("×{:.1}", held as f64 / capped as f64),
                format!("{:.0} MiB", mib(inkling_core::fixture::resident_bytes())),
            );
        }
    }

    /// The window of layer `layer` under the checkpoint's 5:1 split: layers 5,
    /// 11, 17 and every sixth after them are global and the rest are capped.
    fn window_of(layer: usize) -> usize {
        if layer % 6 == 5 { 0 } else { SLIDING_WINDOW }
    }

    /// What one layer's keys and values would occupy with room for `keys` of
    /// them, through [`KeyValues::capacity_for`] rather than beside it — so a
    /// change to how a span grows moves this column with the one it is compared
    /// against.
    fn span_bytes_for(config: AttentionConfig, keys: usize) -> u64 {
        let slots = KeyValues::capacity_for(keys);
        2 * (config.kv_heads * slots * config.head_dim) as u64 * size_of::<f32>() as u64
    }

    /// **What cutting the key span across threadgroups is worth, and where it
    /// turns** — which is what [`WANTED_GROUPS`] is fitted on rather than
    /// reasoned about.
    ///
    /// One split is the kernel this repo ran for eleven milestones: 32
    /// threadgroups at a decode step's one query, on a machine with 80 cores.
    /// Every column past it is the same walk over the same tiles cut more ways,
    /// and each pays a second dispatch to fold what the splits left — so the
    /// turn is where the fold starts costing more than the parallelism buys,
    /// which is the shape of finding `ROWS_A_TILE` and `COLUMNS_A_TILE` both
    /// are.
    ///
    /// **The fold is inside the figure**, because a batch holds both dispatches
    /// and the device's clock is read across it. A column that timed the walk
    /// alone would rank the widest cut first at every shape and be wrong about
    /// all of them.
    ///
    /// Both kinds of layer, because they are not the same question. A global
    /// layer's live range is the whole span and every split gets a share of it;
    /// a windowed layer's is its last 512 keys, and the splits are cut over the
    /// span rather than over the live range — see SPLIT in the kernel, where
    /// that is what keeps the loop bound exact — so past a context of about four
    /// windows its live keys land in one split and the rest return. That is
    /// visible here as a windowed row that stops improving, and it is the next
    /// thing to take.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_the_split_over_the_key_span_is_worth() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        const ROUNDS: usize = 3;
        const CUTS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

        let cost = |keys: usize, sliding: usize, queries: usize, splits: usize| -> Duration {
            let case = blocked(keys, sliding, queries);
            let layer = case.wrapped(&device, &attention);
            layer.hold(0, keys).expect("the span reserves");
            layer.span().appended(keys);
            layer.split_into(Some(splits));

            let mut q = device.buffer(&case.q).expect("the queries upload");
            let mut rel = device.buffer(&case.rel).expect("the features upload");
            let mut span = layer.span();
            let calls = readings_of(keys);
            let mut held = Vec::with_capacity(calls);
            crate::testing::device_time(&device, calls, |batch| {
                held.push(
                    layer
                        .encode_over(batch, &mut span, &mut q, &mut rel, None, keys - queries)
                        .expect("the step encodes"),
                );
            })
        };

        // A decode step at both kinds of layer and two contexts, and the two
        // block widths a speculative round proposes — `k = 2` verifies three
        // rows and the deepest chain nine, which are 96 and 288 threadgroups
        // where a decode step is 32. **A block is where the predicate could
        // most easily be wrong**: its grid is already wide, so a split there
        // buys less parallelism and costs the same fold.
        let shapes = [
            (769, 0, 1),
            (769, SLIDING_WINDOW, 1),
            (8192, 0, 1),
            (8192, SLIDING_WINDOW, 1),
            (8192, 0, 3),
            (8192, SLIDING_WINDOW, 3),
            (8192, 0, 9),
            (8192, SLIDING_WINDOW, 9),
        ];
        let mut taken = shapes.map(|(keys, sliding, queries)| {
            CUTS.map(|splits| {
                cost(keys, sliding, queries, splits);
                Vec::new()
            })
        });
        for _ in 0..ROUNDS {
            for (each, (keys, sliding, queries)) in taken.iter_mut().zip(shapes) {
                for (readings, splits) in each.iter_mut().zip(CUTS) {
                    readings.push(cost(keys, sliding, queries, splits));
                }
            }
        }

        eprintln!(
            "{:>26}{}",
            "splits a call",
            CUTS.map(|c| format!("{c:>10}")).concat()
        );
        for ((keys, sliding, queries), each) in shapes.iter().zip(&taken) {
            let window = match sliding {
                0 => "global".to_string(),
                _ => format!("window {sliding}"),
            };
            let cells: String = each
                .iter()
                .map(|readings| {
                    let mean = readings.iter().sum::<Duration>() / readings.len() as u32;
                    format!("{:>10}", format!("{mean:.2?}"))
                })
                .collect();
            eprintln!("{keys:>6} keys, {queries:>2} rows, {window:>11}{cells}");
        }
        for (keys, queries) in [(769, 1), (8192, 1), (8192, 3), (8192, 9), (8192, 769)] {
            eprintln!(
                "  the shipped predicate cuts {queries:>3} rows over {keys:>4} keys \
                 — {:>5} threadgroups — into {} splits",
                32 * queries,
                splits_for(32 * queries, keys, 128),
            );
        }
    }

    /// **What the attention step costs as the context grows into a coding
    /// one**, out to 65536 keys — 85 times the longest context this repo has
    /// ever quoted a decode figure at.
    ///
    /// **This is the one row of a decode step that is not flat in the context**,
    /// which `which_kernels_own_a_decode_step_at_each_context` establishes over
    /// the three lengths the cross-engine table uses: every other kernel moves
    /// by under 20% between 97 and 769 keys while this one goes 3.93 ms to
    /// 17.35. So the shape of *this* curve is the shape of the step's, and the
    /// two halves of it are shaped differently on purpose — 35 layers cap at a
    /// 512-token window and 7 do not.
    ///
    /// **The per-key column is what says which.** A row whose per-key cost is
    /// constant is walking the span; one whose per-key cost falls as `1/keys` is
    /// flat in the context. Nothing here asserts a slope — the numbers go to
    /// stderr — because the point of the table is the shape rather than a bound
    /// anyone would have to maintain.
    ///
    /// The span is held on the device across the readings rather than handed
    /// over per call, which is the difference between measuring this kernel and
    /// measuring a 268 MB copy: [`LayerAttention::encode`] uploads the whole
    /// span per call, and the fixed 128 readings the sweep above takes would be
    /// 34 GB of buffers for one command buffer to retain at this size. What the
    /// layer holds is what a decode step reads — see
    /// [`LayerAttention::encode_over`] — and the contents are the zeroes the
    /// span was allocated with, because what a key costs does not depend on
    /// what is in it.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_the_attention_step_costs_as_the_context_grows() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        const ROUNDS: usize = 3;

        let cost = |keys: usize, sliding: usize| -> Duration {
            let case = windowed(keys, sliding);
            let layer = case.wrapped(&device, &attention);
            layer.hold(0, keys).expect("the span reserves");
            layer.span().appended(keys);

            let mut q = device.buffer(&case.q).expect("the query uploads");
            let mut rel = device.buffer(&case.rel).expect("the features upload");
            let mut span = layer.span();
            let calls = readings_of(keys);
            let mut held = Vec::with_capacity(calls);
            crate::testing::device_time(&device, calls, |batch| {
                held.push(
                    layer
                        .encode_over(batch, &mut span, &mut q, &mut rel, None, keys - 1)
                        .expect("the step encodes"),
                );
            })
        };

        let mut taken = LONG_CONTEXTS.map(|keys| {
            // Warm, for the reason the sweep above warms: the first dispatch of
            // a fresh pipeline pays for the driver's first look at its buffers.
            cost(keys, 0);
            cost(keys, SLIDING_WINDOW);
            [const { Vec::new() }; 2]
        });
        for _ in 0..ROUNDS {
            for (each, keys) in taken.iter_mut().zip(LONG_CONTEXTS) {
                each[0].push(cost(keys, 0));
            }
            for (each, keys) in taken.iter_mut().zip(LONG_CONTEXTS) {
                each[1].push(cost(keys, SLIDING_WINDOW));
            }
        }

        let mean = |of: &Vec<Duration>| of.iter().sum::<Duration>() / of.len() as u32;
        eprintln!(
            "{:>7}{:>12}{:>10}{:>12}{:>10}{:>12}{:>14}",
            "keys", "global", "a key", "window 512", "a key", "the stack", "a decode step"
        );
        for (keys, each) in LONG_CONTEXTS.iter().zip(&taken) {
            let (global, window) = (mean(&each[0]), mean(&each[1]));
            let stack = SLIDING_LAYERS as u32 * window + GLOBAL_LAYERS as u32 * global;
            let a_key = |of: Duration| of.as_secs_f64() * 1e6 / *keys as f64;
            eprintln!(
                "{keys:>7}{:>12}{:>10}{:>12}{:>10}{:>12}{:>14}",
                format!("{global:.2?}"),
                format!("{:.3}µs", a_key(global)),
                format!("{window:.2?}"),
                format!("{:.3}µs", a_key(window)),
                format!("{stack:.2?}"),
                format!("{:.1}ms", stack.as_secs_f64() * 1e3),
            );
        }
    }

    /// The prompt lengths a prefill's own attention is priced at, which are the
    /// four a coding turn opens somewhere inside.
    const PREFILL_CONTEXTS: [usize; 4] = [2048, 4096, 8192, 16384];

    /// **What a prefill's attention costs on each kind of layer**, which is the
    /// same question the sweep above asks of a decode step and is a different
    /// shape entirely: there one query row walks a span, here `n` of them do.
    ///
    /// **What the two columns should be is arithmetic, and the point of the
    /// table is whether they are.** A windowed layer's row `i` may read its last
    /// 512 keys, so 35 of them are `n × 512` and linear in the prompt; a global
    /// layer's row `i` may read `i + 1`, so 7 of them are `n²/2` and quadratic.
    /// A prefill that walked full spans on all 42 would be about six times the
    /// work at 16384 — and this is what says whether that is so rather than an
    /// engine's whole prefill wall, which has a MoE and a hundred projections in
    /// it as well.
    ///
    /// **The per-token and per-token² columns are what say which term a row
    /// is.** A row whose cost divided by `n` is constant is linear; one whose
    /// cost divided by `n²` is constant is quadratic. Neither is asserted — the
    /// numbers go to stderr — because the shape is the finding and a bound on it
    /// is a bound somebody would have to maintain.
    ///
    /// One dispatch a reading rather than the sweep above's several: a call at
    /// this shape allocates `[n, heads × head_dim]` for its answer, which at
    /// 16384 tokens is 268 MB that the command buffer holds until it completes.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_prefills_attention_costs_as_the_prompt_grows() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        const ROUNDS: usize = 3;

        let cost = |tokens: usize, sliding: usize| -> Duration {
            let case = blocked(tokens, sliding, tokens);
            let layer = case.wrapped(&device, &attention);
            layer.hold(0, tokens).expect("the span reserves");
            layer.span().appended(tokens);

            let mut q = device.buffer(&case.q).expect("the queries upload");
            let mut rel = device.buffer(&case.rel).expect("the features upload");
            let mut span = layer.span();
            let mut held = None;
            crate::testing::device_time(&device, 1, |batch| {
                held = Some(
                    layer
                        .encode_over(batch, &mut span, &mut q, &mut rel, None, 0)
                        .expect("the prefill encodes"),
                );
            })
        };

        let mut taken = PREFILL_CONTEXTS.map(|_| [const { Vec::new() }; 2]);
        for round in 0..ROUNDS {
            for (each, tokens) in taken.iter_mut().zip(PREFILL_CONTEXTS) {
                for (readings, sliding) in each.iter_mut().zip([0, SLIDING_WINDOW]) {
                    // Warm on the first round rather than beside every reading:
                    // the first dispatch of a fresh pipeline pays for the
                    // driver's first look at its buffers, and at this shape a
                    // warming call is seconds rather than microseconds.
                    let taken = cost(tokens, sliding);
                    if round > 0 {
                        readings.push(taken);
                    }
                }
            }
        }

        let mean = |of: &Vec<Duration>| of.iter().sum::<Duration>() / of.len() as u32;
        eprintln!(
            "{:>7}{:>12}{:>11}{:>11}{:>12}{:>11}{:>11}{:>12}",
            "tokens",
            "global",
            "a token",
            "a token²",
            "window 512",
            "a token",
            "a token²",
            "the stack"
        );
        for (tokens, each) in PREFILL_CONTEXTS.iter().zip(&taken) {
            let (global, window) = (mean(&each[0]), mean(&each[1]));
            let stack = SLIDING_LAYERS as u32 * window + GLOBAL_LAYERS as u32 * global;
            let a_token = |of: Duration| of.as_secs_f64() * 1e6 / *tokens as f64;
            let a_square = |of: Duration| of.as_secs_f64() * 1e9 / (*tokens * *tokens) as f64;
            eprintln!(
                "{tokens:>7}{:>12}{:>11}{:>11}{:>12}{:>11}{:>11}{:>12}",
                format!("{global:.2?}"),
                format!("{:.2}µs", a_token(global)),
                format!("{:.3}ns", a_square(global)),
                format!("{window:.2?}"),
                format!("{:.2}µs", a_token(window)),
                format!("{:.3}ns", a_square(window)),
                format!("{stack:.2?}"),
            );
        }
    }

    /// **Whether a prefill's attention is waiting on the keys it reads**, which
    /// is the premise under every proposal to make this kernel read fewer of
    /// them and is the one thing a bandwidth column computed from a *declared*
    /// byte count cannot say.
    ///
    /// The declared figure is the reads the walk issues: at 16384 tokens the
    /// seven global layers issue 30.8 TB against a span of 134 MB apiece, which
    /// over their device time is 699 GB/s against this machine's 819. Read as a
    /// bandwidth that says the kernel is at the memory and the only lever left
    /// is to read less. Read as an amplification it says nothing at all — 32
    /// query heads and their neighbouring rows walk almost the same keys at
    /// almost the same time, so a tile fetched for one is in cache for the rest,
    /// and 699 GB/s of *issued* reads can sit on far less traffic. **This is the
    /// same distinction [`crate::PackedBank::moves`] forced on the matmul rows**,
    /// met on the other kernel.
    ///
    /// So it is settled by a mutation rather than by an argument: the same
    /// kernel with every key and every value read from slot zero. It walks the
    /// same tiles, scores the same number of keys, takes the same barriers and
    /// does the same arithmetic — and its whole working set is one 16 KB tile
    /// that never leaves the cache. **What separates the two is the memory and
    /// nothing else.**
    ///
    /// The answer it gives is wrong, and the case asserts that it is: a mutation
    /// that read the right keys after all would report the walk costs nothing
    /// and be measuring the shipped kernel twice.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn whether_a_prefills_attention_is_waiting_on_the_keys_it_reads() {
        let Some(device) = device() else { return };
        const TOKENS: [usize; 3] = [2048, 4096, 8192];

        let walking = FusedAttention::new(&device).expect("the kernel compiles");
        let cached =
            FusedAttention::from_source(&device, &reading_one_slot()).expect("the mutant compiles");
        assert_ne!(
            blocked(97, 0, 97).on_the_device(&device, &walking),
            blocked(97, 0, 97).on_the_device(&device, &cached),
            "reading one slot answered what reading the span answers"
        );
        let cost = |attention: &FusedAttention, tokens, sliding| {
            a_prefill_costs(&device, attention, tokens, sliding)
        };

        eprintln!(
            "{:>7}{:>14}{:>12}{:>12}{:>10}",
            "tokens", "layer", "the span", "one slot", "of it"
        );
        for tokens in TOKENS {
            for (what, sliding) in [("global", 0), ("window 512", SLIDING_WINDOW)] {
                let (span, slot) = (
                    cost(&walking, tokens, sliding),
                    cost(&cached, tokens, sliding),
                );
                eprintln!(
                    "{tokens:>7}{what:>14}{:>12}{:>12}{:>10}",
                    format!("{span:.2?}"),
                    format!("{slot:.2?}"),
                    format!("{:.0}%", 1e2 * slot.as_secs_f64() / span.as_secs_f64()),
                );
            }
        }
    }

    /// The prompt lengths the limiter table below is taken at — the shortest
    /// and the longest of [`PREFILL_CONTEXTS`] that a sweep of nine arms can
    /// afford three passes of.
    const BOUND_TOKENS: [usize; 2] = [2048, 8192];

    /// The four dispatches every table below is read across: both prompt lengths
    /// on both kinds of layer, as `(tokens, sliding, what to call it)`.
    fn bound_cells() -> Vec<(usize, usize, &'static str)> {
        BOUND_TOKENS
            .iter()
            .flat_map(|&tokens| {
                [("global", 0), ("window 512", SLIDING_WINDOW)]
                    .map(|(what, sliding)| (tokens, sliding, what))
            })
            .collect()
    }

    /// What one prefill-shaped dispatch of `attention` costs on the device — `n`
    /// query rows over `n` keys, the shape
    /// [`what_a_prefills_attention_costs_as_the_prompt_grows`] prices and the
    /// one both tables below mutate underneath.
    ///
    /// The best of `ROUNDS` after a warming call, for the reason every sweep
    /// here warms: the first dispatch of a fresh pipeline pays for the driver's
    /// first look at its buffers, and at this shape that is seconds.
    fn a_prefill_costs(
        device: &Device,
        attention: &FusedAttention,
        tokens: usize,
        sliding: usize,
    ) -> Duration {
        const ROUNDS: usize = 2;
        let case = blocked(tokens, sliding, tokens);
        let layer = case.wrapped(device, attention);
        layer.hold(0, tokens).expect("the span reserves");
        layer.span().appended(tokens);
        let mut q = device.buffer(&case.q).expect("the queries upload");
        let mut rel = device.buffer(&case.rel).expect("the features upload");
        let mut span = layer.span();
        let mut held = None;
        let mut taken = Duration::MAX;
        for round in 0..=ROUNDS {
            let each = crate::testing::device_time(device, 1, |batch| {
                held = Some(
                    layer
                        .encode_over(batch, &mut span, &mut q, &mut rel, None, 0)
                        .expect("the prefill encodes"),
                );
            });
            taken = if round == 0 { taken } else { taken.min(each) };
        }
        taken
    }

    /// One arm of the limiter table: a name, and the shipped source with one
    /// thing taken out of it.
    ///
    /// **Every arm here answers wrongly and is meant to.** What each measures is
    /// the shipped kernel minus one term, so an arm that still answered what the
    /// kernel answers would be the kernel measured twice under another name —
    /// which is the failure the slot-zero case above already guards against, met
    /// once per term rather than once.
    struct Without {
        what: &'static str,
        source: String,
        /// Whether the arm's answer must differ from the shipped kernel's.
        ///
        /// True everywhere but the barrier arm: taking the barriers out is a
        /// race rather than a different arithmetic, and a race that happens to
        /// resolve the shipped way at one shape is not evidence the source was
        /// unchanged.
        answers_differently: bool,
    }

    /// The shipped source with every key and every value read from slot zero:
    /// the same tiles, the same barriers and the same arithmetic over a 16 KB
    /// working set that never leaves cache, so what separates it from the kernel
    /// is the memory and nothing else.
    fn reading_one_slot() -> String {
        let staged = crate::testing::instead_of(
            &source(),
            "values_of + (ulong)first * shape.head_dim",
            "values_of + (ulong)0 * shape.head_dim",
        );
        crate::testing::instead_of(
            &staged,
            "keys_of + (ulong)j * shape.head_dim",
            "keys_of + (ulong)0 * shape.head_dim",
        )
    }

    /// The nine arms, each the shipped source with one term of the walk removed.
    ///
    /// **The replacements are cheap rather than absent, so that nothing an arm
    /// removes can take something else with it.** A term deleted outright would
    /// let the compiler drop every load that fed it and the arm would price two
    /// things; a term replaced by one instruction over the same operands leaves
    /// the reads where they are and prices itself.
    fn without_each_term() -> Vec<Without> {
        let shipped = source();
        let mut arms = Vec::new();
        let mut arm = |what, source: String, answers_differently| {
            assert_ne!(source, shipped, "{what}: the mutation changed nothing");
            arms.push(Without {
                what,
                source,
                answers_differently,
            });
        };

        arm("the keys and values", reading_one_slot(), true);

        // The band, which is `d_rel` device reads and `d_rel` multiplies made by
        // lane 0 of a simdgroup for every key the other 31 lanes scored — and is
        // the whole of what this kernel does that the reference's does not,
        // since mlx-vlm reads a mask somebody else materialised. `fmin` of the
        // two operands is one instruction that cannot be folded away and lands
        // in [0, 1], so nothing downstream overflows.
        arm(
            "the band it derives",
            crate::testing::instead_of(
                &shipped,
                "banded_entry(proj, features, shape, position - (int)j, tau);",
                "fmin(tau, (float)(position - (int)j));",
            ),
            true,
        );

        // The transcendental, twice: once at the precision the reference uses
        // and once at no precision at all. `fast::exp2` is a hardware
        // instruction and `precise::exp` a range reduction around one, so the
        // first arm is the cost of the accuracy and the second the cost of the
        // function.
        //
        // **Both rewrite the fold's exponential as well as the walk's**, and
        // that is inert rather than intended: `splits_for` gives one split at
        // every shape here, so no dispatch of `attention_combine` is encoded.
        arm(
            "the exp's precision",
            crate::testing::instead_of(&shipped, "precise::exp(", "fast::exp2("),
            true,
        );
        arm(
            "the exp",
            crate::testing::instead_of(
                &crate::testing::instead_of(
                    &shipped,
                    "using namespace metal;",
                    "using namespace metal;\n\
                     inline float cheap_exp(float x) { return fmax(1.0f + x, 0.0f); }",
                ),
                "precise::exp(",
                "cheap_exp(",
            ),
            true,
        );

        // Every barrier, which is the four a tile takes, the one that publishes
        // the staged query row ahead of the walk, and the fold's — the last of
        // which no dispatch here encodes.
        arm(
            "the barriers",
            crate::testing::instead_of(
                &shipped,
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                ";",
            ),
            false,
        );

        // The two serial reductions over the tile, which every one of the 256
        // threads runs for itself: 32 threadgroup reads and 32 operations
        // apiece, twice a tile, for two numbers.
        arm(
            "the tile's two reductions",
            crate::testing::instead_of(
                &crate::testing::instead_of(
                    &shipped,
                    "float top = peak;\n        for (uint s = 0; s < held; ++s) {",
                    "float top = peak;\n        for (uint s = 0; s < 1u; ++s) {",
                ),
                "float sum = 0.0f;\n        for (uint s = 0; s < held; ++s) {",
                "float sum = 0.0f;\n        for (uint s = 0; s < 1u; ++s) {",
            ),
            true,
        );

        // The weighting, which is the tile's values against the tile's weights:
        // 32 threadgroup reads and 32 multiply-adds on each of the 128 threads a
        // 128-channel head leaves busy.
        arm(
            "the weighting",
            crate::testing::instead_of(
                &shipped,
                "for (uint s = 0; s < held; ++s) {\n                acc += scores[s]",
                "for (uint s = 0; s < 1u; ++s) {\n                acc += scores[s]",
            ),
            true,
        );

        // The scoring dot, which is four multiply-adds a lane and the four key
        // reads under them — so this arm takes three quarters of the key traffic
        // with it and is read beside the slot-zero arm rather than alone.
        arm(
            "three quarters of the dot",
            crate::testing::instead_of(
                &shipped,
                "for (uint d = lane; d < shape.head_dim; d += width) {",
                "for (uint d = lane; d < width; d += width) {",
            ),
            true,
        );

        // The cross-lane reduction behind it: five shuffle steps for each of the
        // four keys a simdgroup scores in a tile.
        arm(
            "the simd_sum",
            crate::testing::instead_of(&shipped, "            dot = simd_sum(dot);\n", ""),
            true,
        );

        arms
    }

    /// **What a prefill's attention is bound by, one term at a time** — the
    /// question A3's slot-zero mutation answered for memory alone, asked of
    /// every other candidate on the same denominator.
    ///
    /// A3 established that the keys and values are 15 to 19% of this kernel and
    /// that a lever dividing the reads could never pay. What it did not
    /// establish is what the other four fifths are, and three milestones in a
    /// row have now picked a lever from a number that turned out to be the wrong
    /// denominator. **So the instrument generalises rather than the finding**:
    /// each arm is the shipped source with exactly one term replaced by
    /// something that costs an instruction and cannot be folded away, and the
    /// column that matters is what the clock does.
    ///
    /// **The shares do not sum to one and are not meant to.** Removing a term
    /// removes the instructions that issue it, the registers it held and
    /// whatever it was waiting on, and two terms that were waiting on each other
    /// each look like the whole of the wait. What the table ranks is which terms
    /// are worth anything at all, and the two that are worth nothing are as much
    /// of the finding as the two that are.
    ///
    /// Both kinds of layer, because the band is bounded by `rel_extent` and a
    /// windowed layer's live keys are all inside it where a global layer's are
    /// mostly not — so a term that is a rounding error on one row can be most of
    /// the other.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_prefills_attention_is_bound_by() {
        let Some(device) = device() else { return };

        let shipped = FusedAttention::new(&device).expect("the kernel compiles");
        let answered =
            |attention: &FusedAttention| blocked(97, 0, 97).on_the_device(&device, attention);
        let want = answered(&shipped);
        let cost = |attention: &FusedAttention, tokens, sliding| {
            a_prefill_costs(&device, attention, tokens, sliding)
        };

        let cells = bound_cells();
        let header: String = cells
            .iter()
            .map(|(tokens, _, what)| format!("{:>21}", format!("{tokens} {what}")))
            .collect();
        eprintln!("  {:<28}{header}", "without");

        let whole: Vec<Duration> = cells
            .iter()
            .map(|&(tokens, sliding, _)| cost(&shipped, tokens, sliding))
            .collect();
        let row = |what: &str, taken: &[Duration]| {
            let cells: String = taken
                .iter()
                .zip(&whole)
                .map(|(each, whole)| {
                    format!(
                        "{:>12}{:>9}",
                        format!("{each:.2?}"),
                        format!("{:.0}%", 1e2 * each.as_secs_f64() / whole.as_secs_f64()),
                    )
                })
                .collect();
            eprintln!("  {what:<28}{cells}");
        };
        row("nothing — the kernel", &whole);

        for arm in without_each_term() {
            let mutant =
                FusedAttention::from_source(&device, &arm.source).expect("the mutant compiles");
            if arm.answers_differently {
                assert_ne!(
                    answered(&mutant),
                    want,
                    "{}: the mutation answered what the kernel answers",
                    arm.what
                );
            }
            let taken: Vec<Duration> = cells
                .iter()
                .map(|&(tokens, sliding, _)| cost(&mutant, tokens, sliding))
                .collect();
            row(arm.what, &taken);
        }
    }

    /// The entry a split call runs: the same walk, staging a whole tile of
    /// values rather than declaring memory nobody reads.
    ///
    /// **What it stages is the one thing that decides how much memory the walk
    /// declares**: the four live arrays are 3 KiB and a tile of values is 16,
    /// which is two steps past the occupancy turn. The tile itself does not move
    /// — [`TILED_VALUES`] bounds that in both entries — so what separates the
    /// two is the memory and where a value is read from, and not the number of
    /// keys a tile holds.
    fn staging_the_tiles_values() -> String {
        super::source(STAGED_BY_A_SPLIT_CALL, NO_RESIDENCY)
    }

    /// The shipped source with a tile's two reductions taken by one thread and
    /// handed to the other 255 through threadgroup memory.
    ///
    /// **The same two floats in the same order, computed once instead of 256
    /// times.** Every thread of a threadgroup needs a tile's largest score and
    /// its total, and each of them arrives at both by walking the tile itself —
    /// 256 threads running the same serial chain over the same 32 scores for a
    /// scalar every one of them ends up holding. One thread walking it and
    /// storing the result is the same chain over the same operands in the same
    /// order, so what the others read is the float they would have computed:
    /// bit-safe by construction rather than by tolerance, and
    /// [`a_tile_reduced_by_one_thread_is_reduced_by_all_of_them_bit_for_bit`]
    /// holds it to that.
    ///
    /// **It costs no barrier in the tile.** The maximum needs its readers held
    /// off until the one thread has read every score, which is exactly what the
    /// barrier already standing between the maximum and the rescaling does — the
    /// store lands before it and the loads after it. The total is read by nobody
    /// until the walk is over, so it crosses once at the end rather than once a
    /// tile, which is the one barrier this adds and it is per dispatch.
    ///
    /// The two floats are 8 bytes beside the 12.5 KiB an unsplit call declares,
    /// reported as 16 — a kibibyte inside the near edge of the occupancy
    /// plateau, and a column of the table below rather than an assumption.
    fn reduced_by_one_thread() -> String {
        let declared = crate::testing::instead_of(
            &source(),
            "    threadgroup float scores[MOST_SIMDGROUPS * KEYS_PER_SIMD];",
            "    threadgroup float scores[MOST_SIMDGROUPS * KEYS_PER_SIMD];\n    threadgroup \
             float reduced[2];",
        );
        let most = crate::testing::instead_of(
            &declared,
            "        float top = peak;\n        for (uint s = 0; s < held; ++s) {\n            \
             top = fmax(top, scores[s]);\n        }\n",
            "        if (local == 0) {\n            float most = peak;\n            for (uint s \
             = 0; s < held; ++s) {\n                most = fmax(most, scores[s]);\n            \
             }\n            reduced[0] = most;\n        }\n        \
             threadgroup_barrier(mem_flags::mem_threadgroup);\n        const float top = \
             reduced[0];\n",
        );
        let once = crate::testing::instead_of(
            &most,
            "        const float rescale = precise::exp(peak - top);\n        peak = top;\n       \
             \x20threadgroup_barrier(mem_flags::mem_threadgroup);\n",
            "        const float rescale = precise::exp(peak - top);\n        peak = top;\n",
        );
        let summed = crate::testing::instead_of(
            &once,
            "        float sum = 0.0f;\n        for (uint s = 0; s < held; ++s) {\n            \
             sum += scores[s];\n        }\n        total = total * rescale + sum;\n",
            "        if (local == 0) {\n            float sum = 0.0f;\n            for (uint s = \
             0; s < held; ++s) {\n                sum += scores[s];\n            }\n            \
             total = total * rescale + sum;\n        }\n",
        );
        crate::testing::instead_of(
            &summed,
            "    if (shape.splits == 1u) {",
            "    if (local == 0) {\n        reduced[1] = total;\n    }\n    \
             threadgroup_barrier(mem_flags::mem_threadgroup);\n    total = reduced[1];\n\n    if \
             (shape.splits == 1u) {",
        )
    }

    /// **A tile reduced by one thread is a tile reduced by all of them, bit for
    /// bit**, which is what would have made the choice between them a rate and
    /// never an answer.
    ///
    /// The claim is that a broadcast is a copy: the two scalars come off the
    /// same serial chain over the same scores in the same ascending order, and
    /// what a thread reads out of threadgroup memory is the float it would have
    /// computed for itself. So this is `assert_eq!` on the bits rather than a
    /// tolerance — `-0.0` and `0.0` compare equal as floats and are two
    /// different answers — over the same cases and both paths
    /// [`a_value_weighted_where_it_lies_is_a_staged_one_bit_for_bit`] drives,
    /// because the peak and the total are what a split call's partials carry and
    /// a fold is where a wrong one would show.
    ///
    /// **Kept though the arm is not shipped**, because what it establishes is
    /// what makes the table below a table of rates: two arms answering different
    /// bits would be two kernels and their times would not be comparable at all.
    #[test]
    fn a_tile_reduced_by_one_thread_is_reduced_by_all_of_them_bit_for_bit() {
        let Some(device) = device() else { return };
        let all = FusedAttention::from_source(&device, &source()).expect("the kernel compiles");
        let broadcast = reduced_by_one_thread();
        assert_ne!(broadcast, source(), "the arm changed nothing");
        let one =
            FusedAttention::from_source(&device, &broadcast).expect("the broadcast arm compiles");

        let mut cases = Case::synthetic();
        cases.extend([
            blocked(2048, 0, 2048),
            blocked(2048, SLIDING_WINDOW, 2048),
            blocked(600, SLIDING_WINDOW, 13),
            blocked(97, 0, 97),
            blocked(1200, 0, 1),
            blocked(1200, SLIDING_WINDOW, 1),
        ]);
        cases.extend(
            Case::all(ACTIVATIONS)
                .expect("the committed capture")
                .into_iter()
                .map(|(case, _)| case),
        );

        let mut elements = 0;
        for case in &cases {
            for splits in [1, 8] {
                let want = case.cut(&device, &all, splits);
                let got = case.cut(&device, &one, splits);
                let apart = want
                    .iter()
                    .zip(&got)
                    .position(|(want, got)| want.to_bits() != got.to_bits());
                assert_eq!(
                    apart,
                    None,
                    "{} in {splits}: one thread reducing answered {:?} where all of them answered \
                     {:?}",
                    case.name,
                    apart.map(|at| got[at]),
                    apart.map(|at| want[at]),
                );
                elements += want.len();
            }
        }
        eprintln!(
            "{} cases agree bit for bit over {elements} elements",
            cases.len()
        );
    }

    /// **What reducing a tile in one thread costs**, which is the measurement
    /// A4's list asked for and A5 declined to infer — and which answers the
    /// opposite of what the model does.
    ///
    /// A4 priced the two reductions at 23 to 29% of the attention rows: every
    /// one of 256 threads walks a tile's 32 scores twice for two scalars, and
    /// that is the largest term of the walk after the keys. One thread reducing
    /// frees 255 threads' issue slots and puts the same serial chain on the
    /// critical path with everyone else waiting at a barrier for it, so whether
    /// it pays is a measurement rather than an inference either way.
    ///
    /// **Here it pays and in the model it does not, and the model is the
    /// arbiter.** This dispatch is 5.6 and 7.2% faster at the two global cells;
    /// the same kernel inside a 769-token prefill is 7.8 and 8.2% *slower* on
    /// the two attention rows, and a paired 2048-token prefill is 2.9% slower on
    /// the device's own clock. What separates them is what else is running: a
    /// dispatch measured alone, three rounds over one set of keys, is warm and
    /// issue-bound, where the same dispatch between two matmuls that stream a
    /// terabyte is reading from memory at 88 to 95% of this part's peak — and
    /// issue slots freed under a bandwidth ceiling buy nothing while the
    /// serialization is still paid.
    ///
    /// **So this table is kept for the caution rather than for the number.** It
    /// is the instrument A4's whole attention limiter table is taken on, and
    /// this is the first arm measured on both sides of it.
    ///
    /// The declared column is here because it moves: the arm holds two floats
    /// the kernel does not, and a change of sixteen bytes that crossed the
    /// occupancy turn would be measuring the turn rather than the reduction.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_reducing_a_tile_in_one_thread_costs() {
        let Some(device) = device() else { return };
        let cells = bound_cells();
        let header: String = cells
            .iter()
            .map(|(tokens, _, what)| format!("{:>17}", format!("{tokens} {what}")))
            .collect();
        eprintln!(
            "  {:<24}{:>14}{header}",
            "the two reductions", "a threadgroup"
        );

        for (what, arm) in [
            ("every thread — shipped", source()),
            ("one thread, broadcast", reduced_by_one_thread()),
        ] {
            let attention = FusedAttention::from_source(&device, &arm).expect("the arm compiles");
            let held = attention.global.threadgroup_memory();
            let taken: String = cells
                .iter()
                .map(|&(tokens, sliding, _)| {
                    let each = a_prefill_costs(&device, &attention, tokens, sliding);
                    format!("{:>17}", format!("{each:.2?}"))
                })
                .collect();
            eprintln!(
                "  {what:<24}{:>14}{taken}",
                format!("{:.2} KiB", held as f64 / 1024.0),
            );
        }
    }

    /// **How many threadgroups of this kernel a core holds, what holding a
    /// different number is worth, and which of the two things that decide it is
    /// doing the work** — the occupancy term, turned by a knob at each end.
    ///
    /// A threadgroup here is one query row of one head, and its four live arrays
    /// are 3 KiB of the 32 an Apple GPU allows a declaration. What else it
    /// declares decides how many of it a core holds, so this sweeps that from
    /// both sides: a walk that stages a tile and one that stages nothing and
    /// declares the same bytes for nobody to read. Every arm walks the same keys
    /// in the same tiles with the same instructions — [`TILED_VALUES`] bounds the
    /// tile in all of them — which is what "changes nothing else" has to mean
    /// for the rows to be readable against each other, and what
    /// `a_value_weighted_where_it_lies_is_a_staged_one_bit_for_bit` holds to the
    /// bits.
    ///
    /// **The staged rows stop where a tile stops fitting**, which is what makes
    /// them three rather than twelve: a threadgroup that stages copies a whole
    /// tile whatever it declared, so an arm below [`TILED_VALUES`] would be a
    /// walk writing past its own array rather than a smaller staging. The
    /// declaration is what this sweeps and the staging is what it cannot.
    ///
    /// **19 KiB against 19 KiB is the row worth reading twice.** The two arms
    /// are within 0.2% there, on a walk whose values come from threadgroup
    /// memory in one and from device memory in the other — so at this shape the
    /// staging buys nothing at all and the whole of what it does to the row is
    /// declare 16 KiB. That is what leaves an unsplit call free to declare
    /// something else instead. It is not true at a decode step's shape, which is
    /// why the other entry exists.
    ///
    /// **The two shipped entries are rows of this table**, at 12.5 KiB staging
    /// nothing and 19 KiB staging a tile.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn how_many_threadgroups_of_a_prefills_attention_a_core_holds() {
        let Some(device) = device() else { return };
        /// Floats a threadgroup declares beside its four live arrays: finely
        /// across the turn, and past it at either end.
        const DECLARED: [usize; 14] = [
            256, 1024, 1536, 2048, 2112, 2176, 2432, 2560, 2624, 2688, 3072, 4096, 5120, 7168,
        ];

        let most = device.most_threadgroup_bytes();
        let cells = bound_cells();
        let header: String = cells
            .iter()
            .map(|(tokens, _, what)| format!("{:>17}", format!("{tokens} {what}")))
            .collect();
        eprintln!(
            "  a threadgroup may declare {} KiB of a core's own memory",
            most / 1024
        );
        eprintln!("  {:<15}{:>14}{header}", "the values", "a threadgroup");

        for floats in DECLARED {
            for (what, arm) in [
                (
                    "where they lie",
                    super::source(STAGED_BY_AN_UNSPLIT_CALL, floats),
                ),
                ("staged", super::source(floats, NO_RESIDENCY)),
            ] {
                if what == "staged" && floats < TILED_VALUES {
                    continue;
                }
                let attention =
                    FusedAttention::from_source(&device, &arm).expect("the arm compiles");
                let held = attention.global.threadgroup_memory();
                let taken: String = cells
                    .iter()
                    .map(|&(tokens, sliding, _)| {
                        let each = a_prefill_costs(&device, &attention, tokens, sliding);
                        format!("{:>17}", format!("{each:.2?}"))
                    })
                    .collect();
                eprintln!(
                    "  {what:<15}{:>14}{taken}",
                    format!("{:.2} KiB", held as f64 / 1024.0),
                );
            }
        }
    }

    /// A layer whose span grows past what it starts with, and a sequence long
    /// enough to make it: `LEAST_KEYS` slots and a few more.
    ///
    /// Narrow heads and a small band, because what this is about is the span
    /// rather than the arithmetic — the arithmetic is settled above against
    /// mlx-vlm's own masks.
    fn streaming(keys: usize) -> (Case, Vec<f32>, Vec<f32>) {
        let (heads, kv_heads, d_rel, extent) = (4, 2, 8, 256);
        let config = AttentionConfig {
            hidden: heads * HEAD_DIM,
            heads,
            kv_heads,
            head_dim: HEAD_DIM,
            d_rel,
            sliding: 0,
            rms_norm_eps: 1e-6,
            log_scaling: None,
        };
        // `[keys, kv_heads * head_dim]`, which is the layout the projections
        // produce and the span is appended from.
        let (k, v) = (
            values(keys * kv_heads * HEAD_DIM, 2),
            values(keys * kv_heads * HEAD_DIM, 3),
        );
        let case = Case {
            name: format!("a sequence of {keys} keys"),
            proj: values(d_rel * extent, 4),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            rel: Vec::new(),
            taus: None,
            config,
            queries: 1,
            keys,
            q_offset: 0,
            mask: None,
        };
        (case, k, v)
    }

    /// One call through the span the layer keeps: the keys appended, the query
    /// and the relative features handed over as buffers, and the answer read
    /// back.
    fn through_the_span(
        device: &Device,
        layer: &LayerAttention<'_>,
        held: usize,
        k: &[f32],
        v: &[f32],
        q: &[f32],
        rel: &[f32],
    ) -> Vec<f32> {
        let rows = k.len() / (layer.config().kv_channels());
        layer.hold(held, rows).expect("the span grows");
        layer.append(k, v);

        let mut batch = device.batch().expect("a command buffer opens");
        let mut q = device.buffer(q).expect("the query uploads");
        let mut rel = device.buffer(rel).expect("the features upload");
        let out = layer
            .encode_over(&mut batch, &mut layer.span(), &mut q, &mut rel, None, held)
            .expect("the step encodes");
        batch.wait().expect("the batch completes");
        out.to_vec()
    }

    /// **The claim the residency rests on.** A sequence fed to a layer a chunk
    /// at a time, against the same keys and values handed over whole on every
    /// call, which is what [`LayerAttention::encode`] still does for a caller
    /// holding its own copy of the span.
    ///
    /// Exact equality rather than a tolerance, because it has to be: the kernel
    /// reads the same floats in the same order either way, and the only thing
    /// the span changes is the stride between one KV head's keys and the next.
    /// A difference of even a bit would mean the stride reached the arithmetic.
    ///
    /// The chunks straddle the growth: the span starts with `LEAST_KEYS` slots
    /// and the sequence outruns them, so the reallocation and the copy of every
    /// key already there are on the path this measures.
    #[test]
    fn a_span_the_layer_keeps_answers_the_span_handed_over_whole() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let keys = LEAST_KEYS + 6;
        let (case, k, v) = streaming(keys);
        let layer = case.wrapped(&device, &attention);
        let (heads, kv_heads) = (case.config.heads, case.config.kv_heads);
        let (head_dim, d_rel) = (case.config.head_dim, case.config.d_rel);
        let channels = kv_heads * head_dim;

        // A prefill that fills most of the span, then decode steps that outrun
        // it — which is the shape a generation has and the one that reallocates
        // mid-sequence.
        let chunks: Vec<usize> = std::iter::once(LEAST_KEYS - 3)
            .chain(std::iter::repeat_n(1, 9))
            .collect();
        assert_eq!(chunks.iter().sum::<usize>(), keys);

        let mut held = 0;
        for (chunk, rows) in chunks.iter().enumerate() {
            let (q, rel) = (
                values(heads * rows * head_dim, 5 + chunk),
                values(rows * heads * d_rel, 9 + chunk),
            );
            let span = held * channels..(held + rows) * channels;
            let streamed =
                through_the_span(&device, &layer, held, &k[span.clone()], &v[span], &q, &rel);
            held += rows;
            assert_eq!(layer.held(), held);

            let whole = layer
                .forward(Step {
                    q: &q,
                    k: &split_heads(&k[..held * channels], kv_heads, head_dim),
                    v: &split_heads(&v[..held * channels], kv_heads, head_dim),
                    rel: &rel,
                    taus: None,
                    q_offset: held - rows,
                })
                .expect("the dispatch completes");
            assert_eq!(streamed, whole, "chunk {chunk} of {}", case.name);
        }
        assert!(held > LEAST_KEYS, "the span never had to grow");
    }

    /// A sequence that has seen nothing is a sequence starting, and the span is
    /// emptied for it — so what the last sequence left behind cannot reach this
    /// one's answer.
    ///
    /// The two sequences are given different keys, which is what makes a span
    /// that carried over a wrong answer rather than the same one.
    #[test]
    fn a_sequence_that_has_seen_no_keys_starts_the_span_over() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let (case, k, v) = streaming(8);
        let layer = case.wrapped(&device, &attention);
        let (heads, head_dim) = (case.config.heads, case.config.head_dim);
        let channels = case.config.kv_channels();
        let (q, rel) = (
            values(heads * head_dim, 5),
            values(heads * case.config.d_rel, 9),
        );

        let first = &k[..4 * channels];
        let got = through_the_span(&device, &layer, 0, first, &v[..4 * channels], &q, &rel);
        assert_eq!(layer.held(), 4);

        // A second sequence over the same four keys, after the first has left
        // its own behind.
        let alone = through_the_span(&device, &layer, 0, first, &v[..4 * channels], &q, &rel);
        assert_eq!(layer.held(), 4, "the span started over");
        assert_eq!(alone, got);
    }

    /// Keys a rejected speculative token left in the span are keys nobody
    /// indexes, and the answer says so: attending after a rewind is attending
    /// over what the sequence had before those keys.
    ///
    /// Exact equality, because the kernel's loop bound is what a rewind moves
    /// and the floats it reads are the same ones. The slots the rejected keys
    /// wrote are left where they are — the next call overwrites them, and until
    /// it does nothing reaches them.
    #[test]
    fn keys_a_rewind_took_back_are_keys_the_step_does_not_attend_over() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let (case, k, v) = streaming(8);
        let layer = case.wrapped(&device, &attention);
        let (heads, head_dim) = (case.config.heads, case.config.head_dim);
        let channels = case.config.kv_channels();
        let (q, rel) = (
            values(heads * head_dim, 5),
            values(heads * case.config.d_rel, 9),
        );

        let (kept, rejected) = (4, 2);
        let of = |range: std::ops::Range<usize>| {
            (
                k[range.start * channels..range.end * channels].to_vec(),
                v[range.start * channels..range.end * channels].to_vec(),
            )
        };
        let (first_k, first_v) = of(0..kept);
        let (wrong_k, wrong_v) = of(kept..kept + rejected);

        through_the_span(&device, &layer, 0, &first_k, &first_v, &q, &rel);
        let want = through_the_span(&device, &layer, kept, &wrong_k, &wrong_v, &q, &rel);
        assert_eq!(layer.held(), kept + rejected);

        layer.rewind(rejected);
        assert_eq!(layer.held(), kept, "the span gave the keys back");
        let got = through_the_span(&device, &layer, kept, &wrong_k, &wrong_v, &q, &rel);
        assert_eq!(got, want, "the same keys again are the same answer");

        layer.rewind(rejected);
        let alone = through_the_span(&device, &layer, kept, &first_k, &first_v, &q, &rel);
        assert_ne!(alone, want, "keys the step attended over regardless");
    }

    /// And a sequence whose count the span does not match is two sequences
    /// through one layer, which is refused where it is asked for rather than
    /// answered over the other one's keys.
    #[test]
    #[should_panic(expected = "a sequence at 7 keys against a span holding 4")]
    fn a_span_holding_another_sequences_keys_is_refused() {
        let Some(device) = device() else {
            panic!("a sequence at 7 keys against a span holding 4")
        };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let (case, k, v) = streaming(8);
        let layer = case.wrapped(&device, &attention);
        let channels = case.config.kv_channels();

        layer.hold(0, 4).expect("the span grows");
        layer.append(&k[..4 * channels], &v[..4 * channels]);
        layer.hold(7, 1).expect("the span grows");
    }

    /// The two shapes a wrapped layer refuses, both of which the kernel would
    /// otherwise index past rather than fault on — a GPU read off the end of a
    /// buffer answers with whatever is there.
    #[test]
    fn a_shape_the_kernel_would_index_past_is_refused_where_it_is_handed_over() {
        let Some(device) = device() else { return };
        let attention = FusedAttention::new(&device).expect("the kernel compiles");
        let case = synthetic("decode");
        let wrapped = |config, proj: &[f32]| {
            LayerAttention::new(&device, &attention, config, proj).map(|_| ())
        };

        let err = wrapped(case.config, &case.proj[..case.proj.len() - 1])
            .expect_err("a partial band is refused");
        assert!(matches!(err, AttentionError::PartialBand { d_rel, .. } if d_rel == 16));

        // Two heads over three KV heads: the group is `2 / 3` — zero — so every
        // query head would read past the keys rather than share one.
        let err = wrapped(
            AttentionConfig {
                heads: 2,
                kv_heads: 3,
                ..case.config
            },
            &case.proj,
        )
        .expect_err("heads that do not group are refused");
        assert!(matches!(err, AttentionError::UngroupedHeads { heads, .. } if heads == 2));

        assert!(wrapped(case.config, &case.proj).is_ok());
    }
}
