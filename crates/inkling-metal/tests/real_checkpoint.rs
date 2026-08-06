//! The packed matmul against a real Inkling-Small checkpoint, which is far too
//! large to commit. Set `INKLINGRS_CHECKPOINT` to a checkpoint directory to run
//! these; unset, each reports a skip and passes. `just test-full` sets it, to an
//! absolute path — a relative one resolves against each test process's own
//! working directory and fails every one of them.
//!
//! What the hermetic cases in `matmul::tests` cannot settle is the shape and the
//! contents of a trained weight. `lm_head` is `[201024, 4096]` — 411 MB of codes
//! under 26 MB of scales, 3.3 GB once decoded — so only a real checkpoint
//! carries a tensor whose group scales, code distribution and sheer reduction
//! count are the ones the engine will actually run against, and only a real
//! checkpoint asks whether one dispatch of that size survives the GPU watchdog.
//!
//! It is also the only place the two backends can be run against each other in
//! anger. `inkling-core` cannot reach a Metal device — the dependency points the
//! other way — so the engine driven from end to end with its head on the GPU is
//! a case that has to live here, beside the kernel it is measuring.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use inkling_core::fixture::{self, ACTIVATIONS, deviation, indices};
use inkling_core::generate::{Picked, Proposer, Round};
use inkling_core::model::Batched;
use inkling_core::moe::{Gate, GateWeights, MoeConfig, SparseMoe};
use inkling_core::mtp::{CheckpointHeads, MtpProposer};
use inkling_core::ops::linear;
use inkling_core::profile::{self, Op, Profile};
use inkling_core::quant::{BITS, dequantize_blocks_into};
use inkling_core::workload::{DECODED, STRUCTURED_PROMPT, SWEPT, tiled};
use inkling_core::{
    AttentionCache, AttentionStep, BandedMask, Bf16, Checkpoint, CheckpointWeights, Dtype, Ending,
    LayerStep, ModelCache, Packed as CorePacked, Projections, Sdpa, ShortConv, Tail, TensorView,
    Tokenizer, split_heads,
};
// The ceiling every "of peak" column here is a fraction of, read from the crate
// rather than written down again: `what_a_prefills_blocked_matmul_is_bound_by`
// divides its roofline by the same figure, and two spellings are two that can
// drift. `what_a_streaming_read_achieves_on_this_machine` is what measures it,
// on whatever host it runs on.
use inkling_metal::kernel::MEMORY_BANDWIDTH;
use inkling_metal::ordering::Groups;
use inkling_metal::trace::{self, Difference, Encoded};
use inkling_metal::{
    DISPATCHES_A_SUBMISSION, DenseMatmul, DenseWeight, Device, ExpertGrouping, ExpertKernels,
    LayerKernels, LayerProjections, LayerRouter, MetalError, ModelHeads, ModelLayers, ModelTail,
    MoeCombine, Numerics, PackedBank, PackedMatmul, PackedProjection, RoundTrip, Router,
    RouterWeights, StackShape, SwiGlu, TailWeights, WANTED_GROUPS,
};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

/// Which numerics the measurements in this file run the engine under.
///
/// **A variable rather than an argument, because these are `#[ignore]`d
/// measurements and libtest has no way to pass one.** The per-kernel table below
/// is the arbiter for what a change to a dispatch is worth, and a change behind
/// [`Numerics`] can only be arbitrated by the same table taken on the other
/// side of it — so the same run has to be reachable both ways:
///
///     INKLINGRS_NUMERICS=production just test-timing
///
/// **Unset is the reference**, which is what every figure this file records was
/// taken under and what a run that says nothing gets.
const NUMERICS_VAR: &str = "INKLINGRS_NUMERICS";

/// The numerics this process measures under, refused rather than defaulted where
/// the word is not one — a table headed "reference" that ran something else is
/// the one failure mode worth a panic here.
fn numerics() -> Numerics {
    match std::env::var(NUMERICS_VAR) {
        Err(_) => Numerics::default(),
        Ok(name) => Numerics::parse(&name).unwrap_or_else(|| {
            let every: Vec<&str> = Numerics::EVERY.into_iter().map(Numerics::named).collect();
            panic!("{NUMERICS_VAR}={name} is none of {}", every.join(", "))
        }),
    }
}

/// The projection this measures, which is the largest in the model and the one
/// M3 routes through the kernel first.
const LM_HEAD: &str = "language_model.lm_head";

/// One MoE layer's routed bank, `[256, 2048, 4096]`, for the other shape a
/// projection comes in.
const ROUTED_EXPERTS: &str = "language_model.model.layers.2.mlp.switch_mlp.gate_proj";

/// The same layer's router gate, `[258, 4096]`, which is the one weight a
/// matmul reads that the quantiser left in bfloat16.
const ROUTER_GATE: &str = "language_model.model.layers.2.mlp.gate_weight";

const HIDDEN: usize = 4096;

/// The layer whose whole attention is driven below.
///
/// A *global* one — layers 5, 11, 17 and up are full attention — because its
/// band reaches every key an eight-token sequence has, where a sliding layer's
/// 512-token window covers all of them and covers nothing else. The layer's MLP
/// is not reached here; what is wrapped is its five attention projections.
const LAYER: usize = 5;

const CODES_PER_WORD: usize = u32::BITS as usize / BITS;

/// How far a dispatch may land from the CPU's answer over trained weights.
///
/// The same account as `matmul::tests::TOLERANCE`, and a tighter number.
/// Decoding is exact on both sides, so summation order is all that separates
/// them, and what separates them is the *CPU's* drift over a serial 4096-long
/// f32 reduction rather than the kernel's over a tree — so this is a bound on
/// the oracle rather than on the kernel either way.
///
/// A sixth of the synthetic bound, and knowably so rather than by luck. A
/// serial reduction's drift is set by how far its running sum wanders from its
/// terms, and `lm_head`'s scale bytes span `0x74..=0x7e` across the whole
/// tensor while the 128 groups *within* a row span a median of one byte. A
/// trained row's terms are therefore closer in magnitude than a synthetic row's,
/// and the serial loop gives up an order of magnitude less on them.
///
/// Worst observed when this landed: 2.9e-7 over eight rows of 201024 outputs,
/// and 1.1e-7 over one routed expert. Over fifty sampled head rows the kernel
/// drifts 3.3e-8 from an f64 accumulation of the same products where the CPU
/// drifts 1.6e-7, so the kernel is the closer of the two by a factor of five and
/// what the two disagree by is the CPU's error arriving whole. 1e-6 is a factor
/// of three and a half in hand.
///
/// Against the weakest mutation it has to catch — each packed word's eight codes
/// read from the top down, which is the one fact about the format a decoder can
/// invert while still producing weights of the right magnitude — 3.2, seven
/// decades above.
const TOLERANCE: f32 = 1e-6;

/// How far the dense matmul may land from the CPU's answer over a trained
/// bfloat16 gate.
///
/// The same account again, and a looser number than [`TOLERANCE`] for a reason
/// the format decides. Widening bfloat16 is exact on both sides, so summation
/// order is still all that separates them — but a gate's values carry no block
/// scale to hold a row's terms near each other, and the trained ones span
/// decades within a single row where a packed row's groups span a byte. A
/// serial f32 reduction gives up more on terms that far apart, and it is the
/// serial reduction that gives it up: the kernel sums 128 products a lane and
/// reduces 32 lanes in a tree.
///
/// Worst observed when this landed: 7.1e-7 over eight rows of 258 outputs,
/// which leaves a factor of four in hand. Against the mutation it has to catch
/// — a value's two bytes read the other way round, which is the one fact about
/// the format a kernel can invert — `dense::tests` measures 4.0, six decades
/// above.
const GATE_TOLERANCE: f32 = 3e-6;

/// How far a layer answered whole may land from the same layer's pieces run
/// apart.
///
/// The two run the same five projections through the same kernel and the same
/// attention step through the same kernel, so neither the multiplies nor the
/// softmax is what separates them. What does is the two short convolutions:
/// on one side they are a dispatch and on the other they are
/// `LayerStep::convolved` here, and Metal compiles `acc += w * v` with fast math
/// on and may contract it to an FMA — one rounding a tap where the CPU takes
/// two. Every kernel in this tree is held to the CPU within a bound for a reason
/// of this shape; this one is four taps wide and then goes through a softmax.
///
/// Worst observed when this landed: 2.7e-7, about four f32 ulps of the layer's
/// output peak, which leaves a factor of three and a half. What the bound has to
/// tell that from is a span read under the wrong stride, and the assertion below
/// says what that scale is: the same row against a layer that has seen nothing
/// lands 1.1 away, six decades above.
const LAYER_TOLERANCE: f32 = 1e-6;

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os(CHECKPOINT_VAR).map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: {CHECKPOINT_VAR} is unset");
    }
    dir
}

/// The device, or `None` with a reported skip — the same bargain the crate's own
/// tests strike, for a machine that has no Metal device.
fn device() -> Option<Device> {
    match Device::open() {
        Ok(device) => Some(device),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: this machine has no Metal device");
            None
        }
        Err(err) => panic!("the default device opens: {err}"),
    }
}

/// One packed tensor of the checkpoint, as the two views that pair.
struct Packed<'a> {
    codes: TensorView<'a>,
    scales: TensorView<'a>,
}

impl<'a> Packed<'a> {
    fn open(ckpt: &'a Checkpoint, name: &str) -> Self {
        let of = |suffix: &str| {
            ckpt.tensor(&format!("{name}.{suffix}"))
                .unwrap_or_else(|err| panic!("checkpoint holds {name}.{suffix}: {err}"))
        };
        let (codes, scales) = (of("weight"), of("scales"));
        assert_eq!(codes.dtype(), Dtype::U32, "{name} is packed");
        assert_eq!(scales.dtype(), Dtype::U8, "{name}'s scales are bytes");
        Self { codes, scales }
    }

    fn out_dim(&self) -> usize {
        self.codes.shape()[0]
    }

    fn in_dim(&self) -> usize {
        self.codes.data().len() * 2 / self.out_dim()
    }

    /// The packed bytes of one weight row, and the scale bytes that go with it.
    fn row(&self, index: usize) -> (&'a [u8], &'a [u8]) {
        let stride = |view: &TensorView<'a>| view.data().len() / self.out_dim();
        let (codes, scales) = (stride(&self.codes), stride(&self.scales));
        (
            &self.codes.data()[index * codes..][..codes],
            &self.scales.data()[index * scales..][..scales],
        )
    }

    fn upload<'m>(&self, device: &'m Device, matmul: &'m PackedMatmul) -> PackedProjection<'m> {
        PackedProjection::upload(
            device,
            matmul,
            self.in_dim(),
            self.out_dim(),
            self.codes.data(),
            self.scales.data(),
        )
        .expect("the checkpoint's two tensors pair")
    }
}

/// How each packed row is read before it is multiplied, so that a mutation can
/// be a different reading of the same bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reading {
    /// The format as `inkling_core::quant` documents it: within a little-endian
    /// word, code `i` occupies bits `4i..4i+4`.
    Documented,
    /// Each word's eight codes read from the top down — code `i` taken from
    /// where code `7 - i` lives. The same mutation
    /// `matmul::tests::reading_each_words_nibbles_from_the_top_down_is_a_different_answer`
    /// makes to the kernel, made here to the bytes instead, so that what the
    /// bound is measured against on trained weights is the bug it is measured
    /// against on synthetic ones.
    WordsReversed,
}

/// One packed word with its nibble order inverted: the bytes reversed, which
/// exchanges the four pairs, and then each byte's own two nibbles exchanged.
fn reverse_words(codes: &[u8]) -> Vec<u8> {
    codes
        .chunks_exact(size_of::<u32>())
        .flat_map(|word| {
            let mut word: [u8; 4] = word.try_into().expect("chunked into words");
            word.reverse();
            word.map(|byte| byte.rotate_right(4))
        })
        .collect()
}

/// `x @ wᵀ` on the CPU, a weight row at a time.
///
/// This is the oracle, and it is the path the engine runs today: one row of the
/// head decoded into a buffer, multiplied against every row of `x` through
/// [`linear`], and dropped. Decoding the whole tensor first would be 3.3 GB and
/// would not be what M3 is replacing.
fn on_the_cpu(packed: &Packed<'_>, x: &[f32], reading: Reading) -> Vec<f32> {
    let (in_dim, out_dim) = (packed.in_dim(), packed.out_dim());
    let rows = x.len() / in_dim;
    let mut out = vec![0.0; rows * out_dim];
    let mut weight = vec![0.0; in_dim];

    for col in 0..out_dim {
        let (codes, scales) = packed.row(col);
        let read = match reading {
            Reading::Documented => codes.to_vec(),
            Reading::WordsReversed => reverse_words(codes),
        };
        dequantize_blocks_into(&read, scales, &mut weight)
            .unwrap_or_else(|err| panic!("head row {col} decodes: {err}"));

        for (row, value) in linear(x, &weight, in_dim).into_iter().enumerate() {
            out[row * out_dim + col] = value;
        }
    }
    out
}

/// The same multiply accumulated in f64, over the rows the caller names.
///
/// Neither side computes this. Decoding is exact on both — a table lookup times
/// a power of two — so the products are the same f32s either way and summation
/// order is the only thing left to differ about; accumulating with 29 bits of
/// headroom is what says which of the two orders is drifting, and so whether a
/// disagreement is float noise or a bug.
///
/// Over a sample rather than the whole tensor, because 201024 rows in f64 buys
/// nothing the worst of a few thousand does not already say.
fn exactly(packed: &Packed<'_>, x: &[f32], cols: &[usize]) -> Vec<f64> {
    let in_dim = packed.in_dim();
    let mut weight = vec![0.0; in_dim];
    let mut out = Vec::with_capacity(cols.len());

    for col in cols {
        let (codes, scales) = packed.row(*col);
        dequantize_blocks_into(codes, scales, &mut weight)
            .unwrap_or_else(|err| panic!("head row {col} decodes: {err}"));
        out.push(
            x.iter()
                .zip(&weight)
                .map(|(x, w)| f64::from(*x) * f64::from(*w))
                .sum::<f64>(),
        );
    }
    out
}

/// How far an answer lands from the exact one, as a fraction of the exact
/// values' peak.
fn drift(got: &[f32], exact: &[f64]) -> f64 {
    assert_eq!(got.len(), exact.len(), "length");
    let scale = exact.iter().fold(0.0f64, |peak, w| peak.max(w.abs()));
    got.iter().zip(exact).fold(0.0f64, |worst, (got, exact)| {
        worst.max((f64::from(*got) - exact).abs())
    }) / scale
}

/// The hidden state the reference's own forward pass ended with, which is what
/// the head is driven by in anger. `[8, 4096]`.
fn normed_state() -> Vec<f32> {
    fixture::f32s(&fixture::tensor(&fixture::open(ACTIVATIONS), "norm_out"))
}

/// One row of it, which is the decode-step shape and the width every projection
/// below the head takes.
fn x_row() -> Vec<f32> {
    normed_state()[..HIDDEN].to_vec()
}

/// Every 4093rd row of the head, which over 201024 rows is 50 spread across the
/// whole tensor and prime to every stride the layout has.
fn sampled_rows(out_dim: usize) -> Vec<usize> {
    (0..out_dim).step_by(4093).collect()
}

/// What the hermetic cases cannot settle: that a trained `[201024, 4096]` weight,
/// with the group scales and the code distribution quantisation actually
/// produced, multiplies to what the CPU makes of the same bytes.
///
/// Eight rows of `x` rather than one, because the reference's own normed state
/// is eight and because a kernel that took its row index off the wrong axis
/// would still fill the buffer.
#[test]
fn the_packed_matmul_reproduces_the_cpu_over_the_real_head() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");

    let head = Packed::open(&ckpt, LM_HEAD);
    assert_eq!(head.in_dim(), HIDDEN, "the head's input width");
    let x = normed_state();
    let rows = x.len() / HIDDEN;
    assert!(rows > 1, "one row would not settle the row indexing");

    let started = Instant::now();
    let got = head
        .upload(&device, &matmul)
        .multiply(&x)
        .expect("the dispatch completes");
    eprintln!(
        "{rows} x {} through [{}, {HIDDEN}] on the GPU in {:?}",
        HIDDEN,
        head.out_dim(),
        started.elapsed()
    );

    let started = Instant::now();
    let want = on_the_cpu(&head, &x, Reading::Documented);
    eprintln!("the same on the CPU in {:?}", started.elapsed());

    let worst = deviation(&got, &want);
    assert!(
        worst > 0.0,
        "an exact match would mean the two are not summing independently"
    );

    // Which of the two is drifting, over a sample of the head's rows against the
    // first row of `x`. The kernel sums 128 products a lane and reduces 32 lanes
    // in a tree where the CPU sums 4096 serially, so the kernel has to be the
    // closer of the two to the exact answer — a dispatch that merely sat inside
    // the bound while drifting further than a serial f32 loop would be one
    // hiding a mistake in a tolerance.
    let cols = sampled_rows(head.out_dim());
    let exact = exactly(&head, &x[..HIDDEN], &cols);
    let sample = |answer: &[f32]| -> Vec<f32> {
        let first_row = &answer[..head.out_dim()];
        cols.iter().map(|col| first_row[*col]).collect()
    };
    let (mine, theirs) = (drift(&sample(&got), &exact), drift(&sample(&want), &exact));
    eprintln!(
        "worst deviation {worst:e}; over {} sampled rows the kernel drifts {mine:e} against \
         the CPU's {theirs:e}",
        cols.len()
    );

    assert!(worst <= TOLERANCE, "deviation {worst:e}");
    assert!(mine < theirs, "{mine:e} against the CPU's {theirs:e}");

    // The nibble order, which is the one fact about the format a decoder can
    // invert while still producing weights of the right magnitude — and so the
    // weakest mutation the bound above has to be able to tell from float noise.
    let reversed = deviation(&got, &on_the_cpu(&head, &x, Reading::WordsReversed));
    eprintln!("against the head read from the top nibble down: {reversed:e}");
    assert!(reversed > TOLERANCE, "deviation {reversed:e}");
}

/// What one dispatch of the largest projection in the model costs, which is the
/// number M3 is buying.
///
/// `[1, 4096] @ [201024, 4096]ᵀ` is the decode-step shape: one token, every row
/// of the head. It is also the shape `Device::run`'s watchdog note is about —
/// one command buffer doing the whole projection is exactly what the GPU
/// watchdog stops if it takes too long, and whether it does is a measurement
/// rather than a thing to pre-emptively tile around. If it ever starts to, this
/// fails with `kIOGPUCommandBufferCallbackErrorTimeout` rather than hanging.
///
/// Nothing here asserts a speed. What is asserted is that the dispatch completes
/// and that it is not somehow slower than the loop it replaces; the numbers go
/// to stderr for the commit message to quote.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn one_dispatch_does_an_lm_head_shaped_multiply_without_meeting_the_watchdog() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");

    let head = Packed::open(&ckpt, LM_HEAD);
    let x = x_row();
    let weight_bytes = head.codes.data().len() + head.scales.data().len();

    let started = Instant::now();
    let projection = head.upload(&device, &matmul);
    let uploaded = started.elapsed();

    // Twice, and the second is what is reported: the first dispatch of a fresh
    // pipeline pays for the driver's first look at these buffers, which is a
    // cost a decode loop pays once and not per token.
    let mut dispatched = std::time::Duration::ZERO;
    for _ in 0..2 {
        let started = Instant::now();
        projection.multiply(&x).expect("the dispatch completes");
        dispatched = started.elapsed();
    }

    let started = Instant::now();
    on_the_cpu(&head, &x, Reading::Documented);
    let on_the_cpu = started.elapsed();

    let gib = (1u64 << 30) as f64;
    eprintln!(
        "[1, {HIDDEN}] @ [{}, {HIDDEN}]^T: {:.2} GiB of packed weights uploaded in {uploaded:?}, \
         dispatched in {dispatched:?} ({:.0} GB/s of weights consumed), against {on_the_cpu:?} on \
         the CPU — {:.0}x",
        head.out_dim(),
        weight_bytes as f64 / gib,
        weight_bytes as f64 / dispatched.as_secs_f64() / 1e9,
        on_the_cpu.as_secs_f64() / dispatched.as_secs_f64(),
    );
    assert!(dispatched < on_the_cpu, "the kernel bought nothing");
}

/// How many tokens the end-to-end case decodes, which is the whole of what the
/// fixture recorded and what `inkling-core` asserts on the CPU path.
const GENERATED: usize = 8;

/// What the whole process may hold resident with the whole model on the device.
///
/// A wrapped weight is read by the GPU through the mapping, and those pages do
/// not join *this process's* resident set the way a `dequantize_blocks_into` of
/// the same bytes does — so each handover has taken the peak down rather than
/// up. Measured over the same eight-token generation: 20.77 GiB with only the
/// head on the device, 2.44 GiB with the experts there too, 0.19 GiB once the
/// layers' own projections and input layernorms were, and 0.12 GiB now that the
/// routers' gates are.
///
/// **32 GiB was slack and 1 GiB is a claim.** What is left resident is what the
/// CPU still reads: the embedding rows a prompt asked for, and every layer's
/// bfloat16 tensors — which are now held widened rather than widened again on
/// every step. Everything else is pages the GPU reads where the checkpoint
/// mapped them. The 981 MB layer scratch is still allocated and is never
/// written, so it never faults in either.
///
/// Holding those widened is where the bound had to be re-derived rather than
/// waved at, and where the derivation then paid for itself. Widening every
/// layer's bfloat16 tensors at load took the peak from 0.13 GiB to 0.30, and
/// the 179 MB between them was arithmetic rather than a measurement: forty
/// routers' `[258, 4096]` gates are 169 MB of float32 — 95% of it — and the
/// norms, convolution kernels and relative-position projections of forty-two
/// layers are the other 9.8 MB.
///
/// The gates then went to the device as the bfloat16 they are, so on this path
/// nothing widens them at all and the peak is 0.12 GiB — under where it started.
/// What that says about the other 9.8 MB is that it was never the term.
///
/// A bound this tight is a regression test with a name: a path that went back
/// to decoding a layer's projections into that scratch lands at 2.44 GiB and
/// fails here. What the number still does not measure is how much memory the
/// machine is using — the pages are in the unified buffer cache either way, and
/// what changed is whose they are.
///
/// **The keys and values are the first thing under this bound that grows with
/// the sequence**, and the first that is allocated rather than mapped. Each
/// layer keeps a `[kv_heads, capacity, head_dim]` float32 span of each — 4 KB a
/// key slot a layer for the pair at Inkling's shape — so the 64 slots a span
/// starts with are 21 MB across the stack whatever the prompt is, and a
/// generation past 64 tokens doubles that and doubles it again. It is not new
/// memory: what it replaces is the same span in a `Vec<f32>` the CPU path grew
/// beside a copy of all of it made onto the device on every layer of every
/// step. Measured over the same eight-token generation the peak went 0.13 GiB
/// to 0.14, which is the 64 slots minus the 16 keys' worth of vector. The same
/// generation measures 0.17 now, 0.15 of which it was before a layer took the
/// two residual paths' four convolution windows and the buffers between their
/// dispatches.
///
/// So the bound now covers a *span*, and what it can no longer be read as is a
/// claim about a long context: at 4096 tokens the two spans are 1.4 GiB and
/// this fails — correctly, because at that point the resident set is the KV
/// cache and the number a bound should hold is the cache's own, not this one.
/// Eight tokens is what this case generates and 1 GiB is four decades of
/// headroom over what they cost.
const RESIDENT_BOUND: u64 = 1 << 30;

/// What one decode step dispatches, and how many command buffers it submits
/// them in, from the shape of the model rather than as two numbers to keep in
/// step with it.
///
/// Both are asserted rather than printed, and the second is the one this
/// milestone's last commit is about. **A step that stopped batching would
/// dispatch exactly as much and submit twice as often**, so a bound on the
/// dispatches alone would watch the number that cannot change while the one
/// that pays for it doubled underneath.
///
/// **A whole layer is one submission**, and twenty-one dispatches on a layer
/// that routes. Nine are its attention: the input layernorm, the four
/// projections that consume what it produced, the one paired short convolution
/// behind `k` and `v`, the one paired head norm over `q` and the convolved `k`,
/// the attention step and `o_proj`. Three more are the two residual paths around the
/// MLP — the layer's two short convolutions, each of which adds the value its
/// block began with as it writes, and the second norm between them. The last
/// nine are the MLP: the router's gate, the top-k over what it produced, each
/// bank's pair and down, the one activation both banks share, the softmax over
/// the eight logits that selection named, and both banks' rows weighted by it. A
/// dense layer is sixteen, its feed-forward network four where a MoE layer's two
/// banks and the router around them are nine. The head is one of each.
///
/// **Four of those dispatches are a pair.** The banks' `gate` and `up` are one
/// where the call is untiled — [`PackedPair`](inkling_metal::PackedPair), and a
/// dense layer's are two projections that were never a bank. The key and value
/// convolutions are one; the query norm and the key norm are one wherever they
/// share a threadgroup; and the two banks' activations are one, which is what
/// the layer dispatching both pairs before either activation buys. Every one of
/// the four is two calls that read different rows against different weights into
/// different places, and none of them reads what the other writes.
///
/// **Not one of them costs a round trip.** The whole chain from the hidden state
/// a layer is handed to the one it passes on is buffers a next dispatch reads —
/// including the four values that outlive the call, three convolutions' windows
/// and the span of keys and values, which is why they had to become the layer's
/// before the rest could follow. And what a layer passes on is what the next
/// layer reads, so the command buffer does not end where the layer does either.
///
/// **What decides how many command buffers it does end in is
/// `DISPATCHES_A_SUBMISSION`**, and the same greedy rule is walked here: a run
/// commits at the first layer boundary past that many dispatches and carries on
/// encoding into the next buffer without waiting for it. So the count is a
/// scheduling decision rather than a property of the stack — which is exactly
/// why it is asserted, since a layer that started forcing a *wait* would leave
/// this number where it is and the step nowhere near it.
///
/// **The last layer is a layer with something after it**, and that is the four
/// dispatches added here and the submission that is not: the final norm,
/// `lm_head` and the two the argmax over its output is go into the buffer layer
/// 41 left open, so the run does not end at the layer and the head does not open
/// one of its own. Whether that buffer is the one layer 41 encoded into or a
/// fresh one is the same greedy rule, walked past the end of the stack rather
/// than stopping short of it.
///
/// A prefill wide enough that one layer reaches the bytes a run may hold is
/// still one a layer, and deliberately — see `ModelLayers::carries`, where what
/// a merged run holds is traded against what it saves. Such a call ends its run
/// at the last layer whatever the tail is, and the tail runs where it always
/// did. This counts a decode step, whose forty-two layers are far under that
/// budget.
fn per_step(layers: u64, dense: u64) -> (u64, u64) {
    let width = |layer: u64| if layer < dense { 12 + 4 } else { 12 + 9 };
    let (mut dispatches, mut encoded, mut submissions) = (0, 0, 0);
    for layer in 0..layers {
        dispatches += width(layer);
        encoded += width(layer);
        if encoded >= DISPATCHES_A_SUBMISSION as u64 {
            submissions += 1;
            encoded = 0;
        }
    }
    (dispatches + 4, submissions + 1)
}

/// What a run of the engine is asked for, since standing the model up is the
/// same whichever of the two regimes the case below is about.
///
/// **The profile follows from `generated` rather than being a knob beside it.**
/// A run that decodes nothing is a prefill and its one step is what there is to
/// charge; a run that decodes is charged its decode steps and the prefill in
/// front of them is cleared. A profile summed over both would describe neither:
/// the first step is hundreds of rows through the same kernels every later one
/// puts a single row through.
#[derive(Clone, Copy)]
struct Asked {
    /// How many tokens to prefill, or the recorded prompt's own length.
    prompt: Option<usize>,
    /// How many tokens the run produces, the prefill's own being the first.
    generated: usize,
    /// Decode steps thrown away after the prefill before the profile starts.
    ///
    /// **The step after a long prefill is not a decode step.** The cross-engine
    /// table records step 1 at 736 and 783 ms against medians of 32.6 and 36.4
    /// at the two long prompts: it is what the prefill deferred — 13741 MiB over
    /// 1278 buffers released when its command buffer completes — arriving on the
    /// step after it. A profile that charged it would put that release in the
    /// rows of every step it was divided by.
    settle: usize,
    /// Whether the device times each dispatch.
    sampling: bool,
    /// Whether every dispatch a step encodes is written down, so that two steps
    /// can be held against each other — see [`inkling_metal::trace`].
    tracing: bool,
}

impl Asked {
    /// The recorded prompt and the oracle's continuation, nothing sampling —
    /// what every case about a decode step starts from.
    fn decoding() -> Self {
        Self {
            prompt: None,
            generated: GENERATED,
            settle: 0,
            sampling: false,
            tracing: false,
        }
    }

    /// The same, over a prompt tiled to `prompt` tokens — **a decode step at a
    /// context somebody might have**, where [`Asked::decoding`] is one at the
    /// eight tokens the recorded prompt has.
    fn decoding_at(prompt: usize) -> Self {
        Self {
            prompt: Some(prompt),
            generated: 1 + SETTLED + CHARGED,
            settle: SETTLED,
            ..Self::decoding()
        }
    }

    /// One prefill of `prompt` tokens and no decode step at all.
    fn prefilling(prompt: usize) -> Self {
        Self {
            prompt: Some(prompt),
            generated: 1,
            settle: 0,
            sampling: false,
            tracing: false,
        }
    }

    fn sampled(self) -> Self {
        Self {
            sampling: true,
            ..self
        }
    }

    fn traced(self) -> Self {
        Self {
            tracing: true,
            ..self
        }
    }

    /// Whether the profile that comes back is the prefill's rather than the
    /// decode steps'.
    fn charges_the_prefill(&self) -> bool {
        self.generated == 1
    }

    /// Steps whose accounts are cleared rather than charged: the prefill, and
    /// whatever it deferred onto the steps behind it.
    fn discarded(&self) -> usize {
        1 + self.settle
    }
}

/// Decode steps a context's table throws away before it starts charging.
///
/// **Two at every length, including the ones that do not need it.** Only a long
/// prompt has a step to discard — the cross-engine table records step 1 at 736
/// and 783 ms at 385 and 769 tokens against medians of 32.6 and 36.4, where at
/// 97 the longest step is 25.55 ms and falls at step 125. Discarding a fixed two
/// is what keeps the three rows means over the same steps of the same
/// generation, which is the whole use they are put to; discarding by length
/// would make the shortest row a mean over something the other two are not.
///
/// Two rather than one because a span doubling is also a step nobody wants in a
/// mean, and where it falls is the prompt's rather than this constant's.
const SETTLED: usize = 2;

/// Decode steps a context's table is the mean of, which is what the recorded
/// prompt's own generation charges: [`GENERATED`] tokens are one prefill and
/// this many decode steps, so the two tables are means over the same number.
const CHARGED: usize = GENERATED - 1;

/// One run of the engine with every weight it has a kernel for on the device,
/// over the prompt the activation capture recorded.
///
/// Held together rather than driven twice, because standing the model up and
/// generating from it is what both of the cases below start with and neither is
/// about: one asks what came out, the other asks what it cost.
struct OnTheDevice {
    /// How many of the stack's layers had banks, projections and a dense
    /// feed-forward network wrapped, and what wrapping all of it took.
    expert_layers: usize,
    /// The first layer the checkpoint has banks for, which is where the dense
    /// ones stop.
    first_routed: usize,
    projection_layers: usize,
    dense_layers: usize,
    wrapped: Duration,
    prompt: usize,
    /// What this run was asked for, which is what says whether the profile
    /// below is the prefill's or the decode steps'.
    asked: Asked,
    /// What each step took, the prompt's prefill first.
    steps: Vec<Duration>,
    /// The running `(dispatches, submissions, allocations, allocated bytes)`
    /// the run opened at and each step was reached at, so there is one more of
    /// these than there are steps and a step is the difference across it.
    ///
    /// The bytes are the last of the four and the newest, and they are what a
    /// merged run is bounded by: a decode step's are what a run of forty-two
    /// layers holds until its one command buffer completes, which is the number
    /// `ModelLayers::carries` decides against a budget. The count of buffers
    /// cannot say it — a layer allocates the same buffers for one row as for
    /// seven hundred.
    submitted: Vec<(u64, u64, u64, u64)>,
    peak: u64,
    got: Vec<usize>,
    /// What the steps [`Asked::charges_the_prefill`] names spent, by operation,
    /// divided by however many of them there were. The other regime's accounts
    /// are cleared rather than folded in: a prefill dispatches the same kernels
    /// over hundreds of rows where a decode step puts one row through them.
    profile: Profile,
    /// Every command buffer those same steps waited for, cleared the same way —
    /// and one row per shape of submission where the profile has a sum, because
    /// a decode step's two are 1076 dispatches and one and the wait means
    /// something different in each.
    round_trips: Vec<RoundTrip>,
    /// What each step encoded, dispatch by dispatch, where the run was asked to
    /// write it down. One entry a step, the prefill's first.
    traces: Vec<Vec<Encoded>>,
    /// Where each traced step committed a command buffer, as indices into that
    /// step's own trace — which is what says how many of the gaps between its
    /// groups are barriers somebody has to encode. See
    /// [`Groups::over`](inkling_metal::ordering::Groups::over).
    boundaries: Vec<Vec<usize>>,
    /// Barriers the engine encoded over the steps the profile charges, which is
    /// what the division derived from those steps has to agree with.
    barriers: u64,
}

impl OnTheDevice {
    /// The run every case here starts from, with the device's own clock left
    /// where it is.
    fn generate(dir: &Path, device: &Device) -> Self {
        Self::running(dir, device, Asked::decoding())
    }

    /// The same run, with each dispatch timed on the device if `sampling`.
    ///
    /// **Sampling is a parameter rather than a setting**, because the only way
    /// to say what it costs is to run the same thing both ways — see
    /// `what_timing_each_dispatch_costs`.
    fn generating(dir: &Path, device: &Device, sampling: bool) -> Self {
        Self::maybe_sampling(dir, device, Asked::decoding(), sampling)
    }

    /// One prefill of `prompt` tokens and nothing after it, which is the step
    /// the cases below are about.
    fn prefilling(dir: &Path, device: &Device, prompt: usize, sampling: bool) -> Self {
        Self::maybe_sampling(dir, device, Asked::prefilling(prompt), sampling)
    }

    /// The decode steps after a prompt tiled to `prompt` tokens, which is the
    /// same regime [`OnTheDevice::generating`] charges at a context a user would
    /// actually have.
    fn decoding_at(dir: &Path, device: &Device, prompt: usize, sampling: bool) -> Self {
        Self::maybe_sampling(dir, device, Asked::decoding_at(prompt), sampling)
    }

    fn maybe_sampling(dir: &Path, device: &Device, asked: Asked, sampling: bool) -> Self {
        Self::running(dir, device, if sampling { asked.sampled() } else { asked })
    }

    fn running(dir: &Path, device: &Device, asked: Asked) -> Self {
        device
            .time_each_dispatch(asked.sampling)
            .expect("the device times a dispatch");
        let kernels =
            LayerKernels::compiling(device, numerics()).expect("the layer kernels compile");
        let matmul = kernels.matmul();
        let dense = DenseMatmul::new(device).expect("the dense matmul compiles");
        let swiglu = SwiGlu::new(device).expect("the swiglu compiles");
        let router = Router::new(device).expect("the router compiles");
        let grouping = ExpertGrouping::new(device).expect("the grouping compiles");
        let weighing = RouterWeights::new(device).expect("the weighting compiles");
        let combine = MoeCombine::new(device).expect("the combine compiles");
        let config = fixture::config(dir).text_config;
        let ckpt = Checkpoint::open(dir).expect("checkpoint opens");
        let recorded = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));
        let ids = match asked.prompt {
            None => recorded,
            Some(tokens) => tiled(&recorded, tokens),
        };

        let weights =
            CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
        let started = Instant::now();
        let head = PackedProjection::wrap_packed(
            device,
            matmul,
            &weights.head_packed(),
            weights.head().vocab(),
        )
        .expect("the head wraps");
        let banks = weights.expert_banks();
        let packed = weights.layer_projections();
        let layers = ModelLayers::wrap(
            device,
            &kernels,
            ExpertKernels {
                matmul,
                dense: &dense,
                swiglu: &swiglu,
                router: &router,
                grouping: &grouping,
                weights: &weighing,
                combine: &combine,
            },
            &packed,
            &banks,
            wrap_tail(device, &kernels, &weights, &config),
            StackShape {
                layers: config.num_hidden_layers,
                dim: config.hidden_size,
                slack: 0,
                slots: 1,
            },
        )
        .expect("the layers wrap");

        let mut run = Self {
            expert_layers: layers.expert_layers(),
            first_routed: banks[0].layer,
            projection_layers: layers.layers(),
            dense_layers: layers.dense_layers(),
            wrapped: started.elapsed(),
            prompt: ids.len(),
            asked,
            steps: Vec::new(),
            submitted: Vec::new(),
            peak: fixture::resident_bytes(),
            got: Vec::new(),
            profile: Profile::default(),
            round_trips: Vec::new(),
            traces: Vec::new(),
            boundaries: Vec::new(),
            barriers: 0,
        };
        device.record_round_trips(true);
        trace::record(asked.tracing);

        // Once, before the loop rather than inside it — though "once" is now 6
        // ms for 137 GB, so what that used to be defending against is gone.
        let weights = weights
            .with_head(Box::new(head))
            .with_backend(Box::new(layers));
        let generator = weights.generator();

        let read = |run: &mut Self| run.submitted.push(counters(device));
        read(&mut run);
        // Where the charged steps start, on the counter the barriers are read
        // off. It moves with every step the profile discards and stops when the
        // profile does, so what is left is the barriers of the steps charged.
        let mut charged_barriers = device.barriers();
        let mut step = Instant::now();
        generator.stream(
            &mut ModelCache::new(&config),
            &ids,
            Ending {
                budget: asked.generated,
                eos: None,
            },
            &weights,
            |id| {
                run.steps.push(step.elapsed());
                if asked.tracing {
                    // The boundaries first: taking the trace clears both.
                    run.boundaries.push(trace::submissions());
                    run.traces.push(trace::take());
                }
                read(&mut run);
                run.peak = run.peak.max(fixture::resident_bytes());
                run.got.push(id);
                if run.steps.len() <= asked.discarded() && !asked.charges_the_prefill() {
                    profile::take();
                    device.round_trips();
                    charged_barriers = device.barriers();
                }
                step = Instant::now();
                ControlFlow::Continue(())
            },
        );
        run.barriers = device.barriers() - charged_barriers;
        run.profile = profile::take().per_step(run.charged_steps());
        run.round_trips = device.round_trips();
        device.record_round_trips(false);
        trace::record(false);
        run
    }

    fn decode_steps(&self) -> u32 {
        (self.steps.len() - 1) as u32
    }

    /// How many steps the profile was summed over, which is one for a prefill.
    fn charged_steps(&self) -> u32 {
        if self.asked.charges_the_prefill() {
            1
        } else {
            self.decode_steps() - self.asked.settle as u32
        }
    }

    /// The index into [`Self::steps`] of the first step the profile describes.
    fn charged_from(&self) -> usize {
        if self.asked.charges_the_prefill() {
            0
        } else {
            self.asked.discarded()
        }
    }

    /// What one step of the regime the profile describes took: the prefill, or
    /// the mean of the decode steps after it.
    ///
    /// The prompt's prefill is the first step and every later one is a single
    /// decode; a mean over the two describes neither, which is what the slice
    /// here is for.
    fn each_charged_step(&self) -> Duration {
        self.steps[self.charged_from()..].iter().sum::<Duration>() / self.charged_steps()
    }

    /// The median of the charged steps.
    ///
    /// **A mean that is not the median is a reading of something else**, which
    /// the cross-engine table found the hard way: its mean at a 769-token prompt
    /// is 42.58 ms against a median of 36.44, and what separates them is the one
    /// step the prefill deferred onto. [`Asked::settle`] takes that step out, so
    /// the two should agree here — and printing both is what says whether it
    /// did.
    fn median_charged_step(&self) -> Duration {
        let mut charged: Vec<Duration> = self.steps[self.charged_from()..].to_vec();
        charged.sort_unstable();
        charged.get(charged.len() / 2).copied().unwrap_or_default()
    }

    /// The `(dispatches, submissions, allocations, allocated bytes)` of the last
    /// step the profile describes, which is the difference between the two
    /// running totals either side of it.
    fn per_charged_step(&self) -> (u64, u64, u64, u64) {
        let [.., before, after] = &self.submitted[..=self.steps.len()] else {
            panic!("the counters are read once before the loop and once a step, so there are two")
        };
        since(*before, *after)
    }

    /// The same two for a decode step, which is what the cases about one call
    /// them — and which a run that decoded nothing does not have.
    fn each_decode_step(&self) -> Duration {
        self.decoded().each_charged_step()
    }

    fn per_decode_step(&self) -> (u64, u64, u64, u64) {
        self.decoded().per_charged_step()
    }

    fn decoded(&self) -> &Self {
        assert!(
            !self.asked.charges_the_prefill(),
            "a run that decoded nothing has no decode step to price"
        );
        self
    }

    /// The keys the first and last charged decode step attend over.
    ///
    /// **A decode step's cost is a function of this and the record did not say
    /// it.** A prefill of `prompt` tokens leaves that many keys, and the `i`th
    /// decode step after it appends one and attends over `prompt + i` — so the
    /// charged steps span a range rather than sitting at a context, and a table
    /// that named neither is what let eleven milestones quote an eight-token
    /// figure as a decode figure.
    fn context(&self) -> (usize, usize) {
        let first = self.prompt + self.charged_from();
        (first, first + self.charged_steps() as usize - 1)
    }

    /// This run's charged regime, for the tables that read one.
    fn measured(&self) -> Measured<'_> {
        let (first, last) = self.context();
        Measured {
            regime: if self.asked.charges_the_prefill() {
                format!("prefill of {} tokens", self.prompt)
            } else {
                format!(
                    "decode step, a {}-token prompt and {first} to {last} keys",
                    self.prompt
                )
            },
            step: self.each_charged_step(),
            steps: self.charged_steps(),
            counters: self.per_charged_step(),
            profile: &self.profile,
            round_trips: &self.round_trips,
        }
    }
}

/// The device's four running totals, which one unit of work is the difference
/// across.
fn counters(device: &Device) -> (u64, u64, u64, u64) {
    (
        device.dispatches(),
        device.submissions(),
        device.allocations(),
        device.allocated_bytes(),
    )
}

/// What happened between two readings of them.
fn since(before: (u64, u64, u64, u64), after: (u64, u64, u64, u64)) -> (u64, u64, u64, u64) {
    (
        after.0 - before.0,
        after.1 - before.1,
        after.2 - before.2,
        after.3 - before.3,
    )
}

/// The engine, with every weight it has a kernel for on the GPU, against the
/// tokens mlx-vlm generated from the same prompt.
///
/// **This is the assertion with teeth, and it is the same one the CPU path
/// makes.** `inkling-core`'s `the_generated_tokens_match_the_oracle` establishes
/// that the eight recorded ids are what this engine decodes; what this says is
/// that moving where the head, the experts and now every layer's own
/// projections multiply does not change one of them. Every generated token is an
/// argmax over a distribution 42 layers of accumulated bfloat16 have already
/// moved, and two of the eight recorded positions carry a top-1/top-2 margin
/// *narrower* than that accumulated deviation — so arithmetic that is better is
/// not thereby guaranteed to agree, and this is where that would show.
///
/// Each handover raises the stakes over the last. The head is one multiply at
/// the end and a different summation order there moves a logit; an expert's
/// output is added into a residual that forty more layers then read; and a
/// layer's projections are *every* layer's, on the path every token takes,
/// including the queries and keys a softmax then amplifies the difference
/// between. That the eight tokens do not move is the finding.
///
/// A token that stops agreeing is a finding rather than a bound to widen. The
/// kernel sums 128 products a lane and reduces 32 lanes in a tree where the CPU
/// sums 4096 serially, and `the_packed_matmul_reproduces_the_cpu_over_the_real_head`
/// measures the kernel as the *closer* of the two to an f64 accumulation — so a
/// flipped token would mean a position where the reference's own bfloat16 logits
/// are tied or all but tied, which the recorded `logits_topk_values` at that
/// position settles.
///
/// The timings go to stderr rather than into an assertion.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn the_generated_tokens_match_the_oracle_with_the_model_on_the_device() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    let config = fixture::config(&dir).text_config;
    let oracle = indices(&fixture::tensor(
        &fixture::open(ACTIVATIONS),
        "greedy_continuation",
    ));
    let want = &oracle[..GENERATED];

    let run = OnTheDevice::generate(&dir, &device);
    eprintln!(
        "the head, {} MoE layers' banks and {} layers' projections ({} of them dense) wrapped in \
         {:?}",
        run.expert_layers, run.projection_layers, run.dense_layers, run.wrapped
    );
    assert_eq!(
        run.expert_layers,
        config.num_hidden_layers - run.first_routed,
        "every layer past the dense ones has banks here"
    );
    assert_eq!(
        run.projection_layers, config.num_hidden_layers,
        "every layer has its projections here"
    );
    assert_eq!(
        run.dense_layers + run.expert_layers,
        config.num_hidden_layers,
        "a layer has a feed-forward network or banks, never both and never neither"
    );

    let (dispatches, submissions, allocations, bytes) = run.per_decode_step();
    eprintln!(
        "{} tokens prefilled in {:.2?}, {} decoded at {:.2?}/token, peak RSS {:.2} GiB\
         \n  {dispatches} dispatches a decode step in {submissions} submissions over \
         {allocations} buffers of {:.1} MiB, which is {:.0} dispatches a submission\
         \n  got  {:?}\n  want {want:?}",
        run.prompt,
        run.steps[0],
        run.decode_steps(),
        run.each_decode_step(),
        run.peak as f64 / (1u64 << 30) as f64,
        bytes as f64 / (1u64 << 20) as f64,
        dispatches as f64 / submissions as f64,
        run.got,
    );
    assert_eq!(
        (dispatches, submissions),
        per_step(config.num_hidden_layers as u64, run.dense_layers as u64),
        "a decode step's dispatches, and the command buffers they went in"
    );
    assert!(allocations > 0, "a decode step allocated nothing");

    let agreed = run.got.iter().zip(want).take_while(|(a, b)| a == b).count();
    assert_eq!(run.got, want, "{agreed} of {GENERATED} tokens agree");
    assert!(
        run.peak < RESIDENT_BOUND,
        "peak RSS {} bytes is over the bound of {RESIDENT_BOUND}",
        run.peak
    );
}

/// **The device tail against the host tail, on real logits — which is what has
/// to hold before any figure this milestone reports means anything.**
///
/// Moving the final norm and the muP divide onto the GPU is the one piece of
/// tail work this project deferred every time it came up, and the reason was
/// numerical rather than structural: Apple silicon has no f64 to accumulate a
/// sum of squares in, a norm is a reduction, and a reduction that reassociates
/// differently moves a logit. A moved logit at the top of the distribution is a
/// different token.
///
/// So the two tails are run over the same rows and compared where it matters.
/// The same prompt goes through two stacks that differ in nothing but who runs
/// the last three operations — the layers are the same wrapped weights and the
/// same dispatches, so what they hand the tail is bit for bit the same hidden
/// state — and what is asserted is the **argmax**, at every row of the block and
/// at the eight decode steps behind it. The logits themselves are held to a
/// tolerance, and the margin at each position is printed beside it: what says a
/// tail is safe is not that the logits agree to an ulp but that the gap between
/// the first and second logit is decades wider than the disagreement.
///
/// The divide is the half of this that *is* exact — see `inkling_metal::tail`,
/// where the multiplier is folded into the norm's weight only when doing so
/// moves no bit — so what this measures is the norm's own reassociation, which
/// is the same one every other norm in this engine already has.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn the_device_tail_takes_the_token_the_host_tail_takes() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    let config = fixture::config(&dir).text_config;
    let text = &config;
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);
    let ids = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));

    let held = gpu.wrap(&ckpt, text, 0);
    let apart = gpu.without_a_tail(&ckpt, text);
    let want = Tail {
        block: 1,
        chained: true,
        logits: true,
    };

    // The prompt, and then a decode step at a time — because the two regimes
    // hand the tail different shapes: a prefill's block is one row out of many
    // and a decode step's is the whole call, which is the arm that folds.
    let (device_cache, host_cache) = (&mut ModelCache::new(text), &mut ModelCache::new(text));
    let (mut on_device, mut on_host) = (ids.clone(), ids.clone());
    let (mut worst, mut narrowest) = (0.0f32, f32::INFINITY);

    for step in 0..=GENERATED {
        let ours = held
            .generator()
            .tailed(device_cache, &on_device, want, &held);
        let theirs = apart.generator().tailed(host_cache, &on_host, want, &apart);

        let (ours_id, theirs_id) = (ours.picks[0], theirs.picks[0]);
        assert_eq!(
            ours.logits.len(),
            theirs.logits.len(),
            "step {step}: the two tails answered different widths"
        );

        // **The argmax on the device against the argmax on this side, over the
        // one row both of them ranked** — which is exact rather than close, and
        // is the only claim in this file of which that is true. The two tails'
        // logits differ by ulps and the assertion below is about a margin; these
        // two rank the same 200058 floats, so a disagreement would be the tie
        // rule and nothing else.
        assert_eq!(
            ours_id,
            inkling_core::Generator::picks(&ours.logits, 1)[0],
            "step {step}: the device argmax and the host's disagree about one row"
        );
        assert_eq!(
            theirs_id,
            inkling_core::Generator::picks(&theirs.logits, 1)[0],
            "step {step}: the host tail's own ids are not its logits'"
        );

        // What the argmax had to survive: how far the two tails' logits are
        // apart, against how far the best logit is from the second best.
        let apartness = deviation(&ours.logits, &theirs.logits);
        let margin = margin(&theirs.logits);
        let (at_the_top, longest) = tied(&theirs.logits);
        eprintln!(
            "step {step:>2}: token {theirs_id:>6}  logits apart {apartness:.3e}  \
             margin {margin:.3e}  normed apart {:.3e}  ids at the peak {at_the_top}  \
             longest tied run {longest}",
            deviation(&ours.normed, &theirs.normed)
        );
        assert_eq!(
            ours_id, theirs_id,
            "step {step}: the device tail took a different token"
        );
        worst = worst.max(apartness);
        narrowest = narrowest.min(margin);

        on_device = vec![ours_id];
        on_host = vec![theirs_id];
    }

    eprintln!(
        "worst disagreement {worst:.3e} against the narrowest margin {narrowest:.3e}, \
         a factor of {:.0}",
        narrowest / worst
    );
    assert!(
        worst < narrowest,
        "the two tails disagree by more than a token's margin"
    );
}

/// How many ids hold a row's largest logit, and the longest run of ids anywhere
/// in it that agree bit for bit.
///
/// **Whether the tie rule is hypothetical is a measurement and not an opinion**,
/// and this is the measurement. The first number is how many ways the argmax
/// could have gone at that position; the second is how often 200058 float32
/// logits collide at all, which is what says a rule about equal values is a rule
/// this vocabulary reaches.
fn tied(logits: &[f32]) -> (usize, usize) {
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let at_the_top = sorted.iter().take_while(|x| **x == sorted[0]).count();
    let (mut longest, mut run) = (1usize, 1usize);
    for pair in sorted.windows(2) {
        run = match pair[0].to_bits() == pair[1].to_bits() {
            true => run + 1,
            false => 1,
        };
        longest = longest.max(run);
    }
    (at_the_top, longest)
}

/// How far the best logit of a row is from the second best, relative to the
/// row's own peak — the scale `deviation` reports a disagreement on, so that
/// the two are comparable.
fn margin(logits: &[f32]) -> f32 {
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("logits are finite"));
    let peak = sorted[0].abs().max(sorted[sorted.len() - 1].abs());
    (sorted[0] - sorted[1]) / peak
}

/// A checkpoint and a device that times each of its dispatches, or nothing with
/// a reported skip.
///
/// The three cases that read a kernel table all need the same three things and
/// all decline for the same three reasons, and none of them is a property of the
/// code under test: no checkpoint, no Metal device, or a device that cannot
/// timestamp a stage boundary — which this hardware answers `true` for and
/// others need not.
fn sampling_device() -> Option<(PathBuf, Device)> {
    let dir = checkpoint_dir()?;
    let device = device()?;
    if !device.times_a_pass() {
        eprintln!("skipping: this device does not sample at a stage boundary");
        return None;
    }
    Some((dir, device))
}

/// What a sampled run's kernel rows say the device did: how many dispatches
/// were timed and what they declared they moved.
///
/// Both together, because the three tables that read them read them for one
/// thing — the count is what says the table describes the whole step, and the
/// bytes are what the achieved column divides — and a case that took one without
/// the other would be describing a fraction of a step in bytes.
fn what_was_sampled(profile: &Profile) -> (u64, u64) {
    let kernels = profile.kernels();
    assert!(!kernels.is_empty(), "nothing was sampled");
    (
        kernels.iter().map(|(_, each)| each.calls).sum(),
        kernels.iter().map(|(_, each)| each.bytes).sum(),
    )
}

/// One regime's repeated unit of work, in the shape the two tables below read
/// it: what one of them took, what it dispatched and submitted, and where its
/// time went.
///
/// **The tables are written against this rather than against the run that
/// produced it**, because a decode step, a prefill and a chain of heads are
/// three regimes measured three different ways — a generation's steps, one
/// prefill, and a round repeated against a warm cache — and the whole use the
/// tables are put to is reading one against another.
struct Measured<'a> {
    /// What a row of the header calls the unit, which is the only thing in here
    /// that knows which regime this is.
    regime: String,
    step: Duration,
    /// How many of them the profile was summed over, which is what the round
    /// trips divide by.
    steps: u32,
    /// The `(dispatches, submissions, allocations, allocated bytes)` of one.
    counters: (u64, u64, u64, u64),
    profile: &'a Profile,
    round_trips: &'a [RoundTrip],
}

/// A step's rows, and the kernels underneath them where the device was asked
/// which of its dispatches owns which of the milliseconds.
///
/// One table rather than two, because the second half only means anything
/// against the first: a kernel's device time is a share of what the device
/// executed for, which is a share of the wait, which is a share of the step.
///
/// **One function rather than one per regime**, because the question a prefill
/// or a chain of heads is read for is which rows differ from a decode step's —
/// and two tables that could rank, divide or round differently could not answer
/// it.
///
/// **The `groups` column is the one B2 was missing.** Its attention step was
/// read as a dispatch count and the count turned out to be 14% of what the
/// change was worth: what a narrow dispatch costs is the machine it leaves idle,
/// which is its grid against [`WANTED_GROUPS`] and is a column no table here
/// carried. A row at 2% of the wanted threadgroups is a kernel with a batched
/// version worth taking whatever its share says; a row already at 100% has
/// nothing there to win.
fn step_table(run: &Measured<'_>) -> String {
    let &Measured {
        step,
        counters: (dispatches, submissions, allocations, bytes),
        ..
    } = run;
    let regime = &run.regime;
    let accounted = run.profile.total();

    let share = |part: Duration| 100.0 * part.as_secs_f64() / step.as_secs_f64();
    let mut table = vec![format!(
        "a {step:.2?} {regime}, {dispatches} dispatches in {submissions} submissions over \
         {allocations} buffers of {:.1} MiB\n  {:<18}{:>7}{:>12}{:>8}",
        bytes as f64 / (1u64 << 20) as f64,
        "operation",
        "calls",
        "self time",
        "share"
    )];
    for (op, calls, elapsed) in run.profile.rows() {
        table.push(format!(
            "  {:<18}{calls:>7}{:>12}{:>7.1}%",
            op.name(),
            format!("{elapsed:.2?}"),
            share(elapsed)
        ));
    }
    table.push(format!(
        "  {:<18}{:>7}{:>12}{:>7.1}%",
        "unaccounted",
        "",
        format!("{:.2?}", step.saturating_sub(accounted)),
        share(step.saturating_sub(accounted))
    ));
    // Of the step and not of the wait, which it may now exceed: a run commits
    // part way through and keeps encoding, so a command buffer executes while
    // this process is charging its time to `dispatch encode` rather than to
    // `submit and wait`. The share of the step is what does not depend on that.
    table.push(format!(
        "  of which the device reported executing for {:.2?}, {:.1}% of the step and {:.1}% of \
         the {:.2?} spent waiting for it",
        run.profile.gpu(),
        share(run.profile.gpu()),
        100.0 * run.profile.gpu().as_secs_f64()
            / run.profile.elapsed(Op::Submit).as_secs_f64().max(f64::MIN),
        run.profile.elapsed(Op::Submit),
    ));

    let kernels = run.profile.kernels();
    if kernels.is_empty() {
        return table.join("\n");
    }
    // Of the passes rather than of the command buffer, because what is left
    // over is the pass boundaries the sampling itself adds — an artefact of
    // asking, printed on its own line rather than folded into a kernel's share
    // of a step nobody runs.
    let sampled = run.profile.dispatched();
    let executing = |part: Duration| 100.0 * part.as_secs_f64() / sampled.as_secs_f64();
    table.push(format!(
        "  {:<18}{:>7}{:>12}{:>8}{:>10}{:>9}{:>14}{:>11}{:>8}",
        "kernel", "calls", "device", "share", "groups", "of want", "moved", "achieved", "of peak"
    ));
    for (kernel, dispatches) in &kernels {
        let achieved = dispatches.bytes_per_second();
        let groups = dispatches.groups_a_dispatch();
        table.push(format!(
            "  {kernel:<18}{:>7}{:>12}{:>7.1}%{:>10}{:>8.0}%{:>14}{:>11}{:>7.0}%",
            dispatches.calls,
            format!("{:.2?}", dispatches.elapsed),
            executing(dispatches.elapsed),
            format!("{groups:.0}"),
            100.0 * groups / WANTED_GROUPS as f64,
            format!("{:.2} MB", dispatches.bytes as f64 / 1e6),
            format!("{:.0} GB/s", achieved / 1e9),
            100.0 * achieved / MEMORY_BANDWIDTH,
        ));
    }
    // **The two clocks need not fall the same way round and which one is larger
    // is itself a reading.** A command buffer's own clock runs from before its
    // first dispatch to after its last, so what the passes inside it leave over
    // is the device's gaps between them — until a pass span's own
    // over-reporting is the larger of the two, which is where a table of few
    // long dispatches lands and a table of many short ones does not.
    let (gpu, gap) = (run.profile.gpu(), run.profile.gpu().saturating_sub(sampled));
    table.push(if gap > Duration::ZERO {
        format!(
            "  the passes account for {sampled:.2?} of the {gpu:.2?} the command buffers \
             reported, and the {gap:.2?} between them is what a pass boundary a dispatch costs"
        )
    } else {
        format!(
            "  the passes claim {sampled:.2?} of the {gpu:.2?} the command buffers reported, \
             which is {:.2?} more than the buffers themselves clocked and is the span's own \
             over-reporting arriving above the gaps it is measured beside",
            sampled.saturating_sub(gpu),
        )
    });
    table.join("\n")
}

/// What the waits inside a step were waiting on, one row per shape of
/// submission.
///
/// **Grouped by the dispatches in a submission and not by which came first**,
/// because that is what tells a decode step's two apart — 1076 dispatches and
/// one — and what tells a prefill's forty-two apart from the head's, without
/// this having to know the order. A mean over shapes that differ that much
/// describes none of them.
fn round_trip_table(run: &Measured<'_>) -> String {
    let steps = run.steps;
    let mut shapes: BTreeMap<usize, Vec<RoundTrip>> = BTreeMap::new();
    for trip in run.round_trips {
        shapes.entry(trip.dispatches).or_default().push(*trip);
    }

    let mut table = vec![format!(
        "a {:.2?} step, of which {:.2?} is the {} submissions it waits for\n  \
         {:<11}{:>8}{:>11}{:>11}{:>10}{:>11}{:>14}",
        run.step,
        run.profile.elapsed(Op::Submit),
        run.round_trips.len() as u32 / steps,
        "dispatches",
        "a step",
        "waited",
        "scheduled",
        "queued",
        "executed",
        "unattributed",
    )];
    let mean = |trips: &[RoundTrip], of: fn(&RoundTrip) -> Duration| {
        trips.iter().map(of).sum::<Duration>() / steps
    };
    for (dispatches, trips) in &shapes {
        table.push(format!(
            "  {dispatches:<11}{:>8}{:>11}{:>11}{:>10}{:>11}{:>14}",
            trips.len() as u32 / steps,
            format!("{:.2?}", mean(trips, |trip| trip.waited)),
            format!("{:.2?}", mean(trips, |trip| trip.scheduled)),
            format!("{:.2?}", mean(trips, |trip| trip.queued)),
            format!("{:.2?}", mean(trips, |trip| trip.executed)),
            format!("{:.2?}", mean(trips, |trip| trip.unattributed())),
        ));
    }
    table.join("\n")
}

/// Where a decode step's time actually goes, as a table.
///
/// **This is the measurement the next several commits are ordered by**, and
/// what makes it worth a case of its own is that every prediction this project
/// has made about its own cost has been wrong: `lm_head` was 7.6% of a step
/// rather than the 54% its parameter count implied, and the bandwidth model
/// that said a packed matmul would be memory-bound died against a kernel
/// running at 4% of the bandwidth.
///
/// Three numbers frame the rows. **The wall time** of a decode step is what
/// there is to divide up. **The rows** are self time — see
/// [`inkling_core::profile`] — so they sum to what is accounted for and the
/// remainder is what nothing here has a scope around. And **the GPU's own
/// clock** is inside the *step* rather than inside `submit and wait`: it was
/// inside the wait while a submission was a stall, and a run that commits part
/// way through and keeps encoding has the device executing against a row this
/// process is charging elsewhere. The two figures crossing is what that change
/// looks like from here, and the step is the denominator that survives it.
///
/// Nothing asserts a share. What is asserted is that the accounting adds up —
/// the rows cannot exceed the wall time they were measured inside, the device
/// cannot have executed for longer than the step that waited for it, and what
/// the rows leave over stays small enough for the table to be a description of
/// the step rather than of a fraction of it.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn where_a_decode_step_spends_its_time() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    let run = OnTheDevice::generate(&dir, &device);
    let step = run.each_decode_step();
    let accounted = run.profile.total();
    eprintln!("{}", step_table(&run.measured()));

    assert!(
        accounted <= step,
        "the rows sum to {accounted:.2?} inside a {step:.2?} step"
    );
    assert!(
        run.profile.gpu() < step,
        "the device reported executing for {:.2?} of a {step:.2?} step",
        run.profile.gpu()
    );
    // **A decode step reads no weight at all.** Every packed one is a dispatch
    // against bytes nothing decodes, and every bfloat16 one was widened at
    // construction — so the row that was 500 calls and 12.4 ms is not a smaller
    // row, it is absent. A path that went back to widening a layer's norms, its
    // convolution kernels or its router's gate per step lands at 500 again.
    assert_eq!(
        run.profile.calls(Op::Decode),
        0,
        "a decode step widened or decoded a weight"
    );
    // A table that named a third of the step would be describing something
    // else. The bound is loose because what it guards is that the scopes are
    // still where the ops are, not the share any of them holds.
    assert!(
        accounted > step / 2,
        "only {accounted:.2?} of a {step:.2?} step is accounted for"
    );
}

/// **What the CPU is waiting for while it waits**, which is the question the
/// `submit and wait` row above cannot answer either.
///
/// That row is three quarters of a decode step, and a milestone that reads it
/// as three quarters of a step spent asking rather than working would go and
/// remove submissions. The device's own clock already says otherwise — the row
/// prints what share of it the GPU was executing for — and what this adds is
/// the rest of the division, per submission rather than summed: the driver
/// turning a committed buffer into work the GPU can start, the queue, the
/// execution, and what none of the three claim.
///
/// **One row per shape of submission and not one per submission**, because a
/// decode step's two are 1076 dispatches and one, and a mean over the pair
/// describes neither. Grouped by the dispatches in them, which is what tells
/// them apart without this having to know which came first.
///
/// Nothing asserts a share. What is asserted is that the trips describe the same
/// waits the profile does, so that a table read against the one above is read
/// against the same step.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn what_a_decode_steps_round_trips_are_waiting_on() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    let run = OnTheDevice::generate(&dir, &device);
    let steps = run.decode_steps();
    eprintln!("{}", round_trip_table(&run.measured()));

    let waited: Duration = run.round_trips.iter().map(|trip| trip.waited).sum();
    let executed: Duration = run.round_trips.iter().map(|trip| trip.executed).sum();
    // Against the device's count and not the profile's: `Op::Submit` is charged
    // at both ends of a command buffer that is committed and waited for
    // separately, so its calls are the ends and these are the buffers.
    assert_eq!(
        run.round_trips.len() as u64,
        run.per_decode_step().1 * u64::from(steps),
        "a trip was recorded for a submission the device did not count, or the other way round"
    );
    // The two clocks around one wait: `Op::Submit` is a scope that closes after
    // the trip is recorded, so it is the larger of the two by whatever the
    // recording itself takes, and a trip claiming more than the scope around it
    // would mean the wall time here is not this wait's.
    assert!(
        waited / steps <= run.profile.elapsed(Op::Submit),
        "the trips waited {:.2?} inside a {:.2?} row",
        waited / steps,
        run.profile.elapsed(Op::Submit)
    );
    // Exactly, and not within a tolerance: a wait reads the device's clock once
    // and charges the same duration to both accounts, so a disagreement here is
    // a trip or a submission one of the two did not see rather than a rounding.
    assert_eq!(
        executed / steps,
        run.profile.gpu(),
        "the trips and the profile disagree about what the device executed for"
    );
}

/// **Which kernels own the milliseconds the device executes for**, which is the
/// question the `submit and wait` row above cannot answer.
///
/// A decode step is 1077 dispatches across nine distinct kernels, and until this
/// landed the only figure any of them had was the 26 ms the pair of command
/// buffers reported between them. This project's record on dividing that number
/// up by reasoning is poor and written down: `lm_head` was predicted at 54% of a
/// step and measured at 7.6%, the 4.9 GB/s dequantisation model stopped
/// describing the step the moment a different kernel dominated it, and M9's
/// premise about which dispatches forced a submission was wrong about which two
/// they were.
///
/// **What a row is.** Each dispatch runs as a compute pass of its own — the only
/// grain this hardware samples at, see `inkling_metal::sampling` — and a row
/// is the sum of those passes' spans for one kernel. Those spans are the
/// dispatch's own execution and *not* what the pass boundary around it costs:
/// against an unsampled command buffer of the same dispatches, a span
/// over-reports by around a microsecond and the boundary's own cost lands in the
/// gap between passes, which is the `between passes` row. So the ranking is the
/// finding and the absolute figures carry that bias, stated rather than hidden.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn which_kernels_own_a_decode_step() {
    let Some((dir, device)) = sampling_device() else {
        return;
    };

    // The unsampled step first, because what the rows below are worth is how
    // close they sum to the device time of a step nobody was asking about.
    let unsampled = OnTheDevice::generate(&dir, &device).profile.gpu();
    let run = OnTheDevice::generating(&dir, &device, true);
    eprintln!("{}", step_table(&run.measured()));
    eprintln!(
        "  against {unsampled:.2?} of device time with nothing sampling, so the rows carry \
         {:+.1}% of asking",
        100.0 * (run.profile.dispatched().as_secs_f64() / unsampled.as_secs_f64() - 1.0)
    );

    // **The bytes are declared and not derived**, so a formula that dropped a
    // factor would move the whole bandwidth column and nothing would say so.
    // What checks it is a figure this repo has from somewhere else entirely: a
    // token reads six of each MoE layer's 256 experts and both shared ones plus
    // every layer's own projections, which the checkpoint's shapes put at 5.9 GB
    // of packed bytes.
    let (timed, moved) = what_was_sampled(&run.profile);
    assert!(
        (5e9..7e9).contains(&(moved as f64)),
        "a decode step moved {:.2} GB, where the checkpoint's active weights are 5.9",
        moved as f64 / 1e9
    );
    // Every dispatch the step encoded came back with a pair of timestamps. A
    // device that dropped one writes `MTLCounterErrorValue` and this side
    // charges it nothing, which would be a row quietly short rather than a
    // failure — so the count is what says the table describes the whole step.
    let (dispatches, ..) = run.per_decode_step();
    assert_eq!(
        timed, dispatches,
        "a decode step's dispatches were not all timed"
    );
    assert!(
        run.profile.dispatched() <= run.profile.gpu(),
        "the passes claim {:.2?} of a command buffer the device clocked at {:.2?}",
        run.profile.dispatched(),
        run.profile.gpu()
    );
}

/// **Which kernels own a decode step at a context somebody actually has**,
/// which is the same table as the case above at the three lengths the
/// cross-engine comparison decodes over rather than at the recorded prompt's
/// eight.
///
/// **The case above is a true table about a context nobody has.** Every decode
/// figure this repo has ever quoted was taken over the eight-token prompt the
/// activation capture recorded — the head merge, the device tail, the packed
/// heads, the device argmax — and `just bench-engines` then found the reference
/// flat in the context and this engine not: 23.5 ms a token at 97 keys against
/// 42.6 at 769, where mlx-vlm reads 23.0 and 23.7. A step that grows with the
/// context has a row in this table that grows with it, and one table at one
/// length cannot say which.
///
/// **Three lengths and not two, because the two questions differ.** A prefill is
/// diagnosed at two — 97 tokens is the only length whose layers still merge into
/// one run, which is a scheduling difference and not a kernel one. Here every
/// length runs the same fourteen submissions over the same 1078 dispatches, so
/// the 97-token row is a row about the kernels like the other two and is the one
/// the reference is indistinguishable from.
///
/// Nothing asserts a share or a slope. What is asserted is what the case above
/// asserts — that every dispatch was timed and that the passes stay inside the
/// command buffers they were cut out of — at each of the three.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn which_kernels_own_a_decode_step_at_each_context() {
    let Some((dir, device)) = sampling_device() else {
        return;
    };

    for &prompt in &PREFILL_WALL_LENGTHS {
        let unsampled = OnTheDevice::decoding_at(&dir, &device, prompt, false);
        let run = OnTheDevice::decoding_at(&dir, &device, prompt, true);
        eprintln!("{}", step_table(&run.measured()));
        eprintln!(
            "  against a {:.2?} unsampled step of {:.2?} device time, so the rows carry \
             {:+.1}% of asking",
            unsampled.each_charged_step(),
            unsampled.profile.gpu(),
            100.0
                * (run.profile.dispatched().as_secs_f64() / unsampled.profile.gpu().as_secs_f64()
                    - 1.0)
        );

        let (timed, moved) = what_was_sampled(&run.profile);
        // The same bound the eight-token case makes, and it holds at every
        // length for the reason the prefill table gives from the other side: a
        // decode step reads the model once whatever the context, and what grows
        // with the context is the span attention walks rather than the weights.
        assert!(
            (5e9..7e9).contains(&(moved as f64)),
            "a decode step at {prompt} tokens moved {:.2} GB, where the checkpoint's active \
             weights are 5.9",
            moved as f64 / 1e9
        );
        let (dispatches, ..) = run.per_decode_step();
        assert_eq!(
            timed, dispatches,
            "a decode step at {prompt} tokens did not have all its dispatches timed"
        );
        assert!(
            run.profile.dispatched() <= run.profile.gpu(),
            "at {prompt} tokens the passes claim {:.2?} of a command buffer the device clocked \
             at {:.2?}",
            run.profile.dispatched(),
            run.profile.gpu()
        );
    }
}

/// The widths the batched table is taken at, which are B2's own sweep's without
/// the two it can say nothing new about: 1 is the case that must not move, and
/// 32 is where the curve it is read against ends.
const BATCHED_WIDTHS: [usize; 5] = [1, 4, 8, 16, 32];

/// What the profile files a layer's attention step under — two rows for one
/// entry, because the 35 windowed layers and the 7 global ones are a number
/// about neither summed.
const ATTENTION_ROWS: [&str; 2] = ["windowed attention", "global attention"];

/// **The batched per-kernel table B2 named and did not take**, at five widths
/// and with the column that would have predicted its own win.
///
/// B2 shared the attention step between a batch's sequences and left four
/// dispatches a layer that are still a slot's own — the key and value
/// convolutions as one pair, the two head norms as one, and the two residual
/// convolutions. Their count is 168 a slot against the step's 42, which on B1's
/// reading makes them four times the prize; on the reading B2 established they
/// are worth what their grids are short of the machine by, and nothing had
/// measured that. **What decides whether they are worth a kernel change is the
/// `groups` column against `WANTED_GROUPS`, not the `calls` one.**
///
/// **Both arms at every width**, for the reason the context table gives: a
/// sampled step is a pass a dispatch and the per-slot kernels are the numerous
/// ones, so the cost of asking grows with the batch and a table that did not
/// print the unsampled step beside it could not say by how much.
///
/// Nothing here asserts a share or a grid ratio between kernels — that is the
/// reading, and the reading is the report's. What is asserted is what the two
/// tables above assert, plus the one thing the new column can be wrong about:
/// **a batched attention dispatch's grid is the batch's rows and not a slot's.**
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn which_kernels_own_a_batched_step() {
    let Some((dir, device)) = sampling_device() else {
        return;
    };
    let config = fixture::config(&dir).text_config;
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);
    let recorded = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));

    let mut alone = 0.0;
    for slots in BATCHED_WIDTHS {
        let unsampled = batched_steps(&device, &gpu, &ckpt, &config, &recorded, slots, false);
        let run = batched_steps(&device, &gpu, &ckpt, &config, &recorded, slots, true);
        eprintln!("{}", step_table(&run.measured()));
        eprintln!(
            "  against a {:.2?} unsampled step of {:.2?} device time, so the rows carry \
             {:+.1}% of asking",
            unsampled.step,
            unsampled.profile.gpu(),
            100.0
                * (run.profile.dispatched().as_secs_f64() / unsampled.profile.gpu().as_secs_f64()
                    - 1.0)
        );

        let (timed, moved) = what_was_sampled(&run.profile);
        let moved = moved as f64;
        // **What a batch reads is between one step's bytes and N of them**, and
        // both ends are the claim rather than one: the projections and both
        // shared experts are read once between the sequences, and the routed
        // banks are not — sixteen tokens draw about 80 distinct experts of 256
        // where one draws six, so the bytes grow with the batch and grow slower
        // than it. A batch of one is held to the checkpoint's own 5.9 GB, which
        // is the figure the other four are read against.
        if slots == 1 {
            alone = moved;
            assert!(
                (5e9..7e9).contains(&moved),
                "a batch of one moved {:.2} GB, where the checkpoint's active weights are 5.9",
                moved / 1e9
            );
        } else {
            assert!(
                (alone..slots as f64 * alone).contains(&moved),
                "a batch of {slots} moved {:.2} GB against {:.2} for one sequence, which is \
                 outside the {:.2} to {:.2} a shared read has to land in",
                moved / 1e9,
                alone / 1e9,
                alone / 1e9,
                slots as f64 * alone / 1e9
            );
        }
        let (dispatches, ..) = run.counters;
        assert_eq!(
            timed, dispatches,
            "a batch of {slots} did not have all its dispatches timed"
        );
        assert!(
            run.profile.dispatched() <= run.profile.gpu(),
            "at a batch of {slots} the passes claim {:.2?} of command buffers the device clocked \
             at {:.2?}",
            run.profile.dispatched(),
            run.profile.gpu()
        );

        // **A threadgroup to each query row of each head, over the whole
        // batch.** The recorded prompt is short enough that no span has the
        // tiles to be cut, so every attention dispatch is exactly one grid of
        // `heads * slots` — which is the arithmetic B2's 32 dispatches of 32
        // becoming one of 1024 is, and the one thing the column above can be
        // wrong about. A row that read a slot's grid here would leave the table
        // saying the step is as short of the machine at 32 as at 1.
        let rows = run.profile.kernels();
        for name in ATTENTION_ROWS {
            let Some((_, attention)) = rows.iter().find(|(kernel, _)| *kernel == name) else {
                panic!("a batch of {slots} ran no {name}");
            };
            assert_eq!(
                attention.groups_a_dispatch(),
                (config.num_attention_heads * slots) as f64,
                "a batch of {slots} dispatched {name} over {} threadgroups",
                attention.groups_a_dispatch()
            );
        }
    }
}

/// `slots` sequences prefilled apart and then decoded together, timed over the
/// steps after them.
///
/// **The prompts are rotated rather than copied**, for the reason `bench batch`
/// gives: the routing is the prompt's, and a batch of identical sequences sends
/// every row of every step to the same six experts — the one distribution a
/// grouped dispatch is least like the real one at.
///
/// **The prefills and the first [`SETTLED`] steps are outside the clock.** A
/// width is a fresh wrap, and a slot is buffers this device has not allocated
/// before, so what the first steps of a width pay for belongs to the wrap: the
/// batch of one read 25.9 ms against its own 16.4 unsettled.
fn batched_steps(
    device: &Device,
    gpu: &Kernels<'_>,
    ckpt: &Checkpoint,
    config: &inkling_core::TextConfig,
    ids: &[usize],
    slots: usize,
    sampling: bool,
) -> BatchedSteps {
    device
        .time_each_dispatch(sampling)
        .expect("the device times a dispatch");
    let weights = gpu.wrap_batch(ckpt, config, slots);
    let generator = weights.generator();
    let mut caches: Vec<ModelCache> = (0..slots)
        .map(|slot| ModelCache::in_slot(config, 0, slot))
        .collect();
    let want = Tail {
        block: 1,
        chained: false,
        logits: false,
    };
    let mut pending: Vec<usize> = caches
        .iter_mut()
        .enumerate()
        .map(|(slot, cache)| {
            let mut prompt = ids.to_vec();
            prompt.rotate_left(slot % ids.len());
            generator.tailed(cache, &prompt, want, &weights).picks[0]
        })
        .collect();

    let step = |pending: &[usize], caches: &mut [ModelCache]| -> Vec<usize> {
        let feeding: Vec<[usize; 1]> = pending.iter().map(|id| [*id]).collect();
        let mut batch: Vec<Batched<'_>> = caches
            .iter_mut()
            .zip(&feeding)
            .map(|(cache, ids)| Batched { cache, ids })
            .collect();
        generator
            .step_batch(&mut batch, &weights)
            .iter()
            .map(Picked::last)
            .collect()
    };

    for _ in 0..SETTLED {
        pending = step(&pending, &mut caches);
    }

    profile::take();
    device.record_round_trips(true);
    device.round_trips();
    let before = counters(device);
    let started = Instant::now();
    for _ in 0..CHARGED {
        pending = step(&pending, &mut caches);
    }
    let elapsed = started.elapsed();
    let charged = CHARGED as u32;
    let (dispatches, submissions, allocations, bytes) = since(before, counters(device));
    let each = |total: u64| total / u64::from(charged);
    let steps = BatchedSteps {
        slots,
        prompt: ids.len(),
        step: elapsed / charged,
        counters: (
            each(dispatches),
            each(submissions),
            each(allocations),
            each(bytes),
        ),
        profile: profile::take().per_step(charged),
        round_trips: device.round_trips(),
    };
    device.record_round_trips(false);
    steps
}

/// One batched decode step, priced the way a single sequence's is.
struct BatchedSteps {
    slots: usize,
    prompt: usize,
    step: Duration,
    counters: (u64, u64, u64, u64),
    profile: Profile,
    round_trips: Vec<RoundTrip>,
}

impl BatchedSteps {
    /// The batch as the tables read a regime, which is where it is put beside a
    /// decode step, a prefill and a chain of heads.
    fn measured(&self) -> Measured<'_> {
        let first = self.prompt + SETTLED;
        Measured {
            regime: format!(
                "step over a batch of {}, {}-token prompts and {first} to {} keys",
                self.slots,
                self.prompt,
                first + CHARGED - 1
            ),
            step: self.step,
            steps: CHARGED as u32,
            counters: self.counters,
            profile: &self.profile,
            round_trips: &self.round_trips,
        }
    }
}

/// **What the step after a prefill is paying for**, which every decode figure in
/// this file takes out and none of them has ever priced.
///
/// [`Asked::settle`] discards it and says in one line what it is; the
/// cross-engine table records it at 736 and 783 ms and loses two rows of its
/// mean to it. **A user waits for it either way**, so what it is made of decides
/// whether it can be moved off the path a token is on — and a sentence in a doc
/// comment is not a measurement.
///
/// **What it is not is work.** It dispatches what the step behind it dispatches,
/// submits what it submits and allocates what it allocates — the first table
/// below is those three columns beside the wall — and the device executes for
/// 19 ms of the 845. It is not a compilation either: every pipeline this engine
/// has was created before the prefill ran.
///
/// **What it is, is scheduling.** The round-trip table divides a submission into
/// what the driver took to turn a committed buffer into work the GPU can start,
/// what it then spent queued, and what it executed for — and at a 769-token
/// prompt the twelve middle submissions of this one step are **741 ms
/// scheduled against 19 ms executed**, where the same twelve of the step behind
/// it are 1.9 ms scheduled. Making a step's buffers ready to run is what costs,
/// and a long prefill is what leaves them needing it.
///
/// **It is a threshold and not a slope**, which is the second table's finding
/// and the reason the cross-engine figures at 385 and 769 tokens are so nearly
/// equal: what the step pays for is the step's own residency being re-taken,
/// and that is the model's size rather than the prompt's.
///
/// **`settle` is zero here**, which is what makes this the one case in the file
/// that sees the step. Nothing else does and nothing else should.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn what_the_step_after_a_prefill_is_paying_for() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    eprintln!(
        "{:>8}{:>10}{:>11}{:>10}{:>10}{:>12}",
        "context", "prefill", "step 1", "step 2", "step 3", "allocated"
    );
    for prompt in DEFERRED_CONTEXTS {
        let run = OnTheDevice::running(
            &dir,
            &device,
            Asked {
                prompt: Some(prompt),
                generated: 1 + STEPS_AFTER_A_PREFILL,
                settle: 0,
                sampling: false,
                tracing: false,
            },
        );
        let (.., bytes) = since(run.submitted[0], run.submitted[1]);
        eprintln!(
            "{prompt:>8}{:>10}{:>11}{:>10}{:>10}{:>12}",
            format!("{:.2?}", run.steps[0]),
            format!("{:.2?}", run.steps[1]),
            format!("{:.2?}", run.steps[2]),
            format!("{:.2?}", run.steps[3]),
            format!("{:.0} MiB", bytes as f64 / (1u64 << 20) as f64),
        );

        // **The step after a prefill is not a decode step, and this is the shape
        // of that rather than its duration.** What it dispatches is what the
        // step after it dispatches, so nothing it does is extra work — which is
        // what makes the duration a question about waiting.
        let (dispatches, ..) = since(run.submitted[1], run.submitted[2]);
        let (after, ..) = since(run.submitted[2], run.submitted[3]);
        assert_eq!(
            dispatches, after,
            "the step after a {prompt}-token prefill dispatched more than the step after it"
        );
    }

    // The step alone, so that the profile and the trips describe it rather than
    // a mean over it and the steps that are not paying for anything: one
    // prefill, one decode step, nothing to divide the account by.
    let alone = OnTheDevice::running(
        &dir,
        &device,
        Asked {
            prompt: Some(*DEFERRED_CONTEXTS.last().expect("the sweep has a longest")),
            generated: 2,
            settle: 0,
            sampling: false,
            tracing: false,
        },
    );
    eprintln!("{}", step_table(&alone.measured()));
    eprintln!("{}", round_trip_table(&alone.measured()));
}

/// The prompts the deferred step is looked for at, which straddle the length it
/// appears at: 97 tokens is the cross-engine table's shortest and has no such
/// step, and the two longest are where it records 736 and 783 ms.
const DEFERRED_CONTEXTS: [usize; 5] = [97, 193, 385, 769, 1537];

/// Steps a generation runs past its prefill where the question is which of them
/// is slow, which is three: the one that pays for the prefill, and two behind it
/// to say what a step costs when nothing is paying for anything.
const STEPS_AFTER_A_PREFILL: usize = 3;

/// The contexts a decode step is priced at once the workload is a coding one.
///
/// **Every decode figure this repo has ever taken tops out at 769 tokens** —
/// the cross-engine table's longest prompt — and a coding turn opens at
/// thousands and grows all session. Whether a step is linear in the context or
/// plateaus cannot be told apart over an eightfold range, and it is what decides
/// whether the engine is usable at 32k at all.
///
/// Stopping at 8192 is a wall-time judgement rather than a claim: a prefill here
/// is the cost of reaching a row, and the two lengths past this one are 5 and 14
/// minutes of it. `what_the_attention_step_costs_as_the_context_grows` carries
/// the shape out to 65536 for the price of a dispatch, and this is what says the
/// shape it carries is the step's.
const GROWN_CONTEXTS: [usize; 6] = [97, 385, 769, 2048, 4096, 8192];

/// **What a decode step costs, and what a sequence holds, as the context
/// grows.**
///
/// Three columns nothing in this repo had: a decode step past 769 tokens, the
/// peak resident set beside it, and both against the same lengths the reference
/// is swept over by `reference/scripts/context_sweep.py`. The last is the one
/// that had to be measured rather than assumed — the cross-engine table found
/// mlx-vlm flat from 97 to 769 and this file read that as flat, where the
/// reference's own sweep puts it at 24 ms to 769 and 78 ms from 2048 on.
///
/// **The memory column is a claim about the architecture and not about this
/// run.** 35 of the 42 layers cap at a 512-token window, so a sequence's keys
/// and values should grow on 7 of them; `KeyValues::reserve` allocates against
/// the keys a sequence has seen instead, and
/// `what_a_context_costs_in_keys_and_values` is where that is weighed exactly.
/// What this adds is what the whole process holds around it.
///
/// Nothing asserts a slope. What is asserted is that each row generated the
/// tokens it was asked for, so a row is not a run that stopped early.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn what_a_decode_step_costs_as_the_context_grows() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    eprintln!(
        "{:>8}{:>11}{:>11}{:>11}{:>11}{:>11}{:>10}",
        "context", "prefill", "a token", "median", "device", "tokens/s", "peak"
    );
    for context in GROWN_CONTEXTS {
        let run = OnTheDevice::decoding_at(&dir, &device, context, false);
        let step = run.each_charged_step();
        assert_eq!(
            run.got.len(),
            run.asked.generated,
            "the run at {context} tokens stopped after {} of {} tokens",
            run.got.len(),
            run.asked.generated
        );
        eprintln!(
            "{context:>8}{:>11}{:>11}{:>11}{:>11}{:>11}{:>10}",
            format!("{:.0?}", run.steps[0]),
            format!("{step:.2?}"),
            format!("{:.2?}", run.median_charged_step()),
            format!("{:.2?}", run.profile.gpu()),
            format!("{:.1}", 1.0 / step.as_secs_f64()),
            format!("{:.2} GiB", run.peak as f64 / (1u64 << 30) as f64),
        );
    }
}

/// The contexts the sequence question is asked at: the recorded prompt's own,
/// one in the middle, and the longest this file measures.
///
/// **Three rather than one because the answer may depend on the context and
/// that is the whole risk.** A decode step's attention reads every key there is,
/// so a grid that followed the span would make the sequence a different sequence
/// at every step — which is exactly what a table taken at one length could not
/// see.
const TRACED_CONTEXTS: [usize; 3] = [97, 2048, 8192];

/// **What actually changes between one decode step and the next**, which every
/// claim about encoding a step's dispatches once and reusing them rests on and
/// which nothing in this repo had asked.
///
/// A step writes down each dispatch it encodes — the entry, the pipeline, the
/// grid, and what filled every argument slot — and consecutive steps are held
/// against each other. What comes back is three numbers that mean three
/// different things:
///
/// - **commands changed** is a dispatch that is not the same command at all:
///   another entry, another pipeline, another grid. Any of those and the
///   sequence has to be written rather than patched, and a *count* that moved
///   means there is no one sequence to reuse.
/// - **bound** is a slot naming a different allocation, which is a patch —
///   `setKernelBuffer:` on one command.
/// - **inline** is a slot whose bytes changed. An indirect command has no
///   inline binding at all, so each of these is an allocation the engine does
///   not currently make, and their count is what that would cost.
///
/// The last column is the other half: what the Metal calls encoding a step cost
/// against what the profile charges the whole `dispatch encode` row. **A
/// sequence encoded once removes the first and not the second**, so the two
/// figures are what any projection has to be built on.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn what_changes_between_two_decode_steps() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    eprintln!(
        "{:>8}{:>12}{:>10}{:>10}{:>9}{:>9}{:>13}{:>12}",
        "context", "dispatches", "commands", "slots", "bound", "inline", "metal encode", "the row"
    );
    for context in TRACED_CONTEXTS {
        let run = OnTheDevice::running(&dir, &device, Asked::decoding_at(context).traced());
        // The prefill is the first trace and is a different sequence from every
        // step behind it; the steps this is about are the ones the profile
        // charges.
        let steps = &run.traces[run.charged_from()..];
        assert!(steps.len() >= 2, "two steps to hold against each other");

        let differences: Vec<Difference> = steps
            .windows(2)
            .map(|pair| Difference::between(&pair[0], &pair[1]))
            .collect();
        let worst =
            |of: fn(&Difference) -> usize| differences.iter().map(of).max().unwrap_or_default();
        let metal: Duration = steps
            .iter()
            .map(|step| {
                step.iter()
                    .map(|dispatch| dispatch.encoding)
                    .sum::<Duration>()
            })
            .sum::<Duration>()
            / steps.len() as u32;

        eprintln!(
            "{context:>8}{:>12}{:>10}{:>10}{:>9}{:>9}{:>13}{:>12}",
            steps[0].len(),
            worst(|difference| difference.commands_changed.len()),
            worst(|difference| difference.slots),
            worst(|difference| difference.bound_changed.len()),
            worst(|difference| difference.inline_changed.len()),
            format!("{metal:.2?}"),
            format!("{:.2?}", run.profile.elapsed(Op::Encode)),
        );

        // Which kernels the patches fall on, since what a patch costs is one
        // `setKernelBuffer:` and what an inline argument costs is an allocation
        // the engine does not make today.
        let mut inline: BTreeMap<&str, usize> = BTreeMap::new();
        let mut bound: BTreeMap<&str, usize> = BTreeMap::new();
        let last = differences.last().expect("a pair");
        for (at, _) in &last.inline_changed {
            *inline.entry(steps[0][*at].entry.as_str()).or_default() += 1;
        }
        for (at, _) in &last.bound_changed {
            *bound.entry(steps[0][*at].entry.as_str()).or_default() += 1;
        }
        for (what, rows) in [("inline", &inline), ("bound", &bound)] {
            let named: Vec<String> = rows
                .iter()
                .map(|(entry, count)| format!("{entry} {count}"))
                .collect();
            eprintln!("  {what:>6} slots that changed: {}", named.join(", "));
        }

        assert_eq!(
            differences
                .iter()
                .filter(|difference| difference.reusable())
                .count(),
            differences.len(),
            "at {context} tokens some pair of steps is not the same sequence: {differences:?}"
        );
    }
}

/// **What a decode step's dispatches actually have to wait for**, which is where
/// a concurrent pass puts a barrier and nowhere else.
///
/// This is what a step's groups come to, derived rather than read off the
/// layer code: which allocations filled each dispatch's slots, and which of
/// those slots the kernel's own source declares it may write. It is the same
/// division [`Batch`](inkling_metal::Batch) encodes, through the same
/// [`Open`](inkling_metal::ordering) — so a count here that moved without the
/// engine moving with it is not possible.
///
/// **The counts and not a price.** What removing these barriers is worth is a
/// paired benchmark and not a count times a constant, for the reason
/// [`inkling_metal::ordering`] gives at length: the fixed-cost reading of
/// `what_a_barrier_costs_the_device_against_the_dispatches_it_separates` gets
/// the sign wrong.
///
/// **At three contexts, because a span that grows is the one thing that could
/// change the shape.** `what_changes_between_two_decode_steps` says the sequence
/// is the same commands at 97, 2048 and 8192 keys; this says whether the
/// dependencies between them are too.
///
/// The count here is 918 where `where_a_decode_step_spends_its_time` reports 876,
/// and the gap is one `attention_combine` a layer: that table is taken at the
/// structured prompt's own 9 to 15 keys, which is short enough that a layer's
/// attention needs no second dispatch to fold its key tiles together. Every
/// context this covers is past that, so 918 is what a step of any length anyone
/// has makes.
#[test]
#[ignore = "needs a checkpoint: `INKLINGRS_CHECKPOINT=… cargo test -- --ignored`"]
fn what_a_decode_steps_dispatches_have_to_wait_for() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    ordering_header("context");
    for context in TRACED_CONTEXTS {
        let run = OnTheDevice::running(&dir, &device, Asked::decoding_at(context).traced());
        // The prefill is a different sequence from every step behind it; the
        // steps this is about are the ones the profile charges.
        let from = run.charged_from();
        let steps: Vec<Groups> = run.traces[from..]
            .iter()
            .zip(&run.boundaries[from..])
            .map(|(step, boundaries)| Groups::over(step, boundaries))
            .collect();
        assert!(steps.len() >= 2, "two steps to hold against each other");
        assert!(
            steps.windows(2).all(|pair| pair[0] == pair[1]),
            "at {context} keys two steps divide into different groups"
        );
        let step = &steps[0];

        derived_barriers_are_the_encoded_ones(&run, step, &format!("{context} keys"));
        ordering_row(context, step);
        eprintln!("  {}", spread(step));
        for (held, group) in step.shapes() {
            eprintln!("  {held:>6} of {}", group.join(", "));
        }
    }
}

/// The header the two ordering tables share, and one row of either.
///
/// Written once because the two are the same eight columns about the same
/// division of the same kind of sequence, and a prefill's row that drifted from
/// a decode step's would be two tables a reader cannot hold side by side —
/// which is the whole use they are put to.
fn ordering_header(first: &str) {
    eprintln!(
        "{:>8}{:>12}{:>8}{:>9}{:>10}{:>9}{:>9}{:>11}",
        first, "dispatches", "passes", "groups", "barriers", "a group", "widest", "of them"
    );
}

fn ordering_row(at: usize, groups: &Groups) {
    eprintln!(
        "{at:>8}{:>12}{:>8}{:>9}{:>10}{:>9}{:>9}{:>11}",
        groups.dispatches(),
        groups.passes(),
        groups.groups(),
        groups.barriers(),
        format!("{:.2}", groups.average()),
        groups.widest(),
        format!(
            "{:.0}%",
            100.0 * groups.barriers() as f64 / groups.dispatches() as f64
        ),
    );
}

/// **The one assertion that makes either table a fact about the engine rather
/// than a report about a recording.**
///
/// The division and the barriers [`Batch`](inkling_metal::Batch) encoded come
/// out of the same `Open` asked the same question, so they have to be the same
/// number — and a milestone whose whole claim is that the barriers left are the
/// ones a dependency needs cannot leave that unchecked. **A missing barrier is a
/// race that is correct most of the time**, so what this guards against is a
/// failure that would otherwise be found by a wrong token months later.
///
/// Every charged step and not their average: the sequence is the same commands
/// every step — `what_changes_between_two_decode_steps` is what says so — so the
/// barriers are the same count every step, and an average would let one deviant
/// step hide behind the others. A prefill charges one step and the arithmetic is
/// the same one.
fn derived_barriers_are_the_encoded_ones(run: &OnTheDevice, groups: &Groups, at: &str) {
    assert_eq!(
        run.barriers,
        groups.barriers() as u64 * u64::from(run.charged_steps()),
        "at {at} the engine encoded {} barriers over {} steps where the division needs {} each",
        run.barriers,
        run.charged_steps(),
        groups.barriers()
    );
}

/// **What a prefill's dispatches have to wait for, which nobody had asked.**
///
/// D4 derived a decode step's dependency graph, made its dispatches concurrent
/// and measured the step 17.254 → 16.287 ms. The mechanism is in the encoder —
/// [`Batch`](inkling_metal::Batch) opens every pass `MTLDispatchTypeConcurrent`
/// and barriers what the open group's slots say to — so **it already reaches a
/// prefill and has since the day it landed**; nothing about it is gated on a
/// shape. What had never been read is what it finds there.
///
/// **The reason to expect more of it here is arithmetic and the reason to
/// measure it is that the arithmetic has been wrong before.** T2's limiter table
/// puts barriers at 10% of the prefill matmul, and a prefill issues over a
/// thousand dispatches with far more independent work than a one-row step, where
/// 546 of 670 groups were singletons. Whether that independence is *in the
/// sequence* is what a width distribution says and a mean does not — see
/// [`Groups::widths`](inkling_metal::ordering::Groups::widths).
///
/// **At the same three lengths the decode table uses**, so the two can be read
/// down one column: a prefill of 97, 2048 and 8192 tokens against a decode step
/// at 97, 2048 and 8192 keys.
#[test]
#[ignore = "needs a checkpoint: `INKLINGRS_CHECKPOINT=… cargo test -- --ignored`"]
fn what_a_prefills_dispatches_have_to_wait_for() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    ordering_header("tokens");
    for tokens in TRACED_CONTEXTS {
        let run = OnTheDevice::running(&dir, &device, Asked::prefilling(tokens).traced());
        // A prefill is the whole of what this run does, so its one trace is the
        // one the profile charges and the barriers are all of them.
        let prefill = Groups::over(&run.traces[0], &run.boundaries[0]);

        derived_barriers_are_the_encoded_ones(&run, &prefill, &format!("{tokens} tokens"));
        ordering_row(tokens, &prefill);
        // **The distribution rather than the shapes.** A decode step's groups
        // are a few dozen distinct entry lists a reader can hold against the
        // layer they come from; a prefill's are hundreds, and what the question
        // needs is how many dispatches a group holds rather than which.
        eprintln!("  {}", spread(&prefill));
    }
}

/// A division's width distribution as one line: how many of its groups are a
/// single dispatch, and how the rest divide.
///
/// **The singletons first because they are the finding.** A group of one is a
/// dispatch a concurrent pass found nothing to overlap with, so the share of
/// them is what says how much of a sequence the ordering can reach at all — and
/// it is the one column on which a prefill and a decode step read alike.
fn spread(groups: &Groups) -> String {
    let widths = groups.widths();
    let singletons = widths
        .iter()
        .find(|(width, _)| *width == 1)
        .map_or(0, |(_, count)| *count);
    format!(
        "{singletons} of {} groups are one dispatch, and the rest divide {}",
        groups.groups(),
        widths
            .iter()
            .filter(|(width, _)| *width > 1)
            .map(|(width, count)| format!("{count}×{width}"))
            .collect::<Vec<String>>()
            .join(", ")
    )
}

/// The three prompt lengths this file's prefill figures are quoted at, and the
/// three the README compares against the reference.
const PREFILL_WALL_LENGTHS: [usize; 3] = [97, 385, 769];

/// The two of those a prefill is *diagnosed* at. The 97-token one is left out on
/// purpose: it is the only length short enough that a run of layers still
/// merges, so its rows carry a scheduling difference the other two do not, and
/// the question there is what a prefill's kernels cost rather than how they are
/// submitted. A wall time is the one figure that scheduling belongs in, which is
/// why the sweep above keeps all three.
const PREFILL_LENGTHS: &[usize] = &[PREFILL_WALL_LENGTHS[1], PREFILL_WALL_LENGTHS[2]];

/// **Where a prefill's time goes, kernel by kernel**, which is the same
/// question `which_kernels_own_a_decode_step` asks of the other regime and has
/// never been asked of this one.
///
/// **The shapes are entirely different and the ranking was not assumed to
/// survive them.** A decode step is one row through every kernel in the model;
/// a prefill is hundreds through the same ones, and the model's own arithmetic
/// says the two cannot scale alike — a token reads six of each MoE layer's 256
/// experts, so 769 tokens name up to every expert a layer has where one token
/// names six. Whether that arrives as bandwidth, as occupancy or as neither is
/// what the table answers.
///
/// **Against a standing hypothesis, and it is on record as a hypothesis.** M9
/// left behind the claim that every `(head, query)` threadgroup re-reading all
/// keys is the next order of magnitude here. Nothing in this repo has measured
/// it, and this project's estimates have a poor record — `lm_head` was
/// predicted at 54% of a step and measured 7.6%, and the dequantisation
/// bandwidth model died the moment a different kernel dominated. So the table
/// is taken first and the hypothesis is read off it rather than the other way
/// round.
///
/// Two lengths rather than one, because a row that is flat in the tokens and a
/// row that is quadratic in them are the same row at a single length. Nothing
/// asserts a share; what is asserted is that the accounting adds up and that
/// every dispatch was timed.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn where_a_prefill_spends_its_time() {
    let Some((dir, device)) = sampling_device() else {
        return;
    };

    for &tokens in PREFILL_LENGTHS {
        // **Two unsampled prefills and not one, and they go either side of
        // nothing.** A decode step's table takes one, to say how close the rows
        // sum to the device time of a step nobody was asking about. A prefill
        // needs the pair, because here that comparison comes out the wrong way
        // round — the sampled wall time is the *shorter* — and a single
        // unsampled run could not tell an effect of sampling from this
        // machine's own state or from the first read of a length faulting its
        // pages in. Two that agree with each other can.
        let first = OnTheDevice::prefilling(&dir, &device, tokens, false);
        let unsampled = OnTheDevice::prefilling(&dir, &device, tokens, false);
        let run = OnTheDevice::prefilling(&dir, &device, tokens, true);
        eprintln!(
            "{}\n  against {:.2?} then {:.2?} of prefill and {:.2?} then {:.2?} of device time \
             with nothing sampling, so the rows carry {:+.1}% of asking\n{}",
            step_table(&run.measured()),
            first.each_charged_step(),
            unsampled.each_charged_step(),
            first.profile.gpu(),
            unsampled.profile.gpu(),
            100.0
                * (run.profile.dispatched().as_secs_f64() / unsampled.profile.gpu().as_secs_f64()
                    - 1.0),
            round_trip_table(&run.measured()),
        );

        a_sampled_prefills_accounting_adds_up(&run, tokens);
        let (_, submissions, ..) = run.per_charged_step();
        // **A prefill of hundreds of tokens merges nothing**, because one of
        // its layers alone passes the bytes a run may hold — so it is a
        // submission a layer where a decode step is two for the whole stack,
        // and a run that started merging them would be a different engine from
        // the one these rows describe. The two dense layers are 18 dispatches
        // each and share one, which is why this is not a layer plus a head
        // exactly.
        let layers = fixture::config(&dir).text_config.num_hidden_layers as u64;
        assert!(
            submissions >= layers,
            "a prefill of {tokens} tokens went in {submissions} submissions, which is fewer than \
             its {layers} layers and so is a merged run rather than a layer at a time"
        );
    }
}

/// What a sampled prefill's rows have to be true of for the table above them to
/// describe the prefill rather than a fraction of it: the rows are self time
/// inside the wall time they were measured in, the device cannot have executed
/// for longer than the prefill that waited for it, and every dispatch came back
/// with a pair of timestamps.
///
/// The last is the one that fails quietly on its own. A device that dropped a
/// timestamp writes `MTLCounterErrorValue` and this side charges it nothing,
/// which is a row short rather than a failure.
fn a_sampled_prefills_accounting_adds_up(run: &OnTheDevice, tokens: usize) {
    let (timed, _) = what_was_sampled(&run.profile);
    let (step, accounted) = (run.each_charged_step(), run.profile.total());
    assert!(
        accounted <= step,
        "the rows sum to {accounted:.2?} inside a {step:.2?} prefill of {tokens} tokens"
    );
    assert!(
        run.profile.gpu() < step,
        "the device reported executing for {:.2?} of a {step:.2?} prefill of {tokens} tokens",
        run.profile.gpu()
    );
    let (dispatches, ..) = run.per_charged_step();
    assert_eq!(
        timed, dispatches,
        "a prefill of {tokens} tokens did not have all its dispatches timed"
    );
}

/// The prompts a prefill is diagnosed at once the workload is a coding one.
///
/// The four `what_a_decode_step_costs_as_the_context_grows` stops inside and
/// past, and its own list stops where this one does for a reason of its own —
/// a decode step at 65536 keys is a dispatch and a prefill to it is half an
/// hour, so the two lists are the same question asked at different prices.
///
/// **The two under a thousand are where a coding session actually lives.** A
/// turn's opening prompt is thousands of tokens and is prefilled once; every
/// turn after it adds about 321, and K1's kept cache is what makes the delta
/// rather than the whole prompt the thing that gets prefilled. So the lengths
/// this engine is asked at most often are the two the table used to open above.
const LONG_PREFILLS: [usize; 6] = [321, 512, 2048, 4096, 8192, 16384];

/// **Where a prefill's time goes at a prompt a coding turn opens with**, which
/// is the same table as `where_a_prefill_spends_its_time` at 385 and 769 tokens
/// and is the only one that can say which of its terms grow faster than the
/// prompt.
///
/// **The attention rows are two here and one there.** 35 of this checkpoint's
/// layers stop at a 512-key window and 7 reach every key the sequence has, so
/// the arithmetic says one term should be `n × 512` and the other `n²/2` — and
/// summed into a single row the two are a number about neither. `Kernel::under`
/// is what splits them, and it splits them everywhere rather than here.
///
/// **One sampled run a length and no unsampled one beside it**, which is a
/// departure from the two-length table above and is a wall-clock judgement
/// rather than a claim: these four prefills are 3.5 minutes of device time
/// between them and the pair would be 7. What the pair buys there is telling
/// an effect of sampling from this machine's own state, and what stands in for
/// it is `bench prefill` at the same four lengths — quoted in the README beside
/// these rows rather than run inside this process, so a reader can see both.
/// The row that says whether that substitution held is the sampled wall time
/// printed here against the unsampled one printed there.
///
/// Nothing asserts a share or a slope. What is asserted is what the shorter
/// table asserts: the rows are self time inside the wall time they were
/// measured in, the device cannot have executed for longer than the prefill that
/// waited for it, and every dispatch came back with a pair of timestamps — so a
/// row is not quietly short.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn where_a_long_prefill_spends_its_time() {
    let Some((dir, device)) = sampling_device() else {
        return;
    };

    for &tokens in &LONG_PREFILLS {
        let run = OnTheDevice::prefilling(&dir, &device, tokens, true);
        eprintln!("{}", step_table(&run.measured()));
        a_sampled_prefills_accounting_adds_up(&run, tokens);
    }
}

/// **What a prefill costs at each of the three lengths, and what it reads to do
/// it.**
///
/// The wall time this repo's prefill claims are made in, beside the two figures
/// that say where it goes: the device's own clock, and the bytes the dispatches
/// declare they move. Nothing here has produced those three together before —
/// `where_a_prefill_spends_its_time` divides one length up by kernel, and the
/// wall times the README quotes were taken by hand.
///
/// **Bytes a token is the column to read.** A prefill's whole gap to the
/// reference is byte count rather than execution — the packed matmul is nearer
/// this machine's bandwidth at prefill shape than anywhere else this file
/// measures — so what a change to it has to move is this column, and what a
/// change to it has to *not* move is the tokens. A prefill that amortised
/// nothing reads the same bytes a token whatever the prompt length; one that
/// amortises something reads fewer as the prompt grows, and the shape of that
/// column is the finding.
///
/// **Three runs a length: two unsampled and one asked.** The bytes are declared
/// per dispatch and only reach the profile when the device is timing each of
/// them, and timing each of them costs — see `what_timing_each_dispatch_costs`
/// — so the wall time is read off the runs that were not asked and the bytes off
/// the run that was. Two unsampled and not one for the reason
/// `where_a_prefill_spends_its_time` gives: the first run of a length is the
/// first read of the pages that length's rows fault in, and a single figure
/// cannot tell that from this machine's state. Both are printed, and a pair that
/// disagree is the reading rather than a mean of them.
///
/// The dispatch count is checked across the sampled run and the unsampled one,
/// which is the premise the row rests on: the bytes describe the same dispatches
/// the wall time was spent on, because sampling changes where the passes are cut
/// and nothing about what they read.
///
/// **What a prefill holds is the allocation columns and not a resident set.**
/// Six prefills in one process share a resident set that only grows, so what
/// RSS says here is what the process has touched rather than what a prefill
/// costs — `the_whole_stack_holds_its_resident_set_under_a_bound` is where a
/// bound on that is asserted, in a process that runs one pass. What is a
/// prefill's own is what the device allocated for it, which is counted per
/// dispatch and freed as each layer's command buffer completes: the buffers
/// column is how many and the MiB column is how much, and a change that made a
/// layer hold something new moves both.
///
/// Nothing asserts a duration. What is asserted is that every dispatch was
/// timed, so the byte column describes the whole prefill.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn what_a_prefill_costs_at_each_length() {
    let Some((dir, device)) = sampling_device() else {
        return;
    };

    eprintln!(
        "  {:<12}{:>9}{:>9}{:>9}{:>12}{:>12}{:>10}{:>12}{:>10}{:>10}",
        "prefill",
        "wall",
        "again",
        "device",
        "dispatches",
        "submissions",
        "buffers",
        "allocated",
        "moved",
        "a token"
    );
    for tokens in PREFILL_WALL_LENGTHS {
        let first = OnTheDevice::prefilling(&dir, &device, tokens, false);
        let run = OnTheDevice::prefilling(&dir, &device, tokens, false);
        let sampled = OnTheDevice::prefilling(&dir, &device, tokens, true);
        let (timed, moved) = what_was_sampled(&sampled.profile);
        let (dispatches, submissions, buffers, bytes) = run.per_charged_step();

        eprintln!(
            "  {:<12}{:>9}{:>9}{:>9}{:>12}{:>12}{:>10}{:>12}{:>10}{:>10}",
            format!("{tokens} tokens"),
            format!("{:.2?}", first.each_charged_step()),
            format!("{:.2?}", run.each_charged_step()),
            format!("{:.2?}", run.profile.gpu()),
            dispatches,
            submissions,
            buffers,
            format!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64),
            format!("{:.0} GB", moved as f64 / 1e9),
            format!("{:.0} MB", moved as f64 / 1e6 / tokens as f64),
        );

        assert_eq!(
            timed, dispatches,
            "a prefill of {tokens} tokens read its bytes off dispatches the run it is timed \
             beside did not make"
        );
        assert!(
            run.profile.gpu() < run.each_charged_step(),
            "the device reported executing for {:.2?} of a {:.2?} prefill of {tokens} tokens",
            run.profile.gpu(),
            run.each_charged_step()
        );
    }
}

/// **What asking costs**, over seven alternating pairs.
///
/// The instrumentation must not change what it measures, and on this hardware it
/// does: a dispatch can only be timed by being a compute pass of its own, and a
/// pass boundary is not free. So the honest thing is to run the same step both
/// ways rather than to claim the difference is small — and the reason
/// `Device::time_each_dispatch` is off unless somebody asks for it.
///
/// Alternating rather than one run each, because T1 established that this
/// machine's own state moves a decode step by more than the effect a single pair
/// would be measuring.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn what_timing_each_dispatch_costs() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    if !device.times_a_pass() {
        eprintln!("skipping: this device does not sample at a stage boundary");
        return;
    }
    const PAIRS: usize = 7;
    assert!(
        !device.timing_each_dispatch(),
        "a device that has just opened is already sampling"
    );

    let mut off = Vec::new();
    let mut on = Vec::new();
    let mut dispatches = 0;
    for _ in 0..PAIRS {
        for (sampling, taken) in [(false, &mut off), (true, &mut on)] {
            let run = OnTheDevice::generating(&dir, &device, sampling);
            dispatches = run.per_decode_step().0;
            taken.push((run.each_decode_step(), run.profile.gpu()));
        }
    }

    let mean = |taken: &[(Duration, Duration)], of: fn(&(Duration, Duration)) -> Duration| {
        taken.iter().map(of).sum::<Duration>() / taken.len() as u32
    };
    let (step_off, step_on) = (mean(&off, |run| run.0), mean(&on, |run| run.0));
    let (gpu_off, gpu_on) = (mean(&off, |run| run.1), mean(&on, |run| run.1));
    eprintln!(
        "over {PAIRS} alternating pairs, a decode step\n  unsampled {step_off:.2?}, of which the \
         device executed for {gpu_off:.2?}\n  sampled   {step_on:.2?}, of which the device \
         executed for {gpu_on:.2?}\n  timing each dispatch costs {:.2?} a step and {:.2?} of \
         device time, which is {:.0} µs a dispatch\n  the pairs: {:.2?}",
        step_on.saturating_sub(step_off),
        gpu_on.saturating_sub(gpu_off),
        1e6 * (gpu_on.saturating_sub(gpu_off)).as_secs_f64() / dispatches as f64,
        off.iter()
            .zip(&on)
            .map(|(off, on)| (off.0, on.0))
            .collect::<Vec<(Duration, Duration)>>(),
    );

    assert!(
        step_on >= step_off,
        "sampling made the step faster, which is a measurement of something else"
    );
    assert!(
        off.iter().zip(&on).all(|(off, on)| on.0 > off.0),
        "a pair moved the other way, so the mean is describing this machine's own state: \
         {off:.2?} against {on:.2?}"
    );
}

/// The router's own selection against the CPU's, over the trained gate of the
/// captured MoE layer and the hidden state the reference ran through it.
///
/// **This is the one thing the device now decides that the fixtures cannot
/// reach.** `inkling_core::moe` pins the whole routing computation to mlx-vlm —
/// the selection, the weights, the two scales — and the weights are still
/// computed there. What moved is the top-k, so what has to be said here is that
/// the same 256 scores rank to the same six on both sides, over the trained gate
/// rather than over synthetic logits.
///
/// **The set and not the order.** `mx.argpartition` states nothing about the
/// order the k come back in and MLX's own two streams disagree about it, so the
/// engine depends on the set alone — see `SparseMoe::route`. The sets are
/// compared sorted, which is what both sides promise.
///
/// The two `sigmoid`s are the GPU's `exp` and libm's and agree to a few ulps, so
/// the selection can only part company where two scores straddling the last slot
/// are that close. `the_trained_selection_clears_the_gates_float32_drift` in
/// `inkling_core` measures the trained margin at four times the drift a float32
/// gate already introduces, which is decades above a few ulps — and that is a
/// property of these eight tokens rather than a guarantee, which is why this is
/// stated rather than assumed.
#[test]
fn the_router_selects_the_experts_the_cpu_selects_over_a_real_gate() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = fixture::config(&dir).text_config;
    let matmul = DenseMatmul::new(&device).expect("the dense matmul compiles");
    let kernel = Router::new(&device).expect("the router compiles");

    let module = "language_model.model.layers.2.mlp";
    let of = |name: &str| {
        ckpt.tensor(&format!("{module}.{name}"))
            .unwrap_or_else(|err| panic!("the checkpoint holds {name}: {err}"))
            .to_f32()
            .unwrap_or_else(|| panic!("{name} widens"))
    };
    let correction_bias = of("e_score_correction_bias");
    let moe_config = MoeConfig::for_layer(&config, 2).expect("a MoE layer has a router");
    assert_eq!(
        correction_bias.len(),
        moe_config.n_routed,
        "one bias per routed expert"
    );

    // The gate's own logits, formed by the kernel that will feed the router in
    // anger rather than by the CPU — so what this compares is the two rankings
    // of one row of scores and not two rows.
    let gate = Bf16::open(&ckpt, ROUTER_GATE).expect("the checkpoint holds the gate");
    let logits = DenseWeight::wrap(&device, &matmul, &gate)
        .expect("the gate wraps")
        .multiply(&normed_state())
        .expect("the dispatch completes");

    let weighing = RouterWeights::new(&device).expect("the weighting compiles");
    let router = LayerRouter::new(
        &device,
        &kernel,
        &weighing,
        moe_config,
        &correction_bias,
        of("global_scale")[0],
    )
    .expect("the router stands up");
    let got = router.select(&logits).expect("the dispatch completes");

    let want = SparseMoe::new(
        moe_config,
        GateWeights {
            gate: Gate::Backend {
                hidden: gate.in_dim(),
            },
            correction_bias: &correction_bias,
            global_scale: of("global_scale")[0],
        },
    )
    .route(&logits);

    let sets = |picked: &[usize]| -> Vec<Vec<usize>> {
        picked
            .chunks_exact(moe_config.top_k)
            .map(|row| {
                let mut row = row.to_vec();
                row.sort_unstable();
                row
            })
            .collect()
    };
    let mine: Vec<usize> = got.iter().map(|expert| *expert as usize).collect();
    eprintln!(
        "layer 2, {} tokens: the device selected {:?}",
        mine.len() / moe_config.top_k,
        &mine[..moe_config.top_k]
    );
    assert_eq!(sets(&mine), sets(want.experts()), "the selected sets");
    assert!(
        sets(&mine).iter().any(|row| *row != sets(&mine)[0]),
        "eight tokens that all routed the same way would say nothing"
    );
}

/// The four tensors one layer's attention reads that are not projections,
/// widened out of the checkpoint.
///
/// `CheckpointWeights` widens these for the layer it stands up and does not hand
/// them out — the CPU path reads them through `AttentionWeights` and the device
/// path through `LayerStep` — so a case that drives a layer's attention from
/// outside the stack opens them itself.
struct HeadTensors {
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_sconv: Vec<f32>,
    v_sconv: Vec<f32>,
}

impl HeadTensors {
    fn open(ckpt: &Checkpoint, layer: usize) -> Self {
        let of = |name: &str| {
            ckpt.tensor(&format!(
                "language_model.model.layers.{layer}.self_attn.{name}"
            ))
            .unwrap_or_else(|err| panic!("the checkpoint holds {name}: {err}"))
            .to_f32()
            .unwrap_or_else(|| panic!("{name} widens"))
        };
        Self {
            q_norm: of("q_norm.weight"),
            k_norm: of("k_norm.weight"),
            k_sconv: of("k_sconv.conv.weight"),
            v_sconv: of("v_sconv.conv.weight"),
        }
    }

    /// One call's step, over the layer these came from.
    fn step<'a>(
        &'a self,
        layer: &'a inkling_core::LayerPacked<'_>,
        x: &'a [f32],
        q_offset: usize,
    ) -> LayerStep<'a> {
        let shape = layer.config;
        let channels = shape.kv_channels();
        LayerStep {
            sdpa: Sdpa::new(shape.heads, shape.kv_heads, shape.head_dim),
            mask: BandedMask::new(shape.d_rel, &layer.rel_proj, shape.sliding),
            x,
            input_layernorm: Some(&layer.input_layernorm),
            eps: shape.rms_norm_eps,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            k_sconv: ShortConv::new(channels, &self.k_sconv),
            v_sconv: ShortConv::new(channels, &self.v_sconv),
            // The floor is 128000 tokens, so nothing eight rows can reach makes
            // a `tau` that is not exactly 1 — and both paths would scale alike
            // anyway, the queries being scaled before either hands them over.
            q_taus: None,
            bias_taus: None,
            q_offset,
        }
    }
}

/// One layer's whole attention against its own pieces run apart, driven a chunk
/// at a time through one cache.
///
/// **The claim the seam rests on, and it has to be made on a real layer.**
/// `LayerProjections::layer` answers everything between a hidden state and
/// `o_proj` — five projections whose widths have to pair with each other, with
/// two convolutions of the key's own channel count, with two head norms of the
/// head's, and with a band of the layer's `d_rel`. The hermetic cases in
/// `projections::tests` are cut from three fixture tensors that map from 4096 to
/// 64, 64 and 2, and no assignment of those to five slots is a layer: there is
/// nothing there for `o_proj` to map back from. So the shape this is about is
/// the checkpoint's, and this is where the checkpoint is.
///
/// What it is measured against is the same five projections and the same four
/// operations with the keys held *here* — which is the path
/// `Attention::attend` takes when a backend answers `None`. Exact equality
/// rather than a tolerance: both run the same kernels over the same floats, and
/// the only thing that differs is whether the span was copied over for the call
/// or left where the layer put it. A stride that reached the arithmetic would
/// show in the last bits.
///
/// Four chunks — a five-token prefill and three single decodes — because a
/// prefill alone would say nothing about the cache and a decode alone nothing
/// about a call with rows to convolve against each other.
#[test]
fn a_layers_whole_attention_matches_its_pieces_run_apart() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = fixture::config(&dir).text_config;
    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");

    let kernels = LayerKernels::compile(&device).expect("the layer kernels compile");
    let packed = weights.layer_projections();
    let layer = &packed[LAYER];
    let five = LayerProjections::wrap(&device, &kernels, layer, 0).expect("the layer wraps");

    let shape = layer.config;
    let heads = HeadTensors::open(&ckpt, LAYER);

    // The reference's own normed state, which is eight rows of the width every
    // one of these projections maps from.
    let x = normed_state();
    let fresh = || AttentionCache::new(shape, config.sconv_kernel_size);
    let (mut resident, mut apart) = (fresh(), fresh());
    let (mut keys, mut values) = (Vec::new(), Vec::new());
    let (mut at, mut last) = (0, Vec::new());
    let mut worst = 0.0f32;

    for rows in [5, 1, 1, 1] {
        let call = &x[at * HIDDEN..(at + rows) * HIDDEN];
        let step = heads.step(layer, call, at);

        let fused = five
            .layer(&mut resident, step)
            .expect("the layer answers for itself");

        let projected = five.normed_qkvr(call, &layer.input_layernorm, shape.rms_norm_eps);
        let convolved = step.convolved(&mut apart, &projected);
        keys.extend(convolved.k);
        values.extend(convolved.v);
        let whole = five.attend(AttentionStep {
            sdpa: step.sdpa,
            mask: step.mask,
            q: &convolved.q,
            k: &split_heads(&keys, shape.kv_heads, shape.head_dim),
            v: &split_heads(&values, shape.kv_heads, shape.head_dim),
            rel: &projected.r,
            taus: None,
            q_offset: at,
        });

        let deviation = deviation(&fused, &whole);
        assert!(
            deviation <= LAYER_TOLERANCE,
            "{rows} rows at offset {at}: deviation {deviation:e}"
        );
        worst = worst.max(deviation);
        at += rows;
        assert_eq!(resident.seen(), at, "the sequence's count");
        last = fused;
    }

    // And what the sequence carried is load-bearing. The same last row through a
    // layer that has seen nothing attends over itself alone, out of empty
    // convolution windows and at position zero — a different answer rather than
    // a near one, which is what says the two paths above agreed about something.
    let alone = five
        .layer(&mut fresh(), heads.step(layer, &x[(at - 1) * HIDDEN..], 0))
        .expect("the layer answers for itself");
    let carried = deviation(&alone, &last);
    eprintln!("worst deviation over four calls: {worst:e}, against {carried:e} for the state");
    assert!(carried > LAYER_TOLERANCE, "deviation {carried:e}");
}

/// The other shape in the model, which the head does not cover: one expert of a
/// `[256, 2048, 4096]` routed bank, 2048 rows where the head has 201024.
///
/// The gather is what the kernel does with that leading axis, and it is the
/// thing a synthetic bank cannot settle: a stride of 4 MB across 1.06 GiB of
/// wrapped mapping, on the bytes the engine will actually index. So the rows
/// here name experts that are far apart and repeat one of them, and each is
/// checked against decoding that expert's own slice.
///
/// The bank is wrapped rather than copied, which is what makes it cheap enough
/// to state on a real one at all: 1.06 GiB in 50 microseconds instead of 130
/// milliseconds and a second copy.
#[test]
fn the_gathered_matmul_reproduces_the_cpu_over_a_routed_bank() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");

    let packed = CorePacked::open(&ckpt, ROUTED_EXPERTS).expect("the checkpoint holds the bank");
    let bank = Packed::open(&ckpt, ROUTED_EXPERTS);
    let &[experts, out_dim, packed_width] = bank.codes.shape() else {
        panic!(
            "a bank is [experts, out, in/8], got {:?}",
            bank.codes.shape()
        )
    };
    assert_eq!(packed_width * CODES_PER_WORD, HIDDEN, "an expert's width");
    assert!(experts > 1, "a bank of one would not need the gather");

    let started = Instant::now();
    let resident = PackedBank::wrap(&device, &matmul, &packed, HIDDEN).expect("the bank wraps");
    let bytes = bank.codes.data().len() + bank.scales.data().len();
    eprintln!(
        "{ROUTED_EXPERTS}: {experts} experts of [{out_dim}, {HIDDEN}], {:.2} GiB wrapped in {:.2?}",
        bytes as f64 / (1u64 << 30) as f64,
        started.elapsed()
    );
    assert_eq!(resident.experts(), experts);
    assert_eq!(resident.out_dim(), out_dim);

    // Far apart in the bank and one of them repeated, which is the shape a
    // token's six-of-256 has. The last is the last expert with a neighbour to
    // be told apart from, which is as far into the bank as this can reach and
    // still check what it checks.
    let chosen: Vec<u32> = vec![6, 200, 6, experts as u32 - 2];
    let x: Vec<f32> = chosen.iter().flat_map(|_| x_row()).collect();

    let started = Instant::now();
    let got = resident
        .multiply(&chosen, &x)
        .expect("the dispatch completes");
    eprintln!(
        "{} rows gathered out of the bank in {:.2?}",
        chosen.len(),
        started.elapsed()
    );
    assert_eq!(got.len(), chosen.len() * out_dim);

    let slice = |view: &TensorView<'_>, index: usize| {
        let stride = view.data().len() / experts;
        view.data()[index * stride..][..stride].to_vec()
    };
    let mut weight = vec![0.0; out_dim * HIDDEN];
    let mut worst = 0.0f32;
    for (row, expert) in chosen.iter().enumerate() {
        let expert = *expert as usize;
        dequantize_blocks_into(
            &slice(&bank.codes, expert),
            &slice(&bank.scales, expert),
            &mut weight,
        )
        .expect("the expert decodes");
        let want = linear(&x_row(), &weight, HIDDEN);
        let mine = deviation(&got[row * out_dim..][..out_dim], &want);
        assert!(mine <= TOLERANCE, "expert {expert}: {mine:e}");
        worst = worst.max(mine);

        // Its neighbour in the bank is the same shape and different weights, so
        // a stride an expert out would run and be quietly wrong.
        dequantize_blocks_into(
            &slice(&bank.codes, expert + 1),
            &slice(&bank.scales, expert + 1),
            &mut weight,
        )
        .expect("the neighbour decodes");
        let want = linear(&x_row(), &weight, HIDDEN);
        let neighbour = deviation(&got[row * out_dim..][..out_dim], &want);
        assert!(
            neighbour > TOLERANCE,
            "experts {expert} and {} deviate by only {neighbour:e}",
            expert + 1
        );
    }
    eprintln!(
        "worst deviation over {} gathered rows: {worst:e}",
        chosen.len()
    );

    assert_eq!(
        got[..out_dim],
        got[2 * out_dim..3 * out_dim],
        "one expert, one input, twice"
    );
    assert_ne!(got[..out_dim], got[out_dim..2 * out_dim]);
}

/// The one weight in the model the quantiser left alone, on the checkpoint's
/// own bytes: a router's `[258, 4096]` bfloat16 gate against what the CPU makes
/// of the same tensor widened.
///
/// What the hermetic cases in `dense::tests` cannot settle is the same thing
/// they cannot settle for the packed matmul — a trained weight's spread of
/// magnitudes, and where the checkpoint put it. The second is the sharper half
/// here: the quant's shard headers are not padded, so a tensor can begin at an
/// odd byte and a wrap promising two-byte elements could not be pointed at one.
/// Reading it a byte at a time can, and the deviation is what says the pair of
/// bytes widens to the value the CPU widened.
///
/// The last two rows are the shared experts and the two above them are routed —
/// four rows the layer treats differently and the kernel does not — so a
/// dispatch that stopped short of the shared pair, or started past the routed
/// ones, is a wrong length rather than a near miss.
#[test]
fn the_dense_matmul_reproduces_the_cpu_over_a_real_router_gate() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let matmul = DenseMatmul::new(&device).expect("the dense matmul compiles");

    let gate = Bf16::open(&ckpt, ROUTER_GATE).expect("the checkpoint holds the gate");
    assert_eq!(gate.in_dim(), HIDDEN, "the gate maps from the hidden width");
    assert!(
        gate.out_dim() > 256,
        "{} rows is not the routed experts and the shared ones",
        gate.out_dim()
    );
    assert_eq!(
        gate.bytes().as_ptr() as usize % size_of::<u16>(),
        1,
        "an even-aligned tensor would not exercise the reading this needs"
    );

    let started = Instant::now();
    let resident = DenseWeight::wrap(&device, &matmul, &gate).expect("the gate wraps");
    eprintln!(
        "{ROUTER_GATE}: [{}, {HIDDEN}] bfloat16, {:.2} MB wrapped in {:.2?}",
        gate.out_dim(),
        gate.bytes().len() as f64 / 1e6,
        started.elapsed()
    );

    // Eight rows, because the reference's own normed state is eight and a
    // kernel that took its row index off the wrong axis would still fill the
    // buffer.
    let x = normed_state();
    let rows = x.len() / HIDDEN;
    let got = resident.multiply(&x).expect("the dispatch completes");
    assert_eq!(
        got.len(),
        rows * gate.out_dim(),
        "a logit per expert per row"
    );

    let widened = ckpt
        .tensor(ROUTER_GATE)
        .expect("the gate is there")
        .to_f32()
        .expect("the gate widens");
    let want = linear(&x, &widened, HIDDEN);
    let deviation = deviation(&got, &want);
    eprintln!("[{rows}, {HIDDEN}] through the gate: deviation {deviation:e}");
    assert!(deviation <= GATE_TOLERANCE, "deviation {deviation:e}");
    assert!(
        deviation > 0.0,
        "an exact match would mean the two are not summing independently"
    );
}

/// How many depths the block's cost is priced over, which is every one the
/// checkpoint ships heads for.
const DEPTHS: usize = 8;

/// How many times the sweep runs every depth, round-robin.
const PASSES: usize = 3;

/// A block's cost is measured this many times and averaged, after one run that
/// is thrown away — the first pass over a shape pays for the driver's first
/// look at the buffers it binds.
const REPEATS: usize = 12;

/// Everything a speculative round is made of, priced separately: what the
/// machinery costs a run that never speculates, what a verify block costs
/// against a warm cache, and what the two together buy over a real generation.
///
/// **The three are measured apart because they answer different questions.** A
/// block's cost is the model's and grows with the tokens in it; acceptance is
/// the workload's; and the speedup is what acceptance makes of the cost. The
/// study measured all three against mlx-vlm and found the block's cost the one
/// that decides it — 10.5 ms an extra token against a 31.8 ms step — so the
/// second table is the one to read.
///
/// Every depth wraps the layers with the slack *it* needs rather than the
/// deepest one's, because that is the configuration a run of that depth has:
/// the windows a rejected token is taken back out of are wider by the depth,
/// and what that costs is the first table.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn what_a_speculative_round_costs_and_what_it_buys() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    let config = fixture::config(&dir);
    let text = &config.text_config;
    let mtp = config.mtp_config.as_ref().expect("an mtp_config");
    let tokenizer = Tokenizer::open(&dir, &config).expect("the tokenizer opens");
    let ids: Vec<usize> = tokenizer
        .encode(STRUCTURED_PROMPT)
        .expect("the prompt encodes")
        .into_iter()
        .map(|id| id as usize)
        .collect();

    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);

    // The heads are 4.2 GiB of mapping, and a page joins the resident set when
    // something reads it — so the first generation that speculates pays for
    // faulting them in off disk and every later one does not. Warmed here, out
    // of the clock, because that cost belongs to a run's first token and not to
    // the depth that happened to be measured first.
    {
        let held = gpu.wrap(&ckpt, text, SWEPT);
        let heads = gpu.heads(&ckpt, text, mtp, gpu.tail(&held, text));
        Decoded::at(SWEPT, &held, &heads, text, &ids[..4]);
    }

    // What the machinery costs when nothing speculates: the same generation,
    // over layers whose windows keep enough to take four tokens back and over
    // layers that keep nothing.
    eprintln!("\nwith nothing speculating, over {DECODED} tokens");
    eprintln!("{:>7}  {:>10}", "slack", "ms/token");
    let mut idle = Vec::new();
    for slack in [0, 4] {
        let held = gpu.wrap(&ckpt, text, slack);
        let heads = gpu.heads(&ckpt, text, mtp, gpu.tail(&held, text));
        let run = Decoded::at(0, &held, &heads, text, &ids);
        eprintln!("{slack:>7}  {:>10.2}", run.step.as_secs_f64() * 1e3);
        idle.push(run);
    }
    let decode = idle[0].step;

    let held = gpu.wrap(&ckpt, text, 0);
    eprintln!(
        "\na verify block, against a warm cache of {} tokens",
        ids.len()
    );
    eprintln!(
        "{:>7}  {:>10}  {:>10}  {:>12}",
        "tokens", "block", "xdecode", "submissions"
    );
    let mut block = Vec::new();
    for tokens in 1..=DEPTHS + 1 {
        let (cost, submissions) = time_block(&device, &held, text, &ids, tokens);
        eprintln!(
            "{tokens:>7}  {:>10.2?}  {:>10.3}  {submissions:>12}",
            cost,
            cost.as_secs_f64() / decode.as_secs_f64()
        );
        block.push(cost);
    }
    let extra = (block[DEPTHS].as_secs_f64() - block[0].as_secs_f64()) / DEPTHS as f64;
    eprintln!("an extra token in the block: {:.2} ms", extra * 1e3);

    eprintln!("\nthe chain of heads, over one row");
    eprintln!("{:>7}  {:>10}  {:>10}", "heads", "chain", "xdecode");
    for depth in 1..=DEPTHS {
        let heads = gpu.heads(&ckpt, text, mtp, gpu.tail(&held, text));
        let cost = time_chain(&device, &held, &heads, text, &ids, depth).each;
        eprintln!(
            "{depth:>7}  {:>10.2?}  {:>10.3}",
            cost,
            cost.as_secs_f64() / decode.as_secs_f64()
        );
    }

    // **Round-robin over the depths rather than a run apiece**, for the reason
    // `.config/nextest.toml` records: a number taken once is a number about
    // whatever else the machine was doing. Every pass runs every depth, so a
    // drift that moves one moves them all, and what is reported is each depth's
    // best pass — the one that shared the machine with the least.
    let mut passes: Vec<Vec<Decoded>> = Vec::new();
    for _ in 0..PASSES {
        let mut pass = Vec::new();
        for depth in 0..=SWEPT {
            let held = gpu.wrap(&ckpt, text, depth);
            let heads = gpu.heads(&ckpt, text, mtp, gpu.tail(&held, text));
            pass.push(Decoded::at(depth, &held, &heads, text, &ids));
        }
        passes.push(pass);
    }
    let runs: Vec<&Decoded> = (0..=SWEPT)
        .map(|depth| {
            passes
                .iter()
                .map(|pass| &pass[depth])
                .min_by_key(|run| run.step)
                .expect("a pass")
        })
        .collect();
    let decode = runs[0].step;

    eprintln!("\nwhat the loop banked, over {DECODED} tokens");
    eprintln!(
        "{:>3}  {:>10}  {:>9}  {:>8}  {:>18}  accepted",
        "k", "ms/token", "tok/round", "speedup", "passes"
    );
    for (depth, run) in runs.iter().enumerate() {
        let spread: Vec<String> = passes
            .iter()
            .map(|pass| format!("{:.1}", pass[depth].step.as_secs_f64() * 1e3))
            .collect();
        eprintln!(
            "{:>3}  {:>10.2}  {:>9.3}  {:>8.3}  {:>18}  {}",
            run.depth,
            run.step.as_secs_f64() * 1e3,
            run.tokens_per_round(),
            decode.as_secs_f64() / run.step.as_secs_f64(),
            spread.join(" "),
            run.acceptance()
        );
    }

    // The property the whole thing rests on, and the reason the tokens are kept
    // rather than only timed: a latency optimisation that moved a token would
    // be a wrong engine, and every depth here ran the real heads against the
    // real rollback.
    for run in &runs[1..] {
        assert_eq!(
            run.tokens, runs[0].tokens,
            "speculating {} deep changed the tokens",
            run.depth
        );
    }
}

/// **Which kernels own the milliseconds a chain of heads costs**, and in how
/// many submissions — the question the sweep above prints one number for.
///
/// That number was the one figure of this engine's that had gone backwards, and
/// this is the table that said where it went. Four kernel milestones went into
/// the main step and none of them touched the chain: a decode step fell 30%
/// while the chain moved 37.92 ms to 37.63, so the depth that paid 1.31× paid
/// 1.00×. What the sweep could not say is *why*, and this project's record at
/// answering that by reasoning is written down — `lm_head` was predicted at 54%
/// of a step and measured 7.6%.
///
/// What it said was 88 dispatches in 48 submissions, 1.8 a submission against a
/// decode step's 71.8, and two thirds of a 43 ms chain waiting for them. It is
/// 160 dispatches in 16 now — a head and the `lm_head` behind it — for the
/// reason the same table gives a decode step's 15: what a backend is handed
/// whole it can encode into one command buffer. The chain is put through the
/// same two tables a decode step and a prefill go through, at the depth the
/// sweep prices it at.
///
/// Nothing asserts a share. What is asserted is that the table describes the
/// chain — every dispatch timed, and the rows summing inside the wall time they
/// were measured in.
#[test]
#[ignore = "a measurement: `just test-timing`, or `just test-full`"]
fn which_kernels_own_a_chain_of_heads() {
    let Some((dir, device)) = sampling_device() else {
        return;
    };

    let config = fixture::config(&dir);
    let text = &config.text_config;
    let mtp = config.mtp_config.as_ref().expect("an mtp_config");
    let tokenizer = Tokenizer::open(&dir, &config).expect("the tokenizer opens");
    let ids: Vec<usize> = tokenizer
        .encode(STRUCTURED_PROMPT)
        .expect("the prompt encodes")
        .into_iter()
        .map(|id| id as usize)
        .collect();

    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);
    let held = gpu.wrap(&ckpt, text, 0);
    let heads = gpu.heads(&ckpt, text, mtp, gpu.tail(&held, text));

    // The unsampled chain first, for the reason `which_kernels_own_a_decode_step`
    // takes one: what the rows below are worth is how close they sum to the
    // device time of a chain nobody was asking about. It is also what faults the
    // heads' 4.2 GiB of mapping in — `time_chain` runs a round before its own
    // clock starts, so that cost lands on neither of the two figures.
    device
        .time_each_dispatch(false)
        .expect("the device times a dispatch");
    device.record_round_trips(true);
    let unsampled = time_chain(&device, &held, &heads, text, &ids, DEPTHS);
    device
        .time_each_dispatch(true)
        .expect("the device times a dispatch");
    let sampled = time_chain(&device, &held, &heads, text, &ids, DEPTHS);
    device
        .time_each_dispatch(false)
        .expect("the device times a dispatch");
    device.record_round_trips(false);

    eprintln!("{}", step_table(&sampled.measured()));
    eprintln!(
        "  against a {:.2?} chain and {:.2?} of device time with nothing sampling, so the rows \
         carry {:+.1}% of asking",
        unsampled.each,
        unsampled.profile.gpu(),
        100.0
            * (sampled.profile.dispatched().as_secs_f64() / unsampled.profile.gpu().as_secs_f64()
                - 1.0)
    );
    eprintln!("{}", round_trip_table(&unsampled.measured()));

    let (timed, moved) = what_was_sampled(&sampled.profile);
    let (dispatches, submissions, ..) = sampled.counters;
    eprintln!(
        "  {:.2} GB over {dispatches} dispatches in {submissions} submissions, which is {:.1} \
         dispatches a submission against a decode step's {:.1}",
        moved as f64 / 1e9,
        dispatches as f64 / submissions as f64,
        1077.0 / 15.0,
    );

    // Every dispatch the chain encoded came back with a pair of timestamps. A
    // device that dropped one writes `MTLCounterErrorValue` and this side
    // charges it nothing, which would be a row quietly short rather than a
    // failure — so the count is what says the table describes the whole chain.
    assert_eq!(timed, dispatches, "a chain's dispatches were not all timed");
    // A chain reads `lm_head` once a head — 3.5 GB over the eight — and each
    // head's own weights once, which the checkpoint's shapes put at 4.5 GB where
    // they are the BF16 original's and a quarter of that where they have since
    // been packed. The bound holds for either, because which of the two a shard
    // is is not this case's question: what it guards is that the byte column
    // describes a chain rather than a fraction of one.
    assert!(
        (4e9..12e9).contains(&(moved as f64)),
        "a chain of {DEPTHS} heads moved {:.2} GB, where the shapes put its \
         weights between 4.6 and 8.0",
        moved as f64 / 1e9,
    );
    let accounted = sampled.profile.total();
    assert!(
        accounted <= sampled.each,
        "the rows sum to {accounted:.2?} inside a {:.2?} chain",
        sampled.each
    );
}

/// **The test this milestone lives or dies by, on the real checkpoint.**
///
/// The same prompt, generated alone and generated inside a batch, produces
/// identical tokens: at every position of the batch, beside neighbours whose
/// prompts are different lengths, and beside a neighbour that finishes early.
///
/// **Every existing check here is blind to what a batch breaks.** If sequence
/// A's span, convolution window or rows leak into B, both continuations are
/// still fluent text and the recorded eight-token continuation still passes —
/// the sequence that carries it is one of the batch's, and it is checked
/// against the oracle here as well, but agreeing with the oracle is not the
/// claim. The claim is that the *other* sequences are unmoved too, and only a
/// token-for-token comparison against each of them run alone says so.
///
/// One of the prompts is the oracle's own, so this is the recorded
/// continuation asserted from inside a batch as well as beside one.
#[test]
fn a_generation_in_a_batch_produces_what_it_produces_alone_on_the_device() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let config = fixture::config(&dir).text_config;
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);

    let recorded = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));
    let oracle = indices(&fixture::tensor(
        &fixture::open(ACTIVATIONS),
        "greedy_continuation",
    ));
    // Three prompts of three different lengths, the first of them the oracle's.
    let prompts: Vec<Vec<usize>> = vec![
        recorded.clone(),
        recorded[..recorded.len() / 2].to_vec(),
        recorded[recorded.len() - 3..].to_vec(),
    ];
    // The third stops two tokens early, which is a neighbour leaving the batch
    // while the others go on.
    let counts = [BATCHED, BATCHED, BATCHED - 2];

    let alone: Vec<Vec<usize>> = {
        let weights = gpu.wrap(&ckpt, &config, 0);
        let generator = weights.generator();
        prompts
            .iter()
            .zip(&counts)
            .map(|(prompt, count)| {
                generator.generate(&mut ModelCache::new(&config), prompt, *count, &weights)
            })
            .collect()
    };
    assert_eq!(
        alone[0],
        oracle[..BATCHED],
        "the recorded continuation, generated alone"
    );
    assert_ne!(alone[0], alone[1], "two generations to tell apart");

    for order in [vec![0, 1, 2], vec![2, 1, 0], vec![1, 0], vec![0, 2]] {
        let weights = gpu.wrap_batch(&ckpt, &config, order.len());
        let generator = weights.generator();
        let mut caches: Vec<ModelCache> = (0..order.len())
            .map(|slot| ModelCache::in_slot(&config, 0, slot))
            .collect();
        let ids: Vec<&[usize]> = order.iter().map(|at| prompts[*at].as_slice()).collect();
        let budgets: Vec<usize> = order.iter().map(|at| counts[*at]).collect();
        let batched = generator.generate_batch(&mut caches, &ids, &budgets, &weights);

        for (at, seq) in order.iter().enumerate() {
            assert_eq!(
                batched[at], alone[*seq],
                "sequence {seq} at position {at} of {order:?}"
            );
        }
    }
}

/// **What a batch costs in memory, stated and bounded.**
///
/// N sequences means N KV caches, and that is the whole of what a slot adds: a
/// span and four convolution windows in every layer, against weights that are
/// read once for every sequence in flight and held once. So what this asserts is
/// that the device's own accounting of that state is *linear* in the slots and
/// that a slot costs what the checkpoint's shapes say it does.
///
/// The arithmetic, from the config alone and at a span holding its least
/// capacity: a layer's keys and values are `2 * kv_heads * capacity * head_dim`
/// floats, and its four windows are `2 * (taps - 1) * channels` floats apiece
/// over the key's width twice and the hidden width twice.
///
/// **A windowed layer is charged the same as a global one**, which is a finding
/// rather than an interface — see `LayerAttention::span_bytes`. It is what makes
/// the bound below a bound on every layer rather than on seven of forty-two.
#[test]
fn a_slot_costs_a_span_and_four_windows_in_every_layer() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let config = fixture::config(&dir).text_config;
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);

    let recorded = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));
    let mut held = Vec::new();
    for slots in [1, 2, 4] {
        let weights = gpu.wrap_batch(&ckpt, &config, slots);
        let generator = weights.generator();
        let want = Tail {
            block: 1,
            chained: false,
            logits: false,
        };
        for slot in 0..slots {
            let mut cache = ModelCache::in_slot(&config, 0, slot);
            generator.tailed(&mut cache, &recorded, want, &weights);
        }
        let bytes = weights.held_bytes();
        eprintln!(
            "batch {slots}: {:.1} MiB of spans and windows, {:.1} a sequence",
            bytes as f64 / (1u64 << 20) as f64,
            bytes as f64 / (1u64 << 20) as f64 / slots as f64,
        );
        held.push(bytes);
    }

    // A slot is a slot: two of them hold twice what one does, and four twice
    // what two do. A layer that held its state anywhere but in the slots would
    // break this before it broke a token.
    assert_eq!(held[1], 2 * held[0], "two slots against one");
    assert_eq!(held[2], 4 * held[0], "four slots against one");

    // What one of them is, from the checkpoint's own shapes. The span is at the
    // least capacity a span has, which the recorded prompt does not exceed.
    let layers = config.num_hidden_layers as u64;
    let kv = (config.num_key_value_heads * config.head_dim) as u64;
    let taps = (config.sconv_kernel_size - 1) as u64;
    let floats = |values: u64| values * size_of::<f32>() as u64;
    let span = floats(2 * LEAST_SPAN * kv);
    let windows = floats(2 * taps * (2 * kv + 2 * config.hidden_size as u64));
    assert_eq!(
        held[0],
        layers * (span + windows),
        "a slot against the shapes the checkpoint gives it"
    );
}

/// Key slots a span has room for before a sequence makes it grow, which is
/// `LEAST_KEYS` in `inkling_metal::attention` and is not public — spelled here
/// so that the arithmetic above is checkable and a constant that moved would
/// fail this rather than quietly widen a bound.
const LEAST_SPAN: u64 = 64;

/// **The derived and the encoded barrier counts still agree at batch > 1**, which
/// is what makes contamination through a missing barrier a test failure rather
/// than a paragraph.
///
/// D4's claim is that the barriers a step encodes are the ones its dependency
/// graph needs and no others, and the graph is derived from each kernel's Metal
/// source at compile time. A batch adds edges that graph never saw: a
/// convolution reads the run of the projections' rows that is its own and writes
/// into a span the step after it binds, and there are N of each — so a division
/// that was right for one sequence is not thereby right for sixteen.
///
/// **A missing barrier is a race that is correct most of the time**, which is
/// the same reason the single-sequence case exists: what it would otherwise cost
/// is a wrong token months later, in the one place this milestone cannot afford
/// one.
///
/// Two steps at each width, because the sequence is the same commands every step
/// and a division that differed between two of them would be an ordering that
/// depends on something other than the shape.
#[test]
fn the_barriers_a_batched_step_encodes_are_the_ones_its_dependencies_need() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let config = fixture::config(&dir).text_config;
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);
    let recorded = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));

    let mut dispatched = Vec::new();
    for slots in [1, 2, 4] {
        let weights = gpu.wrap_batch(&ckpt, &config, slots);
        let generator = weights.generator();
        let mut caches: Vec<ModelCache> = (0..slots)
            .map(|slot| ModelCache::in_slot(&config, 0, slot))
            .collect();
        let want = Tail {
            block: 1,
            chained: false,
            logits: false,
        };
        let mut pending: Vec<usize> = caches
            .iter_mut()
            .enumerate()
            .map(|(slot, cache)| {
                let mut prompt = recorded.clone();
                let by = slot % prompt.len();
                prompt.rotate_left(by);
                generator.tailed(cache, &prompt, want, &weights).picks[0]
            })
            .collect();

        trace::record(true);
        let mut divisions = Vec::new();
        let mut by_symbol: BTreeMap<String, usize> = BTreeMap::new();
        for _ in 0..2 {
            let before = device.barriers();
            let feeding: Vec<[usize; 1]> = pending.iter().map(|id| [*id]).collect();
            let mut batch: Vec<Batched<'_>> = caches
                .iter_mut()
                .zip(&feeding)
                .map(|(cache, ids)| Batched { cache, ids })
                .collect();
            pending = generator
                .step_batch(&mut batch, &weights)
                .iter()
                .map(Picked::last)
                .collect();
            let encoded = device.barriers() - before;
            // The boundaries first: taking the trace clears both.
            let boundaries = trace::submissions();
            let traced = trace::take();
            let mut counted: BTreeMap<String, usize> = BTreeMap::new();
            for dispatch in &traced {
                *counted.entry(dispatch.symbol.clone()).or_default() += 1;
            }
            by_symbol = counted;
            let groups = Groups::over(&traced, &boundaries);
            assert_eq!(
                encoded,
                groups.barriers() as u64,
                "a batch of {slots} encoded {encoded} barriers where its division needs {}",
                groups.barriers()
            );
            eprintln!(
                "batch {slots}: {} dispatches in {} groups, {} barriers",
                groups.dispatches(),
                groups.groups(),
                groups.barriers()
            );
            divisions.push(groups);
        }
        trace::record(false);
        assert_eq!(
            divisions[0], divisions[1],
            "a batch of {slots} divides two steps differently"
        );
        // **The step and the three convolutions are one dispatch a layer at
        // every width**, which is the milestone's own claim and the one the
        // count above states only in aggregate: a per-row slot index is what
        // lets N sequences over N spans be one dispatch, and a kernel that went
        // back to one a slot would still satisfy a total that another had
        // meanwhile given up.
        eprintln!("slots {slots} by symbol: {by_symbol:?}");
        for entry in PER_CALL {
            let (name, each) = entry;
            assert_eq!(
                by_symbol.get(*name).copied(),
                Some(each * config.num_hidden_layers),
                "a batch of {slots} against {each} {name} a layer whatever the batch"
            );
        }
        // And the one that is still a sequence's own, which is what a slot now
        // costs and where the remaining per-slot dispatches are.
        for entry in PER_SLOT {
            let (name, each) = entry;
            assert_eq!(
                by_symbol.get(*name).copied(),
                Some(each * config.num_hidden_layers * slots),
                "a batch of {slots} against {each} {name} a layer a slot"
            );
        }
        dispatched.push((slots, divisions[0].dispatches()));
    }

    // **What a slot costs in dispatches, which is the other half of what it
    // costs.** The five weights every sequence reads are one dispatch each
    // whatever the batch is, and so is the attention step: it reads no weight at
    // all, and what used to make it a dispatch a slot was a span per allocation
    // rather than anything about the work. **The three convolutions are the
    // same story one layer down** — what made them a dispatch a slot was two
    // window allocations a slot, and their windows are now runs of one. So a
    // slot adds the head norms alone, one a layer, and a dispatch that quietly
    // became per sequence would show up here before it showed up in a timing.
    //
    // **The upper bound is the rows and not the slots**, and it is why this is a
    // range rather than an equality: a routed layer's grouped dispatch decides
    // its own shape from the rows it is given, and at four rows it takes one
    // dispatch more than at one — 40 of them, which is one a MoE layer, at every
    // width from four up. That is the batch feeding the block more rows, which
    // is what a batch is for.
    let routed = (0..config.num_hidden_layers)
        .filter(|layer| !config.layer_is_dense(*layer))
        .count();
    let (_, alone) = dispatched[0];
    for (slots, dispatches) in &dispatched[1..] {
        let added = dispatches - alone;
        let slotted = config.num_hidden_layers * (slots - 1);
        assert!(
            (slotted..=slotted + routed).contains(&added),
            "a batch of {slots} adds {added} dispatches where its slots need {slotted} \
             and its rows may add one to each of {routed} routed layers"
        );
        eprintln!("slots {slots}: {dispatches} dispatches, {added} over one");
    }
}

/// The compiled entry a layer's attention step runs, which the profile charges
/// to two different rows — see [`trace::Encoded::symbol`].
const ATTENTION_STEP: &str = "fused_attention";

/// What a layer dispatches once whatever the batch is: the attention step, the
/// key and value convolutions as one paired dispatch, and the two on the layer's
/// residual paths. Each carries a seat per sequence — a run of the rows, a run
/// of the spans or windows, and where the answer goes.
const PER_CALL: &[(&str, usize)] = &[
    (ATTENTION_STEP, 1),
    ("short_conv_pair", 1),
    ("short_conv", 2),
];

/// What is left per sequence once those are shared, and how many dispatches of
/// each a layer takes: the query and key head norms, as one paired dispatch.
const PER_SLOT: &[(&str, usize)] = &[("rms_norm_pair", 1)];

/// How many tokens each sequence of the batched case generates.
///
/// Short, because the case runs every ordering of three sequences and each
/// token is a decode step of the whole stack — and long enough that a step
/// reads what more than one step before it left, a convolution's window being
/// three inputs deep.
const BATCHED: usize = 4;

/// The kernels a speculative run compiles, held so that the weights wrapped
/// against them can be built once per depth.
struct Kernels<'d> {
    device: &'d Device,
    layers: LayerKernels,
    dense: DenseMatmul,
    swiglu: SwiGlu,
    router: Router,
    grouping: ExpertGrouping,
    weights: RouterWeights,
    combine: MoeCombine,
}

impl<'d> Kernels<'d> {
    /// Under [`numerics`], the way [`OnTheDevice::running`] compiles its own —
    /// so that the batched table is reachable on both sides of the flag, which
    /// is what a table that is anyone's arbiter has to be. Unset is the
    /// reference, which is what every gated case here is written against.
    fn compile(device: &'d Device) -> Self {
        Self {
            device,
            layers: LayerKernels::compiling(device, numerics()).expect("the layer kernels compile"),
            dense: DenseMatmul::new(device).expect("the dense matmul compiles"),
            swiglu: SwiGlu::new(device).expect("the swiglu compiles"),
            router: Router::new(device).expect("the router compiles"),
            grouping: ExpertGrouping::new(device).expect("the grouping compiles"),
            weights: RouterWeights::new(device).expect("the weighting compiles"),
            combine: MoeCombine::new(device).expect("the combine compiles"),
        }
    }

    /// The whole model on the device, over layers that can give `slack`
    /// timesteps back.
    fn wrap<'a>(
        &'a self,
        ckpt: &'a Checkpoint,
        config: &'a inkling_core::TextConfig,
        slack: usize,
    ) -> CheckpointWeights<'a>
    where
        'd: 'a,
    {
        self.wrapping(ckpt, config, slack, true, 1)
    }

    /// The same, holding `slots` sequences at once — see
    /// [`StackShape::slots`](inkling_metal::StackShape).
    fn wrap_batch<'a>(
        &'a self,
        ckpt: &'a Checkpoint,
        config: &'a inkling_core::TextConfig,
        slots: usize,
    ) -> CheckpointWeights<'a>
    where
        'd: 'a,
    {
        self.wrapping(ckpt, config, 0, true, slots)
    }

    /// The same, with the final norm and the muP divide left on the CPU and
    /// `lm_head` in a submission of its own — which is where all three were, and
    /// what the tail is held against.
    fn without_a_tail<'a>(
        &'a self,
        ckpt: &'a Checkpoint,
        config: &'a inkling_core::TextConfig,
    ) -> CheckpointWeights<'a>
    where
        'd: 'a,
    {
        self.wrapping(ckpt, config, 0, false, 1)
    }

    fn wrapping<'a>(
        &'a self,
        ckpt: &'a Checkpoint,
        config: &'a inkling_core::TextConfig,
        slack: usize,
        tail: bool,
        slots: usize,
    ) -> CheckpointWeights<'a>
    where
        'd: 'a,
    {
        let mapped = CheckpointWeights::open(ckpt, config).expect("the checkpoint's weights map");
        let head = PackedProjection::wrap_packed(
            self.device,
            self.layers.matmul(),
            &mapped.head_packed(),
            mapped.head().vocab(),
        )
        .expect("the head wraps");
        let banks = mapped.expert_banks();
        let packed = mapped.layer_projections();
        let layers = ModelLayers::wrap(
            self.device,
            &self.layers,
            ExpertKernels {
                matmul: self.layers.matmul(),
                dense: &self.dense,
                swiglu: &self.swiglu,
                router: &self.router,
                grouping: &self.grouping,
                weights: &self.weights,
                combine: &self.combine,
            },
            &packed,
            &banks,
            tail.then(|| wrap_tail(self.device, &self.layers, &mapped, config))
                .flatten(),
            StackShape {
                layers: config.num_hidden_layers,
                dim: config.hidden_size,
                slack,
                slots,
            },
        )
        .expect("the layers wrap");
        mapped
            .with_head(Box::new(head))
            .with_backend(Box::new(layers))
    }

    /// The model's tail, for a caller wrapping the heads that run in front of
    /// it — see [`wrap_tail`].
    fn tail<'a>(
        &'a self,
        weights: &CheckpointWeights<'a>,
        config: &inkling_core::TextConfig,
    ) -> Option<ModelTail<'a>>
    where
        'd: 'a,
    {
        wrap_tail(self.device, &self.layers, weights, config)
    }

    /// The eight heads on the device.
    fn heads<'a>(
        &'a self,
        ckpt: &'a Checkpoint,
        config: &inkling_core::TextConfig,
        mtp: &inkling_core::config::MtpConfig,
        tail: Option<ModelTail<'a>>,
    ) -> CheckpointHeads<'a> {
        let heads = CheckpointHeads::open(ckpt, config, mtp).expect("the heads open");
        let held = heads.head_projections();
        let wrapped = ModelHeads::wrap(
            self.device,
            &self.layers,
            &self.dense,
            &self.swiglu,
            &held,
            tail,
            inkling_core::mtp::FRONTIER,
            1,
        )
        .expect("the heads wrap");
        heads.with_backend(Box::new(wrapped))
    }
}

/// The model's tail on the device: its final norm, the muP divide and
/// `lm_head`, which is what closes a decode step's one command buffer and a
/// head's.
///
/// Built here rather than taken from a shape, because what it is made of is
/// four things out of the checkpoint and the point of assembling them in one
/// place is that they cannot come from different ones.
fn wrap_tail<'a>(
    device: &'a Device,
    kernels: &'a LayerKernels,
    weights: &CheckpointWeights<'a>,
    config: &inkling_core::TextConfig,
) -> Option<ModelTail<'a>> {
    ModelTail::wrap(
        device,
        kernels.norm(),
        kernels.matmul(),
        kernels.argmax(),
        &TailWeights {
            norm: weights.final_norm().to_vec(),
            eps: config.rms_norm_eps,
            mup: weights.head().mup(),
            head: weights.head_packed(),
            vocab: weights.head().vocab(),
        },
    )
    .expect("the tail wraps")
}

/// One generation at one depth: what it produced, what it cost, and what its
/// heads guessed right.
struct Decoded {
    depth: usize,
    tokens: Vec<usize>,
    step: Duration,
    rounds: usize,
    rates: Vec<f64>,
}

impl Decoded {
    fn at(
        depth: usize,
        weights: &CheckpointWeights<'_>,
        heads: &CheckpointHeads<'_>,
        config: &inkling_core::TextConfig,
        ids: &[usize],
    ) -> Self {
        let generator = weights.generator();
        let cache = &mut ModelCache::speculating(config, depth);
        let ending = Ending {
            budget: DECODED,
            eos: None,
        };
        let mut proposer = MtpProposer::new(heads, generator, weights, depth);
        let mut tokens = Vec::new();

        // The prefill is a step of another price and is not in the mean — the
        // clock starts once the prompt is in the cache, which is where a
        // decode-step figure is comparable to any other in this file.
        let mut started = None;
        generator.speculate(cache, ids, ending, weights, &mut proposer, |id| {
            started.get_or_insert_with(Instant::now);
            tokens.push(id);
            ControlFlow::Continue(())
        });
        let elapsed = started.expect("a token").elapsed();
        Self {
            depth,
            rounds: proposer.rounds(),
            step: elapsed / (tokens.len() - 1).max(1) as u32,
            rates: proposer.rates(),
            tokens,
        }
    }

    /// Tokens a round banked, which is what acceptance buys before the cost of
    /// having guessed is taken off it.
    ///
    /// Counted rather than inferred: a round is a forward pass, and the
    /// proposer is asked for guesses once in each of them.
    fn tokens_per_round(&self) -> f64 {
        self.tokens.len() as f64 / self.rounds as f64
    }

    /// What [`MtpProposer::rates`] measured, as a row of a table.
    fn acceptance(&self) -> String {
        self.rates
            .iter()
            .map(|rate| format!("{:.0}%", 100.0 * rate))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// **A head's own argmax against the same argmax taken here**, which is the
/// claim [`Guessed::guess`] rests on and the one the tail's own case cannot
/// make: that case compares two tails behind the *stack*, and a head's guess
/// comes out of the head's command buffer instead.
///
/// The same chain twice over the same round, differing in nothing but whether
/// the heads were wrapped with the model's tail behind them. With it, each
/// head's guess is a dispatch over the row `lm_head` just wrote; without it,
/// the head answers with rows and `Generator::id_from_hidden` runs the same
/// final norm, the same divide, the same projection and `top_k` back on this
/// side. Every guess of every head has to be the same id — exactly, since what
/// separates the two is an argmax and not an accumulation.
///
/// Eight heads and not one, because a chain guesses from its own guesses: a
/// head whose guess moved would feed the next head a different token, so a
/// single disagreement anywhere shows up here as every head after it
/// disagreeing too.
#[test]
fn a_heads_own_argmax_takes_the_token_the_host_takes() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };

    let config = fixture::config(&dir);
    let text = &config.text_config;
    let mtp = config.mtp_config.as_ref().expect("an mtp_config");
    let tokenizer = Tokenizer::open(&dir, &config).expect("the tokenizer opens");
    let ids: Vec<usize> = tokenizer
        .encode(STRUCTURED_PROMPT)
        .expect("the prompt encodes")
        .into_iter()
        .map(|id| id as usize)
        .collect();

    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let gpu = Kernels::compile(&device);
    let held = gpu.wrap(&ckpt, text, 0);
    let generator = held.generator();

    // The row a round would chain from, which both chains are given.
    let cache = &mut ModelCache::speculating(text, 0);
    let hidden = generator
        .tailed(
            cache,
            &ids,
            Tail {
                block: 1,
                chained: true,
                logits: false,
            },
            &held,
        )
        .normed;
    let width = text.hidden_size;
    let row = &hidden[hidden.len() - width..];

    // The tail behind the heads, and no tail behind them — a loop rather than
    // two spellings of a round, so that what differs between the two answers is
    // the argument and not the code that produced them.
    let mut guessed = Vec::new();
    for tail in [gpu.tail(&held, text), None] {
        let heads = gpu.heads(&ckpt, text, mtp, tail);
        let mut proposer = MtpProposer::new(&heads, generator, &held, DEPTHS);
        guessed.push(
            proposer
                .propose(Round {
                    hidden: row,
                    next: &ids[..1],
                    depth: DEPTHS,
                })
                .to_vec(),
        );
    }

    let (there, here) = (&guessed[0], &guessed[1]);
    eprintln!("a chain of {DEPTHS} heads guessed {there:?}");
    assert_eq!(there.len(), DEPTHS, "a guess a head");
    assert_eq!(there, here, "a head's own argmax took a different token");
}

/// What `depth` heads cost to run over one row, which is what a round pays to
/// have guessed, and where that went.
///
/// A row rather than a block, because it is the shape a round has when nothing
/// was accepted — and because what grows with the rows is the head's own
/// attention rather than its weights, which are read once whatever the row
/// count.
///
/// **The accounts are cleared after the warm round and read after the timed
/// ones**, so what comes back describes the same rounds the duration does: the
/// prefill in front of them is a submission a layer and would drown every row.
fn time_chain(
    device: &Device,
    weights: &CheckpointWeights<'_>,
    heads: &CheckpointHeads<'_>,
    config: &inkling_core::TextConfig,
    ids: &[usize],
    depth: usize,
) -> Chain {
    let generator = weights.generator();
    let cache = &mut ModelCache::speculating(config, 0);
    let hidden = generator
        .tailed(
            cache,
            ids,
            Tail {
                block: 1,
                chained: true,
                logits: false,
            },
            weights,
        )
        .normed;
    let width = config.hidden_size;
    let row = &hidden[hidden.len() - width..];

    let mut proposer = MtpProposer::new(heads, generator, weights, depth);
    let round = |proposer: &mut MtpProposer<'_, CheckpointWeights<'_>>| {
        proposer.propose(Round {
            hidden: row,
            next: &ids[..1],
            depth,
        });
    };
    round(&mut proposer);

    profile::take();
    device.round_trips();
    let before = counters(device);
    let started = Instant::now();
    for _ in 0..REPEATS {
        round(&mut proposer);
    }
    let elapsed = started.elapsed();
    let (dispatches, submissions, allocations, bytes) = since(before, counters(device));
    let each = |total: u64| total / REPEATS as u64;
    Chain {
        depth,
        each: elapsed / REPEATS as u32,
        counters: (
            each(dispatches),
            each(submissions),
            each(allocations),
            each(bytes),
        ),
        profile: profile::take().per_step(REPEATS as u32),
        round_trips: device.round_trips(),
    }
}

/// One chain of heads, priced the way a decode step is.
struct Chain {
    depth: usize,
    each: Duration,
    counters: (u64, u64, u64, u64),
    profile: Profile,
    round_trips: Vec<RoundTrip>,
}

impl Chain {
    /// The chain as the tables read a regime, which is where it is put beside a
    /// decode step and a prefill.
    ///
    /// The round trips are what the timed rounds recorded between them, so they
    /// divide by the rounds exactly as the profile already has.
    fn measured(&self) -> Measured<'_> {
        Measured {
            regime: format!("chain of {} heads over one row", self.depth),
            step: self.each,
            steps: REPEATS as u32,
            counters: self.counters,
            profile: &self.profile,
            round_trips: &self.round_trips,
        }
    }
}

/// What a forward pass over `tokens` costs against a cache holding `ids`, which
/// is what a round pays to verify a block of that many.
///
/// Real decoded tokens rather than filler, for the reason the study gives: a
/// block of one token repeated reaches the same six experts a single-token step
/// does, which prices the block at a decode step and flatters speculation
/// several-fold.
fn time_block(
    device: &Device,
    weights: &CheckpointWeights<'_>,
    config: &inkling_core::TextConfig,
    ids: &[usize],
    tokens: usize,
) -> (Duration, u64) {
    let generator = weights.generator();
    let mut decoded = Vec::new();
    let cache = &mut ModelCache::speculating(config, 0);
    generator.stream(
        cache,
        ids,
        Ending {
            budget: tokens + 1,
            eos: None,
        },
        weights,
        |id| {
            decoded.push(id);
            ControlFlow::Continue(())
        },
    );

    // A fresh cache each time, so every repeat is the same block against the
    // same warm prompt rather than one sequence growing under the clock.
    let block = &decoded[..tokens];
    // The counters bracket the block alone, which is why the warm cache is
    // filled before either of them starts: a prefill is a submission a layer,
    // and counting it here would drown the number this column exists for.
    let run = || {
        let cache = &mut ModelCache::speculating(config, 0);
        generator.logits(cache, ids, weights);
        let submitted = device.submissions();
        let started = Instant::now();
        generator.logits(cache, block, weights);
        (started.elapsed(), device.submissions() - submitted)
    };
    run();
    let (elapsed, submissions): (Vec<Duration>, Vec<u64>) = (0..REPEATS).map(|_| run()).unzip();
    (
        elapsed.iter().sum::<Duration>() / REPEATS as u32,
        submissions[0],
    )
}
