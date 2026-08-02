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
use inkling_core::generate::{Proposer, Round};
use inkling_core::moe::{Gate, GateWeights, MoeConfig, SparseMoe};
use inkling_core::mtp::{CheckpointHeads, MtpProposer};
use inkling_core::ops::linear;
use inkling_core::profile::{self, Op, Profile};
use inkling_core::quant::{BITS, dequantize_blocks_into};
use inkling_core::{
    AttentionCache, AttentionStep, BandedMask, Bf16, Checkpoint, CheckpointWeights, Dtype, Ending,
    LayerStep, ModelCache, Packed as CorePacked, Projections, Sdpa, ShortConv, TensorView,
    Tokenizer, split_heads,
};
use inkling_metal::{
    DISPATCHES_A_SUBMISSION, DenseMatmul, DenseWeight, Device, ExpertKernels, LayerKernels,
    LayerProjections, LayerRouter, MetalError, ModelHeads, ModelLayers, MoeCombine, PackedBank,
    PackedMatmul, PackedProjection, RoundTrip, Router, RouterWeights, StackShape, SwiGlu,
};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

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
/// **A whole layer is one submission**, and twenty-six dispatches on a layer
/// that routes. Eleven are its attention: the input layernorm, the four
/// projections that consume what it produced, the two short convolutions behind
/// `k` and `v`, the two head norms over `q` and the convolved `k`, the attention
/// step and `o_proj`. Three more are the two residual paths around the MLP — the
/// layer's two short convolutions, each of which adds the value its block began
/// with as it writes, and the second norm between them. The last twelve are the
/// MLP: the router's gate, the top-k over what it produced, each bank's gate,
/// up, activation and down, the softmax over the eight logits that selection
/// named, and both banks' rows weighted by it. A dense layer is eighteen, its
/// feed-forward network four where a MoE layer's two banks and the router around
/// them are twelve. The head is one of each.
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
/// A prefill wide enough that one layer reaches the bytes a run may hold is
/// still one a layer, and deliberately — see `ModelLayers::carries`, where what
/// a merged run holds is traded against what it saves. This counts a decode
/// step, whose forty-two layers are far under that budget.
fn per_step(layers: u64, dense: u64) -> (u64, u64) {
    let width = |layer: u64| if layer < dense { 14 + 4 } else { 14 + 12 };
    let (mut dispatches, mut encoded, mut submissions) = (0, 0, 1);
    for layer in 0..layers {
        dispatches += width(layer);
        encoded += width(layer);
        if layer + 1 < layers && encoded >= DISPATCHES_A_SUBMISSION as u64 {
            submissions += 1;
            encoded = 0;
        }
    }
    (dispatches + 1, submissions + 1)
}

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
    /// What each step took, the prompt's prefill first.
    steps: Vec<Duration>,
    /// The running `(dispatches, submissions, allocations, allocated bytes)`
    /// each step was reached at.
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
    /// What the **decode** steps spent, by operation. The prefill's accounts
    /// are cleared before the first of them: it dispatches the same kernels
    /// over more rows, and what a profile is read for is the steady state.
    profile: Profile,
    /// Every command buffer the **decode** steps waited for, cleared after the
    /// prefill for the reason the profile is — and one row where the profile
    /// has a sum, because a step's two submissions are 1076 dispatches and one
    /// and the wait means something different in each.
    round_trips: Vec<RoundTrip>,
}

impl OnTheDevice {
    /// The run every case here starts from, with the device's own clock left
    /// where it is.
    fn generate(dir: &Path, device: &Device) -> Self {
        Self::generating(dir, device, false)
    }

    /// The same run, with each dispatch timed on the device if `sampling`.
    ///
    /// **Sampling is a parameter rather than a setting**, because the only way
    /// to say what it costs is to run the same thing both ways — see
    /// `what_timing_each_dispatch_costs`.
    fn generating(dir: &Path, device: &Device, sampling: bool) -> Self {
        device
            .time_each_dispatch(sampling)
            .expect("the device times a dispatch");
        let kernels = LayerKernels::compile(device).expect("the layer kernels compile");
        let matmul = kernels.matmul();
        let dense = DenseMatmul::new(device).expect("the dense matmul compiles");
        let swiglu = SwiGlu::new(device).expect("the swiglu compiles");
        let router = Router::new(device).expect("the router compiles");
        let weighing = RouterWeights::new(device).expect("the weighting compiles");
        let combine = MoeCombine::new(device).expect("the combine compiles");
        let config = fixture::config(dir).text_config;
        let ckpt = Checkpoint::open(dir).expect("checkpoint opens");
        let ids = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));

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
                weights: &weighing,
                combine: &combine,
            },
            &packed,
            &banks,
            StackShape {
                layers: config.num_hidden_layers,
                dim: config.hidden_size,
                slack: 0,
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
            steps: Vec::new(),
            submitted: Vec::new(),
            peak: fixture::resident_bytes(),
            got: Vec::new(),
            profile: Profile::default(),
            round_trips: Vec::new(),
        };
        device.record_round_trips(true);

        // Once, before the loop rather than inside it — though "once" is now 6
        // ms for 137 GB, so what that used to be defending against is gone.
        let weights = weights
            .with_head(Box::new(head))
            .with_backend(Box::new(layers));
        let generator = weights.generator();

        let mut step = Instant::now();
        generator.stream(
            &mut ModelCache::new(&config),
            &ids,
            Ending {
                budget: GENERATED,
                eos: None,
            },
            &weights,
            |id| {
                run.steps.push(step.elapsed());
                run.submitted.push((
                    device.dispatches(),
                    device.submissions(),
                    device.allocations(),
                    device.allocated_bytes(),
                ));
                run.peak = run.peak.max(fixture::resident_bytes());
                run.got.push(id);
                if run.steps.len() == 1 {
                    profile::take();
                    device.round_trips();
                }
                step = Instant::now();
                ControlFlow::Continue(())
            },
        );
        run.profile = profile::take().per_step(run.decode_steps());
        run.round_trips = device.round_trips();
        device.record_round_trips(false);
        run
    }

    fn decode_steps(&self) -> u32 {
        (self.steps.len() - 1) as u32
    }

    /// The prompt's prefill is the first step and every later one is a single
    /// decode; a mean over the two describes neither.
    fn each_decode_step(&self) -> Duration {
        let (_, decode) = self.steps.split_first().expect("a step per token");
        decode.iter().sum::<Duration>() / self.decode_steps()
    }

    /// The `(dispatches, submissions, allocations, allocated bytes)` of the last
    /// decode step, which is the difference between the two running totals
    /// either side of it.
    fn per_decode_step(&self) -> (u64, u64, u64, u64) {
        let [.., before, after] = self.submitted.as_slice() else {
            panic!("a step per token, and more than one of them")
        };
        (
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2,
            after.3 - before.3,
        )
    }
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

/// What this machine's memory will hand a kernel per second, as the ceiling the
/// achieved column is a fraction of.
///
/// **Written down rather than measured, and it is the one number in the table
/// that is.** Metal has no API that answers it: an M3 Ultra's 819 GB/s is the
/// part's stated figure, and no kernel reaches it — M2's isolated matmul,
/// which is the closest thing here to a pure streaming read, measured 284 to
/// 424. So a row at 40% of this is not 40% of what is achievable; what the
/// column is for is telling a kernel that is near the machine from one that is
/// a decade off it.
const MEMORY_BANDWIDTH: f64 = 819e9;

/// A decode step's rows, and the kernels underneath them where the device was
/// asked which of its dispatches owns which of the milliseconds.
///
/// One table rather than two, because the second half only means anything
/// against the first: a kernel's device time is a share of what the device
/// executed for, which is a share of the wait, which is a share of the step.
fn decode_step_table(run: &OnTheDevice) -> String {
    let step = run.each_decode_step();
    let (dispatches, submissions, allocations, bytes) = run.per_decode_step();
    let accounted = run.profile.total();

    let share = |part: Duration| 100.0 * part.as_secs_f64() / step.as_secs_f64();
    let mut table = vec![format!(
        "a {step:.2?} decode step, {dispatches} dispatches in {submissions} submissions over \
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
        "  {:<18}{:>7}{:>12}{:>8}{:>11}{:>11}{:>8}",
        "kernel", "calls", "device", "share", "moved", "achieved", "of peak"
    ));
    for (kernel, dispatches) in &kernels {
        let achieved = dispatches.bytes_per_second();
        table.push(format!(
            "  {kernel:<18}{:>7}{:>12}{:>7.1}%{:>11}{:>11}{:>7.0}%",
            dispatches.calls,
            format!("{:.2?}", dispatches.elapsed),
            executing(dispatches.elapsed),
            format!("{:.2} MB", dispatches.bytes as f64 / 1e6),
            format!("{:.0} GB/s", achieved / 1e9),
            100.0 * achieved / MEMORY_BANDWIDTH,
        ));
    }
    table.push(format!(
        "  the passes account for {sampled:.2?} of the {:.2?} the command buffers reported, and \
         the {:.2?} between them is what a pass boundary a dispatch costs",
        run.profile.gpu(),
        run.profile.gpu().saturating_sub(sampled),
    ));
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
    eprintln!("{}", decode_step_table(&run));

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
    let mut shapes: BTreeMap<usize, Vec<RoundTrip>> = BTreeMap::new();
    for trip in &run.round_trips {
        shapes.entry(trip.dispatches).or_default().push(*trip);
    }

    eprintln!(
        "a {:.2?} decode step, of which {:.2?} is the {} submissions it waits for\n  \
         {:<11}{:>8}{:>11}{:>11}{:>10}{:>11}{:>14}",
        run.each_decode_step(),
        run.profile.elapsed(Op::Submit),
        run.round_trips.len() as u32 / steps,
        "dispatches",
        "a step",
        "waited",
        "scheduled",
        "queued",
        "executed",
        "unattributed",
    );
    let mean = |trips: &[RoundTrip], of: fn(&RoundTrip) -> Duration| {
        trips.iter().map(of).sum::<Duration>() / steps
    };
    for (dispatches, trips) in &shapes {
        eprintln!(
            "  {dispatches:<11}{:>8}{:>11}{:>11}{:>10}{:>11}{:>14}",
            trips.len() as u32 / steps,
            format!("{:.2?}", mean(trips, |trip| trip.waited)),
            format!("{:.2?}", mean(trips, |trip| trip.scheduled)),
            format!("{:.2?}", mean(trips, |trip| trip.queued)),
            format!("{:.2?}", mean(trips, |trip| trip.executed)),
            format!("{:.2?}", mean(trips, |trip| trip.unattributed())),
        );
    }

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
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    if !device.times_a_pass() {
        eprintln!("skipping: this device does not sample at a stage boundary");
        return;
    }

    // The unsampled step first, because what the rows below are worth is how
    // close they sum to the device time of a step nobody was asking about.
    let unsampled = OnTheDevice::generate(&dir, &device).profile.gpu();
    let run = OnTheDevice::generating(&dir, &device, true);
    eprintln!("{}", decode_step_table(&run));
    eprintln!(
        "  against {unsampled:.2?} of device time with nothing sampling, so the rows carry \
         {:+.1}% of asking",
        100.0 * (run.profile.dispatched().as_secs_f64() / unsampled.as_secs_f64() - 1.0)
    );

    let kernels = run.profile.kernels();
    assert!(!kernels.is_empty(), "nothing was sampled");
    // **The bytes are declared and not derived**, so a formula that dropped a
    // factor would move the whole bandwidth column and nothing would say so.
    // What checks it is a figure this repo has from somewhere else entirely: a
    // token reads six of each MoE layer's 256 experts and both shared ones plus
    // every layer's own projections, which the checkpoint's shapes put at 5.9 GB
    // of packed bytes.
    let moved: u64 = kernels.iter().map(|(_, each)| each.bytes).sum();
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
        kernels.iter().map(|(_, each)| each.calls).sum::<u64>(),
        dispatches,
        "a decode step's dispatches were not all timed"
    );
    assert!(
        run.profile.dispatched() <= run.profile.gpu(),
        "the passes claim {:.2?} of a command buffer the device clocked at {:.2?}",
        run.profile.dispatched(),
        run.profile.gpu()
    );
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

/// The prompt the speculative measurement runs against.
///
/// Templated, because the model answers a turn and continues raw text — and
/// what a head is guessing about is what the model is doing. The acceptance
/// study measured six regimes and found the spread between them larger than the
/// spread between depths; this is one of them, the structured one, so what it
/// says about acceptance is "on text like this" and nothing wider.
const SPECULATIVE_PROMPT: &str = "<|message_user|><|content_text|>Count from 1 to 30. Put each on \
     its own line in exactly the form 'Line N: N squared is M'. No \
     commentary.<|end_message|><|message_model|>";

/// How many tokens each depth decodes, which is what the acceptance and the
/// throughput are both measured over.
const SPECULATED_TOKENS: usize = 64;

/// How many depths the block's cost is priced over, which is every one the
/// checkpoint ships heads for.
const DEPTHS: usize = 8;

/// How deep the sweep of real generations goes.
///
/// Four, where the block is priced to eight: the study's pooled optimum was 2
/// and its deepest paying depth 6, and every depth here is a whole generation
/// rather than a repeat of one block.
const SWEPT: usize = 4;

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
        .encode(SPECULATIVE_PROMPT)
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
        let heads = gpu.heads(&ckpt, text, mtp);
        Decoded::at(SWEPT, &held, &heads, text, &ids[..4]);
    }

    // What the machinery costs when nothing speculates: the same generation,
    // over layers whose windows keep enough to take four tokens back and over
    // layers that keep nothing.
    eprintln!("\nwith nothing speculating, over {SPECULATED_TOKENS} tokens");
    eprintln!("{:>7}  {:>10}", "slack", "ms/token");
    let mut idle = Vec::new();
    for slack in [0, 4] {
        let held = gpu.wrap(&ckpt, text, slack);
        let heads = gpu.heads(&ckpt, text, mtp);
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
        let heads = gpu.heads(&ckpt, text, mtp);
        let cost = time_chain(&held, &heads, text, &ids, depth);
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
            let heads = gpu.heads(&ckpt, text, mtp);
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

    eprintln!("\nwhat the loop banked, over {SPECULATED_TOKENS} tokens");
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

/// The kernels a speculative run compiles, held so that the weights wrapped
/// against them can be built once per depth.
struct Kernels<'d> {
    device: &'d Device,
    layers: LayerKernels,
    dense: DenseMatmul,
    swiglu: SwiGlu,
    router: Router,
    weights: RouterWeights,
    combine: MoeCombine,
}

impl<'d> Kernels<'d> {
    fn compile(device: &'d Device) -> Self {
        Self {
            device,
            layers: LayerKernels::compile(device).expect("the layer kernels compile"),
            dense: DenseMatmul::new(device).expect("the dense matmul compiles"),
            swiglu: SwiGlu::new(device).expect("the swiglu compiles"),
            router: Router::new(device).expect("the router compiles"),
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
                weights: &self.weights,
                combine: &self.combine,
            },
            &packed,
            &banks,
            StackShape {
                layers: config.num_hidden_layers,
                dim: config.hidden_size,
                slack,
            },
        )
        .expect("the layers wrap");
        mapped
            .with_head(Box::new(head))
            .with_backend(Box::new(layers))
    }

    /// The eight heads on the device.
    fn heads<'a>(
        &'a self,
        ckpt: &'a Checkpoint,
        config: &inkling_core::TextConfig,
        mtp: &inkling_core::config::MtpConfig,
    ) -> CheckpointHeads<'a> {
        let heads = CheckpointHeads::open(ckpt, config, mtp).expect("the heads open");
        let held = heads.head_projections();
        let wrapped = ModelHeads::wrap(self.device, &self.dense, self.layers.attention(), &held)
            .expect("the heads wrap");
        heads.with_backend(Box::new(wrapped))
    }
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
            budget: SPECULATED_TOKENS,
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

/// What `depth` heads cost to run over one row, which is what a round pays to
/// have guessed.
///
/// A row rather than a block, because it is the shape a round has when nothing
/// was accepted — and because what grows with the rows is the head's own
/// attention rather than its weights, which are read once whatever the row
/// count.
fn time_chain(
    weights: &CheckpointWeights<'_>,
    heads: &CheckpointHeads<'_>,
    config: &inkling_core::TextConfig,
    ids: &[usize],
    depth: usize,
) -> Duration {
    let generator = weights.generator();
    let cache = &mut ModelCache::speculating(config, 0);
    let hidden = generator
        .model()
        .final_norm(&generator.model().forward(cache, ids, weights));
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
    let started = Instant::now();
    for _ in 0..REPEATS {
        round(&mut proposer);
    }
    started.elapsed() / REPEATS as u32
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
