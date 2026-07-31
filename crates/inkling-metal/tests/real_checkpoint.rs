//! The packed matmul against a real Inkling-Small checkpoint, which is far too
//! large to commit. Set `INKLINGRS_CHECKPOINT` to a checkpoint directory to run
//! these; unset, each reports a skip and passes.
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

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use inkling_core::fixture::{self, ACTIVATIONS, deviation, indices};
use inkling_core::ops::linear;
use inkling_core::quant::{BITS, dequantize_blocks_into};
use inkling_core::{
    Checkpoint, CheckpointWeights, Dtype, Ending, ModelCache, Packed as CorePacked, TensorView,
};
use inkling_metal::{
    Device, MetalError, ModelExperts, ModelProjections, PackedBank, PackedMatmul, PackedProjection,
};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

/// The projection this measures, which is the largest in the model and the one
/// M3 routes through the kernel first.
const LM_HEAD: &str = "language_model.lm_head";

/// One MoE layer's routed bank, `[256, 2048, 4096]`, for the other shape a
/// projection comes in.
const ROUTED_EXPERTS: &str = "language_model.model.layers.2.mlp.switch_mlp.gate_proj";

const HIDDEN: usize = 4096;

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
/// head on the device, 2.44 GiB with the experts there too, and 0.18 GiB now
/// that the layers' own projections are.
///
/// **32 GiB was slack and 1 GiB is a claim.** What is left resident is what the
/// CPU still reads: the embedding rows a prompt asked for, the norms, the
/// convolution kernels and the routers' bfloat16 gates. Everything else is
/// pages the GPU reads where the checkpoint mapped them. The 981 MB layer
/// scratch is still allocated and is now never written, so it never faults in
/// either — which is the sharpest way to say what this commit did.
///
/// A bound this tight is a regression test with a name: a path that went back
/// to decoding a layer's projections into that scratch lands at 2.44 GiB and
/// fails here. What the number still does not measure is how much memory the
/// machine is using — the pages are in the unified buffer cache either way, and
/// what changed is whose they are.
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
/// Per layer: five attention projections in two submissions — the four that
/// share the normed hidden state, then `o_proj` — and per MoE layer six expert
/// dispatches in four, being each bank's gate and up together and then its
/// down. A dense layer's feed-forward network is three in two. The head is one
/// of each.
fn per_step(layers: u64, dense: u64) -> (u64, u64) {
    let moe = layers - dense;
    (
        5 * layers + 3 * dense + 6 * moe + 1,
        2 * layers + 2 * dense + 4 * moe + 1,
    )
}

/// The running `(dispatches, submissions)` of the last decode step, which is the
/// difference between the two totals either side of it.
///
/// The prefill is skipped: it dispatches the same kernels over more rows, and
/// what the count is being read for is the steady state.
fn per_decode_step(submitted: &[(u64, u64)]) -> (u64, u64) {
    let [.., before, after] = submitted else {
        panic!("a step per token, and more than one of them")
    };
    (after.0 - before.0, after.1 - before.1)
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
fn the_generated_tokens_match_the_oracle_with_the_model_on_the_device() {
    let Some(dir) = checkpoint_dir() else { return };
    let Some(device) = device() else { return };
    let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");

    let config = fixture::config(&dir).text_config;
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let activations = fixture::open(ACTIVATIONS);
    let ids = indices(&fixture::tensor(&activations, "input_ids"));
    let oracle = indices(&fixture::tensor(&activations, "greedy_continuation"));
    let want = &oracle[..GENERATED];

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let vocab = weights.head().vocab();
    let started = Instant::now();
    let head = PackedProjection::wrap_packed(&device, &matmul, &weights.head_packed(), vocab)
        .expect("the head wraps");
    let banks = weights.expert_banks();
    let experts = ModelExperts::wrap(
        &device,
        &matmul,
        &banks,
        config.num_hidden_layers,
        config.hidden_size,
    )
    .expect("the banks wrap");

    let packed = weights.layer_projections();
    let projections = ModelProjections::wrap(&device, &matmul, &packed, config.num_hidden_layers)
        .expect("the projections wrap");
    eprintln!(
        "{vocab} rows of the head, {} MoE layers' banks and {} layers' projections ({} of them \
         dense) wrapped in {:?}",
        experts.layers(),
        projections.layers(),
        projections.dense_layers(),
        started.elapsed()
    );
    assert_eq!(
        experts.layers(),
        config.num_hidden_layers - banks[0].layer,
        "every layer past the dense ones has banks here"
    );
    assert_eq!(
        projections.layers(),
        config.num_hidden_layers,
        "every layer has its projections here"
    );
    assert_eq!(
        projections.dense_layers() + experts.layers(),
        config.num_hidden_layers,
        "a layer has a feed-forward network or banks, never both and never neither"
    );

    // Read before the handover moves it, because what a decode step submits is
    // decided by how many layers are dense and how many route.
    let dense_layers = projections.dense_layers();

    // Once, before the loop rather than inside it — though "once" is now 6 ms
    // for 137 GB, so what that used to be defending against is gone.
    let weights = weights
        .with_head(Box::new(head))
        .with_experts(Box::new(experts))
        .with_projections(Box::new(projections));
    let generator = weights.generator();

    let mut steps: Vec<Duration> = Vec::new();
    // What each step submitted, as the running totals it was reached at. A
    // decode step's dispatch count is what the model's shape decides and its
    // submission count is what the batching decides, and the two are the whole
    // of the granularity question.
    let mut submitted: Vec<(u64, u64)> = Vec::new();
    let mut got = Vec::new();
    let mut step = Instant::now();
    let mut peak = fixture::resident_bytes();
    generator.stream(
        &mut ModelCache::new(&config),
        &ids,
        Ending {
            budget: GENERATED,
            eos: None,
        },
        &weights,
        |id| {
            steps.push(step.elapsed());
            submitted.push((device.dispatches(), device.submissions()));
            peak = peak.max(fixture::resident_bytes());
            got.push(id);
            step = Instant::now();
            ControlFlow::Continue(())
        },
    );

    // The prompt's prefill is the first step and every later one is a single
    // decode; a mean over the two describes neither.
    let (prefill, decode) = steps.split_first().expect("a step per token");
    let each = decode.iter().sum::<Duration>() / decode.len() as u32;
    let (dispatches, submissions) = per_decode_step(&submitted);
    eprintln!(
        "{} tokens prefilled in {prefill:.2?}, {} decoded at {each:.2?}/token, peak RSS {:.2} GiB\
         \n  {dispatches} dispatches a decode step in {submissions} submissions, which at 206 µs \
         a submission is {:.1?} of round trip\
         \n  got  {got:?}\n  want {want:?}",
        ids.len(),
        decode.len(),
        peak as f64 / (1u64 << 30) as f64,
        Duration::from_micros(206) * submissions as u32,
    );
    assert_eq!(
        (dispatches, submissions),
        per_step(config.num_hidden_layers as u64, dense_layers as u64),
        "a decode step's dispatches, and the command buffers they went in"
    );

    let agreed = got.iter().zip(want).take_while(|(a, b)| a == b).count();
    assert_eq!(got, want, "{agreed} of {GENERATED} tokens agree");
    assert!(
        peak < RESIDENT_BOUND,
        "peak RSS {peak} bytes is over the bound of {RESIDENT_BOUND}"
    );
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
/// to state on a real one at all: 1.06 GiB in 40 microseconds instead of 130
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
