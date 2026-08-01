//! The routed expert matmuls, which were 73% of a decode step.
//!
//! The CPU path decodes about 44 GB to answer one token: 32 of them are the
//! experts, 9 the layer projections and 3.3 the head. This is the 32, and taking
//! it takes a decode step from 8.21 s to 3.19 s.
//!
//! **Nothing is uploaded.** The forty MoE layers' banks are 137 GB of packed
//! bytes — the whole checkpoint but for its two ends — and a policy that copied
//! them onto the device would take two minutes at load and hold a second copy of
//! the model. [`Device::wrap`](crate::Device::wrap) is the other side of that:
//! every bank is handed over where the checkpoint mapped it, all 137 GB in 5.6
//! ms and with no resident set of its own, and what the GPU then reads are the
//! file's own pages. Wrapping a bank nobody routes to costs nothing, so *every*
//! bank is wrapped at load and the residency question — upload at construction,
//! upload lazily, or map — stops being one. The resident set goes down rather
//! than up: 20.8 GiB to 2.4 GiB over the same four-token generation.
//!
//! **One dispatch a projection, indexed by the gathered expert list.** The other
//! way to spend a layer is a dispatch per selected expert, and the arithmetic is
//! the same either way — the same six banks read, the same 250 untouched. What
//! differs is what surrounds it. Six experts by three projections by forty
//! layers is 720 dispatches for the routed banks alone, against the 13
//! microseconds one expert's 4 MB takes to read at 267 GB/s. Gathered, a layer
//! is six dispatches — gate, up and down of each bank — and a step is 240.
//!
//! What that costs is worth stating, because it is what the term became rather
//! than what it stopped being. Measured over all forty layers at decode shape,
//! with each of the 240 dispatches submitted on its own: 72 ms once the pages
//! are resident, which is 300 microseconds a dispatch against the 16 ms its 4.28
//! GB would take at the bandwidth — so the expert term is now bound by
//! dispatching and not by reading. Three command buffers a layer is what that
//! reading led to, and the figure is the one that argued for them rather than a
//! measurement of what they cost. Cold it is 892 ms, which is the checkpoint
//! arriving from disk at 5 GB/s and is a cost the CPU path paid too, under six
//! seconds of dequantisation that hid it.
//!
//! Two things would move it. The shared bank is two experts and gets three
//! dispatches of its own, the same as the routed bank's six; and gate and up
//! read different tensors but the same rows. Merging either pair is a dispatch
//! count halved, and neither was worth doing before the 9 GB of layer
//! projections that were 78% of a step — which `crate::projections` has since
//! taken, leaving dispatching the term rather than the arithmetic.
//!
//! **The SwiGLU is a fourth dispatch and not a fourth submission.** Between
//! `gate_proj` and `down_proj` sits `silu(gate) * up` over `[rows, 2048]`, which
//! for a decode step is eight rows — 16384 multiplies against the 4.3 GB the
//! dispatches around it read, so as arithmetic it is free wherever it runs. What
//! it cost on the CPU was the command buffer it closed: `down` reads what the
//! pair produced, so an activation this process had to see meant reading both
//! outputs back, multiplying them, and copying the answer over again as `down`'s
//! input. Encoded between them — see [`crate::SwiGlu`] — a bank is four
//! dispatches in one command buffer and nothing it computes is ever a
//! `Vec<f32>` here.

use inkling_core::layer::Experts;
use inkling_core::moe::{BankRows, Gathered, Routed, Rows};
use inkling_core::profile::{self, Op};
use inkling_core::weights::{LayerBanks, PackedExperts};

use crate::buffer::Buffer;
use crate::dense::{DenseMatmul, DenseWeight};
use crate::device::Device;
use crate::kernel::Batch;
use crate::matmul::{MatmulError, PackedBank, PackedMatmul, Pending};
use crate::router::{LayerRouter, Router};
use crate::swiglu::SwiGlu;

/// One `SwitchGLU`'s three banks on the device: `[experts, hidden_dim, dim]`
/// gate and up projections beside `[experts, dim, hidden_dim]` down projections.
///
/// The mirror of [`PackedExperts`], which is the same three banks left in the
/// mapping — and holds the same relation to it that
/// [`PackedProjection`](crate::PackedProjection) holds to
/// [`PackedRows`](inkling_core::PackedRows): the arithmetic is the checkpoint's,
/// and what changes is that no weight is ever decoded to memory.
#[derive(Debug)]
pub struct ExpertBanks<'a> {
    gate_proj: PackedBank<'a>,
    up_proj: PackedBank<'a>,
    down_proj: PackedBank<'a>,
    /// The activation between the first two and the third, which is not a
    /// weight and belongs to no bank — one pipeline serves the whole model.
    swiglu: &'a SwiGlu,
}

impl<'a> ExpertBanks<'a> {
    /// Three banks that are one `SwitchGLU`, however they reached the device.
    ///
    /// The shapes are checked here rather than assumed, and this is the only
    /// place they can be: `gate_proj` and `up_proj` both map `dim` to the width
    /// between and `down_proj` maps it back, so three banks of a plausible
    /// shape that are not each other's — one layer's `down_proj` beside
    /// another's `gate_proj`, or `up_proj` from a bank of a different width —
    /// would run and be quietly wrong. `silu(gate) * up` is a zip, so a `up`
    /// narrower than `gate` would not even be a length mismatch downstream; it
    /// would be a truncation.
    pub fn new(
        gate_proj: PackedBank<'a>,
        up_proj: PackedBank<'a>,
        down_proj: PackedBank<'a>,
        swiglu: &'a SwiGlu,
    ) -> Result<Self, MatmulError> {
        let pair = |what, got, expected| match got == expected {
            true => Ok(()),
            false => Err(MatmulError::MismatchedBanks {
                what,
                expected,
                got,
            }),
        };
        pair("experts of up_proj", up_proj.experts(), gate_proj.experts())?;
        pair(
            "experts of down_proj",
            down_proj.experts(),
            gate_proj.experts(),
        )?;
        pair(
            "the width up_proj maps from",
            up_proj.in_dim(),
            gate_proj.in_dim(),
        )?;
        pair(
            "the width up_proj maps to",
            up_proj.out_dim(),
            gate_proj.out_dim(),
        )?;
        pair(
            "the width down_proj maps from",
            down_proj.in_dim(),
            gate_proj.out_dim(),
        )?;
        pair(
            "the width down_proj maps to",
            down_proj.out_dim(),
            gate_proj.in_dim(),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            swiglu,
        })
    }

    /// Wrap a checkpoint's three banks, `dim` wide in and out.
    ///
    /// The width between is read off `gate_proj` rather than taken, because it
    /// is the one dimension the checkpoint's shapes do not state directly — and
    /// [`ExpertBanks::new`] is then what says the other two agree about it.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        swiglu: &'a SwiGlu,
        banks: &PackedExperts<'a>,
        dim: usize,
    ) -> Result<Self, MatmulError> {
        let gate_proj = PackedBank::wrap(device, matmul, &banks.gate_proj(), dim)?;
        let hidden_dim = gate_proj.out_dim();
        Self::new(
            gate_proj,
            PackedBank::wrap(device, matmul, &banks.up_proj(), dim)?,
            PackedBank::wrap(device, matmul, &banks.down_proj(), hidden_dim)?,
            swiglu,
        )
    }

    pub fn experts(&self) -> usize {
        self.gate_proj.experts()
    }

    /// The width in and out, which is the layer's hidden size.
    pub fn dim(&self) -> usize {
        self.gate_proj.in_dim()
    }

    /// The width between, which is `moe_intermediate_size`.
    pub fn hidden_dim(&self) -> usize {
        self.gate_proj.out_dim()
    }

    /// Every gathered row through the expert it named, as the SwiGLU MLP an
    /// expert is.
    ///
    /// Four dispatches over the same expert list, in one command buffer: `x @
    /// gate^T` and `x @ up^T` read the same rows and nothing of each other, and
    /// the activation and `down` each read what the dispatch before them wrote.
    ///
    /// One bank on its own, which is what a caller with a single bank in hand
    /// wants — and what [`LayerExperts`] measures its own schedule against,
    /// since a layer that runs both banks answers what running each of them
    /// answers.
    pub fn forward(&self, gathered: Gathered<'_>) -> Result<Vec<f32>, MatmulError> {
        let chosen = chosen(gathered);
        let mut batch = self.device().batch()?;
        let out = self.encode(&mut batch, &chosen, gathered.rows())?;
        batch.wait()?;
        Ok(out.take())
    }

    /// The whole bank encoded into `batch`: the pair, the activation between
    /// them, and `down` over what it produced.
    ///
    /// **Nothing here waits for this side.** Every value between the rows handed
    /// in and the rows handed back is a buffer the next dispatch reads, so a
    /// caller with something else to put in the same command buffer may — which
    /// is what a MoE layer's two banks are to each other.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        chosen: &[u32],
        rows: &[f32],
    ) -> Result<Pending, MatmulError> {
        let glu = self.encode_glu(batch, chosen, rows)?;
        let Some(mut activated) = self.activated(batch, glu)? else {
            return Ok(Pending::empty());
        };
        self.down_proj.encode_over(batch, chosen, &mut activated)
    }

    /// The same bank over every row of `x`, once per expert `chosen` names in
    /// turn — see [`PackedBank::encode_repeating`].
    ///
    /// **The shared bank's rows are not laid out either.** Every token goes
    /// through every shared expert, so what the bank runs is the hidden state
    /// read over again rather than a `[n_shared * tokens, hidden]` tensor
    /// somebody built and copied over twice — once for `gate` and once for `up`.
    /// What it reads instead is the buffer the router's gate and the routed bank
    /// are already reading.
    pub(crate) fn encode_repeated(
        &self,
        batch: &mut Batch<'_>,
        chosen: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        let glu = [
            self.gate_proj.encode_repeating(batch, chosen, x)?,
            self.up_proj.encode_repeating(batch, chosen, x)?,
        ];
        let Some(mut activated) = self.activated(batch, glu)? else {
            return Ok(Pending::empty());
        };
        self.down_proj.encode_over(batch, chosen, &mut activated)
    }

    /// The same bank over an expert list a dispatch already left on the device,
    /// `per_source` of its rows reading each row of `x` — see
    /// [`PackedBank::encode_picked`].
    ///
    /// **The rows are never laid out.** `gate` and `up` read the hidden state
    /// itself at the stride the routing implies, and only what they produced is
    /// a tensor of this bank's own shape — which `down` then reads one row to a
    /// row, the expert list being the same list either way.
    pub(crate) fn encode_picked(
        &self,
        batch: &mut Batch<'_>,
        chosen: &mut Buffer<u32>,
        x: &mut Buffer<f32>,
        per_source: usize,
    ) -> Result<Pending, MatmulError> {
        let glu = [
            self.gate_proj.encode_picked(batch, chosen, x, per_source)?,
            self.up_proj.encode_picked(batch, chosen, x, per_source)?,
        ];
        let Some(mut activated) = self.activated(batch, glu)? else {
            return Ok(Pending::empty());
        };
        self.down_proj
            .encode_picked(batch, chosen, &mut activated, 1)
    }

    pub fn device(&self) -> &'a Device {
        self.gate_proj.device()
    }

    /// `silu(gate) * up`, encoded over the pair's own outputs and left in the
    /// gate's buffer — `None` for a bank no row named, which dispatched neither.
    fn activated(
        &self,
        batch: &mut Batch<'_>,
        [gate, up]: [Pending; 2],
    ) -> Result<Option<Buffer<f32>>, MatmulError> {
        let (Some(mut gate), Some(mut up)) = (gate.into_buffer(), up.into_buffer()) else {
            return Ok(None);
        };
        self.swiglu.encode(batch, &mut gate, &mut up)?;
        Ok(Some(gate))
    }

    /// The two dispatches that read the gathered rows, encoded into `batch`.
    ///
    /// **A dispatch that reads what these rows were gathered from costs no
    /// submission at all.** `gate` and `up` read the hidden state and nothing
    /// this bank produces, so anything else with the same input can be encoded
    /// beside them — the router's gate is exactly that, and 225 microseconds a
    /// submission is what forty of them would otherwise cost a step.
    fn encode_glu(
        &self,
        batch: &mut Batch<'_>,
        chosen: &[u32],
        rows: &[f32],
    ) -> Result<[Pending; 2], MatmulError> {
        Ok([
            self.gate_proj.encode(batch, chosen, rows)?,
            self.up_proj.encode(batch, chosen, rows)?,
        ])
    }
}

/// The expert each row goes through, as the kernel indexes them.
fn chosen(gathered: Gathered<'_>) -> Vec<u32> {
    gathered
        .experts()
        .iter()
        .map(|expert| {
            u32::try_from(*expert).unwrap_or_else(|_| panic!("expert {expert} is a wide index"))
        })
        .collect()
}

/// The four kernels a MoE layer dispatches through, compiled once for the whole
/// model.
///
/// Held together because that is what they are: none of the four names a shape,
/// so one pipeline each serves all forty layers, and a layer standing itself up
/// needs all four. Borrowed rather than owned because the first of them is not
/// the experts' — `crate::matmul` is the same kernel every projection and the
/// head dispatch through, and a second compilation of it would be a second
/// pipeline for one source string.
#[derive(Debug, Clone, Copy)]
pub struct ExpertKernels<'a> {
    pub matmul: &'a PackedMatmul,
    /// The gate, which is the one weight in the model the quantiser left alone.
    pub dense: &'a DenseMatmul,
    /// The activation between a bank's two halves.
    pub swiglu: &'a SwiGlu,
    /// The top-k over what the gate produced.
    pub router: &'a Router,
}

/// One MoE layer's two banks and the router that chooses between them.
///
/// The routed bank is 256 experts of which a token reads six and the shared bank
/// is two every token reads, and nothing else separates them — the same four
/// dispatches over the same expert list, differing in how much of the bank the
/// list names.
///
/// **The gate and the top-k over it are here because of what they can be
/// dispatched beside.** Neither is an expert's weight — the gate is `[258,
/// 4096]` of bfloat16 and a correction bias is 1 KB, against 3.2 GiB of packed
/// banks — and both belong to the layer rather than to either bank. What they
/// buy by being here is that the whole layer is one command buffer: the gate
/// reads the hidden state the shared bank reads, the top-k reads what the gate
/// wrote, and the routed bank indexes its experts out of what the top-k wrote.
#[derive(Debug)]
pub struct LayerExperts<'a> {
    routed: ExpertBanks<'a>,
    shared: ExpertBanks<'a>,
    gate: DenseWeight<'a>,
    router: LayerRouter<'a>,
}

impl<'a> LayerExperts<'a> {
    /// The one device the banks, the gate and the router were wrapped on, which
    /// is what lets a command buffer opened here hold dispatches against any of
    /// them.
    fn device(&self) -> &'a Device {
        self.shared.device()
    }

    /// [`Experts::banks`]'s one command buffer, with the errors a dispatch can
    /// fail with still in hand.
    ///
    /// **`route` is asked after the wait and not between two submissions**,
    /// because nothing it answers decides a dispatch: the experts are picked
    /// here and the rows they name are the hidden state read at a stride, so
    /// what this side is left to do with the gate's logits is weight a selection
    /// that has already run.
    ///
    /// **Neither bank's rows are a tensor this copies.** The hidden state is
    /// uploaded once and read four ways: by the gate, by the routed bank's two
    /// halves at the stride the routing implies, and by the shared bank's two
    /// halves over again once per shared expert. What `shared` still carries is
    /// the expert each of those rows goes through; the rows themselves are
    /// `SparseMoe::shared_rows`'s promise that they are `x` laid end to end
    /// after itself, which is what makes the modulo the right reading of them.
    fn encode(
        &self,
        x: &[f32],
        shared: Gathered<'_>,
        route: &mut dyn FnMut(Option<Routed<'_>>) -> Option<Rows>,
    ) -> Result<BankRows, MatmulError> {
        let tokens = x.len() / self.shared.dim();
        assert_eq!(
            chosen(shared),
            self.shared_chosen(tokens),
            "the shared bank's rows against the passes over the hidden state they are"
        );

        let mut hidden = self.device().buffer(x)?;
        let mut batch = self.device().batch()?;
        let dispatched = self.encode_into(&mut batch, &mut hidden, tokens)?;
        batch.wait()?;

        let Answered {
            logits,
            picked,
            banks,
        } = dispatched.answered();
        let gathered = route(Some(Routed::Picked {
            logits: &logits,
            experts: &picked,
        }));
        assert!(
            gathered.is_none(),
            "the layer gathered rows for a bank that had already run"
        );
        Ok(banks)
    }

    /// The layer's ten dispatches encoded into `batch`, over a hidden state
    /// already on the device.
    ///
    /// **Nothing between the gate and the routed bank's last dispatch waits for
    /// this side**, which is what lets a caller with something else in the same
    /// command buffer put it there — and what a whole decoder layer is, since
    /// the norm that produced `x` is three dispatches back in the same buffer.
    pub(crate) fn encode_into(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        tokens: usize,
    ) -> Result<Dispatched, MatmulError> {
        let mut logits = self.gate.encode_over(batch, x)?.buffer();
        let mut picked = self.router.encode(batch, &mut logits)?;
        let shared = self
            .shared
            .encode_repeated(batch, &self.shared_chosen(tokens), x)?;
        let routed =
            self.routed
                .encode_picked(batch, &mut picked, x, self.router.config().top_k)?;
        Ok(Dispatched {
            logits,
            picked,
            shared,
            routed,
        })
    }

    /// The expert each of the shared bank's rows goes through, which is
    /// `SparseMoe::shared_rows`'s own list: every token once per shared expert,
    /// expert-major.
    ///
    /// Derived rather than taken, because the rows it names are not a tensor
    /// anybody hands over any more — see [`ExpertBanks::encode_repeated`]. A
    /// caller that still has the layer's `Gathered` in hand is the one that
    /// checks the two agree.
    fn shared_chosen(&self, tokens: usize) -> Vec<u32> {
        let rows = tokens * self.router.config().n_shared;
        (0..rows).map(|row| (row / tokens.max(1)) as u32).collect()
    }

    /// Wrap one layer's banks, its gate and its router.
    ///
    /// **The three widths are checked against each other here, and this is the
    /// only place they can be.** The router picks out of `0..n_routed` and
    /// writes those indices where nothing on this side sees them, so what keeps
    /// [`PackedBank::encode_picked`] inside the bank it indexes is that the
    /// router's `n_routed` is the routed bank's own expert count. A gate of
    /// another layer's width would be a distribution over experts this layer
    /// does not have.
    pub fn wrap(
        device: &'a Device,
        kernels: ExpertKernels<'a>,
        banks: &LayerBanks<'a>,
        dim: usize,
    ) -> Result<Self, MatmulError> {
        let routed = ExpertBanks::wrap(device, kernels.matmul, kernels.swiglu, &banks.routed, dim)?;
        let shared = ExpertBanks::wrap(device, kernels.matmul, kernels.swiglu, &banks.shared, dim)?;
        let gate = DenseWeight::wrap(device, kernels.dense, &banks.gate_weight)?;
        let config = banks.config;
        let pair = |what, got, expected| match got == expected {
            true => Ok(()),
            false => Err(MatmulError::MismatchedBanks {
                what,
                expected,
                got,
            }),
        };
        pair(
            "the routed experts of the bank",
            routed.experts(),
            config.n_routed,
        )?;
        pair(
            "the shared experts of the bank",
            shared.experts(),
            config.n_shared,
        )?;
        pair(
            "the rows of the gate",
            gate.out_dim(),
            config.n_routed + config.n_shared,
        )?;
        pair("the width the gate maps from", gate.in_dim(), dim)?;

        Ok(Self {
            routed,
            shared,
            gate,
            router: LayerRouter::new(device, kernels.router, config, &banks.correction_bias)?,
        })
    }
}

/// What a MoE layer's ten dispatches will have left on the device once the
/// command buffer they were encoded into completes.
#[derive(Debug)]
pub struct Dispatched {
    logits: Buffer<f32>,
    picked: Buffer<u32>,
    shared: Pending,
    routed: Pending,
}

/// The same four read back: the gate's logits, the experts the top-k picked out
/// of them, and the rows both banks answered.
///
/// All four together, because they are read for one thing — the weights the
/// scatter carries and the rows it carries them over — and a caller that took
/// some without the rest would be weighting a selection it did not have.
#[derive(Debug)]
pub struct Answered {
    pub logits: Vec<f32>,
    pub picked: Vec<usize>,
    pub banks: BankRows,
}

impl Dispatched {
    pub fn answered(self) -> Answered {
        let _timed = profile::scope(Op::Readback);
        Answered {
            logits: self.logits.to_vec(),
            picked: self.picked.as_slice().iter().map(|e| *e as usize).collect(),
            banks: BankRows {
                routed: self.routed.take(),
                shared: self.shared.take(),
            },
        }
    }
}

/// The seam [`inkling_core::layer`] names, so that a layer running its MoE does
/// not know whether an expert was ever decoded.
///
/// Infallible where [`ExpertBanks::forward`] is not, for the reason
/// [`PackedProjection`](crate::PackedProjection)'s side of the same bargain is:
/// a dispatch that does not complete is not a condition a decode step can do
/// anything about.
impl Experts for LayerExperts<'_> {
    fn routed(&self, gathered: Gathered<'_>) -> Vec<f32> {
        through(&self.routed, gathered)
    }

    fn shared(&self, gathered: Gathered<'_>) -> Vec<f32> {
        through(&self.shared, gathered)
    }

    /// The whole layer — the gate, the top-k over it, both banks and the SwiGLU
    /// inside each — in one command buffer.
    ///
    /// **Ten dispatches in one submission, where the same ten asked for a piece
    /// at a time are four.** Every value between the hidden state this is handed
    /// and the two banks' rows is a buffer the next dispatch reads: the gate's
    /// logits, the six indices the top-k took out of them, each bank's two
    /// halves and the activation between them. Nothing in the middle is a value
    /// this process forms or reads.
    ///
    /// **The rows the routed bank runs are not a tensor anyone built.** A token
    /// reads six experts, so its six rows are one row of the hidden state read
    /// six times — see
    /// [`PackedBank::encode_picked`](crate::PackedBank::encode_picked) — and
    /// laying them out end to end is an integer divide inside the kernel rather
    /// than a gather anyone runs.
    fn banks(
        &self,
        x: &[f32],
        shared: Gathered<'_>,
        route: &mut dyn FnMut(Option<Routed<'_>>) -> Option<Rows>,
    ) -> BankRows {
        self.encode(x, shared, route)
            .unwrap_or_else(|err| panic!("the layer's experts did not run: {err}"))
    }

    /// Yes, which is what keeps 4.2 MB of float32 a layer from being widened on
    /// the other side of the seam for nobody to read.
    fn gates(&self) -> bool {
        true
    }
}

fn through(banks: &ExpertBanks<'_>, gathered: Gathered<'_>) -> Vec<f32> {
    banks
        .forward(gathered)
        .unwrap_or_else(|err| panic!("the expert matmul did not run: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::fixture::deviation;
    use inkling_core::moe::{ExpertBank, MoeConfig};
    use inkling_core::quant::{GROUP_SIZE, dequantize_blocks};

    use crate::dense::testing::narrowed;
    use crate::matmul::testing::{Case, Noise, pack};
    use crate::testing::device;

    /// Narrow enough to run three banks of it in a unit test, and wide enough
    /// that a reduction over it cancels: the checkpoint's widths are 4096 and
    /// 2048, and nothing here depends on which.
    const DIM: usize = 128;
    const HIDDEN_DIM: usize = 64;
    const EXPERTS: usize = 3;

    /// The same account as `matmul::tests::TOLERANCE`, over a shorter reduction
    /// and one more of them: an expert is three multiplies deep, so what
    /// separates the two sides is three summation orders rather than one.
    const TOLERANCE: f32 = 6e-6;

    /// The shape a synthetic layer routes under: three routed experts and the
    /// shared pair, two of the three per token — the shape the layer's own 256,
    /// 2 and 6 is, small enough to run three banks of in a unit test.
    const CONFIG: MoeConfig = MoeConfig {
        n_routed: EXPERTS,
        n_shared: 2,
        top_k: 2,
        route_scale: 8.0,
    };

    /// Rows a synthetic router's gate has, which is every expert it chooses
    /// between.
    const GATE_ROWS: usize = CONFIG.n_routed + CONFIG.n_shared;

    /// A correction bias that is not all one value, so that a router which
    /// dropped it would rank differently.
    fn correction_bias() -> Vec<f32> {
        (0..CONFIG.n_routed)
            .map(|i| (i as f32 - 1.0) / 8.0)
            .collect()
    }

    /// What a call cost the device: `(dispatches, submissions, allocations)`,
    /// which is what the granularity question is settled in.
    fn spent<T>(device: &Device, run: impl FnOnce() -> T) -> (T, (u64, u64, u64)) {
        let before = (
            device.dispatches(),
            device.submissions(),
            device.allocations(),
        );
        let got = run();
        (
            got,
            (
                device.dispatches() - before.0,
                device.submissions() - before.1,
                device.allocations() - before.2,
            ),
        )
    }

    /// A synthetic gate, bfloat16 as the checkpoint stores one.
    fn gate<'a>(device: &'a Device, dense: &'a DenseMatmul, seed: u32) -> DenseWeight<'a> {
        let mut noise = Noise(seed);
        let values: Vec<f32> = (0..GATE_ROWS * DIM).map(|_| noise.signed()).collect();
        DenseWeight::upload(device, dense, DIM, GATE_ROWS, &narrowed(&values))
            .expect("the gate's shape pairs")
    }

    /// One synthetic `SwitchGLU`: three banks of `EXPERTS` experts, held both
    /// packed and decoded so that the same arithmetic can be run either way.
    struct Banks {
        gate: Case,
        up: Case,
        down: Case,
    }

    impl Banks {
        /// Three banks whose codes differ, which they have to: against three
        /// identical banks, exchanging two of them would change nothing.
        fn new() -> Self {
            Self {
                gate: Case::seeded(0x5eed_1111, DIM, EXPERTS * HIDDEN_DIM, 1),
                up: Case::seeded(0x5eed_2222, DIM, EXPERTS * HIDDEN_DIM, 1),
                down: Case::seeded(0x5eed_3333, HIDDEN_DIM, EXPERTS * DIM, 1),
            }
        }

        fn upload<'a>(
            &self,
            device: &'a Device,
            matmul: &'a PackedMatmul,
            swiglu: &'a SwiGlu,
            gate: &Case,
            up: &Case,
        ) -> Result<ExpertBanks<'a>, MatmulError> {
            let bank = |case: &Case, in_dim, out_dim| {
                PackedBank::upload(
                    device,
                    matmul,
                    EXPERTS,
                    in_dim,
                    out_dim,
                    &pack(&case.codes),
                    &case.scales,
                )
            };
            ExpertBanks::new(
                bank(gate, DIM, HIDDEN_DIM)?,
                bank(up, DIM, HIDDEN_DIM)?,
                bank(&self.down, HIDDEN_DIM, DIM)?,
                swiglu,
            )
        }

        /// The same experts as decoded float32, through the CPU's own
        /// [`ExpertBank`] — which is the oracle, because it is what
        /// `inkling_core` pins to mlx-vlm.
        fn on_the_cpu(&self, expert: usize, rows: &[f32]) -> Vec<f32> {
            let decode = |case: &Case| {
                dequantize_blocks(&pack(&case.codes), &case.scales).expect("the case decodes")
            };
            let (gate, up, down) = (decode(&self.gate), decode(&self.up), decode(&self.down));
            ExpertBank::new(EXPERTS, DIM, &gate, &up, &down)
                .expert(expert)
                .forward(rows)
        }
    }

    /// Rows of `x`, one per assignment, spread over both signs.
    fn rows(count: usize) -> Vec<f32> {
        (0..count * DIM)
            .map(|i| ((i * 37 % 71) as f32 - 35.0) / 35.0)
            .collect()
    }

    /// The whole of what this module composes: three gathered dispatches and a
    /// SwiGLU between them are the expert the CPU decodes and runs.
    ///
    /// Every row goes through a different expert and one is repeated, which is
    /// what a decode step's routing looks like — so this pins the gather and the
    /// composition together rather than one at a time.
    ///
    /// **What it costs is asserted beside what it answers**, because the two
    /// move independently and only one of them is visible in the values. Four
    /// dispatches in one command buffer, and five allocations for them: the rows
    /// copied over for `gate` and again for `up`, an output each for the three
    /// multiplies, and nothing at all for the activation — which writes into the
    /// buffer `gate` produced and hands it to `down`. A path that read the pair
    /// back and copied the product over again is the same four values and six
    /// allocations in two command buffers.
    #[test]
    fn three_banks_and_a_swiglu_are_the_expert_the_cpu_decodes() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let banks = Banks::new();
        let resident = banks
            .upload(&device, &matmul, &swiglu, &banks.gate, &banks.up)
            .expect("the three banks pair");
        assert_eq!(resident.experts(), EXPERTS);
        assert_eq!(resident.dim(), DIM);
        assert_eq!(resident.hidden_dim(), HIDDEN_DIM);

        let chosen = [2usize, 0, 2];
        let x = rows(chosen.len());
        let (got, spent) = spent(&device, || {
            resident
                .forward(Gathered::new(DIM, &chosen, &x))
                .expect("the dispatches complete")
        });
        assert_eq!(got.len(), chosen.len() * DIM);
        assert_eq!(
            spent,
            (4, 1, 5),
            "the bank's dispatches, buffers and memory"
        );

        for (row, expert) in chosen.iter().enumerate() {
            let want = banks.on_the_cpu(*expert, &x[row * DIM..][..DIM]);
            let deviation = deviation(&got[row * DIM..][..DIM], &want);
            assert!(deviation <= TOLERANCE, "row {row}: deviation {deviation:e}");
        }
        assert_ne!(
            got[..DIM],
            got[DIM..2 * DIM],
            "two experts that agreed would prove nothing"
        );
    }

    /// `silu` goes on the gate projection and not on the up projection, which
    /// [`inkling_core::ops::swiglu`] is the authority on and which a backend
    /// running the two as separate dispatches can get backwards while producing
    /// two projections of exactly the right shape.
    #[test]
    fn exchanging_the_gate_and_up_banks_changes_the_answer() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let banks = Banks::new();

        let chosen = [1usize];
        let x = rows(1);
        let through = |gate: &Case, up: &Case| {
            banks
                .upload(&device, &matmul, &swiglu, gate, up)
                .expect("the three banks pair")
                .forward(Gathered::new(DIM, &chosen, &x))
                .expect("the dispatches complete")
        };

        let want = banks.on_the_cpu(chosen[0], &x);
        assert!(deviation(&through(&banks.gate, &banks.up), &want) <= TOLERANCE);

        let swapped = deviation(&through(&banks.up, &banks.gate), &want);
        assert!(swapped > TOLERANCE, "deviation {swapped:e}");
    }

    /// Three banks that are not each other's is the mistake the shapes exist to
    /// catch, and it has to be caught here: `silu(gate) * up` is a zip, so a
    /// narrower `up` would truncate the answer rather than fail.
    #[test]
    fn banks_that_do_not_pair_are_refused() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let banks = Banks::new();

        // An `up_proj` of half the width, which is another layer's bank as far
        // as anything but the shape can tell.
        let narrow = Case::seeded(0x5eed_4444, DIM, EXPERTS * (HIDDEN_DIM / 2), 1);
        let bank = |case: &Case, in_dim, out_dim| {
            PackedBank::upload(
                &device,
                &matmul,
                EXPERTS,
                in_dim,
                out_dim,
                &pack(&case.codes),
                &case.scales,
            )
            .expect("the case's shapes pair")
        };

        let err = ExpertBanks::new(
            bank(&banks.gate, DIM, HIDDEN_DIM),
            bank(&narrow, DIM, HIDDEN_DIM / 2),
            bank(&banks.down, HIDDEN_DIM, DIM),
            &swiglu,
        )
        .expect_err("the banks do not pair");
        assert!(
            matches!(
                err,
                MatmulError::MismatchedBanks {
                    expected: HIDDEN_DIM,
                    got: 32,
                    ..
                }
            ),
            "{err}"
        );
        assert_eq!(GROUP_SIZE, 32, "the case's widths are whole groups");
    }

    /// The whole layer asked for at once is the same four answers as the gate,
    /// the top-k, the shared bank and the routed bank asked for apart — **in the
    /// same ten dispatches and three fewer submissions.**
    ///
    /// Both halves are the commit. That the answers agree says the schedule
    /// changed no arithmetic; that the dispatch count does not move while the
    /// submission count does is the whole reason for scheduling at all, and it
    /// is the half a test of the values alone would let slip.
    ///
    /// The saved round trips are the whole of the schedule. The gate reads the
    /// hidden state the shared bank's pair reads; the top-k reads what the gate
    /// wrote; and the routed bank indexes its experts out of what the top-k
    /// wrote and its rows out of the hidden state itself. Not one of the four
    /// waits for this side, so over forty MoE layers that is 120 round trips a
    /// decode step does not take.
    ///
    /// Five fewer buffers, too: the hidden state is uploaded once for the gate,
    /// both of the routed bank's halves and both of the shared bank's to read,
    /// where the parts apart copy it over for each of them — and the shared
    /// bank's copies are `n_shared` times the size, being every token once per
    /// shared expert.
    #[test]
    fn the_whole_layer_costs_three_fewer_submissions_than_its_four_parts_apart() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let dense = DenseMatmul::new(&device).expect("the dense matmul compiles");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let kernel = Router::new(&device).expect("the router compiles");
        let bias = correction_bias();
        let banks = Banks::new();
        let layer = LayerExperts {
            routed: banks
                .upload(&device, &matmul, &swiglu, &banks.gate, &banks.up)
                .expect("the three banks pair"),
            shared: banks
                .upload(&device, &matmul, &swiglu, &banks.gate, &banks.up)
                .expect("the three banks pair"),
            gate: gate(&device, &dense, 0x5eed_6666),
            router: LayerRouter::new(&device, &kernel, CONFIG, &bias)
                .expect("the router stands up"),
        };

        // Two tokens, and the shape `SparseMoe::shared_rows` hands the shared
        // bank: every token once per shared expert, expert-major — which is `x`
        // laid end to end after itself, and is what the layer reads at a modulo
        // rather than copying over.
        let x = rows(2);
        let shared_chosen = [0usize, 0, 1, 1];
        let shared_x: Vec<f32> = x.iter().chain(&x).copied().collect();
        let shared = Gathered::new(DIM, &shared_chosen, &shared_x);

        let mut asked = None;
        let (got, together) = spent(&device, || {
            layer.banks(&x, shared, &mut |routed| {
                let Some(Routed::Picked { logits, experts }) = routed else {
                    panic!("the backend picked the experts and said so")
                };
                asked = Some((logits.to_vec(), experts.to_vec()));
                None
            })
        });
        let (logits, picked) = asked.expect("the layer was routed");

        // The rows the routed bank ran, laid out: each token's own row, once per
        // slot it selected. What the whole layer never builds.
        let routed_x: Vec<f32> = (0..picked.len())
            .flat_map(|row| x[(row / CONFIG.top_k) * DIM..][..DIM].iter().copied())
            .collect();

        let ((apart_logits, apart_picked, apart_rows), apart) = spent(&device, || {
            let logits = layer.gate.multiply(&x).expect("the dispatch completes");
            let picked = layer
                .router
                .select(&logits)
                .expect("the dispatch completes");
            let shared = layer.shared(shared);
            let routed = layer.routed(Gathered::new(DIM, &widened(&picked), &routed_x));
            (logits, picked, BankRows { routed, shared })
        });

        assert_eq!(got, apart_rows, "both banks' rows");
        assert_eq!(logits, apart_logits, "the logits");
        assert_eq!(picked, widened(&apart_picked), "the selection");
        assert!(
            picked.iter().any(|expert| *expert != picked[0]),
            "a selection of one expert would say nothing about the stride: {picked:?}"
        );

        assert_eq!(together.0, apart.0, "the same dispatches");
        assert_eq!(together, (10, 1, 9), "the layer's own cost");
        assert_eq!(apart, (10, 4, 14), "what the four parts cost apart");
    }

    /// The selection as the layer hands it over, which is how a device index
    /// reaches a side that counts in `usize`.
    fn widened(picked: &[u32]) -> Vec<usize> {
        picked.iter().map(|expert| *expert as usize).collect()
    }

    /// Which banks a layer answers with, which is the one thing a stack of these
    /// can get wrong that a single layer cannot: two layers built from the same
    /// gate, up and down but different routed banks have to disagree.
    ///
    /// Which *index* answers with which layer is the stack's own question, and
    /// is `projections::tests`' — see
    /// `a_layer_this_does_not_hold_and_a_layer_past_the_stack_are_left_to_the_cpu`,
    /// which is where every layer on the device is held now.
    #[test]
    fn two_layers_built_from_different_banks_do_not_answer_alike() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let dense = DenseMatmul::new(&device).expect("the dense matmul compiles");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let kernel = Router::new(&device).expect("the router compiles");
        let bias = correction_bias();
        let banks = Banks::new();
        let layer = |routed: &Case| {
            let bank = |case: &Case, in_dim, out_dim| {
                PackedBank::upload(
                    &device,
                    &matmul,
                    EXPERTS,
                    in_dim,
                    out_dim,
                    &pack(&case.codes),
                    &case.scales,
                )
                .expect("the case's shapes pair")
            };
            LayerExperts {
                routed: ExpertBanks::new(
                    bank(routed, DIM, HIDDEN_DIM),
                    bank(&banks.up, DIM, HIDDEN_DIM),
                    bank(&banks.down, HIDDEN_DIM, DIM),
                    &swiglu,
                )
                .expect("the banks pair"),
                shared: ExpertBanks::new(
                    bank(&banks.gate, DIM, HIDDEN_DIM),
                    bank(&banks.up, DIM, HIDDEN_DIM),
                    bank(&banks.down, HIDDEN_DIM, DIM),
                    &swiglu,
                )
                .expect("the banks pair"),
                gate: gate(&device, &dense, 0x5eed_5555),
                router: LayerRouter::new(&device, &kernel, CONFIG, &bias)
                    .expect("the router stands up"),
            }
        };

        // Two layers whose routed banks differ, which is what a stack holds and
        // what an index off by one confuses.
        let (first, second) = (layer(&banks.gate), layer(&banks.up));
        let chosen = [1usize];
        let x = rows(1);
        let of = |layer: &LayerExperts<'_>| layer.routed(Gathered::new(DIM, &chosen, &x));
        assert_ne!(of(&first), of(&second));
    }
}
