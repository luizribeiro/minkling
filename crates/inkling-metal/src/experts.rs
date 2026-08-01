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
//! **The SwiGLU stays on the CPU.** Between `gate_proj` and `down_proj` sits
//! `silu(gate) * up` over `[rows, 2048]`, which for a decode step is eight rows
//! — 16384 multiplies against the 4.3 GB the dispatches around it read. A kernel
//! for it would be a fourth dispatch a bank to save nothing measurable, and the
//! buffers it would avoid touching are shared storage the CPU addresses anyway.

use inkling_core::layer::Experts;
use inkling_core::moe::{BankRows, Gathered, Rows};
use inkling_core::ops::swiglu;
use inkling_core::weights::{ExpertBackend, LayerBanks, PackedExperts};

use crate::dense::{DenseMatmul, DenseWeight};
use crate::device::Device;
use crate::kernel::Batch;
use crate::matmul::{MatmulError, PackedBank, PackedMatmul, Pending};

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
        banks: &PackedExperts<'a>,
        dim: usize,
    ) -> Result<Self, MatmulError> {
        let gate_proj = PackedBank::wrap(device, matmul, &banks.gate_proj(), dim)?;
        let hidden_dim = gate_proj.out_dim();
        Self::new(
            gate_proj,
            PackedBank::wrap(device, matmul, &banks.up_proj(), dim)?,
            PackedBank::wrap(device, matmul, &banks.down_proj(), hidden_dim)?,
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
    /// Three dispatches over the same expert list, in two command buffers:
    /// `x @ gate^T` and `x @ up^T` read the same rows and nothing of each other,
    /// so they are submitted together, and `silu(gate) * up` through `down`
    /// follows because it is what they produced.
    ///
    /// One bank on its own, which is what a caller with a single bank in hand
    /// wants — and what [`LayerExperts`] measures its own schedule against,
    /// since a layer that runs both banks answers what running each of them
    /// answers.
    pub fn forward(&self, gathered: Gathered<'_>) -> Result<Vec<f32>, MatmulError> {
        let chosen = chosen(gathered);
        let mut batch = self.device().batch()?;
        let glu = self.encode_glu(&mut batch, &chosen, gathered.rows())?;
        batch.wait()?;

        let mut batch = self.device().batch()?;
        let out = self.encode_down(&mut batch, &chosen, &activated(glu))?;
        batch.wait()?;
        Ok(out.take())
    }

    pub fn device(&self) -> &'a Device {
        self.gate_proj.device()
    }

    /// The two dispatches that read the gathered rows, encoded into `batch`.
    ///
    /// **A dispatch that reads what these rows were gathered from costs no
    /// submission at all.** `gate` and `up` read the hidden state and nothing
    /// this bank produces, so anything else with the same input can be encoded
    /// beside them — the router's gate is exactly that, and 225 microseconds a
    /// submission is what forty of them would otherwise cost a step.
    pub fn encode_glu(
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

    /// The third dispatch, over what the SwiGLU between them produced.
    ///
    /// **What this reads is all that ties it to `encode_glu`'s command buffer,
    /// and it is nothing another bank produces.** So a bank whose pair has
    /// already been waited for can have this encoded beside the *next* bank's
    /// pair, which is what a MoE layer's two banks are to each other.
    pub fn encode_down(
        &self,
        batch: &mut Batch<'_>,
        chosen: &[u32],
        activated: &[f32],
    ) -> Result<Pending, MatmulError> {
        self.down_proj.encode(batch, chosen, activated)
    }
}

/// `silu(gate) * up`, which is what the pair [`ExpertBanks::encode_glu`]
/// dispatched becomes before `down` reads it.
fn activated([gate, up]: [Pending; 2]) -> Vec<f32> {
    let (mut gate, up) = (gate.take(), up.take());
    swiglu(&mut gate, &up);
    gate
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

/// One MoE layer's two banks and the router that chooses between them.
///
/// The routed bank is 256 experts of which a token reads six and the shared bank
/// is two every token reads, and nothing else separates them — the same three
/// dispatches over the same gathered list, differing in how much of the bank the
/// list names.
///
/// **The gate is here because of what it can be dispatched beside.** It is not
/// an expert's weight at all — `[258, 4096]` of bfloat16 against 3.2 GiB of
/// packed banks — and it belongs to the layer rather than to either of them. But
/// its output weights what the shared bank produces without deciding what the
/// shared bank runs, so the one place it can be encoded for free is the command
/// buffer that bank already opens. Holding it anywhere else would mean a
/// submission of its own: 40 a step, and 9 ms of the 28 the gate costs.
#[derive(Debug)]
pub struct LayerExperts<'a> {
    routed: ExpertBanks<'a>,
    shared: ExpertBanks<'a>,
    gate: DenseWeight<'a>,
}

impl<'a> LayerExperts<'a> {
    /// The one device both banks and the gate were wrapped on, which is what
    /// lets a command buffer opened here hold dispatches against any of them.
    fn device(&self) -> &'a Device {
        self.shared.device()
    }

    /// [`Experts::banks`]'s three command buffers, with the errors a dispatch
    /// can fail with still in hand.
    ///
    /// **`route` sits between the first buffer and the second, and that
    /// placement is the schedule.** It turns the gate's logits — which the
    /// first buffer produced — into the rows the routed bank runs, so by the
    /// time the second buffer is opened everything that goes in it is in hand.
    fn encode(
        &self,
        x: &[f32],
        shared: Gathered<'_>,
        route: &mut dyn FnMut(Option<&[f32]>) -> Rows,
    ) -> Result<BankRows, MatmulError> {
        let shared_chosen = chosen(shared);

        let mut batch = self.device().batch()?;
        let logits = self.gate.encode(&mut batch, x)?;
        let shared_glu = self
            .shared
            .encode_glu(&mut batch, &shared_chosen, shared.rows())?;
        batch.wait()?;

        let shared_activated = activated(shared_glu);
        let routed_rows = route(Some(&logits.take()));
        let routed = routed_rows.gathered();
        let routed_chosen = chosen(routed);

        let mut batch = self.device().batch()?;
        let shared_out = self
            .shared
            .encode_down(&mut batch, &shared_chosen, &shared_activated)?;
        let routed_glu = self
            .routed
            .encode_glu(&mut batch, &routed_chosen, routed.rows())?;
        batch.wait()?;

        let routed_activated = activated(routed_glu);
        let mut batch = self.device().batch()?;
        let routed_out = self
            .routed
            .encode_down(&mut batch, &routed_chosen, &routed_activated)?;
        batch.wait()?;

        Ok(BankRows {
            routed: routed_out.take(),
            shared: shared_out.take(),
        })
    }

    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        dense: &'a DenseMatmul,
        banks: &LayerBanks<'a>,
        dim: usize,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            routed: ExpertBanks::wrap(device, matmul, &banks.routed, dim)?,
            shared: ExpertBanks::wrap(device, matmul, &banks.shared, dim)?,
            gate: DenseWeight::wrap(device, dense, &banks.gate_weight)?,
        })
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

    /// The whole layer — the gate, both banks, and the SwiGLU between each
    /// bank's halves — in three command buffers.
    ///
    /// **Seven dispatches in three submissions, where a layer whose banks run
    /// one after the other is seven in four.** Each bank alone takes two
    /// buffers, because `down` reads what `gate` and `up` produced and has to
    /// wait for them. The shared bank's `down` and the routed bank's `gate` and
    /// `up` are not in that relation to each other: `down` reads the shared
    /// halves this side already holds, and the routed pair reads rows gathered
    /// out of the hidden state under a routing taken from logits the *first*
    /// buffer produced. Neither waits for the other, so they go in one.
    fn banks(
        &self,
        x: &[f32],
        shared: Gathered<'_>,
        route: &mut dyn FnMut(Option<&[f32]>) -> Rows,
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

/// Every MoE layer's banks on the device, which for Inkling-Small is forty
/// layers of 3.2 GiB.
///
/// All of them, at load, and that is the residency policy stated: 137 GB
/// wrapped in about six milliseconds, holding nothing. The alternatives — copy
/// every layer at construction, or copy a layer the first time a token routes
/// into it — are both answers to a question about how much of 137 GB to move,
/// and the answer here is none of it.
#[derive(Debug)]
pub struct ModelExperts<'a> {
    /// Indexed by layer, `None` where the layer is dense. A map would be the
    /// same thing said less directly: the stack asks by index, forty-two times a
    /// forward pass.
    layers: Vec<Option<LayerExperts<'a>>>,
}

impl<'a> ModelExperts<'a> {
    /// Wrap every bank `banks` names, `dim` wide in and out.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        dense: &'a DenseMatmul,
        banks: &[LayerBanks<'a>],
        layers: usize,
        dim: usize,
    ) -> Result<Self, MatmulError> {
        let mut wrapped: Vec<Option<LayerExperts<'a>>> = (0..layers).map(|_| None).collect();
        for bank in banks {
            wrapped[bank.layer] = Some(LayerExperts::wrap(device, matmul, dense, bank, dim)?);
        }
        Ok(Self { layers: wrapped })
    }

    /// How many of the stack's layers have banks here, which is how many are
    /// MoE.
    pub fn layers(&self) -> usize {
        self.layers.iter().flatten().count()
    }
}

impl ExpertBackend for ModelExperts<'_> {
    fn layer(&self, layer: usize) -> Option<&dyn Experts> {
        Some(self.layers.get(layer)?.as_ref()? as &dyn Experts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::fixture::deviation;
    use inkling_core::moe::ExpertBank;
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

    /// Rows a synthetic router's gate has: the routed experts and the shared
    /// pair, which is the shape the layer's own `[258, 4096]` is.
    const GATE_ROWS: usize = EXPERTS + 2;

    /// What a call cost the device: `(dispatches, submissions)`, which is the
    /// pair the granularity question is settled in.
    fn spent<T>(device: &Device, run: impl FnOnce() -> T) -> (T, (u64, u64)) {
        let (dispatches, submissions) = (device.dispatches(), device.submissions());
        let got = run();
        (
            got,
            (
                device.dispatches() - dispatches,
                device.submissions() - submissions,
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
    #[test]
    fn three_banks_and_a_swiglu_are_the_expert_the_cpu_decodes() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let banks = Banks::new();
        let resident = banks
            .upload(&device, &matmul, &banks.gate, &banks.up)
            .expect("the three banks pair");
        assert_eq!(resident.experts(), EXPERTS);
        assert_eq!(resident.dim(), DIM);
        assert_eq!(resident.hidden_dim(), HIDDEN_DIM);

        let chosen = [2usize, 0, 2];
        let x = rows(chosen.len());
        let got = resident
            .forward(Gathered::new(DIM, &chosen, &x))
            .expect("the dispatches complete");
        assert_eq!(got.len(), chosen.len() * DIM);

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
        let banks = Banks::new();

        let chosen = [1usize];
        let x = rows(1);
        let through = |gate: &Case, up: &Case| {
            banks
                .upload(&device, &matmul, gate, up)
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

    /// The whole layer asked for at once is the same three answers as the gate,
    /// the shared bank and the routed bank asked for apart — **in the same seven
    /// dispatches and two fewer submissions.**
    ///
    /// Both halves are the commit. That the answers agree says the schedule
    /// changed no arithmetic; that the dispatch count does not move while the
    /// submission count does is the whole reason for scheduling at all, and it
    /// is the half a test of the values alone would let slip.
    ///
    /// The two saved round trips are two different merges. The gate rides in the
    /// buffer the shared bank's `gate` and `up` open, because it reads the same
    /// hidden state they do; and the shared bank's `down` rides beside the
    /// routed bank's `gate` and `up`, because by then the routing that names the
    /// routed rows has been taken from logits the first buffer produced. Over
    /// forty MoE layers that is 80 round trips a decode step does not take.
    #[test]
    fn the_whole_layer_costs_two_fewer_submissions_than_its_three_parts_apart() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let dense = DenseMatmul::new(&device).expect("the dense matmul compiles");
        let banks = Banks::new();
        let layer = LayerExperts {
            routed: banks
                .upload(&device, &matmul, &banks.gate, &banks.up)
                .expect("the three banks pair"),
            shared: banks
                .upload(&device, &matmul, &banks.gate, &banks.up)
                .expect("the three banks pair"),
            gate: gate(&device, &dense, 0x5eed_6666),
        };

        // Two tokens, and the shape `SparseMoe::forward` hands each bank: every
        // token once per shared expert, and one row per routed assignment,
        // expert-ascending.
        let x = rows(2);
        let shared_chosen = [0usize, 0, 1, 1];
        let shared_x = rows(shared_chosen.len());
        let shared = Gathered::new(DIM, &shared_chosen, &shared_x);
        let routed_chosen = [0usize, 1, 2];
        let routed_x = rows(routed_chosen.len());

        let mut asked = None;
        let (got, together) = spent(&device, || {
            layer.banks(&x, shared, &mut |logits| {
                asked = Some(logits.expect("the backend holds the gate").to_vec());
                Rows::new(DIM, routed_chosen.to_vec(), routed_x.clone())
            })
        });

        let ((logits, apart_rows), apart) = spent(&device, || {
            let logits = layer.gate.multiply(&x).expect("the dispatch completes");
            let shared = layer.shared(shared);
            let routed = layer.routed(Gathered::new(DIM, &routed_chosen, &routed_x));
            (logits, BankRows { routed, shared })
        });

        assert_eq!(got, apart_rows, "both banks' rows");
        assert_eq!(asked.expect("the layer was routed"), logits, "the logits");

        assert_eq!(together.0, apart.0, "the same dispatches");
        assert_eq!(together, (7, 3), "the layer's own cost");
        assert_eq!(apart, (7, 5), "what the three parts cost apart");
    }

    /// Which layer's banks answer for which layer, which is the one thing a
    /// model-wide backend can get wrong that a single layer's cannot.
    ///
    /// The dense layers have no banks and are `None` here rather than absent, so
    /// that the stack can ask by index — and a layer past the stack is `None`
    /// too rather than an index off the end.
    #[test]
    fn a_dense_layer_and_a_layer_past_the_stack_have_no_banks() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let dense = DenseMatmul::new(&device).expect("the dense matmul compiles");
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
                )
                .expect("the banks pair"),
                shared: ExpertBanks::new(
                    bank(&banks.gate, DIM, HIDDEN_DIM),
                    bank(&banks.up, DIM, HIDDEN_DIM),
                    bank(&banks.down, HIDDEN_DIM, DIM),
                )
                .expect("the banks pair"),
                gate: gate(&device, &dense, 0x5eed_5555),
            }
        };

        // A stack of four whose first two are dense, which is Inkling's shape.
        let mut layers: Vec<Option<LayerExperts<'_>>> = (0..4).map(|_| None).collect();
        layers[2] = Some(layer(&banks.gate));
        layers[3] = Some(layer(&banks.up));
        let experts = ModelExperts { layers };

        assert_eq!(experts.layers(), 2, "two of the four are MoE");
        assert!(experts.layer(0).is_none(), "a dense layer");
        assert!(experts.layer(1).is_none());
        assert!(experts.layer(2).is_some());
        assert!(experts.layer(4).is_none(), "past the stack");

        // And the two MoE layers are not each other's, which is what an index
        // off by one would produce.
        let chosen = [1usize];
        let x = rows(1);
        let of = |layer: usize| {
            experts
                .layer(layer)
                .expect("a MoE layer")
                .routed(Gathered::new(DIM, &chosen, &x))
        };
        assert_ne!(of(2), of(3));
    }
}
