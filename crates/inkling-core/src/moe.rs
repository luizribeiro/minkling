//! The sigmoid-gated mixture of experts every layer past the second is built
//! from: 256 routed experts of which each token reads six, plus two shared
//! experts every token reads.
//!
//! Four details of the gate decide the answer, and getting any of them wrong
//! leaves a model that still routes, still runs and still generates:
//!
//! - **The correction bias selects and does not weight.** `sigmoid(logits) +
//!   e_score_correction_bias` picks the top-k; the weights are a softmax over
//!   the *raw* logits of the experts it picked. Adding the bias twice moves the
//!   weights by a fifth, carrying the biased score into them by a half.
//! - **The weights are one softmax over the routed and the shared together.**
//!   The chosen routed logits and both shared logits are concatenated, put
//!   through `log(sigmoid(x))` and normalised across all eight at once.
//!   Normalising the six alone leaves the shared experts three times too heavy.
//! - **Two scales multiply the result**, `route_scale` from the config and a
//!   learned `global_scale` from the checkpoint. Layer 2's `global_scale` is
//!   0.00704, so a port applying `route_scale` alone runs 142x hot.
//! - **The shared experts are the last rows of `gate_weight`.** It is `[258,
//!   4096]` with the 256 routed rows first, and reading either end for the
//!   other still produces a distribution over 258 experts.
//!
//! The gate is pinned to mlx-vlm by `reference/fixtures/moe.safetensors`: the
//! trained `[258, 4096]` gate of the layer the activation capture covers, which
//! makes the whole trained routing computation reproducible without the
//! checkpoint, and synthetic float32 cases through the reference module itself.
//! The experts are `[256, 2048, 4096]`, 25 GB in float32, so the trained ones
//! are left to the checkpoint-gated tests in `tests/real_checkpoint.rs`.

use std::collections::BTreeMap;

use crate::config::TextConfig;
use crate::ops::{self, DenseMlp, linear, softmax};
use crate::profile::{self, Op};

/// The shapes and scalars `InklingSparseMoE.__init__` reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeConfig {
    pub n_routed: usize,
    pub n_shared: usize,
    /// Routed experts per token — `num_experts_per_tok`, six in Inkling-Small.
    pub top_k: usize,
    pub route_scale: f32,
}

impl MoeConfig {
    /// What one layer's MLP is. Layers below `dense_mlp_idx` build an
    /// `InklingDenseMLP` instead and have no router at all, so they have no
    /// config here either.
    pub fn for_layer(config: &TextConfig, layer: usize) -> Option<Self> {
        (!config.layer_is_dense(layer)).then_some(Self {
            n_routed: config.n_routed_experts,
            n_shared: config.n_shared_experts,
            top_k: config.num_experts_per_tok,
            route_scale: config.route_scale,
        })
    }
}

/// The gate's tensors, as the checkpoint stores them.
#[derive(Debug, Clone, Copy)]
pub struct GateWeights<'a> {
    /// Where the `[n_routed + n_shared, hidden]` projection multiplies.
    pub gate: Gate<'a>,
    /// `[n_routed]`, added to the sigmoid of the routed logits before the
    /// top-k, and nowhere else.
    pub correction_bias: &'a [f32],
    /// The layer's learned output scale, which multiplies every weight
    /// alongside `route_scale`.
    pub global_scale: f32,
}

/// Where the router's gate multiplies, which is the same seam
/// [`AttentionProjections`](crate::attention::AttentionProjections) and
/// [`DenseMlp`] are — over the one weight in the model that is not packed.
#[derive(Debug, Clone, Copy)]
pub enum Gate<'a> {
    /// `[n_routed + n_shared, hidden]` row-major and widened to float32, the
    /// routed experts first and the shared ones last. Multiplied here.
    Widened(&'a [f32]),
    /// Held by whatever runs the layer's experts, which answers with the logits
    /// beside the shared bank's rows — see
    /// [`Experts::gated_shared`](crate::layer::Experts::gated_shared).
    ///
    /// Only the width it maps from is carried, because that is the only thing
    /// about it this side still needs: a backend reading the checkpoint's own
    /// bfloat16 means the tensor is never widened at all, and the 4.2 MB of
    /// float32 a layer that would take is the point.
    Backend { hidden: usize },
}

/// Which experts a token reads and with what weight: `[tokens, top_k]` routed
/// experts and their weights, and `[tokens, n_shared]` shared weights.
///
/// The `top_k` of a token are held best-first, which is an order this port
/// chooses and not one the reference states — see [`SparseMoe::route`]. Nothing
/// downstream depends on it: [`Routing::routed_batches`] regroups by expert.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    top_k: usize,
    n_shared: usize,
    experts: Vec<usize>,
    weights: Vec<f32>,
    shared: Vec<f32>,
}

/// One expert and the work routed to it: the token rows that chose it, each
/// with the weight it gave, in ascending token order.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBatch {
    pub expert: usize,
    pub tokens: Vec<(usize, f32)>,
}

/// Every row one bank has to run, and the expert each row goes through:
/// `[rows, dim]` gathered out of the hidden state beside `[rows]` expert
/// indices.
///
/// **The whole bank's work in one value, which is what a gathered dispatch
/// needs.** A backend that decodes an expert wants them one at a time and
/// [`Gathered::batches`] is that; a backend that indexes 137 GB of packed banks
/// from a list wants them all at once and cannot get there from a call per
/// expert — 40 layers times 8 experts times 3 projections is 960 dispatches a
/// decode step, and at the 170 microseconds a dispatch costs to encode, commit
/// and wait for, that is 0.2 s of a step doing nothing.
///
/// Expert-ascending and grouped, which is what makes the second reading
/// possible without giving up the first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gathered<'a> {
    dim: usize,
    experts: &'a [usize],
    rows: &'a [f32],
}

impl<'a> Gathered<'a> {
    pub fn new(dim: usize, experts: &'a [usize], rows: &'a [f32]) -> Self {
        assert!(dim > 0, "a bank maps from some width");
        assert_eq!(
            rows.len(),
            experts.len() * dim,
            "{} values are not {} rows of {dim}",
            rows.len(),
            experts.len()
        );
        Self { dim, experts, rows }
    }

    /// The width a row is, which for every bank in the model is the hidden size.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }

    /// `[rows]`, the expert each row goes through.
    pub fn experts(&self) -> &'a [usize] {
        self.experts
    }

    /// `[rows, dim]`, the rows themselves.
    pub fn rows(&self) -> &'a [f32] {
        self.rows
    }

    /// The runs of one expert, for a backend that decodes an expert before it
    /// can multiply against it and cannot afford to decode one twice.
    ///
    /// A run and not a scan, because the rows arrive expert-ascending — so what
    /// this promises is that no expert appears in two of them.
    pub fn batches(&self) -> impl Iterator<Item = (usize, &'a [f32])> {
        let mut at = 0;
        std::iter::from_fn(move || {
            let expert = *self.experts.get(at)?;
            let run = self.experts[at..]
                .iter()
                .take_while(|next| **next == expert)
                .count();
            let rows = &self.rows[at * self.dim..][..run * self.dim];
            at += run;
            Some((expert, rows))
        })
    }
}

/// The two halves `InklingSparseMoE` adds together on the way out. They are
/// separate because the reference records them separately, and because only the
/// routed half needs the 256-expert bank.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeOutput {
    pub routed: Vec<f32>,
    pub shared: Vec<f32>,
}

impl MoeOutput {
    pub fn total(&self) -> Vec<f32> {
        let _timed = profile::scope(Op::Residual);
        assert_eq!(self.routed.len(), self.shared.len(), "halves");
        self.routed
            .iter()
            .zip(&self.shared)
            .map(|(routed, shared)| routed + shared)
            .collect()
    }
}

/// One bank of experts held in memory: `[experts, hidden_dim, dim]` gate and up
/// projections beside `[experts, dim, hidden_dim]` down projections, the layout
/// `SwitchLinear` stores.
///
/// Inkling's routed bank is 25 GB in float32 and cannot be held this way, which
/// is why [`SparseMoe::forward`] asks for experts through a function rather than
/// taking a bank. The shared pair fits, and so does any expert decoded on
/// demand.
#[derive(Debug, Clone, Copy)]
pub struct ExpertBank<'a> {
    dim: usize,
    hidden_dim: usize,
    gate_proj: &'a [f32],
    up_proj: &'a [f32],
    down_proj: &'a [f32],
}

impl<'a> ExpertBank<'a> {
    pub fn new(
        experts: usize,
        dim: usize,
        gate_proj: &'a [f32],
        up_proj: &'a [f32],
        down_proj: &'a [f32],
    ) -> Self {
        assert!(experts > 0, "a bank needs at least one expert");
        assert_eq!(
            gate_proj.len() % (experts * dim),
            0,
            "{} gate weights are not {experts} whole experts of width {dim}",
            gate_proj.len()
        );
        assert_eq!(up_proj.len(), gate_proj.len(), "up against gate");
        assert_eq!(down_proj.len(), gate_proj.len(), "down against gate");

        Self {
            dim,
            hidden_dim: gate_proj.len() / (experts * dim),
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// One expert as the SwiGLU MLP it is.
    ///
    /// Its scale is 1: an expert carries none of its own, and the layer's
    /// `global_scale` is already folded into the routing weights.
    pub fn expert(&self, index: usize) -> DenseMlp<'a> {
        let projection = self.hidden_dim * self.dim;
        let at = index * projection..(index + 1) * projection;
        DenseMlp::new(
            self.dim,
            &self.gate_proj[at.clone()],
            &self.up_proj[at.clone()],
            &self.down_proj[at],
            1.0,
        )
    }
}

/// One MoE layer, from the hidden state its post-attention layernorm produced
/// to the routed and shared halves of its output.
#[derive(Debug, Clone, Copy)]
pub struct SparseMoe<'a> {
    config: MoeConfig,
    hidden: usize,
    weights: GateWeights<'a>,
}

impl<'a> SparseMoe<'a> {
    pub fn new(config: MoeConfig, weights: GateWeights<'a>) -> Self {
        let experts = config.n_routed + config.n_shared;
        assert!(
            config.n_shared > 0,
            "a layer needs at least one shared expert"
        );
        assert!(
            (1..=config.n_routed).contains(&config.top_k),
            "{} experts per token do not come out of {}",
            config.top_k,
            config.n_routed
        );
        assert_eq!(
            weights.correction_bias.len(),
            config.n_routed,
            "one correction bias per routed expert"
        );

        let hidden = match weights.gate {
            Gate::Widened(weight) => {
                assert_eq!(
                    weight.len() % experts,
                    0,
                    "{} gate weights are not whole rows of {experts} experts",
                    weight.len()
                );
                weight.len() / experts
            }
            Gate::Backend { hidden } => hidden,
        };

        Self {
            config,
            hidden,
            weights,
        }
    }

    pub fn config(&self) -> MoeConfig {
        self.config
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// `[tokens, hidden]` in, `[tokens, n_routed + n_shared]` gate logits out.
    ///
    /// A panic where the gate is a backend's, and reachable only through a
    /// backend that claims to hold one and then answers with no logits — which
    /// is a contradiction rather than a case, since nothing here has the weight
    /// to fall back to.
    pub fn gate(&self, x: &[f32]) -> Vec<f32> {
        match self.weights.gate {
            Gate::Widened(weight) => linear(x, weight, self.hidden),
            Gate::Backend { .. } => {
                panic!("the gate is the experts' backend's, and it answered with no logits")
            }
        }
    }

    /// The routing one row of gate logits per token implies.
    ///
    /// Each token's experts come back best-first, ties going to the lower
    /// index. The set is the reference's; the order is not. `mx.argpartition`
    /// promises only that the k-th element lands where a sort would put it, and
    /// two MLX streams do return the k before it in two different orders for
    /// one input — so the order the fixture recorded belongs to the kernel that
    /// ran. Ties are the reference's too: over sixteen hundred rows of heavily
    /// tied scores, on both streams, `argpartition` selected exactly the set a
    /// stable descending sort selects.
    pub fn route(&self, logits: &[f32]) -> Routing {
        self.route_as(logits, Reading::REFERENCE)
    }

    /// `[tokens, hidden]` in, the routed and shared halves of `[tokens, hidden]`
    /// out.
    ///
    /// The experts are asked for rather than held: `routed` and `shared` are
    /// each called once, with every row their bank has to run and the expert
    /// each goes through, and return the `[rows, hidden]` those rows came out
    /// as. The gathering is here so that a bank of 256 experts is asked about
    /// only the six a token chose, and the grouping is here — see
    /// [`Gathered`] — so that a caller which decodes a 33 MB MXFP4 expert still
    /// decodes it a single time however many tokens routed to it.
    ///
    /// **The shared bank runs before the gate has answered, and that is what
    /// `shared` is asked two things at once for.** Which rows the shared bank
    /// has is not the router's to say — every token goes through every shared
    /// expert, always — so the routing decides only the weight each of those
    /// rows is scaled by on the way back. A backend that dispatches both can
    /// therefore put the gate's multiply in the same command buffer as the
    /// shared bank's, which at 206 microseconds a submission is the difference
    /// between moving the gate onto a device and paying for the privilege.
    ///
    /// It answers with `None` where it holds no gate, and then the layer's own
    /// weight is multiplied here — which is what the CPU path does and what
    /// every case below drives.
    pub fn forward(
        &self,
        x: &[f32],
        routed: impl FnOnce(Gathered<'_>) -> Vec<f32>,
        shared: impl FnOnce(&[f32], Gathered<'_>) -> (Option<Vec<f32>>, Vec<f32>),
    ) -> MoeOutput {
        let (experts, rows) = self.shared_rows(x);
        let (logits, answered) = shared(x, Gathered::new(self.hidden, &experts, &rows));
        let logits = logits.unwrap_or_else(|| self.gate(x));

        let routing = profile::timed(Op::Router, || self.route(&logits));
        MoeOutput {
            routed: self.combine(x, &routing.routed_batches(), routed),
            shared: self.scatter(x.len(), &routing.shared_batches(), &answered),
        }
    }

    /// The rows the shared bank has to run, which every token gives it whatever
    /// the gate says: `[n_shared * tokens, hidden]`, one shared expert after
    /// the other.
    ///
    /// The same rows [`SparseMoe::gather`] would produce from
    /// [`Routing::shared_batches`], in the same order, and formed without the
    /// routing — which is the whole of what lets the gate share a command
    /// buffer with the bank its weights will scale.
    fn shared_rows(&self, x: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let _timed = profile::scope(Op::Gather);
        assert_eq!(
            x.len() % self.hidden,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.hidden
        );

        let tokens = x.len() / self.hidden;
        let mut experts = Vec::with_capacity(self.config.n_shared * tokens);
        let mut rows = Vec::with_capacity(self.config.n_shared * x.len());
        for expert in 0..self.config.n_shared {
            experts.extend(std::iter::repeat_n(expert, tokens));
            rows.extend_from_slice(x);
        }
        (experts, rows)
    }

    /// [`SparseMoe::route`], with the ways of misreading the gate named apart.
    ///
    /// Each of them produces a distribution over experts that a running model
    /// cannot be read to disagree with, so the tests drive them from here to
    /// show that none reproduces mlx-vlm.
    fn route_as(&self, logits: &[f32], reading: Reading) -> Routing {
        let MoeConfig {
            n_routed,
            n_shared,
            top_k,
            route_scale,
        } = self.config;
        let width = n_routed + n_shared;
        assert_eq!(
            logits.len() % width,
            0,
            "{} logits are not whole rows of {width} experts",
            logits.len()
        );

        let scale = route_scale * self.weights.global_scale;
        let mut routing = Routing {
            top_k,
            n_shared,
            experts: Vec::new(),
            weights: Vec::new(),
            shared: Vec::new(),
        };
        let mut row = vec![0.0; top_k + n_shared];

        for logits in logits.chunks_exact(width) {
            let (routed, shared) = reading.split(logits, n_routed, n_shared);
            let scores: Vec<f32> = routed
                .iter()
                .zip(self.weights.correction_bias)
                .map(|(logit, bias)| sigmoid(*logit) + bias)
                .collect();
            let picked = ops::top_k(&scores, top_k);

            let (chosen, rest) = row.split_at_mut(top_k);
            for (slot, expert) in chosen.iter_mut().zip(&picked) {
                let weighted = reading.weighted_logit(
                    routed[*expert],
                    self.weights.correction_bias[*expert],
                    scores[*expert],
                );
                *slot = log_sigmoid(weighted);
            }
            for (slot, logit) in rest.iter_mut().zip(shared) {
                *slot = log_sigmoid(*logit);
            }
            reading.normalise(&mut row, top_k);

            routing.experts.extend(picked);
            routing
                .weights
                .extend(row[..top_k].iter().map(|w| w * scale));
            routing
                .shared
                .extend(row[top_k..].iter().map(|w| w * scale));
        }
        routing
    }

    /// Every expert's contribution to the tokens that routed to it, summed into
    /// `[tokens, hidden]`.
    fn combine(
        &self,
        x: &[f32],
        batches: &[ExpertBatch],
        apply: impl FnOnce(Gathered<'_>) -> Vec<f32>,
    ) -> Vec<f32> {
        let (experts, rows) = self.gather(x, batches);
        let got = apply(Gathered::new(self.hidden, &experts, &rows));
        self.scatter(x.len(), batches, &got)
    }

    /// The rows a set of batches names, gathered out of the hidden state:
    /// `[assignments, hidden]` beside the expert each row goes through.
    fn gather(&self, x: &[f32], batches: &[ExpertBatch]) -> (Vec<usize>, Vec<f32>) {
        let _timed = profile::scope(Op::Gather);
        assert_eq!(
            x.len() % self.hidden,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.hidden
        );

        let assignments = batches.iter().map(|batch| batch.tokens.len()).sum();
        let mut experts = Vec::with_capacity(assignments);
        let mut rows = Vec::with_capacity(assignments * self.hidden);
        for batch in batches {
            for (token, _) in &batch.tokens {
                experts.push(batch.expert);
                rows.extend_from_slice(&x[token * self.hidden..][..self.hidden]);
            }
        }
        (experts, rows)
    }

    /// What a bank answered, weighted and summed back into `[tokens, hidden]`.
    ///
    /// The other half of [`SparseMoe::gather`], and separate from it because
    /// only this half needs the routing: `got` is one row per assignment in the
    /// order the batches name them, and what the routing supplies is the weight
    /// each carries.
    fn scatter(&self, len: usize, batches: &[ExpertBatch], got: &[f32]) -> Vec<f32> {
        let _timed = profile::scope(Op::Gather);
        let assignments: usize = batches.iter().map(|batch| batch.tokens.len()).sum();
        assert_eq!(
            got.len(),
            assignments * self.hidden,
            "the bank answered {} values for {assignments} rows",
            got.len(),
        );

        let mut out = vec![0.0; len];
        let mut answered = got.chunks_exact(self.hidden);
        for batch in batches {
            for (token, weight) in &batch.tokens {
                let y = answered.next().expect("a row per assignment");
                for (out, y) in out[token * self.hidden..][..self.hidden].iter_mut().zip(y) {
                    *out += weight * y;
                }
            }
        }
        out
    }
}

impl Routing {
    pub fn tokens(&self) -> usize {
        self.experts.len() / self.top_k
    }

    /// `[tokens, top_k]` selected routed experts.
    pub fn experts(&self) -> &[usize] {
        &self.experts
    }

    /// `[tokens, top_k]` weights, aligned with [`Routing::experts`].
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// `[tokens, n_shared]` weights for the always-on experts.
    pub fn shared_weights(&self) -> &[f32] {
        &self.shared
    }

    /// The routed work, expert-ascending, skipping experts no token chose.
    ///
    /// Regrouping here rather than at the call site is what makes the answer
    /// independent of the order [`SparseMoe::route`] returned each token's
    /// experts in — the one thing about the reference's selection that cannot
    /// be pinned.
    pub fn routed_batches(&self) -> Vec<ExpertBatch> {
        let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
        let rows = self.experts.chunks_exact(self.top_k);
        for (token, (experts, weights)) in
            rows.zip(self.weights.chunks_exact(self.top_k)).enumerate()
        {
            for (expert, weight) in experts.iter().zip(weights) {
                by_expert.entry(*expert).or_default().push((token, *weight));
            }
        }
        by_expert
            .into_iter()
            .map(|(expert, tokens)| ExpertBatch { expert, tokens })
            .collect()
    }

    /// The shared work: every shared expert, every token, always.
    pub fn shared_batches(&self) -> Vec<ExpertBatch> {
        (0..self.n_shared)
            .map(|expert| ExpertBatch {
                expert,
                tokens: self
                    .shared
                    .chunks_exact(self.n_shared)
                    .enumerate()
                    .map(|(token, weights)| (token, weights[expert]))
                    .collect(),
            })
            .collect()
    }
}

/// How a port reads the gate.
///
/// Only [`Reading::REFERENCE`] is ever constructed outside the tests below, and
/// that is the point of the type: every other value is a misreading that still
/// selects experts and still weights them, so the tests drive each one from
/// here to show that none of them reproduces mlx-vlm. The `dead_code`
/// allowances below say exactly that — the engine has one reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reading {
    weighted_by: WeightedBy,
    normalise: Normalise,
    shared_rows: SharedRows,
}

/// What the weight of a chosen routed expert is computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum WeightedBy {
    /// The raw gate logit, which is what `InklingSparseMoE` gathers.
    RawLogit,
    /// The logit plus the correction bias — the bias counted twice, once in the
    /// selection and once here.
    LogitPlusBias,
    /// The biased score the selection ranked on, carried through whole.
    BiasedScore,
}

/// What one softmax spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Normalise {
    /// One softmax over the chosen routed experts and the shared ones together.
    Jointly,
    /// The routed experts normalised among themselves and the shared ones among
    /// themselves, which leaves each group summing to the full scale.
    Separately,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum SharedRows {
    /// The last `n_shared` rows of `gate_weight`, which is where they are.
    Last,
    /// The first `n_shared` rows, which are routed experts 0 and 1.
    First,
}

impl Reading {
    const REFERENCE: Self = Self {
        weighted_by: WeightedBy::RawLogit,
        normalise: Normalise::Jointly,
        shared_rows: SharedRows::Last,
    };

    /// One token's gate logits as its routed and shared halves.
    fn split<'a>(
        &self,
        logits: &'a [f32],
        n_routed: usize,
        n_shared: usize,
    ) -> (&'a [f32], &'a [f32]) {
        match self.shared_rows {
            SharedRows::Last => (&logits[..n_routed], &logits[n_routed..]),
            SharedRows::First => (&logits[n_shared..], &logits[..n_shared]),
        }
    }

    fn weighted_logit(&self, logit: f32, bias: f32, score: f32) -> f32 {
        match self.weighted_by {
            WeightedBy::RawLogit => logit,
            WeightedBy::LogitPlusBias => logit + bias,
            WeightedBy::BiasedScore => score,
        }
    }

    fn normalise(&self, row: &mut [f32], top_k: usize) {
        match self.normalise {
            Normalise::Jointly => softmax(row),
            Normalise::Separately => {
                let (routed, shared) = row.split_at_mut(top_k);
                softmax(routed);
                softmax(shared);
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `log(sigmoid(x))`, which mlx-vlm writes as `-logaddexp(0, -x)`.
///
/// Shifted for the same reason `logaddexp` is: `exp(-x)` overflows below about
/// -88 in float32, where the answer is simply `x`, and the naive form would
/// return an infinity through a softmax that has no way back from one.
fn log_sigmoid(x: f32) -> f32 {
    x.min(0.0) - (-x.abs()).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{self, ACTIVATIONS, Bank, deviation, indices};

    /// Synthetic routers and the sequences mlx-vlm ran through them, beside the
    /// trained gate of the captured MoE layer, from `just dump-moe-fixture`.
    const FIXTURE: &str = "moe.safetensors";

    const CASES: [&str; 2] = ["main", "tie"];

    /// The worst disagreement across a routing's weights, routed and shared
    /// together, which is where every misreading of the gate lands.
    fn weight_deviation(routing: &Routing, topk_w: &[f32], shared_gammas: &[f32]) -> f32 {
        deviation(routing.weights(), topk_w).max(deviation(routing.shared_weights(), shared_gammas))
    }

    /// The synthetic cases are float32 end to end, so only summation order
    /// separates this from MLX — the same bound, for the same reason, as the
    /// RMSNorm, MLP, sconv, mask and attention cases. The gate is one
    /// projection and a softmax over eight numbers, and each expert is one
    /// SwiGLU MLP, so nothing here reduces over more than the hidden width.
    /// Worst observed when this landed: 3.9e-7 across both cases and every
    /// recorded tensor, a factor of two in hand. The weakest mutation these
    /// tests rely on catching — the correction bias added into the weights as
    /// well as into the selection — moves the answer by 1.2e-1, five decades
    /// above this bound.
    const TOLERANCE: f32 = 1e-6;

    /// The trained routing recomputed from the recorded gate logits, which are
    /// the reference's own bfloat16 answers widened. Everything after them is
    /// float32 in both implementations, so this is an arithmetic bound like the
    /// synthetic one rather than a dtype one. Worst observed: 2.3e-7.
    const TRAINED_TOLERANCE: f32 = 1e-6;

    /// The trained routing recomputed from the hidden state through the
    /// committed gate. mlx-vlm forms those logits in bfloat16 and this port in
    /// float32 over a 4096-wide reduction, so they part company by about a
    /// thousandth and everything downstream inherits it — a dtype's gap, and
    /// the same bound, for the same reason, as the trained attention and masks.
    /// Worst observed: 1.5e-3 on the shared weights.
    const GATE_TOLERANCE: f32 = 6e-3;

    /// One synthetic case: the router mlx-vlm was built as, its two expert
    /// banks, and everything the tap saw.
    struct Case {
        name: String,
        config: MoeConfig,
        hidden: usize,
        gate_weight: Vec<f32>,
        correction_bias: Vec<f32>,
        global_scale: f32,
        routed: Bank,
        shared: Bank,
        x: Vec<f32>,
        logits: Vec<f32>,
        scores: Vec<f32>,
        topk_idx: Vec<usize>,
        topk_w: Vec<f32>,
        shared_gammas: Vec<f32>,
        routed_out: Vec<f32>,
        shared_out: Vec<f32>,
        out: Vec<f32>,
    }

    impl Case {
        fn load(case: &str) -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of = |name: &str| fixture::f32s(&fixture::tensor(&ckpt, &format!("{case}.{name}")));
            let recorded = of("config");
            let &[n_routed, n_shared, top_k, route_scale] = recorded.as_slice() else {
                panic!("{case}: config carries four scalars, got {recorded:?}")
            };
            let config = MoeConfig {
                n_routed: n_routed as usize,
                n_shared: n_shared as usize,
                top_k: top_k as usize,
                route_scale,
            };

            let gate_weight = of("gate_weight");
            let hidden = gate_weight.len() / (config.n_routed + config.n_shared);
            let bank = |module: &str, experts| {
                Bank::load(&ckpt, &format!("{case}.{module}"), experts, hidden)
            };
            Self {
                name: case.to_string(),
                hidden,
                config,
                gate_weight,
                correction_bias: of("e_score_correction_bias"),
                global_scale: of("global_scale")[0],
                routed: bank("switch_mlp", config.n_routed),
                shared: bank("shared_experts", config.n_shared),
                x: fixture::f32s(&fixture::tensor(&ckpt, "x")),
                logits: of("gate_logits"),
                scores: of("gate_scores_biased"),
                topk_idx: indices(&fixture::tensor(&ckpt, &format!("{case}.topk_idx"))),
                topk_w: of("topk_w"),
                shared_gammas: of("shared_gammas"),
                routed_out: of("routed_out"),
                shared_out: of("shared_out"),
                out: of("out"),
            }
        }

        fn all() -> Vec<Self> {
            CASES.iter().map(|case| Self::load(case)).collect()
        }

        fn moe(&self) -> SparseMoe<'_> {
            self.with(GateWeights {
                gate: Gate::Widened(&self.gate_weight),
                correction_bias: &self.correction_bias,
                global_scale: self.global_scale,
            })
        }

        fn with<'a>(&self, weights: GateWeights<'a>) -> SparseMoe<'a> {
            SparseMoe::new(self.config, weights)
        }

        fn routing(&self) -> Routing {
            self.moe().route(&self.logits)
        }

        fn forward(&self) -> MoeOutput {
            self.forward_gated(None)
        }

        /// The same layer with the gate answered for by whoever runs the shared
        /// bank, which is what a backend holding the gate on a device does.
        fn forward_gated(&self, logits: Option<Vec<f32>>) -> MoeOutput {
            self.moe().forward(
                &self.x,
                |gathered| self.routed.gathered(gathered),
                |_, gathered| (logits, self.shared.gathered(gathered)),
            )
        }

        fn weight_deviation(&self, routing: &Routing) -> f32 {
            weight_deviation(routing, &self.topk_w, &self.shared_gammas)
        }

        fn read_as(&self, reading: Reading) -> Routing {
            self.moe().route_as(&self.logits, reading)
        }
    }

    /// Held to the recorded order and not only to the recorded set, which is
    /// more than the reference promises: `argpartition` returns the top-k
    /// unordered, and the fixture's order is the Metal kernel's. It is also
    /// this port's order, so the two agree — but a fixture regenerated on a
    /// backend that partitions differently would fail here, and that would be a
    /// dump to re-read rather than a routing bug.
    #[test]
    fn the_synthetic_routers_reproduce_mlx() {
        for case in Case::all() {
            let routing = case.routing();
            assert_eq!(routing.experts(), case.topk_idx, "{}: experts", case.name);

            let deviation = case.weight_deviation(&routing);
            assert!(
                deviation <= TOLERANCE,
                "{}: weights {deviation:e}",
                case.name
            );
        }
    }

    #[test]
    fn the_synthetic_layers_reproduce_mlx() {
        for case in Case::all() {
            let got = case.forward();
            for (what, got, want) in [
                ("routed", &got.routed, &case.routed_out),
                ("shared", &got.shared, &case.shared_out),
                ("total", &got.total(), &case.out),
            ] {
                let deviation = deviation(got, want);
                assert!(
                    deviation <= TOLERANCE,
                    "{}: {what} deviation {deviation:e}",
                    case.name
                );
            }
        }
    }

    /// The rows the shared bank is given before the gate has answered are the
    /// rows the routing would have gathered for it.
    ///
    /// **This is what the whole reordering rests on.** The shared bank's work
    /// is formed from `x` and `n_shared` alone, so it can be asked for in the
    /// same breath as the gate — and if that ever stopped agreeing with what
    /// `shared_batches` names, every shared expert's output would be scattered
    /// back under the wrong token's weight while remaining a tensor of exactly
    /// the right shape.
    #[test]
    fn the_shared_rows_are_the_ones_the_routing_would_have_gathered() {
        for case in Case::all() {
            let moe = case.moe();
            let (experts, rows) = moe.shared_rows(&case.x);
            let (want_experts, want_rows) = moe.gather(&case.x, &case.routing().shared_batches());

            assert_eq!(
                experts, want_experts,
                "{}: the expert of each row",
                case.name
            );
            assert_eq!(rows, want_rows, "{}: the rows", case.name);
            assert!(
                experts.len() > case.config.n_shared,
                "{}: a single token would not order anything",
                case.name
            );
        }
    }

    /// The logits a backend answers with are the ones the routing is taken
    /// from, and the layer's own gate is not multiplied at all.
    ///
    /// Both halves are needed. Handed what the layer would have computed the
    /// answer is unchanged, so the seam moves nothing on its own; handed
    /// anything else the routing follows it, so a path that quietly ignored the
    /// backend and multiplied here would fail the second — while producing a
    /// perfectly good MoE layer.
    #[test]
    fn the_gate_a_backend_answers_with_is_the_one_the_routing_reads() {
        let case = Case::load("main");
        let mine = case.moe().gate(&case.x);
        assert_eq!(case.forward_gated(Some(mine.clone())), case.forward());

        // The same logits with each token's row reversed, which is a
        // distribution over the same experts and a different routing.
        let width = case.config.n_routed + case.config.n_shared;
        let elsewhere: Vec<f32> = mine
            .chunks_exact(width)
            .flat_map(|row| row.iter().rev().copied())
            .collect();
        assert_ne!(case.forward_gated(Some(elsewhere)), case.forward());
    }

    /// A gate a backend holds is a width and no values, and a layer stood up
    /// that way routes from what the backend answered.
    ///
    /// **The width is the whole of what is left here, and that is the point.**
    /// A `[258, 4096]` gate is 4.2 MB of float32 a layer and 169 MB over the
    /// stack, so a backend multiplying against the checkpoint's own bfloat16
    /// means nothing widens it — but the layer still has to know how wide a row
    /// of its input is, and that is the one thing this carries.
    #[test]
    fn a_layer_whose_gate_a_backend_holds_routes_from_the_logits_it_answered() {
        let case = Case::load("main");
        let logits = case.moe().gate(&case.x);
        let moe = SparseMoe::new(
            case.config,
            GateWeights {
                gate: Gate::Backend {
                    hidden: case.hidden,
                },
                ..case.moe().weights
            },
        );

        assert_eq!(moe.hidden(), case.hidden, "the width comes off the gate");
        let got = moe.forward(
            &case.x,
            |gathered| case.routed.gathered(gathered),
            |_, gathered| (Some(logits), case.shared.gathered(gathered)),
        );
        assert_eq!(got, case.forward());
    }

    /// The contradiction that arm admits: a backend that says it holds the gate
    /// and then answers with no logits. There is no weight on this side to fall
    /// back to — not widening it is the whole reason the arm exists — so it is
    /// a panic rather than a quiet second reading of the router.
    #[test]
    #[should_panic(expected = "the gate is the experts' backend's")]
    fn a_backend_that_holds_the_gate_and_answers_with_nothing_is_refused() {
        let case = Case::load("main");
        SparseMoe::new(
            case.config,
            GateWeights {
                gate: Gate::Backend {
                    hidden: case.hidden,
                },
                ..case.moe().weights
            },
        )
        .forward(
            &case.x,
            |gathered| case.routed.gathered(gathered),
            |_, gathered| (None, case.shared.gathered(gathered)),
        );
    }

    /// The first trap. `e_score_correction_bias` ranks the experts and then
    /// takes no further part: the weights come from the raw logits of whatever
    /// it ranked highest. Both ways of letting it through — adding it to the
    /// logit, or carrying the whole biased score across — leave a router that
    /// picks the same experts and weights them wrong.
    #[test]
    fn the_correction_bias_selects_the_experts_and_does_not_weight_them() {
        let case = Case::load("main");
        for weighted_by in [WeightedBy::LogitPlusBias, WeightedBy::BiasedScore] {
            let leaked = case.read_as(Reading {
                weighted_by,
                ..Reading::REFERENCE
            });
            assert_eq!(
                leaked.experts(),
                case.topk_idx,
                "{weighted_by:?}: selection"
            );

            let deviation = case.weight_deviation(&leaked);
            assert!(deviation > TOLERANCE, "{weighted_by:?}: {deviation:e}");
        }
    }

    /// The other half of that trap: the bias is not decoration. Dropping it
    /// selects different experts, so a port that never loaded it is reading a
    /// different sixth of the model.
    #[test]
    fn dropping_the_correction_bias_selects_different_experts() {
        let case = Case::load("main");
        let unbiased = vec![0.0; case.config.n_routed];
        let routing = case
            .with(GateWeights {
                correction_bias: &unbiased,
                ..case.moe().weights
            })
            .route(&case.logits);

        assert_ne!(routing.experts(), case.topk_idx);
    }

    /// The second trap. One softmax spans the chosen routed experts and both
    /// shared ones, so the eight weights sum to `route_scale * global_scale`
    /// together. Normalising the two groups apart leaves each summing to that,
    /// which is a router that still runs and reads the shared experts far too
    /// heavily.
    #[test]
    fn the_weights_are_one_softmax_over_the_routed_and_the_shared_together() {
        for case in Case::all() {
            let scale = case.config.route_scale * case.global_scale;
            let routing = case.routing();
            for token in 0..routing.tokens() {
                let total: f32 = routing.weights()[token * case.config.top_k..]
                    [..case.config.top_k]
                    .iter()
                    .chain(
                        &routing.shared_weights()[token * case.config.n_shared..]
                            [..case.config.n_shared],
                    )
                    .sum();
                assert!(
                    (total - scale).abs() <= TOLERANCE * scale,
                    "{}: token {token} sums to {total}, not {scale}",
                    case.name
                );
            }

            let apart = case.read_as(Reading {
                normalise: Normalise::Separately,
                ..Reading::REFERENCE
            });
            let deviation = case.weight_deviation(&apart);
            assert!(deviation > TOLERANCE, "{}: {deviation:e}", case.name);
        }
    }

    /// The third trap, at synthetic scale; the trained scale is two decades
    /// worse and has a test of its own.
    #[test]
    fn dropping_either_scale_changes_the_answer() {
        let case = Case::load("main");
        assert_ne!(case.global_scale, 1.0, "a scale of 1 would prove nothing");
        assert_ne!(case.config.route_scale, 1.0);

        for (what, moe) in [
            (
                "global_scale",
                case.with(GateWeights {
                    global_scale: 1.0,
                    ..case.moe().weights
                }),
            ),
            (
                "route_scale",
                SparseMoe::new(
                    MoeConfig {
                        route_scale: 1.0,
                        ..case.config
                    },
                    case.moe().weights,
                ),
            ),
        ] {
            let deviation = case.weight_deviation(&moe.route(&case.logits));
            assert!(deviation > TOLERANCE, "{what}: deviation {deviation:e}");
        }
    }

    /// The fourth trap. `gate_weight` is `[n_routed + n_shared, hidden]` with
    /// the shared experts last. Read from the front they are still two rows of
    /// the right width, the routed experts are still `n_routed` rows, and every
    /// token still routes.
    #[test]
    fn the_shared_experts_are_the_last_rows_of_the_gate() {
        let case = Case::load("main");
        let flipped = case.read_as(Reading {
            shared_rows: SharedRows::First,
            ..Reading::REFERENCE
        });

        assert_ne!(flipped.experts(), case.topk_idx, "selection");
        assert!(case.weight_deviation(&flipped) > TOLERANCE);
    }

    /// What the reference does when two experts tie at the top-k boundary.
    ///
    /// `mx.argpartition` states nothing about it, so the answer is empirical:
    /// over sixteen hundred rows of heavily tied scores, on both the Metal and
    /// the CPU stream, it selects exactly the set a stable descending sort
    /// selects — the lower index wins. The `tie` case is that claim at its
    /// sharpest: two experts share a gate row and a bias, so their scores agree
    /// bit for bit, and their rank straddles the last slot, so exactly one of
    /// them can be selected.
    ///
    /// What cannot be pinned is the *order* the k come back in, which differs
    /// between MLX's own two streams. Nothing downstream reads it.
    #[test]
    fn a_tie_at_the_last_slot_goes_to_the_lower_index() {
        let case = Case::load("tie");
        let top_k = case.config.top_k;

        // The pair that ties, as the recorded biased scores show them.
        let tied: Vec<usize> = (0..case.config.n_routed)
            .filter(|expert| case.scores[*expert] == case.scores[case.topk_idx[top_k - 1]])
            .collect();
        assert_eq!(tied.len(), 2, "the case has no tie: {tied:?}");
        assert_eq!(
            case.topk_idx[top_k - 1],
            tied[0],
            "the higher index took the slot"
        );

        assert_eq!(case.routing().experts(), case.topk_idx);
    }

    /// Which order a token's experts come back in cannot be the reference's, so
    /// nothing downstream may depend on it. Regrouping by expert is what makes
    /// that true: a routing whose slots are reversed batches identically, and
    /// so combines to the same bits.
    #[test]
    fn the_combination_does_not_depend_on_the_order_within_a_token() {
        let case = Case::load("main");
        let routing = case.routing();

        let mut reversed = routing.clone();
        for token in 0..reversed.tokens() {
            reversed.experts[token * routing.top_k..][..routing.top_k].reverse();
            reversed.weights[token * routing.top_k..][..routing.top_k].reverse();
        }

        assert_ne!(reversed.experts(), routing.experts());
        assert_eq!(reversed.routed_batches(), routing.routed_batches());
    }

    /// Every expert with work appears in exactly one run of the gathered rows,
    /// whatever number of tokens routed to it. A caller that decodes a 33 MB
    /// MXFP4 expert per run cannot afford otherwise, and the 256-expert bank
    /// cannot be materialised to avoid the question.
    ///
    /// The runs are the whole of what a gathered call promises beyond the flat
    /// list, so this is where that promise is stated: expert-ascending, once
    /// each, and every assignment inside one of them.
    #[test]
    fn every_expert_appears_in_one_run_with_all_of_its_tokens() {
        let case = Case::load("main");
        let (mut asked, mut rows) = (Vec::new(), 0);

        let out = case.moe().forward(
            &case.x,
            |gathered| {
                for (expert, batch) in gathered.batches() {
                    asked.push(expert);
                    rows += batch.len() / case.hidden;
                }
                case.routed.gathered(gathered)
            },
            |_, gathered| (None, case.shared.gathered(gathered)),
        );

        let mut distinct = case.topk_idx.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(asked, distinct, "in expert order, once each");
        assert_eq!(rows, case.topk_idx.len(), "every assignment was served");
        assert!(
            asked.len() < rows,
            "a case where no expert repeats would prove nothing"
        );
        assert!(out.routed.iter().any(|y| *y != 0.0));
    }

    /// The flat list beside the runs: a backend that indexes a bank rather than
    /// decoding it reads `experts()` and `rows()` and never asks for a batch, so
    /// the two readings have to describe the same work.
    #[test]
    fn the_gathered_rows_are_the_runs_laid_end_to_end() {
        let case = Case::load("main");
        let mut seen = None;
        case.moe().forward(
            &case.x,
            |gathered| {
                let runs: Vec<usize> = gathered
                    .batches()
                    .flat_map(|(expert, rows)| {
                        std::iter::repeat_n(expert, rows.len() / gathered.dim())
                    })
                    .collect();
                assert_eq!(runs, gathered.experts(), "the runs against the flat list");

                let concatenated: Vec<f32> = gathered
                    .batches()
                    .flat_map(|(_, rows)| rows.iter().copied())
                    .collect();
                assert_eq!(concatenated, gathered.rows(), "the runs against the rows");

                seen = Some(gathered.len());
                case.routed.gathered(gathered)
            },
            |_, gathered| (None, case.shared.gathered(gathered)),
        );

        assert_eq!(seen, Some(case.topk_idx.len()), "one row per assignment");
    }

    /// The two ways a gather can be malformed, which are the two things
    /// [`Gathered`] promises a backend: that a row is `dim` wide, and that there
    /// is one of them per expert index. A backend indexes the pair against each
    /// other and would read whichever ran out first.
    #[test]
    #[should_panic(expected = "12 values are not 2 rows of 4")]
    fn a_gather_whose_rows_do_not_pair_with_its_experts_is_refused() {
        Gathered::new(4, &[0, 1], &[0.0; 12]);
    }

    #[test]
    #[should_panic(expected = "a bank maps from some width")]
    fn a_gather_of_no_width_is_refused() {
        Gathered::new(0, &[], &[]);
    }

    /// The trained gate of the captured MoE layer, and the routing mlx-vlm
    /// recorded for it.
    struct Trained {
        layer: usize,
        config: MoeConfig,
        gate_weight: Vec<f32>,
        correction_bias: Vec<f32>,
        global_scale: f32,
        x: Vec<f32>,
        logits: Vec<f32>,
        scores: Vec<f32>,
        topk_idx: Vec<usize>,
        topk_w: Vec<f32>,
        shared_gammas: Vec<f32>,
    }

    impl Trained {
        fn load() -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of =
                |name: &str| fixture::f32s(&fixture::tensor(&ckpt, &format!("trained.{name}")));
            let shape = of("config");
            let &[n_routed, n_shared, top_k, route_scale] = shape.as_slice() else {
                panic!("trained config carries four scalars, got {shape:?}")
            };
            let layer = of("layer")[0] as usize;

            // The gate is a weight and keeps the checkpoint's own bfloat16;
            // everything the reference computed from it was widened on capture.
            let gate_weight = fixture::tensor(&ckpt, "trained.gate_weight")
                .to_f32()
                .expect("the gate widens");

            let activations = fixture::open(ACTIVATIONS);
            let recorded =
                |name: &str| fixture::f32s(&fixture::layer_tensor(&activations, layer, name));
            Self {
                layer,
                config: MoeConfig {
                    n_routed: n_routed as usize,
                    n_shared: n_shared as usize,
                    top_k: top_k as usize,
                    route_scale,
                },
                gate_weight,
                correction_bias: of("e_score_correction_bias"),
                global_scale: of("global_scale")[0],
                x: recorded("post_attention_ln_out"),
                logits: recorded("gate_logits"),
                scores: recorded("gate_scores_biased"),
                topk_idx: indices(&fixture::layer_tensor(&activations, layer, "topk_idx")),
                topk_w: recorded("topk_w"),
                shared_gammas: recorded("shared_gammas"),
            }
        }

        fn moe(&self) -> SparseMoe<'_> {
            SparseMoe::new(
                self.config,
                GateWeights {
                    gate: Gate::Widened(&self.gate_weight),
                    correction_bias: &self.correction_bias,
                    global_scale: self.global_scale,
                },
            )
        }

        fn weight_deviation(&self, routing: &Routing) -> f32 {
            weight_deviation(routing, &self.topk_w, &self.shared_gammas)
        }
    }

    /// The whole trained routing computation, without the checkpoint: the
    /// reference's own gate logits in, its recorded selection and weights out.
    #[test]
    fn the_trained_router_reproduces_the_recorded_routing() {
        let trained = Trained::load();
        let routing = trained.moe().route(&trained.logits);

        assert_eq!(routing.experts(), trained.topk_idx);
        let deviation = trained.weight_deviation(&routing);
        assert!(deviation <= TRAINED_TOLERANCE, "deviation {deviation:e}");
    }

    /// The gate projection and the row layout, which the recorded logits skip
    /// over. mlx-vlm forms them in bfloat16 and this port in float32, so the
    /// selection survives only because the trained scores are not close at the
    /// boundary — which is worth stating, since it is a property of these eight
    /// tokens and not a guarantee.
    #[test]
    fn the_committed_gate_reproduces_the_recorded_routing_from_the_hidden_state() {
        let trained = Trained::load();
        let moe = trained.moe();
        let routing = moe.route(&moe.gate(&trained.x));

        assert_eq!(
            routing.experts(),
            trained.topk_idx,
            "layer {}",
            trained.layer
        );
        let deviation = trained.weight_deviation(&routing);
        assert!(deviation <= GATE_TOLERANCE, "deviation {deviation:e}");
    }

    /// Why the test above can demand an exact selection at all. The gap between
    /// the last selected score and the first rejected one has to stay clear of
    /// the drift a float32 gate introduces, and it does, by a factor of about
    /// five. A regenerated capture that lost that margin would make the
    /// selection a coin toss, and this says so rather than letting the other
    /// test fail mysteriously.
    #[test]
    fn the_trained_selection_clears_the_gates_float32_drift() {
        let trained = Trained::load();
        let MoeConfig {
            n_routed,
            n_shared,
            top_k,
            ..
        } = trained.config;
        let logits = trained.moe().gate(&trained.x);

        let (mut drift, mut margin) = (0.0f32, f32::INFINITY);
        for (row, want) in logits
            .chunks_exact(n_routed + n_shared)
            .zip(trained.scores.chunks_exact(n_routed))
        {
            let mut scores: Vec<f32> = row[..n_routed]
                .iter()
                .zip(&trained.correction_bias)
                .map(|(logit, bias)| sigmoid(*logit) + bias)
                .collect();
            drift = scores
                .iter()
                .zip(want)
                .fold(drift, |worst, (got, want)| worst.max((got - want).abs()));

            scores.sort_unstable_by(|a, b| b.total_cmp(a));
            margin = margin.min(scores[top_k - 1] - scores[top_k]);
        }

        assert!(
            margin > 4.0 * drift,
            "margin {margin:e} against a drift of {drift:e}"
        );
    }

    /// The third trap at trained scale. `global_scale` is 0.00704 and
    /// `route_scale` is 8, so their product is 0.0563 — the sum every recorded
    /// weight row lands on. A port that applied `route_scale` alone would run
    /// every expert 142 times too hot, which is the largest single error any of
    /// these traps can produce and the one worth naming.
    #[test]
    fn dropping_the_trained_global_scale_is_a_hundred_and_forty_twofold_error() {
        let trained = Trained::load();
        assert!(
            (trained.global_scale - 0.007_042_4).abs() < 1e-7,
            "global_scale is {}",
            trained.global_scale
        );

        let scale = trained.config.route_scale * trained.global_scale;
        for token in 0..trained.topk_w.len() / trained.config.top_k {
            let total: f32 = trained.topk_w[token * trained.config.top_k..][..trained.config.top_k]
                .iter()
                .chain(
                    &trained.shared_gammas[token * trained.config.n_shared..]
                        [..trained.config.n_shared],
                )
                .sum();
            assert!(
                (total - scale).abs() <= 1e-6,
                "token {token} sums to {total}"
            );
        }

        let unscaled = SparseMoe::new(
            trained.config,
            GateWeights {
                global_scale: 1.0,
                ..trained.moe().weights
            },
        )
        .route(&trained.logits);
        let deviation = trained.weight_deviation(&unscaled);
        assert!(
            (deviation - (trained.global_scale.recip() - 1.0)).abs() < 1.0,
            "deviation {deviation:e}"
        );
    }

    /// `log(sigmoid(x))` is the one place the router can return an infinity,
    /// and a softmax has no way back from one. mlx-vlm's `-logaddexp(0, -x)` is
    /// shifted for the same reason this is.
    #[test]
    fn the_log_sigmoid_stays_finite_at_both_ends() {
        for x in [-1e30, -200.0, -1.0, 0.0, 1.0, 200.0, 1e30] {
            let got = log_sigmoid(x);
            assert!(got.is_finite(), "log_sigmoid({x}) is {got}");
            assert!(got <= 0.0, "log_sigmoid({x}) is {got}");
        }
        assert_eq!(log_sigmoid(0.0), -std::f32::consts::LN_2);
        assert_eq!(log_sigmoid(-200.0), -200.0);
    }
}
