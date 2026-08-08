//! The two ops a dense decoder layer is built from: RMSNorm, and the SwiGLU
//! feed-forward network.
//!
//! Both are pinned to mlx-vlm by `reference/fixtures/ops.safetensors`. Unlike
//! MXFP4 dequantisation neither can be pinned exactly — each reduces over its
//! whole feature axis, so summation order alone moves the last bits — and the
//! tests bound the disagreement instead of demanding equality.

use std::fmt::Debug;

use crate::profile::{self, Op};

/// Root-mean-square normalisation over the last axis, `x * rsqrt(mean(x²) +
/// eps) * weight`, with one row per `weight.len()` values of `x`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let _timed = profile::scope(Op::RmsNorm);
    assert_eq!(
        x.len() % weight.len(),
        0,
        "{} values are not whole rows of {}",
        x.len(),
        weight.len()
    );

    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(weight.len()) {
        let scale = inverse_rms(row, eps);
        out.extend(row.iter().zip(weight).map(|(x, w)| x * scale * w));
    }
    out
}

/// `1 / sqrt(mean(row²) + eps)`.
///
/// The sum accumulates in f64 for range rather than for precision: values above
/// roughly 3e18 square past f32's maximum, and a row that overflows there
/// normalises to zero rather than to something obviously wrong. MLX accumulates
/// in f32 and does flush such a row.
///
/// Apple GPUs have no double at all, so `inkling_metal::norm` cannot buy that
/// range the way this does. It buys it the way [`softmax`] below already does —
/// scaling each row before squaring — over a power of two rather than over the
/// peak itself, so that the division is exact and the factor cancels out of the
/// answer instead of being multiplied back into it.
///
/// `a_row_that_squares_past_f32_still_normalises` below is what a port that
/// scaled nothing fails, and the kernel keeps that case of its own. A failure
/// there is the intended signal: the answer is to scale the kernel's row, never
/// to lower this accumulator to match it.
///
/// `eps` sits under the square root, which is what keeps an all-zero row from
/// dividing by zero: it scales by `1/sqrt(eps)` and stays zero.
fn inverse_rms(row: &[f32], eps: f32) -> f32 {
    let sum: f64 = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    (sum / row.len() as f64 + f64::from(eps)).sqrt().recip() as f32
}

/// The three projections a SwiGLU feed-forward network multiplies through,
/// wherever their weights live.
///
/// The same seam [`crate::attention::Projections`] is, over the three that make
/// an MLP rather than the five that make an attention layer: a dense layer's
/// are `3 x [16384, 4096]`, which is 1.61 GB of float32 a decode step, and a
/// backend holding them may never decode any of it.
///
/// `Debug` because [`DenseMlp`] derives it.
pub trait MlpProjections: Debug {
    fn gate_proj(&self) -> &dyn Projection;
    fn up_proj(&self) -> &dyn Projection;
    fn down_proj(&self) -> &dyn Projection;

    /// The two that consume the same input, in one call — the same bargain
    /// [`crate::attention::Projections::qkvr`] is, over the pair rather than the
    /// four. `down_proj` is not here because it multiplies what the activation
    /// makes of these two.
    fn gate_up(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (self.gate_proj().forward(x), self.up_proj().forward(x))
    }
}

/// A dense layer's feed-forward network: a SwiGLU MLP times a learned
/// `global_scale`.
///
/// mlx-vlm applies that trailing scale in `InklingDenseMLP`, outside the
/// `SwiGLUMLP` body it shares with other models, so it is easy to leave out and
/// it is not 1.
#[derive(Debug, Clone, Copy)]
pub struct DenseMlp<'a> {
    projections: Held<'a>,
    global_scale: f32,
}

#[derive(Debug, Clone, Copy)]
enum Held<'a> {
    /// Weights decoded to float32 and multiplied here, which is the path every
    /// other one is checked against.
    Decoded(Decoded<'a>),
    /// A backend holding the weights itself, which may never decode them.
    Backend(&'a dyn MlpProjections),
}

/// Three decoded weights with the widths they map between settled.
#[derive(Debug, Clone, Copy)]
struct Decoded<'a> {
    gate_proj: DenseProjection<'a>,
    up_proj: DenseProjection<'a>,
    down_proj: DenseProjection<'a>,
}

impl MlpProjections for Decoded<'_> {
    fn gate_proj(&self) -> &dyn Projection {
        &self.gate_proj
    }

    fn up_proj(&self) -> &dyn Projection {
        &self.up_proj
    }

    fn down_proj(&self) -> &dyn Projection {
        &self.down_proj
    }
}

impl<'a> DenseMlp<'a> {
    /// Projections are `[out, in]` row-major, the layout `nn.Linear` stores.
    pub fn new(
        dim: usize,
        gate_proj: &'a [f32],
        up_proj: &'a [f32],
        down_proj: &'a [f32],
        global_scale: f32,
    ) -> Self {
        assert_eq!(
            gate_proj.len() % dim,
            0,
            "{} gate weights are not whole rows of {dim}",
            gate_proj.len()
        );
        assert_eq!(up_proj.len(), gate_proj.len(), "up against gate");
        assert_eq!(down_proj.len(), gate_proj.len(), "down against gate");

        let hidden_dim = gate_proj.len() / dim;
        Self::over(
            Held::Decoded(Decoded {
                gate_proj: DenseProjection::new(dim, gate_proj),
                up_proj: DenseProjection::new(dim, up_proj),
                down_proj: DenseProjection::new(hidden_dim, down_proj),
            }),
            dim,
            hidden_dim,
            global_scale,
        )
    }

    /// The same MLP over three projections a backend answers for, mapping from
    /// `dim` through `hidden_dim` and back.
    ///
    /// Both widths are the caller's to state rather than the projections' to
    /// report, for the reason [`AttentionConfig`](crate::AttentionConfig)'s
    /// `hidden` is: they are what the three are checked against, and a width
    /// read off one of the weights being checked would agree with itself.
    pub fn backend(
        dim: usize,
        hidden_dim: usize,
        projections: &'a dyn MlpProjections,
        global_scale: f32,
    ) -> Self {
        Self::over(Held::Backend(projections), dim, hidden_dim, global_scale)
    }

    /// Whether the three are one MLP's of these two widths, which has to be
    /// settled here: the activation zips `gate` against `up`, so an `up_proj`
    /// narrower than `gate_proj` would truncate the answer rather than fail, and
    /// a `down_proj` that maps back to another width would only show up in the
    /// residual add a layer later.
    fn over(projections: Held<'a>, dim: usize, hidden_dim: usize, global_scale: f32) -> Self {
        let mlp = Self {
            projections,
            global_scale,
        };
        let three = mlp.projections();
        for (name, projection, from, to) in [
            ("gate_proj", three.gate_proj(), dim, hidden_dim),
            ("up_proj", three.up_proj(), dim, hidden_dim),
            ("down_proj", three.down_proj(), hidden_dim, dim),
        ] {
            assert_eq!(projection.in_dim(), from, "the width {name} maps from");
            assert_eq!(projection.out_dim(), to, "the width {name} maps to");
        }
        mlp
    }

    fn projections(&self) -> &dyn MlpProjections {
        match &self.projections {
            Held::Decoded(decoded) => decoded,
            Held::Backend(projections) => *projections,
        }
    }

    /// The width this maps between, which for a layer's own MLP is the hidden
    /// size and for one expert of a bank is the same.
    pub fn dim(&self) -> usize {
        self.projections().gate_proj().in_dim()
    }

    /// `[rows, dim]` in, `[rows, dim]` out.
    ///
    /// Every row at once rather than a row at a time. The arithmetic is
    /// identical — a projection multiplies each row against every weight row
    /// independently — and it is three calls for the whole batch rather than
    /// three per row, which is what a backend where a call is a dispatch needs.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let three = self.projections();
        assert_eq!(
            x.len() % self.dim(),
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.dim()
        );

        let (mut gate, up) = three.gate_up(x);
        swiglu(&mut gate, &up);
        self.scaled(three.down_proj().forward(&gate))
    }

    /// The trailing `global_scale` this network's rows carry.
    ///
    /// Handed out because it is the one thing in this network that a backend
    /// which dispatched all three still owes: mlx-vlm applies it in
    /// `InklingDenseMLP`, outside the `SwiGLUMLP` body it shares with other
    /// models, so it is easy to leave out and it is not 1. A backend whose next
    /// dispatch reads these rows applies it where that dispatch reads them —
    /// see [`ShortConv`](crate::sconv::ShortConv), which is what reads them in
    /// a layer.
    pub fn scale(&self) -> f32 {
        self.global_scale
    }

    /// The same scale applied here, for rows this side holds.
    fn scaled(&self, rows: Vec<f32>) -> Vec<f32> {
        rows.iter().map(|y| y * self.global_scale).collect()
    }
}

/// `y = x @ wᵀ`, for `[rows, in_dim]` against the `[out, in]` row-major weight
/// `nn.Linear` stores. None of Inkling's projections carries a bias.
pub fn linear(x: &[f32], weight: &[f32], in_dim: usize) -> Vec<f32> {
    let _timed = profile::scope(Op::Linear);
    assert_eq!(
        x.len() % in_dim,
        0,
        "{} values are not whole rows of {in_dim}",
        x.len()
    );
    assert_eq!(
        weight.len() % in_dim,
        0,
        "{} weights are not whole rows of {in_dim}",
        weight.len()
    );

    let mut out = Vec::with_capacity(x.len() / in_dim * (weight.len() / in_dim));
    for x in x.chunks_exact(in_dim) {
        out.extend(
            weight
                .chunks_exact(in_dim)
                .map(|row| x.iter().zip(row).map(|(x, w)| x * w).sum::<f32>()),
        );
    }
    out
}

/// `y = x @ wᵀ`, for a weight held however its owner holds it.
///
/// [`linear`] above takes the weight as a `&[f32]`, and that signature is a
/// claim as much as a convenience: it says the weight has been decoded into
/// memory. Every projection in an Inkling checkpoint is MXFP4 — the whole model
/// is 130.6 GiB packed and 1.1 TB decoded — so the claim is one the CPU path
/// pays for a run at a time, through [`Scratch`](crate::quant::Scratch), and one
/// a GPU kernel that decodes codes in registers should not have to pay at all.
///
/// This is the operation with that claim taken out. A caller written against it
/// is handed weights it cannot see the storage of, which is what lets the same
/// caller run against decoded floats or against packed codes.
pub trait Projection {
    /// The width a row of the input has to be.
    fn in_dim(&self) -> usize;

    /// The width a row of the output comes out.
    fn out_dim(&self) -> usize;

    /// `[rows, in_dim]` in, `[rows, out_dim]` out.
    fn forward(&self, x: &[f32]) -> Vec<f32>;
}

/// A projection over weights already decoded to float32, `[out_dim, in_dim]`
/// row-major — the layout `nn.Linear` stores and the one [`linear`] takes.
#[derive(Debug, Clone, Copy)]
pub struct DenseProjection<'a> {
    in_dim: usize,
    weight: &'a [f32],
}

impl<'a> DenseProjection<'a> {
    pub fn new(in_dim: usize, weight: &'a [f32]) -> Self {
        assert!(in_dim > 0, "a projection maps from some width");
        assert_eq!(
            weight.len() % in_dim,
            0,
            "{} weights are not whole rows of {in_dim}",
            weight.len()
        );
        Self { in_dim, weight }
    }
}

impl Projection for DenseProjection<'_> {
    fn in_dim(&self) -> usize {
        self.in_dim
    }

    fn out_dim(&self) -> usize {
        self.weight.len() / self.in_dim
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        linear(x, self.weight, self.in_dim)
    }
}

/// The indices of the `k` largest values, largest first, ties going to the
/// lower index.
///
/// The tie-break is not arbitrary at either end of the model that uses this. The
/// router picks six of 256 gate scores, and mlx-vlm's `mx.argpartition` leaves a
/// selection whose *order* is more than the reference promises — see
/// [`SparseMoe::route`](crate::moe::SparseMoe::route). The tests compare a
/// ranking of logits against `mx.argsort`, which is stable and ascending, so
/// `argsort(-x)` leaves equal values in index order; agreeing with that matters
/// because the reference's logits are bfloat16, three significant digits over
/// 200058 values, and exact ties are ordinary.
///
/// Partitioned before it is sorted, so ranking the top six of 256 — or the top
/// thirty-two of 200058 — costs a linear pass and a sort of `k`.
pub fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(values.len());
    if k == 0 {
        return Vec::new();
    }
    let rank = |a: &usize, b: &usize| values[*b].total_cmp(&values[*a]).then(a.cmp(b));
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.select_nth_unstable_by(k - 1, rank);
    order.truncate(k);
    order.sort_unstable_by(rank);
    order
}

/// Softmax in place, over a row's largest entry.
///
/// The shift is what lets attention's mask carry a magnitude rather than an
/// infinity: a row whose keys are all masked shifts to zeros and leaves a
/// uniform distribution, where an unshifted `exp` would leave zeros and divide
/// by their sum. It is also what the router's `exp(x - logsumexp(x))` is, that
/// form being the shift written out.
pub fn softmax(row: &mut [f32]) {
    let peak = row.iter().fold(f32::NEG_INFINITY, |peak, x| peak.max(*x));
    let mut total = 0.0;
    for x in row.iter_mut() {
        *x = (*x - peak).exp();
        total += *x;
    }
    for x in row.iter_mut() {
        *x /= total;
    }
}

/// `silu(gate) * up`, written over `gate`.
///
/// The activation goes on the gate projection and not on the up projection:
/// `SwiGLUMLP` computes `swiglu(gate_proj(x), up_proj(x))`, and mlx-vlm's
/// `swiglu` is `silu(gate) * x`. The two are interchangeable to anyone reading
/// generated text and not to the numbers.
///
/// Public because a backend that runs the two projections as dispatches still
/// has to join them, and a second spelling of which operand the activation goes
/// on is the one mistake this whole comment exists to prevent.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    let _timed = profile::scope(Op::Swiglu);
    // A zip over unequal lengths is a truncation and not a panic, which for two
    // projections of a width that disagreed would be an answer of the shorter
    // one's shape rather than an error. [`DenseMlp::new`] settles this at
    // construction; a caller assembling the two out of separate dispatches has
    // nowhere else to.
    assert_eq!(gate.len(), up.len(), "the gate against what gates it");
    for (gate, up) in gate.iter_mut().zip(up) {
        *gate = silu(*gate) * up;
    }
}

/// `x * sigmoid(x)`, as one division.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::fixture::{self, NORM_CASES, OPS as FIXTURE, deviation, norm_case, norm_eps};

    /// 1e-6 is a few tens of f32 ulps at that scale. Both ops reduce over their
    /// feature axis, so their summation order and MLX's part company in the
    /// last bits and no tighter bound is honest; much looser would stop telling
    /// a rounding difference from a wrong formula. Measured when this landed:
    /// 1.8e-7 worst across the RMSNorm cases and 2.0e-7 for the MLP, so the
    /// bound has a factor of five in hand. Needing more than that is a bug
    /// signal, not a reason to widen it.
    const TOLERANCE: f32 = 1e-6;

    struct Mlp {
        dim: usize,
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
        global_scale: f32,
        input: Vec<f32>,
        output: Vec<f32>,
    }

    impl Mlp {
        fn load(ckpt: &Checkpoint) -> Self {
            let of = |field| fixture::f32s(&fixture::tensor(ckpt, &format!("mlp.{field}")));
            let input = fixture::tensor(ckpt, "mlp.input");
            Self {
                dim: *input.shape().last().expect("input has a last axis"),
                gate_proj: of("gate_proj.weight"),
                up_proj: of("up_proj.weight"),
                down_proj: of("down_proj.weight"),
                global_scale: of("global_scale")[0],
                input: fixture::f32s(&input),
                output: of("output"),
            }
        }

        fn with(&self, gate_proj: &[f32], up_proj: &[f32], global_scale: f32) -> Vec<f32> {
            DenseMlp::new(self.dim, gate_proj, up_proj, &self.down_proj, global_scale)
                .forward(&self.input)
        }

        fn forward(&self) -> Vec<f32> {
            self.with(&self.gate_proj, &self.up_proj, self.global_scale)
        }

        fn hidden_dim(&self) -> usize {
            self.gate_proj.len() / self.dim
        }

        /// The same three weights as a backend's, cut to `hidden_dim` of the
        /// width between — which at the full width is this MLP and at any other
        /// is three projections that agree with each other and not with the
        /// layer.
        fn handed(&self, hidden_dim: usize) -> Handed<'_> {
            let between = hidden_dim * self.dim;
            Handed {
                three: Decoded {
                    gate_proj: DenseProjection::new(self.dim, &self.gate_proj[..between]),
                    up_proj: DenseProjection::new(self.dim, &self.up_proj[..between]),
                    down_proj: DenseProjection::new(hidden_dim, &self.down_proj[..between]),
                },
                gate_up: Cell::new(0),
            }
        }
    }

    /// An [`MlpProjections`] that is not this module's — the three answered by
    /// something the MLP cannot see inside, which is the whole of what a backend
    /// is from here.
    #[derive(Debug)]
    struct Handed<'a> {
        three: Decoded<'a>,
        /// How many times the MLP asked for the two that share an input
        /// together, which is what says it did not ask for them one at a time.
        gate_up: Cell<usize>,
    }

    impl MlpProjections for Handed<'_> {
        fn gate_up(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
            self.gate_up.set(self.gate_up.get() + 1);
            (self.gate_proj().forward(x), self.up_proj().forward(x))
        }

        fn gate_proj(&self) -> &dyn Projection {
            self.three.gate_proj()
        }

        fn up_proj(&self) -> &dyn Projection {
            self.three.up_proj()
        }

        fn down_proj(&self) -> &dyn Projection {
            self.three.down_proj()
        }
    }

    #[test]
    fn rms_norm_reproduces_mlx_for_every_shape() {
        let ckpt = fixture::open(FIXTURE);
        let eps = norm_eps(&ckpt);
        let mut widths = Vec::new();

        for case in NORM_CASES {
            let (x, weight, want) = norm_case(&ckpt, case);
            let deviation = deviation(&rms_norm(&x, &weight, eps), &want);
            assert!(deviation <= TOLERANCE, "{case}: deviation {deviation:e}");
            widths.push(weight.len());
        }

        // A regenerated fixture that quietly lost its ragged case would let a
        // future SIMD path assume the feature axis is a whole number of lanes.
        assert!(
            widths.iter().any(|width| width % 8 != 0),
            "no case has a last axis that is not a multiple of 8: {widths:?}"
        );
    }

    /// `eps` exists for this row, and it is the one place the normalisation
    /// divides by zero.
    #[test]
    fn an_all_zero_row_normalises_to_zero_rather_than_nan() {
        let ckpt = fixture::open(FIXTURE);
        let (x, weight, _) = norm_case(&ckpt, "norm_zero_row");
        let width = weight.len();

        let zeroed = x
            .chunks_exact(width)
            .position(|row| row.iter().all(|x| *x == 0.0))
            .expect("the fixture carries an all-zero row");

        let got = rms_norm(&x, &weight, norm_eps(&ckpt));
        assert!(got.iter().all(|y| y.is_finite()), "{got:?}");
        assert!(
            got[zeroed * width..(zeroed + 1) * width]
                .iter()
                .all(|y| *y == 0.0),
            "{:?}",
            &got[zeroed * width..(zeroed + 1) * width]
        );
    }

    /// The `norm_large` case stays inside f32 only because MLX's accumulator
    /// does and the fixture has to remain an oracle. A row ten times larger
    /// squares past f32's maximum, where an f32 accumulator returns a row of
    /// zeros — a wrong answer that looks like a well-behaved one.
    #[test]
    fn a_row_that_squares_past_f32_still_normalises() {
        let width = 32;
        let got = rms_norm(&vec![1e20; width], &vec![1.0; width], 1e-6);
        assert!(
            got.iter().all(|y| (y - 1.0).abs() <= TOLERANCE),
            "uniform row should normalise to its weight: {got:?}"
        );
    }

    /// The shift by the row's peak is invisible in the answer and is the whole
    /// point of the implementation: a row of large entries has to normalise
    /// rather than overflow, and one of equal entries has to come out uniform
    /// however large they are.
    #[test]
    fn softmax_normalises_without_overflowing() {
        let mut row = [90.0, 91.0, 92.0];
        softmax(&mut row);
        assert!(
            (row.iter().sum::<f32>() - 1.0).abs() <= TOLERANCE,
            "{row:?}"
        );
        assert!(row[0] < row[1] && row[1] < row[2], "{row:?}");

        let mut flat = [1e30; 4];
        softmax(&mut flat);
        assert_eq!(flat, [0.25; 4]);
    }

    /// Two rows of three against two weight rows of three, whose axes are
    /// unequal so that a weight read transposed would not even fit.
    const PROJECTION_IN_DIM: usize = 3;
    const PROJECTION_X: [f32; 6] = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
    const PROJECTION_WEIGHT: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    /// `[rows, in]` against a `[out, in]` weight, one output row per input row.
    #[test]
    fn a_projection_maps_each_row_against_every_weight_row() {
        assert_eq!(
            linear(&PROJECTION_X, &PROJECTION_WEIGHT, PROJECTION_IN_DIM),
            [1.0, 3.0, 10.0, 30.0]
        );
    }

    /// The seam over a decoded weight is [`linear`] and nothing else, and the
    /// two shapes it reports are what a caller that cannot see the weight has to
    /// go on. A projection that read them off the wrong axis would report 2 and
    /// 3 the other way round.
    #[test]
    fn a_dense_projection_is_the_linear_it_reports_the_shape_of() {
        let projection = DenseProjection::new(PROJECTION_IN_DIM, &PROJECTION_WEIGHT);

        assert_eq!(projection.in_dim(), PROJECTION_IN_DIM);
        assert_eq!(projection.out_dim(), 2);
        assert_eq!(
            projection.forward(&PROJECTION_X),
            linear(&PROJECTION_X, &PROJECTION_WEIGHT, PROJECTION_IN_DIM)
        );
    }

    #[test]
    #[should_panic(expected = "are not whole rows of 3")]
    fn a_dense_projection_over_a_ragged_weight_is_refused() {
        DenseProjection::new(PROJECTION_IN_DIM, &[1.0; 7]);
    }

    /// The width divides, so it has to be refused before it does — a projection
    /// from nothing would panic on the remainder rather than on the shape.
    #[test]
    #[should_panic(expected = "a projection maps from some width")]
    fn a_dense_projection_from_no_width_is_refused() {
        DenseProjection::new(0, &[]);
    }

    #[test]
    fn the_dense_mlp_reproduces_mlx() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        let deviation = deviation(&mlp.forward(), &mlp.output);
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }

    #[test]
    fn dropping_global_scale_changes_the_answer() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        assert_ne!(mlp.global_scale, 1.0, "a scale of 1 would prove nothing");

        let unscaled = mlp.with(&mlp.gate_proj, &mlp.up_proj, 1.0);
        assert!(deviation(&unscaled, &mlp.output) > TOLERANCE);
    }

    /// The seam: an MLP whose three projections are answered by a backend is the
    /// same MLP.
    ///
    /// Exact rather than bounded, because the backend here multiplies the same
    /// weights through the same [`linear`] — what changes is only who was asked.
    #[test]
    fn an_mlp_whose_projections_come_from_a_backend_is_the_same_mlp() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        let handed = mlp.handed(mlp.hidden_dim());

        assert_eq!(
            DenseMlp::backend(mlp.dim, mlp.hidden_dim(), &handed, mlp.global_scale)
                .forward(&mlp.input),
            mlp.forward()
        );
    }

    /// The two that share an input are asked for in one call, not two.
    ///
    /// The same claim [`crate::attention`]'s own spy makes about the four:
    /// where a multiply is a dispatch, asking twice is a round trip that the
    /// arithmetic did not need. An MLP that went back to `gate_proj()` and
    /// `up_proj()` separately would produce the same answer, so the count is
    /// what says otherwise.
    #[test]
    fn an_mlp_asks_a_backend_for_the_two_that_share_an_input_together() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        let handed = mlp.handed(mlp.hidden_dim());

        DenseMlp::backend(mlp.dim, mlp.hidden_dim(), &handed, mlp.global_scale).forward(&mlp.input);

        assert_eq!(handed.gate_up.get(), 1, "one call a forward");
    }

    /// Three projections that agree with each other but not with the layer are
    /// refused, which is the mistake only a stated width catches: they are
    /// another MLP's, and every check they could be put to among themselves
    /// passes.
    #[test]
    #[should_panic(expected = "the width gate_proj maps to")]
    fn three_projections_of_another_mlps_width_are_refused() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        let handed = mlp.handed(mlp.hidden_dim() / 2);

        DenseMlp::backend(mlp.dim, mlp.hidden_dim(), &handed, mlp.global_scale);
    }

    /// And a `down_proj` that maps back to another width is refused too, where
    /// the two that come before it are this layer's. `silu(gate) * up` is a zip
    /// and the residual add is another, so a width that disagreed would truncate
    /// an answer rather than fail — a layer later, and somewhere else.
    #[test]
    #[should_panic(expected = "the width down_proj maps to")]
    fn a_down_projection_that_maps_back_to_another_width_is_refused() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        let mut handed = mlp.handed(mlp.hidden_dim());
        handed.three.down_proj =
            DenseProjection::new(mlp.hidden_dim(), &mlp.down_proj[..mlp.down_proj.len() / 2]);

        DenseMlp::backend(mlp.dim, mlp.hidden_dim(), &handed, mlp.global_scale);
    }

    /// `silu` is not symmetric in its two operands, so exchanging the
    /// projections has to move the output.
    #[test]
    fn swapping_gate_and_up_changes_the_answer() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        let swapped = mlp.with(&mlp.up_proj, &mlp.gate_proj, mlp.global_scale);
        assert!(deviation(&swapped, &mlp.output) > TOLERANCE);
    }
}
