//! The two ops a dense decoder layer is built from: RMSNorm, and the SwiGLU
//! feed-forward network.
//!
//! Both are pinned to mlx-vlm by `reference/fixtures/ops.safetensors`. Unlike
//! MXFP4 dequantisation neither can be pinned exactly — each reduces over its
//! whole feature axis, so summation order alone moves the last bits — and the
//! tests bound the disagreement instead of demanding equality.

/// Root-mean-square normalisation over the last axis, `x * rsqrt(mean(x²) +
/// eps) * weight`, with one row per `weight.len()` values of `x`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
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
/// Apple GPUs have no double at all, so a Metal RMSNorm cannot buy that range
/// the way this does. It has to buy it the way [`softmax`] below already does —
/// divide each row by its largest magnitude before squaring, and put the factor
/// back after the reciprocal square root — which holds the same range entirely
/// in f32. Until that kernel exists, `a_row_that_squares_past_f32_still_normalises`
/// below is what a naive port fails: that failure is the intended signal, and the
/// answer is to scale the kernel's row, never to lower this accumulator to
/// match it.
///
/// `eps` sits under the square root, which is what keeps an all-zero row from
/// dividing by zero: it scales by `1/sqrt(eps)` and stays zero.
fn inverse_rms(row: &[f32], eps: f32) -> f32 {
    let sum: f64 = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    (sum / row.len() as f64 + f64::from(eps)).sqrt().recip() as f32
}

/// A dense layer's feed-forward network: a SwiGLU MLP times a learned
/// `global_scale`.
///
/// mlx-vlm applies that trailing scale in `InklingDenseMLP`, outside the
/// `SwiGLUMLP` body it shares with other models, so it is easy to leave out and
/// it is not 1.
#[derive(Debug, Clone, Copy)]
pub struct DenseMlp<'a> {
    dim: usize,
    hidden_dim: usize,
    gate_proj: &'a [f32],
    up_proj: &'a [f32],
    down_proj: &'a [f32],
    global_scale: f32,
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

        Self {
            dim,
            hidden_dim: gate_proj.len() / dim,
            gate_proj,
            up_proj,
            down_proj,
            global_scale,
        }
    }

    /// The width this maps between, which for a layer's own MLP is the hidden
    /// size and for one expert of a bank is the same.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `[rows, dim]` in, `[rows, dim]` out.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len() % self.dim,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.dim
        );

        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(self.dim) {
            let mut gate = linear(row, self.gate_proj, self.dim);
            swiglu(&mut gate, &linear(row, self.up_proj, self.dim));
            out.extend(
                linear(&gate, self.down_proj, self.hidden_dim)
                    .iter()
                    .map(|y| y * self.global_scale),
            );
        }
        out
    }
}

/// `y = x @ wᵀ`, for `[rows, in_dim]` against the `[out, in]` row-major weight
/// `nn.Linear` stores. None of Inkling's projections carries a bias.
pub fn linear(x: &[f32], weight: &[f32], in_dim: usize) -> Vec<f32> {
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
fn swiglu(gate: &mut [f32], up: &[f32]) {
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
    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::fixture::{self, deviation};

    /// Synthetic inputs and mlx-vlm's answers to them, from
    /// `just dump-op-fixture`.
    const FIXTURE: &str = "ops.safetensors";

    const NORM_CASES: [&str; 5] = [
        "norm_wide",
        "norm_odd",
        "norm_batched",
        "norm_zero_row",
        "norm_large",
    ];

    /// 1e-6 is a few tens of f32 ulps at that scale. Both ops reduce over their
    /// feature axis, so their summation order and MLX's part company in the
    /// last bits and no tighter bound is honest; much looser would stop telling
    /// a rounding difference from a wrong formula. Measured when this landed:
    /// 1.8e-7 worst across the RMSNorm cases and 2.0e-7 for the MLP, so the
    /// bound has a factor of five in hand. Needing more than that is a bug
    /// signal, not a reason to widen it.
    const TOLERANCE: f32 = 1e-6;

    fn eps(ckpt: &Checkpoint) -> f32 {
        fixture::f32s(&fixture::tensor(ckpt, "rms_norm_eps"))[0]
    }

    /// A case's input, weight and mlx-vlm's output for it.
    fn norm_case(ckpt: &Checkpoint, case: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let of = |field| fixture::f32s(&fixture::tensor(ckpt, &format!("{case}.{field}")));
        (of("input"), of("weight"), of("output"))
    }

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
    }

    #[test]
    fn rms_norm_reproduces_mlx_for_every_shape() {
        let ckpt = fixture::open(FIXTURE);
        let eps = eps(&ckpt);
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

        let got = rms_norm(&x, &weight, eps(&ckpt));
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

    /// `silu` is not symmetric in its two operands, so exchanging the
    /// projections has to move the output.
    #[test]
    fn swapping_gate_and_up_changes_the_answer() {
        let mlp = Mlp::load(&fixture::open(FIXTURE));
        let swapped = mlp.with(&mlp.up_proj, &mlp.gate_proj, mlp.global_scale);
        assert!(deviation(&swapped, &mlp.output) > TOLERANCE);
    }
}
