//! Scaled dot-product attention: the softmax over `q·kᵀ` that the banded mask
//! biases and the short convolutions position.
//!
//! Two of its details are the kind that produce fluent text and wrong numbers.
//! The scale is `1 / head_dim` and not the conventional `1 / sqrt(head_dim)`,
//! and the eight KV heads are shared across the thirty-two query heads in
//! contiguous blocks of four rather than by striding. Either mistake runs, and
//! neither is visible in anything but the tensors.
//!
//! Pinned to mlx-vlm by the `q_norm_out`, `k_norm_out`, `v_sconv_out`, `mask`
//! and `sdpa_out` tensors of `reference/fixtures/layer_activations.safetensors`,
//! which between them carry everything the step consumes and produces. No
//! weights are involved, so this needs no checkpoint.
//!
//! Everything here takes one sequence at a time: batching is the scheduler's,
//! and a batch of sequences is a loop over these.

/// The softmax attention step, over `[heads, queries, head_dim]` queries and
/// `[kv_heads, keys, head_dim]` keys and values.
#[derive(Debug, Clone, Copy)]
pub struct Sdpa {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Sdpa {
    /// `kv_heads` divides `heads`: grouped-query attention, in which each KV
    /// head is read by `heads / kv_heads` query heads.
    pub fn new(heads: usize, kv_heads: usize, head_dim: usize) -> Self {
        assert!(kv_heads > 0, "attention needs at least one KV head");
        assert_eq!(
            heads % kv_heads,
            0,
            "{heads} query heads do not divide into {kv_heads} groups"
        );
        assert!(head_dim > 0, "a head needs at least one channel");

        Self {
            heads,
            kv_heads,
            head_dim,
            scale: 1.0 / head_dim as f32,
        }
    }

    /// The logit scale, which `InklingAttention` sets to `1 / head_dim`.
    ///
    /// Not `1 / sqrt(head_dim)`. Attention under the conventional scale is still
    /// a distribution over the same keys, just a flatter one, so the model keeps
    /// generating and only the numbers say otherwise.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The KV head that query head `head` reads.
    ///
    /// `mx.fast.scaled_dot_product_attention` repeats each KV head over a
    /// contiguous block of query heads, so heads 0..4 all read KV head 0. The
    /// other reading — query head `h` takes KV head `h % kv_heads` — pairs every
    /// query head with a key of the right shape and produces a plausible
    /// distribution over the wrong keys.
    pub fn kv_head(&self, head: usize) -> usize {
        head / (self.heads / self.kv_heads)
    }

    /// `q` is `[heads, queries, head_dim]`, `k` and `v` are `[kv_heads, keys,
    /// head_dim]` and `mask` is the additive `[heads, queries, keys]` the banded
    /// mask produced. Out comes `[heads, queries, head_dim]`.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], mask: &[f32]) -> Vec<f32> {
        let query_stride = self.heads * self.head_dim;
        assert_eq!(
            q.len() % query_stride,
            0,
            "{} values are not whole queries of {query_stride}",
            q.len()
        );
        let queries = q.len() / query_stride;

        let key_stride = self.kv_heads * self.head_dim;
        assert_eq!(
            k.len() % key_stride,
            0,
            "{} values are not whole keys of {key_stride}",
            k.len()
        );
        let keys = k.len() / key_stride;
        assert_eq!(v.len(), k.len(), "values against keys");
        assert_eq!(mask.len(), self.heads * queries * keys, "mask");

        let mut out = vec![0.0; q.len()];
        let mut weights = vec![0.0; keys];
        for head in 0..self.heads {
            let at = self.kv_head(head) * keys * self.head_dim;
            let (k, v) = (&k[at..], &v[at..]);
            for i in 0..queries {
                let at = (head * queries + i) * self.head_dim;
                let q = &q[at..at + self.head_dim];
                let mask = &mask[(head * queries + i) * keys..][..keys];

                for (j, (weight, mask)) in weights.iter_mut().zip(mask).enumerate() {
                    let key = &k[j * self.head_dim..][..self.head_dim];
                    *weight = dot(q, key) * self.scale + mask;
                }
                softmax(&mut weights);

                let out = &mut out[at..at + self.head_dim];
                for (weight, value) in weights.iter().zip(v.chunks_exact(self.head_dim)) {
                    for (out, value) in out.iter_mut().zip(value) {
                        *out += weight * value;
                    }
                }
            }
        }
        out
    }
}

/// `[rows, heads * head_dim]` — the layout a projection produces — as `[heads,
/// rows, head_dim]`, the layout attention reads.
///
/// `InklingAttention` writes this as a reshape into `[B, L, H, D]` followed by a
/// transpose to `[B, H, L, D]`, inline at each of its three call sites.
pub fn split_heads(x: &[f32], heads: usize, head_dim: usize) -> Vec<f32> {
    let stride = heads * head_dim;
    assert_eq!(
        x.len() % stride,
        0,
        "{} values are not whole rows of {stride}",
        x.len()
    );
    let mut out = Vec::with_capacity(x.len());
    for head in 0..heads {
        for row in x.chunks_exact(stride) {
            out.extend_from_slice(&row[head * head_dim..][..head_dim]);
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Softmax in place, over a row's largest entry.
///
/// The shift is what lets the mask carry a magnitude rather than an infinity: a
/// row whose keys are all masked shifts to zeros and leaves a uniform
/// distribution, where an unshifted `exp` would leave zeros and divide by their
/// sum.
fn softmax(row: &mut [f32]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{self, deviation};
    use crate::mask::{MASKED, is_masked};

    /// The forward pass `just dump-activations` recorded.
    const ACTIVATIONS: &str = "layer_activations.safetensors";

    const CAPTURED_LAYERS: [usize; 2] = [0, 2];

    /// The recorded step ran in bfloat16 on trained numbers, so its output is
    /// this computation rounded once on the way out, and the tolerance is that
    /// quantum rather than an arithmetic one: 2^-9 = 2.0e-3 relative, measured
    /// against the tensor's largest value rather than each entry's own
    /// magnitude, which puts it slightly above the ceiling. The same bound, for
    /// the same reason, as the trained masks.
    ///
    /// Worst observed when this landed: 2.2e-3 on layer 0. The weakest mutation
    /// these tests rely on catching, the conventional `1/sqrt(head_dim)` scale,
    /// moves the answer by 1.0 — over two decades above this bound.
    const TOLERANCE: f32 = 6e-3;

    /// One captured layer's attention step: everything it consumed, and what
    /// mlx-vlm produced from it.
    struct Case {
        name: String,
        sdpa: Sdpa,
        queries: usize,
        keys: usize,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        mask: Vec<f32>,
        want: Vec<f32>,
    }

    impl Case {
        /// `v` is the one input the fixture holds in the projection's own
        /// layout: `k_norm_out` and `q_norm_out` were taken as attention passed
        /// them to the kernel, already split into heads, and `v_sconv_out` was
        /// taken one step earlier.
        fn load(layer: usize) -> Self {
            let activations = fixture::open(ACTIVATIONS);
            let of = |name: &str| fixture::tensor(&activations, &format!("layer{layer}.{name}"));

            let q = of("q_norm_out");
            let k = of("k_norm_out");
            let &[_, heads, queries, head_dim] = q.shape() else {
                panic!("q_norm_out is [batch, heads, queries, head_dim]")
            };
            let &[_, kv_heads, keys, _] = k.shape() else {
                panic!("k_norm_out is [batch, kv_heads, keys, head_dim]")
            };

            Self {
                name: format!("layer{layer}"),
                sdpa: Sdpa::new(heads, kv_heads, head_dim),
                queries,
                keys,
                q: fixture::f32s(&q),
                k: fixture::f32s(&k),
                v: split_heads(&fixture::f32s(&of("v_sconv_out")), kv_heads, head_dim),
                mask: fixture::f32s(&of("mask")),
                want: fixture::f32s(&of("sdpa_out")),
            }
        }

        fn all() -> Vec<Self> {
            CAPTURED_LAYERS.iter().copied().map(Self::load).collect()
        }

        fn forward(&self) -> Vec<f32> {
            self.sdpa.forward(&self.q, &self.k, &self.v, &self.mask)
        }

        fn deviation(&self, got: &[f32]) -> f32 {
            deviation(got, &self.want)
        }

        /// The same keys and values with one KV head per query head, so a
        /// grouping rule becomes an ordinary gather and can be written out.
        fn ungrouped(&self, kv_head: impl Fn(usize) -> usize) -> (Sdpa, Vec<f32>, Vec<f32>) {
            let span = self.keys * self.sdpa.head_dim;
            let gather = |kv: &[f32]| {
                (0..self.sdpa.heads)
                    .flat_map(|head| kv[kv_head(head) * span..][..span].to_vec())
                    .collect()
            };
            (
                Sdpa::new(self.sdpa.heads, self.sdpa.heads, self.sdpa.head_dim),
                gather(&self.k),
                gather(&self.v),
            )
        }
    }

    #[test]
    fn the_captured_layers_reproduce_the_reference_attention() {
        let mut worst = 0.0f32;
        for case in Case::all() {
            let deviation = case.deviation(&case.forward());
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
            worst = worst.max(deviation);
        }
        assert!(
            worst > 0.0,
            "a run that matched exactly would mean bfloat16 rounding vanished"
        );
    }

    /// `InklingAttention` sets `scale = 1.0 / self.head_dim`. Writing the
    /// conventional form is the single most likely way to port this wrong.
    #[test]
    fn the_logit_scale_is_one_over_head_dim_not_its_square_root() {
        for case in Case::all() {
            let head_dim = case.sdpa.head_dim as f32;
            assert_eq!(case.sdpa.scale(), 1.0 / head_dim);

            let conventional = Sdpa {
                scale: head_dim.sqrt().recip(),
                ..case.sdpa
            };
            let got = conventional.forward(&case.q, &case.k, &case.v, &case.mask);
            let deviation = case.deviation(&got);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
        }
    }

    /// Each KV head serves a contiguous block of query heads: with 32 query
    /// heads over 8 KV heads, query heads 0..4 all read KV head 0.
    ///
    /// Stated against an attention with one KV head per query head, which has no
    /// grouping to get wrong: gathering the KV heads under the rule and handing
    /// them over ungrouped has to reproduce the grouped answer exactly, and the
    /// striding rule has to miss.
    #[test]
    fn each_kv_head_serves_a_contiguous_block_of_query_heads() {
        for case in Case::all() {
            let group = case.sdpa.heads / case.sdpa.kv_heads;
            assert_eq!((case.sdpa.heads, case.sdpa.kv_heads, group), (32, 8, 4));

            let (ungrouped, k, v) = case.ungrouped(|head| head / group);
            assert_eq!(
                ungrouped.forward(&case.q, &k, &v, &case.mask),
                case.forward(),
                "{}: blocks of {group}",
                case.name
            );

            let (ungrouped, k, v) = case.ungrouped(|head| head % case.sdpa.kv_heads);
            let deviation = case.deviation(&ungrouped.forward(&case.q, &k, &v, &case.mask));
            assert!(
                deviation > TOLERANCE,
                "{}: striding deviates by {deviation:e}",
                case.name
            );
        }
    }

    /// A key the mask rules out contributes nothing to its query, whatever the
    /// value at that key holds.
    ///
    /// This is the test that fails if the mask is added to the softmax's output
    /// rather than to its input: post-softmax, a masked key carries a weight of
    /// about `-1e30` instead of one of about zero, and the value behind it
    /// dominates the answer rather than vanishing from it.
    #[test]
    fn a_masked_key_cannot_reach_its_query() {
        for case in Case::all() {
            let head_dim = case.sdpa.head_dim;
            let last = case.keys - 1;
            let masked: Vec<usize> = (0..case.queries)
                .filter(|i| is_masked(case.mask[i * case.keys + last]))
                .collect();
            assert!(!masked.is_empty(), "{}: nothing is masked", case.name);

            let mut v = case.v.clone();
            for kv in 0..case.sdpa.kv_heads {
                for value in &mut v[(kv * case.keys + last) * head_dim..][..head_dim] {
                    *value = 1e6;
                }
            }

            let want = case.forward();
            let got = case.sdpa.forward(&case.q, &case.k, &v, &case.mask);
            for head in 0..case.sdpa.heads {
                for &i in &masked {
                    let at = (head * case.queries + i) * head_dim;
                    assert_eq!(
                        got[at..at + head_dim],
                        want[at..at + head_dim],
                        "{}: head {head} query {i}",
                        case.name
                    );
                }
            }
            assert_ne!(got, want, "{}: the value went unread", case.name);
        }
    }

    /// A row with no key it may attend to still has to leave softmax with finite
    /// numbers, which is why the mask carries a magnitude rather than an
    /// infinity and why the softmax shifts by the row's largest entry.
    #[test]
    fn a_row_with_every_key_masked_stays_finite() {
        let (heads, head_dim, keys) = (1, 4, 3);
        let sdpa = Sdpa::new(heads, heads, head_dim);
        let got = sdpa.forward(
            &vec![1.0; heads * head_dim],
            &vec![1.0; heads * keys * head_dim],
            &vec![2.0; heads * keys * head_dim],
            &vec![MASKED; heads * keys],
        );
        assert_eq!(got, vec![2.0; heads * head_dim]);
    }
}
