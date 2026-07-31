//! An attention layer, and the scaled dot product at the middle of it.
//!
//! Two details of that middle step are the kind that produce fluent text and
//! wrong numbers. The scale is `1 / head_dim` and not the conventional
//! `1 / sqrt(head_dim)`, and the eight KV heads are shared across the
//! thirty-two query heads in contiguous blocks of four rather than by striding.
//! Either mistake runs, and neither is visible in anything but the tensors.
//!
//! The step is pinned to mlx-vlm by the `q_norm_out`, `k_norm_out`,
//! `v_sconv_out`, `mask` and `sdpa_out` tensors of
//! `reference/fixtures/layer_activations.safetensors`, which between them carry
//! everything it consumes and produces and involve no weights at all. The layer
//! around it is pinned by `reference/fixtures/attention.safetensors`, whose
//! synthetic cases are float32 throughout and reach the one branch a recorded
//! forward pass cannot: log scaling, which fires past position 128000.
//!
//! Everything here takes one sequence at a time: batching is the scheduler's,
//! and a batch of sequences is a loop over these.

use crate::config::TextConfig;
use crate::mask::{BandedMask, is_masked};
use crate::ops::{linear, rms_norm, softmax};
use crate::sconv::{ConvState, ShortConv};

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

/// The positional scaling a global layer applies to long contexts.
///
/// Below the floor `tau` is exactly 1 and the whole branch is inert, which is
/// why no recorded forward pass can reach it: the checkpoint's floor is 128000
/// tokens.
#[derive(Debug, Clone, Copy)]
pub struct LogScaling {
    floor: f32,
    alpha: f32,
}

impl LogScaling {
    /// The checkpoint's `log_scaling_n_floor` and `log_scaling_alpha`.
    ///
    /// `InklingAttention` leaves this off on a sliding layer whatever the
    /// config says, so a sliding layer has none rather than one that never
    /// fires.
    pub fn new(floor: f32, alpha: f32) -> Self {
        assert!(floor > 0.0, "a floor of {floor} would divide by zero");
        Self { floor, alpha }
    }

    /// `1 + alpha * ln(max(position / floor, 1))`.
    ///
    /// `position` counts from zero — query `i` of a call at cache offset `o`
    /// sits at `i + o` — and the reference counts the first token as 1, so the
    /// ratio is taken against `position + 1`.
    pub fn tau(&self, position: usize) -> f32 {
        1.0 + self.alpha * (((position + 1) as f32 / self.floor).max(1.0)).ln()
    }
}

/// The shapes and scalars `InklingAttention.__init__` derives for one layer.
/// Sliding and global layers read different fields of the config for all of
/// them, so which set a layer got is not recoverable from the tensors.
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    /// The layer's window, or zero on a global layer.
    pub sliding: usize,
    pub rms_norm_eps: f32,
    pub log_scaling: Option<LogScaling>,
}

impl AttentionConfig {
    /// What `InklingAttention.__init__` reads for one layer.
    ///
    /// A sliding layer takes its head counts, its head width and its window
    /// from the `swa_` fields and a global layer from the plain ones. The two
    /// sets hold the same numbers in Inkling-Small, so nothing about that
    /// checkpoint can tell a port that read the wrong one — but they are
    /// separate fields for a reason, and mlx-vlm's own defaults give a sliding
    /// layer twice a global layer's KV heads.
    ///
    /// Log scaling is a global layer's alone, whatever the checkpoint sets
    /// `log_scaling_n_floor` to.
    pub fn for_layer(config: &TextConfig, layer: usize) -> Self {
        let sliding = config.layer_is_sliding(layer);
        let either = |swa, global| if sliding { swa } else { global };

        Self {
            heads: either(config.swa_num_attention_heads, config.num_attention_heads),
            kv_heads: either(config.swa_num_key_value_heads, config.num_key_value_heads),
            head_dim: either(config.swa_head_dim, config.head_dim),
            d_rel: config.d_rel,
            sliding: either(config.sliding_window_size, 0),
            rms_norm_eps: config.rms_norm_eps,
            log_scaling: config
                .log_scaling_n_floor
                .filter(|_| !sliding)
                .map(|floor| LogScaling::new(floor, config.log_scaling_alpha)),
        }
    }
}

/// One attention layer's tensors, as the checkpoint stores them: the five
/// projections `[out, in]` row-major and without bias, a per-head-channel
/// RMSNorm weight for the queries and one for the keys, a kernel per short
/// convolution, and the mask's `[d_rel, rel_extent]` projection.
#[derive(Debug, Clone, Copy)]
pub struct AttentionWeights<'a> {
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub r_proj: &'a [f32],
    pub o_proj: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub k_sconv: &'a [f32],
    pub v_sconv: &'a [f32],
    pub rel_proj: &'a [f32],
}

/// Everything one sequence carries between calls: the two short convolutions'
/// windows, and the keys and values already attended over.
///
/// The keys and values are held in the layout their projections produced,
/// `[keys, kv_heads * head_dim]`, so a call appends to them and splits into
/// heads once at the end. What is cached is the *normed* key and the
/// *convolved* value, which is where the reference caches them too.
#[derive(Debug, Clone)]
pub struct AttentionCache {
    k_sconv: ConvState,
    v_sconv: ConvState,
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// One attention layer, from the hidden state its input layernorm produced to
/// the one `o_proj` returns.
#[derive(Debug, Clone, Copy)]
pub struct Attention<'a> {
    sdpa: Sdpa,
    mask: BandedMask<'a>,
    k_sconv: ShortConv<'a>,
    v_sconv: ShortConv<'a>,
    hidden: usize,
    rms_norm_eps: f32,
    log_scaling: Option<LogScaling>,
    weights: AttentionWeights<'a>,
}

impl<'a> Attention<'a> {
    pub fn new(config: AttentionConfig, weights: AttentionWeights<'a>) -> Self {
        let (heads, kv_heads, head_dim) = (config.heads, config.kv_heads, config.head_dim);
        let sdpa = Sdpa::new(heads, kv_heads, head_dim);

        let hidden = weights.q_proj.len() / (heads * head_dim);
        for (name, weight, rows) in [
            ("q_proj", weights.q_proj, heads * head_dim),
            ("k_proj", weights.k_proj, kv_heads * head_dim),
            ("v_proj", weights.v_proj, kv_heads * head_dim),
            ("r_proj", weights.r_proj, heads * config.d_rel),
            ("o_proj", weights.o_proj, hidden),
        ] {
            assert_eq!(
                weight.len(),
                rows * hidden,
                "{name} against a hidden {hidden}"
            );
        }
        assert_eq!(weights.q_norm.len(), head_dim, "q_norm");
        assert_eq!(weights.k_norm.len(), head_dim, "k_norm");

        Self {
            sdpa,
            mask: BandedMask::new(config.d_rel, weights.rel_proj, config.sliding),
            k_sconv: ShortConv::new(kv_heads * head_dim, weights.k_sconv),
            v_sconv: ShortConv::new(kv_heads * head_dim, weights.v_sconv),
            hidden,
            rms_norm_eps: config.rms_norm_eps,
            log_scaling: config.log_scaling,
            weights,
        }
    }

    /// The state a sequence starts from: empty convolution windows and no keys.
    pub fn cache(&self) -> AttentionCache {
        AttentionCache {
            k_sconv: self.k_sconv.state(),
            v_sconv: self.v_sconv.state(),
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// `[queries, hidden]` in and out, continuing from `cache` and leaving this
    /// call's keys, values and convolution windows behind in it.
    pub fn forward(&self, cache: &mut AttentionCache, x: &[f32]) -> Vec<f32> {
        self.attend(cache, x, self.log_scaling, self.log_scaling)
    }

    /// [`Attention::forward`], with log scaling's two multiplications named
    /// apart.
    ///
    /// The reference applies the same `tau` to the queries and to the mask's
    /// unmasked entries: both or neither. They are separable, and either one
    /// alone is a plausible misreading that leaves a model which still
    /// generates, so the tests drive this to show that neither alone reproduces
    /// mlx-vlm.
    fn attend(
        &self,
        cache: &mut AttentionCache,
        x: &[f32],
        on_queries: Option<LogScaling>,
        on_biases: Option<LogScaling>,
    ) -> Vec<f32> {
        assert_eq!(
            x.len() % self.hidden,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.hidden
        );
        let (heads, head_dim) = (self.sdpa.heads, self.sdpa.head_dim);
        let offset = cache.keys.len() / (self.sdpa.kv_heads * head_dim);

        let project = |weight| linear(x, weight, self.hidden);
        let norm = |x: &[f32], weight| rms_norm(x, weight, self.rms_norm_eps);

        // The key is normed after its convolution, not before, and the value is
        // convolved and never normed.
        let k = self
            .k_sconv
            .forward(&mut cache.k_sconv, &project(self.weights.k_proj), None);
        let v = self
            .v_sconv
            .forward(&mut cache.v_sconv, &project(self.weights.v_proj), None);
        let mut q = split_heads(
            &norm(&project(self.weights.q_proj), self.weights.q_norm),
            heads,
            head_dim,
        );
        let rel = project(self.weights.r_proj);

        cache.keys.extend(norm(&k, self.weights.k_norm));
        cache.values.extend(v);
        let keys = cache.keys.len() / (self.sdpa.kv_heads * head_dim);

        let mut mask = self.mask.forward(&rel, 1, heads, offset, keys);
        let queries = x.len() / self.hidden;
        if let Some(log) = on_queries {
            for (q, tau) in q.chunks_exact_mut(head_dim).zip(taus(log, queries, offset)) {
                for q in q {
                    *q *= tau;
                }
            }
        }
        // An entry the mask ruled out keeps the constant it carries: scaling
        // -1e30 would overflow, and it rules the key out at any magnitude.
        if let Some(log) = on_biases {
            for (row, tau) in mask.chunks_exact_mut(keys).zip(taus(log, queries, offset)) {
                for bias in row.iter_mut().filter(|entry| !is_masked(**entry)) {
                    *bias *= tau;
                }
            }
        }

        let out = self.sdpa.forward(
            &q,
            &split_heads(&cache.keys, self.sdpa.kv_heads, head_dim),
            &split_heads(&cache.values, self.sdpa.kv_heads, head_dim),
            &mask,
        );
        linear(
            &merge_heads(&out, heads, head_dim),
            self.weights.o_proj,
            heads * head_dim,
        )
    }
}

/// One `tau` per query of a call, repeated for as many heads as ask for it.
///
/// Both tensors log scaling touches are `[heads, queries, stride]` with the
/// query minor, so cycling over the queries walks their rows in order.
fn taus(log: LogScaling, queries: usize, offset: usize) -> impl Iterator<Item = f32> {
    let taus: Vec<f32> = (0..queries).map(|i| log.tau(offset + i)).collect();
    taus.into_iter().cycle()
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

/// The inverse of [`split_heads`]: `[heads, rows, head_dim]` back to `[rows,
/// heads * head_dim]`, which is what `o_proj` reads.
pub fn merge_heads(x: &[f32], heads: usize, head_dim: usize) -> Vec<f32> {
    let stride = heads * head_dim;
    assert_eq!(
        x.len() % stride,
        0,
        "{} values are not whole rows of {stride}",
        x.len()
    );
    let rows = x.len() / stride;

    let mut out = Vec::with_capacity(x.len());
    for row in 0..rows {
        for head in 0..heads {
            out.extend_from_slice(&x[(head * rows + row) * head_dim..][..head_dim]);
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, deviation};
    use crate::mask::MASKED;

    /// Synthetic layers and the sequences mlx-vlm ran through them, from
    /// `just dump-attention-fixture`.
    const FIXTURE: &str = "attention.safetensors";

    /// Each case, and the weight set it was built on. `global_inert` and
    /// `global_scaled` share theirs and differ only in their floor, which is
    /// what makes their disagreement log scaling and nothing else.
    const SYNTHETIC: [(&str, &str); 3] = [
        ("sliding", "sliding"),
        ("global_inert", "global"),
        ("global_scaled", "global"),
    ];

    /// The synthetic cases are float32 end to end, so only summation order
    /// separates this from MLX — the same bound, for the same reason, as the
    /// RMSNorm, MLP, sconv and mask cases. They run a longer chain than any of
    /// those, though: five projections, two convolutions, two norms, a mask and
    /// a softmax, each feeding the next, so the same bound holds less of it in
    /// reserve. Worst observed when this landed: 4.1e-7, a factor of two in
    /// hand, against a weakest mutation of 2.9e-2 five decades above.
    const TOLERANCE: f32 = 1e-6;

    /// The recorded step ran in bfloat16 on trained numbers, so its output is
    /// this computation rounded once on the way out, and the tolerance is that
    /// quantum rather than an arithmetic one: 2^-9 = 2.0e-3 relative, measured
    /// against the tensor's largest value rather than each entry's own
    /// magnitude, which puts it slightly above the ceiling. The same bound, for
    /// the same reason, as the trained masks.
    ///
    /// Worst observed when this landed: 2.8e-3 on layer 5. The weakest mutation
    /// these tests rely on catching, the conventional `1/sqrt(head_dim)` scale,
    /// moves the answer by 1.0 — over two decades above this bound.
    const TRAINED_TOLERANCE: f32 = 6e-3;

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
            let of = |name: &str| fixture::layer_tensor(&activations, layer, name);

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
                deviation <= TRAINED_TOLERANCE,
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
                deviation > TRAINED_TOLERANCE,
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
                deviation > TRAINED_TOLERANCE,
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

    /// The two reshapes attention brackets itself with. Written out for one
    /// small tensor, because a transpose that is wrong in both directions is a
    /// round trip that still holds.
    #[test]
    fn splitting_and_merging_heads_are_the_transposes_they_claim() {
        let (rows, heads, head_dim) = (2, 3, 2);
        let x: Vec<f32> = (0..(rows * heads * head_dim) as u16)
            .map(f32::from)
            .collect();

        // [rows, heads, head_dim] read down the rows of each head in turn.
        assert_eq!(
            split_heads(&x, heads, head_dim),
            [0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]
        );
        assert_eq!(
            merge_heads(&split_heads(&x, heads, head_dim), heads, head_dim),
            x
        );
    }

    /// One synthetic layer's weights, owned so the borrowed [`AttentionWeights`]
    /// can be handed out repeatedly.
    struct Tensors {
        q_proj: Vec<f32>,
        k_proj: Vec<f32>,
        v_proj: Vec<f32>,
        r_proj: Vec<f32>,
        o_proj: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        k_sconv: Vec<f32>,
        v_sconv: Vec<f32>,
        rel_proj: Vec<f32>,
    }

    impl Tensors {
        fn load(ckpt: &Checkpoint, kind: &str) -> Self {
            let of = |name: &str| fixture::f32s(&fixture::tensor(ckpt, &format!("{kind}.{name}")));
            Self {
                q_proj: of("q_proj.weight"),
                k_proj: of("k_proj.weight"),
                v_proj: of("v_proj.weight"),
                r_proj: of("r_proj.weight"),
                o_proj: of("o_proj.weight"),
                q_norm: of("q_norm.weight"),
                k_norm: of("k_norm.weight"),
                k_sconv: of("k_sconv.conv.weight"),
                v_sconv: of("v_sconv.conv.weight"),
                rel_proj: of("rel_proj"),
            }
        }

        fn view(&self) -> AttentionWeights<'_> {
            AttentionWeights {
                q_proj: &self.q_proj,
                k_proj: &self.k_proj,
                v_proj: &self.v_proj,
                r_proj: &self.r_proj,
                o_proj: &self.o_proj,
                q_norm: &self.q_norm,
                k_norm: &self.k_norm,
                k_sconv: &self.k_sconv,
                v_sconv: &self.v_sconv,
                rel_proj: &self.rel_proj,
            }
        }
    }

    /// One synthetic case: the layer `InklingAttention` was built as, the two
    /// sequences it was driven with, and what it produced for each.
    struct Layer {
        name: String,
        config: AttentionConfig,
        weights: Tensors,
        x: Vec<f32>,
        continue_x: Vec<f32>,
        prefill_out: Vec<f32>,
        continue_out: Vec<f32>,
    }

    impl Layer {
        /// `config` is the `[heads, kv_heads, head_dim, d_rel, sliding,
        /// rel_extent, rms_norm_eps, log_floor, log_alpha]` the dump script
        /// recorded — everything `InklingAttention.__init__` derived from the
        /// config and the layer index, which the shapes do not carry. A
        /// `log_floor` of zero is the `None` a sliding layer gets.
        fn load(case: &str, kind: &str) -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of = |name: &str| fixture::f32s(&fixture::tensor(&ckpt, name));
            let recorded = of(&format!("{case}.config"));
            let &[
                heads,
                kv_heads,
                head_dim,
                d_rel,
                sliding,
                rel_extent,
                eps,
                floor,
                alpha,
            ] = recorded.as_slice()
            else {
                panic!("{case}: config carries nine scalars, got {recorded:?}")
            };

            let weights = Tensors::load(&ckpt, kind);
            assert_eq!(
                weights.rel_proj.len(),
                (d_rel * rel_extent) as usize,
                "{case}: rel_proj against its recorded extent"
            );

            Self {
                name: case.to_string(),
                config: AttentionConfig {
                    heads: heads as usize,
                    kv_heads: kv_heads as usize,
                    head_dim: head_dim as usize,
                    d_rel: d_rel as usize,
                    sliding: sliding as usize,
                    rms_norm_eps: eps,
                    log_scaling: (floor > 0.0).then(|| LogScaling::new(floor, alpha)),
                },
                weights,
                x: of("x"),
                continue_x: of("continue_x"),
                prefill_out: of(&format!("{case}.prefill_out")),
                continue_out: of(&format!("{case}.continue_out")),
            }
        }

        fn all() -> Vec<Self> {
            SYNTHETIC
                .iter()
                .map(|(case, kind)| Self::load(case, kind))
                .collect()
        }

        fn attention(&self) -> Attention<'_> {
            Attention::new(self.config, self.weights.view())
        }

        /// The prefill alone, under weights this layer's own may have been
        /// mutated into.
        fn run(&self, weights: AttentionWeights<'_>) -> Vec<f32> {
            let attention = Attention::new(self.config, weights);
            attention.forward(&mut attention.cache(), &self.x)
        }

        /// The prefill and the continuation, against one cache, as the dump
        /// script drove the reference.
        fn forward(&self) -> (Vec<f32>, Vec<f32>) {
            self.with(self.config.log_scaling, self.config.log_scaling)
        }

        fn with(
            &self,
            on_queries: Option<LogScaling>,
            on_biases: Option<LogScaling>,
        ) -> (Vec<f32>, Vec<f32>) {
            let attention = self.attention();
            let cache = &mut attention.cache();
            (
                attention.attend(cache, &self.x, on_queries, on_biases),
                attention.attend(cache, &self.continue_x, on_queries, on_biases),
            )
        }

        /// The worst of the two calls' deviations. The continuation is where the
        /// cache's offset enters, so a prefill that matched alone would say
        /// nothing about decoding.
        fn deviation(&self, (prefill, rest): &(Vec<f32>, Vec<f32>)) -> f32 {
            deviation(prefill, &self.prefill_out).max(deviation(rest, &self.continue_out))
        }

        /// Which absolute positions the two calls put queries at.
        fn positions(&self) -> std::ops::Range<usize> {
            0..(self.x.len() + self.continue_x.len()) / self.hidden()
        }

        fn hidden(&self) -> usize {
            self.weights.q_proj.len() / (self.config.heads * self.config.head_dim)
        }
    }

    fn synthetic(case: &str) -> Layer {
        Layer::all()
            .into_iter()
            .find(|layer| layer.name == case)
            .unwrap_or_else(|| panic!("no {case} case"))
    }

    #[test]
    fn the_synthetic_layers_reproduce_mlx() {
        for layer in Layer::all() {
            let deviation = layer.deviation(&layer.forward());
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// A sliding layer has no log scaling at all, however the checkpoint sets
    /// `log_scaling_n_floor`: `InklingAttention` reads that field only on a
    /// global layer. The dump script builds the sliding case from the same
    /// floor the `global_scaled` case keeps.
    #[test]
    fn a_sliding_layer_has_no_log_scaling() {
        assert!(synthetic("sliding").config.log_scaling.is_none());
        assert!(synthetic("global_scaled").config.log_scaling.is_some());
    }

    /// Below the floor `tau` is exactly 1, so the whole branch is inert and the
    /// layer is the same layer without it — bit for bit, not within a tolerance.
    ///
    /// This is why the committed activations cannot pin log scaling. The
    /// checkpoint's floor is 128000 tokens; every recorded forward pass runs
    /// here, where a port with no log scaling at all agrees exactly.
    #[test]
    fn log_scaling_is_inert_below_its_floor() {
        let layer = synthetic("global_inert");
        let log = layer.config.log_scaling.expect("a floor");
        for position in layer.positions() {
            assert_eq!(log.tau(position), 1.0, "position {position}");
        }
        assert_eq!(layer.with(None, None), layer.forward());
    }

    /// Past the floor `tau` multiplies the queries **and** the mask's unmasked
    /// entries. Either alone leaves a layer that still runs and still attends,
    /// and neither reproduces the reference.
    #[test]
    fn log_scaling_multiplies_the_queries_and_the_biases_together() {
        let layer = synthetic("global_scaled");
        let log = layer.config.log_scaling.expect("a floor");
        assert!(
            layer.positions().any(|position| log.tau(position) > 1.0),
            "no position clears the floor"
        );

        for (what, halved) in [
            ("neither", layer.with(None, None)),
            ("the queries alone", layer.with(Some(log), None)),
            ("the biases alone", layer.with(None, Some(log))),
        ] {
            let deviation = layer.deviation(&halved);
            assert!(deviation > TOLERANCE, "{what}: deviation {deviation:e}");
        }
    }

    /// `global_inert` and `global_scaled` are one weight set and one input under
    /// two floors, so what separates their recorded outputs is log scaling and
    /// nothing else — which is what makes the test above a test of log scaling
    /// rather than of the layer around it.
    #[test]
    fn the_two_global_cases_differ_only_by_their_floor() {
        let (inert, scaled) = (synthetic("global_inert"), synthetic("global_scaled"));
        assert_eq!(inert.weights.q_proj, scaled.weights.q_proj);
        assert_eq!(inert.weights.rel_proj, scaled.weights.rel_proj);
        assert!(deviation(&inert.prefill_out, &scaled.prefill_out) > TOLERANCE);
    }

    /// The continuation reads the cache: the keys and values of the prefill, and
    /// the last `kernel_size - 1` inputs of each short convolution. A layer that
    /// dropped it would still answer, over its own three tokens.
    #[test]
    fn the_continuation_attends_over_the_cached_prefill() {
        for layer in Layer::all() {
            let attention = layer.attention();
            let fresh = attention.forward(&mut attention.cache(), &layer.continue_x);
            let deviation = deviation(&fresh, &layer.continue_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// The two norms are interchangeable, and no fixture can tell them apart.
    ///
    /// RMSNorm divides a row by its own RMS and then multiplies by its weight,
    /// so a logit contracts the two weights elementwise — `sum_d q_d k_d gq_d
    /// gk_d` — and depends on the pair rather than on which is which. Nothing
    /// downstream sees either one again: the values carry no norm at all.
    ///
    /// Stated as a test because it bounds what this fixture settles. A port
    /// that exchanged them is not wrong, and one that dropped either is caught
    /// below. Held to the tolerance rather than to equality only because the
    /// products are formed in the other order.
    #[test]
    fn exchanging_the_query_and_key_norms_is_a_no_op() {
        for layer in Layer::all() {
            let mut weights = layer.weights.view();
            std::mem::swap(&mut weights.q_norm, &mut weights.k_norm);

            let deviation = deviation(&layer.run(weights), &layer.prefill_out);
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// Their product is not interchangeable with one. A trained RMSNorm weight
    /// sits near 1, so a port that never loaded one is off by a few percent per
    /// channel and still generates.
    #[test]
    fn dropping_either_norm_changes_the_answer() {
        for layer in Layer::all() {
            let ones = vec![1.0; layer.config.head_dim];
            for (what, dropped) in [("q_norm", true), ("k_norm", false)] {
                let mut weights = layer.weights.view();
                *(if dropped {
                    &mut weights.q_norm
                } else {
                    &mut weights.k_norm
                }) = &ones;

                let deviation = deviation(&layer.run(weights), &layer.prefill_out);
                assert!(
                    deviation > TOLERANCE,
                    "{}: {what} deviation {deviation:e}",
                    layer.name
                );
            }
        }
    }

    /// The key's convolution and the value's are separate kernels over separate
    /// cache slots. They have the same shape, so exchanging them runs.
    #[test]
    fn exchanging_the_key_and_value_convolutions_changes_the_answer() {
        for layer in Layer::all() {
            let mut weights = layer.weights.view();
            std::mem::swap(&mut weights.k_sconv, &mut weights.v_sconv);

            let deviation = deviation(&layer.run(weights), &layer.prefill_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }
}
