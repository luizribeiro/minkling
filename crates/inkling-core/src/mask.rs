//! The banded relative-position mask: the additive `[B, H, LQ, S]` tensor
//! attention adds to its logits.
//!
//! It carries three things at once — the causal mask, a sliding layer's window
//! cap, and the learned bias over the last `rel_extent` positions. Inkling has
//! no RoPE, so together with the short convolutions that bias is the whole of
//! what tells attention where a token sits.
//!
//! Pinned to mlx-vlm by `reference/fixtures/mask.safetensors`, whose synthetic
//! cases are float32 throughout, and by the trained `rel_proj` in that bundle
//! against the masks recorded in `layer_activations.safetensors`.

/// The additive constant a masked position carries.
///
/// A magnitude rather than an infinity: a row that is masked end to end still
/// has to leave softmax with finite numbers.
pub const MASKED: f32 = -1e30;

/// Whether an entry was masked rather than biased.
///
/// The threshold is an order of magnitude below [`MASKED`] because the constant
/// does not survive the model's dtype: bfloat16 rounds it to
/// `-1.0002555517425873e30`, so a recorded mask never holds `MASKED` exactly.
/// mlx-vlm tests `mask > -1e29` for the same reason, when log scaling has to
/// rescale the biases and leave the masked entries alone.
pub fn is_masked(entry: f32) -> bool {
    entry <= -1e29
}

/// One attention layer's mask configuration: the learned projection over
/// backward distances, and the window the layer attends over.
#[derive(Debug, Clone, Copy)]
pub struct BandedMask<'a> {
    d_rel: usize,
    rel_extent: usize,
    sliding: usize,
    proj: &'a [f32],
}

impl<'a> BandedMask<'a> {
    /// `proj` is the checkpoint's own `rel_proj`: `[d_rel, rel_extent]`
    /// row-major, one row of per-distance coefficients per relative feature.
    ///
    /// `sliding` is the layer's window, or zero for a global layer.
    /// `InklingAttention` sets a sliding layer's `rel_extent` to the same
    /// `sliding_window_size` it sets the window to, which leaves the band and
    /// the window coincident and the outside-the-band case unreachable; only a
    /// global layer, where the window is zero, ever produces it.
    pub fn new(d_rel: usize, proj: &'a [f32], sliding: usize) -> Self {
        assert_eq!(
            proj.len() % d_rel,
            0,
            "{} coefficients are not whole rows of {d_rel}",
            proj.len()
        );
        let rel_extent = proj.len() / d_rel;
        assert!(rel_extent > 0, "a band needs at least one distance");

        Self {
            d_rel,
            rel_extent,
            sliding,
            proj,
        }
    }

    /// `rel` is `[batch, queries, heads, d_rel]` row-major — the shape `r_proj`
    /// produces, query-major and head-minor, before attention transposes
    /// anything. Out comes `[batch, heads, queries, keys]`.
    ///
    /// `q_offset` is the KV cache's offset: query `i` sits at absolute position
    /// `i + q_offset`, and `keys` counts the whole cached span. During prefill
    /// the offset is zero and every mask is indexed the same whether or not it
    /// is applied, which is what makes dropping it a decode-only bug.
    pub fn forward(
        &self,
        rel: &[f32],
        batch: usize,
        heads: usize,
        q_offset: usize,
        keys: usize,
    ) -> Vec<f32> {
        let stride = batch * heads * self.d_rel;
        assert_eq!(
            rel.len() % stride,
            0,
            "{} values are not whole queries of {stride}",
            rel.len()
        );
        let queries = rel.len() / stride;

        let mut out = vec![0.0; batch * heads * queries * keys];
        for b in 0..batch {
            for h in 0..heads {
                for i in 0..queries {
                    let at = ((b * queries + i) * heads + h) * self.d_rel;
                    let rel = &rel[at..at + self.d_rel];
                    let at = ((b * heads + h) * queries + i) * keys;
                    let position = (i + q_offset) as isize;
                    for (j, out) in out[at..at + keys].iter_mut().enumerate() {
                        *out = self.entry(rel, position - j as isize);
                    }
                }
            }
        }
        out
    }

    /// One query-key pair, from its backward distance. The four cases are
    /// ordered: a position past the window is masked whether or not the band
    /// still covers it, and a position before the sequence starts is masked
    /// before anything indexes the projection with it.
    fn entry(&self, rel: &[f32], dist: isize) -> f32 {
        let Ok(dist) = usize::try_from(dist) else {
            return MASKED;
        };
        if self.sliding > 0 && dist >= self.sliding {
            return MASKED;
        }
        if dist >= self.rel_extent {
            return 0.0;
        }
        rel.iter()
            .zip(self.proj[dist..].iter().step_by(self.rel_extent))
            .map(|(rel, proj)| rel * proj)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::TensorView;
    use crate::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, deviation};

    /// Synthetic cases and the trained projections, from
    /// `just dump-mask-fixture`.
    const FIXTURE: &str = "mask.safetensors";

    /// The synthetic cases, and the branches each was placed to reach. Named
    /// here as well as in the dump script so a case that is retuned until it
    /// stops covering its branch fails rather than passes quietly.
    const SYNTHETIC: [(&str, &[u8]); 5] = [
        ("sliding_window", &[1, 2, 3]),
        ("global_band", &[1, 3, 4]),
        ("decode", &[2, 3]),
        ("prefill", &[1, 3]),
        ("narrow_window", &[1, 2, 3]),
    ];

    /// The synthetic cases are float32 end to end, and this and the Metal
    /// kernel both accumulate the `d_rel` products in float32 in the same
    /// order, so only how each compiler contracts that loop separates them.
    /// 1e-6 is the same bound, for the same reason, as the RMSNorm, MLP and
    /// sconv cases. Worst observed when this landed: 1.5e-7, a couple of ulps
    /// and a factor of seven in hand.
    const TOLERANCE: f32 = 1e-6;

    /// The trained masks cannot be held anywhere near that. The model runs in
    /// bfloat16, so the recorded mask is this computation rounded once on the
    /// way out, and a bias of a few units carries a quantum of a few
    /// hundredths.
    ///
    /// Worst observed when this landed: 2.9e-3 on layer 2, against a ceiling of
    /// 2^-9 = 2.0e-3 — bfloat16's relative quantum, which is measured here
    /// against the largest entry of the tensor rather than each entry's own
    /// magnitude and so comes out slightly above it. The weakest mutation these
    /// tests rely on catching, reading layer 2's `rel_proj` transposed, moves
    /// the answer by 0.93: two decades above this bound, where the bfloat16
    /// noise sits a factor of two below it. What this settles is the
    /// projection's layout and the trained numbers; the synthetic cases settle
    /// the arithmetic.
    const TRAINED_TOLERANCE: f32 = 6e-3;

    /// [`MASKED`] after a round trip through bfloat16, which is how it comes
    /// back out of a mask the model itself computed.
    const BF16_MASKED: f32 = -1.000_255_55e30;

    /// One configuration of the op: the inputs, the scalars `InklingAttention`
    /// would have passed, and the mask mlx-vlm produced from them.
    struct Case {
        name: String,
        batch: usize,
        heads: usize,
        queries: usize,
        d_rel: usize,
        q_offset: usize,
        keys: usize,
        sliding: usize,
        rel_extent: usize,
        rel: Vec<f32>,
        proj: Vec<f32>,
        want: Vec<f32>,
    }

    impl Case {
        /// `rel` is `[batch, queries, heads, d_rel]`, `proj` is `[d_rel,
        /// rel_extent]`, and `config` is the `[q_offset, keys, sliding,
        /// rel_extent]` the dump script recorded.
        fn new(
            name: &str,
            rel: &TensorView<'_>,
            proj: &TensorView<'_>,
            config: &[f32],
            want: Vec<f32>,
        ) -> Self {
            let &[batch, queries, heads, d_rel] = rel.shape() else {
                panic!(
                    "{name}: rel is [batch, queries, heads, d_rel], got {:?}",
                    rel.shape()
                )
            };
            let &[q_offset, keys, sliding, rel_extent] = config else {
                panic!("{name}: config is [q_offset, keys, sliding, rel_extent]")
            };
            let rel_extent = rel_extent as usize;
            assert_eq!(proj.shape(), [d_rel, rel_extent], "{name}: proj");
            assert_eq!(
                want.len(),
                batch * heads * queries * keys as usize,
                "{name}: mask"
            );

            Self {
                name: name.to_string(),
                batch,
                heads,
                queries,
                d_rel,
                q_offset: q_offset as usize,
                keys: keys as usize,
                sliding: sliding as usize,
                rel_extent,
                rel: fixture::f32s(rel),
                proj: fixture::f32s(proj),
                want,
            }
        }

        fn synthetic() -> Vec<Self> {
            let ckpt = fixture::open(FIXTURE);
            SYNTHETIC
                .iter()
                .map(|(name, _)| {
                    let of = |field| fixture::tensor(&ckpt, &format!("{name}.{field}"));
                    let config = fixture::f32s(&of("config"));
                    let extent = config[3] as usize;
                    Self::new(
                        name,
                        &of("rel"),
                        &fixture::tensor(&ckpt, &format!("proj{extent}")),
                        &config,
                        fixture::f32s(&of("mask")),
                    )
                })
                .collect()
        }

        /// The captured layers' own masks, rebuilt from the `r_proj` output
        /// attention fed the kernel and the checkpoint's `rel_proj`.
        fn trained() -> Vec<Self> {
            let ckpt = fixture::open(FIXTURE);
            let activations = fixture::open(ACTIVATIONS);
            CAPTURED_LAYERS
                .iter()
                .map(|&layer| {
                    let of = |name: &str| fixture::layer_tensor(&ckpt, layer, name);
                    let recorded = |name: &str| fixture::layer_tensor(&activations, layer, name);
                    Self::new(
                        &format!("layer{layer}"),
                        &recorded("r_proj_out"),
                        &of("rel_proj"),
                        &fixture::f32s(&of("config")),
                        fixture::f32s(&recorded("mask")),
                    )
                })
                .collect()
        }

        fn with(&self, proj: &[f32], rel: &[f32], q_offset: usize) -> Vec<f32> {
            BandedMask::new(self.d_rel, proj, self.sliding)
                .forward(rel, self.batch, self.heads, q_offset, self.keys)
        }

        fn forward(&self) -> Vec<f32> {
            self.with(&self.proj, &self.rel, self.q_offset)
        }

        /// Which of the four cases a backward distance falls in, written out of
        /// the distance alone so it agrees with the op only where the op is
        /// right.
        fn branch(&self, dist: isize) -> u8 {
            if dist < 0 {
                1
            } else if self.sliding > 0 && dist >= self.sliding as isize {
                2
            } else if dist < self.rel_extent as isize {
                3
            } else {
                4
            }
        }

        /// Every entry of a `[batch, heads, queries, keys]` mask, paired with
        /// the backward distance of the position it sits at.
        fn entries<'t>(&'t self, mask: &'t [f32]) -> impl Iterator<Item = (isize, f32)> + 't {
            mask.iter().enumerate().map(move |(at, value)| {
                let (i, j) = ((at / self.keys) % self.queries, at % self.keys);
                ((i + self.q_offset) as isize - j as isize, *value)
            })
        }

        /// The entries of one branch, which is empty exactly when the case does
        /// not reach it.
        fn branch_entries<'t>(
            &'t self,
            mask: &'t [f32],
            want: u8,
        ) -> impl Iterator<Item = f32> + 't {
            self.entries(mask)
                .filter(move |(dist, _)| self.branch(*dist) == want)
                .map(|(_, value)| value)
        }

        /// The op's answer, checked against mlx-vlm's.
        ///
        /// Masked entries are compared as a pattern and the rest by
        /// [`deviation`]. Splitting the two is what makes the trained masks
        /// usable: their masked entries read back as [`BF16_MASKED`], and one
        /// tensor-wide metric would be measuring that constant's round trip at
        /// a scale thirty orders of magnitude above the biases it buries.
        fn deviation(&self, got: &[f32]) -> f32 {
            assert_eq!(got.len(), self.want.len(), "{}: length", self.name);
            let (mut biased, mut want) = (Vec::new(), Vec::new());
            for (at, (got, expected)) in got.iter().zip(&self.want).enumerate() {
                assert_eq!(
                    is_masked(*got),
                    is_masked(*expected),
                    "{}: entry {at} is {got:e}, reference has {expected:e}",
                    self.name
                );
                if !is_masked(*expected) {
                    biased.push(*got);
                    want.push(*expected);
                }
            }
            deviation(&biased, &want)
        }
    }

    /// `[d_rel, rel_extent]` read as `[rel_extent, d_rel]`. The buffer is the
    /// same length either way, so the mistake indexes in bounds and produces a
    /// mask of the right shape and the wrong numbers.
    fn transposed(values: &[f32], rows: usize) -> Vec<f32> {
        let cols = values.len() / rows;
        (0..cols)
            .flat_map(|c| (0..rows).map(move |r| values[r * cols + c]))
            .collect()
    }

    fn masked_pattern(mask: &[f32]) -> Vec<bool> {
        mask.iter().map(|entry| is_masked(*entry)).collect()
    }

    #[test]
    fn the_synthetic_cases_reproduce_mlx() {
        for case in Case::synthetic() {
            let deviation = case.deviation(&case.forward());
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
        }
    }

    #[test]
    fn the_trained_projections_reproduce_the_reference_masks() {
        let mut worst = 0.0f32;
        for case in Case::trained() {
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

    /// The cases between them have to reach all four branches, or the tests
    /// below that name a branch are testing an empty set.
    #[test]
    fn the_cases_reach_the_branches_they_were_placed_to_reach() {
        let mut reached = Vec::new();
        for (case, want) in Case::synthetic().iter().zip(SYNTHETIC.map(|(_, b)| b)) {
            let mut branches: Vec<u8> = case
                .entries(&case.want)
                .map(|(dist, _)| case.branch(dist))
                .collect();
            branches.sort_unstable();
            branches.dedup();
            assert_eq!(branches, want, "{}", case.name);
            reached.extend(branches);
        }
        reached.sort_unstable();
        reached.dedup();
        assert_eq!(reached, [1, 2, 3, 4]);
    }

    /// Branch 1. A key ahead of the query is masked whatever the band would
    /// have said about it — and it is masked *before* the band is consulted,
    /// which is what keeps a negative distance from indexing the projection.
    #[test]
    fn a_key_the_query_cannot_see_yet_is_masked() {
        for case in Case::synthetic().iter().chain(&Case::trained()) {
            let got = case.forward();
            let causal: Vec<f32> = case.branch_entries(&got, 1).collect();
            assert!(
                causal.iter().all(|entry| *entry == MASKED),
                "{}: {:?}",
                case.name,
                causal.iter().find(|entry| **entry != MASKED)
            );
        }
    }

    /// Branch 2. A sliding layer sets its band to the same width as its window,
    /// so past the window the band has nothing to say either and dropping the
    /// cap would leave a plain zero rather than an obviously wrong bias.
    #[test]
    fn a_key_older_than_the_window_is_masked() {
        for case in Case::synthetic() {
            let got = case.forward();
            let capped: Vec<f32> = case.branch_entries(&got, 2).collect();
            assert!(
                capped.iter().all(|entry| *entry == MASKED),
                "{}: {:?}",
                case.name,
                capped.iter().find(|entry| **entry != MASKED)
            );
        }
    }

    /// Branch 2 before branch 3. The two overlap only when the window is
    /// narrower than the band, which no Inkling layer configures and which the
    /// `narrow_window` case exists to arrange: every distance from the window
    /// edge to the band edge is masked, and would be a learned bias if the
    /// kernel consulted the band first.
    ///
    /// Branches 1 and 2 cannot be ordered against each other at all: a negative
    /// distance is never past a positive window, so no position reaches both.
    #[test]
    fn the_window_cap_outranks_the_band() {
        let case = Case::synthetic()
            .into_iter()
            .find(|case| case.sliding > 0 && case.rel_extent > case.sliding)
            .expect("a case whose window is narrower than its band");

        let got = case.forward();
        let overlap = case.sliding as isize..case.rel_extent as isize;
        let past_the_window: Vec<f32> = case
            .entries(&got)
            .filter(|(dist, _)| overlap.contains(dist))
            .map(|(_, entry)| entry)
            .collect();

        assert!(
            !past_the_window.is_empty(),
            "{}: no distance falls between the window and the band",
            case.name
        );
        assert!(
            past_the_window.iter().all(|entry| *entry == MASKED),
            "{}: {:?}",
            case.name,
            past_the_window.iter().find(|entry| **entry != MASKED)
        );
    }

    /// Branch 3, and the window edge falling strictly inside the key span: the
    /// same row carries masked and biased entries, so a row is never uniformly
    /// one or the other and an implementation that decided per row would fail.
    #[test]
    fn a_sliding_row_holds_both_masked_and_biased_keys() {
        let case = Case::synthetic()
            .into_iter()
            .find(|case| case.name == "sliding_window")
            .expect("the sliding-window case");
        assert!(
            case.keys > case.sliding,
            "{}: the window fills the span",
            case.name
        );

        let got = case.forward();
        for row in got.chunks_exact(case.keys) {
            assert!(row.iter().any(|entry| is_masked(*entry)), "{}", case.name);
            assert!(row.iter().any(|entry| !is_masked(*entry)), "{}", case.name);
        }
    }

    /// Branch 4. In context, outside the band: exactly zero, neither masked nor
    /// a bias. The dump script checks no learned bias is exactly zero, so the
    /// value tells the two apart.
    #[test]
    fn a_key_in_context_but_outside_the_band_contributes_nothing() {
        let case = Case::synthetic()
            .into_iter()
            .find(|case| case.sliding == 0)
            .expect("a global case, the only kind that reaches outside the band");

        let got = case.forward();
        let outside: Vec<f32> = case.branch_entries(&got, 4).collect();
        assert!(!outside.is_empty(), "{}", case.name);
        assert!(outside.iter().all(|entry| *entry == 0.0), "{}", case.name);
        assert!(
            case.branch_entries(&got, 3).all(|entry| entry != 0.0),
            "{}: a learned bias is indistinguishable from outside the band",
            case.name
        );
    }

    /// The captured layers cover both attention configurations, so the trained
    /// cases are a global layer's band as well as a sliding layer's window. The
    /// dump script refuses a set that stops covering both; stated again here
    /// because a fixture that quietly lost its global layer would leave every
    /// test below still passing over sliding layers alone.
    #[test]
    fn the_trained_cases_cover_a_sliding_and_a_global_layer() {
        let trained = Case::trained();
        let (global, sliding): (Vec<&Case>, Vec<&Case>) =
            trained.iter().partition(|case| case.sliding == 0);

        // A global layer's band outruns a sliding layer's, which is the whole
        // of why branch 4 is reachable on one and not the other.
        let narrowest = global
            .iter()
            .map(|case| case.rel_extent)
            .min()
            .expect("a captured layer is global");
        let widest = sliding
            .iter()
            .map(|case| case.rel_extent)
            .max()
            .expect("a captured layer is sliding");
        assert!(
            narrowest > widest,
            "a global band of {narrowest} does not outrun a sliding one of {widest}"
        );
    }

    /// A sliding layer sets `rel_extent` to its own `sliding_window_size`, so
    /// the band ends exactly where the window does and nothing is ever in
    /// context and outside the band. Branch 4 is a global layer's alone.
    #[test]
    fn a_sliding_layer_never_reaches_outside_its_band() {
        let sliding: Vec<Case> = Case::synthetic()
            .into_iter()
            .chain(Case::trained())
            .filter(|case| case.sliding > 0 && case.rel_extent == case.sliding)
            .collect();
        assert!(
            !sliding.is_empty(),
            "no case is configured as a sliding layer"
        );

        for case in sliding {
            assert_eq!(
                case.branch_entries(&case.want, 4).count(),
                0,
                "{}",
                case.name
            );
        }
    }

    /// The bias is indexed by the backward distance and not by the key's
    /// position, so a mask built without the cache's offset is wrong the moment
    /// decoding starts — and right, entry for entry, throughout prefill, which
    /// is where a port is usually checked.
    #[test]
    fn ignoring_the_cache_offset_changes_the_answer() {
        let mut checked = 0;
        for case in Case::synthetic() {
            if case.q_offset == 0 {
                continue;
            }
            let dropped = case.with(&case.proj, &case.rel, 0);
            assert_ne!(
                masked_pattern(&dropped),
                masked_pattern(&case.want),
                "{}",
                case.name
            );
            checked += 1;
        }
        assert!(checked > 0, "no case has a nonzero offset");
    }

    /// `rel_proj` is `[d_rel, rel_extent]`: a row per relative feature, a
    /// column per distance. Read the other way up it still covers the buffer
    /// and still produces a mask of the right shape.
    #[test]
    fn transposing_the_projection_changes_the_answer() {
        for case in Case::synthetic().iter().chain(&Case::trained()) {
            let swapped = transposed(&case.proj, case.d_rel);
            let deviation = case.deviation(&case.with(&swapped, &case.rel, case.q_offset));
            assert!(
                deviation > TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
        }
    }

    /// `rel` is `[batch, queries, heads, d_rel]` — query-major, head-minor,
    /// which is the opposite of the head-major layout every other tensor in
    /// attention uses by the time the mask is built.
    #[test]
    fn transposing_the_query_and_head_axes_of_rel_changes_the_answer() {
        for case in Case::synthetic().iter().chain(&Case::trained()) {
            if case.queries == 1 || case.heads == 1 {
                continue;
            }
            let swapped: Vec<f32> = case
                .rel
                .chunks_exact(case.queries * case.heads * case.d_rel)
                .flat_map(|batch| transposed(batch, case.queries).into_iter())
                .collect();
            let deviation = case.deviation(&case.with(&case.proj, &swapped, case.q_offset));
            assert!(
                deviation > TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
        }
    }

    /// How the masked constant is handled, stated as a test.
    ///
    /// The synthetic cases are float32, so mlx-vlm writes `-1e30` into them
    /// unchanged and they pin the constant exactly. The trained masks were
    /// computed in bfloat16, which has eight bits of mantissa and no `-1e30`,
    /// so they hold [`BF16_MASKED`] instead. Rather than widen a tolerance to
    /// swallow a 2.6e26 disagreement, or emit a constant chosen to match one
    /// checkpoint's dtype, masked entries are compared through [`is_masked`] —
    /// mlx-vlm's own test, and the only property of the constant that attention
    /// depends on.
    #[test]
    fn a_masked_entry_is_the_constant_the_reference_wrote() {
        assert_ne!(MASKED, BF16_MASKED, "the round trip would be invisible");

        for case in Case::synthetic() {
            let masked: Vec<f32> = case
                .want
                .iter()
                .copied()
                .filter(|entry| is_masked(*entry))
                .collect();
            assert!(!masked.is_empty(), "{}", case.name);
            assert!(masked.iter().all(|entry| *entry == MASKED), "{}", case.name);
        }

        for case in Case::trained() {
            let masked: Vec<f32> = case
                .want
                .iter()
                .copied()
                .filter(|entry| is_masked(*entry))
                .collect();
            assert!(!masked.is_empty(), "{}", case.name);
            assert!(
                masked.iter().all(|entry| *entry == BF16_MASKED),
                "{}: {:?}",
                case.name,
                masked.iter().find(|entry| **entry != BF16_MASKED)
            );
        }
    }
}
