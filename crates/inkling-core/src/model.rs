//! The whole language model: `InklingModel.__call__`, which is the embedding,
//! forty-two decoder layers against forty-two caches, and a final norm.
//!
//! Nothing here is a new op. What this decides is where the weights live, and
//! that answer governs every later one.
//!
//! # Weights are addressed, never held
//!
//! [`DecoderWeights`](crate::layer::DecoderWeights) takes `&[f32]`. That is
//! right for one layer and impossible for forty-two: Inkling-Small is 276B
//! parameters, so the same slices for the whole stack are about 1.1 TB of
//! float32 against a 512 GiB host, and the mxfp4 checkpoint they were decoded
//! from is 130.6 GiB. The stack cannot eagerly dequantise, and there are two
//! ways not to.
//!
//! **Dequantise a layer at a time, transiently.** Decode a layer's weights, run
//! it, drop them. It bounds the peak, but at the wrong granularity: a MoE
//! layer's routed bank is 256 experts, 25 GB of float32, of which one token
//! reaches six. Forty layers of that is a terabyte of decoding per forward pass
//! to use two percent of what it produced, and a peak set by a bank rather than
//! by a matmul.
//!
//! **Keep the checkpoint packed and decode what is touched**, which is what
//! this does. [`Experts`](crate::layer::Experts) and [`Embed`] already work this
//! way — an expert is decoded when a token routes to it, a row of the embedding
//! table when a token asks for it — and both say the same thing about the
//! interface: a weight is reached through an index, not handed over as a slice.
//! Extending that to the projections means the packed bytes stay mapped and a
//! matmul decodes into a [`Scratch`](crate::quant::Scratch) the pass allocated
//! once. The peak is then one layer's working set, the 250 experts no token
//! chose are never read at all, and the whole of the decision is expressed by
//! [`ModelWeights`] taking an index rather than by anything in the stack.
//!
//! # What that costs, stated plainly
//!
//! A projection is not an expert. Every token touches all of `q_proj`, so
//! "decode what is touched" decodes it whole, on every layer, on every forward
//! pass: 9.0 GB of projections — 176 MB of attention across 42 layers, plus the
//! two dense FFNs at 805 MB each — and 32 GB of experts, the eight a single
//! token routes to on each of the 40 MoE layers. Call it 41 GB of
//! dequantisation to decode one token. That is ruinous for decode throughput,
//! and it is ruinous under per-layer dequantisation too, only more so: the same
//! pass would decode all 256 experts of every layer instead of eight, for 1 TB.
//!
//! Neither option is the answer to throughput, because the answer is not to
//! decode at all. MLX never materialises a float weight: `mx.quantized_matmul`
//! and `mx.gather_qmm` multiply against the packed codes. Keeping the
//! checkpoint packed is what leaves room for that — the Metal backend replaces
//! decode-then-multiply with a quantised matmul over the same bytes, and
//! nothing above this line moves. Dequantising per layer would instead fix
//! "the weights are float32 somewhere" into the interface, which is the
//! assumption a quantised kernel exists to remove.
//!
//! # The final norm is not the stack's
//!
//! `InklingModel.__call__` ends `return h if skip_final_norm else self.norm(h)`,
//! but its only caller passes the flag: `LanguageModel.__call__` takes
//! `skip_final_norm=True` and applies `self.model.norm` itself, one line later,
//! on the way to the logits. The two values are not interchangeable — the
//! pre-norm state is what `return_hidden` hands the MTP heads and what
//! `speculative_verify_hidden` returns, while the normed one exists to be turned
//! into logits — so they are two methods here, [`Model::forward`] and
//! [`Model::final_norm`], rather than one method behind a flag.
//!
//! Pinned to mlx-vlm by `reference/fixtures/stack.safetensors`, a five-layer
//! synthetic model driven through the reference's own `InklingModel`, and by the
//! recorded `layers_out` and `norm_out` of
//! `reference/fixtures/layer_activations.safetensors` for the trained
//! forty-two.

use crate::attention::AttentionConfig;
use crate::config::TextConfig;
use crate::embed::Embed;
use crate::layer::{DecoderCache, Hidden, Passed};
use crate::ops::rms_norm;

/// The model's weights, reached through an index rather than held as slices.
///
/// Both methods are asked a question about one index and answer with values
/// that outlive nothing: a row of the embedding table, and a hidden state. What
/// an implementation decodes to answer, and what it drops afterwards, is its
/// own — which is the point. See the module documentation for why the stack
/// cannot be handed weights instead.
pub trait ModelWeights {
    /// Row `id` of `embed_tokens`, `[hidden]` long.
    fn embedding_row(&self, id: usize) -> Vec<f32>;

    /// Take the last `rows` timesteps back out of everything a sequence has
    /// left behind, wherever that is.
    ///
    /// **One call rather than two, because a sequence's state is in two places
    /// and both have to move.** `cache` is what this side holds and a backend
    /// running the layers holds the rest — its own key spans and its own
    /// convolution windows — so a caller that rewound the cache and forgot the
    /// backend would leave a sequence whose position is one thing here and
    /// another there, and which still answers.
    fn rewind(&self, cache: &mut ModelCache, rows: usize) {
        cache.rewind(rows);
    }

    /// Layer `index` over `[tokens, hidden]`, continuing from `cache` and
    /// leaving this call's keys and convolution windows behind in it.
    ///
    /// Running the layer rather than lending it is what keeps a layer's decoded
    /// weights from having to outlive the call that touched them.
    ///
    /// **The state between two layers need not be a value here**, which is what
    /// [`Hidden`] is: a backend that runs layer `index` and will run `index + 1`
    /// can leave what the first produced where the second reads it and answer
    /// with a count. The stack asks in order and hands each answer straight to
    /// the next layer, so the only thing it needs of a hidden state it never
    /// sees is that the last layer's is a value again.
    fn run_layer(&self, index: usize, cache: &mut DecoderCache, x: Hidden<'_>) -> Passed;
}

/// Everything one sequence carries between calls to the model: one
/// [`DecoderCache`] per layer, which is what `LanguageModel.make_cache`
/// allocates as `[CacheList(KVCache(), ArraysCache(4)) for _ in layers]`.
#[derive(Debug, Clone)]
pub struct ModelCache {
    layers: Vec<DecoderCache>,
}

impl ModelCache {
    /// The state a sequence starts from, built from the config alone — every
    /// layer's window widths and key strides are shapes, and asking a layer for
    /// them would mean decoding it first.
    pub fn new(config: &TextConfig) -> Self {
        Self::speculating(config, 0)
    }

    /// The same, able to give back `slack` timesteps in every layer.
    ///
    /// What that costs is `slack` more timesteps in each of a layer's four
    /// convolution windows and nothing anywhere else — 21 KB a layer at the
    /// checkpoint's widths and a slack of eight, which is a tenth of what the
    /// same layer's keys cost at 64 tokens. A generation that speculates
    /// nothing asks for none, so a decode step carries what it always carried.
    pub fn speculating(config: &TextConfig, slack: usize) -> Self {
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|layer| {
                    DecoderCache::speculating(
                        AttentionConfig::for_layer(config, layer),
                        config.hidden_size,
                        config.sconv_kernel_size,
                        slack,
                    )
                })
                .collect(),
        }
    }

    /// One layer's state, for a caller that runs a layer rather than the stack
    /// — which the multi-token prediction heads are: a head is a decoder layer
    /// and carries a decoder layer's cache. See [`crate::mtp`].
    pub fn layer(&mut self, index: usize) -> &mut DecoderCache {
        &mut self.layers[index]
    }

    /// Take back the last `rows` timesteps in every layer, which is what a
    /// speculative round does with the tokens the model did not agree with.
    ///
    /// **Every layer or none.** A stack rewound unevenly would attend over a
    /// different number of keys per layer for the same position, and would
    /// still answer.
    pub fn rewind(&mut self, rows: usize) {
        for layer in &mut self.layers {
            layer.rewind(rows);
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

/// The model around its layers: `embed_tokens`, the optional `embed_norm`, and
/// the final `norm`.
///
/// The layers themselves are not here — see [`ModelWeights`].
#[derive(Debug, Clone, Copy)]
pub struct Model<'a> {
    layers: usize,
    embed: Embed<'a>,
    norm: &'a [f32],
    rms_norm_eps: f32,
}

impl<'a> Model<'a> {
    /// `embed_norm` is the weight of the norm the embedding may end with, which
    /// `use_embed_norm` decides the existence of; `norm` is the final one.
    pub fn new(config: &TextConfig, embed_norm: Option<&'a [f32]>, norm: &'a [f32]) -> Self {
        assert_eq!(
            embed_norm.is_some(),
            config.use_embed_norm,
            "an embed_norm weight against what use_embed_norm asks for"
        );
        if let Some(weight) = embed_norm {
            assert_eq!(weight.len(), config.hidden_size, "embed_norm");
        }
        assert_eq!(norm.len(), config.hidden_size, "the final norm");

        Self {
            layers: config.num_hidden_layers,
            embed: Embed::new(embed_norm, config.rms_norm_eps),
            norm,
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The width every hidden state this model passes around is a row of.
    pub fn hidden(&self) -> usize {
        self.norm.len()
    }

    /// `[tokens]` ids in, the `[tokens, hidden]` hidden state the last layer
    /// produced out — *before* the final norm, which is
    /// [`Model::final_norm`]'s.
    pub fn forward(
        &self,
        cache: &mut ModelCache,
        ids: &[usize],
        weights: &impl ModelWeights,
    ) -> Vec<f32> {
        assert_eq!(
            cache.layers.len(),
            self.layers,
            "a cache per layer of the model"
        );

        let mut h = Passed::Rows(self.embed.forward(ids, |id| weights.embedding_row(id)));
        for (index, cache) in cache.layers.iter_mut().enumerate() {
            h = weights.run_layer(index, cache, h.handed());
        }
        h.rows()
    }

    /// The final `norm`, which the stack leaves for its caller to apply — see
    /// the module documentation for why the reference does the same.
    pub fn final_norm(&self, h: &[f32]) -> Vec<f32> {
        rms_norm(h, self.norm, self.rms_norm_eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{self, LayerTensors, Stack, deviation};
    use crate::profile::{self, Op};

    /// The synthetic stack is float32 end to end, so only summation order
    /// separates it from MLX — the same bound, for the same reason, as the
    /// synthetic layer it repeats five of.
    ///
    /// Five layers do not cost five times one layer's error, and measurably do
    /// not: worst observed when this landed, 4.0e-7, against the single
    /// synthetic layer's 5.3e-7. What accumulates in absolute terms is measured
    /// here against a tensor that grew with it — the stack's output peaks at
    /// 45.5 where a layer's input peaks at 4.1 — and [`deviation`] scales by
    /// that peak. The trained forty-two say the same thing at a larger scale;
    /// see `LAYER_TOLERANCE` in `tests/real_checkpoint.rs`.
    ///
    /// A factor of two and a half in hand, against the weakest mutation these
    /// tests rely on catching — the MoE pair of layers exchanged — at 6.7e-1,
    /// six decades above.
    const TOLERANCE: f32 = 1e-6;

    /// What the reference produced for each of the two calls it drove the
    /// synthetic stack with, beside the weights [`Stack`] holds.
    struct Recorded {
        calls: [Call; 2],
    }

    /// One call's two answers: the hidden state the layers produced, and what
    /// the final norm made of it.
    struct Call {
        what: &'static str,
        layers_out: Vec<f32>,
        norm_out: Vec<f32>,
    }

    impl Recorded {
        fn load() -> Self {
            let ckpt = fixture::open(fixture::STACK);
            let call = |what: &'static str| Call {
                what,
                layers_out: fixture::f32s(&fixture::tensor(&ckpt, &format!("{what}.layers_out"))),
                norm_out: fixture::f32s(&fixture::tensor(&ckpt, &format!("{what}.norm_out"))),
            };
            Self {
                calls: [call("prefill"), call("continue")],
            }
        }

        /// The worst deviation of either call, over both the pre-norm and the
        /// normed answer. The continuation is where the cache enters, so a
        /// prefill that matched alone would say nothing about decoding.
        fn deviation(&self, stack: &Stack) -> f32 {
            let model = stack.model();
            forward(stack)
                .iter()
                .zip(&self.calls)
                .fold(0.0f32, |worst, (got, want)| {
                    worst
                        .max(deviation(got, &want.layers_out))
                        .max(deviation(&model.final_norm(got), &want.norm_out))
                })
        }
    }

    /// The prefill and the continuation, against one cache, as the dump script
    /// drove the reference.
    fn forward(stack: &Stack) -> [Vec<f32>; 2] {
        let model = stack.model();
        let cache = &mut ModelCache::new(&stack.config);
        [
            model.forward(cache, &stack.ids, stack),
            model.forward(cache, &stack.continue_ids, stack),
        ]
    }

    #[test]
    fn the_synthetic_stack_reproduces_mlx() {
        let deviation = Recorded::load().deviation(&Stack::load());
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
        assert!(
            deviation > 0.0,
            "float32 summation order cannot agree exactly"
        );
    }

    /// Adjacent layers a stack could exchange and still run: same attention
    /// config and same MLP, and so the same shapes throughout.
    ///
    /// Any other pair differs in shape, so exchanging it trips an assertion
    /// inside the layer rather than changing an answer — which would say
    /// nothing about order. Derived from the config rather than listed, so that
    /// a fixture regenerated over a different layer plan cannot leave the pairs
    /// behind; `dump_stack_fixture.py` derives the same two and checks that
    /// each of them moves the reference's own answer.
    fn interchangeable(config: &TextConfig) -> Vec<(usize, usize)> {
        let pairs: Vec<(usize, usize)> = (1..config.num_hidden_layers)
            .map(|b| (b - 1, b))
            .filter(|&(a, b)| {
                config.layer_is_sliding(a) == config.layer_is_sliding(b)
                    && config.layer_is_dense(a) == config.layer_is_dense(b)
            })
            .collect();
        let kinds: Vec<bool> = pairs
            .iter()
            .map(|&(a, _)| config.layer_is_dense(a))
            .collect();
        assert!(
            kinds.contains(&true) && kinds.contains(&false),
            "the exchangeable pairs {pairs:?} cover only one MLP"
        );
        pairs
    }

    /// What a fixture of one layer cannot say: that the layers run in order,
    /// each against its own weights. Exchanging two of them leaves a stack that
    /// still runs — the same layers, the same caches, the same arithmetic — and
    /// only the numbers say otherwise.
    #[test]
    fn running_two_layers_out_of_order_changes_the_answer() {
        let recorded = Recorded::load();
        for (a, b) in interchangeable(&Stack::load().config) {
            let deviation = recorded.deviation(&Stack::load().exchanging(a, b));
            assert!(
                deviation > TOLERANCE,
                "exchanging layers {a} and {b} deviates by only {deviation:e}"
            );
        }
    }

    /// The stack covers both MLPs and both attentions, which is what makes it a
    /// stack rather than one layer repeated. A fixture that quietly stopped
    /// doing so would leave the per-layer wiring untested.
    #[test]
    fn the_synthetic_stack_covers_both_mlps_and_both_attentions() {
        let stack = Stack::load();
        let indices = 0..stack.config.num_hidden_layers;
        assert_eq!(stack.layers().len(), indices.len(), "a layer per index");

        let dense: Vec<bool> = stack.layers().iter().map(LayerTensors::is_dense).collect();
        let sliding: Vec<bool> = indices.map(|i| stack.config.layer_is_sliding(i)).collect();
        for (what, kinds) in [("MLPs", dense), ("attentions", sliding)] {
            assert_eq!(
                kinds
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                2,
                "the stack covers both {what}: {kinds:?}"
            );
        }
    }

    /// The two sets of head fields differ in this config, so a sliding layer's
    /// keys are a different width from a global layer's. A stack that read one
    /// set for every layer would build a cache of the wrong stride.
    #[test]
    fn the_cache_takes_each_layers_own_attention_config() {
        let stack = Stack::load();
        let config = &stack.config;
        assert_ne!(
            config.swa_num_key_value_heads * config.swa_head_dim,
            config.num_key_value_heads * config.head_dim,
            "a config whose two sets agreed could not settle this"
        );

        let cache = ModelCache::new(config);
        assert_eq!(cache.len(), config.num_hidden_layers);
        assert!(!cache.is_empty());
    }

    /// The continuation reads the cache, so a stack that allocated a fresh one
    /// per call would still answer — over its own tokens, from no keys.
    #[test]
    fn the_continuation_reads_what_the_prefill_cached() {
        let stack = Stack::load();
        let recorded = Recorded::load();

        let fresh = stack.model().forward(
            &mut ModelCache::new(&stack.config),
            &stack.continue_ids,
            &stack,
        );
        let deviation = deviation(&fresh, &recorded.calls[1].layers_out);
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// The stack returns the state *before* the final norm, which is what
    /// `LanguageModel.__call__` asks it for. Returning the normed state instead
    /// leaves a model that still runs and still makes logits, and would hand the
    /// MTP heads the wrong tensor.
    #[test]
    fn the_stack_stops_before_the_final_norm() {
        let stack = Stack::load();
        let model = stack.model();
        for got in forward(&stack).iter().zip(&Recorded::load().calls) {
            let (got, want) = got;
            let normed = model.final_norm(got);
            let deviation = deviation(&normed, &want.layers_out);
            assert!(
                deviation > TOLERANCE,
                "{}: the norm moves the answer by only {deviation:e}",
                want.what
            );
        }
    }

    /// The property the whole speculative loop rests on, stated on the stack:
    /// tokens fed, taken back and replaced are the same sequence as tokens that
    /// were never fed.
    ///
    /// Exact equality rather than a tolerance, for the reason
    /// [`crate::generate`]'s split test demands it — both sides multiply the
    /// same numbers in the same order. That is what makes speculation a latency
    /// optimisation rather than an approximation: what a rejected token leaves
    /// behind in five layers of keys and twenty convolution windows is nothing
    /// at all.
    ///
    /// Every layer, because a rewind that missed one is a stack that still
    /// answers — see [`ModelCache::rewind`] — and the values are what say
    /// otherwise.
    #[test]
    fn rewinding_the_tokens_a_pass_fed_leaves_the_stack_where_they_found_it() {
        let stack = Stack::load();
        let sequence = stack.sequence();
        let model = stack.model();

        for split in 1..sequence.len() {
            let taken = sequence.len() - split;
            let wrong: Vec<usize> = sequence[split..]
                .iter()
                .map(|id| (id + 1) % stack.config.vocab_size)
                .collect();
            assert_ne!(wrong, sequence[split..], "tokens a rewind has to undo");

            let cache = &mut ModelCache::speculating(&stack.config, taken);
            model.forward(cache, &sequence[..split], &stack);
            model.forward(cache, &wrong, &stack);
            cache.rewind(taken);
            let after = model.forward(cache, &sequence[split..], &stack);

            let clean = &mut ModelCache::speculating(&stack.config, taken);
            model.forward(clean, &sequence[..split], &stack);
            let want = model.forward(clean, &sequence[split..], &stack);
            assert_eq!(after, want, "{taken} tokens taken back at {split}");
        }
    }

    /// A stack that never speculates asks for no slack, and asking one that did
    /// not to give a token back is refused rather than answered out of a window
    /// that no longer holds it.
    #[test]
    #[should_panic(expected = "a rewind of 1 against 0")]
    fn rewinding_a_cache_that_kept_no_slack_is_refused() {
        let stack = Stack::load();
        let cache = &mut ModelCache::new(&stack.config);
        stack.model().forward(cache, &stack.ids, &stack);
        cache.rewind(1);
    }

    /// What a forward pass is made of, counted rather than described.
    ///
    /// [`crate::profile`] exists to say where a step's time goes, and it can
    /// only say it about operations something opened a scope around. This is
    /// what says the scopes are where they are claimed to be: a stack of five
    /// layers runs four RMSNorms and four short convolutions a layer, one mask
    /// and one attention step, and a residual add for each of its two residuals
    /// — plus the embedding's own norm, and one more add on each layer that
    /// routes, where the routed and shared halves are summed.
    ///
    /// Exact counts rather than "was reached", because the mistake worth
    /// catching is a scope on the wrong side of a loop. One opened per row of a
    /// call, or per expert rather than per layer, would still charge the right
    /// op and would make every figure in the table depend on a shape the table
    /// does not print.
    ///
    /// The set is exact too, which makes this fail when an op is instrumented
    /// that this stack reaches — deliberately, because a row appearing in the
    /// table is a change to what the table means.
    #[test]
    fn a_forward_pass_opens_one_scope_per_operation_it_runs() {
        let stack = Stack::load();
        let layers = stack.config.num_hidden_layers as u64;
        let routed = (0..stack.config.num_hidden_layers)
            .filter(|layer| !stack.config.layer_is_dense(*layer))
            .count() as u64;

        profile::take();
        stack
            .model()
            .forward(&mut ModelCache::new(&stack.config), &stack.ids, &stack);
        let profile = profile::take();

        for (op, want) in [
            (Op::RmsNorm, 1 + 4 * layers),
            (Op::Sconv, 4 * layers),
            (Op::Mask, layers),
            (Op::Sdpa, layers),
            (Op::Residual, 2 * layers + routed),
            (Op::Router, routed),
            (Op::Gather, 4 * routed),
        ] {
            assert_eq!(profile.calls(op), want, "{op}");
        }

        // The stack decodes nothing — its weights are the fixture's, already
        // float32 — and it neither dispatches nor samples, so the ops it does
        // not reach are as much of the claim as the ones it does.
        let reached: Vec<Op> = profile.rows().iter().map(|(op, ..)| *op).collect();
        for op in Op::ALL {
            assert_eq!(
                reached.contains(&op),
                matches!(
                    op,
                    Op::RmsNorm
                        | Op::Sconv
                        | Op::Mask
                        | Op::Sdpa
                        | Op::Residual
                        | Op::Router
                        | Op::Gather
                        | Op::Linear
                        | Op::Swiglu
                ),
                "{op}"
            );
        }
    }

    /// `use_embed_norm` says whether the embedding ends with a norm, and a
    /// weight handed in against a config that clears it — or withheld against a
    /// config that sets it — is a checkpoint and a config that disagree.
    #[test]
    #[should_panic(expected = "use_embed_norm")]
    fn withholding_embed_norm_from_a_config_that_asks_for_it_is_refused() {
        let stack = Stack::load();
        assert!(stack.config.use_embed_norm, "the fixture normalises");
        Model::new(&stack.config, None, &stack.norm);
    }

    #[test]
    #[should_panic(expected = "the final norm")]
    fn a_final_norm_of_the_wrong_width_is_refused() {
        let stack = Stack::load();
        Model::new(&stack.config, Some(&stack.embed_norm), &stack.norm[1..]);
    }

    /// A cache is one sequence's, and a cache built for a different model has
    /// the wrong number of layers. Caught here rather than by the layer that
    /// would be handed the wrong one.
    #[test]
    #[should_panic(expected = "a cache per layer")]
    fn a_cache_of_the_wrong_depth_is_refused() {
        let stack = Stack::load();
        let mut shallow = stack.config.clone();
        shallow.num_hidden_layers -= 1;
        stack
            .model()
            .forward(&mut ModelCache::new(&shallow), &stack.ids, &stack);
    }
}
