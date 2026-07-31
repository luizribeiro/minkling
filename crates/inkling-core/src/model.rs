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
use crate::layer::DecoderCache;
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

    /// Layer `index` over `[tokens, hidden]`, continuing from `cache` and
    /// leaving this call's keys and convolution windows behind in it.
    ///
    /// Running the layer rather than lending it is what keeps a layer's decoded
    /// weights from having to outlive the call that touched them.
    fn run_layer(&self, index: usize, cache: &mut DecoderCache, x: &[f32]) -> Vec<f32>;
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
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|layer| {
                    DecoderCache::new(
                        AttentionConfig::for_layer(config, layer),
                        config.hidden_size,
                        config.sconv_kernel_size,
                    )
                })
                .collect(),
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

        let mut h = self.embed.forward(ids, |id| weights.embedding_row(id));
        for (index, cache) in cache.layers.iter_mut().enumerate() {
            h = weights.run_layer(index, cache, &h);
        }
        h
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
    use crate::config::Config;
    use crate::fixture::{self, LayerTensors, deviation, indices};
    use crate::layer::DecoderLayer;

    /// A four-layer synthetic model and the two calls mlx-vlm drove it with,
    /// from `just dump-stack-fixture`.
    const FIXTURE: &str = "stack.safetensors";

    /// The config that model was built from, in the checkpoint's own spelling so
    /// that the same JSON stands the reference and this port up.
    const FIXTURE_CONFIG: &str = "stack.json";

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

    /// The whole synthetic model: the config it was built from, its two
    /// stack-level norms, its embedding table, its layers, and what the
    /// reference produced.
    struct Stack {
        config: TextConfig,
        embed_norm: Vec<f32>,
        norm: Vec<f32>,
        table: Vec<f32>,
        layers: Vec<LayerTensors>,
        ids: Vec<usize>,
        continue_ids: Vec<usize>,
        calls: [Call; 2],
    }

    /// One call's two answers: the hidden state the layers produced, and what
    /// the final norm made of it.
    struct Call {
        what: &'static str,
        layers_out: Vec<f32>,
        norm_out: Vec<f32>,
    }

    impl Stack {
        fn load() -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of = |name: &str| fixture::f32s(&fixture::tensor(&ckpt, name));
            let config = serde_json::from_str::<Config>(&fixture::read(FIXTURE_CONFIG))
                .expect("the recorded config parses")
                .text_config;
            let call = |what: &'static str| Call {
                what,
                layers_out: of(&format!("{what}.layers_out")),
                norm_out: of(&format!("{what}.norm_out")),
            };

            Self {
                layers: (0..config.num_hidden_layers)
                    .map(|layer| LayerTensors::load(&ckpt, &format!("layers.{layer}")))
                    .collect(),
                embed_norm: of("embed_norm.weight"),
                norm: of("norm.weight"),
                table: of("embed_tokens.weight"),
                ids: indices(&fixture::tensor(&ckpt, "input_ids")),
                continue_ids: indices(&fixture::tensor(&ckpt, "continue_ids")),
                calls: [call("prefill"), call("continue")],
                config,
            }
        }

        fn model(&self) -> Model<'_> {
            Model::new(&self.config, Some(&self.embed_norm), &self.norm)
        }

        /// The prefill and the continuation, against one cache, as the dump
        /// script drove the reference.
        fn forward(&self, weights: &impl ModelWeights) -> [Vec<f32>; 2] {
            let model = self.model();
            let cache = &mut ModelCache::new(&self.config);
            [
                model.forward(cache, &self.ids, weights),
                model.forward(cache, &self.continue_ids, weights),
            ]
        }

        /// The worst deviation of either call, over both the pre-norm and the
        /// normed answer. The continuation is where the cache enters, so a
        /// prefill that matched alone would say nothing about decoding.
        fn deviation(&self, weights: &impl ModelWeights) -> f32 {
            let model = self.model();
            self.forward(weights)
                .iter()
                .zip(&self.calls)
                .fold(0.0f32, |worst, (got, want)| {
                    worst
                        .max(deviation(got, &want.layers_out))
                        .max(deviation(&model.final_norm(got), &want.norm_out))
                })
        }
    }

    /// The synthetic model's weights, held whole — four layers of width 32,
    /// which is what makes a stack testable without the 131 GB checkpoint.
    ///
    /// The layer this builds per call is built from the *config*, not from
    /// anything the fixture recorded per layer: which attention config and which
    /// MLP each index gets is exactly what a stack has to get right.
    struct Held<'a> {
        config: &'a TextConfig,
        layers: &'a [LayerTensors],
        table: &'a [f32],
        order: Vec<usize>,
    }

    impl<'a> Held<'a> {
        fn new(stack: &'a Stack) -> Self {
            Self {
                config: &stack.config,
                layers: &stack.layers,
                table: &stack.table,
                order: (0..stack.layers.len()).collect(),
            }
        }

        /// The same weights with two layers' tensors exchanged, which is the
        /// mutation a stack that ran its layers out of order would make.
        fn exchanging(mut self, a: usize, b: usize) -> Self {
            self.order.swap(a, b);
            self
        }

        fn layer(&self, index: usize) -> (&'a LayerTensors, DecoderLayer<'_>) {
            let tensors = &self.layers[self.order[index]];
            let config = AttentionConfig::for_layer(self.config, index);
            (
                tensors,
                DecoderLayer::new(config, tensors.view(), tensors.mlp()),
            )
        }
    }

    impl ModelWeights for Held<'_> {
        fn embedding_row(&self, id: usize) -> Vec<f32> {
            let hidden = self.config.hidden_size;
            self.table[id * hidden..][..hidden].to_vec()
        }

        fn run_layer(&self, index: usize, cache: &mut DecoderCache, x: &[f32]) -> Vec<f32> {
            let (experts, layer) = self.layer(index);
            layer.forward(cache, x, experts)
        }
    }

    #[test]
    fn the_synthetic_stack_reproduces_mlx() {
        let stack = Stack::load();
        let deviation = stack.deviation(&Held::new(&stack));
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
        let stack = Stack::load();
        for (a, b) in interchangeable(&stack.config) {
            let deviation = stack.deviation(&Held::new(&stack).exchanging(a, b));
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
        assert_eq!(stack.layers.len(), indices.len(), "a layer per index");

        let dense: Vec<bool> = stack.layers.iter().map(LayerTensors::is_dense).collect();
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
        let model = stack.model();
        let weights = Held::new(&stack);

        let fresh = model.forward(
            &mut ModelCache::new(&stack.config),
            &stack.continue_ids,
            &weights,
        );
        let deviation = deviation(&fresh, &stack.calls[1].layers_out);
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
        for got in stack.forward(&Held::new(&stack)).iter().zip(&stack.calls) {
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
        stack.model().forward(
            &mut ModelCache::new(&shallow),
            &stack.ids,
            &Held::new(&stack),
        );
    }
}
