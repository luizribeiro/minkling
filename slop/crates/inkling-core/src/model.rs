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
//! **It is not this side's either, where a backend will have it.** The norm
//! reads what the last layer wrote and `lm_head` reads what the norm wrote, so
//! a device that ran the stack can run both without either value crossing back
//! — which is [`ModelWeights::tail`], and is why [`Model::forward`] answers
//! with a [`Passed`] rather than with rows. What is here is what runs when
//! nobody else will.
//!
//! Pinned to mlx-vlm by `reference/fixtures/stack.safetensors`, a five-layer
//! synthetic model driven through the reference's own `InklingModel`, and by the
//! recorded `layers_out` and `norm_out` of
//! `reference/fixtures/layer_activations.safetensors` for the trained
//! forty-two.

use crate::attention::AttentionConfig;
use crate::config::TextConfig;
use crate::embed::Embed;
use crate::head::{Tail, Tailed};
use crate::layer::{DecoderCache, Hidden, LayerMark, Passed, Seat};
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

    /// Where the sequence in flight is now, everywhere its state lives, as
    /// something [`ModelWeights::resume`] can put it back to.
    ///
    /// **What this buys that [`ModelWeights::rewind`] does not is a distance.**
    /// A rewind is bounded by the slack a cache was built with, which is sized
    /// for the handful of tokens a speculative round takes back; a mark reaches
    /// back over a whole generation for the cost of four windows a layer,
    /// because it carries the windows rather than shifting them. That is what
    /// lets a conversation's prompt be kept across a request while the reply
    /// that followed it is dropped.
    ///
    /// **Between runs, and this is where that has to be said.** Whatever wrote
    /// the state has to have finished writing it, which on a device means the
    /// command buffer has completed — see
    /// [`LayerBackend::mark`](crate::weights::LayerBackend::mark).
    fn mark(&self, cache: &ModelCache) -> Mark {
        Mark::new(cache.mark(), None)
    }

    /// The state the sequence had when `mark` was taken, everywhere it lives.
    fn resume(&self, cache: &mut ModelCache, mark: &Mark) {
        cache.resume(mark.cache());
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

    /// The same layer over several sequences advancing together, `x` being the
    /// `[rows, hidden]` of the whole call with each sequence's rows following
    /// the last's.
    ///
    /// **Defaulted to a call a sequence, because that is what a batch means.**
    /// Sequence `s`'s rows through layer `index` against sequence `s`'s cache
    /// is the answer a batch has to produce, whatever it does to produce it —
    /// so the default is the definition, and an implementation that has
    /// somewhere to run the layer where the weights are read once overrides it
    /// with something faster and not with something else. See
    /// [`DecoderLayer::forward_batch`](crate::layer::DecoderLayer::forward_batch).
    fn run_layer_batch(&self, index: usize, seats: &mut [Seat<'_>], x: Hidden<'_>) -> Passed {
        let rows = x.rows();
        let hidden = match seats.iter().map(|seat| seat.queries).sum::<usize>() {
            0 => panic!("a forward pass over no tokens"),
            total => rows.len() / total,
        };
        let mut out = Vec::with_capacity(rows.len());
        let mut from = 0;
        for seat in seats.iter_mut() {
            let take = seat.queries * hidden;
            let own = Hidden::Rows(&rows[from..from + take]);
            out.extend(self.run_layer(index, seat.cache, own).rows());
            from += take;
        }
        assert_eq!(
            from,
            rows.len(),
            "a sequence's rows for every row of the call"
        );
        Passed::Rows(out)
    }

    /// The final norm, the muP divide and `lm_head` behind the `rows` the last
    /// layer left where they lie, and `None` from a backend that answered that
    /// layer with rows.
    ///
    /// **The mirror of [`Passed::Carried`] at the other end of the stack.** A
    /// layer's output stays on a device because the layer after it reads it
    /// there; the last layer's output has no layer after it, and what reads it
    /// is the tail — so a backend holding the tail is a backend for which the
    /// stack has one more thing to carry to, and the only value that crosses
    /// back is what a token is taken from.
    ///
    /// Defaulted to `None` for the reason
    /// [`LayerBackend::decoder`](crate::weights::LayerBackend::decoder) is
    /// defaulted: on this side the norm and the divide are loops over slices
    /// and what folding them would buy is a round trip rather than arithmetic.
    fn tail(&self, rows: usize, want: Tail) -> Option<Tailed> {
        let (_, _) = (rows, want);
        None
    }
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

    /// The same, for a sequence whose state on a backend is that backend's slot
    /// `slot` rather than its first — see
    /// [`AttentionCache::in_slot`](crate::AttentionCache::in_slot), which is
    /// where the whole of what a slot is is said.
    ///
    /// **This is how a batch is expressed on this side and it is the whole of
    /// it.** N sequences advancing together are N of these, each naming the slot
    /// of the backend that holds its span and its windows; a sequence advancing
    /// alone is slot zero, which is what [`ModelCache::new`] leaves here. So a
    /// batch of one is not a second path — it is this call with the number
    /// nobody had to pass.
    pub fn in_slot(config: &TextConfig, slack: usize, slot: usize) -> Self {
        Self {
            layers: Self::speculating(config, slack)
                .layers
                .into_iter()
                .map(|layer| layer.in_slot(slot))
                .collect(),
        }
    }

    /// Which of a backend's slots holds the rest of this sequence's state.
    pub fn slot(&self) -> usize {
        self.layers.first().map_or(0, DecoderCache::slot)
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

    /// Where this sequence is now, as something that can put it back here later.
    ///
    /// **What this can do that [`ModelCache::rewind`] cannot is reach past the
    /// slack**, and what it costs is the difference: a rewind gives back
    /// timesteps a cache was built holding room for, and a mark carries the
    /// windows themselves — `kernel_size - 1` timesteps of each of a layer's
    /// four convolutions, whatever the distance being reached back over. The
    /// keys are not carried at all, because a key is addressed by its position;
    /// see [`AttentionCache::mark`](crate::AttentionCache::mark).
    ///
    /// Every layer or none, for the reason [`ModelCache::rewind`] is: a stack
    /// resumed unevenly would attend over a different number of keys per layer
    /// for the same position, and would still answer.
    pub fn mark(&self) -> CacheMark {
        CacheMark::new(self.layers.iter().map(DecoderCache::mark).collect())
    }

    /// The state this sequence had when `mark` was taken.
    pub fn resume(&mut self, mark: &CacheMark) {
        assert_eq!(
            mark.layers.len(),
            self.layers.len(),
            "a mark per layer of the cache"
        );
        for (layer, mark) in self.layers.iter_mut().zip(&mark.layers) {
            layer.resume(mark);
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

/// One sequence of a batch, as the stack sees it: the state that is its own and
/// the ids it is feeding this call.
///
/// The cache carries which of a backend's slots holds the rest of that
/// sequence's state — see [`ModelCache::in_slot`] — so a batch is built by
/// naming caches and nothing else has to be kept in step.
pub struct Batched<'a> {
    pub cache: &'a mut ModelCache,
    pub ids: &'a [usize],
}

/// Where a run of layers was, one [`LayerMark`] apiece.
///
/// A list rather than a map from layer index, because both ends of it walk the
/// layers they hold in order and a mark that landed on the wrong layer is one
/// whose widths would not fit. The layers a *backend* holds are a subset of the
/// stack's, so a mark taken there is shorter than one taken here — which is why
/// the two are counted against whoever they came from and not against each
/// other.
#[derive(Debug, Clone)]
pub struct CacheMark {
    layers: Vec<LayerMark>,
}

impl CacheMark {
    pub fn new(layers: Vec<LayerMark>) -> Self {
        Self { layers }
    }

    pub fn layers(&self) -> &[LayerMark] {
        &self.layers
    }
}

/// Where a sequence was, in both of the places its state lives.
///
/// **One value rather than two, for the reason
/// [`ModelWeights::rewind`] is one call rather than two.** A sequence's state is
/// in a [`ModelCache`] and in whatever backend ran its layers, and a caller that
/// marked one and resumed the other would leave a sequence whose position is one
/// thing here and another there — and which still answers.
#[derive(Debug, Clone)]
pub struct Mark {
    cache: CacheMark,
    backend: Option<CacheMark>,
}

impl Mark {
    pub fn new(cache: CacheMark, backend: Option<CacheMark>) -> Self {
        Self { cache, backend }
    }

    pub fn cache(&self) -> &CacheMark {
        &self.cache
    }

    /// The backend's own half, and `None` where the backend held no state to
    /// mark — which is every backend that holds only weights.
    pub fn backend(&self) -> Option<&CacheMark> {
        self.backend.as_ref()
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
    ///
    /// **Answered rather than handed over**, because the last layer's rows are
    /// the one hidden state whose reader may be on the same device that wrote
    /// them: [`ModelWeights::tail`] is what reads them there, and a stack that
    /// carried them answers with a count. A caller with no tail to ask for
    /// takes [`Passed::rows`], which is what every fixture here does.
    pub fn forward(
        &self,
        cache: &mut ModelCache,
        ids: &[usize],
        weights: &impl ModelWeights,
    ) -> Passed {
        self.forward_batch(&mut [Batched { cache, ids }], weights)
    }

    /// The same stack over several sequences advancing together: every
    /// sequence's ids embedded into one call, forty-two layers over all of
    /// them, and the rows they produced in the order the batch names them.
    ///
    /// **A batch of one is this call and not a simpler one beside it**, for the
    /// reason [`DecoderLayer::forward_batch`](crate::layer::DecoderLayer::forward_batch)
    /// gives: what a batch buys is that a layer's weights are read once for
    /// every sequence in it, and a single request that took another path would
    /// be a second copy of every shape-keyed decision to keep in step.
    ///
    /// The rows are laid out sequence by sequence, which is what makes each
    /// sequence's run of them contiguous — the thing every dispatch that reads
    /// one sequence's rows out of the call indexes by.
    pub fn forward_batch(&self, batch: &mut [Batched<'_>], weights: &impl ModelWeights) -> Passed {
        assert!(!batch.is_empty(), "a forward pass over no sequences");
        let mut ids = Vec::new();
        for sequence in batch.iter() {
            assert_eq!(
                sequence.cache.layers.len(),
                self.layers,
                "a cache per layer of the model"
            );
            assert!(!sequence.ids.is_empty(), "a forward pass over no tokens");
            ids.extend_from_slice(sequence.ids);
        }

        let mut h = Passed::Rows(self.embeddings(&ids, weights));
        for index in 0..self.layers {
            let mut seats: Vec<Seat<'_>> = batch
                .iter_mut()
                .map(|sequence| Seat {
                    queries: sequence.ids.len(),
                    cache: sequence.cache.layer(index),
                })
                .collect();
            let handed = h.handed();
            let next = weights.run_layer_batch(index, &mut seats, handed);
            h = next;
        }
        h
    }

    /// `[tokens]` ids as the `[tokens, hidden]` the stack would have started
    /// from, which is `InklingModel.embed`: a row of the table per id, through
    /// `embed_norm` where the config asks for one.
    ///
    /// Here rather than inside [`Model::forward`] alone because the MTP heads
    /// read it too — a head is handed the embedding of a token one position
    /// further ahead than the one the stack is running — and what they are
    /// handed has to be the same value, from the same norm, as what the stack
    /// begins with. See [`crate::mtp`], where feeding a head the *unnormed* row
    /// instead is one of the wirings the acceptance study ruled out.
    pub fn embeddings(&self, ids: &[usize], weights: &impl ModelWeights) -> Vec<f32> {
        self.embed.forward(ids, |id| weights.embedding_row(id))
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
            model.forward(cache, &stack.ids, stack).rows(),
            model.forward(cache, &stack.continue_ids, stack).rows(),
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

        let fresh = stack
            .model()
            .forward(
                &mut ModelCache::new(&stack.config),
                &stack.continue_ids,
                &stack,
            )
            .rows();
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
            let after = model.forward(cache, &sequence[split..], &stack).rows();

            let clean = &mut ModelCache::speculating(&stack.config, taken);
            model.forward(clean, &sequence[..split], &stack);
            let want = model.forward(clean, &sequence[split..], &stack).rows();
            assert_eq!(after, want, "{taken} tokens taken back at {split}");
        }
    }

    /// The property the whole of a prefix cache rests on, and it is the rewind
    /// test's said over a distance no slack was bought for: a stack marked, fed
    /// tokens, and resumed is the stack that was marked.
    ///
    /// **The cache keeps no slack at all here**, which is what separates this
    /// from [`ModelCache::rewind`]: every one of these resumes is a reach a
    /// rewind would refuse. That is the whole reason a mark exists — a
    /// conversation's reply is hundreds of tokens and a speculative round's
    /// slack is eight.
    ///
    /// Exact equality, for the reason the rewind test demands it: both sides
    /// multiply the same numbers in the same order, so what a resumed pass
    /// leaves behind is not close to what a fresh one does, it is the same.
    #[test]
    fn resuming_a_mark_leaves_the_stack_where_the_mark_was_taken() {
        let stack = Stack::load();
        let sequence = stack.sequence();
        let model = stack.model();

        for split in 1..sequence.len() {
            let wrong: Vec<usize> = sequence[split..]
                .iter()
                .map(|id| (id + 1) % stack.config.vocab_size)
                .collect();
            assert_ne!(wrong, sequence[split..], "tokens a resume has to undo");

            let cache = &mut ModelCache::new(&stack.config);
            model.forward(cache, &sequence[..split], &stack);
            let mark = cache.mark();
            model.forward(cache, &wrong, &stack);
            cache.resume(&mark);
            let after = model.forward(cache, &sequence[split..], &stack).rows();

            let clean = &mut ModelCache::new(&stack.config);
            model.forward(clean, &sequence[..split], &stack);
            let want = model.forward(clean, &sequence[split..], &stack).rows();
            assert_eq!(
                after,
                want,
                "a mark at {split} resumed over {}",
                wrong.len()
            );
        }
    }

    /// **The same sequence, run alone and run inside a batch, produces
    /// identical rows** — stated on the stack, where the rows are all there is
    /// to compare.
    ///
    /// On this side a batch is a call a sequence and so cannot fail; what this
    /// pins is everything around those calls, which a backend that runs them in
    /// one pass inherits: that the ids are laid out sequence by sequence, that
    /// each sequence's rows come back where its own are, and that a sequence is
    /// answered the same wherever in the batch it sits. The kernels' own case
    /// is `a_sequence_in_a_batch_produces_what_it_produces_alone`.
    ///
    /// Exact equality, because both arms multiply the same numbers in the same
    /// order — the only thing a batch changes is which row of the call each of
    /// them is at.
    #[test]
    fn a_sequence_in_a_batch_produces_what_it_produces_alone() {
        let stack = Stack::load();
        let model = stack.model();
        let sequence = stack.sequence();
        // Two prompts of different lengths, so that a neighbour's length is one
        // of the things a batch has to be right about.
        let prompts: [&[usize]; 3] = [&sequence[..3], &sequence[3..], &sequence[..1]];

        let alone: Vec<Vec<f32>> = prompts
            .iter()
            .map(|ids| {
                model
                    .forward(&mut ModelCache::new(&stack.config), ids, &stack)
                    .rows()
            })
            .collect();
        assert_ne!(alone[0], alone[2], "two sequences to tell apart");

        for order in [vec![0, 1, 2], vec![2, 1, 0], vec![1, 2], vec![2, 1]] {
            let mut caches: Vec<ModelCache> = order
                .iter()
                .enumerate()
                .map(|(slot, _)| ModelCache::in_slot(&stack.config, 0, slot))
                .collect();
            let mut batch: Vec<Batched<'_>> = caches
                .iter_mut()
                .zip(&order)
                .map(|(cache, at)| Batched {
                    cache,
                    ids: prompts[*at],
                })
                .collect();
            let rows = model.forward_batch(&mut batch, &stack).rows();

            let mut from = 0;
            for at in &order {
                let take = prompts[*at].len() * model.hidden();
                assert_eq!(
                    rows[from..from + take],
                    alone[*at][..],
                    "sequence {at} of {order:?}"
                );
                from += take;
            }
            assert_eq!(from, rows.len(), "a sequence's rows for every row");
        }
    }

    /// A batch of no sequences is a forward pass with nothing to answer, for the
    /// reason a pass over no tokens is.
    #[test]
    #[should_panic(expected = "a forward pass over no sequences")]
    fn a_forward_pass_over_no_sequences_is_refused() {
        let stack = Stack::load();
        stack.model().forward_batch(&mut [], &stack);
    }

    /// A mark says where a sequence *was*. Asking to be put somewhere it has not
    /// reached is asking for keys nothing has computed, and is refused rather
    /// than answered out of a span whose count would then outrun its keys.
    #[test]
    #[should_panic(expected = "a mark at")]
    fn resuming_a_mark_the_sequence_has_not_reached_is_refused() {
        let stack = Stack::load();
        let model = stack.model();
        let ahead = &mut ModelCache::new(&stack.config);
        model.forward(ahead, &stack.ids, &stack);
        let mark = ahead.mark();

        // A sequence one token in, asked to be put back where a longer one was.
        let behind = &mut ModelCache::new(&stack.config);
        model.forward(behind, &stack.ids[..1], &stack);
        behind.resume(&mark);
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
