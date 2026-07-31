//! The attention projections and the two dense layers' feed-forward networks,
//! which are 66% of a decode step — 52% for the five projections of every layer
//! and 14% for the two feed-forward networks.
//!
//! **The term is the multiply, not the decode.** The experts came here because
//! decoding them was 32 GB a step; these come here for a different reason. Of a
//! step, `dequantize_blocks_into` is 12% and `ops::linear` is 66%, and it is
//! these weights that most of both is spent on — so what a dispatch replaces
//! here is mostly the arithmetic. A serial f32 dot product is what no compiler
//! may vectorise, because f32 addition is not associative and reassociating it
//! is not a transformation LLVM is allowed to make; the 9 GB of dequantisation
//! that fed it comes off as well, but it was never the larger half.
//!
//! **Every one of these is read in full by every token.** That is what separates
//! them from the experts: a routed bank is 256 experts of which a step reads six,
//! so wrapping one is mostly a promise not to read it, and the gather is what
//! keeps that promise. A projection has no such axis — there is nothing here to
//! be selective about — so these are [`PackedProjection`]s, which is a bank of
//! one expert every row goes through, and the only thing that stays packed is
//! the weight itself.
//!
//! **One dispatch a projection.** Five a layer for attention and three for each
//! of the two dense layers, which is 216 dispatches a decode step against the
//! MoE's 240. Whether that is the right granularity is a question about
//! dispatching rather than about arithmetic, and it is asked and answered
//! separately: the four projections that consume the same normed hidden state
//! could be one submission, and so could a feed-forward network's gate and up.

use inkling_core::attention::Projections;
use inkling_core::ops::{MlpProjections, Projection};
use inkling_core::weights::{LayerPacked, Packed, PackedAttention, PackedMlp, ProjectionBackend};

use crate::device::Device;
use crate::matmul::{MatmulError, PackedMatmul, PackedProjection};

/// One attention layer's five projections on the device.
///
/// The mirror of [`DecodedProjections`](inkling_core::DecodedProjections), and
/// it holds the same relation to it that
/// [`ExpertBanks`](crate::ExpertBanks) holds to
/// [`PackedExperts`](inkling_core::PackedExperts): the arithmetic is the
/// checkpoint's, and what changes is that no weight is ever decoded to memory.
#[derive(Debug)]
pub struct LayerProjections<'a> {
    q_proj: PackedProjection<'a>,
    k_proj: PackedProjection<'a>,
    v_proj: PackedProjection<'a>,
    r_proj: PackedProjection<'a>,
    o_proj: PackedProjection<'a>,
}

impl<'a> LayerProjections<'a> {
    /// Wrap a layer's five projections where the checkpoint mapped them.
    ///
    /// Nothing checks here that they are one layer's, and nothing here could:
    /// what the five widths have to be is
    /// [`AttentionConfig`](inkling_core::AttentionConfig)'s to say, and
    /// [`Attention::new`](inkling_core::Attention::new) is where they are asked
    /// — of whichever backend answered, so that the two cannot differ.
    ///
    /// Which is why the mapping from name to tensor is what this module's tests
    /// are about. Two of the five pairs the checkpoint gives are the same shape
    /// both ways round — `q_proj` against `o_proj`, `k_proj` against `v_proj` —
    /// so a slot filled from the wrong name is a layer that stands up, checks
    /// out and attends to the wrong thing.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &PackedAttention<'a>,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            q_proj: whole(device, matmul, &packed.q_proj)?,
            k_proj: whole(device, matmul, &packed.k_proj)?,
            v_proj: whole(device, matmul, &packed.v_proj)?,
            r_proj: whole(device, matmul, &packed.r_proj)?,
            o_proj: whole(device, matmul, &packed.o_proj)?,
        })
    }
}

impl Projections for LayerProjections<'_> {
    fn q_proj(&self) -> &dyn Projection {
        &self.q_proj
    }

    fn k_proj(&self) -> &dyn Projection {
        &self.k_proj
    }

    fn v_proj(&self) -> &dyn Projection {
        &self.v_proj
    }

    fn r_proj(&self) -> &dyn Projection {
        &self.r_proj
    }

    fn o_proj(&self) -> &dyn Projection {
        &self.o_proj
    }
}

/// One dense layer's feed-forward network on the device.
///
/// `3 x [16384, 4096]`, which is the widest weight in the model below the head
/// and four and a half times a layer's five attention projections together. Two
/// layers of forty-two have one.
#[derive(Debug)]
pub struct DenseFfn<'a> {
    gate_proj: PackedProjection<'a>,
    up_proj: PackedProjection<'a>,
    down_proj: PackedProjection<'a>,
}

impl<'a> DenseFfn<'a> {
    /// Wrap a dense layer's three where the checkpoint mapped them. Whether they
    /// pair is [`DenseMlp`](inkling_core::DenseMlp)'s to say, and `gate_proj`
    /// against `up_proj` is the pair that pairs either way round.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &PackedMlp<'a>,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            gate_proj: whole(device, matmul, &packed.gate_proj)?,
            up_proj: whole(device, matmul, &packed.up_proj)?,
            down_proj: whole(device, matmul, &packed.down_proj)?,
        })
    }
}

impl MlpProjections for DenseFfn<'_> {
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

/// A whole packed tensor as the projection it is: every row, and one expert
/// every row of a call goes through.
///
/// The head is the only weight that wants fewer rows than it holds — its 966
/// padding rows are the truncation [`inkling_core::head`] describes — and a
/// layer's projections have no padding to stop short of.
fn whole<'a>(
    device: &'a Device,
    matmul: &'a PackedMatmul,
    packed: &Packed<'a>,
) -> Result<PackedProjection<'a>, MatmulError> {
    PackedProjection::wrap_packed(device, matmul, packed, packed.slices())
}

/// Every layer's own projections on the device, which for Inkling-Small is 42
/// layers of five and two of three more.
///
/// 9.0 GB of packed bytes, wrapped where the checkpoint mapped them and holding
/// no resident set of their own — the same bargain
/// [`ModelExperts`](crate::ModelExperts) strikes over its 137 GB, and cheap
/// enough for the same reason that there is no residency question to answer.
#[derive(Debug)]
pub struct ModelProjections<'a> {
    /// Indexed by layer, `None` where nothing here answers for one — which is a
    /// layer the CPU keeps, and is how a partial handover stays expressible.
    layers: Vec<Option<Layer<'a>>>,
}

/// One layer's own projections: attention's five, and the feed-forward network
/// of a layer that has one.
#[derive(Debug)]
struct Layer<'a> {
    attention: LayerProjections<'a>,
    dense_mlp: Option<DenseFfn<'a>>,
}

impl<'a> ModelProjections<'a> {
    /// Wrap every projection `packed` names, over a stack of `layers`.
    ///
    /// The stack's length is stated rather than read off the last entry, for the
    /// reason [`ModelExperts::wrap`](crate::ModelExperts::wrap) states it: a
    /// backend answering for none of the last layers would otherwise report a
    /// shorter stack than the model has, and "past the stack" and "left to the
    /// CPU" would stop being answerable apart.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &[LayerPacked<'a>],
        layers: usize,
    ) -> Result<Self, MatmulError> {
        let mut wrapped: Vec<Option<Layer<'a>>> = (0..layers).map(|_| None).collect();
        for layer in packed {
            wrapped[layer.layer] = Some(Layer {
                attention: LayerProjections::wrap(device, matmul, &layer.attention)?,
                dense_mlp: layer
                    .dense_mlp
                    .map(|mlp| DenseFfn::wrap(device, matmul, &mlp))
                    .transpose()?,
            });
        }
        Ok(Self { layers: wrapped })
    }

    /// How many of the stack's layers have their attention projections here.
    pub fn layers(&self) -> usize {
        self.layers.iter().flatten().count()
    }

    /// How many of those also have a feed-forward network here, which is how
    /// many are dense.
    pub fn dense_layers(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .filter(|layer| layer.dense_mlp.is_some())
            .count()
    }

    fn layer(&self, layer: usize) -> Option<&Layer<'a>> {
        self.layers.get(layer)?.as_ref()
    }
}

/// The seam [`inkling_core::weights`] names, so that a layer standing itself up
/// does not know whether its projections were ever decoded.
impl ProjectionBackend for ModelProjections<'_> {
    fn attention(&self, layer: usize) -> Option<&dyn Projections> {
        Some(&self.layer(layer)?.attention as &dyn Projections)
    }

    fn dense_mlp(&self, layer: usize) -> Option<&dyn MlpProjections> {
        Some(self.layer(layer)?.dense_mlp.as_ref()? as &dyn MlpProjections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::Checkpoint;
    use inkling_core::fixture::{self, deviation};
    use inkling_core::ops::DenseProjection;

    use crate::testing::device;

    /// The packed tensors `just dump-quant-fixture` cut out of the checkpoint
    /// that are weights, each with the values MLX decoded it to stored beside
    /// it.
    ///
    /// They are what a hermetic test has to hand out, and they are what makes
    /// the assertions below possible at all: a slot's answer is checked against
    /// *the tensor that slot was named*, so a slot filled from the wrong name is
    /// a different weight rather than a different width.
    ///
    /// The fixture's fourth, `code_grid`, is left out. It is every code under
    /// every scale byte, `0xff` included, so it decodes to infinities and is a
    /// decoder's case rather than a weight anything can be measured against.
    const TENSORS: [&str; 3] = ["dense_ffn", "vocab_padding", "routed_expert"];

    /// The same account as `matmul::tests::TOLERANCE`: decoding is exact on both
    /// sides, so what separates a dispatch from the CPU is summation order, and
    /// what a wrong tensor produces is decades away rather than ulps.
    const TOLERANCE: f32 = 6e-6;

    fn packed<'a>(ckpt: &'a Checkpoint, name: &str) -> Packed<'a> {
        Packed::open(ckpt, name).expect("the fixture holds the slice packed")
    }

    /// A row spread over both signs, so that a reduction cancels the way a
    /// trained one does.
    fn row(in_dim: usize) -> Vec<f32> {
        (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect()
    }

    /// That every projection of `named` answers with the tensor it was named,
    /// against MLX's own decoding of that tensor.
    fn each_answers(ckpt: &Checkpoint, named: &[(&str, &dyn Projection)]) {
        for (name, projection) in named {
            let x = row(projection.in_dim());
            let weight = fixture::f32s(&fixture::tensor(ckpt, &format!("{name}.dequantized")));
            let want = DenseProjection::new(x.len(), &weight).forward(&x);

            assert_eq!(projection.out_dim(), want.len(), "{name} maps to");
            let deviation = deviation(&projection.forward(&x), &want);
            assert!(deviation <= TOLERANCE, "{name}: deviation {deviation:e}");
        }
    }

    /// The two tensors the last two slots take, which is what the rounds below
    /// exchange: `(r_proj, o_proj)`.
    const LAST_TWO: [(&str, &str); 2] = [(TENSORS[0], TENSORS[1]), (TENSORS[1], TENSORS[0])];

    /// One attention layer's five tensors: the first three are the fixture's
    /// three, and the last two are whichever round this is.
    fn attention<'a>(ckpt: &'a Checkpoint, (r_proj, o_proj): (&str, &str)) -> PackedAttention<'a> {
        PackedAttention {
            q_proj: packed(ckpt, TENSORS[0]),
            k_proj: packed(ckpt, TENSORS[1]),
            v_proj: packed(ckpt, TENSORS[2]),
            r_proj: packed(ckpt, r_proj),
            o_proj: packed(ckpt, o_proj),
        }
    }

    /// Each of the five names wraps the tensor it was given.
    ///
    /// This is the mistake the widths cannot catch. `q_proj` and `o_proj` are
    /// both `[4096, 4096]` in the checkpoint and `k_proj` and `v_proj` are both
    /// `[1024, 4096]`, so either pair exchanged here produces a layer that
    /// stands up, passes every shape check there is, and attends to the wrong
    /// thing.
    ///
    /// Three distinct tensors over five slots means two slots have to repeat two
    /// others, and which two is what the rounds exchange: the first round
    /// repeats at `(q, r)` and `(k, o)`, the second at `(k, r)` and `(q, o)`. No
    /// pair of slots holds the same weight in both, so every one of the ten
    /// exchanges of two names is a wrong answer in at least one round.
    #[test]
    fn each_of_an_attention_layers_five_names_wraps_the_tensor_it_was_given() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let ckpt = fixture::open(fixture::MXFP4);

        for (r_proj, o_proj) in LAST_TWO {
            let five =
                LayerProjections::wrap(&device, &matmul, &attention(&ckpt, (r_proj, o_proj)))
                    .expect("the five wrap");

            each_answers(
                &ckpt,
                &[
                    (TENSORS[0], five.q_proj()),
                    (TENSORS[1], five.k_proj()),
                    (TENSORS[2], five.v_proj()),
                    (r_proj, five.r_proj()),
                    (o_proj, five.o_proj()),
                ],
            );
        }
    }

    /// And each of a feed-forward network's three does. `silu` goes on the gate
    /// and not on the up, and the two are the same shape, so a network wrapped
    /// with them exchanged is one of exactly the right widths and the wrong
    /// activation.
    #[test]
    fn each_of_a_feed_forward_networks_three_names_wraps_the_tensor_it_was_given() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let ckpt = fixture::open(fixture::MXFP4);

        let three = DenseFfn::wrap(
            &device,
            &matmul,
            &PackedMlp {
                gate_proj: packed(&ckpt, TENSORS[0]),
                up_proj: packed(&ckpt, TENSORS[1]),
                down_proj: packed(&ckpt, TENSORS[2]),
            },
        )
        .expect("the three wrap");

        each_answers(
            &ckpt,
            &[
                (TENSORS[0], three.gate_proj()),
                (TENSORS[1], three.up_proj()),
                (TENSORS[2], three.down_proj()),
            ],
        );
    }

    /// Which layer's projections answer for which layer, and which layers this
    /// answers for at all.
    ///
    /// A layer nothing here holds is `None` rather than absent, because that is
    /// what the CPU path reads as "decode them yourself" — and a layer past the
    /// stack is `None` too rather than an index off the end. The stack here is
    /// two longer than the last layer wrapped, which is the case that says the
    /// two answers are not the same answer.
    #[test]
    fn a_layer_this_does_not_hold_and_a_layer_past_the_stack_are_left_to_the_cpu() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let ckpt = fixture::open(fixture::MXFP4);

        // Inkling's shape with a hole punched in it: a dense layer, a gap the
        // CPU keeps, a MoE layer, and two more layers nothing was handed.
        const LAYERS: usize = 5;
        let packed_layers = [
            LayerPacked {
                layer: 0,
                attention: attention(&ckpt, LAST_TWO[0]),
                dense_mlp: Some(PackedMlp {
                    gate_proj: packed(&ckpt, TENSORS[0]),
                    up_proj: packed(&ckpt, TENSORS[1]),
                    down_proj: packed(&ckpt, TENSORS[2]),
                }),
            },
            LayerPacked {
                layer: 2,
                attention: attention(&ckpt, LAST_TWO[1]),
                dense_mlp: None,
            },
        ];
        let projections = ModelProjections::wrap(&device, &matmul, &packed_layers, LAYERS)
            .expect("the layers wrap");

        assert_eq!(projections.layers(), 2, "two of the five");
        assert_eq!(projections.dense_layers(), 1);
        assert!(projections.dense_mlp(0).is_some());
        assert!(
            projections.dense_mlp(2).is_none(),
            "a layer that routes to experts has no feed-forward network"
        );
        assert!(
            projections.attention(1).is_none(),
            "a layer left to the CPU"
        );
        assert!(projections.attention(4).is_none(), "the last of the stack");
        assert!(projections.attention(LAYERS).is_none(), "past the stack");

        // And the two layers wrapped are not each other's, which is what an
        // index off by one would produce. They differ in `o_proj` alone.
        for (layer, (_, o_proj)) in [(0, LAST_TWO[0]), (2, LAST_TWO[1])] {
            let five = projections.attention(layer).expect("a wrapped layer");
            each_answers(&ckpt, &[(o_proj, five.o_proj())]);
        }
    }
}
