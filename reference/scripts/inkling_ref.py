"""Shared setup for the reference scripts: checkpoint loading with the
eos/pad-token fix-up every Inkling checkpoint needs, shard-level access for the
scripts that want a few tensors rather than a model, the seeded draws the
synthetic fixtures stand a module up from, and byte formatting."""

import json
from functools import lru_cache

import mlx.core as mx
import numpy as np
from mlx_vlm import load

# The decoder layers `dump_activations.py` captures, and so the layers every
# trained fixture is cut from. Three scripts have to agree on them — the
# activations, the sconv kernels and the mask projections are all indexed by
# layer, and a bundle that names a layer the others do not is a fixture the
# Rust side cannot open.
#
# Layer 0 is dense and sliding, 2 is MoE and sliding, 5 is MoE and global. That
# covers both MLPs and both attention configurations; `check_layer_coverage`
# refuses a set that stops doing so.
CAPTURED_LAYERS = (0, 2, 5)


def gib(n_bytes):
    return n_bytes / (1 << 30)


@lru_cache(maxsize=1)
def index_of(model_path):
    return json.loads((model_path / "model.safetensors.index.json").read_text())


def load_shard(model_path, shard):
    """`mx.load` is lazy, so a shard costs only the tensors actually evaluated."""
    return mx.load(str(model_path / shard))


def checkpoint_tensor(model_path, name):
    """One named tensor, straight out of the shard that owns it. Materialising
    the 130 GiB model to read a few hundred KiB has crashed this host before."""
    return load_shard(model_path, index_of(model_path)["weight_map"][name])[name]


def f32(values):
    return mx.array(values, dtype=mx.float32)


def projection(rng, out_dim, in_dim):
    """An `nn.Linear` weight, `[out, in]`, scaled by fan-in so what it produces
    lands in the same range the trained activations do — where an activation is
    still curved, and where a bias is still comparable to a logit."""
    return f32(rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim))


def gamma(rng, dims):
    """A trained RMSNorm weight sits near 1 and is not uniform, so reading it as
    a scalar — or dropping it — has to change the answer."""
    return f32(1.0 + 0.5 * rng.standard_normal(dims))


def taps(rng, channels, kernel_size):
    """A short convolution's kernel, `[channels, kernel_size, 1]` as `nn.Conv1d`
    and the checkpoint both store it.

    Drawn so no two taps of a channel are close in magnitude: a kernel read in
    reversed time order still produces plausible numbers, and only an asymmetric
    kernel makes it produce different ones. The ramp is about the four-fold one
    the trained kernels show, which also keeps the convolution and the residual
    within a factor of a few of each other — a residual lost in the noise would
    be a residual nothing tests."""
    magnitude = 0.4 * 1.6 ** np.arange(kernel_size)
    signs = rng.choice([-1.0, 1.0], size=(channels, kernel_size))
    spread = 1.0 + 0.25 * rng.standard_normal((channels, kernel_size))
    return f32((signs * spread * magnitude)[..., None])


# The global scales the synthetic layers are drawn with. Not the checkpoint's —
# layer 2's 0.007 would put every recorded output two decades below its input —
# but far enough from one that dropping either shows.
DENSE_GLOBAL_SCALE = 0.6
MOE_GLOBAL_SCALE = 0.35


def attention_parameters(rng, config, layer_index):
    """Every tensor `InklingAttention` holds for one layer index.

    Every width comes from the config the way `InklingAttention.__init__` reads
    it, including which of the two sets of head fields the layer gets and how
    wide its band is — a sliding layer's is its window, a global layer's is
    `rel_extent`. A fixture whose layers differ from each other is the only kind
    that can say a stack ran them in order, and this is where they differ.

    `rel_proj` is contracted over `d_rel` rather than over its own width, so it
    is scaled by that; `nn.Linear` leaves it zeroed, which would make the mask a
    plain causal one."""
    sliding = config.layer_is_sliding(layer_index)
    either = lambda swa, glob: swa if sliding else glob  # noqa: E731
    n_heads = either(config.swa_num_attention_heads, config.num_attention_heads)
    n_kv = either(config.swa_num_key_value_heads, config.num_key_value_heads)
    head_dim = either(config.swa_head_dim, config.head_dim)
    rel_extent = either(config.sliding_window_size, config.rel_extent)

    hidden, d_rel = config.hidden_size, config.d_rel
    kernel, kv_width = config.sconv_kernel_size, n_kv * head_dim
    return {
        "q_proj.weight": projection(rng, n_heads * head_dim, hidden),
        "k_proj.weight": projection(rng, kv_width, hidden),
        "v_proj.weight": projection(rng, kv_width, hidden),
        "r_proj.weight": projection(rng, n_heads * d_rel, hidden),
        "o_proj.weight": projection(rng, hidden, n_heads * head_dim),
        "q_norm.weight": gamma(rng, head_dim),
        "k_norm.weight": gamma(rng, head_dim),
        "k_sconv.conv.weight": taps(rng, kv_width, kernel),
        "v_sconv.conv.weight": taps(rng, kv_width, kernel),
        "rel_proj": f32(rng.standard_normal((d_rel, rel_extent)) / np.sqrt(d_rel)),
    }


def dense_parameters(rng, config):
    """`InklingDenseMLP`: a SwiGLU MLP at the dense FFN width, times a learned
    output scale."""
    hidden, width = config.hidden_size, config.intermediate_size
    return {
        "gate_proj.weight": projection(rng, width, hidden),
        "up_proj.weight": projection(rng, width, hidden),
        "down_proj.weight": projection(rng, hidden, width),
        "global_scale": f32([DENSE_GLOBAL_SCALE]),
    }


def expert_bank(rng, config, count):
    """The three `SwitchLinear` projections one `SwitchGLU` holds, `[experts,
    out, in]`."""
    hidden, width = config.hidden_size, config.moe_intermediate_size
    return {
        f"{name}.weight": mx.stack(
            [projection(rng, out_dim, in_dim) for _ in range(count)]
        )
        for name, (out_dim, in_dim) in (
            ("gate_proj", (width, hidden)),
            ("up_proj", (width, hidden)),
            ("down_proj", (hidden, width)),
        )
    }


def moe_parameters(rng, config):
    """`InklingSparseMoE`: a `[n_routed + n_shared, hidden]` gate with the shared
    experts last, a selection-only correction bias over the range the trained one
    spans, and the two expert banks."""
    routed, shared = config.n_routed_experts, config.n_shared_experts
    return {
        "gate_weight": projection(rng, routed + shared, config.hidden_size),
        "e_score_correction_bias": f32(rng.uniform(0.05, 0.8, routed)),
        "global_scale": f32([MOE_GLOBAL_SCALE]),
        **{f"switch_mlp.{k}": v for k, v in expert_bank(rng, config, routed).items()},
        **{
            f"shared_experts.{k}": v
            for k, v in expert_bank(rng, config, shared).items()
        },
    }


def layer_parameters(rng, config, layer_index):
    """Every tensor `InklingDecoderLayer` holds, for one layer index."""
    dense = config.layer_is_dense(layer_index)
    mlp = dense_parameters(rng, config) if dense else moe_parameters(rng, config)
    hidden, kernel = config.hidden_size, config.sconv_kernel_size
    return {
        **{
            f"self_attn.{name}": w
            for name, w in attention_parameters(rng, config, layer_index).items()
        },
        "input_layernorm.weight": gamma(rng, hidden),
        "post_attention_layernorm.weight": gamma(rng, hidden),
        "attn_sconv.conv.weight": taps(rng, hidden, kernel),
        "mlp_sconv.conv.weight": taps(rng, hidden, kernel),
        **{f"mlp.{name}": w for name, w in mlp.items()},
    }


def tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def byte_level_chars():
    """The GPT-2 byte-level alphabet, byte to character: the printable Latin-1
    bytes stand for themselves and the other 68 are lifted to U+0100 and up, in
    byte order. Inkling's vocabulary is spelled in it, so a piece is not text —
    it is these characters standing for the bytes the piece contributes."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table = {b: chr(b) for b in printable}
    lifted = (b for b in range(256) if b not in table)
    for offset, b in enumerate(lifted):
        table[b] = chr(256 + offset)
    return table


BYTE_CHARS = byte_level_chars()
CHAR_BYTES = {c: b for b, c in BYTE_CHARS.items()}


def piece_bytes(piece):
    """One vocabulary entry as the bytes it stands for. An added token is
    literal text rather than byte-level, but every character it is spelled with
    is its own byte-level spelling, so one mapping serves both."""
    return bytes(CHAR_BYTES[c] for c in piece)


def _resolve_eos_token(model, processor):
    # The checkpoint lists every special token under additional_special_tokens
    # and names no eos, so mlx-vlm's `pad_token = eos_token` fallback assigns
    # None and padding fails. Resolve eos from the model config instead.
    tok = tokenizer(processor)
    if tok.eos_token is None:
        tok.eos_token = tok.convert_ids_to_tokens(model.config.eos_token_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


def load_model(path):
    model, processor = load(path)
    mx.eval(model.parameters())
    _resolve_eos_token(model, processor)
    return model, processor
