use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    #[serde(default)]
    pub mtp_config: Option<MtpConfig>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: Option<usize>,
    pub model_max_length: usize,
    pub rms_norm_eps: f32,
    pub use_embed_norm: bool,
    pub logits_mup_width_multiplier: f32,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub swa_head_dim: usize,
    pub sliding_window_size: usize,
    pub local_layer_ids: Vec<usize>,

    /// Position is carried by these plus `sconv`; the model has no RoPE.
    pub d_rel: usize,
    pub rel_extent: usize,

    pub log_scaling_n_floor: Option<f32>,
    pub log_scaling_alpha: f32,

    pub use_sconv: bool,
    pub sconv_kernel_size: usize,

    /// Layers `[0, dense_mlp_idx)` use a dense FFN; the rest are MoE.
    pub dense_mlp_idx: usize,
    /// Width of a dense FFN layer.
    pub dense_intermediate_size: usize,
    /// Width of one routed or shared expert. The checkpoint spells this
    /// `intermediate_size`, which reads as the dense width and has been
    /// transposed with it before; the unambiguous name is load-bearing.
    #[serde(rename = "intermediate_size")]
    pub moe_intermediate_size: usize,

    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub route_scale: f32,
    pub use_gate_bias: bool,
    pub norm_after_topk: bool,
    pub shared_expert_sink: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MtpConfig {
    pub num_nextn_predict_layers: usize,
    pub local_layer_ids: Vec<usize>,
}

/// Per-sequence KV cost. Sliding layers are bounded by the window, so only the
/// global layers scale with sequence length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvFootprint {
    pub bytes_per_token: usize,
    pub fixed_bytes: usize,
}

impl KvFootprint {
    pub fn bytes_at(&self, tokens: usize) -> usize {
        self.fixed_bytes + self.bytes_per_token * tokens
    }
}

impl TextConfig {
    pub fn layer_is_sliding(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }

    pub fn layer_is_dense(&self, layer: usize) -> bool {
        layer < self.dense_mlp_idx
    }

    pub fn global_layers(&self) -> Vec<usize> {
        (0..self.num_hidden_layers)
            .filter(|&i| !self.layer_is_sliding(i))
            .collect()
    }

    pub fn num_sliding_layers(&self) -> usize {
        self.num_hidden_layers - self.global_layers().len()
    }

    pub fn kv_footprint(&self, dtype_bytes: usize) -> KvFootprint {
        let global = self.global_layers().len();
        let sliding = self.num_sliding_layers();
        KvFootprint {
            bytes_per_token: global * 2 * self.num_key_value_heads * self.head_dim * dtype_bytes,
            fixed_bytes: sliding
                * 2
                * self.swa_num_key_value_heads
                * self.swa_head_dim
                * self.sliding_window_size
                * dtype_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::AttentionConfig;

    /// Trimmed from `thinkingmachines/Inkling-Small`, values verbatim.
    const INKLING_SMALL: &str = r#"{
      "eos_token_id": 200006,
      "text_config": {
        "model_max_length": 1048576, "hidden_size": 4096, "num_hidden_layers": 42,
        "vocab_size": 201024, "unpadded_vocab_size": 200058,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "swa_num_attention_heads": 32, "swa_num_key_value_heads": 8, "swa_head_dim": 128,
        "sliding_window_size": 512,
        "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,
                            24,25,26,27,28,30,31,32,33,34,36,37,38,39,40],
        "d_rel": 16, "rel_extent": 1024,
        "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06, "use_embed_norm": true,
        "logits_mup_width_multiplier": 16.0,
        "use_sconv": true, "sconv_kernel_size": 4,
        "dense_mlp_idx": 2, "dense_intermediate_size": 16384, "intermediate_size": 2048,
        "n_routed_experts": 256, "num_experts_per_tok": 6, "n_shared_experts": 2,
        "route_scale": 8.0, "use_gate_bias": true, "norm_after_topk": true,
        "shared_expert_sink": true
      },
      "mtp_config": { "num_nextn_predict_layers": 8, "local_layer_ids": [0,2,4,5,6,7] }
    }"#;

    fn cfg() -> Config {
        serde_json::from_str(INKLING_SMALL).expect("parses")
    }

    #[test]
    fn ffn_widths_are_not_transposed() {
        let t = cfg().text_config;
        assert_eq!(t.dense_intermediate_size, 16384);
        assert_eq!(t.moe_intermediate_size, 2048);
    }

    #[test]
    fn attention_layers_alternate_five_local_then_one_global() {
        let t = cfg().text_config;
        assert_eq!(t.global_layers(), vec![5, 11, 17, 23, 29, 35, 41]);
        assert_eq!(t.num_sliding_layers(), 35);
    }

    /// Which of the two sets of head fields an attention layer reads.
    /// Inkling-Small sets them to the same numbers, so the checkpoint cannot
    /// tell a port that read the wrong set from one that read the right one;
    /// this moves them apart first, the way mlx-vlm's own defaults do.
    #[test]
    fn a_sliding_layer_reads_the_swa_head_fields_and_a_global_one_the_plain_ones() {
        let mut t = cfg().text_config;
        assert_eq!(
            [
                t.swa_num_attention_heads,
                t.swa_num_key_value_heads,
                t.swa_head_dim
            ],
            [t.num_attention_heads, t.num_key_value_heads, t.head_dim],
            "a checkpoint whose two sets differ would settle this on its own"
        );
        t.swa_num_attention_heads = 16;
        t.swa_num_key_value_heads = 4;
        t.swa_head_dim = 64;

        let (local, global) = (0, 5);
        assert!(t.layer_is_sliding(local) && !t.layer_is_sliding(global));

        let sliding = AttentionConfig::for_layer(&t, local);
        assert_eq!(
            [sliding.heads, sliding.kv_heads, sliding.head_dim],
            [16, 4, 64]
        );
        assert_eq!(sliding.sliding, t.sliding_window_size);
        assert!(
            sliding.log_scaling.is_none(),
            "log scaling is a global layer's"
        );

        let full = AttentionConfig::for_layer(&t, global);
        assert_eq!([full.heads, full.kv_heads, full.head_dim], [32, 8, 128]);
        assert_eq!(full.sliding, 0, "a global layer has no window");
        assert!(full.log_scaling.is_some());
    }

    #[test]
    fn only_first_two_layers_are_dense() {
        let t = cfg().text_config;
        let dense: Vec<_> = (0..t.num_hidden_layers)
            .filter(|&i| t.layer_is_dense(i))
            .collect();
        assert_eq!(dense, vec![0, 1]);
    }

    #[test]
    fn kv_footprint_scales_only_with_global_layers() {
        let t = cfg().text_config;
        let kv = t.kv_footprint(2);

        assert_eq!(kv.bytes_per_token, 7 * 2 * 8 * 128 * 2);
        assert_eq!(kv.fixed_bytes, 35 * 2 * 8 * 128 * 512 * 2);

        // A full 1M-token context stays under 30 GiB, which is what makes
        // deep batching viable on a single machine.
        let at_1m = kv.bytes_at(1_048_576);
        assert!(at_1m < 30 << 30, "1M ctx = {at_1m} bytes");
    }

    #[test]
    fn mtp_layers_are_present_in_config() {
        let m = cfg().mtp_config.expect("mtp_config");
        assert_eq!(m.num_nextn_predict_layers, 8);
        assert_eq!(m.local_layer_ids, vec![0, 2, 4, 5, 6, 7]);
    }
}
