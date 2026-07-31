//! `inspect`: what a config says the model is, and what a sequence of it costs.

use std::path::Path;

use anyhow::Result;

use crate::config;

pub fn run(path: &Path) -> Result<()> {
    let cfg = config::read(path)?;
    let t = &cfg.text_config;
    let kv = t.kv_footprint(2);

    println!("layers          {}", t.num_hidden_layers);
    println!("global layers   {:?}", t.global_layers());
    println!("dense FFN       {}", t.dense_intermediate_size);
    println!("expert FFN      {}", t.moe_intermediate_size);
    println!(
        "experts         {} (top-{})",
        t.n_routed_experts, t.num_experts_per_tok
    );
    println!("kv/token        {} B", kv.bytes_per_token);
    println!(
        "kv @ 1M ctx     {:.1} GiB",
        kv.bytes_at(1 << 20) as f64 / (1u64 << 30) as f64
    );
    Ok(())
}
