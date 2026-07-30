use anyhow::Result;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_default();
    if path.is_empty() {
        eprintln!("usage: inklingrs <path-to-config.json>");
        std::process::exit(2);
    }

    let cfg: inkling_core::Config = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
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
