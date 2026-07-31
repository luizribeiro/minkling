//! Reading a `config.json`, which is where every command starts.

use std::path::Path;

use anyhow::{Context, Result};
use inkling_core::Config;

/// The whole config, not just its `text_config`: the architecture is under that
/// key and the end-of-sequence id is beside it.
///
/// Both failures name the file. A checkpoint directory holds thirty shards and
/// four json files, and "No such file or directory" on its own leaves a caller
/// guessing which of them a typo cost them.
pub fn read(path: &Path) -> Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}
