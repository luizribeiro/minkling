mod api;
mod engine;
mod server;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use inkling_inference::{DEFAULT_REUSE_TOKENS, Numerics, Options};

const DEFAULT_ADDRESS: &str = "127.0.0.1:8080";
const DEFAULT_MAX_TOKENS: usize = 64;

/// Run Inkling models behind a small, reviewed host.
#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the model over HTTP.
    Serve(Serve),
}

#[derive(Debug, Args)]
struct Serve {
    /// Checkpoint directory to load.
    checkpoint: PathBuf,

    /// Address to listen on.
    #[arg(long, default_value = DEFAULT_ADDRESS)]
    address: SocketAddr,

    /// Maximum tokens per request, also used when a request sets no limit.
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS, value_parser = positive)]
    max_tokens: usize,

    /// Metal kernel numerics.
    #[arg(long, default_value = "reference", value_parser = parse_numerics)]
    numerics: Numerics,

    /// Maximum prompt positions to retain between turns; zero disables reuse.
    #[arg(long, default_value_t = DEFAULT_REUSE_TOKENS)]
    reuse_tokens: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve(serve) => {
            let inference = engine::Client::start(Options {
                checkpoint: serve.checkpoint,
                max_tokens: serve.max_tokens,
                numerics: serve.numerics,
                reuse_tokens: serve.reuse_tokens,
            })?;
            server::run(serve.address, Arc::new(inference)).await?;
            Ok(())
        }
    }
}

fn positive(value: &str) -> Result<usize, String> {
    match value.parse() {
        Ok(value) if value > 0 => Ok(value),
        _ => Err("must be a count of at least one".to_string()),
    }
}

fn parse_numerics(value: &str) -> Result<Numerics, String> {
    Numerics::parse(value).ok_or_else(|| "must be reference, production, or rounded".to_string())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};

    #[test]
    fn serve_is_a_command() {
        let cli =
            Cli::try_parse_from(["minkling", "serve", "checkpoint"]).expect("serve should parse");

        assert!(matches!(cli.command, Command::Serve(_)));
    }
}
