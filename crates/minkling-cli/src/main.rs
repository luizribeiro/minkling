mod server;

use std::net::SocketAddr;

use clap::{Args, Parser, Subcommand};

const DEFAULT_ADDRESS: &str = "127.0.0.1:8080";

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
    /// Address to listen on.
    #[arg(long, default_value = DEFAULT_ADDRESS)]
    address: SocketAddr,
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve(serve) => server::run(serve.address).await,
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};

    #[test]
    fn serve_is_a_command() {
        let cli = Cli::try_parse_from(["minkling", "serve"]).expect("serve should parse");

        assert!(matches!(cli.command, Command::Serve(_)));
    }
}
