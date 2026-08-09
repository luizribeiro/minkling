use clap::{Parser, Subcommand};

/// Run Inkling models behind a small, reviewed host.
#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the model over HTTP.
    Serve,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve => {}
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};

    #[test]
    fn serve_is_a_command() {
        let cli = Cli::try_parse_from(["minkling", "serve"]).expect("serve should parse");

        assert!(matches!(cli.command, Command::Serve));
    }
}
