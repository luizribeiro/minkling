use std::process::ExitCode;

use inkling_cli::args::{Command, usage};
use inkling_cli::{generate, inspect, serve};

/// What an invocation that was never understood exits with, apart from one that
/// ran and failed. A caller scripting this can tell a typo from a config that
/// would not load.
const MISUSED: u8 = 2;

fn main() -> ExitCode {
    let command = match Command::parse(std::env::args().skip(1)) {
        Ok(command) => command,
        Err(err) => {
            eprintln!("{err}\n\n{}", usage());
            return ExitCode::from(MISUSED);
        }
    };

    let ran = match command {
        Command::Inspect { config } => inspect::run(&config),
        Command::Generate(args) => generate::run(&args),
        Command::Serve(args) => serve::run(&args),
    };
    match ran {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::FAILURE
        }
    }
}
