//! What the binary was asked to do.
//!
//! Hand-rolled rather than derived from a schema. The whole surface is a handful
//! of words, and what a parser crate would buy — help text generated from the
//! same declaration, shell completions, subcommand trees — is not what this
//! needs. What it does need is that a wrong invocation says which word was wrong,
//! and that the saying of it is a value a test can match on rather than a
//! message a test has to grep.

use std::path::PathBuf;

/// What to run, and what against.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    /// Summarise a config's architecture and its KV cost.
    Inspect { config: PathBuf },
}

pub const USAGE: &str = "usage:\n  inklingrs inspect <config.json>";

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ArgError {
    #[error("no command given")]
    NoCommand,

    #[error("{0} is not a command")]
    UnknownCommand(String),

    #[error("{command} takes {what}")]
    Missing {
        command: &'static str,
        what: &'static str,
    },

    #[error("unexpected argument {0}")]
    Unexpected(String),
}

impl Command {
    /// The arguments after the program's own name.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, ArgError> {
        let mut args = args.into_iter();
        let command = args.next().ok_or(ArgError::NoCommand)?;
        match command.as_str() {
            "inspect" => inspect(args),
            _ => Err(ArgError::UnknownCommand(command)),
        }
    }
}

fn inspect(mut args: impl Iterator<Item = String>) -> Result<Command, ArgError> {
    let config = args.next().ok_or(ArgError::Missing {
        command: "inspect",
        what: "a path to a config.json",
    })?;
    match args.next() {
        Some(extra) => Err(ArgError::Unexpected(extra)),
        None => Ok(Command::Inspect {
            config: PathBuf::from(config),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Command, ArgError> {
        Command::parse(args.iter().map(|arg| arg.to_string()))
    }

    #[test]
    fn inspect_takes_the_path_it_is_given() {
        assert_eq!(
            parse(&["inspect", "models/Inkling-Small-mxfp4/config.json"]),
            Ok(Command::Inspect {
                config: PathBuf::from("models/Inkling-Small-mxfp4/config.json"),
            })
        );
    }

    #[test]
    fn no_arguments_at_all_is_refused() {
        assert_eq!(parse(&[]), Err(ArgError::NoCommand));
    }

    /// A path where a command belongs, which is what the invocation this binary
    /// used to accept — `inklingrs <config.json>` — now looks like.
    #[test]
    fn a_command_this_binary_does_not_have_is_refused() {
        assert_eq!(
            parse(&["config.json"]),
            Err(ArgError::UnknownCommand("config.json".to_string()))
        );
    }

    #[test]
    fn inspect_without_a_path_is_refused() {
        assert!(matches!(
            parse(&["inspect"]),
            Err(ArgError::Missing {
                command: "inspect",
                ..
            })
        ));
    }

    /// Silently ignoring the rest would run against the first of two configs a
    /// caller meant to compare.
    #[test]
    fn inspect_refuses_a_second_path() {
        assert_eq!(
            parse(&["inspect", "one.json", "two.json"]),
            Err(ArgError::Unexpected("two.json".to_string()))
        );
    }
}
