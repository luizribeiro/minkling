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
    /// Continue a prompt, a token at a time.
    Generate(Generate),
    /// Answer chat completions over HTTP against a loaded checkpoint.
    Serve(Serve),
}

/// A generation, as a command line describes one.
#[derive(Debug, PartialEq, Eq)]
pub struct Generate {
    /// The checkpoint directory — its `config.json`, its `tokenizer.json` and
    /// its shards — rather than any one file of it, because a generation needs
    /// all three and only the directory names all three.
    pub checkpoint: PathBuf,
    /// The text to continue, sent to the tokenizer as it stands. See
    /// [`crate::generate`] for why nothing here templates it.
    pub prompt: String,
    /// How many tokens may be decoded before the budget ends the generation.
    pub max_tokens: usize,
}

/// A server, as a command line describes one.
#[derive(Debug, PartialEq, Eq)]
pub struct Serve {
    /// The checkpoint directory, loaded once at startup and served against.
    pub checkpoint: PathBuf,
    /// What to listen on, as `tiny_http` takes it.
    pub address: String,
    /// The budget for a request that names none of its own.
    pub max_tokens: usize,
}

/// How many tokens `generate` decodes when the caller names no budget.
///
/// Eight, which is what the oracle's recorded continuation is. A short prompt
/// and eight tokens is about two minutes end to end on the CPU path — half of
/// it the prefill, half of it seven decode steps at 9 s each. A handful, in
/// other words, because a paragraph is an hour.
pub const DEFAULT_MAX_TOKENS: usize = 8;

/// The budget a request that names none is served under.
///
/// Larger than `generate`'s, because a templated turn spends tokens on
/// `<|content_thinking|>` before it says anything, and smaller than any chat
/// server's would be, because 64 tokens is ten minutes at 9.2 s a step. A client
/// that wants more asks for more.
pub const DEFAULT_SERVE_MAX_TOKENS: usize = 64;

/// Where the server listens when nobody says. Loopback: this is one process
/// holding a 16.7 GiB model with no authentication of any kind in front of it,
/// and putting that on every interface is not a default anyone should get by
/// omission.
pub const DEFAULT_ADDRESS: &str = "127.0.0.1:8080";

pub const USAGE: &str = "usage:\n  \
    inklingrs inspect <config.json>\n  \
    inklingrs generate <checkpoint-dir> --prompt <text> [--max-tokens <n>]\n  \
    inklingrs serve <checkpoint-dir> [--address <host:port>] [--max-tokens <n>]";

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

    #[error("{0} takes a value")]
    MissingValue(String),

    #[error("{0} is not a count of at least one")]
    NotACount(String),
}

impl Command {
    /// The arguments after the program's own name.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, ArgError> {
        let mut args = args.into_iter();
        let command = args.next().ok_or(ArgError::NoCommand)?;
        match command.as_str() {
            "inspect" => inspect(args),
            "generate" => generate(args),
            "serve" => serve(args),
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

/// The value of a flag, which is whatever follows it — a prompt beginning with
/// a dash is a prompt and not a flag.
fn value(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<String, ArgError> {
    args.next()
        .ok_or_else(|| ArgError::MissingValue(flag.to_string()))
}

/// A budget, which is a count of at least one. Nothing at all is decoded under a
/// budget of zero — not even the prompt's prefill, which only the first step
/// runs — so it is a mistake rather than a request, in either command.
fn count(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<usize, ArgError> {
    let count = value(flag, args)?;
    match count.parse() {
        Ok(parsed) if parsed > 0 => Ok(parsed),
        _ => Err(ArgError::NotACount(count)),
    }
}

fn generate(args: impl Iterator<Item = String>) -> Result<Command, ArgError> {
    let mut args = args.into_iter();
    let mut checkpoint = None;
    let mut prompt = None;
    let mut max_tokens = DEFAULT_MAX_TOKENS;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prompt" | "-p" => prompt = Some(value(&arg, &mut args)?),
            "--max-tokens" | "-n" => max_tokens = count(&arg, &mut args)?,
            _ if arg.starts_with('-') => return Err(ArgError::Unexpected(arg)),
            _ if checkpoint.is_none() => checkpoint = Some(PathBuf::from(arg)),
            _ => return Err(ArgError::Unexpected(arg)),
        }
    }

    Ok(Command::Generate(Generate {
        checkpoint: checkpoint.ok_or(ArgError::Missing {
            command: "generate",
            what: "a path to a checkpoint directory",
        })?,
        // An empty prompt is no prompt: it encodes to no tokens, and a forward
        // pass over none is refused several layers down where the message would
        // read as a fault rather than as a mistake anyone made here.
        prompt: prompt
            .filter(|text| !text.is_empty())
            .ok_or(ArgError::Missing {
                command: "generate",
                what: "--prompt <text>",
            })?,
        max_tokens,
    }))
}

fn serve(args: impl Iterator<Item = String>) -> Result<Command, ArgError> {
    let mut args = args.into_iter();
    let mut checkpoint = None;
    let mut address = DEFAULT_ADDRESS.to_string();
    let mut max_tokens = DEFAULT_SERVE_MAX_TOKENS;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--address" | "-a" => address = value(&arg, &mut args)?,
            "--max-tokens" | "-n" => max_tokens = count(&arg, &mut args)?,
            _ if arg.starts_with('-') => return Err(ArgError::Unexpected(arg)),
            _ if checkpoint.is_none() => checkpoint = Some(PathBuf::from(arg)),
            _ => return Err(ArgError::Unexpected(arg)),
        }
    }

    Ok(Command::Serve(Serve {
        checkpoint: checkpoint.ok_or(ArgError::Missing {
            command: "serve",
            what: "a path to a checkpoint directory",
        })?,
        address,
        max_tokens,
    }))
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

    fn generated(args: &[&str]) -> Result<Generate, ArgError> {
        match parse(args)? {
            Command::Generate(generate) => Ok(generate),
            other => panic!("{args:?} parsed as {other:?}"),
        }
    }

    #[test]
    fn generate_takes_a_checkpoint_a_prompt_and_a_budget() {
        assert_eq!(
            generated(&[
                "generate",
                "models/small",
                "--prompt",
                "Once",
                "--max-tokens",
                "3"
            ]),
            Ok(Generate {
                checkpoint: PathBuf::from("models/small"),
                prompt: "Once".to_string(),
                max_tokens: 3,
            })
        );
    }

    /// The flags may come before the path as readily as after it. Nothing here
    /// is positional except the one path.
    #[test]
    fn generate_reads_its_flags_in_any_order() {
        assert_eq!(
            generated(&["generate", "-n", "3", "-p", "Once", "models/small"]),
            generated(&[
                "generate",
                "models/small",
                "--prompt",
                "Once",
                "--max-tokens",
                "3"
            ]),
        );
    }

    /// A budget nobody named still has to be one a caller can wait out. At
    /// 9.2 s a token an unbounded default would be a hang.
    #[test]
    fn a_generation_that_names_no_budget_gets_the_default() {
        assert_eq!(
            generated(&["generate", "models/small", "--prompt", "Once"])
                .expect("parses")
                .max_tokens,
            DEFAULT_MAX_TOKENS
        );
    }

    /// A prompt is text and not a flag, so what follows `--prompt` is taken as
    /// it stands — which is what lets a prompt carry the model's own turn
    /// markers, and what stops one that opens with a dash from being read as an
    /// unknown flag.
    #[test]
    fn a_prompt_is_whatever_follows_the_flag() {
        for prompt in ["-p", "--max-tokens", "<|message_model|>", "-- hush --"] {
            assert_eq!(
                generated(&["generate", "models/small", "--prompt", prompt])
                    .expect("parses")
                    .prompt,
                prompt
            );
        }
    }

    #[test]
    fn generate_without_a_checkpoint_is_refused() {
        assert!(matches!(
            parse(&["generate", "--prompt", "Once"]),
            Err(ArgError::Missing {
                command: "generate",
                ..
            })
        ));
    }

    /// A generation with nothing to continue. Refused here rather than at the
    /// tokenizer, which would have loaded a 130 GiB checkpoint first.
    #[test]
    fn generate_without_a_prompt_is_refused() {
        for args in [
            vec!["generate", "models/small"],
            vec!["generate", "models/small", "--prompt", ""],
        ] {
            assert!(
                matches!(
                    parse(&args),
                    Err(ArgError::Missing {
                        command: "generate",
                        ..
                    })
                ),
                "{args:?}"
            );
        }
    }

    #[test]
    fn a_flag_at_the_end_of_the_line_with_nothing_after_it_is_refused() {
        assert_eq!(
            parse(&["generate", "models/small", "--prompt"]),
            Err(ArgError::MissingValue("--prompt".to_string()))
        );
    }

    /// A budget of zero decodes nothing at all — not even the prefill, which
    /// only the first step runs — so it is a mistake rather than a request.
    #[test]
    fn a_budget_that_is_not_a_count_of_at_least_one_is_refused() {
        for count in ["0", "-1", "many", "3.5", ""] {
            assert_eq!(
                parse(&["generate", "models/small", "-p", "Once", "-n", count]),
                Err(ArgError::NotACount(count.to_string())),
                "{count:?}"
            );
        }
    }

    #[test]
    fn generate_refuses_a_flag_it_does_not_have() {
        assert_eq!(
            parse(&[
                "generate",
                "models/small",
                "-p",
                "Once",
                "--temperature",
                "0.7"
            ]),
            Err(ArgError::Unexpected("--temperature".to_string()))
        );
    }

    #[test]
    fn generate_refuses_a_second_checkpoint() {
        assert_eq!(
            parse(&["generate", "models/small", "models/other", "-p", "Once"]),
            Err(ArgError::Unexpected("models/other".to_string()))
        );
    }

    fn serving(args: &[&str]) -> Result<Serve, ArgError> {
        match parse(args)? {
            Command::Serve(serve) => Ok(serve),
            other => panic!("{args:?} parsed as {other:?}"),
        }
    }

    #[test]
    fn serve_takes_a_checkpoint_an_address_and_a_budget() {
        assert_eq!(
            serving(&["serve", "models/small", "-a", "0.0.0.0:9000", "-n", "3"]),
            Ok(Serve {
                checkpoint: PathBuf::from("models/small"),
                address: "0.0.0.0:9000".to_string(),
                max_tokens: 3,
            })
        );
    }

    /// A server nobody gave an address is on loopback, not on every interface.
    /// There is no authentication in front of it, so the difference is who can
    /// reach a 16.7 GiB model, and it should not turn on an argument being
    /// forgotten.
    #[test]
    fn a_server_that_names_no_address_listens_on_loopback() {
        let serve = serving(&["serve", "models/small"]).expect("parses");
        assert_eq!(serve.address, DEFAULT_ADDRESS);
        assert!(serve.address.starts_with("127.0.0.1:"), "{}", serve.address);
        assert_eq!(serve.max_tokens, DEFAULT_SERVE_MAX_TOKENS);
    }

    #[test]
    fn serve_without_a_checkpoint_is_refused() {
        assert!(matches!(
            parse(&["serve"]),
            Err(ArgError::Missing {
                command: "serve",
                ..
            })
        ));
    }

    /// The budget is a count of at least one wherever it is written, which is
    /// the one rule both commands share rather than each having its own.
    #[test]
    fn a_budget_that_is_not_a_count_is_refused_by_either_command() {
        for count in ["0", "-1", "many"] {
            assert_eq!(
                parse(&["serve", "models/small", "-n", count]),
                Err(ArgError::NotACount(count.to_string())),
                "{count:?}"
            );
            assert_eq!(
                parse(&["generate", "models/small", "-p", "Once", "-n", count]),
                Err(ArgError::NotACount(count.to_string())),
                "{count:?}"
            );
        }
    }

    #[test]
    fn serve_refuses_a_flag_it_does_not_have() {
        assert_eq!(
            parse(&["serve", "models/small", "--prompt", "Once"]),
            Err(ArgError::Unexpected("--prompt".to_string()))
        );
    }
}
