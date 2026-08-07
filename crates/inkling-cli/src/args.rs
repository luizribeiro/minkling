//! What the binary was asked to do.
//!
//! Hand-rolled rather than derived from a schema. The whole surface is a handful
//! of words, and what a parser crate would buy — help text generated from the
//! same declaration, shell completions, subcommand trees — is not what this
//! needs. What it does need is that a wrong invocation says which word was wrong,
//! and that the saying of it is a value a test can match on rather than a
//! message a test has to grep.

use std::path::PathBuf;

use inkling_metal::Numerics;

use inkling_core::DEFAULT_BOUND;

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

/// Where the model's weights are multiplied against.
///
/// A runtime flag rather than a Cargo feature, and the reason is what the CPU
/// path is for. It is the oracle every kernel in this tree is validated against,
/// so a disagreement between the two is settled by putting the same prompt
/// through both — which a flag allows and a build-time choice would turn into a
/// rebuild. It is also the first thing a report about a wrong token has to say,
/// and a flag can be printed back where a feature has to be remembered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Every weight decoded on the way through, a row at a time — 9.0 s a
    /// token. The path every fixture in the tree pins.
    Cpu,
    /// `lm_head`, the experts and every layer's own projections multiplied on
    /// the GPU against codes that are never decoded, which is 0.055 s a token.
    ///
    /// The default, because the two produce the same tokens and this is the
    /// faster of them. Asking for it explicitly is still worth allowing: a
    /// command line that says which backend ran is a command line that can be
    /// pasted into a bug report.
    #[default]
    Metal,
}

impl Backend {
    fn parse(name: &str) -> Result<Self, ArgError> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            _ => Err(ArgError::UnknownBackend(name.to_string())),
        }
    }
}

/// Which arithmetic the device's innermost accumulation may use, and the refusal
/// that keeps the word from being asked of a backend that has none.
///
/// **`--numerics production --backend cpu` is a mistake rather than a request.**
/// The CPU path is the oracle both device paths are measured against and it has
/// exactly one arithmetic; a run that took the flag and dropped it would print a
/// command line saying something other than what it did, which is the same rule
/// `bench` refuses a `--context` on a prefill under.
fn numerics(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<Numerics, ArgError> {
    let name = value(flag, args)?;
    Numerics::parse(&name).ok_or(ArgError::UnknownNumerics { word: name })
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
    /// Where the weights are multiplied against.
    pub backend: Backend,
    /// Which arithmetic the device's innermost accumulation uses, which the
    /// backend above has to be `metal` for anyone to have asked.
    pub numerics: Numerics,
    /// How many tokens a round guesses ahead with the multi-token prediction
    /// heads, and zero for a generation that decodes one at a time.
    ///
    /// A number rather than a flag because the depth that pays is the whole
    /// question — the study measured the best pooled depth at 2 and the payoff
    /// varying sixfold across workloads — and because the heads are 4.2 GiB a
    /// caller has a right not to load.
    pub speculate: usize,
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
    /// Where the weights are multiplied against.
    pub backend: Backend,
    /// Which arithmetic the device's innermost accumulation uses.
    pub numerics: Numerics,
    /// The most positions a conversation is kept at between requests, and zero
    /// for a server that keeps nothing — which is the arm every figure for this
    /// is measured against.
    pub reuse_tokens: usize,
    /// How many requests the engine advances together.
    ///
    /// One is a server that answers in the order the requests arrived and keeps
    /// a conversation between them; more is the scheduler, which does neither.
    /// See [`crate::serve`], where what the two paths are is stated.
    pub slots: usize,
}

/// How many requests `serve` advances together when nobody says.
///
/// **One, which is the serial path and not a scheduler of width one.** The two
/// differ in what they can do besides batch — a slot's cache belongs to the slot
/// and is fresh at every handover, so an engine of any width forgets the
/// conversation between turns, and keeping it is worth more than any kernel in
/// this tree. A default that quietly gave that up to reach a batch nobody asked
/// for would be a default that made the common case slower.
pub const DEFAULT_SLOTS: usize = 1;

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
/// server's would be, because 64 tokens is ten seconds on the device path and
/// ten minutes on the CPU's. A client that wants more asks for more.
pub const DEFAULT_SERVE_MAX_TOKENS: usize = 64;

/// Where the server listens when nobody says. Loopback: this is one process
/// holding a 16.7 GiB model with no authentication of any kind in front of it,
/// and putting that on every interface is not a default anyone should get by
/// omission.
pub const DEFAULT_ADDRESS: &str = "127.0.0.1:8080";

/// The words `--numerics` takes, spelled as a command line offers them.
///
/// **Read off [`Numerics::EVERY`] rather than written out**, so that a word
/// added to the flag reaches the usage line, the refusal below and the two
/// commands' help at once. A third word arrived with nothing but this function
/// and one match arm, and the two places that used to spell the list are what
/// would otherwise have gone on offering two.
fn every_numerics() -> String {
    Numerics::EVERY
        .into_iter()
        .map(Numerics::named)
        .collect::<Vec<&str>>()
        .join("|")
}

pub fn usage() -> String {
    let numerics = every_numerics();
    format!(
        "usage:\n  \
        inklingrs inspect <config.json>\n  \
        inklingrs generate <checkpoint-dir> --prompt <text> [--max-tokens <n>] \
            [--backend cpu|metal] [--numerics {numerics}] [--speculate <k>]\n  \
        inklingrs serve <checkpoint-dir> [--address <host:port>] [--max-tokens <n>] \
            [--backend cpu|metal] [--numerics {numerics}] \
            [--reuse-tokens <n>] [--slots <n>]"
    )
}

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

    #[error("{0} is not a depth, which is zero or more")]
    NotADepth(String),

    #[error("{0} is not a count of positions to keep, which is zero or more")]
    NotABound(String),

    #[error("{0} is not a backend, which is cpu or metal")]
    UnknownBackend(String),

    #[error("{word} is not numerics, which is one of {}", every_numerics())]
    UnknownNumerics { word: String },

    #[error("--numerics {0} takes a device: the cpu backend has one arithmetic")]
    NumericsOffDevice(&'static str),
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

/// A count that may be zero, which is the one number [`count`] refuses and which
/// two flags mean something by: `--speculate 0` is a generation that decodes one
/// token at a time, and `--reuse-tokens 0` is a server that keeps nothing. Both
/// are exactly what the flag's absence asks for in one case and the arm a
/// measurement is taken against in the other.
///
/// `wrong` is the refusal, because the two say different things about what a
/// number that would not parse was meant to be.
fn zeroable(
    flag: &str,
    args: &mut impl Iterator<Item = String>,
    wrong: fn(String) -> ArgError,
) -> Result<usize, ArgError> {
    let number = value(flag, args)?;
    number.parse().map_err(|_| wrong(number))
}

fn generate(args: impl Iterator<Item = String>) -> Result<Command, ArgError> {
    let mut args = args.into_iter();
    let mut checkpoint = None;
    let mut prompt = None;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut backend = Backend::default();
    let mut asked = None;
    let mut speculate = 0;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prompt" | "-p" => prompt = Some(value(&arg, &mut args)?),
            "--max-tokens" | "-n" => max_tokens = count(&arg, &mut args)?,
            "--backend" | "-b" => backend = Backend::parse(&value(&arg, &mut args)?)?,
            "--numerics" => asked = Some(numerics(&arg, &mut args)?),
            "--speculate" | "-k" => speculate = zeroable(&arg, &mut args, ArgError::NotADepth)?,
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
        backend,
        numerics: on_device(backend, asked)?,
        speculate,
    }))
}

/// The numerics a run was asked for, refused where the backend has none to
/// select between.
fn on_device(backend: Backend, asked: Option<Numerics>) -> Result<Numerics, ArgError> {
    match (backend, asked) {
        (Backend::Cpu, Some(numerics)) => Err(ArgError::NumericsOffDevice(numerics.named())),
        (_, asked) => Ok(asked.unwrap_or_default()),
    }
}

fn serve(args: impl Iterator<Item = String>) -> Result<Command, ArgError> {
    let mut args = args.into_iter();
    let mut checkpoint = None;
    let mut address = DEFAULT_ADDRESS.to_string();
    let mut max_tokens = DEFAULT_SERVE_MAX_TOKENS;
    let mut backend = Backend::default();
    let mut asked = None;
    let mut reuse_tokens = DEFAULT_BOUND;
    let mut slots = DEFAULT_SLOTS;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--address" | "-a" => address = value(&arg, &mut args)?,
            "--max-tokens" | "-n" => max_tokens = count(&arg, &mut args)?,
            "--slots" => slots = count(&arg, &mut args)?,
            "--reuse-tokens" => reuse_tokens = zeroable(&arg, &mut args, ArgError::NotABound)?,
            "--backend" | "-b" => backend = Backend::parse(&value(&arg, &mut args)?)?,
            "--numerics" => asked = Some(numerics(&arg, &mut args)?),
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
        backend,
        numerics: on_device(backend, asked)?,
        reuse_tokens,
        slots,
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
                backend: Backend::Metal,
                numerics: Numerics::Reference,
                speculate: 0,
            })
        );
    }

    /// The heads are off unless a depth asks for them, which is what keeps a
    /// generation that names none from loading 4.2 GiB it will not multiply.
    #[test]
    fn generate_speculates_only_when_a_depth_says_to() {
        let with = |flag: &[&str]| {
            let mut args = vec!["generate", "models/small", "--prompt", "Once"];
            args.extend_from_slice(flag);
            generated(&args).expect("parses").speculate
        };
        assert_eq!(with(&[]), 0);
        assert_eq!(with(&["--speculate", "2"]), 2);
        assert_eq!(with(&["-k", "8"]), 8);
        // A depth of zero is what the absence of the flag means, so naming it
        // is a request rather than the mistake a budget of zero is.
        assert_eq!(with(&["--speculate", "0"]), 0);
    }

    /// A depth that is not a number is still refused, which is what says the
    /// zero above is a case and not a parse that gave up.
    #[test]
    fn a_speculation_depth_that_is_not_a_number_is_refused() {
        for depth in ["-1", "two", ""] {
            assert_eq!(
                generated(&["generate", "models/small", "-p", "Once", "-k", depth]),
                Err(ArgError::NotADepth(depth.to_string())),
                "--speculate {depth}"
            );
        }
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
    /// 9.0 s a token on the CPU path an unbounded default would be a hang.
    #[test]
    fn a_generation_that_names_no_budget_gets_the_default() {
        assert_eq!(
            generated(&["generate", "models/small", "--prompt", "Once"])
                .expect("parses")
                .max_tokens,
            DEFAULT_MAX_TOKENS
        );
    }

    /// The backend is a word and each command reads the same two, which is what
    /// makes "run it on the CPU" the same edit to either command line.
    #[test]
    fn either_command_takes_a_backend_by_name() {
        for backend in [("cpu", Backend::Cpu), ("metal", Backend::Metal)] {
            let (name, want) = backend;
            assert_eq!(
                generated(&["generate", "models/small", "-p", "Once", "--backend", name])
                    .expect("parses")
                    .backend,
                want
            );
            assert_eq!(
                serving(&["serve", "models/small", "-b", name])
                    .expect("parses")
                    .backend,
                want
            );
        }
    }

    /// What a run nobody said anything about gets, which is the fast one. The
    /// CPU path is what a fixture is compared against and not what a caller
    /// should have to wait for by omission.
    #[test]
    fn a_command_that_names_no_backend_runs_on_the_gpu() {
        assert_eq!(
            generated(&["generate", "models/small", "-p", "Once"])
                .expect("parses")
                .backend,
            Backend::Metal
        );
        assert_eq!(
            serving(&["serve", "models/small"]).expect("parses").backend,
            Backend::Metal
        );
    }

    /// A backend this binary does not have is a mistake and not a fallback: a
    /// typo silently answered by the default would report the timings of a
    /// backend nobody asked for.
    #[test]
    fn a_backend_this_binary_does_not_have_is_refused() {
        for name in ["gpu", "gpU", "gpu ", "cuda", ""] {
            assert_eq!(
                parse(&["generate", "models/small", "-p", "Once", "-b", name]),
                Err(ArgError::UnknownBackend(name.to_string())),
                "{name:?}"
            );
        }
    }

    /// The numerics are a word the same way the backend is, and each command
    /// reads the same list.
    ///
    /// **Walked off [`Numerics::EVERY`] rather than spelled here**, so that a
    /// fourth word is a case that runs it rather than a case that keeps passing
    /// without it.
    #[test]
    fn either_command_takes_its_numerics_by_name() {
        for want in Numerics::EVERY {
            let name = want.named();
            assert_eq!(
                generated(&["generate", "models/small", "-p", "Once", "--numerics", name])
                    .expect("parses")
                    .numerics,
                want
            );
            assert_eq!(
                serving(&["serve", "models/small", "--numerics", name])
                    .expect("parses")
                    .numerics,
                want
            );
        }
    }

    /// **Every word the flag takes is offered by the line that says what it
    /// takes**, which is the drift a usage string spelled by hand invites: a
    /// word reachable on the command line and absent from the help is one
    /// nobody finds.
    #[test]
    fn the_usage_line_and_the_refusal_offer_every_word() {
        let (usage, refused) = (
            usage(),
            ArgError::UnknownNumerics {
                word: "half".into(),
            }
            .to_string(),
        );
        for numerics in Numerics::EVERY {
            assert!(usage.contains(numerics.named()), "usage: {numerics:?}");
            assert!(refused.contains(numerics.named()), "refusal: {numerics:?}");
        }
    }

    /// **What a run nobody said anything about gets, which is the reference.**
    /// The whole of what the other side of this flag is for is being measured,
    /// and a default that drifted to it would make every figure in this file a
    /// figure about arithmetic nobody asked for.
    #[test]
    fn a_command_that_names_no_numerics_runs_the_reference() {
        assert_eq!(
            generated(&["generate", "models/small", "-p", "Once"])
                .expect("parses")
                .numerics,
            Numerics::Reference
        );
        assert_eq!(
            serving(&["serve", "models/small"])
                .expect("parses")
                .numerics,
            Numerics::Reference
        );
    }

    #[test]
    fn numerics_this_binary_does_not_have_are_refused() {
        for name in ["fast", "mma", "Reference", "reference ", ""] {
            assert_eq!(
                parse(&["generate", "models/small", "-p", "Once", "--numerics", name]),
                Err(ArgError::UnknownNumerics {
                    word: name.to_string()
                }),
                "{name:?}"
            );
        }
    }

    /// **The CPU path has one arithmetic, so a word choosing between two is a
    /// mistake there rather than a request.** A run that took the flag and
    /// dropped it would print a command line saying something other than what it
    /// did — and this flag's whole job is to be readable off the line that ran.
    #[test]
    fn numerics_asked_of_the_cpu_backend_are_refused() {
        for name in Numerics::EVERY.map(Numerics::named) {
            for command in [
                vec!["generate", "models/small", "-p", "Once"],
                vec!["serve", "models/small"],
            ] {
                let mut line = command.clone();
                line.extend(["--backend", "cpu", "--numerics", name]);
                assert!(
                    matches!(parse(&line), Err(ArgError::NumericsOffDevice(_))),
                    "{line:?}"
                );
            }
        }
    }

    /// And the refusal is about the pair rather than about either word: the same
    /// flag on the device path parses, and the CPU path with no flag parses.
    #[test]
    fn the_cpu_backend_takes_no_numerics_and_needs_none() {
        assert_eq!(
            generated(&["generate", "models/small", "-p", "Once", "-b", "cpu"])
                .expect("parses")
                .numerics,
            Numerics::Reference
        );
        assert_eq!(
            generated(&[
                "generate",
                "models/small",
                "-p",
                "Once",
                "-b",
                "metal",
                "--numerics",
                "production"
            ])
            .expect("parses")
            .numerics,
            Numerics::Production
        );
    }

    /// **The pair is read after the whole line rather than as the words arrive**,
    /// so which of the two was written first cannot decide whether the run is
    /// refused — and a repeated backend is the last one, the way every other
    /// flag here is.
    #[test]
    fn which_of_the_two_words_came_first_decides_nothing() {
        assert!(matches!(
            parse(&[
                "generate",
                "models/small",
                "-p",
                "Once",
                "--numerics",
                "production",
                "-b",
                "cpu"
            ]),
            Err(ArgError::NumericsOffDevice(_))
        ));
        assert_eq!(
            generated(&[
                "generate",
                "models/small",
                "-p",
                "Once",
                "--numerics",
                "production",
                "-b",
                "metal"
            ])
            .expect("parses")
            .numerics,
            Numerics::Production
        );
        assert!(matches!(
            parse(&[
                "generate",
                "models/small",
                "-p",
                "Once",
                "-b",
                "metal",
                "--numerics",
                "production",
                "-b",
                "cpu"
            ]),
            Err(ArgError::NumericsOffDevice(_))
        ));
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
                backend: Backend::Metal,
                numerics: Numerics::Reference,
                reuse_tokens: DEFAULT_BOUND,
                slots: DEFAULT_SLOTS,
            })
        );
    }

    /// The flag that reaches the scheduler. One is the serial path and its kept
    /// conversation, which is what a server that says nothing gets.
    #[test]
    fn serve_takes_the_slots_the_engine_advances_together() {
        assert_eq!(
            serving(&["serve", "models/small"]).map(|it| it.slots),
            Ok(1)
        );
        assert_eq!(
            serving(&["serve", "models/small", "--slots", "8"]).map(|it| it.slots),
            Ok(8)
        );
        // An engine with no slots seats nobody, which the scheduler refuses one
        // layer down and a command line should refuse before it loads 16.7 GiB.
        assert_eq!(
            serving(&["serve", "models/small", "--slots", "0"]),
            Err(ArgError::NotACount("0".to_string()))
        );
    }

    /// The one number `--max-tokens` refuses and this one means something by: a
    /// server that keeps nothing between requests, which is the arm every figure
    /// for the kept cache is measured against.
    #[test]
    fn serve_takes_no_positions_to_keep_as_keeping_nothing() {
        assert_eq!(
            serving(&["serve", "models/small", "--reuse-tokens", "0"])
                .map(|serve| serve.reuse_tokens),
            Ok(0)
        );
        assert_eq!(
            serving(&["serve", "models/small", "--reuse-tokens", "many"]),
            Err(ArgError::NotABound("many".to_string()))
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
