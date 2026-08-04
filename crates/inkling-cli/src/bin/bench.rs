//! Two builds of this engine, weighed against each other in one sitting.
//!
//! # What it is for
//!
//! Every figure in the README is paired and alternating: build A, build B, run
//! them one after the other with the order flipped each pair, and report whether
//! the ranges overlap. That discipline is what makes a 3% claim a claim rather
//! than a coin, and nothing here changes it.
//!
//! What it changes is the price. A flip used to mean checking out the other ref
//! and rebuilding the Metal crate — several times a milestone, across three
//! prompt lengths and up to seven pairs — and the rebuild bought nothing, because
//! the two binaries do not change between pairs. So this binary is both halves of
//! the arrangement: `alternate` drives two executables that were each built once,
//! and `decode`, `prefill` and `sweep` are what those executables do.
//!
//! # The protocol between the two halves
//!
//! An arm prints one reading a line to stdout — `name value unit` — and anything
//! it wants to say to a human on stderr. That is the whole contract, which is
//! what lets the harness be about alternation and statistics and know nothing
//! about what a decode step is. It also means an arm from an older commit works
//! here for as long as it prints the same names.
//!
//! # What it does not do
//!
//! It does not run measurements beside each other. `.config/nextest.toml` records
//! what a number taken beside another test is worth — M12 lost a day to a 15 ms
//! device-time regression that was the suite around it — and one process at a
//! time is as true here as it is there. One arm runs, then the other. Each opens
//! one Metal device, which at a second apiece is the reason a run measures as
//! much as it can once it has one.

use std::fmt::Write as _;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use inkling_cli::args::Backend;
use inkling_cli::kept::{DEFAULT_BOUND, Kept};
use inkling_cli::{backend, config, session};
use inkling_core::generate::{Proposer, Round};
use inkling_core::mtp::{CheckpointHeads, MtpProposer};
use inkling_core::workload::{
    BEST, CORPUS, DECODED, DIFFERENTIAL, REALISTIC, STRUCTURED_PROMPT, SWEPT, Session, tiled,
};
use inkling_core::{
    Checkpoint, CheckpointWeights, Ending, ModelCache, TextConfig, Tokenizer, profile,
};
use inkling_metal::{FusedAttention, Numerics, PackedMatmul};

/// How long a prefill's prompt is, when nobody says. The middle of the three
/// lengths this repo quotes.
const PREFILLED: usize = 385;

/// How many pairs `alternate` runs, when nobody says. Seven, which is what every
/// paired figure in the README was taken over.
const PAIRS: usize = 7;

const USAGE: &str = "usage:\n  \
    bench decode  <checkpoint> [--tokens <n>] [--context <n>] [--numerics <which>]\n  \
    bench prefill <checkpoint> [--tokens <n>] [--numerics <which>]\n  \
    bench sweep   <checkpoint> [--tokens <n>] [--depth <k>] [--numerics <which>]\n  \
    bench engines <checkpoint> [--depth <k>] [--numerics <which>]\n  \
    bench session <checkpoint> [--tokens <n>] [--reuse-tokens <n>] [--numerics <which>]\n  \
    bench guesses <checkpoint> <checkpoint> [--tokens <n>] [--depth <k>]\n  \
    bench diverge <checkpoint> [--tokens <n>]\n  \
    bench alternate [--pairs <n>] <a> <b> -- <arguments for both>";

/// What an invocation nobody could parse exits with, apart from one that ran and
/// failed.
const MISUSED: u8 = 2;

fn main() -> ExitCode {
    let job = match Job::parse(std::env::args().skip(1)) {
        Ok(job) => job,
        Err(err) => {
            eprintln!("{err:#}\n\n{USAGE}");
            return ExitCode::from(MISUSED);
        }
    };
    match job.run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::FAILURE
        }
    }
}

/// One number an arm reported, as the harness reads it back.
#[derive(Debug, Clone, PartialEq)]
struct Reading {
    name: String,
    value: f64,
    unit: String,
}

impl Reading {
    fn new(name: impl Into<String>, value: f64, unit: &str) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.to_string(),
        }
    }

    /// As it crosses between the two halves. Four decimals because the smallest
    /// thing anybody compares here is a hundredth of a millisecond and the
    /// rounding should not be this line's.
    fn line(&self) -> String {
        format!("{} {:.4} {}", self.name, self.value, self.unit)
    }
}

/// Every reading of one run, in the order the arm printed them.
fn readings(printed: &str) -> Result<Vec<Reading>> {
    printed
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            match (fields.next(), fields.next(), fields.next(), fields.next()) {
                (Some(name), Some(value), Some(unit), None) => Ok(Reading::new(
                    name,
                    value
                        .parse()
                        .with_context(|| format!("{value} is not a number, in {line:?}"))?,
                    unit,
                )),
                _ => bail!("{line:?} is not `name value unit`"),
            }
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum What {
    Decode,
    Prefill,
    Sweep,
    /// Everything a cross-engine table quotes, from one generation per
    /// (prompt, generated) pair. The one measurement here another engine can
    /// also report, which is what makes it the one taken against one.
    Engines,
    /// A simulated coding session, turn by turn.
    ///
    /// **The one measurement here whose subject is what happens *between* two
    /// requests**, which is why it is a session rather than a prompt: keeping a
    /// cache is worth nothing on a measurement of one call, and a prefill of a
    /// given length cannot be made to say otherwise.
    ///
    /// Its arms are the same binary told to keep a different number of
    /// positions, which is the shape `bench-numerics` puts one word through.
    Session,
}

#[derive(Debug, PartialEq, Eq)]
enum Job {
    /// One arm's worth of work: a checkpoint, and what to time against it.
    Measure {
        what: What,
        checkpoint: PathBuf,
        /// Tokens decoded, or — for a prefill — tokens in the prompt.
        tokens: usize,
        /// Keys the sequence already holds when the steps being timed run, which
        /// is the one axis a decode step moves along: the README's own table
        /// puts a step 1.45 times its 97-key figure at 8192 keys, on one build.
        ///
        /// **Zero is the structured prompt as it stands** rather than a context
        /// of no keys, which is what every decode figure in this repo was taken
        /// at before there was a flag for it.
        context: usize,
        depth: usize,
        /// Which arithmetic this arm's innermost accumulation uses.
        ///
        /// **This is what makes the two numerics pairable with one build.** An
        /// arm is a command line, so the thing that differs between them can be
        /// a word rather than an executable — the same shape `bench-weights`
        /// puts two checkpoints through.
        numerics: Numerics,
        /// Positions a session keeps between its turns, and zero for the arm
        /// that keeps nothing — which is the server as it was.
        reuse: usize,
    },
    /// The harness: two commands that already exist, run against each other.
    Alternate {
        pairs: usize,
        /// Each arm's own command line — an executable, and any arguments only
        /// that arm takes. Two builds against one checkpoint is a path apiece;
        /// one build against two checkpoints is the same path twice with a
        /// different directory after it, which is the shape a change to the
        /// weights rather than to the code has.
        arms: [Vec<String>; 2],
        args: Vec<String>,
    },
    /// Two chains of heads asked the same question at every round of one
    /// generation, and how often they answer it differently.
    Guesses {
        checkpoints: [PathBuf; 2],
        tokens: usize,
        depth: usize,
    },
    /// The same prompts through both numerics, and where their tokens part
    /// company.
    Diverge { checkpoint: PathBuf, tokens: usize },
}

impl Job {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let what = match args.next().as_deref() {
            Some("decode") => Some(What::Decode),
            Some("prefill") => Some(What::Prefill),
            Some("sweep") => Some(What::Sweep),
            Some("engines") => Some(What::Engines),
            Some("session") => Some(What::Session),
            Some("alternate") => return Self::alternating(args),
            Some("diverge") => return Self::diverging(args),
            Some("guesses") => None,
            Some(word) => bail!(
                "{word} is not one of decode, prefill, sweep, engines, session, guesses, \
                 diverge or alternate"
            ),
            None => bail!("no measurement given"),
        };

        let mut checkpoints = Vec::new();
        let mut tokens = None;
        let mut context = None;
        // An `Option` for the reason `context` above is one: the refusal below
        // is about whether the word was *given*, and a plain `Numerics` would
        // let `--numerics reference` through by being equal to the default.
        let mut numerics = None;
        // An `Option` for the reason `context` is one: what the refusal below is
        // about is whether the number was *given*, and zero is a number this
        // measurement means something by.
        let mut reuse = None;
        // A sweep runs every depth up to its own, where a cross-engine table
        // quotes one beside `k = 0` — so the default depth is what the flag
        // means to the measurement asking for it.
        let mut depth = match what {
            Some(What::Engines) => BEST,
            _ => SWEPT,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokens" | "-n" => tokens = Some(count(&arg, &mut args)?),
                "--context" | "-c" => context = Some(count(&arg, &mut args)?),
                "--depth" | "-k" => depth = count(&arg, &mut args)?,
                "--numerics" => numerics = Some(which(&arg, &mut args)?),
                "--reuse-tokens" => reuse = Some(positions(&arg, &mut args)?),
                _ if arg.starts_with('-') => bail!("unexpected argument {arg}"),
                _ => checkpoints.push(PathBuf::from(arg)),
            }
        }
        let Some(what) = what else {
            let [a, b] = <[PathBuf; 2]>::try_from(checkpoints).map_err(|given: Vec<PathBuf>| {
                anyhow::anyhow!("guesses takes two checkpoints, given {}", given.len())
            })?;
            if context.is_some() {
                bail!("guesses takes no --context: it asks both chains the same rounds");
            }
            // The heads are the arms of this one and the numerics are shared by
            // both of them, so a word naming one of the two would be a word this
            // measurement dropped.
            if numerics.is_some() {
                bail!("guesses takes no --numerics: its two arms are two sets of heads");
            }
            return Ok(Self::Guesses {
                checkpoints: [a, b],
                tokens: tokens.unwrap_or(DECODED),
                depth,
            });
        };
        let [checkpoint] =
            <[PathBuf; 1]>::try_from(checkpoints).map_err(|given: Vec<PathBuf>| {
                anyhow::anyhow!(
                    "a measurement takes one checkpoint directory, given {}",
                    given.len()
                )
            })?;
        // **A cross-engine table's lengths are the table's own.** It runs
        // [`REALISTIC`], which is four (prompt, generated) pairs rather than one
        // length — so a `--tokens` handed to it could only be dropped, and a
        // measurement that silently ignores the number it was given is one whose
        // rows say something other than what was asked for.
        if what == What::Engines && tokens.is_some() {
            bail!(
                "engines takes no --tokens: it runs {} pairs",
                REALISTIC.len()
            );
        }
        // **Only a decode step has a context to be taken at.** A prefill's
        // context is its own prompt and `--tokens` already says how long that
        // is; a sweep and a cross-engine table each fix their own prompt because
        // acceptance is the workload's. Silently dropping the number would leave
        // a row saying something other than what was asked for, which is the
        // same rule the length above is refused under.
        if what != What::Decode && context.is_some() {
            bail!("{what:?} takes no --context: only a decode step has one to be taken at");
        }
        // **Only a session has a between-requests to keep anything across.**
        // Every other measurement here is one call or a series of them against
        // caches of its own, so a number of positions to keep could only be
        // dropped — the same rule the two above are refused under.
        if what != What::Session && reuse.is_some() {
            bail!("{what:?} takes no --reuse-tokens: it makes one request");
        }
        Ok(Self::Measure {
            what,
            checkpoint,
            tokens: tokens.unwrap_or(match what {
                What::Prefill => PREFILLED,
                What::Session => Session::OPENING,
                _ => DECODED,
            }),
            context: context.unwrap_or(0),
            depth,
            numerics: numerics.unwrap_or_default(),
            reuse: reuse.unwrap_or(DEFAULT_BOUND),
        })
    }

    /// `diverge`, which takes one checkpoint and runs both numerics itself —
    /// so it is the one measurement here that names neither of them.
    fn diverging(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let mut checkpoints = Vec::new();
        let mut tokens = DIFFERENTIAL;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokens" | "-n" => tokens = count(&arg, &mut args)?,
                _ if arg.starts_with('-') => bail!("unexpected argument {arg}"),
                _ => checkpoints.push(PathBuf::from(arg)),
            }
        }
        let [checkpoint] =
            <[PathBuf; 1]>::try_from(checkpoints).map_err(|given: Vec<PathBuf>| {
                anyhow::anyhow!(
                    "diverge takes one checkpoint directory, given {}",
                    given.len()
                )
            })?;
        Ok(Self::Diverge { checkpoint, tokens })
    }

    fn alternating(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let mut pairs = PAIRS;
        let mut arms = Vec::new();
        let mut shared = Vec::new();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--pairs" | "-p" => pairs = count(&arg, &mut args)?,
                // Everything past the separator belongs to the arms, flags and
                // all, which is what lets them take flags this does not have.
                "--" => {
                    shared.extend(args.by_ref());
                    break;
                }
                _ if arg.starts_with('-') => bail!("unexpected argument {arg}"),
                // An arm is a command line rather than a path, so that what
                // differs between the two can be an argument as readily as an
                // executable.
                _ => arms.push(arg.split_whitespace().map(str::to_string).collect()),
            }
        }
        let [a, b] = <[Vec<String>; 2]>::try_from(arms).map_err(|arms: Vec<Vec<String>>| {
            anyhow::anyhow!("alternate takes two commands, given {}", arms.len())
        })?;
        Ok(Self::Alternate {
            pairs,
            arms: [a, b],
            args: shared,
        })
    }

    fn run(&self) -> Result<()> {
        match self {
            Self::Measure {
                what,
                checkpoint,
                tokens,
                context,
                depth,
                numerics,
                reuse,
            } => {
                let asked = Asked {
                    tokens: *tokens,
                    context: *context,
                    depth: *depth,
                    numerics: *numerics,
                    reuse: *reuse,
                };
                for reading in measure(*what, checkpoint, asked)? {
                    println!("{}", reading.line());
                }
                Ok(())
            }
            Self::Alternate { pairs, arms, args } => alternate(*pairs, arms, args),
            Self::Guesses {
                checkpoints,
                tokens,
                depth,
            } => {
                for reading in guesses(checkpoints, *tokens, *depth)? {
                    println!("{}", reading.line());
                }
                Ok(())
            }
            Self::Diverge { checkpoint, tokens } => {
                for reading in diverge(checkpoint, *tokens)? {
                    println!("{}", reading.line());
                }
                Ok(())
            }
        }
    }
}

/// Which numerics a word names.
fn which(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<Numerics> {
    let name = args
        .next()
        .with_context(|| format!("{flag} takes a value"))?;
    Numerics::parse(&name)
        .with_context(|| format!("{name} is not numerics, which is reference or production"))
}

/// A count of at least one, which every number this takes is: no measurement is
/// defined over zero tokens, zero pairs or a sweep of no depths.
/// A count of positions to keep, which may be zero: keeping nothing is the arm
/// a session is measured against, and it is the one number [`count`] refuses.
fn positions(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<usize> {
    let value = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("{flag} takes a value"))?;
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("{value} is not a count of positions, after {flag}"))
}

fn count(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<usize> {
    let value = args
        .next()
        .with_context(|| format!("{flag} takes a value"))?;
    match value.parse() {
        Ok(count) if count > 0 => Ok(count),
        _ => bail!("{value} is not a count of at least one, after {flag}"),
    }
}

/// How many of a generation's own tokens [`Generated`] keeps, so that a run can
/// be said to have generated what another run generated.
const OPENING: usize = 8;

/// One generation, and what it cost.
struct Generated {
    /// Every step, the prompt's prefill first.
    steps: Vec<Duration>,
    /// What the device executed for, per step of the regime being charged.
    gpu: Duration,
    tokens: usize,
    rounds: usize,
    rates: Vec<f64>,
    /// The first [`OPENING`] tokens it produced.
    ///
    /// **What makes a cross-engine figure a comparison of engines rather than
    /// of workloads.** Two engines given the same prompt and asked for the same
    /// number of tokens are comparable whatever they answer, but two that answer
    /// the same thing are comparable without an argument — so the answer is
    /// carried beside the duration rather than assumed to be the same one.
    opening: Vec<usize>,
}

impl Generated {
    /// What one step of the regime charged took: the prefill, or the mean of the
    /// decode steps after it.
    ///
    /// The two are not the same price, and a mean over both describes neither —
    /// which is why the clock a decode figure comes from starts at the first
    /// token rather than at the call.
    fn step(&self) -> Duration {
        match self.steps.split_first() {
            Some((prefill, [])) => *prefill,
            Some((_, decode)) => decode.iter().sum::<Duration>() / decode.len() as u32,
            None => Duration::ZERO,
        }
    }

    /// The first token, which is the prompt's prefill and whatever the engine
    /// does behind it to name a token.
    fn first(&self) -> Duration {
        self.steps.first().copied().unwrap_or_default()
    }

    /// Prompt and answer together — **the wall a user actually waits**, and the
    /// only figure here in which prefill and decode are weighed against each
    /// other rather than quoted apart.
    fn wall(&self) -> Duration {
        self.steps.iter().sum()
    }

    /// Tokens a round banked, which is what acceptance buys before the cost of
    /// having guessed comes off it.
    fn per_round(&self) -> f64 {
        self.tokens as f64 / self.rounds.max(1) as f64
    }

    /// The median of the steps after the prefill.
    ///
    /// **A mean that is not the median is a reading of something else**, and
    /// [`Generated::step`] is a mean. A decode step at a fixed context is flat
    /// here to a few tenths, so a pair whose mean sits well above its own median
    /// has a step in it that is not a decode step — which a table printing only
    /// the mean could not tell from an engine that is uniformly slower.
    fn median(&self) -> Duration {
        let mut after: Vec<Duration> = self.steps.iter().skip(1).copied().collect();
        after.sort_unstable();
        after.get(after.len() / 2).copied().unwrap_or_default()
    }

    /// The longest step after the prefill, and which one it was.
    ///
    /// Which one matters as much as how long: a generation's own first step is
    /// where anything the prefill deferred lands, and one anywhere else is this
    /// machine rather than this engine.
    ///
    /// Step zero is the prefill and is never the answer, so a generation with no
    /// step after it answers `(0, 0)` — a position this cannot otherwise return,
    /// which is what makes the sentinel readable rather than a duration of zero
    /// standing in for one nobody measured.
    fn worst(&self) -> (usize, Duration) {
        self.steps
            .iter()
            .enumerate()
            .skip(1)
            .max_by_key(|(_, step)| **step)
            .map(|(at, step)| (at, *step))
            .unwrap_or_default()
    }
}

fn millis(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1e3
}

/// What this run of the engine is asked to time, taken once against a checkpoint
/// this process opens exactly one device for.
/// The prompt the steps being timed run behind, tiled to the context asked for.
///
/// **What a decode step costs is a function of the keys behind it**, and that is
/// the one thing about it a caller can choose: the same build is 19.97 ms a
/// token at 97 keys and 29.02 at 8192. A context of zero is the prompt as it
/// stands, which is the 34 keys every decode figure in this repo was taken over
/// before there was a flag for it.
fn behind_a_step(prompt: &[usize], context: usize) -> Vec<usize> {
    match context {
        0 => prompt.to_vec(),
        context => tiled(prompt, context),
    }
}

/// The numbers a measurement was asked for, which differ per measurement and are
/// carried together rather than as six positional arguments one of which every
/// caller passes zero for.
#[derive(Debug, Clone, Copy)]
struct Asked {
    tokens: usize,
    context: usize,
    depth: usize,
    numerics: Numerics,
    reuse: usize,
}

fn measure(what: What, dir: &Path, asked: Asked) -> Result<Vec<Reading>> {
    let Asked {
        tokens,
        context,
        depth,
        numerics,
        reuse,
    } = asked;
    let config = config::of_checkpoint(dir)?;
    let text = &config.text_config;
    let tokenizer = Tokenizer::open(dir, &config)?;
    let prompt: Vec<usize> = tokenizer
        .encode(STRUCTURED_PROMPT)?
        .into_iter()
        .map(|id| id as usize)
        .collect();

    let gpu = backend::open(Backend::Metal, numerics)?;
    let ckpt = Checkpoint::open(dir)?;

    // Which depths this wants weights wrapped at: a sweep every one up to its
    // own, a cross-engine table `k = 0` and the depth that pays best beside it,
    // and everything else nothing but zero.
    let depths: Vec<usize> = match what {
        What::Sweep => (0..=depth).collect(),
        What::Engines if depth > 0 => vec![0, depth],
        _ => vec![0],
    };
    // Thrown away, because the first generation of a process faults in the
    // pages the rest of them read — 4.2 GiB of it once heads are mapped — and
    // that belongs to a run's first token rather than to whichever arm ran
    // first.
    let warm = prompt[..prompt.len().min(8)].to_vec();

    let mut taken = Vec::new();
    let mut unspeculated = None;
    for depth in depths {
        // Wrapped at the depth being measured rather than at the deepest one:
        // the windows a rejected token is taken back out of are wider by the
        // depth, so this is the configuration a run of that depth actually has.
        let weights = backend::weights(gpu.as_ref(), &ckpt, text, depth)?;
        let tail = backend::tail_weights(&weights, text);
        let heads = backend::heads(gpu.as_ref(), &ckpt, &config, depth, &tail)?;
        let timed =
            |ids: &[usize], budget| generate(&weights, heads.as_ref(), text, ids, budget, depth);

        timed(&warm, 2);

        match what {
            // **Three readings off one generation and not three runs of it.**
            // The first step is the prefill, the mean of the ones after it is a
            // decode step, and the sum is the wall a user waits — so a table
            // that quotes all three quotes them about the same generation
            // rather than about three sittings of one.
            //
            // **And the whole pass is run twice, with the first thrown away**,
            // because what a pair costs depends on what ran before it: the same
            // 97-token prefill is 1.17 s where a 769-token generation preceded
            // it and 0.59 s where a shorter one did. Warming at one length
            // cannot settle a table of four lengths; two passes give every pair
            // the same predecessor in the pass that counts.
            What::Engines => {
                for measured in [false, true] {
                    for (prompted, generated) in REALISTIC {
                        let felt = timed(&tiled(&prompt, prompted), generated);
                        if !measured {
                            continue;
                        }
                        let pair = format!("{prompted}x{generated}.k{depth}");
                        // The spread only where a step is a step. A round of
                        // depth `k` banks what it accepted all at once, so most
                        // of a speculating run's steps are nothing and the
                        // round's whole cost lands on one of them — a median of
                        // 42 nanoseconds, which is a reading about the sink and
                        // not about the engine. The mean over them is still
                        // milliseconds a token, which is what the table quotes.
                        let (worst, longest) = felt.worst();
                        eprintln!(
                            "{pair}: {} tokens, {}first eight {:?}",
                            felt.tokens,
                            match depth {
                                0 => format!(
                                    "steps p50 {:.2?}, longest {:.2?} at step {worst}, ",
                                    felt.median(),
                                    longest
                                ),
                                _ => String::new(),
                            },
                            felt.opening,
                        );
                        for (name, taken_by) in [
                            ("wall", felt.wall()),
                            ("first", felt.first()),
                            ("token", felt.step()),
                        ] {
                            taken.push(Reading::new(
                                format!("{pair}.{name}"),
                                millis(taken_by),
                                "ms",
                            ));
                        }
                    }
                }
            }
            What::Decode => {
                let run = timed(&behind_a_step(&prompt, context), tokens);
                taken.push(Reading::new("decode", millis(run.step()), "ms"));
                taken.push(Reading::new("device", millis(run.gpu), "ms"));
            }
            What::Prefill => {
                // A prefill is one step and its prompt is the measurement, so
                // the prompt is tiled to the length asked for.
                let ids = tiled(&prompt, tokens);
                let run = timed(&ids, 1);
                // The second of two, for the reason the timing tier prints both:
                // the first prefill of a length is the one that faults its pages
                // in, and what a prefill costs warm is the figure every other
                // one here is comparable to.
                let again = timed(&ids, 1);
                taken.push(Reading::new("prefill", millis(again.step()), "ms"));
                taken.push(Reading::new("device", millis(again.gpu), "ms"));
                taken.push(Reading::new("cold", millis(run.step()), "ms"));
            }
            // **The one arm here that measures more than one request.** What a
            // kept cache is worth is the difference between a turn that
            // re-prefills its whole conversation and one that prefills what was
            // added to it, and neither number exists in a measurement of a
            // single call.
            //
            // Every turn is a row, because the shape is the finding: turn one is
            // cold in both arms and every turn after it is where they part.
            What::Session => {
                let plan = Session::new(tokens);
                let mut kept = Kept::new(text, reuse);
                let turns = session::run(&weights.generator(), &weights, &mut kept, plan, &prompt);

                eprintln!(
                    "session: {} turns keeping {reuse}, first eight {:?}",
                    turns.len(),
                    &session::tokens(&turns)[..OPENING.min(plan.generated * plan.turns)]
                );
                for (at, turn) in turns.iter().enumerate() {
                    eprintln!(
                        "  turn {at}: {} tokens, {} reused, {:.2} s wall, {:.2} s to first",
                        turn.prompt,
                        turn.reused,
                        turn.wall.as_secs_f64(),
                        turn.first.as_secs_f64(),
                    );
                    taken.push(Reading::new(
                        format!("turn{at}.wall"),
                        millis(turn.wall),
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("turn{at}.first"),
                        millis(turn.first),
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("turn{at}.prefilled"),
                        (turn.prompt - turn.reused) as f64,
                        "tokens",
                    ));
                }
                // The figure a user feels, which is the one nobody here has ever
                // produced: the whole conversation, end to end.
                taken.push(Reading::new(
                    "session",
                    millis(turns.iter().map(|turn| turn.wall).sum()),
                    "ms",
                ));
            }
            What::Sweep => {
                let run = timed(&prompt, tokens);
                let step = run.step();
                let unspeculated = *unspeculated.get_or_insert(step);
                taken.push(Reading::new(format!("k{depth}"), millis(step), "ms"));
                taken.push(Reading::new(
                    format!("k{depth}.device"),
                    millis(run.gpu),
                    "ms",
                ));
                // Against this run's own `k = 0` and not against another
                // sitting's: a sweep whose speedup row is divided by a figure
                // taken an hour earlier carries the drift between the two.
                taken.push(Reading::new(
                    format!("k{depth}.speedup"),
                    unspeculated.as_secs_f64() / step.as_secs_f64(),
                    "x",
                ));
                taken.push(Reading::new(
                    format!("k{depth}.tokens"),
                    run.per_round(),
                    "/round",
                ));
                for (at, rate) in run.rates.iter().enumerate() {
                    taken.push(Reading::new(
                        format!("k{depth}.accept{}", at + 1),
                        100.0 * rate,
                        "%",
                    ));
                }
            }
        }
    }
    Ok(taken)
}

/// Two chains of heads asked the same question at every round, and how often
/// they answer it differently.
///
/// **This is the gate a change to the heads has to pass before any timing claim,
/// and it is not the tokens.** No token can move: the model verifies every
/// guess, so a worse head produces a rejected guess rather than a wrong output.
/// What is at risk is acceptance, and acceptance is the whole of the speedup —
/// so what has to be held against the original is the guesses themselves.
///
/// **One generation and one stack.** The first checkpoint's heads are the ones
/// the generation runs on, so the rounds are its rounds; the second is asked the
/// same [`Round`] at every one of them and its answer is compared and thrown
/// away. Both are chained from the same hidden states through the same stack and
/// the same embeddings, so the heads are the only thing that differs — and both
/// chains see every round, so neither one's own carried state falls behind.
struct Held<P: Proposer> {
    /// The chain whose guesses the generation runs on.
    ran: P,
    /// The chain asked the same question and answered nowhere.
    against: P,
    /// Per depth: how many guesses the two were both asked for, and how many of
    /// those they answered differently.
    asked: Vec<usize>,
    diverged: Vec<usize>,
    guessed: Vec<usize>,
}

impl<P: Proposer> Held<P> {
    fn new(ran: P, against: P, depth: usize) -> Self {
        Self {
            ran,
            against,
            asked: vec![0; depth],
            diverged: vec![0; depth],
            guessed: Vec::new(),
        }
    }
}

impl<P: Proposer> Proposer for Held<P> {
    fn depth(&self) -> usize {
        self.ran.depth()
    }

    fn propose(&mut self, round: Round<'_>) -> &[usize] {
        let against: Vec<usize> = self.against.propose(round).to_vec();
        self.guessed = self.ran.propose(round).to_vec();
        for (at, (ran, against)) in self.guessed.iter().zip(&against).enumerate() {
            self.asked[at] += 1;
            self.diverged[at] += usize::from(ran != against);
        }
        &self.guessed
    }
}

/// One generation of `budget` tokens, timed a step at a time.
fn generate(
    weights: &CheckpointWeights<'_>,
    heads: Option<&CheckpointHeads<'_>>,
    config: &TextConfig,
    ids: &[usize],
    budget: usize,
    depth: usize,
) -> Generated {
    let generator = weights.generator();
    let cache = &mut ModelCache::speculating(config, depth);
    let ending = Ending { budget, eos: None };
    let mut proposer = heads
        .filter(|_| depth > 0)
        .map(|heads| MtpProposer::new(heads, generator, weights, depth));

    let mut steps = Vec::new();
    let mut opening = Vec::new();
    let mut tokens = 0;
    // Cleared here so that what comes back is this generation's: wrapping the
    // model charges the same accounts, and on a prefill there is no later step
    // for it to be small beside.
    profile::take();
    let mut step = Instant::now();
    {
        let mut sink = |token: usize| {
            steps.push(step.elapsed());
            if opening.len() < OPENING {
                opening.push(token);
            }
            // The prefill's accounts, dropped where the prefill's duration is:
            // a decode figure is about the steps after it.
            if steps.len() == 1 && budget > 1 {
                profile::take();
            }
            tokens += 1;
            step = Instant::now();
            ControlFlow::Continue(())
        };
        match proposer.as_mut() {
            Some(proposer) => generator.speculate(cache, ids, ending, weights, proposer, &mut sink),
            None => generator.stream(cache, ids, ending, weights, &mut sink),
        };
    }

    let charged = u32::try_from(steps.len().saturating_sub(1).max(1)).unwrap_or(1);
    Generated {
        gpu: profile::take().per_step(charged).gpu(),
        rounds: proposer.as_ref().map_or(tokens, MtpProposer::rounds),
        rates: proposer
            .map(|proposer| proposer.rates())
            .unwrap_or_default(),
        steps,
        tokens,
        opening,
    }
}

/// The two checkpoints' heads held against each other over one generation.
///
/// The stack, the embeddings and the tokenizer are the *first* checkpoint's for
/// both chains. That is not a shortcut: a checkpoint whose heads were quantised
/// afterwards is the same stack with a different shard beside it, and reading
/// the stack twice would be reading the same bytes twice under a second name.
fn guesses(dirs: &[PathBuf; 2], tokens: usize, depth: usize) -> Result<Vec<Reading>> {
    let config = config::of_checkpoint(&dirs[0])?;
    let text = &config.text_config;
    let tokenizer = Tokenizer::open(&dirs[0], &config)?;
    let ids: Vec<usize> = tokenizer
        .encode(STRUCTURED_PROMPT)?
        .into_iter()
        .map(|id| id as usize)
        .collect();

    let gpu = backend::open(Backend::Metal, Numerics::default())?;
    let stack = Checkpoint::open(&dirs[0])?;
    let beside = Checkpoint::open(&dirs[1])?;
    let weights = backend::weights(gpu.as_ref(), &stack, text, depth)?;
    let tail = backend::tail_weights(&weights, text);
    let ran = backend::heads(gpu.as_ref(), &stack, &config, depth, &tail)?
        .context("the first checkpoint has no heads to guess with")?;
    let against = backend::heads(gpu.as_ref(), &beside, &config, depth, &tail)?
        .context("the second checkpoint has no heads to guess with")?;

    let generator = weights.generator();
    let mut held = Held::new(
        MtpProposer::new(&ran, generator, &weights, depth),
        MtpProposer::new(&against, generator, &weights, depth),
        depth,
    );
    let cache = &mut ModelCache::speculating(text, depth);
    let mut banked = 0;
    generator.speculate(
        cache,
        &ids,
        Ending {
            budget: tokens,
            eos: None,
        },
        &weights,
        &mut held,
        |_| {
            banked += 1;
            ControlFlow::Continue(())
        },
    );

    let mut taken = vec![Reading::new("tokens", banked as f64, "banked")];
    for (at, (asked, diverged)) in held.asked.iter().zip(&held.diverged).enumerate() {
        taken.push(Reading::new(
            format!("asked{}", at + 1),
            *asked as f64,
            "guesses",
        ));
        taken.push(Reading::new(
            format!("diverged{}", at + 1),
            *diverged as f64,
            "guesses",
        ));
    }
    let (asked, diverged): (usize, usize) = (held.asked.iter().sum(), held.diverged.iter().sum());
    taken.push(Reading::new("asked", asked as f64, "guesses"));
    taken.push(Reading::new("diverged", diverged as f64, "guesses"));
    taken.push(Reading::new(
        "diverged.share",
        100.0 * diverged as f64 / asked.max(1) as f64,
        "%",
    ));
    Ok(taken)
}

/// The corpus through both numerics, and where the two continuations part
/// company.
///
/// **This is the instrument the flag exists for.** Under the reference the
/// engine's answer is checkable against a recorded array of bits, and every
/// gated case in the tree checks it. Under the production numerics there is no
/// such array and there cannot be one — a matrix instruction's summation order
/// is not this side's to record — so what stands in for the oracle is a second
/// implementation: two GPU paths that share every tiling decision, every
/// predicate and every dispatch, and differ only in how the innermost sum is
/// carried. Where those two agree, the structure around the sum is agreed by two
/// independent accumulations; where they disagree, the disagreement is between
/// the arithmetic and nothing else, and the position it first appears at is
/// where to look.
///
/// **What is reported is leading agreement and not a count of differing
/// tokens.** Two free-running generations that part company at step 12 have
/// nothing comparable after step 12 — the shorter path is now continuing a
/// different sentence — so "how many of the 64 differ" is a number about the two
/// continuations rather than about the arithmetic. How far they got before the
/// first disagreement is the number that means something, and it is per prompt
/// because a prompt is what decides how close two logits get.
///
/// **One device at a time.** The two paths run in sequence rather than side by
/// side: each wraps the whole model, and holding two of those at once would
/// double a resident set this repo bounds a test on for nothing — nothing here
/// is timed.
fn diverge(dir: &Path, tokens: usize) -> Result<Vec<Reading>> {
    let config = config::of_checkpoint(dir)?;
    let text = &config.text_config;
    let tokenizer = Tokenizer::open(dir, &config)?;
    let ckpt = Checkpoint::open(dir)?;

    let prompts = CORPUS
        .iter()
        .map(|prompt| {
            Ok(tokenizer
                .encode(prompt)?
                .into_iter()
                .map(|id| id as usize)
                .collect::<Vec<usize>>())
        })
        .collect::<Result<Vec<_>>>()?;

    // **A prompt shorter than this reaches no entry the flag selects**, so its
    // two continuations are one kernel compared to itself — perfect agreement
    // that would stay perfect however the arithmetic behind the flag changed.
    // Checked here rather than beside the corpus because this is where the
    // tokenizer is: a prompt's length in tokens is the only length that decides
    // it, and a byte count beside the text is a proxy that has already been
    // wrong once.
    //
    // **Both floors and not the larger of them by luck.** The matmul's entries
    // are reached by a call of rows and the attention block by a call of query
    // rows, and a prefill is one length that has to clear both — so the corpus
    // is held to whichever is higher rather than to the one that happened to be
    // written down when the corpus was assembled.
    let reaches = PackedMatmul::SHORTEST_BLOCKED_CALL.max(FusedAttention::SHORTEST_BLOCKED_CALL);
    for (at, ids) in prompts.iter().enumerate() {
        if ids.len() < reaches {
            bail!(
                "prompt {} is {} tokens, under the {reaches} a call needs to reach the entries \
                 this measurement exists to compare",
                at + 1,
                ids.len(),
            );
        }
    }

    let mut answers = Vec::new();
    for numerics in [Numerics::Reference, Numerics::Production] {
        let gpu = backend::open(Backend::Metal, numerics)?;
        let weights = backend::weights(gpu.as_ref(), &ckpt, text, 0)?;
        let generator = weights.generator();
        let mut ran = Vec::new();
        for (at, ids) in prompts.iter().enumerate() {
            let cache = &mut ModelCache::speculating(text, 0);
            let mut continued = Vec::new();
            // No end-of-sequence token, so both paths spend the whole budget:
            // one that stopped early would be shorter than the other for a
            // reason that is not a disagreement about a token.
            let ending = Ending {
                budget: tokens,
                eos: None,
            };
            generator.stream(cache, ids, ending, &weights, &mut |token| {
                continued.push(token);
                ControlFlow::Continue(())
            });
            eprintln!(
                "{}: prompt {} of {}, {} tokens from {} prompted",
                numerics.named(),
                at + 1,
                prompts.len(),
                continued.len(),
                ids.len()
            );
            ran.push(continued);
        }
        answers.push(ran);
    }
    let [reference, production] = <[Vec<Vec<usize>>; 2]>::try_from(answers)
        .map_err(|_| anyhow::anyhow!("a differential run answers twice"))?;

    for (at, (was, is)) in reference.iter().zip(&production).enumerate() {
        let agreed = agreement(was, is);
        match agreed < was.len() {
            true => eprintln!(
                "prompt {} parted at token {}: reference {:?}, production {:?}",
                at + 1,
                agreed + 1,
                &was[agreed..was.len().min(agreed + 4)],
                &is[agreed..is.len().min(agreed + 4)],
            ),
            false => eprintln!("prompt {} agreed for all {} tokens", at + 1, was.len()),
        }
    }
    Ok(parted(&reference, &production))
}

/// How many tokens two continuations agreed on before the first that they did
/// not.
///
/// **Leading agreement rather than a count of differing positions**, and the
/// reason is what a free-running generation is: two paths that part company at
/// step 12 have nothing comparable after step 12, because each is now continuing
/// a different sentence. "How many of the 64 differ" is a number about the two
/// continuations; how far they got before the first disagreement is a number
/// about the arithmetic.
fn agreement(was: &[usize], is: &[usize]) -> usize {
    was.iter().zip(is).take_while(|(a, b)| a == b).count()
}

/// The corpus's readings, out of what the two paths answered.
///
/// Split from the run above so that the arithmetic is checkable without a
/// device: what a differential sitting reports is the whole of what it is for,
/// and a share computed the wrong way would be wrong in the direction nobody
/// looks at.
fn parted(reference: &[Vec<usize>], production: &[Vec<usize>]) -> Vec<Reading> {
    let mut taken = Vec::new();
    let (mut apart, mut generated, mut agreed_over) = (0, 0, 0);
    for (at, (was, is)) in reference.iter().zip(production).enumerate() {
        let agreed = agreement(was, is);
        let named = format!("prompt{}", at + 1);
        taken.push(Reading::new(
            format!("{named}.tokens"),
            was.len() as f64,
            "t",
        ));
        taken.push(Reading::new(format!("{named}.agreed"), agreed as f64, "t"));
        generated += was.len();
        agreed_over += agreed;
        apart += usize::from(agreed < was.len());
    }
    taken.push(Reading::new("prompts", reference.len() as f64, "n"));
    taken.push(Reading::new("parted", apart as f64, "n"));
    taken.push(Reading::new("tokens", generated as f64, "t"));
    taken.push(Reading::new("agreed", agreed_over as f64, "t"));
    taken.push(Reading::new(
        "agreed.share",
        100.0 * agreed_over as f64 / generated.max(1) as f64,
        "%",
    ));
    taken
}

/// Two executables, run against each other for `pairs` pairs with the order
/// flipped each pair.
fn alternate(pairs: usize, arms: &[Vec<String>; 2], args: &[String]) -> Result<()> {
    let mut taken: [Vec<Vec<Reading>>; 2] = [Vec::new(), Vec::new()];
    for pair in 0..pairs {
        // Flipped, so that neither arm always runs on the other's warm page
        // cache and a drift over the sitting lands on both of them.
        let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            let readings = ask(&arms[arm], args)
                .with_context(|| format!("running {} for pair {}", named(&arms[arm]), pair + 1))?;
            eprintln!(
                "pair {} of {pairs}, arm {}: {}",
                pair + 1,
                ["a", "b"][arm],
                readings
                    .iter()
                    .map(|reading| reading.line())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            taken[arm].push(readings);
        }
    }
    print!("{}", report(arms, &compare(&taken)?));
    Ok(())
}

/// One run of one arm.
///
/// Its stdout is the readings and is read here; its stderr is inherited, which
/// is the other half of the protocol: what an arm says to a human — which
/// backend it wrapped, what it is part way through — reaches the terminal while
/// it is still running, and a sweep that takes a minute an arm is not silent for
/// it.
fn ask(arm: &[String], args: &[String]) -> Result<Vec<Reading>> {
    let [program, own @ ..] = arm else {
        bail!("an arm with no executable in it")
    };
    // An arm is one shell word split on whitespace, so a program under a
    // directory whose name has a space in it arrives here cut in half — and
    // would otherwise run something else, or the right thing with a stray
    // argument. Anything that names a path has to be one.
    if program.contains('/') && !Path::new(program).is_file() {
        bail!("{program} is not a file, in the arm {:?}", named(arm));
    }
    let ran = Command::new(program)
        // The shared arguments first and the arm's own after them: the shared
        // ones open with the measurement's name and an arm's own are what that
        // measurement is taken against, which is where a positional goes.
        .args(args)
        .args(own)
        .stderr(Stdio::inherit())
        .output()
        .with_context(|| format!("{program} did not start"))?;
    if !ran.status.success() {
        bail!("{program} exited {}, saying what is above", ran.status);
    }
    readings(&String::from_utf8_lossy(&ran.stdout))
}

/// An arm as it is written back to whoever asked for it.
fn named(arm: &[String]) -> String {
    arm.join(" ")
}

/// What the two arms said about one metric.
#[derive(Debug, PartialEq)]
struct Comparison {
    name: String,
    unit: String,
    /// The mean, then the smallest and largest reading, for each arm in turn.
    arms: [(f64, f64, f64); 2],
    /// Whether the two ranges lie across each other, which is what says the
    /// difference between the means is not this machine's own state.
    overlap: bool,
    /// How many pairs moved the way the means did.
    agreed: usize,
    pairs: usize,
}

impl Comparison {
    /// What b is against a, as a percentage. Divided by a mean rather than by
    /// anything hard-coded, and every metric an arm reports here is a duration,
    /// a rate or a share of one — none of which a run that produced a reading at
    /// all can report as zero.
    fn change(&self) -> f64 {
        100.0 * (self.arms[1].0 / self.arms[0].0 - 1.0)
    }

    /// The standard this file's own README states for a real effect: every pair
    /// moving the same way, and the two ranges not overlapping.
    ///
    /// **Neither test means anything over one pair** — a range of one reading is
    /// a point, and one pair cannot disagree with itself — so a single pair is
    /// never a claim however far apart the two readings fall. Two can, and two is
    /// still not many: a null run of one build against itself over two pairs
    /// reports the ranges apart and every pair agreeing at 0.6%, and over seven
    /// it reports neither. Seven is the default for that reason.
    fn stands(&self) -> bool {
        self.pairs > 1 && !self.overlap && self.agreed == self.pairs
    }
}

fn compare(taken: &[Vec<Vec<Reading>>; 2]) -> Result<Vec<Comparison>> {
    let names: Vec<(String, String)> = taken[0]
        .first()
        .context("no pairs were run")?
        .iter()
        .map(|reading| (reading.name.clone(), reading.unit.clone()))
        .collect();
    for arm in taken {
        for run in arm {
            let ran: Vec<(String, String)> = run
                .iter()
                .map(|reading| (reading.name.clone(), reading.unit.clone()))
                .collect();
            if ran != names {
                bail!(
                    "the arms do not report the same readings: {:?} against {:?}",
                    names.iter().map(|(name, _)| name).collect::<Vec<_>>(),
                    ran.iter().map(|(name, _)| name).collect::<Vec<_>>()
                );
            }
        }
    }

    Ok(names
        .iter()
        .enumerate()
        .map(|(at, (name, unit))| {
            let series =
                |arm: usize| -> Vec<f64> { taken[arm].iter().map(|run| run[at].value).collect() };
            let (a, b) = (series(0), series(1));
            let stats = |values: &[f64]| {
                (
                    values.iter().sum::<f64>() / values.len() as f64,
                    values.iter().copied().fold(f64::INFINITY, f64::min),
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            };
            let (first, second) = (stats(&a), stats(&b));
            let rose = second.0 > first.0;
            Comparison {
                name: name.clone(),
                unit: unit.clone(),
                arms: [first, second],
                overlap: first.1 <= second.2 && second.1 <= first.2,
                agreed: a
                    .iter()
                    .zip(&b)
                    .filter(|(one, two)| (**two > **one) == rose && **two != **one)
                    .count(),
                pairs: a.len(),
            }
        })
        .collect())
}

fn report(arms: &[Vec<String>; 2], compared: &[Comparison]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\na  {}\nb  {}\n", named(&arms[0]), named(&arms[1]));
    let _ = writeln!(
        out,
        "  {:<16}{:>6}{:>11}{:>11}{:>9}  {:<17}{:<17}{:<9}{:<12}claim",
        "", "unit", "a", "b", "change", "a range", "b range", "ranges", "pairs"
    );
    for row in compared {
        let range = |arm: usize| format!("{:.3}-{:.3}", row.arms[arm].1, row.arms[arm].2);
        let pairs = format!("{} of {}", row.agreed, row.pairs);
        let _ = writeln!(
            out,
            "  {:<16}{:>6}{:>11.3}{:>11.3}{:>8.1}%  {:<17}{:<17}{:<9}{:<12}{}",
            row.name,
            row.unit,
            row.arms[0].0,
            row.arms[1].0,
            row.change(),
            range(0),
            range(1),
            if row.overlap { "across" } else { "apart" },
            pairs,
            if row.stands() {
                "every pair the same way, ranges apart"
            } else {
                "no claim"
            }
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    #[test]
    fn a_reading_is_a_name_a_number_and_a_unit() {
        assert_eq!(
            readings("decode 20.86 ms\ndevice 18.17 ms\n").expect("parses"),
            [
                Reading::new("decode", 20.86, "ms"),
                Reading::new("device", 18.17, "ms")
            ]
        );
    }

    /// Blank lines are a formatting choice an arm is allowed; anything else on
    /// stdout is an arm that is not speaking this protocol, and taking a mean
    /// over what could be parsed out of it would report a measurement nobody
    /// made.
    #[test]
    fn a_line_that_is_not_a_reading_is_refused_rather_than_skipped() {
        assert_eq!(readings("\ndecode 20.86 ms\n\n").expect("parses").len(), 1);
        for line in ["decode 20.86", "decode fast ms", "decode 20.86 ms please"] {
            assert!(readings(line).is_err(), "{line:?} parsed");
        }
    }

    fn generated(steps: &[u64], tokens: usize, rounds: usize) -> Generated {
        Generated {
            steps: steps.iter().map(|ms| Duration::from_millis(*ms)).collect(),
            gpu: Duration::ZERO,
            tokens,
            rounds,
            rates: Vec::new(),
            opening: Vec::new(),
        }
    }

    /// **The wall a user waits is the whole call and the prefill is inside it**,
    /// where the decode figure beside it deliberately leaves that first step out
    /// — so the three readings a cross-engine table takes off one generation are
    /// three different cuts of the same list rather than three runs of it.
    #[test]
    fn a_generations_three_readings_are_three_cuts_of_the_same_steps() {
        let run = generated(&[1000, 20, 22, 24], 4, 4);
        assert_eq!(run.first(), Duration::from_millis(1000));
        assert_eq!(run.step(), Duration::from_millis(22));
        assert_eq!(run.wall(), Duration::from_millis(1066));
        // Neither of the two the mean is read against carries the prefill: one
        // that did would report the first step as the longest at every length
        // and say nothing at any of them.
        assert_eq!(run.median(), Duration::from_millis(22));
        assert_eq!(run.worst(), (3, Duration::from_millis(24)));
    }

    /// A generation whose longest step is its *first* is the case this column
    /// exists for — what a prefill deferred, arriving on the step after it —
    /// and it has to be told from the same duration landing anywhere else.
    #[test]
    fn the_longest_step_is_reported_with_where_it_fell() {
        assert_eq!(
            generated(&[1000, 800, 22, 24], 4, 4).worst(),
            (1, Duration::from_millis(800))
        );
    }

    /// A prefill has no step after it, and both readings taken over those steps
    /// have to say so rather than report a zero that reads like a measurement.
    /// Step zero is the prefill, so it is a position `worst` cannot otherwise
    /// answer with.
    #[test]
    fn a_run_with_no_step_after_its_prefill_reports_no_spread() {
        for run in [generated(&[1000], 1, 1), generated(&[], 0, 0)] {
            assert_eq!(run.median(), Duration::ZERO);
            assert_eq!(run.worst(), (0, Duration::ZERO));
        }
    }

    /// The prefill is a step of another price and the decode figure is the mean
    /// of the ones after it. A mean over both would describe neither, and it is
    /// the prefill — twenty times a decode step — that would carry it.
    #[test]
    fn a_decode_figure_leaves_the_prefill_in_front_of_it_out() {
        assert_eq!(
            generated(&[1000, 20, 22, 24], 4, 4).step(),
            Duration::from_millis(22)
        );
    }

    /// One step is a prefill and has no decode step to be the mean of, which is
    /// what `bench prefill` runs.
    #[test]
    fn a_run_of_one_step_is_that_step() {
        assert_eq!(generated(&[1000], 1, 1).step(), Duration::from_millis(1000));
        assert_eq!(generated(&[], 0, 0).step(), Duration::ZERO);
    }

    /// Tokens a round is what acceptance banked before the cost of having
    /// guessed comes off it, and a run that speculated nothing banked one.
    #[test]
    fn tokens_a_round_is_what_was_banked_over_the_rounds_that_banked_it() {
        assert!((generated(&[1, 1], 64, 35).per_round() - 1.829).abs() < 1e-3);
        assert!((generated(&[1, 1], 64, 64).per_round() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_measurement_takes_a_checkpoint_and_its_own_defaults() {
        assert_eq!(
            Job::parse(["sweep".to_string(), "models/small".to_string()]).expect("parses"),
            Job::Measure {
                what: What::Sweep,
                checkpoint: PathBuf::from("models/small"),
                tokens: DECODED,
                context: 0,
                depth: SWEPT,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
            }
        );
        assert_eq!(
            Job::parse(["prefill".to_string(), "models/small".to_string()]).expect("parses"),
            Job::Measure {
                what: What::Prefill,
                checkpoint: PathBuf::from("models/small"),
                tokens: PREFILLED,
                context: 0,
                depth: SWEPT,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
            }
        );
    }

    /// **A cross-engine table's default depth is not a sweep's**, because the
    /// flag does not mean the same thing to the two: a sweep runs every depth up
    /// to the one it is given and a table quotes the one that pays best beside
    /// `k = 0`.
    #[test]
    fn a_cross_engine_table_defaults_to_the_depth_that_pays_rather_than_the_deepest() {
        assert_eq!(
            Job::parse(["engines".to_string(), "models/small".to_string()]).expect("parses"),
            Job::Measure {
                what: What::Engines,
                checkpoint: PathBuf::from("models/small"),
                tokens: DECODED,
                context: 0,
                depth: BEST,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
            }
        );
        assert_ne!(BEST, SWEPT);
        // And it takes no length, because it runs four of them.
        assert!(
            Job::parse(["engines", "models/small", "--tokens", "769"].map(str::to_string)).is_err()
        );
    }

    /// **Only a session has a between-requests to keep anything across**, so it
    /// is the only measurement here the number means anything to — and zero is
    /// one of the numbers it means something by, which is why the arm that keeps
    /// nothing is a command line rather than a second binary.
    #[test]
    fn only_a_session_takes_a_count_of_positions_to_keep() {
        assert_eq!(
            Job::parse(["session", "models/small", "--reuse-tokens", "0"].map(str::to_string))
                .expect("parses"),
            Job::Measure {
                what: What::Session,
                checkpoint: PathBuf::from("models/small"),
                tokens: Session::OPENING,
                context: 0,
                depth: SWEPT,
                numerics: Numerics::default(),
                reuse: 0,
            }
        );
        for what in ["decode", "prefill", "sweep", "engines"] {
            assert!(
                Job::parse([what, "models/small", "--reuse-tokens", "0"].map(str::to_string))
                    .is_err(),
                "{what} took a count of positions to keep"
            );
        }
    }

    /// **The numerics are an arm's own word and every measurement takes it**,
    /// which is what makes the two paths pairable out of one build: an arm is a
    /// command line, so `--numerics reference` against `--numerics production`
    /// is the same arrangement `bench-weights` puts two checkpoints through.
    #[test]
    fn every_measurement_takes_the_numerics_its_arm_runs_under() {
        for what in ["decode", "prefill", "sweep", "engines"] {
            let job =
                Job::parse([what, "models/small", "--numerics", "production"].map(str::to_string))
                    .unwrap_or_else(|err| panic!("{what} takes numerics: {err:#}"));
            assert!(
                matches!(
                    job,
                    Job::Measure {
                        numerics: Numerics::Production,
                        ..
                    }
                ),
                "{what} dropped the word"
            );
        }
    }

    /// And a measurement nobody said anything to runs the reference, which is
    /// what makes an older arm — one built before this flag existed — comparable
    /// to a new one asked for nothing.
    #[test]
    fn a_measurement_that_names_no_numerics_runs_the_reference() {
        assert!(matches!(
            Job::parse(["decode".to_string(), "models/small".to_string()]).expect("parses"),
            Job::Measure {
                numerics: Numerics::Reference,
                ..
            }
        ));
    }

    /// **The differential run is the one measurement that names neither**, since
    /// it runs both itself — so a word choosing one of them is a mistake there.
    #[test]
    fn a_differential_run_takes_one_checkpoint_and_no_numerics() {
        assert_eq!(
            Job::parse(["diverge".to_string(), "models/small".to_string()]).expect("parses"),
            Job::Diverge {
                checkpoint: PathBuf::from("models/small"),
                tokens: DIFFERENTIAL,
            }
        );
        for name in ["reference", "production"] {
            assert!(
                Job::parse(["diverge", "models/small", "--numerics", name].map(str::to_string))
                    .is_err(),
                "{name} was taken by a measurement that runs both"
            );
        }
        assert!(Job::parse(["diverge".to_string()]).is_err());
    }

    /// The same refusal on `guesses`, and **about whether the word was given
    /// rather than about which of the two it named** — a check against the
    /// default would let `--numerics reference` through for being equal to it.
    #[test]
    fn the_heads_measurement_takes_no_numerics_under_either_word() {
        for name in ["reference", "production"] {
            assert!(
                Job::parse(
                    ["guesses", "models/a", "models/b", "--numerics", name].map(str::to_string)
                )
                .is_err(),
                "{name} was taken by a measurement whose arms are two sets of heads"
            );
        }
    }

    /// **What a differential sitting reports is the whole of what it is for**,
    /// so the arithmetic behind it is checked without a device rather than
    /// trusted because the run is expensive.
    ///
    /// The cases are the four shapes a pair of continuations has: agreeing
    /// throughout, parting at the very first token, parting at the last one, and
    /// parting in the middle.
    #[test]
    fn leading_agreement_is_counted_up_to_the_first_token_that_differs() {
        assert_eq!(agreement(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(agreement(&[1, 2, 3], &[9, 2, 3]), 0);
        assert_eq!(agreement(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(agreement(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(agreement(&[], &[]), 0);
    }

    /// **A prompt that parted is counted once however many of its tokens
    /// differ**, and the share is over tokens rather than over prompts: the two
    /// answer different questions and a sitting that reported one of them under
    /// the other's name would read as agreement it did not have.
    #[test]
    fn a_differential_sitting_reports_prompts_apart_and_tokens_agreed() {
        let reference = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 9, 9, 9]];
        let production = vec![vec![1, 2, 3, 4], vec![5, 0, 0, 0], vec![0, 0, 0, 0]];
        let taken = parted(&reference, &production);
        let read = |name: &str| {
            taken
                .iter()
                .find(|reading| reading.name == name)
                .unwrap_or_else(|| panic!("{name} was not reported"))
                .value
        };
        assert_eq!(read("prompts"), 3.0);
        assert_eq!(read("parted"), 2.0);
        assert_eq!(read("tokens"), 12.0);
        // Four, one and none, which is five of twelve.
        assert_eq!(read("agreed"), 5.0);
        assert!((read("agreed.share") - 100.0 * 5.0 / 12.0).abs() < 1e-9);
        assert_eq!(read("prompt2.agreed"), 1.0);
        assert_eq!(read("prompt3.tokens"), 4.0);
    }

    /// Nothing generated is not agreement, and the share it would be divided by
    /// is the one number here that could be a zero.
    #[test]
    fn a_differential_sitting_over_nothing_reports_no_agreement_rather_than_dividing_by_zero() {
        let taken = parted(&[], &[]);
        let share = taken
            .iter()
            .find(|reading| reading.name == "agreed.share")
            .expect("a share is reported");
        assert_eq!(share.value, 0.0);
    }

    /// **A context of zero is the prompt and any other is the prompt tiled to
    /// it**, which is what decides the length the steps being timed run behind.
    ///
    /// The rejection case above says the flag reaches the right measurement;
    /// this says it does something when it gets there, without opening a device
    /// to find out.
    #[test]
    fn a_step_is_timed_behind_the_context_it_was_given() {
        let prompt = [11, 22, 33];
        assert_eq!(behind_a_step(&prompt, 0), prompt);
        assert_eq!(behind_a_step(&prompt, 8).len(), 8);
        assert_eq!(behind_a_step(&prompt, 2).len(), 2);
    }

    /// **A decode step is the one measurement here with a context to be taken
    /// at**, and the flag is refused everywhere else rather than dropped.
    ///
    /// What a step costs is a function of the keys behind it — 19.9 ms a token
    /// at 97 and 28.7 at 8192 — so a paired comparison of two builds taken only
    /// at the structured prompt's own 34 keys is a comparison at the one length
    /// a user does not have. A prefill's context is its prompt and `--tokens`
    /// already says how long that is; a sweep and a cross-engine table fix their
    /// own prompts because acceptance is the workload's.
    #[test]
    fn only_a_decode_step_takes_a_context_to_be_taken_at() {
        assert_eq!(
            Job::parse(["decode", "models/small", "--context", "8192"].map(str::to_string))
                .expect("parses"),
            Job::Measure {
                what: What::Decode,
                checkpoint: PathBuf::from("models/small"),
                tokens: DECODED,
                context: 8192,
                depth: SWEPT,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
            }
        );
        for what in ["prefill", "sweep", "engines", "session"] {
            assert!(
                Job::parse([what, "models/small", "--context", "8192"].map(str::to_string))
                    .is_err(),
                "{what} took a context it has no use for"
            );
        }
        assert!(
            Job::parse(
                ["guesses", "models/a", "models/b", "--context", "8192"].map(str::to_string)
            )
            .is_err()
        );
    }

    /// The flags the recipe passes through, and the separator that says which
    /// side of the harness the rest of the line belongs to.
    #[test]
    fn alternate_takes_two_arms_and_hands_everything_past_the_separator_to_both() {
        let parsed = Job::parse(
            [
                "alternate",
                "--pairs",
                "3",
                "a/bench",
                "b/bench",
                "--",
                "prefill",
                "--tokens",
                "769",
            ]
            .map(str::to_string),
        )
        .expect("parses");
        assert_eq!(
            parsed,
            Job::Alternate {
                pairs: 3,
                arms: [vec!["a/bench".to_string()], vec!["b/bench".to_string()]],
                args: ["prefill", "--tokens", "769"].map(str::to_string).to_vec(),
            }
        );
    }

    #[test]
    fn an_invocation_that_names_one_arm_or_three_is_refused() {
        for arms in [vec!["a"], vec!["a", "b", "c"]] {
            let mut args = vec!["alternate".to_string()];
            args.extend(arms.iter().map(|arm| arm.to_string()));
            assert!(Job::parse(args).is_err(), "{arms:?} parsed");
        }
    }

    #[test]
    fn a_count_that_is_not_one_is_refused() {
        for value in ["0", "-1", "many", ""] {
            assert!(
                Job::parse(["decode".to_string(), "-n".to_string(), value.to_string()]).is_err(),
                "{value:?} parsed"
            );
        }
    }

    /// An executable that prints one reading and records that it ran, which is
    /// what the alternation itself is checked against.
    fn arm(dir: &Path, name: &str, value: f64) -> PathBuf {
        let path = dir.join(name);
        fs::write(
            &path,
            format!(
                "#!/bin/sh\necho {name} >> {}\necho 'decode {value} ms'\n",
                dir.join("order").display()
            ),
        )
        .expect("the arm is written");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("the arm is runnable");
        path
    }

    /// An arm's command line, as the harness holds one.
    fn command(path: &Path) -> Vec<String> {
        vec![path.display().to_string()]
    }

    /// The arms swap places every pair, which is the whole reason this runs them
    /// itself rather than running all of one and then all of the other.
    #[test]
    fn the_order_of_the_two_arms_flips_each_pair() {
        let dir = tempfile::tempdir().expect("a directory");
        let a = arm(dir.path(), "a", 20.0);
        let b = arm(dir.path(), "b", 19.0);

        alternate(3, &[command(&a), command(&b)], &[]).expect("the arms run");

        assert_eq!(
            fs::read_to_string(dir.path().join("order")).expect("the order is recorded"),
            "a\nb\nb\na\na\nb\n"
        );
    }

    /// **An arm may carry arguments of its own**, which is what a change to the
    /// weights rather than to the code looks like here: one executable, two
    /// checkpoints, and the measurement shared between them.
    #[test]
    fn an_arm_may_be_a_command_line_rather_than_a_path() {
        let dir = tempfile::tempdir().expect("a directory");
        let arm = dir.path().join("arm");
        // Reads its last argument, which is where an arm's own land.
        fs::write(
            &arm,
            "#!/bin/sh\nfor value; do :; done\necho \"decode $value ms\"\n",
        )
        .expect("the arm is written");
        fs::set_permissions(&arm, fs::Permissions::from_mode(0o755)).expect("the arm is runnable");

        let with = |value: &str| vec![arm.display().to_string(), value.to_string()];
        alternate(2, &[with("20"), with("19")], &[]).expect("the arms run");

        // The shared arguments come first, so an arm's own are the last words
        // of its command line however many are shared.
        let readings = ask(&with("20"), &["ignored".to_string()]).expect("the arm answers");
        assert_eq!(readings, [Reading::new("decode", 20.0, "ms")]);
    }

    /// A proposer that answers from a list, which is what the two chains of a
    /// `guesses` run are to the counting between them.
    struct Said {
        rounds: std::cell::Cell<usize>,
        said: Vec<Vec<usize>>,
    }

    impl Said {
        fn new(said: &[&[usize]]) -> Self {
            Self {
                rounds: std::cell::Cell::new(0),
                said: said.iter().map(|round| round.to_vec()).collect(),
            }
        }
    }

    impl Proposer for Said {
        fn depth(&self) -> usize {
            self.said.first().map_or(0, Vec::len)
        }

        fn propose(&mut self, _: Round<'_>) -> &[usize] {
            let round = self.rounds.get();
            self.rounds.set(round + 1);
            &self.said[round]
        }
    }

    /// **The gate counts per depth**, because a chain that agrees about its
    /// first guess and diverges deeper is a different result from one that
    /// diverges at the first: the first head is where most of the acceptance is.
    #[test]
    fn holding_two_chains_together_counts_where_they_answered_differently() {
        let mut held = Held::new(
            Said::new(&[&[1, 2, 3], &[4, 5, 6]]),
            Said::new(&[&[1, 9, 3], &[4, 5, 7]]),
            3,
        );
        let round = Round {
            hidden: &[],
            next: &[],
            depth: 3,
        };

        // The guesses the generation runs on are the first chain's, whatever the
        // second said: a comparison that fed the second's back would be
        // measuring a generation nobody asked for.
        assert_eq!(held.propose(round), [1, 2, 3]);
        assert_eq!(held.propose(round), [4, 5, 6]);

        assert_eq!(held.asked, [2, 2, 2]);
        assert_eq!(held.diverged, [0, 1, 1]);
    }

    /// A round shallower than the chain — the last round of a generation whose
    /// budget is running out — is counted at the depths it reached and at no
    /// others.
    #[test]
    fn a_round_that_asked_for_fewer_guesses_is_counted_at_the_depths_it_asked_at() {
        let mut held = Held::new(Said::new(&[&[1]]), Said::new(&[&[2]]), 3);
        held.propose(Round {
            hidden: &[],
            next: &[],
            depth: 1,
        });

        assert_eq!(held.asked, [1, 0, 0]);
        assert_eq!(held.diverged, [1, 0, 0]);
    }

    /// The one measurement that takes two checkpoints rather than one, and the
    /// one that takes neither more nor fewer.
    #[test]
    fn guesses_takes_two_checkpoints() {
        assert_eq!(
            Job::parse(["guesses", "models/one", "models/two", "-k", "2"].map(str::to_string))
                .expect("parses"),
            Job::Guesses {
                checkpoints: [PathBuf::from("models/one"), PathBuf::from("models/two")],
                tokens: DECODED,
                depth: 2,
            }
        );
        for given in [vec!["models/one"], vec!["a", "b", "c"]] {
            let mut args = vec!["guesses".to_string()];
            args.extend(given.iter().map(|dir| dir.to_string()));
            assert!(Job::parse(args).is_err(), "{given:?} parsed");
        }
        // And a measurement takes one, where it used to take the first and
        // ignore the rest.
        assert!(Job::parse(["decode", "a", "b"].map(str::to_string)).is_err());
    }

    /// An arm is split on whitespace, so a program under a directory with a
    /// space in its name is one this cannot run — and says so rather than
    /// running whatever the first half names.
    #[test]
    fn an_arm_whose_path_was_cut_in_half_is_refused() {
        let err = ask(&["/no/such/bench".to_string()], &[]).expect_err("refused");
        assert!(format!("{err:#}").contains("is not a file"), "{err:#}");
    }

    /// One pair is two readings, and two readings always lie apart and always
    /// agree with themselves. What the report says about them has to be that
    /// they say nothing.
    #[test]
    fn one_pair_is_never_a_claim() {
        let compared = compare(&[readings_of(&[20.9]), readings_of(&[20.0])]).expect("compares");
        assert_eq!((compared[0].agreed, compared[0].pairs), (1, 1));
        assert!(!compared[0].overlap);
        assert!(!compared[0].stands());
    }

    fn readings_of(values: &[f64]) -> Vec<Vec<Reading>> {
        values
            .iter()
            .map(|value| vec![Reading::new("decode", *value, "ms")])
            .collect()
    }

    /// The standard the README states: every pair moving the same way and the
    /// two ranges not overlapping.
    #[test]
    fn ranges_that_lie_apart_with_every_pair_agreeing_are_a_claim() {
        let compared = compare(&[
            readings_of(&[20.9, 21.0, 20.8]),
            readings_of(&[20.0, 20.1, 19.9]),
        ])
        .expect("compares");

        let [row] = &compared[..] else {
            panic!("one metric, {compared:?}")
        };
        assert!((row.arms[0].0 - 20.9).abs() < 1e-9, "{:?}", row.arms[0]);
        assert!((row.arms[1].0 - 20.0).abs() < 1e-9, "{:?}", row.arms[1]);
        assert_eq!((row.arms[0].1, row.arms[0].2), (20.8, 21.0));
        assert_eq!((row.arms[1].1, row.arms[1].2), (19.9, 20.1));
        assert!(!row.overlap);
        assert_eq!((row.agreed, row.pairs), (3, 3));
        assert!(row.stands());
        assert!((row.change() - -4.306).abs() < 1e-3, "{}", row.change());
    }

    /// A pair falling the other way is no claim at all, however the means read —
    /// which is the case this repo has published more than once.
    #[test]
    fn a_pair_that_falls_the_other_way_is_not_a_claim() {
        let compared = compare(&[
            readings_of(&[20.9, 19.8, 20.8]),
            readings_of(&[20.0, 20.1, 19.9]),
        ])
        .expect("compares");

        assert_eq!((compared[0].agreed, compared[0].pairs), (2, 3));
        assert!(compared[0].overlap);
        assert!(!compared[0].stands());
        assert!(
            report(&[vec!["a".to_string()], vec!["b".to_string()]], &compared).contains("no claim")
        );
    }

    /// Two arms reporting different metrics cannot be compared row for row, and
    /// lining them up by position would report one measurement under another's
    /// name.
    #[test]
    fn arms_that_report_different_readings_are_refused() {
        let mut b = readings_of(&[20.0]);
        b[0][0].name = "prefill".to_string();
        assert!(compare(&[readings_of(&[20.9]), b]).is_err());
    }
}
