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

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use inkling_cli::args::Backend;
use inkling_cli::kept::{DEFAULT_BOUND, Kept};
use inkling_cli::{backend, config, session};
use inkling_core::generate::{Generator, Picked, Proposer, Round};
use inkling_core::head::Tail;
use inkling_core::model::Batched;
use inkling_core::mtp::{CheckpointHeads, MtpProposer};
use inkling_core::workload::{
    BEST, CORPUS, DECODED, DIFFERENTIAL, REALISTIC, STRUCTURED_PROMPT, SWEPT, Session, tiled,
};
use inkling_core::{
    Checkpoint, CheckpointWeights, Ending, ModelCache, TextConfig, Tokenizer, profile,
};
use inkling_metal::trace;
use inkling_metal::{FusedAttention, Numerics, PackedMatmul};

/// How long a prefill's prompt is, when nobody says. The middle of the three
/// lengths this repo quotes.
const PREFILLED: usize = 385;

/// Steps a width runs and discards before the ones it is timed over.
const SETTLED: usize = 2;

/// The widest batch the batch sweep runs, when nobody says.
///
/// Doubling from one, so that the curve is read on a log axis and the row that
/// says where it stops is the row where doubling stopped halving the token.
const WIDEST: usize = 16;

/// How many pairs `alternate` runs, when nobody says. Seven, which is what every
/// paired figure in the README was taken over.
const PAIRS: usize = 7;

/// Units the clock measurement times, when nobody says.
///
/// Enough decode steps that the run is minutes rather than seconds: what it is
/// asking is whether the part holds a clock under sustained load, and a question
/// about sustained load cannot be put to a run that does not sustain one.
const TICKED: usize = 600;

/// Parts a clock run's units are reported in.
///
/// **A drift is a shape and not a number**, so a run that quoted only its mean
/// could not tell a part that ramped once from one that fell throughout. Five is
/// enough parts to see which, and few enough that each holds tens of units at
/// the default length.
const PARTS: usize = 5;

const USAGE: &str = "usage:\n  \
    bench decode  <checkpoint> [--tokens <n>] [--context <n>] [--numerics <which>]\n  \
    bench prefill <checkpoint> [--tokens <n>] [--numerics <which>]\n  \
    bench sweep   <checkpoint> [--tokens <n>] [--depth <k>] [--numerics <which>]\n  \
    bench engines <checkpoint> [--depth <k>] [--numerics <which>]\n  \
    bench session <checkpoint> [--tokens <n>] [--reuse-tokens <n>] [--numerics <which>]\n  \
    bench batch   <checkpoint> [--tokens <n>] [--context <n>] [--batch <n>]\n  \
    bench clock   <checkpoint> [--tokens <n>] [--context <n>] [--idle <ms>] \
[--prefill <n>] [--batch <n>]\n  \
    bench guesses <checkpoint> <checkpoint> [--tokens <n>] [--depth <k>]\n  \
    bench diverge <checkpoint> [--tokens <n>] [--against <which>]\n  \
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
    /// What a decode step costs as more sequences are decoded through it, which
    /// is the curve this engine's batching exists to produce.
    ///
    /// **The one measurement here that sweeps how many requests are in flight.**
    /// A decode step reads 5.9 GB of weights to produce one token, and at batch
    /// N the same read produces N — so the row that matters is not the step but
    /// the token, and where the two stop diverging is where a batch stops
    /// paying.
    Batch,
    /// What the same unit of work costs the device over and over, and how much
    /// of the wall around each one the device was busy for.
    ///
    /// **The one measurement here whose subject is the machine rather than the
    /// engine.** An Apple GPU moves between power states, and a decode step is
    /// milliseconds of work with a host gap after it where a prefill is a
    /// minute of continuous work — so two figures taken at those two duty
    /// cycles are taken at whatever clocks the part chose to hold, and nothing
    /// else here says what those were.
    ///
    /// The work is fixed, so the device column is the clock read backwards: the
    /// same dispatches over the same shapes cost more device time only if the
    /// part was running slower. `--idle` is the lever — the same steps with
    /// host time deliberately left between them, which is a lower duty cycle at
    /// identical work.
    Clock,
}

#[derive(Debug, PartialEq, Eq)]
enum Job {
    /// One arm's worth of work: a checkpoint, and what to time against it.
    Measure {
        what: What,
        checkpoint: PathBuf,
        /// Tokens decoded, or — for a prefill — tokens in the prompt, or — for
        /// the clock measurement — units of its own work it repeats.
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
        /// How many sequences a unit of work decodes through, and nothing at all
        /// for the measurements that decode one.
        ///
        /// **Two measurements mean two things by it and both of them are a
        /// width.** The sweep runs every width doubling up to this one; the
        /// clock measurement runs *this* width and repeats it, which is what
        /// lets a gap be held constant while the work either side of it grows.
        widest: Option<usize>,
        /// Host time left between one unit of work and the next, which only the
        /// clock measurement has anywhere to put.
        idle: Duration,
        /// The prompt each of the clock measurement's units prefills, and zero
        /// for the decode step it charges otherwise.
        prefill: usize,
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
    /// The same prompts through the reference and one word behind the flag, and
    /// where their tokens part company.
    Diverge {
        checkpoint: PathBuf,
        tokens: usize,
        /// Which word the reference is held against, which is `production`
        /// unless a command line says otherwise.
        ///
        /// **A word rather than both words**, because the two behind the flag
        /// are two different questions. `production` asks what summation order
        /// is worth and has answered 384 of 384 at every milestone; `rounded`
        /// asks what a rounded operand is worth, which is a larger perturbation
        /// and need not answer the same. A run that put both through at once
        /// would report one line for two claims.
        against: Numerics,
    },
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
            Some("batch") => Some(What::Batch),
            Some("clock") => Some(What::Clock),
            Some("alternate") => return Self::alternating(args),
            Some("diverge") => return Self::diverging(args),
            Some("guesses") => None,
            Some(word) => bail!(
                "{word} is not one of decode, prefill, sweep, engines, session, batch, \
                 clock, guesses, diverge or alternate"
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
        // The widest batch a sweep runs, which only the batch sweep takes.
        let mut widest = None;
        // The two the clock measurement takes and nothing else has anywhere to
        // put, `Option` for the reason the numbers above are: zero is a duty
        // cycle this means something by, and so is a prompt of no tokens.
        let mut idle = None;
        let mut prefill = None;
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
                "--batch" | "-b" => widest = Some(count(&arg, &mut args)?),
                // A gap of nothing is the arm every other measurement here runs
                // at and is what this one compares against, so zero is a number
                // it means something by — see `positions`, which is the other
                // flag that takes one.
                "--idle" => {
                    idle = Some(Duration::from_millis(parsed(
                        &arg,
                        &mut args,
                        "a gap in milliseconds",
                        |_| true,
                    )? as u64));
                }
                "--prefill" => prefill = Some(count(&arg, &mut args)?),
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
        if !matches!(what, What::Decode | What::Batch | What::Clock) && context.is_some() {
            bail!("{what:?} takes no --context: only a decode step has one to be taken at");
        }
        // **Only the clock measurement has a duty cycle it is asking about.**
        // Every other row here is taken at whatever occupancy its own work makes
        // and says so nowhere, which is the whole of what this measurement
        // exists to fix — so the two levers are refused to the measurements that
        // could only drop them, under the same rule as the numbers above.
        if what != What::Clock && (idle.is_some() || prefill.is_some()) {
            bail!("{what:?} takes no --idle or --prefill: it does not vary its own duty cycle");
        }
        // **A prefill's unit is its own prompt.** Its keys are the tokens
        // `--prefill` names and there is no step behind a context, so a number
        // given for one could only be dropped — the same rule as above.
        if prefill.is_some() && context.is_some() {
            bail!("a clock run over prefills takes no --context: its keys are its own prompt");
        }
        // **Only the two measurements that have a width take one.** Every other
        // one here decodes a single sequence, so a width handed to it could only
        // be dropped — the same rule the numbers above are refused under.
        if !matches!(what, What::Batch | What::Clock) && widest.is_some() {
            bail!("{what:?} takes no --batch: it decodes one sequence");
        }
        // **A prefill already fills the machine**, which is why the batching
        // milestone batched the other regime — so a width given to the arm whose
        // unit is a prompt would be a width nothing could do anything with.
        if prefill.is_some() && widest.is_some() {
            bail!("a clock run over prefills takes no --batch: a prefill fills the machine alone");
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
                What::Clock => TICKED,
                _ => DECODED,
            }),
            context: context.unwrap_or(0),
            depth,
            numerics: numerics.unwrap_or_default(),
            reuse: reuse.unwrap_or(DEFAULT_BOUND),
            widest,
            idle: idle.unwrap_or_default(),
            prefill: prefill.unwrap_or(0),
        })
    }

    /// `diverge`, which takes one checkpoint and opens both arms itself — so
    /// the word it takes names the second of them rather than the run.
    fn diverging(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let mut checkpoints = Vec::new();
        let mut tokens = DIFFERENTIAL;
        let mut against = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokens" | "-n" => tokens = count(&arg, &mut args)?,
                "--against" => against = Some(which(&arg, &mut args)?),
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
        let against = against.unwrap_or(Numerics::Production);
        // **The reference is one arm of this and cannot be the other.** A run
        // that took it would compare a path to itself and report perfect
        // agreement, which is what a differential run looks like when it has
        // measured nothing.
        if !against.compiles_the_entries() {
            bail!(
                "--against {} is the arm every run of this already has: a differential run of the \
                 reference against itself agrees perfectly and says nothing",
                against.named()
            );
        }
        Ok(Self::Diverge {
            checkpoint,
            tokens,
            against,
        })
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
                widest,
                idle,
                prefill,
            } => {
                let asked = Asked {
                    tokens: *tokens,
                    context: *context,
                    depth: *depth,
                    numerics: *numerics,
                    reuse: *reuse,
                    widest: *widest,
                    idle: *idle,
                    prefill: *prefill,
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
            Self::Diverge {
                checkpoint,
                tokens,
                against,
            } => {
                for reading in diverge(checkpoint, *tokens, *against)? {
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
    Numerics::parse(&name).with_context(|| {
        let every: Vec<&str> = Numerics::EVERY.into_iter().map(Numerics::named).collect();
        format!(
            "{name} is not numerics, which is one of {}",
            every.join("|")
        )
    })
}

/// A count of positions to keep, which may be zero: keeping nothing is the arm
/// a session is measured against, and it is the one number [`count`] refuses.
fn positions(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<usize> {
    parsed(flag, args, "a count of positions", |_| true)
}

/// A count of at least one, which every number this takes is: no measurement is
/// defined over zero tokens, zero pairs or a sweep of no depths.
fn count(flag: &str, args: &mut impl Iterator<Item = String>) -> Result<usize> {
    parsed(flag, args, "a count of at least one", |count| *count > 0)
}

/// The next argument as a number `wanted` accepts, and a refusal naming both the
/// word and what it was meant to be.
fn parsed(
    flag: &str,
    args: &mut impl Iterator<Item = String>,
    what: &str,
    wanted: impl Fn(&usize) -> bool,
) -> Result<usize> {
    let value = args
        .next()
        .with_context(|| format!("{flag} takes a value"))?;
    match value.parse() {
        Ok(number) if wanted(&number) => Ok(number),
        _ => bail!("{value} is not {what}, after {flag}"),
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
    /// How many sequences a unit of work decodes through: the widest of the
    /// sweep's doubling, or the one width the clock measurement repeats.
    widest: Option<usize>,
    /// Host time the clock measurement leaves between its units, ignored by
    /// everything else.
    idle: Duration,
    /// The prompt each of the clock measurement's units prefills, and zero for
    /// the decode step it charges otherwise.
    prefill: usize,
}

fn measure(what: What, dir: &Path, asked: Asked) -> Result<Vec<Reading>> {
    let Asked {
        tokens,
        context,
        depth,
        numerics,
        reuse,
        widest,
        idle,
        prefill,
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
        let weights = backend::weights(gpu.as_ref(), &ckpt, text, depth, 1)?;
        let tail = backend::tail_weights(&weights, text);
        let heads = backend::heads(gpu.as_ref(), &ckpt, &config, depth, &tail, 1)?;
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
            // **The amortisation curve, and it is the result of this
            // milestone.** Every other row here is what one token costs; these
            // are what a token costs as more sequences are decoded through the
            // same weight reads, doubling from one. The step grows and the token
            // falls, and where the token stops falling is where a batch stops
            // paying — which is a shape rather than a number, so the sweep
            // prints every width rather than the best one.
            //
            // Wrapped once per width, because the slots are allocated at wrap
            // time: a slot is a span and four convolution windows in every
            // layer, and a stack wrapped for sixteen holds sixteen of them
            // whether or not sixteen sequences are in flight.
            What::Batch => {
                for slots in widths(widest.unwrap_or(WIDEST)) {
                    let weights = backend::weights(gpu.as_ref(), &ckpt, text, 0, slots)?;
                    let ids = behind_a_step(&prompt, context);
                    let run = batched(&weights, text, &ids, slots, tokens);
                    // A step is what the batch waits; a token is what one
                    // sequence in it waits, which is the step divided by the
                    // sequences that took a token out of it.
                    taken.push(Reading::new(
                        format!("batch{slots}.step"),
                        millis(run.step),
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("batch{slots}.token"),
                        millis(run.step) / slots as f64,
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("batch{slots}.device"),
                        millis(run.gpu),
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("batch{slots}.rate"),
                        slots as f64 / run.step.as_secs_f64(),
                        "tok/s",
                    ));
                    eprintln!(
                        "batch {slots}: step {:.2?}, token {:.2?}, device {:.2?}, {:.1} tok/s",
                        run.step,
                        run.step / slots as u32,
                        run.gpu,
                        slots as f64 / run.step.as_secs_f64(),
                    );
                }
            }
            // **The same work over and over, and what each of them cost the
            // device.** Everything else here quotes a mean over a run and so
            // cannot say whether the run held its speed; this quotes the run in
            // parts, and the device column is what a clock is read off where the
            // work is fixed — see [`Unit`] for which of the two arms that is
            // true of, because the drift column means different things under
            // them.
            What::Clock => {
                let unit = match (prefill, widest) {
                    (0, None) => Unit::Step(behind_a_step(&prompt, context)),
                    (0, Some(slots)) => Unit::Batch(behind_a_step(&prompt, context), slots),
                    (length, _) => Unit::Prefill(tiled(&prompt, length)),
                };
                // Wrapped for the width being repeated, for the reason the sweep
                // wraps per width: a slot is a span and four convolution windows
                // in every layer, allocated when the stack is wrapped rather
                // than when a sequence sits in one.
                let wrapped = match widest {
                    None => None,
                    Some(slots) => Some(backend::weights(gpu.as_ref(), &ckpt, text, 0, slots)?),
                };
                let ticks = ticked(
                    wrapped.as_ref().unwrap_or(&weights),
                    text,
                    &unit,
                    tokens,
                    idle,
                );
                eprintln!(
                    "clock: {} {}, {} idle between them",
                    ticks.len(),
                    unit.over(ticks.len()),
                    match idle.is_zero() {
                        true => "no".to_string(),
                        false => format!("{:.0?}", idle),
                    },
                );
                taken.extend(clocked(&ticks));
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
                        "  turn {at}: {} tokens, {} reused, {:.2} s wall, {:.2} s to first, \
                         {:.2} ms bookkeeping",
                        turn.prompt,
                        turn.reused,
                        turn.wall.as_secs_f64(),
                        turn.first.as_secs_f64(),
                        millis(turn.bookkeeping),
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
                // **What the arrangement costs whether it hits or misses**, which
                // is the question a feature like this owes about its own bad
                // case. Per turn rather than for the session, because it is a
                // per-request cost and the number a server is judged on is what
                // one request pays.
                taken.push(Reading::new(
                    "bookkeeping",
                    millis(turns.iter().map(|turn| turn.bookkeeping).sum())
                        / turns.len().max(1) as f64,
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

/// The widths a batch sweep runs, doubling from one up to `widest`.
///
/// Doubling rather than every width, because what the curve says is where
/// halving stops: a token that halves from one width to the next is a batch
/// that is still free, and the row where it stops is the answer. The widest is
/// included whether or not it is a power of two.
fn widths(widest: usize) -> Vec<usize> {
    let mut widths: Vec<usize> = std::iter::successors(Some(1usize), |n| Some(n * 2))
        .take_while(|n| *n <= widest)
        .collect();
    if widths.last() != Some(&widest) {
        widths.push(widest);
    }
    widths
}

/// What one batched decode step costs, over `slots` copies of `ids` decoded
/// together.
struct Amortised {
    /// The wall one step of the whole batch takes, meaned over the steps.
    step: Duration,
    /// The device time inside it, on the same account every other row here
    /// reads.
    gpu: Duration,
}

/// `slots` sequences in slots of their own, prefilled and settled, as the thing
/// that advances all of them one step.
///
/// **The prompts are made distinct rather than copied**, because the routing is
/// the prompt's: a batch of identical sequences routes every row of every step
/// to the same six experts, which is the one distribution a grouped dispatch is
/// least like the real one at. Each sequence's prompt is rotated by its own slot,
/// so the sequences are the same length and not the same tokens.
struct Seated<'a> {
    generator: Generator<'a>,
    caches: Vec<ModelCache>,
    pending: Vec<usize>,
}

impl<'a> Seated<'a> {
    /// `slots` sequences prefilled one at a time and stepped [`SETTLED`] times
    /// before whoever asked for them starts timing.
    ///
    /// The prefills are not timed by anyone. A prefill of any length already
    /// fills the machine — which is why this engine batches the other regime —
    /// so what a batch is worth is the steps after them, and a mean that
    /// included N prefills would be a measurement of the prompts.
    ///
    /// **The settling steps are thrown away for the reason the warm generation
    /// above the sweep is.** A width is a fresh wrap — a slot is buffers this
    /// device has not allocated before — so the first steps of a width pay for
    /// allocating its spans and its windows, which belongs to the wrap rather
    /// than to the step. The device's own clock does not see it and the wall
    /// does: unsettled, the batch of one read 25.9 ms against its own 16.4.
    fn new(
        weights: &'a CheckpointWeights<'_>,
        config: &TextConfig,
        ids: &[usize],
        slots: usize,
    ) -> Self {
        let generator = weights.generator();
        let want = Tail {
            block: 1,
            chained: false,
            logits: false,
        };
        let mut caches: Vec<ModelCache> = (0..slots)
            .map(|slot| ModelCache::in_slot(config, 0, slot))
            .collect();
        let pending: Vec<usize> = caches
            .iter_mut()
            .enumerate()
            .map(|(slot, cache)| {
                let mut own = ids.to_vec();
                own.rotate_left(slot % ids.len());
                generator.tailed(cache, &own, want, weights).picks[0]
            })
            .collect();

        let mut seated = Self {
            generator,
            caches,
            pending,
        };
        for _ in 0..SETTLED {
            seated.step(weights);
        }
        seated
    }

    /// One step of the whole batch, each sequence fed the id its own last step
    /// named.
    fn step(&mut self, weights: &CheckpointWeights<'_>) {
        let feeding: Vec<[usize; 1]> = self.pending.iter().map(|id| [*id]).collect();
        let mut batch: Vec<Batched<'_>> = self
            .caches
            .iter_mut()
            .zip(&feeding)
            .map(|(cache, ids)| Batched { cache, ids })
            .collect();
        self.pending = self
            .generator
            .step_batch(&mut batch, weights)
            .iter()
            .map(Picked::last)
            .collect();
    }
}

/// `slots` sequences prefilled one at a time and then decoded together, timed
/// over the decode steps alone.
fn batched(
    weights: &CheckpointWeights<'_>,
    config: &TextConfig,
    ids: &[usize],
    slots: usize,
    budget: usize,
) -> Amortised {
    let mut seated = Seated::new(weights, config, ids, slots);

    profile::take();
    let started = Instant::now();
    for _ in 0..budget {
        seated.step(weights);
    }
    let wall = started.elapsed();
    let charged = u32::try_from(budget.max(1)).unwrap_or(1);
    Amortised {
        step: wall / charged,
        gpu: profile::take().per_step(charged).gpu(),
    }
}

/// The work a clock run repeats, which is one decode step behind a context or
/// one prefill of a prompt.
///
/// **Two units rather than two measurements**, because the question is the same
/// one either way and only the duty cycle differs: a decode step is milliseconds
/// of device work with a host gap after it, and a prefill of thousands of tokens
/// is a minute of it with nothing else in the way.
///
/// **Only a prefill's units are the same work**, and that decides what the drift
/// column is worth under each of them:
///
/// - A prefill's unit is a prompt of a fixed length through a cache that starts
///   empty every time, so every one of them is the same dispatches over the same
///   shapes. A unit that took the device longer was run on a slower part, and
///   nothing else can have moved.
/// - A decode step's unit walks one more key than the one before it, because the
///   steps are one generation and that is what a generation does. This repo has
///   measured what those keys cost — a step at 8192 keys is 1.45× its 97-key
///   figure — so a drift over a run of them is that slope plus whatever the
///   clock did, and the two are not separated here. [`Unit::over`] prints the
///   range so that a reader is told which figure they are holding.
///
/// **What the step arm is for is the pair rather than the drift**: two runs of
/// it over the same keys, one with `--idle` and one without, have the same slope
/// under both arms and differ only in occupancy.
///
/// **The batched unit is the same step over more sequences**, and it is what
/// separates a gap from the occupancy it produces. A step of one sequence is
/// 15 ms of work and a step of thirty-two is 223, so the same gap left after
/// each of them is two very different duty cycles at the same gap — which is
/// the one arrangement that says which of the two a slower clock is a function
/// of. Its keys grow exactly as the single-sequence arm's do, one a step in
/// every slot.
enum Unit {
    Step(Vec<usize>),
    Batch(Vec<usize>, usize),
    Prefill(Vec<usize>),
}

impl Unit {
    /// What a run of `units` of this was over, as a reader needs it stated: the
    /// keys a decode run walked between its first timed step and its last, or
    /// the one prompt length every prefill of a run shares.
    ///
    /// The first timed step is the one after the prompt's own prefill, so the
    /// range opens one key past the prompt.
    fn over(&self, units: usize) -> String {
        match self {
            Self::Step(ids) => format!(
                "decode steps over {} to {} keys",
                ids.len() + 1,
                ids.len() + units
            ),
            Self::Batch(ids, slots) => format!(
                "{slots}-wide decode steps over {} to {} keys a sequence",
                ids.len() + 1 + SETTLED,
                ids.len() + SETTLED + units
            ),
            Self::Prefill(ids) => format!("prefills of {} tokens", ids.len()),
        }
    }
}

/// One repetition of a clock run's unit, on the two clocks that can see it.
///
/// **The wall is the whole period and the device is the work inside it**, so
/// their ratio is the duty cycle the unit was run at — which is what `--idle`
/// moves and is the figure this measurement exists to put beside every other
/// one. A gap left between two units belongs to the period and to neither
/// unit's work.
#[derive(Debug, Clone, Copy)]
struct Tick {
    wall: Duration,
    gpu: Duration,
}

/// `count` repetitions of `unit`, back to back, with `idle` of host time
/// deliberately left between them.
///
/// **What each arm discards before the ones that are timed is its own**, and
/// every arm discards something for the reason the prefill measurement takes the
/// second of two: the first unit of a shape is the one that faults in the pages
/// the rest of them read, and a page fault charged to the first unit is a drift
/// this would report as the part warming up. A generation's own first tick is
/// the prompt's prefill, which is not a step at all; a batched run's is
/// [`Seated`]'s settling, which it has already run and thrown away.
fn ticked(
    weights: &CheckpointWeights<'_>,
    config: &TextConfig,
    unit: &Unit,
    count: usize,
    idle: Duration,
) -> Vec<Tick> {
    profile::take();
    let mut ticking = Ticking::new(idle, count + 1);
    let warmed = match unit {
        // Every step of one generation rather than a generation apiece, so that
        // the gap `--idle` leaves falls between two decode steps — which is the
        // occupancy this arm exists to move. What it costs is the growing key
        // count [`Unit`] describes.
        Unit::Step(ids) => {
            let generator = weights.generator();
            let cache = &mut ModelCache::speculating(config, 0);
            let ending = Ending {
                budget: count + 1,
                eos: None,
            };
            let mut sink = |_| {
                ticking.tick(profile::take().gpu());
                ControlFlow::Continue(())
            };
            generator.stream(cache, ids, ending, weights, &mut sink);
            1
        }
        // The gap falls between two steps of one batch, as it does between two
        // steps of one generation: what moves is how much work the period holds
        // either side of it.
        Unit::Batch(ids, slots) => {
            let mut seated = Seated::new(weights, config, ids, *slots);
            // The prefills and the settling steps off both accounts before the
            // first timed one, which is what the single-sequence arm's
            // discarded first tick does for it. Unsettled, the first period is
            // the seating — 1.30 s of wall around 24 ms of device, which is a
            // duty cycle of 1.9% this measurement would report as the machine's.
            ticking.settled();
            for _ in 0..count {
                seated.step(weights);
                ticking.tick(profile::take().gpu());
            }
            0
        }
        Unit::Prefill(ids) => {
            for _ in 0..count + 1 {
                let run = generate(weights, None, config, ids, 1, 0);
                ticking.tick(run.gpu);
            }
            1
        }
    };
    let mut ticks = ticking.ticks;
    ticks.drain(..warmed.min(ticks.len()));
    ticks
}

/// The clock a run of units is timed on, and the gap it leaves after each of
/// them.
///
/// A type rather than a closure because both arms of [`ticked`] keep the same
/// two pieces of state and one of them reaches it from inside a sink the
/// generator drives.
struct Ticking {
    at: Instant,
    idle: Duration,
    ticks: Vec<Tick>,
}

impl Ticking {
    fn new(idle: Duration, count: usize) -> Self {
        Self {
            at: Instant::now(),
            idle,
            ticks: Vec::with_capacity(count),
        }
    }

    /// Both clocks back to zero after work nobody is timing, which is what the
    /// arms that discard a first unit get by discarding it: the first period
    /// opens where the first timed unit does and not where the run did.
    fn settled(&mut self) {
        profile::take();
        self.at = Instant::now();
    }

    /// One unit finished for `gpu` of device time, the gap it is followed by,
    /// and the period the two of them are.
    ///
    /// The sleep is behind a test so that a run asking for no gap makes no
    /// syscall a run of any other measurement here would not: what this is
    /// timing is a decode step, and a decode step's own host side is 8% of it.
    fn tick(&mut self, gpu: Duration) {
        if !self.idle.is_zero() {
            std::thread::sleep(self.idle);
        }
        self.ticks.push(Tick {
            wall: self.at.elapsed(),
            gpu,
        });
        self.at = Instant::now();
    }
}

/// What a run of identical units says about the clock underneath them.
///
/// The run in [`PARTS`] parts, each part's mean device time and duty cycle, and
/// the last part against the first — which is the number that says whether the
/// part held its speed, and in which direction it did not.
fn clocked(ticks: &[Tick]) -> Vec<Reading> {
    let mean = |part: &[Tick]| match part.is_empty() {
        true => (Duration::ZERO, Duration::ZERO),
        false => (
            part.iter().map(|tick| tick.gpu).sum::<Duration>() / part.len() as u32,
            part.iter().map(|tick| tick.wall).sum::<Duration>() / part.len() as u32,
        ),
    };
    let (whole_gpu, whole_wall) = mean(ticks);
    // **[`PARTS`] parts wherever there are units for them**, cut at the
    // proportion rather than by a fixed chunk: a chunk wide enough for the last
    // part to be short is one that leaves a run of eleven reporting four parts
    // and a run of ten reporting five, so the shape a reader compares would
    // depend on a count nobody chose for its divisibility.
    let cuts = PARTS.min(ticks.len());
    let parts: Vec<(Duration, Duration)> = (0..cuts)
        .map(|at| mean(&ticks[at * ticks.len() / cuts..(at + 1) * ticks.len() / cuts]))
        .collect();

    eprintln!("  part      device       wall    duty    against the first");
    let mut readings = Vec::new();
    for (at, (gpu, wall)) in parts.iter().enumerate() {
        eprintln!(
            "  {:>4}   {:>9.4?}  {:>9.4?}  {:>5.1}%    {}",
            at + 1,
            gpu,
            wall,
            duty(*gpu, *wall),
            match at {
                0 => String::new(),
                _ => format!("{:+.2}%", against(*gpu, parts[0].0)),
            },
        );
        readings.push(Reading::new(
            format!("part{}.device", at + 1),
            millis(*gpu),
            "ms",
        ));
    }
    let drift = parts
        .last()
        .map_or(0.0, |(last, _)| against(*last, parts[0].0));
    eprintln!(
        "  whole: device {:.4?}, wall {:.4?}, duty {:.1}%, drift {:+.2}%",
        whole_gpu,
        whole_wall,
        duty(whole_gpu, whole_wall),
        drift,
    );
    readings.extend([
        Reading::new("clock.device", millis(whole_gpu), "ms"),
        Reading::new("clock.wall", millis(whole_wall), "ms"),
        Reading::new("clock.duty", duty(whole_gpu, whole_wall), "%"),
        Reading::new("clock.drift", drift, "%"),
    ]);
    readings
}

/// How much of the wall around a unit of work the device was busy for, which is
/// the number a figure taken at an unstated occupancy is missing.
fn duty(gpu: Duration, wall: Duration) -> f64 {
    match wall.is_zero() {
        true => 0.0,
        false => 100.0 * gpu.as_secs_f64() / wall.as_secs_f64(),
    }
}

/// One duration against another as a percentage, positive where the first is the
/// longer — which for a fixed amount of work is the slower clock.
fn against(taken: Duration, first: Duration) -> f64 {
    match first.is_zero() {
        true => 0.0,
        false => 100.0 * (taken.as_secs_f64() / first.as_secs_f64() - 1.0),
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
    let weights = backend::weights(gpu.as_ref(), &stack, text, depth, 1)?;
    let tail = backend::tail_weights(&weights, text);
    let ran = backend::heads(gpu.as_ref(), &stack, &config, depth, &tail, 1)?
        .context("the first checkpoint has no heads to guess with")?;
    let against = backend::heads(gpu.as_ref(), &beside, &config, depth, &tail, 1)?
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

/// The corpus through the reference and one word behind the flag, and where the
/// two continuations part company.
///
/// **This is the instrument the flag exists for.** Under the reference the
/// engine's answer is checkable against a recorded array of bits, and every
/// gated case in the tree checks it. Behind the flag there is no such array and
/// there cannot be one — a matrix instruction's summation order is not this
/// side's to record — so what stands in for the oracle is a second
/// implementation: two GPU paths that share every tiling decision, every
/// predicate and every dispatch, and differ only in how the innermost sum is
/// carried. Where those two agree, the structure around the sum is agreed by two
/// independent accumulations; where they disagree, the disagreement is between
/// the arithmetic and nothing else, and the position it first appears at is
/// where to look.
///
/// **What `--against rounded` asks is a different question in the same shape.**
/// There the two paths differ by a rounded operand rather than by a summation
/// order, so a parting is not evidence of a bug — it is the answer. The corpus,
/// the floors, the dispatch-list check and the reported leading agreement are
/// the same either way, which is the point of taking the word as an argument
/// rather than writing a second instrument.
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
fn diverge(dir: &Path, tokens: usize, against: Numerics) -> Result<Vec<Reading>> {
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
    let mut dispatched = Vec::new();
    for numerics in [Numerics::Reference, against] {
        let gpu = backend::open(Backend::Metal, numerics)?;
        let weights = backend::weights(gpu.as_ref(), &ckpt, text, 0, 1)?;
        let generator = weights.generator();
        let mut ran = Vec::new();
        let mut ran_entries = Dispatched::default();
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
            // **Every prompt, and the first two steps of each.** Which entries
            // a step goes through is decided by the shapes it dispatches, and a
            // decode step's shapes are the step before it's — so the third step
            // of a generation is the second described again. The *prompt* is a
            // shape and is the one that decides whether a call clears a blocked
            // entry's height, which is why the record is per prompt rather than
            // taken once: the corpus is six lengths as well as six
            // distributions, and only its longest member reaches the grouped
            // entry.
            //
            // **What this would miss is an entry gated on the context rather
            // than on the call**, and what says there is none is that neither
            // gate is handed the context to read: `PackedMatmul::blocks` turns
            // on a call's rows and `splits_for` on its output elements, and both
            // are the checkpoint's own shapes at every length a sequence
            // reaches. An entry that took such a gate would need a record over
            // more than the two regimes, and this is the line that would have to
            // move.
            let mut steps = 0;
            trace::record(true);
            generator.stream(cache, ids, ending, &weights, &mut |token| {
                if steps < REGIMES {
                    steps += 1;
                    ran_entries.step(steps, trace::take());
                    trace::record(steps < REGIMES);
                }
                continued.push(token);
                ControlFlow::Continue(())
            });
            trace::record(false);
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
        eprintln!(
            "{}: a prefill ran {}; a decode step ran {}",
            numerics.named(),
            named_entries(&ran_entries.prefill),
            named_entries(&ran_entries.decode),
        );
        answers.push(ran);
        dispatched.push(ran_entries);
    }
    let [reference, behind] = <[Vec<Vec<usize>>; 2]>::try_from(answers)
        .map_err(|_| anyhow::anyhow!("a differential run answers twice"))?;
    let [under_reference, under_flag] = <[Dispatched; 2]>::try_from(dispatched)
        .map_err(|_| anyhow::anyhow!("a differential run answers twice"))?;

    let selected = under_flag.beyond(&under_reference);
    eprintln!(
        "{} selected {} at a prefill and {} at a decode step",
        against.named(),
        named_entries(&selected.prefill),
        named_entries(&selected.decode),
    );
    unreached(&selected)?;

    for (at, (was, is)) in reference.iter().zip(&behind).enumerate() {
        let agreed = agreement(was, is);
        match agreed < was.len() {
            true => eprintln!(
                "prompt {} parted at token {}: reference {:?}, {} {:?}",
                at + 1,
                agreed + 1,
                &was[agreed..was.len().min(agreed + 4)],
                against.named(),
                &is[agreed..is.len().min(agreed + 4)],
            ),
            false => eprintln!("prompt {} agreed for all {} tokens", at + 1, was.len()),
        }
    }
    let mut taken = parted(&reference, &behind);
    taken.extend(reached(&selected));
    Ok(taken)
}

/// The entries one arm of a differential run dispatched, kept apart by the two
/// regimes a generation has.
///
/// **A prefill and a decode step reach different entries and are gated
/// differently**, which is the whole reason this is two sets and not one. A
/// prefill's calls bring rows and clear a height; a decode step's bring one row
/// and clear nothing, so an entry it reaches is reached by every generation
/// there has ever been and by no prompt in particular. A single set would report
/// the union and hide which of the two was empty.
#[derive(Debug, Default, PartialEq, Eq)]
struct Dispatched {
    prefill: BTreeSet<String>,
    decode: BTreeSet<String>,
}

/// Steps of a generation the record is kept over, which is one of each regime.
///
/// A third step is the second described again — a decode step's shapes are the
/// step before it's — so what a longer record would add is `Encoded` and not
/// entries.
const REGIMES: usize = 2;

impl Dispatched {
    /// Step `at` of one generation folded in, counting from one: the first is
    /// that generation's prefill, the second its first decode step, and every
    /// later one is the second over again.
    ///
    /// The prompt enters the cache on the first step — a generation prefills
    /// there rather than in a step of its own — so the two regimes are the first
    /// two steps of one generation rather than two generations.
    fn step(&mut self, at: usize, encoded: Vec<inkling_metal::trace::Encoded>) {
        let into = match at {
            1 => &mut self.prefill,
            _ => &mut self.decode,
        };
        into.extend(encoded.into_iter().map(|dispatch| dispatch.symbol));
    }

    /// What this arm ran that `other` did not, which under the production
    /// numerics is exactly the entries the flag selected.
    fn beyond(&self, other: &Self) -> Self {
        let apart = |a: &BTreeSet<String>, b: &BTreeSet<String>| a.difference(b).cloned().collect();
        Self {
            prefill: apart(&self.prefill, &other.prefill),
            decode: apart(&self.decode, &other.decode),
        }
    }
}

/// A set of entries as a line of a report, and `nothing` where it is empty —
/// which is a reading and not a blank.
fn named_entries(entries: &BTreeSet<String>) -> String {
    match entries.is_empty() {
        true => "nothing".to_string(),
        false => entries.iter().cloned().collect::<Vec<_>>().join(", "),
    }
}

/// That the corpus reached every entry the flag selects, or which of them it
/// did not.
///
/// **This is the check the token-length floor above cannot make.** That floor
/// asks whether a *prompt* is long enough to be given a blocked entry, which is
/// the only question while every entry behind the flag is gated on a height.
/// An entry a decode step dispatches is gated on nothing — a call of one row
/// reaches it — so no length says whether a differential run ran it, and a run
/// that never did would report its 384 agreeing argmaxes exactly as loudly as
/// one that ran it at every step.
///
/// So what is held is what was *dispatched*, against the list each kernel keeps
/// of what it put behind the flag. A corpus that reaches nothing fails here
/// rather than passing quietly, and so does an entry added behind the flag and
/// never given a shape that reaches it.
fn unreached(selected: &Dispatched) -> Result<()> {
    let ran: BTreeSet<&str> = selected
        .prefill
        .iter()
        .chain(&selected.decode)
        .map(String::as_str)
        .collect();
    let missing: Vec<&str> = behind_the_flag()
        .filter(|entry| !ran.contains(entry))
        .collect();
    match missing.is_empty() {
        true => Ok(()),
        false => bail!(
            "the corpus reached {} of the {} entries behind the flag: nothing in it dispatched {}, \
             so its agreement is a check on this harness rather than on that arithmetic",
            ran.len(),
            behind_the_flag().count(),
            missing.join(", "),
        ),
    }
}

/// Every entry the two kernels that take the flag put behind it.
///
/// One reading of the two lists, because a check against them and a count of
/// them are the same question asked twice — and a refusal that said "3 of 3"
/// while naming a missing entry is the shape that drift would take.
fn behind_the_flag() -> impl Iterator<Item = &'static str> {
    PackedMatmul::BEHIND_THE_FLAG
        .iter()
        .chain(FusedAttention::BEHIND_THE_FLAG)
        .copied()
}

/// What the flag reached, as readings.
///
/// **Counts rather than names, because the protocol between an arm and the
/// harness is `name value unit`** — the names go to stderr, where a human reads
/// them, and what crosses to a report is how many of each there were. A run that
/// reached the flag at a decode step and one that did not are two different
/// measurements, and a report that could not tell them apart would quote the
/// second under the first's heading.
fn reached(selected: &Dispatched) -> Vec<Reading> {
    vec![
        Reading::new("selected.prefill", selected.prefill.len() as f64, "n"),
        Reading::new("selected.decode", selected.decode.len() as f64, "n"),
    ]
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

    use inkling_metal::trace::Encoded;

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
                widest: None,
                idle: Duration::ZERO,
                prefill: 0,
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
                widest: None,
                idle: Duration::ZERO,
                prefill: 0,
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
                widest: None,
                idle: Duration::ZERO,
                prefill: 0,
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
                widest: None,
                idle: Duration::ZERO,
                prefill: 0,
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

    /// **Only the two measurements that decode more than one sequence take a
    /// width**, and every other one refuses it rather than dropping it — which
    /// is the rule the numbers above are refused under.
    ///
    /// Both take a context for the reason a decode step does: a batch is decode
    /// steps, and a step is the one measurement here with a context to be taken
    /// at.
    #[test]
    fn only_the_measurements_that_decode_a_batch_take_a_width() {
        assert_eq!(
            Job::parse(
                ["batch", "models/small", "--batch", "8", "--context", "8192"].map(str::to_string)
            )
            .expect("parses"),
            Job::Measure {
                what: What::Batch,
                checkpoint: PathBuf::from("models/small"),
                tokens: DECODED,
                context: 8192,
                depth: SWEPT,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
                widest: Some(8),
                idle: Duration::ZERO,
                prefill: 0,
            }
        );
        assert!(
            matches!(
                Job::parse(["clock", "models/small", "--batch", "32"].map(str::to_string))
                    .expect("parses"),
                Job::Measure {
                    what: What::Clock,
                    widest: Some(32),
                    ..
                }
            ),
            "a clock run took no width to repeat"
        );
        for what in ["decode", "prefill", "sweep", "engines", "session"] {
            assert!(
                Job::parse([what, "models/small", "--batch", "8"].map(str::to_string)).is_err(),
                "{what} took a width to sweep"
            );
        }
        // **A prefill fills the machine on its own**, so the arm whose unit is a
        // prompt has nowhere to put a width — refused rather than dropped, like
        // the context that arm is already refused.
        assert!(
            Job::parse(
                ["clock", "models/small", "--prefill", "2048", "--batch", "8"].map(str::to_string)
            )
            .is_err(),
            "a clock run over prefills took a width"
        );
    }

    /// The widths a sweep runs double from one, and the widest is run whether or
    /// not doubling reaches it.
    #[test]
    fn a_batch_sweep_doubles_from_one_and_ends_at_the_width_it_was_given() {
        assert_eq!(widths(1), vec![1]);
        assert_eq!(widths(8), vec![1, 2, 4, 8]);
        assert_eq!(widths(12), vec![1, 2, 4, 8, 12]);
    }

    /// **Only the clock measurement varies its own duty cycle**, which is the
    /// rule the two levers no other measurement has anywhere to put are refused
    /// under — and a run that silently dropped the gap it was told to leave
    /// would report a figure about a duty cycle nobody asked for.
    #[test]
    fn only_a_clock_run_varies_its_own_duty_cycle() {
        assert_eq!(
            Job::parse(
                ["clock", "models/small", "--idle", "40", "--context", "8192"].map(str::to_string)
            )
            .expect("parses"),
            Job::Measure {
                what: What::Clock,
                checkpoint: PathBuf::from("models/small"),
                tokens: TICKED,
                context: 8192,
                depth: SWEPT,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
                widest: None,
                idle: Duration::from_millis(40),
                prefill: 0,
            }
        );
        for what in ["decode", "prefill", "sweep", "engines", "session", "batch"] {
            for lever in [["--idle", "40"], ["--prefill", "2048"]] {
                let given = [what, "models/small"].map(str::to_string);
                assert!(
                    Job::parse(given.into_iter().chain(lever.map(str::to_string))).is_err(),
                    "{what} took {} it has no use for",
                    lever[0]
                );
            }
        }
    }

    /// **A prefill's keys are its own prompt**, so the two lengths cannot both
    /// be given: a run told to prefill 2048 tokens behind a context of 8192 has
    /// been given two answers to one question.
    #[test]
    fn a_clock_run_over_prefills_takes_no_context() {
        assert_eq!(
            Job::parse(["clock", "models/small", "--prefill", "2048"].map(str::to_string))
                .expect("parses"),
            Job::Measure {
                what: What::Clock,
                checkpoint: PathBuf::from("models/small"),
                tokens: TICKED,
                context: 0,
                depth: SWEPT,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
                widest: None,
                idle: Duration::ZERO,
                prefill: 2048,
            }
        );
        assert!(
            Job::parse(
                [
                    "clock",
                    "models/small",
                    "--prefill",
                    "2048",
                    "--context",
                    "8192"
                ]
                .map(str::to_string)
            )
            .is_err(),
            "a run over prefills took a context as well as a prompt"
        );
        // Zero is the word for the other unit and cannot also be a prompt
        // length, so it is refused rather than taken for either.
        assert!(
            Job::parse(["clock", "models/small", "--prefill", "0"].map(str::to_string)).is_err(),
            "a prefill of no tokens was taken for a unit"
        );
    }

    /// **A gap of nothing is a gap this measurement means something by** — it is
    /// the arm every other measurement here runs at and the one the idled arm is
    /// compared against — so `--idle 0` is a run rather than a refusal, unlike
    /// every count in this parser.
    #[test]
    fn a_gap_of_no_milliseconds_is_the_arm_the_others_are_compared_against() {
        let job = Job::parse(["clock", "models/small", "--idle", "0"].map(str::to_string))
            .expect("a gap of nothing parses");
        assert!(
            matches!(job, Job::Measure { idle, .. } if idle.is_zero()),
            "{job:?}"
        );
    }

    /// **The key range is what tells a reader which figure they are holding**,
    /// and the two decode arms open it at different keys: a generation's first
    /// timed step is the one after the prompt's own prefill, and a batched one's
    /// is the one after [`Seated`]'s settling steps as well.
    #[test]
    fn a_decode_run_says_which_keys_its_steps_walked() {
        let ids = vec![0; 34];
        assert_eq!(
            Unit::Step(ids.clone()).over(200),
            "decode steps over 35 to 234 keys"
        );
        assert_eq!(
            Unit::Batch(ids, 32).over(200),
            format!(
                "32-wide decode steps over {} to {} keys a sequence",
                35 + SETTLED,
                234 + SETTLED
            )
        );
        assert_eq!(
            Unit::Prefill(vec![0; 2048]).over(20),
            "prefills of 2048 tokens"
        );
    }

    /// **A period opens where the timed work does.** The batched arm seats its
    /// sequences after the run's clock has started — prefills and settling
    /// steps, over a second of them — and a period that held those would report
    /// the seating as a duty cycle the machine never ran at.
    #[test]
    fn the_first_period_opens_after_the_work_nobody_timed() {
        let mut ticking = Ticking::new(Duration::ZERO, 4);
        let opened = ticking.at;
        let seating = Duration::from_millis(2);
        std::thread::sleep(seating);
        ticking.settled();
        assert!(
            ticking.at.duration_since(opened) >= seating,
            "the seating is inside the first period"
        );
    }

    fn ticks(gpu: &[u64], wall: &[u64]) -> Vec<Tick> {
        gpu.iter()
            .zip(wall)
            .map(|(gpu, wall)| Tick {
                gpu: Duration::from_micros(*gpu),
                wall: Duration::from_micros(*wall),
            })
            .collect()
    }

    fn reading(readings: &[Reading], name: &str) -> f64 {
        readings
            .iter()
            .find(|reading| reading.name == name)
            .unwrap_or_else(|| panic!("{name} is among {:?}", names(readings)))
            .value
    }

    fn names(readings: &[Reading]) -> Vec<&str> {
        readings
            .iter()
            .map(|reading| reading.name.as_str())
            .collect()
    }

    /// A reading against what it should be, to a tolerance far below anything
    /// this measurement reports: the durations go through `f64` seconds and
    /// back, so an exact comparison is asserting the rounding rather than the
    /// arithmetic.
    #[track_caller]
    fn reads(readings: &[Reading], name: &str, want: f64) {
        let got = reading(readings, name);
        assert!(
            (got - want).abs() < 1e-6,
            "{name} reads {got} rather than {want}"
        );
    }

    /// **A run is reported in parts because a drift is a shape**, and the parts
    /// are what separates a part that ramped once from one that fell throughout
    /// — two runs a mean cannot tell apart.
    #[test]
    fn a_clock_run_is_reported_in_parts_and_the_last_against_the_first() {
        // Ten units at 10 ms rising to 12, which is a part-per-two run whose
        // last part is 20% above its first.
        let rising = ticks(
            &[
                10_000, 10_000, 10_500, 10_500, 11_000, 11_000, 11_500, 11_500, 12_000, 12_000,
            ],
            &[20_000; 10],
        );
        let readings = clocked(&rising);
        assert_eq!(
            names(&readings),
            [
                "part1.device",
                "part2.device",
                "part3.device",
                "part4.device",
                "part5.device",
                "clock.device",
                "clock.wall",
                "clock.duty",
                "clock.drift",
            ]
        );
        reads(&readings, "part1.device", 10.0);
        reads(&readings, "part5.device", 12.0);
        reads(&readings, "clock.drift", 20.0);
        reads(&readings, "clock.device", 11.0);
    }

    /// **The duty cycle is the device against the whole period**, which is what
    /// makes a gap deliberately left between two units visible in the figure
    /// rather than only in the flag that asked for it.
    #[test]
    fn the_duty_cycle_is_the_device_against_the_period_the_gap_is_inside() {
        let busy = ticks(&[18_400; 4], &[20_000; 4]);
        let idled = ticks(&[18_400; 4], &[60_000; 4]);
        reads(&clocked(&busy), "clock.duty", 92.0);
        reads(&clocked(&idled), "clock.duty", 100.0 * 18.4 / 60.0);
        // And the device column is what says the two ran at the same clock,
        // which is the whole of what the idle arm is for.
        reads(&clocked(&idled), "clock.device", 18.4);
        reads(&clocked(&busy), "clock.device", 18.4);
    }

    /// **The parts are cut at the proportion**, so a run reports as many of them
    /// as it has units for and the shape a reader compares does not depend on
    /// whether the count happened to divide.
    #[test]
    fn a_run_reports_as_many_parts_as_it_has_units_for() {
        let parts = |units: usize| {
            clocked(&ticks(&vec![1_000; units], &vec![2_000; units]))
                .iter()
                .filter(|reading| reading.name.starts_with("part"))
                .count()
        };
        assert_eq!(parts(1), 1);
        assert_eq!(parts(4), 4);
        assert_eq!(parts(5), PARTS);
        // The counts a fixed chunk width gets wrong: eleven over chunks of three
        // is four parts, and sixteen over chunks of four is four.
        assert_eq!(parts(11), PARTS);
        assert_eq!(parts(16), PARTS);
        assert_eq!(parts(600), PARTS);
    }

    /// Every unit lands in exactly one part, so what the parts are is a cut of
    /// the run rather than a second sample of it — which is what makes a drift
    /// between two of them a statement about the run that was taken.
    #[test]
    fn the_parts_are_a_cut_of_the_run_rather_than_a_sample_of_it() {
        // Eleven units, which no chunk width divides, rising by a tenth of a
        // millisecond each.
        let gpu: Vec<u64> = (0..11).map(|at| 10_000 + at * 100).collect();
        let readings = clocked(&ticks(&gpu, &[20_000; 11]));
        reads(&readings, "clock.device", 10.5);
        // Two units in the first part and three in the last, which is where
        // cutting at the proportion puts them.
        reads(&readings, "part1.device", 10.05);
        reads(&readings, "part5.device", 10.9);
        reads(&readings, "clock.drift", 100.0 * (10.9 / 10.05 - 1.0));
    }

    /// A run of nothing reports zeroes rather than dividing by them, which is
    /// the shape every other reading here answers an empty run with.
    #[test]
    fn a_clock_run_over_no_units_divides_by_nothing() {
        let readings = clocked(&[]);
        reads(&readings, "clock.device", 0.0);
        reads(&readings, "clock.duty", 0.0);
        reads(&readings, "clock.drift", 0.0);
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

    /// **The differential run opens both arms itself**, so `--numerics` — which
    /// every other measurement takes to mean "run under this" — is a mistake
    /// here. What it takes instead is `--against`, which names the *second* arm.
    #[test]
    fn a_differential_run_takes_one_checkpoint_and_no_numerics() {
        assert_eq!(
            Job::parse(["diverge".to_string(), "models/small".to_string()]).expect("parses"),
            Job::Diverge {
                checkpoint: PathBuf::from("models/small"),
                tokens: DIFFERENTIAL,
                against: Numerics::Production,
            }
        );
        for name in Numerics::EVERY.map(Numerics::named) {
            assert!(
                Job::parse(["diverge", "models/small", "--numerics", name].map(str::to_string))
                    .is_err(),
                "{name} was taken by a measurement that opens both arms"
            );
        }
        assert!(Job::parse(["diverge".to_string()]).is_err());
    }

    /// **Every word behind the flag is an arm this can be pointed at, and the
    /// reference is not one.** A differential run of the reference against
    /// itself agrees perfectly, which is what a run that measured nothing looks
    /// like — so it is refused rather than reported.
    #[test]
    fn a_differential_run_is_pointed_at_a_word_behind_the_flag() {
        for against in Numerics::EVERY {
            let parsed = Job::parse(
                ["diverge", "models/small", "--against", against.named()].map(str::to_string),
            );
            match against.compiles_the_entries() {
                true => assert_eq!(
                    parsed.expect("parses"),
                    Job::Diverge {
                        checkpoint: PathBuf::from("models/small"),
                        tokens: DIFFERENTIAL,
                        against,
                    }
                ),
                false => assert!(parsed.is_err(), "{against:?} was taken as a second arm"),
            }
        }
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

    /// The two regimes of a generation, out of the two steps that are them:
    /// the prompt enters the cache on the first step, so step one is a prefill
    /// and every step after it is a decode step.
    #[test]
    fn the_first_step_of_a_generation_is_the_prefill_and_the_rest_are_decode_steps() {
        let mut ran = Dispatched::default();
        ran.step(1, vec![encoded("mma_matmul_rows"), encoded("rms_norm")]);
        assert_eq!(
            ran.decode,
            BTreeSet::new(),
            "a prefill is not a decode step"
        );
        ran.step(2, vec![encoded("packed_matmul"), encoded("rms_norm")]);
        ran.step(3, vec![encoded("packed_matmul_pair")]);
        assert_eq!(ran.prefill, named(&["mma_matmul_rows", "rms_norm"]));
        assert_eq!(
            ran.decode,
            named(&["packed_matmul", "packed_matmul_pair", "rms_norm"]),
        );
    }

    /// **The corpus's six prompts fold into one pair of sets**, because what a
    /// prompt decides is which entries it reaches and not which entries exist:
    /// only the longest member is given the grouped entry, and a run holding a
    /// set per prompt would have to say which of the six the flag was checked
    /// against.
    #[test]
    fn what_every_prompt_reached_folds_into_one_pair_of_sets() {
        let mut ran = Dispatched::default();
        ran.step(1, vec![encoded("mma_matmul_rows")]);
        ran.step(2, vec![encoded("packed_matmul")]);
        ran.step(1, vec![encoded("mma_matmul_grouped")]);
        ran.step(2, vec![encoded("packed_matmul")]);
        assert_eq!(
            ran.prefill,
            named(&["mma_matmul_grouped", "mma_matmul_rows"]),
        );
        assert_eq!(ran.decode, named(&["packed_matmul"]));
    }

    /// **What the flag selected is what one arm ran and the other did not**,
    /// per regime — a kernel both arms dispatch is a kernel the flag does not
    /// reach, whichever of the two is running it.
    #[test]
    fn the_entries_the_flag_selected_are_the_ones_only_the_production_arm_ran() {
        let mut reference = Dispatched::default();
        reference.step(1, vec![encoded("packed_matmul_rows"), encoded("rms_norm")]);
        reference.step(2, vec![encoded("packed_matmul"), encoded("rms_norm")]);
        let mut production = Dispatched::default();
        production.step(1, vec![encoded("mma_matmul_rows"), encoded("rms_norm")]);
        production.step(2, vec![encoded("packed_matmul"), encoded("rms_norm")]);

        let selected = production.beyond(&reference);
        assert_eq!(selected.prefill, named(&["mma_matmul_rows"]));
        assert_eq!(
            selected.decode,
            BTreeSet::new(),
            "a decode step both arms ran the same way selected nothing"
        );
    }

    /// **The same on the axis the floor cannot watch**, which is the whole of
    /// what this milestone put behind the flag: two arms whose prefills are
    /// identical and whose decode steps are not, and an entry selected there and
    /// nowhere else. A run reaching the flag only at a decode step is a run this
    /// has to accept, and one reaching it nowhere is a run it has to refuse —
    /// so the two are the same fixture with one entry moved.
    #[test]
    fn an_entry_only_a_decode_step_ran_is_selected_there_and_reported_there() {
        let mut reference = Dispatched::default();
        reference.step(1, vec![encoded("packed_matmul_rows")]);
        reference.step(2, vec![encoded("packed_matmul"), encoded("rms_norm")]);
        let mut production = Dispatched::default();
        production.step(1, vec![encoded("packed_matmul_rows")]);
        production.step(2, vec![encoded("split_matmul"), encoded("rms_norm")]);

        let selected = production.beyond(&reference);
        assert_eq!(selected.prefill, BTreeSet::new());
        assert_eq!(selected.decode, named(&["split_matmul"]));
        assert_eq!(
            reached(&selected)
                .iter()
                .map(|reading| (reading.name.as_str(), reading.value))
                .collect::<Vec<_>>(),
            vec![("selected.prefill", 0.0), ("selected.decode", 1.0)],
        );
    }

    /// **The check the token-length floor cannot make.** Every entry each kernel
    /// says it put behind the flag has to have been dispatched by something the
    /// corpus ran, or the run's agreement is a check on this harness.
    #[test]
    fn a_run_that_reached_every_entry_behind_the_flag_is_the_one_that_passes() {
        let every: Vec<Encoded> = PackedMatmul::BEHIND_THE_FLAG
            .iter()
            .chain(FusedAttention::BEHIND_THE_FLAG)
            .map(|entry| encoded(entry))
            .collect();
        let mut selected = Dispatched::default();
        selected.step(1, every.clone());
        // The second regime is where a decode-path entry would land, and an
        // empty one is not a failure on its own: what fails is an entry
        // *nothing* reached, wherever it should have been reached.
        selected.step(2, Vec::new());
        assert!(unreached(&selected).is_ok());

        assert_eq!(
            reached(&selected)
                .iter()
                .map(|reading| reading.value)
                .collect::<Vec<_>>(),
            vec![every.len() as f64, 0.0],
        );
    }

    /// And the failure it exists for, one entry at a time: a corpus that reached
    /// everything but one is a corpus that says nothing about that one.
    #[test]
    fn a_run_that_reached_no_entry_behind_the_flag_is_refused() {
        assert!(
            unreached(&Dispatched::default()).is_err(),
            "a run that dispatched nothing the reference did not is not a differential run"
        );
        let behind: Vec<&str> = PackedMatmul::BEHIND_THE_FLAG
            .iter()
            .chain(FusedAttention::BEHIND_THE_FLAG)
            .copied()
            .collect();
        assert!(behind.len() > 1, "one entry cannot leave one unreached");
        for left in &behind {
            let mut selected = Dispatched::default();
            selected.step(
                1,
                behind
                    .iter()
                    .filter(|entry| entry != &left)
                    .map(|entry| encoded(entry))
                    .collect(),
            );
            let refused = unreached(&selected).expect_err("{left} was left unreached");
            assert!(
                format!("{refused}").contains(left),
                "the refusal does not name {left}: {refused}"
            );
        }
    }

    /// A set of entries reads as a line either way round, and an empty one reads
    /// as a word rather than as a blank — which is what a report of "a decode
    /// step ran " would be.
    #[test]
    fn an_empty_set_of_entries_is_named_rather_than_left_blank() {
        assert_eq!(named_entries(&BTreeSet::new()), "nothing");
        assert_eq!(named_entries(&named(&["b", "a"])), "a, b");
    }

    /// One dispatch of `entry`, as the trace would have recorded it. Only the
    /// symbol is read here; the rest is what an `Encoded` has to carry.
    fn encoded(entry: &str) -> Encoded {
        Encoded {
            entry: entry.to_string(),
            symbol: entry.to_string(),
            pipeline: 0,
            slots: Vec::new(),
            threads: 0,
            threads_per_group: 0,
            encoding: Duration::ZERO,
        }
    }

    fn named(entries: &[&str]) -> BTreeSet<String> {
        entries.iter().map(|entry| entry.to_string()).collect()
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
                widest: None,
                idle: Duration::ZERO,
                prefill: 0,
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
