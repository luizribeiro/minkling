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

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use inkling_cli::args::Backend;
use inkling_cli::{backend, config, session};
use inkling_core::generate::{
    Alone, BatchProposer, Generator, Picked, Proposer, Round, Seated as SeatedRound,
};
use inkling_core::head::Tail;
use inkling_core::model::Batched;
use inkling_core::mtp::{CheckpointHeads, MtpProposer};
use inkling_core::workload::{
    BEST, CORPUS, DECODED, DIFFERENTIAL, REALISTIC, STRUCTURED_PROMPT, SWEPT, Session, tiled,
    turned,
};
use inkling_core::{
    Checkpoint, CheckpointWeights, Continuous, DEFAULT_BOUND, Ending, Kept, ModelCache, Request,
    TextConfig, Tokenizer, profile,
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
    bench batch   <checkpoint> [--tokens <n>] [--context <n>] [--batch <n>] [--depth <k>]\n  \
    bench clock   <checkpoint> [--tokens <n>] [--context <n>] [--idle <ms>] \
[--every <ms>] [--burst <n>] [--keep-warm <ms>] [--prefill <n>] [--batch <n>]\n  \
    bench joining <checkpoint> [--tokens <n>] [--context <n>] [--batch <n>] \
[--admit <n>]\n  \
    bench fleet   <checkpoint> [--tokens <n>] [--context <n>] [--batch <n>] \
[--admit <n>] [--agents <n>] [--every <ms>]\n  \
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
    /// What a request waits for its first token when a batch is already
    /// running, against the same request at an idle engine and against the same
    /// request waiting for the batch to drain.
    ///
    /// **The one measurement here whose subject is a latency rather than a
    /// rate.** Every other row in this file is what some quantity of work costs;
    /// this is what one request waits, which is what a fleet of agents feels and
    /// is the number a static batch has no way to improve — a slot that fills
    /// sooner earns the same throughput sooner, and earns a wait that is not the
    /// batch's remaining budget.
    Joining,
    /// A fleet of agents asking at irregular intervals, through one engine.
    ///
    /// **The one measurement here that reports a distribution.** A mean over a
    /// fleet's requests describes none of them: what a scheduler does to the
    /// request that arrived at the wrong moment is the whole of the difference
    /// between admitting continuously and admitting in batches, and only the
    /// tail says it.
    Fleet,
    /// Several conversations, each taking several turns, through one engine.
    ///
    /// **The one measurement here whose subject is what happens between two
    /// requests at width greater than one**, which is the join of the two
    /// measurements either side of it and was not a workload this engine could
    /// run at all. [`What::Session`] is one conversation through one slot;
    /// [`What::Fleet`] is many requests that are each a single turn. A fleet of
    /// coding agents is neither: it is several conversations, each coming back
    /// with the last turn and a little more, and what a slot has already
    /// prefilled for one of them is the whole of what it is worth.
    ///
    /// `--agents 1` is a session at width N and is what K1's width-one figures
    /// are read against; `--agents k` is the fleet nobody had measured.
    Conversations,
}

/// How a request reaches a slot, which is the one thing the two arms of the two
/// measurements above differ in.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Admitting {
    /// Into a free slot as soon as there is one, which is what
    /// [`Continuous`](inkling_core::Continuous) does.
    Continuously,
    /// Only into an engine that has drained: up to `slots` requests admitted
    /// together, decoded together, and the next of them admitted when the last
    /// of the batch finishes.
    ///
    /// **This is `generate_batch`'s admission rule and not its prefill.** There
    /// the prompts enter one sequence at a time and no sequence decodes until
    /// all of them are in; here a batch's prompts fill together, in the same
    /// calls, because that is what this engine does with several sequences
    /// admitted at once. What the difference reaches is the prefill phase of a
    /// batch and nothing after it — one call's worth of shape per batch, against
    /// a wait this measures in hundreds of decode steps — so it is named here
    /// rather than corrected, and the README says what it is worth.
    InBatches,
}

impl Admitting {
    fn named(self) -> &'static str {
        match self {
            Self::Continuously => "joining",
            Self::InBatches => "draining",
        }
    }
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
        /// What is left between two bursts of work and how many units are in
        /// one, which only the clock measurement has anywhere to put.
        shape: Shape,
        /// The prompt each of the clock measurement's units prefills, and zero
        /// for the decode step it charges otherwise.
        prefill: usize,
        /// Prompt rows one step carries, over every request filling in it.
        ///
        /// **The knob the two latency measurements trade on.** A prompt fed
        /// whole is one call the sequences already in flight wait the whole of;
        /// spread over this many rows a step it takes as many steps, and what
        /// any one of them costs those sequences is bounded by this.
        admit: usize,
        /// How many requests a fleet makes, and nothing at all for the
        /// measurements that make one.
        agents: usize,
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
            Some("joining") => Some(What::Joining),
            Some("fleet") => Some(What::Fleet),
            Some("conversations") => Some(What::Conversations),
            Some("alternate") => return Self::alternating(args),
            Some("diverge") => return Self::diverging(args),
            Some("guesses") => None,
            Some(word) => bail!(
                "{word} is not one of decode, prefill, sweep, engines, session, batch, \
                 clock, joining, fleet, conversations, guesses, diverge or alternate"
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
        // The three the clock measurement takes and nothing else has anywhere to
        // put, `Option` for the reason the numbers above are: zero is a duty
        // cycle this means something by, and so is a prompt of no tokens.
        let mut idle = None;
        let mut every = None;
        let mut burst = None;
        let mut warm = None;
        let mut prefill = None;
        // The two the latency measurements take, `Option` for the reason above:
        // what the refusals below are about is whether the number was given.
        let mut admit = None;
        let mut agents = None;
        // A sweep runs every depth up to its own, where a cross-engine table
        // quotes one beside `k = 0` — so the default depth is what the flag
        // means to the measurement asking for it.
        let mut depth = match what {
            Some(What::Engines) => BEST,
            // **A batch sweep's default depth is none, and that is a number
            // rather than an absence**: `k = 0` is the column every other one
            // is read against, and it is the column `bench batch` itself
            // reports. See [`What::Batch`] in [`measure`].
            Some(What::Batch) => 0,
            _ => SWEPT,
        };
        let mut asked_depth = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokens" | "-n" => tokens = Some(count(&arg, &mut args)?),
                "--context" | "-c" => context = Some(count(&arg, &mut args)?),
                "--depth" | "-k" => {
                    depth = count(&arg, &mut args)?;
                    asked_depth = true;
                }
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
                // An interval of no milliseconds is not a request rate, which is
                // what separates this from the gap above: the arm a rate is
                // compared against is a gap of nothing, and it already has a
                // word.
                "--every" => {
                    every = Some(Duration::from_millis(count(&arg, &mut args)? as u64));
                }
                "--burst" => burst = Some(count(&arg, &mut args)?),
                // A keep-warm every no milliseconds is a busy loop rather than
                // a dispatch into a gap, and the arm it would be compared
                // against is a run with no gap at all.
                "--keep-warm" => {
                    warm = Some(Duration::from_millis(count(&arg, &mut args)? as u64));
                }
                "--prefill" => prefill = Some(count(&arg, &mut args)?),
                // A step carrying no prompt rows never enters a prompt into a
                // cache at all, which is a joining request nobody answers.
                "--admit" => admit = Some(count(&arg, &mut args)?),
                "--agents" => agents = Some(count(&arg, &mut args)?),
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
        //
        // The two latency measurements take one for a different reason and it is
        // worth saying which: their context is the *joining prompt's* length,
        // which is the whole of what a time to first token is a function of.
        if !matches!(
            what,
            What::Decode | What::Batch | What::Clock | What::Joining | What::Fleet
        ) && context.is_some()
        {
            bail!("{what:?} takes no --context: only a decode step has one to be taken at");
        }
        // **Only the clock measurement has a duty cycle it is asking about.**
        // Every other row here is taken at whatever occupancy its own work makes
        // and says so nowhere, which is the whole of what this measurement
        // exists to fix — so the two levers are refused to the measurements that
        // could only drop them, under the same rule as the numbers above.
        //
        // **A fleet takes `--every` and none of the rest**, and it means by it
        // what the clock measurement means: a rate rather than a gap. What
        // arrives every so often there is a unit of work and here it is a
        // request, which is the same sentence about a server said at the two
        // ends of it.
        let rated = matches!(what, What::Clock | What::Fleet);
        if (what != What::Clock
            && (idle.is_some() || burst.is_some() || warm.is_some() || prefill.is_some()))
            || (!rated && every.is_some())
        {
            bail!(
                "{what:?} takes no --idle, --every, --burst or --prefill: it does not vary its \
                 own duty cycle"
            );
        }
        // **A gap and an interval are two answers to one question**, and the
        // second is not the first plus arithmetic a reader can do in their head:
        // what a run told both would have to decide is whether the interval
        // holds the gap or follows it.
        if idle.is_some() && every.is_some() {
            bail!("--idle and --every are two answers to one question: a gap, or a rate to fit it");
        }
        // **A burst longer than the run is a run with no gap in it.** The gap
        // falls behind every `burst`-th unit, so a run of four units in bursts
        // of eight leaves none at all — and would report the duty cycle of the
        // back-to-back arm under a header announcing the gap it was asked for.
        if burst.is_some_and(|burst| burst > tokens.unwrap_or(TICKED)) {
            bail!(
                "a burst of {} does not fit in {} units: the run would leave no gap between two \
                 of them",
                burst.unwrap_or_default(),
                tokens.unwrap_or(TICKED)
            );
        }
        // **A gap of nothing has nowhere to put a dispatch.** The lever exists
        // to fill an idle device, so a run that asked for one without asking
        // for a gap is asking for a busy loop between two units that are
        // already back to back.
        if warm.is_some() && every.is_none() && idle.is_none_or(|idle| idle.is_zero()) {
            bail!("--keep-warm needs a gap to dispatch into: give --idle or --every");
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
        if !matches!(
            what,
            What::Batch | What::Clock | What::Joining | What::Fleet | What::Conversations
        ) && widest.is_some()
        {
            bail!("{what:?} takes no --batch: it decodes one sequence");
        }
        // **Only the two latency measurements have a request that joins.**
        // Every other one here prefills before it decodes and never puts a
        // prompt row in a call with a decode row in it, so a chunk given to one
        // could only be dropped — the same rule the numbers above are refused
        // under. `--agents` is a fleet's alone for the same reason: everything
        // else makes one request.
        if !matches!(what, What::Joining | What::Fleet | What::Conversations) && admit.is_some() {
            bail!("{what:?} takes no --admit: nothing joins a batch it is running");
        }
        if !matches!(what, What::Fleet | What::Conversations) && agents.is_some() {
            bail!("{what:?} takes no --agents: it makes one request");
        }
        // **An engine of no slots seats nobody**, refused here rather than in
        // the panic one layer down, for the reason a fleet's own width is.
        if what == What::Conversations && widest == Some(0) {
            bail!(
                "conversations takes a --batch of at least one: an engine with no slots seats \
                 nobody"
            );
        }
        // **A slot free for the joiner is what a joining measurement needs.** An
        // engine of one slot has none, so the request would be measuring a queue
        // rather than a batch it joined — and the arm that measures the queue is
        // the one this already runs beside it.
        if what == What::Joining && widest.is_some_and(|slots| slots < 2) {
            bail!("joining takes a --batch of at least two: one of the slots is the joiner's");
        }
        // An engine of no slots is refused below this line rather than here, and
        // a measurement should say what it takes rather than panic inside the
        // thing it is measuring.
        if what == What::Fleet && widest == Some(0) {
            bail!("fleet takes a --batch of at least one: an engine with no slots seats nobody");
        }
        // **A batch that finishes inside its own settling is not a batch this
        // can take a step of.** The two latency measurements settle the
        // sequences already in flight over [`SETTLED`] steps and quote what one
        // of those cost, so a budget that runs out inside them would have the
        // row named "the batch's step" be a mean over calls to an empty engine.
        if matches!(what, What::Joining | What::Fleet)
            && tokens.is_some_and(|tokens| tokens <= SETTLED)
        {
            bail!(
                "{what:?} takes more than {SETTLED} tokens a request: the sequences it holds in \
                 flight are settled over {SETTLED} steps"
            );
        }
        // A prompt entering no rows a step never enters a cache at all, which is
        // a request nothing here would ever answer.
        if admit == Some(0) {
            bail!("--admit 0 is a prompt that never enters a cache");
        }
        // **A prefill already fills the machine**, which is why the batching
        // milestone batched the other regime — so a width given to the arm whose
        // unit is a prompt would be a width nothing could do anything with.
        if prefill.is_some() && widest.is_some() {
            bail!("a clock run over prefills takes no --batch: a prefill fills the machine alone");
        }
        // **Only the three measurements that speculate take a depth.** A batch
        // sweep runs the chain at the depth it is given, a sweep walks up to it
        // and a cross-engine table quotes one beside `k = 0`; everything else
        // here decodes a token at a time, so a depth handed to it could only be
        // dropped — the same rule the numbers above are refused under.
        //
        // Asked of the flag rather than of the value, because the default is a
        // number too and a run that gave one explicitly meant it.
        if !matches!(what, What::Batch | What::Sweep | What::Engines) && asked_depth {
            bail!("{what:?} takes no --depth: it decodes a token at a time");
        }
        // **Only the two measurements with a between-requests keep anything
        // across one.** Every other measurement here is one call or a series of
        // them against caches of its own, so a number of positions to keep could
        // only be dropped — the same rule the two above are refused under.
        if !matches!(what, What::Session | What::Conversations) && reuse.is_some() {
            bail!("{what:?} takes no --reuse-tokens: it makes one request");
        }
        Ok(Self::Measure {
            what,
            checkpoint,
            tokens: tokens.unwrap_or(match what {
                What::Prefill => PREFILLED,
                What::Session | What::Conversations => Session::OPENING,
                What::Clock => TICKED,
                What::Fleet | What::Joining => ASKED,
                _ => DECODED,
            }),
            context: context.unwrap_or(0),
            depth,
            numerics: numerics.unwrap_or_default(),
            reuse: reuse.unwrap_or(DEFAULT_BOUND),
            widest,
            shape: Shape {
                gap: match every {
                    Some(period) => Gap::Every(period),
                    None => Gap::After(idle.unwrap_or_default()),
                },
                burst: burst.unwrap_or(1),
                warm,
            },
            prefill: prefill.unwrap_or(0),
            admit: admit.unwrap_or(ADMITTED),
            agents: agents.unwrap_or(match what {
                What::Conversations => CONVERSATIONS,
                _ => AGENTS,
            }),
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
                shape,
                prefill,
                admit,
                agents,
            } => {
                let asked = Asked {
                    tokens: *tokens,
                    context: *context,
                    depth: *depth,
                    numerics: *numerics,
                    reuse: *reuse,
                    widest: *widest,
                    shape: *shape,
                    prefill: *prefill,
                    admit: *admit,
                    agents: *agents,
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
    /// The shape the clock measurement puts its units in, ignored by everything
    /// else.
    shape: Shape,
    /// The prompt each of the clock measurement's units prefills, and zero for
    /// the decode step it charges otherwise.
    prefill: usize,
    /// Prompt rows a joining request feeds in one step, which only the two
    /// latency measurements have anywhere to put.
    admit: usize,
    /// How many requests a fleet makes.
    agents: usize,
}

fn measure(what: What, dir: &Path, asked: Asked) -> Result<Vec<Reading>> {
    let Asked {
        tokens,
        context,
        depth,
        numerics,
        reuse,
        widest,
        shape,
        prefill,
        admit,
        agents,
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
    // a batch sweep the one it was asked for, and everything else nothing but
    // zero.
    //
    // **The binding below is named for what it is rather than shadowing the
    // flag**, because the two are not the same number at every measurement and
    // a row that read the wrong one would be a table of a depth nobody asked
    // for — which is exactly what a batch sweep did until the missing row said
    // so. See "The one the review could not have found" for the first time this
    // file paid for a number being a number of the wrong thing.
    let depths: Vec<usize> = match what {
        What::Sweep => (0..=depth).collect(),
        What::Engines if depth > 0 => vec![0, depth],
        What::Batch => vec![depth],
        _ => vec![0],
    };
    // Thrown away, because the first generation of a process faults in the
    // pages the rest of them read — 4.2 GiB of it once heads are mapped — and
    // that belongs to a run's first token rather than to whichever arm ran
    // first.
    let warm = prompt[..prompt.len().min(8)].to_vec();

    let mut taken = Vec::new();
    let mut unspeculated = None;
    for wrapped_at in depths {
        // Wrapped at the depth being measured rather than at the deepest one:
        // the windows a rejected token is taken back out of are wider by the
        // depth, so this is the configuration a run of that depth actually has.
        let weights = backend::weights(gpu.as_ref(), &ckpt, text, wrapped_at, 1)?;
        let tail = backend::tail_weights(&weights, text);
        let heads = backend::heads(gpu.as_ref(), &ckpt, &config, wrapped_at, &tail, 1)?;
        let timed = |ids: &[usize], budget| {
            generate(&weights, heads.as_ref(), text, ids, budget, wrapped_at)
        };

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
                        let pair = format!("{prompted}x{generated}.k{wrapped_at}");
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
                            match wrapped_at {
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
                // **Recording for the whole sweep rather than for the cell that
                // needs it**, because which cell that is is what the sweep is
                // being run to find out. What it costs is a record a submission
                // — some tens of thousands over a sitting — and the timestamps
                // behind it are read off buffers that have already completed.
                let recording = gpu.as_ref().map(backend::Gpu::device);
                if let Some(device) = recording {
                    device.record_round_trips(true);
                }
                for slots in widths(widest.unwrap_or(WIDEST)) {
                    let weights = backend::weights(gpu.as_ref(), &ckpt, text, wrapped_at, slots)?;
                    let ids = behind_a_step(&prompt, context);
                    // **The batch's own row, at every width and at every
                    // depth.** A step is what the batch waits; a token is what
                    // one sequence in it waits, which is the step divided by
                    // the sequences that took a token out of it.
                    // **Settled before either arm is timed.** Whichever ran
                    // first at a width would otherwise pay for the device
                    // coming up from idle, and that is a bias in one direction
                    // rather than noise — which is the finding C3's occupancy
                    // loop had to correct, arriving here on a different axis.
                    speculated(
                        &weights,
                        None,
                        text,
                        &ids,
                        Speculating {
                            slots,
                            depth: 0,
                            tokens: SETTLING,
                            device: None,
                        },
                    );
                    let run = batched(&weights, text, &ids, slots, tokens);
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
                        "batch {slots}: step {:.2?}, token {:.2?}, device {:.2?}, duty {:.1}%, \
                         {:.1} tok/s",
                        run.step,
                        run.step / slots as u32,
                        run.gpu,
                        duty(run.gpu, run.step),
                        slots as f64 / run.step.as_secs_f64(),
                    );

                    // **And the same width through the speculative loop, at
                    // `k = 0` as well as at the depth asked for.** The `k = 0`
                    // arm is the same work through a different code path, so
                    // its rate is a figure this measurement does not get to
                    // choose: the two rows above and below have to agree, and a
                    // milestone that only printed the speculating one would
                    // have nothing to notice a wrong denominator with.
                    let tail = backend::tail_weights(&weights, text);
                    let heads =
                        backend::heads(gpu.as_ref(), &ckpt, &config, wrapped_at, &tail, slots)?;
                    for k in 0..=wrapped_at {
                        let held = (k > 0).then_some(heads.as_ref()).flatten();
                        let run = speculated(
                            &weights,
                            held,
                            text,
                            &ids,
                            Speculating {
                                slots,
                                depth: k,
                                tokens,
                                device: recording,
                            },
                        );
                        let rate = run.tokens as f64 / run.wall.as_secs_f64();
                        let name = format!("batch{slots}.k{k}");
                        taken.push(Reading::new(format!("{name}.rate"), rate, "tok/s"));
                        taken.push(Reading::new(
                            format!("{name}.token"),
                            millis(run.wall) / run.tokens.max(1) as f64,
                            "ms",
                        ));
                        taken.push(Reading::new(
                            format!("{name}.duty"),
                            duty(run.gpu, run.wall),
                            "%",
                        ));
                        taken.push(Reading::new(
                            format!("{name}.submissions"),
                            run.submissions,
                            "/round",
                        ));
                        taken.push(Reading::new(
                            format!("{name}.round"),
                            run.tokens as f64 / run.rounds as f64,
                            "/round",
                        ));
                        // Per round rather than over the run, which is what
                        // makes it the same reading as the row above: at
                        // `k = 0` a round is a step.
                        taken.push(Reading::new(
                            format!("{name}.device"),
                            millis(run.gpu) / run.rounds as f64,
                            "ms",
                        ));
                        // **The wait divided, which is the column a duty cycle
                        // is the ratio of and cannot break up.** `executed` is
                        // the row above accumulated a second way, so the two
                        // are a check on each other; the rest are where a wall
                        // that is neither encode nor execution went.
                        for (row, taken_by) in [
                            ("scheduled", run.divided.scheduled),
                            ("queued", run.divided.queued),
                            ("executed", run.divided.executed),
                            ("idle", run.divided.idle),
                            ("unattributed", run.divided.unattributed),
                        ] {
                            taken.push(Reading::new(
                                format!("{name}.{row}"),
                                millis(taken_by),
                                "ms",
                            ));
                        }
                        taken.push(Reading::new(
                            format!("{name}.allocations"),
                            run.allocations,
                            "/round",
                        ));
                        for (at, rate) in run.rates.iter().enumerate() {
                            taken.push(Reading::new(
                                format!("{name}.accept{}", at + 1),
                                100.0 * rate,
                                "%",
                            ));
                        }
                        eprintln!(
                            "batch {slots} k{k}: {:.1} tok/s, token {:.2?}, device {:.2?}, \
                             duty {:.1}%, {:.1} submissions a round — waited {:.2?}, encoded \
                             {:.2?} — {} tokens over {} rounds ({:.2}/round), {} ragged, \
                             acceptance {}",
                            rate,
                            run.wall / run.tokens.max(1) as u32,
                            run.gpu / run.rounds as u32,
                            duty(run.gpu, run.wall),
                            run.submissions,
                            run.waited,
                            run.encode,
                            run.tokens,
                            run.rounds,
                            run.tokens as f64 / run.rounds as f64,
                            run.ragged,
                            run.rates
                                .iter()
                                .map(|rate| format!("{:.0}%", 100.0 * rate))
                                .collect::<Vec<_>>()
                                .join(" "),
                        );
                        eprintln!(
                            "    sampled here {:.1}x a round; {}",
                            run.rows
                                .iter()
                                .find(|(op, _, _)| matches!(op, profile::Op::Sample))
                                .map_or(0, |(_, calls, _)| *calls),
                            run.rows
                                .iter()
                                .take(6)
                                .map(|(op, calls, took)| format!(
                                    "{} {calls}x {took:.2?}",
                                    op.name()
                                ))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                        eprintln!(
                            "    a round's submissions: executed {:.2?} + idle {:.2?} tile it; \
                             scheduled {:.2?} and queued {:.2?} a buffer's own life, \
                             unattributed {:.2?}, over {:.1} buffers allocated",
                            run.divided.executed,
                            run.divided.idle,
                            run.divided.scheduled,
                            run.divided.queued,
                            run.divided.unattributed,
                            run.allocations,
                        );
                    }
                }
                if let Some(device) = recording {
                    device.record_round_trips(false);
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
                    // A batch of one is the wrap this loop already made, and a
                    // second one is a second binding over the same mapped pages
                    // for nothing.
                    None | Some(1) => None,
                    Some(slots) => Some(backend::weights(gpu.as_ref(), &ckpt, text, 0, slots)?),
                };
                // One row of the model's own width, which is the smallest
                // dispatch this crate has a kernel for.
                let warm = match (shape.warm, gpu.as_ref()) {
                    (Some(_), Some(gpu)) => Some(gpu.keep_warm(text.hidden_size)?),
                    _ => None,
                };
                let dispatch = warm
                    .as_ref()
                    .map(|warm| move || warm.dispatch().expect("a keep-warm dispatch runs"));
                let run = ticked(
                    wrapped.as_ref().unwrap_or(&weights),
                    text,
                    &unit,
                    tokens,
                    shape,
                    dispatch.as_ref().map(|warm| warm as &dyn Fn()),
                );
                let ticks = &run.ticks;
                eprintln!(
                    "clock: {} {}, {}",
                    ticks.len(),
                    unit.over(ticks.len()),
                    shape.said(),
                );
                taken.extend(clocked(ticks, shape.burst));
                taken.extend(after_a_gap(ticks, shape.burst));
                // **What the lever cost, beside what it bought.** A keep-warm
                // is device time spent on nothing, and a run that reported only
                // what it saved would be quoting one side of a trade.
                if shape.warm.is_some() {
                    eprintln!(
                        "  kept warm: {} dispatches, {:.4?} of device between them",
                        run.dispatched, run.kept,
                    );
                    taken.push(Reading::new("warm.device", millis(run.kept), "ms"));
                    taken.push(Reading::new(
                        "warm.dispatches",
                        run.dispatched as f64,
                        "count",
                    ));
                }
            }
            // **The wait a request makes, which is the number nobody had.**
            // Every row above says what some quantity of work costs; this says
            // what one request waits for its first token with a batch already
            // running, and the arm beside it is the same request waiting for
            // that batch to drain — which is what static batching makes it do.
            What::Joining => {
                let ids = tiled(&prompt, context.max(prompt.len()));
                let slots = widest.unwrap_or(WIDEST);
                let weights = backend::weights(gpu.as_ref(), &ckpt, text, 0, slots)?;
                eprintln!(
                    "joining: {slots} slots, a {}-token prompt over {admit} prompt rows a \
                     step, {tokens} tokens a request",
                    ids.len(),
                );
                let engine = Engine {
                    weights: &weights,
                    config: text,
                    prompt: &ids,
                    slots,
                    admit,
                    tokens,
                    reuse: 0,
                };
                // **[`WARM`] runs thrown away.** A wall this reports is what a
                // request waits, so it is the one figure here that cannot be
                // read off the device's own clock — and the host side of a
                // fresh wrap takes three runs to settle.
                for _ in 0..WARM {
                    joined(&engine, 0, Admitting::Continuously);
                }
                for held in occupancies(slots) {
                    // **And one throwaway at every occupancy.** The loop below
                    // runs `Continuously` first at each of them, so without this
                    // the first call of every "joiner beside `held` decoders"
                    // shape lands inside a timed run — and inside the same one
                    // every time, which is a bias in one direction rather than
                    // noise.
                    joined(&engine, held, Admitting::Continuously);
                    for policy in [Admitting::Continuously, Admitting::InBatches] {
                        // At an idle engine the two policies are the same run,
                        // and a second one of it would be a row saying the same
                        // thing under a different name.
                        if held == 0 && policy == Admitting::InBatches {
                            continue;
                        }
                        let run = joined(&engine, held, policy);
                        let row = format!("{}{held}", policy.named());
                        eprintln!(
                            "  {held} decoding, {}: first token in {:.0} ms over {} steps, \
                             {:.2?} of device at {:.1}% duty{}",
                            policy.named(),
                            millis(run.ttft),
                            run.steps,
                            run.gpu,
                            duty(run.gpu, run.ttft),
                            match (run.settled, run.mixed) {
                                (Some(settled), Some(mixed)) => format!(
                                    "; the batch's step {:.2?} against {:.2?} with the prompt \
                                     riding in it",
                                    settled, mixed
                                ),
                                _ => String::new(),
                            },
                        );
                        taken.push(Reading::new(format!("{row}.ttft"), millis(run.ttft), "ms"));
                        taken.push(Reading::new(format!("{row}.steps"), run.steps as f64, "n"));
                        taken.push(Reading::new(format!("{row}.device"), millis(run.gpu), "ms"));
                        // **The duty cycle beside the figure**, which is the
                        // rule every table in this file is read under: what a
                        // wall says depends on what the part was clocked at
                        // while it ran, and the ratio is what says so.
                        taken.push(Reading::new(
                            format!("{row}.duty"),
                            duty(run.gpu, run.ttft),
                            "%",
                        ));
                        // **What the joining prompt costs the sequences already
                        // in flight**, which is the other half of the trade and
                        // is a row rather than a sentence.
                        if let (Some(settled), Some(mixed)) = (run.settled, run.mixed) {
                            taken.push(Reading::new(format!("{row}.step"), millis(settled), "ms"));
                            taken.push(Reading::new(format!("{row}.riding"), millis(mixed), "ms"));
                        }
                    }
                }
            }
            // **The fleet shape, and the distribution rather than the mean.**
            // What a scheduler does to the request that arrived at the wrong
            // moment is the whole of what separates the two policies, and a mean
            // over the fleet describes none of them.
            What::Fleet => {
                let ids = tiled(&prompt, context.max(prompt.len()));
                let slots = widest.unwrap_or(WIDEST);
                let weights = backend::weights(gpu.as_ref(), &ckpt, text, 0, slots)?;
                let every = shape.gap.every().unwrap_or(ARRIVAL);
                let arrivals: Vec<Duration> = (0..agents).map(|at| every * at as u32).collect();
                let engine = Engine {
                    weights: &weights,
                    config: text,
                    prompt: &ids,
                    slots,
                    admit,
                    tokens,
                    reuse: 0,
                };
                let asked: usize = (0..agents).map(|at| engine.asking_at(at).count).sum();
                eprintln!(
                    "fleet: {agents} requests one every {:.0?}, {asked} tokens between them \
                     around a budget of {tokens}, {slots} slots, {}-token prompts over {admit} \
                     prompt rows a step",
                    every,
                    ids.len(),
                );
                // Thrown away, for the reason the joining arm's are.
                for _ in 0..WARM {
                    joined(&engine, 0, Admitting::Continuously);
                }
                for policy in [Admitting::Continuously, Admitting::InBatches] {
                    let run = fleeted(&engine, &arrivals, policy);
                    assert_eq!(
                        run.tokens, asked,
                        "every request of the fleet answered in full"
                    );
                    let name = policy.named();
                    for (what, said, waits) in [
                        ("first", "to the first token", run.waits(|felt| felt.first)),
                        ("whole", "to the whole answer", run.waits(|felt| felt.last)),
                    ] {
                        eprintln!(
                            "  {name}, {said}: p50 {:.0} ms, p90 {:.0} ms, worst {:.0} ms",
                            millis(percentile(&waits, 0.5)),
                            millis(percentile(&waits, 0.9)),
                            millis(percentile(&waits, 1.0)),
                        );
                        for (at, q) in [("p50", 0.5), ("p90", 0.9), ("worst", 1.0)] {
                            taken.push(Reading::new(
                                format!("{name}.{what}.{at}"),
                                millis(percentile(&waits, q)),
                                "ms",
                            ));
                        }
                    }
                    eprintln!(
                        "  {name}: {:.1} tok/s over {:.1?}, {} steps carrying {} rows, \
                         {:.1}% duty",
                        run.rate(),
                        run.wall,
                        run.steps,
                        run.rows,
                        run.duty(),
                    );
                    taken.push(Reading::new(format!("{name}.rate"), run.rate(), "tok/s"));
                    taken.push(Reading::new(format!("{name}.wall"), millis(run.wall), "ms"));
                    taken.push(Reading::new(format!("{name}.steps"), run.steps as f64, "n"));
                    taken.push(Reading::new(format!("{name}.rows"), run.rows as f64, "n"));
                    taken.push(Reading::new(format!("{name}.duty"), run.duty(), "%"));
                }
            }
            // **The join of the two measurements either side of this one**: a
            // conversation coming back turn after turn, several of them at once,
            // through the engine that seats several. Neither figure existed —
            // `session` keeps a conversation through one slot and `fleet` runs
            // many requests that are each one turn.
            //
            // Every turn is a row, because the shape is the finding: turn one is
            // cold in both arms and every turn after it is where they part.
            What::Conversations => {
                let plan = Session::new(tokens);
                let slots = widest.unwrap_or(agents);
                let weights = backend::weights(gpu.as_ref(), &ckpt, text, 0, slots)?;
                let engine = Engine {
                    weights: &weights,
                    config: text,
                    prompt: &prompt,
                    slots,
                    admit,
                    tokens,
                    reuse,
                };
                let run = talking(&engine, plan, agents);
                eprintln!(
                    "conversations: {agents} conversations of {} turns, {slots} slots, opening \
                     {tokens}, keeping {reuse}, {admit} prompt rows a step",
                    plan.turns,
                );
                for at in 0..run.turns() {
                    let (wall, first) = (
                        run.waits(at, |had| had.last),
                        run.waits(at, |had| had.first),
                    );
                    eprintln!(
                        "  turn {at}: {} tokens prefilled over {agents}, p50 {:.2} s wall, \
                         worst {:.2} s, p50 {:.2} s to first",
                        run.prefilled(at),
                        percentile(&wall, 0.5).as_secs_f64(),
                        percentile(&wall, 1.0).as_secs_f64(),
                        percentile(&first, 0.5).as_secs_f64(),
                    );
                    taken.push(Reading::new(
                        format!("turn{at}.wall"),
                        millis(percentile(&wall, 0.5)),
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("turn{at}.worst"),
                        millis(percentile(&wall, 1.0)),
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("turn{at}.first"),
                        millis(percentile(&first, 0.5)),
                        "ms",
                    ));
                    taken.push(Reading::new(
                        format!("turn{at}.prefilled"),
                        run.prefilled(at) as f64,
                        "tokens",
                    ));
                }
                eprintln!(
                    "  whole: {:.1?}, {:.1} tok/s, {} steps carrying {} rows, {:.1}% duty, \
                     {:.2} ms a turn of bookkeeping",
                    run.wall,
                    run.rate(),
                    run.steps,
                    run.rows,
                    run.duty(),
                    millis(run.bookkeeping) / (run.turns() * agents).max(1) as f64,
                );
                // The figure a fleet of users feels, which is the one nobody
                // here has ever produced: every conversation, end to end.
                taken.push(Reading::new("whole", millis(run.wall), "ms"));
                taken.push(Reading::new("device", millis(run.gpu), "ms"));
                taken.push(Reading::new("duty", run.duty(), "%"));
                taken.push(Reading::new("rate", run.rate(), "tok/s"));
                // Per turn rather than for the run, because it is a per-request
                // cost and the number a server is judged on is what one request
                // pays — the same reading `session` takes at width one.
                taken.push(Reading::new(
                    "bookkeeping",
                    millis(run.bookkeeping) / (run.turns() * agents).max(1) as f64,
                    "ms",
                ));
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
                taken.push(Reading::new(format!("k{wrapped_at}"), millis(step), "ms"));
                taken.push(Reading::new(
                    format!("k{wrapped_at}.device"),
                    millis(run.gpu),
                    "ms",
                ));
                // Against this run's own `k = 0` and not against another
                // sitting's: a sweep whose speedup row is divided by a figure
                // taken an hour earlier carries the drift between the two.
                taken.push(Reading::new(
                    format!("k{wrapped_at}.speedup"),
                    unspeculated.as_secs_f64() / step.as_secs_f64(),
                    "x",
                ));
                taken.push(Reading::new(
                    format!("k{wrapped_at}.tokens"),
                    run.per_round(),
                    "/round",
                ));
                for (at, rate) in run.rates.iter().enumerate() {
                    taken.push(Reading::new(
                        format!("k{wrapped_at}.accept{}", at + 1),
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

/// How many sequences are already decoding when a request arrives, at each
/// occupancy a `slots`-wide engine has room to leave a slot free at.
///
/// Zero first, which is the idle engine and is the arm every other row is read
/// against; then doubling, for [`widths`]'s reason — what the rows say is where
/// the wait stops halving, and a row where it stopped is the answer. The
/// fullest is `slots - 1`, because a request arriving at an engine with no free
/// slot is a queueing measurement rather than a joining one.
fn occupancies(slots: usize) -> Vec<usize> {
    let mut held = vec![0];
    if slots > 1 {
        held.extend(widths(slots - 1));
    }
    held
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

/// What a batch of speculating sequences came to, over the rounds after its
/// prompts were in the caches.
struct Speculated {
    /// The wall from the first round to the last.
    wall: Duration,
    gpu: Duration,
    /// Tokens the run produced inside that wall, which is every token but the
    /// one each prompt's own pass produced.
    tokens: usize,
    /// Rounds the whole batch took, which is what a token divides by to say
    /// what a round banked.
    rounds: usize,
    /// Acceptance per depth, over every sequence together.
    rates: Vec<f64>,
    /// Rounds whose seats were not all the same length, which is what says the
    /// ragged path was driven at all.
    ragged: usize,
    /// Submissions a round made, and the wall this process spent inside them
    /// against the encode ahead of them.
    ///
    /// **What a duty column cannot say on its own is which side of a round is
    /// waiting**, and a chain's rounds have two kinds of submission in them:
    /// the verify's run of layers, and one a head. So the three are here beside
    /// the device's own clock.
    submissions: f64,
    encode: Duration,
    waited: Duration,
    /// The same wait divided into the things a submission does, per round.
    divided: Divided,
    /// Buffers a round asked the device for.
    ///
    /// **The one column that separates a driver charging for work from a driver
    /// charging for memory.** Every buffer a command buffer names has to be
    /// resident before the GPU can start on it, and a round that allocates is a
    /// round handing the driver something it has never made resident before —
    /// so a `scheduled` that moved without this moving is a driver doing more
    /// with the same buffers.
    allocations: f64,
    /// Every op the round was charged, heaviest first — which is what says
    /// *which* side of a round a wall went to when the duty column says only
    /// that one did.
    rows: Vec<(profile::Op, u64, Duration)>,
}

/// A round's submissions as the driver's own clock divides them.
///
/// **A duty cycle is a ratio and cannot be broken up**, and C4 quoted a cell at
/// 44.6% of it — 1.63 s a round inside `submit and wait` against 755 ms of
/// execution and 7.8 ms of encode — with nowhere to put the difference. These
/// are the columns that say where it went.
///
/// **`executed` and `idle` tile the round and the other two do not.** One queue
/// runs one buffer at a time, so the stretches it spends executing and the gaps
/// between them account for the round's clock exactly once. `scheduled` and
/// `queued` are each buffer's own life, and a caller that leaves three in flight
/// has three of those covering one stretch — so they are read as per-submission
/// costs and never summed against a wall.
#[derive(Debug, Default, Clone, Copy)]
struct Divided {
    /// The driver turning a committed buffer into work, which grows with the
    /// dispatches in it.
    scheduled: Duration,
    /// How long a buffer then sat before the GPU picked it up.
    queued: Duration,
    /// What the GPU was executing, summed over the round's buffers.
    ///
    /// **The same quantity [`Speculated::gpu`] holds, accumulated a second
    /// way** — the profile sums it as each buffer completes and this sums the
    /// records — so the two agreeing is a figure neither of them chooses.
    executed: Duration,
    /// What the GPU spent between one buffer of the round and the next, summed:
    /// the device standing still with the round unfinished.
    idle: Duration,
    /// The commit reaching the driver and this thread being woken, which is
    /// what none of the buffer's own three account for.
    unattributed: Duration,
}

impl Divided {
    /// Every round trip of a run, summed and charged to one round.
    fn over(trips: &[inkling_metal::RoundTrip], rounds: u32) -> Self {
        let summed = |of: fn(&inkling_metal::RoundTrip) -> Duration| {
            trips.iter().map(of).sum::<Duration>() / rounds
        };
        Self {
            scheduled: summed(|trip| trip.scheduled),
            queued: summed(|trip| trip.queued),
            executed: summed(|trip| trip.executed),
            idle: summed(|trip| trip.idle),
            unattributed: summed(inkling_metal::RoundTrip::unattributed),
        }
    }
}

/// A proposer that starts the clock the first time it is asked.
///
/// **A batched speculative run prefills inside its own loop**, so a wall taken
/// around the call is a prefill's and the rounds' summed — and the prompts here
/// are 34 keys apiece, which is a prefill worth more than the rounds it opens.
/// The proposer is asked once the prompts are in the caches and once a round
/// after that, which is exactly where `bench sweep`'s own clock starts and what
/// makes this figure a decode figure.
struct Opened<'p, P> {
    proposer: &'p mut P,
    began: Option<Instant>,
    /// Rounds the clock covers, which is what it is asked once for.
    ///
    /// Counted here rather than taken from the chain, because the `k = 0` arm
    /// has no chain to count and its rounds are what make its device column
    /// comparable to the row above it.
    rounds: usize,
    /// The device whose round trips the clock covers, where one is recording.
    ///
    /// Here rather than around the call for the reason the profile is: the
    /// record has to be cleared where the clock starts, or a run's division of
    /// its wait carries its own prefill's submissions.
    device: Option<&'p inkling_metal::Device>,
    /// Buffers the device had allocated when the clock started, which is what a
    /// round's own are counted against.
    allocated: u64,
}

impl<'p, P> Opened<'p, P> {
    fn over(proposer: &'p mut P, device: Option<&'p inkling_metal::Device>) -> Self {
        Self {
            proposer,
            began: None,
            rounds: 0,
            device,
            allocated: 0,
        }
    }
}

impl<P: BatchProposer> BatchProposer for Opened<'_, P> {
    fn depth(&self) -> usize {
        self.proposer.depth()
    }

    /// **Asked once the prompts are in the caches, and once a round after
    /// that**, so counting the calls counts the rounds the clock covers: every
    /// call but the last is followed by a verify, and so is the last, since the
    /// loop stops on a verify that leaves nobody to ask.
    fn propose_batch(&mut self, rounds: &[SeatedRound<'_>]) -> Vec<Vec<usize>> {
        if self.began.is_none() {
            profile::take();
            if let Some(device) = self.device {
                device.round_trips();
                self.allocated = device.allocations();
            }
            self.began = Some(Instant::now());
        }
        self.rounds += 1;
        self.proposer.propose_batch(rounds)
    }
}

impl<P> Opened<'_, P> {
    /// The wall from the first round to now, the device time inside it, every
    /// submission that made it up, and the buffers those submissions were
    /// handed that the device had not seen before.
    fn clock(
        &self,
    ) -> (
        Duration,
        profile::Profile,
        Vec<inkling_metal::RoundTrip>,
        u64,
    ) {
        (
            self.began.expect("a round").elapsed(),
            profile::take(),
            self.device
                .map(inkling_metal::Device::round_trips)
                .unwrap_or_default(),
            self.device
                .map_or(0, |device| device.allocations() - self.allocated),
        )
    }
}

/// What one speculative run is asked for: how wide, how deep, how many tokens
/// each of its sequences wants, and whose submissions to divide.
#[derive(Clone, Copy)]
struct Speculating<'a> {
    slots: usize,
    depth: usize,
    tokens: usize,
    /// The device recording round trips, and `None` for a run nobody is
    /// dividing — which is the settling, whose submissions are not the ones
    /// being reported.
    device: Option<&'a inkling_metal::Device>,
}

/// `run.slots` sequences prefilled together and then decoded together at depth
/// `run.depth`, timed over the rounds alone.
///
/// **The same harness at every depth including none**, which is what makes the
/// `k = 0` column a check rather than a restatement: it is a different code
/// path from [`batched`] over the same work, so the two agreeing is a figure
/// this measurement does not get to choose and the two disagreeing is a defect.
fn speculated(
    weights: &CheckpointWeights<'_>,
    heads: Option<&CheckpointHeads<'_>>,
    config: &TextConfig,
    ids: &[usize],
    run: Speculating<'_>,
) -> Speculated {
    let Speculating {
        slots,
        depth,
        tokens,
        device,
    } = run;
    let generator = weights.generator();
    // The prompts are made distinct rather than copied, for [`Seated`]'s
    // reason: a batch of identical sequences routes every row of every step to
    // the same six experts.
    let prompts: Vec<Vec<usize>> = (0..slots)
        .map(|slot| {
            let mut own = ids.to_vec();
            own.rotate_left(slot % ids.len());
            own
        })
        .collect();
    let asked: Vec<&[usize]> = prompts.iter().map(Vec::as_slice).collect();
    let counts = vec![tokens; slots];
    let mut caches: Vec<ModelCache> = (0..slots)
        .map(|slot| ModelCache::in_slot(config, depth, slot))
        .collect();

    let mut chain =
        heads.map(|heads| MtpProposer::batched(heads, generator, weights, depth, slots));
    let mut alone = Alone;
    let (produced, wall, spent, trips, allocated, rounds, rates, ragged) = match chain.as_mut() {
        Some(chain) => {
            let mut opened = Opened::over(chain, device);
            let produced =
                generator.speculate_batch(&mut caches, &asked, &counts, weights, &mut opened);
            let (wall, spent, trips, allocated) = opened.clock();
            (
                produced,
                wall,
                spent,
                trips,
                allocated,
                opened.rounds,
                opened.proposer.rates(),
                opened.proposer.ragged(),
            )
        }
        None => {
            let mut opened = Opened::over(&mut alone, device);
            let produced =
                generator.speculate_batch(&mut caches, &asked, &counts, weights, &mut opened);
            let (wall, spent, trips, allocated) = opened.clock();
            (
                produced,
                wall,
                spent,
                trips,
                allocated,
                opened.rounds,
                Vec::new(),
                0,
            )
        }
    };

    let rounds = rounds.max(1);
    Speculated {
        wall,
        gpu: spent.gpu(),
        // Every token but the one each prompt's own pass produced, which landed
        // before the clock started.
        tokens: produced.iter().map(Vec::len).sum::<usize>() - slots,
        rounds,
        rates,
        ragged,
        submissions: spent.calls(profile::Op::Submit) as f64 / rounds as f64,
        encode: spent.elapsed(profile::Op::Encode) / rounds as u32,
        waited: spent.elapsed(profile::Op::Submit) / rounds as u32,
        divided: Divided::over(&trips, u32::try_from(rounds).unwrap_or(1)),
        allocations: allocated as f64 / rounds as f64,
        rows: spent.per_step(u32::try_from(rounds).unwrap_or(1)).rows(),
    }
}

/// Tokens a width is settled over before either of the batch sweep's two arms
/// is timed.
///
/// **A wall is what one of the two reports and a wall takes settling**, which
/// this file already had the figure for at another shape: an unsettled batch of
/// one read 25.9 ms against its own 16.4, and a width whose first arm pays that
/// is a width whose two arms are not the same measurement. Eight, which is past
/// the first dispatch after a gap that C2 priced the toll on and short beside
/// the sixty-four either arm then runs.
const SETTLING: usize = 8;

/// Runs of the idle arm the two latency measurements throw away before they
/// report anything.
///
/// **Three, because a wall is what these report and a wall takes three.** Every
/// other measurement here quotes the device's own clock beside its wall and can
/// say which of the two moved; a time to first token is a wall and nothing else,
/// so what settles it has to be settled before the clock starts. At a 32-row
/// chunk into two slots the same run read **823.9 ms, 455.0, 248.7, then 267.7,
/// 267.6 and 267.9** — against a device column of 254.5, 243.0, 237.9, 237.5,
/// 237.6 and 237.6, which had settled by the third and moved 0.1% after it.
const WARM: usize = 3;

/// Prompt rows one step carries, over every request filling in it, when nobody
/// says.
///
/// **The knob a joining request's own wait trades against the wait it puts on
/// the sequences it joins**, and 128 is where both are least at this
/// checkpoint's shapes. A 385-token prompt joining seven decoders reads, at 16,
/// 32, 64, 128 and 384 rows a step: **4864, 3885, 3269, 2252 and 3636 ms to its
/// own first token**, and **199, 317, 532, 725 and 2750 ms on each step the
/// seven decoders take while it is filling**, against their own 73.6 ms.
///
/// The two ends are two different prices. A narrow budget pays call overhead
/// once per chunk — 25 calls where 2 would do — and a whole prompt makes one
/// call the decoders wait 37 of their own steps inside. What the budget does
/// *not* buy is the total: the delay summed over the decoders is 1.96 to 2.99 s
/// whatever it is, which is the prefill's own work and has to be paid. **It is
/// a bound on the jitter of one token, and it is worth about 3.7× of that.**
const ADMITTED: usize = 128;

/// Requests a fleet makes, when nobody says. Two per slot at the default width,
/// so that a request waits for one to finish as well as arriving into a free
/// slot.
const AGENTS: usize = 16;

/// What one request of a fleet asks for, when nobody says — a few hundred
/// tokens, which is the shape an agent's turn has and is long enough that the
/// wake toll amortises to nothing over it.
const ASKED: usize = 200;

/// Conversations a fleet of them takes, when nobody says.
///
/// Four, against a width that seats all of them at once — what the arm is for is
/// a conversation coming back to its own slot, and more conversations than slots
/// would be measuring the eviction instead. `--agents 8 --batch 4` is that
/// measurement and it is a different one.
const CONVERSATIONS: usize = 4;

/// What one turn of one conversation cost the client that asked for it.
#[derive(Debug, Clone, Copy)]
struct Spoke {
    /// Tokens the turn sent, which grows every turn.
    prompt: usize,
    /// Tokens of it the slot already held, and zero on a cold turn.
    reused: usize,
    /// What the client waited for its first token, from submitting the turn.
    first: Duration,
    /// What it waited for its last.
    last: Duration,
}

/// A fleet of conversations, each taking several turns through one engine.
struct Talked {
    /// Per conversation, per turn.
    spoke: Vec<Vec<Spoke>>,
    wall: Duration,
    gpu: Duration,
    steps: usize,
    rows: usize,
    tokens: usize,
    /// What keeping the conversations cost over the whole run, hit or miss —
    /// see [`Stepped::bookkeeping`](inkling_core::Stepped::bookkeeping).
    bookkeeping: Duration,
}

impl Talked {
    /// What every conversation waited on turn `at`, sorted — which is the
    /// distribution, and a mean over a fleet describes none of it.
    fn waits(&self, at: usize, of: impl Fn(&Spoke) -> Duration) -> Vec<Duration> {
        let mut waits: Vec<Duration> = self.spoke.iter().map(|spoke| of(&spoke[at])).collect();
        waits.sort_unstable();
        waits
    }

    /// One turn's rows over every conversation, which is what the two arms part
    /// company on: a cold turn prefills its whole prompt and a kept one prefills
    /// what was added to it.
    fn prefilled(&self, at: usize) -> usize {
        self.spoke
            .iter()
            .map(|spoke| spoke[at].prompt - spoke[at].reused)
            .sum()
    }

    fn turns(&self) -> usize {
        self.spoke.first().map_or(0, Vec::len)
    }

    fn rate(&self) -> f64 {
        self.tokens as f64 / self.wall.as_secs_f64()
    }

    fn duty(&self) -> f64 {
        duty(self.gpu, self.wall)
    }
}

/// A fleet of conversations through one engine, turn by turn.
///
/// **A turn of every conversation is submitted together and the engine drained
/// before the next**, which is what makes the per-turn rows comparable across
/// the two arms: the same prompts in the same order, and the only thing that
/// differs between the arms is what the slots had already prefilled. Arrivals
/// spread over a clock are [`fleeted`]'s subject and would put a second variable
/// in this one.
///
/// Each conversation opens on its own rotation of the pool, so no two of them
/// share a prefix and a slot matched is a slot matched on its own conversation's
/// tokens rather than on an opening they happen to have in common.
fn talking(engine: &Engine<'_, '_>, plan: Session, agents: usize) -> Talked {
    let Engine {
        weights, prompt, ..
    } = *engine;
    let generator = weights.generator();
    let mut seating = engine.seating();
    let pools: Vec<Vec<usize>> = (0..agents)
        .map(|at| {
            prompt
                .iter()
                .copied()
                .cycle()
                .skip(at * plan.added)
                .take(prompt.len())
                .collect()
        })
        .collect();

    let mut produced: Vec<Vec<Vec<usize>>> = vec![Vec::new(); agents];
    let mut spoke: Vec<Vec<Spoke>> = vec![Vec::new(); agents];
    let (mut steps, mut rows, mut tokens, mut bookkeeping) = (0, 0, 0, Duration::ZERO);

    profile::take();
    let started = Instant::now();
    for turn in 0..plan.turns {
        let prompts: Vec<Vec<usize>> = (0..agents)
            .map(|at| plan.prompt(&pools[at], turn, &produced[at]))
            .collect();
        let asking: Vec<Request> = prompts
            .iter()
            .map(|ids| Request {
                prompt: ids.clone(),
                count: plan.generated,
            })
            .collect();

        // **The clock is read here rather than by the driver**, because when a
        // ticket's first token landed is what this measures and the driving is
        // what it shares — see [`turned`], which is one implementation of the
        // ticket bookkeeping for the three callers that drive conversations.
        let submitted = Instant::now();
        let mut felt: HashMap<usize, (Duration, Duration)> = HashMap::new();
        let answered = turned(&mut seating, &generator, weights, &asking, |stepped| {
            steps += 1;
            rows += stepped.rows();
            tokens += stepped.decoding;
            bookkeeping += stepped.bookkeeping;
            let now = submitted.elapsed();
            for ticket in &stepped.first {
                felt.entry(*ticket).or_insert((now, now)).0 = now;
            }
            for answer in &stepped.done {
                felt.entry(answer.ticket).or_insert((now, now)).1 = now;
            }
        });

        for (at, answer) in answered.into_iter().enumerate() {
            assert_eq!(
                answer.produced.len(),
                plan.generated,
                "conversation {at} was answered short on turn {turn}"
            );
            let (first, last) = felt
                .get(&answer.seat.ticket)
                .copied()
                .expect("a wait for every conversation answered");
            spoke[at].push(Spoke {
                prompt: prompts[at].len(),
                reused: answer.seat.reused,
                first,
                last,
            });
            produced[at].push(answer.produced);
        }
    }

    Talked {
        spoke,
        wall: started.elapsed(),
        gpu: profile::take().gpu(),
        steps,
        rows,
        tokens,
        bookkeeping,
    }
}

/// How long a fleet leaves between two arrivals, when nobody says.
const ARRIVAL: Duration = Duration::from_millis(200);

/// One request's life, as the thing that asked for it feels it.
#[derive(Debug, Clone, Copy)]
struct Felt {
    /// When it was made, from the run's start.
    arrived: Duration,
    /// What it waited for its first token, and `None` for one that never got
    /// there.
    first: Option<Duration>,
    /// What it waited for its last.
    last: Option<Duration>,
}

/// What a fleet of requests through one engine came to.
struct Fleeted {
    felt: Vec<Felt>,
    /// The whole run, from the first arrival being made to the last request
    /// being answered.
    wall: Duration,
    gpu: Duration,
    steps: usize,
    /// Rows every step of the run carried between them, which is what the
    /// engine was actually charged for.
    rows: usize,
    /// Tokens it produced, which is what the rate divides.
    tokens: usize,
}

impl Fleeted {
    /// Tokens a second over the whole run, which is the throughput column
    /// C2's static table is the other half of.
    fn rate(&self) -> f64 {
        self.tokens as f64 / self.wall.as_secs_f64()
    }

    fn duty(&self) -> f64 {
        duty(self.gpu, self.wall)
    }

    /// The waits, sorted, for whichever of the two a caller is quoting.
    fn waits(&self, of: impl Fn(&Felt) -> Option<Duration>) -> Vec<Duration> {
        let mut waits: Vec<Duration> = self.felt.iter().filter_map(of).collect();
        waits.sort_unstable();
        waits
    }
}

/// The `q`th percentile of a sorted run of durations, by nearest rank.
///
/// **A value that was measured rather than one interpolated between two that
/// were**, which is what a tail of sixteen requests can support: an
/// interpolated p90 of sixteen samples is a number no request waited.
fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// What both latency measurements hold still: the width of the engine, how many
/// rows a joining prompt enters in, the prompt itself, and what one request
/// asks for.
///
/// **Every arm of both is this held and one thing moved** — the occupancy a
/// request arrives at, or the policy that admits it — which is the rule C2's
/// tables are built on and the one C1 nearly published a clock drift for
/// breaking.
struct Engine<'a, 'w> {
    weights: &'a CheckpointWeights<'w>,
    config: &'a TextConfig,
    prompt: &'a [usize],
    slots: usize,
    admit: usize,
    tokens: usize,
    /// Positions a slot keeps its conversation at once the sequence holding it
    /// leaves, and zero for an engine that keeps none — which is what both
    /// latency measurements run at, because their requests are one turn each
    /// and a conversation kept between them would be a conversation nobody
    /// comes back for.
    reuse: usize,
}

impl Engine<'_, '_> {
    fn seating(&self) -> Continuous<'_> {
        Continuous::keeping(self.config, self.slots, self.admit, self.reuse)
    }

    /// What every request of the joining measurement asks for, which is one
    /// prompt and one budget: two requests of different lengths would be two
    /// measurements with the wait attributed to whichever of them the reader
    /// assumed.
    fn asking(&self) -> Request {
        Request {
            prompt: self.prompt.to_vec(),
            count: self.tokens,
        }
    }

    /// What request `at` of a *fleet* asks for, which is a budget that varies.
    ///
    /// **A fleet whose every request asks for the same number of tokens is the
    /// one shape a static batch is not penalised by, and taking the measurement
    /// on it would report the difference between the two policies as zero for a
    /// reason that belongs to the fixture.** A batch of identical budgets
    /// finishes together — every sequence produces its last token on the same
    /// step — so no slot ever frees early and continuous admission has nothing
    /// to admit into that draining does not offer at the same moment. Measured
    /// that way, 24 requests of 200 tokens read 46.2 s to a first token against
    /// draining's 46.4, which is a null result about the fixture.
    ///
    /// The rule is [`asked_for`], which is a free function so that what a fleet
    /// asks for can be checked without a checkpoint behind it.
    fn asking_at(&self, at: usize) -> Request {
        Request {
            prompt: self.prompt.to_vec(),
            count: asked_for(self.tokens, at),
        }
    }
}

/// What request `at` of a fleet asks for, around a budget of `tokens`.
///
/// An eighth of the budget to twice it, walked by a stride coprime with the
/// period so that the order requests arrive in is not the order of their
/// lengths — a fleet whose short requests all came first would be measuring one
/// arrangement of a queue rather than a policy.
///
/// **The period is [`AGENTS`] and not less**, which is what a shorter one cost:
/// a period of eight over a default fleet of sixteen is two identical halves,
/// and a distribution drawn from a workload that repeats itself is a
/// distribution with half the shape it says it has.
fn asked_for(tokens: usize, at: usize) -> usize {
    tokens / 8 * (1 + at * 7 % AGENTS)
}

/// A fleet of requests arriving on `arrivals`, each asking for what
/// [`Engine::asking_at`] says, through one engine.
///
/// **The arrivals are wall-clock and the engine is not driven ahead of them**: a
/// request that has not been made yet is not in the queue, and an engine with
/// nothing seated sleeps until the next one. That is what makes the wait this
/// reports a wait rather than a position in a list.
fn fleeted(engine: &Engine<'_, '_>, arrivals: &[Duration], policy: Admitting) -> Fleeted {
    let Engine { weights, slots, .. } = *engine;
    let generator = weights.generator();
    let asking: Vec<Request> = (0..arrivals.len()).map(|at| engine.asking_at(at)).collect();
    let mut engine = engine.seating();
    let mut felt: Vec<Felt> = arrivals
        .iter()
        .map(|arrived| Felt {
            arrived: *arrived,
            first: None,
            last: None,
        })
        .collect();

    profile::take();
    let started = Instant::now();
    let (mut next, mut steps, mut rows, mut tokens_out) = (0, 0, 0, 0);
    loop {
        // **A batch is admitted only into an engine that has drained**, which is
        // the whole of what the static arm is: `slots` requests at a time, and
        // the ones that arrived while they ran wait for the last of them.
        let taking = match policy {
            Admitting::Continuously => usize::MAX,
            Admitting::InBatches if engine.idle() => slots,
            Admitting::InBatches => 0,
        };
        let mut took = 0;
        while took < taking && next < felt.len() && felt[next].arrived <= started.elapsed() {
            assert_eq!(
                engine.submit(asking[next].clone()),
                next,
                "a ticket per request, in arrival order"
            );
            next += 1;
            took += 1;
        }

        if engine.idle() {
            let Some(waiting) = felt.get(next) else { break };
            // Nothing is in flight and the next request has not been made, so
            // there is no work to do and no clock to hold. What this leaves is
            // the gap a wake is charged for — see the README's toll.
            std::thread::sleep(waiting.arrived.saturating_sub(started.elapsed()));
            continue;
        }

        let stepped = engine.step(&generator, weights);
        steps += 1;
        rows += stepped.rows();
        tokens_out += stepped.decoding;
        let now = started.elapsed();
        for ticket in &stepped.first {
            felt[*ticket].first = Some(now - felt[*ticket].arrived);
        }
        for answer in &stepped.done {
            felt[answer.ticket].last = Some(now - felt[answer.ticket].arrived);
        }
    }

    Fleeted {
        felt,
        wall: started.elapsed(),
        gpu: profile::take().gpu(),
        steps,
        rows,
        tokens: tokens_out,
    }
}

/// What one request waited for its first token, and what the batch it joined
/// paid for it.
struct Joined {
    /// From the request being made to its first token reaching whoever asked.
    ttft: Duration,
    /// Steps the engine ran inside that, which is what the wait is made of.
    steps: usize,
    gpu: Duration,
    /// What one step of the batch cost while the joining prompt was riding in
    /// it, and what the same batch's step cost with nothing joining. **The
    /// price the sequences already in flight pay is the difference.**
    mixed: Option<Duration>,
    settled: Option<Duration>,
}

/// `held` sequences settled and decoding, then one more request made — and what
/// it waited.
///
/// **The held sequences are settled before the clock starts** for the reason
/// [`Seated::new`] throws its own settling steps away: a width is a fresh wrap,
/// and the first steps of one pay for allocating its spans and its windows.
///
/// `held` of zero is the same request at an idle engine, which is the arm every
/// other one here is read against — and it is the same code rather than a
/// second path, so what differs between the two rows is the batch and nothing
/// else.
fn joined(engine: &Engine<'_, '_>, held: usize, policy: Admitting) -> Joined {
    let Engine { weights, slots, .. } = *engine;
    assert!(held < slots, "{held} held sequences in {slots} slots");
    let generator = weights.generator();
    let asking = engine.asking();
    let mut engine = engine.seating();
    for _ in 0..held {
        engine.submit(asking.clone());
    }

    let mut settled = None;
    if held > 0 {
        while engine.step(&generator, weights).filling > 0 {}
        let at = Instant::now();
        for _ in 0..SETTLED {
            engine.step(&generator, weights);
        }
        settled = Some(at.elapsed() / u32::try_from(SETTLED).unwrap_or(1));
    }

    profile::take();
    let at = Instant::now();
    let mut steps = 0;
    if policy == Admitting::InBatches {
        // The wait a request makes of a static batch, measured rather than
        // reasoned about: it is admitted when the batch it arrived behind has
        // decoded the last token any of its sequences asked for.
        while !engine.idle() {
            engine.step(&generator, weights);
            steps += 1;
        }
    }
    let ticket = engine.submit(asking);

    let (mut mixed, mut riding) = (Duration::ZERO, 0u32);
    loop {
        let before = Instant::now();
        let stepped = engine.step(&generator, weights);
        let took = before.elapsed();
        steps += 1;
        // Only the steps that carried the joining prompt *and* the batch it
        // joined, which is the comparison `settled` is the other half of. The
        // step that produces the first token carries no prompt rows and belongs
        // to neither.
        if stepped.filling > 0 && stepped.decoding == held {
            mixed += took;
            riding += 1;
        }
        if stepped.first.contains(&ticket) {
            break;
        }
    }
    Joined {
        ttft: at.elapsed(),
        steps,
        gpu: profile::take().gpu(),
        mixed: (riding > 0).then(|| mixed / riding),
        settled,
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

/// `count` repetitions of `unit`, back to back, with whatever host time `gap`
/// deliberately leaves between them.
///
/// **What each arm discards before the ones that are timed is its own**, and
/// every arm discards something for the reason the prefill measurement takes the
/// second of two: the first unit of a shape is the one that faults in the pages
/// the rest of them read, and a page fault charged to the first unit is a drift
/// this would report as the part warming up. A generation's own first tick is
/// the prompt's prefill, which is not a step at all; a batched run's is
/// [`Seated`]'s settling, which it has already run and thrown away.
fn ticked<'a>(
    weights: &CheckpointWeights<'_>,
    config: &TextConfig,
    unit: &Unit,
    count: usize,
    shape: Shape,
    warm: Option<&'a dyn Fn()>,
) -> Ticking<'a> {
    profile::take();
    let mut ticking = Ticking::new(shape, warm, count);
    match unit {
        // Every step of one generation rather than a generation apiece, so that
        // the gap falls between two decode steps — which is the occupancy this
        // arm exists to move. What it costs is the growing key count [`Unit`]
        // describes.
        Unit::Step(ids) => {
            let generator = weights.generator();
            let cache = &mut ModelCache::speculating(config, 0);
            let ending = Ending {
                budget: count + 1,
                eos: None,
            };
            // The prompt's own prefill, which is not a decode step: it opens the
            // run rather than being timed inside it.
            let mut opened = false;
            let mut sink = |_| {
                let gpu = profile::take().gpu();
                match opened {
                    true => ticking.tick(gpu),
                    false => {
                        opened = true;
                        ticking.settled();
                    }
                }
                ControlFlow::Continue(())
            };
            generator.stream(cache, ids, ending, weights, &mut sink);
        }
        // The gap falls between two steps of one batch, as it does between two
        // steps of one generation: what moves is how much work the period holds
        // either side of it.
        Unit::Batch(ids, slots) => {
            let mut seated = Seated::new(weights, config, ids, *slots);
            ticking.settled();
            for _ in 0..count {
                seated.step(weights);
                ticking.tick(profile::take().gpu());
            }
        }
        Unit::Prefill(ids) => {
            // The repetition that faults the pages in, which every later one
            // reads without paying for.
            generate(weights, None, config, ids, 1, 0);
            ticking.settled();
            for _ in 0..count {
                let run = generate(weights, None, config, ids, 1, 0);
                ticking.tick(run.gpu);
            }
        }
    }
    ticking
}

/// What a unit of work cost by where in its burst it ran.
///
/// **This is what separates a clock that follows the occupancy from one that
/// ramps after a gap**, and a mean cannot tell them apart. A part that pays a
/// fixed ramp for having been idle pays it once a burst: the unit after the gap
/// is dear and the ones behind it are not. A part running a lower clock because
/// the period is mostly idle pays the same on every unit of the burst, however
/// long ago the gap was.
///
/// Whole bursts only. A run whose count does not divide would otherwise report
/// its early positions over one more burst than its late ones, which is a
/// difference between the columns that is nothing but where the run stopped.
fn after_a_gap(ticks: &[Tick], burst: usize) -> Vec<Reading> {
    let whole = ticks.len() / burst.max(1) * burst.max(1);
    if burst < 2 || whole == 0 {
        return Vec::new();
    }
    let at = |position: usize| {
        let held: Vec<Duration> = ticks[..whole]
            .iter()
            .skip(position)
            .step_by(burst)
            .map(|tick| tick.gpu)
            .collect();
        held.iter().sum::<Duration>() / held.len() as u32
    };
    let positions: Vec<Duration> = (0..burst).map(at).collect();
    let furthest = *positions.last().expect("a burst of at least two positions");

    eprintln!("  in burst     device    against the {burst}th, which is furthest from the gap");
    let mut readings = Vec::new();
    for (position, held) in positions.iter().enumerate() {
        eprintln!(
            "  {:>8}  {:>9.4?}    {:+.2}%",
            position + 1,
            held,
            against(*held, furthest)
        );
        readings.push(Reading::new(
            format!("at{}.device", position + 1),
            millis(*held),
            "ms",
        ));
    }
    readings
}

/// What a run leaves between one unit of work and the next.
///
/// **A gap and an interval are the same sleep counted from two different ends,
/// and a server knows the second one.** Requests arrive every 200 ms whatever a
/// step costs, so what the device is left idle for is the interval less the work
/// — and at a width where the work is longer than the interval there is no idle
/// at all. That last case is why both words exist: from the sleep's side a gap
/// of nothing and an interval already spent are the same instruction, and about
/// a server they are two different sentences. One says the operator asked for no
/// gap; the other says the operator asked for 200 ms between requests and the
/// batch cannot answer them that fast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gap {
    /// Host time left after every unit, whatever the unit cost.
    After(Duration),
    /// A period a unit is answered inside, idle for whatever is left of it.
    Every(Duration),
}

impl Gap {
    /// What to sleep for at the end of a period `spent` of which has gone on the
    /// work.
    fn idle(&self, spent: Duration) -> Duration {
        match self {
            Self::After(idle) => *idle,
            Self::Every(period) => period.saturating_sub(spent),
        }
    }

    /// The interval `--every` named, and `None` where a gap was asked for
    /// instead — which is what a fleet's arrivals are spaced by and is the one
    /// of the two words that means a rate.
    fn every(&self) -> Option<Duration> {
        match self {
            Self::Every(period) => Some(*period),
            Self::After(_) => None,
        }
    }

    /// As the header says it, which is the flag that asked for it.
    fn said(&self) -> String {
        match self {
            Self::After(idle) if idle.is_zero() => "no idle between them".to_string(),
            Self::After(idle) => format!("{idle:.0?} idle between them"),
            Self::Every(period) => format!("one of them every {period:.0?}"),
        }
    }
}

/// The shape a run puts its work in: what it leaves between two gaps, and how
/// many units it answers between them.
///
/// **A burst is what makes the gap a server's rather than a metronome's.** A
/// request arriving every 200 ms is not one step every 200 ms — it is however
/// many steps that request needs, back to back, and then an idle device until
/// the next one. The two are the same duty cycle at different gap lengths,
/// which is the pair that says which of the two a slower clock is a function
/// of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    gap: Gap,
    burst: usize,
    /// How often a gap is punctuated by a dispatch that exists only to be one,
    /// and nothing for a gap left empty — which is every figure this repo has
    /// ever taken.
    warm: Option<Duration>,
}

impl Shape {
    /// How the header says it: the gap, the units it falls between, and
    /// whatever is dispatched inside it.
    fn said(&self) -> String {
        let mut said = match self.burst {
            1 => self.gap.said(),
            burst => format!("{}, {burst} of them a burst", self.gap.said()),
        };
        if let Some(every) = self.warm {
            let _ = write!(said, ", kept warm every {every:.0?}");
        }
        said
    }
}

/// The clock a run of units is timed on, and the gap it leaves between bursts of
/// them.
///
/// A type rather than a closure because both arms of [`ticked`] keep the same
/// state and one of them reaches it from inside a sink the generator drives.
struct Ticking<'a> {
    /// When the unit now running began, which is what its own period is
    /// measured from.
    at: Instant,
    /// When the burst now running began, which is what a rate is measured from:
    /// a request answered by four steps is one request whatever the four cost.
    opened: Instant,
    shape: Shape,
    /// What is submitted into a gap, for the runs that fill theirs.
    ///
    /// A closure rather than the [`KeepWarm`] itself, so that what is being
    /// tested here — that a gap is dispatched into and is still a gap of the
    /// length it was asked for — can be tested without a device.
    warm: Option<&'a dyn Fn()>,
    /// What those submissions cost the device, kept apart from what the units
    /// cost it: a keep-warm charged to the work it was meant to protect would
    /// be a lever paying for itself out of its own pocket.
    kept: Duration,
    dispatched: usize,
    ticks: Vec<Tick>,
}

impl<'a> Ticking<'a> {
    fn new(shape: Shape, warm: Option<&'a dyn Fn()>, count: usize) -> Self {
        let now = Instant::now();
        Self {
            at: now,
            opened: now,
            shape,
            warm,
            kept: Duration::ZERO,
            dispatched: 0,
            ticks: Vec::with_capacity(count),
        }
    }

    /// A gap of `idle`, empty or punctuated by dispatches that exist only to be
    /// dispatches.
    ///
    /// The device time they cost is taken off the account here, so that the
    /// unit after the gap is charged for itself alone — and reported, because
    /// what the lever costs is half of what it is worth.
    ///
    /// **A subdivided gap comes back closer to the deadline than one sleep
    /// does**, because a `sleep` on this host overshoots what it was asked for
    /// and the chunks that follow absorb it: a 200 ms gap left empty measures
    /// 240 ms of period against 235 for the same gap kept warm every 20 ms. So
    /// the two arms of that pair do not idle for exactly as long as each other,
    /// and the arm that idles *less* is the kept-warm one — which is the
    /// direction that flatters it.
    fn idled(&mut self, idle: Duration) {
        let Some(warm) = self.warm else {
            std::thread::sleep(idle);
            return;
        };
        let Some(every) = self.shape.warm else {
            std::thread::sleep(idle);
            return;
        };
        let until = Instant::now() + idle;
        while let Some(left) = until.checked_duration_since(Instant::now()) {
            std::thread::sleep(every.min(left));
            if Instant::now() >= until {
                break;
            }
            warm();
            self.dispatched += 1;
        }
        self.kept += profile::take().gpu();
    }

    /// Both clocks back to zero after work nobody is timing, and a full gap
    /// before the first unit that is.
    ///
    /// **The gap is left here for the same reason it is left between two units**:
    /// every timed unit is preceded by one, so the first is too. A run that
    /// opened straight out of its own prefills would have its first unit — and
    /// with a burst, its whole first position — measured off a device that had
    /// been busy right up to it, which is the one thing this measurement is
    /// about.
    fn settled(&mut self) {
        let idle = self.shape.gap.idle(Duration::ZERO);
        if !idle.is_zero() {
            self.idled(idle);
        }
        self.kept = Duration::ZERO;
        self.dispatched = 0;
        profile::take();
        self.at = Instant::now();
        self.opened = self.at;
    }

    /// One unit finished for `gpu` of device time, whatever gap follows it, and
    /// the period the two of them are.
    ///
    /// The sleep is behind a test so that a run asking for no gap makes no
    /// syscall a run of any other measurement here would not: what this is
    /// timing is a decode step, and a decode step's own host side is 8% of it.
    fn tick(&mut self, gpu: Duration) {
        let ending = (self.ticks.len() + 1) % self.shape.burst == 0;
        let idle = match ending {
            true => self.shape.gap.idle(self.opened.elapsed()),
            false => Duration::ZERO,
        };
        if !idle.is_zero() {
            self.idled(idle);
        }
        self.ticks.push(Tick {
            wall: self.at.elapsed(),
            gpu,
        });
        self.at = Instant::now();
        if ending {
            self.opened = self.at;
        }
    }
}

/// What a run of identical units says about the clock underneath them.
///
/// The run in [`PARTS`] parts, each part's mean device time and duty cycle, and
/// the last part against the first — which is the number that says whether the
/// part held its speed, and in which direction it did not.
///
/// **Whole bursts, for the reason [`after_a_gap`] takes whole bursts.** The unit
/// after a gap costs about three and a half milliseconds more than the ones
/// behind it, so a part holding four of those where the next part holds three is
/// a part that reads slower for no reason but where the cut fell — 128 units in
/// bursts of eight report a drift of −0.98% that way, on units that are all
/// identical and a clock that never moved. Cut on burst boundaries and every
/// part holds the same proportion of gaps, which is what makes the column a
/// reading about the machine.
fn clocked(ticks: &[Tick], burst: usize) -> Vec<Reading> {
    let mean = |part: &[Tick]| match part.is_empty() {
        true => (Duration::ZERO, Duration::ZERO),
        false => (
            part.iter().map(|tick| tick.gpu).sum::<Duration>() / part.len() as u32,
            part.iter().map(|tick| tick.wall).sum::<Duration>() / part.len() as u32,
        ),
    };
    let burst = burst.max(1);
    // The units of whole bursts, which for a run of one unit a gap is every unit
    // it has.
    let ticks = &ticks[..ticks.len() / burst * burst];
    let (whole_gpu, whole_wall) = mean(ticks);
    // **[`PARTS`] parts wherever there are bursts for them**, cut at the
    // proportion rather than by a fixed chunk: a chunk wide enough for the last
    // part to be short is one that leaves a run of eleven reporting four parts
    // and a run of ten reporting five, so the shape a reader compares would
    // depend on a count nobody chose for its divisibility.
    let bursts = ticks.len() / burst;
    let cuts = PARTS.min(bursts);
    let parts: Vec<(Duration, Duration)> = (0..cuts)
        .map(|at| mean(&ticks[at * bursts / cuts * burst..(at + 1) * bursts / cuts * burst]))
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
    let median = middle(ticks);
    eprintln!(
        "  whole: device {:.4?}, median {:.4?}, wall {:.4?}, duty {:.1}%, drift {:+.2}%",
        whole_gpu,
        median,
        whole_wall,
        duty(whole_gpu, whole_wall),
        drift,
    );
    readings.extend([
        Reading::new("clock.device", millis(whole_gpu), "ms"),
        Reading::new("clock.median", millis(median), "ms"),
        Reading::new("clock.wall", millis(whole_wall), "ms"),
        Reading::new("clock.duty", duty(whole_gpu, whole_wall), "%"),
        Reading::new("clock.drift", drift, "%"),
    ]);
    readings
}

/// The middle unit's device time.
///
/// **A mean that is not near its median is a reading of something else**, which
/// is the rule [`Generated::median`] states and the one an idled run needs most:
/// a run whose gap costs it one dear unit a burst has a mean above every unit it
/// is a mean of, and a single stalled unit does the same to a run with no gap at
/// all. The two are told apart by the pair of columns and by nothing else here.
fn middle(ticks: &[Tick]) -> Duration {
    let mut held: Vec<Duration> = ticks.iter().map(|tick| tick.gpu).collect();
    held.sort_unstable();
    held.get(held.len() / 2).copied().unwrap_or_default()
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
                shape: Shape {
                    gap: Gap::After(Duration::ZERO),
                    burst: 1,
                    warm: None,
                },
                prefill: 0,
                admit: ADMITTED,
                agents: AGENTS,
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
                shape: Shape {
                    gap: Gap::After(Duration::ZERO),
                    burst: 1,
                    warm: None,
                },
                prefill: 0,
                admit: ADMITTED,
                agents: AGENTS,
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
                shape: Shape {
                    gap: Gap::After(Duration::ZERO),
                    burst: 1,
                    warm: None,
                },
                prefill: 0,
                admit: ADMITTED,
                agents: AGENTS,
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
                shape: Shape {
                    gap: Gap::After(Duration::ZERO),
                    burst: 1,
                    warm: None,
                },
                prefill: 0,
                admit: ADMITTED,
                agents: AGENTS,
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

    /// **A fleet of conversations takes both of the flags its two parents take**,
    /// which is the whole of what it is: a width and a number of agents from the
    /// fleet, a count of positions to keep from the session.
    ///
    /// Stated because the refusals above are written as a list of measurements
    /// rather than as a property, so a measurement added to the list of one flag
    /// and not the others is refused a flag it needs — and the refusal reads
    /// exactly like the flag being wrong.
    #[test]
    fn a_fleet_of_conversations_takes_a_width_a_count_of_agents_and_a_bound() {
        let parsed = Job::parse(
            [
                "conversations",
                "models/small",
                "--reuse-tokens",
                "0",
                "--batch",
                "4",
                "--agents",
                "3",
                "--admit",
                "64",
                "--tokens",
                "2048",
            ]
            .map(str::to_string),
        )
        .expect("parses");
        let Job::Measure {
            what,
            tokens,
            reuse,
            widest,
            admit,
            agents,
            ..
        } = parsed
        else {
            panic!("a measurement")
        };
        assert_eq!(what, What::Conversations);
        assert_eq!(
            (tokens, reuse, widest, admit, agents),
            (2048, 0, Some(4), 64, 3)
        );

        // And the defaults are the workload's: a session's opening, a slot per
        // conversation, and a conversation kept.
        let Ok(Job::Measure {
            tokens,
            reuse,
            agents,
            widest,
            ..
        }) = Job::parse(["conversations", "models/small"].map(str::to_string))
        else {
            panic!("a measurement")
        };
        assert_eq!(
            (tokens, reuse, agents, widest),
            (Session::OPENING, DEFAULT_BOUND, CONVERSATIONS, None)
        );

        // An engine with no slots seats nobody, refused here rather than in the
        // panic one layer down.
        assert!(
            Job::parse(["conversations", "models/small", "--batch", "0"].map(str::to_string))
                .is_err(),
            "an engine of no slots was accepted"
        );
    }

    /// **A depth given on the command line is the depth the measurement runs
    /// at**, and one given to a measurement that decodes a token at a time is
    /// refused rather than dropped.
    ///
    /// **Stated because a default cannot state it.** A batch sweep carried a
    /// depth of four and ran at none — the flag's value never reached the call,
    /// which shadowed it — and every row it printed was plausible: the table
    /// was a `k = 0` table under a header saying otherwise. Nothing about the
    /// default was wrong and nothing about the refusals was wrong, so what has
    /// to be asserted is the number arriving.
    #[test]
    fn a_batch_sweeps_depth_is_the_one_the_command_line_asked_for() {
        let depth = |what: &str, k: &str| match Job::parse(
            [what, "models/small", "--depth", k].map(str::to_string),
        ) {
            Ok(Job::Measure { depth, .. }) => Some(depth),
            _ => None,
        };
        assert_eq!(depth("batch", "3"), Some(3));
        assert_eq!(depth("sweep", "2"), Some(2));
        assert_eq!(depth("engines", "5"), Some(5));

        for what in ["decode", "prefill", "session", "clock", "joining", "fleet"] {
            assert!(
                depth(what, "3").is_none(),
                "{what} took a depth it decodes no block at"
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
    ///
    /// **A batch sweep's default depth is none**, which is the column every
    /// other one is read against — see [`What::Batch`] in [`measure`], and
    /// `a_batch_sweeps_depth_is_the_one_the_command_line_asked_for` for the
    /// half of this that a default cannot say.
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
                depth: 0,
                numerics: Numerics::default(),
                reuse: DEFAULT_BOUND,
                widest: Some(8),
                shape: Shape {
                    gap: Gap::After(Duration::ZERO),
                    burst: 1,
                    warm: None,
                },
                prefill: 0,
                admit: ADMITTED,
                agents: AGENTS,
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
                shape: Shape {
                    gap: Gap::After(Duration::from_millis(40)),
                    burst: 1,
                    warm: None,
                },
                prefill: 0,
                admit: ADMITTED,
                agents: AGENTS,
            }
        );
        for what in ["decode", "prefill", "sweep", "engines", "session", "batch"] {
            for lever in [["--idle", "40"], ["--prefill", "2048"], ["--burst", "4"]] {
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
                shape: Shape {
                    gap: Gap::After(Duration::ZERO),
                    burst: 1,
                    warm: None,
                },
                prefill: 2048,
                admit: ADMITTED,
                agents: AGENTS,
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

    /// **The occupancies a joining request is measured at**: the idle engine
    /// first, then doubling, and never an engine with no slot for it.
    ///
    /// The last is the one worth asserting rather than reading: a request
    /// arriving at a full engine measures a queue, and the arm that measures the
    /// queue is the draining one this runs beside it.
    #[test]
    fn a_joining_request_is_measured_against_an_engine_with_a_slot_for_it() {
        assert_eq!(occupancies(1), vec![0]);
        assert_eq!(occupancies(2), vec![0, 1]);
        assert_eq!(occupancies(8), vec![0, 1, 2, 4, 7]);
        assert_eq!(occupancies(32), vec![0, 1, 2, 4, 8, 16, 31]);
        for slots in [1, 2, 8, 32] {
            assert!(
                occupancies(slots).iter().all(|held| *held < slots),
                "an occupancy with no slot left for the joiner at {slots}"
            );
        }
    }

    /// **A fleet's requests do not all ask for the same thing**, because a batch
    /// of identical budgets finishes together and a static batch is not
    /// penalised by a slot that never frees early. What that would report is a
    /// null result about the fixture, so the spread is asserted rather than
    /// assumed — and so is that the order they arrive in is not the order of
    /// their lengths.
    #[test]
    fn a_fleets_requests_do_not_all_ask_for_the_same_thing() {
        // Over the default fleet's own size, which is where the property has to
        // hold: a pattern that repeats inside one fleet is a fleet with fewer
        // distinct requests in it than its size says.
        let counts: Vec<usize> = (0..AGENTS).map(|at| asked_for(200, at)).collect();
        assert_eq!(
            counts,
            vec![
                25, 200, 375, 150, 325, 100, 275, 50, 225, 400, 175, 350, 125, 300, 75, 250
            ]
        );
        let mut sorted = counts.clone();
        sorted.sort_unstable();
        assert_ne!(counts, sorted, "the short requests all arrived first");
        assert_eq!(
            counts.iter().collect::<BTreeSet<_>>().len(),
            counts.len(),
            "two requests of one length inside one fleet"
        );
    }

    /// **A percentile is a wait some request actually had.** Interpolating
    /// between two of sixteen samples would report a tail nobody waited, which
    /// is the one thing a tail is quoted for.
    #[test]
    fn a_percentile_is_one_of_the_waits_it_is_taken_over() {
        let waits: Vec<Duration> = [10u64, 20, 30, 40].map(Duration::from_millis).to_vec();
        assert_eq!(percentile(&waits, 0.5), Duration::from_millis(20));
        assert_eq!(percentile(&waits, 0.9), Duration::from_millis(40));
        assert_eq!(percentile(&waits, 1.0), Duration::from_millis(40));
        // The lowest rank is the first sample rather than a step before it.
        assert_eq!(percentile(&waits, 0.0), Duration::from_millis(10));
        assert_eq!(percentile(&[], 0.5), Duration::ZERO);
    }

    /// **The two latency measurements' flags are theirs alone**, for the reason
    /// every other flag here is refused to the measurements that could only drop
    /// it: a row that silently ignored the number it was given says something
    /// other than what was asked for.
    #[test]
    fn the_flags_a_joining_request_takes_are_refused_to_what_does_not_join() {
        let parsed = |args: &[&str]| Job::parse(args.iter().map(|arg| arg.to_string()));
        assert!(parsed(&["joining", "models/small", "--admit", "64"]).is_ok());
        assert!(parsed(&["fleet", "models/small", "--agents", "8"]).is_ok());
        assert!(parsed(&["fleet", "models/small", "--every", "200"]).is_ok());
        assert!(
            parsed(&["batch", "models/small", "--admit", "64"]).is_err(),
            "a sweep took a chunk for a prompt nothing in it joins with"
        );
        assert!(
            parsed(&["joining", "models/small", "--agents", "8"]).is_err(),
            "a measurement of one request took a fleet's size"
        );
        assert!(
            parsed(&["fleet", "models/small", "--idle", "200"]).is_err(),
            "a fleet took a clock run's gap"
        );
        assert!(
            parsed(&["joining", "models/small", "--every", "200"]).is_err(),
            "a measurement of one request took an arrival rate"
        );
        // One slot is an engine with none free for the joiner, and a prompt
        // entering no rows a step never enters a cache at all.
        assert!(parsed(&["joining", "models/small", "--batch", "1"]).is_err());
        assert!(parsed(&["joining", "models/small", "--admit", "0"]).is_err());
        // A fleet has no joiner to leave a slot for, so one slot is a width it
        // means something by — and no slots is an engine that seats nobody.
        assert!(parsed(&["fleet", "models/small", "--batch", "1"]).is_ok());
        assert!(parsed(&["fleet", "models/small", "--batch", "0"]).is_err());
        // **A budget that runs out inside the settling is a batch whose step
        // these would report as a mean over calls to an empty engine.**
        assert!(
            parsed(&["fleet", "models/small", "--tokens", &SETTLED.to_string()]).is_err(),
            "a request that finishes inside the settling was taken"
        );
        assert!(
            parsed(&[
                "joining",
                "models/small",
                "--tokens",
                &(SETTLED + 1).to_string()
            ])
            .is_ok()
        );
    }

    /// **A rate is a gap the work is taken out of**, which is the shape a server
    /// has: requests arrive on their own schedule and what the device is left
    /// idle for is whatever the step did not spend. A rate the work outruns
    /// leaves no gap at all rather than a negative one — the saturated server,
    /// which reads as the duty cycle it produced.
    #[test]
    fn a_rate_is_the_interval_less_whatever_the_work_spent() {
        let period = Duration::from_millis(200);
        let step = Duration::from_millis(15);
        assert_eq!(Gap::Every(period).idle(step), period - step);
        assert_eq!(
            Gap::Every(period).idle(Duration::from_millis(223)),
            Duration::ZERO
        );
        // Where a gap is what it says whatever the work cost, which is the lever
        // the published sweep was taken with.
        assert_eq!(Gap::After(period).idle(step), period);
        assert_eq!(Gap::After(period).idle(Duration::from_millis(223)), period);
        assert_eq!(Gap::After(Duration::ZERO).idle(step), Duration::ZERO);
    }

    /// **A gap and a rate are two answers to one question**, so a run given both
    /// is refused rather than served one of them — and a rate of no milliseconds
    /// is not a rate, unlike the gap of none every idled arm is compared
    /// against.
    #[test]
    fn a_clock_run_takes_a_gap_or_a_rate_and_not_both() {
        assert!(
            matches!(
                Job::parse(["clock", "models/small", "--every", "200"].map(str::to_string))
                    .expect("parses"),
                Job::Measure {
                    shape: Shape { gap: Gap::Every(period), .. },
                    ..
                } if period == Duration::from_millis(200)
            ),
            "an interval did not reach the run"
        );
        for given in [
            ["clock", "models/small", "--every", "200", "--idle", "8"].as_slice(),
            ["clock", "models/small", "--every", "0"].as_slice(),
            ["decode", "models/small", "--every", "200"].as_slice(),
        ] {
            assert!(
                Job::parse(given.iter().map(|word| word.to_string())).is_err(),
                "{given:?} was taken"
            );
        }
    }

    /// **A part is whole bursts or it is a reading about where the cut fell.**
    /// The unit after a gap is the dear one, so a part holding four of them
    /// where the next holds three reads slower on units that are identical —
    /// 128 units in bursts of eight, cut at the proportion, report a drift of
    /// about a percent that way.
    #[test]
    fn the_parts_of_a_bursted_run_hold_the_same_share_of_gaps() {
        // Sixteen bursts of eight, every burst dear in its first position and
        // flat behind it: a run whose clock never moved.
        let mut held = Vec::new();
        for _ in 0..16 {
            held.push(19_000);
            held.extend([15_500; 7]);
        }
        let bursted = ticks(&held, &[20_000; 128]);

        let readings = clocked(&bursted, 8);
        reads(&readings, "clock.drift", 0.0);
        for part in 1..=PARTS {
            reads(
                &readings,
                &format!("part{part}.device"),
                (19.0 + 7.0 * 15.5) / 8.0,
            );
        }
        // And told nothing about the burst, the same run reports a drift it
        // does not have — which is the reading this is here to keep out.
        let blind = reading(&clocked(&bursted, 1), "clock.drift");
        assert!(
            blind < -0.5,
            "a proportional cut hid its own artefact: {blind}"
        );
    }

    /// **The three things at once**: a burst, a count that does not divide it,
    /// and fewer bursts than there are parts. Every part still holds whole
    /// bursts, so no part is a different shape from its neighbours and the tail
    /// that belongs to no burst is in none of them.
    #[test]
    fn a_run_of_a_few_uneven_bursts_still_cuts_its_parts_on_them() {
        // Three bursts of four and a tail of two, dear in every first position.
        let mut held = Vec::new();
        for _ in 0..3 {
            held.push(19_000);
            held.extend([15_500; 3]);
        }
        held.extend([15_500; 2]);
        let readings = clocked(&ticks(&held, &[20_000; 14]), 4);

        let parts: Vec<&str> = names(&readings)
            .into_iter()
            .filter(|name| name.starts_with("part"))
            .collect();
        assert_eq!(parts.len(), 3, "three bursts cut into {parts:?}");
        for part in 1..=3 {
            reads(
                &readings,
                &format!("part{part}.device"),
                (19.0 + 3.0 * 15.5) / 4.0,
            );
        }
        // The two units of the unfinished burst are in no part and in no mean:
        // a whole-run figure that held them would carry three tolls where the
        // shape it reports carries three in twelve.
        reads(&readings, "clock.device", (19.0 + 3.0 * 15.5) / 4.0);
        reads(&readings, "clock.drift", 0.0);
    }

    /// **A gap that is dispatched into is still a gap.** What the lever is for
    /// is to leave the device something to do without shortening the idle it is
    /// being asked about, so a run that came back early would be measuring a
    /// different gap from the one the arm it is paired against left.
    #[test]
    fn a_gap_kept_warm_is_dispatched_into_and_is_still_the_gap_it_was_asked_for() {
        const GAP: Duration = Duration::from_millis(100);
        const EVERY: Duration = Duration::from_millis(20);
        let dispatched = std::cell::Cell::new(0usize);
        let warm = || dispatched.set(dispatched.get() + 1);
        let mut ticking = Ticking::new(
            Shape {
                gap: Gap::After(GAP),
                burst: 1,
                warm: Some(EVERY),
            },
            Some(&warm),
            2,
        );

        let started = Instant::now();
        ticking.tick(Duration::ZERO);
        assert!(
            started.elapsed() >= GAP,
            "the gap came back early: {:?}",
            started.elapsed()
        );
        // At least one and no more than the gap holds, which is what a cadence
        // is: the count itself is the host's to decide, because a sleep of 20 ms
        // on this machine is 20 ms and however much longer it feels like.
        // And the gap is a gap rather than a cadence run until something else
        // stops it: a loop that slept `every` without watching the deadline
        // would still be dispatching a minute later.
        assert!(
            started.elapsed() < 2 * GAP,
            "the gap ran long: {:?}",
            started.elapsed()
        );
        assert!(
            (1..=(GAP.as_millis() / EVERY.as_millis()) as usize).contains(&dispatched.get()),
            "{} dispatches in a {GAP:?} gap at one every {EVERY:?}",
            dispatched.get()
        );
        assert_eq!(dispatched.get(), ticking.dispatched, "counted twice");
    }

    /// **A gap of nothing has nowhere to put a dispatch**, so the lever is
    /// refused there rather than turned into a busy loop between two units that
    /// are already back to back — and refused, like every other one, to the
    /// measurements that do not vary their own duty cycle.
    #[test]
    fn a_keep_warm_needs_a_gap_to_dispatch_into() {
        assert!(
            matches!(
                Job::parse(
                    ["clock", "models/small", "--idle", "200", "--keep-warm", "20"]
                        .map(str::to_string)
                )
                .expect("parses"),
                Job::Measure {
                    shape: Shape { warm: Some(every), .. },
                    ..
                } if every == Duration::from_millis(20)
            ),
            "a keep-warm did not reach the run"
        );
        for given in [
            ["clock", "models/small", "--keep-warm", "20"].as_slice(),
            ["clock", "models/small", "--idle", "0", "--keep-warm", "20"].as_slice(),
            [
                "clock",
                "models/small",
                "--every",
                "200",
                "--keep-warm",
                "0",
            ]
            .as_slice(),
            ["decode", "models/small", "--keep-warm", "20"].as_slice(),
        ] {
            assert!(
                Job::parse(given.iter().map(|word| word.to_string())).is_err(),
                "{given:?} was taken"
            );
        }
        // A rate leaves whatever the work did not spend, which is a gap to
        // dispatch into.
        assert!(
            Job::parse(
                [
                    "clock",
                    "models/small",
                    "--every",
                    "200",
                    "--keep-warm",
                    "20"
                ]
                .map(str::to_string)
            )
            .is_ok(),
            "a rate had nowhere to put a keep-warm"
        );
    }

    /// **One stalled unit moves the mean and not the middle**, which is the pair
    /// of columns a reader needs to tell a run that was uniformly slower from a
    /// run that was interrupted once — and an idled run is where that matters,
    /// because the gap makes one unit a burst dear by design.
    #[test]
    fn a_stalled_unit_moves_the_mean_and_not_the_middle() {
        let mut held = vec![15_000; 9];
        held.push(150_000);
        let stalled = ticks(&held, &[20_000; 10]);
        let readings = clocked(&stalled, 1);
        reads(&readings, "clock.median", 15.0);
        reads(&readings, "clock.device", 28.5);
    }

    /// **A burst longer than the run leaves no gap at all**, and a run that
    /// took it would report the back-to-back arm's duty cycle under a header
    /// announcing a gap — so it is refused, like every other lever that could
    /// only be dropped.
    #[test]
    fn a_burst_longer_than_the_run_is_refused() {
        assert!(
            Job::parse(
                [
                    "clock",
                    "models/small",
                    "--tokens",
                    "4",
                    "--idle",
                    "200",
                    "--burst",
                    "8"
                ]
                .map(str::to_string)
            )
            .is_err(),
            "a burst of eight fitted in four units"
        );
        // A burst of exactly the run is one gap behind one burst, which is a
        // shape rather than a mistake.
        assert!(
            Job::parse(
                [
                    "clock",
                    "models/small",
                    "--tokens",
                    "8",
                    "--idle",
                    "200",
                    "--burst",
                    "8"
                ]
                .map(str::to_string)
            )
            .is_ok(),
            "a burst of the whole run was refused"
        );
    }

    /// **A burst is units back to back and one gap behind them**, which is the
    /// shape a request has and a metronome does not. A run that slept after
    /// every unit of a burst would be reporting a gap the server never left.
    #[test]
    fn a_gap_falls_behind_a_burst_and_not_inside_one() {
        const GAP: Duration = Duration::from_millis(40);
        let mut ticking = Ticking::new(
            Shape {
                gap: Gap::After(GAP),
                burst: 3,
                warm: None,
            },
            None,
            6,
        );
        let started = Instant::now();
        for _ in 0..6 {
            ticking.tick(Duration::from_micros(100));
        }
        let whole = started.elapsed();

        for ending in [2, 5] {
            assert!(
                ticking.ticks[ending].wall >= GAP,
                "the period closing burst {} holds no gap: {:?}",
                ending / 3 + 1,
                ticking.ticks[ending].wall
            );
        }
        // Two gaps for six units, which is what separates a burst from a
        // metronome: one after each of six would be 240 ms.
        assert!(
            whole < 4 * GAP,
            "six units left more than two gaps: {whole:?}"
        );
    }

    /// **What a unit cost by where in its burst it ran is what a mean cannot
    /// say**: a part that ramps once after a gap and a part running slower for
    /// the whole period report the same mean and a different table.
    #[test]
    fn a_burst_says_what_each_of_its_positions_cost() {
        // Four bursts of three, each dear in its first position and flat behind
        // it — a ramp paid once a gap.
        let ramped = ticks(&[20_000, 15_000, 15_000].repeat(4), &[20_000; 12]);
        let readings = after_a_gap(&ramped, 3);
        assert_eq!(names(&readings), ["at1.device", "at2.device", "at3.device"]);
        reads(&readings, "at1.device", 20.0);
        reads(&readings, "at3.device", 15.0);
        // Whole bursts only: a thirteenth unit belongs to a burst that did not
        // finish, and counting it would put one more reading in the first
        // position than in the last.
        let mut uneven = ramped.clone();
        uneven.push(ticks(&[90_000], &[90_000])[0]);
        reads(&after_a_gap(&uneven, 3), "at1.device", 20.0);
        // And a run of one unit a gap has one position, which is the mean it
        // already reports.
        assert!(after_a_gap(&ramped, 1).is_empty());
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
            matches!(job, Job::Measure { shape, .. } if shape.gap == Gap::After(Duration::ZERO)),
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
        let mut ticking = Ticking::new(
            Shape {
                gap: Gap::After(Duration::ZERO),
                burst: 1,
                warm: None,
            },
            None,
            4,
        );
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
        let readings = clocked(&rising, 1);
        assert_eq!(
            names(&readings),
            [
                "part1.device",
                "part2.device",
                "part3.device",
                "part4.device",
                "part5.device",
                "clock.device",
                "clock.median",
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
        reads(&clocked(&busy, 1), "clock.duty", 92.0);
        reads(&clocked(&idled, 1), "clock.duty", 100.0 * 18.4 / 60.0);
        // And the device column is what says the two ran at the same clock,
        // which is the whole of what the idle arm is for.
        reads(&clocked(&idled, 1), "clock.device", 18.4);
        reads(&clocked(&busy, 1), "clock.device", 18.4);
    }

    /// **The parts are cut at the proportion**, so a run reports as many of them
    /// as it has units for and the shape a reader compares does not depend on
    /// whether the count happened to divide.
    #[test]
    fn a_run_reports_as_many_parts_as_it_has_units_for() {
        let parts = |units: usize| {
            clocked(&ticks(&vec![1_000; units], &vec![2_000; units]), 1)
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
        let readings = clocked(&ticks(&gpu, &[20_000; 11]), 1);
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
        let readings = clocked(&[], 1);
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
                shape: Shape {
                    gap: Gap::After(Duration::ZERO),
                    burst: 1,
                    warm: None,
                },
                prefill: 0,
                admit: ADMITTED,
                agents: AGENTS,
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
