//! `generate`: a prompt in, decoded text out as the tokens arrive.
//!
//! This is the first thing in the tree that closes the loop. Everything under it
//! has been checked against a tensor or against a recorded id; here a sentence
//! goes in and a sentence comes out, and the parts have to agree about what a
//! token is on the way.
//!
//! # The prompt is sent as it stands
//!
//! The checkpoint ships a `chat_template.jinja` that wraps a conversation in
//! `<|message_user|>`, `<|content_text|>` and the rest, and the model was
//! trained on that structure. Nothing here applies it, and that is a decision
//! rather than an omission.
//!
//! The reason is what this command is for. Every fixture in the tree — the
//! activation dump, the recorded logits, the oracle's greedy continuation — was
//! captured from an untemplated prompt, so a command that templated silently
//! could not be compared against any of them, and the one end-to-end assertion
//! that this port and mlx-vlm produce the same text would have nothing to assert
//! against. A debugging tool that rewrites its input cannot settle a
//! disagreement about the input.
//!
//! Templating is not lost by leaving it out, either, because the vocabulary
//! spells the turn markers and the tokenizer parses them back out of ordinary
//! text — `tokenizer_cases.json`'s `turn` case is a whole templated message that
//! round-trips. A caller who wants the turn structure writes it into the prompt,
//! where a reader can see it, rather than getting it applied behind one.
//!
//! What the model does either way is worth recording here, because it is what
//! the server will have to decide about. Both were measured against this
//! checkpoint.
//!
//! **Untemplated, it continues the text and nothing more.** `The lighthouse
//! keeper counted the ships that passed` is continued with ` by. The lighthouse
//! keeper counted 8` — prose in the prompt's own register, no turn marker, no
//! answer to anything. Nothing in such a prompt puts the model in a turn it
//! could end, so `<|content_model_end_sampling|>` never arrives and the budget
//! is what stops it. The stopping rule is correct and simply never fires.
//!
//! **Templated, the turn structure comes back immediately.** The same command,
//! handed the sixteen tokens the template produces for one `Hi` —
//! `<|message_system|><|content_text|>Thinking effort level:
//! 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|><|message_model|>`
//! — answers with `<|content_thinking|>The user said`, which is the framing the
//! smoke test saw the model reach for unprompted. So the template is what
//! decides whether this is a completion engine or a chat one, and a server that
//! forwards a user's text unwrapped is asking for the first.

use std::io::{self, Write};
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use inkling_core::mtp::MtpProposer;
use inkling_core::{Checkpoint, Ending, ModelCache, ModelWeights, Stop, Tokenizer};

use crate::LABEL;
use crate::args::Generate;
use crate::{backend, config};

pub fn run(args: &Generate) -> Result<()> {
    let config = config::of_checkpoint(&args.checkpoint)?;
    let tokenizer = Tokenizer::open(&args.checkpoint, &config)?;

    let ids: Vec<usize> = tokenizer
        .encode(&args.prompt)?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    if ids.is_empty() {
        bail!("the prompt encodes to no tokens, so there is nothing to continue");
    }
    eprintln!("{:<LABEL$}{} tokens", "prompt", ids.len());

    // Before the checkpoint is mapped, so that a backend this machine cannot
    // give ends the command in a millisecond rather than after a 130 GiB load.
    let gpu = backend::open(args.backend, args.numerics)?;
    let checkpoint = Checkpoint::open(&args.checkpoint)?;
    let weights = backend::weights(
        gpu.as_ref(),
        &checkpoint,
        &config.text_config,
        args.speculate,
        1,
    )?;
    let tail = backend::tail_weights(&weights, &config.text_config);
    let heads = backend::heads(gpu.as_ref(), &checkpoint, &config, args.speculate, &tail, 1)?;
    let generator = weights.generator();
    let ending = Ending {
        budget: args.max_tokens,
        eos: Some(tokenizer.eos() as usize),
    };
    // The slack every layer's convolution windows keep is the depth this run
    // speculates to, because that is the most a round can have to take back.
    let cache = &mut ModelCache::speculating(&config.text_config, args.speculate);

    // The clock starts here rather than a line earlier: the first step is the
    // prompt's prefill and is reported as such, and everything before it — the
    // mapping, the tokenizer, the cache — is setup nobody times a generation by.
    let mut text = tokenizer.stream();
    let mut out = Output::new(io::stdout().lock());
    let mut proposer = heads
        .as_ref()
        .map(|heads| MtpProposer::new(heads, generator, &weights, args.speculate));
    let mut sink = |id: usize| out.push(text.push(id as u32).map_err(anyhow::Error::from));
    let stop = match proposer.as_mut() {
        Some(proposer) => generator.speculate(cache, &ids, ending, &weights, proposer, &mut sink),
        None => generator.stream(cache, &ids, ending, &weights, &mut sink),
    };
    if let Some(proposer) = &proposer {
        report_acceptance(proposer);
    }
    // Bytes the last token left half a character with. A generation the budget
    // cut off mid-character has them, and holding them back would lose them.
    out.finish(&text.finish(), stop, tokenizer.eos())
}

/// What the heads guessed and what the model agreed with, per depth.
///
/// **Acceptance is a property of the workload, not of the engine**, and it is
/// the number that decides the depth worth running — the study measured 99.7%
/// at the first head on enumeration against 44.9% on prose. So it goes to
/// stderr beside the timings rather than into a table anywhere, and a run
/// reports its own.
fn report_acceptance<W: ModelWeights>(proposer: &MtpProposer<'_, W>) {
    let (accepted, proposed) = proposer.accepted();
    let curve: Vec<String> = proposer
        .rates()
        .iter()
        .map(|rate| format!("{:.0}%", 100.0 * rate))
        .collect();
    let banked: usize = accepted.iter().sum();
    eprintln!(
        "{:<LABEL$}{banked} of {} guesses accepted — by depth {}",
        "mtp",
        proposed.iter().sum::<usize>(),
        curve.join(" ")
    );
}

/// Where a generation goes: the text to `out` as each token arrives, what it
/// cost to stderr once it is over.
///
/// The two streams are separate so that the first can be piped. A caller that
/// wants only the continuation redirects stdout and keeps the timings on the
/// terminal, and nothing has to be stripped back out of the text.
///
/// It holds no tokenizer. What arrives here is already the text a token
/// contributed — which for a token that completed no character is none of it —
/// so that everything below is about writing and timing and can be driven by a
/// test that has no checkpoint to detokenize against.
struct Output<W: Write> {
    out: W,
    /// When the step now running began. The first covers the prompt's prefill
    /// and every later one a single decode; they are not the same price and are
    /// reported apart.
    step: Instant,
    steps: Vec<Duration>,
    /// What stopped the text from reaching `out`, if anything did. A sink
    /// cannot fail loudly from inside the loop — it can only decline the next
    /// token — so it fails quietly and says so once the loop is over.
    failed: Option<anyhow::Error>,
}

impl<W: Write> Output<W> {
    fn new(out: W) -> Self {
        Self {
            out,
            step: Instant::now(),
            steps: Vec::new(),
            failed: None,
        }
    }

    /// One token's worth of text, and the step that produced it.
    ///
    /// A `Result` because the token had to be spelled out of the vocabulary
    /// before there was any text to write, and either half can fail. The step is
    /// recorded before either is looked at: a step that ended in a failure still
    /// cost nine seconds, and a report that dropped it would be missing the
    /// prefill — the most expensive step of the run — precisely when the first
    /// token is the one that went wrong.
    fn push(&mut self, text: Result<String>) -> ControlFlow<()> {
        self.steps.push(self.step.elapsed());
        let wrote = text.and_then(|text| self.emit(&text));
        self.step = Instant::now();
        match wrote {
            Ok(()) => ControlFlow::Continue(()),
            Err(err) => {
                self.failed = Some(err);
                ControlFlow::Break(())
            }
        }
    }

    /// Written and flushed. Stdout is line-buffered and a token is rarely a
    /// line, so without the flush a nine-second step would surface whenever the
    /// model happened to end a sentence.
    ///
    /// Nothing at all for the empty string, which is what a token that only
    /// completed part of a character contributes: there is no reason to flush
    /// a stream nothing was written to.
    fn emit(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.out
            .write_all(text.as_bytes())
            .and_then(|()| self.out.flush())
            .context("writing the generated text")
    }

    /// The trailing text, the report, and whatever the sink could not say from
    /// inside the loop.
    ///
    /// The report is printed either way. A generation that died on a closed pipe
    /// still cost what it cost, and that is the first thing anyone debugging one
    /// wants to know.
    fn finish(mut self, tail: &str, stop: Stop, eos: u32) -> Result<()> {
        if self.failed.is_none() {
            if let Err(err) = self.emit(tail) {
                self.failed = Some(err);
            }
        }
        eprintln!("\n{}", self.report(stop, eos));
        match self.failed.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// What the run cost and what ended it, as it reaches stderr.
    fn report(&self, stop: Stop, eos: u32) -> String {
        let generated = self.steps.len();
        let mut lines = Vec::new();

        if let Some((prefill, decode)) = self.steps.split_first() {
            lines.push(format!("{:<LABEL$}{prefill:.2?}", "prefill"));
            // No mean over all of them: one prefill and `generated - 1` decode
            // steps produced them, and a mean over the two regimes describes
            // neither.
            if let Some(each) = decode
                .iter()
                .sum::<Duration>()
                .checked_div(decode.len() as u32)
            {
                lines.push(format!(
                    "{:<LABEL$}{each:.2?}/token over {} tokens",
                    "decode",
                    decode.len()
                ));
            }
            lines.push(format!(
                "{:<LABEL$}{:.2?}",
                "total",
                self.steps.iter().sum::<Duration>()
            ));
        }

        lines.push(format!(
            "{:<LABEL$}{}",
            "stopped",
            match stop {
                Stop::EndOfSequence =>
                    format!("on the end-of-sequence id {eos}, {generated} tokens in"),
                Stop::Budget => format!("on the budget, {generated} tokens in"),
                Stop::Sink => format!("on the failure below, {generated} tokens in"),
            }
        ));
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;

    use super::*;

    /// The checkpoint's end-of-sequence id, which the report only ever prints.
    const EOS: u32 = 200006;

    /// A stream that remembers what reached it and when, standing in for a
    /// terminal or a pipe.
    #[derive(Default)]
    struct Pipe {
        /// What each flush made visible, in order. One entry per flush is what
        /// says the text arrived a token at a time rather than in a batch at
        /// the end.
        flushed: Vec<String>,
        pending: String,
        writes: usize,
        /// Which write fails, counted from one. `None` never fails.
        breaks_at: Option<usize>,
    }

    impl Write for &mut Pipe {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if self.breaks_at == Some(self.writes) {
                return Err(io::Error::other("the pipe closed"));
            }
            self.pending
                .push_str(std::str::from_utf8(buf).expect("the text is utf8"));
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushed.push(std::mem::take(&mut self.pending));
            Ok(())
        }
    }

    /// The whole point of the command: a token's text is out and flushed before
    /// the next step is asked for, rather than accumulated and printed at the
    /// end. At nine seconds a step the difference is a minute of an empty
    /// terminal.
    #[test]
    fn every_token_is_flushed_before_the_next_one_is_asked_for() {
        let mut pipe = Pipe::default();
        let mut out = Output::new(&mut pipe);

        for text in [" by", ".", " The"] {
            assert_eq!(out.push(Ok(text.to_string())), ControlFlow::Continue(()));
        }
        out.finish("", Stop::Budget, EOS).expect("finishes");

        assert_eq!(pipe.flushed, [" by", ".", " The"]);
    }

    /// A token that completed no character contributes nothing, and nothing is
    /// not a write. Three tokens spell one `日` and only the third of them has
    /// any text to show for it.
    #[test]
    fn a_token_that_completed_no_character_writes_nothing() {
        let mut pipe = Pipe::default();
        let mut out = Output::new(&mut pipe);

        for text in ["", "", "日"] {
            assert_eq!(out.push(Ok(text.to_string())), ControlFlow::Continue(()));
        }
        out.finish("", Stop::Budget, EOS).expect("finishes");

        assert_eq!(pipe.flushed, ["日"]);
    }

    /// A pipe that closed ends the generation rather than being written to
    /// eight more times over the next seventy seconds, and the failure is the
    /// command's answer.
    #[test]
    fn a_write_that_fails_ends_the_generation_and_surfaces() {
        let mut pipe = Pipe {
            breaks_at: Some(2),
            ..Pipe::default()
        };
        let mut out = Output::new(&mut pipe);

        assert_eq!(out.push(Ok(" by".to_string())), ControlFlow::Continue(()));
        assert_eq!(out.push(Ok(".".to_string())), ControlFlow::Break(()));
        let err = out
            .finish(" The", Stop::Sink, EOS)
            .expect_err("the failure surfaces");

        assert!(format!("{err:#}").contains("the pipe closed"), "{err:#}");
        assert_eq!(pipe.flushed, [" by"], "it wrote past the failure");
    }

    /// A token the vocabulary cannot spell ends the generation the same way a
    /// closed pipe does — and the step that produced it is still counted. It is
    /// the *first* step that fails this way if any does, and that step is the
    /// prefill, so a report that dropped it would lose the most expensive
    /// measurement of the run to the one failure most likely to reach it.
    #[test]
    fn a_token_that_cannot_be_spelled_ends_the_generation_and_is_still_counted() {
        let mut pipe = Pipe::default();
        let mut out = Output::new(&mut pipe);

        let unspellable = anyhow!("no token with id 4096 in this vocabulary");
        assert_eq!(out.push(Err(unspellable)), ControlFlow::Break(()));

        let report = out.report(Stop::Sink, EOS);
        assert!(report.contains("prefill"), "{report}");
        assert!(
            report.contains("on the failure below, 1 tokens in"),
            "{report}"
        );

        let err = out
            .finish("", Stop::Sink, EOS)
            .expect_err("the failure surfaces");
        assert!(format!("{err:#}").contains("4096"), "{err:#}");
        assert!(
            pipe.flushed.is_empty(),
            "it wrote a token it could not spell"
        );
    }

    /// The first step is the prompt's prefill and every later one a decode, and
    /// the report has to say so: a mean over all of them would describe a price
    /// nothing was ever charged.
    #[test]
    fn the_report_tells_the_prefill_apart_from_the_decode_steps() {
        let mut pipe = Pipe::default();
        let mut out = Output::new(&mut pipe);
        for text in [" by", ".", " The"] {
            assert_eq!(out.push(Ok(text.to_string())), ControlFlow::Continue(()));
        }

        let report = out.report(Stop::Budget, EOS);
        assert!(report.contains("prefill"), "{report}");
        assert!(report.contains("/token over 2 tokens"), "{report}");
        assert!(report.contains("on the budget, 3 tokens in"), "{report}");
    }

    /// One token is a prefill and no decode at all, so there is no per-token
    /// decode cost to report and none is invented.
    #[test]
    fn a_generation_of_one_token_reports_no_decode_step() {
        let mut pipe = Pipe::default();
        let mut out = Output::new(&mut pipe);
        assert_eq!(out.push(Ok(" by".to_string())), ControlFlow::Continue(()));

        let report = out.report(Stop::EndOfSequence, EOS);
        assert!(report.contains("prefill"), "{report}");
        assert!(!report.contains("/token"), "{report}");
        assert!(
            report.contains(&format!("on the end-of-sequence id {EOS}, 1 tokens in")),
            "{report}"
        );
    }
}
