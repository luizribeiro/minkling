//! A simulated coding session, driven turn by turn and timed.
//!
//! **The figure a kept cache is worth cannot be taken on one prompt.** A prefill
//! of a given length says what one prompt costs; what a user feels is the same
//! conversation coming back turn after turn with a little added each time, and
//! there is no "between requests" in a measurement of one. So this drives
//! several, through [`Kept::turn`] — the same function the server's request loop
//! calls, so that what is measured is what is served.
//!
//! What the shape of the session is belongs to
//! [`Session`](inkling_core::workload::Session), beside the rest of what this
//! repo measures over: two arms that disagreed about the workload would be two
//! measurements, however alike the tables looked.

use std::time::{Duration, Instant};

use inkling_core::workload::Session;
use inkling_core::{Ending, Generator, ModelWeights};

use inkling_core::Kept;

/// What one turn of a session cost, and what it produced.
#[derive(Debug, Clone)]
pub struct Turn {
    /// Tokens in the prompt this turn sent, which grows every turn.
    pub prompt: usize,
    /// Tokens of that prompt the cache already held, which is the whole of what
    /// this measures: zero on a cold turn and nearly the prompt on a warm one.
    pub reused: usize,
    /// What the client waited, prompt in to last token out.
    pub wall: Duration,
    /// What it waited for the *first* token, which is the prefill and one decode
    /// step. The row a kept cache moves.
    pub first: Duration,
    /// The ids the model produced, which is what says a kept cache changed no
    /// answer.
    pub produced: Vec<usize>,
    /// What the arrangement cost the turn whether it hit or missed — see
    /// [`Served::bookkeeping`](crate::kept::Served::bookkeeping).
    pub bookkeeping: Duration,
}

/// The whole session, turn by turn.
///
/// `ids` is the pool the prompts are tiled out of — this repo's own workload
/// prompt, so that a session and a prefill are measured over the same tokens.
pub fn run(
    generator: &Generator<'_>,
    weights: &impl ModelWeights,
    kept: &mut Kept<'_>,
    session: Session,
    ids: &[usize],
) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::with_capacity(session.turns);
    let mut produced: Vec<Vec<usize>> = Vec::with_capacity(session.turns);

    for turn in 0..session.turns {
        let prompt = session.prompt(ids, turn, &produced);
        let ending = Ending {
            budget: session.generated,
            eos: None,
        };

        let started = Instant::now();
        let mut first = None;
        let mut reply = Vec::with_capacity(session.generated);
        let served = kept.turn(generator, weights, &prompt, ending, |id| {
            first.get_or_insert_with(|| started.elapsed());
            reply.push(id);
            std::ops::ControlFlow::Continue(())
        });

        turns.push(Turn {
            prompt: prompt.len(),
            reused: served.reused,
            wall: started.elapsed(),
            first: first.unwrap_or_default(),
            produced: reply.clone(),
            bookkeeping: served.bookkeeping,
        });
        produced.push(reply);
    }
    turns
}

/// Every token the session produced, in order — what two arms are held against
/// each other by.
pub fn tokens(turns: &[Turn]) -> Vec<usize> {
    turns
        .iter()
        .flat_map(|turn| turn.produced.clone())
        .collect()
}
