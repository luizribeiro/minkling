//! A kept cache against the cache it replaces: **the same tokens, or it is not
//! an optimisation.**
//!
//! Everything the arrangement is made of is settled where it lives — the mark
//! against the synthetic stack and against a run of layers on the device, the
//! matching and the bound against their own cases. What only a real checkpoint
//! can settle is the claim the whole thing rests on: that a conversation served
//! against a cache carried from the request before it produces the ids a
//! conversation served from nothing produces, over a session of several turns
//! and every token of every one of them.
//!
//! **Ids and not text.** A reply that reads the same is a reply that might have
//! moved a token; the ids are what the argmax named, and equality on them is the
//! whole claim. This is what says a kept cache is a latency optimisation rather
//! than an approximation — the same thing `speculation_changes_no_token` says
//! about the other one.
//!
//! Gated on `INKLINGRS_CHECKPOINT`; unset, it reports a skip and passes.
//! `just test-full` sets it.
//!
//! **Nothing here asserts a duration.** This runs beside a whole suite, which is
//! what `.config/nextest.toml` says a measurement must not be — the timings are
//! `bench session`'s, alternating, in a sitting of their own.

use std::path::PathBuf;

use inkling_cli::args::Backend;
use inkling_cli::{backend, config, session};
use inkling_core::workload::{STRUCTURED_PROMPT, Session};
use inkling_core::{Checkpoint, DEFAULT_BOUND, Kept, Tokenizer};
use inkling_metal::Numerics;

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

/// How long a session this case drives.
///
/// Short in the opening and full in the turns: what has to be exercised is the
/// matching, the delta prefill and the resume, once per turn, and every one of
/// those runs at 320 tokens exactly as it runs at 16384. A session opening at
/// the length `bench session` measures would be minutes inside a suite that has
/// a hundred other cases to run.
const TURNS: Session = Session {
    opening: 320,
    turns: 4,
    added: 48,
    generated: 8,
};

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os(CHECKPOINT_VAR).map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: {CHECKPOINT_VAR} is unset");
    }
    dir
}

/// The claim, and the check that the two arms were actually different arms.
#[test]
fn a_session_served_from_a_kept_cache_produces_the_tokens_a_cold_one_produces() {
    let Some(dir) = checkpoint_dir() else { return };

    let config = config::of_checkpoint(&dir).expect("a config");
    let tokenizer = Tokenizer::open(&dir, &config).expect("a tokenizer");
    let ids: Vec<usize> = tokenizer
        .encode(STRUCTURED_PROMPT)
        .expect("the workload prompt encodes")
        .into_iter()
        .map(|id| id as usize)
        .collect();

    let gpu = backend::open(Backend::default(), Numerics::default()).expect("a backend");
    let ckpt = Checkpoint::open(&dir).expect("the checkpoint opens");
    let weights =
        backend::weights(gpu.as_ref(), &ckpt, &config.text_config, 0, 1).expect("the weights wrap");
    let generator = weights.generator();

    // Cold first and warm second, against the same weights and the same device.
    // A cache that is never kept is what the server did before any of this
    // existed, which is what makes it the arm rather than a contrivance.
    let cold = {
        let mut kept = Kept::new(&config.text_config, 0);
        session::run(&generator, &weights, &mut kept, TURNS, &ids)
    };
    let warm = {
        let mut kept = Kept::new(&config.text_config, DEFAULT_BOUND);
        session::run(&generator, &weights, &mut kept, TURNS, &ids)
    };

    // The two arms were different arms. Without this the case would pass just as
    // well against a `Kept` that never matched anything.
    let reused: Vec<usize> = warm.iter().map(|turn| turn.reused).collect();
    assert_eq!(reused[0], 0, "the first turn had nothing to reuse");
    for (turn, held) in reused.iter().enumerate().skip(1) {
        assert_eq!(
            *held,
            warm[turn - 1].prompt - 1,
            "turn {turn} reused something other than the whole of the turn before it"
        );
    }
    assert!(
        cold.iter().all(|turn| turn.reused == 0),
        "the cold arm kept a cache: {:?}",
        cold.iter().map(|turn| turn.reused).collect::<Vec<_>>()
    );

    // The claim.
    assert_eq!(
        session::tokens(&warm),
        session::tokens(&cold),
        "a kept cache moved a token"
    );

    // And that the session it produced them over was a session: every turn
    // decoded its whole budget, so none of the equality above is two empty
    // sequences agreeing.
    for (turn, produced) in warm.iter().map(|turn| &turn.produced).enumerate() {
        assert_eq!(produced.len(), TURNS.generated, "turn {turn} decoded short");
    }
}
