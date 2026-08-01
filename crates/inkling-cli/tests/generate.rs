//! The `generate` command as a caller meets it: argv in, bytes on two streams
//! out.
//!
//! Everything the command is made of is tested where it lives — the loop against
//! the synthetic stack, the tokenizer against the whole vocabulary, the argument
//! parser against its own errors. What only the binary can settle is that they
//! were wired to each other: that the text a caller types reaches the tokenizer,
//! that the ids it makes reach the model, and that what comes back is text on
//! stdout rather than a debug print of a vector.
//!
//! The case that needs weights sets `INKLINGRS_CHECKPOINT` to a checkpoint
//! directory; unset, it reports a skip and passes, the way
//! `inkling-core`'s own checkpoint tests do. `just test-full` sets it.

use std::path::PathBuf;
use std::process::{Command, Output};

use inkling_core::Tokenizer;
use inkling_core::fixture::{self, ACTIVATIONS, indices};
use tempfile::TempDir;

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os(CHECKPOINT_VAR).map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: {CHECKPOINT_VAR} is unset");
    }
    dir
}

fn inklingrs(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_inklingrs"))
        .args(args)
        .output()
        .expect("the binary runs")
}

fn stdout(output: &Output) -> &str {
    std::str::from_utf8(&output.stdout).expect("the generated text is utf8")
}

fn stderr(output: &Output) -> &str {
    std::str::from_utf8(&output.stderr).expect("the report is utf8")
}

/// An id tensor of the activation dump, as the tokenizer wants them.
fn recorded(activations: &inkling_core::Checkpoint, name: &str) -> Vec<u32> {
    indices(&fixture::tensor(activations, name))
        .iter()
        .map(|&id| id as u32)
        .collect()
}

/// How many tokens the end-to-end case decodes.
///
/// Three: the recorded prompt prefilled, then two decode steps, which is about
/// eighty seconds. `inkling-core` already asserts all eight ids the fixture
/// recorded against the oracle; what this adds is the text at either end of
/// them, and five more tokens would not make that truer — they would only make
/// it a minute longer.
const GENERATED: usize = 3;

/// The milestone, as a command: the sentence the fixtures were captured from
/// goes in, and the text mlx-vlm continued it with comes out.
///
/// The prompt is passed as *text*, which is the part no earlier test could
/// reach. Every fixture in the tree holds ids, and the command holds none — it
/// encodes what it was handed — so this is where the tokenizer's encode and the
/// recorded `input_ids` have to be the same eight numbers. They are asserted to
/// be, first and separately, because a mismatch there and a wrong continuation
/// are different faults and the second would hide the first.
///
/// **Both backends, and that is the point of running it twice.** `--backend` is
/// what keeps the CPU path selectable, and a path nothing runs is a path that
/// stops working quietly. The two are asserted against the same recorded text
/// rather than against each other, so a run where both drifted the same way is
/// still a failure. The cost is one more prefill: about eighty seconds a
/// backend, of which the stack is nearly all.
#[test]
fn either_backend_writes_the_oracles_continuation_of_the_recorded_prompt() {
    let Some(dir) = checkpoint_dir() else { return };
    let tokenizer =
        Tokenizer::open(&dir, &fixture::config(&dir)).expect("the checkpoint's tokenizer opens");
    let activations = fixture::open(ACTIVATIONS);

    let ids = recorded(&activations, "input_ids");
    let prompt = tokenizer.decode(&ids).expect("the recorded ids decode");
    assert_eq!(
        tokenizer.encode(&prompt).expect("the prompt encodes"),
        ids,
        "the recorded prompt does not survive being spelled out and read back"
    );

    let oracle = recorded(&activations, "greedy_continuation");
    let want = tokenizer
        .decode(&oracle[..GENERATED])
        .expect("the continuation decodes");

    for backend in ["metal", "cpu"] {
        let output = inklingrs(&[
            "generate",
            dir.to_str().expect("a printable checkpoint path"),
            "--prompt",
            &prompt,
            "--max-tokens",
            &GENERATED.to_string(),
            "--backend",
            backend,
        ]);
        eprintln!("{prompt:?} on {backend} ->\n{}", stderr(&output));

        assert!(output.status.success(), "{}", stderr(&output));
        assert_eq!(stdout(&output), want, "on {backend}");
        assert!(
            stderr(&output).contains(backend),
            "the report does not say which backend ran: {}",
            stderr(&output)
        );
    }
}

/// A path that is not a checkpoint, which is what a typo makes. It has to be
/// refused by name rather than by a panic out of a loader.
#[test]
fn a_directory_that_holds_no_config_is_refused() {
    let dir = TempDir::new().expect("a temporary directory");
    let output = inklingrs(&[
        "generate",
        dir.path().to_str().expect("a printable path"),
        "--prompt",
        "Once",
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("config.json"),
        "{}",
        stderr(&output)
    );
    assert!(stdout(&output).is_empty(), "it wrote text anyway");
}

/// An invocation nobody could have meant exits differently from one that ran and
/// failed, so that a script can tell a typo from a checkpoint that would not
/// load. The parser's own cases say which message; this says what it costs.
#[test]
fn a_misuse_exits_apart_from_a_failure() {
    let output = inklingrs(&["generate", "models/Inkling-Small-mxfp4"]);

    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("usage:"), "{}", stderr(&output));
}
