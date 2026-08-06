//! The two halves of `bench` against each other: an arm that measures, and the
//! harness that alternates two of them.
//!
//! The harness' own arithmetic — the means, the ranges, whether they overlap,
//! how many pairs moved the same way — is tested where it lives, against arms
//! that are shell scripts printing numbers a test chose. What only the binary
//! can settle is that the *real* arm speaks the same protocol: that
//! `bench decode` prints lines the harness can read, under the names a report
//! and a milestone quote. A change to either side that broke that would leave
//! every unit test passing and `just bench` reporting nothing.
//!
//! Gated on `INKLINGRS_CHECKPOINT` like every other case that needs weights;
//! unset, it reports a skip and passes. `just test-full` sets it.
//!
//! **Nothing here asserts a duration.** These runs decode two tokens beside a
//! whole test suite, which is what `.config/nextest.toml` says a measurement
//! must not be — what is asserted is that a number arrived and under which name.

use std::path::PathBuf;
use std::process::{Command, Output};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os(CHECKPOINT_VAR).map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: {CHECKPOINT_VAR} is unset");
    }
    dir
}

fn bench(args: &[&str]) -> Output {
    let ran = Command::new(env!("CARGO_BIN_EXE_bench"))
        .args(args)
        .output()
        .expect("the binary runs");
    assert!(
        ran.status.success(),
        "bench {args:?} exited {} saying:\n{}",
        ran.status,
        String::from_utf8_lossy(&ran.stderr)
    );
    ran
}

fn stdout(output: &Output) -> &str {
    std::str::from_utf8(&output.stdout).expect("the readings are utf8")
}

/// Every reading, as the harness reads them: `name value unit`, and a value that
/// is a number. Parsed here rather than imported, because what this case is
/// about is the two sides agreeing about a format neither of them owns.
fn readings(printed: &str) -> Vec<(String, f64, String)> {
    printed
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let [name, value, unit] = fields[..] else {
                panic!("{line:?} is not `name value unit`")
            };
            (
                name.to_string(),
                value.parse().unwrap_or_else(|_| panic!("{line:?}")),
                unit.to_string(),
            )
        })
        .collect()
}

fn named(printed: &str) -> Vec<String> {
    readings(printed)
        .into_iter()
        .map(|(name, ..)| name)
        .collect()
}

/// Two tokens, which is the fewest that has a decode step in it at all: the
/// first is the prompt's prefill and the second is the step this reports. What
/// is being checked is the wiring, and sixty-two more of them would only make it
/// a minute longer.
const DECODED: &str = "2";

/// How many a sweep decodes, which is not two: a round proposes `k` and verifies
/// `k + 1`, so a budget that cannot hold a whole round never runs one and there
/// is no acceptance to report. Four is the fewest that has a round in it.
const SPECULATED: &str = "4";

#[test]
fn a_decode_run_reports_a_step_and_what_the_device_executed_for() {
    let Some(dir) = checkpoint_dir() else { return };
    let ran = bench(&["decode", "--tokens", DECODED, &dir.display().to_string()]);

    let readings = readings(stdout(&ran));
    assert_eq!(
        readings
            .iter()
            .map(|(name, ..)| name.as_str())
            .collect::<Vec<_>>(),
        ["decode", "device"],
        "{}",
        stdout(&ran)
    );
    for (name, value, unit) in readings {
        assert!(value > 0.0, "{name} is {value}");
        assert_eq!(unit, "ms", "{name} is in {unit}");
    }
}

/// A prefill is the one measurement whose prompt is the parameter, and it
/// reports both the first pass at a length and the second — the first is the one
/// that faults its pages in.
#[test]
fn a_prefill_run_reports_the_warm_pass_and_the_cold_one() {
    let Some(dir) = checkpoint_dir() else { return };
    let ran = bench(&["prefill", "--tokens", "64", &dir.display().to_string()]);

    assert_eq!(named(stdout(&ran)), ["prefill", "device", "cold"]);
}

/// The sweep is what a milestone's `k` table is read off, so the names it prints
/// are the table's columns: the step, what the device executed for, the speedup
/// against this run's own `k = 0`, the tokens a round banked, and acceptance at
/// every depth the round guessed at.
#[test]
fn a_sweep_reports_a_row_a_depth_with_acceptance_at_every_depth() {
    let Some(dir) = checkpoint_dir() else { return };
    let ran = bench(&[
        "sweep",
        "--tokens",
        SPECULATED,
        "--depth",
        "1",
        &dir.display().to_string(),
    ]);

    assert_eq!(
        named(stdout(&ran)),
        [
            "k0",
            "k0.device",
            "k0.speedup",
            "k0.tokens",
            "k1",
            "k1.device",
            "k1.speedup",
            "k1.tokens",
            "k1.accept1",
        ]
    );
    // The unspeculated row is divided by itself, which is what says the speedups
    // are this run's own rather than another sitting's.
    let speedup = readings(stdout(&ran))
        .into_iter()
        .find(|(name, ..)| name == "k0.speedup")
        .expect("the unspeculated speedup");
    assert!((speedup.1 - 1.0).abs() < 1e-9, "{speedup:?}");
}

/// The whole arrangement, over one pair: the harness runs the arm twice, reads
/// what it printed, and reports a row per reading. Against itself, because what
/// is being settled is that the two halves fit — a difference between two runs
/// of one binary is the machine's own state and is not a claim about anything.
#[test]
fn the_harness_alternates_the_real_arm_against_itself() {
    let Some(dir) = checkpoint_dir() else { return };
    let arm = env!("CARGO_BIN_EXE_bench");
    let ran = bench(&[
        "alternate",
        "--pairs",
        "1",
        arm,
        arm,
        "--",
        "decode",
        "--tokens",
        DECODED,
        &dir.display().to_string(),
    ]);

    let report = stdout(&ran);
    for row in ["decode", "device"] {
        assert!(report.contains(row), "no {row} row in:\n{report}");
    }
    assert!(report.contains("1 of 1"), "{report}");
}

/// **The clock measurement reports a part at a time and a whole beside them**,
/// which is what a drift is read off — and the gap it was told to leave falls
/// inside the period rather than beside it, or the duty cycle it reports is
/// about a lever nobody pulled.
///
/// Four units, because what this asserts is the shape of the report rather than
/// anything about the machine, and a short decode run reaches every reading a
/// long one does.
///
/// **The gap is the one duration asserted here and it is asserted as a floor**,
/// which is what keeps this inside the rule at the top of this file: a period
/// holding a 200 ms sleep is at least 200 ms because `sleep` says so, on any
/// machine and under any load. What the gap does to the *pair* of arms is a
/// comparison of two durations and belongs where the arithmetic lives — see
/// `the_duty_cycle_is_the_device_against_the_period_the_gap_is_inside`. This run
/// decodes beside a whole suite, and the reading that failed here first was the
/// unidled arm's wall at 510 ms against a device of 21.
#[test]
fn a_clock_run_reports_every_part_and_the_period_the_gap_is_inside() {
    let Some(dir) = checkpoint_dir() else { return };
    const GAP: f64 = 200.0;
    let ran = bench(&[
        "clock",
        "--tokens",
        "4",
        "--idle",
        "200",
        &dir.display().to_string(),
    ]);
    let reported = readings(stdout(&ran));

    assert_eq!(
        named(stdout(&ran)),
        [
            "part1.device",
            "part2.device",
            "part3.device",
            "part4.device",
            "clock.device",
            "clock.wall",
            "clock.duty",
            "clock.drift",
        ]
    );
    let value = |name: &str| {
        reported
            .iter()
            .find(|(had, ..)| had == name)
            .unwrap_or_else(|| panic!("{name} is reported"))
            .1
    };
    assert!(
        value("clock.wall") > GAP,
        "the gap is inside the period: {reported:?}"
    );
    // And the device time is the work rather than the period, so it is under the
    // gap the period is over — which is the separation the whole measurement
    // rests on.
    assert!(
        value("clock.device") < GAP && value("clock.device") > 0.0,
        "the gap is beside the work rather than inside it: {reported:?}"
    );
}

/// **The third unit is a step of a batch**, which is what puts a gap of a fixed
/// length beside two very different amounts of work — the arrangement that says
/// whether a slower clock is a function of the gap or of the occupancy the gap
/// produced.
///
/// Two sequences and two steps, because what this asserts is that the arm runs,
/// reports under the same names, and says the width it repeated.
#[test]
fn a_clock_run_over_a_batch_reports_the_width_it_repeated() {
    let Some(dir) = checkpoint_dir() else { return };
    let ran = bench(&[
        "clock",
        "--batch",
        "2",
        "--tokens",
        "2",
        &dir.display().to_string(),
    ]);

    assert_eq!(
        named(stdout(&ran)),
        [
            "part1.device",
            "part2.device",
            "clock.device",
            "clock.wall",
            "clock.duty",
            "clock.drift"
        ]
    );
    let said = String::from_utf8_lossy(&ran.stderr).to_string();
    assert!(said.contains("2-wide decode steps"), "{said}");
}

/// **The other unit is a whole prefill**, which is the arm whose repetitions are
/// the same work every time and so the arm a drift means anything under — and
/// nothing else in this file runs it.
///
/// A prompt of 32 tokens and two of them, because what this asserts is that the
/// arm runs and reports rather than anything about how long it took.
#[test]
fn a_clock_run_over_prefills_reports_the_same_readings() {
    let Some(dir) = checkpoint_dir() else { return };
    let ran = bench(&[
        "clock",
        "--prefill",
        "32",
        "--tokens",
        "2",
        &dir.display().to_string(),
    ]);

    assert_eq!(
        named(stdout(&ran)),
        [
            "part1.device",
            "part2.device",
            "clock.device",
            "clock.wall",
            "clock.duty",
            "clock.drift"
        ]
    );
    // The durations, and not the drift beside them: a part slower than the first
    // is a positive drift and a part faster than it is a negative one, which is
    // the whole of what that column is for.
    for (name, value, unit) in readings(stdout(&ran)) {
        if unit == "ms" {
            assert!(value > 0.0, "{name} is {value}");
        }
    }
    // The prompt is the unit, so the header says which length was repeated
    // rather than a range of keys — the two arms report different things there
    // and a reader has to be able to tell which one they are holding.
    let said = String::from_utf8_lossy(&ran.stderr).to_string();
    assert!(said.contains("prefills of 32 tokens"), "{said}");
}
