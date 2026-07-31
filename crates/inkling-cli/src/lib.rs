//! The commands the binary offers, as a library it drives.
//!
//! Nothing here is a second abstraction over `inkling-core`. It is the same
//! modules the binary had, in a target that a test can link against — which is
//! what a server needs and a pair of print-and-exit commands did not. A request
//! handler is worth driving from a test directly, in-process, rather than by
//! binding a port and speaking HTTP to it, and only a library can be called that
//! way.
//!
//! It is also the shape the split the README promises takes: `inkling-serve`
//! moves out of here once the batching scheduler is more than a request loop,
//! and moving a module is a smaller thing than lifting one out of a binary.

pub mod args;
pub mod chat;
pub mod config;
pub mod generate;
pub mod inspect;
pub mod openai;
pub mod serve;

/// Reading a response the way a client does. Behind a feature so that the
/// checkpoint-gated test in `tests/` — which links this crate from outside and
/// cannot see a `cfg(test)` module — and the unit tests here share one copy.
#[cfg(feature = "test-support")]
pub mod wire;
