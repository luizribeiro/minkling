//! The model loop behind a host that owns its transport and threads.

pub mod backend;
pub mod chat;
pub mod config;
pub mod openai;
pub mod stop;
#[cfg(any(test, feature = "test-support"))]
pub mod wire;

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use inkling_core::{Checkpoint, Ending, Kept, ModelWeights, Stop, Tokenizer};
use serde_json::Value;

use crate::chat::{Channel, Channels, MARKERS, Reading, Routed};
use crate::openai::{ChatRequest, Completion, Finish};
use crate::stop::Stops;

pub use inkling_metal::Numerics;

pub const DEFAULT_REUSE_TOKENS: usize = inkling_core::DEFAULT_BOUND;

/// Where the model's weights are multiplied against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Every weight is decoded on the CPU as it is used.
    Cpu,
    /// The packed model runs on Metal.
    #[default]
    Metal,
}

impl Backend {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cpu" => Some(Self::Cpu),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }
}

pub struct Options {
    pub checkpoint: PathBuf,
    pub max_tokens: usize,
    pub numerics: Numerics,
    pub reuse_tokens: usize,
}

pub type Response = std::result::Result<Value, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Failed(String),
}

pub enum StreamEvent {
    /// A token produced no frame, but gives the host a cancellation checkpoint.
    Pulse,
    /// One or more complete SSE frames.
    Frame(String),
    /// Generation failed after the response had already started.
    Failed(String),
}

pub struct Request {
    body: Value,
    delivery: Delivery,
}

enum Delivery {
    Collected(Box<dyn FnOnce(Response) + Send>),
    Streaming {
        started: Box<dyn FnOnce(std::result::Result<(), Error>) -> bool + Send>,
        event: Box<dyn FnMut(StreamEvent) -> ControlFlow<()> + Send>,
    },
}

impl Request {
    pub fn collected(body: Value, answer: impl FnOnce(Response) + Send + 'static) -> Self {
        Self {
            body,
            delivery: Delivery::Collected(Box::new(answer)),
        }
    }

    pub fn streaming(
        body: Value,
        started: impl FnOnce(std::result::Result<(), Error>) -> bool + Send + 'static,
        event: impl FnMut(StreamEvent) -> ControlFlow<()> + Send + 'static,
    ) -> Self {
        Self {
            body,
            delivery: Delivery::Streaming {
                started: Box::new(started),
                event: Box::new(event),
            },
        }
    }
}

pub fn run(
    options: Options,
    requests: Receiver<Request>,
    ready: Sender<std::result::Result<String, String>>,
) -> Result<()> {
    let result = run_loaded(options, requests, &ready);
    if let Err(error) = &result {
        let _ = ready.send(Err(format!("{error:#}")));
    }
    result
}

fn run_loaded(
    options: Options,
    requests: Receiver<Request>,
    ready: &Sender<std::result::Result<String, String>>,
) -> Result<()> {
    let model = model_name(&options.checkpoint);
    let config = config::of_checkpoint(&options.checkpoint)?;
    let tokenizer = Tokenizer::open(&options.checkpoint, &config)?;
    let markers = markers(&tokenizer)?;
    let gpu = backend::open(Backend::Metal, options.numerics)?;

    eprintln!("loading {}", options.checkpoint.display());
    let checkpoint = Checkpoint::open(&options.checkpoint)?;
    let weights = backend::weights(gpu.as_ref(), &checkpoint, &config.text_config, 0, 1)?;
    let generator = weights.generator();
    let mut kept = Kept::new(&config.text_config, options.reuse_tokens);
    let mut served = 0_u64;

    ready
        .send(Ok(model.clone()))
        .map_err(|_| anyhow::anyhow!("the host stopped during inference startup"))?;

    for request in requests {
        served += 1;
        serve(
            request,
            options.max_tokens,
            served,
            &model,
            &tokenizer,
            &markers,
            &generator,
            &weights,
            &mut kept,
        );
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "the model references stay borrowed on the worker stack while request values vary"
)]
fn serve(
    request: Request,
    default_max_tokens: usize,
    served: u64,
    model: &str,
    tokenizer: &Tokenizer,
    markers: &[(u32, String, Reading)],
    generator: &inkling_core::Generator<'_>,
    weights: &impl ModelWeights,
    kept: &mut Kept<'_>,
) {
    let prepared = prepare(&request.body, default_max_tokens, served, model, tokenizer);

    match request.delivery {
        Delivery::Collected(answer) => {
            let response = prepared.and_then(|prepared| {
                generate(prepared, tokenizer, markers, generator, weights, kept, None)
            });
            answer(response);
        }
        Delivery::Streaming { started, mut event } => {
            let prepared = match prepared {
                Ok(prepared) => prepared,
                Err(error) => {
                    let _ = started(Err(error));
                    return;
                }
            };
            if !started(Ok(())) {
                return;
            }
            if let Err(error) = generate(
                prepared,
                tokenizer,
                markers,
                generator,
                weights,
                kept,
                Some(event.as_mut()),
            ) {
                let _ = event(StreamEvent::Failed(error.to_string()));
            }
        }
    }
}

struct Prepared {
    ids: Vec<usize>,
    ending: Ending,
    completion: Completion,
    stops: Stops,
}

fn prepare(
    body: &Value,
    default_max_tokens: usize,
    served: u64,
    model: &str,
    tokenizer: &Tokenizer,
) -> std::result::Result<Prepared, Error> {
    let asked = ChatRequest::parse(&body.to_string()).map_err(invalid)?;
    let prompt = chat::prompt(&asked.messages, asked.declared()).map_err(invalid)?;
    let ids: Vec<usize> = tokenizer
        .encode(&prompt)
        .map_err(invalid)?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let budget = asked.max_tokens(default_max_tokens);
    if budget > default_max_tokens {
        return Err(Error::Invalid(format!(
            "max_tokens must be at most {default_max_tokens}"
        )));
    }
    let ending = Ending {
        budget,
        eos: Some(tokenizer.eos() as usize),
    };
    let created = now();
    let completion = Completion::new(
        format!("chatcmpl-{created}{served:04}"),
        created,
        model.to_string(),
        ids.len(),
    )
    .reporting_usage(asked.stream && asked.wants_usage());
    Ok(Prepared {
        ids,
        ending,
        completion,
        stops: Stops::new(asked.stopping()),
    })
}

fn generate(
    prepared: Prepared,
    tokenizer: &Tokenizer,
    markers: &[(u32, String, Reading)],
    generator: &inkling_core::Generator<'_>,
    weights: &impl ModelWeights,
    kept: &mut Kept<'_>,
    stream: Option<&mut dyn FnMut(StreamEvent) -> ControlFlow<()>>,
) -> Response {
    let mut reply = Reply {
        text: tokenizer.stream(),
        channels: Channels::new(markers.iter().cloned()),
        completion: prepared.completion,
        stops: prepared.stops,
        struck: false,
        failed: None,
        cancelled: false,
        stream,
    };

    if reply.open().is_break() {
        return Ok(Value::Null);
    }
    let generated = kept.turn(generator, weights, &prepared.ids, prepared.ending, |id| {
        reply.push(id)
    });
    reply.finish(generated.stop)
}

struct Reply<'a, 'stream> {
    text: inkling_core::Detokenizer<'a>,
    channels: Channels,
    completion: Completion,
    stops: Stops,
    struck: bool,
    failed: Option<String>,
    cancelled: bool,
    stream: Option<&'stream mut dyn FnMut(StreamEvent) -> ControlFlow<()>>,
}

impl Reply<'_, '_> {
    fn open(&mut self) -> ControlFlow<()> {
        let frame = self.completion.opening();
        self.emit(Some(frame))
    }

    fn push(&mut self, id: usize) -> ControlFlow<()> {
        let decoded = match self.text.push(id as u32) {
            Ok(decoded) => decoded,
            Err(error) => {
                self.failed = Some(error.to_string());
                return ControlFlow::Break(());
            }
        };
        let routed = self.channels.route(id as u32, &decoded);
        let routed = self.cut(routed);
        let frame = self.completion.push(routed);
        if self.emit(frame).is_break() || self.struck {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn finish(mut self, stop: Stop) -> Response {
        if self.cancelled {
            return Ok(Value::Null);
        }
        if let Some(message) = self.failed.take() {
            return Err(Error::Failed(message));
        }

        let tail = self.text.finish();
        let routed = self.channels.finish(&tail);
        let routed = self.cut(routed);
        let frame = self.completion.tail(routed);
        if self.emit(frame).is_break() {
            return Ok(Value::Null);
        }

        let held = self.stops.finish();
        if !held.is_empty() {
            let frame = self.completion.tail(Routed::Text(Channel::Content, held));
            if self.emit(frame).is_break() {
                return Ok(Value::Null);
            }
        }

        let finish = match (self.struck, stop, self.completion.called()) {
            (true, _, _) => Finish::Stop,
            (false, Stop::EndOfSequence, true) => Finish::ToolCalls,
            (false, Stop::EndOfSequence, false) => Finish::Stop,
            (false, Stop::Budget | Stop::Sink, _) => Finish::Length,
        };
        let closing = self.completion.closing(finish);
        if self.emit(Some(closing)).is_break() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&self.completion.collected(finish))
            .map_err(|error| Error::Failed(error.to_string()))
    }

    fn emit(&mut self, frame: Option<String>) -> ControlFlow<()> {
        let Some(stream) = &mut self.stream else {
            return ControlFlow::Continue(());
        };
        let event = match frame {
            Some(frame) => StreamEvent::Frame(frame),
            None => StreamEvent::Pulse,
        };
        if stream(event).is_break() {
            self.cancelled = true;
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn cut(&mut self, routed: Routed) -> Routed {
        if self.struck {
            return Routed::Nothing;
        }
        let Routed::Text(Channel::Content, text) = routed else {
            return routed;
        };
        let taken = self.stops.take(&text);
        self.struck = taken.struck;
        Routed::Text(Channel::Content, taken.shown)
    }
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::Invalid(error.to_string())
}

fn markers(tokenizer: &Tokenizer) -> Result<Vec<(u32, String, Reading)>> {
    let eos = tokenizer.eos();
    MARKERS
        .into_iter()
        .map(|(marker, reading)| {
            let id = match marker {
                "<|content_model_end_sampling|>" => eos,
                _ => tokenizer
                    .id_of(marker)
                    .with_context(|| format!("the tokenizer has no {marker}"))?,
            };
            Ok((id, marker.to_string(), reading))
        })
        .collect()
}

fn model_name(checkpoint: &Path) -> String {
    checkpoint
        .file_name()
        .unwrap_or(checkpoint.as_os_str())
        .to_string_lossy()
        .into_owned()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

/// The width shared diagnostic labels are padded to.
pub(crate) const LABEL: usize = 9;
