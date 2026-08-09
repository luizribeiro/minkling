//! The model loop behind a host that owns its transport and threads.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use inkling_core::{Checkpoint, Ending, Kept, ModelWeights, Stop, Tokenizer};
use serde_json::Value;

use crate::args::Backend;
use crate::chat::{self, Channel, Channels, MARKERS, Reading, Routed};
use crate::openai::{ChatRequest, Completion, Finish};
use crate::stop::Stops;
use crate::{backend, config};

pub use inkling_metal::Numerics;

pub const DEFAULT_REUSE_TOKENS: usize = inkling_core::DEFAULT_BOUND;

pub struct Options {
    pub checkpoint: PathBuf,
    pub max_tokens: usize,
    pub numerics: Numerics,
    pub reuse_tokens: usize,
}

pub type Response = std::result::Result<Value, Error>;

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Failed(String),
}

pub struct Request {
    body: Value,
    answer: Box<dyn FnOnce(Response) + Send>,
}

impl Request {
    pub fn new(body: Value, answer: impl FnOnce(Response) + Send + 'static) -> Self {
        Self {
            body,
            answer: Box::new(answer),
        }
    }

    fn answer(self, response: Response) {
        (self.answer)(response);
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
        let response = complete(
            &request.body,
            options.max_tokens,
            served,
            &model,
            &tokenizer,
            &markers,
            &generator,
            &weights,
            &mut kept,
        );
        request.answer(response);
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "the model references stay borrowed on the worker stack while request values vary"
)]
fn complete(
    body: &Value,
    default_max_tokens: usize,
    served: u64,
    model: &str,
    tokenizer: &Tokenizer,
    markers: &[(u32, String, Reading)],
    generator: &inkling_core::Generator<'_>,
    weights: &impl ModelWeights,
    kept: &mut Kept<'_>,
) -> Response {
    let asked = ChatRequest::parse(&body.to_string()).map_err(invalid)?;
    if asked.stream {
        return Err(Error::Invalid(
            "streaming responses are not implemented by minkling yet".to_string(),
        ));
    }

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
    );
    let mut reply = Reply {
        text: tokenizer.stream(),
        channels: Channels::new(markers.iter().cloned()),
        completion,
        stops: Stops::new(asked.stopping()),
        struck: false,
        failed: None,
    };

    let generated = kept.turn(generator, weights, &ids, ending, |id| reply.push(id));
    reply.finish(generated.stop)
}

struct Reply<'a> {
    text: inkling_core::Detokenizer<'a>,
    channels: Channels,
    completion: Completion,
    stops: Stops,
    struck: bool,
    failed: Option<String>,
}

impl Reply<'_> {
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
        let _ = self.completion.push(routed);
        match self.struck {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    }

    fn finish(mut self, stop: Stop) -> Response {
        if let Some(message) = self.failed.take() {
            return Err(Error::Failed(message));
        }

        let tail = self.text.finish();
        let routed = self.channels.finish(&tail);
        let routed = self.cut(routed);
        let _ = self.completion.tail(routed);

        let held = self.stops.finish();
        if !held.is_empty() {
            let _ = self.completion.tail(Routed::Text(Channel::Content, held));
        }

        let finish = match (self.struck, stop, self.completion.called()) {
            (true, _, _) => Finish::Stop,
            (false, Stop::EndOfSequence, true) => Finish::ToolCalls,
            (false, Stop::EndOfSequence, false) => Finish::Stop,
            (false, Stop::Budget | Stop::Sink, _) => Finish::Length,
        };
        serde_json::from_str(&self.completion.collected(finish))
            .map_err(|error| Error::Failed(error.to_string()))
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
