use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;

use anyhow::{Context, Result, anyhow};
use inkling_cli::inference::{self, Options, Request};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::api::{Completion, Inference, InferenceError};

const QUEUED_REQUESTS: usize = 16;

#[derive(Clone)]
pub struct Client {
    model: String,
    requests: SyncSender<Request>,
}

impl Client {
    pub fn start(options: Options) -> Result<Self> {
        let (send, receive) = mpsc::sync_channel(QUEUED_REQUESTS);
        let (ready, started) = mpsc::channel();

        thread::Builder::new()
            .name("inkling-inference".to_string())
            .spawn(move || {
                if let Err(error) = inference::run(options, receive, ready) {
                    eprintln!("inference stopped: {error:#}");
                }
            })
            .context("starting the inference worker")?;

        let model = started
            .recv()
            .context("the inference worker stopped during startup")?
            .map_err(|message| anyhow!(message))?;

        Ok(Self {
            model,
            requests: send,
        })
    }
}

impl Inference for Client {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn complete(&self, body: Value) -> Completion<'_> {
        let (send, receive) = oneshot::channel();
        let request = Request::new(body, move |response| {
            let _ = send.send(response);
        });

        match self.requests.try_send(request) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Box::pin(std::future::ready(Err(InferenceError::unavailable(
                    "the inference queue is full",
                ))));
            }
            Err(TrySendError::Disconnected(_)) => {
                return Box::pin(std::future::ready(Err(InferenceError::unavailable(
                    "the inference worker is not running",
                ))));
            }
        }

        Box::pin(answer(receive))
    }
}

async fn answer(receive: oneshot::Receiver<inference::Response>) -> Result<Value, InferenceError> {
    let response = receive
        .await
        .map_err(|_| InferenceError::unavailable("the inference worker stopped"))?;
    response.map_err(|error| match error {
        inference::Error::Invalid(message) => InferenceError::bad_request(message),
        inference::Error::Failed(message) => InferenceError::internal(message),
    })
}
