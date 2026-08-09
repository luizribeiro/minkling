use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;

use anyhow::{Context, Result, anyhow};
use futures_util::stream;
use inkling_cli::inference::{self, Options, Request, StreamEvent};
use serde_json::Value;
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::api::{Completion, CompletionFuture, FrameStream, Inference, InferenceError};

const QUEUED_REQUESTS: usize = 16;
const BUFFERED_CHUNKS: usize = 8;

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

    fn complete(&self, body: Value) -> CompletionFuture<'_> {
        if body["stream"].as_bool() == Some(true) {
            return self.stream(body);
        }

        let (send, receive) = oneshot::channel();
        let request = Request::collected(body, move |response| {
            let _ = send.send(response);
        });

        if let Err(error) = self.enqueue(request) {
            return Box::pin(std::future::ready(Err(error)));
        }

        Box::pin(async move { answer(receive).await.map(Completion::Collected) })
    }
}

impl Client {
    fn stream(&self, body: Value) -> CompletionFuture<'_> {
        let (ready, started) = oneshot::channel();
        let (send, receive) = async_mpsc::channel(BUFFERED_CHUNKS);
        let request = Request::streaming(
            body,
            move |result| ready.send(result).is_ok(),
            move |event| deliver(&send, event),
        );

        if let Err(error) = self.enqueue(request) {
            return Box::pin(std::future::ready(Err(error)));
        }

        Box::pin(async move {
            let started = started
                .await
                .map_err(|_| InferenceError::unavailable("the inference worker stopped"))?;
            started.map_err(inference_error)?;
            Ok(Completion::Streaming(frames(receive)))
        })
    }

    fn enqueue(&self, request: Request) -> Result<(), InferenceError> {
        match self.requests.try_send(request) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                Err(InferenceError::unavailable("the inference queue is full"))
            }
            Err(TrySendError::Disconnected(_)) => Err(InferenceError::unavailable(
                "the inference worker is not running",
            )),
        }
    }
}

fn deliver(
    send: &async_mpsc::Sender<Result<String, String>>,
    event: StreamEvent,
) -> std::ops::ControlFlow<()> {
    match event {
        StreamEvent::Pulse => match send.is_closed() {
            true => std::ops::ControlFlow::Break(()),
            false => std::ops::ControlFlow::Continue(()),
        },
        StreamEvent::Frame(frame) => match send.try_send(Ok(frame)) {
            Ok(()) => std::ops::ControlFlow::Continue(()),
            Err(_) => std::ops::ControlFlow::Break(()),
        },
        StreamEvent::Failed(message) => {
            let _ = send.try_send(Err(message));
            std::ops::ControlFlow::Break(())
        }
    }
}

fn frames(receive: async_mpsc::Receiver<Result<String, String>>) -> FrameStream {
    Box::pin(stream::unfold(receive, |mut receive| async move {
        receive.recv().await.map(|frame| (frame, receive))
    }))
}

fn inference_error(error: inference::Error) -> InferenceError {
    match error {
        inference::Error::Invalid(message) => InferenceError::bad_request(message),
        inference::Error::Failed(message) => InferenceError::internal(message),
    }
}

async fn answer(receive: oneshot::Receiver<inference::Response>) -> Result<Value, InferenceError> {
    let response = receive
        .await
        .map_err(|_| InferenceError::unavailable("the inference worker stopped"))?;
    response.map_err(inference_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropping_a_stream_cancels_at_the_next_token() {
        let (send, receive) = async_mpsc::channel(1);
        drop(receive);

        assert!(deliver(&send, StreamEvent::Pulse).is_break());
    }

    #[test]
    fn a_full_frame_buffer_cancels_instead_of_growing() {
        let (send, _receive) = async_mpsc::channel(1);

        assert!(deliver(&send, StreamEvent::Frame("one".to_string())).is_continue());
        assert!(deliver(&send, StreamEvent::Frame("two".to_string())).is_break());
    }
}
