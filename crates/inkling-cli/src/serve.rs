//! `serve`: the loaded checkpoint behind an OpenAI-compatible endpoint.
//!
//! `POST /v1/chat/completions`, streaming and collected, and `GET /v1/models`.
//! What makes it more than a socket in front of [`crate::generate`] is
//! [`crate::chat`]: the messages are written out as the turn structure the model
//! was trained on, without which nothing puts the model in a turn it could end
//! and every request runs to `max_tokens`.
//!
//! # One request at a time, and the next one waits
//!
//! The checkpoint is loaded once — 0.3 s and 0.35 GiB peak — and every request is
//! served against it in the order it arrived. There is no batching here, and
//! saying otherwise would be the substantial lie: continuous batching is the
//! reason this engine exists and it is a scheduler, not a request loop.
//!
//! So a second client waits rather than fails. `tiny_http` accepts and parses on
//! a thread of its own and hands requests over one at a time, so a request that
//! arrives mid-generation is queued and answered when the one before it is
//! finished. At 0.055 s a token that is a wait a client can sit through, and at
//! the CPU path's 9.0 s it is a long one — honest either way.
//!
//! # A request prefills what the last one did not
//!
//! A conversation comes back turn after turn with a little added each time, so a
//! server that keeps nothing re-prefills the whole of it every turn. Whether the
//! conversation a client sent is the one the cache holds is
//! [`Kept`](crate::kept::Kept)'s decision — one conversation, and an exact
//! extension of it — and this loop is what follows from it: prefill the part
//! that is new, mark the cache where the prompt ends, generate, and put the
//! cache back at the mark.
//!
//! **The mark is what keeps that decision honest.** A generation moves the cache
//! past the prompt, and the reply cannot be recorded in its place: what a client
//! sends back is not what the model streamed, because the turn structure renders
//! a thinking channel as a message of its own where the model emits it inside
//! one. A cache that had recorded the reply would match a position in the middle
//! of it that no mark stands at.
//!
//! # Why `tiny_http`
//!
//! A blocking accept loop is what a server holding one model and decoding one
//! request at a time actually is, so an async runtime would buy an executor to
//! hide a step behind that the process has nothing else to do during, and
//! hiding it is not possible. `tiny_http` is MIT/Apache-2.0 over four
//! transitive crates — `ascii`, `chunked_transfer`,
//! `httpdate`, `log` — against roughly a hundred for an `axum`/`hyper`/`tokio`
//! stack, which matters in a tree where `tokenizers` has so far been the only
//! heavy dependency.
//!
//! It parses the request and owns the socket; it does not write the event
//! stream. A `tiny_http::Response` is built around a `Read` of known length, and
//! an event stream is neither — it is produced by a loop pushing frames as
//! tokens arrive. So [`Body`] takes the socket by
//! [`into_writer`](tiny_http::Request::into_writer) and writes the completion's
//! head, its chunk framing and its frames itself, and the ordinary responses — a
//! refusal, a listing — go out through `tiny_http` where the length is known
//! before the first byte.
//!
//! Writing the response by hand means writing its framing by hand, and the
//! framing is the one thing here that is not free to get wrong: see
//! [`EVENT_STREAM`] for why it has to be chunked rather than a body the close of
//! the connection ends.

use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use inkling_core::{Checkpoint, CheckpointWeights, Ending, Generator, Stop, Tokenizer};
use tiny_http::{Header, Method, Request, Response, Server};

use crate::args::Serve;
use crate::chat::{self, Channel, Channels, MARKERS};
use crate::kept::Kept;
use crate::openai::{ChatRequest, Completion, Finish, RequestError};
use crate::{backend, config};

const COMPLETIONS: &str = "/v1/chat/completions";
const MODELS: &str = "/v1/models";

/// The largest request body this reads.
///
/// A mebibyte of JSON is on the order of a quarter of a million tokens, and
/// prefill is quadratic in the prompt: a conversation that large would not be
/// answered this century on the CPU path, so nothing is lost by refusing one.
/// What it buys is that a `Content-Length` a client made up cannot be turned
/// into that many bytes of memory in a process already holding 16.7 GiB of
/// weights.
const LARGEST_BODY: usize = 1 << 20;

/// The head of an event stream.
///
/// `Transfer-Encoding: chunked` rather than a `Content-Length`, which is not
/// known when the first frame goes out — and the whole point is that the first
/// frame goes out nine seconds in rather than three minutes in.
///
/// Not `Connection: close`, which would be the other way to end a body of
/// unknown length and which this cannot honour. The writer
/// [`into_writer`](tiny_http::Request::into_writer) hands over is one end of a
/// socket `tiny_http` keeps for the next request on the same connection, so
/// dropping it closes nothing. A client told to read until the connection closes
/// would wait for an end that never arrives, which is precisely what the first
/// `curl` against this did.
const EVENT_STREAM: &str = "HTTP/1.1 200 OK\r\n\
     Content-Type: text/event-stream\r\n\
     Cache-Control: no-cache\r\n\
     Transfer-Encoding: chunked\r\n\r\n";

/// One piece of a chunked body: its length in hexadecimal, and then the bytes.
fn chunked(text: &str) -> String {
    format!("{:x}\r\n{text}\r\n", text.len())
}

/// The empty chunk that ends a chunked body, which is what tells a client the
/// stream is over rather than merely quiet.
const LAST_CHUNK: &str = "0\r\n\r\n";

fn json_head(status: &str, length: usize) -> String {
    format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {length}\r\n\r\n"
    )
}

/// Seconds since the epoch, which is what OpenAI's `created` is.
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

/// The name this checkpoint answers to, which is the directory it was loaded
/// from. A client that lists models and then names one back gets the same
/// string either way.
fn model_name(checkpoint: &Path) -> String {
    checkpoint
        .file_name()
        .unwrap_or(checkpoint.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// The marker ids of this vocabulary, which the reply is split on.
///
/// Refused at startup rather than at the first request. A checkpoint whose
/// vocabulary spells these differently is one whose replies would carry
/// `<|end_message|><|content_model_end_sampling|>` on the end of every message,
/// and a server should not have to serve one request to find that out.
fn markers(tokenizer: &Tokenizer) -> Result<Vec<(u32, String, Option<Channel>)>> {
    // The config's end-of-sequence id, which is `<|content_model_end_sampling|>`
    // for this checkpoint and is asked for by id rather than assumed to be —
    // `Tokenizer` takes it from the config precisely because the vocabulary's
    // own files name none.
    let eos = tokenizer.piece(tokenizer.eos()).ok_or_else(|| {
        anyhow!(
            "the end-of-sequence id {} is not in the vocabulary",
            tokenizer.eos()
        )
    })?;

    let mut markers: Vec<(u32, String, Option<Channel>)> = Vec::new();
    let named = MARKERS.iter().map(|(marker, channel)| (*marker, *channel));
    for (marker, channel) in named.chain([(eos.as_str(), None)]) {
        let id = tokenizer.id_of(marker).ok_or_else(|| {
            anyhow!("this vocabulary has no {marker}, so it is not an Inkling one")
        })?;
        if !markers.iter().any(|(seen, ..)| *seen == id) {
            markers.push((id, marker.to_string(), channel));
        }
    }
    Ok(markers)
}

/// A response whose length is known before it is written, which is every one of
/// them but the event stream. `tiny_http` writes these.
fn json(status: u16, body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("a well-formed header");
    Response::from_string(body)
        .with_status_code(status)
        .with_header(header)
}

/// Where a completion goes as it is decoded, in whichever form was asked for.
///
/// One type for both, because there is one generation behind them. The
/// difference is when the bytes leave: a stream writes each frame as its token
/// arrives, and a collected body writes once at the end — but both are the same
/// [`Completion`] being pushed the same deltas, so the second cannot drift from
/// the first.
///
/// Failing is quiet. A sink cannot fail loudly from inside the generation loop —
/// it can only decline the next token — so a failure is remembered and surfaces
/// once the loop is over, exactly as [`crate::generate`] does it.
struct Body<W: Write> {
    out: W,
    completion: Completion,
    streaming: bool,
    failed: Option<anyhow::Error>,
    /// Whether the failure was the socket's own.
    ///
    /// Both kinds of failure end the generation, and only this one ends the
    /// response. A client that hung up cannot be told anything; one that is
    /// still there is owed the end of the body it was promised — a chunked body
    /// that stops without its last chunk leaves a client reading a stream that
    /// has already finished.
    disconnected: bool,
}

impl<W: Write> Body<W> {
    /// The head, for a stream, and nothing at all for a collected body — whose
    /// `Content-Length` is not known until the last token.
    fn open(mut out: W, completion: Completion, streaming: bool) -> Result<Self> {
        if streaming {
            out.write_all(EVENT_STREAM.as_bytes())
                .context("writing the response head")?;
        }
        let mut body = Self {
            out,
            completion,
            streaming,
            failed: None,
            disconnected: false,
        };
        if streaming {
            let opening = body.completion.opening();
            let _ = body.write(&chunked(&opening));
        }
        Ok(body)
    }

    /// Written and flushed. A frame held in a buffer is a frame the client
    /// cannot see, and the next one is nine seconds away.
    fn write(&mut self, text: &str) -> ControlFlow<()> {
        // A connection that has gone does not come back, and a second attempt
        // would only replace the first failure's message with its own.
        if self.disconnected {
            return ControlFlow::Break(());
        }
        let wrote = self
            .out
            .write_all(text.as_bytes())
            .and_then(|()| self.out.flush());
        match wrote {
            Ok(()) => ControlFlow::Continue(()),
            Err(err) => {
                self.disconnected = true;
                self.failed.get_or_insert_with(|| {
                    anyhow::Error::from(err).context("writing to the client")
                });
                ControlFlow::Break(())
            }
        }
    }

    /// One token's worth of text, and whether the generation should go on.
    ///
    /// [`ControlFlow::Break`] is what reaches `Generator::stream` as
    /// [`Stop::Sink`], and it is why a client that hung up costs one more decode
    /// step rather than the whole budget.
    fn push(&mut self, channel: Channel, text: &str) -> ControlFlow<()> {
        let frame = self.completion.push(channel, text);
        self.emit(frame)
    }

    /// A delta on its way out, which is a frame only for a stream: a collected
    /// body has already taken it and has nothing to send until the end.
    fn emit(&mut self, frame: Option<String>) -> ControlFlow<()> {
        match (frame, self.streaming) {
            (Some(frame), true) => self.write(&chunked(&frame)),
            _ => ControlFlow::Continue(()),
        }
    }

    /// A token that could not be spelled out of the vocabulary, which ends the
    /// generation without ending the connection.
    fn fail(&mut self, err: impl Into<anyhow::Error>) -> ControlFlow<()> {
        self.failed.get_or_insert_with(|| err.into());
        ControlFlow::Break(())
    }

    /// The last of the text, and then whichever ending the caller asked for.
    ///
    /// A generation that failed has no ending to report — neither of OpenAI's
    /// two reasons is true of it — so the response is finished rather than
    /// concluded: a stream gets its last chunk and nothing else, and a collected
    /// body that never went out at all becomes the failure it hit. Both are
    /// worse answers than a reply, and both are better than a client left
    /// reading a response that has already stopped.
    fn close(mut self, tail: (Channel, &str), finish: Finish) -> Result<()> {
        let rest = match self.failed.is_some() {
            true if self.streaming => LAST_CHUNK.to_string(),
            true => {
                let body = crate::openai::error(&format!(
                    "{:#}",
                    self.failed.as_ref().expect("a failure")
                ));
                format!(
                    "{}{body}",
                    json_head("500 Internal Server Error", body.len())
                )
            }
            false => {
                // The `Break` this can return says the write failed, and
                // `self.failed` is where that is read off below.
                let leftover = self.completion.tail(tail.0, tail.1);
                let _ = self.emit(leftover);
                match self.streaming {
                    true => format!("{}{LAST_CHUNK}", chunked(&self.completion.closing(finish))),
                    false => {
                        let body = self.completion.collected(finish);
                        format!("{}{body}", json_head("200 OK", body.len()))
                    }
                }
            }
        };
        let _ = self.write(&rest);
        match self.failed.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

/// Everything a request is answered against, loaded once.
struct Engine<'a> {
    tokenizer: Tokenizer,
    weights: &'a CheckpointWeights<'a>,
    generator: Generator<'a>,
    markers: Vec<(u32, String, Option<Channel>)>,
    model: String,
    max_tokens: usize,
    served: u64,
    /// The conversation the last request left behind, which the next one either
    /// extends or replaces.
    kept: Kept<'a>,
}

impl Engine<'_> {
    /// What a request that could not be understood is answered with. Its message
    /// is the one the parser or the turn structure produced, so a client is told
    /// which role or which field it was.
    fn refuse(&self, request: Request, status: u16, message: &str) -> Result<()> {
        request
            .respond(json(status, crate::openai::error(message)))
            .context("answering a request that was refused")
    }

    /// A completion, streamed or collected.
    ///
    /// Everything that can be refused is refused before the socket is taken over
    /// for the response, so a bad request gets a status code and a message rather
    /// than an event stream carrying an apology.
    fn complete(&mut self, mut request: Request) -> Result<()> {
        if request.body_length().is_some_and(|len| len > LARGEST_BODY) {
            let message = format!("a request body is at most {LARGEST_BODY} bytes");
            return self.refuse(request, 413, &message);
        }

        let mut body = String::new();
        if let Err(err) = request
            .as_reader()
            .take(LARGEST_BODY as u64)
            .read_to_string(&mut body)
        {
            return self.refuse(
                request,
                400,
                &format!("the request body is not text: {err}"),
            );
        }

        let asked = match ChatRequest::parse(&body) {
            Ok(asked) => asked,
            Err(err) => return self.refuse(request, 400, &err.to_string()),
        };
        let prompt = match chat::prompt(&asked.messages) {
            Ok(prompt) => prompt,
            Err(err) => return self.refuse(request, 400, &RequestError::from(err).to_string()),
        };

        let ids: Vec<usize> = match self.tokenizer.encode(&prompt) {
            Ok(ids) => ids.into_iter().map(|id| id as usize).collect(),
            Err(err) => return self.refuse(request, 400, &err.to_string()),
        };
        eprintln!(
            "{COMPLETIONS} {} messages, {} prompt tokens, budget {}{}",
            asked.messages.len(),
            ids.len(),
            asked.max_tokens(self.max_tokens),
            if asked.stream { ", streamed" } else { "" }
        );

        self.served += 1;
        let created = now();
        let completion = Completion::new(
            format!("chatcmpl-{created}{:04}", self.served),
            created,
            self.model.clone(),
            ids.len(),
        );
        self.generate(
            &asked,
            &ids,
            Body::open(request.into_writer(), completion, asked.stream)?,
        )
    }

    /// The loop: what the last request did not already hold of the prompt
    /// prefilled — which is [`Kept::turn`] — then a token at a time into `out`
    /// until the model ends its turn, the budget runs out, or the client hangs
    /// up.
    fn generate(
        &mut self,
        asked: &ChatRequest,
        ids: &[usize],
        mut out: Body<impl Write>,
    ) -> Result<()> {
        let ending = Ending {
            budget: asked.max_tokens(self.max_tokens),
            eos: Some(self.tokenizer.eos() as usize),
        };
        let mut text = self.tokenizer.stream();
        let mut channels = Channels::new(self.markers.iter().cloned());

        let (weights, generator) = (self.weights, self.generator);
        let (stop, _) = self.kept.turn(&generator, weights, ids, ending, |id| {
            match text.push(id as u32) {
                Ok(decoded) => {
                    let (channel, decoded) = channels.route(id as u32, &decoded);
                    out.push(channel, &decoded)
                }
                Err(err) => out.fail(err),
            }
        });

        // Bytes the last token left half a character with, which a budget that
        // cut the reply off mid-character has and holding back would lose.
        let tail = text.finish();
        out.close((channels.current(), &tail), finish(stop))
    }
}

/// Which of OpenAI's two endings a stop is.
///
/// [`Stop::Sink`] is neither. The client is gone or the write failed, so there is
/// nothing left to tell it why — the failure surfaces on the server's own stderr
/// instead.
fn finish(stop: Stop) -> Finish {
    match stop {
        Stop::EndOfSequence => Finish::Stop,
        Stop::Budget | Stop::Sink => Finish::Length,
    }
}

pub fn run(args: &Serve) -> Result<()> {
    let config = config::of_checkpoint(&args.checkpoint)?;
    let tokenizer = Tokenizer::open(&args.checkpoint, &config)?;
    let markers = markers(&tokenizer)?;

    let speculation = 0;

    // The device before the checkpoint, so that a backend this machine cannot
    // give ends the process before a client can wait on a server that was going
    // to fail.
    let gpu = backend::open(args.backend, args.numerics)?;
    eprintln!("loading {}", args.checkpoint.display());
    let checkpoint = Checkpoint::open(&args.checkpoint)?;
    let weights = backend::weights(gpu.as_ref(), &checkpoint, &config.text_config, speculation)?;
    let generator = weights.generator();

    let mut engine = Engine {
        tokenizer,
        weights: &weights,
        generator,
        markers,
        model: model_name(&args.checkpoint),
        max_tokens: args.max_tokens,
        served: 0,
        kept: Kept::new(&config.text_config, args.reuse_tokens),
    };

    let server = Server::http(&args.address)
        .map_err(|err| anyhow!("cannot listen on {}: {err}", args.address))?;
    eprintln!(
        "serving {} on http://{}, one request at a time",
        engine.model,
        server.server_addr()
    );

    for request in server.incoming_requests() {
        let path = request
            .url()
            .split('?')
            .next()
            .unwrap_or_default()
            .to_string();
        let served = match (request.method(), path.as_str()) {
            (Method::Post, COMPLETIONS) => engine.complete(request),
            (Method::Get, MODELS) => request
                .respond(json(200, crate::openai::models(&engine.model, now())))
                .context("answering a model listing"),
            (Method::Get | Method::Post, _) => engine.refuse(
                request,
                404,
                &format!("{path} is not an endpoint this serves"),
            ),
            (method, _) => {
                let message = format!("{method} is not a method this serves");
                engine.refuse(request, 405, &message)
            }
        };
        // One request failing is not the server failing. A closed connection is
        // the commonest way for one to, and the next client is still waiting.
        if let Err(err) = served {
            eprintln!("request failed: {err:#}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{dechunked, frames, payload};

    /// A socket that remembers what reached it, and can close under the writer
    /// the way a client that hung up does.
    #[derive(Default)]
    struct Socket {
        written: String,
        writes: usize,
        /// Which write fails, counted from one. `None` never fails.
        breaks_at: Option<usize>,
    }

    impl Write for &mut Socket {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.writes += 1;
            if self.breaks_at == Some(self.writes) {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            self.written
                .push_str(std::str::from_utf8(buf).expect("a response is utf8"));
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    const MODEL: &str = "Inkling-Small-mxfp4";

    fn completion() -> Completion {
        Completion::new(
            "chatcmpl-1".to_string(),
            1_774_000_000,
            MODEL.to_string(),
            16,
        )
    }

    /// A whole reply written to a socket, the way `generate` drives one.
    fn served(socket: &mut Socket, streaming: bool, deltas: &[(Channel, &str)]) -> Result<()> {
        let mut body = Body::open(socket, completion(), streaming).expect("the head goes out");
        for (channel, text) in deltas {
            assert_eq!(body.push(*channel, text), ControlFlow::Continue(()));
        }
        body.close((Channel::Content, ""), Finish::Stop)
    }

    const REPLY: [(Channel, &str); 3] = [
        (Channel::Thinking, "Weigh it up."),
        (Channel::Content, "Hello"),
        (Channel::Content, "."),
    ];

    /// What a client asking for a stream gets: the event-stream head, and then
    /// frames.
    #[test]
    fn a_streamed_completion_is_an_event_stream_that_terminates() {
        let mut socket = Socket::default();
        served(&mut socket, true, &REPLY).expect("it is written");

        let (head, body) = socket
            .written
            .split_once("\r\n\r\n")
            .expect("a head and a body");
        assert!(head.contains("Content-Type: text/event-stream"), "{head}");
        assert!(!head.contains("Content-Length"), "{head}");

        let frames = dechunked(body);
        assert!(frames.starts_with("data: {"), "{frames}");
        assert!(frames.ends_with("data: [DONE]\n\n"), "{frames}");
    }

    /// The framing a body of unknown length needs. `Connection: close` is the
    /// other way to end one and this cannot honour it — the socket belongs to
    /// `tiny_http`, which keeps it for the next request — so a client told to
    /// read to the end of the connection would hang on a stream that had already
    /// finished. The empty chunk is what says it finished.
    #[test]
    fn a_stream_is_chunked_because_this_cannot_close_the_connection() {
        let mut socket = Socket::default();
        served(&mut socket, true, &REPLY).expect("it is written");

        let (head, body) = socket
            .written
            .split_once("\r\n\r\n")
            .expect("a head and a body");
        assert!(head.contains("Transfer-Encoding: chunked"), "{head}");
        assert!(!head.contains("Connection: close"), "{head}");
        assert!(body.ends_with(LAST_CHUNK), "{body:?} never ends");
    }

    /// A chunk declares the *bytes* it carries, not the characters. A reply in
    /// any language but English would otherwise leave a client reading a chunk
    /// short and every frame after it out of step.
    #[test]
    fn a_chunk_declares_the_bytes_it_carries() {
        let text = "Café, 日本語, 🙂.";
        assert_eq!(chunked(text), format!("{:x}\r\n{text}\r\n", text.len()));
        assert!(text.len() > text.chars().count(), "nothing is multi-byte");
        assert_eq!(dechunked(&format!("{}{LAST_CHUNK}", chunked(text))), text);
    }

    /// A frame per token, flushed as it is decoded. A reply buffered until the
    /// end is a client watching nothing for as long as the whole budget takes —
    /// which is the whole reason the streaming path is the primary one.
    #[test]
    fn every_frame_is_written_as_its_token_arrives() {
        let mut socket = Socket::default();
        let mut body = Body::open(&mut socket, completion(), true).expect("the head goes out");

        // The head and the opening role frame.
        let opened = body.out.writes;
        for (channel, text) in REPLY {
            let before = body.out.writes;
            assert_eq!(body.push(channel, text), ControlFlow::Continue(()));
            assert_eq!(body.out.writes, before + 1, "{text:?} was buffered");
        }
        assert!(opened >= 2, "the stream opened in one write");
    }

    /// The collected form is one JSON body with a length, and nothing at all
    /// reaches the socket before the last token — there is no length to declare
    /// until then.
    #[test]
    fn a_collected_completion_is_one_json_body_with_its_length_declared() {
        let mut socket = Socket::default();
        served(&mut socket, false, &REPLY).expect("it is written");

        let (head, body) = socket
            .written
            .split_once("\r\n\r\n")
            .expect("a head and a body");
        assert!(head.contains("Content-Type: application/json"), "{head}");
        assert!(
            head.contains(&format!("Content-Length: {}", body.len())),
            "{head} declares a length {} is not",
            body.len()
        );

        let parsed: serde_json::Value = serde_json::from_str(body).expect("a json body");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello.");
        assert_eq!(
            parsed["choices"][0]["message"]["reasoning_content"],
            "Weigh it up."
        );
    }

    /// A collected request writes nothing at all until it is finished, so a
    /// client cannot mistake a half-written body for a whole one.
    #[test]
    fn a_collected_completion_writes_nothing_until_it_is_whole() {
        let mut socket = Socket::default();
        let mut body = Body::open(&mut socket, completion(), false).expect("nothing goes out");
        for (channel, text) in REPLY {
            assert_eq!(body.push(channel, text), ControlFlow::Continue(()));
        }
        assert_eq!(body.out.writes, 0);
        body.close((Channel::Content, ""), Finish::Stop)
            .expect("it is written");
        assert_eq!(socket.writes, 1);
    }

    /// The case `Stop::Sink` exists for. A client that hung up makes the next
    /// write fail, and the failure has to come back as a `Break` — which is what
    /// `Generator::stream` reads as a reason to stop rather than spend the rest
    /// of the budget on a reply nobody will read.
    #[test]
    fn a_client_that_hung_up_ends_the_generation_at_the_next_token() {
        let mut socket = Socket {
            // The head, the role frame, one token, and then the connection is
            // gone.
            breaks_at: Some(4),
            ..Socket::default()
        };
        let mut body = Body::open(&mut socket, completion(), true).expect("the head goes out");

        assert_eq!(
            body.push(Channel::Content, "Hello"),
            ControlFlow::Continue(())
        );
        assert_eq!(body.push(Channel::Content, "."), ControlFlow::Break(()));
        assert_eq!(body.push(Channel::Content, " More"), ControlFlow::Break(()));

        let err = body
            .close((Channel::Content, ""), Finish::Length)
            .expect_err("the failure surfaces");
        assert!(
            format!("{err:#}").contains("writing to the client"),
            "{err:#}"
        );
    }

    /// A token the vocabulary cannot spell ends the generation the same way a
    /// closed connection does — but the connection is not closed, so the body
    /// it promised still has to end. A stream that stopped without its last
    /// chunk would leave a client reading one that had already finished.
    ///
    /// There is no reason to report with it. Neither `stop` nor `length` is
    /// true of a generation that failed, so the last chunk goes out alone and
    /// the reason reaches the server's own stderr.
    #[test]
    fn a_token_that_cannot_be_spelled_still_ends_the_stream_it_opened() {
        let mut socket = Socket::default();
        let mut body = Body::open(&mut socket, completion(), true).expect("the head goes out");

        let unspellable = anyhow!("no token with id 4096 in this vocabulary");
        assert_eq!(body.fail(unspellable), ControlFlow::Break(()));
        let err = body
            .close((Channel::Content, ""), Finish::Length)
            .expect_err("the failure surfaces");
        assert!(format!("{err:#}").contains("4096"), "{err:#}");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        assert!(written.ends_with(LAST_CHUNK), "{written:?} never ends");

        // The role frame the stream opened with, and nothing after it: no
        // reason, and no `[DONE]` claiming the reply is whole.
        let sent = dechunked(written);
        assert_eq!(frames(&sent).len(), 1, "{sent:?}");
        assert_eq!(
            payload(&sent).expect("a chunk")["choices"][0]["finish_reason"],
            serde_json::Value::Null
        );
    }

    /// The same failure before a collected body has written anything. Nothing
    /// has gone out, so what goes out is the failure — a client left with an
    /// open connection and no response at all could not tell that from a server
    /// still thinking.
    #[test]
    fn a_collected_completion_that_failed_answers_with_the_failure() {
        let mut socket = Socket::default();
        let mut body = Body::open(&mut socket, completion(), false).expect("nothing goes out");
        assert_eq!(
            body.push(Channel::Content, "The"),
            ControlFlow::Continue(())
        );
        assert_eq!(
            body.fail(anyhow!("no token with id 4096 in this vocabulary")),
            ControlFlow::Break(())
        );
        body.close((Channel::Content, ""), Finish::Length)
            .expect_err("the failure surfaces");

        let (head, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        assert!(head.starts_with("HTTP/1.1 500"), "{head}");
        assert!(
            head.contains(&format!("Content-Length: {}", written.len())),
            "{head} declares a length {} is not",
            written.len()
        );
        let refusal: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert!(
            refusal["error"]["message"]
                .as_str()
                .expect("a message")
                .contains("4096"),
            "{refusal}"
        );
    }

    /// A client that hung up gets nothing more, because there is nothing there
    /// to write to. It is the one failure that leaves a body unfinished, and it
    /// leaves it unfinished for a reason no other failure has.
    #[test]
    fn a_stream_to_a_client_that_hung_up_is_not_written_to_again() {
        let mut socket = Socket {
            breaks_at: Some(3),
            ..Socket::default()
        };
        let mut body = Body::open(&mut socket, completion(), true).expect("the head goes out");
        assert_eq!(body.push(Channel::Content, "Hello"), ControlFlow::Break(()));
        body.close((Channel::Content, ""), Finish::Length)
            .expect_err("the failure surfaces");

        assert_eq!(socket.writes, 3, "it wrote past the disconnect");
    }

    /// The half a character a budget can cut a reply off in the middle of. It
    /// belongs to whichever channel was open, and it is the last thing written.
    #[test]
    fn the_text_left_over_at_the_end_reaches_the_client() {
        let mut socket = Socket::default();
        let mut body = Body::open(&mut socket, completion(), false).expect("nothing goes out");
        assert_eq!(
            body.push(Channel::Content, "The"),
            ControlFlow::Continue(())
        );
        body.close((Channel::Content, "\u{fffd}"), Finish::Length)
            .expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert_eq!(parsed["choices"][0]["message"]["content"], "The\u{fffd}");
        assert_eq!(parsed["choices"][0]["finish_reason"], "length");
    }

    /// A client that hung up is not a turn the model ended, and the difference
    /// only matters where it is written down: nothing goes back to that client,
    /// but a reply cut short is a reply cut short.
    #[test]
    fn the_endings_a_generation_can_have_map_onto_the_two_openai_has() {
        assert_eq!(finish(Stop::EndOfSequence), Finish::Stop);
        assert_eq!(finish(Stop::Budget), Finish::Length);
        assert_eq!(finish(Stop::Sink), Finish::Length);
    }

    /// The name a model is listed and answered under is the directory it came
    /// from, not the path a caller happened to type.
    #[test]
    fn a_checkpoint_is_named_after_the_directory_it_was_loaded_from() {
        assert_eq!(model_name(Path::new("models/Inkling-Small-mxfp4")), MODEL);
        assert_eq!(model_name(Path::new("models/Inkling-Small-mxfp4/")), MODEL);
    }
}
