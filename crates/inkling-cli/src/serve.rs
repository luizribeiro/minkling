//! `serve`: the loaded checkpoint behind an OpenAI-compatible endpoint.
//!
//! `POST /v1/chat/completions`, streaming and collected, and `GET /v1/models`.
//! What makes it more than a socket in front of [`crate::generate`] is
//! [`crate::chat`]: the messages are written out as the turn structure the model
//! was trained on, without which nothing puts the model in a turn it could end
//! and every request runs to `max_tokens`.
//!
//! # Two paths, and `--slots` is which
//!
//! The checkpoint is loaded once — 0.3 s and 0.35 GiB peak — and served against
//! by one of two arrangements. They are not a fast path and a slow one; they
//! trade different things, and neither dominates.
//!
//! **`--slots 1`, the default: one request at a time, and a conversation kept
//! between them.** Requests are answered in the order they arrived, and what the
//! last one left in the cache the next one starts from — which at 16384 tokens
//! is 33 s of prefill not paid again, and is worth more than any kernel in this
//! tree. A second client waits rather than fails.
//!
//! **`--slots N`: the scheduler, and the conversation is not kept.** This is
//! [`Continuous`], the engine the batching figures were taken on, with the
//! socket in front of it: N requests advance together, a request that arrives
//! mid-generation joins the batch it found rather than waiting for it to drain,
//! and a slot a request vacates is filled from the front of the queue in the
//! very next step.
//!
//! **What the second gives up is [`Kept`](crate::kept::Kept), and that is not a
//! decision made here.** A slot's cache is the *slot's* — the device holds one
//! span and one window pair per layer per slot, and [`Continuous`] hands every
//! joining sequence a fresh one precisely so that it does not attend over the
//! keys of whoever sat there before. There is no position in that arrangement
//! for a conversation to be resumed to. So a scheduled server re-prefills every
//! turn, and a single-slot one does not, and which of those a workload wants is
//! the question `--slots` asks. A fleet of agents at irregular times wants the
//! scheduler; one coding session sending its context back turn after turn wants
//! the cache.
//!
//! # A request prefills what the last one did not, on the serial path
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
//! # Threads, and where the model stays
//!
//! **The model never moves.** One thread owns the weights, the generator and
//! the scheduler, and it does nothing but step: there is one device and one set
//! of slots, so a second thread touching either would be two callers writing the
//! same spans. Around it are connection threads, which parse a request, hand the
//! prompt over, and then do nothing but write frames as tokens come back — none
//! of them ever sees a weight.
//!
//! What travels between them is ids. A prompt goes in on [`Wanted`] and tokens
//! come back on [`Dispatch`], one per step that decoded for that ticket, which is
//! what lets a client see its reply as it is produced rather than when the batch
//! it is in finishes.
//!
//! **And a connection thread gives its seat back however it leaves.** See
//! [`Seat`]: a client that hangs up mid-generation is not an edge case in a
//! fleet, it is the ordinary way a request ends, and a seat that outlived one
//! would decode a whole budget nobody reads while the request behind it waits.
//!
//! # Why `tiny_http`
//!
//! A blocking accept loop is what this is on both paths — a connection thread
//! waiting for tokens has nothing else to do, and an async runtime would buy an
//! executor to hide a step behind that nothing can be hidden behind.
//! `tiny_http` is MIT/Apache-2.0 over four
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

use std::collections::HashMap;
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use inkling_core::schedule::Request as SeatRequest;
use inkling_core::{
    Checkpoint, CheckpointWeights, Continuous, Detokenizer, Ending, Generator, Kept, ModelWeights,
    Stepped, Stop, TextConfig, Tokenizer,
};
use tiny_http::{Header, Method, Request, Response, Server};

use crate::args::Serve;
use crate::chat::{self, Channel, Channels, MARKERS, Reading, Routed};
use crate::openai::{ChatRequest, Completion, Finish, RequestError};
use crate::stop::Stops;
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
fn markers(tokenizer: &Tokenizer) -> Result<Vec<(u32, String, Reading)>> {
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

    let mut markers: Vec<(u32, String, Reading)> = Vec::new();
    let named = MARKERS.iter().map(|(marker, channel)| (*marker, *channel));
    // The end of the turn leaves the reader where the end of a message does:
    // in content, which is where text nobody has named a channel for goes.
    let ends = [(eos.as_str(), Reading::Channel(Channel::Content))];
    for (marker, channel) in named.chain(ends) {
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
    /// Where the client asked the reply to be cut, and the text held back until
    /// it is known whether it begins a cut. See [`crate::stop`] for what it is
    /// matched against and why nothing goes out that a match would retract.
    stops: Stops,
    /// Whether a stop sequence matched.
    ///
    /// An ending of its own, and one [`Stop`] has no word for: the model did not
    /// end its turn and the budget did not run out. It reaches the generation as
    /// [`Stop::Sink`] — the same `Break` a client that hung up produces, because
    /// there is one way to say "no more tokens" — and that is precisely why it
    /// has to be remembered here. `Sink` maps onto `length`, and a reply cut
    /// where the client asked is not a reply that was cut short.
    struck: bool,
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
    fn open(mut out: W, completion: Completion, streaming: bool, stops: Stops) -> Result<Self> {
        if streaming {
            out.write_all(EVENT_STREAM.as_bytes())
                .context("writing the response head")?;
        }
        let mut body = Self {
            out,
            completion,
            streaming,
            stops,
            struck: false,
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

    /// One token's worth of whatever it turned out to be, and whether the
    /// generation should go on.
    ///
    /// [`ControlFlow::Break`] is what reaches `Generator::stream` as
    /// [`Stop::Sink`], and it is why a client that hung up costs one more decode
    /// step rather than the whole budget.
    fn push(&mut self, routed: Routed) -> ControlFlow<()> {
        let routed = self.cut(routed);
        let frame = self.completion.push(routed);
        let wrote = self.emit(frame);
        match self.struck {
            true => ControlFlow::Break(()),
            false => wrote,
        }
    }

    /// `routed` with anything past a stop sequence cut off, and anything that
    /// might yet begin one held back.
    ///
    /// **Only content is matched, and everything else passes through
    /// untouched** — which is what keeps a client's `content` one string across
    /// a thinking channel the model opened in the middle of it. See
    /// [`crate::stop`] for why `content` is the field a `stop` is written
    /// against.
    fn cut(&mut self, routed: Routed) -> Routed {
        if self.struck {
            return Routed::Nothing;
        }
        let Routed::Text(Channel::Content, text) = &routed else {
            return routed;
        };
        let taken = self.stops.take(text);
        self.struck = taken.struck;
        Routed::Text(Channel::Content, taken.shown)
    }

    /// Which of OpenAI's endings this reply had.
    ///
    /// A stop sequence outranks whatever the generation reported, because what
    /// the generation reported is the `Break` this handed it.
    fn ending(&self, stop: Stop) -> Finish {
        match self.struck {
            true => Finish::Stop,
            false => finish(stop, self.completion.called()),
        }
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

    /// The last of it, and then the ending the stop maps onto.
    ///
    /// The ending is decided after the tail rather than before it, because the
    /// tail can be the call a budget cut short — and a turn that asked for a
    /// tool ends for a different reason from one that did not.
    ///
    /// A generation that failed has no ending to report — none of OpenAI's
    /// reasons is true of it — so the response is finished rather than
    /// concluded: a stream gets its last chunk and nothing else, and a collected
    /// body that never went out at all becomes the failure it hit. Both are
    /// worse answers than a reply, and both are better than a client left
    /// reading a response that has already stopped.
    fn close(mut self, tail: Routed, stop: Stop) -> Result<()> {
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
                // The tail is text like any other and can be the second half of
                // a sequence, so it goes through the matching rather than round
                // it. The `Break` this can return says the write failed, and
                // `self.failed` is where that is read off below.
                let cut = self.cut(tail);
                let leftover = self.completion.tail(cut);
                let _ = self.emit(leftover);

                // Text held against a sequence that never matched was text all
                // along, and there is nothing left to resolve it. A generation a
                // sequence *did* match holds nothing: everything behind the
                // match was cut where the match was found.
                let held = self.stops.finish();
                if !held.is_empty() {
                    let frame = self.completion.tail(Routed::Text(Channel::Content, held));
                    let _ = self.emit(frame);
                }

                let finish = self.ending(stop);
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

/// A reply on its way out: the tokens decoded, routed into their channels, and
/// framed into the body.
///
/// **One type for both of the ways a request is served**, because what differs
/// between answering a request alone and answering one of eight is where the
/// tokens come from and nothing whatever about what is done with them. A second
/// copy of this loop is where the two paths would drift, and what a drift would
/// look like is a client seeing `<|content_text|>` in its `content` on one of
/// them and not the other.
struct Turn<'a, W: Write> {
    text: Detokenizer<'a>,
    channels: Channels,
    body: Body<W>,
}

impl<'a, W: Write> Turn<'a, W> {
    fn new(shared: &'a Shared, body: Body<W>) -> Self {
        Self {
            text: shared.tokenizer.stream(),
            channels: Channels::new(shared.markers.iter().cloned()),
            body,
        }
    }

    /// One decoded token on its way to the client, and whether the generation
    /// should go on.
    fn push(&mut self, id: usize) -> ControlFlow<()> {
        match self.text.push(id as u32) {
            Ok(decoded) => self.body.push(self.channels.route(id as u32, &decoded)),
            Err(err) => self.body.fail(err),
        }
    }

    /// The response finished, however the generation ended.
    fn close(mut self, stop: Stop) -> Result<()> {
        // Bytes the last token left half a character with, which a budget that
        // cut the reply off mid-character has and holding back would lose, and
        // with them the call a budget cut short.
        let held = self.text.finish();
        self.body.close(self.channels.finish(&held), stop)
    }
}

/// Everything every request is answered against, and the only state a
/// connection thread holds.
///
/// Read-only once the server is up, which is what lets one copy serve every
/// thread at once: a vocabulary, the marker ids read out of it, the name the
/// checkpoint answers to, and the budget a request that names none is served
/// under.
struct Shared {
    tokenizer: Tokenizer,
    markers: Vec<(u32, String, Reading)>,
    model: String,
    max_tokens: usize,
    /// How many completions have been handed out, which is what makes their ids
    /// distinct. Atomic because on the scheduled path several connection threads
    /// mint one at once.
    served: AtomicU64,
}

/// A request read and understood: the ids of its prompt, what ends it, and the
/// response it will be written into.
struct Understood<'a, W: Write> {
    ids: Vec<usize>,
    ending: Ending,
    turn: Turn<'a, W>,
}

/// What a request that could not be understood is answered with. Its message is
/// the one the parser or the turn structure produced, so a client is told which
/// role or which field it was.
fn refuse(request: Request, status: u16, message: &str) -> Result<()> {
    request
        .respond(json(status, crate::openai::error(message)))
        .context("answering a request that was refused")
}

impl Shared {
    /// A completion parsed, templated and encoded, or `None` where the request
    /// has already been refused.
    ///
    /// **Everything that can be refused is refused before the socket is taken
    /// over for the response**, so a bad request gets a status code and a
    /// message rather than an event stream carrying an apology.
    fn understand(&self, mut request: Request) -> Result<Option<Understood<'_, Wire>>> {
        if request.body_length().is_some_and(|len| len > LARGEST_BODY) {
            let message = format!("a request body is at most {LARGEST_BODY} bytes");
            return refuse(request, 413, &message).map(|()| None);
        }

        let mut body = String::new();
        if let Err(err) = request
            .as_reader()
            .take(LARGEST_BODY as u64)
            .read_to_string(&mut body)
        {
            let message = format!("the request body is not text: {err}");
            return refuse(request, 400, &message).map(|()| None);
        }

        let asked = match ChatRequest::parse(&body) {
            Ok(asked) => asked,
            Err(err) => return refuse(request, 400, &err.to_string()).map(|()| None),
        };
        let prompt = match chat::prompt(&asked.messages, asked.declared()) {
            Ok(prompt) => prompt,
            Err(err) => {
                let message = RequestError::from(err).to_string();
                return refuse(request, 400, &message).map(|()| None);
            }
        };
        let ids: Vec<usize> = match self.tokenizer.encode(&prompt) {
            Ok(ids) => ids.into_iter().map(|id| id as usize).collect(),
            Err(err) => return refuse(request, 400, &err.to_string()).map(|()| None),
        };

        let budget = asked.max_tokens(self.max_tokens);
        eprintln!(
            "{COMPLETIONS} {} messages, {} prompt tokens, budget {budget}{}",
            asked.messages.len(),
            ids.len(),
            if asked.stream { ", streamed" } else { "" }
        );

        let created = now();
        let served = self.served.fetch_add(1, Ordering::Relaxed) + 1;
        // The counts are on the collected body already, and are written after
        // the last token where they are known — so `include_usage` is a question
        // only a stream has, and a client that sent it without asking for one is
        // answered with what it wanted rather than refused for how it asked.
        let completion = Completion::new(
            format!("chatcmpl-{created}{served:04}"),
            created,
            self.model.clone(),
            ids.len(),
        )
        .reporting_usage(asked.stream && asked.wants_usage());

        let body = Body::open(
            request.into_writer(),
            completion,
            asked.stream,
            Stops::new(asked.stopping()),
        )?;
        Ok(Some(Understood {
            ids,
            ending: Ending {
                budget,
                eos: Some(self.tokenizer.eos() as usize),
            },
            turn: Turn::new(self, body),
        }))
    }
}

/// The serial path: one request at a time, and the conversation the last one
/// left behind.
struct Engine<'a> {
    shared: &'a Shared,
    weights: &'a CheckpointWeights<'a>,
    generator: Generator<'a>,
    /// The conversation the last request left behind, which the next one either
    /// extends or replaces.
    kept: Kept<'a>,
}

impl Engine<'_> {
    /// What the last request did not already hold of the prompt prefilled —
    /// which is [`Kept::turn`] — then a token at a time until the model ends its
    /// turn, the budget runs out, or the client hangs up.
    fn complete(&mut self, request: Request) -> Result<()> {
        let Some(understood) = self.shared.understand(request)? else {
            return Ok(());
        };
        let Understood {
            ids,
            ending,
            mut turn,
        } = understood;

        let (weights, generator) = (self.weights, self.generator);
        let served = self
            .kept
            .turn(&generator, weights, &ids, ending, |id| turn.push(id));
        turn.close(served.stop)
    }
}

/// The socket a response is written down, which is what `tiny_http` hands over
/// when the length of a body is not known before its first byte.
type Wire = Box<dyn Write + Send>;

/// What a connection thread asks the engine for, and what it tells it
/// afterwards.
enum Wanted {
    /// A prompt to generate from, and where to send its tokens.
    Seat(Seating),
    /// A ticket nobody is waiting for any more. See [`Seat`] for why every path
    /// out of a connection thread sends one.
    Release(usize),
}

/// A request on its way into a slot.
struct Seating {
    ids: Vec<usize>,
    budget: usize,
    reply: Sender<Dispatch>,
}

/// What the engine sends one connection thread.
enum Dispatch {
    /// The ticket this request was seated under, which is the first thing it
    /// hears and the name it gives its seat back by.
    Seated(usize),
    /// One token, on the step that produced it.
    Token(usize),
    /// The generation is over, and why.
    Done(Stop),
}

/// A ticket the engine is holding on one connection's behalf.
///
/// **Every path out of a connection thread gives the ticket back**, which is
/// the whole of what [`Drop`] is doing here: a client that hangs up, a frame
/// that will not write, a body that would not parse, a panic unwinding — all of
/// them leave the scope, and a seat that outlived one would go on decoding a
/// budget nobody reads while the request behind it waits for a slot.
///
/// That is not an edge case under the load a scheduler exists for. A fleet of
/// agents cancels requests; it is the ordinary way one ends.
struct Seat<'a> {
    ticket: usize,
    engine: &'a Sender<Wanted>,
}

impl Drop for Seat<'_> {
    fn drop(&mut self) {
        // An engine that has already stopped has no seats left to give back.
        let _ = self.engine.send(Wanted::Release(self.ticket));
    }
}

/// Prompt rows one step carries at most, over every seat that is filling.
///
/// **A latency knob and not a correctness one**: it changes no token — which
/// [`Continuous`]'s own `the_rows_a_prompt_is_fed_in_change_no_token` asserts —
/// and what it buys is a bound on the jitter one arrival costs the sequences
/// already decoding. A 385-token prompt fed whole makes their next token wait
/// 2.75 s against their own 73.6 ms; spread over this budget the worst one step
/// costs them is 725 ms, and 199 at a budget of 16. Narrower is not free either:
/// the same prompt in 24 chunks of 16 is 4.54 s of device where whole it is
/// 1.78.
///
/// 128 because that is the budget the README's fleet table was taken at, so a
/// server run at this width is being read against a measurement of it.
const ADMIT: usize = 128;

/// How many connections a scheduled server reads at once, for each slot it has.
///
/// A thread here is blocked on a socket rather than on the model — it parses a
/// request, hands the prompt over, and then does nothing but wait for tokens —
/// so what this bounds is how many clients can be mid-request and not how much
/// work the engine does. Two apiece: one seated and one queued behind it, which
/// is the deepest queue a slot can drain without the engine ever standing idle.
/// Connections past that wait in the listener's own backlog, which is where a
/// server with a bounded number of threads has to put them.
const READERS_A_SLOT: usize = 2;

/// One connection answered against the scheduler: parse it, hand the prompt to
/// the engine, and write frames until the tokens stop.
fn answer(shared: &Shared, engine: &Sender<Wanted>, request: Request) -> Result<()> {
    let Some(understood) = shared.understand(request)? else {
        return Ok(());
    };
    let Understood {
        ids,
        ending,
        mut turn,
    } = understood;

    // **A request wanting no tokens is answered here rather than seated.** The
    // scheduler walks its queue past one without ever answering it — a slot held
    // by a request that is already finished is a slot standing empty in front of
    // the one behind it — so a client that reached it would wait for ever, where
    // the serial path answers the same request with an empty reply. Both
    // entrances refuse a budget of zero already, and neither of them is
    // something this loop can see.
    if ending.budget == 0 {
        return turn.close(Stop::Budget);
    }

    let (reply, dispatched) = mpsc::channel();
    let seating = Seating {
        ids,
        budget: ending.budget,
        reply,
    };
    engine
        .send(Wanted::Seat(seating))
        .map_err(|_| anyhow!("the engine has stopped and cannot seat a request"))?;

    // The ticket before any token, because it is the name this gives the seat
    // back by — and from here on there is a seat to give back.
    let seated = dispatched.recv();
    let Ok(Dispatch::Seated(ticket)) = seated else {
        return Err(anyhow!("the engine seated a request without saying where"));
    };
    let seat = Seat { ticket, engine };

    // `Sink` is what a stream that simply stopped arriving is: the engine
    // dropped the other end without saying why, which is what a released seat
    // looks like from here.
    let mut stop = Stop::Sink;
    while let Ok(dispatch) = dispatched.recv() {
        match dispatch {
            Dispatch::Token(id) if turn.push(id).is_break() => break,
            Dispatch::Token(_) => {}
            Dispatch::Done(ended) => {
                stop = ended;
                break;
            }
            // Seated once, above. A second is the engine contradicting itself
            // and there is nothing to be done with it but stop.
            Dispatch::Seated(_) => break,
        }
    }

    // Given back before the last frame rather than after it, so the request
    // behind this one is admitted into the slot while this one is still
    // writing. A panic between here and the end is a seat already returned.
    drop(seat);
    turn.close(stop)
}

/// The scheduler behind the socket.
///
/// One thread — this one — owning the weights, the generator and the slots, and
/// doing nothing but stepping them; `readers` connection threads around it
/// doing nothing but sockets. See the module documentation for why the model
/// does not move.
fn schedule(
    config: &TextConfig,
    shared: &Arc<Shared>,
    weights: &impl ModelWeights,
    generator: &Generator<'_>,
    server: Arc<Server>,
    slots: usize,
) -> Result<()> {
    let (asking, asked) = mpsc::channel::<Wanted>();
    for _ in 0..slots * READERS_A_SLOT {
        let (server, shared, asking) = (Arc::clone(&server), Arc::clone(shared), asking.clone());
        std::thread::spawn(move || {
            while let Ok(request) = server.recv() {
                route(&shared, request, |request| {
                    answer(&shared, &asking, request)
                });
            }
        });
    }
    // The last handle this thread holds, so that the loop below sees the channel
    // closed once every connection thread has gone rather than never.
    drop(asking);
    stepping(
        config,
        weights,
        generator,
        asked,
        shared.tokenizer.eos() as usize,
        slots,
    )
}

/// The loop itself: admit whatever arrived, step, and send the step's tokens to
/// whoever is waiting for them.
///
/// **Apart from the threads it runs between**, so that what a test drives is
/// this rather than a copy of it. The interleavings worth being careful about
/// are all here — a request arriving while the engine is busy, a request
/// arriving while it is idle, and the last client going away — and none of them
/// is about a socket.
fn stepping(
    config: &TextConfig,
    weights: &impl ModelWeights,
    generator: &Generator<'_>,
    asked: Receiver<Wanted>,
    eos: usize,
    slots: usize,
) -> Result<()> {
    let mut engine = Continuous::new(config, slots, ADMIT);
    let mut live: HashMap<usize, Sender<Dispatch>> = HashMap::new();

    loop {
        // Blocked rather than spinning where there is nothing to step, which is
        // most of what a server does. **Only where there is nothing**: an engine
        // with a seat in it must not wait on a channel to advance it, or a
        // request already running would stall until the next one arrived.
        if engine.idle() {
            match asked.recv() {
                Ok(wanted) => take(&mut engine, &mut live, weights, wanted),
                // Every connection thread has gone and nothing is seated, which
                // is the only arrangement this can stop in.
                Err(_) => return Ok(()),
            }
        }
        for wanted in asked.try_iter() {
            take(&mut engine, &mut live, weights, wanted);
        }
        // Everything that arrived may have been a release, and a step over no
        // seats is a forward pass over no rows.
        if engine.idle() {
            continue;
        }
        let stepped = engine.step(generator, weights);
        dispatch(&mut engine, &mut live, weights, stepped, eos);
    }
}

/// One thing a connection thread asked for, done.
fn take(
    engine: &mut Continuous<'_>,
    live: &mut HashMap<usize, Sender<Dispatch>>,
    weights: &impl ModelWeights,
    wanted: Wanted,
) {
    match wanted {
        Wanted::Seat(seating) => {
            let ticket = engine.submit(SeatRequest {
                prompt: seating.ids,
                count: seating.budget,
            });
            match seating.reply.send(Dispatch::Seated(ticket)) {
                Ok(()) => {
                    live.insert(ticket, seating.reply);
                }
                // Gone before it was ever seated, so nothing will read its
                // tokens and the slot should not be spent producing them.
                Err(_) => {
                    engine.release(ticket, weights);
                }
            }
        }
        Wanted::Release(ticket) => {
            engine.release(ticket, weights);
            live.remove(&ticket);
        }
    }
}

/// What one step produced, sent to whoever is waiting for it.
///
/// **The end-of-sequence id is read here rather than by the connection thread**,
/// and that is worth a line. A scheduler stops a request when its budget runs
/// out and knows nothing about a vocabulary, so a server that let the client's
/// own thread notice the terminator would free the seat a round trip later —
/// and the engine would have built another step into it by then. Read here, the
/// slot is empty before the next step is built.
fn dispatch(
    engine: &mut Continuous<'_>,
    live: &mut HashMap<usize, Sender<Dispatch>>,
    weights: &impl ModelWeights,
    stepped: Stepped,
    eos: usize,
) {
    for (ticket, id) in &stepped.produced {
        let Some(reply) = live.get(ticket).cloned() else {
            continue;
        };
        // The terminator reaches the client like any other token, which is what
        // the serial path does with it too: the routing knows it as a marker and
        // shows nothing, and a sink that swallowed it would leave the two paths
        // counting different numbers of tokens.
        let gone = reply.send(Dispatch::Token(*id)).is_err();
        let ended = *id == eos;
        if ended {
            let _ = reply.send(Dispatch::Done(Stop::EndOfSequence));
        }
        // A send that failed is a connection thread already gone; its own
        // release is on its way and this only frees the seat a step sooner.
        if ended || gone {
            engine.release(*ticket, weights);
            live.remove(ticket);
        }
    }
    for answered in stepped.done {
        if let Some(reply) = live.remove(&answered.ticket) {
            let _ = reply.send(Dispatch::Done(Stop::Budget));
        }
    }
}

/// Where a request goes, which is the same four answers on both paths — and
/// what becomes of a failure, which is the same on both too.
///
/// **One request failing is not the server failing.** A closed connection is
/// much the commonest way for one to, and there is either a next client waiting
/// or a slot to give back.
fn route(shared: &Shared, request: Request, complete: impl FnOnce(Request) -> Result<()>) {
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_string();
    let served = match (request.method(), path.as_str()) {
        (Method::Post, COMPLETIONS) => complete(request),
        (Method::Get, MODELS) => request
            .respond(json(200, crate::openai::models(&shared.model, now())))
            .context("answering a model listing"),
        (Method::Get | Method::Post, _) => refuse(
            request,
            404,
            &format!("{path} is not an endpoint this serves"),
        ),
        (method, _) => {
            let message = format!("{method} is not a method this serves");
            refuse(request, 405, &message)
        }
    };
    if let Err(err) = served {
        eprintln!("request failed: {err:#}");
    }
}

/// Which of OpenAI's endings a stop is.
///
/// A turn the model ended has two of them and `called` is what tells them
/// apart: a client branches on `tool_calls` to run a tool where it would
/// otherwise show the reply, so reporting `stop` for a turn that asked for one
/// is a reply the client renders as an empty message.
///
/// [`Stop::Sink`] is none of them. The client is gone or the write failed, so
/// there is nothing left to tell it why — the failure surfaces on the server's
/// own stderr instead.
fn finish(stop: Stop, called: bool) -> Finish {
    match stop {
        Stop::EndOfSequence if called => Finish::ToolCalls,
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
    // **The slots are asked for where the layers are wrapped**, because a slot
    // is a span and four convolution windows in every layer and is allocated
    // then rather than when a sequence sits in one. A server that asked the
    // scheduler for more than this would be refused one layer down.
    let weights = backend::weights(
        gpu.as_ref(),
        &checkpoint,
        &config.text_config,
        speculation,
        args.slots,
    )?;
    let generator = weights.generator();

    let shared = Arc::new(Shared {
        tokenizer,
        markers,
        model: model_name(&args.checkpoint),
        max_tokens: args.max_tokens,
        served: AtomicU64::new(0),
    });

    let server = Server::http(&args.address)
        .map_err(|err| anyhow!("cannot listen on {}: {err}", args.address))?;
    eprintln!(
        "serving {} on http://{}, {}",
        shared.model,
        server.server_addr(),
        match args.slots {
            1 => "one request at a time, keeping the conversation between them".to_string(),
            slots => format!("{slots} requests at a time, keeping no conversation"),
        }
    );

    if args.slots > 1 {
        return schedule(
            &config.text_config,
            &shared,
            &weights,
            &generator,
            Arc::new(server),
            args.slots,
        );
    }

    let mut engine = Engine {
        shared: &shared,
        weights: &weights,
        generator,
        kept: Kept::new(&config.text_config, args.reuse_tokens),
    };
    for request in server.incoming_requests() {
        route(&shared, request, |request| engine.complete(request));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use inkling_core::fixture::Stack;
    use inkling_core::head::LmHead;
    use inkling_core::ops::DenseProjection;

    use super::*;
    use crate::chat::text;
    use crate::wire::{dechunked, delta, frames, payload, payloads};

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

    /// A body over a socket, for the cases whose subject is not the stop
    /// sequences — which is every one that was here before them.
    fn opened<W: Write>(out: W, streaming: bool) -> Result<Body<W>> {
        Body::open(out, completion(), streaming, Stops::default())
    }

    /// A body that cuts its reply where a client asked it to.
    fn cutting<W: Write>(out: W, streaming: bool, at: &[&str]) -> Body<W> {
        let stops = Stops::new(at.iter().map(|word| word.to_string()).collect());
        Body::open(out, completion(), streaming, stops).expect("the head goes out")
    }

    /// A whole reply written to a socket, the way `generate` drives one.
    fn served(socket: &mut Socket, streaming: bool, reply: &[Routed]) -> Result<()> {
        let mut body = opened(socket, streaming).expect("the head goes out");
        for routed in reply {
            assert_eq!(body.push(routed.clone()), ControlFlow::Continue(()));
        }
        body.close(Routed::Nothing, Stop::EndOfSequence)
    }

    fn reply() -> Vec<Routed> {
        vec![
            text(Channel::Thinking, "Weigh it up."),
            text(Channel::Content, "Hello"),
            text(Channel::Content, "."),
        ]
    }

    /// What a client asking for a stream gets: the event-stream head, and then
    /// frames.
    #[test]
    fn a_streamed_completion_is_an_event_stream_that_terminates() {
        let mut socket = Socket::default();
        served(&mut socket, true, &reply()).expect("it is written");

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
        served(&mut socket, true, &reply()).expect("it is written");

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
        let mut body = opened(&mut socket, true).expect("the head goes out");

        // The head and the opening role frame.
        let opened = body.out.writes;
        for routed in reply() {
            let before = body.out.writes;
            assert_eq!(body.push(routed.clone()), ControlFlow::Continue(()));
            assert_eq!(body.out.writes, before + 1, "{routed:?} was buffered");
        }
        assert!(opened >= 2, "the stream opened in one write");
    }

    /// The collected form is one JSON body with a length, and nothing at all
    /// reaches the socket before the last token — there is no length to declare
    /// until then.
    #[test]
    fn a_collected_completion_is_one_json_body_with_its_length_declared() {
        let mut socket = Socket::default();
        served(&mut socket, false, &reply()).expect("it is written");

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
        let mut body = opened(&mut socket, false).expect("nothing goes out");
        for routed in reply() {
            assert_eq!(body.push(routed), ControlFlow::Continue(()));
        }
        assert_eq!(body.out.writes, 0);
        body.close(Routed::Nothing, Stop::EndOfSequence)
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
        let mut body = opened(&mut socket, true).expect("the head goes out");

        assert_eq!(
            body.push(text(Channel::Content, "Hello")),
            ControlFlow::Continue(())
        );
        assert_eq!(
            body.push(text(Channel::Content, ".")),
            ControlFlow::Break(())
        );
        assert_eq!(
            body.push(text(Channel::Content, " More")),
            ControlFlow::Break(())
        );

        let err = body
            .close(Routed::Nothing, Stop::Budget)
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
        let mut body = opened(&mut socket, true).expect("the head goes out");

        let unspellable = anyhow!("no token with id 4096 in this vocabulary");
        assert_eq!(body.fail(unspellable), ControlFlow::Break(()));
        let err = body
            .close(Routed::Nothing, Stop::Budget)
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
        let mut body = opened(&mut socket, false).expect("nothing goes out");
        assert_eq!(
            body.push(text(Channel::Content, "The")),
            ControlFlow::Continue(())
        );
        assert_eq!(
            body.fail(anyhow!("no token with id 4096 in this vocabulary")),
            ControlFlow::Break(())
        );
        body.close(Routed::Nothing, Stop::Budget)
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
        let mut body = opened(&mut socket, true).expect("the head goes out");
        assert_eq!(
            body.push(text(Channel::Content, "Hello")),
            ControlFlow::Break(())
        );
        body.close(Routed::Nothing, Stop::Budget)
            .expect_err("the failure surfaces");

        assert_eq!(socket.writes, 3, "it wrote past the disconnect");
    }

    /// The half a character a budget can cut a reply off in the middle of. It
    /// belongs to whichever channel was open, and it is the last thing written.
    #[test]
    fn the_text_left_over_at_the_end_reaches_the_client() {
        let mut socket = Socket::default();
        let mut body = opened(&mut socket, false).expect("nothing goes out");
        assert_eq!(
            body.push(text(Channel::Content, "The")),
            ControlFlow::Continue(())
        );
        body.close(text(Channel::Content, "\u{fffd}"), Stop::Budget)
            .expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert_eq!(parsed["choices"][0]["message"]["content"], "The\u{fffd}");
        assert_eq!(parsed["choices"][0]["finish_reason"], "length");
    }

    /// **A stop sequence ends the turn and says `stop`.** The trap this is
    /// about: a match reaches the generation as the same `Break` a client that
    /// hung up produces, which is [`Stop::Sink`], and `Sink` maps onto `length`
    /// — so a reply cut exactly where the client asked would be reported as one
    /// the budget cut short, and a client would resume a message that is
    /// finished.
    #[test]
    fn a_stop_sequence_ends_the_reply_and_reports_that_it_ended() {
        let mut socket = Socket::default();
        let mut body = cutting(&mut socket, false, &["\nUser:"]);

        assert_eq!(
            body.push(text(Channel::Content, "The answer.")),
            ControlFlow::Continue(())
        );
        assert_eq!(
            body.push(text(Channel::Content, "\nUser:")),
            ControlFlow::Break(()),
            "the generation was allowed to run on past the sequence"
        );
        // The budget is what the generation reports once the sink has broken,
        // and it is not what the client is told.
        body.close(Routed::Nothing, Stop::Sink)
            .expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["choices"][0]["message"]["content"], "The answer.");
    }

    /// The stop text is not in the output, and neither is anything the model
    /// produced behind it — including the bytes a detokenizer was still holding,
    /// which reach [`Body::close`] after the match and are past it.
    #[test]
    fn the_stop_text_and_everything_behind_it_is_cut() {
        let mut socket = Socket::default();
        let mut body = cutting(&mut socket, false, &["END"]);

        assert_eq!(
            body.push(text(Channel::Content, "Done. ")),
            ControlFlow::Continue(())
        );
        assert_eq!(
            body.push(text(Channel::Content, "END and more")),
            ControlFlow::Break(())
        );
        body.close(text(Channel::Content, " a tail"), Stop::Sink)
            .expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Done. ");
        assert!(!written.contains("END"), "{written}");
        assert!(!written.contains("a tail"), "{written}");
    }

    /// **Nothing is framed that a match would have to take back.** A frame is
    /// gone the moment it is flushed, so the tokens that could still be the
    /// first half of a sequence must reach the socket as no write at all —
    /// which is the claim, and the reply adding up correctly afterwards is not
    /// enough to make it.
    #[test]
    fn a_stream_frames_nothing_a_stop_sequence_would_have_to_retract() {
        let mut socket = Socket::default();
        let mut body = cutting(&mut socket, true, &["\nUser:"]);

        let before = body.out.writes;
        assert_eq!(
            body.push(text(Channel::Content, "The answer.")),
            ControlFlow::Continue(())
        );
        assert_eq!(body.out.writes, before + 1, "the plain text was held");

        // Each of these could still turn into the sequence, so none of them may
        // reach the socket.
        let held = body.out.writes;
        for token in ["\n", "Us", "er"] {
            assert_eq!(
                body.push(text(Channel::Content, token)),
                ControlFlow::Continue(())
            );
            assert_eq!(body.out.writes, held, "{token:?} was framed too early");
        }
        assert_eq!(
            body.push(text(Channel::Content, ":")),
            ControlFlow::Break(())
        );

        body.close(Routed::Nothing, Stop::Sink)
            .expect("it is written");
        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let sent = dechunked(written);
        let content: String = payloads(&sent)
            .iter()
            .filter_map(delta)
            .map(|(_, text)| text)
            .collect();
        assert_eq!(content, "The answer.");
    }

    /// Text that only looked like a sequence is released, and a generation that
    /// ends with the ambiguity outstanding still owes the client the text. A
    /// reply quietly missing its last few bytes is worse than a stop that did
    /// not fire.
    #[test]
    fn text_held_against_a_sequence_that_never_matched_still_reaches_the_client() {
        let mut socket = Socket::default();
        let mut body = cutting(&mut socket, false, &["\nUser:"]);

        for token in ["Answer.", "\nUse", "r"] {
            assert_eq!(
                body.push(text(Channel::Content, token)),
                ControlFlow::Continue(())
            );
        }
        body.close(Routed::Nothing, Stop::EndOfSequence)
            .expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Answer.\nUser");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    /// **The decision, asserted: a sequence is matched against what the client
    /// sees.** The model reasons in the client's own words, so a `stop` of
    /// `"\nUser:"` would fire inside the thinking of half the requests carrying
    /// one — cutting the turn off before the answer had started. `content` is
    /// the field the rule was written against, and `reasoning_content` is not
    /// it.
    #[test]
    fn a_stop_sequence_is_not_matched_against_the_thinking_channel() {
        let mut socket = Socket::default();
        let mut body = cutting(&mut socket, false, &["\nUser:"]);

        for routed in [
            text(Channel::Thinking, "They will say \nUser: next."),
            text(Channel::Content, "The answer."),
        ] {
            assert_eq!(
                body.push(routed.clone()),
                ControlFlow::Continue(()),
                "{routed:?} ended the turn"
            );
        }
        body.close(Routed::Nothing, Stop::EndOfSequence)
            .expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        let message = &parsed["choices"][0]["message"];
        assert_eq!(message["content"], "The answer.");
        assert_eq!(message["reasoning_content"], "They will say \nUser: next.");
    }

    /// The other half of that decision. The model can open a thinking channel in
    /// the middle of its content, and what the client's `content` holds is the
    /// two pieces joined — so the join is where a sequence spanning it exists,
    /// and holding back across the interruption is what finds it.
    #[test]
    fn a_stop_sequence_spanning_a_thinking_interruption_is_matched_in_the_join() {
        let mut socket = Socket::default();
        let mut body = cutting(&mut socket, false, &["\nUser:"]);

        assert_eq!(
            body.push(text(Channel::Content, "Answer.\nUs")),
            ControlFlow::Continue(())
        );
        assert_eq!(
            body.push(text(Channel::Thinking, "Second thoughts.")),
            ControlFlow::Continue(()),
            "the interruption ended the turn"
        );
        assert_eq!(
            body.push(text(Channel::Content, "er: again")),
            ControlFlow::Break(())
        );
        body.close(Routed::Nothing, Stop::Sink)
            .expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Answer.");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    /// A request that named no `stop` is answered exactly as it was before there
    /// were any: nothing held, nothing cut, and the reason the generation's own.
    #[test]
    fn a_request_that_named_no_stop_is_answered_as_it_always_was() {
        let mut socket = Socket::default();
        served(&mut socket, false, &reply()).expect("it is written");

        let (_, written) = socket.written.split_once("\r\n\r\n").expect("a body");
        let parsed: serde_json::Value = serde_json::from_str(written).expect("a json body");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello.");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    /// A client that hung up is not a turn the model ended, and the difference
    /// only matters where it is written down: nothing goes back to that client,
    /// but a reply cut short is a reply cut short.
    #[test]
    fn the_endings_a_generation_can_have_map_onto_the_ones_openai_has() {
        assert_eq!(finish(Stop::EndOfSequence, false), Finish::Stop);
        assert_eq!(finish(Stop::EndOfSequence, true), Finish::ToolCalls);
        assert_eq!(finish(Stop::Budget, false), Finish::Length);
        assert_eq!(finish(Stop::Sink, false), Finish::Length);
    }

    /// A budget that ran out is a budget that ran out, whatever the turn had
    /// asked for. A client told `tool_calls` runs a tool it was handed half of;
    /// one told `length` knows the reply is not whole.
    #[test]
    fn a_budget_that_ran_out_mid_call_is_still_a_budget_that_ran_out() {
        assert_eq!(finish(Stop::Budget, true), Finish::Length);
        assert_eq!(finish(Stop::Sink, true), Finish::Length);
    }

    /// The scheduler over the synthetic stack, with the pieces `schedule` holds
    /// around it. Everything below drives the same three functions the server
    /// does — [`take`], [`Continuous::step`] and [`dispatch`] — because a test
    /// that drove its own loop would be testing its own loop.
    struct Seated<'a> {
        engine: Continuous<'a>,
        live: HashMap<usize, Sender<Dispatch>>,
        stack: &'a Stack,
        head: &'a DenseProjection<'a>,
    }

    /// How many tokens the cases below ask for.
    const COUNT: usize = 4;

    impl<'a> Seated<'a> {
        fn new(stack: &'a Stack, head: &'a DenseProjection<'a>, slots: usize) -> Self {
            Self {
                engine: Continuous::new(&stack.config, slots, 2),
                live: HashMap::new(),
                stack,
                head,
            }
        }

        fn generator(&self) -> Generator<'a> {
            Generator::new(
                self.stack.model(),
                LmHead::for_config(&self.stack.config),
                self.head,
            )
        }

        /// A request handed over the way a connection thread hands one over, and
        /// the ticket and the receiving end it gets back.
        fn seat(&mut self, prompt: &[usize], budget: usize) -> (usize, Receiver<Dispatch>) {
            let (reply, dispatched) = mpsc::channel();
            let seating = Seating {
                ids: prompt.to_vec(),
                budget,
                reply,
            };
            take(
                &mut self.engine,
                &mut self.live,
                self.stack,
                Wanted::Seat(seating),
            );
            match dispatched.recv() {
                Ok(Dispatch::Seated(ticket)) => (ticket, dispatched),
                other => panic!("the engine seated nothing: {:?}", other.is_ok()),
            }
        }

        /// One step, and everything it produced sent on.
        fn step(&mut self, eos: usize) {
            let generator = self.generator();
            let stepped = self.engine.step(&generator, self.stack);
            dispatch(&mut self.engine, &mut self.live, self.stack, stepped, eos);
        }

        /// The engine run until it has nothing left.
        fn drain(&mut self, eos: usize) {
            while !self.engine.idle() {
                self.step(eos);
            }
        }
    }

    /// What one client saw: its tokens, and how it was told the turn ended.
    fn received(dispatched: &Receiver<Dispatch>) -> (Vec<usize>, Option<Stop>) {
        let mut tokens = Vec::new();
        while let Ok(dispatch) = dispatched.recv() {
            match dispatch {
                Dispatch::Token(id) => tokens.push(id),
                Dispatch::Done(stop) => return (tokens, Some(stop)),
                Dispatch::Seated(ticket) => panic!("seated twice, as {ticket}"),
            }
        }
        (tokens, None)
    }

    /// The same generation run alone, which is what a client's tokens are held
    /// against.
    fn alone(stack: &Stack, prompt: &[usize], count: usize) -> Vec<usize> {
        let head = stack.head();
        let generator = Generator::new(stack.model(), LmHead::for_config(&stack.config), &head);
        generator.generate(
            &mut inkling_core::ModelCache::new(&stack.config),
            prompt,
            count,
            stack,
        )
    }

    /// **Two clients at once, each sent its own request's tokens and nobody
    /// else's.** This is the whole of what wiring the socket to the scheduler
    /// has to get right on the way back: the engine produces a step's worth of
    /// tokens for every seat at once, and which connection each belongs to is a
    /// lookup that could be wrong in a way that reads as a plausible reply.
    ///
    /// Held against each generation run alone, so it says the tokens are right
    /// and not only that the two clients got different ones.
    #[test]
    fn two_clients_are_each_sent_the_tokens_of_their_own_request() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let prompts = [sequence[..3].to_vec(), sequence[3..].to_vec()];
        let want: Vec<Vec<usize>> = prompts
            .iter()
            .map(|prompt| alone(&stack, prompt, COUNT))
            .collect();
        assert_ne!(want[0], want[1], "two generations to tell apart");

        let mut seated = Seated::new(&stack, &head, 2);
        let clients: Vec<(usize, Receiver<Dispatch>)> = prompts
            .iter()
            .map(|prompt| seated.seat(prompt, COUNT))
            .collect();

        // An id the synthetic stack cannot produce, so nothing here ends on a
        // terminator and every request ends on its budget.
        seated.drain(usize::MAX);
        drop(seated);

        for (at, (_, dispatched)) in clients.iter().enumerate() {
            let (tokens, stop) = received(dispatched);
            assert_eq!(tokens, want[at], "client {at}");
            assert_eq!(stop, Some(Stop::Budget), "client {at}");
        }
    }

    /// **A client that hung up frees its slot, and the request behind it is
    /// admitted into the slot it left.** A seat held for a connection that has
    /// gone decodes a whole budget nobody reads, and under the load a scheduler
    /// exists for that is not an edge case.
    ///
    /// The hanging up here is the receiving end going away, which is what a
    /// connection thread leaving its scope does — see [`Seat`], which is the
    /// other end of the same event.
    #[test]
    fn a_client_that_hung_up_frees_its_slot_for_the_request_behind_it() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let (abandoned, behind) = (sequence[..3].to_vec(), sequence[3..].to_vec());
        let want = alone(&stack, &behind, COUNT);

        let mut seated = Seated::new(&stack, &head, 1);
        let (_, gone) = seated.seat(&abandoned, COUNT);
        let (_, waiting) = seated.seat(&behind, COUNT);

        // Far enough in that the abandoned seat holds a filled prompt and a
        // token of its own, which is the state one is actually abandoned in.
        seated.step(usize::MAX);
        seated.step(usize::MAX);
        assert_eq!(seated.engine.seated(), 1);
        assert_eq!(seated.engine.waiting(), 1);

        drop(gone);
        // The step that notices, which is the next one to send it a token. The
        // seat is given up in that step and filled in the one after it, because
        // a slot is admitted into at the top of a step.
        seated.step(usize::MAX);
        assert_eq!(seated.engine.seated(), 0, "the abandoned seat was kept");
        assert_eq!(seated.engine.waiting(), 1);

        seated.step(usize::MAX);
        assert_eq!(
            (seated.engine.seated(), seated.engine.waiting()),
            (1, 0),
            "the freed slot did not take the request behind it"
        );

        seated.drain(usize::MAX);
        drop(seated);
        let (tokens, stop) = received(&waiting);
        assert_eq!(tokens, want, "the request behind read the abandoned keys");
        assert_eq!(stop, Some(Stop::Budget));
    }

    /// **The terminator frees the seat on the step that produced it**, rather
    /// than a round trip later.
    ///
    /// The scheduler stops a request when its budget runs out and knows nothing
    /// about a vocabulary, so a server that left the client's own thread to
    /// notice the end of a turn would give the seat back after the engine had
    /// already built another step into it. Asserted as the slot being free
    /// before the next step: the request behind is seated on it.
    #[test]
    fn a_turn_the_model_ended_frees_its_seat_before_the_next_step() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let (first, second) = (sequence[..3].to_vec(), sequence[3..].to_vec());

        // The stack's own first token for this prompt, used as the id that ends
        // a turn — so the terminator arrives on the first decode step rather
        // than at a budget nothing here would reach.
        let eos = alone(&stack, &first, 1)[0];

        let mut seated = Seated::new(&stack, &head, 1);
        let (ticket, ending) = seated.seat(&first, COUNT);
        let (_, behind) = seated.seat(&second, COUNT);

        // Up to and including the step that decodes the terminator, which is
        // the first step to produce anything at all. The budget is four tokens
        // and this must end on the first of them.
        let mut steps = 0;
        while seated.live.contains_key(&ticket) {
            seated.step(eos);
            steps += 1;
            assert!(steps < COUNT, "the terminator did not end the turn");
        }
        assert_eq!(
            (seated.engine.seated(), seated.engine.waiting()),
            (0, 1),
            "the ended turn kept its seat"
        );

        let (tokens, stop) = received(&ending);
        assert_eq!(tokens, [eos], "the terminator reaches the client");
        assert_eq!(stop, Some(Stop::EndOfSequence));

        // And the next step is the one the request behind it is seated on,
        // rather than the step after a round trip through its client.
        seated.step(eos);
        assert_eq!((seated.engine.seated(), seated.engine.waiting()), (1, 0));

        seated.drain(eos);
        drop(seated);
        assert_eq!(received(&behind).1, Some(Stop::Budget));
    }

    /// **The loop itself, driven across a thread boundary.** Everything above
    /// calls [`take`] and [`dispatch`] straight, which is the loop's pieces and
    /// not the loop: what [`stepping`] adds is *when* it blocks, and a scheduler
    /// that blocked at the wrong moment would stall a request already running
    /// until an unrelated one happened to arrive.
    ///
    /// So the requests are submitted from another thread, and the second is
    /// submitted only after the first has produced a token — which is the
    /// interleaving that matters, a request arriving while the engine is busy
    /// rather than while it is waiting. The loop returns when the last sender is
    /// dropped, and that it returns at all is half of what this asserts.
    #[test]
    fn the_loop_answers_a_request_that_arrives_while_it_is_already_running() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = Generator::new(stack.model(), LmHead::for_config(&stack.config), &head);
        let sequence = stack.sequence();
        let prompts = [sequence[..3].to_vec(), sequence[3..].to_vec()];
        let want: Vec<Vec<usize>> = prompts
            .iter()
            .map(|prompt| alone(&stack, prompt, COUNT))
            .collect();
        assert_ne!(want[0], want[1], "two generations to tell apart");

        let (asking, asked) = mpsc::channel();
        let (first, listening) = mpsc::channel();
        let (second, joining) = mpsc::channel();

        let client = std::thread::spawn(move || {
            let seat = |prompt: &[usize]| {
                let (reply, dispatched) = mpsc::channel();
                let seating = Seating {
                    ids: prompt.to_vec(),
                    budget: COUNT,
                    reply,
                };
                asking
                    .send(Wanted::Seat(seating))
                    .expect("the engine is up");
                match dispatched.recv() {
                    Ok(Dispatch::Seated(_)) => dispatched,
                    _ => panic!("the engine seated nothing"),
                }
            };

            let one = seat(&prompts[0]);
            // Not until the engine has actually produced something, so the
            // second request joins a batch that is running.
            let started = one.recv().expect("a first token");
            let two = seat(&prompts[1]);

            first.send((started, received(&one))).expect("a listener");
            second.send(received(&two)).expect("a listener");
            // Every sender gone is what ends the loop.
        });

        stepping(&stack.config, &stack, &generator, asked, usize::MAX, 2)
            .expect("the loop ends when its last client does");
        client.join().expect("the client thread");

        let (started, (rest, stop)) = listening.recv().expect("the first request");
        let Dispatch::Token(opened) = started else {
            panic!("the first request ended before it began")
        };
        let mut tokens = vec![opened];
        tokens.extend(rest);
        assert_eq!(tokens, want[0], "the request that was already running");
        assert_eq!(stop, Some(Stop::Budget));

        let (tokens, stop) = joining.recv().expect("the second request");
        assert_eq!(tokens, want[1], "the request that joined it");
        assert_eq!(stop, Some(Stop::Budget));
    }

    /// **There is no path out of a connection thread that does not give the
    /// ticket back.** A write that fails, a body that would not parse, a client
    /// that hung up, a panic unwinding — a seat that survived any of them is a
    /// slot decoding a budget nobody reads.
    ///
    /// The panic is the case a guard exists for, because it is the one no
    /// `return` covers.
    #[test]
    fn every_way_out_of_a_connection_gives_the_seat_back() {
        let (engine, asked) = mpsc::channel();

        let leaving = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _seat = Seat {
                ticket: 7,
                engine: &engine,
            };
            panic!("a connection thread came apart");
        }));
        assert!(leaving.is_err(), "the panic was swallowed");

        {
            let _seat = Seat {
                ticket: 9,
                engine: &engine,
            };
        }

        let given: Vec<usize> = asked
            .try_iter()
            .map(|wanted| match wanted {
                Wanted::Release(ticket) => ticket,
                Wanted::Seat(_) => panic!("a seat asked for rather than given back"),
            })
            .collect();
        assert_eq!(given, [7, 9]);
    }

    /// The name a model is listed and answered under is the directory it came
    /// from, not the path a caller happened to type.
    #[test]
    fn a_checkpoint_is_named_after_the_directory_it_was_loaded_from() {
        assert_eq!(model_name(Path::new("models/Inkling-Small-mxfp4")), MODEL);
        assert_eq!(model_name(Path::new("models/Inkling-Small-mxfp4/")), MODEL);
    }
}
