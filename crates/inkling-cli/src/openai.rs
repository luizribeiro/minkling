//! The wire shapes of `/v1/chat/completions`, and the framing that carries them.
//!
//! Nothing here knows what a model is. What arrives is a request to parse and
//! then text a token at a time; what leaves is either a stream of
//! `text/event-stream` frames or one JSON body. Which of the two a caller asked
//! for changes almost nothing, and that is deliberate — see [`Completion`].
//!
//! # Streaming is the path, not a mode
//!
//! A decode step is 9.2 s. Twenty tokens is three minutes, so a client that
//! cannot see the reply until it is finished is looking at a hung server for as
//! long as the reply takes. Everything here is built around the token that just
//! arrived; the collected form is the same tokens, added up.
//!
//! # The thinking channel goes in a field of its own
//!
//! The model opens `<|content_thinking|>` unprompted — `generate`'s docs measured
//! it doing so on the first templated request anyone made of this checkpoint —
//! and the tokenizer renders special tokens literally rather than swallowing
//! them. So a server has to decide what a client sees, and all three answers are
//! real behaviour rather than cosmetics.
//!
//! Left in `content`, an OpenAI client renders `<|content_thinking|>The user
//! said` verbatim, markers and reasoning and all, because `content` is the field
//! every one of them prints. Stripped, the reasoning is gone and a client that
//! sends the reply back in the next request sends a turn the model never took.
//!
//! So it is split: the answer in `content`, the reasoning in
//! `reasoning_content`, and the markers themselves in neither. `reasoning_content`
//! rather than any other name because it is the field the checkpoint's *own*
//! `chat_template.jinja` reads an assistant turn's thinking out of, which makes
//! the round trip closed — [`crate::chat`] writes back what this wrote out.

use serde::Deserialize;

use crate::chat::{Channel, ChatError, Message};

/// What a client asked for.
///
/// Unknown fields are accepted and ignored, which is what an OpenAI client
/// needs — they all send fields this does not have. Two exceptions are refused
/// rather than ignored, because ignoring them changes the shape of the answer
/// and not just its wording: `tools`, whose disappearance turns a tool call into
/// a refusal to make one, and an `n` above one, which asks for choices this
/// would silently return one of.
///
/// The sampling parameters — `temperature`, `top_p`, and the rest — are among
/// the ignored. Decoding is greedy and only greedy; see
/// [`greedy`](inkling_core::greedy) for why a sampler over this port's logits
/// would draw from an order that is legitimately not the reference's.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI's older name for the budget.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI's newer name for the same thing. Both are read; the newer wins.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    #[error("the request body is not the JSON this takes: {0}")]
    Malformed(String),

    #[error("this server implements no tools")]
    Tools,

    #[error("this server returns one choice, so n must be 1")]
    Choices,

    #[error("max_tokens must be a count of at least one")]
    NotACount,

    #[error(transparent)]
    Chat(#[from] ChatError),
}

impl ChatRequest {
    pub fn parse(body: &str) -> Result<Self, RequestError> {
        let request: Self =
            serde_json::from_str(body).map_err(|err| RequestError::Malformed(err.to_string()))?;
        if request.tools.is_some() {
            return Err(RequestError::Tools);
        }
        if request.n.is_some_and(|choices| choices != 1) {
            return Err(RequestError::Choices);
        }
        if request.budget() == Some(0) {
            return Err(RequestError::NotACount);
        }
        Ok(request)
    }

    /// The budget this request named, under either of OpenAI's two names for
    /// it, or `None` if it named none.
    ///
    /// A budget of zero is refused rather than served, for the reason `generate`
    /// refuses one: no token at all is decoded, not even the prompt's prefill,
    /// so it is a mistake rather than a request.
    fn budget(&self) -> Option<usize> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// The budget to decode under, which is the server's own where the request
    /// named none.
    pub fn max_tokens(&self, default: usize) -> usize {
        self.budget().unwrap_or(default)
    }
}

/// Why a completion ended, as OpenAI spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// The budget ran out with the model still going.
    Length,
}

impl Finish {
    fn reason(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
        }
    }
}

/// The frame that ends an event stream, which is OpenAI's and not the SSE
/// specification's.
pub const DONE: &str = "data: [DONE]\n\n";

/// One JSON value as a `text/event-stream` frame.
///
/// A frame is `data: `, the payload, and the blank line that ends the event. The
/// payload is compact JSON, so it holds no newline of its own — every control
/// character in a string is escaped — and nothing here has to escape a payload
/// that would otherwise end its own frame early.
fn frame(payload: &serde_json::Value) -> String {
    format!("data: {payload}\n\n")
}

/// A completion being written out, in both of the forms at once.
///
/// The streaming and the collected response are not two code paths here. Every
/// delta is added to the reply *and* framed, so the body a non-streaming request
/// gets is by construction the frames a streaming one would have got, added up.
/// The alternative — a collected path that re-runs the generation and joins the
/// text itself — is where the two forms drift apart, and the drift is invisible
/// until a client compares them.
///
/// The cost of doing both is a `String` per reply, against 9.2 s a token.
#[derive(Debug)]
pub struct Completion {
    id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    content: String,
    reasoning: String,
}

impl Completion {
    pub fn new(id: String, created: u64, model: String, prompt_tokens: usize) -> Self {
        Self {
            id,
            created,
            model,
            prompt_tokens,
            completion_tokens: 0,
            content: String::new(),
            reasoning: String::new(),
        }
    }

    fn envelope(&self, object: &str, choice: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "object": object,
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        })
    }

    /// The frame that opens the stream: the role, and no text yet.
    ///
    /// OpenAI sends it, and clients that build a message out of the deltas need
    /// something to build it on before any text arrives.
    pub fn opening(&self) -> String {
        frame(&self.envelope(
            "chat.completion.chunk",
            serde_json::json!({
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null,
            }),
        ))
    }

    /// One token's worth of text, added to the reply and framed.
    ///
    /// The token is counted whether or not it showed anything. What the budget
    /// spends is decode steps, and a step that produced no visible text cost the
    /// same 9.2 s as one that did.
    pub fn push(&mut self, channel: Channel, text: &str) -> Option<String> {
        self.completion_tokens += 1;
        self.delta(channel, text)
    }

    /// Text no token contributed: the bytes a detokenizer was still holding when
    /// the generation ended, which a budget that cut a reply off mid-character
    /// leaves behind.
    ///
    /// Added and framed like any other delta, and counted against nothing. No
    /// decode step produced it — it is the tail of ones already charged for —
    /// and `usage` a client reconciles against a budget has to say so.
    pub fn tail(&mut self, channel: Channel, text: &str) -> Option<String> {
        self.delta(channel, text)
    }

    /// `None` where there is no text — a marker, or a token that only completed
    /// part of a character. Both are ordinary and neither is an event: a frame
    /// carrying an empty delta is one a client has to be careful about for no
    /// reason.
    fn delta(&mut self, channel: Channel, text: &str) -> Option<String> {
        if text.is_empty() {
            return None;
        }

        let field = match channel {
            Channel::Thinking => {
                self.reasoning.push_str(text);
                "reasoning_content"
            }
            Channel::Content => {
                self.content.push_str(text);
                "content"
            }
        };
        Some(frame(&self.envelope(
            "chat.completion.chunk",
            serde_json::json!({
                "index": 0,
                "delta": {field: text},
                "finish_reason": null,
            }),
        )))
    }

    /// The frames that end the stream: why it ended, and then the terminator.
    pub fn closing(&self, finish: Finish) -> String {
        let last = frame(&self.envelope(
            "chat.completion.chunk",
            serde_json::json!({
                "index": 0,
                "delta": {},
                "finish_reason": finish.reason(),
            }),
        ));
        format!("{last}{DONE}")
    }

    /// The whole reply as one body, which is every delta that was pushed.
    ///
    /// `reasoning_content` is omitted rather than empty when the model opened no
    /// thinking channel, so that a reply with none looks like an ordinary
    /// OpenAI one.
    pub fn collected(&self, finish: Finish) -> String {
        let mut message = serde_json::json!({
            "role": "assistant",
            "content": self.content,
        });
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = self.reasoning.as_str().into();
        }

        let mut body = self.envelope(
            "chat.completion",
            serde_json::json!({
                "index": 0,
                "message": message,
                "finish_reason": finish.reason(),
            }),
        );
        body["usage"] = serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        });
        body.to_string()
    }
}

/// What `GET /v1/models` answers: the one checkpoint this process loaded.
///
/// One, because loading it is ~30 s and 16.7 GiB and a process holds exactly one.
/// A client that lists models to pick one has a list of length one to pick from,
/// which is the honest answer.
pub fn models(model: &str, created: u64) -> String {
    serde_json::json!({
        "object": "list",
        "data": [{
            "id": model,
            "object": "model",
            "created": created,
            "owned_by": "inklingrs",
        }],
    })
    .to_string()
}

/// A refusal, in the shape OpenAI clients unpack one from.
pub fn error(message: &str) -> String {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
        },
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use inkling_core::Utf8Stream;

    use super::*;
    use crate::wire::{delta, frames, payload, payloads};

    const MODEL: &str = "Inkling-Small-mxfp4";
    const CREATED: u64 = 1_774_000_000;
    const PROMPT_TOKENS: usize = 16;

    fn completion() -> Completion {
        Completion::new(
            "chatcmpl-1".to_string(),
            CREATED,
            MODEL.to_string(),
            PROMPT_TOKENS,
        )
    }

    fn request(body: serde_json::Value) -> Result<ChatRequest, RequestError> {
        ChatRequest::parse(&body.to_string())
    }

    fn refused(extra: serde_json::Value) -> RequestError {
        request(asking(extra)).expect_err("the request is refused")
    }

    fn asking(extra: serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({"messages": [{"role": "user", "content": "Hi"}]});
        for (key, value) in extra.as_object().expect("an object") {
            body[key] = value.clone();
        }
        body
    }

    /// A whole streamed reply, as it reaches a socket.
    fn streamed(completion: &mut Completion, deltas: &[(Channel, &str)], finish: Finish) -> String {
        let mut stream = completion.opening();
        for (channel, text) in deltas {
            if let Some(chunk) = completion.push(*channel, text) {
                stream.push_str(&chunk);
            }
        }
        stream.push_str(&completion.closing(finish));
        stream
    }

    const REPLY: [(Channel, &str); 5] = [
        (Channel::Thinking, "Weigh"),
        (Channel::Thinking, " it up."),
        (Channel::Content, "Caf"),
        (Channel::Content, "é."),
        (Channel::Content, ""),
    ];

    /// What a client parsing the stream sees, frame by frame: a role to build
    /// on, one delta per token that had text, a reason, and the terminator.
    #[test]
    fn a_stream_opens_with_a_role_and_ends_with_a_reason_and_a_terminator() {
        let stream = streamed(&mut completion(), &REPLY, Finish::Stop);
        // `payloads` drops the terminator and refuses a stream without one, so
        // a chunk count here is a count of the chunks alone: a role, and four
        // tokens that had text, and the reason. The fifth token contributed
        // nothing and is not a frame.
        let chunks = payloads(&stream);
        assert_eq!(chunks.len(), 6, "{chunks:?}");

        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["object"], "chat.completion.chunk");
        assert_eq!(chunks[0]["model"], MODEL);

        let last = chunks.last().expect("a last chunk");
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["choices"][0]["delta"], serde_json::json!({}));
    }

    /// A budget that ran out and a model that ended its turn are different
    /// endings, and a client resuming the first would resume a message the model
    /// considers finished.
    #[test]
    fn a_budget_that_ran_out_is_reported_apart_from_a_turn_the_model_ended() {
        let stream = streamed(&mut completion(), &REPLY, Finish::Length);
        let last = payloads(&stream).pop().expect("a last chunk");
        assert_eq!(last["choices"][0]["finish_reason"], "length");
    }

    /// The two channels reach a client in two fields, and neither carries the
    /// other's text.
    #[test]
    fn thinking_and_content_arrive_in_fields_of_their_own() {
        let stream = streamed(&mut completion(), &REPLY, Finish::Stop);
        let deltas: Vec<(String, String)> = payloads(&stream).iter().filter_map(delta).collect();

        assert_eq!(
            deltas,
            [
                ("reasoning_content".to_string(), "Weigh".to_string()),
                ("reasoning_content".to_string(), " it up.".to_string()),
                ("content".to_string(), "Caf".to_string()),
                ("content".to_string(), "é.".to_string()),
            ]
        );
    }

    /// A token that completed no character is not an event. Clients differ in
    /// what they make of an empty delta, and there is nothing for one to make.
    #[test]
    fn a_token_that_contributed_no_text_is_not_framed() {
        let mut completion = completion();
        assert_eq!(completion.push(Channel::Content, ""), None);
        assert!(completion.push(Channel::Content, "Hi").is_some());
    }

    /// The claim the whole type exists to make: the body a non-streaming request
    /// gets is the stream, added up. Asserted by adding the stream up rather
    /// than by restating the text, so that a change to either path has to keep
    /// them equal.
    #[test]
    fn the_collected_body_is_the_stream_added_up() {
        let mut completion = completion();
        let stream = streamed(&mut completion, &REPLY, Finish::Stop);

        let mut content = String::new();
        let mut reasoning = String::new();
        for chunk in payloads(&stream) {
            match delta(&chunk) {
                Some((field, text)) if field == "content" => content.push_str(&text),
                Some((_, text)) => reasoning.push_str(&text),
                None => {}
            }
        }

        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Stop)).expect("a json body");
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], content);
        assert_eq!(message["reasoning_content"], reasoning);
        assert_eq!(message["role"], "assistant");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    /// A reply the model opened no thinking channel for looks like an ordinary
    /// OpenAI one rather than one with an empty field bolted on.
    #[test]
    fn a_reply_without_thinking_carries_no_reasoning_field() {
        let mut completion = completion();
        assert!(completion.push(Channel::Content, "Hello.").is_some());
        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Stop)).expect("a json body");

        assert_eq!(
            body["choices"][0]["message"]["reasoning_content"],
            serde_json::Value::Null
        );
        assert_eq!(body["choices"][0]["message"]["content"], "Hello.");
    }

    /// Every token is counted, including the ones that showed nothing — a
    /// marker, or half a character. Each cost a decode step, and usage that
    /// counted only the visible ones would understate what the request spent.
    #[test]
    fn usage_counts_every_token_the_budget_was_charged_for() {
        let mut completion = completion();
        for (channel, text) in REPLY {
            let _ = completion.push(channel, text);
        }
        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Stop)).expect("a json body");

        assert_eq!(body["usage"]["prompt_tokens"], PROMPT_TOKENS);
        assert_eq!(body["usage"]["completion_tokens"], REPLY.len());
        assert_eq!(body["usage"]["total_tokens"], PROMPT_TOKENS + REPLY.len());
    }

    /// What no token produced is not counted as one. The bytes left over at the
    /// end of a reply are the tail of tokens already charged for, so counting
    /// them again reports a completion one token longer than the budget allowed —
    /// which is what a client reconciling `usage` against `max_tokens` reads as
    /// a server that overran.
    #[test]
    fn the_text_left_over_at_the_end_is_not_a_token() {
        let mut completion = completion();
        let _ = completion.push(Channel::Content, "The");
        assert!(completion.tail(Channel::Content, "\u{fffd}").is_some());

        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Length)).expect("a json body");
        assert_eq!(body["usage"]["completion_tokens"], 1);
        assert_eq!(body["choices"][0]["message"]["content"], "The\u{fffd}");
    }

    /// The guarantee [`Utf8Stream`] makes, carried through the framing: a
    /// character whose bytes arrived across three tokens reaches the client
    /// whole, in one frame, rather than as three replacement characters spread
    /// over three.
    ///
    /// Driven a byte at a time, which is the worst split there is and the one a
    /// byte-level vocabulary produces whenever no merge covers a character.
    #[test]
    fn a_character_split_across_tokens_reaches_the_client_in_one_frame() {
        let text = "Café, 日本語, 🙂.";
        let mut stream = Utf8Stream::new();
        let mut completion = completion();

        let mut frames = String::new();
        for byte in text.as_bytes() {
            if let Some(chunk) = completion.push(Channel::Content, &stream.push(&[*byte])) {
                frames.push_str(&chunk);
            }
        }
        assert!(stream.finish().is_empty(), "the text ends mid-character");

        frames.push_str(&completion.closing(Finish::Stop));
        let arrived: String = payloads(&frames)
            .iter()
            .filter_map(delta)
            .map(|(_, text)| text)
            .collect();
        assert_eq!(arrived, text);
        assert!(
            !arrived.contains(char::REPLACEMENT_CHARACTER),
            "{arrived:?}"
        );
    }

    /// A frame's payload is one line however many the text has, so a reply
    /// carrying a blank line does not terminate its own frame two tokens early.
    #[test]
    fn text_with_newlines_in_it_stays_inside_one_frame() {
        let mut completion = completion();
        let chunk = completion
            .push(Channel::Content, "one\n\ntwo")
            .expect("a frame");

        assert_eq!(frames(&chunk).len(), 1, "{chunk:?}");
        let (_, text) = payload(&chunk).as_ref().and_then(delta).expect("a delta");
        assert_eq!(text, "one\n\ntwo");
    }

    #[test]
    fn a_body_that_is_not_json_is_refused() {
        assert!(matches!(
            ChatRequest::parse("{\"messages\":"),
            Err(RequestError::Malformed(_))
        ));
    }

    /// The fields every OpenAI client sends and this does not have. Ignored
    /// rather than refused, because refusing them refuses every client there is.
    #[test]
    fn the_sampling_parameters_are_accepted_and_ignored() {
        let request = request(asking(
            serde_json::json!({"temperature": 0.7, "top_p": 0.9, "model": "gpt-4", "seed": 1}),
        ))
        .expect("parses");
        assert_eq!(request.messages.len(), 1);
        assert!(!request.stream);
    }

    /// Two fields that change the shape of the answer rather than its wording,
    /// which is why these are the two that are refused.
    #[test]
    fn tools_and_a_second_choice_are_refused() {
        assert_eq!(
            refused(serde_json::json!({"tools": []})),
            RequestError::Tools
        );
        assert_eq!(refused(serde_json::json!({"n": 2})), RequestError::Choices);
        assert!(request(asking(serde_json::json!({"n": 1}))).is_ok());
    }

    /// The newer name wins where a client sends both, which is what a client
    /// sending both means by it.
    #[test]
    fn the_budget_comes_from_either_name_and_falls_back_to_the_servers() {
        const DEFAULT: usize = 64;
        let budget = |extra| request(asking(extra)).expect("parses").max_tokens(DEFAULT);

        assert_eq!(budget(serde_json::json!({"max_tokens": 3})), 3);
        assert_eq!(budget(serde_json::json!({"max_completion_tokens": 5})), 5);
        assert_eq!(
            budget(serde_json::json!({"max_tokens": 3, "max_completion_tokens": 5})),
            5
        );
        assert_eq!(budget(serde_json::json!({})), DEFAULT);
    }

    #[test]
    fn a_budget_of_nothing_is_refused() {
        assert_eq!(
            refused(serde_json::json!({"max_tokens": 0})),
            RequestError::NotACount
        );
    }

    /// A role the turn structure cannot map is a bad request and says which
    /// role, rather than arriving as a generation nobody can read.
    #[test]
    fn a_role_that_cannot_be_mapped_surfaces_as_a_request_error() {
        let body = serde_json::json!({"messages": [{"role": "sudo", "content": "Hi"}]});
        let request = ChatRequest::parse(&body.to_string()).expect("the body parses");
        let refused = crate::chat::prompt(&request.messages).expect_err("the role is refused");

        assert_eq!(refused, ChatError::UnknownRole("sudo".to_string()));
        let message = RequestError::from(refused).to_string();
        assert!(message.contains("sudo"), "{message}");
    }

    /// The listing a client picks a model from, which has one entry because a
    /// process holds one checkpoint.
    #[test]
    fn the_model_listing_names_the_checkpoint_that_was_loaded() {
        let listing: serde_json::Value =
            serde_json::from_str(&models(MODEL, CREATED)).expect("a json body");

        assert_eq!(listing["object"], "list");
        assert_eq!(listing["data"].as_array().expect("a list").len(), 1);
        assert_eq!(listing["data"][0]["id"], MODEL);
        assert_eq!(listing["data"][0]["object"], "model");
    }

    #[test]
    fn a_refusal_carries_its_message_where_a_client_looks_for_one() {
        let refused: serde_json::Value =
            serde_json::from_str(&error("the tool role needs tool calling")).expect("a json body");

        assert_eq!(
            refused["error"]["message"],
            "the tool role needs tool calling"
        );
        assert_eq!(refused["error"]["type"], "invalid_request_error");
    }
}
