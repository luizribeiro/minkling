//! The wire shapes and framing of `/v1/chat/completions`.
//!
//! Nothing here knows what a model is. What arrives is a request to parse and
//! then text a token at a time; what leaves is either a stream of
//! `text/event-stream` frames or one JSON body. Which of the two a caller asked
//! for changes almost nothing, and that is deliberate — see [`Completion`].
//!
//! # Streaming is the path, not a mode
//!
//! A decode step is 0.055 s on the device path and 9.0 s on the CPU's, so a
//! client that cannot see the reply until it is finished is looking at a hung
//! server for as long as the reply takes. Everything here is built around the
//! token that just arrived; the collected form is the same tokens, added up.
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
//!
//! # A tool call is a third shape, and it is not text
//!
//! The model spells a call out in its own turn — see [`crate::chat`] for the
//! markers — and what a client needs is `tool_calls`, a `finish_reason` of
//! `tool_calls`, and an id per call for the result to answer by. None of the
//! three is text, so none of them goes through the two channels: a call reaches
//! [`Completion`] already assembled, and what it produces is a delta of its own.
//!
//! The id is minted here because the model does not produce one. It only has to
//! be unique inside a conversation — a client sends it back on the `tool`
//! message, and [`crate::chat`] looks the tool's name up by it — so it is the
//! completion's own id and the call's index in it.

use serde::Deserialize;

use crate::chat::{Call, Channel, ChatError, Message, Routed};
use crate::stop::MOST_SEQUENCES;

/// What a client asked for.
///
/// Unknown fields are accepted and ignored, which is what an OpenAI client
/// needs — they all send fields this does not have. Two exceptions are refused
/// rather than ignored, because ignoring them changes the shape of the answer
/// and not just its wording: a `tool_choice` this does not implement, whose
/// disappearance turns "you must call a tool" into a paragraph about calling
/// one, and an `n` above one, which asks for choices this would silently return
/// one of.
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
    /// The specs to declare, held as they arrived: [`crate::chat`] builds the
    /// declaration out of whatever the client sent, the way the template does,
    /// rather than out of a shape this insisted on first.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Where to cut the reply, as a string or a list of them. Held as it
    /// arrived because OpenAI spells it both ways and the two mean the same
    /// thing; see [`crate::stop`] for what it is matched against.
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

/// What a client wants of the stream beyond the reply itself.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// Which of OpenAI's `tool_choice` values this implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    /// The model decides, which is the default and what a declaration alone
    /// expresses.
    Auto,
    /// The tools are not declared at all, which is what leaves the model
    /// nothing to call.
    None,
}

impl Choice {
    /// What `tool_choice` asked for, or which value it is that this does not
    /// implement.
    ///
    /// `required` and a named function are the two it does not. Both are a
    /// constraint on what the model may produce, and there is nothing in a
    /// prompt that constrains it — honouring either needs the sampler to refuse
    /// the tokens that would leave the turn, which is a decode-time mechanism
    /// this engine has no flag for. Refused rather than treated as `auto`,
    /// because a client that asked for a call and got a sentence has no way to
    /// tell that from a model that would not call one.
    fn parse(asked: Option<&serde_json::Value>) -> Result<Self, RequestError> {
        let Some(asked) = asked else {
            return Ok(Choice::Auto);
        };
        match asked.as_str() {
            Some("auto") => Ok(Choice::Auto),
            Some("none") => Ok(Choice::None),
            _ => Err(RequestError::ToolChoice(asked.to_string())),
        }
    }
}

/// The sequences a `stop` names, or which value it is that cannot be one.
///
/// **An empty sequence is refused rather than dropped.** It matches before the
/// first token of every reply, so a request carrying one would be answered with
/// nothing at all — which a client reads as a model that had nothing to say
/// rather than as a `stop` it should not have sent.
///
/// The count is OpenAI's own limit and is refused rather than truncated, for
/// the reason a `tool_choice` this cannot honour is refused: a client whose
/// fifth sequence was quietly dropped gets a reply running past a stop it asked
/// for, which is the failure `stop` exists to prevent.
fn sequences(asked: Option<&serde_json::Value>) -> Result<Vec<String>, RequestError> {
    let Some(asked) = asked else {
        return Ok(Vec::new());
    };
    let named: Vec<String> = match asked {
        serde_json::Value::Null => Vec::new(),
        serde_json::Value::String(one) => vec![one.clone()],
        serde_json::Value::Array(many) => many
            .iter()
            .map(|one| match one.as_str() {
                Some(sequence) => Ok(sequence.to_string()),
                None => Err(RequestError::Stop(asked.to_string())),
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(RequestError::Stop(asked.to_string())),
    };

    if named.len() > MOST_SEQUENCES {
        return Err(RequestError::Stop(asked.to_string()));
    }
    if named.iter().any(String::is_empty) {
        return Err(RequestError::EmptyStop);
    }
    Ok(named)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    #[error("the request body is not the JSON this takes: {0}")]
    Malformed(String),

    #[error("this server implements tool_choice auto and none, not {0}")]
    ToolChoice(String),

    #[error("this server returns one choice, so n must be 1")]
    Choices,

    #[error("max_tokens must be a count of at least one")]
    NotACount,

    #[error("stop takes a string or a list of at most {MOST_SEQUENCES} strings, not {0}")]
    Stop(String),

    #[error("a stop sequence must not be empty: it would match before the first token")]
    EmptyStop,

    #[error(transparent)]
    Chat(#[from] ChatError),
}

impl ChatRequest {
    pub fn parse(body: &str) -> Result<Self, RequestError> {
        let request: Self =
            serde_json::from_str(body).map_err(|err| RequestError::Malformed(err.to_string()))?;
        Choice::parse(request.tool_choice.as_ref())?;
        sequences(request.stop.as_ref())?;
        if request.n.is_some_and(|choices| choices != 1) {
            return Err(RequestError::Choices);
        }
        if request.budget() == Some(0) {
            return Err(RequestError::NotACount);
        }
        Ok(request)
    }

    /// Where this request's reply is cut, which is nothing for a request that
    /// named no `stop`.
    ///
    /// `stop` was checked when the request was parsed, so a value that is not
    /// sequences never reaches here.
    pub fn stopping(&self) -> Vec<String> {
        sequences(self.stop.as_ref()).unwrap_or_default()
    }

    /// Whether the client asked for token counts on the stream, which is a
    /// question only a stream has.
    pub fn wants_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .is_some_and(|options| options.include_usage)
    }

    /// The specs the prompt declares, which is none of them where the caller
    /// asked for none.
    ///
    /// `tool_choice` was checked when the request was parsed, so a value this
    /// does not implement never reaches here.
    pub fn declared(&self) -> &[serde_json::Value] {
        match Choice::parse(self.tool_choice.as_ref()) {
            Ok(Choice::None) => &[],
            _ => self.tools.as_deref().unwrap_or_default(),
        }
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
    /// The model ended its turn having asked for a tool, which is what a client
    /// branches on to run one rather than to show the reply.
    ToolCalls,
}

impl Finish {
    fn reason(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
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
/// The cost of doing both is a `String` per reply, against a decode step.
#[derive(Debug)]
pub struct Completion {
    id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    content: String,
    reasoning: String,
    /// The calls the model asked for, each as the object a collected body
    /// carries — a streamed one is the same object with its index beside it.
    calls: Vec<serde_json::Value>,
    /// Whether the stream carries token counts, which is
    /// `stream_options.include_usage`.
    ///
    /// A question only a stream has. The collected body has reported usage since
    /// there was one — it is written after the last token, when the counts are
    /// known — where a stream's frames all go out before that, so a client
    /// reading counts off a stream needs a frame that exists for nothing else.
    reporting: bool,
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
            calls: Vec::new(),
            reporting: false,
        }
    }

    /// The same completion with token counts on its stream, which is what
    /// `stream_options.include_usage` asks for.
    pub fn reporting_usage(mut self, wanted: bool) -> Self {
        self.reporting = wanted;
        self
    }

    /// Whether the model asked for a tool, which is what decides between two of
    /// OpenAI's reasons for a turn the model ended.
    pub fn called(&self) -> bool {
        !self.calls.is_empty()
    }

    fn envelope(&self, object: &str, choices: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "object": object,
            "created": self.created,
            "model": self.model,
            "choices": choices,
        })
    }

    /// One chunk of a stream, around the one choice this returns.
    ///
    /// **`usage` is on every chunk once a client has asked for counts**, null
    /// until the one that carries them. That is the shape OpenAI sends and it is
    /// not decoration: a client reading the field off each chunk as it arrives
    /// finds a key rather than an absence, so the chunk that finally has counts
    /// is the same shape as the ones before it.
    fn chunk(&self, choice: serde_json::Value) -> serde_json::Value {
        let mut body = self.envelope("chat.completion.chunk", serde_json::json!([choice]));
        if self.reporting {
            body["usage"] = serde_json::Value::Null;
        }
        body
    }

    /// What the request spent, which is the same three counts however they
    /// leave — on the collected body, or on a stream's last chunk.
    fn usage(&self) -> serde_json::Value {
        serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        })
    }

    /// The frame that opens the stream: the role, and no text yet.
    ///
    /// OpenAI sends it, and clients that build a message out of the deltas need
    /// something to build it on before any text arrives.
    pub fn opening(&self) -> String {
        frame(&self.chunk(serde_json::json!({
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": null,
        })))
    }

    /// One token's worth of whatever it turned out to be, added to the reply and
    /// framed.
    ///
    /// The token is counted whether or not it showed anything. What the budget
    /// spends is decode steps, and a step that produced no visible text cost the
    /// same as one that did.
    pub fn push(&mut self, routed: Routed) -> Option<String> {
        self.completion_tokens += 1;
        self.add(routed)
    }

    /// What no token contributed: the bytes a detokenizer was still holding when
    /// the generation ended, and the call a budget cut short — both of which
    /// [`Channels`](crate::chat::Channels) hands back at the end.
    ///
    /// Added and framed like any other delta, and counted against nothing. No
    /// decode step produced it — it is the tail of ones already charged for —
    /// and `usage` a client reconciles against a budget has to say so.
    pub fn tail(&mut self, routed: Routed) -> Option<String> {
        self.add(routed)
    }

    /// `None` where there is nothing to show — a marker, a token that only
    /// completed part of a character, or text held back until it is known
    /// whether it names a tool. All three are ordinary and none is an event: a
    /// frame carrying an empty delta is one a client has to be careful about for
    /// no reason.
    fn add(&mut self, routed: Routed) -> Option<String> {
        match routed {
            Routed::Nothing => None,
            Routed::Text(channel, text) => self.delta(channel, &text),
            Routed::Call(call) => Some(self.invocation(call)),
        }
    }

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
        Some(frame(&self.chunk(serde_json::json!({
            "index": 0,
            "delta": {field: text},
            "finish_reason": null,
        }))))
    }

    /// A call, kept for the collected body and framed for the stream.
    ///
    /// `index` is on the streamed copy and not on the kept one, which is where
    /// OpenAI puts it: a client building a message out of deltas needs to know
    /// which call a delta belongs to, and one reading a whole message has the
    /// array's own order.
    fn invocation(&mut self, call: Call) -> String {
        let index = self.calls.len();
        let made = serde_json::json!({
            "id": format!("call_{}_{index}", self.id),
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        });
        self.calls.push(made.clone());

        let mut streamed = made;
        streamed["index"] = index.into();
        frame(&self.chunk(serde_json::json!({
            "index": 0,
            "delta": {"tool_calls": [streamed]},
            "finish_reason": null,
        })))
    }

    /// The frames that end the stream: why it ended, what it spent if the client
    /// asked, and then the terminator.
    ///
    /// **The counts come after the reason and not with it**, in a chunk whose
    /// `choices` is empty. That is OpenAI's shape and the reason is the one every
    /// streaming client is written around: a client assembles its message out of
    /// `choices[0].delta`, and counts hung off the chunk that carries the reason
    /// would be counts a client has to look for in a chunk it may already have
    /// stopped reading. An empty `choices` says this chunk is not part of the
    /// message.
    pub fn closing(&self, finish: Finish) -> String {
        let last = frame(&self.chunk(serde_json::json!({
            "index": 0,
            "delta": {},
            "finish_reason": finish.reason(),
        })));
        let counts = match self.reporting {
            false => String::new(),
            true => {
                let mut chunk = self.envelope("chat.completion.chunk", serde_json::json!([]));
                chunk["usage"] = self.usage();
                frame(&chunk)
            }
        };
        format!("{last}{counts}{DONE}")
    }

    /// The whole reply as one body, which is every delta that was pushed.
    ///
    /// `reasoning_content` is omitted rather than empty when the model opened no
    /// thinking channel, so that a reply with none looks like an ordinary
    /// OpenAI one. `content` is null rather than empty on a turn that was
    /// nothing but calls, which is the shape OpenAI sends one in and the shape
    /// [`crate::chat`] reads back — an empty string there is a message the
    /// template renders as an empty one rather than as no message at all.
    pub fn collected(&self, finish: Finish) -> String {
        let mut message = serde_json::json!({
            "role": "assistant",
            "content": self.content,
        });
        if self.content.is_empty() && self.called() {
            message["content"] = serde_json::Value::Null;
        }
        if !self.calls.is_empty() {
            message["tool_calls"] = self.calls.clone().into();
        }
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = self.reasoning.as_str().into();
        }

        let mut body = self.envelope(
            "chat.completion",
            serde_json::json!([{
                "index": 0,
                "message": message,
                "finish_reason": finish.reason(),
            }]),
        );
        body["usage"] = self.usage();
        body.to_string()
    }
}

/// What `GET /v1/models` answers: the one checkpoint this process loaded.
///
/// One, because a process holds exactly one, and it holds it whole: the banks are
/// wrapped where the checkpoint mapped them rather than copied, so what a second
/// would cost is another 137 GB of mapping and not another 0.35 GiB.
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
    use crate::chat::{invoked, text};
    use crate::wire::{calls, delta, frames, payload, payloads};

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
    fn streamed(completion: &mut Completion, reply: &[Routed], finish: Finish) -> String {
        let mut stream = completion.opening();
        for routed in reply {
            if let Some(chunk) = completion.push(routed.clone()) {
                stream.push_str(&chunk);
            }
        }
        stream.push_str(&completion.closing(finish));
        stream
    }

    fn reply() -> Vec<Routed> {
        vec![
            text(Channel::Thinking, "Weigh"),
            text(Channel::Thinking, " it up."),
            text(Channel::Content, "Caf"),
            text(Channel::Content, "é."),
            Routed::Nothing,
        ]
    }

    /// What a client parsing the stream sees, frame by frame: a role to build
    /// on, one delta per token that had text, a reason, and the terminator.
    #[test]
    fn a_stream_opens_with_a_role_and_ends_with_a_reason_and_a_terminator() {
        let stream = streamed(&mut completion(), &reply(), Finish::Stop);
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
        let stream = streamed(&mut completion(), &reply(), Finish::Length);
        let last = payloads(&stream).pop().expect("a last chunk");
        assert_eq!(last["choices"][0]["finish_reason"], "length");
    }

    /// The two channels reach a client in two fields, and neither carries the
    /// other's text.
    #[test]
    fn thinking_and_content_arrive_in_fields_of_their_own() {
        let stream = streamed(&mut completion(), &reply(), Finish::Stop);
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
        assert_eq!(completion.push(text(Channel::Content, "")), None);
        assert!(completion.push(text(Channel::Content, "Hi")).is_some());
    }

    /// The claim the whole type exists to make: the body a non-streaming request
    /// gets is the stream, added up. Asserted by adding the stream up rather
    /// than by restating the text, so that a change to either path has to keep
    /// them equal.
    #[test]
    fn the_collected_body_is_the_stream_added_up() {
        let mut completion = completion();
        let stream = streamed(&mut completion, &reply(), Finish::Stop);

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
        assert!(completion.push(text(Channel::Content, "Hello.")).is_some());
        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Stop)).expect("a json body");

        assert_eq!(
            body["choices"][0]["message"]["reasoning_content"],
            serde_json::Value::Null
        );
        assert_eq!(body["choices"][0]["message"]["content"], "Hello.");
    }

    /// The milestone, as a client reads it: the call arrives in a delta of its
    /// own and again in the collected body, the two carry the same id, and the
    /// reason says a tool was asked for rather than that the model had its say.
    #[test]
    fn a_call_reaches_a_client_as_tool_calls_and_not_as_text() {
        let mut completion = completion();
        let asked = [invoked("get_weather", "{\"city\":\"Paris\"}")];
        let stream = streamed(&mut completion, &asked, Finish::ToolCalls);

        let chunks = payloads(&stream);
        let streamed: Vec<serde_json::Value> = chunks.iter().flat_map(calls).collect();
        assert_eq!(streamed.len(), 1, "{chunks:?}");
        assert_eq!(streamed[0]["index"], 0);
        assert_eq!(streamed[0]["type"], "function");
        assert_eq!(streamed[0]["function"]["name"], "get_weather");
        assert_eq!(streamed[0]["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert!(
            chunks.iter().all(|chunk| delta(chunk).is_none()),
            "{chunks:?}"
        );

        let last = chunks.last().expect("a last chunk");
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");

        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::ToolCalls)).expect("a json body");
        let message = &body["choices"][0]["message"];
        assert_eq!(message["tool_calls"][0]["id"], streamed[0]["id"]);
        assert_eq!(
            message["tool_calls"][0]["function"],
            streamed[0]["function"]
        );
        // OpenAI's own shape for a turn that was nothing but calls, and the one
        // `crate::chat` reads back as a message with no content of its own.
        assert_eq!(message["content"], serde_json::Value::Null);
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    /// Two calls out of one turn, which is what an agent asking for two files
    /// at once produces. A client builds its array by `index`, and answers each
    /// call by its own id — so two calls sharing either would be one call.
    #[test]
    fn two_calls_are_told_apart_by_their_index_and_by_their_id() {
        let mut completion = completion();
        let asked = [invoked("a", "{}"), invoked("b", "{\"x\":1}")];
        let stream = streamed(&mut completion, &asked, Finish::ToolCalls);

        let made: Vec<serde_json::Value> = payloads(&stream).iter().flat_map(calls).collect();
        assert_eq!(made.len(), 2, "{made:?}");
        assert_eq!(made[0]["index"], 0);
        assert_eq!(made[1]["index"], 1);
        assert_ne!(made[0]["id"], made[1]["id"]);
        assert_eq!(made[1]["function"]["name"], "b");
    }

    /// A turn that said something *and* called a tool keeps both. The text is
    /// the answer and the call is the ask, and a client that got only one of
    /// them is missing the other.
    #[test]
    fn a_turn_that_answered_and_called_keeps_both() {
        let mut completion = completion();
        let asked = [text(Channel::Content, "Looking."), invoked("a", "{}")];
        let _ = streamed(&mut completion, &asked, Finish::ToolCalls);

        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::ToolCalls)).expect("a json body");
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], "Looking.");
        assert_eq!(message["tool_calls"].as_array().expect("calls").len(), 1);
    }

    /// A reply nobody asked a tool for carries no `tool_calls` key at all,
    /// rather than an empty array a client has to check the length of.
    #[test]
    fn a_reply_without_a_call_carries_no_tool_calls_field() {
        let mut completion = completion();
        let _ = streamed(&mut completion, &reply(), Finish::Stop);
        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Stop)).expect("a json body");

        assert_eq!(
            body["choices"][0]["message"]["tool_calls"],
            serde_json::Value::Null
        );
        assert_eq!(body["choices"][0]["message"]["content"], "Café.");
    }

    /// Every token is counted, including the ones that showed nothing — a
    /// marker, or half a character. Each cost a decode step, and usage that
    /// counted only the visible ones would understate what the request spent.
    #[test]
    fn usage_counts_every_token_the_budget_was_charged_for() {
        let mut completion = completion();
        for routed in reply() {
            let _ = completion.push(routed);
        }
        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Stop)).expect("a json body");

        assert_eq!(body["usage"]["prompt_tokens"], PROMPT_TOKENS);
        assert_eq!(body["usage"]["completion_tokens"], reply().len());
        assert_eq!(body["usage"]["total_tokens"], PROMPT_TOKENS + reply().len());
    }

    /// **The chunk `include_usage` exists for**: after the reason, before the
    /// terminator, with an empty `choices` so a client assembling a message out
    /// of `choices[0].delta` reads it as no part of the message.
    #[test]
    fn a_stream_that_asked_for_usage_ends_with_a_chunk_carrying_it() {
        let mut completion = completion().reporting_usage(true);
        let stream = streamed(&mut completion, &reply(), Finish::Stop);

        let chunks = payloads(&stream);
        let last = chunks.last().expect("a last chunk");
        assert_eq!(
            last["choices"].as_array().expect("an array").len(),
            0,
            "{last}"
        );
        assert_eq!(last["object"], "chat.completion.chunk");
        assert_eq!(last["usage"]["prompt_tokens"], PROMPT_TOKENS);
        assert_eq!(last["usage"]["completion_tokens"], reply().len());
        assert_eq!(last["usage"]["total_tokens"], PROMPT_TOKENS + reply().len());

        // The reason is the chunk before it, and it is still a chunk of the
        // message: a client that stopped reading at the reason would have the
        // whole reply and none of the counts.
        let reason = &chunks[chunks.len() - 2];
        assert_eq!(reason["choices"][0]["finish_reason"], "stop");
    }

    /// **The counts on the stream are the counts on the body.** Two ways out of
    /// one completion, and a client that reconciles them against a budget has to
    /// find them equal — which they are by construction here, and this is what
    /// says the construction survived.
    #[test]
    fn a_stream_and_a_collected_body_report_the_same_usage() {
        let mut completion = completion().reporting_usage(true);
        let stream = streamed(&mut completion, &reply(), Finish::Stop);
        let last = payloads(&stream).pop().expect("a last chunk");

        let body: serde_json::Value =
            serde_json::from_str(&completion.collected(Finish::Stop)).expect("a json body");
        assert_eq!(last["usage"], body["usage"]);
    }

    /// **A client gets counts only where it asked for them.** The chunk is
    /// OpenAI's opt-in, and a stream that grew one unasked is a stream whose
    /// last frame before the terminator is not the one every existing client
    /// expects.
    #[test]
    fn a_stream_that_did_not_ask_for_usage_carries_none() {
        let stream = streamed(&mut completion(), &reply(), Finish::Stop);
        let chunks = payloads(&stream);

        assert!(
            chunks.iter().all(|chunk| chunk.get("usage").is_none()),
            "{chunks:?}"
        );
        let last = chunks.last().expect("a last chunk");
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
    }

    /// Every chunk carries the key once it was asked for, null until the one
    /// that has the counts. A client reading `usage` off each chunk as it
    /// arrives finds a field rather than an absence, which is the shape OpenAI
    /// sends and the reason the last chunk is not the only one shaped
    /// differently.
    #[test]
    fn every_chunk_carries_the_usage_key_once_a_client_has_asked_for_it() {
        let mut completion = completion().reporting_usage(true);
        let stream = streamed(&mut completion, &reply(), Finish::Stop);

        let mut chunks = payloads(&stream);
        let counted = chunks.pop().expect("a last chunk");
        assert!(!chunks.is_empty(), "nothing before the counts");
        for chunk in &chunks {
            assert_eq!(chunk["usage"], serde_json::Value::Null, "{chunk}");
            assert!(chunk.get("usage").is_some(), "{chunk} has no usage key");
        }
        assert_ne!(counted["usage"], serde_json::Value::Null);
    }

    /// What no token produced is not counted as one. The bytes left over at the
    /// end of a reply are the tail of tokens already charged for, so counting
    /// them again reports a completion one token longer than the budget allowed —
    /// which is what a client reconciling `usage` against `max_tokens` reads as
    /// a server that overran.
    #[test]
    fn the_text_left_over_at_the_end_is_not_a_token() {
        let mut completion = completion();
        let _ = completion.push(text(Channel::Content, "The"));
        assert!(
            completion
                .tail(text(Channel::Content, "\u{fffd}"))
                .is_some()
        );

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
        let split = "Café, 日本語, 🙂.";
        let mut stream = Utf8Stream::new();
        let mut completion = completion();

        let mut frames = String::new();
        for byte in split.as_bytes() {
            if let Some(chunk) = completion.push(text(Channel::Content, &stream.push(&[*byte]))) {
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
        assert_eq!(arrived, split);
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
            .push(text(Channel::Content, "one\n\ntwo"))
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

    /// A second choice changes the shape of the answer rather than its wording,
    /// which is why it is refused where `temperature` is ignored.
    #[test]
    fn a_second_choice_is_refused() {
        assert_eq!(refused(serde_json::json!({"n": 2})), RequestError::Choices);
        assert!(request(asking(serde_json::json!({"n": 1}))).is_ok());
    }

    const WEATHER: fn() -> serde_json::Value =
        || serde_json::json!([{"type": "function", "function": {"name": "get_weather"}}]);

    /// The declaration reaches the prompt, which is the whole of what accepting
    /// `tools` means here.
    #[test]
    fn the_tools_a_request_names_are_the_tools_the_prompt_declares() {
        let asked = request(asking(serde_json::json!({"tools": WEATHER()}))).expect("parses");
        assert_eq!(asked.declared(), WEATHER().as_array().expect("a list"));
        assert!(
            request(asking(serde_json::json!({})))
                .expect("parses")
                .declared()
                .is_empty()
        );
    }

    /// `none` is the one `tool_choice` with something to do rather than
    /// something to refuse, and what it does is leave the specs out: a model
    /// that was never told the tools has none to reach for.
    #[test]
    fn tool_choice_none_declares_no_tools_at_all() {
        let extra = serde_json::json!({"tools": WEATHER(), "tool_choice": "none"});
        let asked = request(asking(extra)).expect("parses");
        assert!(asked.declared().is_empty());

        let extra = serde_json::json!({"tools": WEATHER(), "tool_choice": "auto"});
        let auto = request(asking(extra)).expect("parses");
        assert_eq!(auto.declared().len(), 1);
    }

    /// The two `tool_choice` values this does not implement. Both are a
    /// constraint on what the model may produce and nothing in a prompt is one,
    /// so both are named back — a client that asked for a call and got a
    /// sentence cannot tell that from a model that would not call one.
    #[test]
    fn a_tool_choice_this_cannot_honour_is_refused_rather_than_ignored() {
        for asked in [
            serde_json::json!("required"),
            serde_json::json!({"type": "function", "function": {"name": "get_weather"}}),
            serde_json::json!(7),
        ] {
            let extra = serde_json::json!({"tools": WEATHER(), "tool_choice": asked});
            assert_eq!(
                refused(extra),
                RequestError::ToolChoice(asked.to_string()),
                "{asked}"
            );
        }
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

    /// OpenAI spells `stop` both ways and means the same thing by them, so both
    /// reach the matching as the same list.
    #[test]
    fn stop_is_read_from_a_string_and_from_a_list_alike() {
        let stopping = |extra| request(asking(extra)).expect("parses").stopping();

        assert_eq!(
            stopping(serde_json::json!({"stop": "\nUser:"})),
            ["\nUser:"]
        );
        assert_eq!(
            stopping(serde_json::json!({"stop": ["\nUser:", "###"]})),
            ["\nUser:", "###"]
        );
        assert!(stopping(serde_json::json!({})).is_empty());
        assert!(stopping(serde_json::json!({"stop": null})).is_empty());
    }

    /// **An empty sequence matches before the first token of every reply**, so a
    /// request carrying one would be answered with nothing at all — which a
    /// client reads as a model that had nothing to say rather than as a `stop`
    /// it should not have sent.
    #[test]
    fn an_empty_stop_sequence_is_refused() {
        assert_eq!(
            refused(serde_json::json!({"stop": ""})),
            RequestError::EmptyStop
        );
        assert_eq!(
            refused(serde_json::json!({"stop": ["fine", ""]})),
            RequestError::EmptyStop
        );
    }

    /// A `stop` this cannot read is named back rather than ignored, for the
    /// reason a `tool_choice` this cannot honour is: a client whose sequence was
    /// quietly dropped gets a reply running past a stop it asked for, which is
    /// the failure `stop` exists to prevent.
    #[test]
    fn a_stop_that_is_not_sequences_is_refused_rather_than_ignored() {
        for asked in [
            serde_json::json!(7),
            serde_json::json!(["fine", 7]),
            serde_json::json!({"where": "here"}),
            // OpenAI's own limit, and one past it.
            serde_json::json!(["a", "b", "c", "d", "e"]),
        ] {
            assert_eq!(
                refused(serde_json::json!({"stop": asked})),
                RequestError::Stop(asked.to_string()),
                "{asked}"
            );
        }
        assert!(request(asking(serde_json::json!({"stop": ["a", "b", "c", "d"]}))).is_ok());
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
        let refused = crate::chat::prompt(&request.messages, &[]).expect_err("the role is refused");

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
