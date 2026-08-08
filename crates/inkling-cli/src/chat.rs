//! The turn structure, written out from a messages array.
//!
//! `generate` sends its prompt to the tokenizer as it stands, and records what
//! that costs: nothing in an untemplated prompt puts the model in a turn it could
//! end, so `<|content_model_end_sampling|>` never arrives and the budget is what
//! stops it. That is the right answer for a debugging tool and the wrong one for
//! a server — a client that asks a question and gets its own sentence continued
//! to `max_tokens` is looking at a broken model.
//!
//! So this is where the turn structure is applied, and it is the reason the
//! server is more than a socket in front of [`crate::generate`].
//!
//! # Hard-coded, not interpreted
//!
//! The checkpoint ships a `chat_template.jinja` and nothing here reads it. A
//! Jinja engine is a large dependency to carry for a template whose whole output
//! is a dozen literal markers in a fixed order, and one that ran the real
//! template would accept requests — images, audio — that the engine underneath
//! cannot serve anyway.
//!
//! What that costs is that a checkpoint whose template differs is templated
//! wrongly and silently, so the divergences are worth stating rather than
//! discovering. Against `models/Inkling-Small-mxfp4/chat_template.jinja`, this
//! reproduces the template exactly for the messages it accepts. It does not
//! implement:
//!
//! - **Content parts that are not text.** A `content` that is a list of text
//!   parts is written out the way the template writes one — a message *apiece*
//!   rather than one message of joined text — because that is how most OpenAI
//!   clients send a user turn and refusing it made this server unusable from
//!   them. `input_image` and `input_audio` parts are still refused, where the
//!   template renders a placeholder message for each: the engine is text-only,
//!   and a prompt that says an image was attached to a model that was handed
//!   none is a worse answer than a 400. An empty list is the template's own
//!   nothing and is treated here exactly as an absent `content` is — refused
//!   on a turn that carries nothing else, and rendered as the calls alone on
//!   an assistant turn that made some.
//! - **Content parts on a `tool` message.** The template emits a tool result's
//!   `content` only where it is a string and renders an empty result for a
//!   list, which is a silent mangle rather than a format. Refused here.
//! - **`reasoning_effort`.** The template maps six names onto numbers and accepts
//!   a float, defaulting to 0.9 when the caller names none. Only the default is
//!   emitted here, which is the string `generate`'s docs measured the template
//!   producing.
//! - **A `content` that is absent or null on a message that carries nothing
//!   else.** The template emits nothing at all for such a message, not even its
//!   role marker. Refused here, because a message that silently contributes
//!   nothing to the prompt is a request the client will not recognise the answer
//!   to. An assistant message whose `content` is null *and* which carries
//!   `tool_calls` is the shape a client replays a call in, and is accepted.
//! - **A number the two JSON writers spell differently.** A spec is serialised
//!   the way the template's `tojson` serialises one — keys sorted at every
//!   depth, `(",", ":")` separators, non-ASCII left as it stands — and the two
//!   agree on every value a JSON Schema has except an exponent: Python writes
//!   `1e+30` where `serde_json` writes `1e30`.
//!
//! # Tools
//!
//! The template puts the specs in a leading system message of their own, ahead
//! of even the thinking-effort one, as `tool_declare<|content_xml|>` and the
//! specs as JSON. **The serialisation is load-bearing**: the model was trained
//! on one spelling of a given spec, so the keys are sorted and the separators
//! carry no spaces, and a spec whose keys arrive in another order is the same
//! prompt.
//!
//! A call is a model message of its own — `<|message_model|>` and the function's
//! name, then `<|content_invoke_tool_json|>` and `{"name":…,"args":…}` — and a
//! result is a `<|message_tool|>` message named for the tool it came from. The
//! name of a result is the message's own `name` where it has one, and otherwise
//! is looked up by `tool_call_id` in the calls the conversation already carries,
//! which is what the template does and what makes an OpenAI client's replay
//! land the same way twice.
//!
//! **`arguments` arrives as a JSON string and leaves as an object.** Every
//! OpenAI client sends the arguments of a call it is replaying as a string, and
//! the template refuses one — "canonicalize upstream", it says, because the
//! string a client kept is not the sorted, space-free spelling the model was
//! trained on. So a string is parsed here and written back out canonically, and
//! `chat_template_cases.json` records both forms against the one prompt.
//!
//! # Where the thinking-effort message goes
//!
//! Not simply first. The template emits it before the first message whose role is
//! *not* `system`, so a caller's own system prompt precedes it, and emits it at
//! the end of a conversation that never had one. That ordering is reproduced
//! rather than approximated: it is a system message either way, and the model was
//! trained on it in that position.

use serde::Deserialize;
use serde_json::Value;

/// The markers, as the vocabulary spells them.
const SYSTEM: &str = "<|message_system|>";
const USER: &str = "<|message_user|>";
const MODEL: &str = "<|message_model|>";
const TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_XML: &str = "<|content_xml|>";
const CONTENT_INVOKE: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";

/// What the template names the system message the specs are declared in. Not a
/// marker — it is ordinary text between the role marker and the channel one.
const TOOL_DECLARE: &str = "tool_declare";

/// The template's default `reasoning_effort`, written the way Jinja writes the
/// float 0.9.
const THINKING_EFFORT: &str = "Thinking effort level: 0.9";

/// One message of a conversation, as a request carries it.
///
/// `role` is a string rather than an enum so that a role this does not map can
/// be named back to the caller. Deserialising it into a closed set would make
/// every unknown role the same parse error.
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    /// Whatever the client sent, which is only a string here — see the module
    /// docs on content parts.
    #[serde(default)]
    pub content: Option<Value>,
    /// An assistant turn's thinking, which is the field the checkpoint's own
    /// template reads it from and the field [`crate::openai`] streams it back
    /// out under. The two together are what lets a client feed a reply it was
    /// given back into the next request unchanged.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// The calls an assistant turn made, which a client replays on the turn
    /// after them.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Which call a `tool` message answers, and the only thing most clients
    /// send to say which tool it came from.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// The tool a `tool` message came from, where the client names it outright.
    #[serde(default)]
    pub name: Option<String>,
}

/// One call in an assistant turn, as OpenAI carries one.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    /// What the result of this call will name it by. Optional because the
    /// template does not need it — a result may name its tool directly — and
    /// because nothing here mints one for a call a client sent.
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Called,
}

/// The function half of a call: which one, and with what.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Called {
    #[serde(default)]
    pub name: Option<Value>,
    /// The JSON string an OpenAI client sends, or the object the template
    /// takes. See the module docs on canonicalising the first into the second.
    #[serde(default)]
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChatError {
    #[error("a conversation needs at least one message")]
    NoMessages,

    #[error("{0:?} is not a role; this server takes system, user, assistant and tool")]
    UnknownRole(String),

    #[error("the content of the {0} message is not a string; this server takes no content parts")]
    ContentNotText(String),

    #[error(
        "the {0} message carries a content part that is not text; this engine serves text alone"
    )]
    ContentPartNotText(String),

    #[error("a tool call needs a function name, which must be a string")]
    CallWithoutAName,

    #[error("the arguments of the {0} call are not an object")]
    ArgumentsNotAnObject(String),

    #[error("the arguments of the {name} call are a string that is not JSON: {why}")]
    ArgumentsNotJson { name: String, why: String },

    #[error("a tool specification needs a name")]
    SpecWithoutAName,
}

/// A role the turn structure has a marker for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    System,
    User,
    /// `assistant` on the wire. The model's own turns are opened by
    /// `<|message_model|>`, and the two names are worth keeping apart: what a
    /// client calls the role is not what the vocabulary calls it.
    Model,
    Tool,
}

impl Role {
    fn marker(self) -> &'static str {
        match self {
            Role::System => SYSTEM,
            Role::User => USER,
            Role::Model => MODEL,
            Role::Tool => TOOL,
        }
    }

    fn parse(role: &str) -> Result<Self, ChatError> {
        match role {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Model),
            "tool" => Ok(Role::Tool),
            _ => Err(ChatError::UnknownRole(role.to_string())),
        }
    }
}

impl Message {
    /// The message's text, or `None` where it carries no `content` at all.
    ///
    /// A `content` that is present and is not a string is refused rather than
    /// flattened. This is the rule the template applies to a `tool` message
    /// alone; every other role takes [`Message::texts`].
    fn text(&self) -> Result<Option<&str>, ChatError> {
        match &self.content {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text)),
            Some(_) => Err(ChatError::ContentNotText(self.role.clone())),
        }
    }

    /// The messages this one's `content` writes out, which is one apiece for
    /// the parts of a list and one for a plain string. `None` where the
    /// message carries no `content` at all.
    ///
    /// **A list is not joined**, because the template does not join one: each
    /// part gets its own role marker and its own `<|end_message|>`, and a
    /// conversation whose parts arrived as one message is a prompt the model
    /// was not trained on. An empty list yields `None` rather than an empty
    /// list of messages, so that it is refused where a missing `content` is.
    fn texts(&self) -> Result<Option<Vec<&str>>, ChatError> {
        match &self.content {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(vec![text])),
            Some(Value::Array(parts)) if parts.is_empty() => Ok(None),
            Some(Value::Array(parts)) => parts
                .iter()
                .map(|part| self.part(part))
                .collect::<Result<Vec<&str>, ChatError>>()
                .map(Some),
            Some(_) => Err(ChatError::ContentNotText(self.role.clone())),
        }
    }

    /// One content part as the text it contributes.
    ///
    /// The template reads a bare string as text, and reads an object as text
    /// where it names no `type` at all or names one of the two the format
    /// spells text with. A `type` that is present and is anything else — an
    /// image, an audio clip, a word this vocabulary has no channel for, or a
    /// value that is not even a string — is refused. `text` missing from a
    /// text part is the empty string rather than a refusal, which is the
    /// template's own default and not a guess.
    fn part<'a>(&self, part: &'a Value) -> Result<&'a str, ChatError> {
        let refused = || Err(ChatError::ContentPartNotText(self.role.clone()));
        match part {
            Value::String(text) => Ok(text),
            Value::Object(fields) => match fields.get("type") {
                None => Ok(text_of(fields)),
                Some(Value::String(named)) if TEXT_PARTS.contains(&named.as_str()) => {
                    Ok(text_of(fields))
                }
                Some(_) => refused(),
            },
            _ => refused(),
        }
    }

    fn calls(&self) -> &[ToolCall] {
        self.tool_calls.as_deref().unwrap_or_default()
    }
}

/// The two words the format spells a text part's `type` with.
const TEXT_PARTS: [&str; 2] = ["text", "input_text"];

/// A text part's own text, which the template defaults to the empty string
/// where the key is missing or is not a string.
fn text_of(part: &serde_json::Map<String, Value>) -> &str {
    part.get("text").and_then(Value::as_str).unwrap_or_default()
}

/// Jinja's own notion of truth, which is what every `x if x is defined and x
/// else <default>` in the template falls back on: null, false, zero, and every
/// empty string, list and object are the default's.
fn truthy(value: Option<&Value>) -> Option<&Value> {
    value.filter(|value| match value {
        Value::Null => false,
        Value::Bool(is) => *is,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    })
}

/// One message written out, which is the same five parts for every role and
/// every channel: whose turn it is, what it is named, which channel it is in,
/// the text, and the end of the message.
///
/// Only a tool's result and a call to one carry the name; everything else
/// passes an empty one.
fn named(marker: &str, name: &str, channel: &str, content: &str, out: &mut String) {
    out.push_str(marker);
    out.push_str(name);
    out.push_str(channel);
    out.push_str(content);
    out.push_str(END_MESSAGE);
}

fn message(marker: &str, channel: &str, content: &str, out: &mut String) {
    named(marker, "", channel, content, out);
}

/// The conversation as a prompt, ending in the marker that opens the model's own
/// turn.
///
/// That last marker is the whole point. It is what a template calls
/// `add_generation_prompt`, and it is what puts the model in a turn it can end —
/// without it the stopping rule is correct and never fires.
///
/// `tools` are the specs to declare, which the template puts ahead of the
/// conversation and ahead of the thinking-effort message.
pub fn prompt(messages: &[Message], tools: &[Value]) -> Result<String, ChatError> {
    if messages.is_empty() {
        return Err(ChatError::NoMessages);
    }

    let mut out = String::new();
    declare(tools, &mut out)?;

    let mut effort_emitted = false;
    for sent in messages {
        let role = Role::parse(&sent.role)?;
        if !effort_emitted && role != Role::System {
            message(SYSTEM, CONTENT_TEXT, THINKING_EFFORT, &mut out);
            effort_emitted = true;
        }
        match role {
            Role::Tool => result(sent, messages, &mut out)?,
            _ => turn(sent, role, &mut out)?,
        }
    }
    if !effort_emitted {
        message(SYSTEM, CONTENT_TEXT, THINKING_EFFORT, &mut out);
    }

    out.push_str(MODEL);
    Ok(out)
}

/// One turn of the conversation that is not a tool's result: the thinking of a
/// model turn, then the text, then the calls it made, then the marker that says
/// the model ended it.
fn turn(sent: &Message, role: Role, out: &mut String) -> Result<(), ChatError> {
    let content = sent.texts()?;
    let calls = sent.calls();
    // A message with no content is only a message at all when it carries calls,
    // and only an assistant turn carries those.
    if content.is_none() && !(role == Role::Model && !calls.is_empty()) {
        return Err(ChatError::ContentNotText(sent.role.clone()));
    }

    if let (Role::Model, Some(thinking)) = (role, &sent.reasoning_content) {
        message(MODEL, CONTENT_THINKING, thinking, out);
    }
    for text in content.into_iter().flatten() {
        message(role.marker(), CONTENT_TEXT, text, out);
    }
    if role == Role::Model {
        for call in calls {
            invocation(call, out)?;
        }
        out.push_str(END_SAMPLING);
    }
    Ok(())
}

/// A call the model made, replayed: the function's name beside the marker that
/// opens the turn, and again inside the envelope the marker opens.
///
/// The name has to be a string and does not have to be a *filled* one, which is
/// the template's own check and not Jinja truthiness. An empty name is a call
/// the model never named, which the reading side produces whenever neither the
/// text before the marker nor the envelope carried one — so refusing it here
/// would 400 a conversation on a turn this server itself wrote.
fn invocation(call: &ToolCall, out: &mut String) -> Result<(), ChatError> {
    let name = call
        .function
        .name
        .as_ref()
        .and_then(Value::as_str)
        .ok_or(ChatError::CallWithoutAName)?;
    let arguments = arguments(name, call.function.arguments.as_ref())?;
    let body = format!(
        "{{\"name\":{},\"args\":{arguments}}}",
        Value::String(name.to_string())
    );
    named(MODEL, name, CONTENT_INVOKE, &body, out);
    Ok(())
}

/// A call's arguments as the template writes them: an object, with its keys
/// sorted and no spaces between them.
///
/// The string an OpenAI client sends is parsed and written back out that way —
/// see the module docs. What a client kept is its own spelling of the
/// arguments, and the model was trained on one spelling.
fn arguments(name: &str, value: Option<&Value>) -> Result<String, ChatError> {
    let Some(value) = truthy(value) else {
        return Ok("{}".to_string());
    };
    let parsed;
    let object = match value {
        Value::String(text) => {
            parsed = serde_json::from_str(text).map_err(|err| ChatError::ArgumentsNotJson {
                name: name.to_string(),
                why: err.to_string(),
            })?;
            &parsed
        }
        object => object,
    };
    match object.is_object() {
        true => Ok(object.to_string()),
        false => Err(ChatError::ArgumentsNotAnObject(name.to_string())),
    }
}

/// A tool's result, named for the tool it came from.
///
/// The name is the message's own where it has one and is otherwise looked up by
/// `tool_call_id` among the calls the conversation carries, which is the walk
/// the template does. An id that names no call leaves the message unnamed —
/// that is what the template does with one, and a prompt with an unnamed tool
/// message is a prompt the model was trained to read.
fn result(sent: &Message, messages: &[Message], out: &mut String) -> Result<(), ChatError> {
    let content = sent.text()?.unwrap_or_default();
    named(TOOL, tool_name(sent, messages), CONTENT_TEXT, content, out);
    Ok(())
}

fn tool_name<'a>(sent: &'a Message, messages: &'a [Message]) -> &'a str {
    if let Some(name) = sent.name.as_deref().filter(|name| !name.is_empty()) {
        return name;
    }
    let Some(id) = sent.tool_call_id.as_deref().filter(|id| !id.is_empty()) else {
        return "";
    };
    messages
        .iter()
        .filter(|message| message.role == "assistant")
        .flat_map(Message::calls)
        .filter(|call| call.id.as_deref() == Some(id))
        .filter_map(|call| call.function.name.as_ref().and_then(Value::as_str))
        .next_back()
        .unwrap_or_default()
}

/// The specs, in the leading system message the template declares them in.
///
/// Nothing at all where there are none, which is what the template's `if tools`
/// does with an empty list: a request that named no tool is the request it was
/// before this existed, prompt for prompt.
fn declare(tools: &[Value], out: &mut String) -> Result<(), ChatError> {
    if tools.is_empty() {
        return Ok(());
    }
    let specs: Vec<Value> = tools.iter().map(spec).collect::<Result<_, _>>()?;
    let specs = Value::Array(specs).to_string();
    named(SYSTEM, TOOL_DECLARE, CONTENT_XML, &specs, out);
    Ok(())
}

/// One spec, as the four keys the template builds out of whatever the client
/// sent: a tool with no `function` of its own *is* the function, and a spec
/// that names neither a description nor parameters gets an empty one of each
/// rather than losing the key.
fn spec(tool: &Value) -> Result<Value, ChatError> {
    let function = tool.get("function").unwrap_or(tool);
    let name = function
        .get("name")
        .ok_or(ChatError::SpecWithoutAName)?
        .clone();
    Ok(serde_json::json!({
        "description": truthy(function.get("description")).cloned().unwrap_or_else(|| Value::String(String::new())),
        "name": name,
        "parameters": truthy(function.get("parameters")).cloned().unwrap_or_else(|| serde_json::json!({})),
        "type": truthy(tool.get("type")).cloned().unwrap_or_else(|| Value::String("function".to_string())),
    }))
}

/// Which of the model's two channels a token's text belongs to.
///
/// The model opens one and then the other inside a single turn: the `turn` case
/// of `tokenizer_cases.json` is `<|message_model|><|content_thinking|>Weigh it
/// up.<|content_text|>Café, 日本語, 🙂.<|end_message|>`, one message carrying
/// both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Thinking,
    Content,
}

/// What the model is spelling out now, which is what decides where the text
/// arriving belongs.
///
/// Two of the three are not channels at all. The template puts a call's
/// function name between `<|message_model|>` and the marker that opens the
/// call, so text there is a name and not a message; and what the marker opens
/// is an envelope — `{"name":…,"args":…}` — that a client has no use for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// A channel a client renders.
    Channel(Channel),
    /// The name of a tool the model is about to call.
    Name,
    /// The JSON of the call itself.
    Invocation,
}

/// The markers a vocabulary is asked for, and what each puts the reader into.
///
/// `<|end_message|>` and the end-of-sequence id are here for what they are *not*:
/// text. The tokenizer renders special tokens literally rather than swallowing
/// them, and both of these reach a sink like any other token — so a server that
/// did not name them would put `<|end_message|><|content_model_end_sampling|>` on
/// the end of every reply it sent. `<|message_model|>` is here for the same
/// reason and one more: the model emits it in the middle of a turn that calls
/// two tools, and what follows it is a name.
///
/// An ended message leaves the reader in [`Channel::Content`], which is where
/// text nobody has named a channel for goes.
pub const MARKERS: [(&str, Reading); 6] = [
    (CONTENT_THINKING, Reading::Channel(Channel::Thinking)),
    (CONTENT_TEXT, Reading::Channel(Channel::Content)),
    (CONTENT_INVOKE, Reading::Invocation),
    (MODEL, Reading::Name),
    (END_MESSAGE, Reading::Channel(Channel::Content)),
    (END_SAMPLING, Reading::Channel(Channel::Content)),
];

/// A call the model made, as a client is handed one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub name: String,
    /// The arguments as the JSON a client hands to the tool, which is the
    /// `args` of the model's envelope — not the envelope.
    pub arguments: String,
}

/// What a token turned out to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Routed {
    /// Text a client renders, in the channel it belongs to.
    Text(Channel, String),
    /// A call, whole. See [`Channels`] on why it arrives whole rather than in
    /// fragments.
    Call(Call),
    /// Nothing to show: a marker, half a character, or text held back until it
    /// is known whether it names a tool.
    Nothing,
}

/// What the model is spelling out, kept across tokens.
///
/// It starts on [`Reading::Name`], because that is where the prompt leaves it:
/// the prompt ends with `<|message_model|>` and the template puts a call's name
/// straight after one. A reply that is not a call opens a channel marker as its
/// first token, so the held text is empty and nothing is delayed by starting
/// there.
///
/// # A call arrives whole
///
/// OpenAI's streaming shape allows a call to arrive in fragments, and this does
/// not send it that way. Two reasons, and the first is enough: the envelope the
/// template writes is `{"name":…,"args":…}`, and a client's `arguments` is the
/// `args` alone — which cannot be cut out of a stream of fragments without
/// having the whole of it. The second is that a half-written `arguments` is a
/// half-written JSON object, which no client can do anything with anyway.
///
/// What that costs is a wait for the call's own tokens, which is the same wait
/// the client had before it could parse them.
#[derive(Debug)]
pub struct Channels {
    markers: Vec<(u32, String, Reading)>,
    reading: Reading,
    /// The text of whatever is being held back: a name, or a call's envelope.
    held: String,
    /// The name of the call being read, once the marker that opens it has said
    /// that is what the held text was.
    name: String,
}

impl Channels {
    pub fn new(markers: impl IntoIterator<Item = (u32, String, Reading)>) -> Self {
        Self {
            markers: markers.into_iter().collect(),
            reading: Reading::Name,
            held: String::new(),
            name: String::new(),
        }
    }

    /// Where `text` — the text token `id` contributed — goes, and what of it a
    /// client should see.
    ///
    /// A marker's own literal is not part of the message and is cut off. What
    /// can precede it is not nothing, though: a detokenizer holds back the bytes
    /// of a character it has not finished, and a special token is what releases
    /// them, so the text of a marker token can be a replacement character and
    /// then the marker. Those bytes belong to whatever was being read when they
    /// were held back, which is why the switch happens after they are handed
    /// back and not before.
    pub fn route(&mut self, id: u32, text: &str) -> Routed {
        let marker = self
            .markers
            .iter()
            .find(|(marker, ..)| *marker == id)
            .map(|(_, literal, opens)| (literal.clone(), *opens));
        let Some((literal, opens)) = marker else {
            return self.arriving(text);
        };

        let released = text.strip_suffix(literal.as_str()).unwrap_or(text);
        let carried = self.arriving(released);
        let left = self.leaving(opens);
        self.reading = opens;
        match carried {
            Routed::Nothing => left,
            carried => carried,
        }
    }

    /// The last of it: the bytes a detokenizer was still holding when the
    /// generation ended, and whatever a message the model never closed had held
    /// back.
    ///
    /// A budget that ran out mid-call leaves an envelope that will not parse,
    /// and the call goes out with that text as its arguments rather than not at
    /// all — a client that reconciles `finish_reason: "length"` against a call
    /// it cannot parse is being told what happened.
    pub fn finish(&mut self, held: &str) -> Routed {
        let carried = self.arriving(held);
        let left = self.leaving(Reading::Channel(Channel::Content));
        self.reading = Reading::Channel(Channel::Content);
        match carried {
            Routed::Nothing => left,
            carried => carried,
        }
    }

    /// Text with no marker on it, which is shown where a channel is open and
    /// held back where one is not.
    fn arriving(&mut self, text: &str) -> Routed {
        match self.reading {
            Reading::Channel(channel) if !text.is_empty() => {
                Routed::Text(channel, text.to_string())
            }
            Reading::Channel(_) => Routed::Nothing,
            _ => {
                self.held.push_str(text);
                Routed::Nothing
            }
        }
    }

    /// What the state being left owes, now that `opens` says what came next.
    ///
    /// A name is only a name if the marker that opens a call follows it;
    /// anything else and it was text the model put where the template puts a
    /// name, which is shown rather than dropped.
    fn leaving(&mut self, opens: Reading) -> Routed {
        match (self.reading, opens) {
            (Reading::Name, Reading::Invocation) => {
                self.name = std::mem::take(&mut self.held);
                Routed::Nothing
            }
            (Reading::Name, _) => {
                self.name.clear();
                match std::mem::take(&mut self.held) {
                    text if text.is_empty() => Routed::Nothing,
                    text => Routed::Text(Channel::Content, text),
                }
            }
            (Reading::Invocation, _) => Routed::Call(call(
                std::mem::take(&mut self.name),
                &std::mem::take(&mut self.held),
            )),
            (Reading::Channel(_), _) => Routed::Nothing,
        }
    }
}

/// A token's worth of text, and a call the model made, as the tests either side
/// of this module spell them.
///
/// Here rather than in each of the three, because three copies of a two-line
/// constructor are three places a change to [`Routed`] has to land.
#[cfg(test)]
pub(crate) fn text(channel: Channel, text: &str) -> Routed {
    Routed::Text(channel, text.to_string())
}

#[cfg(test)]
pub(crate) fn invoked(name: &str, arguments: &str) -> Routed {
    Routed::Call(Call {
        name: name.to_string(),
        arguments: arguments.to_string(),
    })
}

/// The call the model spelled out, read out of the two places it put it.
///
/// The name is what came before the marker, which is where the template puts
/// it; the arguments are the `args` of the envelope after it. A payload that is
/// not that envelope is not thrown away — its text becomes the arguments as it
/// stands, so a client's own parse fails on what the model actually said rather
/// than on something invented here.
fn call(named: String, payload: &str) -> Call {
    let body: Option<Value> = serde_json::from_str(payload).ok();
    let args = body
        .as_ref()
        .and_then(|body| body.get("args"))
        .filter(|args| args.is_object());
    let name = match named.is_empty() {
        false => named,
        true => body
            .as_ref()
            .and_then(|body| body.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    };
    Call {
        name,
        arguments: args.map_or_else(|| payload.to_string(), Value::to_string),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// What `just dump-chat-template-fixture` recorded the checkpoint's own
    /// template producing.
    const TEMPLATE_CASES: &str = "chat_template_cases.json";

    #[derive(Deserialize)]
    struct TemplateCases {
        cases: BTreeMap<String, Rendered>,
        refused: BTreeMap<String, Rendered>,
        canonicalised: BTreeMap<String, Canonicalised>,
    }

    #[derive(Deserialize)]
    struct Rendered {
        messages: Vec<Message>,
        #[serde(default)]
        tools: Vec<Value>,
        #[serde(default)]
        prompt: String,
    }

    /// A conversation the template refuses and this canonicalises instead: the
    /// shape a client sends, the shape the template takes, and the one prompt
    /// both have to reach.
    #[derive(Deserialize)]
    struct Canonicalised {
        sent: Vec<Message>,
        messages: Vec<Message>,
        #[serde(default)]
        tools: Vec<Value>,
        prompt: String,
    }

    fn template_cases() -> TemplateCases {
        serde_json::from_str(&inkling_core::fixture::read(TEMPLATE_CASES))
            .expect("the recorded template cases parse")
    }

    /// The claim the whole module rests on, checked rather than asserted: what
    /// this writes by hand is what `chat_template.jinja` renders, prompt for
    /// prompt, for every conversation it accepts.
    ///
    /// This is what makes hard-coding the structure defensible. A Jinja engine
    /// is a large dependency for a dozen literal markers in a fixed order, and
    /// the only thing it would buy is this agreement — so the agreement is
    /// recorded from the template and reproduced here, and a checkpoint that
    /// changes its template fails a test rather than serving prompts the model
    /// was never trained on.
    #[test]
    fn every_recorded_case_reproduces_what_the_checkpoints_own_template_renders() {
        let recorded = template_cases();
        assert!(recorded.cases.len() >= 25, "the fixture went missing cases");

        for (name, case) in &recorded.cases {
            assert_eq!(
                prompt(&case.messages, &case.tools).as_deref(),
                Ok(case.prompt.as_str()),
                "{name}"
            );
        }
    }

    /// A conversation the template raises on is one this refuses. The messages
    /// are recorded rather than restated so that the two sides disagree loudly
    /// if the template ever starts accepting one.
    #[test]
    fn a_conversation_the_template_refuses_is_refused_here_too() {
        let recorded = template_cases();
        assert!(!recorded.refused.is_empty(), "nothing was recorded refused");

        for (name, case) in &recorded.refused {
            assert!(prompt(&case.messages, &case.tools).is_err(), "{name}");
        }
    }

    /// The one shape the two sides deliberately disagree about. A client sends
    /// a call's `arguments` as a JSON string and the template refuses one,
    /// saying to canonicalise upstream — so this is upstream, and what it has
    /// to produce from the string is the prompt the template produces from the
    /// object.
    #[test]
    fn a_calls_arguments_reach_the_same_prompt_as_a_string_and_as_an_object() {
        let recorded = template_cases();
        assert!(
            !recorded.canonicalised.is_empty(),
            "nothing was recorded canonicalised"
        );

        for (name, case) in &recorded.canonicalised {
            assert_eq!(
                prompt(&case.messages, &case.tools).as_deref(),
                Ok(case.prompt.as_str()),
                "{name}: the form the template accepts"
            );
            assert_eq!(
                prompt(&case.sent, &case.tools).as_deref(),
                Ok(case.prompt.as_str()),
                "{name}: the form a client sends"
            );
        }
    }

    fn sent(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Some(Value::String(content.to_string())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// An assistant turn that called one tool, which is the shape a client
    /// replays and the shape this server emits.
    fn called(id: &str, name: &str, arguments: Value) -> Message {
        Message {
            role: "assistant".to_string(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: Some(id.to_string()),
                function: Called {
                    name: Some(Value::String(name.to_string())),
                    arguments: Some(arguments),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn prompted(messages: &[Message]) -> String {
        prompt(messages, &[]).expect("a conversation this maps")
    }

    fn declared(tools: &[Value]) -> String {
        prompt(&[sent("user", "Hi")], tools).expect("a conversation this maps")
    }

    const WEATHER: fn() -> Value = || {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look a city's weather up.",
                "parameters": {"type": "object"},
            },
        })
    };

    /// The structure the whole server rests on, spelled out in full for the
    /// commonest request there is. Every other case below states its difference
    /// from this one.
    #[test]
    fn a_system_and_a_user_message_become_the_turn_the_model_was_trained_on() {
        assert_eq!(
            prompted(&[sent("system", "Be brief."), sent("user", "Hi")]),
            "<|message_system|><|content_text|>Be brief.<|end_message|>\
             <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
             <|message_user|><|content_text|>Hi<|end_message|>\
             <|message_model|>"
        );
    }

    /// The caller's own system message comes first and the thinking effort after
    /// it, which is the ordering the template's `effort_emitted` flag produces
    /// and not the one "prepend a system message" would.
    #[test]
    fn the_thinking_effort_follows_a_system_message_and_precedes_a_user_one() {
        let prompt = prompted(&[sent("system", "Be brief."), sent("user", "Hi")]);
        let effort = prompt.find(THINKING_EFFORT).expect("the effort message");
        assert!(prompt.find("Be brief.").expect("the system message") < effort);
        assert!(effort < prompt.find("Hi").expect("the user message"));
    }

    /// The same request without a system message of its own. The effort message
    /// is emitted anyway — the model was trained with it — and it opens the
    /// prompt.
    #[test]
    fn a_user_message_alone_still_gets_the_thinking_effort() {
        assert_eq!(
            prompted(&[sent("user", "Hi")]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
             <|message_user|><|content_text|>Hi<|end_message|>\
             <|message_model|>"
        );
    }

    /// A conversation that is nothing but system messages never reaches the
    /// point the template emits the effort at, so the template emits it at the
    /// end instead. Odd-looking and worth reproducing: it is a real request —
    /// a system prompt and nothing else — and diverging here would put the
    /// effort message somewhere the model never saw one.
    #[test]
    fn a_conversation_of_system_messages_alone_gets_the_effort_last() {
        assert_eq!(
            prompted(&[sent("system", "Be brief.")]),
            "<|message_system|><|content_text|>Be brief.<|end_message|>\
             <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
             <|message_model|>"
        );
    }

    /// A reply already given is a message like any other, and it carries the
    /// marker that says the model *ended* it. Without that, the prompt reads as
    /// a model turn still in progress and the next user message arrives inside
    /// it.
    #[test]
    fn a_prior_reply_is_closed_with_the_marker_that_ends_a_model_turn() {
        assert_eq!(
            prompted(&[
                sent("user", "Hi"),
                sent("assistant", "Hello."),
                sent("user", "Again?"),
            ]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
             <|message_user|><|content_text|>Hi<|end_message|>\
             <|message_model|><|content_text|>Hello.<|end_message|><|content_model_end_sampling|>\
             <|message_user|><|content_text|>Again?<|end_message|>\
             <|message_model|>"
        );
    }

    /// What the server streamed out under `reasoning_content` goes back in as a
    /// model message of its own, ahead of the reply it led to. This is the round
    /// trip that makes splitting the thinking out of `content` lossless: a
    /// client can hand back what it was given.
    #[test]
    fn an_assistant_turns_thinking_precedes_its_content_as_its_own_message() {
        let mut reply = sent("assistant", "Hello.");
        reply.reasoning_content = Some("Weigh it up.".to_string());

        assert_eq!(
            prompted(&[sent("user", "Hi"), reply]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
             <|message_user|><|content_text|>Hi<|end_message|>\
             <|message_model|><|content_thinking|>Weigh it up.<|end_message|>\
             <|message_model|><|content_text|>Hello.<|end_message|><|content_model_end_sampling|>\
             <|message_model|>"
        );
    }

    /// Thinking on a role that has none is not a model turn, so it is not
    /// emitted. The template reads the field off assistant messages alone.
    #[test]
    fn thinking_on_a_user_message_is_not_emitted() {
        let mut asked = sent("user", "Hi");
        asked.reasoning_content = Some("Weigh it up.".to_string());
        assert_eq!(prompted(&[asked]), prompted(&[sent("user", "Hi")]));
    }

    /// A role nobody can map. Named back to the caller, because the difference
    /// between a typo and a role this does not have is the caller's to see.
    #[test]
    fn a_role_the_turn_structure_has_no_marker_for_is_refused() {
        assert_eq!(
            prompt(&[sent("developer", "Be brief.")], &[]),
            Err(ChatError::UnknownRole("developer".to_string()))
        );
    }

    /// A `content` that is neither a string nor a list is refused. The list is
    /// the shape most OpenAI clients send a user turn in and is written out
    /// above; a number is a client this cannot serve either way.
    #[test]
    fn content_that_is_neither_a_string_nor_a_list_is_refused() {
        for content in [serde_json::json!(7), serde_json::json!({"text": "Hi"})] {
            let mut asked = sent("user", "Hi");
            asked.content = Some(content.clone());
            assert_eq!(
                prompt(&[asked], &[]),
                Err(ChatError::ContentNotText("user".to_string())),
                "{content}"
            );
        }
    }

    /// The parts the engine cannot serve. The template renders a placeholder
    /// message for an image and for an audio clip, and raises on a type it has
    /// no channel for — this refuses all three, because a prompt that tells a
    /// text-only model an image was attached is a worse answer than a 400.
    #[test]
    fn a_content_part_that_is_not_text_is_refused() {
        for part in [
            serde_json::json!({"type": "image_url", "image_url": {"url": "data:,"}}),
            serde_json::json!({"type": "input_image", "image_url": "data:,"}),
            serde_json::json!({"type": "input_audio", "input_audio": {"data": ""}}),
            serde_json::json!({"type": "file", "file": {"file_id": "f"}}),
            serde_json::json!({"type": 7}),
            serde_json::json!({"type": null}),
            serde_json::json!(7),
        ] {
            let mut asked = sent("user", "Hi");
            asked.content = Some(serde_json::json!([{"type": "text", "text": "Hi"}, part]));
            assert_eq!(
                prompt(&[asked], &[]),
                Err(ChatError::ContentPartNotText("user".to_string())),
                "{part}"
            );
        }
    }

    /// An empty list is the template's own nothing, and this treats it as an
    /// absent `content`: a turn carrying nothing else is refused, and the one
    /// shape the template still renders — an assistant turn whose calls go out
    /// alone — reaches the prompt those calls alone produce.
    #[test]
    fn an_empty_content_list_is_an_absent_one() {
        for role in ["user", "assistant"] {
            let mut asked = sent(role, "");
            asked.content = Some(serde_json::json!([]));
            assert_eq!(
                prompt(&[asked], &[]),
                Err(ChatError::ContentNotText(role.to_string())),
                "{role}"
            );
        }

        let calling = |content: Value| {
            let mut turn = sent("assistant", "");
            turn.content = Some(content);
            turn.tool_calls = Some(vec![
                serde_json::from_value(serde_json::json!({
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": {}}
                }))
                .expect("the call parses"),
            ]);
            prompt(&[sent("user", "Hi"), turn], &[])
        };
        let absent = calling(Value::Null);
        assert!(
            absent
                .as_deref()
                .is_ok_and(|it| it.contains(CONTENT_INVOKE)),
            "the calls go out alone: {absent:?}"
        );
        assert_eq!(calling(serde_json::json!([])), absent);
    }

    /// A tool result's content parts, which the template renders as an empty
    /// result rather than as text. Refused, so that a client sending one is
    /// told rather than answered from a prompt its result vanished out of.
    #[test]
    fn a_tool_results_content_parts_are_refused() {
        let mut result = sent("tool", "");
        result.name = Some("get_weather".to_string());
        result.content = Some(serde_json::json!([{"type": "text", "text": "17C"}]));
        assert_eq!(
            prompt(&[sent("user", "Hi"), result], &[]),
            Err(ChatError::ContentNotText("tool".to_string()))
        );
    }

    /// A message with no `content` key at all and nothing else in it. The
    /// template emits not even its role marker, which is a request the client
    /// will not recognise the answer to.
    #[test]
    fn a_message_with_no_content_and_no_calls_is_refused() {
        for role in ["assistant", "user"] {
            let mut asked = sent(role, "");
            asked.content = None;
            assert_eq!(
                prompt(&[asked], &[]),
                Err(ChatError::ContentNotText(role.to_string())),
                "{role}"
            );
        }
    }

    #[test]
    fn a_conversation_with_no_messages_is_refused() {
        assert_eq!(prompt(&[], &[]), Err(ChatError::NoMessages));
    }

    /// The declaration goes ahead of the effort message and ahead of the
    /// conversation, which is not where "a system message" would put it: the
    /// template emits it before the loop over the messages runs at all.
    #[test]
    fn the_tool_declaration_precedes_even_the_thinking_effort() {
        let prompt = declared(&[WEATHER()]);
        let declaration = prompt.find(TOOL_DECLARE).expect("the declaration");
        assert_eq!(declaration, SYSTEM.len(), "{prompt}");
        assert!(declaration < prompt.find(THINKING_EFFORT).expect("the effort"));
    }

    /// A request that named no tool is the request it was before any of this
    /// existed, prompt for prompt. An empty list is what a client that has
    /// nothing to offer sends, and the template's `if tools` reads it as none.
    #[test]
    fn a_request_with_no_tools_is_the_prompt_it_was_without_them() {
        assert_eq!(declared(&[]), prompted(&[sent("user", "Hi")]));
    }

    /// The serialisation, which is what the model was trained on: keys sorted
    /// at every depth and not one space between them. A spec whose keys arrive
    /// in another order is the same prompt, and a spec written any other way is
    /// a prompt the model never saw.
    #[test]
    fn a_specs_keys_are_sorted_at_every_depth_and_spaced_nowhere() {
        let unsorted = serde_json::json!({
            "function": {"name": "z", "description": "d", "parameters": {"z": 1, "a": {"n": 2, "m": 1}}},
            "type": "function",
        });
        assert_eq!(
            declared(&[unsorted]),
            format!(
                "<|message_system|>tool_declare<|content_xml|>\
                 [{{\"description\":\"d\",\"name\":\"z\",\
                 \"parameters\":{{\"a\":{{\"m\":1,\"n\":2}},\"z\":1}},\
                 \"type\":\"function\"}}]<|end_message|>\
                 <|message_system|><|content_text|>{THINKING_EFFORT}<|end_message|>\
                 <|message_user|><|content_text|>Hi<|end_message|>\
                 <|message_model|>"
            )
        );
    }

    /// A tool with no `function` of its own *is* the function, and a spec that
    /// names neither a description nor parameters gets an empty one of each
    /// rather than losing the key. Both are the template's own defaults, and a
    /// spec missing a key is a different prompt from one carrying an empty one.
    #[test]
    fn a_spec_keeps_all_four_keys_however_few_the_client_sent() {
        let bare = declared(&[serde_json::json!({"name": "a"})]);
        assert!(
            bare.contains(
                "[{\"description\":\"\",\"name\":\"a\",\"parameters\":{},\"type\":\"function\"}]"
            ),
            "{bare}"
        );
        assert_eq!(
            bare,
            declared(&[serde_json::json!({"type": "function", "function": {"name": "a"}})]),
        );
    }

    /// A spec with nothing to name is not a spec. Refused rather than declared
    /// under an empty name, which is what the template's own `tojson` of an
    /// undefined name raises on.
    #[test]
    fn a_spec_without_a_name_is_refused() {
        assert_eq!(
            prompt(
                &[sent("user", "Hi")],
                &[serde_json::json!({"description": "d"})]
            ),
            Err(ChatError::SpecWithoutAName)
        );
    }

    /// The round trip a client replays: the call, then the result, then the
    /// marker that opens the next model turn. The name appears twice in a call
    /// — beside the marker and inside the envelope — because that is where the
    /// template puts it both times.
    #[test]
    fn a_call_and_its_result_are_the_two_messages_the_template_writes() {
        let mut result = sent("tool", "17C");
        result.tool_call_id = Some("call_1".to_string());

        assert_eq!(
            prompted(&[
                sent("user", "Weather in Paris?"),
                called(
                    "call_1",
                    "get_weather",
                    serde_json::json!({"city": "Paris"})
                ),
                result,
            ]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
             <|message_user|><|content_text|>Weather in Paris?<|end_message|>\
             <|message_model|>get_weather<|content_invoke_tool_json|>\
             {\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>\
             <|content_model_end_sampling|>\
             <|message_tool|>get_weather<|content_text|>17C<|end_message|>\
             <|message_model|>"
        );
    }

    /// A result names its tool by the id of the call it answers, and the id is
    /// looked up among the calls the conversation already carries. A client
    /// that sends the name outright says the same thing, and an id that names
    /// no call leaves the message unnamed rather than refused — all three are
    /// what the template does.
    #[test]
    fn a_results_tool_is_named_by_its_own_name_by_its_id_or_by_neither() {
        let call = called("call_1", "get_weather", serde_json::json!({}));
        let result = |name: Option<&str>, id: Option<&str>| {
            let mut result = sent("tool", "17C");
            result.name = name.map(str::to_string);
            result.tool_call_id = id.map(str::to_string);
            prompted(&[call.clone(), result])
        };

        let by_id = result(None, Some("call_1"));
        assert!(
            by_id.contains("<|message_tool|>get_weather<|content_text|>17C"),
            "{by_id}"
        );
        assert_eq!(result(Some("get_weather"), None), by_id);
        let unnamed = result(None, Some("call_gone"));
        assert!(
            unnamed.contains("<|message_tool|><|content_text|>17C"),
            "{unnamed}"
        );
    }

    /// A tool that returned nothing is a tool that returned nothing. The
    /// template renders an empty message rather than raising, and a result
    /// dropped here would be a conversation the model reads as never having
    /// called the tool.
    #[test]
    fn a_result_with_no_content_is_an_empty_message_rather_than_a_refusal() {
        let mut result = sent("tool", "");
        result.content = None;
        result.name = Some("get_weather".to_string());
        assert!(
            prompted(&[result])
                .contains("<|message_tool|>get_weather<|content_text|><|end_message|>")
        );
    }

    /// A turn that called two tools closes once. The marker ends the assistant's
    /// turn rather than each message inside it, so a second one would put the
    /// model in a turn it had already left.
    #[test]
    fn two_calls_in_one_turn_are_closed_by_one_end_of_sampling() {
        let mut turn = called("call_1", "a", serde_json::json!({}));
        turn.tool_calls.as_mut().expect("calls").push(ToolCall {
            id: Some("call_2".to_string()),
            function: Called {
                name: Some(Value::String("b".to_string())),
                arguments: None,
            },
        });

        let prompt = prompted(&[sent("user", "Hi"), turn]);
        assert_eq!(prompt.matches(END_SAMPLING).count(), 1, "{prompt}");
        assert_eq!(prompt.matches(CONTENT_INVOKE).count(), 2, "{prompt}");
        // Arguments the client left out are an empty object, not a missing key.
        assert!(prompt.contains("{\"name\":\"b\",\"args\":{}}"), "{prompt}");
    }

    /// The canonicalisation, stated on its own: the string every OpenAI client
    /// sends and the object the template takes are the same prompt, whatever
    /// spacing and whatever key order the client kept.
    #[test]
    fn a_calls_arguments_are_canonicalised_whichever_way_they_arrive() {
        let object = serde_json::json!({"city": "Paris", "units": "C"});
        let string = Value::String("{\"units\": \"C\", \"city\": \"Paris\"}".to_string());
        assert_eq!(
            prompted(&[called("call_1", "a", string)]),
            prompted(&[called("call_1", "a", object)])
        );
    }

    /// Arguments nobody can make an object of. A string that is not JSON is a
    /// client bug and a number is a different one, and both are named back
    /// rather than turned into an empty call the model would answer.
    #[test]
    fn arguments_that_are_not_an_object_are_refused() {
        let refused = |arguments| prompt(&[called("call_1", "a", arguments)], &[]);

        assert!(matches!(
            refused(Value::String("not json".to_string())),
            Err(ChatError::ArgumentsNotJson { .. })
        ));
        assert_eq!(
            refused(serde_json::json!(7)),
            Err(ChatError::ArgumentsNotAnObject("a".to_string()))
        );
        assert_eq!(
            refused(serde_json::json!("[1,2]")),
            Err(ChatError::ArgumentsNotAnObject("a".to_string()))
        );
    }

    /// Arguments a client left out, or left empty in any of the ways there are.
    /// The template's `if defined and truthy else {}` reads every one of them as
    /// no arguments, and an empty string is the one a reader would not guess.
    #[test]
    fn arguments_a_client_left_empty_are_an_empty_object() {
        let empty = prompted(&[called("call_1", "a", serde_json::json!({}))]);
        for arguments in [
            serde_json::json!(null),
            serde_json::json!(""),
            serde_json::json!([]),
        ] {
            assert_eq!(prompted(&[called("call_1", "a", arguments)]), empty);
        }
    }

    /// The boundary the template draws around a call's name, which is "defined
    /// and is a string" and not Jinja truthiness. Null and absent are refused;
    /// **an empty name is a name**, and it has to be — it is what the reading
    /// side produces when neither the text before the marker nor the envelope
    /// carried one, so refusing it would 400 a conversation on a turn this
    /// server itself wrote.
    #[test]
    fn a_call_needs_a_name_that_is_a_string_and_not_one_that_is_filled() {
        let named = |name| {
            let mut turn = called("call_1", "a", serde_json::json!({}));
            turn.tool_calls.as_mut().expect("calls")[0].function.name = name;
            prompt(&[turn], &[])
        };

        assert_eq!(named(None), Err(ChatError::CallWithoutAName));
        assert_eq!(named(Some(Value::Null)), Err(ChatError::CallWithoutAName));
        assert_eq!(
            named(Some(serde_json::json!(7))),
            Err(ChatError::CallWithoutAName)
        );
        assert!(
            named(Some(Value::String(String::new())))
                .expect("an empty name is a name")
                .contains(
                    "<|message_model|><|content_invoke_tool_json|>{\"name\":\"\",\"args\":{}}"
                ),
        );
    }

    /// The loop the empty name closes: a call the model named nowhere is read
    /// out as one, handed to a client as one, and replayed into a prompt as
    /// one. Asserted as the round trip rather than as three literals, so that a
    /// side that starts refusing what the other produces fails here.
    #[test]
    fn a_call_the_model_named_nowhere_still_replays_into_a_prompt() {
        let read = routed(
            &mut channels(),
            &[
                (MODEL_ID, MODEL),
                (INVOKE_ID, CONTENT_INVOKE),
                (WORD, "{\"args\":{}}"),
                (END_ID, END_MESSAGE),
            ],
        );
        assert_eq!(read, [invoked("", "{}")]);

        let Routed::Call(made) = read[0].clone() else {
            panic!("a call")
        };
        let replayed = called("call_1", &made.name, Value::String(made.arguments));
        assert!(prompted(&[replayed]).contains(CONTENT_INVOKE));
    }
    /// The marker ids of a vocabulary the tests can spell out. Which numbers
    /// they are decides nothing; that [`Channels`] is told them rather than
    /// guessing is the whole of what it needs.
    const THINKING_ID: u32 = 10;
    const TEXT_ID: u32 = 11;
    const END_ID: u32 = 12;
    const INVOKE_ID: u32 = 13;
    const MODEL_ID: u32 = 14;
    const WORD: u32 = 1;

    fn channels() -> Channels {
        Channels::new([
            (
                THINKING_ID,
                CONTENT_THINKING.to_string(),
                Reading::Channel(Channel::Thinking),
            ),
            (
                TEXT_ID,
                CONTENT_TEXT.to_string(),
                Reading::Channel(Channel::Content),
            ),
            (
                END_ID,
                END_MESSAGE.to_string(),
                Reading::Channel(Channel::Content),
            ),
            (INVOKE_ID, CONTENT_INVOKE.to_string(), Reading::Invocation),
            (MODEL_ID, MODEL.to_string(), Reading::Name),
        ])
    }

    fn routed(channels: &mut Channels, tokens: &[(u32, &str)]) -> Vec<Routed> {
        tokens
            .iter()
            .map(|(id, text)| channels.route(*id, text))
            .filter(|routed| *routed != Routed::Nothing)
            .collect()
    }

    /// A whole reply, routed the way the `turn` fixture case spells one: thinking
    /// first, then the answer, and not one marker in either.
    #[test]
    fn a_turn_is_split_into_the_two_channels_it_declares() {
        assert_eq!(
            routed(
                &mut channels(),
                &[
                    (THINKING_ID, CONTENT_THINKING),
                    (WORD, "Weigh it up."),
                    (TEXT_ID, CONTENT_TEXT),
                    (WORD, "Café."),
                    (END_ID, END_MESSAGE),
                ]
            ),
            [
                text(Channel::Thinking, "Weigh it up."),
                text(Channel::Content, "Café."),
            ]
        );
    }

    /// A reply that opens no channel at all is still a reply. It is held back
    /// while it could be the name of a tool — which is what the prompt's last
    /// marker leaves the reader expecting — and reaches content once the reply
    /// ends without a call.
    #[test]
    fn text_before_any_marker_reaches_content_when_the_reply_ends() {
        let mut channels = channels();
        assert_eq!(channels.route(WORD, "Hello."), Routed::Nothing);
        assert_eq!(channels.finish(""), text(Channel::Content, "Hello."));
    }

    /// The bytes a detokenizer was holding when a marker arrived come back with
    /// the marker's own text, and they are the message's while the marker is
    /// not. They belong to the channel that was open when they were held back.
    #[test]
    fn bytes_released_by_a_marker_reach_the_channel_that_was_open() {
        let mut channels = channels();
        channels.route(THINKING_ID, CONTENT_THINKING);
        assert_eq!(
            channels.route(TEXT_ID, &format!("\u{fffd}{CONTENT_TEXT}")),
            text(Channel::Thinking, "\u{fffd}")
        );
        assert_eq!(
            channels.route(WORD, "Hello."),
            text(Channel::Content, "Hello.")
        );
    }

    /// A message that ended leaves the reader in content, which is where text
    /// nobody has named a channel for goes. The thinking channel it was in
    /// belonged to a message that is over.
    #[test]
    fn a_message_that_ended_leaves_the_reader_in_content() {
        let mut channels = channels();
        channels.route(THINKING_ID, CONTENT_THINKING);
        channels.route(END_ID, END_MESSAGE);
        assert_eq!(channels.route(WORD, "more"), text(Channel::Content, "more"));
    }

    /// A marker's text is what the vocabulary spells it, and a token whose text
    /// merely *reads* like one is not one. Only the id decides.
    #[test]
    fn an_ordinary_token_that_reads_like_a_marker_is_not_one() {
        let mut channels = channels();
        channels.route(TEXT_ID, CONTENT_TEXT);
        assert_eq!(
            channels.route(WORD, CONTENT_THINKING),
            text(Channel::Content, CONTENT_THINKING)
        );
    }

    /// The milestone's own case: the model opens a turn, names a tool, and
    /// spells the envelope out. What reaches the client is the call — the name
    /// from beside the marker and the `args` from inside the envelope — and not
    /// one character of either in `content`.
    #[test]
    fn a_call_the_model_spelled_out_arrives_as_a_call_and_not_as_prose() {
        assert_eq!(
            routed(
                &mut channels(),
                &[
                    (THINKING_ID, CONTENT_THINKING),
                    (WORD, "Weigh it up."),
                    (END_ID, END_MESSAGE),
                    (MODEL_ID, MODEL),
                    (WORD, "get_"),
                    (WORD, "weather"),
                    (INVOKE_ID, CONTENT_INVOKE),
                    (WORD, "{\"name\":\"get_weather\",\"args\":"),
                    (WORD, "{\"city\":\"Paris\"}}"),
                    (END_ID, END_MESSAGE),
                ]
            ),
            [
                text(Channel::Thinking, "Weigh it up."),
                invoked("get_weather", "{\"city\":\"Paris\"}"),
            ]
        );
    }

    /// Two calls out of one turn, which is the shape an agent asking for two
    /// files at once produces. Each is closed by its own `<|end_message|>` and
    /// the turn by the end-of-sampling marker.
    #[test]
    fn two_calls_in_one_turn_arrive_as_two_calls() {
        assert_eq!(
            routed(
                &mut channels(),
                &[
                    (MODEL_ID, MODEL),
                    (WORD, "a"),
                    (INVOKE_ID, CONTENT_INVOKE),
                    (WORD, "{\"name\":\"a\",\"args\":{\"x\":1}}"),
                    (END_ID, END_MESSAGE),
                    (MODEL_ID, MODEL),
                    (WORD, "b"),
                    (INVOKE_ID, CONTENT_INVOKE),
                    (WORD, "{\"name\":\"b\",\"args\":{}}"),
                    (END_ID, END_MESSAGE),
                ]
            ),
            [invoked("a", "{\"x\":1}"), invoked("b", "{}")]
        );
    }

    /// An envelope the model got wrong. Its text becomes the arguments as it
    /// stands, so the client's own parse fails on what the model actually said
    /// — the alternative is a call with arguments nobody generated.
    #[test]
    fn an_envelope_that_is_not_the_one_the_template_writes_is_passed_through() {
        for (payload, arguments) in [
            // Not JSON at all.
            ("{\"city\": ", "{\"city\": "),
            // JSON, but not the envelope: no `args` to take.
            ("{\"city\":\"Paris\"}", "{\"city\":\"Paris\"}"),
            // An `args` that is not an object.
            ("{\"name\":\"a\",\"args\":7}", "{\"name\":\"a\",\"args\":7}"),
        ] {
            let mut channels = channels();
            assert_eq!(
                routed(
                    &mut channels,
                    &[
                        (MODEL_ID, MODEL),
                        (WORD, "a"),
                        (INVOKE_ID, CONTENT_INVOKE),
                        (WORD, payload),
                        (END_ID, END_MESSAGE),
                    ]
                ),
                [invoked("a", arguments)],
                "{payload}"
            );
        }
    }

    /// A call the model never named, which the envelope names anyway. The
    /// template writes the name twice and either one is enough to say which
    /// tool the client is being asked for.
    #[test]
    fn a_call_with_no_name_beside_the_marker_takes_the_envelopes() {
        assert_eq!(
            routed(
                &mut channels(),
                &[
                    (MODEL_ID, MODEL),
                    (INVOKE_ID, CONTENT_INVOKE),
                    (WORD, "{\"name\":\"a\",\"args\":{}}"),
                    (END_ID, END_MESSAGE),
                ]
            ),
            [invoked("a", "{}")]
        );
    }

    /// A budget that ran out in the middle of a call. The call goes out with
    /// the envelope as far as the model got it, rather than not at all: a
    /// client reconciling `finish_reason` against arguments it cannot parse is
    /// being told what happened, and one told nothing is not.
    #[test]
    fn a_call_the_budget_cut_short_still_reaches_the_client() {
        let mut channels = channels();
        let cut = routed(
            &mut channels,
            &[
                (MODEL_ID, MODEL),
                (WORD, "a"),
                (INVOKE_ID, CONTENT_INVOKE),
                (WORD, "{\"name\":\"a\",\"args\":{\"cit"),
            ],
        );
        assert_eq!(cut, []);
        assert_eq!(
            channels.finish("\u{fffd}"),
            invoked("a", "{\"name\":\"a\",\"args\":{\"cit\u{fffd}")
        );
    }

    /// The two sides against each other: the envelope the prompt writes for a
    /// call is the envelope this reads one out of. Neither is asserted against
    /// a literal here — what is asserted is that they agree, so a change to
    /// either has to keep them agreeing.
    #[test]
    fn a_call_written_into_a_prompt_is_the_call_read_back_out_of_one() {
        let arguments = serde_json::json!({"city": "Paris", "units": "C"});
        let written = prompted(&[called("call_1", "get_weather", arguments.clone())]);
        let envelope = written
            .split_once(CONTENT_INVOKE)
            .and_then(|(_, rest)| rest.split_once(END_MESSAGE))
            .expect("an invocation")
            .0;

        assert_eq!(
            routed(
                &mut channels(),
                &[
                    (MODEL_ID, MODEL),
                    (WORD, "get_weather"),
                    (INVOKE_ID, CONTENT_INVOKE),
                    (WORD, envelope),
                    (END_ID, END_MESSAGE),
                ]
            ),
            [invoked("get_weather", &arguments.to_string())]
        );
    }
}
