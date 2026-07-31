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
//! is seven literal markers in a fixed order, and one that ran the real template
//! would accept requests — tools, images, audio — that the engine underneath
//! cannot serve anyway.
//!
//! What that costs is that a checkpoint whose template differs is templated
//! wrongly and silently, so the divergences are worth stating rather than
//! discovering. Against `models/Inkling-Small-mxfp4/chat_template.jinja`, this
//! reproduces the template exactly for the messages it accepts. It does not
//! implement:
//!
//! - **`tools` and the `tool` role.** The template declares tool specs in a
//!   leading system message and renders tool calls as their own model messages.
//!   Refused rather than dropped — a request whose tools vanished gets an answer
//!   that reads as a refusal to use them.
//! - **Content parts.** A `content` that is a list — text parts, `input_image`,
//!   `input_audio` — is refused. The engine is text-only.
//! - **`reasoning_effort`.** The template maps six names onto numbers and accepts
//!   a float, defaulting to 0.9 when the caller names none. Only the default is
//!   emitted here, which is the string `generate`'s docs measured the template
//!   producing.
//! - **A `content` that is absent or null.** The template emits nothing at all
//!   for such a message, not even its role marker. Refused here, because a
//!   message that silently contributes nothing to the prompt is a request the
//!   client will not recognise the answer to.
//!
//! # Where the thinking-effort message goes
//!
//! Not simply first. The template emits it before the first message whose role is
//! *not* `system`, so a caller's own system prompt precedes it, and emits it at
//! the end of a conversation that never had one. That ordering is reproduced
//! rather than approximated: it is a system message either way, and the model was
//! trained on it in that position.

use serde::Deserialize;

/// The markers, as the vocabulary spells them.
const SYSTEM: &str = "<|message_system|>";
const USER: &str = "<|message_user|>";
const MODEL: &str = "<|message_model|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";

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
    pub content: Option<serde_json::Value>,
    /// An assistant turn's thinking, which is the field the checkpoint's own
    /// template reads it from and the field [`crate::openai`] streams it back
    /// out under. The two together are what lets a client feed a reply it was
    /// given back into the next request unchanged.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChatError {
    #[error("a conversation needs at least one message")]
    NoMessages,

    #[error("{0:?} is not a role; this server takes system, user and assistant")]
    UnknownRole(String),

    #[error("the {role} role needs {needs}, which this server does not implement")]
    UnsupportedRole {
        role: &'static str,
        needs: &'static str,
    },

    #[error("the content of the {0} message is not a string; this server takes no content parts")]
    ContentNotText(String),
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
}

impl Role {
    fn marker(self) -> &'static str {
        match self {
            Role::System => SYSTEM,
            Role::User => USER,
            Role::Model => MODEL,
        }
    }

    fn parse(role: &str) -> Result<Self, ChatError> {
        match role {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Model),
            "tool" => Err(ChatError::UnsupportedRole {
                role: "tool",
                needs: "tool calling",
            }),
            _ => Err(ChatError::UnknownRole(role.to_string())),
        }
    }
}

impl Message {
    fn text(&self) -> Result<&str, ChatError> {
        match self.content.as_ref().and_then(serde_json::Value::as_str) {
            Some(text) => Ok(text),
            None => Err(ChatError::ContentNotText(self.role.clone())),
        }
    }
}

/// One message written out, which is the same four parts for every role and
/// every channel: whose turn it is, which channel it is in, the text, and the
/// end of the message.
fn message(marker: &str, channel: &str, content: &str, out: &mut String) {
    out.push_str(marker);
    out.push_str(channel);
    out.push_str(content);
    out.push_str(END_MESSAGE);
}

/// The conversation as a prompt, ending in the marker that opens the model's own
/// turn.
///
/// That last marker is the whole point. It is what a template calls
/// `add_generation_prompt`, and it is what puts the model in a turn it can end —
/// without it the stopping rule is correct and never fires.
pub fn prompt(messages: &[Message]) -> Result<String, ChatError> {
    if messages.is_empty() {
        return Err(ChatError::NoMessages);
    }

    let mut out = String::new();
    let mut effort_emitted = false;
    for sent in messages {
        let role = Role::parse(&sent.role)?;
        let content = sent.text()?;

        if !effort_emitted && role != Role::System {
            message(SYSTEM, CONTENT_TEXT, THINKING_EFFORT, &mut out);
            effort_emitted = true;
        }

        if let (Role::Model, Some(thinking)) = (role, &sent.reasoning_content) {
            message(MODEL, CONTENT_THINKING, thinking, &mut out);
        }
        message(role.marker(), CONTENT_TEXT, content, &mut out);
        if role == Role::Model {
            out.push_str(END_SAMPLING);
        }
    }
    if !effort_emitted {
        message(SYSTEM, CONTENT_TEXT, THINKING_EFFORT, &mut out);
    }

    out.push_str(MODEL);
    Ok(out)
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

/// The markers a vocabulary is asked for, and the channel each opens. `None`
/// closes the message rather than opening anything.
///
/// `<|end_message|>` and the end-of-sequence id are here for what they are *not*:
/// text. The tokenizer renders special tokens literally rather than swallowing
/// them, and both of these reach a sink like any other token — so a server that
/// did not name them would put `<|end_message|><|content_model_end_sampling|>` on
/// the end of every reply it sent.
pub const MARKERS: [(&str, Option<Channel>); 4] = [
    (CONTENT_THINKING, Some(Channel::Thinking)),
    (CONTENT_TEXT, Some(Channel::Content)),
    (END_MESSAGE, None),
    (END_SAMPLING, None),
];

/// Which channel the text arriving now belongs to, kept across tokens.
///
/// It starts on [`Channel::Content`]. The model's first token after
/// `<|message_model|>` is ordinarily a channel marker, but a reply that opened
/// with neither is a reply, and putting it somewhere a client renders beats
/// dropping it.
#[derive(Debug)]
pub struct Channels {
    markers: Vec<(u32, String, Option<Channel>)>,
    current: Channel,
}

impl Channels {
    pub fn new(markers: impl IntoIterator<Item = (u32, String, Option<Channel>)>) -> Self {
        Self {
            markers: markers.into_iter().collect(),
            current: Channel::Content,
        }
    }

    /// The channel that is open now, which is where text nobody handed an id for
    /// belongs — the bytes a detokenizer was still holding when the generation
    /// ended.
    pub fn current(&self) -> Channel {
        self.current
    }

    /// Where `text` — the text token `id` contributed — goes, and what of it a
    /// client should see.
    ///
    /// A marker's own literal is not part of the message and is cut off. What
    /// can precede it is not nothing, though: a detokenizer holds back the bytes
    /// of a character it has not finished, and a special token is what releases
    /// them, so the text of a marker token can be a replacement character and
    /// then the marker. Those bytes belong to the channel that was open when
    /// they were held back, which is why the switch happens after they are
    /// handed back and not before.
    pub fn route(&mut self, id: u32, text: &str) -> (Channel, String) {
        let was = self.current;
        let Some((_, literal, opens)) = self.markers.iter().find(|(marker, ..)| *marker == id)
        else {
            return (was, text.to_string());
        };

        let released = text
            .strip_suffix(literal.as_str())
            .unwrap_or(text)
            .to_string();
        if let Some(channel) = opens {
            self.current = *channel;
        }
        (was, released)
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
    }

    #[derive(Deserialize)]
    struct Rendered {
        messages: Vec<Message>,
        #[serde(default)]
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
    /// is a large dependency for seven literal markers in a fixed order, and the
    /// only thing it would buy is this agreement — so the agreement is recorded
    /// from the template and reproduced here, and a checkpoint that changes its
    /// template fails a test rather than serving prompts the model was never
    /// trained on.
    #[test]
    fn every_recorded_case_reproduces_what_the_checkpoints_own_template_renders() {
        let recorded = template_cases();
        assert!(recorded.cases.len() >= 8, "the fixture went missing cases");

        for (name, case) in &recorded.cases {
            assert_eq!(
                prompt(&case.messages).as_deref(),
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
            assert!(prompt(&case.messages).is_err(), "{name}");
        }
    }

    fn sent(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Some(serde_json::Value::String(content.to_string())),
            reasoning_content: None,
        }
    }

    fn prompted(messages: &[Message]) -> String {
        prompt(messages).expect("a conversation this maps")
    }

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
            prompt(&[sent("developer", "Be brief.")]),
            Err(ChatError::UnknownRole("developer".to_string()))
        );
    }

    /// `tool` is a role the *template* has a marker for and this server does
    /// not, which is a different answer from "no such role" and says so.
    #[test]
    fn the_tool_role_is_refused_as_unimplemented_rather_than_unknown() {
        assert_eq!(
            prompt(&[sent("tool", "42")]),
            Err(ChatError::UnsupportedRole {
                role: "tool",
                needs: "tool calling"
            })
        );
    }

    /// Content parts — the shape an OpenAI client sends an image in, and the
    /// shape several send plain text in. Refused by name rather than flattened,
    /// which is what silently mangling one would look like from the outside.
    #[test]
    fn content_that_is_not_a_string_is_refused() {
        for content in [
            serde_json::json!([{"type": "text", "text": "Hi"}]),
            serde_json::json!(null),
            serde_json::json!(7),
        ] {
            let sent = Message {
                role: "user".to_string(),
                content: Some(content.clone()),
                reasoning_content: None,
            };
            assert_eq!(
                prompt(&[sent]),
                Err(ChatError::ContentNotText("user".to_string())),
                "{content}"
            );
        }
    }

    /// A message with no `content` key at all, which is what an assistant turn
    /// carrying only tool calls looks like.
    #[test]
    fn a_message_with_no_content_at_all_is_refused() {
        let sent = Message {
            role: "assistant".to_string(),
            content: None,
            reasoning_content: None,
        };
        assert_eq!(
            prompt(&[sent]),
            Err(ChatError::ContentNotText("assistant".to_string()))
        );
    }

    #[test]
    fn a_conversation_with_no_messages_is_refused() {
        assert_eq!(prompt(&[]), Err(ChatError::NoMessages));
    }

    /// The marker ids of a vocabulary the tests can spell out. Which numbers
    /// they are decides nothing; that [`Channels`] is told them rather than
    /// guessing is the whole of what it needs.
    const THINKING_ID: u32 = 10;
    const TEXT_ID: u32 = 11;
    const END_ID: u32 = 12;
    const WORD: u32 = 1;

    fn channels() -> Channels {
        Channels::new([
            (
                THINKING_ID,
                CONTENT_THINKING.to_string(),
                Some(Channel::Thinking),
            ),
            (TEXT_ID, CONTENT_TEXT.to_string(), Some(Channel::Content)),
            (END_ID, END_MESSAGE.to_string(), None),
        ])
    }

    /// A whole reply, routed the way the `turn` fixture case spells one: thinking
    /// first, then the answer, and not one marker in either.
    #[test]
    fn a_turn_is_split_into_the_two_channels_it_declares() {
        let mut channels = channels();
        let routed: Vec<(Channel, String)> = [
            (THINKING_ID, CONTENT_THINKING),
            (WORD, "Weigh it up."),
            (TEXT_ID, CONTENT_TEXT),
            (WORD, "Café."),
            (END_ID, END_MESSAGE),
        ]
        .into_iter()
        .map(|(id, text)| channels.route(id, text))
        .collect();

        assert_eq!(
            routed,
            [
                (Channel::Content, String::new()),
                (Channel::Thinking, "Weigh it up.".to_string()),
                (Channel::Thinking, String::new()),
                (Channel::Content, "Café.".to_string()),
                (Channel::Content, String::new()),
            ]
        );
    }

    /// A reply that opens with no marker at all is still a reply. Content is
    /// where a client renders it, and where dropping it would be silent.
    #[test]
    fn text_before_any_marker_is_content() {
        assert_eq!(
            channels().route(WORD, "Hello."),
            (Channel::Content, "Hello.".to_string())
        );
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
            (Channel::Thinking, "\u{fffd}".to_string())
        );
        assert_eq!(
            channels.route(WORD, "Hello."),
            (Channel::Content, "Hello.".to_string())
        );
    }

    /// An id that only ends things leaves the channel where it was, so a marker
    /// list that names it is not the same as one that switches on it.
    #[test]
    fn a_marker_that_opens_nothing_leaves_the_channel_alone() {
        let mut channels = channels();
        channels.route(THINKING_ID, CONTENT_THINKING);
        channels.route(END_ID, END_MESSAGE);
        assert_eq!(
            channels.route(WORD, "more"),
            (Channel::Thinking, "more".to_string())
        );
    }

    /// A marker's text is what the vocabulary spells it, and a token whose text
    /// merely *ends* the same way is not one. Only the id decides.
    #[test]
    fn an_ordinary_token_that_reads_like_a_marker_is_not_one() {
        assert_eq!(
            channels().route(WORD, CONTENT_THINKING),
            (Channel::Content, CONTENT_THINKING.to_string())
        );
    }
}
