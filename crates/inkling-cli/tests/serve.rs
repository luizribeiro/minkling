//! The `serve` command as a client meets it: a socket in, HTTP out.
//!
//! Everything the server is made of is tested where it lives — the turn
//! structure against the prompts `chat_template.jinja` renders, the framing
//! against its own frames, the loop against the synthetic stack. What only a
//! running process can settle is that a real request reaches the real model and
//! comes back as text: that the messages were templated, that the reply is the
//! model's answer rather than the prompt continued, and that not one turn marker
//! leaked into the field a client renders.
//!
//! Gated on `INKLINGRS_CHECKPOINT`, which is 0.3 s to load and 0.35 GiB peak.
//! Unset, it reports a skip and passes; `just test-full` sets it.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use inkling_cli::chat::{self, Message};
use inkling_cli::wire::{calls, dechunked, delta, payloads};
use inkling_core::{DEFAULT_BOUND, Kept, Tokenizer};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

/// How many tokens the end-to-end case decodes.
///
/// Two, against a prefill of the twenty-odd tokens a templated `Hi` makes. The
/// reply is not a sentence at that budget and is not meant to be — what it
/// settles is that the turn structure reached the model and that what came back
/// is HTTP, and a longer one would only cost minutes to settle the same thing.
const GENERATED: usize = 2;

/// The budget the tool cases decode under.
///
/// **A whole turn rather than a fragment of one, which is the difference from
/// [`GENERATED`].** What a tool case has to settle is that the model reached the
/// end of a call it started, and a budget that cut one short would settle
/// nothing about the shape a client reads — so this is generous enough for the
/// thinking the model opens with and the call after it. The measured turn is 62
/// tokens; the round trip that answers it is 18.
const CALLED: usize = 128;

/// The tool the end-to-end cases declare, and the question that reaches for it.
fn weather() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look a city's current weather up.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city name."}},
                "required": ["city"],
            },
        },
    }])
}

const ASKED: &str = "What is the weather in Paris right now?";

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os(CHECKPOINT_VAR).map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: {CHECKPOINT_VAR} is unset");
    }
    dir
}

/// A server for the duration of a test, killed however the test ends.
struct Serving {
    child: Child,
    address: String,
}

impl Drop for Serving {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Serving {
    /// Started on a port the operating system picks, and waited for.
    ///
    /// Port zero rather than a number chosen here: a fixed port is a test that
    /// fails when anything else on the machine happens to hold it. What the
    /// server bound is read back off the line it prints once the checkpoint is
    /// loaded, which is also what makes this wait for the load rather than race
    /// it.
    fn start(checkpoint: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_inklingrs"))
            .args([
                "serve",
                checkpoint.to_str().expect("a printable checkpoint path"),
                "--address",
                "127.0.0.1:0",
            ])
            .stderr(Stdio::piped())
            .spawn()
            .expect("the binary runs");

        let mut log = BufReader::new(child.stderr.take().expect("stderr is piped"));
        let mut address = None;
        let mut line = String::new();
        while log.read_line(&mut line).expect("the server logs") > 0 {
            eprint!("{line}");
            if let Some((_, listening)) = line.trim_end().split_once("on http://") {
                address = Some(listening.split(',').next().unwrap_or(listening).to_string());
                break;
            }
            line.clear();
        }

        // Drained from here on. The server writes a line per request and a
        // pipe nobody reads eventually blocks the process writing to it.
        std::thread::spawn(move || {
            let mut rest = String::new();
            let _ = log.read_to_string(&mut rest);
            eprint!("{rest}");
        });

        Self {
            child,
            address: address.expect("the server says what it bound"),
        }
    }

    /// One request, and the whole response.
    ///
    /// `Connection: close` is the *request's*, which `tiny_http` honours by
    /// closing the connection once this one is answered — so reading to the end
    /// of the socket is reading to the end of the response. The server's own
    /// framing is checked separately, and is what a client that keeps the
    /// connection open relies on instead.
    fn request(&self, head: &str, body: &str) -> String {
        let mut socket = TcpStream::connect(&self.address).expect("the server is listening");
        let request = format!(
            "{head} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            self.address,
            body.len()
        );
        socket
            .write_all(request.as_bytes())
            .expect("the request goes out");
        socket.flush().expect("the request is flushed");

        let mut response = Vec::new();
        socket
            .read_to_end(&mut response)
            .expect("the server answers");
        String::from_utf8(response).expect("the response is utf8")
    }
}

fn head_and_body(response: &str) -> (&str, &str) {
    response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("{response:?} is not an http response"))
}

/// The milestone, as a client sees it: a chat request in, an event stream of the
/// model's own answer out.
///
/// The assertion that matters is that the reply opens a thinking channel.
/// `generate`'s docs record what an untemplated prompt does — the model continues
/// the text and `<|content_model_end_sampling|>` never arrives — so a server that
/// forwarded the user's words unwrapped would answer a `Hi` with prose about
/// greetings. Nothing but the turn structure puts the model in a turn, and a
/// model in a turn is what opens a channel.
///
/// Everything else this server does is checked here too rather than in a test of
/// its own, because a second test is a second server holding the whole model, and
/// `cargo nextest` would run the two at once.
#[test]
fn a_chat_request_is_answered_with_the_models_own_turn_streamed_back() {
    let Some(dir) = checkpoint_dir() else { return };
    let serving = Serving::start(&dir);

    let listing = serving.request("GET /v1/models", "");
    let (head, body) = head_and_body(&listing);
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let listed: serde_json::Value = serde_json::from_str(body).expect("a json listing");
    let model = listed["data"][0]["id"].as_str().expect("a model id");
    assert_eq!(
        model,
        dir.file_name()
            .expect("a named directory")
            .to_string_lossy()
    );

    let asked = serde_json::json!({
        "model": model,
        "stream": true,
        "max_tokens": GENERATED,
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let answered = serving.request("POST /v1/chat/completions", &asked.to_string());
    let (head, body) = head_and_body(&answered);
    eprintln!("{answered}");

    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(head.contains("Content-Type: text/event-stream"), "{head}");
    assert!(head.contains("Transfer-Encoding: chunked"), "{head}");
    assert!(
        body.ends_with("0\r\n\r\n"),
        "the stream never ends: {body:?}"
    );

    let payloads = payloads(&dechunked(body));
    assert_eq!(payloads[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(payloads[0]["model"], model);
    assert_eq!(
        payloads.last().expect("a last chunk")["choices"][0]["finish_reason"],
        "length",
        "{GENERATED} tokens is not a whole turn, so the budget is what ends it"
    );

    let mut content = String::new();
    let mut reasoning = String::new();
    for (field, text) in payloads.iter().filter_map(delta) {
        match field.as_str() {
            "content" => content.push_str(&text),
            _ => reasoning.push_str(&text),
        }
    }
    eprintln!("content {content:?}, reasoning {reasoning:?}");

    // The turn structure reached the model: it opened a channel, which is
    // something an untemplated prompt never gives it a reason to do.
    assert!(
        !reasoning.is_empty(),
        "the model opened no thinking channel, so it was not put in a turn"
    );
    for text in [&content, &reasoning] {
        assert!(
            !text.contains("<|"),
            "a marker reached the client: {text:?}"
        );
    }

    // A role nobody can map, refused by name and before a decode step is spent
    // on it. The turn structure's own tests say which message; this says the
    // status code a client reads it off, and that the server is still serving
    // afterwards.
    let asked = serde_json::json!({
        "messages": [{"role": "developer", "content": "Be brief."}],
    });
    let refused = serving.request("POST /v1/chat/completions", &asked.to_string());
    let (head, body) = head_and_body(&refused);
    assert!(head.starts_with("HTTP/1.1 400"), "{head}");
    let refusal: serde_json::Value = serde_json::from_str(body).expect("a json body");
    let message = refusal["error"]["message"].as_str().expect("a message");
    assert!(message.contains("developer"), "{message}");

    let missing = serving.request("GET /v1/embeddings", "");
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
}

/// The milestone, as an agent meets it: a question and a tool, and back comes
/// the call rather than a paragraph about calling one.
///
/// What only the real model can settle is that the declaration reached it in a
/// shape it recognises. Every marker and every separator is held against
/// `chat_template.jinja` where the turn structure lives, and a spec serialised
/// some other way would still be a valid prompt — it would simply be one the
/// model was never trained on, and the only symptom is an answer that reaches
/// for nothing. So this asserts on the *call*: its name, and arguments that
/// parse and carry the city that was asked about.
///
/// Then the round trip, which is the second half of what a client does with a
/// call: hand the result back and get an answer that used it. A conversation
/// that did not round-trip would be answered — the model always answers — and
/// answered about the wrong thing.
///
/// A server of its own rather than a case inside
/// [`a_chat_request_is_answered_with_the_models_own_turn_streamed_back`],
/// because `just test-full` runs a process a test and the two are what that
/// buys: one holds the model for the plain path and one for this. Both are
/// gated on the checkpoint.
#[test]
fn a_tool_a_request_declares_is_a_tool_the_model_calls_and_answers_from() {
    let Some(dir) = checkpoint_dir() else { return };
    let serving = Serving::start(&dir);

    let asked = serde_json::json!({
        "stream": true,
        "max_tokens": CALLED,
        "tools": weather(),
        "messages": [{"role": "user", "content": ASKED}],
    });
    let answered = serving.request("POST /v1/chat/completions", &asked.to_string());
    let (head, body) = head_and_body(&answered);
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");

    let payloads = payloads(&dechunked(body));
    let made: Vec<serde_json::Value> = payloads.iter().flat_map(calls).collect();
    eprintln!("{made:?}");
    assert_eq!(made.len(), 1, "the model asked for no tool: {payloads:?}");
    assert_eq!(made[0]["function"]["name"], "get_weather");
    assert_eq!(
        payloads.last().expect("a last chunk")["choices"][0]["finish_reason"],
        "tool_calls",
        "a turn that asked for a tool is not a turn that had its say"
    );

    // The arguments as a client uses them: parsed, and about the city asked
    // for. A model handed the envelope rather than its `args` would fail here
    // on a key it never sent.
    let arguments = made[0]["function"]["arguments"]
        .as_str()
        .expect("arguments are a string");
    let arguments: serde_json::Value =
        serde_json::from_str(arguments).expect("arguments are the JSON a tool takes");
    assert_eq!(arguments["city"], "Paris", "{arguments}");

    // Not one character of the call in the field a client renders, which is
    // what the whole of the reading side is for.
    let content: String = payloads
        .iter()
        .filter_map(delta)
        .filter(|(field, _)| field == "content")
        .map(|(_, text)| text)
        .collect();
    assert!(!content.contains("get_weather"), "{content:?}");
    assert!(!content.contains("<|"), "{content:?}");

    // The round trip. The call goes back as the client was handed it — id,
    // name, and `arguments` as the JSON *string* every OpenAI client keeps —
    // and the result beside it.
    let replayed = serde_json::json!({
        "max_tokens": CALLED,
        "tools": weather(),
        "messages": [
            {"role": "user", "content": ASKED},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": made[0]["id"],
                "type": "function",
                "function": made[0]["function"],
            }]},
            {"role": "tool", "tool_call_id": made[0]["id"], "content": "17C, light rain."},
        ],
    });
    let answered = serving.request("POST /v1/chat/completions", &replayed.to_string());
    let (head, body) = head_and_body(&answered);
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let answer: serde_json::Value = serde_json::from_str(body).expect("a json body");
    let said = answer["choices"][0]["message"]["content"]
        .as_str()
        .expect("an answer");
    eprintln!("{said:?}");
    assert!(
        said.contains("17"),
        "the result never reached the model: {said:?}"
    );
    assert_eq!(answer["choices"][0]["finish_reason"], "stop");

    // A `tool_choice` this does not implement, refused before a decode step is
    // spent on it — and the server still serving afterwards.
    let refused = serde_json::json!({
        "tools": weather(),
        "tool_choice": "required",
        "messages": [{"role": "user", "content": ASKED}],
    });
    let refused = serving.request("POST /v1/chat/completions", &refused.to_string());
    let (head, body) = head_and_body(&refused);
    assert!(head.starts_with("HTTP/1.1 400"), "{head}");
    let refusal: serde_json::Value = serde_json::from_str(body).expect("a json body");
    let message = refusal["error"]["message"].as_str().expect("a message");
    assert!(message.contains("required"), "{message}");
}

/// **K1's kept cache across a tool round-trip**, which is the shape the cache
/// exists for and the one nothing had checked it on.
///
/// A turn that answers a call is the turn before it with three things added —
/// the call, the marker that ends the model's turn, and the result — so it is an
/// exact extension, which is the only thing [`Kept`] can serve from. That is a
/// claim about *tokens* and not about markers: the two prompts could agree
/// character for character and still part company at the first token if a merge
/// straddled the boundary, so it is asked of the vocabulary rather than of the
/// strings.
///
/// Gated on the checkpoint for the tokenizer alone, which is milliseconds. No
/// weights are loaded and nothing is generated.
#[test]
fn a_tool_round_trip_is_an_exact_extension_of_the_turn_it_answers() {
    let Some(dir) = checkpoint_dir() else { return };
    let config = inkling_cli::config::of_checkpoint(&dir).expect("the checkpoint's config");
    let tokenizer = Tokenizer::open(&dir, &config).expect("the checkpoint's vocabulary");

    let asked: Vec<Message> =
        serde_json::from_value(serde_json::json!([{"role": "user", "content": ASKED}]))
            .expect("messages");
    let answered: Vec<Message> = serde_json::from_value(serde_json::json!([
        {"role": "user", "content": ASKED},
        {"role": "assistant", "content": null, "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "17C, light rain."},
    ]))
    .expect("messages");

    let tools = weather();
    let tools = tools.as_array().expect("a list");
    let ids = |messages: &[Message]| -> Vec<usize> {
        let prompt = chat::prompt(messages, tools).expect("a conversation this maps");
        tokenizer
            .encode(&prompt)
            .expect("a prompt this vocabulary spells")
            .into_iter()
            .map(|id| id as usize)
            .collect()
    };
    let (first, second) = (ids(&asked), ids(&answered));

    // What the server keeps after the first turn: the prompt bar its last
    // token, which the generation feeds rather than the prefill.
    let mut kept = Kept::new(&config.text_config, DEFAULT_BOUND);
    kept.keep(&first[..first.len() - 1]);
    assert_eq!(
        kept.matching(&second),
        first.len() - 1,
        "the round trip re-prefills the turn it answers"
    );
}
