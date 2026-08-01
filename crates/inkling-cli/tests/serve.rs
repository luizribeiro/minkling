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

use inkling_cli::wire::{dechunked, delta, payloads};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

/// How many tokens the end-to-end case decodes.
///
/// Two, against a prefill of the twenty-odd tokens a templated `Hi` makes: about
/// seventy-five seconds. The reply is not a sentence at that budget and is not
/// meant to be — what it settles is that the turn structure reached the model and
/// that what came back is HTTP, and a longer one would only cost minutes to
/// settle the same thing.
const GENERATED: usize = 2;

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
