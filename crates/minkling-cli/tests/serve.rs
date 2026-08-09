use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const CHECKPOINT: &str = "INKLINGRS_CHECKPOINT";

struct Server {
    child: Child,
    address: String,
}

impl Server {
    fn start(checkpoint: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_minkling"))
            .args([
                "serve",
                checkpoint,
                "--address",
                "127.0.0.1:0",
                "--max-tokens",
                "64",
                "--numerics",
                "production",
                "--reuse-tokens",
                "0",
            ])
            .stderr(Stdio::piped())
            .spawn()
            .expect("minkling should start");

        let mut log = BufReader::new(child.stderr.take().expect("stderr should be piped"));
        let mut line = String::new();
        let address = loop {
            line.clear();
            assert_ne!(
                log.read_line(&mut line).expect("startup should be logged"),
                0,
                "minkling stopped before listening"
            );
            eprint!("{line}");
            if let Some((_, address)) = line.trim_end().split_once("listening on http://") {
                break address.to_string();
            }
        };

        std::thread::spawn(move || {
            let mut rest = String::new();
            let _ = log.read_to_string(&mut rest);
            eprint!("{rest}");
        });

        Self { child, address }
    }

    fn request(&self, method: &str, path: &str, body: &str) -> String {
        self.request_with_timeout(method, path, body, Duration::from_secs(10))
    }

    fn request_with_timeout(
        &self,
        method: &str,
        path: &str,
        body: &str,
        timeout: Duration,
    ) -> String {
        let mut socket = TcpStream::connect(&self.address).expect("minkling should be listening");
        socket
            .set_read_timeout(Some(timeout))
            .expect("the response timeout should be set");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            self.address,
            body.len()
        );
        socket
            .write_all(request.as_bytes())
            .expect("the request should be written");

        let mut response = String::new();
        socket
            .read_to_string(&mut response)
            .expect("the response should be read");
        response
    }

    fn disconnect_stream(&self, body: &str) {
        let mut socket = TcpStream::connect(&self.address).expect("minkling should be listening");
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("the response timeout should be set");
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            self.address,
            body.len()
        );
        socket
            .write_all(request.as_bytes())
            .expect("the streaming request should be written");

        let mut response = BufReader::new(socket);
        let mut line = String::new();
        loop {
            line.clear();
            response
                .read_line(&mut line)
                .expect("the response head should be read");
            if line == "\r\n" {
                break;
            }
        }

        line.clear();
        response
            .read_line(&mut line)
            .expect("the first chunk size should be read");
        let size = usize::from_str_radix(line.trim(), 16).expect("the chunk size should be hex");
        let mut frame = vec![0_u8; size];
        response
            .read_exact(&mut frame)
            .expect("the opening frame should be read");
        assert!(frame.starts_with(b"data: {"), "{frame:?}");
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn json_body(response: &str) -> serde_json::Value {
    let (head, body) = response
        .split_once("\r\n\r\n")
        .expect("the response should have a head and body");
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    serde_json::from_str(body).expect("the response should be JSON")
}

fn event_body(response: &str) -> String {
    let (head, body) = response
        .split_once("\r\n\r\n")
        .expect("the response should have a head and body");
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(head.contains("content-type: text/event-stream"), "{head}");
    dechunk(body)
}

fn dechunk(mut body: &str) -> String {
    let mut decoded = String::new();
    loop {
        let (size, rest) = body
            .split_once("\r\n")
            .expect("a chunk should start with its size");
        let size = usize::from_str_radix(size, 16).expect("the chunk size should be hex");
        if size == 0 {
            return decoded;
        }
        let (chunk, rest) = rest.split_at(size);
        decoded.push_str(chunk);
        body = rest
            .strip_prefix("\r\n")
            .expect("a chunk should end with a line break");
    }
}

#[test]
fn a_chat_completion_crosses_the_reviewed_host_boundary() {
    let Some(checkpoint) = std::env::var_os(CHECKPOINT) else {
        eprintln!("skipping: {CHECKPOINT} is unset");
        return;
    };
    let checkpoint = PathBuf::from(checkpoint);
    let checkpoint = match checkpoint.is_relative() {
        true => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(checkpoint),
        false => checkpoint,
    };
    let checkpoint = std::fs::canonicalize(checkpoint).expect("the checkpoint should exist");
    let server = Server::start(&checkpoint.to_string_lossy());

    let models = json_body(&server.request("GET", "/v1/models", ""));
    let model = models["data"][0]["id"]
        .as_str()
        .expect("the model should be named");

    let asked = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 2,
    });
    let completion = json_body(&server.request("POST", "/v1/chat/completions", &asked.to_string()));

    assert_eq!(completion["object"], "chat.completion");
    assert_eq!(completion["model"], model);
    assert_eq!(completion["choices"][0]["finish_reason"], "length");
    assert_eq!(completion["usage"]["completion_tokens"], 2);
    let message = &completion["choices"][0]["message"];
    assert_eq!(message["role"], "assistant");
    assert!(
        !message.to_string().contains("<|"),
        "model markers reached the client: {message}"
    );

    let streamed = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 2,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let stream = event_body(&server.request("POST", "/v1/chat/completions", &streamed.to_string()));
    assert!(stream.starts_with("data: {"), "{stream:?}");
    assert!(stream.ends_with("data: [DONE]\n\n"), "{stream:?}");
    let payloads: Vec<serde_json::Value> = stream
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str(payload).expect("a stream payload should be JSON"))
        .collect();
    assert_eq!(payloads[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        payloads[payloads.len() - 2]["choices"][0]["finish_reason"],
        "length"
    );
    assert_eq!(
        payloads.last().expect("a usage payload")["usage"]["completion_tokens"],
        2
    );

    let invalid_stream = serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 65,
        "stream": true,
    });
    let refused = server.request("POST", "/v1/chat/completions", &invalid_stream.to_string());
    assert!(refused.starts_with("HTTP/1.1 400"), "{refused}");
    assert!(
        refused.contains("max_tokens must be at most 64"),
        "{refused}"
    );

    let abandoned = serde_json::json!({
        "messages": [{"role": "user", "content": "Count slowly."}],
        "max_tokens": 64,
        "stream": true,
    });
    server.disconnect_stream(&abandoned.to_string());

    let started = Instant::now();
    let follow_up = json_body(&server.request_with_timeout(
        "POST",
        "/v1/chat/completions",
        &asked.to_string(),
        Duration::from_millis(2_500),
    ));
    assert_eq!(follow_up["usage"]["completion_tokens"], 2);
    assert!(
        started.elapsed() < Duration::from_millis(2_500),
        "the disconnected stream kept the worker for {:?}",
        started.elapsed()
    );
}
