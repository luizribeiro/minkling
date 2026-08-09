use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

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
                "2",
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
        let mut socket = TcpStream::connect(&self.address).expect("minkling should be listening");
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
}
