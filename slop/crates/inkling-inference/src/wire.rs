//! Reading a response the way a client does for tests.
//!
//! The server writes its own framing — see [`crate::serve`] for why — and a test
//! that asserts on the frames has to take that framing apart first. Both the
//! unit tests inside [`crate::openai`] and the checkpoint-gated test in
//! `tests/serve.rs` need the same taking-apart, and the second links this crate
//! from outside and cannot see a `cfg(test)` module. A feature that every test
//! target turns on and nothing else does keeps one copy of it, the way
//! `inkling-core`'s `fixture` module does for the reference bundles.
//!
//! Everything here checks as it reads. A body this returns from is one a client
//! could have read; a body it panics on is one that would have left a client
//! waiting or reading the framing as though it were the message.

/// A chunked body put back together.
///
/// The size lines are hexadecimal and count *bytes*, which is what makes this
/// worth reading rather than assuming: a reply in any language but English has
/// more bytes than characters, and a client that confused the two would read
/// every frame after the first one out of step.
pub fn dechunked(body: &str) -> String {
    let mut rest = body;
    let mut text = String::new();
    loop {
        let (size, tail) = rest.split_once("\r\n").expect("a chunk size line");
        let size = usize::from_str_radix(size, 16).expect("a hexadecimal chunk size");
        if size == 0 {
            assert_eq!(tail, "\r\n", "the last chunk is not terminated");
            return text;
        }
        text.push_str(&tail[..size]);
        rest = tail[size..]
            .strip_prefix("\r\n")
            .expect("a chunk terminator");
    }
}

/// Every frame of an event stream, the blank line that ends each one kept.
///
/// A stream that does not end on one ends mid-frame, and a client reading it
/// would be waiting for the rest.
pub fn frames(stream: &str) -> Vec<String> {
    assert!(stream.ends_with("\n\n"), "{stream:?} ends mid-frame");
    stream.split_inclusive("\n\n").map(str::to_string).collect()
}

/// The payload of one frame, or `None` for the terminator.
pub fn payload(frame: &str) -> Option<serde_json::Value> {
    let data = frame
        .strip_prefix("data: ")
        .and_then(|frame| frame.strip_suffix("\n\n"))
        .unwrap_or_else(|| panic!("{frame:?} is not a frame"));
    match data {
        "[DONE]" => None,
        json => Some(serde_json::from_str(json).expect("a frame carries json")),
    }
}

/// Every chunk of an event stream, the terminator dropped — which is what a
/// client that builds a message out of the deltas iterates over.
pub fn payloads(stream: &str) -> Vec<serde_json::Value> {
    let frames = frames(stream);
    let (last, chunks) = frames.split_last().expect("a stream has frames");
    assert_eq!(payload(last), None, "{stream:?} is not terminated");
    chunks
        .iter()
        .map(|frame| payload(frame))
        .collect::<Option<_>>()
        .expect("only the last frame terminates")
}

/// The delta of a chunk, as the field it landed in and the text it carried.
///
/// `tool_calls` is not one of the fields: a call is not text, and a client that
/// appended it to a message would print JSON at a user. [`calls`] is where one
/// is read off instead.
pub fn delta(chunk: &serde_json::Value) -> Option<(String, String)> {
    let delta = &chunk["choices"][0]["delta"];
    for field in ["content", "reasoning_content"] {
        if let Some(text) = delta[field].as_str() {
            return Some((field.to_string(), text.to_string()));
        }
    }
    None
}

/// The calls a chunk carried, which is what a client builds `tool_calls` out
/// of. Empty for every chunk that is not one.
pub fn calls(chunk: &serde_json::Value) -> Vec<serde_json::Value> {
    chunk["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}
