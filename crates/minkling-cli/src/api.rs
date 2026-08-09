use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::Value;

#[cfg(test)]
use std::future::ready;

const LARGEST_BODY: usize = 1 << 20;

pub type Completion<'a> = Pin<Box<dyn Future<Output = Result<Value, InferenceError>> + Send + 'a>>;

pub trait Inference: Send + Sync + 'static {
    fn model(&self) -> Option<&str>;

    fn complete(&self, request: Value) -> Completion<'_>;
}

#[cfg(test)]
pub struct Unavailable;

#[cfg(test)]
impl Inference for Unavailable {
    fn model(&self) -> Option<&str> {
        None
    }

    fn complete(&self, _request: Value) -> Completion<'_> {
        Box::pin(ready(Err(InferenceError::unavailable(
            "the inference engine is not connected yet",
        ))))
    }
}

#[derive(Debug)]
pub struct InferenceError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl InferenceError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            message: message.into(),
        }
    }
}

impl IntoResponse for InferenceError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorEnvelope::new(self.kind, self.message)),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ApiError,
}

impl ErrorEnvelope {
    fn new(kind: &'static str, message: String) -> Self {
        Self {
            error: ApiError { message, kind },
        }
    }
}

#[derive(Serialize)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
}

#[derive(Serialize)]
struct Models {
    object: &'static str,
    data: Vec<Model>,
}

#[derive(Serialize)]
struct Model {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

pub fn router(inference: Arc<dyn Inference>) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(complete))
        .layer(DefaultBodyLimit::max(LARGEST_BODY))
        .with_state(inference)
}

async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}

async fn models(State(inference): State<Arc<dyn Inference>>) -> Json<Models> {
    let data = inference
        .model()
        .map(|id| Model {
            id: id.to_string(),
            object: "model",
            created: now(),
            owned_by: "minkling",
        })
        .into_iter()
        .collect();

    Json(Models {
        object: "list",
        data,
    })
}

async fn complete(
    State(inference): State<Arc<dyn Inference>>,
    Json(request): Json<Value>,
) -> Result<Json<Value>, InferenceError> {
    inference.complete(request).await.map(Json)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::future::ready;

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use super::*;

    struct Fake;

    impl Inference for Fake {
        fn model(&self) -> Option<&str> {
            Some("Inkling-Small-mxfp4")
        }

        fn complete(&self, _request: Value) -> Completion<'_> {
            Box::pin(ready(Ok(serde_json::json!({
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            }))))
        }
    }

    async fn send(request: Request<Body>) -> (StatusCode, Value) {
        let response = router(Arc::new(Fake))
            .oneshot(request)
            .await
            .expect("the router should answer");
        let status = response.status();
        let body = to_bytes(response.into_body(), LARGEST_BODY)
            .await
            .expect("the response body should be readable");
        let body = serde_json::from_slice(&body).expect("the response should be JSON");
        (status, body)
    }

    #[tokio::test]
    async fn health_is_live() {
        let request = Request::get("/healthz")
            .body(Body::empty())
            .expect("the request should be valid");
        let (status, body) = send(request).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, serde_json::json!({"status": "ok"}));
    }

    #[tokio::test]
    async fn the_loaded_model_is_listed() {
        let request = Request::get("/v1/models")
            .body(Body::empty())
            .expect("the request should be valid");
        let (status, body) = send(request).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "Inkling-Small-mxfp4");
        assert_eq!(body["data"][0]["owned_by"], "minkling");
    }

    #[tokio::test]
    async fn a_chat_completion_reaches_inference() {
        let request = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "messages": [{"role": "user", "content": "Hi"}],
                })
                .to_string(),
            ))
            .expect("the request should be valid");
        let (status, body) = send(request).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["content"], "hello");
    }

    #[tokio::test]
    async fn a_disconnected_engine_is_a_service_error() {
        let request = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"messages":[]}"#))
            .expect("the request should be valid");
        let response = router(Arc::new(Unavailable))
            .oneshot(request)
            .await
            .expect("the router should answer");
        let status = response.status();
        let body = to_bytes(response.into_body(), LARGEST_BODY)
            .await
            .expect("the response body should be readable");
        let body: Value = serde_json::from_slice(&body).expect("the error response should be JSON");

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "server_error");
    }
}
