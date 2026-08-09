use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;

use crate::api::{self, Inference};

pub async fn run(address: SocketAddr, inference: Arc<dyn Inference>) -> std::io::Result<()> {
    let listener = TcpListener::bind(address).await?;
    let address = listener.local_addr()?;

    eprintln!("listening on http://{address}");

    axum::serve(listener, api::router(inference))
        .with_graceful_shutdown(shutdown_signal())
        .await
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        eprintln!("failed to listen for a shutdown signal: {error}");
    }
}
