use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;

use crate::api::{self, Unavailable};

pub async fn run(address: SocketAddr) -> std::io::Result<()> {
    let listener = TcpListener::bind(address).await?;
    let address = listener.local_addr()?;

    eprintln!("listening on http://{address}");

    axum::serve(listener, api::router(Arc::new(Unavailable)))
        .with_graceful_shutdown(shutdown_signal())
        .await
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        eprintln!("failed to listen for a shutdown signal: {error}");
    }
}
