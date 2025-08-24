//! Logging configuration and utilities

use crate::errors::Result;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize logging for the service
pub fn init(service_name: &str, environment: &str) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let formatting_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true);

    if environment == "production" {
        // JSON formatting for production
        tracing_subscriber::registry()
            .with(env_filter)
            .with(formatting_layer.json())
            .init();
    } else {
        // Pretty formatting for development
        tracing_subscriber::registry()
            .with(env_filter)
            .with(formatting_layer.pretty())
            .init();
    }

    tracing::info!(
        service = service_name,
        environment = environment,
        "Logging initialized"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_logging() {
        let result = init("test-service", "development");
        assert!(result.is_ok());
    }
}