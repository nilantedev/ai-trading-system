//! Trading Common Library
//!
//! Shared Rust utilities and libraries for the AI trading system.
//! Provides high-performance, type-safe building blocks for trading applications.

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod logging;
pub mod types;
pub mod errors;
pub mod utils;

#[cfg(feature = "redis")]
pub mod redis;

#[cfg(feature = "metrics")]
pub mod metrics;

pub mod database;
pub mod security;

// Re-export commonly used types
pub use chrono::{DateTime, Utc};
pub use rust_decimal::Decimal;
pub use serde::{Deserialize, Serialize};
pub use uuid::Uuid;

pub use crate::errors::{Result, TradingError};
pub use crate::types::*;

/// Version information for the trading common library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: BuildInfo = BuildInfo {
    version: VERSION,
    git_hash: match option_env!("GIT_HASH") {
        Some(hash) => hash,
        None => "unknown",
    },
    build_timestamp: match option_env!("BUILD_TIMESTAMP") {
        Some(timestamp) => timestamp,
        None => "unknown",
    },
    rust_version: env!("CARGO_PKG_RUST_VERSION"),
};

/// Build information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Library version
    pub version: &'static str,
    /// Git commit hash
    pub git_hash: &'static str,
    /// Build timestamp
    pub build_timestamp: &'static str,
    /// Rust version used to build
    pub rust_version: &'static str,
}

/// Initialize the trading common library
///
/// This function should be called once at application startup to initialize
/// logging, metrics, and other shared resources.
///
/// # Arguments
///
/// * `service_name` - Name of the service using this library
/// * `environment` - Environment (e.g., "development", "production")
///
/// # Returns
///
/// Returns `Ok(())` on successful initialization, or a `TradingError` on failure.
///
/// # Examples
///
/// ```rust
/// use trading_common::init;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     init("my-trading-service", "production").await?;
///     // Your application code here
///     Ok(())
/// }
/// ```
pub async fn init(service_name: &str, environment: &str) -> Result<()> {
    // Initialize logging
    logging::init(service_name, environment)?;

    // Initialize metrics if enabled
    #[cfg(feature = "metrics")]
    metrics::init(service_name).await?;

    tracing::info!(
        service = service_name,
        environment = environment,
        version = VERSION,
        "Trading common library initialized"
    );

    Ok(())
}

/// Health check information for the library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Overall health status
    pub status: HealthStatus,
    /// Version information
    pub version: &'static str,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Number of active connections (if applicable)
    pub active_connections: usize,
    /// Memory usage information
    pub memory_usage: MemoryUsage,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Service is healthy
    Healthy,
    /// Service has warnings but is functional
    Warning,
    /// Service is unhealthy
    Unhealthy,
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Allocated memory in bytes
    pub allocated: usize,
    /// Total system memory in bytes
    pub total: usize,
    /// Memory usage percentage
    pub usage_percent: f64,
}

/// Get health check information for the library
///
/// # Returns
///
/// Returns a `HealthCheck` struct with current status information.
pub fn health_check() -> HealthCheck {
    let uptime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Get memory usage (simplified)
    let memory_usage = MemoryUsage {
        allocated: 0,     // Would be populated with actual memory stats
        total: 0,         // Would be populated with system memory
        usage_percent: 0.0,
    };

    HealthCheck {
        status: HealthStatus::Healthy,
        version: VERSION,
        uptime_seconds: uptime,
        active_connections: 0, // Would be populated with actual connection count
        memory_usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_info() {
        assert_eq!(BUILD_INFO.version, VERSION);
        assert!(!BUILD_INFO.version.is_empty());
    }

    #[test]
    fn test_health_check() {
        let health = health_check();
        assert!(matches!(health.status, HealthStatus::Healthy));
        assert_eq!(health.version, VERSION);
    }

    #[tokio::test]
    async fn test_init() {
        let result = init("test-service", "test").await;
        assert!(result.is_ok());
    }
}