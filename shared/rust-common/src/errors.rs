//! Error types and utilities for the trading system

use thiserror::Error;

/// Result type alias for trading operations
pub type Result<T> = std::result::Result<T, TradingError>;

/// Main error type for the trading system
#[derive(Error, Debug)]
pub enum TradingError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { 
        /// Error message
        message: String,
        /// Configuration key that caused the error
        key: Option<String>,
    },

    /// Database operation errors
    #[error("Database error: {message}")]
    Database {
        /// Error message
        message: String,
        /// Database operation that failed
        operation: Option<String>,
        /// Table or collection name
        table: Option<String>,
    },

    /// Redis cache errors
    #[error("Redis error: {message}")]
    Redis {
        /// Error message
        message: String,
        /// Redis command that failed
        command: Option<String>,
    },

    /// Network/HTTP errors
    #[error("Network error: {message}")]
    Network {
        /// Error message
        message: String,
        /// HTTP status code if applicable
        status_code: Option<u16>,
        /// URL that caused the error
        url: Option<String>,
    },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization {
        /// Error message
        message: String,
        /// Data type being serialized
        data_type: Option<String>,
    },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation {
        /// Error message
        message: String,
        /// Field that failed validation
        field: Option<String>,
        /// Invalid value
        value: Option<String>,
    },

    /// Authentication errors
    #[error("Authentication error: {message}")]
    Authentication {
        /// Error message
        message: String,
    },

    /// Authorization errors
    #[error("Authorization error: {message}")]
    Authorization {
        /// Error message
        message: String,
        /// Required permission
        required_permission: Option<String>,
    },

    /// Trading-specific errors
    #[error("Trading error: {message}")]
    Trading {
        /// Error message
        message: String,
        /// Symbol involved in the error
        symbol: Option<String>,
        /// Order ID if applicable
        order_id: Option<String>,
    },

    /// Risk management errors
    #[error("Risk management error: {message}")]
    RiskManagement {
        /// Error message
        message: String,
        /// Type of risk violation
        risk_type: Option<String>,
        /// Current value that exceeded limit
        current_value: Option<f64>,
        /// Maximum allowed value
        limit_value: Option<f64>,
    },

    /// Market data errors
    #[error("Market data error: {message}")]
    MarketData {
        /// Error message
        message: String,
        /// Data provider
        provider: Option<String>,
        /// Symbol
        symbol: Option<String>,
        /// Data type (e.g., "quote", "trade", "options_chain")
        data_type: Option<String>,
    },

    /// AI/ML model errors
    #[error("AI model error: {message}")]
    AiModel {
        /// Error message
        message: String,
        /// Model name
        model_name: Option<String>,
        /// Model operation
        operation: Option<String>,
    },

    /// Circuit breaker errors
    #[error("Circuit breaker active: {service}")]
    CircuitBreaker {
        /// Service with active circuit breaker
        service: String,
        /// Number of consecutive failures
        failure_count: usize,
    },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        /// Error message
        message: String,
        /// Service being rate limited
        service: Option<String>,
        /// Retry after duration in seconds
        retry_after: Option<u64>,
    },

    /// Timeout errors
    #[error("Operation timed out: {message}")]
    Timeout {
        /// Error message
        message: String,
        /// Operation that timed out
        operation: Option<String>,
        /// Timeout duration in seconds
        timeout_seconds: Option<u64>,
    },

    /// Internal server errors
    #[error("Internal error: {message}")]
    Internal {
        /// Error message
        message: String,
        /// Error code for debugging
        error_code: Option<String>,
    },

    /// External service errors
    #[error("External service error: {service} - {message}")]
    ExternalService {
        /// Service name
        service: String,
        /// Error message
        message: String,
        /// HTTP status code if applicable
        status_code: Option<u16>,
    },

    /// Generic IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// HTTP client errors
    #[error("HTTP error: {0}")]
    Http(#[from] hyper::Error),
}

impl TradingError {
    /// Create a new configuration error
    pub fn config<S: Into<String>>(message: S, key: Option<String>) -> Self {
        Self::Config {
            message: message.into(),
            key,
        }
    }

    /// Create a new database error
    pub fn database<S: Into<String>>(
        message: S,
        operation: Option<String>,
        table: Option<String>,
    ) -> Self {
        Self::Database {
            message: message.into(),
            operation,
            table,
        }
    }

    /// Create a new validation error
    pub fn validation<S: Into<String>>(
        message: S,
        field: Option<String>,
        value: Option<String>,
    ) -> Self {
        Self::Validation {
            message: message.into(),
            field,
            value,
        }
    }

    /// Create a new trading error
    pub fn trading<S: Into<String>>(
        message: S,
        symbol: Option<String>,
        order_id: Option<String>,
    ) -> Self {
        Self::Trading {
            message: message.into(),
            symbol,
            order_id,
        }
    }

    /// Create a new risk management error
    pub fn risk_management<S: Into<String>>(
        message: S,
        risk_type: Option<String>,
        current_value: Option<f64>,
        limit_value: Option<f64>,
    ) -> Self {
        Self::RiskManagement {
            message: message.into(),
            risk_type,
            current_value,
            limit_value,
        }
    }

    /// Create a new circuit breaker error
    pub fn circuit_breaker<S: Into<String>>(service: S, failure_count: usize) -> Self {
        Self::CircuitBreaker {
            service: service.into(),
            failure_count,
        }
    }

    /// Create a new timeout error
    pub fn timeout<S: Into<String>>(
        message: S,
        operation: Option<String>,
        timeout_seconds: Option<u64>,
    ) -> Self {
        Self::Timeout {
            message: message.into(),
            operation,
            timeout_seconds,
        }
    }

    /// Get error category for metrics and logging
    pub fn category(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config",
            Self::Database { .. } => "database",
            Self::Redis { .. } => "redis",
            Self::Network { .. } => "network",
            Self::Serialization { .. } => "serialization",
            Self::Validation { .. } => "validation",
            Self::Authentication { .. } => "authentication",
            Self::Authorization { .. } => "authorization",
            Self::Trading { .. } => "trading",
            Self::RiskManagement { .. } => "risk_management",
            Self::MarketData { .. } => "market_data",
            Self::AiModel { .. } => "ai_model",
            Self::CircuitBreaker { .. } => "circuit_breaker",
            Self::RateLimit { .. } => "rate_limit",
            Self::Timeout { .. } => "timeout",
            Self::Internal { .. } => "internal",
            Self::ExternalService { .. } => "external_service",
            Self::Io(_) => "io",
            Self::Json(_) => "json",
            Self::Http(_) => "http",
        }
    }

    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network { .. }
                | Self::Timeout { .. }
                | Self::ExternalService { .. }
                | Self::RateLimit { .. }
                | Self::Io(_)
                | Self::Http(_)
        )
    }

    /// Get retry delay in seconds for retryable errors
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimit { retry_after, .. } => *retry_after,
            Self::Network { .. } => Some(5),
            Self::Timeout { .. } => Some(10),
            Self::ExternalService { .. } => Some(30),
            _ if self.is_retryable() => Some(5),
            _ => None,
        }
    }
}

/// Result type for configuration operations
pub type ConfigResult<T> = std::result::Result<T, TradingError>;
/// Result type for database operations
pub type DatabaseResult<T> = std::result::Result<T, TradingError>;
/// Result type for trading operations  
pub type TradingResult<T> = std::result::Result<T, TradingError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categories() {
        let config_err = TradingError::config("test", None);
        assert_eq!(config_err.category(), "config");

        let trading_err = TradingError::trading("test", None, None);
        assert_eq!(trading_err.category(), "trading");
    }

    #[test]
    fn test_retryable_errors() {
        let network_err = TradingError::Network {
            message: "connection failed".to_string(),
            status_code: None,
            url: None,
        };
        assert!(network_err.is_retryable());

        let validation_err = TradingError::validation("invalid field", None, None);
        assert!(!validation_err.is_retryable());
    }

    #[test]
    fn test_error_display() {
        let err = TradingError::trading("Order failed", Some("AAPL".to_string()), None);
        let error_string = format!("{}", err);
        assert!(error_string.contains("Trading error"));
        assert!(error_string.contains("Order failed"));
    }
}