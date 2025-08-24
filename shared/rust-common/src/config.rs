//! Configuration management for Rust services

use serde::{Deserialize, Serialize};
use std::env;

use crate::errors::{Result, TradingError};

/// Configuration structure for Rust services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Service configuration
    pub service: ServiceConfig,
    /// Database configuration
    pub database: DatabaseConfig,
    /// Trading configuration
    pub trading: TradingConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Service-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Service name
    pub name: String,
    /// Service version
    pub version: String,
    /// Host to bind to
    pub host: String,
    /// Port to bind to
    pub port: u16,
    /// Environment (development, staging, production)
    pub environment: String,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Redis URL
    pub redis_url: String,
    /// PostgreSQL URL
    pub postgres_url: Option<String>,
    /// QuestDB configuration
    pub questdb: QuestDbConfig,
}

/// QuestDB configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestDbConfig {
    /// QuestDB host
    pub host: String,
    /// QuestDB HTTP port
    pub port: u16,
    /// QuestDB username
    pub username: Option<String>,
    /// QuestDB password
    pub password: Option<String>,
}

/// Trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Paper trading mode
    pub paper_trading: bool,
    /// Maximum position size
    pub max_position_size: f64,
    /// Risk limit percentage
    pub risk_limit_percent: f64,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    /// JSON formatting
    pub json: bool,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let service = ServiceConfig {
            name: env::var("SERVICE_NAME").unwrap_or_else(|_| "trading-service".to_string()),
            version: env::var("SERVICE_VERSION")
                .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string()),
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("PORT")
                .unwrap_or_else(|_| "8000".to_string())
                .parse()
                .map_err(|e| TradingError::config(format!("Invalid port: {}", e), Some("PORT".to_string())))?,
            environment: env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()),
        };

        let database = DatabaseConfig {
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379/0".to_string()),
            postgres_url: env::var("POSTGRES_URL").ok(),
            questdb: QuestDbConfig {
                host: env::var("QUESTDB_HOST").unwrap_or_else(|_| "localhost".to_string()),
                port: env::var("QUESTDB_PORT")
                    .unwrap_or_else(|_| "9000".to_string())
                    .parse()
                    .map_err(|e| {
                        TradingError::config(format!("Invalid QuestDB port: {}", e), Some("QUESTDB_PORT".to_string()))
                    })?,
                username: env::var("QUESTDB_USERNAME").ok(),
                password: env::var("QUESTDB_PASSWORD").ok(),
            },
        };

        let trading = TradingConfig {
            paper_trading: env::var("PAPER_TRADING")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            max_position_size: env::var("MAX_POSITION_SIZE")
                .unwrap_or_else(|_| "10000.0".to_string())
                .parse()
                .unwrap_or(10000.0),
            risk_limit_percent: env::var("RISK_LIMIT_PERCENT")
                .unwrap_or_else(|_| "2.0".to_string())
                .parse()
                .unwrap_or(2.0),
        };

        let logging = LoggingConfig {
            level: env::var("LOG_LEVEL").unwrap_or_else(|_| "INFO".to_string()),
            json: env::var("LOG_JSON")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
        };

        Ok(Config {
            service,
            database,
            trading,
            logging,
        })
    }

    /// Check if running in production environment
    pub fn is_production(&self) -> bool {
        self.service.environment.to_lowercase() == "production"
    }

    /// Check if running in development environment
    pub fn is_development(&self) -> bool {
        self.service.environment.to_lowercase() == "development"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_env() {
        // Set test environment variables
        env::set_var("SERVICE_NAME", "test-service");
        env::set_var("PORT", "3000");
        env::set_var("ENVIRONMENT", "test");

        let config = Config::from_env().unwrap();
        assert_eq!(config.service.name, "test-service");
        assert_eq!(config.service.port, 3000);
        assert_eq!(config.service.environment, "test");

        // Clean up
        env::remove_var("SERVICE_NAME");
        env::remove_var("PORT");
        env::remove_var("ENVIRONMENT");
    }
}