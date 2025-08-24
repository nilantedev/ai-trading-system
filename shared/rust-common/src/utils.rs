//! Utility functions and helpers

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Calculate percentage change
pub fn percentage_change(old_value: Decimal, new_value: Decimal) -> Option<f64> {
    if old_value.is_zero() {
        return None;
    }
    
    let change = new_value - old_value;
    let percentage = (change / old_value) * Decimal::from(100);
    percentage.to_f64()
}

/// Round decimal to specified precision
pub fn round_decimal(value: Decimal, precision: u32) -> Decimal {
    value.round_dp(precision)
}

/// Get current timestamp
pub fn current_timestamp() -> DateTime<Utc> {
    Utc::now()
}

/// Parse timestamp from string
pub fn parse_timestamp(timestamp_str: &str) -> Result<DateTime<Utc>, chrono::ParseError> {
    DateTime::parse_from_rfc3339(timestamp_str)
        .map(|dt| dt.with_timezone(&Utc))
}

/// Convert HashMap to query string
pub fn map_to_query_string(params: &HashMap<String, String>) -> String {
    if params.is_empty() {
        return String::new();
    }

    params
        .iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect::<Vec<_>>()
        .join("&")
}

/// Simple rate limiter using timestamps
pub struct RateLimiter {
    requests: Vec<DateTime<Utc>>,
    max_requests: usize,
    window_seconds: u64,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(max_requests: usize, window_seconds: u64) -> Self {
        Self {
            requests: Vec::new(),
            max_requests,
            window_seconds,
        }
    }

    /// Check if a request is allowed
    pub fn is_allowed(&mut self) -> bool {
        let now = Utc::now();
        let window_start = now - chrono::Duration::seconds(self.window_seconds as i64);

        // Remove old requests
        self.requests.retain(|&timestamp| timestamp > window_start);

        if self.requests.len() >= self.max_requests {
            false
        } else {
            self.requests.push(now);
            true
        }
    }

    /// Get remaining requests in current window
    pub fn remaining_requests(&self) -> usize {
        self.max_requests.saturating_sub(self.requests.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_percentage_change() {
        let old = Decimal::from(100);
        let new = Decimal::from(110);
        let change = percentage_change(old, new).unwrap();
        assert!((change - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_round_decimal() {
        let value = Decimal::from_str("123.456789").unwrap();
        let rounded = round_decimal(value, 2);
        assert_eq!(rounded, Decimal::from_str("123.46").unwrap());
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(2, 60);
        
        assert!(limiter.is_allowed());
        assert!(limiter.is_allowed());
        assert!(!limiter.is_allowed()); // Should be blocked
        
        assert_eq!(limiter.remaining_requests(), 0);
    }

    #[test]
    fn test_map_to_query_string() {
        let mut params = HashMap::new();
        params.insert("key1".to_string(), "value1".to_string());
        params.insert("key2".to_string(), "value2".to_string());
        
        let query_string = map_to_query_string(&params);
        assert!(query_string.contains("key1=value1"));
        assert!(query_string.contains("key2=value2"));
    }
}