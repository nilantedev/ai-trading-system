//! Security utilities and authentication

use crate::errors::{Result, TradingError};

/// Hash a password using Argon2
pub fn hash_password(password: &str) -> Result<String> {
    let salt = argon2::password_hash::SaltString::generate(&mut argon2::password_hash::rand_core::OsRng);
    let argon2 = argon2::Argon2::default();
    
    argon2::PasswordHasher::hash_password(&argon2, password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|e| TradingError::Internal {
            message: format!("Password hashing failed: {}", e),
            error_code: Some("HASH_ERROR".to_string()),
        })
}

/// Verify a password against a hash
pub fn verify_password(password: &str, hash: &str) -> Result<bool> {
    let parsed_hash = argon2::PasswordHash::new(hash)
        .map_err(|e| TradingError::Internal {
            message: format!("Invalid password hash: {}", e),
            error_code: Some("HASH_PARSE_ERROR".to_string()),
        })?;

    let argon2 = argon2::Argon2::default();
    
    match argon2::PasswordVerifier::verify_password(&argon2, password.as_bytes(), &parsed_hash) {
        Ok(()) => Ok(true),
        Err(argon2::password_hash::Error::Password) => Ok(false),
        Err(e) => Err(TradingError::Internal {
            message: format!("Password verification failed: {}", e),
            error_code: Some("VERIFY_ERROR".to_string()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_hashing() {
        let password = "test_password_123";
        let hash = hash_password(password).unwrap();
        
        assert!(verify_password(password, &hash).unwrap());
        assert!(!verify_password("wrong_password", &hash).unwrap());
    }
}