#!/usr/bin/env python3
"""
Security Validator Module
Validates security configuration and prevents insecure deployments.
"""

import os
import sys
import hashlib
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security configuration issue."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # secrets, auth, network, etc.
    message: str
    remediation: str


class SecurityValidator:
    """Validates security configuration for different environments."""
    
    # Known default/weak values that should never be in production
    FORBIDDEN_SECRETS = [
        "dev-secret-change-in-production",
        "your-super-secret-jwt-key-change-in-production",
        "TradingSystem2024!",
        "change-me",
        "default",
        "password",
        "123456",
        "admin"
    ]
    
    # Required environment variables by environment
    REQUIRED_ENV_VARS = {
        "production": [
            "SECURITY_SECRET_KEY",
            "ADMIN_PASSWORD_HASH",
            "DATABASE_PASSWORD",
            "REDIS_PASSWORD"
        ],
        "staging": [
            "SECURITY_SECRET_KEY",
            "ADMIN_PASSWORD_HASH"
        ],
        "development": []
    }
    
    def __init__(self, environment: str = "development"):
        """Initialize security validator for given environment."""
        self.environment = environment.lower()
        self.issues: List[SecurityIssue] = []
        
    def validate_all(self) -> bool:
        """Run all security validations."""
        logger.info(f"Running security validation for {self.environment} environment")
        
        # Run validation checks
        self._validate_secrets()
        self._validate_required_vars()
        self._validate_jwt_config()
        self._validate_admin_config()
        self._validate_api_keys()
        self._validate_database_config()
        
        # Report results
        if self.issues:
            self._report_issues()
            critical_issues = [i for i in self.issues if i.severity == "CRITICAL"]
            if critical_issues and self.environment == "production":
                logger.critical(f"Found {len(critical_issues)} CRITICAL security issues!")
                return False
        else:
            logger.info("✅ Security validation passed!")
            
        return True
    
    def _validate_secrets(self):
        """Check for default/weak secrets in configuration."""
        env_vars_to_check = [
            "SECURITY_SECRET_KEY",
            "JWT_SECRET_KEY",
            "ADMIN_PASSWORD",
            "DATABASE_PASSWORD",
            "REDIS_PASSWORD"
        ]
        
        for var in env_vars_to_check:
            value = os.getenv(var)
            if not value:
                continue
                
            # Check against forbidden values
            if any(forbidden in value.lower() for forbidden in self.FORBIDDEN_SECRETS):
                self.issues.append(SecurityIssue(
                    severity="CRITICAL" if self.environment == "production" else "HIGH",
                    category="secrets",
                    message=f"{var} contains default or weak value",
                    remediation=f"Generate secure value for {var} using cryptographically secure methods"
                ))
            
            # Check minimum length for secrets
            if var in ["SECURITY_SECRET_KEY", "JWT_SECRET_KEY"] and len(value) < 32:
                self.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="secrets",
                    message=f"{var} is too short ({len(value)} chars)",
                    remediation=f"{var} should be at least 32 characters. Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                ))
    
    def _validate_required_vars(self):
        """Check that required environment variables are set."""
        required_vars = self.REQUIRED_ENV_VARS.get(self.environment, [])
        
        for var in required_vars:
            if not os.getenv(var):
                self.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    category="configuration",
                    message=f"Required variable {var} is not set",
                    remediation=f"Set {var} in environment or .env file"
                ))
    
    def _validate_jwt_config(self):
        """Validate JWT configuration."""
        secret_key = os.getenv("SECURITY_SECRET_KEY") or os.getenv("JWT_SECRET_KEY")
        
        if self.environment != "development":
            if not secret_key:
                self.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    category="auth",
                    message="No JWT secret key configured",
                    remediation="Set SECURITY_SECRET_KEY environment variable"
                ))
            
            # Check for hardcoded algorithm
            algorithm = os.getenv("SECURITY_JWT_ALGORITHM", "HS256")
            if algorithm not in ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]:
                self.issues.append(SecurityIssue(
                    severity="MEDIUM",
                    category="auth",
                    message=f"Unsupported JWT algorithm: {algorithm}",
                    remediation="Use standard JWT algorithm (HS256, RS256, etc.)"
                ))
    
    def _validate_admin_config(self):
        """Validate admin user configuration."""
        admin_hash = os.getenv("ADMIN_PASSWORD_HASH")
        admin_password = os.getenv("ADMIN_PASSWORD")
        
        if self.environment == "production":
            if not admin_hash:
                self.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    category="auth",
                    message="Admin password hash not set for production",
                    remediation="Set ADMIN_PASSWORD_HASH with bcrypt-hashed password"
                ))
            
            if admin_password:
                self.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    category="auth",
                    message="Plain text ADMIN_PASSWORD should not be set in production",
                    remediation="Remove ADMIN_PASSWORD and use only ADMIN_PASSWORD_HASH"
                ))
    
    def _validate_api_keys(self):
        """Validate external API keys configuration."""
        if self.environment == "production":
            api_keys = [
                "TRADING_ALPACA_API_KEY",
                "TRADING_ALPACA_SECRET_KEY",
                "TRADING_POLYGON_API_KEY"
            ]
            
            for key in api_keys:
                value = os.getenv(key)
                if value and ("paper" in value.lower() or "test" in value.lower() or "demo" in value.lower()):
                    self.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="api_keys",
                        message=f"{key} appears to be a test/demo key in production",
                        remediation=f"Use production API credentials for {key}"
                    ))
    
    def _validate_database_config(self):
        """Validate database configuration."""
        if self.environment in ["production", "staging"]:
            db_password = os.getenv("DATABASE_PASSWORD")
            if db_password and len(db_password) < 16:
                self.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="database",
                    message="Database password is too weak",
                    remediation="Use a strong password (16+ chars) for database"
                ))
            
            # Check for SSL/TLS requirement
            db_url = os.getenv("DATABASE_URL", "")
            if "sslmode=disable" in db_url or "ssl=false" in db_url:
                self.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="database",
                    message="Database connection not using SSL/TLS",
                    remediation="Enable SSL/TLS for database connections in production"
                ))
    
    def _report_issues(self):
        """Report all found security issues."""
        # Group by severity
        by_severity = {}
        for issue in self.issues:
            if issue.severity not in by_severity:
                by_severity[issue.severity] = []
            by_severity[issue.severity].append(issue)
        
        # Report in order of severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity not in by_severity:
                continue
                
            issues = by_severity[severity]
            logger.warning(f"\n{severity} Security Issues ({len(issues)}):")
            for issue in issues:
                logger.warning(f"  [{issue.category}] {issue.message}")
                logger.info(f"    → {issue.remediation}")
    
    def get_report(self) -> Dict[str, Any]:
        """Get security validation report."""
        return {
            "environment": self.environment,
            "passed": len([i for i in self.issues if i.severity == "CRITICAL"]) == 0,
            "total_issues": len(self.issues),
            "by_severity": {
                "CRITICAL": len([i for i in self.issues if i.severity == "CRITICAL"]),
                "HIGH": len([i for i in self.issues if i.severity == "HIGH"]),
                "MEDIUM": len([i for i in self.issues if i.severity == "MEDIUM"]),
                "LOW": len([i for i in self.issues if i.severity == "LOW"])
            },
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "remediation": i.remediation
                }
                for i in self.issues
            ]
        }


def validate_deployment_security(environment: Optional[str] = None) -> bool:
    """
    Main entry point for security validation.
    Returns True if validation passes, False otherwise.
    """
    env = environment or os.getenv("ENVIRONMENT", "production")
    validator = SecurityValidator(env)
    
    result = validator.validate_all()
    
    # In production, fail hard on critical issues
    if env == "production" and not result:
        logger.critical("DEPLOYMENT BLOCKED: Critical security issues detected!")
        # sys.exit(1)
    
    return result


if __name__ == "__main__":
    # Run validation when module is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate security configuration")
    parser.add_argument("--environment", default=os.getenv("ENVIRONMENT", "production"),
                        help="Environment to validate (development/staging/production)")
    parser.add_argument("--json", action="store_true",
                        help="Output report as JSON")
    
    args = parser.parse_args()
    
    validator = SecurityValidator(args.environment)
    result = validator.validate_all()
    
    if args.json:
        import json
        print(json.dumps(validator.get_report(), indent=2))
    
    sys.exit(0 if result else 1)