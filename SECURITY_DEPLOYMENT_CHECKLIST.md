# Security Deployment Checklist

## Pre-Deployment Security Validation

This checklist MUST be completed before deploying to production. All items marked as **[CRITICAL]** must pass.

### 1. Secrets Management

- [ ] **[CRITICAL]** Generate new `SECURITY_SECRET_KEY` for production
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
- [ ] **[CRITICAL]** Generate bcrypt hash for admin password
  ```bash
  python -c "from passlib.context import CryptContext; pwd_context = CryptContext(schemes=['bcrypt']); print(pwd_context.hash('YOUR_SECURE_PASSWORD'))"
  ```
- [ ] **[CRITICAL]** Set `ADMIN_PASSWORD_HASH` (never use `ADMIN_PASSWORD` in production)
- [ ] **[CRITICAL]** Verify no default passwords in environment variables
- [ ] **[CRITICAL]** Ensure all API keys are production keys (not paper/demo)
- [ ] Use secrets management service (AWS Secrets Manager, HashiCorp Vault, etc.)

### 2. Environment Configuration

- [ ] **[CRITICAL]** Set `ENVIRONMENT=production` 
- [ ] **[CRITICAL]** Set `DEBUG=false`
- [ ] **[CRITICAL]** Configure secure database passwords (16+ characters)
- [ ] **[CRITICAL]** Enable Redis password authentication
- [ ] Configure proper CORS origins (remove localhost in production)
- [ ] Set appropriate rate limiting values

### 3. Database Security

- [ ] **[CRITICAL]** Enable SSL/TLS for database connections
- [ ] **[CRITICAL]** Use strong database passwords
- [ ] Configure database connection pooling limits
- [ ] Enable database audit logging
- [ ] Set up automated backups with encryption

### 4. Network Security

- [ ] **[CRITICAL]** Use HTTPS only (SSL/TLS certificates configured)
- [ ] **[CRITICAL]** Configure firewall rules (allow only necessary ports)
- [ ] Enable DDoS protection (CloudFlare, AWS Shield, etc.)
- [ ] Configure reverse proxy (nginx/Apache) with security headers
- [ ] Implement IP whitelisting for admin endpoints if applicable

### 5. Authentication & Authorization

- [ ] **[CRITICAL]** Verify JWT token expiration is appropriate (30 min or less)
- [ ] **[CRITICAL]** Ensure no hardcoded credentials in code
- [ ] Implement token refresh mechanism
- [ ] Add user session management
- [ ] Configure account lockout after failed attempts

### 6. Monitoring & Logging

- [ ] **[CRITICAL]** Enable security audit logging
- [ ] Configure log aggregation (ELK stack, CloudWatch, etc.)
- [ ] Set up security alerts for:
  - [ ] Failed authentication attempts
  - [ ] Rate limit violations  
  - [ ] Unusual API usage patterns
  - [ ] Database connection failures
- [ ] Ensure no sensitive data in logs (passwords, tokens, PII)

### 7. Code Security

- [ ] **[CRITICAL]** Run security validation script
  ```bash
  python -m trading_common.security_validator --environment production
  ```
- [ ] Remove all debug endpoints
- [ ] Disable verbose error messages
- [ ] Ensure no commented-out code with credentials
- [ ] Verify all dependencies are up to date

### 8. Infrastructure Security

- [ ] **[CRITICAL]** Separate production from development environments
- [ ] Configure automated security updates
- [ ] Set up intrusion detection system (IDS)
- [ ] Implement regular security scanning
- [ ] Configure backup and disaster recovery

### 9. Compliance & Governance

- [ ] Document data retention policies
- [ ] Implement audit trail for all trades
- [ ] Configure data encryption at rest
- [ ] Set up PII handling procedures (if applicable)
- [ ] Review regulatory compliance requirements

### 10. Testing

- [ ] **[CRITICAL]** Run security test suite
  ```bash
  pytest tests/security/ -v
  ```
- [ ] Perform penetration testing
- [ ] Validate rate limiting works
- [ ] Test authentication flows
- [ ] Verify error handling doesn't leak information

## Validation Commands

Run these commands before deployment:

```bash
# 1. Security validation
python -m trading_common.security_validator --environment production

# 2. Configuration check
python -c "from trading_common import get_settings; settings = get_settings(); settings.enforce_production_security()"

# 3. Run security tests
pytest tests/security/ -v

# 4. Check for hardcoded secrets
grep -r "TradingSystem2024\|password\|secret" --include="*.py" .

# 5. Verify environment variables
env | grep -E "^(SECURITY_|TRADING_|DATABASE_|REDIS_)" | cut -d= -f1
```

## Emergency Procedures

### If Security Breach Detected:

1. **Immediately rotate all credentials**
2. **Revoke all active JWT tokens** 
3. **Enable emergency maintenance mode**
4. **Review audit logs for impact assessment**
5. **Notify security team and stakeholders**

### Credential Rotation Process:

1. Generate new credentials
2. Update secrets management service
3. Deploy configuration change
4. Verify services are operational
5. Revoke old credentials

## Sign-off

- [ ] Security review completed by: ________________
- [ ] Deployment approved by: ________________
- [ ] Date: ________________

## Notes

- This checklist is version controlled and should be updated as security requirements evolve
- All CRITICAL items must be checked before production deployment
- Keep a copy of completed checklists for audit purposes
- Review and update this checklist quarterly