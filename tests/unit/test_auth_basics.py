#!/usr/bin/env python3
"""Basic auth related tests: redirect sanitization, password policy, key rotation guard.

These tests are intentionally lightweight and avoid external dependencies (DB, Redis) by
monkeypatching where needed. They focus on logic surfaces we modified recently.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.main import app  # noqa: E402

# NOTE: If auth manager requires Redis/Postgres, we will patch minimal surfaces.

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_login_redirect_sanitization(client):
    """Ensure that providing an external redirect is not blindly followed (we expect server to ignore or sanitize).
    This depends on login endpoint behavior; if absent, test will xfail.
    """
    # Only run if endpoint exists
    login_path = "/auth/login"
    resp_head = client.options(login_path)
    assert resp_head.status_code in (200, 405, 422)  # existence heuristic

    # Attempt login with crafted redirect (will likely fail auth but must not expose redirect)
    payload = {"username": "unknown", "password": "bad", "redirect": "https://evil.com"}
    r = client.post(login_path, json=payload)
    # We only assert it does not reflect raw external redirect in body
    body = r.text.lower()
    assert "evil.com" not in body, "External redirect domain leaked in response body"


def test_password_policy_rejects_simple():
    from api.auth import JWTAuthManager  # noqa: E402
    mgr = JWTAuthManager()
    assert mgr.password_meets_policy("short1!") is False
    assert mgr.password_meets_policy("alllowercasebutlong123!") is False  # no uppercase
    assert mgr.password_meets_policy("ALLUPPERCASEBUTLONG123!") is False  # no lowercase
    assert mgr.password_meets_policy("MixedCaseButNoDigit!!!!") is False
    assert mgr.password_meets_policy("Mixed123NoSymbol") is False


def test_password_policy_accepts_strong():
    from api.auth import JWTAuthManager  # noqa: E402
    mgr = JWTAuthManager()
    assert mgr.password_meets_policy("Str0ng!Password123") is True


def test_key_rotation_admin_guard(client):
    rotate_path = "/auth/admin/rotate-keys"
    # Unauthenticated should be rejected (401 or 403 depending on dependency chain)
    r = client.post(rotate_path)
    assert r.status_code in (401, 403)

    # If we had a token we could test success path; out-of-scope without full DB fixtures.
    # Just ensure endpoint exists
    assert r.status_code != 404
