#!/usr/bin/env python3
"""Setup script for trading common library."""

from setuptools import setup, find_packages

# Read the pyproject.toml for basic info, but keep this simple setup.py for build compatibility
setup(
    name="trading-common",
    version="1.0.0.dev0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "structlog>=23.2.0",
        "redis>=5.0.0",
        "asyncpg>=0.29.0",
        "sqlalchemy[asyncio]>=2.0.0",
        "httpx>=0.25.0",
        "python-jose[cryptography]>=3.3.0",
        "prometheus-client>=0.19.0",
        "opentelemetry-api>=1.21.0",
    ],
)