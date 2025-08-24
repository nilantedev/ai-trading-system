# Trading Common Library

Shared Python utilities and libraries for the AI trading system.

## Installation

```bash
pip install -e .
```

## Components

- `config`: Configuration management with Pydantic settings
- `logging`: Structured logging with correlation IDs
- `database`: Database connection pools and utilities
- `messaging`: Message queue abstractions
- `validation`: Data validation and sanitization
- `monitoring`: Prometheus metrics and health checks
- `auth`: Authentication and authorization utilities

## Usage

```python
from trading_common.config import get_settings
from trading_common.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)
```