# Testing Guide - AI Trading System

## Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_auth_simple.py
│   ├── test_http_client.py
│   ├── test_ml_components.py
│   └── ...
├── integration/          # Integration tests
│   ├── test_database_integration.py
│   └── ...
├── smoke/               # Smoke tests
│   └── test_core_endpoints.py
└── fixtures/           # Shared test fixtures
```

## Running Tests

### Quick Test Commands
```bash
# Run all tests
make test

# Run specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-security     # Security tests
make test-coverage     # Generate coverage report

# Using pytest directly
pytest tests/unit -v
pytest tests/integration -v --asyncio-mode=auto
pytest --cov=api --cov=shared --cov-report=html
```

### Test Coverage Requirements
- **Minimum Coverage**: 70% (enforced)
- **Target Coverage**: 85%+
- **Critical Components**: 90%+

## Test Categories

### Unit Tests
Fast, isolated tests for individual components:
- Authentication functions
- HTTP client with circuit breakers
- ML model components
- Data validation
- Utility functions

### Integration Tests
Tests that verify component interactions:
- Database operations
- API endpoint functionality
- External API integration (mocked)
- Message broker communication
- Cache operations

### Smoke Tests
Quick validation of core functionality:
- Health endpoints
- Authentication flow
- Basic API operations
- Critical path validation

### Security Tests
Security-focused validation:
- Input validation
- SQL injection prevention
- Authentication bypass attempts
- Rate limiting verification
- Secret exposure checks

## Writing Tests

### Test Structure Template
```python
import pytest
from unittest.mock import Mock, patch

class TestComponentName:
    """Test suite for ComponentName."""
    
    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        return {"key": "value"}
    
    def test_functionality(self, setup):
        """Test specific functionality."""
        # Arrange
        input_data = setup
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result.status == "success"
        assert result.value > 0
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        result = await async_function()
        assert result is not None
```

### Testing Best Practices

1. **Use Descriptive Names**
   - `test_auth_creates_valid_jwt_token`
   - `test_circuit_breaker_opens_after_failures`

2. **Follow AAA Pattern**
   - Arrange: Set up test data
   - Act: Execute the function
   - Assert: Verify results

3. **Mock External Dependencies**
   ```python
   @patch('httpx.AsyncClient.get')
   async def test_api_call(mock_get):
       mock_get.return_value.json.return_value = {"data": "value"}
       result = await fetch_data()
       assert result["data"] == "value"
   ```

4. **Use Fixtures for Common Setup**
   ```python
   @pytest.fixture
   async def db_session():
       async with get_test_session() as session:
           yield session
   ```

5. **Test Edge Cases**
   - Empty inputs
   - Invalid data types
   - Boundary values
   - Concurrent access
   - Error conditions

## Testing Checklist

### Before Committing
- [ ] All tests pass locally
- [ ] Coverage meets minimum requirements
- [ ] No hardcoded test data
- [ ] Tests are deterministic
- [ ] No test pollution between runs

### Critical Path Tests
- [ ] User authentication flow
- [ ] Order placement and execution
- [ ] Risk limit enforcement
- [ ] Data feed processing
- [ ] Model inference pipeline

### Security Tests
- [ ] Authentication bypass attempts
- [ ] SQL injection prevention
- [ ] Rate limiting enforcement
- [ ] Input validation
- [ ] Secret exposure checks

### Performance Tests
- [ ] API response times < 50ms
- [ ] WebSocket latency < 10ms
- [ ] Database query optimization
- [ ] Memory leak detection
- [ ] Concurrent user handling

## CI/CD Integration

### GitHub Actions Workflow
Tests run automatically on:
- Push to main branch
- Pull request creation
- Daily scheduled runs

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Test Environment Variables
```bash
# Test configuration
export TESTING=true
export DATABASE_URL=postgresql://test_user:password@localhost:5432/test_db
export REDIS_URL=redis://localhost:6379/1
export LOG_LEVEL=DEBUG
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=.:shared/python-common:$PYTHONPATH
```

**Async Test Failures**
```python
# Use proper async test decorator
@pytest.mark.asyncio
async def test_async():
    pass
```

**Database Connection Issues**
```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d
```

**Flaky Tests**
- Check for time-dependent logic
- Ensure proper test isolation
- Mock external dependencies
- Use deterministic data

## Test Metrics

### Current Status
- **Total Tests**: 150+
- **Unit Test Coverage**: 85%
- **Integration Coverage**: 70%
- **Average Test Time**: < 30 seconds
- **Security Tests**: 20+

### Performance Benchmarks
- Unit tests: < 10 seconds
- Integration tests: < 30 seconds
- Full test suite: < 2 minutes
- Coverage report: < 1 minute

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Async Testing Guide](https://pytest-asyncio.readthedocs.io/)

---

Last Updated: November 26, 2024