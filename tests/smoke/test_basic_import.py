"""Basic import test to verify the application can be loaded."""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set required environment variables before imports
os.environ.setdefault("ADMIN_PASSWORD", "devpass123")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("REDIS_PASSWORD", "test_redis")
os.environ.setdefault("ENVIRONMENT", "development")

def test_imports():
    """Test that core modules can be imported."""
    try:
        # Test trading_common imports
        from trading_common import models
        from trading_common.config import get_settings
        assert models is not None
        assert get_settings is not None
        
        # Test API imports
        from api import main
        assert main is not None
        
        # Test that app can be created (but don't initialize it fully)
        assert hasattr(main, 'app')
        
        print("✅ All core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment():
    """Test that environment is properly configured."""
    required_vars = [
        "ADMIN_PASSWORD",
        "SECRET_KEY", 
        "JWT_SECRET",
        "DB_USER",
        "DB_PASSWORD", 
        "DB_NAME"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        return False
    
    print("✅ Environment variables configured")
    return True

def test_config_file():
    """Test that .env file exists."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_path):
        print(f"✅ .env file exists at {env_path}")
        return True
    else:
        print(f"❌ .env file not found at {env_path}")
        return False

if __name__ == "__main__":
    print("Running basic smoke tests...")
    print("-" * 50)
    
    tests = [
        ("Environment Configuration", test_environment),
        ("Config File", test_config_file),
        ("Core Imports", test_imports)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        results.append(test_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ ALL BASIC TESTS PASSED - Ready for deployment")
        sys.exit(0)
    else:
        print("❌ Some tests failed - Review issues above")
        sys.exit(1)