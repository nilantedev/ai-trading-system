"""
Unit tests for secrets manager with comprehensive coverage.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from cryptography.fernet import Fernet

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.secrets_manager import (
    SecretsManager,
    LocalEncryptedProvider,
    HashiCorpVaultProvider,
    AWSSecretsManagerProvider,
    get_secrets_manager,
    get_secret,
    set_secret
)


class TestLocalEncryptedProvider:
    """Test local encrypted provider."""
    
    @pytest.fixture
    def provider(self):
        """Create provider with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalEncryptedProvider(storage_path=tmpdir)
    
    def test_set_and_get_secret(self, provider):
        """Test setting and retrieving a secret."""
        # Arrange
        key = "test_key"
        value = "test_value_123"
        
        # Act
        result = provider.set_secret(key, value)
        retrieved = provider.get_secret(key)
        
        # Assert
        assert result is True
        assert retrieved == value
    
    def test_get_nonexistent_secret(self, provider):
        """Test retrieving non-existent secret returns None."""
        # Act
        result = provider.get_secret("nonexistent")
        
        # Assert
        assert result is None
    
    def test_delete_secret(self, provider):
        """Test deleting a secret."""
        # Arrange
        key = "test_delete"
        provider.set_secret(key, "value")
        
        # Act
        delete_result = provider.delete_secret(key)
        retrieved = provider.get_secret(key)
        
        # Assert
        assert delete_result is True
        assert retrieved is None
    
    def test_list_secrets(self, provider):
        """Test listing all secret keys."""
        # Arrange
        keys = ["key1", "key2", "key3"]
        for key in keys:
            provider.set_secret(key, f"value_{key}")
        
        # Act
        listed = provider.list_secrets()
        
        # Assert
        assert set(listed) == set(keys)
    
    def test_rotate_secret(self, provider):
        """Test rotating a secret."""
        # Arrange
        key = "rotate_test"
        original = "original_value"
        provider.set_secret(key, original)
        
        # Act
        new_value = provider.rotate_secret(key)
        retrieved = provider.get_secret(key)
        
        # Assert
        assert new_value != original
        assert retrieved == new_value
        assert len(new_value) > 0
    
    def test_encryption_persistence(self, provider):
        """Test that secrets are encrypted on disk."""
        # Arrange
        key = "encryption_test"
        value = "sensitive_data"
        provider.set_secret(key, value)
        
        # Act
        file_path = Path(provider.storage_path) / f"{key}.enc"
        raw_content = file_path.read_bytes()
        
        # Assert
        assert value.encode() not in raw_content  # Value should not be in plaintext
        assert len(raw_content) > 0
    
    def test_master_key_derivation(self):
        """Test password-based key derivation."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalEncryptedProvider(
                storage_path=tmpdir,
                master_key="test_password_123"
            )
        
        # Act & Assert
        assert provider.set_secret("test", "value")
        assert provider.get_secret("test") == "value"


class TestHashiCorpVaultProvider:
    """Test HashiCorp Vault provider."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Vault client."""
        client = Mock()
        client.is_authenticated.return_value = True
        client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"value": "test_value"}}
        }
        return client
    
    @patch('shared.security.secrets_manager.hvac.Client')
    def test_get_secret(self, mock_hvac, mock_client):
        """Test retrieving secret from Vault."""
        # Arrange
        mock_hvac.return_value = mock_client
        provider = HashiCorpVaultProvider("http://vault:8200", "token")
        
        # Act
        result = provider.get_secret("test_key")
        
        # Assert
        assert result == "test_value"
        mock_client.secrets.kv.v2.read_secret_version.assert_called_once()
    
    @patch('shared.security.secrets_manager.hvac.Client')
    def test_set_secret(self, mock_hvac, mock_client):
        """Test storing secret in Vault."""
        # Arrange
        mock_hvac.return_value = mock_client
        provider = HashiCorpVaultProvider("http://vault:8200", "token")
        
        # Act
        result = provider.set_secret("test_key", "test_value")
        
        # Assert
        assert result is True
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once()
    
    @patch('shared.security.secrets_manager.hvac.Client')
    def test_authentication_failure(self, mock_hvac):
        """Test handling authentication failure."""
        # Arrange
        mock_client = Mock()
        mock_client.is_authenticated.return_value = False
        mock_hvac.return_value = mock_client
        
        # Act & Assert
        with pytest.raises(ValueError, match="Vault authentication failed"):
            HashiCorpVaultProvider("http://vault:8200", "invalid_token")


class TestSecretsManager:
    """Test main secrets manager."""
    
    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        provider = Mock()
        provider.get_secret.return_value = "test_value"
        provider.set_secret.return_value = True
        provider.delete_secret.return_value = True
        provider.list_secrets.return_value = ["key1", "key2"]
        return SecretsManager(provider=provider)
    
    def test_get_secret_with_cache(self, manager):
        """Test caching behavior."""
        # Act
        result1 = manager.get_secret("test_key", use_cache=True)
        result2 = manager.get_secret("test_key", use_cache=True)
        
        # Assert
        assert result1 == "test_value"
        assert result2 == "test_value"
        # Provider should only be called once due to caching
        manager.provider.get_secret.assert_called_once_with("test_key")
    
    def test_get_secret_without_cache(self, manager):
        """Test bypassing cache."""
        # Act
        result1 = manager.get_secret("test_key", use_cache=False)
        result2 = manager.get_secret("test_key", use_cache=False)
        
        # Assert
        assert result1 == "test_value"
        assert result2 == "test_value"
        # Provider should be called twice
        assert manager.provider.get_secret.call_count == 2
    
    def test_get_required_secret_exists(self, manager):
        """Test getting required secret that exists."""
        # Act
        result = manager.get_required_secret("test_key")
        
        # Assert
        assert result == "test_value"
    
    def test_get_required_secret_missing(self, manager):
        """Test getting required secret that doesn't exist."""
        # Arrange
        manager.provider.get_secret.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Required secret 'missing_key' not found"):
            manager.get_required_secret("missing_key")
    
    def test_get_secret_or_env(self, manager, monkeypatch):
        """Test fallback to environment variable."""
        # Arrange
        manager.provider.get_secret.return_value = None
        monkeypatch.setenv("TEST_ENV_VAR", "env_value")
        
        # Act
        result = manager.get_secret_or_env("missing_key", "TEST_ENV_VAR")
        
        # Assert
        assert result == "env_value"
    
    def test_validate_secrets(self, manager):
        """Test validating multiple secrets."""
        # Arrange
        def get_secret_side_effect(key):
            return "value" if key != "missing" else None
        
        manager.provider.get_secret.side_effect = get_secret_side_effect
        
        # Act
        results = manager.validate_secrets(["key1", "key2", "missing"])
        
        # Assert
        assert results == {
            "key1": True,
            "key2": True,
            "missing": False
        }
    
    def test_clear_cache(self, manager):
        """Test clearing the cache."""
        # Arrange
        manager.get_secret("key1", use_cache=True)
        assert len(manager._cache) > 0
        
        # Act
        manager.clear_cache()
        
        # Assert
        assert len(manager._cache) == 0
    
    def test_generate_secure_password(self, manager):
        """Test password generation."""
        # Act
        password = manager.generate_secure_password(20)
        
        # Assert
        assert len(password) == 20
        assert any(c.isdigit() for c in password)
        assert any(c.isalpha() for c in password)
    
    def test_generate_api_key(self, manager):
        """Test API key generation."""
        # Act
        key = manager.generate_api_key()
        
        # Assert
        assert len(key) > 0
        assert isinstance(key, str)
    
    def test_hash_secret(self, manager):
        """Test secret hashing."""
        # Act
        hash1 = manager.hash_secret("test_value")
        hash2 = manager.hash_secret("test_value")
        hash3 = manager.hash_secret("different_value")
        
        # Assert
        assert hash1 == hash2  # Same input produces same hash
        assert hash1 != hash3  # Different input produces different hash
        assert len(hash1) == 64  # SHA256 produces 64-character hex string


class TestAWSSecretsManagerProvider:
    """Test AWS Secrets Manager provider."""
    
    @patch('shared.security.secrets_manager.boto3.client')
    def test_get_secret(self, mock_boto):
        """Test retrieving secret from AWS."""
        # Arrange
        mock_client = Mock()
        mock_client.get_secret_value.return_value = {"SecretString": "test_value"}
        mock_boto.return_value = mock_client
        
        provider = AWSSecretsManagerProvider()
        
        # Act
        result = provider.get_secret("test_key")
        
        # Assert
        assert result == "test_value"
        mock_client.get_secret_value.assert_called_with(SecretId="test_key")
    
    @patch('shared.security.secrets_manager.boto3.client')
    def test_set_secret_new(self, mock_boto):
        """Test creating new secret in AWS."""
        # Arrange
        mock_client = Mock()
        mock_boto.return_value = mock_client
        
        provider = AWSSecretsManagerProvider()
        
        # Act
        result = provider.set_secret("new_key", "new_value")
        
        # Assert
        assert result is True
        mock_client.create_secret.assert_called_with(
            Name="new_key",
            SecretString="new_value"
        )


class TestSecureConfig:
    """Test secure configuration."""
    
    @patch('config.secure_config.get_secrets_manager')
    def test_config_loads_secrets(self, mock_get_manager):
        """Test that config loads secrets from manager."""
        # Arrange
        mock_manager = Mock()
        mock_manager.get_secret.side_effect = lambda k: f"secret_{k}"
        mock_get_manager.return_value = mock_manager
        
        # Import after patching
        from config.secure_config import SecureConfig
        
        # Act
        config = SecureConfig(environment="development")
        
        # Assert
        assert mock_manager.get_secret.called
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        from config.secure_config import SecureConfig
        from pydantic import SecretStr
        
        # Arrange
        config = SecureConfig(
            db_host="localhost",
            db_port=5432,
            db_name="test_db",
            db_user="test_user",
            db_password=SecretStr("test_pass")
        )
        
        # Act
        url = config.database_url
        
        # Assert
        assert url == "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"
    
    def test_environment_validation(self):
        """Test environment validation."""
        from config.secure_config import SecureConfig, Environment
        
        # Valid environment
        config = SecureConfig(environment="production")
        assert config.environment == Environment.PRODUCTION
        
        # Invalid environment
        with pytest.raises(ValueError):
            SecureConfig(environment="invalid")


class TestIntegrationSecretsFlow:
    """Integration tests for complete secrets flow."""
    
    def test_full_secret_lifecycle(self):
        """Test complete secret lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create provider
            provider = LocalEncryptedProvider(storage_path=tmpdir)
            manager = SecretsManager(provider=provider)
            
            # Create secret
            assert manager.set_secret("lifecycle/test", "initial_value")
            
            # Read secret
            assert manager.get_secret("lifecycle/test") == "initial_value"
            
            # Update secret
            assert manager.set_secret("lifecycle/test", "updated_value")
            assert manager.get_secret("lifecycle/test") == "updated_value"
            
            # Rotate secret
            new_value = manager.rotate_secret("lifecycle/test")
            assert new_value != "updated_value"
            assert manager.get_secret("lifecycle/test") == new_value
            
            # List secrets
            secrets = manager.list_secrets()
            assert "lifecycle/test" in secrets
            
            # Delete secret
            assert manager.delete_secret("lifecycle/test")
            assert manager.get_secret("lifecycle/test") is None
    
    def test_concurrent_access(self):
        """Test concurrent access to secrets."""
        import threading
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalEncryptedProvider(storage_path=tmpdir)
            manager = SecretsManager(provider=provider)
            
            results = []
            
            def worker(worker_id):
                for i in range(10):
                    key = f"concurrent/key_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"
                    manager.set_secret(key, value)
                    retrieved = manager.get_secret(key)
                    results.append(retrieved == value)
            
            # Start multiple threads
            threads = []
            for i in range(5):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()
            
            # Wait for completion
            for t in threads:
                t.join()
            
            # All operations should succeed
            assert all(results)