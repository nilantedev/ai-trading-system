"""
Enterprise-grade secrets management with HashiCorp Vault integration.
Provides secure secret storage, rotation, and access control.
"""

import os
import json
import base64
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import hvac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Optional cloud provider imports
try:
    import boto3
    HAS_AWS = True
except ImportError:
    HAS_AWS = False
    
try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False
    
try:
    from google.cloud import secretmanager
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

logger = logging.getLogger(__name__)


class SecretProvider:
    """Base class for secret providers."""
    
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret value."""
        raise NotImplementedError
    
    def set_secret(self, key: str, value: str) -> bool:
        """Store a secret value."""
        raise NotImplementedError
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        raise NotImplementedError
    
    def list_secrets(self) -> List[str]:
        """List available secret keys."""
        raise NotImplementedError
    
    def rotate_secret(self, key: str) -> str:
        """Rotate a secret and return new value."""
        raise NotImplementedError


class HashiCorpVaultProvider(SecretProvider):
    """HashiCorp Vault integration for production environments."""
    
    def __init__(self, vault_url: str, token: str, mount_point: str = "secret"):
        self.client = hvac.Client(url=vault_url, token=token)
        self.mount_point = mount_point
        
        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed")
    
    def get_secret(self, key: str) -> Optional[str]:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=key, mount_point=self.mount_point
            )
            return response["data"]["data"].get("value")
        except Exception as e:
            logger.error(f"Failed to retrieve secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=key,
                secret={"value": value},
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=key, mount_point=self.mount_point
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                mount_point=self.mount_point
            )
            return response["data"]["keys"]
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    def rotate_secret(self, key: str) -> str:
        """Rotate secret with version history."""
        import secrets
        new_value = secrets.token_urlsafe(32)
        self.set_secret(key, new_value)
        return new_value


class AWSSecretsManagerProvider(SecretProvider):
    """AWS Secrets Manager integration."""
    
    def __init__(self, region_name: str = "us-east-1"):
        if not HAS_AWS:
            raise ImportError("boto3 is required for AWS Secrets Manager. Install with: pip install boto3")
        self.client = boto3.client("secretsmanager", region_name=region_name)
        self.region = region_name
    
    def get_secret(self, key: str) -> Optional[str]:
        try:
            response = self.client.get_secret_value(SecretId=key)
            return response.get("SecretString")
        except Exception as e:
            logger.error(f"Failed to retrieve AWS secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        try:
            self.client.create_secret(Name=key, SecretString=value)
            return True
        except self.client.exceptions.ResourceExistsException:
            try:
                self.client.update_secret(SecretId=key, SecretString=value)
                return True
            except Exception as e:
                logger.error(f"Failed to update AWS secret {key}: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to create AWS secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        try:
            self.client.delete_secret(SecretId=key, ForceDeleteWithoutRecovery=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete AWS secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        try:
            response = self.client.list_secrets()
            return [secret["Name"] for secret in response.get("SecretList", [])]
        except Exception as e:
            logger.error(f"Failed to list AWS secrets: {e}")
            return []
    
    def rotate_secret(self, key: str) -> str:
        """Use AWS automatic rotation."""
        import secrets
        new_value = secrets.token_urlsafe(32)
        self.set_secret(key, new_value)
        return new_value


class AzureKeyVaultProvider(SecretProvider):
    """Azure Key Vault integration."""
    
    def __init__(self, vault_url: str):
        if not HAS_AZURE:
            raise ImportError("azure-keyvault-secrets is required. Install with: pip install azure-keyvault-secrets azure-identity")
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
    
    def get_secret(self, key: str) -> Optional[str]:
        try:
            secret = self.client.get_secret(key)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to retrieve Azure secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        try:
            self.client.set_secret(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to store Azure secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        try:
            poller = self.client.begin_delete_secret(key)
            poller.wait()
            return True
        except Exception as e:
            logger.error(f"Failed to delete Azure secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        try:
            return [secret.name for secret in self.client.list_properties_of_secrets()]
        except Exception as e:
            logger.error(f"Failed to list Azure secrets: {e}")
            return []
    
    def rotate_secret(self, key: str) -> str:
        import secrets
        new_value = secrets.token_urlsafe(32)
        self.set_secret(key, new_value)
        return new_value


class GCPSecretManagerProvider(SecretProvider):
    """Google Cloud Secret Manager integration."""
    
    def __init__(self, project_id: str):
        if not HAS_GCP:
            raise ImportError("google-cloud-secret-manager is required. Install with: pip install google-cloud-secret-manager")
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id
        self.parent = f"projects/{project_id}"
    
    def get_secret(self, key: str) -> Optional[str]:
        try:
            name = f"{self.parent}/secrets/{key}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Failed to retrieve GCP secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        try:
            secret_id = f"{self.parent}/secrets/{key}"
            
            try:
                self.client.get_secret(request={"name": secret_id})
                self.client.add_secret_version(
                    request={
                        "parent": secret_id,
                        "payload": {"data": value.encode("UTF-8")}
                    }
                )
            except:
                self.client.create_secret(
                    request={
                        "parent": self.parent,
                        "secret_id": key,
                        "secret": {"replication": {"automatic": {}}}
                    }
                )
                self.client.add_secret_version(
                    request={
                        "parent": secret_id,
                        "payload": {"data": value.encode("UTF-8")}
                    }
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store GCP secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        try:
            name = f"{self.parent}/secrets/{key}"
            self.client.delete_secret(request={"name": name})
            return True
        except Exception as e:
            logger.error(f"Failed to delete GCP secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        try:
            secrets = self.client.list_secrets(request={"parent": self.parent})
            return [secret.name.split("/")[-1] for secret in secrets]
        except Exception as e:
            logger.error(f"Failed to list GCP secrets: {e}")
            return []
    
    def rotate_secret(self, key: str) -> str:
        import secrets
        new_value = secrets.token_urlsafe(32)
        self.set_secret(key, new_value)
        return new_value


class LocalEncryptedProvider(SecretProvider):
    """Local encrypted storage for development/testing."""
    
    def __init__(self, storage_path: str, master_key: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if master_key:
            self.fernet = Fernet(master_key.encode() if len(master_key) == 44 else self._derive_key(master_key))
        else:
            key_file = self.storage_path / ".master.key"
            if key_file.exists():
                self.fernet = Fernet(key_file.read_bytes())
            else:
                key = Fernet.generate_key()
                key_file.write_bytes(key)
                key_file.chmod(0o600)
                self.fernet = Fernet(key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def get_secret(self, key: str) -> Optional[str]:
        try:
            file_path = self.storage_path / f"{key}.enc"
            if file_path.exists():
                encrypted_data = file_path.read_bytes()
                return self.fernet.decrypt(encrypted_data).decode()
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve local secret {key}: {e}")
            return None
    
    def set_secret(self, key: str, value: str) -> bool:
        try:
            file_path = self.storage_path / f"{key}.enc"
            encrypted_data = self.fernet.encrypt(value.encode())
            file_path.write_bytes(encrypted_data)
            file_path.chmod(0o600)
            return True
        except Exception as e:
            logger.error(f"Failed to store local secret {key}: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        try:
            file_path = self.storage_path / f"{key}.enc"
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete local secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        try:
            return [f.stem for f in self.storage_path.glob("*.enc")]
        except Exception as e:
            logger.error(f"Failed to list local secrets: {e}")
            return []
    
    def rotate_secret(self, key: str) -> str:
        import secrets
        new_value = secrets.token_urlsafe(32)
        self.set_secret(key, new_value)
        return new_value


class SecretsManager:
    """Main secrets management interface with multi-provider support."""
    
    def __init__(self, provider: Optional[SecretProvider] = None):
        if provider:
            self.provider = provider
        else:
            self.provider = self._auto_detect_provider()
        
        self._cache: Dict[str, tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def _auto_detect_provider(self) -> SecretProvider:
        """Auto-detect the best available provider based on environment."""
        
        # Check for Vault
        if os.getenv("VAULT_ADDR") and os.getenv("VAULT_TOKEN"):
            logger.info("Using HashiCorp Vault provider")
            return HashiCorpVaultProvider(
                vault_url=os.getenv("VAULT_ADDR"),
                token=os.getenv("VAULT_TOKEN")
            )
        
        # Check for AWS
        if os.getenv("AWS_SECRET_ACCESS_KEY"):
            logger.info("Using AWS Secrets Manager provider")
            return AWSSecretsManagerProvider(
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            )
        
        # Check for Azure
        if os.getenv("AZURE_KEY_VAULT_URL"):
            logger.info("Using Azure Key Vault provider")
            return AzureKeyVaultProvider(vault_url=os.getenv("AZURE_KEY_VAULT_URL"))
        
        # Check for GCP
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            logger.info("Using GCP Secret Manager provider")
            return GCPSecretManagerProvider(project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))
        
        # Fallback to local encrypted storage
        logger.warning("No cloud provider detected, using local encrypted storage")
        storage_path = os.getenv("LOCAL_SECRETS_PATH", "/tmp/secrets")
        return LocalEncryptedProvider(storage_path=storage_path)
    
    def get_secret(self, key: str, use_cache: bool = True) -> Optional[str]:
        """Get a secret with optional caching."""
        
        # Check cache
        if use_cache and key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                return value
        
        # Retrieve from provider
        value = self.provider.get_secret(key)
        
        # Update cache
        if value and use_cache:
            self._cache[key] = (value, datetime.now())
        
        return value
    
    def get_secret_or_env(self, key: str, env_var: Optional[str] = None) -> Optional[str]:
        """Get secret from provider, fallback to environment variable."""
        value = self.get_secret(key)
        if not value and env_var:
            value = os.getenv(env_var)
        return value
    
    def get_required_secret(self, key: str) -> str:
        """Get a required secret, raise exception if not found."""
        value = self.get_secret(key)
        if not value:
            raise ValueError(f"Required secret '{key}' not found")
        return value
    
    def set_secret(self, key: str, value: str) -> bool:
        """Store a secret."""
        success = self.provider.set_secret(key, value)
        if success:
            self._cache.pop(key, None)
        return success
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        success = self.provider.delete_secret(key)
        if success:
            self._cache.pop(key, None)
        return success
    
    def rotate_secret(self, key: str) -> str:
        """Rotate a secret and return new value."""
        new_value = self.provider.rotate_secret(key)
        self._cache.pop(key, None)
        return new_value
    
    def list_secrets(self) -> List[str]:
        """List available secret keys."""
        return self.provider.list_secrets()
    
    def validate_secrets(self, required_keys: List[str]) -> Dict[str, bool]:
        """Validate that required secrets exist."""
        results = {}
        for key in required_keys:
            results[key] = self.get_secret(key) is not None
        return results
    
    def clear_cache(self):
        """Clear the secrets cache."""
        self._cache.clear()
    
    def generate_secure_password(self, length: int = 32) -> str:
        """Generate a cryptographically secure password."""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def hash_secret(self, value: str) -> str:
        """Create a secure hash of a secret for comparison."""
        return hashlib.sha256(value.encode()).hexdigest()


# Singleton instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create the singleton secrets manager."""
    global _secrets_manager
    if not _secrets_manager:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions
def get_secret(key: str) -> Optional[str]:
    """Get a secret value."""
    return get_secrets_manager().get_secret(key)


def get_required_secret(key: str) -> str:
    """Get a required secret value."""
    return get_secrets_manager().get_required_secret(key)


def set_secret(key: str, value: str) -> bool:
    """Store a secret value."""
    return get_secrets_manager().set_secret(key, value)


def rotate_secret(key: str) -> str:
    """Rotate a secret."""
    return get_secrets_manager().rotate_secret(key)