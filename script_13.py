# Create security utility
security_content = '''"""
Security utilities for the Legal Agent Orchestrator.
Provides encryption, validation, and privacy protection for sensitive legal data.
"""

import hashlib
import hmac
import secrets
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import json
from pathlib import Path

from .logging import get_logger, log_security_event

logger = get_logger(__name__)


class EncryptionManager:
    """Manages encryption and decryption of sensitive data."""
    
    def __init__(self, encryption_key: str):
        """Initialize with encryption key."""
        self.key = encryption_key.encode()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.key[:32].ljust(32, b'0')))
    
    def encrypt_text(self, text: str) -> str:
        """Encrypt a text string."""
        try:
            encrypted = self.fernet.encrypt(text.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt a text string."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary."""
        json_str = json.dumps(data, ensure_ascii=False)
        return self.encrypt_text(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt to a dictionary."""
        json_str = self.decrypt_text(encrypted_data)
        return json.loads(json_str)
    
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()


class HashManager:
    """Manages hashing and verification of data."""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        
        return (
            base64.urlsafe_b64encode(key).decode(),
            base64.urlsafe_b64encode(salt).decode()
        )
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt_bytes = base64.urlsafe_b64decode(salt.encode())
            expected_hash, _ = HashManager.hash_password(password, salt_bytes)
            return hmac.compare_digest(expected_hash, hashed)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Create a secure hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def create_hmac(message: str, key: str) -> str:
        """Create HMAC for message integrity."""
        return hmac.new(
            key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def verify_hmac(message: str, signature: str, key: str) -> bool:
        """Verify HMAC signature."""
        expected = HashManager.create_hmac(message, key)
        return hmac.compare_digest(expected, signature)


class PrivacyPatterns:
    """Patterns for detecting and handling sensitive legal information."""
    
    # Attorney-client privilege indicators
    PRIVILEGE_PATTERNS = [
        r'\\battorney[\\s-]client\\s+privilege\\b',
        r'\\bconfidential\\s+communication\\b',
        r'\\blegal\\s+advice\\b',
        r'\\bwork\\s+product\\b',
        r'\\bprivileged\\s+and\\s+confidential\\b'
    ]
    
    # Personal information patterns
    PII_PATTERNS = {
        'ssn': r'\\b\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{4}\\b',
        'phone': r'\\b\\d{3}[-\\s]?\\d{3}[-\\s]?\\d{4}\\b',
        'email': r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
        'credit_card': r'\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b',
        'driver_license': r'\\b[A-Z]{1,2}\\d{6,8}\\b'
    }
    
    # Financial information patterns
    FINANCIAL_PATTERNS = {
        'bank_account': r'\\b\\d{8,17}\\b',
        'routing_number': r'\\b\\d{9}\\b',
        'tax_id': r'\\b\\d{2}[-\\s]?\\d{7}\\b'
    }
    
    # Legal document patterns
    LEGAL_PATTERNS = {
        'case_number': r'\\b\\d{2,4}[-\\s]?[A-Z]{1,3}[-\\s]?\\d{2,6}\\b',
        'docket_number': r'\\bNo\\.?\\s*\\d{2,4}[-\\s]?\\d{2,6}\\b',
        'citation': r'\\b\\d+\\s+[A-Za-z.]+\\s+\\d+\\b'
    }


class SecurityValidator:
    """Validates content for security and privacy compliance."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.patterns = PrivacyPatterns()
        self.encryption_manager = EncryptionManager(encryption_key) if encryption_key else None
    
    async def validate_input(self, content: str) -> bool:
        """Validate input content for security issues."""
        try:
            # Check for potential injection attacks
            if self._detect_injection_patterns(content):
                log_security_event(
                    "injection_attempt",
                    "HIGH",
                    {"content_length": len(content), "patterns_detected": True}
                )
                return False
            
            # Check for excessive length
            if len(content) > 100000:  # 100KB limit
                log_security_event(
                    "oversized_input",
                    "MEDIUM",
                    {"content_length": len(content)}
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    async def validate_output(self, content: str) -> bool:
        """Validate output content for privacy compliance."""
        try:
            # Check for sensitive information leakage
            sensitive_data = self.detect_sensitive_data(content)
            
            if sensitive_data:
                log_security_event(
                    "sensitive_data_detected",
                    "HIGH",
                    {"data_types": list(sensitive_data.keys())}
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return False
    
    async def validate_privileged_content(self, content: str) -> bool:
        """Special validation for attorney-client privileged content."""
        try:
            # Check for privilege markers
            has_privilege = self._detect_privilege_patterns(content)
            
            if has_privilege:
                log_security_event(
                    "privileged_content_processed",
                    "INFO",
                    {"content_length": len(content), "has_privilege_markers": True}
                )
            
            # Additional validation for privileged content
            return await self.validate_input(content)
            
        except Exception as e:
            logger.error(f"Privileged content validation error: {e}")
            return False
    
    def detect_sensitive_data(self, content: str) -> Dict[str, List[str]]:
        """Detect sensitive data in content."""
        detected = {}
        
        # Check PII patterns
        for data_type, pattern in self.patterns.PII_PATTERNS.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                detected[f"pii_{data_type}"] = matches
        
        # Check financial patterns
        for data_type, pattern in self.patterns.FINANCIAL_PATTERNS.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                detected[f"financial_{data_type}"] = matches
        
        return detected
    
    def _detect_privilege_patterns(self, content: str) -> bool:
        """Detect attorney-client privilege patterns."""
        for pattern in self.patterns.PRIVILEGE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _detect_injection_patterns(self, content: str) -> bool:
        """Detect potential injection attack patterns."""
        injection_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'onload\\s*=',  # Event injection
            r'eval\\s*\\(',  # Code evaluation
            r'exec\\s*\\(',  # Code execution
            r'\\bUNION\\s+SELECT\\b',  # SQL injection
            r'\\bDROP\\s+TABLE\\b',  # SQL injection
            r'\\.\\./.*\\.\\./.*\\.\\./',  # Path traversal
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def redact_sensitive_data(self, content: str) -> str:
        """Redact sensitive data from content."""
        redacted_content = content
        
        # Redact PII
        for data_type, pattern in self.patterns.PII_PATTERNS.items():
            redacted_content = re.sub(
                pattern,
                f"[REDACTED {data_type.upper()}]",
                redacted_content,
                flags=re.IGNORECASE
            )
        
        # Redact financial information
        for data_type, pattern in self.patterns.FINANCIAL_PATTERNS.items():
            redacted_content = re.sub(
                pattern,
                f"[REDACTED {data_type.upper()}]",
                redacted_content,
                flags=re.IGNORECASE
            )
        
        return redacted_content
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for secure storage."""
        # Remove dangerous characters
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized


class AccessControl:
    """Manages access control and permissions."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.hash_manager = HashManager()
    
    def create_access_token(
        self,
        user_id: str,
        permissions: List[str],
        expiry_hours: int = 24
    ) -> str:
        """Create an access token with permissions."""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "issued_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
        }
        
        token_data = json.dumps(payload, sort_keys=True)
        signature = self.hash_manager.create_hmac(token_data, self.secret_key)
        
        token = base64.urlsafe_b64encode(
            json.dumps({"data": token_data, "signature": signature}).encode()
        ).decode()
        
        log_security_event(
            "access_token_created",
            "INFO",
            {"user_id": user_id, "permissions": permissions}
        )
        
        return token
    
    def validate_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode an access token."""
        try:
            token_obj = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
            token_data = token_obj["data"]
            signature = token_obj["signature"]
            
            # Verify signature
            if not self.hash_manager.verify_hmac(token_data, signature, self.secret_key):
                log_security_event(
                    "invalid_token_signature",
                    "MEDIUM",
                    {"token_length": len(token)}
                )
                return None
            
            payload = json.loads(token_data)
            
            # Check expiry
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.now() > expires_at:
                log_security_event(
                    "expired_token_used",
                    "LOW",
                    {"user_id": payload.get("user_id")}
                )
                return None
            
            return payload
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            log_security_event(
                "token_validation_error",
                "MEDIUM",
                {"error": str(e)}
            )
            return None
    
    def check_permission(
        self,
        token: str,
        required_permission: str
    ) -> bool:
        """Check if token has required permission."""
        payload = self.validate_access_token(token)
        if not payload:
            return False
        
        permissions = payload.get("permissions", [])
        has_permission = required_permission in permissions or "admin" in permissions
        
        if not has_permission:
            log_security_event(
                "permission_denied",
                "MEDIUM",
                {
                    "user_id": payload.get("user_id"),
                    "required_permission": required_permission,
                    "user_permissions": permissions
                }
            )
        
        return has_permission


class SecureFileHandler:
    """Handles secure file operations."""
    
    def __init__(
        self,
        encryption_manager: EncryptionManager,
        validator: SecurityValidator
    ):
        self.encryption_manager = encryption_manager
        self.validator = validator
    
    async def store_secure_file(
        self,
        content: bytes,
        filename: str,
        storage_path: Path,
        encrypt: bool = True
    ) -> str:
        """Store a file securely with optional encryption."""
        try:
            # Sanitize filename
            safe_filename = self.validator.sanitize_filename(filename)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.sha256(content).hexdigest()[:8]
            unique_filename = f"{timestamp}_{content_hash}_{safe_filename}"
            
            file_path = storage_path / unique_filename
            
            # Encrypt if requested
            if encrypt and self.encryption_manager:
                content = self.encryption_manager.encrypt_text(
                    base64.b64encode(content).decode()
                ).encode()
            
            # Store file
            file_path.write_bytes(content)
            
            log_security_event(
                "file_stored",
                "INFO",
                {
                    "filename": safe_filename,
                    "size": len(content),
                    "encrypted": encrypt,
                    "path": str(file_path)
                }
            )
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Secure file storage error: {e}")
            raise
    
    async def retrieve_secure_file(
        self,
        file_path: str,
        decrypt: bool = True
    ) -> bytes:
        """Retrieve a securely stored file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            content = path.read_bytes()
            
            # Decrypt if requested
            if decrypt and self.encryption_manager:
                try:
                    decrypted = self.encryption_manager.decrypt_text(content.decode())
                    content = base64.b64decode(decrypted.encode())
                except Exception:
                    # File might not be encrypted
                    pass
            
            log_security_event(
                "file_retrieved",
                "INFO",
                {"file_path": file_path, "size": len(content)}
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Secure file retrieval error: {e}")
            raise


def generate_secure_id() -> str:
    """Generate a cryptographically secure ID."""
    return secrets.token_urlsafe(32)


def secure_compare(a: str, b: str) -> bool:
    """Securely compare two strings to prevent timing attacks."""
    return hmac.compare_digest(a, b)


def generate_salt() -> str:
    """Generate a cryptographic salt."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
'''

with open("legal_agent_orchestrator/utils/security.py", "w") as f:
    f.write(security_content)

print("Security utility created!")