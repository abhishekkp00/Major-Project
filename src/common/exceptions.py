class SecureLoraError(Exception):
    """Base exception class for the Secure Device-Bound LoRA framework."""
    pass


class ConfigError(SecureLoraError, ValueError):
    """Raised when configuration validation or loading fails."""
    pass


class CryptoError(SecureLoraError, ValueError):
    """Raised when cryptographic operations fail (decryption, padding, validation)."""
    pass


class SecurityError(CryptoError):
    """Raised on security violations."""
    pass


class IntegrityValidationError(CryptoError):
    """Raised on integrity validation failures."""
    pass


class SignatureValidationError(CryptoError):
    """Raised on signature verification failures."""
    pass


class DeviceFingerprintError(SecureLoraError, ValueError):
    """Raised when device fingerprint extraction or verification fails."""
    pass


class DeviceAuthorizationError(DeviceFingerprintError):
    """Raised when device authorization verification fails."""
    pass


class PackageError(SecureLoraError, FileNotFoundError):
    """Raised when package parsing, extraction, or completeness checks fail."""
    pass


class IncompletePackageError(PackageError):
    """Raised when a package is missing required files."""
    pass


class InvalidArchiveError(PackageError):
    """Raised when a package archive is invalid or corrupted."""
    pass


class ManifestSchemaError(PackageError):
    """Raised when package_manifest.json schema validation fails."""
    pass


class ReplayAttackError(SecurityError):
    """Raised when an old or duplicated package is submitted (anti-replay check failed)."""
    pass


class PackageExpiredError(SecurityError):
    """Raised when a package is executed past its expiration timestamp."""
    pass


class ModelMismatchError(SecurityError):
    """Raised when a package base_model_id does not match deployment target model."""
    pass


class AdapterMismatchError(SecurityError):
    """Raised when a package adapter_id does not match deployment target adapter."""
    pass



class VerificationError(SecureLoraError, RuntimeError):
    """Raised when Phase 4 pipeline verification fails."""
    pass


class DatasetValidationError(SecureLoraError, ValueError):
    """Raised when uploaded private dataset format or validation checks fail."""
    pass


class AdapterSecurityGateError(SecurityError):
    """Raised when pre-packaging adapter security screening flags an adapter as high-risk."""
    pass





