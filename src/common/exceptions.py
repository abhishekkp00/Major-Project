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


class VerificationError(SecureLoraError, RuntimeError):
    """Raised when Phase 4 pipeline verification fails."""
    pass

