import pytest

from phase4.device_auth import (
    verify_device_binding,
    get_device_bound_key,
    DeviceAuthorizationError
)

@pytest.fixture
def mock_fingerprints():
    return {
        "auth_fp": "3926c635fa8a12607cf843d884442ae151b5253f54529dc053cd6f0cebddfb93",
        "other_fp": "deadbeef" * 8
    }

@pytest.fixture
def test_salt():
    return "demo-integration-salt-abc123xyz"

def test_device_binding_success(mock_fingerprints):
    # Match expected with mock local
    verify_device_binding(
        expected_fingerprint_hash=mock_fingerprints["auth_fp"],
        mock_fingerprint=mock_fingerprints["auth_fp"]
    ) # Should pass without raising

def test_device_binding_unauthorized(mock_fingerprints):
    # Expected differs from local
    with pytest.raises(DeviceAuthorizationError):
        verify_device_binding(
            expected_fingerprint_hash=mock_fingerprints["auth_fp"],
            mock_fingerprint=mock_fingerprints["other_fp"]
        )

def test_device_key_derivation_deterministic(mock_fingerprints, test_salt):
    k1 = get_device_bound_key(test_salt, mock_fingerprints["auth_fp"])
    k2 = get_device_bound_key(test_salt, mock_fingerprints["auth_fp"])
    assert k1 == k2
    assert len(k1) == 32

def test_device_key_derivation_different_salt(mock_fingerprints, test_salt):
    k1 = get_device_bound_key(test_salt, mock_fingerprints["auth_fp"])
    k2 = get_device_bound_key("different-salt", mock_fingerprints["auth_fp"])
    assert k1 != k2

def test_device_key_derivation_different_fingerprint(mock_fingerprints, test_salt):
    k1 = get_device_bound_key(test_salt, mock_fingerprints["auth_fp"])
    k2 = get_device_bound_key(test_salt, mock_fingerprints["other_fp"])
    assert k1 != k2

def test_device_key_derivation_empty_salt(mock_fingerprints):
    with pytest.raises(ValueError):
        get_device_bound_key("", mock_fingerprints["auth_fp"])
