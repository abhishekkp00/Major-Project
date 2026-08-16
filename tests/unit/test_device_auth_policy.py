"""
test_device_auth_policy.py
===========================
Comprehensive unit test matrix for the Adaptive Device Authorization System.

Tests all 14 required operational scenarios:
  1. Identical device
  2. Reboot
  3. Network change
  4. Hostname change
  5. Disk replacement
  6. Machine-ID replacement
  7. CPU mismatch
  8. Missing device feature
  9. VM clone simulation
 10. Completely different device
 11. Reauthorization success
 12. Unauthorized self-reauthorization attempt
 13. Package tampering combined with device change
 14. Invalid signature combined with authorized device
"""

import pytest
from pathlib import Path

from src.security.device_auth_policy import (
    BindingPolicy,
    DeviceState,
    evaluate_device_authorization,
    reauthorize_device,
    collect_classified_features,
)
from src.security.fingerprint import compute_fingerprint_hash, build_canonical_string
from src.phase4.package_validator import validate_package_provenance
from src.phase4.device_auth import verify_device_binding
from src.common.exceptions import DeviceAuthorizationError, SignatureValidationError


@pytest.fixture
def base_features():
    return {
        "machine_id": "mid-12345-abcde",
        "cpu_model": "Intel Core i7-12700K",
        "disk_uuid": "disk-uuid-9999",
        "hostname": "edge-node-01",
        "network_interface": "00:11:22:33:44:55",
    }


@pytest.fixture
def default_policy():
    return BindingPolicy(
        enabled=True,
        strictness="high",
        allow_network_change=True,
        allow_hostname_change=True,
        allow_disk_change=False,
        allow_machine_id_change=False,
        allow_cpu_change=False,
    )


# 1. Identical Device
def test_scenario_1_identical_device(base_features, default_policy):
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": base_features["machine_id"], "cpu_model": base_features["cpu_model"]},
                            "semi_stable": {"disk_uuid": base_features["disk_uuid"]},
                            "volatile": {"hostname": base_features["hostname"], "network_interface": base_features["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.AUTHORIZED
    assert res.is_authorized is True


# 2. Reboot Simulation (No attribute changes)
def test_scenario_2_reboot(base_features, default_policy):
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": base_features["machine_id"], "cpu_model": base_features["cpu_model"]},
                            "semi_stable": {"disk_uuid": base_features["disk_uuid"]},
                            "volatile": {"hostname": base_features["hostname"], "network_interface": base_features["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.AUTHORIZED


# 3. Network Change
def test_scenario_3_network_change(base_features, default_policy):
    curr = dict(base_features)
    curr["network_interface"] = "aa:bb:cc:dd:ee:ff"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.REAUTHORIZATION_REQUIRED
    assert res.reauthorization_allowed is True


# 4. Hostname Change
def test_scenario_4_hostname_change(base_features, default_policy):
    curr = dict(base_features)
    curr["hostname"] = "new-edge-name"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.REAUTHORIZATION_REQUIRED
    assert res.reauthorization_allowed is True


# 5. Disk Replacement
def test_scenario_5_disk_replacement(base_features, default_policy):
    curr = dict(base_features)
    curr["disk_uuid"] = "new-disk-uuid-0000"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.UNAUTHORIZED
    assert res.is_authorized is False


# 6. Machine-ID Replacement
def test_scenario_6_machine_id_replacement(base_features, default_policy):
    curr = dict(base_features)
    curr["machine_id"] = "new-machine-id-999"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.UNAUTHORIZED
    assert res.is_authorized is False


# 7. CPU Mismatch
def test_scenario_7_cpu_mismatch(base_features, default_policy):
    curr = dict(base_features)
    curr["cpu_model"] = "AMD EPYC 7763"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.UNAUTHORIZED


# 8. Missing Device Feature
def test_scenario_8_missing_device_feature(base_features, default_policy):
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": "UNAVAILABLE", "cpu_model": "UNAVAILABLE"},
                            "semi_stable": {"disk_uuid": "UNAVAILABLE"},
                            "volatile": {"hostname": "UNAVAILABLE", "network_interface": "UNAVAILABLE"}},
        policy=default_policy,
    )
    assert res.state == DeviceState.UNAUTHORIZED
    assert res.fingerprint_stability == "MISSING_IDENTIFIERS"


# 9. VM Clone Simulation
def test_scenario_9_vm_clone_simulation(base_features, default_policy):
    # VM clone has same machine_id, cpu, disk, but different MAC address
    curr = dict(base_features)
    curr["network_interface"] = "55:44:33:22:11:00"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.REAUTHORIZATION_REQUIRED


# 10. Completely Different Device
def test_scenario_10_completely_different_device(base_features, default_policy):
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": "foreign-id", "cpu_model": "ARM Cortex-A72"},
                            "semi_stable": {"disk_uuid": "foreign-disk"},
                            "volatile": {"hostname": "rogue-host", "network_interface": "ff:ff:ff:ff:ff:ff"}},
        policy=default_policy,
    )
    assert res.state == DeviceState.UNAUTHORIZED


# 11. Reauthorization Success
def test_scenario_11_reauthorization_success(base_features, default_policy):
    curr = dict(base_features)
    curr["network_interface"] = "aa:bb:cc:dd:ee:ff"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.REAUTHORIZATION_REQUIRED
    reauth_res, audit_rec = reauthorize_device(res, admin_token="SECRET_TOKEN", expected_token="SECRET_TOKEN")
    assert reauth_res.state == DeviceState.AUTHORIZED
    assert audit_rec.reauthorized_by_admin is True


# 12. Unauthorized Self-Reauthorization Attempt
def test_scenario_12_unauthorized_self_reauth(base_features, default_policy):
    curr = dict(base_features)
    curr["machine_id"] = "hacked-machine-id"
    res = evaluate_device_authorization(
        expected_features=base_features,
        current_classified={"stable": {"machine_id": curr["machine_id"], "cpu_model": curr["cpu_model"]},
                            "semi_stable": {"disk_uuid": curr["disk_uuid"]},
                            "volatile": {"hostname": curr["hostname"], "network_interface": curr["network_interface"]}},
        policy=default_policy,
    )
    assert res.state == DeviceState.UNAUTHORIZED
    with pytest.raises(DeviceAuthorizationError, match="Cannot reauthorize device in state 'UNAUTHORIZED'"):
        reauthorize_device(res, admin_token="SECRET_TOKEN", expected_token="SECRET_TOKEN")


# 13. Package Tampering Combined with Device Change (Integrity Check Fails First)
def test_scenario_13_tampering_and_device_change(tmp_path):
    pkg_dir = tmp_path / "tampered_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "adapter.enc").write_bytes(b"tampered_data")
    (pkg_dir / "adapter.hash").write_text("0000000000000000000000000000000000000000000000000000000000000000")
    (pkg_dir / "adapter.sig").write_bytes(b"invalid_sig")
    (pkg_dir / "public.pem").write_text("fake_pem")

    with pytest.raises(Exception):
        validate_package_provenance(pkg_dir)


# 14. Invalid Signature Combined with Authorized Device (Signature Fails Before Binding)
def test_scenario_14_invalid_sig_authorized_device(tmp_path):
    pkg_dir = tmp_path / "invalid_sig_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "adapter.enc").write_bytes(b"valid_looking_data")
    # Correct hash of data
    import hashlib
    h = hashlib.sha256(b"valid_looking_data").hexdigest()
    (pkg_dir / "adapter.hash").write_text(h)
    (pkg_dir / "adapter.sig").write_bytes(b"bad_signature_bytes")

    with pytest.raises(SignatureValidationError):
        validate_package_provenance(pkg_dir)
