"""
test_device_auth_policy.py
===========================
Unit and integration tests for Adaptive Device-Bound Adapter Authorization Engine.
"""

import os
import pytest

from src.common.exceptions import DeviceAuthorizationError
from src.security.device_auth_policy import (
    DeviceState,
    BindingPolicy,
    AuthorizationResult,
    evaluate_device_authorization,
    reauthorize_device,
    collect_classified_features,
    flatten_classified_features,
)


class TestFeatureClassification:
    def test_collect_classified_features_structure(self):
        classified = collect_classified_features()
        assert "stable" in classified
        assert "semi_stable" in classified
        assert "machine_id" in classified["stable"]
        assert "cpu_model" in classified["stable"]
        assert "disk_uuid" in classified["semi_stable"]
        assert "hostname" in classified["semi_stable"]
        assert "network_interface" in classified["semi_stable"]

    def test_flatten_classified_features(self):
        classified = {
            "stable": {"machine_id": "mid123", "cpu_model": "cpu456"},
            "semi_stable": {"disk_uuid": "disk789", "hostname": "host0"},
        }
        flat = flatten_classified_features(classified)
        assert flat == {
            "machine_id": "mid123",
            "cpu_model": "cpu456",
            "disk_uuid": "disk789",
            "hostname": "host0",
        }


class TestAuthorizationStatesAndTransitions:
    @pytest.fixture
    def baseline_classified(self):
        return {
            "stable": {
                "machine_id": "11111111222222223333333344444444",
                "cpu_model": "Intel Core i7-12700K",
            },
            "semi_stable": {
                "disk_uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "hostname": "primary-node",
                "network_interface": "00:11:22:33:44:55",
            },
        }

    @pytest.fixture
    def high_policy(self):
        return BindingPolicy(
            strictness="high",
            allowed_feature_changes={
                "network_interface": True,
                "hostname": False,
                "machine_id": False,
                "disk_uuid": False,
            },
        )

    def test_state_authorized_on_identical_features(self, baseline_classified, high_policy):
        flat_base = flatten_classified_features(baseline_classified)
        result = evaluate_device_authorization(
            expected_features=flat_base,
            current_classified=baseline_classified,
            policy=high_policy,
        )
        assert result.state == DeviceState.AUTHORIZED
        assert result.is_authorized is True
        assert result.fingerprint_stability == "STABLE"
        assert result.reason_for_rejection is None
        assert result.device_changes_detected == []

    def test_state_unauthorized_on_stable_feature_change(self, baseline_classified, high_policy):
        flat_base = flatten_classified_features(baseline_classified)
        modified = {
            "stable": {
                "machine_id": "99999999999999999999999999999999",  # Stable changed!
                "cpu_model": baseline_classified["stable"]["cpu_model"],
            },
            "semi_stable": baseline_classified["semi_stable"].copy(),
        }
        result = evaluate_device_authorization(
            expected_features=flat_base,
            current_classified=modified,
            policy=high_policy,
        )
        assert result.state == DeviceState.UNAUTHORIZED
        assert result.is_authorized is False
        assert result.fingerprint_stability == "UNSTABLE_STABLE_CHANGED"
        assert "machine_id" in result.device_changes_detected
        assert "Sensitive event detected" in result.reason_for_rejection
        assert result.reauthorization_allowed is False

    def test_state_reauthorization_required_on_allowed_semi_stable_change(self, baseline_classified, high_policy):
        flat_base = flatten_classified_features(baseline_classified)
        modified = {
            "stable": baseline_classified["stable"].copy(),
            "semi_stable": {
                "disk_uuid": baseline_classified["semi_stable"]["disk_uuid"],
                "hostname": baseline_classified["semi_stable"]["hostname"],
                "network_interface": "aa:bb:cc:dd:ee:ff",  # MAC changed (allowed by policy)
            },
        }
        result = evaluate_device_authorization(
            expected_features=flat_base,
            current_classified=modified,
            policy=high_policy,
        )
        assert result.state == DeviceState.REAUTHORIZATION_REQUIRED
        assert result.is_authorized is False
        assert result.fingerprint_stability == "SEMI_STABLE_CHANGED"
        assert "network_interface" in result.device_changes_detected
        assert result.reauthorization_allowed is True

    def test_state_unauthorized_on_disallowed_semi_stable_change(self, baseline_classified, high_policy):
        flat_base = flatten_classified_features(baseline_classified)
        modified = {
            "stable": baseline_classified["stable"].copy(),
            "semi_stable": {
                "disk_uuid": baseline_classified["semi_stable"]["disk_uuid"],
                "hostname": "unapproved-hostname-change",  # Hostname changed (disallowed by policy)
                "network_interface": baseline_classified["semi_stable"]["network_interface"],
            },
        }
        result = evaluate_device_authorization(
            expected_features=flat_base,
            current_classified=modified,
            policy=high_policy,
        )
        assert result.state == DeviceState.UNAUTHORIZED
        assert result.is_authorized is False
        assert result.fingerprint_stability == "SEMI_STABLE_CHANGED"
        assert "hostname" in result.device_changes_detected
        assert "Policy rejected change" in result.reason_for_rejection
        assert result.reauthorization_allowed is False

    def test_state_unauthorized_on_all_missing_identifiers(self, high_policy):
        empty_classified = {
            "stable": {"machine_id": "UNAVAILABLE", "cpu_model": "UNAVAILABLE"},
            "semi_stable": {
                "disk_uuid": "UNAVAILABLE",
                "hostname": "UNAVAILABLE",
                "network_interface": "UNAVAILABLE",
            },
        }
        result = evaluate_device_authorization(
            expected_features={"machine_id": "12345"},
            current_classified=empty_classified,
            policy=high_policy,
        )
        assert result.state == DeviceState.UNAUTHORIZED
        assert result.fingerprint_stability == "MISSING_IDENTIFIERS"
        assert "All hardware and OS identifiers are UNAVAILABLE" in result.reason_for_rejection


class TestReauthorizationWorkflow:
    def test_reauthorization_successful_with_valid_token(self):
        eval_result = AuthorizationResult(
            state=DeviceState.REAUTHORIZATION_REQUIRED,
            is_authorized=False,
            fingerprint_generation_time_ms=1.5,
            feature_availability={"network_interface": True},
            fingerprint_stability="SEMI_STABLE_CHANGED",
            reason_for_rejection="Network changed",
            device_changes_detected=["network_interface"],
            reauthorization_allowed=True,
        )
        approved = reauthorize_device(eval_result, admin_token="my-secret-admin-token", expected_token="my-secret-admin-token")
        assert approved.state == DeviceState.AUTHORIZED
        assert approved.is_authorized is True
        assert approved.reauthorized_by_admin is True

    def test_reauthorization_failed_with_invalid_token(self):
        eval_result = AuthorizationResult(
            state=DeviceState.REAUTHORIZATION_REQUIRED,
            is_authorized=False,
            fingerprint_generation_time_ms=1.5,
            feature_availability={"network_interface": True},
            fingerprint_stability="SEMI_STABLE_CHANGED",
            reason_for_rejection="Network changed",
            device_changes_detected=["network_interface"],
            reauthorization_allowed=True,
        )
        with pytest.raises(DeviceAuthorizationError, match="Invalid admin reauthorization token"):
            reauthorize_device(eval_result, admin_token="wrong-token", expected_token="my-secret-admin-token")

    def test_reauthorization_rejected_for_unauthorized_state(self):
        eval_result = AuthorizationResult(
            state=DeviceState.UNAUTHORIZED,
            is_authorized=False,
            fingerprint_generation_time_ms=1.5,
            feature_availability={"machine_id": True},
            fingerprint_stability="UNSTABLE_STABLE_CHANGED",
            reason_for_rejection="Stable ID changed",
            device_changes_detected=["machine_id"],
            reauthorization_allowed=False,
        )
        with pytest.raises(DeviceAuthorizationError, match="Cannot reauthorize device in state 'UNAUTHORIZED'"):
            reauthorize_device(eval_result, admin_token="my-secret-admin-token", expected_token="my-secret-admin-token")


class TestInstrumentationMetrics:
    def test_result_contains_all_instrumentation_fields(self):
        result = evaluate_device_authorization()
        res_dict = result.to_dict()

        required_keys = [
            "state",
            "is_authorized",
            "fingerprint_generation_time_ms",
            "feature_availability",
            "fingerprint_stability",
            "reason_for_rejection",
            "device_changes_detected",
            "reauthorization_allowed",
            "reauthorized_by_admin",
        ]
        for key in required_keys:
            assert key in res_dict
        assert isinstance(result.fingerprint_generation_time_ms, float)
        assert isinstance(result.feature_availability, dict)
