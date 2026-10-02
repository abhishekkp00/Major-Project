"""
tests/unit/test_pii_name_masking.py
====================================
Regression tests for the dynamic grammatical name-masking bug fix in
HybridPIIEngine.mask().

Bug: the replacement string contained a backreference (\\1) that re-injected
the captured name back into the output.  The fix replaces the entire match
(trigger phrase + name) with the plain placeholder "[GIVENNAME]".

These tests use enable_ml=False so they are fast, deterministic, and isolated
from whichever (optional) ML backend happens to be installed.
"""

import pytest
from src.security.pii_engine import HybridPIIEngine

PLACEHOLDER = "[GIVENNAME]"


@pytest.fixture(scope="module")
def engine():
    """Regex-only engine — no ML backend, fast and deterministic."""
    return HybridPIIEngine(enable_ml=False)


# ---------------------------------------------------------------------------
# Positive cases — the name MUST NOT appear in the output
# ---------------------------------------------------------------------------

class TestNameRetentionBugFixed:

    def test_my_name_is_john_doe(self, engine):
        text = "my name is John Doe"
        masked, counts = engine.mask(text)
        assert "John" not in masked, f"First name leaked into output: {masked!r}"
        assert "Doe" not in masked, f"Last name leaked into output: {masked!r}"
        assert PLACEHOLDER in masked, f"Placeholder missing from output: {masked!r}"
        assert counts.get("PERSON", 0) >= 1

    def test_contact_trigger_is_the_manager(self, engine):
        # "John Doe is the manager" has no trigger; use contact to make deterministic
        text = "contact John Doe is the manager"
        masked, counts = engine.mask(text)
        assert "John" not in masked, f"First name leaked: {masked!r}"
        assert "Doe" not in masked, f"Last name leaked: {masked!r}"
        assert PLACEHOLDER in masked, f"Placeholder missing: {masked!r}"

    def test_multiple_names_in_one_string(self, engine):
        text = "my name is Alice Smith and contact Bob Jones for support"
        masked, counts = engine.mask(text)
        for name in ("Alice", "Smith", "Bob", "Jones"):
            assert name not in masked, (
                f"Name '{name}' still present in masked output: {masked!r}"
            )
        assert masked.count(PLACEHOLDER) >= 2, (
            f"Expected at least 2 placeholders, got: {masked!r}"
        )
        assert counts.get("PERSON", 0) >= 2

    def test_name_followed_by_punctuation(self, engine):
        text = "my name is Jane Doe, please contact me."
        masked, counts = engine.mask(text)
        assert "Jane" not in masked, f"Name leaked before punctuation: {masked!r}"
        assert "Doe" not in masked, f"Name leaked before punctuation: {masked!r}"
        assert PLACEHOLDER in masked, f"Placeholder missing: {masked!r}"

    def test_name_at_beginning_of_string_with_trigger(self, engine):
        text = "Dr. Emily Clarke will chair the meeting"
        masked, counts = engine.mask(text)
        assert "Emily" not in masked, f"Name leaked: {masked!r}"
        assert "Clarke" not in masked, f"Name leaked: {masked!r}"
        assert PLACEHOLDER in masked, f"Placeholder missing: {masked!r}"

    def test_name_at_end_of_string(self, engine):
        text = "The patient is Mr. David Lee"
        masked, counts = engine.mask(text)
        assert "David" not in masked, f"Name leaked at end: {masked!r}"
        assert "Lee" not in masked, f"Name leaked at end: {masked!r}"
        assert PLACEHOLDER in masked, f"Placeholder missing: {masked!r}"

    def test_no_name_passthrough(self, engine):
        """A string with no name trigger must be returned unchanged."""
        text = "The weather today is sunny and warm."
        masked, counts = engine.mask(text)
        assert masked == text, (
            f"Name-free string was mutated unexpectedly: {masked!r}"
        )
        assert counts.get("PERSON", 0) == 0

    def test_repeated_matches_all_redacted(self, engine):
        """Every occurrence of a name introduction must be redacted, not just the first."""
        text = (
            "my name is Tom Brown. "
            "I am Tom Brown and contact Tom Brown for details."
        )
        masked, counts = engine.mask(text)
        assert "Tom" not in masked, f"Name 'Tom' still present: {masked!r}"
        assert "Brown" not in masked, f"Name 'Brown' still present: {masked!r}"
        assert masked.count(PLACEHOLDER) >= 3, (
            f"Expected at least 3 placeholders, got: {masked!r}"
        )
        assert counts.get("PERSON", 0) >= 3

    def test_placeholder_not_prefixed_by_original_name(self, engine):
        """Explicitly verify the old bug symptom: name + [GIVENNAME] must not appear."""
        text = "my name is John Doe"
        masked, _ = engine.mask(text)
        # Old buggy output: "my name is John Doe [GIVENNAME]"
        assert "John Doe [GIVENNAME]" not in masked, (
            "Bug regression: original name is still prepended to the placeholder"
        )
        assert "John Doe" not in masked, (
            "Bug regression: original name still present in output"
        )
        assert PLACEHOLDER in masked
