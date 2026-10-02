from src.phase1.preprocessing import preprocess_record


def test_preprocess_record_instruction():
    raw = {"instruction": " Tell me  ", "input": " \n", "output": " Something. "}
    cleaned = preprocess_record(raw)
    assert cleaned["instruction"] == "Tell me"
    assert cleaned["input"] == ""
    assert cleaned["output"] == "Something."


def test_preprocess_record_unstructured():
    raw = {"text": "   Some text content.   "}
    cleaned = preprocess_record(raw)
    assert cleaned["text"] == "Some text content."


def test_preprocess_record_masks_email_in_instruction_format():
    """Email in instruction/output fields must be replaced by the PII masker."""
    raw = {
        "instruction": "Contact user john.doe@example.com for details",
        "input": "",
        "output": "Send an email to jane.smith@corp.org",
    }
    result = preprocess_record(raw)
    assert "john.doe@example.com" not in result["instruction"], (
        "preprocess_record must mask PII in the 'instruction' field"
    )
    assert "jane.smith@corp.org" not in result["output"], (
        "preprocess_record must mask PII in the 'output' field"
    )


def test_preprocess_record_masks_phone_in_text_format():
    """Phone number in a causal-LM text record must be masked."""
    raw = {"text": "Call me at 555-867-5309 anytime."}
    result = preprocess_record(raw)
    assert "555-867-5309" not in result["text"], (
        "preprocess_record must mask PII in the 'text' field"
    )
