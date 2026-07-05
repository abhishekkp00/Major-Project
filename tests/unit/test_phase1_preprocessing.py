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
