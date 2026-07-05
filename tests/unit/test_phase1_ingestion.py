import pytest
from pathlib import Path
import json

from src.phase1.ingestion import ingest_directory

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_ingest_directory_empty(tmp_dir):
    records = ingest_directory(tmp_dir)
    assert len(records) == 0


def test_ingest_directory_valid(tmp_dir):
    # CSV file
    csv_file = tmp_dir / "data.csv"
    csv_file.write_text("instruction,input,output\nQ1,in1,A1\nQ2,in2,A2\n", encoding="utf-8")

    # JSON file
    json_file = tmp_dir / "data.json"
    json_file.write_text(json.dumps([{"prompt": "Q3", "response": "A3"}]), encoding="utf-8")

    records = ingest_directory(tmp_dir)
    assert len(records) == 3
