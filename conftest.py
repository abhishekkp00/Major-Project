"""
Root-level conftest.py — adds the project root to sys.path so pytest can
resolve both the `phase3` package and other project modules regardless of
how pytest is invoked (from the project root or from the tests/ directory).
"""
import sys
from pathlib import Path

# Ensure the project root is at the front of sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
