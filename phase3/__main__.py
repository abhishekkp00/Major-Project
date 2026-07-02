"""Allows ``python -m phase3`` to invoke the CLI entry-point."""
from .main import main
import sys

sys.exit(main())
