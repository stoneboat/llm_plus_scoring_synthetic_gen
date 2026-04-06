"""
Pytest configuration for the llm-pe-synth test suite.

Adds the project root to sys.path so that ``from src.X import Y`` works
whether or not the package has been installed with ``pip install -e .``.
This is intentionally belt-and-suspenders: once the package is installed
the sys.path insertion is harmless.
"""

import os
import sys

# Project root is the parent of the tests/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
