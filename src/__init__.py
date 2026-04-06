"""
Private prediction synthetic text generation (Amin et al. 2024).

Phase 1: package boundary established via pyproject.toml.
Phase 2: dataset / prompt / batching layers extracted into sub-packages.

Install with ``pip install -e .`` to make ``from src.X import Y`` work
without per-script sys.path manipulation.

See paper/refactor_phase1_report.md and paper/phase2_migration_report.md
for change history.
"""

__version__ = "0.2.0"
