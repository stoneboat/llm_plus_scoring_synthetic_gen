"""
Private prediction synthetic text generation (Amin et al. 2024).

Phase 1: package boundary established via pyproject.toml.
Phase 2: dataset / prompt / batching layers extracted into sub-packages.
Phase 3: model backend and generation mechanism extracted into src/backends/
and src/mechanisms/; duplicate compute_max_private_tokens removed from config.
Phase 3.5: privacy boundary strengthened into src/privacy/ package with
explicit bounds, events, analyses, accountants, planning, and reporting layers.
Phase 3.5b: PrivacyEvent now carries a PrivacyBound (not a raw rho scalar);
PrivacyBound ABC added; accountant interface made more guarantee-family-neutral.
Phase 4: runtime / orchestration extracted into src/runtime/; artifact writing
and checkpoint I/O extracted into src/artifacts/.
Phase 5: evaluation layer consolidated into src/evaluation/; BERT fine-tuning
and ICL evaluation logic extracted from src/evaluate.py and the evaluation
script into a reusable package.

Install with ``pip install -e .`` to make ``from src.X import Y`` work
without per-script sys.path manipulation.

See paper/refactor_phase1_report.md and paper/phase2_migration_report.md
for change history.
"""

__version__ = "0.5.0"
