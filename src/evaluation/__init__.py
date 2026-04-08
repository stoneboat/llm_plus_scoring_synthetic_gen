"""
Evaluation layer for private synthetic text generation.

Phase 5: extracted and consolidated from src/evaluate.py and
scripts/evaluate_downstream.py.

Owns:
- evaluation-time data loading (test set, real train baseline, synthetic output)
- BERT fine-tuning evaluation
- ICL evaluation + prompt construction
- result saving helpers

Public surface
--------------
Data loading:
    load_test_set           — load real test split via dataset adapter
    load_real_train         — load real train split for baseline comparison
    load_synthetic_for_eval — load synthetic JSONL and extract text/label lists

Fine-tuning:
    finetune_bert           — consolidated BERT fine-tuning evaluator

ICL:
    build_icl_prompt        — build a balanced few-shot ICL prompt
    icl_evaluate            — run full ICL evaluation loop with a local LM

Results:
    save_eval_results       — write evaluation result dict to JSON
"""

from src.evaluation.data import (      # noqa: F401
    load_test_set,
    load_real_train,
    load_synthetic_for_eval,
)
from src.evaluation.finetune import finetune_bert  # noqa: F401
from src.evaluation.icl import (       # noqa: F401
    build_icl_prompt,
    icl_evaluate,
)
from src.evaluation.results import save_eval_results  # noqa: F401

__all__ = [
    "load_test_set",
    "load_real_train",
    "load_synthetic_for_eval",
    "finetune_bert",
    "build_icl_prompt",
    "icl_evaluate",
    "save_eval_results",
]
