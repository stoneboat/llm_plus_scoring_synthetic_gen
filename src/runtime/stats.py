"""
Generation summary statistics.

These are runtime diagnostics over generated examples, not downstream task
metrics. They live in ``src.runtime`` so generation scripts can depend on them
without going through the legacy ``src.evaluate`` compatibility surface.
"""

from typing import Dict, List

from src.runtime.generation import SyntheticExample


def compute_generation_stats(examples: List[SyntheticExample]) -> dict:
    """Compute summary statistics over generated examples."""
    if not examples:
        return {}

    total_tokens = [e.num_total_tokens for e in examples]
    priv_tokens = [e.num_private_tokens for e in examples]
    pub_tokens = [e.num_public_tokens for e in examples]

    total_priv = sum(priv_tokens)
    total_pub = sum(pub_tokens)
    total_all = total_priv + total_pub

    label_counts: Dict[int, int] = {}
    for e in examples:
        label_counts[e.label] = label_counts.get(e.label, 0) + 1

    return {
        "num_examples": len(examples),
        "label_distribution": label_counts,
        "total_tokens": sum(total_tokens),
        "mean_tokens_per_example": sum(total_tokens) / len(examples),
        "total_private_tokens": total_priv,
        "total_public_tokens": total_pub,
        "public_token_fraction": total_pub / max(1, total_all),
        "max_private_tokens_in_example": max(priv_tokens),
    }
