"""
Run-metadata construction for checkpointed generation runs.

``build_run_metadata`` assembles the dict that is written as the
``_metadata`` header of every checkpoint JSONL file.  The schema is
checkpoint_format=1; all field names must remain stable across runs so that
``load_resume_state`` / compatibility checks can compare metadata dicts.
"""


def build_run_metadata(
    dataset: str,
    epsilon: float,
    delta: float,
    batch_size: int,
    clip_bound: float,
    temperature: float,
    public_temperature: float,
    svt_threshold: float,
    svt_noise: float,
    top_k_vocab: int,
    max_private_tokens: int,
    max_total_tokens: int,
    num_source_examples: int,
    seed: int,
    micro_batch_size: int,
    output_path: str,
) -> dict:
    """Assemble the run-metadata payload for a checkpoint JSONL header.

    All parameters are explicit (no argparse namespace) so that this function
    is usable from scripts, sweep runners, and tests alike.

    Args:
        dataset: dataset name (e.g. ``"agnews"``).
        epsilon: target epsilon value from the privacy report.
        delta: resolved delta (1/n if not supplied by the user).
        batch_size: expected batch size *s*.
        clip_bound: logit clipping bound *c*.
        temperature: private-token sampling temperature *τ*.
        public_temperature: public-token sampling temperature.
        svt_threshold: SVT threshold *θ* (``float("-inf")`` if disabled).
        svt_noise: SVT Laplace noise *σ*.
        top_k_vocab: top-k vocabulary restriction (0 = off).
        max_private_tokens: per-batch private-token budget *r*.
        max_total_tokens: absolute maximum output length.
        num_source_examples: number of source examples used (for compat check).
        seed: random seed.
        micro_batch_size: GPU micro-batch size for forward passes.
        output_path: the output file path written into metadata (for traceability).

    Returns:
        Metadata dict with ``checkpoint_format=1``.
    """
    return {
        "dataset": dataset,
        "epsilon": epsilon,
        "delta": delta,
        "batch_size": batch_size,
        "clip_bound": clip_bound,
        "temperature": temperature,
        "public_temperature": public_temperature,
        "svt_threshold": svt_threshold,
        "svt_noise": svt_noise,
        "top_k_vocab": top_k_vocab,
        "max_private_tokens": max_private_tokens,
        "max_total_tokens": max_total_tokens,
        "num_source_examples": num_source_examples,
        "seed": seed,
        "micro_batch_size": micro_batch_size,
        "output_path": output_path,
        "checkpoint_format": 1,
    }
