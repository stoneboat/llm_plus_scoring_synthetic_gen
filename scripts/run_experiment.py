#!/usr/bin/env python3
"""
End-to-end experiment runner for private prediction synthetic text generation.

Usage:
    python scripts/run_experiment.py --dataset agnews --epsilon 1.0
    python scripts/run_experiment.py --dataset agnews --epsilon 1.0 --batch_size 127 --svt_threshold 0.5 --svt_noise 0.2
    python scripts/run_experiment.py --dataset agnews --epsilon 1.0 --num_examples 1000 --max_total_tokens 128

Loads a dataset, constructs prompts, runs Algorithm 1, saves synthetic data,
and prints generation statistics.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# Reduce CUDA memory fragmentation, especially with large-vocabulary models
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Add project root to path so src/ is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    PrivacyConfig,
    GenerationConfig,
    ModelConfig,
    DatasetConfig,
    compute_max_private_tokens,
)
from src.privacy_accounting import privacy_report
from src.generate import BatchDescriptor, SyntheticExample, generate_synthetic_dataset
from src.evaluate import compute_generation_stats
from src.datasets.registry import DATASET_CHOICES, get_adapter


def load_dataset_examples(
    dataset_name: str,
    num_examples: int = None,
    cache_dir: str = "data/datasets",
) -> list:
    """Load dataset from HuggingFace and convert to list of dicts."""
    adapter = get_adapter(dataset_name)
    print(f"Loading {adapter.hf_name} (train split)...")
    examples = adapter.load("train", num_examples=num_examples, cache_dir=cache_dir)
    labels = set(e["label"] for e in examples)
    print(f"  Loaded {len(examples)} examples with {len(labels)} labels")
    return examples


def load_model(model_config: ModelConfig):
    """Load the LLM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_path = model_config.model_name_or_path
    if not os.path.isdir(model_path):
        model_path = model_config.hf_model_id
        print(f"Local model not found at {model_config.model_name_or_path}, "
              f"using HF hub: {model_path}")

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_config.torch_dtype,
        device_map=model_config.device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"  Model loaded on {device}, dtype={next(model.parameters()).dtype}")
    return model, tokenizer


def _jsonl_append_line(handle, record: dict) -> None:
    handle.write(json.dumps(record) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _batch_record(ex: SyntheticExample, descriptor: BatchDescriptor) -> dict:
    return {
        "text": ex.text,
        "label": ex.label,
        "label_name": ex.label_name,
        "num_private_tokens": ex.num_private_tokens,
        "num_public_tokens": ex.num_public_tokens,
        "num_total_tokens": ex.num_total_tokens,
        "batch_id": descriptor.batch_id,
        "batch_index": descriptor.batch_index,
        "total_batches": descriptor.total_batches,
    }


def _load_resume_state(
    output_path: str,
) -> Tuple[Optional[dict], List[SyntheticExample], Set[str], Dict[str, BatchDescriptor]]:
    """Load readable records from a checkpointed JSONL file.

    Ignores a truncated or malformed tail line so the file remains usable after
    an interrupted append.
    """
    metadata = None
    batch_examples: Dict[str, List[SyntheticExample]] = {}
    batch_descriptors: Dict[str, BatchDescriptor] = {}
    completed_batch_ids: List[str] = []

    with open(output_path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: ignoring unreadable JSONL tail at line {line_no} in {output_path}")
                break

            if "_metadata" in record:
                metadata = record["_metadata"]
                continue

            if "_batch_complete" in record:
                info = record["_batch_complete"]
                batch_id = info["batch_id"]
                batch_descriptors[batch_id] = BatchDescriptor(
                    batch_id=batch_id,
                    batch_index=info["batch_index"],
                    total_batches=info["total_batches"],
                    label=info["label"],
                    label_name=info["label_name"],
                    batch_size=info["batch_size"],
                )
                completed_batch_ids.append(batch_id)
                continue

            batch_id = record.get("batch_id")
            if batch_id is None:
                continue

            batch_examples.setdefault(batch_id, []).append(SyntheticExample(
                text=record["text"],
                label=record["label"],
                label_name=record["label_name"],
                num_private_tokens=record["num_private_tokens"],
                num_public_tokens=record["num_public_tokens"],
                num_total_tokens=record["num_total_tokens"],
            ))

            batch_index = record.get("batch_index")
            total_batches = record.get("total_batches")
            if batch_index is not None and total_batches is not None:
                batch_descriptors.setdefault(batch_id, BatchDescriptor(
                    batch_id=batch_id,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    label=record["label"],
                    label_name=record["label_name"],
                    batch_size=record.get("source_batch_size", 0),
                ))

    completed_batch_set = set(completed_batch_ids)
    loaded_examples: List[SyntheticExample] = []
    for batch_id in completed_batch_ids:
        loaded_examples.extend(batch_examples.get(batch_id, []))

    return metadata, loaded_examples, completed_batch_set, batch_descriptors


def _write_metadata_header(output_path: str, metadata: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        _jsonl_append_line(f, {"_metadata": metadata})


def _append_completed_batch(
    output_path: str,
    descriptor: BatchDescriptor,
    examples: List[SyntheticExample],
) -> None:
    with open(output_path, "a") as f:
        for ex in examples:
            record = _batch_record(ex, descriptor)
            record["source_batch_size"] = descriptor.batch_size
            _jsonl_append_line(f, record)

        _jsonl_append_line(f, {
            "_batch_complete": {
                "batch_id": descriptor.batch_id,
                "batch_index": descriptor.batch_index,
                "total_batches": descriptor.total_batches,
                "label": descriptor.label,
                "label_name": descriptor.label_name,
                "batch_size": descriptor.batch_size,
                "num_examples": len(examples),
            }
        })


def _build_run_metadata(
    args,
    report: dict,
    delta: float,
    max_priv: int,
    num_source_examples: int,
    output_path: str,
) -> dict:
    return {
        "dataset": args.dataset,
        "epsilon": report["epsilon"],
        "delta": delta,
        "batch_size": args.batch_size,
        "clip_bound": args.clip_bound,
        "temperature": args.temperature,
        "public_temperature": args.public_temperature,
        "svt_threshold": args.svt_threshold,
        "svt_noise": args.svt_noise,
        "top_k_vocab": args.top_k_vocab,
        "max_private_tokens": max_priv,
        "max_total_tokens": args.max_total_tokens,
        "num_source_examples": num_source_examples,
        "seed": args.seed,
        "micro_batch_size": args.micro_batch_size,
        "output_path": output_path,
        "checkpoint_format": 1,
    }


def _metadata_matches_args(metadata: dict, args, delta: float, max_priv: int, num_source_examples: int) -> bool:
    expected = {
        "dataset": args.dataset,
        "delta": delta,
        "batch_size": args.batch_size,
        "clip_bound": args.clip_bound,
        "temperature": args.temperature,
        "public_temperature": args.public_temperature,
        "svt_threshold": args.svt_threshold,
        "svt_noise": args.svt_noise,
        "top_k_vocab": args.top_k_vocab,
        "max_private_tokens": max_priv,
        "max_total_tokens": args.max_total_tokens,
        "num_source_examples": num_source_examples,
        "seed": args.seed,
        "micro_batch_size": args.micro_batch_size,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return True


def _default_output_prefix(args) -> str:
    return f"{args.dataset}_eps{args.epsilon}_s{args.batch_size}_"


def _resolve_output_path(args) -> str:
    if args.output_path:
        return args.output_path

    prefix = _default_output_prefix(args)
    if args.resume:
        if not os.path.isdir(args.output_dir):
            raise FileNotFoundError(
                f"--resume could not find output directory {args.output_dir}"
            )
        matches = sorted(
            os.path.join(args.output_dir, name)
            for name in os.listdir(args.output_dir)
            if name.startswith(prefix) and name.endswith(".jsonl")
        )
        if not matches:
            raise FileNotFoundError(
                f"--resume could not find an existing checkpoint matching {prefix}*.jsonl"
            )
        return matches[-1]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(args.output_dir, f"{prefix}{timestamp}.jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Private prediction synthetic text generation"
    )

    parser.add_argument(
        "--dataset", type=str, default="agnews",
        choices=DATASET_CHOICES,
        help="Dataset to generate synthetic data for",
    )
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Subsample N examples from the dataset (default: all)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Target (epsilon, delta)-DP epsilon")
    parser.add_argument("--delta", type=float, default=None,
                        help="Target delta (default: 1/n)")
    parser.add_argument("--batch_size", type=int, default=255,
                        help="Expected batch size s")
    parser.add_argument("--clip_bound", type=float, default=10.0,
                        help="Logit clipping bound c")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Sampling temperature tau for private tokens")
    parser.add_argument("--public_temperature", type=float, default=1.5,
                        help="Sampling temperature for public tokens")
    parser.add_argument("--svt_threshold", type=float, default=float("-inf"),
                        help="SVT threshold theta (-inf to disable)")
    parser.add_argument("--svt_noise", type=float, default=0.2,
                        help="SVT Laplace noise sigma")
    parser.add_argument("--max_private_tokens", type=int, default=None,
                        help="Max private tokens r (default: computed from budget)")
    parser.add_argument("--max_total_tokens", type=int, default=256,
                        help="Absolute max output length")
    parser.add_argument("--model_path", type=str, default="data/models/gemma-2-2b-it",
                        help="Path to local model or HF model ID")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--output_dir", type=str, default="data/outputs",
                        help="Directory for output files")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Explicit output JSONL path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing checkpointed output file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--micro_batch_size", type=int, default=32,
                        help="Prompts per GPU forward pass (reduce if OOM, default 32)")
    parser.add_argument("--top_k_vocab", type=int, default=0,
                        help="Restrict sampling to top-k of public logits (0=off, "
                             "paper uses 1024 at high temperature)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run BERT fine-tuning evaluation after generation")
    parser.add_argument("--bert_epochs", type=int, default=5,
                        help="BERT fine-tuning epochs (used with --evaluate)")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Cap test examples for eval (default: all)")

    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    examples = load_dataset_examples(
        args.dataset,
        num_examples=args.num_examples,
        cache_dir="data/datasets",
    )

    # Configure privacy
    delta = args.delta if args.delta else 1.0 / len(examples)

    svt_noise = args.svt_noise if args.svt_threshold != float("-inf") else None

    if args.max_private_tokens is not None:
        max_priv = args.max_private_tokens
    else:
        max_priv = compute_max_private_tokens(
            args.epsilon, delta, args.batch_size,
            args.clip_bound, args.temperature, svt_noise,
        )
        print(f"Computed max_private_tokens = {max_priv} for epsilon={args.epsilon}")

    privacy_config = PrivacyConfig(
        target_epsilon=args.epsilon,
        delta=delta,
        clip_bound=args.clip_bound,
        temperature=args.temperature,
        public_temperature=args.public_temperature,
        svt_threshold=args.svt_threshold,
        svt_noise=args.svt_noise,
    )

    gen_config = GenerationConfig(
        batch_size=args.batch_size,
        max_private_tokens=max_priv,
        max_total_tokens=args.max_total_tokens,
        top_k_vocab=args.top_k_vocab,
    )

    # Print privacy report
    report = privacy_report(
        max_priv, args.clip_bound, args.batch_size,
        args.temperature, delta, svt_noise,
    )
    print(f"\n{'='*60}")
    print(f"Privacy configuration:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # Load model
    model_config = ModelConfig(
        model_name_or_path=args.model_path,
        device=args.device,
    )
    model, tokenizer = load_model(model_config)

    # Prepare output/checkpoint path
    output_path = _resolve_output_path(args)

    metadata = _build_run_metadata(
        args=args,
        report=report,
        delta=delta,
        max_priv=max_priv,
        num_source_examples=len(examples),
        output_path=output_path,
    )

    resumed_examples: List[SyntheticExample] = []
    completed_batch_ids: Set[str] = set()
    existing_metadata = None

    if args.resume:
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"--resume requires an existing output file, but {output_path} was not found"
            )
        existing_metadata, resumed_examples, completed_batch_ids, _ = _load_resume_state(output_path)
        if existing_metadata is None:
            raise ValueError(f"Cannot resume from {output_path}: metadata header is missing")
        if not _metadata_matches_args(existing_metadata, args, delta, max_priv, len(examples)):
            raise ValueError(
                f"Checkpoint metadata in {output_path} does not match the current command"
            )
        print(f"Resuming from {output_path}")
        print(f"  Found {len(completed_batch_ids)} completed batches "
              f"and {len(resumed_examples)} saved synthetic examples")
    else:
        if os.path.exists(output_path):
            raise FileExistsError(
                f"Refusing to overwrite existing output file: {output_path}. "
                "Use --output_path to choose a different file or --resume to continue."
            )
        _write_metadata_header(output_path, metadata)

    # Generate
    print(f"\nStarting generation...")
    t0 = time.time()

    synthetic_data = generate_synthetic_dataset(
        model, tokenizer, examples,
        dataset_name=args.dataset,
        privacy_config=privacy_config,
        gen_config=gen_config,
        text_column="text",
        label_column="label",
        device=args.device,
        verbose=True,
        micro_batch_size=args.micro_batch_size,
        completed_batch_ids=completed_batch_ids,
        batch_callback=lambda descriptor, batch_examples: _append_completed_batch(
            output_path, descriptor, batch_examples,
        ),
    )

    if resumed_examples:
        synthetic_data = resumed_examples + synthetic_data

    elapsed = time.time() - t0
    print(f"\nGeneration took {elapsed:.1f}s")

    print(f"Saved {len(synthetic_data)} examples to {output_path}")

    # Stats
    stats = compute_generation_stats(synthetic_data)
    print(f"\nGeneration statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Optional downstream evaluation
    if args.evaluate:
        from src.evaluate import finetune_and_evaluate

        adapter = get_adapter(args.dataset)
        print(f"\nLoading test set: {adapter.hf_name} (test)...")
        test_examples = adapter.load(
            "test",
            num_examples=args.max_test,
            cache_dir="data/datasets",
        )
        test_texts = [e["text"] for e in test_examples]
        test_labels = [e["label"] for e in test_examples]
        print(f"  {len(test_texts)} test examples")

        synth_texts = [e.text for e in synthetic_data]
        synth_labels = [e.label for e in synthetic_data]

        eval_results = finetune_and_evaluate(
            synth_texts, synth_labels,
            test_texts, test_labels,
            num_labels=adapter.num_labels,
            epochs=args.bert_epochs,
            device=args.device,
        )
        print(f"\nDownstream accuracy: {eval_results['accuracy']:.4f}")
        print(f"Macro F1: {eval_results['macro_f1']:.4f}")

        eval_path = output_path.replace(".jsonl", "_eval.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Eval results saved to {eval_path}")


if __name__ == "__main__":
    main()
