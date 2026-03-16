# AGNews Experiment Report

## Run Summary

This report summarizes the experiment run with:

```bash
python scripts/run_experiment.py \
  --dataset agnews \
  --epsilon 3.0 \
  --num_examples 1000 \
  --batch_size 255 \
  --clip_bound 10.0 \
  --temperature 2.0 \
  --public_temperature 1.5 \
  --svt_threshold 0.5 \
  --svt_noise 0.2 \
  --max_private_tokens 64
```

Output file:

- `data/outputs/agnews_eps3.0_s255_20260316_230302.jsonl`

## Configuration Used

- Dataset: `AGNews`
- Number of source examples: `1000`
- Batch size: `255`
- Clip bound: `10.0`
- Private temperature: `2.0`
- Public temperature: `1.5`
- SVT threshold: `0.5`
- SVT noise: `0.2`
- Max private tokens per synthetic example: `64`
- Target epsilon argument: `3.0`
- Actual reported epsilon for this run: `1.3652`
- Delta: `0.001`

## High-Level Result

The run completed successfully and generated `5` synthetic examples total.

Because the implementation partitions the `1000` input examples into stable label-grouped batches of size about `255`, the run produced one synthetic example per resulting batch:

- `World`: 1 batch
- `Sports`: 1 batch
- `Business`: 1 batch
- `Sci/Tech`: 2 batches

This gave the final label distribution:

- `World`: 1
- `Sports`: 1
- `Business`: 1
- `Sci/Tech`: 2

## Privacy And Token Usage

Reported privacy summary:

- `rho_per_token = 0.000961`
- `total_rho = 0.061515`
- `epsilon = 1.3652`
- `delta = 0.001`

Final generation statistics:

- Number of synthetic examples: `5`
- Total tokens generated: `1073`
- Mean tokens per example: `214.6`
- Total private tokens: `263`
- Total public tokens: `810`
- Public token fraction: `0.7549`
- Max private tokens in any example: `64`

## Batch-Level Summary

- Batch 1 (`World`, size 244): `256` total tokens, `49` private, `207` public, `81%` public
- Batch 2 (`Sports`, size 243): `256` total tokens, `34` private, `222` public, `87%` public
- Batch 3 (`Sci/Tech`, size 126): `141` total tokens, `64` private, `77` public, `55%` public
- Batch 4 (`Sci/Tech`, size 145): `164` total tokens, `64` private, `100` public, `61%` public
- Batch 5 (`Business`, size 242): `256` total tokens, `52` private, `204` public, `80%` public

## Interpretation

This run shows that enabling SVT had a large effect:

- Roughly `75%` of generated tokens were public tokens.
- Only `263` of `1073` total tokens consumed privacy budget.
- Several batches produced long outputs while staying below the private-token cap of `64`.

That is a clear contrast with earlier no-SVT runs, where:

- every token was private,
- each example often stopped exactly at the private-token cap,
- outputs were much shorter and more privacy-expensive.

In this run, SVT allowed the model to keep generating with many "free" public tokens once the private/public distributions were considered close enough.

## Qualitative Output Quality

The output texts are longer than the no-SVT baseline and contain more stretches of recognizable English phrases, topic words, and sentence-like fragments. Examples include:

- `World`: words like "EPA", "policy", "union", "societ", "urban", "recession"
- `Sports`: words like "puck", "Glory", "playbook", "jerseys", "wrestler", "ice"
- `Business`: words like "Competition", "Firmware", "Phone", "Launches", "research", "proposal"

However, the generated text is still far from clean natural-language news articles. Common issues include:

- mixed languages and Unicode fragments,
- sudden topic drift,
- formatting artifacts,
- repeated nonsense tokens,
- incomplete syntax.

So the run appears to be an improvement in **quantity** and **effective privacy efficiency**, but not yet a strong result in **text fluency**.

## What Looks Good

- The pipeline ran end-to-end successfully.
- SVT is clearly active and useful.
- Public-token usage is high, which is exactly the mechanism the paper relies on.
- The generated examples are much longer than the no-SVT version.
- The reported epsilon stayed moderate (`1.3652`) even with these longer outputs.

## Remaining Limitations

- Only `5` synthetic examples were produced because `num_examples=1000` with batch size `255` leads to only a few batches.
- Text quality is still noisy and partially nonsensical.
- The actual realized epsilon is well below the CLI target `3.0`, which means this run is still more private than necessary and may be leaving utility on the table.

## Suggested Next Experiments

- Increase `num_examples` if the goal is to produce a larger synthetic dataset.
- Increase utility further by trying:
  - larger effective epsilon,
  - slightly lower private temperature,
  - alternative SVT settings such as `(0.3, 0.1)` or `(0.7, 0.3)`.
- Compare this run directly against a no-SVT run in a small table:
  - number of examples,
  - mean tokens,
  - public token fraction,
  - qualitative fluency.

## Bottom Line

This experiment is a good sign that the implementation is working in the intended regime:

- batching behaves as expected,
- privacy accounting is coherent,
- SVT substantially reduces privacy-paid tokens,
- longer synthetic outputs become possible.

The main remaining gap is text quality, not mechanism activation.
