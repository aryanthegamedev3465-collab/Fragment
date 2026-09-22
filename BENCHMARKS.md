# Benchmarks — what we measured, exactly how

Everything on this page was run by us, on our sandbox (2-thread cloud vCPU,
CPU-only, PyTorch). Raw result files live next to this document:
[`head_to_head.json`](head_to_head.json) is the unedited output of the run the
README chart is drawn from. The reproduction script is
[`bench_fragment1.py`](bench_fragment1.py).

## Setup

* **Machine:** 2-thread cloud vCPU, no GPU, PyTorch CPU wheels
* **Model:** `FrameXlabs/fragment-1` (4.96M params)
* **Data:** standard public splits, never trained on
  * SST-2 — `stanfordnlp/sst2`, validation split, all 872 rows
  * AG News — `fancyzhx/ag_news`, test split, 2,000 of 7,600 rows (`sample(n=2000, random_state=42)`)
  * Yelp-5 — `fancyzhx/yelp_review_full`, test split, 2,000 of 50,000 rows (`sample(n=2000, random_state=42)`)
* **Templates** (identical to training-time item rendering):

```
noul:   instruction "is the sentiment positive", options @no / @yes
choice: instruction "which topic", criteria "world sports business technology",
        options @0 World; @1 Sports; @2 Business; @3 Technology
score:  instruction "star rating", criteria "1 to 5 stars",
        options @0 1-star; @1 2-star; @2 3-star; @3 4-star; @4 5-star
```

* **Metrics:** accuracy (argmax), ECE with 15 bins, Brier (multiclass, sums over classes), RPS (ranked probability score, ordinal tasks only)

## fragment-1 results (the run behind the README table)

| task | n | accuracy | ECE | Brier | RPS |
|---|---|---|---|---|---|
| `noul` SST-2 | 872 | 0.7672 | 0.0466 | 0.3201 | — |
| `choice` AG News | 2,000 | 0.8835 | 0.0183 | 0.1630 | 0.0437 |
| `score` Yelp-5 | 2,000 | 0.4525 | 0.0412 | 0.6531 | 0.1249 |
| **mean** | | **0.7011** | **0.0354** | | |

Batched (64 at a time): 4,872 questions in 39.2 s on an idle machine ≈ 8 ms per
question.

## The head-to-head (why one model got deleted)

The account had two models and we only wanted to keep the stronger one, so we
ran both over the identical data in the same session. `Fragment v1.1` was the
older model (4-block, 4.9M params); it lost on every task and every calibration
metric, so it was deleted. fragment-1 stayed. Unedited numbers:

| | noul acc | noul ECE | choice acc | choice ECE | score acc | score ECE | total time |
|---|---|---|---|---|---|---|---|
| **fragment-1** (kept) | **0.7672** | 0.0466 | **0.8835** | 0.0183 | **0.4525** | 0.0412 | 39.2 s |
| Fragment v1.1 (deleted) | 0.6216 | 0.0293 | 0.5810 | 0.0769 | 0.3110 | 0.0136 | 72.2 s |

v1.1 posted slightly better ECE on two tasks, but it was losing 14+ accuracy
points everywhere while barely beating random on `score`. Being well-calibrated
around wrong answers is not a virtue. It's gone, with a local backup.

## Fragment v2 in-training smoke numbers

Not benchmarks — a sanity check on the current checkpoint while training runs
(v2 is mid-roadmap-task-1, so weights are half-baked). Measured on the same
2-thread CPU **while the trainer was using it**, which matters:

* single question, batch 1: median 96 ms (2-option noul), 92 ms (4-option choice), 85 ms (5-option score)
* batch of 64: 2,923 ms ≈ 22 questions/second

Real v2 evaluation numbers land when roadmap task 4 (`evaluate.py`) runs against
the finished checkpoint. They'll be added here the same day, whether they're
good or not.

## Comparisons with Laya and Jev (published numbers, not ours)

We did not run Laya and we have no TypeSafe API access. Everything in this
section is what those projects published about themselves, on their own
evaluation setups, which are not the same as ours:

* Jev 1.13.0 (TypeSafe, published): typed-decisions 0.727, AG News 0.910, ECE 0.246, p50 236–276 ms (third-party measured)
* Laya (convaiinnovations, published): typed-decisions 0.766, AG News 0.950, ECE 0.081 after temperature fitting, ~33 ms/question on a T4, 421M params (ModernBERT-large)

Our AG News 0.884 is on the standard test split; theirs are on their own
evaluation of the same task. Close enough to compare in spirit, not in
precision. The typed-decisions benchmark they use has not been run against
Fragment at all yet.

## Reproducing

```bash
pip install torch pandas numpy pyarrow huggingface_hub
python3 benchmarks/bench_fragment1.py --model FrameXlabs/fragment-1
```

Downloads the three public splits, samples with the same seed, runs the model,
prints accuracy / ECE / Brier / RPS and timing. On the same data you should get
the accuracy numbers above to rounding; timing will scale with your hardware.
