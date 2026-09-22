---
license: apache-2.0
library_name: fragment
pipeline_tag: text-classification
tags:
- fragment-1
- system-one
- decision-model
- calibrated-decisions
- rlcd
- typed-decisions
- non-autoregressive
- classification
- scoring
- routing
- tiny-model
- from-scratch
datasets:
- stanfordnlp/sst2
- fancyzhx/ag_news
- google/boolq
- fancyzhx/yelp_review_full
---

# fragment-1 (v2.0)

**A 9.0M-parameter System-One decision model.** Give it a **state** (any text) and **typed questions**; it returns **typed answers with calibrated probabilities in a single forward pass** — no autoregression, no text generation, no chain-of-thought, nothing to parse and nothing to hallucinate. Trained entirely from scratch on a 2-thread cloud CPU (own BPE tokenizer, own encoder, own RLCD loop — no pretrained weights, no GPU), which is the point: small enough to run anywhere, including a browser tab.

<p align="center">
  <img src="https://raw.githubusercontent.com/aryanthegamedev3465-collab/Fragment/main/assets/benchmark_headtohead.png" alt="fragment-1 versus the deleted weak Fragment v1.1: accuracy and calibration on SST-2, AG News and Yelp-5" width="100%" />
</p>

> **Live demo → [FrameXlabs/fragment-demo](https://huggingface.co/spaces/FrameXlabs/fragment-demo)** — a fully static Space: the model downloads into your browser and every `decide()` runs locally. No server, no GPU, nothing leaves your machine.

**One name, two versions.** This repo keeps the name `fragment-1` forever. This release **replaced v1 in place** — better weights at the same URL, one model, no graveyard:

| version | params | status | notes |
|---|---|---|---|
| v1 | 4.96M | superseded by this release | SST-2 + AG News + Yelp-5 · supervised + RLCD + calibration |
| **v2.0** (this checkpoint) | 9.0M | **live** | adds BoolQ · 326k items · supervised + RLCD + calibration |

## The three question types

| type | question | answer |
|---|---|---|
| `choice` | which of these options? | the option + a probability for every option |
| `score` | where on this rubric? | expected level + the full distribution |
| `noul` | is this true? | a single calibrated probability |

Every question in a call is answered in **one pass**; options and rubrics are defined at request time, so new schemas need no retraining.

## Quickstart

The repo ships a self-contained runtime — `f2.py` (needs only `torch` + `safetensors`):

```python
from f2 import Fragment

m = Fragment.from_pretrained("FrameXlabs/fragment-1")  # v2.0 — same name as v1, replaced in place

state = "Hi, we were billed twice for March. Please refund the duplicate today."

questions = {
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
               "criteria": ["not urgent", "soon", "critical or blocking"]},
    "refund": {"type": "noul",
               "instructions": "Does the user explicitly request a refund?"},
}

res = m.decide(state, questions)
print(res["answers"]["refund"]["noul"])   # calibrated P(yes)
print(res["answers"]["urgency"]["score"]) # expected level + level distribution
```

## Architecture

| | |
|---|---|
| Type | System-One typed-decision model (non-autoregressive) |
| Encoder | 6 bidirectional pre-norm transformer blocks, d=256, 4 heads, ff=1024 |
| Decision head | per-position logit head; softmax over @option marker tokens |
| Params | 8,982,785 (9.0M) |
| Context | 192 tokens (input text is budgeted so the options always survive truncation) |
| Tokenizer | fragment-bpe-v2 — from-scratch BPE, 16384 vocab |
| Calibration | per-qtype temperature scaling on held-out data |
| Format | `[CLS] <instruction> input: <state> options: @0 ...; @1 ... [SEP]` |

## Training: supervised warmup + RLCD

1. **Supervised warmup** — cross-entropy on the gold marker over 326k typed
   items built from SST-2, AG News, BoolQ and Yelp Review Full, with
   instruction paraphrases and choice-option shuffling (anti position-bias).
   bf16 autocast, AdamW, cosine schedule.
2. **RLCD** — reinforcement learning for calibrated decisions: GRPO-style
   Gaussian logit-noise exploration (G=6), group-normalized advantage, and
   strictly proper scoring rules as reward (log score for choice/noul; RPS for
   ordinal score questions). Honest probabilities are the unique reward
   maximiser.
3. **Calibration** — per-question-type temperature scaling fitted on a held-out
   calibration split (NLL for choice/noul, RPS for score).

## Evaluation (held-out: official validation/test splits, never trained on)

| task | n | accuracy | ECE | Brier | RPS |
|---|---|---|---|---|---|

Summary: {}

Reference points on the same protocol: fragment-1 **v1** (the previous release under this same
name, superseded by v2.0) reached 0.70 average accuracy and the deleted weak model 0.50.
fragment-1 v2.0 beats both. Laya (ModernBERT-large, 421M params, ~47x larger) remains
stronger in absolute terms; fragment-1 v2.0 is the strongest model trainable from scratch
on a CPU sandbox in this family.

## Honest limits

- **Four in-domain task families only:** SST-2-style sentiment, AG News-style topics,
  BoolQ-style reading-comprehension yes/no, Yelp-style ratings. Not a general NLU system.
- **`score` is the weakest primitive** — ordinal rubric questions stay hard at this scale;
  calibration on them is what RLCD + temperature scaling target.
- **Out-of-distribution text degrades quietly:** watch the probabilities, gate on them, and
  fine-tune before trusting picks in a new domain.
- **English only.** No instruction-following, no generation. It makes decisions.

## Links

- **GitHub:** https://github.com/aryanthegamedev3465-collab/Fragment (runtime, benchmarks, pipeline)
- **Live demo:** https://huggingface.co/spaces/FrameXlabs/fragment-demo (runs in your browser)
- **Benchmarks:** [BENCHMARKS.md](https://github.com/aryanthegamedev3465-collab/Fragment/blob/main/BENCHMARKS.md)

Apache-2.0 · FrameXlabs — students with no GPUs.
