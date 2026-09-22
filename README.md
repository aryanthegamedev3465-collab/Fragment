# Fragment

**Fragment** is a from-scratch **System-One decision model** by
[FrameXlabs](https://huggingface.co/FrameXlabs) — the open, CPU-trained member
of the Jev (TypeSafe AI) / Laya (convaiinnovations) family of non-autoregressive
decision models.

Give it a **state** (any text) and **typed questions** (choice / score / noul);
it returns **typed answers with calibrated probabilities in a single forward
pass**. No text generation, nothing to parse, nothing to hallucinate.

- Model card + weights: https://huggingface.co/FrameXlabs/Fragment
- Current roadmap progress: 0/50
- Held-out benchmark summary: {}

## Architecture

pending first release

## Pipeline

```
prepare_data.py     typed items from SST-2 / BoolQ / AG News / Yelp (+extras)
build_tokenizer.py  from-scratch BPE (16,384 vocab, atomic @option markers)
encode_data.py      compact numpy cache
train_supervised.py cross-entropy on gold markers (bf16, CPU)
train_rlcd.py       RLCD: GRPO-style noise exploration, proper scoring rules
calibrate.py        per-qtype temperature scaling
evaluate.py         held-out benchmark (acc / ECE / Brier / RPS)
improve.py          one improvement round (promote if better)
drive.py            roadmap driver — 50 long-running tasks
release_hf.py       Hub release (one model remains: FrameXlabs/Fragment)
```

## Quickstart

```python
import sys; sys.path.insert(0, "model")
from f2 import Fragment

m = Fragment("./model")
res = m.decide(
    state="We were billed twice for March. Please refund the duplicate today.",
    questions={"refund": {"type": "noul",
                           "instructions": "Does the user request a refund?"}})
print(res["answers"]["refund"]["noul"])
```

## Reproduce

```
python3 f2/prepare_data.py
python3 f2/build_tokenizer.py
python3 f2/encode_data.py
python3 f2/train_supervised.py
python3 f2/train_rlcd.py
python3 f2/calibrate.py
python3 f2/evaluate.py
```

Apache-2.0. Trained and released by FrameXlabs.
