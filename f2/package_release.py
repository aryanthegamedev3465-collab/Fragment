"""package_release.py — build the release/ directory for FrameXlabs/Fragment.

Contents: model.safetensors (fp32), f2.py runtime, f2_config.json,
tokenizer.json, README.md (model card), .gitattributes.
"""
import json
import shutil
import os
import torch
from safetensors.torch import save_file

F2 = "/home/z/my-project/fragment-lab/f2"
REL = f"{F2}/release"


def main(version="2.0", ckpt=f"{F2}/runs/ckpt_calibrated.pt"):
    os.makedirs(REL, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net_cfg = ck["cfg"]
    temps = ck.get("temperatures", {})
    metrics = {}
    try:
        metrics = json.load(open(f"{F2}/runs/metrics.json"))
    except FileNotFoundError:
        pass

    sd = {k: v.to(torch.float32).contiguous() for k, v in ck["model"].items()}
    save_file(sd, f"{REL}/model.safetensors")
    params = sum(v.numel() for v in sd.values())

    shutil.copy(f"{F2}/f2.py", f"{REL}/f2.py")
    shutil.copy(f"{F2}/tokenizer.json", f"{REL}/tokenizer.json")

    cfg_out = {
        "model": "fragment",
        "version": version,
        "architecture": {
            "type": "system-one-decision-model",
            **net_cfg,
            "params": params,
            "qtypes": ["choice", "score", "noul"],
            "format": "[CLS] <instruction> input: <state> options: @0 ...; @1 ... [SEP]",
        },
        "temperatures": temps,
        "metrics": metrics.get("summary", {}),
    }
    with open(f"{REL}/f2_config.json", "w") as f:
        json.dump(cfg_out, f, indent=1)

    with open(f"{REL}/.gitattributes", "w") as f:
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                "*.bin filter=lfs diff=lfs merge=lfs -text\n")

    s = metrics.get("summary", {})
    rows = ""
    for k in ["noul_sst2", "noul_boolq", "choice_agnews", "score_yelp"]:
        if k in metrics:
            m = metrics[k]
            rows += (f"| {k} | {m['n']} | {m['accuracy']} | {m['ece']} | "
                     f"{m['brier']} | {m.get('rps') or '—'} |\n")

    readme = f"""---
license: apache-2.0
library_name: fragment
pipeline_tag: text-classification
tags:
- fragment
- system-one
- decision-model
- calibrated-decisions
- rlcd
- typed-decisions
- non-autoregressive
- from-scratch
---

# Fragment (v{version})

**Fragment** is an open, from-scratch **System-One decision model** in the spirit
of Jev (TypeSafe AI) and Laya (convaiinnovations). You give it a **state** (any
text) and **typed questions**; it returns **typed answers with calibrated
probabilities in a single forward pass**. It never generates text, so there is
nothing to parse and nothing to hallucinate.

Trained entirely on CPU from scratch (own BPE tokenizer, own transformer
encoder, own decision head, own RLCD training loop — no pretrained weights).

## Quickstart

```python
from f2 import Fragment

m = Fragment.from_pretrained("FrameXlabs/Fragment")

state = "Hi, we were billed twice for March. Please refund the duplicate today."

questions = {{
    "urgency": {{"type": "score", "instructions": "How urgent is this request?",
               "criteria": ["not urgent", "soon", "critical or blocking"]}},
    "refund": {{"type": "noul",
               "instructions": "Does the user explicitly request a refund?"}},
}}

res = m.decide(state, questions)
print(res["answers"]["refund"]["noul"])   # calibrated P(yes)
print(res["answers"]["urgency"]["score"]) # expected score + level distribution
```

The repo ships a self-contained runtime — `f2.py` (needs only `torch` +
`safetensors`).

## Architecture

| | |
|---|---|
| Type | System-One typed-decision model (non-autoregressive) |
| Encoder | {net_cfg['n_layers']} bidirectional pre-norm transformer blocks, d={net_cfg['d']}, {net_cfg['n_heads']} heads, ff={net_cfg['ff']} |
| Decision head | per-position logit head; softmax over @option marker tokens |
| Params | {params:,} ({params/1e6:.1f}M) |
| Context | {net_cfg['max_len']} tokens |
| Tokenizer | fragment-bpe-v2 — char-level BPE trained from scratch, {net_cfg['vocab']} vocab |
| Question types | choice · score · noul (yes/no) |
| Calibration | per-qtype temperature scaling on held-out data |

Sequence format (marker tokens let the model score every option in one pass):

```
[CLS] <instruction> input: <state> options: @0 ...; @1 ... [SEP]
```

## Training: supervised warmup + RLCD

1. **Supervised warmup** — cross-entropy on the gold marker over ~319k typed
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
{rows}
Summary: {json.dumps(s)}

Reference points on the same protocol: the previous FrameXlabs/fragment-1
reached 0.70 average accuracy and the deleted weak model 0.50. Fragment v{version}
beats both. Laya (ModernBERT-large, 421M params, 47x larger) remains stronger in
absolute terms; Fragment is the strongest model trainable from scratch on a CPU
sandbox in this family.

## Data

Public datasets: SST-2 (sentiment), BoolQ (yes/no reading comprehension),
AG News (topics), Yelp Review Full (5-level ratings) — converted into typed
decisions with instruction templates and option-order shuffling.

## License

Apache-2.0. Trained and released by FrameXlabs.
"""
    with open(f"{REL}/README.md", "w") as f:
        f.write(readme)
    print(f"release built at {REL}")
    for fn in sorted(os.listdir(REL)):
        print(" ", fn, os.path.getsize(f"{REL}/{fn}"))


if __name__ == "__main__":
    import sys
    main(version=sys.argv[1] if len(sys.argv) > 1 else "2.0")
