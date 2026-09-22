"""Reproduce the fragment-1 held-out benchmark from the README.

Downloads the three public test sets from the Hugging Face Hub, samples them
exactly like the September run (seed 42, n=2000 for AG News and Yelp), runs
fragment-1 over them and prints accuracy / ECE / Brier / RPS plus timing.

The numbers in README.md and BENCHMARKS.md came from this script (run on a
2-thread cloud vCPU). If your hardware differs, expect the timing to differ;
the accuracy numbers should match to rounding.

Usage:
    python3 benchmarks/bench_fragment1.py --model FrameXlabs/fragment-1

Data sources (standard test/validation splits, downloaded as parquet):
    stanfordnlp/sst2              validation  (872 rows)
    fancyzhx/ag_news              test        (7,600 rows, sampled to 2,000)
    fancyzhx/yelp_review_full     test        (50,000 rows, sampled to 2,000)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, for fragment1.py
from fragment1 import Fragment1  # noqa: E402

DATA = os.path.join(HERE, "data")
rng = np.random.RandomState(42)

SOURCES = {
    "sst2_val": ("stanfordnlp/sst2", "validation"),
    "agnews_test": ("fancyzhx/ag_news", "test"),
    "yelp_test": ("fancyzhx/yelp_review_full", "test"),
}


def fetch_parquet(name):
    """Download the exact split parquet from the HF Hub if not already local."""
    path = os.path.join(DATA, f"{name}.parquet")
    if os.path.exists(path):
        return path
    repo, split = SOURCES[name]
    os.makedirs(DATA, exist_ok=True)
    import urllib.request
    url = f"https://huggingface.co/api/datasets/{repo}"
    with urllib.request.urlopen(url) as r:
        sib = json.load(r).get("siblings", [])
    cand = [s["rfilename"] for s in sib
            if s["rfilename"].endswith(".parquet") and f"/{split}/" in s["rfilename"]]
    if not cand:
        raise RuntimeError(f"no parquet found for {repo} split {split}")
    u = f"https://huggingface.co/datasets/{repo}/resolve/main/{cand[0]}"
    print(f"downloading {name}: {u}")
    import subprocess
    subprocess.run(["curl", "-sL", "-o", path, u], check=True)
    return path


# ---------------------------------------------------------------- metrics

def ece15(probs, correct, n_bins=15):
    conf = np.max(probs, axis=1)
    acc = correct.astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.sum() / len(conf) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def rps(probs, gold):
    K = probs.shape[1]
    cp = np.cumsum(probs, axis=1)
    cy = np.zeros_like(cp)
    cy[np.arange(len(gold)), gold] = 1.0
    cy = np.cumsum(cy, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (K - 1)))


def brier(probs, gold):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(gold)), gold] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def summarize(probs, gold):
    probs = np.asarray(probs)
    gold = np.asarray(gold)
    pred = probs.argmax(axis=1)
    correct = pred == gold
    return {"n": int(len(gold)),
            "accuracy": round(float(correct.mean()), 4),
            "ece": round(ece15(probs, correct), 4),
            "brier": round(brier(probs, gold), 4),
            "rps": round(rps(probs, gold), 4) if probs.shape[1] > 2 else None}


# ---------------------------------------------------------------- eval

TEMPLATES = {
    "noul": ("is the sentiment positive", "",
             ["@no", "@yes"]),
    "choice": ("which topic", "world sports business technology",
               ["@0 World", "@1 Sports", "@2 Business", "@3 Technology"]),
    "score": ("star rating", "1 to 5 stars",
              ["@0 1-star", "@1 2-star", "@2 3-star", "@3 4-star", "@4 5-star"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FrameXlabs/fragment-1",
                    help="local dir with fragment-final.pt, or an HF repo id")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()

    m = Fragment1.from_pretrained(args.model)

    sst2 = pd.read_parquet(fetch_parquet("sst2_val")).rename(columns={"sentence": "text"})
    ag = pd.read_parquet(fetch_parquet("agnews_test"))
    yelp = pd.read_parquet(fetch_parquet("yelp_test"))
    ag_s = ag.sample(n=2000, random_state=42)
    yelp_s = yelp.sample(n=2000, random_state=42)
    print(f"sst2={len(sst2)} agnews={len(ag_s)} yelp={len(yelp_s)}")

    res = {}
    t0 = time.time()
    for task, df, labelcol in [("noul", sst2, "label"),
                               ("choice", ag_s, "label"),
                               ("score", yelp_s, "label")]:
        inst, crit, opts = TEMPLATES[task]
        items = [m.encode_item(inst, crit, str(t), opts) for t in df["text"].tolist()]
        probs = m.probs(task, items)
        res[task] = summarize(probs, df[labelcol].tolist())
    res["_time_s"] = round(time.time() - t0, 1)

    print(json.dumps(res, indent=1))
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "results": res}, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
