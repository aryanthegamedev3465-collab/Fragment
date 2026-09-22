"""evaluate.py — held-out benchmark on official validation/test splits.

Protocol (identical to the head-to-head bench used to rank the old models):
  noul   : SST-2 validation (872) + BoolQ validation (1000)
  choice : AG News test (2000 sample, seed 42)
  score  : Yelp-5 test (2000 sample, seed 42)
Metrics: accuracy, ECE(15-bin), Brier, RPS. Uses calibrated temperatures.
Writes runs/metrics.json.
"""
import json
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, "/home/z/my-project/fragment-lab/f2")
from f2model import F2Net, F2Tokenizer, build_item

torch.set_num_threads(2)
F2 = "/home/z/my-project/fragment-lab/f2"
DATA = "/home/z/my-project/fragment-lab/data"

NOUL_OPTS = ["@no: no, the statement does not hold",
             "@yes: yes, the statement holds"]
AG_OPTS = ["@0 World: world news and international events",
           "@1 Sports: sports results and athletics",
           "@2 Business: business, economy, stocks and finance",
           "@3 Technology: technology, computers, internet and science"]
YELP_OPTS = ["@0 1-star: very negative", "@1 2-star: negative",
             "@2 3-star: mixed or neutral", "@3 4-star: positive",
             "@4 5-star: very positive"]

TASKS = {
    "noul_sst2": {"task": "noul", "inst": "is the sentiment of this text positive",
                  "opts": NOUL_OPTS, "gold_map": lambda r: int(r["label"])},
    "noul_boolq": {"task": "noul", "inst": None,  # question is the instruction
                   "opts": NOUL_OPTS, "gold_map": lambda r: 1 if bool(r["answer"]) else 0},
    "choice_agnews": {"task": "choice", "inst": "which topic does this text belong to",
                      "opts": AG_OPTS, "gold_map": lambda r: int(r["label"])},
    "score_yelp": {"task": "score", "inst": "how many stars does this review give",
                   "opts": YELP_OPTS, "gold_map": lambda r: int(r["label"])},
}


def ece15(probs, correct, n_bins=15):
    conf = np.max(probs, 1)
    acc = correct.astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / len(conf) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def rps_np(probs, gold):
    K = probs.shape[1]
    cp = np.cumsum(probs, 1)
    cy = np.zeros_like(cp)
    cy[np.arange(len(gold)), gold] = 1.0
    cy = np.cumsum(cy, 1)
    return float(np.mean(np.sum((cp - cy) ** 2, 1) / (K - 1)))


def brier_np(probs, gold):
    K = probs.shape[1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(gold)), gold] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, 1)))


@torch.no_grad()
def run_task(net, tok, temps, name, spec, rows, textcol="text"):
    items, golds = [], []
    for r in rows:
        text = str(r[textcol])
        inst = spec["inst"] if spec["inst"] is not None else str(r.get("question", "")).strip().rstrip("?")
        seq, pos = build_item(tok, inst, text, spec["opts"])
        items.append((torch.tensor([seq]), pos))
        golds.append(spec["gold_map"](r))
    logits_rows = []
    B = 64
    for i in range(0, len(items), B):
        chunk = items[i: i + B]
        maxlen = max(len(c[0][0]) for c in chunk)
        ids = torch.zeros(len(chunk), maxlen, dtype=torch.long)
        for j, (t, _p) in enumerate(chunk):
            ids[j, : len(t[0])] = t[0]
        logits = net(ids)
        for j, (_t, pos) in enumerate(chunk):
            logits_rows.append(logits[j, pos].tolist())
    T = temps.get(spec["task"], 1.0)
    probs = np.stack([np.exp(np.array(l) / T) for l in logits_rows])
    probs = probs / probs.sum(1, keepdims=True)
    gold = np.array(golds)
    correct = probs.argmax(1) == gold
    return {"n": int(len(gold)),
            "accuracy": round(float(correct.mean()), 4),
            "ece": round(ece15(probs, correct), 4),
            "brier": round(brier_np(probs, gold), 4),
            "rps": round(rps_np(probs, gold), 4) if probs.shape[1] > 2 else None}


def main(ckpt=f"{F2}/runs/ckpt_calibrated.pt"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = F2Net()
    net.load_state_dict(ck["model"])
    net.eval()
    tok = F2Tokenizer(f"{F2}/tokenizer.json")
    temps = ck.get("temperatures", {})

    sst2 = pd.read_parquet(f"{DATA}/sst2_val.parquet").rename(columns={"sentence": "text"})
    boolq = pd.read_parquet(f"{DATA}/boolq_val.parquet").rename(
        columns={"passage": "text"}).sample(n=1000, random_state=42)
    ag = pd.read_parquet(f"{DATA}/agnews_test.parquet").sample(n=2000, random_state=42)
    yelp = pd.read_parquet(f"{DATA}/yelp_test.parquet").sample(n=2000, random_state=42)

    res = {
        "noul_sst2": run_task(net, tok, temps, "noul_sst2", TASKS["noul_sst2"],
                              sst2.to_dict("records")),
        "noul_boolq": run_task(net, tok, temps, "noul_boolq", TASKS["noul_boolq"],
                               boolq.to_dict("records")),
        "choice_agnews": run_task(net, tok, temps, "choice_agnews", TASKS["choice_agnews"],
                                  ag.to_dict("records")),
        "score_yelp": run_task(net, tok, temps, "score_yelp", TASKS["score_yelp"],
                               yelp.to_dict("records")),
    }
    # headline aggregates per qtype (noul = sst2+boolq weighted)
    n_noul = res["noul_sst2"]["n"] + res["noul_boolq"]["n"]
    res["summary"] = {
        "noul_acc": round((res["noul_sst2"]["accuracy"] * res["noul_sst2"]["n"] +
                           res["noul_boolq"]["accuracy"] * res["noul_boolq"]["n"]) / n_noul, 4),
        "choice_acc": res["choice_agnews"]["accuracy"],
        "score_acc": res["score_yelp"]["accuracy"],
        "avg_acc": round((res["noul_sst2"]["accuracy"] * res["noul_sst2"]["n"] +
                          res["noul_boolq"]["accuracy"] * res["noul_boolq"]["n"] +
                          res["choice_agnews"]["accuracy"] * res["choice_agnews"]["n"] +
                          res["score_yelp"]["accuracy"] * res["score_yelp"]["n"]) /
                         (n_noul + res["choice_agnews"]["n"] + res["score_yelp"]["n"]), 4),
        "avg_ece": round(np.mean([res[k]["ece"] for k in
                                  ["noul_sst2", "noul_boolq", "choice_agnews", "score_yelp"]]), 4),
    }
    print(json.dumps(res, indent=1))
    import sys as _sys
    out = _sys.argv[2] if len(_sys.argv) > 2 else f"{F2}/runs/metrics.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    import sys
    main(ckpt=sys.argv[1] if len(sys.argv) > 1 else f"{F2}/runs/ckpt_calibrated.pt")
