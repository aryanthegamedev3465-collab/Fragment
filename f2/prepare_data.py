"""prepare_data.py — build typed-decision items.jsonl from public datasets.

Tasks:
  noul   : SST-2 sentiment (67k) + BoolQ reading-comprehension (9.4k)
  choice : AG News topic classification (120k, options shuffled)
  score  : Yelp Review Full 5-star rating (130k sample)

Each item: {task, text, instruction, options, gold, split}
Splits: train / calib (2% held out for temperature fitting).
Instruction paraphrases are sampled per item.
"""
import json
import os
import random
import subprocess
import pandas as pd

DATA = "/home/z/my-project/fragment-lab/data"
OUT = "/home/z/my-project/fragment-lab/f2/items.jsonl"
rng = random.Random(7)

INST = {
    "sst2": [
        "is the sentiment of this text positive",
        "does this text express a positive sentiment",
        "is this review positive or negative overall",
        "is the sentiment here positive",
        "would readers consider this text positive",
        "is the tone of this text positive",
    ],
    "agnews": [
        "which topic does this text belong to",
        "classify the topic of this text",
        "which category fits this text best",
        "what is this text about",
        "pick the topic of this article",
        "which section would this text appear in",
    ],
    "yelp": [
        "how many stars does this review give",
        "rate this review from one to five stars",
        "how would you rate this review",
        "what star rating does this review deserve",
        "how positive or negative is this review on a five star scale",
        "how many stars does the reviewer give",
    ],
}
AG_OPTIONS = [
    "@0 World: world news and international events",
    "@1 Sports: sports results and athletics",
    "@2 Business: business, economy, stocks and finance",
    "@3 Technology: technology, computers, internet and science",
]
YELP_OPTIONS = [
    "@0 1-star: very negative",
    "@1 2-star: negative",
    "@2 3-star: mixed or neutral",
    "@3 4-star: positive",
    "@4 5-star: very positive",
]
NOUL_OPTIONS = [
    "@no: no, the statement does not hold",
    "@yes: yes, the statement holds",
]

items = []

# ---------------------------------------------------------------- SST-2 (noul)
df = pd.read_parquet(f"{DATA}/sst2_train.parquet").rename(columns={"sentence": "text"})
for text, label in zip(df["text"], df["label"]):
    items.append({
        "task": "noul", "source": "sst2",
        "text": str(text),
        "instruction": rng.choice(INST["sst2"]),
        "options": NOUL_OPTIONS,
        "gold": int(label),  # 1 = positive -> @yes
    })

# ---------------------------------------------------------------- BoolQ (noul)
df = pd.read_parquet(f"{DATA}/boolq_train.parquet")
for passage, question, answer in zip(df["passage"], df["question"], df["answer"]):
    items.append({
        "task": "noul", "source": "boolq",
        "text": str(passage),
        "instruction": str(question).strip().rstrip("?"),
        "options": NOUL_OPTIONS,
        "gold": 1 if bool(answer) else 0,
    })

# ---------------------------------------------------------------- AG News (choice)
df = pd.read_parquet(f"{DATA}/agnews_train.parquet")
for text, label in zip(df["text"], df["label"]):
    # option-order shuffle to prevent position bias (gold remapped)
    order = list(range(4))
    rng.shuffle(order)
    opts = [AG_OPTIONS[i] for i in order]
    gold = order.index(int(label))
    items.append({
        "task": "choice", "source": "agnews",
        "text": str(text),
        "instruction": rng.choice(INST["agnews"]),
        "options": opts,
        "gold": gold,
    })

# ---------------------------------------------------------------- Yelp (score)
df = pd.read_parquet(f"{DATA}/yelp_train.parquet")
df = df.sample(n=min(130_000, len(df)), random_state=7)
for text, label in zip(df["text"], df["label"]):
    items.append({
        "task": "score", "source": "yelp",
        "text": str(text),
        "instruction": rng.choice(INST["yelp"]),
        "options": YELP_OPTIONS,
        "gold": int(label),
    })

rng.shuffle(items)
# 2% calibration split (never trained on; also never overlapping the bench sets,
# which come from official validation/test parquets). Marked BEFORE extras are
# appended so the calib set stays stable across roadmap rounds.
n_calib = max(500, int(0.02 * len(items)))
for i, it in enumerate(items):
    it["split"] = "calib" if i < n_calib else "train"

# ---------------------------------------------------------------- extras
import sys
EXTRA = [a for a in sys.argv[1:] if a.startswith("--extra=")]
if EXTRA:
    wanted = EXTRA[0].split("=", 1)[1].split(",")
    import pyarrow.parquet as pq

    def stream_rows(repo, want_train=True, max_rows=100000):
        import urllib.request, json as _json
        url = f"https://huggingface.co/api/datasets/{repo}?full=true"
        with urllib.request.urlopen(url) as r:
            sib = _json.load(r).get("siblings", [])
        cand = [s["rfilename"] for s in sib
                if s["rfilename"].endswith(".parquet")
                and ("train" in s["rfilename"]) == want_train]
        if not cand:
            raise RuntimeError(f"no parquet found for {repo}")
        path = f"{DATA}/{repo.replace('/', '_')}_{cand[0].split('/')[-1]}"
        if not os.path.exists(path):
            u = f"https://huggingface.co/datasets/{repo}/resolve/main/{cand[0]}"
            subprocess.run(["curl", "-sL", "-o", path, u], check=True)
        pf = pq.ParquetFile(path)
        rows = []
        for batch in pf.iter_batches(batch_size=2048):
            d = batch.to_pylist()
            rows.extend(d)
            if len(rows) >= max_rows:
                break
        return rows[:max_rows]

    if "imdb" in wanted:
        for r in stream_rows("fancyzhx/imdb", max_rows=25000):
            items.append({"task": "noul", "source": "imdb",
                          "text": str(r["text"]),
                          "instruction": rng.choice(INST["sst2"]),
                          "options": NOUL_OPTIONS, "gold": int(r["label"]),
                          "split": "train"})
        print("added imdb")

    if "amazon" in wanted:
        for r in stream_rows("fancyzhx/amazon_polarity", max_rows=60000):
            txt = str(r.get("content") or r.get("text") or "")
            items.append({"task": "noul", "source": "amazon",
                          "text": txt,
                          "instruction": rng.choice(INST["sst2"]),
                          "options": NOUL_OPTIONS, "gold": int(r["label"]),
                          "split": "train"})
        print("added amazon_polarity")

    if "yelp_more" in wanted:
        n0 = len(items)
        dfy = pd.read_parquet(f"{DATA}/yelp_train.parquet")
        dfy = dfy.sample(n=min(240_000, len(dfy)), random_state=7)
        have = set()
        for it in items:
            if it["source"] == "yelp":
                have.add(it["text"][:80])
        for text, label in zip(dfy["text"], dfy["label"]):
            if str(text)[:80] not in have:
                items.append({"task": "score", "source": "yelp",
                              "text": str(text),
                              "instruction": rng.choice(INST["yelp"]),
                              "options": YELP_OPTIONS, "gold": int(label),
                              "split": "train"})
        print(f"added yelp_more: +{len(items)-n0}")

with open(OUT, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")

from collections import Counter
c = Counter((it["task"], it["split"]) for it in items)
print(f"items: {len(items)} -> {OUT}")
for k in sorted(c):
    print(" ", k, c[k])
