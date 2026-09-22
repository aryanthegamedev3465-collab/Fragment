"""encode_data.py — pre-encode items.jsonl into compact numpy arrays.

encoded.npz: ids (N,192 int32), mpos (N,5 int32), n_opt (N int8),
             gold (N int8), is_calib (N int8)
encoded_meta.json: index -> {task, source}
"""
import json
import numpy as np
import sys

sys.path.insert(0, "/home/z/my-project/fragment-lab/f2")
from f2model import F2Tokenizer, MAX_LEN, build_item

items = [json.loads(l) for l in open("/home/z/my-project/fragment-lab/f2/items.jsonl")]
tok = F2Tokenizer("/home/z/my-project/fragment-lab/f2/tokenizer.json")

N = len(items)
ids = np.zeros((N, MAX_LEN), dtype=np.int32)
mpos = np.zeros((N, 5), dtype=np.int32)
n_opt = np.zeros(N, dtype=np.int8)
gold = np.zeros(N, dtype=np.int8)
is_calib = np.zeros(N, dtype=np.int8)
meta = []

for i, it in enumerate(items):
    seq, positions = build_item(tok, it["instruction"], it["text"], it["options"])
    k = min(len(positions), 5)
    ids[i, : len(seq)] = seq[:MAX_LEN]
    if k:
        mpos[i, :k] = positions[:k]
    n_opt[i] = k
    gold[i] = min(it["gold"], 4)
    is_calib[i] = 1 if it["split"] == "calib" else 0
    meta.append({"task": it["task"], "source": it["source"]})
    if (i + 1) % 50000 == 0:
        print(f"encoded {i+1}/{N}")

np.savez_compressed("/home/z/my-project/fragment-lab/f2/encoded.npz",
                    ids=ids, mpos=mpos, n_opt=n_opt, gold=gold, is_calib=is_calib)
with open("/home/z/my-project/fragment-lab/f2/encoded_meta.json", "w") as f:
    json.dump(meta, f)
print(f"encoded {N} items -> encoded.npz")
unk = int((ids[:, :] == tok.unk).sum())
tot = int((ids > 3).sum())
print(f"unk rate: {unk/max(tot,1):.4%}")
