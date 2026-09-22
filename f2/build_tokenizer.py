"""build_tokenizer.py — train the from-scratch BPE tokenizer (16384 vocab).

Uses the fast `tokenizers` lib, then exports to the simple portable format
{vocab: [...], merges: [[a,b],...], added: [...]} used by f2 runtime.
Markers @0..@4, @no, @yes are atomic added tokens.
"""
import json
import random
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

DATA = "/home/z/my-project/fragment-lab/data"
OUT = "/home/z/my-project/fragment-lab/f2/tokenizer.json"

VOCAB = 16384
MARKERS = ["@0", "@1", "@2", "@3", "@4", "@no", "@yes"]
SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]

rng = random.Random(7)
corpus = []

def add_stream(path, column, frac, cap, max_n=None):
    """Stream a parquet in batches (low-RAM) and sample sentences."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    taken = 0
    for batch in pf.iter_batches(batch_size=2048, columns=[column]):
        for v in batch.column(0).to_pylist():
            if rng.random() < frac:
                s = str(v)[:cap]
                if s.strip():
                    corpus.append(s)
                    taken += 1
                    if max_n and taken >= max_n:
                        return
    return

add_stream(f"{DATA}/sst2_train.parquet", "sentence", 0.35, 200, max_n=22000)
add_stream(f"{DATA}/agnews_train.parquet", "text", 0.25, 250, max_n=28000)
add_stream(f"{DATA}/yelp_train.parquet", "text", 0.03, 300, max_n=22000)
add_stream(f"{DATA}/boolq_train.parquet", "passage", 0.45, 300, max_n=4500)

import gc
gc.collect()
print(f"corpus sentences: {len(corpus)}")

tok = Tokenizer(models.BPE())
tok.normalizer = normalizers.Lowercase()
tok.pre_tokenizer = pre_tokenizers.Split(pattern=r"\w+|[^\w\s]", behavior="isolated")
# \w includes '@'? In Rust regex \w = [0-9A-Za-z_] plus unicode -> '@' is NOT \w.
# Markers will be handled as added tokens (atomic), so pre-split on @ is fine.

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB - len(MARKERS) - len(SPECIALS),
    special_tokens=SPECIALS,
    show_progress=False,
    initial_alphabet=list("abcdefghijklmnopqrstuvwxyz0123456789"),
)
tok.train_from_iterator(corpus, trainer)
tok.add_tokens(MARKERS)

# ---------------------------------------------------------------- export
vocab_map = tok.get_vocab()          # token -> id
inv = {i: v for v, i in vocab_map.items()}
max_id = max(inv)
vocab = [inv.get(i, "[UNK]") for i in range(max_id + 1)]

bpe = tok.model
merges = []
# tokenizers serializes merges internally; get them via to_str
spec = json.loads(tok.to_str())
merges = [list(pair) for pair in spec["model"].get("merges", [])]

spec_out = {"vocab": vocab, "merges": merges, "added": MARKERS, "specials": SPECIALS}
with open(OUT, "w") as f:
    json.dump(spec_out, f)
print(f"vocab size: {len(vocab)}  merges: {len(merges)}")

# sanity: round-trip through our runtime tokenizer
import sys
sys.path.insert(0, "/home/z/my-project/fragment-lab/f2")
from f2model import F2Tokenizer
rt = F2Tokenizer(OUT)
enc = rt.encode("options: @0 World: world news; @1 Sports Input: the game was great .")
ids_mark = [rt.id_of[m] for m in MARKERS]
print("encode sample:", enc[:14])
print("marker ids:", ids_mark)
present = [i for i in enc if i in set(ids_mark)]
print("markers present in sample:", present)
assert all(m in rt.id_of for m in MARKERS)
print("tokenizer OK")
