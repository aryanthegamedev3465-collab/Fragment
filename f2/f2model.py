"""f2model.py — shared model + tokenizer for Fragment v2 (from scratch).

Architecture (chosen by CPU speed benchmark, bf16 autocast):
  vocab 16384, d=256, 6 pre-norm transformer blocks, 4 heads, ff=1024,
  max_len 192, per-position logit head, softmax over @option markers.
Item format (proven by the strongest prior model, fragment-1):
  [CLS] <instruction> Input: <text> Options: @0 ...; @1 ... [SEP]
"""
import json
import math
import re
import torch
import torch.nn as nn

SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
MARKERS = ["@0", "@1", "@2", "@3", "@4", "@no", "@yes"]
MAX_LEN = 192


# ---------------------------------------------------------------- tokenizer

class F2Tokenizer:
    r"""Char-level BPE with atomic marker tokens. Mirrors the `tokenizers`
    training config (lowercase, \w+|[^\w\s] pre-split)."""

    def __init__(self, path: str):
        spec = json.load(open(path))
        self.vocab = spec["vocab"]          # list[str]
        self.id_of = {v: i for i, v in enumerate(self.vocab)}
        self.merges = [(a, b) for a, b in spec["merges"]]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.added = set(spec.get("added", []))
        self.pad = self.id_of["[PAD]"]
        self.cls = self.id_of["[CLS]"]
        self.sep = self.id_of["[SEP]"]
        self.unk = self.id_of["[UNK]"]
        self._cache = {}

    def _bpe(self, word: str):
        syms = list(word)
        while len(syms) >= 2:
            best, rank = None, None
            for a, b in zip(syms, syms[1:]):
                r = self.ranks.get((a, b))
                if r is not None and (rank is None or r < rank):
                    best, rank = (a, b), r
            if best is None:
                break
            a, b = best
            out, j = [], 0
            while j < len(syms):
                if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                    out.append(a + b)
                    j += 2
                else:
                    out.append(syms[j])
                    j += 1
            syms = out
        return syms

    def _word_ids(self, word: str):
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [self.id_of.get(s, self.unk) for s in self._bpe(word)]
        if len(self._cache) < 300_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str):
        text = text.lower()
        ids = []
        for w in re.findall(r"@?[a-z0-9]+|[^\sa-z0-9]", text):
            if w in self.added:
                ids.append(self.id_of[w])
            else:
                ids.extend(self._word_ids(w))
        return ids


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d, nhead, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, x, pad):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        h = self.ln2(x)
        return x + self.fc2(nn.functional.gelu(self.fc1(h)))


class F2Net(nn.Module):
    def __init__(self, vocab=16384, d=256, max_len=MAX_LEN, n_layers=6,
                 n_heads=4, ff=1024):
        super().__init__()
        self.d = d
        self.scale = math.sqrt(d)
        self.tok = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.cfg = {"vocab": vocab, "d": d, "max_len": max_len,
                    "n_layers": n_layers, "n_heads": n_heads, "ff": ff}

    def forward(self, ids):
        att = ids != 0
        pad = ~att
        B, L = ids.shape
        h = self.tok(ids) * self.scale + self.pos(torch.arange(L, device=ids.device))[None]
        for blk in self.blocks:
            h = blk(h, pad)
        h = self.ln_f(h)
        return self.head(h).squeeze(-1).float()


# ---------------------------------------------------------------- item builder

def build_item(tok: F2Tokenizer, instruction: str, text: str, options,
               max_len=MAX_LEN):
    """options: list[str] like '@0 World: ...'. Returns (ids, marker_positions)."""
    pre = tok.encode(f"{instruction} input:")
    opt_ids = tok.encode("options: " + "; ".join(options))
    marker_tokens = [o.split()[0].rstrip(":") for o in options]
    marker_ids = [tok.id_of[m] for m in marker_tokens]
    positions, seen = [], set()
    for idx, t in enumerate(opt_ids):
        if t in marker_ids and t not in seen:
            seen.add(t)
            positions.append(idx)
    budget = max_len - 2 - len(pre) - len(opt_ids)
    body = tok.encode(" " + text)[: max(8, budget)]
    ids = [tok.cls] + pre + body + opt_ids + [tok.sep]
    ids = ids[:max_len]
    shift = 1 + len(pre) + len(body)
    abs_pos = [p + shift for p in positions]
    abs_pos = [p for p in abs_pos if p < len(ids) - 1]
    return ids, abs_pos


def collate(items, max_len=MAX_LEN, max_opts=5):
    """items: list of (ids, marker_pos, gold, n_opts)."""
    B = len(items)
    ids = torch.zeros(B, max_len, dtype=torch.long)
    mpos = torch.zeros(B, max_opts, dtype=torch.long)
    mmask = torch.zeros(B, max_opts, dtype=torch.bool)
    gold = torch.zeros(B, dtype=torch.long)
    for j, (seq, positions, g, _n) in enumerate(items):
        ids[j, : len(seq)] = torch.tensor(seq[:max_len])
        k = min(len(positions), max_opts)
        mpos[j, :k] = torch.tensor(positions[:k])
        mmask[j, :k] = True
        gold[j] = g
    return ids, mpos, mmask, gold


def marker_logits(logits, mpos, mmask):
    """logits (B,L) -> (B,5) gathered at marker positions, masked."""
    gathered = torch.gather(logits, 1, mpos)
    return gathered.masked_fill(~mmask, -1e4)
