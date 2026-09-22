#!/usr/bin/env python3
"""f2.py — self-contained Python runtime for FrameXlabs/fragment-1 (v2).

Loads the released safetensors checkpoint + BPE tokenizer and answers typed
questions with calibrated probabilities in a single forward pass.

Usage:
    from f2 import Fragment
    m = Fragment.from_pretrained("FrameXlabs/fragment-1")  # hub or local dir
    res = m.decide(
        state="Hi, we were billed twice for March. Please refund the duplicate.",
        questions={
            "refund": {"type": "noul",
                       "instructions": "Does the user request a refund?"},
            "urgency": {"type": "score",
                        "instructions": "How urgent is this?",
                        "criteria": ["not urgent", "soon", "critical"]},
            "topic": {"type": "choice",
                      "instructions": "Which department should handle this?",
                      "criteria": {"billing": "invoices, payments, refunds",
                                   "technical": "bugs, outages, errors",
                                   "other": "everything else"}},
        })
    print(res["answers"]["refund"]["noul"])       # calibrated P(yes)

Dependencies: torch, safetensors.
"""
import json
import math
import os
import re
from typing import Dict

import torch
import torch.nn as nn

MAX_LEN = 192


class F2Tokenizer:
    def __init__(self, path: str):
        spec = json.load(open(path))
        self.vocab = spec["vocab"]
        self.id_of = {v: i for i, v in enumerate(self.vocab)}
        self.ranks = {(a, b): i for i, (a, b) in enumerate(spec["merges"])}
        self.added = set(spec.get("added", []))
        self.pad = self.id_of["[PAD]"]
        self.cls = self.id_of["[CLS]"]
        self.sep = self.id_of["[SEP]"]
        self.unk = self.id_of["[UNK]"]
        self._cache: Dict[str, list] = {}

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
    def __init__(self, vocab, d, max_len, n_layers, n_heads, ff):
        super().__init__()
        self.scale = math.sqrt(d)
        self.tok = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, ids):
        att = ids != 0
        pad = ~att
        B, L = ids.shape
        h = self.tok(ids) * self.scale + self.pos(torch.arange(L))[None]
        for blk in self.blocks:
            h = blk(h, pad)
        return self.head(self.ln_f(h)).squeeze(-1).float()


def _confidence(p):
    k = len(p)
    if k < 2:
        return 1.0
    ent = -sum(x * math.log(max(x, 1e-12)) for x in p)
    return min(1.0, max(0.0, 1.0 - ent / math.log(k)))


class Fragment:
    def __init__(self, root: str):
        self.cfg = json.load(open(os.path.join(root, "f2_config.json")))
        self.tok = F2Tokenizer(os.path.join(root, "tokenizer.json"))
        from safetensors.torch import load_file
        sd = load_file(os.path.join(root, "model.safetensors"))
        arch = self.cfg["architecture"]
        self.net = F2Net(arch["vocab"], arch["d"], arch["max_len"],
                         arch["n_layers"], arch["n_heads"], arch["ff"])
        self.net.load_state_dict(sd, strict=True)
        self.net.eval()
        self.version = self.cfg.get("version", "2.0")
        self.temps = self.cfg.get("temperatures", {})

    @classmethod
    def from_pretrained(cls, src: str = "FrameXlabs/fragment-1") -> "Fragment":
        if os.path.isdir(src):
            return cls(src)
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo_id=src))

    def _options(self, q):
        t = q["type"]
        if t == "choice":
            crit = q.get("criteria") or {}
            return [f"@{i} {k}: {v}" if v else f"@{i} {k}"
                    for i, (k, v) in enumerate(crit.items())], list(crit.keys())
        if t == "score":
            crit = q.get("criteria") or []
            return [f"@{i} level {i}: {c}" for i, c in enumerate(crit)], None
        return ["@no: no, the statement does not hold",
                "@yes: yes, the statement holds"], None

    @torch.no_grad()
    def decide(self, state: str, questions: Dict) -> Dict:
        import time as _t
        t0 = _t.time()
        answers = {}
        for qid, q in questions.items():
            opts, keys = self._options(q)
            pre = self.tok.encode(f"{q['instructions']} input:")
            opt_ids = self.tok.encode("options: " + "; ".join(opts))
            markers = [o.split()[0].rstrip(":") for o in opts]
            marker_ids = [self.tok.id_of[m] for m in markers]
            positions, seen = [], set()
            for idx, t in enumerate(opt_ids):
                if t in marker_ids and t not in seen:
                    seen.add(t)
                    positions.append(idx)
            budget = MAX_LEN - 2 - len(pre) - len(opt_ids)
            body = self.tok.encode(" " + state)[: max(8, budget)]
            ids = [self.tok.cls] + pre + body + opt_ids + [self.tok.sep]
            ids = ids[:MAX_LEN]
            shift = 1 + len(pre) + len(body)
            abs_pos = [p + shift for p in positions]
            abs_pos = [p for p in abs_pos if p < len(ids) - 1]
            t = torch.tensor([ids])
            logits = self.net(t)[0, abs_pos]
            temp = self.temps.get(q["type"], 1.0) or 1.0
            p = torch.softmax(logits / temp, -1).tolist()
            conf = _confidence(p)
            base = {"type": q["type"], "question": q["instructions"],
                    "probabilities": {}, "confidence": round(conf, 4)}
            if q["type"] == "choice":
                for i, k in enumerate(keys or []):
                    base["probabilities"][k] = round(p[i], 4)
                base["choice"] = (keys or ["?"])[int(max(range(len(p)), key=lambda i: p[i]))]
            elif q["type"] == "score":
                exp = sum(i * p[i] for i in range(len(p)))
                for i in range(len(p)):
                    base["probabilities"][str(i)] = round(p[i], 4)
                base["score"] = round(exp, 4)
            else:
                base["probabilities"] = {"false": round(p[0], 4),
                                         "true": round(p[1], 4)}
                base["noul"] = round(p[1], 4)
            answers[qid] = base
        return {"model": "fragment", "version": self.version, "answers": answers,
                "usage": {"latency_ms": round((_t.time() - t0) * 1000)}}
