"""fragment1.py — single-file runtime for FrameXlabs/fragment-1.

Loads the checkpoint straight from the Hugging Face Hub (or a local folder),
answers typed questions in one forward pass each.

Architecture (from fragment-final.pt state_dict + config.json):
  tok_emb(16384,192)*sqrt(192) + pos_emb(160,192) -> in_ln -> 4 pre-norm blocks
  (ln1->MHA(4 heads), ln2->fc1(768)->exact GELU->fc2) -> ln_f -> head Linear(192->1)
  One logit per position; softmax over the @option-marker logits = decision.
Item format (from the model card):
  [CLS] instruction criteria Input: <text> Options: @0 ...; @1 ... [SEP]

Requires: torch. Optional: huggingface_hub (only if you load by repo id).

Usage:
    from fragment1 import Fragment1

    m = Fragment1.from_pretrained("FrameXlabs/fragment-1")
    res = m.decide(
        state="We were billed twice for March. Please refund the duplicate today.",
        questions={
            "refund": {"type": "noul",
                       "instructions": "Does the user explicitly request a refund?"},
        },
    )
    print(res["answers"]["refund"]["noul"])
"""
import json
import math
import os
import re

import torch
import torch.nn as nn


# ---------------------------------------------------------------- tokenizer

class F1Tokenizer:
    def __init__(self, path):
        spec = json.load(open(path))
        self.vocab = [v.encode("utf-8") if isinstance(v, str) else v for v in spec["vocab"]]
        self.id_of = {v: i for i, v in enumerate(self.vocab)}
        self.ranks = {(a.encode("utf-8"), b.encode("utf-8")): i
                      for i, (a, b) in enumerate(spec["merges"])}
        self.pad, self.cls, self.sep, self.unk = 0, 1, 2, 3

    def _bpe(self, word: str):
        syms = [bytes([b]) for b in word.encode("utf-8")]
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

    def encode(self, text: str):
        ids = []
        # words may include @ and digits so markers like '@0' / '@yes' stay single words
        for w in re.findall(r"[a-z@0-9]+|[^\sa-z0-9@]", text.lower()):
            for s in self._bpe(w):
                ids.append(self.id_of.get(s, self.unk))
        return ids


# ---------------------------------------------------------------- net

class Block(nn.Module):
    def __init__(self, d, nhead, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, x, pad):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        h = self.ln2(x)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(h)))


class F1Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, L = cfg["d_model"], cfg["max_len"]
        self.scale = math.sqrt(d)
        self.tok = nn.Embedding(cfg["vocab_size"], d, padding_idx=0)
        self.pos = nn.Embedding(L, d)
        self.in_ln = nn.LayerNorm(d, eps=1e-5)
        self.blocks = nn.ModuleList([Block(d, cfg["n_heads"], cfg["ff"])
                                     for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.head = nn.Linear(d, 1)

    def forward(self, ids):
        att = (ids != 0)
        pad = ~att
        B, L = ids.shape
        h = self.tok(ids) * self.scale + self.pos(torch.arange(L))[None]
        h = self.in_ln(h)
        for blk in self.blocks:
            h = blk(h, pad)
        h = self.ln_f(h)
        return self.head(h).squeeze(-1).float()


# ---------------------------------------------------------------- model

class Fragment1:
    """Typed-decision model. 4.96M parameters, runs on CPU."""

    def __init__(self, root):
        self.ck = torch.load(f"{root}/fragment-final.pt", map_location="cpu",
                             weights_only=False)
        self.cfg = self.ck["cfg"]
        self.temps = self.ck["temperatures"]
        self.net = F1Net(self.cfg)
        self.net.load_state_dict(self.ck["model"], strict=True)
        self.net.eval()
        self.tok = F1Tokenizer(f"{root}/tokenizer.json")

    @classmethod
    def from_pretrained(cls, src: str = "FrameXlabs/fragment-1") -> "Fragment1":
        """`src` is a local directory containing fragment-final.pt + tokenizer.json,
        or a Hugging Face repo id (needs `huggingface_hub`)."""
        if os.path.isdir(src):
            return cls(src)
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo_id=src))

    def encode_item(self, instruction: str, criteria: str, text: str, options):
        """[CLS] instruction criteria Input: <text> Options: @0 ...; @1 ... [SEP]
        Marker positions are tracked explicitly; input text is budgeted so
        the options (and their markers) always survive truncation."""
        L = self.cfg["max_len"]
        pre = self.tok.encode(f"{instruction} {criteria} Input:")
        opt_str = "; ".join(o for o in options)
        opt_ids = self.tok.encode("Options: " + opt_str)
        marker_ids = [self.tok.id_of[o.split()[0].encode()] for o in options]
        # locate marker ids inside opt_ids (first occurrence of each, in order)
        positions, seen = [], set()
        for idx, t in enumerate(opt_ids):
            if t in marker_ids and t not in seen:
                seen.add(t)
                positions.append(idx)
        budget = L - 2 - len(pre) - len(opt_ids)
        body = self.tok.encode(" " + text)[: max(8, budget)]
        ids = [self.tok.cls] + pre + body + opt_ids + [self.tok.sep]
        ids = ids[:L]
        # marker absolute positions (shifted by pre+body); clamp if truncated
        shift = 1 + len(pre) + len(body)
        abs_pos = [p + shift for p in positions]
        abs_pos = [p for p in abs_pos if p < len(ids) - 1]
        return ids, abs_pos

    @torch.no_grad()
    def batch_logits(self, items):
        """items: list of (ids, [marker positions]). returns list of option logits"""
        L = self.cfg["max_len"]
        out = []
        B = 64
        for i in range(0, len(items), B):
            chunk = items[i: i + B]
            ids = torch.zeros(len(chunk), L, dtype=torch.long)
            for j, (seq, _m) in enumerate(chunk):
                ids[j, : len(seq)] = torch.tensor(seq[:L])
            logits = self.net(ids)
            for j, (seq, midx) in enumerate(chunk):
                if not midx:
                    midx = [min(len(seq), L) - 2]
                out.append(logits[j, midx].tolist())
        return out

    @torch.no_grad()
    def probs(self, task, items):
        """items: list of (ids, markers). Applies task temperature, softmax over markers."""
        lg = self.batch_logits(items)
        T = self.temps.get(task, 1.0)
        res = []
        for row in lg:
            t = torch.tensor(row) / T
            res.append(torch.softmax(t, -1).tolist())
        return res

    # ---------------------------------------------------------------- decide API

    @staticmethod
    def _render_options(q):
        t = q["type"]
        if t == "choice":
            crit = q.get("criteria") or {}
            return ([f"@{i} {k}" + (f": {v}" if v else "")
                     for i, (k, v) in enumerate(crit.items())],
                    list(crit.keys()))
        if t == "score":
            crit = q.get("criteria") or []
            return ([f"@{i} {i}: {c}" for i, c in enumerate(crit)], None)
        if t == "noul":
            return ["@no", "@yes"], None
        raise ValueError(f"unknown question type: {t!r} (use choice/score/noul)")

    @torch.no_grad()
    def decide(self, state: str, questions: dict) -> dict:
        """Ask typed questions about one piece of text.

        questions = {
            "qid": {"type": "noul" | "choice" | "score",
                    "instructions": str,
                    "criteria": ...}   # dict for choice, list for score
        }
        Returns {"answers": {qid: {...}}, "usage": {"latency_ms": float}}.
        """
        import time as _t
        t0 = _t.time()
        answers = {}
        for qid, q in questions.items():
            opts, keys = self._render_options(q)
            ids, pos = self.encode_item(q["instructions"], "", str(state), opts)
            p = self.probs(q["type"], [(ids, pos)])[0]
            top = int(max(range(len(p)), key=lambda i: p[i]))
            out = {"type": q["type"], "question": q["instructions"],
                   "probabilities": {}, "confidence": round(max(p), 4)}
            if q["type"] == "choice":
                for i, k in enumerate(keys):
                    out["probabilities"][k] = round(p[i], 4)
                out["choice"] = keys[top]
            elif q["type"] == "score":
                exp = sum(i * p[i] for i in range(len(p)))
                for i in range(len(p)):
                    out["probabilities"][str(i)] = round(p[i], 4)
                out["score"] = round(exp, 4)
            else:  # noul
                out["probabilities"] = {"false": round(p[0], 4),
                                        "true": round(p[1], 4)}
                out["noul"] = round(p[1], 4)
            answers[qid] = out
        return {"model": "fragment-1", "answers": answers,
                "usage": {"latency_ms": round((_t.time() - t0) * 1000, 1)}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ask fragment-1 typed questions about text")
    ap.add_argument("--model", default="FrameXlabs/fragment-1")
    ap.add_argument("--state", required=True)
    ap.add_argument("--noul", action="append", default=[],
                    help="yes/no question (repeatable)")
    args = ap.parse_args()
    m = Fragment1.from_pretrained(args.model)
    qs = {f"q{i}": {"type": "noul", "instructions": s} for i, s in enumerate(args.noul)}
    print(json.dumps(m.decide(args.state, qs), indent=2))
