"""train_rlcd.py — RLCD: Reinforcement Learning for Calibrated Decisions.

GRPO-style Gaussian logit-noise exploration with group-normalized advantage
against strictly proper scoring rules (log score for choice/noul, RPS for
score). The gradient of E[reward] under logit noise is estimated with the
score-function (ES) estimator:  g = E[A_i * eps_i] / sigma^2, applied to the
marker logits. A small supervised CE anchor keeps training stable.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/z/my-project/fragment-lab/f2")
from f2model import F2Net

torch.set_num_threads(2)
F2 = "/home/z/my-project/fragment-lab/f2"


def rps_torch(probs, gold, mask):
    """Ranked Probability Score per row (ordinal 'score' questions)."""
    K = probs.shape[-1]
    cp = torch.cumsum(probs, -1)
    onehot = F.one_hot(gold, K).float()
    cy = torch.cumsum(onehot, -1)
    per = ((cp - cy) ** 2).sum(-1) / (K - 1)
    if mask is None:
        return per
    return per * mask.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--resume", default=f"{F2}/runs/ckpt_supervised.pt")
    ap.add_argument("--out", default=f"{F2}/runs/ckpt_rlcd.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--sigma_start", type=float, default=1.0)
    ap.add_argument("--sigma_end", type=float, default=0.25)
    ap.add_argument("--ce_anchor", type=float, default=0.3)
    args = ap.parse_args()

    cfg = json.loads(open(args.config).read()) if args.config else {}
    steps = int(cfg.get("steps", args.steps))
    resume = cfg.get("resume", args.resume)
    out = cfg.get("out", args.out)
    batch = int(cfg.get("batch", args.batch))
    G = int(cfg.get("G", args.G))
    sig0 = float(cfg.get("sigma_start", args.sigma_start))
    sig1 = float(cfg.get("sigma_end", args.sigma_end))
    anchor = float(cfg.get("ce_anchor", args.ce_anchor))

    z = np.load(f"{F2}/encoded.npz")
    meta = json.load(open(f"{F2}/encoded_meta.json"))
    tasks = np.array([m["task"] for m in meta])
    train_idx = np.where(z["is_calib"] == 0)[0]
    rng = np.random.RandomState(23)

    net = F2Net()
    ck = torch.load(resume, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    print(f"resumed from {resume}")
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=0.01)

    logf = open(f"{F2}/runs/train_rlcd.log", "a")
    t0 = time.time()
    for step in range(1, steps + 1):
        sigma = sig0 + (sig1 - sig0) * (step / steps)
        sel = rng.choice(train_idx, batch, replace=False)
        ids = torch.tensor(z["ids"][sel], dtype=torch.long)
        mpos = torch.tensor(z["mpos"][sel], dtype=torch.long)
        n_opt = torch.tensor(z["n_opt"][sel], dtype=torch.long)
        gold = torch.tensor(z["gold"][sel], dtype=torch.long)
        task = [tasks[r] for r in sel]
        B, K = mpos.shape
        ar = torch.arange(K)[None].expand(B, K)
        mmask = ar < n_opt[:, None]
        ordinal = torch.tensor([t == "score" for t in task], dtype=torch.bool)

        logits = net(ids)
        zlog = torch.gather(logits, 1, mpos)          # (B,K) real markers only
        real = mmask.float()

        # ---- group exploration at logit level (G samples per item)
        with torch.no_grad():
            eps = torch.randn(G, B, K) * sigma
            zp = zlog.detach().unsqueeze(0) + eps      # (G,B,K)
            probs = torch.softmax(zp, -1)
            gold_p = gold.unsqueeze(0).expand(G, B)
            # strictly proper scoring rules as reward
            r_log = (probs * F.one_hot(gold_p, K).float()).clamp(1e-9, 1).log().sum(-1)
            r_rps = -rps_torch(probs, gold_p, None)
            reward = torch.where(ordinal.unsqueeze(0), r_rps, r_log)  # (G,B)
            A = reward - reward.mean(0, keepdim=True)
            A = A / (reward.std(0, keepdim=True) + 1e-6)
            # ES gradient on the marker logits
            g = (A.unsqueeze(-1) * eps).mean(0) / (sigma ** 2 + 1e-6)
            g = g * real                                # only real marker slots

        # apply manual gradient + supervised anchor
        loss = (zlog * g.detach()).sum() * 0.01 + anchor * F.cross_entropy(
            zlog.masked_fill(~mmask, -1e4), gold)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % 25 == 0:
            with torch.no_grad():
                acc = ((zlog.argmax(-1) == gold).float() * real.sum(-1) / real.sum(-1).clamp(1)).mean()
            msg = f"rlcd step {step}/{steps} sigma {sigma:.3f} acc {acc.item():.4f} loss {loss.item():.4f}"
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

    torch.save({"model": net.state_dict(), "cfg": net.cfg, "step": steps,
                "rlcd": True}, out)
    print(f"saved {out}")
    logf.write(f"saved {out}\n")
    logf.close()


if __name__ == "__main__":
    main()
