"""train_supervised.py — supervised warmup: cross-entropy on gold markers.

Defaults: batch 64, 2 epochs, AdamW 3e-4 -> cosine -> 3e-5, bf16 autocast.
Supports --config overrides (epochs, lr, resume, oversample, out) so roadmap
improvement tasks can continue training the same checkpoint.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/z/my-project/fragment-lab/f2")
from f2model import F2Net, MAX_LEN

torch.set_num_threads(2)
F2 = "/home/z/my-project/fragment-lab/f2"


def load_encoded():
    # materialize arrays once — NpzFile re-decompresses members on every
    # access, which otherwise adds ~0.7s per batch
    zf = np.load(f"{F2}/encoded.npz")
    z = {k: zf[k] for k in zf.files}
    meta = json.load(open(f"{F2}/encoded_meta.json"))
    tasks = np.array([m["task"] for m in meta])
    return z, tasks


def make_loader(z, tasks, idx, batch=64, oversample=None, seed=0):
    """Yield batches of (ids, mpos, mmask, gold) from row indices idx."""
    rng = np.random.RandomState(seed)
    rows = list(idx)
    if oversample:
        for task, factor in oversample.items():
            t_rows = [r for r in idx if tasks[r] == task]
            extra_n = int(len(t_rows) * (factor - 1.0))
            if extra_n > 0:
                rows += list(rng.choice(t_rows, extra_n, replace=True))
    rng.shuffle(rows)
    for i in range(0, len(rows), batch):
        sel = rows[i: i + batch]
        ids = torch.tensor(z["ids"][sel], dtype=torch.long)
        mpos = torch.tensor(z["mpos"][sel], dtype=torch.long)
        n_opt = torch.tensor(z["n_opt"][sel], dtype=torch.long)
        gold = torch.tensor(z["gold"][sel], dtype=torch.long)
        B, K = mpos.shape
        ar = torch.arange(K)[None].expand(B, K)
        mmask = ar < n_opt[:, None]
        yield ids, mpos, mmask, gold


@torch.no_grad()
def quick_val(net, z, tasks, calib_idx, batch=128):
    net.eval()
    per_task = {}
    for i in range(0, len(calib_idx), batch):
        sel = calib_idx[i: i + batch]
        ids = torch.tensor(z["ids"][sel], dtype=torch.long)
        mpos = torch.tensor(z["mpos"][sel], dtype=torch.long)
        n_opt = torch.tensor(z["n_opt"][sel], dtype=torch.long)
        gold = torch.tensor(z["gold"][sel], dtype=torch.long)
        B, K = mpos.shape
        ar = torch.arange(K)[None].expand(B, K)
        mmask = ar < n_opt[:, None]
        logits = net(ids)
        gathered = torch.gather(logits, 1, mpos).masked_fill(~mmask, -1e4)
        pred = gathered.argmax(-1)
        for j, r in enumerate(sel):
            t = tasks[r]
            d = per_task.setdefault(t, [0, 0])
            d[0] += int(pred[j].item() == gold[j].item())
            d[1] += 1
    net.train()
    return {t: round(c / n, 4) for t, (c, n) in per_task.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON config overrides")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=f"{F2}/runs/ckpt_supervised.pt")
    ap.add_argument("--oversample", default=None, help='JSON like {"score":1.3}')
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = json.load(open(args.config))
    epochs = float(cfg.get("epochs", args.epochs))
    lr = float(cfg.get("lr", args.lr))
    batch = int(cfg.get("batch", args.batch))
    resume = cfg.get("resume", args.resume)
    out = cfg.get("out", args.out)
    oversample = cfg.get("oversample") or (
        json.loads(args.oversample) if args.oversample else None)
    seed = int(cfg.get("seed", args.seed))

    z, tasks = load_encoded()
    train_idx = np.where(z["is_calib"] == 0)[0]
    calib_idx = np.where(z["is_calib"] == 1)[0]
    print(f"train={len(train_idx)} calib={len(calib_idx)} epochs={epochs} "
          f"lr={lr} batch={batch} oversample={oversample}")

    net = F2Net()
    start_step = 0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {resume} (step {start_step})")
    elif os.path.exists(f"{F2}/runs/ckpt_supervised_latest.pt"):
        ck = torch.load(f"{F2}/runs/ckpt_supervised_latest.pt",
                        map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        start_step = int(ck.get("step", 0))
        print(f"auto-resumed from periodic checkpoint (step {start_step})")

    steps_total = int(epochs * len(train_idx) / batch) + 1
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(50, int(0.03 * steps_total))

    def lr_at(step):
        if step < warmup:
            return lr * step / warmup
        p = (step - warmup) / max(1, steps_total - warmup)
        p = min(1.0, p)
        return lr * (0.1 ** p)  # cosine-ish decay to 10%

    net.train()
    t0 = time.time()
    step = start_step
    log_path = f"{F2}/runs/train_supervised.log"
    logf = open(log_path, "a")
    best_acc = -1
    for ep in range(int(epochs) + (1 if epochs % 1 else 0)):
        for ids, mpos, mmask, gold in make_loader(z, tasks, train_idx, batch,
                                                  oversample, seed=seed + ep):
            if step >= steps_total:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = net(ids)
            gathered = torch.gather(logits.float(), 1, mpos).masked_fill(~mmask, -1e4)
            loss = F.cross_entropy(gathered, gold)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                dt = (time.time() - t0) / max(1, step - start_step)
                msg = (f"step {step}/{steps_total} loss {loss.item():.4f} "
                       f"lr {lr_at(step):.2e} {dt:.2f}s/step eta "
                       f"{(steps_total-step)*dt/3600:.2f}h")
                print(msg, flush=True)
                logf.write(msg + "\n")
                logf.flush()
            if step % 2000 == 0:
                accs = quick_val(net, z, tasks, calib_idx[:2000])
                msg = f"[val] step {step} {accs}"
                print(msg, flush=True)
                logf.write(msg + "\n")
                logf.flush()
                torch.save({"model": net.state_dict(), "cfg": net.cfg,
                            "step": step, "val": accs},
                           f"{F2}/runs/ckpt_supervised_latest.pt")
        if step >= steps_total:
            break

    accs = quick_val(net, z, tasks, calib_idx)
    torch.save({"model": net.state_dict(), "cfg": net.cfg, "step": step,
                "val": accs}, out)
    print(f"saved {out} step={step} val={accs}", flush=True)
    logf.write(f"saved {out} step={step} val={accs}\n")
    logf.close()


if __name__ == "__main__":
    main()
