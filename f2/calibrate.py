"""calibrate.py — per-task temperature scaling on the held-out calibration split.

choice/noul: minimise NLL. score: minimise RPS (ordinal-aware).
Writes temperatures into the checkpoint (runs/ckpt_calibrated.pt).
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/z/my-project/fragment-lab/f2")
from f2model import F2Net

torch.set_num_threads(2)
F2 = "/home/z/my-project/fragment-lab/f2"


@torch.no_grad()
def collect(net, z, idx, batch=128):
    """Returns {task: (logits (n,K), gold (n,), n_opt)}"""
    out = {}
    for i in range(0, len(idx), batch):
        sel = idx[i: i + batch]
        ids = torch.tensor(z["ids"][sel], dtype=torch.long)
        mpos = torch.tensor(z["mpos"][sel], dtype=torch.long)
        n_opt = torch.tensor(z["n_opt"][sel], dtype=torch.long)
        logits = net(ids)
        gathered = torch.gather(logits, 1, mpos)
        for j, r in enumerate(sel):
            t = META[r]
            K = int(n_opt[j])
            d = out.setdefault(t, ([], [], []))
            d[0].append(gathered[j, :K])
            d[1].append(int(z["gold"][r]))
            d[2].append(K)
    return {t: (torch.stack(l), torch.tensor(g), k) for t, (l, g, k) in out.items()}


def rps_np(probs, gold):
    K = probs.shape[1]
    cp = np.cumsum(probs, 1)
    cy = np.zeros_like(cp)
    cy[np.arange(len(gold)), gold] = 1.0
    cy = np.cumsum(cy, 1)
    return float(np.mean(np.sum((cp - cy) ** 2, 1) / (K - 1)))


def main():
    import sys
    ck_path = sys.argv[1] if len(sys.argv) > 1 else f"{F2}/runs/ckpt_rlcd.pt"
    net = F2Net()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()

    z = np.load(f"{F2}/encoded.npz")
    calib_idx = np.where(z["is_calib"] == 1)[0]
    data = collect(net, z, calib_idx)

    temps = {}
    report = {}
    for t, (logits, gold, k) in data.items():
        best_T, best_s = 1.0, None
        for T in np.arange(0.5, 3.01, 0.02):
            p = torch.softmax(logits / T, -1)
            if t == "score":
                s = rps_np(p.numpy(), gold.numpy())
            else:
                s = F.nll_loss(torch.log(p.clamp(1e-9)).double(), gold).item()
            if best_s is None or s < best_s:
                best_s, best_T = s, float(T)
        temps[t] = round(best_T, 2)
        p = torch.softmax(logits / best_T, -1)
        acc = float((p.argmax(-1) == gold).float().mean())
        report[t] = {"n": len(gold), "calib_acc": round(acc, 4),
                     "objective": round(best_s, 5), "T": round(best_T, 2)}
        print(t, report[t])

    ck["temperatures"] = temps
    ck["calibration"] = {"method": "per-task temperature (grid 0.5-3.0)",
                         "report": report}
    # universal naming: *_rlcd.pt -> *_calibrated.pt (or *_cal.pt fallback)
    if "_rlcd.pt" in ck_path:
        out_path = ck_path.replace("_rlcd.pt", "_calibrated.pt")
    elif ck_path.endswith(".pt"):
        out_path = ck_path[:-3] + "_cal.pt"
    else:
        out_path = f"{F2}/runs/ckpt_calibrated.pt"
    torch.save(ck, out_path)
    with open(f"{F2}/runs/temperatures.json", "w") as f:
        json.dump(temps, f, indent=1)
    print("calibrated ckpt ->", out_path)
    print("temperatures:", temps)


META = None
if __name__ == "__main__":
    META = np.array([m["task"] for m in
                     json.load(open(f"{F2}/encoded_meta.json"))])
    main()
