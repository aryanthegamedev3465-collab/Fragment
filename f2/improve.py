"""improve.py — one improvement round: train -> rlcd -> calibrate -> evaluate.

If the round beats the current best (avg_acc on the held-out benchmark), the
checkpoint is promoted and re-released to the Hub; otherwise it is rolled back.
Usage:
  python3 improve.py --round 3 --title "epoch 3" \
      --train '{"epochs":1,"lr":1e-4}' --rlcd '{"steps":250}'
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

F2 = "/home/z/my-project/fragment-lab/f2"
BEST = f"{F2}/runs/ckpt_calibrated.pt"      # current best (promoted)
STATE = f"{F2}/runs/improve_state.json"


def sh(cmd, log):
    print(f"[improve] $ {cmd}", flush=True)
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
        r = subprocess.run(cmd, shell=True, cwd=F2, stdout=f,
                           stderr=subprocess.STDOUT)
    return r.returncode


def avg_acc(path):
    try:
        return json.load(open(path))["summary"]["avg_acc"]
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--train", default="{}")
    ap.add_argument("--rlcd", default="{}")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_rlcd", action="store_true")
    args = ap.parse_args()

    R = args.round
    log = f"{F2}/runs/improve_round_{R}.log"
    state = {"round": R, "title": args.title,
             "base_avg_acc": avg_acc(f"{F2}/runs/metrics.json")}
    print(f"[improve] round {R}: {args.title} | base avg_acc={state['base_avg_acc']}")

    ck_sup = f"{F2}/runs/ckpt_round{R}_sup.pt"
    ck_rlcd = f"{F2}/runs/ckpt_round{R}_rlcd.pt"
    ck_cal = f"{F2}/runs/ckpt_round{R}_calibrated.pt"
    met = f"{F2}/runs/metrics_round{R}.json"

    if not args.skip_train:
        tcfg = {"resume": BEST, "out": ck_sup, **json.loads(args.train)}
        with open(f"{F2}/runs/tcfg_{R}.json", "w") as f:
            json.dump(tcfg, f)
        if sh(f"python3 train_supervised.py --config runs/tcfg_{R}.json", log) != 0:
            state["status"] = "train_failed"
            json.dump(state, open(STATE, "w"))
            sys.exit(1)

    if not args.skip_rlcd:
        rcfg = {"resume": ck_sup if not args.skip_train else BEST,
                "out": ck_rlcd, **json.loads(args.rlcd)}
        with open(f"{F2}/runs/rcfg_{R}.json", "w") as f:
            json.dump(rcfg, f)
        if sh(f"python3 train_rlcd.py --config runs/rcfg_{R}.json", log) != 0:
            state["status"] = "rlcd_failed"
            json.dump(state, open(STATE, "w"))
            sys.exit(1)
    else:
        shutil.copy(ck_sup if not args.skip_train else BEST, ck_rlcd)

    sh(f"python3 calibrate.py {ck_rlcd}", log)

    sh(f"python3 evaluate.py {ck_cal} {met}", log)
    new_acc = avg_acc(met)
    state["new_avg_acc"] = new_acc
    print(f"[improve] round {R} avg_acc: {state['base_avg_acc']} -> {new_acc}")

    if new_acc > state["base_avg_acc"]:
        shutil.copy(ck_cal, BEST)
        shutil.copy(met, f"{F2}/runs/metrics.json")
        state["status"] = "promoted"
        print("[improve] promoted — re-releasing")
        sh(f"python3 package_release.py 2.{R}", log)
        sh("python3 release_hf.py", log)
    else:
        state["status"] = "rolled_back"
        print("[improve] no improvement — rolled back")
    json.dump(state, open(STATE, "w"), indent=1)


if __name__ == "__main__":
    main()
