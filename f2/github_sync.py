"""github_sync.py — push the Fragment project to GitHub (FrameXlabs).

Creates the repo if needed, syncs code + roadmap + release artifacts.
NEVER commits secrets.json, data/, runs/, or encoded caches.
"""
import json
import os
import subprocess
import sys
import urllib.request

F2 = "/home/z/my-project/fragment-lab/f2"
GH = "/home/z/my-project/fragment-lab/github"
REPO = "Fragment"


def gh_api(method, path, token, body=None):
    req = urllib.request.Request(f"https://api.github.com{path}", method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    data = json.dumps(body).encode() if body else None
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data) as r:
            return json.load(r) if r.status != 204 else {}
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:400]}


def sh(cmd, cwd):
    print(f"[gh] {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")


def main():
    secrets = json.load(open(f"{F2}/secrets.json"))
    token = secrets["github_token"]

    # who am I
    me = gh_api("GET", "/user", token)
    login = me.get("login")
    if not login:
        print("GitHub auth failed:", me)
        sys.exit(1)
    print(f"github login: {login}")

    # create repo (ignore 422 = already exists)
    r = gh_api("POST", "/user/repos", token, {
        "name": REPO,
        "description": "Fragment — from-scratch System-One decision model "
                       "(Jev/Laya family) by FrameXlabs. Typed decisions, "
                       "calibrated probabilities, RLCD-trained.",
        "private": False,
        "has_issues": True,
    })
    print("create repo:", r.get("full_name") or r)

    # prepare working tree
    os.makedirs(GH, exist_ok=True)
    if not os.path.exists(f"{GH}/.git"):
        sh(f'git init -b main "{GH}"', "/")
    with open(f"{GH}/.gitignore", "w") as f:
        f.write("secrets.json\ndata/\nruns/\nencoded*\nitems.jsonl\n"
                "__pycache__/\n*.pt\nnohup.out\n")

    # sync files
    import shutil
    for fn in ["f2model.py", "f2.py", "prepare_data.py", "build_tokenizer.py",
               "encode_data.py", "train_supervised.py", "train_rlcd.py",
               "calibrate.py", "evaluate.py", "improve.py", "drive.py",
               "package_release.py", "release_hf.py", "github_sync.py",
               "verify_release.py", "roadmap.json", "state.json"]:
        src = f"{F2}/{fn}"
        if os.path.exists(src):
            os.makedirs(f"{GH}/f2", exist_ok=True)
            shutil.copy(src, f"{GH}/f2/{fn}")

    # release artifacts (incl. weights when available)
    if os.path.isdir(f"{F2}/release"):
        os.makedirs(f"{GH}/model", exist_ok=True)
        for fn in os.listdir(f"{F2}/release"):
            shutil.copy(f"{F2}/release/{fn}", f"{GH}/model/{fn}")

    # repo README
    metrics = {}
    try:
        metrics = json.load(open(f"{F2}/runs/metrics.json")).get("summary", {})
    except Exception:
        pass
    try:
        state = json.load(open(f"{F2}/state.json"))
        cur = state["current"]
        total = len(json.load(open(f"{F2}/roadmap.json"))["tasks"])
        progress = f"{cur}/{total}"
    except Exception:
        progress = "0/50"
    readme = f"""# Fragment

**Fragment** is a from-scratch **System-One decision model** by
[FrameXlabs](https://huggingface.co/FrameXlabs) — the open, CPU-trained member
of the Jev (TypeSafe AI) / Laya (convaiinnovations) family of non-autoregressive
decision models.

Give it a **state** (any text) and **typed questions** (choice / score / noul);
it returns **typed answers with calibrated probabilities in a single forward
pass**. No text generation, nothing to parse, nothing to hallucinate.

- Model card + weights: https://huggingface.co/FrameXlabs/Fragment
- Current roadmap progress: {progress}
- Held-out benchmark summary: {json.dumps(metrics)}

## Architecture

{json.dumps(json.load(open(f'{GH}/model/f2_config.json'))['architecture'], indent=2) if os.path.exists(f'{GH}/model/f2_config.json') else "pending first release"}

## Pipeline

```
prepare_data.py     typed items from SST-2 / BoolQ / AG News / Yelp (+extras)
build_tokenizer.py  from-scratch BPE (16,384 vocab, atomic @option markers)
encode_data.py      compact numpy cache
train_supervised.py cross-entropy on gold markers (bf16, CPU)
train_rlcd.py       RLCD: GRPO-style noise exploration, proper scoring rules
calibrate.py        per-qtype temperature scaling
evaluate.py         held-out benchmark (acc / ECE / Brier / RPS)
improve.py          one improvement round (promote if better)
drive.py            roadmap driver — 50 long-running tasks
release_hf.py       Hub release (one model remains: FrameXlabs/Fragment)
```

## Quickstart

```python
import sys; sys.path.insert(0, "model")
from f2 import Fragment

m = Fragment("./model")
res = m.decide(
    state="We were billed twice for March. Please refund the duplicate today.",
    questions={{"refund": {{"type": "noul",
                           "instructions": "Does the user request a refund?"}}}})
print(res["answers"]["refund"]["noul"])
```

## Reproduce

```
python3 f2/prepare_data.py
python3 f2/build_tokenizer.py
python3 f2/encode_data.py
python3 f2/train_supervised.py
python3 f2/train_rlcd.py
python3 f2/calibrate.py
python3 f2/evaluate.py
```

Apache-2.0. Trained and released by FrameXlabs.
"""
    with open(f"{GH}/README.md", "w") as f:
        f.write(readme)

    # commit + push
    sh('git add -A', GH)
    sh('git -c user.name="FrameXlabs" -c user.email="aryanthegamedev3465@gmail.com" '
       'commit -m "Fragment: training pipeline, roadmap, release artifacts" '
       '|| true', GH)
    remote = f"https://{login}:{token}@github.com/{login}/{REPO}.git"
    sh(f'git remote set-url origin "{remote}" || git remote add origin "{remote}"', GH)
    sh("git push -u origin main", GH)
    print(f"[gh] pushed -> https://github.com/{login}/{REPO}")
    return f"https://github.com/{login}/{REPO}"


if __name__ == "__main__":
    main()
