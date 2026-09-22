"""github_sync.py — push the Fragment project to GitHub (FrameXlabs).

Creates the repo if needed, syncs code + roadmap + release artifacts.
NEVER commits secrets.json, data/, runs/, or encoded caches.
README.md / BENCHMARKS.md / fragment1.py / assets/ are hand-written and only
patched (progress marker, post-release v1->v2 text swap), never regenerated.

The model keeps the name fragment-1 across versions: links never flip, the
Hub repo is replaced in place when v2 ships.
"""
import json
import os
import re
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


def update_readme(progress, released=False):
    """The README is hand-written. Only two things change automatically:
    the roadmap progress marker and the post-release v1->v2 text swap.
    The model name fragment-1 never flips — links stay valid forever."""
    path = f"{GH}/README.md"
    txt = open(path).read()
    txt = re.sub(r"<!--ROADMAP:\d+/\d+-->", f"<!--ROADMAP:{progress}-->", txt)
    txt = re.sub(r"Current roadmap progress: \*\*task \d+/\d+\*\*",
                 f"Current roadmap progress: **task {progress}**", txt)
    if released:
        for old, new in POST_RELEASE_PATCHES:
            if old in txt:
                txt = txt.replace(old, new)
            elif new not in txt:
                print(f"[gh] WARN: post-release patch not found: {old[:60]!r}")
    with open(path, "w") as f:
        f.write(txt)


# applied once, right after the v2 release swaps the Hub repo in place;
# idempotent — a second run finds the new strings and stays silent
POST_RELEASE_PATCHES = [
    ("When v2 beats v1 on the same held-out benchmark, it ships **under the same name `fragment-1`** — v1 is replaced in place, one model stays on the account, no graveyard. Until then, `fragment-1` v1 is the one to use.",
     "v2 beat v1 on the same held-out benchmark and shipped **under the same name `fragment-1`** — v1 was replaced in place, one model stays on the account, no graveyard. The badge and the quickstart below now point at v2."),
    ("(live) | fragment-1 v2 (training) |",
     "(superseded) | fragment-1 v2 (live) |"),
    ("The release rule is simple: when v2 beats v1 on the same benchmark, it's released **under the same name `FrameXlabs/fragment-1`** and v1 is replaced in place. One model, no confusion — the name `fragment-1` always points at the strongest version.",
     "The release rule was simple: v2 had to beat v1 on the same benchmark, then ship **under the same name `FrameXlabs/fragment-1`** with v1 replaced in place. That's what happened — the name `fragment-1` now points at v2, the strongest version."),
    ("fragment-1 v2 is being trained from scratch by an automated driver running a 50-task roadmap",
     "fragment-1 v2 shipped and replaced v1 in place; the same automated driver keeps running the 50-task roadmap's improvement rounds"),
    ('from fragment1 import Fragment1\n\nm = Fragment1.from_pretrained("FrameXlabs/fragment-1")   # or a local folder',
     'from f2 import Fragment\n\nm = Fragment.from_pretrained("FrameXlabs/fragment-1")   # v2 — same name, v1 replaced in place'),
    ("print(res[\"answers\"][\"topic\"][\"probabilities\"]) # -> {'World': 0.0396, 'Sports': 0.0023, 'Business': 0.9498, 'Technology': 0.0084}",
     "print(res[\"answers\"][\"topic\"][\"probabilities\"]) # full distribution over the four options"),
    ("print(res[\"answers\"][\"sentiment\"][\"noul\"])      # -> 0.51 (genuinely ambiguous headline)",
     "print(res[\"answers\"][\"sentiment\"][\"noul\"])      # calibrated P(positive)"),
    ("pip install torch huggingface_hub",
     "pip install torch safetensors huggingface_hub"),
    ("The weights on the Hub are a plain PyTorch checkpoint (`fragment-final.pt`, plus a float16 copy and a manifest). `fragment1.py` in this repo is the entire runtime — one file, no dependencies beyond torch.",
     "The weights on the Hub ship as one `model.safetensors` with the runtime (`f2.py`), `f2_config.json` and the tokenizer next to it — torch and safetensors, nothing else. The v1 runtime `fragment1.py` stays in this repo for the record."),
    ("fragment-1, run by us, on our sandbox CPU.",
     "fragment-1 **v1** (superseded by v2 under the same name), run by us, on our sandbox CPU. v2 numbers land in [`benchmarks/metrics.json`](benchmarks/metrics.json) on the sync after release evaluation."),
    ("A 5M-parameter encoder, a logit at each option marker, softmax, done.",
     "A 9M-parameter encoder, a logit at each option marker, softmax, done."),
    ("* **It's tiny.** 4.96M parameters. Laya is 421M and it shows.",
     "* **It's tiny.** 8.98M parameters. Laya is 421M and it shows."),
]


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
        "description": "A tiny typed-decision model with calibrated "
                       "probabilities — one forward pass, no text generation. "
                       "Trained from scratch on a CPU by FrameXlabs.",
        "private": False,
        "has_issues": True,
    })
    print("create repo:", r.get("full_name") or r)
    # keep the description fresh if the repo already existed
    gh_api("PATCH", f"/repos/{gh_api('GET', '/user', token).get('login')}/{REPO}", token, {
        "description": "A tiny typed-decision model with calibrated "
                       "probabilities — one forward pass, no text generation. "
                       "Trained from scratch on a CPU by FrameXlabs.",
    })

    # prepare working tree
    os.makedirs(GH, exist_ok=True)
    if not os.path.exists(f"{GH}/.git"):
        sh(f'git init -b main "{GH}"', "/")
    with open(f"{GH}/.gitignore", "w") as f:
        f.write("secrets.json\ndata/\nruns/\nencoded*\nitems.jsonl\n"
                "__pycache__/\n*.pt\nnohup.out\nbenchmarks/data/\n"
                "benchmarks/results.json\n")

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

    # metrics snapshot for anyone reading the repo
    try:
        metrics = json.load(open(f"{F2}/runs/metrics.json"))
        os.makedirs(f"{GH}/benchmarks", exist_ok=True)
        shutil.copy(f"{F2}/runs/metrics.json", f"{GH}/benchmarks/metrics.json")
    except Exception:
        pass

    # roadmap progress (patch the hand-written README, never regenerate it)
    try:
        state = json.load(open(f"{F2}/state.json"))
        cur = state["current"]
        total = len(json.load(open(f"{F2}/roadmap.json"))["tasks"])
        progress = f"{cur}/{total}"
    except Exception:
        progress = "0/50"
    # has the v2 release happened? (local release record, or v2 file
    # signature model.safetensors on the Hub's fragment-1 repo)
    released = os.path.exists(f"{F2}/runs/release_record.json")
    if not released:
        try:
            hf_tok = secrets.get("hf_token", "")
            req = urllib.request.Request(
                "https://huggingface.co/api/models/FrameXlabs/fragment-1",
                headers={"Authorization": f"Bearer {hf_tok}"})
            info = json.load(urllib.request.urlopen(req))
            hub_files = {s.get("rfilename", "") for s in info.get("siblings", [])}
            released = "model.safetensors" in hub_files
        except Exception:
            pass
    update_readme(progress, released)
    print(f"[gh] readme patched: progress={progress}, v2 released={released}")

    # commit + push
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
    sh('git add -A', GH)
    sh(f'git -c user.name="FrameXlabs" -c user.email="aryanthegamedev3465@gmail.com" '
       f'commit -m "Fragment: sync pipeline + progress {stamp}" '
       '|| true', GH)
    remote = f"https://{login}:{token}@github.com/{login}/{REPO}.git"
    sh(f'git remote set-url origin "{remote}" || git remote add origin "{remote}"', GH)
    sh("git push -u origin main", GH)
    print(f"[gh] pushed -> https://github.com/{login}/{REPO}")
    return f"https://github.com/{login}/{REPO}"


if __name__ == "__main__":
    main()
