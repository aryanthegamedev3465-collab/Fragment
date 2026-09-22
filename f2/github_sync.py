"""github_sync.py — push the Fragment project to GitHub (FrameXlabs).

Creates the repo if needed, syncs code + roadmap + release artifacts.
NEVER commits secrets.json, data/, runs/, or encoded caches.
README.md / BENCHMARKS.md / fragment1.py / assets/ are hand-written and only
patched (progress marker, current-model pointer), never regenerated.
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


def update_readme(progress, hf_model="fragment-1"):
    """The README is hand-written. Only two things change automatically:
    the roadmap progress marker and the 'current model' pointer after release."""
    path = f"{GH}/README.md"
    txt = open(path).read()
    txt = re.sub(r"<!--ROADMAP:\d+/\d+-->", f"<!--ROADMAP:{progress}-->", txt)
    txt = re.sub(r"Current roadmap progress: \*\*task \d+/\d+\*\*",
                 f"Current roadmap progress: **task {progress}**", txt)
    if hf_model == "Fragment":
        # release happened: the live model is now FrameXlabs/Fragment
        txt = txt.replace("current%20model-FrameXlabs%2Ffragment--1-yellow",
                          "current%20model-FrameXlabs%2FFragment-yellow")
        txt = txt.replace("https://huggingface.co/FrameXlabs/fragment-1",
                          "https://huggingface.co/FrameXlabs/Fragment")
        txt = txt.replace("[`fragment-1`](https://huggingface.co/FrameXlabs/Fragment) (released)",
                          "[`Fragment`](https://huggingface.co/FrameXlabs/Fragment) (released)")
        txt = txt.replace("# fragment-1 (ours, measured)", "# Fragment (ours, measured)")
    with open(path, "w") as f:
        f.write(txt)


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
    # which model is live on the Hub right now?
    hf_model = "fragment-1"
    try:
        hf_tok = secrets.get("hf_token", "")
        req = urllib.request.Request("https://huggingface.co/api/models?author=FrameXlabs",
                                     headers={"Authorization": f"Bearer {hf_tok}"})
        ids = [m["modelId"] for m in json.load(urllib.request.urlopen(req))]
        if "FrameXlabs/Fragment" in ids:
            hf_model = "Fragment"
    except Exception:
        pass
    update_readme(progress, hf_model)
    print(f"[gh] readme patched: progress={progress}, live model={hf_model}")

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
