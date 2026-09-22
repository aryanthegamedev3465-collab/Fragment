"""drive.py — Fragment roadmap driver (long-running task engine).

Modes:
  python3 drive.py loop    — run tasks sequentially until roadmap is done
  python3 drive.py once    — cron-friendly: restart loop if it died, else report
  python3 drive.py status  — print progress
Locking: runs/drive.lock carries {pid, ts, task}; heartbeat updated per minute.
"""
import json
import os
import subprocess
import sys
import time

F2 = "/home/z/my-project/fragment-lab/f2"
ROADMAP = f"{F2}/roadmap.json"
STATE = f"{F2}/state.json"
LOCK = f"{F2}/runs/drive.lock"
STALE = 300  # seconds without heartbeat = considered dead


def read_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"current": 0, "history": []}


def write_state(s):
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def beat(extra=None):
    d = {"pid": os.getpid(), "ts": time.time()}
    if extra:
        d.update(extra)
    json.dump(d, open(LOCK, "w"))


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def lock_is_fresh():
    try:
        d = json.load(open(LOCK))
        return pid_alive(d["pid"]) and (time.time() - d["ts"]) < STALE
    except Exception:
        return False


def run_loop():
    print("[drive] loop start", flush=True)
    while True:
        state = read_state()
        tasks = json.load(open(ROADMAP))["tasks"]
        cur = state["current"]
        if cur >= len(tasks):
            print("[drive] roadmap complete", flush=True)
            beat({"task": "DONE"})
            break
        t = tasks[cur]
        attempts = state.get(f"attempts_{t['id']}", 0)
        print(f"[drive] task {t['id']}: {t['title']} (attempt {attempts+1})", flush=True)
        beat({"task": f"{t['id']}: {t['title']}"})
        log = f"{F2}/runs/task_{t['id']:02d}.log"
        t0 = time.time()
        with open(log, "w") as lf:
            lf.write(f"# task {t['id']}: {t['title']}\n# cmd: {t['cmd']}\n\n")
            lf.flush()
            proc = subprocess.Popen(t["cmd"], shell=True, cwd=F2, stdout=lf,
                                    stderr=subprocess.STDOUT)
            # heartbeat while the task runs
            while proc.poll() is None:
                beat({"task": f"{t['id']}: {t['title']}"})
                time.sleep(30)
            rc = proc.returncode
        dur = round(time.time() - t0, 1)
        rec = {"id": t["id"], "title": t["title"], "rc": rc,
               "dur_s": dur, "ts": time.time()}
        if rc == 0:
            rec["status"] = "completed"
            state["current"] = cur + 1
            state.pop(f"attempts_{t['id']}", None)
        else:
            attempts += 1
            state[f"attempts_{t['id']}"] = attempts
            if attempts >= 2:
                rec["status"] = "failed_final"
                state["current"] = cur + 1
            else:
                rec["status"] = "failed_retry"
        state["history"].append(rec)
        write_state(state)
        print(f"[drive] task {t['id']} -> {rec['status']} ({dur}s)", flush=True)
    print("[drive] loop end", flush=True)


def once():
    if lock_is_fresh():
        d = json.load(open(LOCK))
        print(f"[drive] busy (pid {d['pid']}, task {d.get('task')}), not starting")
        return 0
    print("[drive] no live driver — launching loop detached")
    subprocess.Popen(["setsid", "nohup", sys.executable, "-u",
                      f"{F2}/drive.py", "loop"],
                     stdout=open(f"{F2}/runs/drive_once.log", "a"),
                     stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(2)
    return 0


def status():
    state = read_state()
    tasks = json.load(open(ROADMAP))["tasks"]
    print(f"current: {state['current']}/{len(tasks)}")
    for h in state["history"][-8:]:
        print(f"  task {h['id']} {h['status']} rc={h['rc']} {h['dur_s']}s")
    try:
        d = json.load(open(LOCK))
        alive = pid_alive(d["pid"])
        age = round(time.time() - d["ts"], 1)
        print(f"lock: pid={d['pid']} alive={alive} age={age}s task={d.get('task')}")
    except Exception:
        print("lock: none")
    # tail current task log
    cur = state["current"]
    if cur < len(tasks):
        log = f"{F2}/runs/task_{tasks[cur]['id']:02d}.log"
        if os.path.exists(log):
            print("--- last log lines ---")
            lines = open(log, "rb").read()[-2000:].decode(errors="replace")
            print(lines)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "loop":
        run_loop()
    elif mode == "once":
        once()
    else:
        status()
