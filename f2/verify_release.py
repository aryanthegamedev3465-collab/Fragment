"""verify_release.py — end-to-end release verification.

1. exactly one model on the Hub: FrameXlabs/Fragment
2. release files present on the Hub
3. f2 runtime loads and decide() produces sane typed answers
"""
import json
import sys

F2 = "/home/z/my-project/fragment-lab/f2"


def main():
    secrets = json.load(open(f"{F2}/secrets.json"))
    from huggingface_hub import HfApi
    api = HfApi(token=secrets["hf_token"])

    models = [m.id for m in api.list_models(author="FrameXlabs")]
    print("models:", models)
    ok = models == ["FrameXlabs/Fragment"]
    print("PASS: exactly one model" if ok else "FAIL: model list wrong")

    info = api.model_info("FrameXlabs/Fragment", files_metadata=True)
    files = {s.rfilename for s in info.siblings}
    need = {"model.safetensors", "f2.py", "f2_config.json", "tokenizer.json", "README.md"}
    missing = need - files
    print("files:", sorted(files))
    print("PASS: files complete" if not missing else f"FAIL: missing {missing}")

    # smoke test from local release dir
    sys.path.insert(0, f"{F2}/release")
    from f2 import Fragment
    m = Fragment(f"{F2}/release")
    res = m.decide(
        state="Hi, we were billed twice for March. Please refund the duplicate "
              "today or we will cancel our plan.",
        questions={
            "refund": {"type": "noul",
                       "instructions": "Does the user explicitly request a refund?"},
            "urgency": {"type": "score",
                        "instructions": "How urgent is this request?",
                        "criteria": ["not urgent", "soon", "critical or blocking"]},
            "topic": {"type": "choice",
                      "instructions": "Which team should handle this request?",
                      "criteria": {"billing": "invoices, payments, refunds",
                                   "technical": "bugs, outages, errors",
                                   "other": "everything else"}},
        })
    print(json.dumps(res, indent=1))
    a = res["answers"]
    sane = (0.0 <= a["refund"]["noul"] <= 1.0 and
            0.0 <= a["urgency"]["score"] <= 2.0 and
            a["topic"]["choice"] in {"billing", "technical", "other"})
    print("PASS: decide() smoke test" if sane else "FAIL: decide() output insane")
    return 0 if (ok and not missing and sane) else 1


if __name__ == "__main__":
    sys.exit(main())
