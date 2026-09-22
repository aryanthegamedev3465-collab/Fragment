"""release_hf.py — release fragment-1 v2 to the Hugging Face Hub.

The model keeps the name fragment-1 (user decision): v2 replaces v1 IN PLACE.

1. upload release/ to FrameXlabs/fragment-1 with delete_patterns="*"
   (single commit: old v1 files are removed and the new files land atomically —
   if the upload fails, the old model is still intact)
2. verify the new release files are present and the old v1 files are gone
3. verify exactly ONE model remains on the account: FrameXlabs/fragment-1
"""
import json
import sys
import time
from huggingface_hub import HfApi

F2 = "/home/z/my-project/fragment-lab/f2"
REPO_ID = "FrameXlabs/fragment-1"

NEW_FILES = {"model.safetensors", "f2.py", "f2_config.json", "tokenizer.json", "README.md"}
OLD_V1_FILES = {"fragment-final.pt", "config.json"}  # v1 artifacts that must be gone


def main():
    secrets = json.load(open(f"{F2}/secrets.json"))
    api = HfApi(token=secrets["hf_token"])

    print(f"uploading release files to {REPO_ID} (replacing v1 in place) ...")
    version = json.load(open(f"{F2}/release/f2_config.json"))["version"]
    for attempt in range(3):
        try:
            api.upload_folder(
                folder_path=f"{F2}/release",
                repo_id=REPO_ID,
                repo_type="model",
                delete_patterns="*",  # wipes old v1 files in the same commit
                commit_message=f"fragment-1 v{version} — replaces v1 in place "
                               f"(same name, one model, no graveyard)",
            )
            break
        except Exception as e:
            print(f"upload attempt {attempt+1} failed: {e}")
            time.sleep(10)
    else:
        print("UPLOAD FAILED — old v1 model untouched, safe to retry")
        sys.exit(1)

    info = api.model_info(REPO_ID, files_metadata=True)
    files = {s.rfilename for s in info.siblings}
    print("files on hub:", sorted(files))
    missing = NEW_FILES - files
    leftover = OLD_V1_FILES & files
    if missing:
        print("MISSING FILES:", missing)
        sys.exit(1)
    if leftover:
        print("OLD V1 FILES STILL PRESENT:", leftover)
        sys.exit(1)

    remaining = sorted(m.id for m in api.list_models(author="FrameXlabs"))
    print("models on account:", remaining)
    if remaining != [REPO_ID]:
        print(f"UNEXPECTED STATE — expected exactly ['{REPO_ID}']")
        sys.exit(1)
    print(f"RELEASE COMPLETE — exactly one model remains: {REPO_ID} (now v{version})")

    with open(f"{F2}/runs/release_record.json", "w") as f:
        json.dump({"released": REPO_ID, "replaced_in_place": "FrameXlabs/fragment-1 v1",
                   "version": version, "remaining": remaining, "ts": time.time()},
                  f, indent=1)


if __name__ == "__main__":
    main()
