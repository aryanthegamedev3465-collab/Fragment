"""release_hf.py — release Fragment to the Hugging Face Hub.

1. create FrameXlabs/Fragment (exist_ok)
2. upload the release/ folder
3. verify files
4. delete the previous strongest model (FrameXlabs/fragment-1)
5. verify exactly ONE model remains
"""
import json
import sys
import time
from huggingface_hub import HfApi

F2 = "/home/z/my-project/fragment-lab/f2"


def main(dry_run=False):
    secrets = json.load(open(f"{F2}/secrets.json"))
    api = HfApi(token=secrets["hf_token"])

    print("creating repo FrameXlabs/Fragment ...")
    api.create_repo(repo_id="FrameXlabs/Fragment", repo_type="model",
                    private=False, exist_ok=True)

    print("uploading release files ...")
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=f"{F2}/release",
                              repo_id="FrameXlabs/Fragment",
                              repo_type="model",
                              commit_message=f"Fragment release "
                                             f"{json.load(open(f'{F2}/release/f2_config.json'))['version']}")
            break
        except Exception as e:
            print(f"upload attempt {attempt+1} failed: {e}")
            time.sleep(10)
    else:
        print("UPLOAD FAILED")
        sys.exit(1)

    info = api.model_info("FrameXlabs/Fragment", files_metadata=True)
    files = [s.rfilename for s in info.siblings]
    print("files on hub:", files)
    needed = {"model.safetensors", "f2.py", "f2_config.json", "tokenizer.json", "README.md"}
    missing = needed - set(files)
    if missing:
        print("MISSING FILES:", missing)
        sys.exit(1)

    print("deleting previous strongest model FrameXlabs/fragment-1 ...")
    api.delete_repo(repo_id="FrameXlabs/fragment-1", repo_type="model")

    remaining = [m.id for m in api.list_models(author="FrameXlabs")]
    print("remaining models:", remaining)
    if remaining != ["FrameXlabs/Fragment"]:
        print("UNEXPECTED STATE — expected exactly ['FrameXlabs/Fragment']")
        sys.exit(1)
    print("RELEASE COMPLETE — exactly one model remains: FrameXlabs/Fragment")

    with open(f"{F2}/runs/release_record.json", "w") as f:
        json.dump({"released": "FrameXlabs/Fragment", "deleted": "FrameXlabs/fragment-1",
                   "remaining": remaining, "ts": time.time()}, f, indent=1)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
