"""Upload local model artifacts to HuggingFace Hub.

Usage:
    HF_TOKEN=hf_... python scripts/upload_to_hf.py [model_name]

Where model_name is one of:
    nli-popia-v2
    sa-compliance-embeddings-v1
    popia-instruct-v0
    all  (uploads every model that has a local out/ directory)

The script reads the upload manifest below and pushes each artifact
folder to the matching labrat-aiko repo with a sensible commit message.

Token authority required: write access to labrat-aiko/<model_name>.
Token can be set via HF_TOKEN env var OR will be read from huggingface_hub's
cached login (huggingface-cli login).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# (local-dir, hf-repo-id, repo-type, commit-msg)
MANIFEST: dict[str, tuple[str, str, str, str]] = {
    "nli-popia-v2": (
        "out/nli-popia-v2",
        "labrat-aiko/nli-popia-v2",
        "model",
        "Initial release of nli-popia-v2 — 10-clause POPIA NLI judge",
    ),
    "sa-compliance-embeddings-v1": (
        "out/sa-compliance-embeddings-v1",
        "labrat-aiko/sa-compliance-embeddings-v1",
        "model",
        "Initial release of sa-compliance-embeddings-v1",
    ),
    "popia-instruct-v0": (
        "out/popia-instruct-v0/adapter",
        "labrat-aiko/popia-instruct-v0",
        "model",
        "Initial release of popia-instruct-v0 (QLoRA adapter on Phi-3-mini)",
    ),
}


def upload_one(name: str) -> None:
    if name not in MANIFEST:
        sys.exit(f"unknown model {name!r} -- choose from {sorted(MANIFEST)} or 'all'")
    local_dir, repo_id, repo_type, msg = MANIFEST[name]
    local = Path(local_dir)
    if not local.exists():
        print(f"[skip] {name}: {local_dir} does not exist (not trained yet?)")
        return

    token = os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id, token=token, repo_type=repo_type, exist_ok=True)
    print(f"[ok] repo ensured: {repo_id}")

    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=msg,
        ignore_patterns=["*_tmp/*", "*.tmp", "checkpoints/*"],
    )
    print(f"[ok] uploaded {local} -> https://huggingface.co/{repo_id}")


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "all"
    if target == "all":
        for name in MANIFEST:
            upload_one(name)
    else:
        upload_one(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
