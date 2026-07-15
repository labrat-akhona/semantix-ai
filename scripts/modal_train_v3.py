"""Cloud-train POPIA-Judge v3 on Modal — deberta-v3-base fine-tune.

Runs `scripts/train_popia_v2.py` against `cross-encoder/nli-deberta-v3-base`
(184M params, ~2.25x the v2 base) on a Modal-provisioned L4 GPU.
~10 min, ~$0.10–0.20 per run on L4 ($0.80/hr). Uploads the trained
artifact to HF as `labrat-aiko/nli-popia-v3`.

Setup (one-time)
----------------
    pip install modal
    modal token new                                         # auth
    modal secret create huggingface HF_TOKEN=hf_yourtoken   # for upload

Run
---
    modal run scripts/modal_train_v3.py                     # defaults
    modal run scripts/modal_train_v3.py --epochs 8 --seed 1
    modal run scripts/modal_train_v3.py --no-push           # train only, no upload

The remote function clones the repo fresh each run from origin/master,
runs the generalized training script with --base-model + --out-dir
overrides, captures the release_gate.json, and (by default) uploads to
HF. Local entrypoint prints the metrics when remote finishes.

ONNX export is skipped on the cloud (it's CPU work and pollutes the GPU
container with extra deps). Run `scripts/calibrate_popia_v2.py` and any
ONNX export locally after the pytorch weights are pulled from HF.
"""

from __future__ import annotations

import json

import modal

APP_NAME = "popia-v3-train"
REPO_URL = "https://github.com/labrat-akhona/semantix-ai.git"
HF_REPO = "labrat-aiko/nli-popia-v3"
BASE_MODEL = "cross-encoder/nli-deberta-v3-base"
OUT_DIR = "out/nli-popia-v3"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.46",
        "datasets",
        "scikit-learn",
        "huggingface_hub",
        "sentencepiece",  # required by deberta-v3 tokenizer
        "accelerate",
        "protobuf",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="L4",
    timeout=60 * 60,  # 1h hard cap
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(epochs: int = 6, seed: int = 42, push_to_hf: bool = True) -> dict:
    import os
    import subprocess
    from pathlib import Path

    print(f"[modal] cloning {REPO_URL}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/work"], check=True)
    os.chdir("/work")

    cmd = [
        "python",
        "scripts/train_popia_v2.py",
        "--base-model",
        BASE_MODEL,
        "--out-dir",
        OUT_DIR,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--skip-quantize",
    ]
    print(f"[modal] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    report_path = Path(OUT_DIR) / "release_gate.json"
    report = json.loads(report_path.read_text())
    print(f"[modal] release_gate.json:\n{json.dumps(report, indent=2)}")

    if push_to_hf:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[modal] HF_TOKEN missing — skipping upload")
        else:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import HfHubHTTPError

            api = HfApi(token=token)
            api.create_repo(
                repo_id=HF_REPO, repo_type="model", exist_ok=True, private=False, token=token
            )
            try:
                api.upload_folder(
                    folder_path=OUT_DIR,
                    repo_id=HF_REPO,
                    repo_type="model",
                    commit_message=f"Initial release: POPIA-Judge v3 ({BASE_MODEL}, seed={seed}, epochs={epochs})",
                    ignore_patterns=["pytorch/checkpoint-*"],  # keep only the best weights
                    token=token,
                )
                print(f"[modal] uploaded to https://huggingface.co/{HF_REPO}")
            except HfHubHTTPError as e:
                print(f"[modal] upload error: {e}")

    return report


@app.local_entrypoint()
def main(epochs: int = 6, seed: int = 42, push: bool = True):
    print(f"[local] kicking off remote train (epochs={epochs}, seed={seed}, push={push})")
    report = train.remote(epochs=epochs, seed=seed, push_to_hf=push)

    print("\n" + "=" * 60)
    print("v3 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Base:    {report.get('base_model')}")
    print(f"Seed:    {report.get('seed')}")
    print(f"Epochs:  {report.get('epochs')}")
    print(
        f"\nv1 holdout macro F1: {report.get('v2_model_v1_f1'):.4f}  (stock {report.get('stock_v1_f1'):.4f})"
    )
    print(
        f"v2 holdout macro F1: {report.get('v2_model_v2_f1'):.4f}  (stock {report.get('stock_v2_f1'):.4f})"
    )
    print(f"Gate:    {'PASS' if report.get('gate_pass') else 'FAIL'}")
    if push:
        print(f"\nUploaded to: https://huggingface.co/{HF_REPO}")
    print(
        f"Next: run scripts/calibrate_popia_v2.py against {HF_REPO} to fit a v3 calibration constant."
    )
