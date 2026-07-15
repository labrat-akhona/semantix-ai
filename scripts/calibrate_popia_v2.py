"""Temperature-scaling calibration for POPIA-Judge v2.

Runs single-parameter temperature scaling (Guo et al., 2017) on the
shipped ONNX artifact. Produces:

  - out/nli-popia-v2/calibration.json — fitted temperature + metadata.
  - papers/popiajudge-arxiv/figures/reliability_{pre,post}.png

The eval set (eval.jsonl + eval_v2.jsonl, 197 pairs) is split per
(clause, label) stratum: 40% for fitting T, 60% held back for measuring
ECE pre/post. Stratification matters here — random splits leave whole
clauses unrepresented in the fit set on small benchmarks.

Reproducible: deterministic split (seed 42), no GPU, CPU-only ONNX.

Usage:
    python scripts/calibrate_popia_v2.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from scipy.optimize import minimize_scalar
from transformers import AutoTokenizer

ONNX_DIR = Path("out/nli-popia-v2/onnx")
MODEL_PATH = ONNX_DIR / "model.onnx"
EVAL_PATHS = [
    Path("out/nli-popia-v2/eval.jsonl"),
    Path("out/nli-popia-v2/eval_v2.jsonl"),
]
FIG_DIR = Path("papers/popiajudge-arxiv/figures")
CAL_JSON = Path("out/nli-popia-v2/calibration.json")

SEED = 42
FIT_FRAC = 0.40
N_BINS = 10

LABEL_TO_ID = {"contradiction": 0, "entailment": 1, "neutral": 2}


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for p in EVAL_PATHS:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(rows: list[dict], fit_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["clause"], r["label"])].append(r)
    rng = random.Random(seed)
    fit: list[dict] = []
    test: list[dict] = []
    for key in sorted(buckets):
        items = buckets[key][:]
        rng.shuffle(items)
        n_fit = max(1, int(round(len(items) * fit_frac)))
        fit.extend(items[:n_fit])
        test.extend(items[n_fit:])
    return fit, test


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def run_inference(model_path: Path, tokenizer, rows: list[dict], batch: int = 16) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_infos = {i.name: i.type for i in sess.get_inputs()}
    out: list[np.ndarray] = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        enc = tokenizer(
            [r["premise"] for r in chunk],
            [r["hypothesis"] for r in chunk],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="np",
        )
        feed = {}
        for name, type_str in input_infos.items():
            if name not in enc:
                continue
            arr = enc[name]
            if type_str == "tensor(int64)" and arr.dtype != np.int64:
                arr = arr.astype(np.int64)
            feed[name] = arr
        logits = sess.run(None, feed)[0]
        out.append(logits)
    return np.concatenate(out, axis=0)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> tuple[float, list[dict]]:
    confidence = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    bins: list[dict] = []
    for j in range(n_bins):
        lo, hi = float(edges[j]), float(edges[j + 1])
        if j == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append(
                {"lo": lo, "hi": hi, "count": 0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0}
            )
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(acc - conf)
        ece += (cnt / n) * gap
        bins.append(
            {"lo": lo, "hi": hi, "count": cnt, "accuracy": acc, "confidence": conf, "gap": gap}
        )
    return float(ece), bins


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        p = softmax(logits, T)
        idx = np.arange(len(labels))
        return float(-np.log(np.clip(p[idx, labels], 1e-12, 1.0)).mean())

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-5})
    return float(res.x), float(res.fun)


def reliability_plot(bins: list[dict], ece: float, title: str, path: Path) -> None:
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["accuracy"] for b in bins]
    cnts = [b["count"] for b in bins]
    bin_width = bins[0]["hi"] - bins[0]["lo"]

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.bar(
        centers,
        accs,
        width=bin_width * 0.9,
        edgecolor="black",
        color="steelblue",
        alpha=0.75,
        label="Empirical accuracy",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax2 = ax.twinx()
    ax2.bar(centers, cnts, width=bin_width * 0.9, color="red", alpha=0.18)
    ax2.set_ylabel("Bin count", color="red", fontsize=9)
    ax2.tick_params(axis="y", colors="red", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    print(f"loaded {len(rows)} eval pairs")

    fit_rows, test_rows = stratified_split(rows, FIT_FRAC, SEED)
    print(f"stratified split: fit={len(fit_rows)}, test={len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))

    print("inference on fit set ...")
    fit_logits = run_inference(MODEL_PATH, tokenizer, fit_rows)
    print("inference on test set ...")
    test_logits = run_inference(MODEL_PATH, tokenizer, test_rows)

    fit_labels = np.array([LABEL_TO_ID[r["label"]] for r in fit_rows], dtype=np.int64)
    test_labels = np.array([LABEL_TO_ID[r["label"]] for r in test_rows], dtype=np.int64)

    # Pre-calibration
    pre_probs = softmax(test_logits, T=1.0)
    pre_ece, pre_bins = compute_ece(pre_probs, test_labels, N_BINS)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reliability_plot(pre_bins, pre_ece, "Pre-calibration (T=1.0)", FIG_DIR / "reliability_pre.png")
    print(f"pre-calibration ECE: {pre_ece:.4f}")

    # Fit T on fit set
    T_star, fit_nll = fit_temperature(fit_logits, fit_labels)
    print(f"fitted temperature T* = {T_star:.4f}  (NLL on fit = {fit_nll:.4f})")

    # Post-calibration
    post_probs = softmax(test_logits, T=T_star)
    post_ece, post_bins = compute_ece(post_probs, test_labels, N_BINS)
    reliability_plot(
        post_bins, post_ece, f"Post-calibration (T={T_star:.3f})", FIG_DIR / "reliability_post.png"
    )
    print(f"post-calibration ECE: {post_ece:.4f}")
    print(
        f"ECE reduction: {pre_ece - post_ece:+.4f}  ({(pre_ece - post_ece) / max(pre_ece, 1e-12) * 100:+.1f}%)"
    )

    # Save calibration constant + metadata
    out = {
        "model": "labrat-aiko/nli-popia-v2",
        "artifact": "onnx/model.onnx",
        "method": "single-parameter temperature scaling (Guo et al. 2017) fit by NLL minimization with scipy.optimize.minimize_scalar (bounded, brent)",
        "temperature": T_star,
        "fit_nll": fit_nll,
        "ece_pre": pre_ece,
        "ece_post": post_ece,
        "n_fit": len(fit_rows),
        "n_test": len(test_rows),
        "n_bins": N_BINS,
        "fit_frac": FIT_FRAC,
        "seed": SEED,
        "eval_paths": [str(p) for p in EVAL_PATHS],
        "test_bins": post_bins,
    }
    CAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    CAL_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {CAL_JSON}")
    print(f"figures: {FIG_DIR}/reliability_pre.png  +  {FIG_DIR}/reliability_post.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
