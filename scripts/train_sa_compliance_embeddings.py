"""Fine-tune sa-compliance-embeddings-v1.

Base: BAAI/bge-small-en-v1.5 (33M params, 384-dim).
Training: MultipleNegativesRankingLoss with in-batch negatives — implemented
inline against raw transformers so the recipe doesn't depend on the
sentence-transformers trainer (which is version-locked to bleeding-edge
transformers).
Data: data/sa_compliance_pairs.jsonl (308 anchor/positive pairs from POPIA Act + scenarios).
Eval: top-k recall on a held-out scenario->section retrieval task against the same base model.

Output: out/sa-compliance-embeddings-v1/  (transformers checkpoint + eval report)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

PAIRS = Path("data/sa_compliance_pairs.jsonl")
CORPUS = Path("data/sa_compliance_corpus.jsonl")
OUT_DIR = Path("out/sa-compliance-embeddings-v1")
BASE_MODEL = "BAAI/bge-small-en-v1.5"

if TYPE_CHECKING:
    import numpy as np

# Held-out clause->section gold (used for retrieval eval, not training).
CLAUSE_TO_SECTIONS: dict[str, list[int]] = {
    "POPIA consent": [11, 12, 18, 69],
    "POPIA minimality / purpose limitation": [10, 13, 14, 15],
    "POPIA security safeguards": [19, 20, 21, 22],
    "POPIA general processing": [8, 9, 4],
    "POPIA breach notification": [22],
    "POPIA cross-border transfers": [72],
    "POPIA data subject rights": [5, 23, 24, 25],
    "POPIA children's information": [34, 35],
    "POPIA special personal information": [26, 27, 28, 29, 30, 31, 32, 33],
    "POPIA automated decision-making": [71],
}


def load_pairs() -> list[dict]:
    return [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]


def load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]


def load_eval_scenarios() -> list[dict]:
    """Use popia_eval_v2.jsonl + popia_eval.jsonl as the eval queries.
    These are NOT in the training pairs (which came from seeds + paraphrases)."""
    rows = []
    for f in ["data/popia_eval.jsonl", "data/popia_eval_v2.jsonl"]:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    # Keep only entailment + contradiction (neutral scenarios don't reliably target a clause)
    return [r for r in rows if r["label"] in {"entailment", "contradiction"}]


class STModel:
    """Thin wrapper around HuggingFace transformers giving a sentence-transformer-like API."""

    def __init__(self, model_name_or_path: str, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):

        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import numpy as np
        import torch
        import torch.nn.functional as F

        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                pooled = self._mean_pool(hidden, enc["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(path))
        self.model.save_pretrained(str(path))


def retrieval_eval(model: STModel, eval_scenarios: list[dict], corpus: list[dict]) -> dict:
    """For each eval scenario, embed (premise) and find its best matches in corpus.
    Gold = corpus sections matching the scenario's clause.
    Report recall@1, recall@3, recall@5, recall@10."""
    import numpy as np

    sec_index = {}
    for i, doc in enumerate(corpus):
        num = int(doc["section_id"].split("§")[1])
        sec_index[num] = i

    queries = [s["premise"] for s in eval_scenarios]
    docs = [d["text"] for d in corpus]

    q_emb = model.encode(queries)
    d_emb = model.encode(docs)

    sims = q_emb @ d_emb.T
    topk = np.argsort(-sims, axis=1)[:, :10]

    correct_at = defaultdict(int)
    total = 0
    for q_idx, scenario in enumerate(eval_scenarios):
        clause = scenario["clause"]
        gold_sections = set(CLAUSE_TO_SECTIONS.get(clause, []))
        gold_doc_indices = {sec_index[s] for s in gold_sections if s in sec_index}
        if not gold_doc_indices:
            continue
        total += 1
        for k in [1, 3, 5, 10]:
            if set(topk[q_idx, :k].tolist()) & gold_doc_indices:
                correct_at[k] += 1

    return {
        "total_queries": total,
        "recall_at_1": correct_at[1] / total if total else 0.0,
        "recall_at_3": correct_at[3] / total if total else 0.0,
        "recall_at_5": correct_at[5] / total if total else 0.0,
        "recall_at_10": correct_at[10] / total if total else 0.0,
    }


def fit_mnr(
    model: STModel, pairs: list[dict], epochs: int, batch_size: int, lr: float, scale: float = 20.0
) -> None:
    """MultipleNegativesRankingLoss training loop.

    For each batch of (anchor, positive) pairs, compute cosine similarities
    between every anchor and every positive in the batch. Diagonal entries
    are the true positives; off-diagonal entries are negatives sampled
    in-batch. Loss = cross-entropy over rows.
    """
    import torch
    import torch.nn.functional as F

    anchors = [p["anchor"] for p in pairs]
    positives = [p["positive"] for p in pairs]
    idx = list(range(len(pairs)))
    rng = random.Random(42)

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)
    total_steps = (len(pairs) // batch_size) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / warmup_steps,
            max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
        ),
    )

    model.model.train()
    step = 0
    for epoch in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            if len(batch_idx) < 2:  # need ≥2 for in-batch negatives
                continue
            a_texts = [anchors[j] for j in batch_idx]
            p_texts = [positives[j] for j in batch_idx]
            a_enc = model.tokenizer(
                a_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(model.device)
            p_enc = model.tokenizer(
                p_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(model.device)
            a_h = model._mean_pool(model.model(**a_enc).last_hidden_state, a_enc["attention_mask"])
            p_h = model._mean_pool(model.model(**p_enc).last_hidden_state, p_enc["attention_mask"])
            a_h = F.normalize(a_h, p=2, dim=1)
            p_h = F.normalize(p_h, p=2, dim=1)
            sims = (a_h @ p_h.T) * scale
            target = torch.arange(sims.size(0), device=sims.device)
            loss = F.cross_entropy(sims, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 10 == 0:
                print(
                    f"  epoch {epoch + 1}/{epochs} step {step}/{total_steps}  loss={loss.item():.4f}"
                )
        print(f"epoch {epoch + 1} done")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    pairs = load_pairs()
    corpus = load_corpus()
    eval_scenarios = load_eval_scenarios()

    print(f"training pairs: {len(pairs)}")
    print(f"corpus sections: {len(corpus)}")
    print(f"eval scenarios (entailment+contradiction): {len(eval_scenarios)}")

    # ---- Baseline retrieval eval ----
    print("\n=== Baseline (stock bge-small-en-v1.5) ===")
    base = STModel(BASE_MODEL)
    base_report = retrieval_eval(base, eval_scenarios, corpus)
    print(json.dumps(base_report, indent=2))

    # ---- Fine-tune ----
    model = STModel(BASE_MODEL)
    fit_mnr(model, pairs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR)
    print(f"trained model saved to {OUT_DIR}")

    # ---- Trained retrieval eval ----
    print("\n=== Trained (sa-compliance-embeddings-v1) ===")
    trained_report = retrieval_eval(model, eval_scenarios, corpus)
    print(json.dumps(trained_report, indent=2))

    # ---- Release gate ----
    gate_pass = (
        trained_report["recall_at_1"] >= base_report["recall_at_1"]
        and trained_report["recall_at_5"] >= base_report["recall_at_5"] + 0.05
    )
    print(
        f"\nrelease gate (recall@5 beats stock by ≥5pp AND recall@1 no regression): "
        f"{'PASS' if gate_pass else 'FAIL'}"
    )

    report = {
        "base_model": BASE_MODEL,
        "training_pairs": len(pairs),
        "epochs": args.epochs,
        "baseline": base_report,
        "trained": trained_report,
        "gate_pass": gate_pass,
        "deltas": {
            "recall_at_1": trained_report["recall_at_1"] - base_report["recall_at_1"],
            "recall_at_3": trained_report["recall_at_3"] - base_report["recall_at_3"],
            "recall_at_5": trained_report["recall_at_5"] - base_report["recall_at_5"],
            "recall_at_10": trained_report["recall_at_10"] - base_report["recall_at_10"],
        },
    }
    (OUT_DIR / "release_gate.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {OUT_DIR / 'release_gate.json'}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
