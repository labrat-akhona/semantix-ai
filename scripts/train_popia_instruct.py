"""QLoRA fine-tune popia-instruct-v0 on Phi-3-mini-4k-instruct.

Base: microsoft/Phi-3-mini-4k-instruct (3.8B, MIT licence).
Training: QLoRA (4-bit nf4) on top of HF Trainer.
Data: data/popia_instruct.jsonl (805 instruction-response examples).

Compute envelope: GTX 1650 (4 GB VRAM). 4-bit Phi-3-mini ≈ 1.9 GB
weights — leaves headroom for LoRA adapters + activations.

Output: out/popia-instruct-v0/  (LoRA adapter weights + tokenizer + merged model)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Quiet some warnings before heavy imports.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("BNB_CUDA_VERSION", "121")

DATA = Path("data/popia_instruct.jsonl")
OUT_DIR = Path("out/popia-instruct-v0")
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def load_examples() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    examples = load_examples()
    print(f"loaded {len(examples)} instruction examples")

    # Hold out 5% for sanity eval (loss + a couple of qualitative generations).
    split = int(len(examples) * 0.95)
    train_examples, eval_examples = examples[:split], examples[split:]
    print(f"train: {len(train_examples)}  eval: {len(eval_examples)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading 4-bit base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def render(ex: dict) -> str:
        # Apply Phi-3 chat template to the messages.
        return tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )

    def tokenize_row(ex):
        text = render(ex)
        enc = tokenizer(text, truncation=True, max_length=args.max_len, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_ds = Dataset.from_list(train_examples).map(
        tokenize_row, remove_columns=["messages"], desc="tokenize train"
    )
    eval_ds = Dataset.from_list(eval_examples).map(
        tokenize_row, remove_columns=["messages"], desc="tokenize eval"
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    if not args.eval_only:
        trainer.train()
        # Save just the LoRA adapter (tiny, ~30 MB)
        model.save_pretrained(str(OUT_DIR / "adapter"))
        tokenizer.save_pretrained(str(OUT_DIR / "adapter"))
        print(f"adapter saved -> {OUT_DIR / 'adapter'}")

    # Sanity: generate against a couple of held-out prompts and log them.
    print("\n=== Qualitative samples ===")
    model.eval()
    samples_to_log: list[dict] = []
    for i, ex in enumerate(eval_examples[:3]):
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        gold = ex["messages"][-1]["content"]
        print(f"\n--- sample {i + 1} ---")
        print(f"USER: {ex['messages'][1]['content'][:150]}")
        print(f"GOLD: {gold[:200]}")
        print(f"PRED: {decoded[:200]}")
        samples_to_log.append({"user": ex["messages"][1]["content"], "gold": gold, "pred": decoded})

    (OUT_DIR / "qualitative_samples.json").write_text(json.dumps(samples_to_log, indent=2))
    print(f"\nqualitative samples written -> {OUT_DIR / 'qualitative_samples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
