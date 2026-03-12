"""
train_adapters.py
Fine-tunes TinyLlama with LoRA / QLoRA on each legal JSONL dataset
and saves adapters to adapters/<jurisdiction>/

Run:
    python train_adapters.py --mode qlora   # 4-bit QLoRA (default, less VRAM)
    python train_adapters.py --mode lora    # fp16 LoRA (faster compute)
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ── CONFIG ────────────────────────────────────────────────────
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

JURISDICTIONS = {
    "eu_gdpr":   "data/eu_gdpr_train.jsonl",
    "us_law":    "data/us_law_train.jsonl",
    "india_law": "data/india_law_train.jsonl",
}

# LoRA hyperparameters
LORA_CONFIG = LoraConfig(
    r=16,                        # rank — higher = more capacity, more VRAM
    lora_alpha=32,               # scaling factor
    target_modules=[             # TinyLlama attention layers to apply LoRA to
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Training hyperparameters
TRAIN_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",            # set to "wandb" if you want experiment tracking
)


# ── HELPERS ───────────────────────────────────────────────────
def load_jsonl(path: str):
    """Load a JSONL file and return list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def format_sample(sample: dict) -> str:
    """
    Convert a JSONL sample to TinyLlama ChatML format.
    Input fields: instruction, input (optional), output
    """
    instruction = sample.get("instruction", "").strip()
    context     = sample.get("input", "").strip()
    answer      = sample.get("output", "").strip()

    user_msg = f"{instruction}\n{context}".strip() if context else instruction

    return (
        f"<|system|>\nYou are a knowledgeable legal advisor. "
        f"Provide clear and accurate legal information.</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n{answer}</s>"
    )


def get_bnb_config():
    """4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ── MAIN TRAINING LOOP ────────────────────────────────────────
def train(mode: str):
    print(f"\n{'='*60}")
    print(f"  Training mode : {mode.upper()}")
    print(f"  Base model    : {BASE_MODEL_ID}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    print(f"Loading TinyLlama in [{mode.upper()}] mode ...")
    if mode == "qlora":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
        )
    else:  # lora — fp16
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False   # required for gradient checkpointing

    # Apply LoRA
    print("Applying LoRA config to TinyLlama ...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Train one adapter per jurisdiction
    for jurisdiction, data_path in JURISDICTIONS.items():
        print(f"\n{'─'*60}")
        print(f"  Training adapter: {jurisdiction}")
        print(f"  Dataset         : {data_path}")
        print(f"{'─'*60}")

        if not os.path.exists(data_path):
            print(f"  ⚠ Dataset not found — skipping {jurisdiction}")
            continue

        # Load & format dataset
        raw      = load_jsonl(data_path)
        texts    = [format_sample(s) for s in raw]
        dataset  = Dataset.from_dict({"text": texts})
        print(f"  Loaded {len(texts)} samples")

        # Output path for this adapter
        output_dir = f"adapters/{jurisdiction}"

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            **TRAIN_ARGS,
        )

        # SFTTrainer handles tokenization + packing
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            args=training_args,
        )

        print(f"  Starting training ...")
        trainer.train()

        # Save LoRA adapter weights
        print(f"  Saving adapter to '{output_dir}/' ...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  ✅ Adapter saved: {jurisdiction}")

    print(f"\n{'='*60}")
    print("  All adapters trained and saved successfully!")
    print("  Now start the server:")
    print("    QUANT_MODE=qlora python server.py   (Windows: set QUANT_MODE=qlora)")
    print(f"{'='*60}\n")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["qlora", "lora"],
        default="qlora",
        help="qlora = 4-bit quantized training (less VRAM), lora = fp16 training",
    )
    args = parser.parse_args()
    train(args.mode)