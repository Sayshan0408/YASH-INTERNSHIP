## ⚖️ Legal Advisor AI

A multi-jurisdiction AI-powered legal Q&A chatbot built with TinyLlama 1.1B, fine-tuned using LoRA and QLoRA adapters across three legal systems — India, United States, and European Union (GDPR).


## Overview
Legal Advisor AI is a two-component web application:

FastAPI Backend (server.py) — loads TinyLlama with per-jurisdiction LoRA adapters and serves answers via REST API
Streamlit Frontend (app.py) — provides a chat interface with mode badges and jurisdiction selector

## Features

🌍 Three Jurisdictions — India Law, US Law, EU GDPR
⚡ QLoRA 4-bit mode — runs on low VRAM GPUs (~0.7 GB)
🔷 LoRA fp16 mode — faster inference on higher VRAM setups
🧩 Per-jurisdiction adapters — each fine-tuned on domain-specific legal data
💬 Chat history — maintains last 2 turns of conversation context
🏷️ Mode badges — every response shows active quantization and adapter status
🔄 Graceful fallback — uses base TinyLlama if adapter is missing

## Project Structure
Legal Advisor AI/
│
├── server.py               # FastAPI backend — model inference + REST API
├── app.py                  # Streamlit chat UI
├── train_adapters.py       # LoRA/QLoRA fine-tuning script
│
├── data/
│   ├── eu_gdpr_train.jsonl     # EU GDPR training dataset
│   ├── us_law_train.jsonl      # US Law training dataset
│   └── india_law_train.jsonl   # India Law training dataset
│
└── adapters/
    ├── eu_gdpr/                # Trained LoRA adapter for EU GDPR
    │   ├── adapter_config.json
    │   └── adapter_model.bin
    ├── us_law/                 # Trained LoRA adapter for US Law
    │   ├── adapter_config.json
    │   └── adapter_model.bin
    └── india_law/              # Trained LoRA adapter for India Law
        ├── adapter_config.json
        └── adapter_model.bin

## How It Works
1. User types legal question in Streamlit UI
2. app.py sends POST /query to FastAPI server
3. server.py selects correct LoRA adapter for the jurisdiction
4. Prompt is built in TinyLlama ChatML format
5. Tokenizer converts text to input_ids tensor
6. TinyLlama generates response tokens one by one
7. Only new tokens (after prompt) are decoded to text
8. Response + metadata returned as JSON to UI
9. UI renders chat bubble with ⚡ QLoRA + 🧩 Adapter badges
