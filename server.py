import os, json, logging, time
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG TOGGLE
#  QUANT_MODE=qlora python server.py   ← 4-bit QLoRA (default)
#  QUANT_MODE=lora  python server.py   ← fp16 LoRA
# ══════════════════════════════════════════════════════════════════════════════
QUANTIZATION_MODE = os.environ.get("QUANT_MODE", "qlora")

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── SPEED SETTINGS (tuned for fast response) ──────────────────────────────────
MAX_NEW_TOKENS_CAP = 128        # hard cap — keeps responses fast
MAX_INPUT_LENGTH   = 512        # shorter context = faster inference
TEMPERATURE_DEFAULT = 0.3

JURISDICTION_META = {
    "eu_gdpr": {
        "name": "European Union (GDPR)",
        "flag": "EU",
        "adapter_path": "adapters/eu_gdpr",
        "countries": [
            "AT","BE","DE","FR","IT","ES","NL","PL","SE","DK",
            "FI","IE","PT","GR","HU","CZ","RO","SK","SI","HR",
            "LT","LV","EE","LU","MT","CY","BG"
        ],
    },
    "us_law": {
        "name": "United States",
        "flag": "US",
        "adapter_path": "adapters/us_law",
        "countries": ["US"],
    },
    "india_law": {
        "name": "India",
        "flag": "IN",
        "adapter_path": "adapters/india_law",
        "countries": ["IN"],
    },
}

DISCLAIMER = "For informational purposes only. Always consult a qualified lawyer."

tokenizer   = None
base_model  = None
lora_models = {}


# ── QUANTIZATION CONFIG ───────────────────────────────────────────────────────
def get_model_load_kwargs() -> dict:
    if QUANTIZATION_MODE == "qlora":
        logger.info("Quantization mode: QLoRA (4-bit NF4 + double quantization)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        return {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
    else:
        logger.info("Quantization mode: LoRA only (fp16, no quantization)")
        return {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_base_model():
    global tokenizer, base_model
    logger.info(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = get_model_load_kwargs()
    logger.info(f"Loading TinyLlama [{QUANTIZATION_MODE.upper()}] ...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)
    base_model.eval()
    logger.info("TinyLlama loaded successfully.")


def load_lora_adapter(jurisdiction: str) -> bool:
    global lora_models
    if jurisdiction in lora_models:
        return lora_models[jurisdiction] is not None

    adapter_path = JURISDICTION_META[jurisdiction]["adapter_path"]
    if not os.path.exists(adapter_path):
        logger.warning(f"[{jurisdiction}] Adapter not found at '{adapter_path}'. Using base model.")
        lora_models[jurisdiction] = None
        return False

    logger.info(f"[{jurisdiction}] Loading LoRA adapter from '{adapter_path}' ...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    lora_model.eval()
    lora_models[jurisdiction] = lora_model
    logger.info(f"[{jurisdiction}] Adapter ready ({QUANTIZATION_MODE.upper()}).")
    return True


def get_model(jurisdiction: str):
    load_lora_adapter(jurisdiction)
    model = lora_models.get(jurisdiction)
    return model if model is not None else base_model


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────
def build_prompt(query: str, jurisdiction_name: str, history: Optional[List[dict]] = None) -> str:
    system_msg = (
        f"You are a legal advisor for {jurisdiction_name}. "
        "Give a short, clear, accurate legal answer in 3-5 sentences. "
        "Always advise consulting a qualified lawyer."
    )
    prompt = f"<|system|>\n{system_msg}</s>\n"

    if history:
        for turn in history[-2:]:   # only last 2 turns to keep prompt short
            prompt += f"<|user|>\n{turn.get('user', '')}</s>\n"
            prompt += f"<|assistant|>\n{turn.get('assistant', '')}</s>\n"

    prompt += f"<|user|>\n{query}</s>\n<|assistant|>\n"
    return prompt


# ── INFERENCE (optimised for speed) ──────────────────────────────────────────
def run_inference(
    query: str,
    jurisdiction: str,
    history: Optional[List[dict]] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
) -> str:
    model = get_model(jurisdiction)
    jurisdiction_name = JURISDICTION_META[jurisdiction]["name"]
    prompt = build_prompt(query, jurisdiction_name, history)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,    # shorter = faster
    ).to(model.device)

    # Cap tokens to prevent slow responses
    safe_max_tokens = min(max_new_tokens, MAX_NEW_TOKENS_CAP)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=safe_max_tokens,
            temperature=max(temperature, 1e-6),
            do_sample=temperature > 0.0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,             # speeds up generation
        )

    input_length  = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_length:]
    response      = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response or "Unable to generate a response. Please consult a legal professional."


# ── LIFESPAN ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    logger.info(f"Starting Legal Advisor AI [{QUANTIZATION_MODE.upper()}] ...")
    load_base_model()
    for key in JURISDICTION_META:
        load_lora_adapter(key)
    logger.info("Legal Advisor AI is ready!")
    yield


# ── FASTAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Legal Advisor AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LegalQueryRequest(BaseModel):
    query: str
    jurisdiction: str
    history: Optional[List[dict]] = None
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.3


class LegalQueryResponse(BaseModel):
    query: str
    jurisdiction: str
    jurisdiction_name: str
    flag: str
    response: str
    tokens_generated: int
    time_seconds: float
    disclaimer: str
    adapter_used: bool
    quant_mode: str


@app.get("/")
async def root():
    return {"service": "Legal Advisor AI", "status": "running", "model": BASE_MODEL_ID, "mode": QUANTIZATION_MODE}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "base_model": BASE_MODEL_ID,
        "quant_mode": QUANTIZATION_MODE,
        "adapters_loaded": {k: (v is not None) for k, v in lora_models.items()},
        "device": str(next(base_model.parameters()).device) if base_model else "not loaded",
    }


@app.get("/jurisdictions")
async def jurisdictions():
    return [
        {
            "key": k, "name": m["name"], "flag": m["flag"],
            "adapter_available": os.path.exists(m["adapter_path"]),
            "adapter_loaded": lora_models.get(k) is not None,
            "countries": m["countries"],
        }
        for k, m in JURISDICTION_META.items()
    ]


@app.post("/query", response_model=LegalQueryResponse)
async def legal_query(req: LegalQueryRequest):
    if req.jurisdiction not in JURISDICTION_META:
        raise HTTPException(status_code=400, detail=f"Unknown jurisdiction: {req.jurisdiction}")

    t0 = time.time()
    response = run_inference(
        query=req.query,
        jurisdiction=req.jurisdiction,
        history=req.history,
        max_new_tokens=req.max_tokens or 128,
        temperature=req.temperature or 0.3,
    )

    meta         = JURISDICTION_META[req.jurisdiction]
    adapter_used = lora_models.get(req.jurisdiction) is not None

    return LegalQueryResponse(
        query=req.query,
        jurisdiction=req.jurisdiction,
        jurisdiction_name=meta["name"],
        flag=meta["flag"],
        response=response,
        tokens_generated=len(response.split()),
        time_seconds=round(time.time() - t0, 3),
        disclaimer=DISCLAIMER,
        adapter_used=adapter_used,
        quant_mode=QUANTIZATION_MODE,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)