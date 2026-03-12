"""
app.py - Legal Advisor UI
Run: streamlit run app.py
"""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Legal Advisor AI", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .user-bubble {
        background: #1e2d40;
        border-left: 4px solid #4A90D9;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        color: #e0e0e0;
    }
    .bot-bubble {
        background: #1a2e1a;
        border-left: 4px solid #4CAF50;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        color: #e0e0e0;
    }
    .disclaimer-box {
        background: #2a2a14;
        border: 1px solid #777700;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.78rem;
        color: #cccc55;
        margin-top: 4px;
    }
    .meta-row { font-size: 0.72rem; color: #666; margin-top: 4px; }
    .welcome-box { text-align: center; padding: 80px 20px; color: #555; }
    .badge-qlora {
        background: #1a1a3a; border: 1px solid #7B68EE;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.72rem; color: #7B68EE; display: inline-block; margin-right: 6px;
    }
    .badge-lora {
        background: #1a3a3a; border: 1px solid #20B2AA;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.72rem; color: #20B2AA; display: inline-block; margin-right: 6px;
    }
    .badge-adapter-on {
        background: #1a3a1a; border: 1px solid #4CAF50;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.72rem; color: #4CAF50; display: inline-block;
    }
    .badge-adapter-off {
        background: #2a2a2a; border: 1px solid #888;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.72rem; color: #888; display: inline-block;
    }
    .mode-info {
        background: #1a1a2e; border: 1px solid #333;
        border-radius: 8px; padding: 10px 14px;
        font-size: 0.8rem; color: #aaa; margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""
if "quant_mode" not in st.session_state:
    st.session_state.quant_mode = "qlora"

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Legal Advisor AI")
    st.markdown("*TinyLlama + LoRA / QLoRA Adapters*")
    st.divider()

    # Quantization Mode Toggle
    st.markdown("### 🔧 Quantization Mode")
    quant_choice = st.radio(
        "Select mode",
        options=["QLoRA (4-bit, less VRAM)", "LoRA (fp16, faster compute)"],
        index=0 if st.session_state.quant_mode == "qlora" else 1,
        label_visibility="collapsed",
    )
    selected_mode = "qlora" if "QLoRA" in quant_choice else "lora"

    if selected_mode == "qlora":
        st.markdown("""
        <div class="mode-info">
            <b>QLoRA</b>: TinyLlama in <b>4-bit NF4</b> via BitsAndBytes.<br>
            ✅ Less VRAM (saves ~75%)<br>
            ⚠️ Slightly slower than fp16
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mode-info">
            <b>LoRA</b>: TinyLlama in <b>fp16</b> half-precision.<br>
            ✅ Faster compute<br>
            ⚠️ Needs more VRAM (~2× vs QLoRA)
        </div>""", unsafe_allow_html=True)

    if selected_mode != st.session_state.quant_mode:
        st.session_state.quant_mode = selected_mode
        st.warning(f"Restart server: `$env:QUANT_MODE=\"{selected_mode}\"; python server.py`")

    st.divider()

    # API Status
    st.markdown("### 📡 API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            server_mode = info.get("quant_mode", "unknown").upper()
            st.success(f"🟢 API Online — **{server_mode}** mode")
            st.caption(f"Model: `{info.get('base_model', 'N/A')}`")
            st.caption(f"Device: `{info.get('device', 'N/A')}`")
            adapters = info.get("adapters_loaded", {})
            if adapters:
                st.markdown("**LoRA Adapters:**")
                for jur, loaded in adapters.items():
                    icon = "🟢" if loaded else "🔴"
                    st.caption(f"{icon} `{jur}` — {'Loaded' if loaded else 'Fallback'}")
            if info.get("quant_mode") != st.session_state.quant_mode:
                st.warning(f"⚠️ Server is **{server_mode}** but UI is set to **{st.session_state.quant_mode.upper()}**")
        else:
            st.error("🔴 API Error")
    except Exception:
        st.error("🔴 API Offline")
        st.caption("Run: `python server.py`")

    st.divider()

    # Jurisdiction
    st.markdown("### 🌍 Jurisdiction")
    jurisdiction_map = {
        "🇮🇳 India":                "india_law",
        "🇺🇸 United States":         "us_law",
        "🇪🇺 European Union (GDPR)": "eu_gdpr",
    }
    selected = st.selectbox("Jurisdiction", list(jurisdiction_map.keys()), label_visibility="collapsed")
    jurisdiction = jurisdiction_map[selected]

    st.divider()

    # Model Settings
    st.markdown("### ⚙️ Model Settings")
    max_tokens  = st.slider("Max New Tokens", 32, 256, 128, 16,
                            help="Keep low (128) for faster responses")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                            help="Higher = more creative. Lower = more deterministic.")

    st.divider()

    # Sample Questions
    st.markdown("### 💡 Try These")
    sample_qs = {
        "india_law": ["What are my fundamental rights?", "How do I file an FIR?",
                      "What is anticipatory bail?", "What are consumer rights in India?"],
        "us_law":    ["What are Miranda rights?", "What is small claims court?",
                      "Difference between felony and misdemeanor?", "What is Chapter 7 bankruptcy?"],
        "eu_gdpr":   ["What is GDPR?", "What are my rights under GDPR?",
                      "What are GDPR fines?", "How to report a data breach?"],
    }
    for q in sample_qs.get(jurisdiction, []):
        if st.button(q, key=q, use_container_width=True):
            st.session_state.prefill = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── MAIN ──────────────────────────────────────────────────────
st.markdown("## ⚖️ Legal Advisor AI")
st.markdown(f"Jurisdiction: **{selected}**")
st.divider()

if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size:3.5rem">⚖️</div>
        <div style="font-size:1.3rem; margin-top:14px; color:#888">Ask a legal question to get started</div>
        <div style="font-size:0.9rem; margin-top:8px; color:#555">Select a jurisdiction and quantization mode from the sidebar</div>
    </div>""", unsafe_allow_html=True)
else:
    for turn in st.session_state.chat_history:
        st.markdown(f"""
        <div class="user-bubble">👤 <strong>You:</strong><br>{turn["user"]}</div>
        """, unsafe_allow_html=True)

        mode = turn.get("quant_mode", "lora")
        mode_badge    = f'<span class="badge-qlora">⚡ QLoRA 4-bit</span>' if mode == "qlora" else f'<span class="badge-lora">🔷 LoRA fp16</span>'
        adapter_badge = f'<span class="badge-adapter-on">🧩 LoRA Adapter — {turn["jurisdiction_name"]}</span>' if turn.get("adapter_used") else f'<span class="badge-adapter-off">⚙️ Base TinyLlama — {turn["jurisdiction_name"]}</span>'

        st.markdown(f"""
        <div class="bot-bubble">
            ⚖️ <strong>Legal Advisor ({turn["jurisdiction_name"]}):</strong><br>
            {mode_badge} {adapter_badge}<br><br>
            {turn["response"]}
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="disclaimer-box">⚠️ {turn["disclaimer"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta-row">⏱ {turn["time_seconds"]}s &nbsp;|&nbsp; 📝 {turn["tokens_generated"]} words</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────
user_input = st.text_area(
    "Question", value=st.session_state.prefill, height=110,
    placeholder="Type your legal question here...", label_visibility="collapsed",
)
st.session_state.prefill = ""

col1, col2 = st.columns([6, 1])
with col2:
    send = st.button("Send ➤", use_container_width=True, type="primary")

if send and user_input.strip():
    history = [{"user": t["user"], "assistant": t["response"]} for t in st.session_state.chat_history[-2:]]
    mode_label = "QLoRA 4-bit" if st.session_state.quant_mode == "qlora" else "LoRA fp16"
    with st.spinner(f"⏳ Running TinyLlama [{mode_label}] — may take 30–60 seconds..."):
        try:
            r = requests.post(
                f"{API_URL}/query",
                json={
                    "query":        user_input.strip(),
                    "jurisdiction": jurisdiction,
                    "history":      history,
                    "max_tokens":   max_tokens,
                    "temperature":  temperature,
                },
                timeout=300,    # ← increased to 300 seconds
            )
            if r.status_code == 200:
                result = r.json()
                st.session_state.chat_history.append({
                    "user":              user_input.strip(),
                    "response":          result["response"],
                    "jurisdiction_name": result["jurisdiction_name"],
                    "disclaimer":        result["disclaimer"],
                    "time_seconds":      result["time_seconds"],
                    "tokens_generated":  result["tokens_generated"],
                    "adapter_used":      result.get("adapter_used", False),
                    "quant_mode":        result.get("quant_mode", "lora"),
                })
                st.rerun()
            else:
                st.error(f"Error {r.status_code}: {r.text}")
        except requests.exceptions.ConnectionError:
            st.error("🔴 Cannot connect to API. Open a new terminal and run: `python server.py`")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The model is taking too long — try a shorter question or reduce Max New Tokens.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
elif send:
    st.warning("Please type a question first.")