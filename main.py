"""
main.py
AutoDev — AI-powered multi-agent software development pipeline.
Powered by Groq (llama-3.1-8b-instant).
"""

import streamlit as st

from agents.config import STAGE_ORDER, AGENT_META
from core.pipeline import run_full_pipeline
from core.history import (
    list_runs,
    search_runs,
    get_run,
    delete_run,
    clear_all_runs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoDev — AI Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
h1,h2,h3,p,span,label { color: #e6edf3 !important; }

.agent-chip {
    display: flex; flex-direction: column; align-items: center;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 14px 8px; transition: all 0.2s;
}
.agent-chip:hover { border-color: #58a6ff; }
.agent-chip .icon  { font-size: 26px; }
.agent-chip .name  {
    font-size: 11px; font-weight: 700; color: #8b949e;
    margin-top: 6px; text-transform: uppercase; letter-spacing: 0.05em;
}

.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-running { background:#0c2d6b; color:#58a6ff; }
.badge-blocked { background:#4a1a1a; color:#f85149; }

[data-testid="stExpander"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; margin-bottom: 10px;
}

.stButton > button {
    background:#21262d; color:#e6edf3;
    border:1px solid #30363d; border-radius:8px; font-weight:600;
}
.stButton > button:hover { background:#30363d; border-color:#58a6ff; color:#58a6ff; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#238636 0%,#2ea043 100%);
    color:#fff; border:none; font-size:16px; border-radius:10px;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#2ea043 0%,#3fb950 100%); color:#fff;
}

textarea {
    background:#0d1117 !important; color:#e6edf3 !important;
    border:1px solid #30363d !important; border-radius:8px !important;
}
hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "loaded_run_id"    not in st.session_state: st.session_state["loaded_run_id"]    = None
if "pipeline_outputs" not in st.session_state: st.session_state["pipeline_outputs"] = {}
if "guardrail_msgs"   not in st.session_state: st.session_state["guardrail_msgs"]   = []


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AutoDev")
    st.caption("Multi-agent AI pipeline · Powered by Groq")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at https://console.groq.com",
    )
    if api_key:
        if api_key.startswith("gsk_"):
            st.success("✅ API key looks valid")
        else:
            st.warning("⚠️ Groq keys usually start with gsk_")

    st.markdown("---")
    st.markdown("### 📂 Run History")
    search_q = st.text_input("🔍 Search", placeholder="filter by keyword...")
    runs = search_runs(search_q) if search_q else list_runs()

    if runs:
        for run in runs[:20]:
            ts  = run.get("timestamp", "")[:16].replace("T", " ")
            req = run.get("requirement", "")
            preview = (req[:45] + "…") if len(req) > 45 else req
            icon = "✅" if run.get("status") == "success" else "⚠️"

            col_a, col_b = st.columns([5, 1])
            with col_a:
                if st.button(f"{icon} {ts}\n{preview}", key=f"load_{run['id']}", use_container_width=True):
                    st.session_state["loaded_run_id"]    = run["id"]
                    st.session_state["pipeline_outputs"] = {}
                    st.rerun()
            with col_b:
                if st.button("🗑", key=f"del_{run['id']}"):
                    delete_run(run["id"])
                    if st.session_state["loaded_run_id"] == run["id"]:
                        st.session_state["loaded_run_id"] = None
                    st.rerun()

        st.markdown("")
        if st.button("🧹 Clear All History", use_container_width=True):
            clear_all_runs()
            st.session_state["loaded_run_id"] = None
            st.rerun()
    else:
        st.caption("No runs yet.")

    st.markdown("---")
    st.markdown("### 🤖 Agents")
    for stage in STAGE_ORDER:
        m = AGENT_META[stage]
        st.markdown(f"**{m['icon']} {m['label']}** — {m['description']}")
        st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🚀 AutoDev")
st.markdown(
    "Enter a software requirement. Six AI agents will **plan, architect, "
    "code, review, test, and document** it automatically using **Groq**."
)
st.markdown("---")

# Agent chips
cols = st.columns(len(STAGE_ORDER))
for idx, stage in enumerate(STAGE_ORDER):
    m = AGENT_META[stage]
    with cols[idx]:
        st.markdown(
            f'<div class="agent-chip">'
            f'<span class="icon">{m["icon"]}</span>'
            f'<span class="name">{m["label"]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# Requirement input
# ─────────────────────────────────────────────────────────────────────────────

prefill = ""
if st.session_state["loaded_run_id"]:
    loaded = get_run(st.session_state["loaded_run_id"])
    if loaded:
        prefill = loaded.get("requirement", "")

requirement = st.text_area(
    "📝 Software Requirement",
    value=prefill,
    height=130,
    placeholder=(
        "Describe your software requirement in detail.\n"
        "Example: Build a REST API in FastAPI with SQLite for a todo app — "
        "CRUD endpoints, JWT auth, and pagination support."
    ),
)

col_run, col_clear = st.columns([2, 1])
with col_run:
    run_btn = st.button("🚀 Run Pipeline", type="primary", disabled=not api_key, use_container_width=True)
with col_clear:
    if st.button("✖ Clear", use_container_width=True):
        st.session_state["loaded_run_id"]    = None
        st.session_state["pipeline_outputs"] = {}
        st.session_state["guardrail_msgs"]   = []
        st.rerun()

if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to enable the pipeline. Free at console.groq.com")

# ─────────────────────────────────────────────────────────────────────────────
# Show loaded history run
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["loaded_run_id"] and not run_btn:
    loaded = get_run(st.session_state["loaded_run_id"])
    if loaded:
        st.markdown("---")
        ts = loaded.get("timestamp", "")[:16].replace("T", " ")
        st.success(f"📂 Viewing run from **{ts}** — *{loaded.get('requirement','')[:60]}*")
        for stage in STAGE_ORDER:
            if stage in loaded.get("outputs", {}):
                m = AGENT_META[stage]
                with st.expander(f"{m['icon']} {m['label']}", expanded=(stage == "developer")):
                    st.markdown(loaded["outputs"][stage])


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and requirement.strip() and api_key:
    st.session_state["loaded_run_id"]    = None
    st.session_state["pipeline_outputs"] = {}
    st.session_state["guardrail_msgs"]   = []

    st.markdown("---")
    progress_bar  = st.progress(0, text="Starting pipeline…")
    status_box    = st.empty()
    stage_slots   = {stage: st.empty() for stage in STAGE_ORDER}
    total         = len(STAGE_ORDER)
    completed     = [0]

    def on_start(stage: str) -> None:
        m = AGENT_META[stage]
        stage_slots[stage].markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:10px">'
            f'<span style="font-size:20px">{m["icon"]}</span> '
            f'<b style="color:#e6edf3">{m["label"]}</b> '
            f'<span class="badge badge-running">⏳ Running…</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        status_box.info(f"Running **{m['label']}**…")

    def on_done(stage: str, output: str) -> None:
        st.session_state["pipeline_outputs"][stage] = output
        stage_slots[stage].empty()
        completed[0] += 1
        pct = int((completed[0] / total) * 100)
        progress_bar.progress(pct, text=f"Completed {completed[0]}/{total} agents…")

    def on_blocked(stage: str, reason: str) -> None:
        st.session_state["guardrail_msgs"].append((stage, reason))
        m = AGENT_META.get(stage, {"icon": "🚫", "label": stage.capitalize()})
        slot = stage_slots.get(stage, st.empty())
        slot.markdown(
            f'<div style="background:#1a0000;border:1px solid #f85149;border-radius:10px;padding:16px;margin-bottom:10px">'
            f'<span style="font-size:20px">{m["icon"]}</span> '
            f'<b style="color:#f85149">{m["label"]}</b> '
            f'<span class="badge badge-blocked">🚫 Blocked</span><br>'
            f'<span style="color:#f85149;font-size:13px">{reason}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    outputs = run_full_pipeline(
        requirement=requirement,
        api_key=api_key,
        on_stage_start=on_start,
        on_stage_done=on_done,
        on_guardrail_block=on_blocked,
    )

    progress_bar.empty()
    status_box.empty()
    for slot in stage_slots.values():
        slot.empty()

    if not outputs:
        st.error("❌ Pipeline was blocked before it could start. Check the message above.")
    else:
        st.success("✅ Pipeline completed! All 6 agents finished.")
        st.balloons()
        for stage in STAGE_ORDER:
            if stage in outputs:
                m = AGENT_META[stage]
                with st.expander(f"{m['icon']} {m['label']}", expanded=(stage == "developer")):
                    st.markdown(outputs[stage])
