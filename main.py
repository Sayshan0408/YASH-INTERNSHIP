"""
main.py
AutoDev — AI-powered multi-agent software development pipeline.
Powered by Groq (llama-3.1-8b-instant).
"""

import streamlit as st

from agents.config import STAGE_ORDER, AGENT_META
from core.pipeline import run_full_pipeline
from core.accuracy import overall_pipeline_score
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
if "accuracy_scores"  not in st.session_state: st.session_state["accuracy_scores"]  = {}


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
        st.session_state["accuracy_scores"]  = {}
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
        pct = min(100, int((completed[0] / total) * 100))
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

    outputs, accuracy_scores = run_full_pipeline(
        requirement=requirement,
        api_key=api_key,
        on_stage_start=on_start,
        on_stage_done=on_done,
        on_guardrail_block=on_blocked,
    )

    st.session_state["accuracy_scores"] = accuracy_scores

    progress_bar.empty()
    status_box.empty()
    for slot in stage_slots.values():
        slot.empty()

    if not outputs:
        st.error("❌ Pipeline was blocked before it could start. Check the message above.")
    else:
        st.success("✅ Pipeline completed! All 6 agents finished.")
        st.balloons()

        # ── Accuracy Dashboard ─────────────────────────────────────────────
        if accuracy_scores:
            pipeline_score = overall_pipeline_score(accuracy_scores)
            grade_color = {"A": "#2ea043", "B": "#58a6ff", "C": "#f0b429", "D": "#f97316", "F": "#f85149"}

            st.markdown("---")
            st.markdown("## 📊 Pipeline Accuracy Report")

            # Overall score banner
            overall_scores = [s.score for s in accuracy_scores.values()]
            avg = pipeline_score
            grade = "A" if avg >= 90 else "B" if avg >= 75 else "C" if avg >= 60 else "D" if avg >= 40 else "F"
            color = grade_color.get(grade, "#888")

            st.markdown(
                f'<div style="background:#161b22;border:2px solid {color};border-radius:14px;'
                f'padding:20px 28px;margin-bottom:18px;display:flex;align-items:center;gap:24px">'
                f'<div style="font-size:48px;font-weight:800;color:{color}">{avg:.0f}</div>'
                f'<div>'
                f'<div style="font-size:22px;font-weight:700;color:#e6edf3">Overall Pipeline Score</div>'
                f'<div style="font-size:14px;color:#8b949e">'
                f'Grade <b style="color:{color}">{grade}</b> · '
                f'{len(accuracy_scores)}/6 agents scored · '
                f'{"Excellent quality 🎉" if avg >= 90 else "Good quality ✅" if avg >= 75 else "Needs improvement ⚠️" if avg >= 60 else "Poor quality ❌"}'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

            # Per-agent score cards
            agent_cols = st.columns(len(accuracy_scores))
            for idx, (stage, score_obj) in enumerate(accuracy_scores.items()):
                m = AGENT_META.get(stage, {"icon": "🤖", "label": stage.capitalize()})
                c = grade_color.get(score_obj.grade, "#888")
                with agent_cols[idx]:
                    st.markdown(
                        f'<div style="background:#161b22;border:1px solid {c};border-radius:10px;'
                        f'padding:12px 10px;text-align:center">'
                        f'<div style="font-size:22px">{m["icon"]}</div>'
                        f'<div style="font-size:11px;color:#8b949e;font-weight:700;text-transform:uppercase;margin:4px 0">'
                        f'{m["label"]}</div>'
                        f'<div style="font-size:28px;font-weight:800;color:{c}">{score_obj.score:.0f}</div>'
                        f'<div style="font-size:13px;color:{c};font-weight:700">Grade {score_obj.grade}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("")

            # Detailed breakdown per agent
            for stage, score_obj in accuracy_scores.items():
                m = AGENT_META.get(stage, {"icon": "🤖", "label": stage.capitalize()})
                c = grade_color.get(score_obj.grade, "#888")
                with st.expander(
                    f"{m['icon']} {m['label']} — {score_obj.score:.0f}/100 (Grade {score_obj.grade})  ·  {score_obj.summary}",
                    expanded=False
                ):
                    for check_name, passed, detail in score_obj.checks:
                        icon = "✅" if passed else "❌"
                        st.markdown(
                            f'<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #21262d">'
                            f'<span style="font-size:16px">{icon}</span>'
                            f'<div>'
                            f'<span style="color:#e6edf3;font-size:13px;font-weight:600">{check_name}</span><br>'
                            f'<span style="color:#8b949e;font-size:12px">{detail}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

        st.markdown("---")
        st.markdown("## 📄 Agent Outputs")
        for stage in STAGE_ORDER:
            if stage in outputs:
                m = AGENT_META[stage]
                with st.expander(f"{m['icon']} {m['label']}", expanded=(stage == "developer")):
                    st.markdown(outputs[stage])
