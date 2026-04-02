"""
app.py
Aviation Incident Report Checker
Verifies AI-generated regulatory summaries against original pilot incident reports
"""

import streamlit as st
import json

st.set_page_config(
    page_title="Aviation Incident Report Checker",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0f1c; }
[data-testid="stSidebar"] { background: #0f1525; border-right: 1px solid #1e3060; }
h1,h2,h3,p,label { color: #dde6ff !important; }

.metric-card {
    background: #0f1525;
    border: 1px solid #1e3060;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-score { font-size: 36px; font-weight: 800; }
.metric-name  { font-size: 12px; color: #7a99cc; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-desc  { font-size: 11px; color: #556688; margin-top: 4px; }

.status-cleared  { background:#0a2a1a; border:2px solid #2ea043; border-radius:14px; padding:24px; text-align:center; }
.status-review   { background:#2a2000; border:2px solid #f0b429; border-radius:14px; padding:24px; text-align:center; }
.status-grounded { background:#2a0a0a; border:2px solid #f85149; border-radius:14px; padding:24px; text-align:center; }

.flag-critical { background:#1a0808; border-left:4px solid #f85149; border-radius:6px; padding:12px; margin:6px 0; font-size:13px; color:#dde6ff; }
.flag-warning  { background:#1a1400; border-left:4px solid #f0b429; border-radius:6px; padding:12px; margin:6px 0; font-size:13px; color:#dde6ff; }
.flag-ok       { background:#081a10; border-left:4px solid #2ea043; border-radius:6px; padding:12px; margin:6px 0; font-size:13px; color:#dde6ff; }

[data-testid="stExpander"] { background:#0f1525; border:1px solid #1e3060; border-radius:10px; }
textarea { background:#0f1525 !important; color:#dde6ff !important; border:1px solid #1e3060 !important; }
.stButton > button { background:#1e3060; color:#dde6ff; border:1px solid #2d4a8a; border-radius:8px; font-weight:600; }
.stButton > button:hover { background:#2d4a8a; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#1a3a7a,#2563eb); color:#fff; border:none; font-size:16px; }
</style>
""", unsafe_allow_html=True)

from utils.extractor import extract_text_from_pdf, extract_text_from_txt
from utils.metrics   import run_all_metrics
from utils.scorer    import aviation_safety_score, get_aviation_flags
from utils.report    import generate_report_text

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈ Aviation Incident Report Checker")
    st.caption("ROUGE · BLEU · METEOR · BERTScore · Exact Match")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown("""
1. Upload the **pilot's incident report** (ground truth)
2. Paste the **AI-generated regulatory summary**
3. Click **Verify Report**
4. Get a **Compliance Score** with flagged omissions
    """)

    st.markdown("---")
    st.markdown("### Score Guide")
    st.markdown("""
- ✅ **85–100** → COMPLIANT
- ⚠️ **60–84**  → REVIEW REQUIRED
- ❌ **0–59**   → NON-COMPLIANT
    """)

    st.markdown("---")
    st.markdown("### Critical Fields Checked")
    fields = [
        "Flight number",
        "Aircraft type",
        "Date & time",
        "Location / airport",
        "Altitude & speed",
        "Weather conditions",
        "Crew actions taken",
        "Injuries reported",
        "Damage assessment",
        "Cause of incident",
    ]
    for f in fields:
        st.markdown(f"- {f}")

    st.markdown("---")
    st.markdown("### Metric Weights")
    for m, w in [("Exact Match","35%"),("BERTScore","25%"),
                 ("ROUGE-L","20%"),("METEOR","10%"),("BLEU","10%")]:
        st.markdown(f"**{m}** — {w}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# ✈ Aviation Incident Report Checker")
st.markdown(
    "Verify AI-generated regulatory summaries against original pilot incident reports. "
    "Catch omissions before they reach aviation authorities."
)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Pilot Incident Report (Ground Truth)")
    mode = st.radio("Input method", ["Upload PDF/TXT", "Paste text"],
                    horizontal=True, label_visibility="collapsed")
    reference_text = ""
    if mode == "Upload PDF/TXT":
        uploaded = st.file_uploader("Upload report", type=["pdf","txt"],
                                    label_visibility="collapsed")
        if uploaded:
            reference_text = (extract_text_from_pdf(uploaded)
                              if uploaded.name.endswith(".pdf")
                              else extract_text_from_txt(uploaded))
            st.success(f"Loaded: {uploaded.name} ({len(reference_text)} chars)")
            with st.expander("Preview"):
                st.text(reference_text[:800] + "..." if len(reference_text) > 800 else reference_text)
    else:
        reference_text = st.text_area(
            "Paste pilot report",
            height=260,
            placeholder="Paste the original pilot incident report here...",
            label_visibility="collapsed",
        )

with col2:
    st.markdown("### AI-Generated Regulatory Summary")
    briefing_text = st.text_area(
        "AI summary",
        height=300,
        placeholder="Paste the AI-generated regulatory summary here...",
        label_visibility="collapsed",
    )

st.markdown("")
ca, cb, cc = st.columns([1,1,2])
with ca:
    if st.button("Load Sample Data", use_container_width=True):
        st.session_state["sample"] = True
        st.rerun()
with cb:
    verify_btn = st.button(
        "Verify Report",
        type="primary",
        use_container_width=True,
        disabled=not (reference_text.strip() and briefing_text.strip()),
    )

# ── Sample loader ─────────────────────────────────────────────────────────────
if st.session_state.get("sample"):
    st.session_state["sample"] = False
    with open("sample_docs/pilot_incident_report.txt", "r") as f:
        sample_ref = f.read()
    with open("sample_docs/ai_summary_with_errors.txt", "r") as f:
        sample_ai = f.read()
    st.info("**Sample loaded!** Copy the text below into the boxes above.")
    with st.expander("Pilot Incident Report (copy to left box)"):
        st.text(sample_ref)
    with st.expander("AI Summary with Errors (copy to right box)"):
        st.text(sample_ai)

# ── Verify ────────────────────────────────────────────────────────────────────
if verify_btn and reference_text.strip() and briefing_text.strip():
    st.markdown("---")
    st.markdown("## Compliance Analysis Report")

    with st.spinner("Running metric analysis... (30-60 seconds for BERTScore)"):
        scores = run_all_metrics(briefing_text, reference_text)
        safety, status_label, status_class = aviation_safety_score(scores)
        flags  = get_aviation_flags(briefing_text, reference_text, scores)

    # ── Status banner ─────────────────────────────────────────────────────────
    icons = {
        "cleared":  ("✅", "#2ea043", "COMPLIANT — Safe to submit to regulators"),
        "review":   ("⚠️", "#f0b429", "REVIEW REQUIRED — Human check needed"),
        "grounded": ("❌", "#f85149", "NON-COMPLIANT — Do not submit"),
    }
    icon, color, label = icons[status_class]

    st.markdown(
        f'<div class="status-{status_class}">'
        f'<div style="font-size:48px">{icon}</div>'
        f'<div style="font-size:30px;font-weight:800;color:{color};margin:8px 0">'
        f'COMPLIANCE SCORE: {safety}/100</div>'
        f'<div style="font-size:18px;color:{color}">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("### Metric Breakdown")
    metric_info = {
        "exact":  ("Exact Match",  "Numbers & codes",   35),
        "bert":   ("BERTScore",    "Semantic meaning",  25),
        "rouge":  ("ROUGE-L",      "Content coverage",  20),
        "meteor": ("METEOR",       "Terminology",       10),
        "bleu":   ("BLEU",         "Phrasing match",    10),
    }
    cols = st.columns(5)
    for idx, (key, (name, desc, weight)) in enumerate(metric_info.items()):
        val = scores[key]
        color = "#2ea043" if val >= 80 else "#f0b429" if val >= 55 else "#f85149"
        with cols[idx]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-name">{name}</div>'
                f'<div class="metric-score" style="color:{color}">{val:.1f}</div>'
                f'<div class="metric-desc">{desc}</div>'
                f'<div class="metric-desc">Weight: {weight}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("")

    # ── Progress bars ─────────────────────────────────────────────────────────
    st.markdown("### Score Visualization")
    for key, (name, desc, weight) in metric_info.items():
        val = scores[key]
        st.markdown(f"**{name}** — {val:.1f}/100")
        st.progress(int(val))
    st.markdown("")

    # ── Critical field checklist ──────────────────────────────────────────────
    st.markdown("### Critical Field Checklist")
    checklist = flags.get("checklist", {})
    cc1, cc2 = st.columns(2)
    items = list(checklist.items())
    half  = len(items) // 2
    for i, (field, present) in enumerate(items):
        col = cc1 if i < half else cc2
        with col:
            icon2  = "✅" if present else "❌"
            color2 = "#2ea043" if present else "#f85149"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'padding:6px 0;border-bottom:1px solid #1e3060;">'
                f'<span style="font-size:16px">{icon2}</span>'
                f'<span style="font-size:13px;color:{color2}">{field}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("")

    # ── Flags ─────────────────────────────────────────────────────────────────
    st.markdown("### Issues Found")
    if flags["critical"]:
        st.markdown("**Critical — must fix before submission:**")
        for f in flags["critical"]:
            st.markdown(f'<div class="flag-critical">🔴 {f}</div>', unsafe_allow_html=True)
    if flags["warnings"]:
        st.markdown("**Warnings — review recommended:**")
        for f in flags["warnings"]:
            st.markdown(f'<div class="flag-warning">🟡 {f}</div>', unsafe_allow_html=True)
    if not flags["critical"] and not flags["warnings"]:
        st.markdown('<div class="flag-ok">✅ No critical issues found. Report appears complete.</div>',
                    unsafe_allow_html=True)
    st.markdown("")

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("### Download Report")
    report = generate_report_text(scores, safety, status_label, flags, briefing_text[:200])
    st.download_button(
        label="Download Compliance Report (.txt)",
        data=report,
        file_name="aviation_compliance_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("Raw Scores (JSON)"):
        st.code(json.dumps({**scores, "compliance_score": safety, "status": status_label}, indent=2))
