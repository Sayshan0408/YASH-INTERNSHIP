import streamlit as st
import requests
import json
import PyPDF2
import io

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Interrogator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=Epilogue:wght@400;500&display=swap');

html, body, [class*="css"]     { font-family: 'Epilogue', sans-serif; }
.main, .block-container        { background: #f5f0e8 !important; }
.block-container               { padding-top: 2rem; max-width: 980px; }

.title-wrap h1                 { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:900; color:#1a1410; letter-spacing:-0.03em; line-height:1; margin:0; }
.title-wrap h1 em              { color:#c0392b; font-style:italic; }
.title-wrap p                  { font-size:.83rem; color:#8a7a6a; margin:.35rem 0 0; }

.step-label                    { display:inline-block; background:#1a1410; color:#f5f0e8; font-family:'Syne',sans-serif; font-size:.58rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; padding:.22rem .7rem; border-radius:100px; margin-bottom:.6rem; }
.pdf-banner                    { background:#1a6a2a; color:#fff; border-radius:10px; padding:.75rem 1.1rem; margin-bottom:1rem; font-size:.88rem; font-weight:600; }

.q-card                        { background:#fff; border:1.5px solid #e0d8cc; border-radius:12px; padding:.9rem 1.1rem; margin-bottom:.5rem; }
.q-card .num                   { font-family:'Syne',sans-serif; font-size:.58rem; font-weight:800; color:#c0392b; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.25rem; }
.q-card .qtext                 { font-size:.91rem; font-weight:500; color:#1a1410; line-height:1.45; }

.sel-q-wrap                    { background:#1a1410; border-radius:12px; padding:.9rem 1.3rem; margin-bottom:1.4rem; }
.sel-q-wrap .lbl               { font-family:'Syne',sans-serif; font-size:.58rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; color:#8a7a6a; margin-bottom:.3rem; }
.sel-q-wrap .qtxt              { font-size:.95rem; color:#f5f0e8; font-weight:500; line-height:1.5; }

.badge-chunk                   { background:#d4820a; color:#fff; padding:.28rem .9rem; border-radius:100px; font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }
.badge-prompt                  { background:#1a4a8a; color:#fff; padding:.28rem .9rem; border-radius:100px; font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }

.info-box                      { background:#faf7f2; border:1px solid #e0d8cc; border-left:4px solid #c0392b; border-radius:8px; padding:.65rem .95rem; font-size:.81rem; color:#4a4038; margin-bottom:.9rem; }

.ans-card                      { background:#fff; border:1.5px solid #e0d8cc; border-radius:12px; padding:1.15rem 1.25rem; margin-bottom:.7rem; }
.ans-card-body                 { font-size:.96rem; line-height:1.85; color:#1a1410; }

div.stButton > button          { background:#1a1410!important; color:#f5f0e8!important; border:none!important; border-radius:9px!important; font-family:'Syne',sans-serif!important; font-weight:800!important; font-size:.83rem!important; padding:.5rem 1.2rem!important; }
div.stButton > button:hover    { background:#c0392b!important; }
div.stTextInput input,
div.stTextArea textarea        { background:#fff!important; border:2px solid #e0d8cc!important; border-radius:9px!important; font-family:'Epilogue',sans-serif!important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in [
    ("questions",  []),
    ("selected_q", None),
    ("pdf_text",   None),
    ("pdf_name",   None),
    ("results",    {}),
    ("generated",  False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Technique Definitions ─────────────────────────────────────────────────────
CHUNKS = {
    "🔢 Sequential": {
        "desc": "Answer broken into ordered, numbered steps — each flows logically from the previous.",
        "sys":  "You are a document analyst. Answer only in English. Use SEQUENTIAL CHUNKING: clearly numbered steps (Step 1, Step 2, Step 3...). Each step must logically follow the previous one.",
    },
    "🌲 Hierarchical": {
        "desc": "Structured broad to specific: Main idea → sub-topics → supporting details.",
        "sys":  "You are a document analyst. Answer only in English. Use HIERARCHICAL CHUNKING: start with the main idea, then break into sub-topics, then specific details.",
    },
    "🧠 Semantic": {
        "desc": "Grouped by meaning — each paragraph covers one conceptual cluster from the document.",
        "sys":  "You are a document analyst. Answer only in English. Use SEMANTIC CHUNKING: group related concepts by meaning and theme. Each paragraph covers exactly one semantic cluster.",
    },
    "⚡ Parallel": {
        "desc": "Multiple perspectives or aspects presented side-by-side for balanced coverage.",
        "sys":  "You are a document analyst. Answer only in English. Use PARALLEL CHUNKING: present 2-3 distinct perspectives or aspects, labeled clearly as Perspective A, Perspective B, etc.",
    },
}

PROMPTS = {
    "🎯 Zero-Shot": {
        "desc": "Direct answer from the document — no examples given. Tests pure comprehension.",
        "msg":  lambda q: f"Answer the following question based on the document content. Respond in English only, 4-6 sentences.\n\nQuestion: {q}",
    },
    "📚 Few-Shot": {
        "desc": "Guided by 2 example Q&A pairs before answering — shapes tone and format.",
        "msg":  lambda q: (
            f"Use these examples as a format guide, then answer the question.\n\n"
            f"Example Q: What is the main topic of this document?\n"
            f"Example A: The document primarily discusses [topic], focusing on [key aspect]. It provides detailed analysis of [subject] and draws conclusions about [outcome].\n\n"
            f"Example Q: What are the key findings?\n"
            f"Example A: The key findings include [finding 1] and [finding 2]. The document concludes that [conclusion].\n\n"
            f"Now answer in English only (4-6 sentences):\nQuestion: {q}"
        ),
    },
    "🔗 Chain-of-Thought": {
        "desc": "Step-by-step reasoning before the final answer — improves accuracy on complex questions.",
        "msg":  lambda q: (
            f"Think through this step by step before answering:\n"
            f"Step 1: Identify which section(s) of the document are relevant\n"
            f"Step 2: Extract the key information from those sections\n"
            f"Step 3: Connect the information to form a complete answer\n"
            f"Step 4: Write a clear final answer in English only (4-6 sentences)\n\n"
            f"Question: {q}"
        ),
    },
    "🎭 Role-Play": {
        "desc": "Expert researcher persona — authoritative, specialist-level response.",
        "msg":  lambda q: (
            f"You are a senior academic researcher and subject matter expert who has thoroughly studied this document. "
            f"Respond with authority and depth in English only (4-6 sentences):\n\nQuestion: {q}"
        ),
    },
}

# ── Groq API ──────────────────────────────────────────────────────────────────
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq(api_key: str, system: str, user: str) -> str:
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "llama-3.3-70b-versatile",
                "messages":    [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens":  800,
                "temperature": 0.7,
            },
            timeout=60,
        )
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return f"API Error: {data['error'].get('message', str(data['error']))}"
        return "No response received."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text[:12000].strip()
    except:
        return ""


def gen_questions(api_key: str, pdf_text: str) -> list:
    system = "You are a document analyst. Generate insightful English questions based on document content. Return only valid JSON."
    user = (
        f"Read this document:\n\n{pdf_text[:8000]}\n\n"
        "Generate exactly 6 specific, insightful questions in English that a reader would ask about this document. "
        "Cover: main ideas, key arguments, methodology, findings, implications, and one specific detail.\n\n"
        "Return ONLY a JSON array of 6 strings, nothing else:\n"
        '["Question 1?","Question 2?","Question 3?","Question 4?","Question 5?","Question 6?"]'
    )
    result = call_groq(api_key, system, user)
    try:
        s = result.find("[")
        e = result.rfind("]") + 1
        if s != -1 and e > s:
            qs = json.loads(result[s:e])
            return [q for q in qs if isinstance(q, str)][:6]
    except:
        pass
    return []


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-wrap">
  <h1>PDF <em>Interrogator</em></h1>
  <p>Upload a PDF → Questions auto-generate → Click one → Answers in 4 Chunking + 4 Prompting techniques</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key from console.groq.com"
    )
    st.markdown("[Get free Groq API key](https://console.groq.com/keys)", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Paste your Groq API key
2. Upload any PDF
3. 6 questions auto-generate
4. Click any question
5. Get answers in 8 technique tabs

**All answers in English.**  
**Free — no credit card needed!**
""")

# ════════════════════════════════
# STEP 1 — UPLOAD
# ════════════════════════════════
st.markdown('<span class="step-label">Step 1 — Upload PDF</span>', unsafe_allow_html=True)

uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded is not None:
    size_kb = round(uploaded.size / 1024, 1)
    st.markdown(
        f'<div class="pdf-banner">✅  <strong>{uploaded.name}</strong>  ·  {size_kb} KB — Loaded successfully!</div>',
        unsafe_allow_html=True,
    )

    if not groq_key:
        st.warning("Paste your Groq API key in the sidebar to continue.")
    elif not groq_key.startswith("gsk_"):
        st.error("Invalid key — Groq API keys start with gsk_")
    else:
        if st.session_state.pdf_name != uploaded.name:
            st.session_state.pdf_name   = uploaded.name
            st.session_state.questions  = []
            st.session_state.selected_q = None
            st.session_state.results    = {}
            st.session_state.generated  = False

            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_pdf_text(uploaded.read())
                st.session_state.pdf_text = pdf_text

            if not pdf_text:
                st.error("Could not extract text. Make sure the PDF is not a scanned image.")
                st.stop()

            with st.spinner("Generating questions from your PDF..."):
                st.session_state.questions = gen_questions(groq_key, pdf_text)

            if not st.session_state.questions:
                st.error("Could not generate questions. Check your Groq API key.")

# ════════════════════════════════
# STEP 2 — QUESTIONS
# ════════════════════════════════
if st.session_state.questions:
    st.markdown("")
    st.markdown('<span class="step-label">Step 2 — Click a Question</span>', unsafe_allow_html=True)
    st.caption("Auto-generated from your PDF — click Ask on any card below")

    col_l, col_r = st.columns(2, gap="medium")
    for i, q in enumerate(st.session_state.questions):
        col = col_l if i % 2 == 0 else col_r
        with col:
            st.markdown(f"""
            <div class="q-card">
              <div class="num">Q{i+1}</div>
              <div class="qtext">{q}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Ask Q{i+1} →", key=f"qb_{i}", use_container_width=True):
                st.session_state.selected_q = q
                st.session_state.results    = {}
                st.session_state.generated  = False
                st.rerun()

    st.markdown("---")
    st.markdown("**Or type your own question:**")
    ca, cb = st.columns([5, 1])
    with ca:
        cq = st.text_input("", placeholder="Ask anything about this PDF...",
                           label_visibility="collapsed", key="cq_inp")
    with cb:
        if st.button("Ask →", key="cq_btn", use_container_width=True):
            if cq.strip():
                st.session_state.selected_q = cq.strip()
                st.session_state.results    = {}
                st.session_state.generated  = False
                st.rerun()

# ════════════════════════════════
# STEP 3 — ANSWERS
# ════════════════════════════════
if st.session_state.selected_q and st.session_state.pdf_text and groq_key:
    st.markdown("")
    st.markdown('<span class="step-label">Step 3 — Answers</span>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sel-q-wrap">
      <div class="lbl">Answering</div>
      <div class="qtxt">"{st.session_state.selected_q}"</div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.generated:
        doc_ctx = f"Document content:\n\n{st.session_state.pdf_text}\n\n"
        total   = 8
        done    = 0
        bar     = st.progress(0, text="Generating answers...")
        res     = {"chunk": {}, "prompt": {}}

        for tname, tinfo in CHUNKS.items():
            user = doc_ctx + f"Answer this question in English only (4-6 sentences):\n\n{st.session_state.selected_q}"
            res["chunk"][tname] = call_groq(groq_key, tinfo["sys"], user)
            done += 1
            bar.progress(done / total, text=f"{done}/8 answers generated...")

        for tname, tinfo in PROMPTS.items():
            user = doc_ctx + tinfo["msg"](st.session_state.selected_q)
            res["prompt"][tname] = call_groq(
                groq_key,
                "You are a document expert. Answer only in English based on the provided document.",
                user
            )
            done += 1
            bar.progress(done / total, text=f"{done}/8 answers generated...")

        bar.progress(1.0, text="All 8 answers ready!")
        st.session_state.results   = res
        st.session_state.generated = True
        st.rerun()

    if st.session_state.generated and st.session_state.results:
        res = st.session_state.results

        # ── CHUNKING TABS ──
        st.markdown('<span class="badge-chunk">✂️ Chunking Techniques</span>', unsafe_allow_html=True)
        st.markdown("")
        for tab, (tname, tinfo) in zip(st.tabs(list(CHUNKS.keys())), CHUNKS.items()):
            with tab:
                st.markdown(f'<div class="info-box"><strong>{tname}</strong> — {tinfo["desc"]}</div>', unsafe_allow_html=True)
                text = res["chunk"].get(tname, "No answer generated.")
                st.markdown(f"""
                <div class="ans-card">
                  <div class="ans-card-body">{text}</div>
                </div>""", unsafe_allow_html=True)

        # ── PROMPTING TABS ──
        st.markdown("")
        st.markdown('<span class="badge-prompt">💬 Prompting Techniques</span>', unsafe_allow_html=True)
        st.markdown("")
        for tab, (tname, tinfo) in zip(st.tabs(list(PROMPTS.keys())), PROMPTS.items()):
            with tab:
                st.markdown(f'<div class="info-box"><strong>{tname}</strong> — {tinfo["desc"]}</div>', unsafe_allow_html=True)
                text = res["prompt"].get(tname, "No answer generated.")
                st.markdown(f"""
                <div class="ans-card">
                  <div class="ans-card-body">{text}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        if st.button("Ask a different question", key="reset_btn"):
            st.session_state.selected_q = None
            st.session_state.results    = {}
            st.session_state.generated  = False
            st.rerun()