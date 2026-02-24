import os
import streamlit as st

st.set_page_config(
    page_title="RFP Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Dark base */
    .stApp {
        background: #050510 !important;
        color: #e0e0e0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d2b 0%, #0a1628 100%) !important;
        border-right: 1px solid rgba(74,158,255,0.25);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* All main content sits above background */
    .block-container {
        position: relative;
        z-index: 2;
    }

    /* Glow layers */
    .glow-bg {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(ellipse at 15% 25%, rgba(74,158,255,0.09) 0%, transparent 50%),
            radial-gradient(ellipse at 85% 75%, rgba(76,175,80,0.07)  0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(120,80,255,0.05) 0%, transparent 65%);
    }

    /* Dot grid */
    .grid-bg {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image: radial-gradient(circle, rgba(74,158,255,0.07) 1px, transparent 1px);
        background-size: 42px 42px;
    }

    /* ── UI Components ── */
    .user-msg {
        background: rgba(30,58,95,0.85);
        border-left: 4px solid #4a9eff;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        backdrop-filter: blur(4px);
    }
    .bot-msg {
        background: rgba(26,42,26,0.85);
        border-left: 4px solid #4caf50;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        backdrop-filter: blur(4px);
    }
    .source-card {
        background: rgba(30,30,46,0.9);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #aaa;
    }
    .summary-card {
        background: rgba(26,26,53,0.92);
        border: 1px solid #4a9eff;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 14px;
        color: #d0e8ff;
        line-height: 1.8;
        backdrop-filter: blur(6px);
    }
    .summary-title {
        color: #4a9eff;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .badge-green { color: #4caf50; font-weight: bold; }
    .badge-red   { color: #f44336; font-weight: bold; }
    .stButton button {
        background: linear-gradient(90deg, #4a9eff, #0070f3) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .pipeline-step {
        background: rgba(30,30,46,0.85);
        border: 1px solid #2a2a3e;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        font-size: 13px;
    }
    .arrow { color: #4a9eff; font-size: 20px; text-align: center; }
</style>

<!-- Background layers -->
<div class="glow-bg"></div>
<div class="grid-bg"></div>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RFP Chatbot")
    st.markdown("*Powered by Qwen2.5-7B + FAISS + LangChain*")
    st.markdown("---")
    st.markdown("### 1️⃣ Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.success("📎 " + uploaded_file.name)
        st.markdown("### 2️⃣ Process Document")
        if st.button("⚙️ Process Document", use_container_width=True):
            with st.spinner("Running ingestion pipeline..."):
                try:
                    os.makedirs("uploads", exist_ok=True)
                    pdf_path = os.path.join("uploads", uploaded_file.name)
                    pdf_bytes = uploaded_file.read()
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_bytes)

                    from ingest import ingest_pdf
                    _, chunk_count = ingest_pdf(pdf_path)
                    st.session_state.chunk_count = chunk_count

                    from bot import load_chain
                    st.session_state.chain = load_chain()
                    st.session_state.doc_processed = True
                    st.session_state.summaries = {}
                    st.success("✅ Ready! " + str(chunk_count) + " chunks indexed.")
                except Exception as e:
                    st.error("❌ Error: " + str(e))

    st.markdown("---")
    if st.session_state.doc_processed:
        st.markdown('<span class="badge-green">● Document Ready</span>', unsafe_allow_html=True)
        st.caption(str(st.session_state.chunk_count) + " chunks in FAISS")
    else:
        st.markdown('<span class="badge-red">● No document loaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Tech Stack")
    st.caption("🔗 LangChain — Framework")
    st.caption("📄 PyPDFLoader — PDF Reading")
    st.caption("✂️ RecursiveCharacterTextSplitter")
    st.caption("🔢 all-MiniLM-L6-v2 — Embeddings")
    st.caption("💾 FAISS — Vector Database")
    st.caption("🤖 Qwen2.5-7B-Instruct — LLM")
    st.caption("🖥️ Streamlit — UI")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔄 Pipeline", "ℹ️ How It Works"])

with tab1:
    st.markdown("""
    <div style="text-align:center; padding: 80px 0 40px 0;">
        <h1 style="
            font-size: 3.2rem;
            font-weight: 900;
            background: linear-gradient(90deg, #4a9eff, #a064ff, #4caf50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            letter-spacing: 1px;
        ">
            Powered by Qwen2.5-7B + FAISS + LangChain
        </h1>
    </div>
    <hr style="border:1px solid rgba(74,158,255,0.12); margin-bottom:18px;">
    """, unsafe_allow_html=True)

    if not st.session_state.doc_processed:
        st.info("👈 Upload and process a PDF from the sidebar to begin.")
    else:
        msg_pairs = []
        i = 0
        while i < len(st.session_state.chat_history):
            msg = st.session_state.chat_history[i]
            if msg["role"] == "user":
                question  = msg["content"]
                answer_msg = st.session_state.chat_history[i + 1] if i + 1 < len(st.session_state.chat_history) else None
                msg_pairs.append((question, answer_msg))
                i += 2
            else:
                i += 1

        for pair_idx, (question, answer_msg) in enumerate(msg_pairs):
            st.markdown(
                '<div class="user-msg">🧑 <b>You:</b> ' + question + '</div>',
                unsafe_allow_html=True
            )

            if answer_msg:
                answer  = answer_msg["content"]
                sources = answer_msg.get("sources", [])

                st.markdown(
                    '<div class="bot-msg">🤖 <b>Assistant:</b> ' + answer + '</div>',
                    unsafe_allow_html=True
                )

                if sources:
                    with st.expander("📚 Source Chunks Used (" + str(len(sources)) + ")"):
                        for idx, doc in enumerate(sources):
                            page = doc.metadata.get("page", "?")
                            st.markdown(
                                '<div class="source-card"><b>Chunk ' + str(idx + 1) +
                                ' — Page ' + str(page) + '</b><br>' +
                                doc.page_content + '</div>',
                                unsafe_allow_html=True
                            )

                # ── Summary Button ─────────────────────────────────────────────
                sum_key = "summary_" + str(pair_idx)
                btn_col, spacer = st.columns([2, 8])
                with btn_col:
                    if st.button("📝 Summarize", key="btn_sum_" + str(pair_idx), use_container_width=True):
                        if sum_key not in st.session_state.summaries:
                            with st.spinner("Generating summary..."):
                                try:
                                    from bot import summarize_answer
                                    summary = summarize_answer(question, answer, sources)
                                    st.session_state.summaries[sum_key] = summary
                                    st.rerun()
                                except Exception as e:
                                    st.error("❌ Summary Error: " + str(e))

                if sum_key in st.session_state.summaries:
                    summary_text = st.session_state.summaries[sum_key]
                    bullets_html = ""
                    for line in summary_text.split("\n"):
                        line = line.strip()
                        if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                            bullets_html += "<li style='margin:4px 0'>" + line.lstrip("•-* ") + "</li>"
                        elif line:
                            bullets_html += "<li style='margin:4px 0'>" + line + "</li>"

                    st.markdown(
                        '<div class="summary-card">'
                        '<div class="summary-title">📋 Summary — ' +
                        question[:60] + ('...' if len(question) > 60 else '') +
                        '</div><ul style="padding-left:18px; margin:0">' +
                        bullets_html + '</ul></div>',
                        unsafe_allow_html=True
                    )
                    if st.button("✖ Hide Summary", key="hide_sum_" + str(pair_idx)):
                        del st.session_state.summaries[sum_key]
                        st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask a question",
                key="user_input",
                label_visibility="collapsed",
                placeholder="e.g. What are the eligibility requirements?"
            )
        with col2:
            send = st.button("Send 🚀", use_container_width=True)

        if send and user_input.strip():
            with st.spinner("Searching document and generating answer..."):
                try:
                    result  = st.session_state.chain.invoke({"query": user_input})
                    answer  = result["result"].strip()
                    sources = result.get("source_documents", [])
                    st.session_state.chat_history.append({"role": "user",      "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                    st.rerun()
                except Exception as e:
                    st.error("❌ Error: " + str(e))

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.summaries    = {}
                st.rerun()

with tab2:
    st.markdown("## 🔄 How Your Question Is Processed")
    st.markdown("### Phase 1 — Document Ingestion (runs once)")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 1, 2])
    with c1:
        st.markdown('<div class="pipeline-step">📄 <b>PDF File</b><br>Your document</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="pipeline-step">✂️ <b>Chunking</b><br>500 chars / 50 overlap</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="pipeline-step">💾 <b>FAISS Index</b><br>Saved to disk</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Phase 2 — Query Pipeline (every question)")
    q1, q2, q3, q4, q5, q6, q7 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
    with q1:
        st.markdown('<div class="pipeline-step">🧑 <b>User Question</b></div>', unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with q3:
        st.markdown('<div class="pipeline-step">🔢 <b>Embed Question</b><br>MiniLM 384-dim</div>', unsafe_allow_html=True)
    with q4:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with q5:
        st.markdown('<div class="pipeline-step">🔍 <b>FAISS Search</b><br>Top-3 chunks</div>', unsafe_allow_html=True)
    with q6:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with q7:
        st.markdown('<div class="pipeline-step">🤖 <b>Qwen2.5-7B</b><br>Generate answer</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Phase 3 — Summary Pipeline (on-demand)")
    s1, s2, s3, s4, s5 = st.columns([2, 0.5, 2, 0.5, 2])
    with s1:
        st.markdown('<div class="pipeline-step">📝 <b>Summarize Button</b><br>Clicked by user</div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="pipeline-step">🧠 <b>Q + Answer + Chunks</b><br>Sent to Qwen</div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with s5:
        st.markdown('<div class="pipeline-step">📋 <b>Bullet Summary</b><br>Shown below answer</div>', unsafe_allow_html=True)

    st.info("🔑 The LLM never reads the full PDF — it only sees the top-3 relevant chunks per question.")

with tab3:
    st.markdown("## ℹ️ Setup & Technology Guide")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📦 Installation (Python 3.11)")
        st.code("py -3.11 -m pip install -r requirements.txt", language="bash")
        st.markdown("### ▶️ Run App")
        st.code("py -3.11 -m streamlit run app.py", language="bash")
        st.markdown("### 🗂️ Project Files")
        st.table({
            "File": ["app.py", "ingest.py", "bot.py", ".env", "faiss_index/", "uploads/"],
            "Role": ["Streamlit UI", "PDF to FAISS", "Retrieval + LLM + Summary", "HuggingFace token", "Auto-created", "Auto-created"]
        })
    with col_r:
        st.markdown("### 🤖 Model — Qwen2.5-7B-Instruct")
        st.table({
            "Property": ["Parameters", "Context", "RAM", "Cost"],
            "Value": ["7B", "128K tokens", "~14GB / ~6GB quantized", "Free"]
        })
        st.markdown("### 💡 Low-RAM Alternative")
        st.code('repo_id = "Qwen/Qwen2.5-3B-Instruct"', language="python")
        st.markdown("### 🔑 .env Setup")
        st.code("HUGGINGFACEHUB_API_TOKEN=hf_your_token_here", language="bash")
        st.markdown("### ✅ Test Questions")
        st.markdown("- What is the project deadline?")
        st.markdown("- What are the eligibility requirements?")
        st.markdown("- What is the total budget?")
        st.markdown("- Who is the point of contact?")
        st.markdown("### 📝 Summary Feature")
        st.markdown("After each answer, click **📝 Summarize** to get a concise bullet-point summary.")