"""
🎓 CampusAI Navigator
Groq · Llama 3.3 70B · Text-based ReAct Agent · Semantic Chunking
"""

import streamlit as st
import streamlit.components.v1 as components
import requests, json, re, os, math
from bs4 import BeautifulSoup
from datetime import datetime

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
MODEL          = "llama-3.3-70b-versatile"
TEMP_REACT     = 0.2
TEMP_FINAL     = 0.7
TOP_P_REACT    = 0.9
TOP_P_FINAL    = 0.9
TOP_K_CHUNKS   = 4
MAX_TOKENS     = 4096
MAX_ITERATIONS = 5

SEARCH_MODES = {"keyword": "🔑 Keyword", "semantic": "🧠 Semantic", "hybrid": "⚡ Hybrid"}
SM_KEYS      = list(SEARCH_MODES.keys())   # ["keyword", "semantic", "hybrid"]

MODES = {
    "general":     "🎓 General",
    "events":      "🎉 Events",
    "faculty":     "👩‍🏫 Faculty",
    "resources":   "📚 Resources",
    "internships": "💼 Internships",
}
QUICK_PROMPTS = {
    "general":     ["What are the most popular student clubs?", "How do I appeal a grade?", "What mental health resources are available?"],
    "events":      ["What hackathons are coming up this month?", "When is the next cultural fest?", "Are there any seminars this week?"],
    "faculty":     ["Which professor is best for ML thesis guidance?", "Who researches NLP / LLMs here?", "How do I contact the HOD?"],
    "resources":   ["Where can I get free software licenses?", "What financial aid is available?", "Is there a writing/tutoring center?"],
    "internships": ["Which companies recruit from this campus?", "How do I register with the placement cell?", "Top internship tips for CS students?"],
}

# ── System prompt (text-based ReAct, no function calling) ─────
SYSTEM_PROMPT = """\
You are CampusAI Navigator, an expert assistant for college students.
Today: {date}
{college}
Focus: {mode_ctx}
Search strategy: {search_hint}

══ STRICT OUTPUT FORMAT ══
Follow this EXACT pattern every time. Never output text outside tags.

<thought>
Your reasoning: what you know, what you need, and why.
</thought>
<action>search_web: your search query here</action>

After seeing results, if you need to read a full page:
<thought>
What you learned and which URL to fetch.
</thought>
<action>fetch_page: https://full-url-here.com</action>

When you have enough information:
<thought>
What you found and why it answers the question.
</thought>
<final_answer>
## Title Here

Your comprehensive Markdown answer with ## headers and bullet points.
Include specific names, numbers, and facts. Minimum 150 words.

## Sources
- [Source Name](https://url)
</final_answer>

RULES:
- Always write <thought> before every <action> or <final_answer>.
- Only use: search_web: query  OR  fetch_page: url  inside <action>.
- Never write plain text outside the tags.
- Never fabricate facts.
"""

# ── CSS ───────────────────────────────────────────────────────
# KEY FIX: The 3 search buttons are placed inside a div with id="sm-row".
# We use CSS nth-child to target each button independently with its colour.
# This is 100% reliable and cannot be overridden by Streamlit's default gold.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --navy:#0d1b2a; --navy-mid:#142332; --navy-card:#1a2d3e; --navy-edge:#1f3347;
  --gold:#c9a84c; --gold-soft:#d4b96a; --gold-dim:#8a6e2f; --cream:#f5f0e8;
  --text:#d6cfc4; --text-dim:#8a9ab0; --text-faint:#4a5a6a;
  --green:#4caf7d; --red:#e05c5c; --blue:#6495ed; --purple:#a78bfa;
  --kw-color:#e05c5c; --sem-color:#a78bfa; --hyb-color:#c9a84c;
  --radius:10px; --shadow:0 4px 24px rgba(0,0,0,.35);
}
html,body,.stApp{background:var(--navy)!important;color:var(--text)!important;font-family:'Source Sans 3',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.5rem 2rem 4rem!important;max-width:900px!important;}
[data-testid="stSidebar"]{background:var(--navy-mid)!important;border-right:1px solid var(--navy-edge);}
[data-testid="stSidebar"] *{color:var(--text)!important;}

/* ─────────────────────────────────────────
   SEARCH MODE BUTTONS  — nth-child approach
   col 1 = Keyword (red), col 2 = Semantic (purple), col 3 = Hybrid (gold)
   .sm-inactive  = dimmed, unselected state
   .sm-active-kw / sm-active-sem / sm-active-hyb = selected glow state
   ───────────────────────────────────────── */

/* Strip Streamlit's default gold from ALL 3 buttons in the sm-row */
div.sm-row div[data-testid="column"]:nth-child(1) .stButton>button,
div.sm-row div[data-testid="column"]:nth-child(2) .stButton>button,
div.sm-row div[data-testid="column"]:nth-child(3) .stButton>button {
  background: var(--navy-card) !important;
  border-radius: 999px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .8rem !important;
  height: 38px !important;
  font-weight: 600 !important;
  transition: all .18s ease !important;
}

/* ── Keyword (col 1) — RED ── */
div.sm-row div[data-testid="column"]:nth-child(1) .stButton>button {
  border: 1.5px solid var(--kw-color) !important;
  color: var(--kw-color) !important;
  background: rgba(224,92,92,.08) !important;
}
div.sm-row div[data-testid="column"]:nth-child(1) .stButton>button:hover {
  background: rgba(224,92,92,.18) !important;
  box-shadow: 0 0 10px rgba(224,92,92,.4) !important;
}

/* ── Semantic (col 2) — PURPLE ── */
div.sm-row div[data-testid="column"]:nth-child(2) .stButton>button {
  border: 1.5px solid var(--sem-color) !important;
  color: var(--sem-color) !important;
  background: rgba(167,139,250,.08) !important;
}
div.sm-row div[data-testid="column"]:nth-child(2) .stButton>button:hover {
  background: rgba(167,139,250,.18) !important;
  box-shadow: 0 0 10px rgba(167,139,250,.4) !important;
}

/* ── Hybrid (col 3) — GOLD ── */
div.sm-row div[data-testid="column"]:nth-child(3) .stButton>button {
  border: 1.5px solid var(--hyb-color) !important;
  color: var(--hyb-color) !important;
  background: rgba(201,168,76,.08) !important;
}
div.sm-row div[data-testid="column"]:nth-child(3) .stButton>button:hover {
  background: rgba(201,168,76,.18) !important;
  box-shadow: 0 0 10px rgba(201,168,76,.4) !important;
}

/* ── ACTIVE state — brighter glow ── */
div.sm-row div[data-testid="column"]:nth-child(1) .stButton>button[data-active="true"],
div.sm-row div[data-testid="column"]:nth-child(1).sm-active .stButton>button {
  background: rgba(224,92,92,.2) !important;
  box-shadow: 0 0 16px rgba(224,92,92,.55) !important;
  font-weight: 800 !important;
}
div.sm-row div[data-testid="column"]:nth-child(2) .stButton>button[data-active="true"],
div.sm-row div[data-testid="column"]:nth-child(2).sm-active .stButton>button {
  background: rgba(167,139,250,.2) !important;
  box-shadow: 0 0 16px rgba(167,139,250,.55) !important;
  font-weight: 800 !important;
}
div.sm-row div[data-testid="column"]:nth-child(3) .stButton>button[data-active="true"],
div.sm-row div[data-testid="column"]:nth-child(3).sm-active .stButton>button {
  background: rgba(201,168,76,.2) !important;
  box-shadow: 0 0 16px rgba(201,168,76,.55) !important;
  font-weight: 800 !important;
}

/* ─────────────────────────────────────────
   CHAT / MESSAGES
   ───────────────────────────────────────── */
.chat-header{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:var(--gold);margin-bottom:.2rem;}
.chat-sub{color:var(--text-dim);font-size:.93rem;margin-bottom:1rem;font-weight:300;}
.status-bar{display:inline-flex;align-items:center;gap:.5rem;background:var(--navy-card);
  border:1px solid var(--navy-edge);border-radius:999px;padding:.28rem 1rem;
  font-size:.76rem;font-family:'JetBrains Mono',monospace;color:var(--text-dim);margin-bottom:.4rem;}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.status-online{background:var(--green);box-shadow:0 0 6px var(--green);}
.status-offline{background:var(--red);}

/* user bubble */
.user-wrap{display:flex;justify-content:flex-end;gap:.7rem;margin-bottom:1rem;animation:fadeUp .3s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.user-avatar{width:34px;height:34px;border-radius:50%;background:var(--gold-dim);
  display:flex;align-items:center;justify-content:center;font-size:.95rem;flex-shrink:0;margin-top:3px;}
.user-bubble{background:linear-gradient(135deg,var(--gold-dim),#6b5020);color:var(--cream);
  border-radius:var(--radius);border-bottom-right-radius:3px;padding:.8rem 1.1rem;
  font-size:.93rem;line-height:1.65;box-shadow:var(--shadow);max-width:78%;}

/* assistant avatar */
.asst-avatar{width:34px;height:34px;border-radius:50%;background:var(--navy-edge);
  border:1px solid var(--gold-dim);display:flex;align-items:center;justify-content:center;font-size:.95rem;}

/* answer block */
.answer-block{background:var(--navy-card);border:1px solid var(--navy-edge);border-radius:var(--radius);
  padding:1.1rem 1.3rem;margin-top:.4rem;line-height:1.78;font-size:.94rem;color:var(--text);}
.answer-block h1,.answer-block h2,.answer-block h3{
  font-family:'Playfair Display',serif;color:var(--gold-soft);margin:.8rem 0 .4rem;}
.answer-block strong{color:var(--cream);}
.answer-block a{color:var(--gold-soft);}
.answer-block ul,.answer-block ol{padding-left:1.4rem;}
.answer-block li{margin-bottom:.3rem;}
.answer-block code{font-family:'JetBrains Mono',monospace;background:var(--navy);
  padding:.1em .4em;border-radius:4px;font-size:.84em;}
.answer-block p{margin:.4rem 0;}

/* sources */
.sources-wrap{margin-top:.7rem;padding-top:.55rem;border-top:1px solid var(--navy-edge);}
.sources-label{font-size:.67rem;font-family:'JetBrains Mono',monospace;color:var(--text-faint);
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.38rem;}
.source-pill{display:inline-flex;align-items:center;gap:.28rem;background:var(--navy);
  border:1px solid var(--navy-edge);border-radius:999px;padding:.2rem .68rem;
  font-size:.74rem;color:var(--text-dim);text-decoration:none;margin:.18rem .15rem 0 0;
  transition:border-color .2s,color .2s;}
.source-pill:hover{border-color:var(--gold-dim);color:var(--gold-soft);}
.pill-dot{width:5px;height:5px;background:var(--gold-dim);border-radius:50%;flex-shrink:0;}

/* ReAct trace */
.react-step{border-radius:8px;padding:.65rem 1rem;margin-bottom:.55rem;
  border-left:3px solid;font-size:.87rem;line-height:1.62;}
.step-label{font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.12em;
  font-weight:700;margin-bottom:.4rem;text-transform:uppercase;
  display:flex;align-items:center;gap:.35rem;flex-wrap:wrap;}
.thought-step{background:rgba(201,168,76,.09);border-color:#c9a84c;}
.thought-step .step-label{color:#c9a84c;}
.thought-body{color:var(--cream);font-size:.9rem;line-height:1.6;}
.action-step{background:rgba(76,175,125,.09);border-color:#4caf7d;}
.action-step .step-label{color:#4caf7d;}
.action-code{font-family:'JetBrains Mono',monospace;font-size:.84em;color:#7effc4;
  background:rgba(76,175,125,.12);padding:.22em .65em;border-radius:5px;
  display:inline-block;margin-top:.2rem;word-break:break-all;}
.observation-step{background:rgba(100,149,237,.09);border-color:#6495ed;}
.observation-step .step-label{color:#6495ed;}

/* badges */
.badge{display:inline-flex;align-items:center;border-radius:4px;padding:.04rem .4rem;
  font-size:.63rem;font-family:'JetBrains Mono',monospace;border:1px solid;
  vertical-align:middle;white-space:nowrap;}
.b-blue  {background:rgba(100,149,237,.13);border-color:rgba(100,149,237,.35);color:#6495ed;}
.b-red   {background:rgba(224,92,92,.1);border-color:rgba(224,92,92,.3);color:#f07070;}
.b-purple{background:rgba(167,139,250,.1);border-color:rgba(167,139,250,.3);color:#a78bfa;}
.b-gold  {background:rgba(201,168,76,.1);border-color:rgba(201,168,76,.3);color:#c9a84c;}
.b-green {background:rgba(76,175,125,.1);border-color:rgba(76,175,125,.3);color:#4caf7d;}

/* expander */
[data-testid="stExpander"]{background:var(--navy-card)!important;
  border:1px solid var(--navy-edge)!important;border-radius:var(--radius)!important;margin-bottom:.55rem!important;}
[data-testid="stExpander"] summary{
  font-family:'JetBrains Mono',monospace!important;font-size:.77rem!important;color:var(--text-dim)!important;}

/* param box */
.param-box{background:var(--navy-card);border:1px solid var(--navy-edge);border-radius:8px;padding:.6rem .85rem;margin-top:.4rem;}
.param-section{font-size:.62rem;font-family:'JetBrains Mono',monospace;color:var(--text-faint);
  letter-spacing:.08em;text-transform:uppercase;margin:.5rem 0 .28rem;}
.param-row{display:flex;justify-content:space-between;margin-bottom:.16rem;
  font-size:.73rem;font-family:'JetBrains Mono',monospace;}
.param-label{color:var(--text-faint);}
.param-value{color:var(--gold-soft);font-weight:700;}
.param-bar-wrap{width:100%;height:4px;background:var(--navy-edge);border-radius:2px;margin-top:.1rem;}
.param-bar{height:4px;background:linear-gradient(90deg,var(--gold-dim),var(--gold));border-radius:2px;}

/* inputs */
.stTextInput input{background:var(--navy-card)!important;border:1px solid var(--navy-edge)!important;
  border-radius:var(--radius)!important;color:var(--text)!important;font-size:.94rem!important;}
.stTextInput input:focus{border-color:var(--gold-dim)!important;
  box-shadow:0 0 0 2px rgba(201,168,76,.15)!important;}
.stTextInput input::placeholder{color:var(--text-faint)!important;}

/* default (gold) button — for Ask→, Clear, Quick Prompts */
.stButton>button{
  background:linear-gradient(135deg,var(--gold),var(--gold-dim))!important;
  color:var(--navy)!important;border:none!important;border-radius:var(--radius)!important;
  font-weight:700!important;transition:opacity .2s,transform .15s!important;height:42px!important;}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important;}

/* misc */
.sect-divider{border-top:1px solid var(--navy-edge);margin:1.1rem 0 .9rem;}
.empty-state{text-align:center;padding:3.5rem 1rem;}
.empty-icon{font-size:3rem;margin-bottom:.8rem;}
.empty-title{font-family:'Playfair Display',serif;font-size:1.35rem;color:var(--text);margin-bottom:.4rem;}
.empty-sub{font-size:.87rem;color:var(--text-faint);}
.footer{text-align:center;font-size:.65rem;font-family:'JetBrains Mono',monospace;
  letter-spacing:.11em;color:var(--text-faint);margin-top:2.5rem;padding-top:.9rem;
  border-top:1px solid var(--navy-edge);}
[data-testid="stMetric"]{background:var(--navy-card)!important;border:1px solid var(--navy-edge)!important;
  border-radius:var(--radius)!important;padding:.55rem .75rem!important;}
[data-testid="stMetricValue"]{color:var(--gold)!important;font-family:'JetBrains Mono',monospace!important;}

/* anchor for auto-scroll to latest answer */
.latest-answer-anchor{height:0;overflow:hidden;}
</style>
"""


# ══════════════════════════════════════════════════════════════
# SEARCH ALGORITHMS
# ══════════════════════════════════════════════════════════════

def _keyword_score(chunk, query):
    q = re.findall(r'\w+', query.lower())
    c = set(re.findall(r'\w+', chunk.lower()))
    return sum(1 for t in q if t in c) / max(len(q), 1)

def _semantic_score(chunk, query):
    def tf(text):
        words = re.findall(r'\w+', text.lower())
        freq = {}
        for w in words: freq[w] = freq.get(w,0)+1
        n = max(len(words),1)
        return {w: c/n for w,c in freq.items()}
    qt, ct = tf(query), tf(chunk)
    vocab  = set(qt)|set(ct)
    dot    = sum(qt.get(w,0)*ct.get(w,0) for w in vocab)
    nq     = math.sqrt(sum(v**2 for v in qt.values()))
    nc     = math.sqrt(sum(v**2 for v in ct.values()))
    return dot/(nq*nc) if nq and nc else 0.0

def _hybrid_score(chunk, query):
    return .5*_keyword_score(chunk,query) + .5*_semantic_score(chunk,query)

def rank_chunks(chunks, query, search_mode="hybrid", top_k=TOP_K_CHUNKS):
    fn = {"keyword":_keyword_score,"semantic":_semantic_score}.get(search_mode,_hybrid_score)
    scored = [(fn(c,query),i,c) for i,c in enumerate(chunks)]
    scored.sort(key=lambda x:-x[0])
    return sorted(scored[:top_k], key=lambda x:x[1])

def semantic_chunk(text, chunk_size=400, overlap=80):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    chunks, cur_words, cur_sents = [], 0, []
    for sent in sents:
        wc = len(sent.split())
        if cur_words+wc > chunk_size and cur_sents:
            chunks.append(" ".join(cur_sents))
            ov, owc = [], 0
            for s in reversed(cur_sents):
                w = len(s.split())
                if owc+w > overlap: break
                ov.insert(0,s); owc+=w
            cur_sents, cur_words = ov, owc
        cur_sents.append(sent); cur_words+=wc
    if cur_sents: chunks.append(" ".join(cur_sents))
    return chunks


# ══════════════════════════════════════════════════════════════
# WEB TOOLS
# ══════════════════════════════════════════════════════════════

def search_web(query, n=5):
    try:
        r = requests.post(
            "https://html.duckduckgo.com/html/", timeout=12,
            data={"q": query, "b": "", "kl": "us-en"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; CampusAI/1.0)"},
        )
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for el in soup.select(".result__body")[:n]:
            t = el.select_one(".result__title")
            s = el.select_one(".result__snippet")
            l = el.select_one(".result__url")
            if t and s:
                out.append({"title":t.get_text(strip=True),
                            "snippet":s.get_text(strip=True),
                            "url":l.get_text(strip=True) if l else ""})
        return out or [{"title":"No results","snippet":"Try a different query.","url":""}]
    except Exception as e:
        return [{"title":"Error","snippet":str(e),"url":""}]

def fetch_page(url, query="", search_mode="hybrid"):
    if not url.startswith("http"):
        url = "https://"+url
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","form"]):
            tag.decompose()
        raw = re.sub(r"\n{3,}","\n\n", soup.get_text(separator="\n",strip=True))
    except Exception as e:
        return {"url":url,"total":0,"selected":0,"content":f"Fetch error: {e}","indices":[]}
    chunks = semantic_chunk(raw)
    if not chunks:
        return {"url":url,"total":0,"selected":0,"content":"(empty)","indices":[]}
    top = (rank_chunks(chunks,query,search_mode,TOP_K_CHUNKS)
           if query else [(0.0,i,c) for i,c in enumerate(chunks[:TOP_K_CHUNKS])])
    content = "\n\n---\n\n".join(f"[Chunk {i+1}] {c}" for _,i,c in top)
    return {"url":url,"total":len(chunks),"selected":len(top),
            "content":content,"indices":[i for _,i,_ in top]}


# ══════════════════════════════════════════════════════════════
# GROQ API (plain chat, no function calling)
# ══════════════════════════════════════════════════════════════

def groq_chat(messages, api_key, temperature=TEMP_REACT, top_p=TOP_P_REACT):
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
        json={"model":MODEL,"messages":messages,"temperature":temperature,
              "top_p":top_p,"max_tokens":MAX_TOKENS},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""

def build_system(college_name, college_url, mode, search_mode):
    college = (
        f"The user is asking about {college_name}"
        +(f" ({college_url})" if college_url else "")+"."
    ) if college_name else "No specific college set — give general campus advice."
    mode_ctx = {
        "general":     "All campus topics — clubs, administration, campus life.",
        "events":      "Events, hackathons, cultural fests, seminars, workshops.",
        "faculty":     "Professors, research interests, office hours, contact details.",
        "resources":   "Libraries, labs, scholarships, mental health, tutoring.",
        "internships": "Placements, recruiters, career tips, registration procedures.",
    }.get(mode,"All campus topics.")
    search_hint = {
        "keyword":  "Use exact keyword queries matching specific names, codes, or terms.",
        "semantic": "Use broad conceptual queries focused on topics and themes.",
        "hybrid":   "Balance exact keywords with broad topic coverage.",
    }.get(search_mode,"")
    return SYSTEM_PROMPT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        college=college, mode_ctx=mode_ctx, search_hint=search_hint,
    )


# ══════════════════════════════════════════════════════════════
# TAG PARSER
# ══════════════════════════════════════════════════════════════

def parse_tags(text):
    blocks = []
    for m in re.finditer(
        r'<(thought|action|final_answer)>(.*?)</\1>',
        text, re.DOTALL|re.IGNORECASE
    ):
        tag, content = m.group(1).lower(), m.group(2).strip()
        if content:
            blocks.append({"tag":tag,"content":content})
    if not blocks and text.strip():
        blocks.append({"tag":"final_answer","content":text.strip()})
    return blocks

def parse_action(s):
    s = s.strip()
    if re.match(r'search_web\s*:', s, re.IGNORECASE):
        return "search_web", re.sub(r'^search_web\s*:\s*','',s,flags=re.IGNORECASE).strip()
    if re.match(r'fetch_page\s*:', s, re.IGNORECASE):
        return "fetch_page", re.sub(r'^fetch_page\s*:\s*','',s,flags=re.IGNORECASE).strip()
    return "search_web", s


# ══════════════════════════════════════════════════════════════
# REACT AGENT
# ══════════════════════════════════════════════════════════════

def run_agent(query, api_key, college_name, college_url, mode, search_mode):
    messages     = [
        {"role":"system","content":build_system(college_name,college_url,mode,search_mode)},
        {"role":"user",  "content":f"Question: {query}"},
    ]
    steps        = []
    sources      = []
    tool_count   = 0
    final_answer = None

    for _ in range(MAX_ITERATIONS):
        try:
            raw = groq_chat(messages, api_key)
        except Exception as e:
            return {"answer":f"❌ Groq error: {e}","sources":[],"tool_calls":tool_count,
                    "steps":steps,"search_mode":search_mode}

        messages.append({"role":"assistant","content":raw})
        blocks = parse_tags(raw)
        if not blocks:
            break

        obs_for_model = None

        for block in blocks:
            tag, content = block["tag"], block["content"]

            if tag == "thought":
                steps.append({"type":"thought","content":content})

            elif tag == "action":
                tool_name, argument = parse_action(content)
                tool_count += 1
                steps.append({"type":"action","content":f"{tool_name}: {argument}","tool":tool_name})

                if tool_name == "search_web":
                    results = search_web(argument, n=5)
                    for r in results:
                        u = r.get("url","")
                        if u and u not in sources: sources.append(u)

                    # Display text (rich Markdown)
                    obs_display = (
                        f"Found **{len(results)}** results "
                        f"[**{search_mode}** search] for: *{argument}*\n\n"
                        + "\n\n".join(
                            f"- **{r['title']}**  \n  {r['snippet']}  \n  🔗 {r['url']}"
                            for r in results[:5]
                        )
                    )
                    # Plain text for model context
                    obs_for_model = (
                        f"Search results for '{argument}':\n"
                        + "\n".join(
                            f"  [{r['title']}] {r['snippet']} — {r['url']}"
                            for r in results[:5]
                        )
                    )
                    chunk_info = False

                elif tool_name == "fetch_page":
                    page = fetch_page(argument, query=query, search_mode=search_mode)
                    if argument not in sources: sources.append(argument)
                    obs_display = (
                        f"**Fetched:** {page['url']}  \n"
                        f"Chunks: {page['total']} total → {page['selected']} selected "
                        f"({search_mode}, top_k={TOP_K_CHUNKS})\n\n{page['content']}"
                    )
                    obs_for_model = f"Page content from {page['url']}:\n{page['content']}"
                    chunk_info = True
                else:
                    obs_display = obs_for_model = "Unknown tool."
                    chunk_info = False

                steps.append({"type":"observation","content":obs_display,
                               "chunk_info":chunk_info,"search_mode":search_mode})

            elif tag == "final_answer":
                final_answer = content
                break

        if final_answer is not None:
            break

        if obs_for_model:
            messages.append({
                "role":"user",
                "content":(
                    f"<observation>\n{obs_for_model}\n</observation>\n\n"
                    "Continue. Write <final_answer> when ready, or another <thought>+<action>."
                )
            })
        else:
            break

    # Fallback synthesis
    if not final_answer:
        obs_texts = [s["content"] for s in steps if s["type"]=="observation"]
        context   = "\n\n".join(obs_texts) if obs_texts else "(no data fetched)"
        try:
            final_answer = groq_chat(
                [
                    {"role":"system","content":(
                        "Write a comprehensive Markdown answer. Use ## headers and bullet points. "
                        "Include specific facts. End with ## Sources listing URLs. "
                        "Minimum 150 words. No tags, just plain Markdown."
                    )},
                    {"role":"user","content":f"Question: {query}\n\nResearch:\n{context}\n\nAnswer:"},
                ],
                api_key, temperature=TEMP_FINAL, top_p=TOP_P_FINAL,
            )
        except Exception as e:
            final_answer = f"⚠️ Could not generate answer: {e}"

    if not final_answer or not final_answer.strip():
        final_answer = "I was unable to generate an answer. Please try again."

    return {"answer":final_answer,"sources":_build_sources(final_answer,sources),
            "tool_calls":tool_count,"steps":steps,"search_mode":search_mode}

def _build_sources(text, raw_urls):
    src, seen = [], set()
    for m in re.finditer(r'\[([^\]]+)\]\((https?://[^\)]+)\)', text):
        t, u = m.group(1), m.group(2)
        if u not in seen:
            seen.add(u); src.append({"title":t,"url":u})
    for u in raw_urls:
        if u and u not in seen:
            seen.add(u)
            src.append({"title":u,"url":u if u.startswith("http") else "https://"+u})
    return src


# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════

def esc(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def sm_badge(mode):
    cfg = {"keyword":("b-red","🔑 keyword"),"semantic":("b-purple","🧠 semantic"),
           "hybrid":("b-gold","⚡ hybrid")}
    cls, lbl = cfg.get(mode,("b-blue",mode))
    return f'<span class="badge {cls}">{lbl}</span>'

def param_bar(label, value, max_val=1.0):
    pct = int((value/max_val)*100)
    return (
        f'<div class="param-row"><span class="param-label">{label}</span>'
        f'<span class="param-value">{value}</span></div>'
        f'<div class="param-bar-wrap">'
        f'<div class="param-bar" style="width:{pct}%"></div></div>'
    )


# ── Search mode selector ──────────────────────────────────────
def render_search_mode_selector():
    """
    Renders 3 pill buttons inside a div.sm-row wrapper.
    CSS nth-child rules above give each column its own colour:
      col-1 = RED (keyword), col-2 = PURPLE (semantic), col-3 = GOLD (hybrid)
    The active button gets extra glow via JS setAttribute.
    """
    current = st.session_state.search_mode

    # Wrap columns in a named div so CSS nth-child can target them
    st.markdown('<div class="sm-row">', unsafe_allow_html=True)
    cols    = st.columns(3)
    changed = False

    sm_items = [
        ("keyword",  "🔑 Keyword"),
        ("semantic", "🧠 Semantic"),
        ("hybrid",   "⚡ Hybrid"),
    ]
    for col, (key, label) in zip(cols, sm_items):
        is_active = (key == current)
        btn_text  = label + (" ✓" if is_active else "")
        with col:
            if st.button(btn_text, key=f"sm_{key}", use_container_width=True):
                if not is_active:
                    st.session_state.search_mode = key
                    changed = True

    st.markdown('</div>', unsafe_allow_html=True)

    if changed:
        st.rerun()

    # JS: add glow to the active button only
    # Each active button text is unique (has " ✓"), so text matching is safe here
    active_label = {"keyword":"🔑 Keyword ✓","semantic":"🧠 Semantic ✓","hybrid":"⚡ Hybrid ✓"}[current]
    glow_color   = {"keyword":"rgba(224,92,92,.6)","semantic":"rgba(167,139,250,.6)","hybrid":"rgba(201,168,76,.6)"}[current]
    safe_label   = active_label.replace("'","\\'")
    js = f"""
<script>
(function(){{
  function styleBtn(){{
    var btns = window.parent.document.querySelectorAll('button');
    btns.forEach(function(b){{
      if(b.innerText.trim()==='{safe_label}'){{
        b.style.setProperty('box-shadow','0 0 18px {glow_color}','important');
        b.style.setProperty('font-weight','800','important');
      }}
    }});
  }}
  styleBtn();
  setTimeout(styleBtn, 300);
  setTimeout(styleBtn, 800);
}})();
</script>"""
    components.html(js, height=0, scrolling=False)


# ── ReAct trace expander ──────────────────────────────────────
def render_react_trace(steps, tool_calls, search_mode):
    if not steps:
        return
    label = (
        f"🧠 ReAct Trace  ·  {tool_calls} tool call{'s' if tool_calls!=1 else ''}  ·  "
        f"T={TEMP_REACT}/top_p={TOP_P_REACT}  ·  top_k={TOP_K_CHUNKS}  ·  {search_mode}"
    )
    with st.expander(label, expanded=False):
        for i, step in enumerate(steps):
            stype = step["type"]
            if stype == "thought":
                st.markdown(
                    f'<div class="react-step thought-step">'
                    f'<div class="step-label">💭 THOUGHT'
                    f'<span class="badge b-gold">step {i+1}</span>'
                    f'<span class="badge b-blue">T={TEMP_REACT}</span>'
                    f'</div>'
                    f'<div class="thought-body">{esc(step["content"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif stype == "action":
                icon  = "🔍" if step.get("tool")=="search_web" else "🌐"
                st.markdown(
                    f'<div class="react-step action-step">'
                    f'<div class="step-label">{icon} ACTION {sm_badge(search_mode)}</div>'
                    f'<div class="action-code">{esc(step["content"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif stype == "observation":
                chunk_badge = (
                    f'<span class="badge b-blue">top_k={TOP_K_CHUNKS}</span>'
                    f'{sm_badge(search_mode)}'
                    if step.get("chunk_info") else sm_badge(search_mode)
                )
                st.markdown(
                    f'<div class="react-step observation-step">'
                    f'<div class="step-label">👁️ OBSERVATION {chunk_badge}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(step["content"])
                st.markdown("</div>", unsafe_allow_html=True)


# ── Message renderer ──────────────────────────────────────────
def render_message(role, content, sources=None, tool_calls=0,
                   steps=None, search_mode="hybrid"):
    steps = steps or []

    if role == "user":
        st.markdown(
            f'<div class="user-wrap">'
            f'<div class="user-avatar">👤</div>'
            f'<div class="user-bubble">{esc(content)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Assistant
    col_av, col_body = st.columns([0.04, 0.96])
    with col_av:
        st.markdown('<div class="asst-avatar" style="margin-top:4px;">🎓</div>',
                    unsafe_allow_html=True)
    with col_body:
        # ReAct trace (collapsed by default)
        if steps and tool_calls > 0:
            render_react_trace(steps, tool_calls, search_mode)

        # Strip Sources section from body (shown as pills)
        clean = re.sub(
            r'\n*(#{1,3}\s*Sources?|^\*\*Sources?\*\*)\s*\n.*',
            '', content, flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
        ).strip()

        # Answer block — always rendered
        st.markdown('<div class="answer-block">', unsafe_allow_html=True)
        st.markdown(clean if clean else "*(No answer generated — please try again.)*")

        if sources:
            pills = "".join(
                f'<a href="{s["url"]}" target="_blank" class="source-pill">'
                f'<span class="pill-dot"></span>{esc(s["title"][:52])}</a>'
                for s in sources
            )
            st.markdown(
                f'<div class="sources-wrap">'
                f'<div class="sources-label">📎 Sources</div>{pills}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<h2 style="font-family:\'Playfair Display\',serif;color:#c9a84c;margin-bottom:.1rem;">'
            '⚙️ CampusAI</h2>',
            unsafe_allow_html=True,
        )
        st.caption("College Event & Resource Navigator")
        st.divider()

        st.markdown("**🔑 Groq API Key**")
        key = st.text_input("Key",type="password",placeholder="gsk_…",
                            label_visibility="collapsed",key="_api_key_input")
        if key: st.session_state.api_key = key
        if not st.session_state.api_key:
            st.markdown('<a href="https://console.groq.com/keys" target="_blank" '
                        'style="font-size:.77rem;color:#c9a84c;">→ Get free key</a>',
                        unsafe_allow_html=True)
        elif st.session_state.api_key.startswith("gsk_"):
            st.success("✓ Key accepted",icon="🔓")
        else:
            st.error("Key should start with gsk_")

        st.divider()

        st.markdown("**🏛️ Your College**")
        st.session_state.college_name = st.text_input(
            "Name",value=st.session_state.college_name,
            placeholder="e.g. Sathyabama, VIT…",label_visibility="collapsed")
        st.session_state.college_url = st.text_input(
            "URL",value=st.session_state.college_url,
            placeholder="e.g. https://sathyabama.ac.in",label_visibility="collapsed")

        st.divider()

        # Sidebar search mode selector (in sync with main buttons)
        st.markdown("**🔍 Search Mode**")
        sm_labels = list(SEARCH_MODES.values())
        sm_idx    = SM_KEYS.index(st.session_state.search_mode) if st.session_state.search_mode in SM_KEYS else 2
        sm_sel    = st.selectbox("SM",sm_labels,index=sm_idx,
                                 label_visibility="collapsed",key="sb_sm")
        sel_key   = SM_KEYS[sm_labels.index(sm_sel)]
        if sel_key != st.session_state.search_mode:
            st.session_state.search_mode = sel_key
            st.rerun()

        # Colour-coded description
        desc_html = {
            "keyword":  '<span style="color:#e05c5c;">🔑 Keyword</span> — exact term matching, best for names & codes.',
            "semantic": '<span style="color:#a78bfa;">🧠 Semantic</span> — contextual similarity, best for topics.',
            "hybrid":   '<span style="color:#c9a84c;">⚡ Hybrid</span> — blended search. ✓ Recommended',
        }.get(st.session_state.search_mode,"")
        st.markdown(f'<div style="font-size:.72rem;color:#8a9ab0;margin-top:-.2rem;">{desc_html}</div>',
                    unsafe_allow_html=True)

        st.divider()

        st.markdown("**🎯 Focus Mode**")
        mode_labels = list(MODES.values())
        mode_keys   = list(MODES.keys())
        idx = mode_keys.index(st.session_state.mode) if st.session_state.mode in mode_keys else 0
        sel = st.selectbox("Mode",mode_labels,index=idx,label_visibility="collapsed")
        st.session_state.mode = mode_keys[mode_labels.index(sel)]

        st.divider()

        st.markdown("**⚡ Quick Prompts**")
        for prompt in QUICK_PROMPTS.get(st.session_state.mode,[]):
            if st.button(prompt,use_container_width=True,key=f"qp_{hash(prompt)}"):
                st.session_state.pending_query = prompt
                st.rerun()

        st.divider()
        c1,c2 = st.columns(2)
        c1.metric("Queries",st.session_state.queries)
        c2.metric("Sources",st.session_state.sources_used)
        st.divider()

        st.markdown("**🧪 Sampling Params**")
        st.markdown(
            f'<div class="param-box">'
            f'<div class="param-section">ReAct Phase</div>'
            f'{param_bar("temperature",TEMP_REACT)}{param_bar("top_p",TOP_P_REACT)}'
            f'<div class="param-section">Final Answer</div>'
            f'{param_bar("temperature",TEMP_FINAL)}{param_bar("top_p",TOP_P_FINAL)}'
            f'<div class="param-section">Chunk Selection</div>'
            f'{param_bar("top_k chunks",TOP_K_CHUNKS,max_val=10)}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("🗑️ Clear Chat",use_container_width=True):
            st.session_state.messages     = []
            st.session_state.queries      = 0
            st.session_state.sources_used = 0
            st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="CampusAI Navigator",page_icon="🎓",
                   layout="wide",initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

for k,v in {
    "messages":[],"college_url":"","college_name":"",
    "mode":"general","search_mode":"hybrid",
    "queries":0,"sources_used":0,
    "api_key":os.environ.get("GROQ_API_KEY",""),
    "pending_query":"",
}.items():
    if k not in st.session_state: st.session_state[k]=v

render_sidebar()

cname = st.session_state.college_name or "Your Campus"
st.markdown(
    f'<div class="chat-header"><em>{cname}</em> Navigator</div>'
    f'<p class="chat-sub">Ask anything about faculty, events, resources — '
    f'ReAct · Semantic Chunking · Llama 3.3 70B on Groq.</p>',
    unsafe_allow_html=True,
)

api_key     = st.session_state.api_key
search_mode = st.session_state.search_mode

# Status bar
online = bool(api_key and api_key.startswith("gsk_"))
dot    = "status-online" if online else "status-offline"
txt    = (
    f"Groq Online · {MODEL} · T={TEMP_REACT}/top_p={TOP_P_REACT} → "
    f"T={TEMP_FINAL}/top_p={TOP_P_FINAL} · top_k={TOP_K_CHUNKS}"
    if online else "Enter your Groq API key in the sidebar to begin"
)
sm_label = SEARCH_MODES.get(search_mode,"")
st.markdown(
    f'<div class="status-bar">'
    f'<span class="status-dot {dot}"></span>{txt}'
    f' &nbsp;·&nbsp; Mode: <b>{st.session_state.mode.title()}</b>'
    f' &nbsp;·&nbsp; Search: <b>{sm_label}</b>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Search mode buttons ───────────────────────────────────────
render_search_mode_selector()

# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(
        msg["role"], msg["content"],
        msg.get("sources"),
        msg.get("tool_calls",0),
        msg.get("steps",[]),
        msg.get("search_mode","hybrid"),
    )

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">🎓</div>'
        '<div class="empty-title">Ask me anything about your campus</div>'
        '<div class="empty-sub">Faculty · Events · Research · Clubs · Internships · Resources</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="sect-divider"></div>', unsafe_allow_html=True)


# ── Query handler ─────────────────────────────────────────────
def handle_query(query):
    query = query.strip()
    if not query: return
    if not st.session_state.api_key or not st.session_state.api_key.startswith("gsk_"):
        st.error("⚠️ Please enter a valid Groq API key in the sidebar.")
        return

    sm   = st.session_state.search_mode
    spin = {"keyword":"🔑 Keyword","semantic":"🧠 Semantic","hybrid":"⚡ Hybrid"}.get(sm,"")

    st.session_state.messages.append({"role":"user","content":query})
    st.session_state.queries += 1

    with st.spinner(f"{spin} search + ReAct reasoning…"):
        result = run_agent(
            query, st.session_state.api_key,
            st.session_state.college_name, st.session_state.college_url,
            st.session_state.mode, sm,
        )

    st.session_state.sources_used += len(result["sources"])
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     result["answer"],
        "sources":     result["sources"],
        "tool_calls":  result["tool_calls"],
        "steps":       result.get("steps",[]),
        "search_mode": result.get("search_mode",sm),
    })
    st.rerun()


if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = ""
    handle_query(q)

# Input row
col_in, col_btn = st.columns([5,1])
with col_in:
    st.text_input("Ask",
                  placeholder='"Which professor is best for ML thesis guidance?"',
                  key="chat_input",label_visibility="collapsed")
with col_btn:
    st.write("")
    if st.button("Ask →",use_container_width=True,key="ask_btn"):
        handle_query(st.session_state.get("chat_input",""))

st.markdown(
    f'<div class="footer">CAMPUSAI · {MODEL.upper()} · GROQ · '
    f'T={TEMP_REACT}/top_p={TOP_P_REACT} → T={TEMP_FINAL}/top_p={TOP_P_FINAL} · '
    f'top_k={TOP_K_CHUNKS} · {search_mode.upper()} SEARCH</div>',
    unsafe_allow_html=True,
)