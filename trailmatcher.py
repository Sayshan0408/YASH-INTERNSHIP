import json
import re
import requests

API_KEY = "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"
MODEL = "llama-3.3-70b-versatile"
URL = "https://api.groq.com/openai/v1/chat/completions"

TRIALS = [
    {"id": 0, "title": "DESTINY-Breast06", "info": "HER2+, Trastuzumab-resistant, Stage 2-4, Age 18-65, Mumbai/Delhi sites."},
    {"id": 1, "title": "HER2CLIMB-02",     "info": "HER2+, Trastuzumab-NAIVE patients only, Stage 1-3, US sites only."},
    {"id": 2, "title": "KEYNOTE-522",      "info": "Triple-negative breast cancer, HER2-negative only, Age 18-75, Global sites."},
    {"id": 3, "title": "SOPHIA Trial",     "info": "HER2+, progressed on prior Trastuzumab, Stage 3-4, Age 30-70, India sites open."},
    {"id": 4, "title": "NALA Trial",       "info": "HER2+ metastatic, Trastuzumab-resistant accepted, Age 18-50, Chennai/Bangalore."},
    {"id": 5, "title": "PHILA Trial",      "info": "HER2+, Trastuzumab-resistant, Stage 2-4, Age 18-65, Mumbai site open."},
]

def llm(prompt, temp=0.0):
    res = requests.post(URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": temp, "max_tokens": 512}, timeout=30)
    data = res.json()
    if "choices" not in data:
        print("API Error:", data)
        return ""
    return data["choices"][0]["message"]["content"].strip()

def hyde(patient):
    hypo = llm(f"Write 2 sentences describing the ideal clinical trial for:\n{patient}\nOnly the passage.", temp=0.4)
    return f"{patient}\n{hypo}"

def multi_query(patient):
    raw = llm(f"Give 3 different medical search queries to find trials for:\n{patient}\nReturn ONLY a JSON array of 3 strings.")
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not m:
        return [patient, patient, patient, patient]
    return [patient] + json.loads(m.group())[:3]

def score(trial, query):
    words = set(re.findall(r'\w+', query.lower()))
    text = (trial["title"] + " " + trial["info"]).lower()
    return sum(1 for w in words if w in text) / max(len(words), 1)

def retrieve(query, top_k=6):
    return [t["id"] for t in sorted(TRIALS, key=lambda t: -score(t, query))[:top_k]]

def rrf(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, tid in enumerate(ranked):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: -scores[x])

def listwise_rerank(candidate_ids, patient):
    candidates = [t for t in TRIALS if t["id"] in candidate_ids]
    trials_text = "\n".join(f"[{i}] {t['title']}: {t['info']}" for i, t in enumerate(candidates))
    raw = llm(f"Patient: {patient}\n\nTrials:\n{trials_text}\n\nRank MOST to LEAST suitable.\nConsider eligibility, location, resistance, disqualifying clauses.\nReturn ONLY a JSON array of indices e.g. [2,0,3,1,4,5]")
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not m:
        return candidates
    return [candidates[i] for i in json.loads(m.group()) if 0 <= i < len(candidates)]

def recommend(top3, patient):
    trials_text = "\n".join(f"- {t['title']}: {t['info']}" for t in top3)
    return llm(f"Patient: {patient}\n\nTop 3 trials:\n{trials_text}\n\nWrite a 4-sentence oncologist recommendation.", temp=0.5)

def match(patient):
    print("1. HyDE: Expanding query...")
    expanded = hyde(patient)
    print("2. Multi-Query: Generating 3 variants...")
    queries = multi_query(patient)
    print("3. Retrieve + RRF: Merging ranked lists...")
    candidates = rrf([retrieve(q) for q in [expanded] + queries])[:6]
    print("4. Listwise Reranking: LLM ranks all candidates...")
    reranked = listwise_rerank(candidates, patient)
    print("5. Generating recommendation...")
    answer = recommend(reranked[:3], patient)
    print("\nTop Matches:", [t["title"] for t in reranked[:3]])
    print("\nRecommendation:\n", answer)

match("""
Age 42, Female. Stage 3 HER2+ Breast Cancer.
Prior Trastuzumab (8 months) — now resistant.
Location: Chennai, India. ECOG 1.
""")