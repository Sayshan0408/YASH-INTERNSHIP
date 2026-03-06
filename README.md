## 🏥 ClinicalMatch AI
AI-powered clinical trial matching using HyDE, Multi-Query RRF, and Listwise LLM Reranking — powered by Groq & Llama 3.3 70B.

## 🧬 What It Does
ClinicalMatch AI takes a patient's clinical profile (age, gender, diagnosis, prior treatment, location, ECOG score) and intelligently matches them to the most suitable clinical trials using a multi-stage RAG pipeline.
No hallucinated eligibility. No irrelevant matches. Just ranked, explainable recommendations.

## 🚀 Pipeline Overview

## Patient Profile
      │
      ▼
① HyDE  ──────────────── Hypothetical Document Expansion
      │
      ▼
② Multi-Query ─────────── 3 LLM-generated search variants
      │
      ▼
③ Retrieve + RRF ──────── Keyword retrieval + Reciprocal Rank Fusion
      │
      ▼
④ Listwise Rerank ─────── LLM ranks all 6 candidates holistically
      │
      ▼
⑤ Recommendation ──────── 4-sentence oncologist-style summary

## 🏥 Supported Disease Categories
CategoryExample Trials Included🎗 Breast CancerDESTINY-Breast06, HER2CLIMB-02, KEYNOTE-522, SOPHIA, NALA, PHILA🫁 Lung CancerFLAURA2, MARIPOSA, PAPILLON, KEYNOTE-789, CHRYSALIS-2, DESTINY-Lung02🔴 Colorectal CancerKRYSTAL-10, PARADIGM, SUNLIGHT, MOUNTAINEER-03, CodeBreaK 300, FRESCO-2🩸 LeukemiaVIALE-A, AGILE, BEAT-AML, LACEWING, MORPHO, SEQUOIA🔵 Prostate CancerPROfound, VISION, TRITON3, MAGNITUDE, PROPEL, TALAPRO-2🟣 Ovarian CancerSOLO-3, MIRASOL, SORAYA, PRIMA, DUO-O, ATALANTE

