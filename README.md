## ✈️ Aviation AI Briefing Checker

Verify AI-generated aviation summaries against original pilot reports using 5 NLP metrics.

A safety-critical tool that checks whether an AI-generated briefing accurately represents the original aviation incident report — catching wrong numbers, missing facts, and meaning drift before they cause harm.

## 🚨 The Problem
Aviation reports contain critical data: flight levels, passenger counts, injury severity, timestamps. When AI summarizes these reports, it can silently change numbers, drop key facts, or drift in meaning — and a human reader may not catch it.
This tool catches those mistakes automatically.

## 🛠️ What It Does
Paste in two texts — the original pilot/incident report and the AI-generated summary — and the checker runs 5 independent NLP checks, combines them into a weighted Safety Score (0–100), and returns a clear verdict:

##  Score            Verdict
## 85–100✅   CLEARED — Safe to use
## 60–84⚠️    REVIEW — Needs human check
## 0–59🚨     GROUNDED — Do not use**

## 📊 The 5 Metrics Explained
## 1. Exact Match — Weight: 35%
Extracts every number from both texts using regex and checks how many from the original appear in the summary.

Why it matters: In aviation, "3 passengers injured" vs "2 passengers injured" is the difference between accurate and dangerous.

## 2. ROUGE-L — Weight: 20%
Measures recall — how much of the original content the AI actually covered. Uses longest common subsequence matching with stemming.

Best at catching: Things the AI forgot to mention entirely.

## 3. BLEU — Weight: 10%
Checks whether the AI used similar phrases (word pairs and triplets) to the original.

Best at catching: Imprecise rewording that changes specific meaning.

## 4. METEOR — Weight: 10%
Like BLEU, but smarter — uses WordNet to recognise synonyms, so "aircraft" and "plane" don't count as wrong.

## 5. BERTScore — Weight: 25%
Converts every word into a 768-dimensional meaning vector using DistilBERT, then compares vectors using cosine similarity. Two sentences can share zero words and still score high if they mean the same thing.

Best at catching: Semantic drift — where the AI changed the overall meaning without changing obvious words.
