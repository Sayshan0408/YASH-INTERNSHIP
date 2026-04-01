## 🤖 AutoDev — AI-Powered Multi-Agent Software Development Pipeline
Turn a plain-English requirement into production-ready code, tests, and documentation — automatically.

## 📌 What is AutoDev?
AutoDev is a multi-agent AI pipeline that simulates a full software development team. You describe what you want to build — AutoDev's six specialized agents handle the rest: planning, architecture, coding, code review, testing, and documentation.
All powered by Groq's LLaMA 3.1 model — fast, free-tier friendly, and no OpenAI dependency.

## 🎬 Demo
Input:  "Build a REST API in FastAPI with SQLite for a todo app —
         CRUD endpoints, JWT auth, and pagination support."

Output: ✅ Project Plan
        ✅ System Architecture + Folder Structure
        ✅ Production-Ready Code (all files)
        ✅ Code Review Report
        ✅ pytest Test Suite
        ✅ README + API Docs
        📊 Per-Agent Accuracy Scores (0–100)


## ✨ Features

6 specialized agents — each expert in its own phase of development
Developer ↔ Reviewer loop — iterative code improvement (up to 3 rounds)
3-layer safety guardrails — input validation, output validation, PII redaction
Per-agent accuracy scoring — grades every agent's output from 0–100
Overall pipeline score — weighted grade across all 6 agents
Run history — all pipeline runs saved and searchable
Clean Streamlit UI — dark-themed, real-time progress updates
No OpenAI — runs entirely on Groq's free tier

## 🛡️ Safety Guardrails
AutoDev has a 3-layer safety system that runs on every request:
Layer 1 — Input Validation (before any agent runs)

Blocks prompt injection attempts
Blocks harmful/malicious content requests
Enforces input length limits (10–5000 chars)

Layer 2 — Output Validation (after each agent responds)

Detects dangerous code patterns
Flags real hardcoded secrets (30+ char random strings)
Skips false positives like password="admin" in example code

Layer 3 — PII Sanitization (final pass on all output)

Redacts SSNs, credit card numbers, emails, phone numbers

## 🌟 Star this repo if AutoDev saved you time!

Built with ❤️ using Groq + Streamlit + LLaMA 3.1
