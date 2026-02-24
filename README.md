An interactive Streamlit-based chatbot for querying and summarizing PDF documents (like RFPs).
Powered by Qwen2.5-7B, LangChain, and FAISS, this app lets you upload a PDF, process it into searchable chunks, and ask natural language questions with concise summaries.

Features:
1-PDF Upload & Processing: Upload RFPs or other documents, automatically chunked and indexed with FAISS.

2-Conversational Q&A: Ask questions in natural language, get answers grounded in the document.

3-Summarization: Generate concise bullet-point summaries of answers.

4-Modern UI: Custom dark-themed Streamlit interface with styled chat bubbles, summaries, and source cards.

5-Tech Stack Transparency: Sidebar shows the underlying components used 

Tech Stack:
1-Streamlit — UI framework

2-LangChain — Orchestration framework

3-PyPDFLoader — PDF parsing

4-RecursiveCharacterTextSplitter — Chunking strategy

5-all-MiniLM-L6-v2 — Embedding model

6-FAISS — Vector database

7-Qwen2.5-7B-Instruct — LLM for answering and summarization

8-HuggingFace Hub — Model hosting
