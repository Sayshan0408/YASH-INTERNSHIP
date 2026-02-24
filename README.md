## 📄 RFP Chatbot
An interactive Streamlit-based chatbot for querying and summarizing PDF documents (like RFPs).
Powered by Qwen2.5-7B, LangChain, and FAISS, this app lets you upload a PDF, process it into searchable chunks, and ask natural language questions with concise summaries.

## 🗂️ Project Structure

File / Folder	Role  
app.py	Streamlit UI
ingest.py	PDF ingestion → FAISS index
bot.py	Retrieval, LLM, summarization
.env	HuggingFace API token
faiss_index/	Auto-created FAISS index
uploads/	Uploaded PDFs

## 🚀 Features

PDF Upload & Processing: Upload RFPs or other documents, automatically chunked and indexed with FAISS.

Conversational Q&A: Ask questions in natural language, get answers grounded in the document.

Summarization: Generate concise bullet-point summaries of answers.

Modern UI: Custom dark-themed Streamlit interface with styled chat bubbles, summaries, and source cards.

Tech Stack Transparency: Sidebar shows the underlying components used.

## Architecture

User → Streamlit UI → Upload PDF
       ↓
   Ingestion Pipeline
   (PyPDFLoader → Chunking → Embeddings → FAISS)
       ↓
   Retrieval Pipeline
   (User Question → Embedding → FAISS Search → Qwen2.5-7B)
       ↓
   Answer + Sources → Streamlit UI
       ↓
   (Optional) Summarization Pipeline → Bullet Summary
