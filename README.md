# LangChain Research – Local RAG Knowledge Base

A lightweight, modular **Retrieval-Augmented Generation (RAG)** system built with **LangChain**, **FAISS**, **HuggingFace embeddings**, and a **local Qwen model**.
This project lets you load documents, embed them, build a vector store, and chat with an LLM that retrieves relevant context from your custom knowledge base.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)]()

---

## ✨ Features

- Load text, Markdown, and PDF documents recursively from a folder
  → via custom loader and text splitter

- Generate embeddings using **HuggingFace sentence transformers**
  → `all-MiniLM-L6-v2`

- Build or load a **FAISS vector store**
  → with persistence for reuse

- Run a **local Qwen LLM** for text generation
  → loaded with Transformers / AutoModelForCausalLM

- Retrieval-augmented chat loop
  → retrieve top-k docs, construct a prompt, and generate an answer

---

## 🧱 Project Architecture

```
langchain-research/
│
├── embedder.py          # HuggingFace embeddings
├── llm.py               # Local Qwen model loader
├── loader.py            # Document loading & splitting
├── vector_store.py      # FAISS index build/load
├── rag_chain.py         # Retrieval + generation chain
└── main.py              # CLI chat app
```

---

## 📚 How It Works

### 1. Load & Split Documents

All `.txt`, `.md`, and `.pdf` files under `../data/docs` are processed and chunked into 500-token windows with overlap.

### 2. Build Embeddings

Embeddings come from:
`sentence-transformers/all-MiniLM-L6-v2`

### 3. Build FAISS Vector Store

A persistent FAISS index is created under `../data/vector_store`.

### 4. Load Local LLM

A local Qwen model is loaded from `../models/qwen3/`.

### 5. Build RAG Answer Function

The system retrieves top-k relevant chunks and feeds them into a generation pipeline.

### 6. Chat Loop

A terminal chat interface runs the retrieval-augmented assistant.

---

## 🚀 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/anto18671/langchain-research
cd langchain-research
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download/Place Local Qwen Model

Place your Qwen model folder here:

```
models/qwen3/
```

Example (if using HuggingFace):

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir models/qwen3
```

---

## ▶️ Running the RAG Chat System

```bash
python main.py
```

You should see:

```
Knowledge Base Chat Ready. Type 'exit' to quit.
```

Ask questions about anything inside your `../data/docs/` directory.

---

## 📁 Expected Directory Layout

```
project-root/
│
├── data/
│   ├── docs/               # Your input documents
│   └── vector_store/       # Auto-generated FAISS index
│
├── models/
│   └── qwen3/              # Your local Qwen model
```

---

## 🔧 Customization

- Adjust **chunk size or overlap** → `loader.py`
- Change embedding model → `embedder.py`
- Swap LLM model or pipeline settings → `llm.py` + `rag_chain.py`
- Modify retrieval `k` → `rag_chain.py`

---

## 📝 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.
