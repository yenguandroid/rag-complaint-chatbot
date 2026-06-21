# Intelligent Complaint Analysis for Financial Services

A RAG-powered chatbot that transforms raw CFPB customer complaint data into actionable insights for CrediTrust Financial's internal teams.

## Overview

This project builds a Retrieval-Augmented Generation (RAG) system that allows product, support, and compliance teams to ask plain-English questions about customer complaints across four financial product categories:

- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/      # CI/CD pipeline
├── data/
│   ├── raw/                # Original CFPB dataset (not tracked)
│   └── processed/          # Cleaned and filtered data
├── vector_store/           # Persisted FAISS/ChromaDB index
├── notebooks/              # EDA and preprocessing notebooks
├── src/                    # Core pipeline modules
│   ├── data_processor.py   # Task 1: EDA & preprocessing
│   ├── embedder.py         # Task 2: Chunking & embedding
│   ├── rag_pipeline.py     # Task 3: RAG core logic
│   └── evaluator.py        # Task 3: Evaluation utilities
├── tests/                  # Unit tests
├── app.py                  # Task 4: Gradio chat interface
└── requirements.txt
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Tasks

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | EDA & Data Preprocessing | ✅ |
| Task 2 | Chunking, Embedding & Vector Store | ✅ |
| Task 3 | RAG Core Logic & Evaluation | ✅ |
| Task 4 | Interactive Chat Interface | ✅ |

## Running the App

```bash
python app.py
```

Then open the URL shown in your terminal (default: http://localhost:7860).

## Data

Download the CFPB complaint dataset and place it in `data/raw/`. The processed output is saved to `data/processed/filtered_complaints.csv`.

## Key Design Decisions

- **Embedding model**: `all-MiniLM-L6-v2` — fast, lightweight, strong semantic similarity performance
- **Chunk size**: 500 characters with 50-character overlap
- **Vector store**: ChromaDB (primary) with FAISS support
