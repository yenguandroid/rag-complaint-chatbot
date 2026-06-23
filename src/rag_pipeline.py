"""
Task 3 — RAG Core Pipeline
CrediTrust Financial Complaint Chatbot
LLM    : Llama3 via Groq API (free, fast, no local download)
Store  : ChromaDB (built in Task 2)
"""

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_chroma                   import Chroma
from langchain_huggingface              import HuggingFaceEmbeddings
from langchain_core.prompts             import PromptTemplate
from langchain_core.runnables           import RunnablePassthrough
from langchain_core.output_parsers      import StrOutputParser
from langchain_groq                     import ChatGroq

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # <-- FIX HERE
CHROMA_DIR = str(BASE_DIR / "vector_store" / "chroma")
COLLECTION_NAME = "cfpb_complaints"
TOP_K           = 5
GROQ_MODEL = "llama-3.1-8b-instant"   # ✅ current, fast, free

# ── Prompt Template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Answer the question below using ONLY the complaint excerpts provided in the context.
If the context does not contain enough information, say: "I don't have enough information in the retrieved complaints to answer this."
Do NOT make up any information outside the context.

Context:
{context}

Question: {question}

Answer:"""


# ── 1. Load Embedding Model ────────────────────────────────────────────────────
def load_embeddings():
    """Load the same embedding model used in Task 2."""
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded.")
    return embeddings


# ── 2. Load Vector Store ───────────────────────────────────────────────────────
def load_vector_store(embeddings, persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME):
    """Load the pre-built ChromaDB vector store from Task 2."""
    print(f"Loading ChromaDB from: {persist_dir} ...")

    print("Persist dir:", persist_dir)
    import os

    print("Absolute path:", os.path.abspath(persist_dir))

    print("Collection:", collection_name)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    count = vectorstore._collection.count()
    print(f"✅ Vector store loaded: {count:,} chunks available.")
    return vectorstore


# ── 3. Load LLM (Groq — free cloud API) ───────────────────────────────────────
def load_llm(groq_api_key: str = None, model_name: str = GROQ_MODEL):
    """
    Load Llama3 via Groq's free API.
    No local download — responses in < 2 seconds.

    Args:
        groq_api_key : your Groq API key from console.groq.com
        model_name   : Groq model to use (default: llama3-8b-8192)
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key required. Get one free at https://console.groq.com\n"
            "Then pass it as: load_llm(groq_api_key='your_key_here')\n"
            "Or set: os.environ['GROQ_API_KEY'] = 'your_key_here'"
        )

    print(f"Loading LLM: {model_name} via Groq API ...")
    llm = ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=512,
    )
    print(f"✅ LLM loaded ({model_name} via Groq — no local download needed).")
    return llm


# ── 4. Build RAG Chain ─────────────────────────────────────────────────────────
def build_rag_chain(vectorstore, llm, k=TOP_K):
    """Build RAG pipeline using LangChain LCEL."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Complaint {i+1} | {doc.metadata.get('product_category','N/A')} | "
            f"{doc.metadata.get('company','N/A')} | {doc.metadata.get('date_received','N/A')}]\n"
            f"{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    chain = (
        {
            "context"  : retriever | format_docs,
            "question" : RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"✅ RAG chain ready (top-k={k}, model={GROQ_MODEL}).")
    return {"chain": chain, "retriever": retriever}


# ── 5. Query Function ──────────────────────────────────────────────────────────
def query_rag(rag_chain, question: str) -> dict:
    """Run a question through the full RAG pipeline."""
    answer      = rag_chain["chain"].invoke(question)
    source_docs = rag_chain["retriever"].invoke(question)

    sources = []
    for doc in source_docs:
        sources.append({
            "text"             : doc.page_content[:300] + "..."
                                 if len(doc.page_content) > 300 else doc.page_content,
            "product_category" : doc.metadata.get("product_category", "N/A"),
            "company"          : doc.metadata.get("company", "N/A"),
            "date_received"    : doc.metadata.get("date_received", "N/A"),
            "complaint_id"     : doc.metadata.get("complaint_id", "N/A"),
            "state"            : doc.metadata.get("state", "N/A"),
            "chunk_index"      : doc.metadata.get("chunk_index", 0),
            "total_chunks"     : doc.metadata.get("total_chunks", 1),
        })

    return {
        "question" : question,
        "answer"   : answer.strip(),
        "sources"  : sources
    }


# ── 6. Pretty Printer ──────────────────────────────────────────────────────────
def print_rag_result(result: dict, show_sources: int = 2):
    """Print a formatted RAG result to console."""
    print("\n" + "="*70)
    print(f"QUESTION:\n  {result['question']}")
    print("-"*70)
    print(f"ANSWER:\n  {result['answer']}")
    print("-"*70)
    print(f"RETRIEVED SOURCES (showing {show_sources} of {len(result['sources'])}):")
    for i, src in enumerate(result["sources"][:show_sources], 1):
        print(f"\n  Source {i}:")
        print(f"    Product  : {src['product_category']}")
        print(f"    Company  : {src['company']}")
        print(f"    Date     : {src['date_received']}")
        print(f"    State    : {src['state']}")
        print(f"    Chunk    : {src['chunk_index']+1}/{src['total_chunks']}")
        print(f"    Excerpt  : {src['text'][:200]}...")
    print("="*70 + "\n")