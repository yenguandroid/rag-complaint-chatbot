
"""
Task 4 — Interactive Chat Interface
CrediTrust Financial — RAG Complaint Chatbot
Framework : Gradio 6.x
"""

import os
import sys
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

sys.path.append('./src')

from rag_pipeline import (
    load_embeddings,
    load_vector_store,
    load_llm,
    build_rag_chain,
    query_rag,
)

# ── Load pipeline once at startup ─────────────────────────────────────────────
print("Initializing RAG pipeline...")
os.environ.setdefault("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
embeddings  = load_embeddings()
vectorstore = load_vector_store(embeddings)
llm         = load_llm()
rag_chain   = build_rag_chain(vectorstore, llm, k=5)
print("✅ Pipeline ready.")


# ── Core query function ────────────────────────────────────────────────────────
def answer_question(question: str, history: list):
    if not question.strip():
        return "", history, "<p style='color:gray;'>Please enter a question.</p>"

    result  = query_rag(rag_chain, question)
    answer  = result["answer"]
    sources = result["sources"]

    # Format sources as HTML cards
    sources_html = "<div style='display:flex; flex-direction:column; gap:12px;'>"
    for i, src in enumerate(sources[:5], 1):
        chunk_label = f"Chunk {src.get('chunk_index', 0)+1} of {src.get('total_chunks', 1)}"
        sources_html += f"""
        <div style='border:1px solid #3B82F6; border-left:4px solid #3B82F6;
                    border-radius:8px; padding:12px 16px; background:#EFF6FF;
                    font-family:Arial,sans-serif; margin-bottom:8px;'>
            <div style='font-weight:bold; color:#1E40AF; margin-bottom:8px;'>
                📄 Source {i}
                <span style='color:#6B7280; font-size:12px; font-weight:normal;'>
                    &nbsp;— {chunk_label}
                </span>
            </div>
            <div style='font-size:12px; color:#374151; margin-bottom:8px; line-height:1.8;'>
                🏷️ <b>Product:</b> {src.get('product_category','N/A')}<br/>
                🏢 <b>Company:</b> {src.get('company','N/A')}<br/>
                📅 <b>Date:</b> {src.get('date_received','N/A')} &nbsp;|&nbsp;
                📍 <b>State:</b> {src.get('state','N/A')}
            </div>
            <div style='background:white; border-radius:4px; padding:8px;
                        font-size:13px; color:#1F2937; line-height:1.5;
                        border:1px solid #DBEAFE;'>
                {src.get('text','')[:400]}...
            </div>
        </div>"""
    sources_html += "</div>"

    # Gradio 6.x dict format
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    return "", history, sources_html


def clear_all():
    return (
        "",    # clear input
        [],    # clear history
        "<p style='color:gray; font-style:italic; padding:20px;'>"
        "Sources will appear here after your first question.</p>"
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="CrediTrust Financial — Complaint Intelligence",
    css="""
        #header { text-align:center; padding:20px 0 10px 0; }
        #header h1 { color:#1E40AF; font-size:2em; margin-bottom:4px; }
        #header p  { color:#6B7280; font-size:0.95em; }
        footer { display:none !important; }
    """
) as demo:

    # Header
    gr.HTML("""
        <div id='header'>
            <h1>🏦 CrediTrust Financial</h1>
            <p>Complaint Intelligence Chatbot — powered by RAG + Llama3</p>
            <p style='font-size:0.85em; color:#9CA3AF;'>
                Ask questions about customer complaints across Credit Cards,
                Personal Loans, Savings Accounts &amp; Money Transfers
            </p>
        </div>
    """)

    gr.Markdown("---")

    with gr.Row():
        # Left: Chat
        with gr.Column(scale=3):
            gr.Markdown("### 💬 Conversation")

            chatbot = gr.Chatbot(
                label="",
                height=220,
                # bubble_full_width=False,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=creditrust"
                ),
                placeholder=(
                    "<div style='text-align:center; color:#9CA3AF; padding:40px;'>"
                    "<h3>👋 Welcome!</h3>"
                    "<p>Ask me anything about customer complaints.</p>"
                    "<p><b>Example questions:</b></p>"
                    "<p>• What are common credit card issues?</p>"
                    "<p>• How do customers describe money transfer problems?</p>"
                    "<p>• What complaints exist about savings accounts?</p>"
                    "</div>"
                ),
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about customer complaints...",
                    label="",
                    scale=5,
                    container=False,
                    lines=1,
                )
                ask_btn = gr.Button("Ask ➤", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Conversation", variant="stop")

            gr.Markdown("**💡 Try these example questions:**")
            with gr.Row():
                ex1 = gr.Button("Credit card issues",           size="sm")
                ex2 = gr.Button("Money transfer problems",      size="sm")
                ex3 = gr.Button("Savings account complaints",   size="sm")
                ex4 = gr.Button("Company responses",            size="sm")

        # Right: Sources
        with gr.Column(scale=2):
            gr.Markdown("### 📚 Retrieved Sources")
            gr.Markdown("*The complaint excerpts used to generate the answer above.*")
            sources_display = gr.HTML(
                value=(
                    "<p style='color:gray; font-style:italic; padding:20px;'>"
                    "Sources will appear here after your first question.</p>"
                )
            )

    gr.Markdown("---")
    gr.HTML("""
        <div style='text-align:center; color:#9CA3AF; font-size:0.8em; padding:8px;'>
            CrediTrust Financial Internal Tool &nbsp;|&nbsp;
            Powered by ChromaDB · all-MiniLM-L6-v2 · Llama3 via Groq &nbsp;|&nbsp;
            34,202 complaint chunks indexed
        </div>
    """)

    # State
    history_state = gr.State([])

    # ── Event wiring ───────────────────────────────────────────────────────────
    ask_btn.click(
        fn=answer_question,
        inputs=[question_box, history_state],
        outputs=[question_box, history_state, sources_display],
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[chatbot],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, history_state],
        outputs=[question_box, history_state, sources_display],
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[chatbot],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_box, history_state, sources_display],
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[chatbot],
    )

    # Example buttons
    ex1.click(fn=lambda: "What are the most common issues customers report with credit cards?", outputs=[question_box])
    ex2.click(fn=lambda: "What problems do customers face with money transfers?",               outputs=[question_box])
    ex3.click(fn=lambda: "What are the main complaints about checking or savings accounts?",    outputs=[question_box])
    ex4.click(fn=lambda: "How do companies typically respond to customer complaints?",          outputs=[question_box])


# ── Launch ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )