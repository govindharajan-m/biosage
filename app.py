
import gradio as gr
import logging
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

rag = RAGEngine()

WELCOME_HTML = """
<div id="welcome-hero">
  <div class="hero-icon">🧬</div>
  <h1 class="hero-title">BioSage</h1>
  <p class="hero-sub">Genetic variant &amp; pathogenicity specialist</p>
</div>
"""

EXAMPLES = [
    "What mutations in BRCA1 are pathogenic?",
    "What causes cystic fibrosis at the molecular level?",
    "List pathogenic variants in the CFTR gene.",
    "What is the inheritance pattern of Huntington disease?",
    "Which genes are associated with hypertrophic cardiomyopathy?",
]

def predict(message, history):
    if not message.strip():
        return history
    logger.info(f"Query: {message}")
    output = rag.answer(message)
    answer = output["answer"]
    sources = output["sources"]

    if sources:
        src_block = "\n\n---\n**Sources:**\n"
        for i, s in enumerate(sources[:3], 1):
            db = s['metadata'].get('source_db', 'Database')
            sid = s['id']
            src_block += f"- [{i}] **{db}** — {sid}\n"
        full = f"{answer}{src_block}"
    else:
        full = answer

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full})
    return history

def use_example(example, history):
    return predict(example, history)

APP_HEAD = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
    height: 100%;
    background: #1a1a1a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e5e5e5 !important;
}

/* Kill all Gradio wrappers */
.gradio-container, .gradio-container > *, .main, .wrap, .contain, footer {
    background: #1a1a1a !important;
    color: #e5e5e5 !important;
    border: none !important;
    box-shadow: none !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

footer { display: none !important; }

/* App shell */
#app-shell {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 0 16px 32px;
}

/* Welcome hero */
#welcome-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 72px 0 40px;
    gap: 8px;
    transition: all 0.3s ease;
}

.hero-icon {
    font-size: 2rem;
    margin-bottom: 4px;
}

.hero-title {
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #ececec !important;
    letter-spacing: -0.04em !important;
    text-align: center !important;
    background: none !important;
    -webkit-text-fill-color: #ececec !important;
}

.hero-sub {
    font-size: 0.9rem !important;
    color: #888 !important;
    font-weight: 400 !important;
    text-align: center !important;
}

/* Chatbot area */
#chatbot-wrap {
    width: 100%;
    max-width: 760px;
}

#chatbot-wrap .chatbot {
    background: transparent !important;
    border: none !important;
}

/* Message bubbles */
#chatbot-wrap .message-wrap { padding: 4px 0 !important; }

#chatbot-wrap [data-testid="user"] .bubble-wrap,
#chatbot-wrap [data-testid="user"] .message {
    background: #2f2f2f !important;
    color: #ececec !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 16px !important;
    max-width: 75% !important;
    margin-left: auto !important;
    border: none !important;
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
}

#chatbot-wrap [data-testid="bot"] .bubble-wrap,
#chatbot-wrap [data-testid="bot"] .message {
    background: transparent !important;
    color: #d4d4d4 !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 16px !important;
    max-width: 85% !important;
    border: none !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

/* Input section */
#input-section {
    width: 100%;
    max-width: 760px;
    margin-top: 8px;
}

#input-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: #2f2f2f;
    border: 1px solid #3d3d3d;
    border-radius: 14px;
    padding: 10px 12px;
    transition: border-color 0.15s;
}

#input-row:focus-within {
    border-color: #555 !important;
}

#msg-input textarea, #msg-input input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    color: #e5e5e5 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    resize: none !important;
    font-family: inherit !important;
    padding: 0 !important;
    box-shadow: none !important;
}

#msg-input textarea::placeholder {
    color: #666 !important;
}

#msg-input .wrap-inner, #msg-input label {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}

#send-btn {
    background: #ececec !important;
    color: #1a1a1a !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    min-width: 80px !important;
    transition: background 0.15s !important;
    box-shadow: none !important;
}

#send-btn:hover {
    background: #d4d4d4 !important;
}

/* Example chips */
#examples-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
    justify-content: center;
}

.example-chip button {
    background: #242424 !important;
    border: 1px solid #383838 !important;
    color: #aaa !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 0.78rem !important;
    font-family: inherit !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
    box-shadow: none !important;
}

.example-chip button:hover {
    background: #2f2f2f !important;
    border-color: #555 !important;
    color: #e5e5e5 !important;
    transform: none !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3d3d3d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
"""

with gr.Blocks(title="BioSage") as demo:

    with gr.Column(elem_id="app-shell"):

        gr.HTML(WELCOME_HTML)

        with gr.Column(elem_id="chatbot-wrap"):
            chatbot = gr.Chatbot(
                show_label=False,
                container=False,
                height=480,
                elem_classes=["chatbot"],
            )

        with gr.Column(elem_id="input-section"):
            with gr.Row(elem_id="input-row"):
                msg = gr.Textbox(
                    placeholder="Ask about a disease, gene, or variant...",
                    show_label=False,
                    container=False,
                    elem_id="msg-input",
                    scale=9,
                    autofocus=True,
                )
                send = gr.Button("Send", elem_id="send-btn", scale=1)

            with gr.Row(elem_id="examples-row"):
                for ex in EXAMPLES:
                    btn = gr.Button(ex, elem_classes=["example-chip"], size="sm")
                    btn.click(fn=lambda h, e=ex: predict(e, h), inputs=[chatbot], outputs=[chatbot])

        msg.submit(predict, [msg, chatbot], [chatbot])
        msg.submit(lambda: "", None, msg)
        send.click(predict, [msg, chatbot], [chatbot])
        send.click(lambda: "", None, msg)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        head=APP_HEAD,
    )
