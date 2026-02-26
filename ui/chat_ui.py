"""
/aiquery — RAG Chat Interface (Gradio Blocks).

Features:
  • Fresh Query      → clears memory, starts a new session.
  • Continue Previous → loads last session history into chatbot.
  • Real-time streaming feel (synchronous Gemini call, fast enough for UX).
  • Clean, professional theme.
"""

import gradio as gr

import database as db
from config import settings
from services.rag_service import answer_query


# ── Helpers ───────────────────────────────────────────────────────────────────

def _db_to_gradio(messages: list[dict]) -> list[dict]:
    """Convert DB messages → Gradio chatbot message format."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ── Event handlers ────────────────────────────────────────────────────────────

def on_mode_change(mode: str, session_state: dict):
    """
    Called when the user switches between Fresh Query / Continue.
    Returns (new session_state, updated chatbot history).
    """
    if mode == "Fresh Query":
        new_sid = db.new_session_id()
        return {"session_id": new_sid}, []

    # Continue — load latest session
    latest_sid = db.get_latest_session()
    if latest_sid:
        history = _db_to_gradio(db.get_conversation(latest_sid))
        return {"session_id": latest_sid}, history

    # No history yet → create fresh
    new_sid = db.new_session_id()
    return {"session_id": new_sid}, []


def on_submit(user_message: str, chat_history: list, session_state: dict):
    """
    Send user message → RAG pipeline → append answer to chat.
    Returns ("", updated_chat_history, updated_session_state).
    """
    user_message = (user_message or "").strip()
    if not user_message:
        return "", chat_history, session_state

    # Ensure we have a session
    if not session_state or "session_id" not in session_state:
        session_state = {"session_id": db.new_session_id()}

    sid = session_state["session_id"]

    # Add user message immediately
    chat_history = list(chat_history) + [{"role": "user", "content": user_message}]

    try:
        answer = answer_query(user_message, sid)
    except Exception as exc:
        answer = f"⚠️ An error occurred: {exc}"

    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history, session_state


# ── UI Builder ────────────────────────────────────────────────────────────────

CHAT_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

CHAT_CSS = """
#chat-header { text-align: center; padding: 10px 0 4px; }
#chat-header h1 { font-size: 1.6rem; margin-bottom: 2px; }
#chat-header p  { color: #6b7280; font-size: 0.9rem; margin: 0; }
#mode-row { display: flex; justify-content: center; padding: 4px 0; }
#chatbot { border-radius: 12px; }
#send-btn { min-width: 90px; }
"""


def create_chat_interface() -> gr.Blocks:
    with gr.Blocks(title="RAG Chat") as chat_app:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Column(elem_id="chat-header"):
            gr.HTML(
                f"""
                <h1>💬 RAG Chat Assistant</h1>
                <p>{settings.DOMAIN_DESCRIPTION}</p>
                """
            )

        # ── Mode selector ──────────────────────────────────────────────────────
        with gr.Row(elem_id="mode-row"):
            mode_radio = gr.Radio(
                choices=["Fresh Query", "Continue Previous Conversation"],
                value="Continue Previous Conversation",
                label="Query Mode",
                interactive=True,
            )

        # ── Chatbot ────────────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
            elem_id="chatbot",
            avatar_images=(None, None),
            placeholder=(
                "<div style='text-align:center;color:#9ca3af;padding:40px'>"
                "<p style='font-size:2rem'>🤖</p>"
                f"<p>Ask me anything about <strong>{settings.DOMAIN_DESCRIPTION}</strong></p>"
                "<p style='font-size:0.85rem'>Select a query mode above, then type your question.</p>"
                "</div>"
            ),
        )

        # ── Input row ──────────────────────────────────────────────────────────
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Type your question here…",
                show_label=False,
                scale=9,
                lines=1,
                max_lines=5,
                autofocus=True,
            )
            send_btn = gr.Button(
                "Send ➤",
                variant="primary",
                scale=1,
                elem_id="send-btn",
            )

        # ── State ──────────────────────────────────────────────────────────────
        session_state = gr.State(value={})

        # ── Wire events ────────────────────────────────────────────────────────
        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio, session_state],
            outputs=[session_state, chatbot],
        )

        send_btn.click(
            fn=on_submit,
            inputs=[msg_box, chatbot, session_state],
            outputs=[msg_box, chatbot, session_state],
        )

        msg_box.submit(
            fn=on_submit,
            inputs=[msg_box, chatbot, session_state],
            outputs=[msg_box, chatbot, session_state],
        )

        # Auto-load history on page open
        chat_app.load(
            fn=on_mode_change,
            inputs=[mode_radio, session_state],
            outputs=[session_state, chatbot],
        )

    return chat_app
