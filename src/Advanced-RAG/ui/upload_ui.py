"""
/documentation — Document Upload Interface (Gradio Blocks).

What it shows:
  • Drag-and-drop file uploader (PDF, DOCX, TXT, MD).
  • Single "Process Document" button.
  • Clean status output — no chunking stats, no memory metrics.
  • On success : "The document has been successfully uploaded and processed
                  into X chunks using sophisticated chunking strategy."
  • On failure : domain rejection message or parse error.
"""

import os
import shutil
import tempfile

import gradio as gr

from config import settings
from services.rag_service import ingest_document


# ── Helpers ───────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

STATUS_SUCCESS_CSS = "color: #16a34a; font-weight: 600;"
STATUS_ERROR_CSS   = "color: #dc2626; font-weight: 600;"


# ── Event handler ─────────────────────────────────────────────────────────────

def on_upload(file_obj):
    """
    Called when the user clicks 'Process Document'.

    Args:
        file_obj: Gradio file object (has .name with the temp path).

    Returns:
        gr.HTML status block.
    """
    if file_obj is None:
        return gr.HTML(
            "<p style='color:#f59e0b;font-weight:600;'>⚠️ Please select a file first.</p>"
        )

    file_path = file_obj.name
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return gr.HTML(
            f"<p style='{STATUS_ERROR_CSS}'>❌ Unsupported file type '{ext}'. "
            f"Allowed: PDF, DOCX, TXT, MD.</p>"
        )

    # Copy to a stable temp location so Gradio's temp cleanup doesn't race us
    tmp_dir = tempfile.mkdtemp()
    stable_path = os.path.join(tmp_dir, os.path.basename(file_path))
    shutil.copy2(file_path, stable_path)

    try:
        success, message = ingest_document(stable_path)
    except Exception as exc:
        success = False
        message = f"Unexpected error during processing: {exc}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if success:
        return gr.HTML(f"<p style='{STATUS_SUCCESS_CSS}'>✅ {message}</p>")
    else:
        return gr.HTML(f"<p style='{STATUS_ERROR_CSS}'>❌ {message}</p>")


# ── UI Builder ────────────────────────────────────────────────────────────────

UPLOAD_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

UPLOAD_CSS = """
#upload-header { text-align: center; padding: 10px 0 4px; }
#upload-header h1 { font-size: 1.6rem; margin-bottom: 2px; }
#upload-header p  { color: #6b7280; font-size: 0.9rem; margin: 0; }
#upload-box { border-radius: 12px; }
#process-btn { margin-top: 8px; }
#status-box { min-height: 48px; padding: 10px 14px;
              background: #f8fafc; border-radius: 8px;
              border: 1px solid #e2e8f0; margin-top: 8px; }
"""


def create_upload_interface() -> gr.Blocks:
    with gr.Blocks(title="Document Upload") as upload_app:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Column(elem_id="upload-header"):
            gr.HTML(
                f"""
                <h1>📄 Document Ingestion</h1>
                <p>Upload documents to the <strong>{settings.DOMAIN_DESCRIPTION}</strong></p>
                """
            )

        # ── Domain info ───────────────────────────────────────────────────────
        gr.Markdown(
            f"> **Accepted domain:** {settings.DOMAIN}  \n"
            f"> Documents outside this domain will be automatically rejected."
        )

        # ── File uploader ─────────────────────────────────────────────────────
        file_input = gr.File(
            label="Drop your document here or click to browse",
            file_types=[".pdf", ".docx", ".txt", ".md"],
            file_count="single",
            elem_id="upload-box",
            height=200,
        )

        # ── Action button ─────────────────────────────────────────────────────
        process_btn = gr.Button(
            "⚙️  Process Document",
            variant="primary",
            size="lg",
            elem_id="process-btn",
        )

        # ── Status output ─────────────────────────────────────────────────────
        status_output = gr.HTML(
            value="<p style='color:#9ca3af;'>Status will appear here after processing.</p>",
            elem_id="status-box",
        )

        # ── Wire event ────────────────────────────────────────────────────────
        process_btn.click(
            fn=on_upload,
            inputs=[file_input],
            outputs=[status_output],
        )

    return upload_app
