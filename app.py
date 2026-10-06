"""Streamlit interface for syllabus-specific doubt resolution."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from src.chunking import split_documents
from src.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K,
    FEEDBACK_FILE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_MODEL,
    UPLOADS_DIR,
)
from src.embeddings import EmbeddingModel
from src.ingestion import load_pdf
from src.llm import Provider
from src.rag_pipeline import PipelineResult, RAGPipeline
from src.retriever import Retriever
from src.utils import get_logger
from src.vector_store import VectorStore

logger = get_logger(__name__)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Create and cache the persistent retrieval pipeline."""
    store = VectorStore()
    return RAGPipeline(Retriever(store, EmbeddingModel()))


def _save_upload(upload: object) -> tuple[Path, str]:
    """Persist a Streamlit-uploaded PDF and return its path and SHA-256 digest."""
    name = Path(str(getattr(upload, "name", "upload.pdf"))).name
    if Path(name).suffix.casefold() != ".pdf":
        raise ValueError(f"{name} is not a PDF.")
    content = getattr(upload, "getvalue")()
    digest = sha256(content).hexdigest()
    path = UPLOADS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path, digest


def _show_sources(result: dict[str, object]) -> None:
    """Render retrieved source passages and their similarity scores."""
    sources = result.get("sources", [])
    if not sources:
        return
    with st.expander("Retrieved syllabus passages"):
        for index, source in enumerate(sources, start=1):
            metadata = source["metadata"]
            st.markdown(
                f"**{index}. {metadata.get('source_file', 'Unknown file')}, "
                f"p.{metadata.get('page', '?')}** — "
                f"similarity {float(source['score']):.3f}"
            )
            st.caption(str(source["text"]))


def _record_feedback(
    message_id: str, question: str, answer: str, vote: str, provider: str
) -> None:
    """Append one feedback vote to the persistent CSV file."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "message_id": message_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "answer": answer,
                "feedback": vote,
                "provider": provider,
            }
        ]
    )
    row.to_csv(
        FEEDBACK_FILE,
        mode="a",
        header=not FEEDBACK_FILE.exists(),
        index=False,
        encoding="utf-8",
    )


def _store_result(result: PipelineResult) -> dict[str, object]:
    """Convert a pipeline result to session-state-friendly message metadata."""
    return {
        "sources": [
            {
                "text": chunk.text,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
            for chunk in result.chunks
        ],
        "provider": result.provider or "Not used",
        "fallback_message": result.fallback_message,
        "message_id": str(uuid.uuid4()),
    }


def _render_assistant_message(message: dict[str, object]) -> None:
    """Display an assistant answer, source expander, and feedback controls."""
    with st.chat_message("assistant"):
        st.markdown(str(message.get("content", "")))
        fallback_message = message.get("fallback_message")
        if fallback_message:
            st.info(str(fallback_message))
        _show_sources(message)
        _render_feedback(message)


def _render_feedback(message: dict[str, object]) -> None:
    """Render helpful/not-helpful buttons and persist a selected vote."""
    message_id = str(message.get("message_id", ""))
    if not message_id or not message.get("question"):
        return
    feedback_cols = st.columns(2)
    for column, vote, label in (
        (feedback_cols[0], "up", "👍 Helpful"),
        (feedback_cols[1], "down", "👎 Not helpful"),
    ):
        if column.button(label, key=f"feedback-{message_id}-{vote}"):
            try:
                _record_feedback(
                    message_id,
                    str(message["question"]),
                    str(message["content"]),
                    vote,
                    str(message.get("provider", "Not used")),
                )
                st.toast("Feedback saved. Thank you!")
            except OSError:
                logger.exception("Could not save feedback")
                st.error("Feedback could not be saved. Please try again.")


def _index_uploads(
    uploads: list[object], subject: str, unit: str, store: VectorStore
) -> None:
    """Save, deduplicate, parse, embed, and index selected PDF uploads."""
    if not uploads:
        st.warning("Select one or more PDF files before indexing.")
        return
    progress = st.progress(0, text="Preparing documents...")
    successes = 0
    failures: list[str] = []
    for index, upload in enumerate(uploads):
        filename = str(getattr(upload, "name", "uploaded PDF"))
        try:
            path, digest = _save_upload(upload)
            if store.is_file_indexed(path.name, digest):
                st.info(f"{path.name} is already indexed and has not changed.")
            else:
                pages = load_pdf(path, subject, unit)
                if not pages:
                    raise ValueError(
                        f"No selectable text was found in {path.name}; "
                        "scanned PDFs require OCR before indexing."
                    )
                chunks = split_documents(pages)
                embeddings = EmbeddingModel().encode(
                    [chunk.page_content for chunk in chunks]
                ).tolist()
                store.add_documents(chunks, embeddings, digest)
                successes += 1
            progress.progress(
                (index + 1) / len(uploads),
                text=f"Processed {index + 1} of {len(uploads)} file(s)",
            )
        except (OSError, ValueError, RuntimeError) as error:
            logger.exception("Indexing failed for %s", filename)
            failures.append(f"{filename}: {error}")
        except Exception:
            logger.exception("Unexpected indexing failure for %s", filename)
            failures.append(
                f"{filename}: processing failed. Check logs/app.log for details."
            )
    progress.empty()
    if successes:
        st.success(f"Indexed {successes} new or updated PDF file(s).")
    if not successes and not failures:
        st.info("No new documents needed indexing.")
    for failure in failures:
        st.error(failure)


def _render_history() -> None:
    """Render conversation history with source expanders and feedback controls."""
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            _render_assistant_message(message)
        else:
            with st.chat_message("user"):
                st.markdown(message["content"])


def _answer_question(
    pipeline: RAGPipeline,
    question: str,
    provider: Provider,
    model: str,
    top_k: int,
    temperature: float,
    subject: str | None,
    unit: str | None,
) -> None:
    """Run one turn, stream its rendered chunks, and persist the turn in session."""
    previous_history = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            result = pipeline.answer(
                question=question,
                history=previous_history,
                provider=provider,
                model=model,
                top_k=top_k,
                temperature=temperature,
                subject=subject,
                unit=unit,
            )
            metadata = _store_result(result)
            if result.fallback_message:
                st.info(result.fallback_message)
            if result.stream_chunks:
                shown_answer = st.write_stream(iter(result.stream_chunks))
            else:
                shown_answer = result.answer
                placeholder.markdown(shown_answer)
            assistant_message: dict[str, object] = {
                "role": "assistant",
                "content": str(shown_answer),
                "question": question,
                **metadata,
            }
            _show_sources(assistant_message)
            _render_feedback(assistant_message)
            st.session_state.messages.append(assistant_message)
            if result.provider:
                st.caption(f"Answered with {result.provider}.")
        except Exception:
            logger.exception("Question answering failed")
            error_message = (
                "I couldn't generate a response right now. Check that Ollama is "
                "running, the selected model is available, and Gemini has a valid "
                "API key if selected. Please try again."
            )
            placeholder.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "question": question,
                    "sources": [],
                    "provider": "Not used",
                    "message_id": str(uuid.uuid4()),
                }
            )


def _unit_options(files: list[dict[str, str | int]]) -> tuple[list[str], list[str]]:
    """Return sorted subject and unit names represented in the indexed files."""
    subjects = sorted(
        {str(item["subject"]) for item in files if item.get("subject")},
        key=str.casefold,
    )
    units = sorted(
        {str(item["unit"]) for item in files if item.get("unit")},
        key=str.casefold,
    )
    return subjects, units


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Syllabus Web Tutor",
        page_icon="📘",
        layout="wide",
    )
    st.title("Syllabus-Specific Web Tutor")
    st.caption("Ask questions grounded only in your uploaded syllabus material.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        pipeline = get_pipeline()
        store = pipeline.retriever.vector_store
        indexed_files = store.list_indexed_files()
    except Exception:
        logger.exception("Could not initialize the tutor's vector store")
        st.error(
            "The document index could not be opened. Check the data/chroma_db "
            "folder and logs/app.log, then reload the app."
        )
        return

    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader(
            "Upload syllabus PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        subject_input = st.text_input("Subject for uploaded files")
        unit_input = st.text_input("Unit for uploaded files")
        if st.button("Index documents", type="primary", use_container_width=True):
            if not subject_input.strip() or not unit_input.strip():
                st.warning("Enter both a subject and unit before indexing.")
            else:
                _index_uploads(
                    list(uploads or []),
                    subject_input.strip(),
                    unit_input.strip(),
                    store,
                )
                indexed_files = store.list_indexed_files()

        st.subheader("Indexed files")
        if indexed_files:
            for item in indexed_files:
                file_cols = st.columns([4, 1])
                file_cols[0].caption(
                    f"{item['source_file']} · {item['subject']} / {item['unit']} "
                    f"· {item['chunks']} chunks"
                )
                if file_cols[1].button(
                    "Delete",
                    key=f"delete-{item['source_file']}",
                    help=f"Remove {item['source_file']} from the index",
                ):
                    store.delete_file(str(item["source_file"]))
                    st.rerun()
        else:
            st.caption("No files have been indexed.")

        st.divider()
        provider = st.selectbox(
            "Answer provider",
            ["Ollama local", "Gemini cloud"],
            help="Gemini sends prompts and retrieved syllabus text to Google's service.",
        )
        provider_choice: Provider = (
            "Ollama local" if provider == "Ollama local" else "Gemini cloud"
        )
        configured_model = (
            OLLAMA_MODEL or DEFAULT_OLLAMA_MODEL
            if provider_choice == "Ollama local"
            else GEMINI_MODEL or DEFAULT_GEMINI_MODEL
        )
        model_options = list(
            dict.fromkeys(
                [
                    configured_model,
                    *(
                        ["llama3.2", "llama3.1", "mistral"]
                        if provider_choice == "Ollama local"
                        else ["gemini-2.0-flash", "gemini-2.5-flash"]
                    ),
                ]
            )
        )
        model = st.selectbox("Model", model_options)
        if provider_choice == "Gemini cloud" and not GEMINI_API_KEY:
            st.warning("Add GEMINI_API_KEY to .env to use Gemini.")
        top_k = st.slider("Retrieved chunks (top_k)", 1, 15, DEFAULT_TOP_K)
        temperature = st.slider(
            "Generation temperature", 0.0, 1.0, 0.2, step=0.05
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    subjects, units = _unit_options(indexed_files)
    if not subjects or not units:
        selected_subject = None
        selected_unit = None
        st.info("Index at least one PDF with a subject and unit to begin.")
    else:
        selection_cols = st.columns(2)
        selected_subject = selection_cols[0].selectbox("Subject", subjects)
        subject_units = sorted(
            {
                str(item["unit"])
                for item in indexed_files
                if item["subject"] == selected_subject and item.get("unit")
            },
            key=str.casefold,
        )
        selected_unit = selection_cols[1].selectbox(
            "Unit", subject_units or units
        )

    ask_tab, quiz_tab, summary_tab = st.tabs(
        ["Ask a Doubt", "Quick Quiz", "Summary"]
    )
    with ask_tab:
        _render_history()
        if question := st.chat_input("Ask a question about the selected syllabus"):
            _answer_question(
                pipeline,
                question,
                provider_choice,
                model,
                top_k,
                temperature,
                selected_subject,
                selected_unit,
            )

    with quiz_tab:
        st.subheader("Quick Quiz")
        st.write("Generate five multiple-choice questions from the selected unit.")
        if st.button("Generate 5-question quiz", disabled=not selected_unit):
            try:
                result = pipeline.quick_quiz(
                    str(selected_subject),
                    str(selected_unit),
                    provider_choice,
                    model=model,
                    temperature=temperature,
                )
                if result.fallback_message:
                    st.info(result.fallback_message)
                st.markdown(result.answer)
                _show_sources(_store_result(result))
            except Exception:
                logger.exception("Quiz generation failed")
                st.error("The quiz could not be generated. Check the selected provider.")

    with summary_tab:
        st.subheader("Unit Summary")
        st.write("Create a concise, cited summary of the selected unit.")
        if st.button("Summarize selected unit", disabled=not selected_unit):
            try:
                result = pipeline.summarize(
                    str(selected_subject),
                    str(selected_unit),
                    provider_choice,
                    model=model,
                    temperature=temperature,
                )
                if result.fallback_message:
                    st.info(result.fallback_message)
                st.markdown(result.answer)
                _show_sources(_store_result(result))
            except Exception:
                logger.exception("Unit summary generation failed")
                st.error("The summary could not be generated. Check the selected provider.")


if __name__ == "__main__":
    main()
