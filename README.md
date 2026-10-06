# A Retrieval-Augmented Generation-Based Web Tutor for Syllabus-Specific Doubt Resolution

This project provides an interactive tutor that answers student questions from
uploaded course PDFs. It retrieves page-level excerpts from the indexed subject
and unit, refuses unsupported questions, and cites its material as
`[filename.pdf, p.N]`. It supports a local Ollama model and Google Gemini.

## Architecture

```text
PDF uploads
    |
    v
PyPDF page extraction -> text cleanup -> page-aware recursive chunks
    |                                           |
    |                                           v
    |                               Sentence Transformer embeddings
    |                                           |
    +-------------------------------------------v
                                   Persistent ChromaDB (cosine distance)
                                                  |
Student question -> follow-up rewrite -> threshold retrieval + unit filters
                                                  |
                                                  v
                              grounded prompt -> Ollama or Gemini
                                                  |
                                                  v
                                  cited answer + source passages
```

```text
app.py                 Streamlit interface: indexing, chat, quiz, summary
evaluate.py            CSV-based retrieval hit-rate and keyword-coverage evaluation
src/config.py          Environment, paths, and model/chunk defaults
src/ingestion.py       PDF page extraction, cleanup, and metadata
src/chunking.py        LangChain recursive, page-aware text chunks
src/embeddings.py      Cached, normalized sentence-transformer vectors
src/vector_store.py    Persistent ChromaDB collection and file operations
src/retriever.py       Similarity threshold, filters, and optional reranking
src/llm.py             Ollama streaming and Gemini generation/fallback
src/rag_pipeline.py    Follow-up rewrite, retrieval, and grounded generation
src/prompts.py         Tutor, quiz, summary, and rewrite instructions
src/utils.py           Application logging and text helpers
```

## Requirements

- Windows with Python **3.11.9**
- Ollama installed and running locally for the local provider
- Optional Gemini API key for the cloud provider
- The Python dependencies listed in `requirements.txt`

The sentence-transformer downloads its model the first time it is used. GPU
inference is selected automatically when CUDA is available; otherwise it runs on
the CPU.

## Setup

From PowerShell, in the project root:

```powershell
py -3.11.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2
```

Edit `.env` to set `GEMINI_API_KEY` if you want to use Gemini. The Ollama model
can be changed with `OLLAMA_MODEL`; the Gemini model can be changed with
`GEMINI_MODEL`. Keep Ollama running (for example, start the Ollama desktop
application or run `ollama serve` in another terminal).

Run the app:

```powershell
streamlit run app.py
```

## Usage

1. In the sidebar, upload one or more text-based PDF notes or textbook files.
2. Enter the subject and unit that describe the uploaded material and select
   **Index documents**. Files are saved under `data/uploads/`; their extracted
   chunks and metadata are persisted under `data/chroma_db/`.
3. Choose the Ollama or Gemini provider and a model. Set the number of retrieved
   chunks and response temperature as desired.
4. In **Ask a Doubt**, select a subject and unit, then submit a question. Answers
   include source citations; expand **Retrieved syllabus passages** to inspect
   excerpts, page numbers, and similarity scores. Follow-up questions are
   rewritten from the latest three conversation turns.
5. Use **Quick Quiz** to make five unit-based MCQs or **Summary** for a cited
   summary of the selected unit.
6. Use the file list in the sidebar to remove indexed files. Helpful/not helpful
   votes are appended to `data/feedback.csv`. **Clear chat** resets the current
   browser session's conversation.

Scanned PDFs without a text layer are skipped with a warning; run OCR on them
before uploading. Pages without extractable text are not indexed.

### Evaluate retrieval

Prepare a UTF-8 CSV with required `question` and `expected_keywords` columns.
Separate multiple expected keywords with semicolons, commas, or `|`. Optional
`subject` and `unit` columns constrain each row to matching indexed metadata.

```powershell
python evaluate.py .\evaluation.csv --output .\evaluation-results.csv --top-k 5
```

The script reports retrieval hit rate (at least one expected keyword appears in
the retrieved passages) and token-level keyword coverage. It evaluates retrieval
without requiring an LLM call. The vector index must already contain the
evaluation syllabus.

## Tests

```powershell
python -m pytest
```

## Troubleshooting

- **Ollama connection failed:** Confirm Ollama is running at
  `http://localhost:11434` and that the selected model is installed with
  `ollama list`. The app tries Gemini as a fallback; without a Gemini key, an
  error message explains that both providers are unavailable.
- **Gemini reports a missing key:** Put a valid `GEMINI_API_KEY` in `.env` in
  the project root, then restart Streamlit.
- **No answer found:** Confirm the PDF is text-selectable, indexed under the
  correct subject and unit, and that the selected filters match. The similarity
  threshold intentionally rejects weak matches rather than allowing a guess.
- **Scanned or image-only PDF:** OCR the file and upload the searchable PDF.
- **Damaged/password-protected PDF:** Export an unlocked, valid PDF and index it
  again.
- **First embedding run is slow or out of memory:** The sentence-transformer
  model downloads on first use. Close other applications, use a smaller batch
  if adapting the embedding code, or allow CPU inference.
- **Stale or corrupt local index:** Stop Streamlit and, after backing up any
  needed data, remove the specific contents of `data/chroma_db` and restart the
  app to create a fresh collection. Re-index the source PDFs afterward.
- **Diagnostic details:** Application errors are logged to `logs/app.log`; the
  interface displays friendly messages without exposing tracebacks.

