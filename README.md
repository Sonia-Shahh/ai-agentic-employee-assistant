# Employee AI Assistant (RAGBot)

Employee AI Assistant is an asynchronous retrieval-augmented generation pipeline for ingesting PDFs and audio, storing embeddings in Qdrant, and answering questions with GPT 5.1-mini plus Whisper-based transcription. It now uses a Pydantic AI agent with pluggable tools (vector_search, web_search via SerpAPI) and post-turn LLM-as-a-Judge scoring powered by pydantic-evals. This README walks through environment setup, data ingestion, interactive querying, and the engineering decisions behind the system.

## 1. Prerequisites

1. **Install Python 3.11+**
   ```powershell
   python --version  # confirm >= 3.11

2. **Ensure pip is up to date**

    ```python -m pip install --upgrade pip

3. **Install ffmpeg + ffprobe (required for audio chunking)**
    Download the Windows build from https://ffmpeg.org/download.html and add the bin folder to your system PATH, then verify:

    ```ffmpeg -version
       ffprobe -version
       ```

4. **Clone this repository**

    ```
    git clone https://github.com/your-org/ragbot.git
    cd ragbot
    ```

## 2. Environment Setup in VS Code Terminal

1. **Create and activate a virtual environment**

    ```python -m venv .venv
        .venv\Scripts\activate
    ```

## 2. **Install project dependencies**

    ``` pip install -r requirements.txt 
    ```
       This installs OpenAI, qdrant-client, PyPDF, Tiktoken, python-dotenv, and other toolchain requirements.

## 3. **Configure Qdrant Cloud and Environment Variables**

1. Create a free Qdrant Cloud cluster at https://qdrant.io/cloud

    Obtain the cluster URL (e.g., https://your-cluster-name-xxxx.a.qdrant.io) and API key.

2. Create a .env file in the repository root

    OPENAI_API_KEY=sk-your-openai-key
    QDRANT_URL=https://your-cluster-name-xxxx.a.qdrant.io
    QDRANT_API_KEY=your-qdrant-api-key
    QDRANT_COLLECTION=ragbot-collection
    SERPAPI_API_KEY=your-serpapi-key
    LOGFIRE_TOKEN=your-logfire-token


    The application automatically loads .env through python-dotenv during startup.

## 4. Data Preparation and Directory Layout

```
ragbot/
├── data/
│   ├── pdf/        # place source PDFs here
│   └── audio/      # place audio/video files here
├── scripts/        # CLI entry points
└── src/            # library code

```

Populate 'data/pdf' with files such as 'Databases_for_GenAI.pdf' and 'data/audio' with matching MP3/MP4/WAV assets.

## 5. Run the Ingestion Pipeline

Ingest all PDFs and audio into the configured Qdrant collection:

```
python -m scripts.run_ingestion --pdf-dir data/pdf --audio-dir data/audio --collection ragbot-collection
```


The ingestion script:

- Reads PDFs asynchronously via PyPDF.
- Transcribes every audio file using Whisper. Long recordings are automatically split into overlapping chunks using ffmpeg and ffprobe before being sent to the OpenAI API.
-Normalizes and tokenizes text, produces embeddings through text-embedding-3-small, ensures the Qdrant collection exists, and upserts chunk metadata with unique UUID identifiers.

## 6. Ask Questions Interactively

Launch the sequential chat loop, which maintains the full conversation history during a session:

```
    python -m scripts.ask_questions --collection ragbot-collection --limit 5 --interactive
```

    - Enter each user turn at the You: prompt.
    - Hit Enter on a blank line to exit.
    - Runtime prints:
        - [answer] … the model response
        - [metrics] running evals...
        - [metrics] {...} containing LLM-judge scores

## 7. Engineering Walkthrough

### 7.1 Ingestion of PDFs
- `src/data_loader/pdf_loader.py` uses `asyncio.to_thread` to call PyPDF for each file, returning `(Path, text)` pairs without blocking the event loop.
- `src/rag/pipeline.py` normalizes whitespace, chunks each document according to the configured token window, embeds the text, and stores it in Qdrant with metadata such as source type, filename, and chunk index.

### 7.2 Audio Transcription and Splitting
- `scripts/run_ingestion.py` passes audio paths to `transcribe_audios()`.
- `src/data_loader/audio_transcriber.py` probes each file with ffprobe. If the recording exceeds the Whisper limit (1,400 seconds), it splits the file using ffmpeg into overlapping slices (configurable via `AUDIO_CHUNK_MAX_SECONDS` and `AUDIO_CHUNK_OVERLAP_SECONDS`).
- Each slice is uploaded sequentially to Whisper (`AsyncOpenAI.audio.transcriptions.create`). The combined transcript is normalized and chunked just like the PDF text, so audio knowledge is searchable alongside documents.

### 7.3 Chunking and Embeddings
- `src/text_processing/chunker.py` performs token-aware splitting with Tiktoken. Overlaps preserve context across neighboring segments.
- `src/embeddings/openai_embeddings.py` batches requests to `text-embedding-3-small`, ensuring the pipeline only proceeds when embeddings are returned successfully.
- `src/vectorstore/qdrant_store.py` uses unique UUIDs per chunk to avoid accidental overwrites. It supports both `query_points` (current `qdrant-client`) and `search` (legacy clients) for neighborhood retrieval.

### 7.4 Chatbot and Prompt Design
- `src/rag/chatbot.py` exposes `retrieve()` and `answer()` methods.
- `answer()` accepts the current question plus the sequential conversation history collected by `scripts/ask_questions.py`.
- The prompt explicitly instructs GPT 5.1-mini to use only the supplied context and history, resolve references carefully, and return “I do not know based on the provided context.” when facts are missing. Conversation history is rendered as alternating user and assistant messages so the model can ground pronouns and follow-up questions correctly.

### 7.5 Pydantic AI Agent, Tools, and Web Search
- Agent factory (`src/agent/agent.py`) builds a Pydantic AI `Agent` with retries and the system prompt.
- Tools are registered in `src/agent/tools.py`:
  - `vector_search`: queries Qdrant with fresh embeddings for the user question.
  - `web_search`: SerpAPI-backed Google search when KB context is insufficient; requires `SERPAPI_API_KEY`.
- SerpAPI client is only created when the key is present; otherwise the tool returns an explanatory message.

### 7.6 LLM-as-a-Judge Metrics
- After every answer, `src/agent/metrics.py` calls Pydantic Evals’ `judge_input_output` with rubric “factually correct, relevant, concise, safe.”
- Judge model defaults to `openai:gpt-5-nano` with deterministic settings; set `OPENAI_API_KEY` for access.
- The metrics payload includes `llm_judge_score`, `llm_judge_pass`, `llm_judge_reason`, and `judge_model`, printed in the interactive loop.
- Optional telemetry: if `LOGFIRE_TOKEN` is set, Logfire instrumentation (configured in `src/agent/agent.py`) sends eval spans/results to the Logfire UI.

## 8. Operational Tips

- **Testing**: Run `python -m compileall src scripts` (or `uv run python -m compileall src scripts`) after edits to catch syntax errors quickly.
- **Environment validation**: Call `python - <<'PY'
from src.config import settings
settings.validate()
print("Settings OK")
PY` to confirm that mandatory keys like `OPENAI_API_KEY` are present.
- **Qdrant hygiene**: When re-ingesting from scratch, either drop the collection via the Qdrant dashboard or point to a new collection name using `--collection`. Because point IDs are UUIDs, re-running ingestion on the same collection appends new data without clobbering existing vectors.

## 9. Reference Commands

| Purpose | Command |
| --- | --- |
| Install dependencies | `uv sync --python 3.11` |
| Run ingestion | `PYTHONPATH=$PWD uv run python -m scripts.run_ingestion --pdf-dir data/pdf --audio-dir data/audio --collection ragbot-collection` |
| Interactive QA | `PYTHONPATH=$PWD uv run python -m scripts.ask_questions --collection ragbot-collection --limit 5 --interactive` |
