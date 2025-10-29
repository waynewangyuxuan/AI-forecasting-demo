# Technical Design Document - RAG-Based Forecasting Pipeline

**Version:** 0.1 (Draft)
**Last Updated:** November 2025
**Status:** Implementation Planning
**Related:** [PRODUCT.md](./PRODUCT.md)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Processing Workflow](#processing-workflow)
5. [Component Design](#component-design)
6. [Data Storage (SQLite)](#data-storage-sqlite)
7. [External Integrations](#external-integrations)
8. [Operational Concerns](#operational-concerns)
9. [Testing Strategy](#testing-strategy)
10. [Implementation Plan](#implementation-plan)
11. [Open Technical Questions](#open-technical-questions)

---

## System Overview
- **Goal:** Deliver a Retrieval-Augmented Generation pipeline that ingests forecasting questions, acquires current web evidence, and outputs probability estimates with transparent reasoning.
- **Scope:** End-to-end CLI-first workflow with optional lightweight API layer. Supports processing of three seed questions while remaining extensible.
- **Constraints:** Complete MVP in 3-6 hours, graceful degradation when APIs fail, SQLite for persistence, no heavy infrastructure dependencies.
- **Non-Goals:** Real-time streaming UI, multi-tenant auth, complex orchestration beyond a single runner, long-term data warehousing.

---

## Architecture

### High-Level Diagram
```
┌──────────────────────┐
│  CLI / REST Driver   │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Orchestration Layer (Python)                 │
│  - Pipeline Runner                                            │
│  - Task Graph + State Tracker                                 │
└─────────┬─────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────┬──────────────────────────┬────────┐
│ Query & Search Module    │ Web Scraper Module       │ Event  │
│ - LLM query expansion    │ - Async fetch & parsing  │ Proc.  │
│ - Google Search API      │ - Content cleaning       │ - LLM  │
└───────┬──────────────────┴───────────────┬──────────┴────────┘
        │                                  │
        ▼                                  ▼
┌──────────────────────────────────────────────────────────────┐
│      Document Store + Embedding Pipeline                     │
│  - Text normalization                                         │
│  - Embedding generation (LLM API)                             │
│  - Similarity clustering + deduplication                      │
└─────────┬─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┬────────────────────┐
│ Timeline Builder                         │ Forecast Generator │
│ - Chronological ordering                 │ - LLM reasoning    │
│ - Source citations                       │ - Probability calc │
└─────────┬────────────────────────────────┴──────────┬─────────┘
          │                                           │
          ▼                                           ▼
┌──────────────────────┐                  ┌────────────────────┐
│ SQLite Persistence   │◄────────────────►│ Reporting Layer    │
│ - Questions          │                  │ - JSON export      │
│ - URLs & documents   │                  │ - CLI summaries    │
│ - Events & forecasts │                  │ - Optional REST    │
└──────────────────────┘                  └────────────────────┘
```

### Deployment Model
- Single Python process executed via CLI command (`python -m forecast_pipeline run <question_id>`)
- Optional FastAPI app exposes REST endpoints for triggering runs and retrieving outputs; same code paths as CLI.
- All state stored in local SQLite file (`data/forecast.db`) to keep footprint light.
- Background tasks handled synchronously with async IO primitives; no external queue.

---

## Technology Stack
- **Language:** Python 3.11
- **Core Libraries:**
  - `typer` for CLI ergonomics
  - `fastapi` + `uvicorn` for optional REST surface
  - `httpx` with async client for HTTP requests
  - `beautifulsoup4` + `readability-lxml` for content extraction
  - `tenacity` for retry policies
  - `sqlalchemy` (Core) for SQLite access
  - `pydantic` for data models and validation
  - `numpy`, `scikit-learn` (or `hdbscan` optional) for clustering
  - `openai`/`google-generativeai` SDKs depending on selected LLM endpoints
- **Configuration:** `pydantic-settings` or `.env` file loaded via `python-dotenv`
- **Jobs & Concurrency:** AsyncIO event loop with bounded concurrency for scraping and embeddings.
- **Artifacts:** Forecast outputs saved as JSON + Markdown under `outputs/<question_id>/`.

---

## Processing Workflow

### Stage 0: Initialization
- Load question definition from SQLite or inline config.
- Instantiate dependency container (clients, DB session, logger, metrics collector).
- Write run metadata (start timestamp, version hash) to `Runs` table.

### Stage 1: Query Generation
- Use Gemini 2.5 Flash (or configured LLM) with tailored prompt to generate ~10 diversified queries.
- Persist queries to `SearchQueries` table with prompt version.
- Apply dedupe heuristic (case-fold + fuzzy matching) before execution.

### Stage 2: Web Search
- Call Google Custom Search JSON API per query (rate-limited, sequential with backoff).
- Store returned URLs + metadata in `SearchResults` table (rank, snippet, query_id).
- Skip URLs already processed for the question (unique SHA256 key).

### Stage 3: Scraping & Content Extraction
- Async fetch with `httpx` respecting robots.txt (using `robotexclusionrulesparser`).
- Parse HTML -> main article text via `readability` fallback to `trafilatura` if needed.
- Normalize (strip boilerplate, lower noise) and store in `Documents` table with content hash.
- Log fetch failures to `Errors` table for transparency.

### Stage 4: Event Extraction
- Chunk documents (max 2k tokens) with overlap for context.
- Prompt LLM to extract timestamped events (ISO date, description, actors, supporting quote).
- Persist results in `Events` table with reference to document and provenance metadata.

### Stage 5: Embedding, Clustering, Deduplication
- Generate embeddings for each event description (OpenAI `text-embedding-3-large` or equivalent).
- Cache embedding vectors on disk (`.npy`) plus store dimension + hash in DB.
- Compute similarity matrix; apply clustering (default: Agglomerative clustering with cosine distance threshold 0.2).
- For each cluster, select canonical event (highest confidence) and merge citations.
- Write cluster summary to `EventClusters` table.

### Stage 6: Timeline Construction
- Order canonical events by timestamp (fallback to publication date if missing).
- Enrich with aggregated metadata (participant tags, sentiment, geographic hints using regex/NER).
- Store final timeline entries in `Timeline` table, linking back to clusters and documents.

### Stage 7: Forecast Generation
- Compose prompt with: original question, key timeline entries, unresolved uncertainties.
- Call LLM (Gemini 2.5 Flash or GPT-4.1) to produce structured JSON output: probability, reasoning chain, caveats.
- Parse and validate JSON using Pydantic schema (`ForecastOutput`), persist to `Forecasts` table.
- Render Markdown summary for CLI output.

### Stage 8: Reporting & Review
- Generate run report (queries, successful scrapes, event counts, dedupe stats, final forecast).
- Persist aggregate metrics in `RunMetrics` table.
- Expose results via CLI or REST endpoint (`GET /runs/{run_id}`) returning JSON + Markdown.

---

## Component Design

### 1. Orchestrator (`pipeline/orchestrator.py`)
- Coordinates stages using a state machine; each stage is idempotent and resumable.
- Maintains run context (question id, run id, logger, DB session).
- Supports `--resume` flag to skip completed stages (based on `Runs.stage_state`).
- Emits structured logs (JSON) through standard logger.

### 2. Query Generation Service (`services/query_generation.py`)
- Accepts question text + history of prior queries to avoid duplicates.
- Encapsulates prompt template and LLM client call.
- Includes post-processing filters (length bounds, keyword coverage heuristics).

### 3. Search Service (`services/search.py`)
- Responsible for Google API interactions.
- Implements adaptive throttling and error categorization (quota vs transient vs fatal).
- Stores raw responses for debugging in `SearchResponses` table.

### 4. Scraper (`services/scraper.py`)
- Worker pool built on asyncio `gather` with semaphore-limited concurrency (default 5).
- Applies domain-specific selectors if known (config-driven YAML file).
- Sanitizes text (remove scripts, compress whitespace) before persistence.

### 5. Document Processor (`services/doc_processor.py`)
- Splits documents into semantic chunks (approx 500 tokens) for downstream LLM calls.
- Detects language and filters non-English content unless flagged otherwise.

### 6. Event Extractor (`services/event_extractor.py`)
- Batched LLM prompts with streaming support; handles retries on malformed JSON via re-prompt.
- Confidence scoring derived from LLM output + heuristics (source recency, snippet overlap).

### 7. Embedding & Clustering (`services/embedding.py`, `services/clustering.py`)
- Caches embeddings keyed by document chunk hash to avoid duplicate calls.
- Clustering strategies configurable; default agglomerative with distance threshold.
- Provides diagnostics (silhouette score) logged per run.

### 8. Timeline Builder (`services/timeline.py`)
- Consolidates cluster outputs into ordered timeline.
- Attaches citations (URL + quote spans) and normalization tags (actors, topics).
- Ensures minimum event coverage (target 15-30) else flags run as LOW_COVERAGE.

### 9. Forecast Generator (`services/forecast.py`)
- Single entry point returning `Forecast` dataclass.
- Handles fallback prompt if initial response missing fields.
- Persists reasoning chain and supporting evidence references.

### 10. Persistence Layer (`db/repository.py`)
- Thin repository pattern atop SQLAlchemy Core with typed functions per table.
- Centralizes migrations (see `db/schema.sql` executed on startup).
- Provides helper for transactional stage execution with rollback on failure.

### 11. Interface Adapters
- **CLI (`cli.py`):** `forecast run <question>` and `forecast report <run_id>` commands.
- **REST (`api.py`):** Minimal FastAPI app with `/questions`, `/runs`, `/forecasts` endpoints; ideal for demo integration.
- **Outputs:** Markdown report generated via Jinja2 template, saved under `outputs/`.

---

## Data Storage (SQLite)

### Database File
- Location: `data/forecast.db`
- Connection string: `sqlite+aiosqlite:///data/forecast.db`

### Tables (Initial Schema)
```sql
CREATE TABLE IF NOT EXISTS Questions (
  id INTEGER PRIMARY KEY,
  question_text TEXT NOT NULL,
  resolution_criteria TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Runs (
  id INTEGER PRIMARY KEY,
  question_id INTEGER NOT NULL,
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT,
  status TEXT CHECK(status IN ('PENDING','RUNNING','FAILED','COMPLETED')),
  stage_state TEXT,
  git_commit TEXT,
  FOREIGN KEY(question_id) REFERENCES Questions(id)
);

CREATE TABLE IF NOT EXISTS SearchQueries (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  query_text TEXT NOT NULL,
  prompt_version TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(run_id, query_text),
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

CREATE TABLE IF NOT EXISTS SearchResults (
  id INTEGER PRIMARY KEY,
  query_id INTEGER NOT NULL,
  url TEXT NOT NULL,
  title TEXT,
  snippet TEXT,
  rank INTEGER,
  UNIQUE(query_id, url),
  FOREIGN KEY(query_id) REFERENCES SearchQueries(id)
);

CREATE TABLE IF NOT EXISTS Documents (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  url TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  raw_content TEXT,
  cleaned_content TEXT,
  fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
  status TEXT,
  UNIQUE(run_id, content_hash),
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

CREATE TABLE IF NOT EXISTS Events (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  event_time TEXT,
  headline TEXT,
  body TEXT,
  actors TEXT,
  confidence REAL,
  raw_response TEXT,
  FOREIGN KEY(document_id) REFERENCES Documents(id)
);

CREATE TABLE IF NOT EXISTS Embeddings (
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL,
  vector BLOB NOT NULL,
  model TEXT,
  dimensions INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(event_id, model),
  FOREIGN KEY(event_id) REFERENCES Events(id)
);

CREATE TABLE IF NOT EXISTS EventClusters (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  label TEXT,
  centroid_event_id INTEGER,
  member_ids TEXT,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

CREATE TABLE IF NOT EXISTS Timeline (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  cluster_id INTEGER,
  event_time TEXT,
  summary TEXT,
  citations TEXT,
  tags TEXT,
  FOREIGN KEY(run_id) REFERENCES Runs(id),
  FOREIGN KEY(cluster_id) REFERENCES EventClusters(id)
);

CREATE TABLE IF NOT EXISTS Forecasts (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  probability REAL,
  reasoning TEXT,
  caveats TEXT,
  raw_response TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

CREATE TABLE IF NOT EXISTS RunMetrics (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  metric_name TEXT,
  metric_value REAL,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);

CREATE TABLE IF NOT EXISTS Errors (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  stage TEXT,
  reference TEXT,
  error_type TEXT,
  message TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(run_id) REFERENCES Runs(id)
);
```

### Data Access Guidelines
- Interact via repository layer; avoid raw SQL in business logic.
- Use transactions per stage; commit after stage success to maintain resumability.
- Index on `Documents.content_hash`, `Events.event_time`, `Timeline.run_id` for faster retrieval.

---

## External Integrations

### LLM Providers
- **Gemini 2.5 Flash:** primary for query generation and forecasting.
- **OpenAI GPT-4.1 or similar:** fallback for event extraction if Gemini quota hit.
- **Embedding Model:** `text-embedding-3-large` (OpenAI) or `text-embedding-004` (Vertex) depending on access.
- Clients wrapped in adapter classes to allow mocking during tests.

### Search API
- Google Custom Search JSON API.
- Configurable keys via environment: `GOOGLE_CSE_ID`, `GOOGLE_API_KEY`.
- Respect per-day quota by tracking usage in `RunMetrics` table.

### Optional Services
- Proxy service (BrightData, ScraperAPI) abstracted via HTTPX transport interface; use only if necessary.
- Local caching of HTTP responses on disk (`.cache/`) to reduce duplicate fetches during development.

### Configuration & Secrets
- `.env` file template (`.env.example`) listing keys.
- Use `secretsmanager.py` helper for loading environment variables with validation.
- Sensitive outputs excluded from logs; store API responses only when `DEBUG=1`.

---

## Operational Concerns

### Logging & Observability
- Structured logging via `structlog` with context (run_id, stage, question_id).
- Per-stage metrics captured (duration, success count, fail count).
- CLI supports `--verbose` flag for trace-level output.

### Error Handling & Retries
- Wrap external calls with exponential backoff (max 3 attempts).
- Classify failures: recoverable (HTTP 5xx, rate limit), permanent (404, parsing errors), data issues.
- Persist detailed errors for demo transparency.
- Provide `forecast resume --run-id <id>` to restart failed runs at last incomplete stage.

### Performance Targets
- Keep total runtime within 3-6 hour window for three questions combined.
- Limit concurrent scrapes to 5 to avoid bans; throttle LLM calls to stay within quota.
- Aim for >80% successful document fetch rate and >70% relevant event extraction rate.

### Security & Compliance
- Respect `robots.txt`; store fetch decision per domain.
- Redact PII by skipping pages flagged via heuristics (presence of user data patterns).
- Ensure all API keys loaded from environment, never hard-coded.

---

## Testing Strategy
- **Unit Tests:**
  - Mocked LLM responses for query generation, event extraction, forecast formatting.
  - Repository layer tests using in-memory SQLite (`sqlite:///:memory:`).
- **Integration Tests:**
  - Dry-run pipeline with cached HTTP fixtures (e.g., `vcrpy`).
  - End-to-end test using local HTML fixtures representing typical articles.
- **Regression Checks:**
  - Snapshot tests on forecast Markdown output to detect prompt drift.
  - Validate deduplication fidelity via known event sets.
- **Operational Tests:**
  - Load test search and scraping modules with 100 URLs to ensure throttling works.
  - Chaos tests injecting failures into LLM calls to confirm retry + resume.

---

## Implementation Plan

### Phase 1: Foundation (Day 0)
- Scaffold repository structure (`pipeline/`, `services/`, `db/`).
- Implement SQLite schema migration script (`python -m db.migrate`).
- Configure logging, settings management, CLI entrypoint skeleton.

### Phase 2: Retrieval (Day 0-1)
- Build query generation service with prompt templates and tests.
- Integrate Google Search API client with caching + persistence.
- Implement scraper with async fetch and basic cleaning; store documents.

### Phase 3: Knowledge Processing (Day 1)
- Implement chunking and event extraction pipeline with mocked LLM interface.
- Add embedding + clustering utilities; validate dedupe metrics on sample data.
- Build timeline builder and ensure outputs persisted correctly.

### Phase 4: Forecasting & Reporting (Day 1-2)
- Implement forecasting prompt and parsing logic.
- Generate Markdown/JSON reports with citations and metrics.
- Add REST endpoints (optional) for retrieving results.

### Phase 5: Hardening & Demo Prep (Day 2)
- Add failure recovery paths (`--resume`, error table UI in CLI).
- Instrument logging and metrics export.
- Populate sample runs for demo readiness; document operational playbook in `META/PROGRESS.md`.

---

## Open Technical Questions
1. **Embedding Provider:** Confirm availability of OpenAI embeddings vs Vertex; adjust adapter accordingly.
2. **Clustering Algorithm:** Is simple agglomerative sufficient or should we integrate HDBSCAN for dense event clusters?
3. **Forecast Output Format:** Should we include confidence intervals or stick to point probability + qualitative confidence level?
4. **Parallel Runs:** Do we need concurrency across questions, or sequential processing acceptable for MVP demo?
5. **Caching Strategy:** Should we persist raw LLM responses beyond debug mode for auditability?

---

**End of Document**
