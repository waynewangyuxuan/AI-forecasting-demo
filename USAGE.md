# AI Forecasting Pipeline - Usage Guide

This guide explains how to use the AI Forecasting Pipeline CLI and orchestration system.

## Table of Contents
- [Installation](#installation)
- [CLI Commands](#cli-commands)
- [Pipeline Stages](#pipeline-stages)
- [Examples](#examples)
- [Output Files](#output-files)
- [Resuming Failed Runs](#resuming-failed-runs)
- [Programmatic Usage](#programmatic-usage)

## Installation

1. **Setup Virtual Environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Environment Variables**:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# - GOOGLE_API_KEY
# - GOOGLE_CSE_ID
# - GOOGLE_GEMINI_API_KEY
# - OPENAI_API_KEY
```

4. **Initialize Database**:
```bash
python cli.py init
```

## CLI Commands

The CLI provides several commands for running and managing forecasts.

### Run a Forecast

Execute the full pipeline on a forecasting question:

```bash
python cli.py run "Will the S&P 500 reach 5000 by end of 2024?"
```

**Options:**
- `--output, -o`: Custom output directory (default: `outputs/run_<id>/`)
- `--max-urls`: Limit URLs per query (default: 10)
- `--max-events`: Limit total events extracted
- `--dry-run`: Run without saving to database
- `--verbose, -v`: Enable verbose logging

**Examples:**
```bash
# Basic run
python cli.py run "Will Apple release a new iPad in Q1 2024?"

# With custom output directory
python cli.py run "Question here" --output ./my_forecasts

# Limit data collection
python cli.py run "Question here" --max-urls 5 --max-events 50

# Dry run (no database changes)
python cli.py run "Question here" --dry-run --verbose
```

### Resume a Failed Run

Resume from the last completed stage:

```bash
python cli.py resume 42
```

**Options:**
- `--verbose, -v`: Enable verbose logging

### Check Run Status

View detailed status of a run:

```bash
python cli.py status 42
```

Shows:
- Question text
- Current status (PENDING, RUNNING, COMPLETED, FAILED)
- Start and completion timestamps
- Current stage
- Completed stages
- Errors (if any)

### List All Runs

View all runs with their status:

```bash
python cli.py list
```

**Options:**
- `--limit, -n`: Maximum runs to show (default: 20)

### Generate Report

Generate a detailed report for a completed run:

```bash
# View markdown report in terminal
python cli.py report 42 --format markdown

# Save JSON report to file
python cli.py report 42 --format json --output report.json
```

**Options:**
- `--format, -f`: Output format (`markdown` or `json`)
- `--output, -o`: Save to file instead of printing to console

### Initialize Database

Create database tables and directories:

```bash
python cli.py init
```

### Version Info

Show version information:

```bash
python cli.py version
```

## Pipeline Stages

The pipeline executes 8 stages sequentially:

### 1. INIT - Initialization
- Initialize all service instances
- Create output directory
- Load configuration
- **Duration:** ~1 second

### 2. QUERY_GEN - Query Generation
- Generate 10 diversified search queries using LLM
- Deduplicate similar queries
- Save to database
- **Duration:** ~5-10 seconds

### 3. SEARCH - Web Search
- Execute each query via Google Custom Search API
- Collect top 10 URLs per query
- Store results with metadata (title, snippet, rank)
- **Duration:** ~20-30 seconds (sequential with rate limiting)

### 4. SCRAPE - Content Extraction
- Fetch HTML from all unique URLs
- Extract main content using readability
- Clean and normalize text
- Deduplicate by content hash
- **Duration:** ~30-60 seconds (async with 5 concurrent requests)

### 5. EVENT_EXTRACT - Event Extraction
- Chunk documents (~500 tokens per chunk)
- Extract timestamped events using LLM
- Parse dates and actors
- Filter by confidence threshold
- **Duration:** ~2-5 minutes (depends on document count)

### 6. CLUSTER - Embedding & Clustering
- Generate embeddings for each event
- Compute similarity matrix
- Cluster similar events (distance threshold: 0.2)
- Select canonical event per cluster
- **Duration:** ~30-60 seconds

### 7. TIMELINE - Timeline Construction
- Order events chronologically
- Merge citations from clustered events
- Extract metadata (actors, topics, locations)
- Assess coverage quality
- **Duration:** ~5-10 seconds

### 8. FORECAST - Forecast Generation
- Compose prompt with question and timeline
- Generate structured forecast with reasoning
- Parse and validate output
- Save probability and caveats
- **Duration:** ~10-20 seconds

**Total Pipeline Duration:** ~4-8 minutes per question (depending on data volume)

## Examples

### Example 1: Basic Forecast

```bash
python cli.py run "Will Tesla stock reach $300 by June 2024?"
```

Output:
```
┌─────────────────────────────────────────────────────────┐
│ AI Forecasting Pipeline                                 │
│                                                         │
│ Question: Will Tesla stock reach $300 by June 2024?   │
└─────────────────────────────────────────────────────────┘

INIT: Initializing pipeline
QUERY_GEN: Generating search queries for: Will Tesla stock...
SEARCH: Searching 10 queries
SCRAPE: Scraping 85 unique URLs
EVENT_EXTRACT: Extracting events from 42 documents
CLUSTER: Clustering 127 events
TIMELINE: Building timeline
FORECAST: Generating forecast

[SUCCESS] Pipeline completed successfully!
[INFO] Run ID: 1

┌────────── Pipeline Summary ──────────┐
│ Metric                   │    Value  │
├──────────────────────────┼───────────┤
│ QUERY success_count      │       10  │
│ SEARCH total_urls        │       85  │
│ SCRAPE documents_saved   │       42  │
│ EVENT total_events       │      127  │
│ CLUSTER clusters         │       34  │
│ TIMELINE timeline_entries│       34  │
└──────────────────────────┴───────────┘

┌─────────── Forecast Preview ────────────┐
│ Probability: 0.45                        │
│                                          │
│ Reasoning:                               │
│ Based on historical volatility and...   │
└──────────────────────────────────────────┘

[INFO] Full report: outputs/run_1/forecast_report.md
```

### Example 2: Resume Failed Run

If a run fails at stage 5 (CLUSTER):

```bash
# First run (fails at CLUSTER)
python cli.py run "Complex question"
# Error occurs during clustering...

# Resume from last checkpoint
python cli.py resume 2
```

The resume command will:
1. Load completed stages from database
2. Skip INIT, QUERY_GEN, SEARCH, SCRAPE, EVENT_EXTRACT
3. Retry CLUSTER stage
4. Continue with remaining stages

### Example 3: Check Status During Run

While a run is executing, check its progress:

```bash
# In another terminal
python cli.py status 3
```

Output shows current stage and completed stages.

### Example 4: Limited Data Collection

For faster testing or lower API costs:

```bash
python cli.py run "Test question" \
  --max-urls 3 \
  --max-events 20 \
  --verbose
```

## Output Files

Each run creates a directory: `outputs/run_<id>/`

### forecast_output.json

Complete structured output:

```json
{
  "question": {
    "id": 1,
    "text": "Will...",
    "resolution_criteria": "..."
  },
  "run": {
    "id": 1,
    "started_at": "2024-01-15T10:30:00",
    "completed_at": "2024-01-15T10:35:42",
    "status": "COMPLETED",
    "git_commit": "abc123..."
  },
  "queries": [
    "Tesla stock price forecast 2024",
    "TSLA technical analysis June 2024",
    ...
  ],
  "timeline": [
    {
      "event_time": "2024-01-10",
      "summary": "Tesla announces...",
      "citations": [
        {
          "url": "https://...",
          "quote": "..."
        }
      ],
      "tags": {
        "actors": ["Tesla", "Elon Musk"],
        "topics": ["earnings", "production"],
        "geographic": ["USA", "China"]
      }
    },
    ...
  ],
  "forecast": {
    "probability": 0.45,
    "reasoning": "Based on...",
    "caveats": [
      "Market volatility could impact...",
      "Limited data on..."
    ]
  },
  "metrics": {
    "QUERY_GEN_duration_seconds": 8.2,
    "SEARCH_total_urls": 85,
    ...
  },
  "errors": [
    {
      "stage": "SCRAPE",
      "type": "ScraperError",
      "message": "Failed to fetch https://..."
    }
  ]
}
```

### forecast_report.md

Human-readable Markdown report:

```markdown
# Forecast Report

**Generated:** 2024-01-15 10:35:42

## Question

Will Tesla stock reach $300 by June 2024?

## Forecast

**Probability:** 0.45

### Reasoning

Based on historical volatility analysis and recent earnings reports...

### Caveats

- Market volatility could significantly impact predictions
- Limited data on upcoming product launches
- Regulatory changes not fully accounted for

## Timeline

**Total Events:** 34

### 2024-01-10

Tesla announces Q4 2023 earnings beat expectations...

**Sources:**
- [https://example.com/article1](https://example.com/article1)
- [https://example.com/article2](https://example.com/article2)

...
```

## Resuming Failed Runs

The pipeline is designed to be **idempotent and resumable**. Each stage checks if it's already completed before running.

### How Resume Works

1. **Stage State Tracking**: After each stage completes successfully, the run's `stage_state` is updated in the database:
   ```json
   {
     "completed_stages": ["INIT", "QUERY_GEN", "SEARCH"],
     "current_stage": "SEARCH",
     "last_updated": "2024-01-15T10:32:15"
   }
   ```

2. **Resume Execution**: When resuming:
   - Load `stage_state` from database
   - Skip stages in `completed_stages` list
   - Start from first incomplete stage
   - Continue through remaining stages

3. **Transactional Safety**: Each stage uses transactions:
   - Commit on success → stage marked complete
   - Rollback on failure → stage not marked complete
   - Can safely retry failed stages

### When to Resume

Use `resume` when:
- Pipeline fails due to API rate limit
- Network connection drops
- LLM service temporarily unavailable
- Unexpected error in one stage

Don't need to resume when:
- Want to regenerate forecast with different data
- Question or parameters changed
- Previous run completed successfully

## Programmatic Usage

Use the orchestrator directly in Python code:

```python
import asyncio
from pipeline import run_forecast_pipeline

async def main():
    run_id = await run_forecast_pipeline(
        question_text="Will the S&P 500 reach 5000 by end of 2024?",
        max_urls=10,
        max_events=100,
        progress_callback=lambda stage, msg: print(f"{stage}: {msg}")
    )
    print(f"Completed! Run ID: {run_id}")

asyncio.run(main())
```

### Using the Orchestrator Class

For more control:

```python
from pipeline import PipelineOrchestrator, RunContext
from db.repository import DatabaseRepository
from db.models import Question, Run, RunStatus

repo = DatabaseRepository()
orchestrator = PipelineOrchestrator(repo)

# Create question and run
question = Question(question_text="Your question")
question_id = repo.create_question(question)

run = Run(question_id=question_id, status=RunStatus.PENDING)
run_id = repo.create_run(run)

# Create context
ctx = RunContext(
    question_id=question_id,
    run_id=run_id,
    question_text="Your question",
    verbose=True
)

# Run pipeline
await orchestrator.run_pipeline(ctx)
```

## Troubleshooting

### "No module named 'typer'" or "No module named 'rich'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "API key not found" or "Invalid API key"

Check your `.env` file has all required keys:
```bash
cat .env
```

Required keys:
- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`
- `GOOGLE_GEMINI_API_KEY`
- `OPENAI_API_KEY`

### "Database is locked"

SQLite doesn't support concurrent writes. Only run one pipeline at a time.

### "Rate limit exceeded"

Google/OpenAI API rate limits hit. Wait a few minutes and use `resume`:
```bash
python cli.py resume <run_id>
```

### Pipeline hangs at SCRAPE stage

Some websites may be slow or unresponsive. The scraper has a 30-second timeout per URL. Increase `max_concurrent_scrapes` in settings if needed.

## Advanced Configuration

Edit `config/settings.py` or set environment variables:

```bash
# Scraping
export MAX_CONCURRENT_SCRAPES=10
export REQUEST_TIMEOUT=60

# LLM
export GEMINI_MODEL=gemini-2.0-flash-exp
export OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Search
export MAX_SEARCH_QUERIES=15
export SEARCH_RESULTS_PER_QUERY=10

# Clustering
export CLUSTERING_THRESHOLD=0.15
```

## Next Steps

- See [TECHNICAL.md](META/TECHNICAL.md) for architecture details
- See [PRODUCT.md](META/PRODUCT.md) for product vision
- See [PROGRESS.md](META/PROGRESS.md) for development status
- Check the [examples/](examples/) directory for sample runs
