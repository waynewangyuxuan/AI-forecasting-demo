# Examples

This directory contains example scripts demonstrating various usage patterns for the AI Forecasting Pipeline.

## Files

### `basic_usage.py`

Demonstrates fundamental usage patterns:

- **Simple Forecast**: Default settings, full pipeline
- **Limited Forecast**: Reduced data collection (faster, cheaper)
- **Dry Run**: Test without database changes
- **Custom Output**: Save results to custom directory
- **Batch Processing**: Process multiple questions sequentially

**Run:**
```bash
python examples/basic_usage.py
```

### `advanced_usage.py`

Shows advanced features and patterns:

- **Custom Service Configuration**: Direct orchestrator usage
- **Resume Failed Runs**: Checkpoint recovery
- **Metrics Analysis**: Detailed performance analysis
- **Run Comparison**: Compare multiple forecasts

**Run:**
```bash
python examples/advanced_usage.py
```

## Prerequisites

Before running examples, ensure:

1. **Dependencies installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **API keys configured:**
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

3. **Database initialized:**
   ```bash
   python cli.py init
   ```

## Example Scenarios

### Scenario 1: Quick Test

Test the pipeline with minimal data collection:

```python
import asyncio
from pipeline import run_forecast_pipeline

async def quick_test():
    run_id = await run_forecast_pipeline(
        question_text="Will Bitcoin reach $50k by June 2024?",
        max_urls=3,
        max_events=20,
        dry_run=True  # Don't save to database
    )
    print(f"Test completed: {run_id}")

asyncio.run(quick_test())
```

### Scenario 2: Production Forecast

Run a full production forecast with monitoring:

```python
import asyncio
from pipeline import run_forecast_pipeline

class ProgressMonitor:
    def __init__(self):
        self.stages_completed = []

    def callback(self, stage, message):
        if stage not in self.stages_completed:
            print(f"✓ Started {stage}")
            self.stages_completed.append(stage)
        print(f"  {message}")

async def production_forecast():
    monitor = ProgressMonitor()

    run_id = await run_forecast_pipeline(
        question_text="Will Apple release new iPad in Q1 2024?",
        verbose=True,
        progress_callback=monitor.callback
    )

    print(f"Completed: Run {run_id}")
    print(f"Stages: {', '.join(monitor.stages_completed)}")

asyncio.run(production_forecast())
```

### Scenario 3: Error Recovery

Handle errors and retry:

```python
import asyncio
from pipeline import run_forecast_pipeline

async def forecast_with_retry(question, max_retries=3):
    for attempt in range(max_retries):
        try:
            run_id = await run_forecast_pipeline(
                question_text=question,
                resume=True if attempt > 0 else False
            )
            return run_id
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            print("Retrying...")

asyncio.run(forecast_with_retry("Your question"))
```

### Scenario 4: Batch Processing with Progress Tracking

Process multiple questions with detailed tracking:

```python
import asyncio
from pipeline import run_forecast_pipeline

async def batch_with_tracking(questions):
    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(questions)}: {question[:50]}...")
        print(f"{'='*60}")

        try:
            run_id = await run_forecast_pipeline(
                question_text=question,
                progress_callback=lambda s, m: print(f"[{s}] {m}")
            )
            results.append({'question': question, 'run_id': run_id, 'success': True})
        except Exception as e:
            results.append({'question': question, 'error': str(e), 'success': False})

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*60}")
    print(f"Batch Complete: {successful}/{len(questions)} successful")
    print(f"{'='*60}")

    return results

questions = [
    "Will Tesla stock reach $300 by June 2024?",
    "Will inflation fall below 2% in 2024?",
]

asyncio.run(batch_with_tracking(questions))
```

### Scenario 5: Custom Orchestrator

Use the orchestrator directly for maximum control:

```python
import asyncio
from pipeline import PipelineOrchestrator, RunContext
from db.repository import DatabaseRepository
from db.models import Question, Run, RunStatus

async def custom_orchestrator_example():
    # Initialize
    repo = DatabaseRepository()
    orchestrator = PipelineOrchestrator(repo)

    # Create question
    question = Question(
        question_text="Will quantum computing be viable by 2025?",
        resolution_criteria="Commercial quantum computer available"
    )
    question_id = repo.create_question(question)

    # Create run
    run = Run(question_id=question_id, status=RunStatus.PENDING)
    run_id = repo.create_run(run)

    # Create context with custom settings
    ctx = RunContext(
        question_id=question_id,
        run_id=run_id,
        question_text=question.question_text,
        max_urls=5,
        max_events=30,
        verbose=True
    )

    # Run pipeline with custom context
    await orchestrator.run_pipeline(ctx)

    # Access results
    forecast = repo.get_forecast_by_run(run_id)
    print(f"Probability: {forecast.probability}")

    return run_id

asyncio.run(custom_orchestrator_example())
```

## Output Examples

### Console Output

```
==============================================================
Question: Will Bitcoin reach $50,000 by June 2024?
==============================================================

[INIT] Initializing pipeline
[QUERY_GEN] Generating search queries for: Will Bitcoin...
[SEARCH] Searching 10 queries
[SEARCH] Query 1/10: Bitcoin price forecast 2024...
[SCRAPE] Scraping 85 unique URLs
[EVENT_EXTRACT] Extracting events from 42 documents
[CLUSTER] Generating embeddings for 127 events
[CLUSTER] Clustering 127 events
[TIMELINE] Building timeline
[FORECAST] Generating forecast

✓ Completed! Run ID: 1
  Report: outputs/run_1/forecast_report.md
  JSON: outputs/run_1/forecast_output.json
```

### JSON Output Structure

```json
{
  "question": {
    "id": 1,
    "text": "Will Bitcoin reach $50,000 by June 2024?"
  },
  "forecast": {
    "probability": 0.42,
    "reasoning": "Based on historical volatility...",
    "caveats": ["Market uncertainty", "Limited predictability"]
  },
  "timeline": [
    {
      "event_time": "2024-01-15",
      "summary": "Bitcoin trading at $45,000...",
      "citations": [...]
    }
  ],
  "metrics": {
    "QUERY_GEN_duration_seconds": 8.2,
    "SEARCH_total_urls": 85,
    "EVENT_total_events": 127
  }
}
```

## Best Practices

### 1. Always Use Progress Callbacks

Monitor pipeline execution:

```python
def progress(stage, message):
    print(f"[{datetime.now().isoformat()}] {stage}: {message}")

await run_forecast_pipeline(
    question_text="Question",
    progress_callback=progress
)
```

### 2. Handle Errors Gracefully

```python
try:
    run_id = await run_forecast_pipeline(question_text="Question")
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    # Try to resume
    if run_id:
        await resume_failed_run(run_id)
```

### 3. Use Dry Run for Testing

```python
# Test without database changes
await run_forecast_pipeline(
    question_text="Test question",
    dry_run=True,
    verbose=True
)
```

### 4. Limit Data for Fast Iteration

```python
# Quick tests with limited data
await run_forecast_pipeline(
    question_text="Question",
    max_urls=3,
    max_events=20
)
```

### 5. Monitor Metrics

```python
from db.repository import DatabaseRepository

repo = DatabaseRepository()
metrics = repo.get_metrics_by_run(run_id)
for metric in metrics:
    print(f"{metric.metric_name}: {metric.metric_value}")
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd /path/to/AI-forecasting-demo
python examples/basic_usage.py
```

Or add project to PYTHONPATH:

```bash
export PYTHONPATH=/path/to/AI-forecasting-demo:$PYTHONPATH
python examples/basic_usage.py
```

### API Rate Limits

If you hit API rate limits, use smaller limits or retry later:

```python
await run_forecast_pipeline(
    question_text="Question",
    max_urls=3,  # Fewer API calls
    max_events=20
)
```

### Database Locked

Only run one pipeline at a time with SQLite:

```bash
# Check for running pipelines
ps aux | grep "python.*cli.py"
```

## Further Reading

- [USAGE.md](../USAGE.md) - Complete CLI documentation
- [TECHNICAL.md](../META/TECHNICAL.md) - Technical architecture
- [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Implementation details

## Contributing

To add new examples:

1. Create a new Python file in this directory
2. Follow the existing structure
3. Include docstrings and comments
4. Add entry to this README
5. Test thoroughly before committing
