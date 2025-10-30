#!/usr/bin/env python3
"""
Advanced usage example for the AI Forecasting Pipeline.

This script demonstrates advanced features like:
- Custom service configuration
- Direct orchestrator usage
- Resume functionality
- Metric analysis
- Custom progress tracking
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from pipeline import PipelineOrchestrator, RunContext, PipelineStage
from db.repository import DatabaseRepository
from db.models import Question, Run, RunStatus
from config.settings import settings


class CustomProgressTracker:
    """
    Custom progress tracker with detailed logging and metrics.
    """

    def __init__(self):
        self.stage_times = {}
        self.current_stage = None
        self.start_time = None

    def on_stage_start(self, stage: str):
        """Called when a stage starts."""
        self.current_stage = stage
        self.stage_times[stage] = {'start': datetime.now()}
        print(f"\n→ Starting {stage}")

    def on_stage_end(self, stage: str):
        """Called when a stage ends."""
        if stage in self.stage_times:
            end_time = datetime.now()
            start_time = self.stage_times[stage]['start']
            duration = (end_time - start_time).total_seconds()
            self.stage_times[stage]['end'] = end_time
            self.stage_times[stage]['duration'] = duration
            print(f"✓ Completed {stage} in {duration:.1f}s")

    def on_progress(self, stage: str, message: str):
        """Called on progress updates."""
        if stage != self.current_stage:
            if self.current_stage:
                self.on_stage_end(self.current_stage)
            self.on_stage_start(stage)

        print(f"  [{stage}] {message}")

    def get_summary(self):
        """Get timing summary."""
        total = sum(s.get('duration', 0) for s in self.stage_times.values())
        return {
            'total_duration': total,
            'stages': self.stage_times
        }


async def advanced_forecast_with_custom_services():
    """
    Run a forecast with custom service configuration.
    """
    print("\n" + "="*60)
    print("Advanced Example: Custom Service Configuration")
    print("="*60)

    # Initialize repository
    repo = DatabaseRepository()

    # Create question
    question = Question(
        question_text="Will quantum computing be commercially viable by 2025?",
        resolution_criteria="A quantum computer must be available for commercial purchase"
    )
    question_id = repo.create_question(question)

    # Create run
    run = Run(
        question_id=question_id,
        status=RunStatus.PENDING
    )
    run_id = repo.create_run(run)

    # Create custom progress tracker
    tracker = CustomProgressTracker()

    # Create context with custom settings
    ctx = RunContext(
        question_id=question_id,
        run_id=run_id,
        question_text=question.question_text,
        max_urls=5,  # Limit for faster execution
        max_events=30,
        verbose=True,
        progress_callback=tracker.on_progress
    )

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(repo)

    try:
        # Run pipeline
        await orchestrator.run_pipeline(ctx)

        # Print summary
        summary = tracker.get_summary()
        print(f"\n{'='*60}")
        print(f"Pipeline Summary")
        print(f"{'='*60}")
        print(f"Total Duration: {summary['total_duration']:.1f}s")
        print(f"\nStage Timings:")
        for stage, times in summary['stages'].items():
            if 'duration' in times:
                print(f"  {stage:20s}: {times['duration']:6.1f}s")

        return run_id

    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def resume_failed_run(run_id: int):
    """
    Resume a failed run from the last checkpoint.

    Args:
        run_id: ID of the run to resume
    """
    print("\n" + "="*60)
    print(f"Resuming Run {run_id}")
    print("="*60)

    repo = DatabaseRepository()
    orchestrator = PipelineOrchestrator(repo)

    # Get run details
    run = repo.get_run_by_id(run_id)
    if not run:
        print(f"✗ Run {run_id} not found")
        return None

    question = repo.get_question_by_id(run.question_id)

    # Parse stage state to see what's completed
    completed_stages = []
    if run.stage_state:
        try:
            stage_state = json.loads(run.stage_state)
            completed_stages = stage_state.get('completed_stages', [])
        except json.JSONDecodeError:
            pass

    print(f"\nQuestion: {question.question_text}")
    print(f"Status: {run.status.value}")
    print(f"Completed stages: {', '.join(completed_stages) if completed_stages else 'None'}")

    if run.status == RunStatus.COMPLETED:
        print("\n⚠ Run already completed")
        return run_id

    # Create context with resume flag
    ctx = RunContext(
        question_id=run.question_id,
        run_id=run_id,
        question_text=question.question_text,
        resume=True,  # Enable resume
        verbose=True,
        progress_callback=lambda s, m: print(f"[{s}] {m}")
    )

    try:
        # Resume pipeline
        print("\n→ Resuming from last checkpoint...")
        await orchestrator.run_pipeline(ctx)

        print(f"\n✓ Successfully resumed and completed run {run_id}")
        return run_id

    except Exception as e:
        print(f"\n✗ Resume failed: {str(e)}")
        return None


async def analyze_run_metrics(run_id: int):
    """
    Analyze and display detailed metrics for a completed run.

    Args:
        run_id: ID of the run to analyze
    """
    print("\n" + "="*60)
    print(f"Metrics Analysis: Run {run_id}")
    print("="*60)

    repo = DatabaseRepository()

    # Get run
    run = repo.get_run_by_id(run_id)
    if not run:
        print(f"✗ Run {run_id} not found")
        return

    question = repo.get_question_by_id(run.question_id)

    # Get all metrics
    metrics = repo.get_metrics_by_run(run_id)
    errors = repo.get_errors_by_run(run_id)

    print(f"\nQuestion: {question.question_text}")
    print(f"Status: {run.status.value}")
    print(f"Started: {run.started_at}")
    print(f"Completed: {run.completed_at}")

    if run.started_at and run.completed_at:
        duration = (run.completed_at - run.started_at).total_seconds()
        print(f"Total Duration: {duration:.1f}s ({duration/60:.1f}m)")

    # Organize metrics by stage
    stage_metrics = {}
    for metric in metrics:
        parts = metric.metric_name.split('_')
        if len(parts) >= 2:
            stage = parts[0]
            metric_name = '_'.join(parts[1:])
            if stage not in stage_metrics:
                stage_metrics[stage] = {}
            stage_metrics[stage][metric_name] = metric.metric_value

    # Display stage metrics
    print(f"\n{'Stage Metrics':=^60}")
    for stage in ['QUERY', 'SEARCH', 'SCRAPE', 'EVENT', 'CLUSTER', 'TIMELINE', 'FORECAST']:
        if stage in stage_metrics:
            print(f"\n{stage}:")
            for key, value in sorted(stage_metrics[stage].items()):
                if 'duration' in key:
                    print(f"  {key:30s}: {value:7.2f}s")
                else:
                    print(f"  {key:30s}: {value:7.0f}")

    # Error analysis
    if errors:
        print(f"\n{'Errors':=^60}")
        print(f"Total Errors: {len(errors)}")

        # Group by stage
        errors_by_stage = {}
        for error in errors:
            stage = error.stage or 'UNKNOWN'
            if stage not in errors_by_stage:
                errors_by_stage[stage] = []
            errors_by_stage[stage].append(error)

        for stage, stage_errors in sorted(errors_by_stage.items()):
            print(f"\n{stage}: {len(stage_errors)} errors")
            for error in stage_errors[:3]:  # Show first 3
                print(f"  - {error.error_type}: {error.message[:80]}")
            if len(stage_errors) > 3:
                print(f"  ... and {len(stage_errors) - 3} more")

    # Data statistics
    print(f"\n{'Data Statistics':=^60}")

    queries = repo.get_search_queries_by_run(run_id)
    documents = repo.get_documents_by_run(run_id)
    events = repo.get_events_for_run(run_id)
    clusters = repo.get_event_clusters_by_run(run_id)
    timeline = repo.get_timeline_for_run(run_id)
    forecast = repo.get_forecast_by_run(run_id)

    print(f"Queries Generated:     {len(queries)}")
    print(f"Documents Scraped:     {len(documents)}")
    print(f"Events Extracted:      {len(events)}")
    print(f"Event Clusters:        {len(clusters)}")
    print(f"Timeline Entries:      {len(timeline)}")
    print(f"Forecast Generated:    {'Yes' if forecast else 'No'}")

    if forecast:
        print(f"\nForecast Probability:  {forecast.probability}")
        print(f"Forecast Caveats:      {len(json.loads(forecast.caveats)) if forecast.caveats else 0}")


async def compare_runs(run_ids: list[int]):
    """
    Compare metrics across multiple runs.

    Args:
        run_ids: List of run IDs to compare
    """
    print("\n" + "="*60)
    print(f"Comparing {len(run_ids)} Runs")
    print("="*60)

    repo = DatabaseRepository()

    comparison_data = []

    for run_id in run_ids:
        run = repo.get_run_by_id(run_id)
        if not run:
            print(f"⚠ Run {run_id} not found, skipping")
            continue

        question = repo.get_question_by_id(run.question_id)
        metrics = repo.get_metrics_by_run(run_id)
        forecast = repo.get_forecast_by_run(run_id)
        events = repo.get_events_for_run(run_id)

        duration = None
        if run.started_at and run.completed_at:
            duration = (run.completed_at - run.started_at).total_seconds()

        comparison_data.append({
            'run_id': run_id,
            'question': question.question_text[:50] + "...",
            'status': run.status.value,
            'duration': duration,
            'events': len(events),
            'probability': forecast.probability if forecast else None
        })

    # Display comparison table
    print(f"\n{'Run ID':<8} {'Status':<10} {'Duration':<10} {'Events':<8} {'Probability':<12} {'Question'}")
    print("-" * 100)

    for data in comparison_data:
        duration_str = f"{data['duration']:.1f}s" if data['duration'] else "N/A"
        prob_str = f"{data['probability']:.2f}" if data['probability'] is not None else "N/A"

        print(
            f"{data['run_id']:<8} "
            f"{data['status']:<10} "
            f"{duration_str:<10} "
            f"{data['events']:<8} "
            f"{prob_str:<12} "
            f"{data['question']}"
        )


async def main():
    """
    Run advanced usage examples.
    """
    print("\nAI Forecasting Pipeline - Advanced Usage Examples")
    print("="*60)

    # Example 1: Custom service configuration
    print("\n\n### Example 1: Custom Service Configuration ###")
    run_id_1 = await advanced_forecast_with_custom_services()

    # Example 2: Analyze metrics (uncomment if you have a run to analyze)
    # print("\n\n### Example 2: Analyze Run Metrics ###")
    # if run_id_1:
    #     await analyze_run_metrics(run_id_1)

    # Example 3: Resume failed run (uncomment and provide run_id to test)
    # print("\n\n### Example 3: Resume Failed Run ###")
    # failed_run_id = 5  # Replace with actual failed run ID
    # await resume_failed_run(failed_run_id)

    # Example 4: Compare multiple runs (uncomment if you have runs to compare)
    # print("\n\n### Example 4: Compare Multiple Runs ###")
    # await compare_runs([1, 2, 3])

    print("\n\nAdvanced examples completed!")


if __name__ == "__main__":
    """
    Run the advanced examples.

    Usage:
        python examples/advanced_usage.py
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
