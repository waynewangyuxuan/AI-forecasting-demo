#!/usr/bin/env python3
"""
Basic usage example for the AI Forecasting Pipeline.

This script demonstrates how to use the orchestrator programmatically
to generate forecasts.
"""

import asyncio
from pathlib import Path
from pipeline import run_forecast_pipeline

# Example questions
EXAMPLE_QUESTIONS = [
    "Will the S&P 500 reach 5000 by the end of 2024?",
    "Will Tesla's stock price exceed $300 by June 2024?",
    "Will Apple release a new iPad model in Q1 2024?",
]


async def simple_forecast(question: str):
    """
    Run a simple forecast with default settings.

    Args:
        question: The forecasting question

    Returns:
        Run ID
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    def progress_callback(stage: str, message: str):
        """Print progress updates."""
        print(f"[{stage}] {message}")

    try:
        run_id = await run_forecast_pipeline(
            question_text=question,
            progress_callback=progress_callback
        )

        print(f"\n✓ Completed! Run ID: {run_id}")
        print(f"  Report: outputs/run_{run_id}/forecast_report.md")
        print(f"  JSON: outputs/run_{run_id}/forecast_output.json")

        return run_id

    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        return None


async def limited_forecast(question: str, max_urls: int = 5, max_events: int = 50):
    """
    Run a forecast with limited data collection (faster, cheaper).

    Args:
        question: The forecasting question
        max_urls: Maximum URLs to scrape per query
        max_events: Maximum events to extract

    Returns:
        Run ID
    """
    print(f"\n{'='*60}")
    print(f"Limited Forecast (max_urls={max_urls}, max_events={max_events})")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    def progress_callback(stage: str, message: str):
        print(f"[{stage}] {message}")

    try:
        run_id = await run_forecast_pipeline(
            question_text=question,
            max_urls=max_urls,
            max_events=max_events,
            progress_callback=progress_callback
        )

        print(f"\n✓ Completed! Run ID: {run_id}")
        return run_id

    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        return None


async def dry_run_forecast(question: str):
    """
    Run a dry-run forecast (no database changes).

    Useful for testing without persisting data.

    Args:
        question: The forecasting question

    Returns:
        Run ID (dummy)
    """
    print(f"\n{'='*60}")
    print(f"Dry Run (no database changes)")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    def progress_callback(stage: str, message: str):
        print(f"[{stage}] {message}")

    try:
        run_id = await run_forecast_pipeline(
            question_text=question,
            dry_run=True,
            verbose=True,
            progress_callback=progress_callback
        )

        print(f"\n✓ Dry run completed! (No data saved)")
        return run_id

    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        return None


async def custom_output_forecast(question: str, output_dir: Path):
    """
    Run a forecast with custom output directory.

    Args:
        question: The forecasting question
        output_dir: Custom output directory path

    Returns:
        Run ID
    """
    print(f"\n{'='*60}")
    print(f"Custom Output Directory: {output_dir}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    def progress_callback(stage: str, message: str):
        print(f"[{stage}] {message}")

    try:
        run_id = await run_forecast_pipeline(
            question_text=question,
            output_dir=output_dir,
            progress_callback=progress_callback
        )

        print(f"\n✓ Completed! Run ID: {run_id}")
        print(f"  Report: {output_dir}/forecast_report.md")
        print(f"  JSON: {output_dir}/forecast_output.json")

        return run_id

    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        return None


async def batch_forecasts(questions: list[str]):
    """
    Run forecasts for multiple questions sequentially.

    Args:
        questions: List of forecasting questions

    Returns:
        List of run IDs
    """
    print(f"\n{'='*60}")
    print(f"Batch Forecast: {len(questions)} questions")
    print(f"{'='*60}\n")

    run_ids = []

    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}/{len(questions)} ---")
        run_id = await simple_forecast(question)
        if run_id:
            run_ids.append(run_id)

    print(f"\n{'='*60}")
    print(f"Batch Completed: {len(run_ids)}/{len(questions)} successful")
    print(f"{'='*60}")

    return run_ids


async def main():
    """
    Main function demonstrating different usage patterns.
    """
    print("\nAI Forecasting Pipeline - Usage Examples")
    print("=" * 60)

    # Example 1: Simple forecast with default settings
    print("\n\n### Example 1: Simple Forecast ###")
    await simple_forecast(EXAMPLE_QUESTIONS[0])

    # Example 2: Limited data collection (faster, cheaper)
    print("\n\n### Example 2: Limited Forecast ###")
    await limited_forecast(EXAMPLE_QUESTIONS[1], max_urls=3, max_events=20)

    # Example 3: Dry run (no database changes)
    print("\n\n### Example 3: Dry Run ###")
    await dry_run_forecast(EXAMPLE_QUESTIONS[2])

    # Example 4: Custom output directory
    print("\n\n### Example 4: Custom Output Directory ###")
    custom_dir = Path("./my_forecasts/custom_run")
    await custom_output_forecast(EXAMPLE_QUESTIONS[0], custom_dir)

    # Example 5: Batch processing
    print("\n\n### Example 5: Batch Processing ###")
    batch_questions = EXAMPLE_QUESTIONS[:2]  # First 2 questions
    await batch_forecasts(batch_questions)

    print("\n\nAll examples completed!")


if __name__ == "__main__":
    """
    Run the examples.

    Usage:
        python examples/basic_usage.py
    """
    # Note: Make sure your .env file is configured with API keys
    # before running this script.

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
