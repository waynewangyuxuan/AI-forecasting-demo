#!/usr/bin/env python3
"""
CLI Interface for AI Forecasting Pipeline.

Provides commands for running forecasts, checking status, generating reports,
and managing the pipeline with rich terminal formatting.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
import structlog

from config.settings import settings
from db.repository import DatabaseRepository
from db.models import RunStatus, Question, Run
from pipeline.orchestrator import run_forecast_pipeline

# Initialize CLI app
app = typer.Typer(
    name="forecast",
    help="AI Forecasting Pipeline - Generate evidence-based forecasts from web data",
    add_completion=False
)

# Rich console for colored output
console = Console()

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def _print_status(message: str, status: str = "info") -> None:
    """
    Print a status message with color coding.

    Args:
        message: Message to print
        status: Status type (info, success, warning, error)
    """
    colors = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red"
    }
    color = colors.get(status, "white")
    prefix = {
        "info": "[INFO]",
        "success": "[SUCCESS]",
        "warning": "[WARNING]",
        "error": "[ERROR]"
    }
    console.print(f"[{color}]{prefix.get(status, '')}[/{color}] {message}")


@app.command()
def run(
    question: str = typer.Argument(..., help="The forecasting question to answer"),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: outputs/run_<id>)"
    ),
    max_urls: Optional[int] = typer.Option(
        None,
        "--max-urls",
        help="Maximum URLs to scrape per query (default: 10)"
    ),
    max_events: Optional[int] = typer.Option(
        None,
        "--max-events",
        help="Maximum events to extract (optional)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without saving to database"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed logs (disables progress bar)"
    )
):
    """
    Run the full forecasting pipeline on a question.

    Example:
        forecast run "Will the S&P 500 reach 5000 by end of 2024?"
    """
    console.print(Panel.fit(
        f"[bold cyan]AI Forecasting Pipeline[/bold cyan]\n\n"
        f"Question: [bold]{question}[/bold]",
        border_style="cyan"
    ))

    # Progress tracking
    current_stage = {"name": "INIT", "message": "Starting..."}

    def progress_callback(stage: str, message: str):
        current_stage["name"] = stage
        current_stage["message"] = message

    try:
        if verbose:
            # Verbose mode: no progress bar, show all logs
            console.print("[yellow]Running in verbose mode - all logs will be displayed[/yellow]")
            console.print()

            async def run_verbose():
                def callback(stage: str, message: str):
                    console.print(f"[cyan]{stage}[/cyan]: {message}")

                run_id = await run_forecast_pipeline(
                    question_text=question,
                    max_urls=max_urls,
                    max_events=max_events,
                    dry_run=dry_run,
                    verbose=verbose,
                    output_dir=output,
                    progress_callback=callback
                )
                return run_id

            run_id = asyncio.run(run_verbose())
        else:
            # Normal mode: with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Running pipeline...", total=8)

                # Wrapper to update progress
                async def run_with_progress():
                    def callback(stage: str, message: str):
                        progress_callback(stage, message)
                        progress.update(task, description=f"[cyan]{stage}[/cyan]: {message}")
                        # Update progress based on stage
                        stage_map = {
                            "INIT": 0, "QUERY_GEN": 1, "SEARCH": 2, "SCRAPE": 3,
                            "EVENT_EXTRACT": 4, "CLUSTER": 5, "TIMELINE": 6, "FORECAST": 7
                        }
                        if stage in stage_map:
                            progress.update(task, completed=stage_map[stage])

                    run_id = await run_forecast_pipeline(
                        question_text=question,
                        max_urls=max_urls,
                        max_events=max_events,
                        dry_run=dry_run,
                        verbose=verbose,
                        output_dir=output,
                        progress_callback=callback
                    )
                    progress.update(task, completed=8)
                    return run_id

                # Run pipeline
                run_id = asyncio.run(run_with_progress())

        # Success!
        console.print()
        _print_status(f"Pipeline completed successfully!", "success")
        _print_status(f"Run ID: {run_id}", "info")

        # Show summary if not dry run
        if not dry_run:
            console.print()
            show_summary(run_id)

    except KeyboardInterrupt:
        console.print()
        _print_status("Pipeline interrupted by user", "warning")
        sys.exit(1)
    except Exception as e:
        console.print()
        _print_status(f"Pipeline failed: {str(e)}", "error")
        logger.exception("pipeline_failed")
        sys.exit(1)


@app.command()
def resume(
    run_id: int = typer.Argument(..., help="The run ID to resume"),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Resume a failed or interrupted run from last checkpoint.

    Example:
        forecast resume 42
    """
    console.print(f"[cyan]Resuming run {run_id}...[/cyan]")

    try:
        repo = DatabaseRepository()
        run = repo.get_run_by_id(run_id)

        if not run:
            _print_status(f"Run {run_id} not found", "error")
            sys.exit(1)

        if run.status == RunStatus.COMPLETED:
            _print_status(f"Run {run_id} already completed", "warning")
            return

        question = repo.get_question_by_id(run.question_id)

        # Progress tracking
        def progress_callback(stage: str, message: str):
            console.print(f"[cyan]{stage}[/cyan]: {message}")

        # Run pipeline with resume flag
        asyncio.run(run_forecast_pipeline(
            question_text=question.question_text,
            resume=True,
            verbose=verbose,
            progress_callback=progress_callback
        ))

        _print_status(f"Run {run_id} resumed successfully!", "success")

    except Exception as e:
        _print_status(f"Resume failed: {str(e)}", "error")
        logger.exception("resume_failed")
        sys.exit(1)


@app.command()
def status(
    run_id: int = typer.Argument(..., help="The run ID to check")
):
    """
    Show status and progress of a run.

    Example:
        forecast status 42
    """
    try:
        repo = DatabaseRepository()
        run = repo.get_run_by_id(run_id)

        if not run:
            _print_status(f"Run {run_id} not found", "error")
            sys.exit(1)

        question = repo.get_question_by_id(run.question_id)

        # Parse stage state
        completed_stages = []
        current_stage = None
        if run.stage_state:
            try:
                stage_state = json.loads(run.stage_state)
                completed_stages = stage_state.get('completed_stages', [])
                current_stage = stage_state.get('current_stage')
            except json.JSONDecodeError:
                pass

        # Create status table
        table = Table(title=f"Run {run_id} Status", show_header=True, header_style="bold cyan")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Question", question.question_text)
        table.add_row("Status", f"[{'green' if run.status == RunStatus.COMPLETED else 'yellow'}]{run.status.value}[/]")
        table.add_row("Started", run.started_at.strftime('%Y-%m-%d %H:%M:%S') if run.started_at else "N/A")
        table.add_row("Completed", run.completed_at.strftime('%Y-%m-%d %H:%M:%S') if run.completed_at else "N/A")
        table.add_row("Current Stage", current_stage or "N/A")
        table.add_row("Completed Stages", ", ".join(completed_stages) if completed_stages else "None")

        console.print(table)

        # Show errors if any
        errors = repo.get_errors_by_run(run_id)
        if errors:
            console.print()
            error_table = Table(title="Errors", show_header=True, header_style="bold red")
            error_table.add_column("Stage", style="yellow")
            error_table.add_column("Type", style="red")
            error_table.add_column("Message", style="white")

            for error in errors[:10]:  # Show first 10 errors
                error_table.add_row(
                    error.stage or "N/A",
                    error.error_type or "N/A",
                    error.message[:80] + "..." if error.message and len(error.message) > 80 else error.message or "N/A"
                )

            console.print(error_table)
            if len(errors) > 10:
                console.print(f"[yellow]... and {len(errors) - 10} more errors[/yellow]")

    except Exception as e:
        _print_status(f"Failed to get status: {str(e)}", "error")
        sys.exit(1)


@app.command("list")
def list_runs(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of runs to show")
):
    """
    List all runs with their status.

    Example:
        forecast list --limit 10
    """
    try:
        repo = DatabaseRepository()
        questions = repo.get_all_questions()

        if not questions:
            _print_status("No runs found", "info")
            return

        # Create table
        table = Table(title="All Runs", show_header=True, header_style="bold cyan")
        table.add_column("Run ID", style="cyan", justify="right")
        table.add_column("Question ID", style="cyan", justify="right")
        table.add_column("Question", style="white", max_width=50)
        table.add_column("Status", style="yellow")
        table.add_column("Started", style="white")

        count = 0
        for question in questions:
            runs = repo.get_runs_by_question(question.id)
            for run in runs:
                if count >= limit:
                    break

                status_color = {
                    RunStatus.COMPLETED: "green",
                    RunStatus.RUNNING: "yellow",
                    RunStatus.FAILED: "red",
                    RunStatus.PENDING: "blue"
                }.get(run.status, "white")

                table.add_row(
                    str(run.id),
                    str(question.id),
                    question.question_text[:47] + "..." if len(question.question_text) > 50 else question.question_text,
                    f"[{status_color}]{run.status.value}[/{status_color}]",
                    run.started_at.strftime('%Y-%m-%d %H:%M') if run.started_at else "N/A"
                )
                count += 1

            if count >= limit:
                break

        console.print(table)

    except Exception as e:
        _print_status(f"Failed to list runs: {str(e)}", "error")
        sys.exit(1)


@app.command()
def report(
    run_id: int = typer.Argument(..., help="The run ID to generate report for"),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format: markdown or json"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (prints to console if not specified)"
    )
):
    """
    Generate a detailed report for a completed run.

    Example:
        forecast report 42 --format markdown
        forecast report 42 --format json --output report.json
    """
    try:
        repo = DatabaseRepository()
        run = repo.get_run_by_id(run_id)

        if not run:
            _print_status(f"Run {run_id} not found", "error")
            sys.exit(1)

        # Check for existing output files
        expected_dir = settings.outputs_dir / f"run_{run_id}"
        if format == "markdown":
            report_file = expected_dir / "forecast_report.md"
        else:
            report_file = expected_dir / "forecast_output.json"

        if report_file.exists():
            # Load existing report
            with open(report_file, 'r') as f:
                content = f.read()

            if output:
                output.write_text(content)
                _print_status(f"Report saved to {output}", "success")
            else:
                if format == "markdown":
                    console.print(Markdown(content))
                else:
                    console.print_json(content)
        else:
            _print_status(f"Report not found at {report_file}", "warning")
            _print_status("Run might not be completed or outputs not generated", "info")

    except Exception as e:
        _print_status(f"Failed to generate report: {str(e)}", "error")
        sys.exit(1)


@app.command()
def init():
    """
    Initialize the database.

    Creates all necessary tables and directories.

    Example:
        forecast init
    """
    try:
        console.print("[cyan]Initializing database...[/cyan]")

        # Ensure data directory exists
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        # Ensure outputs directory exists
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize repository (will create tables)
        from db.migrate import migrate
        migrate()

        _print_status("Database initialized successfully!", "success")
        _print_status(f"Database location: {settings.data_dir / 'forecast.db'}", "info")
        _print_status(f"Outputs directory: {settings.outputs_dir}", "info")

    except Exception as e:
        _print_status(f"Initialization failed: {str(e)}", "error")
        sys.exit(1)


def show_summary(run_id: int):
    """
    Show a summary of the completed run.

    Args:
        run_id: Run ID
    """
    try:
        repo = DatabaseRepository()
        metrics = repo.get_metrics_by_run(run_id)
        forecast = repo.get_forecast_by_run(run_id)

        # Create metrics table
        table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")

        # Organize metrics by stage
        stage_metrics = {}
        for metric in metrics:
            if '_' in metric.metric_name:
                stage = metric.metric_name.split('_')[0]
                metric_name = '_'.join(metric.metric_name.split('_')[1:])
            else:
                stage = "other"
                metric_name = metric.metric_name

            if stage not in stage_metrics:
                stage_metrics[stage] = {}
            stage_metrics[stage][metric_name] = metric.metric_value

        # Display key metrics
        for stage in ["QUERY", "SEARCH", "SCRAPE", "EVENT", "CLUSTER", "TIMELINE"]:
            if stage in stage_metrics:
                for key, value in stage_metrics[stage].items():
                    if 'count' in key or 'total' in key:
                        table.add_row(f"{stage} {key}", f"{int(value)}")
                    elif 'duration' in key:
                        table.add_row(f"{stage} {key}", f"{value:.2f}s")

        console.print()
        console.print(table)

        # Show forecast
        if forecast:
            console.print()
            console.print(Panel.fit(
                f"[bold]Probability:[/bold] {forecast.probability}\n\n"
                f"[bold]Reasoning:[/bold]\n{forecast.reasoning[:200]}...",
                title="Forecast Preview",
                border_style="green"
            ))

            console.print()
            _print_status(f"Full report: outputs/run_{run_id}/forecast_report.md", "info")

    except Exception as e:
        logger.warning("failed_to_show_summary", error=str(e))


@app.command()
def version():
    """
    Show version information.
    """
    console.print(Panel.fit(
        "[bold cyan]AI Forecasting Pipeline[/bold cyan]\n\n"
        "Version: 0.1.0\n"
        "Python: 3.11+",
        border_style="cyan"
    ))


if __name__ == "__main__":
    app()
