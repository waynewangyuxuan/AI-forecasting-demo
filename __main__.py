#!/usr/bin/env python3
"""
Entry point for running the AI Forecasting Pipeline as a module.

Usage:
    python -m forecast_pipeline run "Your question here"
    python -m forecast_pipeline status 42
    python -m forecast_pipeline list
"""

from cli import app

if __name__ == "__main__":
    app()
