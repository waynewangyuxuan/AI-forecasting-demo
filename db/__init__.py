"""
Database package for AI Forecasting Pipeline.

This package handles all database operations including:
- Schema definitions
- Repository layer for data access
- Database migrations and initialization

Main exports:
    - DatabaseRepository: Main repository class for database operations
    - All Pydantic models: Question, Run, SearchQuery, etc.
    - RunStatus: Enum for run status values
    - migrate: Database migration function
"""

# Import models
from db.models import (
    Question,
    Run,
    RunStatus,
    SearchQuery,
    SearchResult,
    Document,
    Event,
    Embedding,
    EventCluster,
    TimelineEntry,
    Forecast,
    RunMetric,
    Error
)

# Import repository
from db.repository import DatabaseRepository

# Import migration function
from db.migrate import migrate, reset_database, get_database_info

# Define public API
__all__ = [
    # Repository
    'DatabaseRepository',

    # Models
    'Question',
    'Run',
    'RunStatus',
    'SearchQuery',
    'SearchResult',
    'Document',
    'Event',
    'Embedding',
    'EventCluster',
    'TimelineEntry',
    'Forecast',
    'RunMetric',
    'Error',

    # Migration functions
    'migrate',
    'reset_database',
    'get_database_info',
]
