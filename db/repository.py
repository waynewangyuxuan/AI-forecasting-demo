"""
Repository layer for AI Forecasting Pipeline database operations.

This module provides a typed interface for database operations using SQLAlchemy Core.
All operations are transaction-safe and use proper type hints.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text,
    REAL, ForeignKey, CheckConstraint, UniqueConstraint, Index, BLOB
)
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.sql import select, insert, update, delete, and_, or_

from db.models import (
    Question, Run, SearchQuery, SearchResult, Document, Event,
    Embedding, EventCluster, TimelineEntry, Forecast, RunMetric, Error,
    RunStatus
)


class DatabaseRepository:
    """
    Repository class for all database operations.

    Uses SQLAlchemy Core for type-safe database access without ORM overhead.
    Supports transactions, connection pooling, and proper error handling.
    """

    def __init__(self, db_path: str = "data/forecast.db"):
        """
        Initialize the repository with a database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine: Engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.metadata = MetaData()
        self._define_tables()

    def _define_tables(self):
        """Define SQLAlchemy table objects for all database tables."""
        self.questions = Table(
            'Questions', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('question_text', Text, nullable=False),
            Column('resolution_criteria', Text),
            Column('created_at', Text)
        )

        self.runs = Table(
            'Runs', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('question_id', Integer, ForeignKey('Questions.id'), nullable=False),
            Column('started_at', Text),
            Column('completed_at', Text),
            Column('status', Text, CheckConstraint("status IN ('PENDING','RUNNING','FAILED','COMPLETED')")),
            Column('stage_state', Text),
            Column('git_commit', Text)
        )

        self.search_queries = Table(
            'SearchQueries', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', Integer, ForeignKey('Runs.id'), nullable=False),
            Column('query_text', Text, nullable=False),
            Column('prompt_version', Text),
            Column('created_at', Text),
            UniqueConstraint('run_id', 'query_text')
        )

        self.search_results = Table(
            'SearchResults', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('query_id', Integer, ForeignKey('SearchQueries.id'), nullable=False),
            Column('url', Text, nullable=False),
            Column('title', Text),
            Column('snippet', Text),
            Column('rank', Integer),
            UniqueConstraint('query_id', 'url')
        )

        self.documents = Table(
            'Documents', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', Integer, ForeignKey('Runs.id'), nullable=False),
            Column('url', Text, nullable=False),
            Column('content_hash', Text, nullable=False),
            Column('raw_content', Text),
            Column('cleaned_content', Text),
            Column('fetched_at', Text),
            Column('status', Text),
            UniqueConstraint('run_id', 'content_hash')
        )

        self.events = Table(
            'Events', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('document_id', Integer, ForeignKey('Documents.id'), nullable=False),
            Column('event_time', Text),
            Column('headline', Text),
            Column('body', Text),
            Column('actors', Text),
            Column('confidence', REAL),
            Column('raw_response', Text)
        )

        self.embeddings = Table(
            'Embeddings', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('event_id', Integer, ForeignKey('Events.id'), nullable=False),
            Column('vector', BLOB, nullable=False),
            Column('model', Text),
            Column('dimensions', Integer),
            Column('created_at', Text),
            UniqueConstraint('event_id', 'model')
        )

        self.event_clusters = Table(
            'EventClusters', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', Integer, ForeignKey('Runs.id'), nullable=False),
            Column('label', Text),
            Column('centroid_event_id', Integer),
            Column('member_ids', Text)
        )

        self.timeline = Table(
            'Timeline', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', Integer, ForeignKey('Runs.id'), nullable=False),
            Column('cluster_id', Integer, ForeignKey('EventClusters.id')),
            Column('event_time', Text),
            Column('summary', Text),
            Column('citations', Text),
            Column('tags', Text)
        )

        self.forecasts = Table(
            'Forecasts', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', Integer, ForeignKey('Runs.id'), nullable=False),
            Column('probability', REAL),
            Column('reasoning', Text),
            Column('caveats', Text),
            Column('raw_response', Text),
            Column('created_at', Text)
        )

        self.run_metrics = Table(
            'RunMetrics', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', Integer, ForeignKey('Runs.id'), nullable=False),
            Column('metric_name', Text),
            Column('metric_value', REAL)
        )

        self.errors = Table(
            'Errors', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', Integer, ForeignKey('Runs.id'), nullable=False),
            Column('stage', Text),
            Column('reference', Text),
            Column('error_type', Text),
            Column('message', Text),
            Column('created_at', Text)
        )

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            Connection: SQLAlchemy connection object
        """
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Automatically commits on success or rolls back on error.

        Yields:
            Connection: SQLAlchemy connection object in transaction
        """
        with self.get_connection() as conn:
            trans = conn.begin()
            try:
                yield conn
                trans.commit()
            except Exception:
                trans.rollback()
                raise

    # Helper methods for datetime conversion
    @staticmethod
    def _to_db_datetime(dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to database string format."""
        return dt.isoformat() if dt else None

    @staticmethod
    def _from_db_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """Convert database string to datetime."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except ValueError:
            return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

    # Questions CRUD operations
    def create_question(self, question: Question) -> int:
        """
        Create a new question.

        Args:
            question: Question model instance

        Returns:
            int: ID of the created question
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.questions).values(
                    question_text=question.question_text,
                    resolution_criteria=question.resolution_criteria,
                    created_at=self._to_db_datetime(question.created_at or datetime.now())
                )
            )
            return result.lastrowid

    def get_question_by_id(self, question_id: int) -> Optional[Question]:
        """
        Get a question by ID.

        Args:
            question_id: Question ID

        Returns:
            Question model or None if not found
        """
        with self.get_connection() as conn:
            result = conn.execute(
                select(self.questions).where(self.questions.c.id == question_id)
            ).fetchone()

            if result:
                return Question(
                    id=result.id,
                    question_text=result.question_text,
                    resolution_criteria=result.resolution_criteria,
                    created_at=self._from_db_datetime(result.created_at)
                )
            return None

    def get_all_questions(self) -> List[Question]:
        """
        Get all questions.

        Returns:
            List of Question models
        """
        with self.get_connection() as conn:
            results = conn.execute(select(self.questions)).fetchall()
            return [
                Question(
                    id=row.id,
                    question_text=row.question_text,
                    resolution_criteria=row.resolution_criteria,
                    created_at=self._from_db_datetime(row.created_at)
                )
                for row in results
            ]

    # Runs CRUD operations
    def create_run(self, run: Run) -> int:
        """
        Create a new run.

        Args:
            run: Run model instance

        Returns:
            int: ID of the created run
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.runs).values(
                    question_id=run.question_id,
                    started_at=self._to_db_datetime(run.started_at or datetime.now()),
                    completed_at=self._to_db_datetime(run.completed_at),
                    status=run.status.value,
                    stage_state=run.stage_state,
                    git_commit=run.git_commit
                )
            )
            return result.lastrowid

    def get_run_by_id(self, run_id: int) -> Optional[Run]:
        """
        Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            Run model or None if not found
        """
        with self.get_connection() as conn:
            result = conn.execute(
                select(self.runs).where(self.runs.c.id == run_id)
            ).fetchone()

            if result:
                return Run(
                    id=result.id,
                    question_id=result.question_id,
                    started_at=self._from_db_datetime(result.started_at),
                    completed_at=self._from_db_datetime(result.completed_at),
                    status=RunStatus(result.status),
                    stage_state=result.stage_state,
                    git_commit=result.git_commit
                )
            return None

    def update_run_status(self, run_id: int, status: RunStatus,
                         completed_at: Optional[datetime] = None,
                         stage_state: Optional[str] = None) -> None:
        """
        Update run status and optionally completion time and stage state.

        Args:
            run_id: Run ID
            status: New status
            completed_at: Completion timestamp (if completed)
            stage_state: Current stage state JSON
        """
        with self.transaction() as conn:
            values = {'status': status.value}
            if completed_at:
                values['completed_at'] = self._to_db_datetime(completed_at)
            if stage_state is not None:
                values['stage_state'] = stage_state

            conn.execute(
                update(self.runs).where(self.runs.c.id == run_id).values(**values)
            )

    def get_runs_by_question(self, question_id: int) -> List[Run]:
        """
        Get all runs for a specific question.

        Args:
            question_id: Question ID

        Returns:
            List of Run models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.runs).where(self.runs.c.question_id == question_id)
                .order_by(self.runs.c.started_at.desc())
            ).fetchall()

            return [
                Run(
                    id=row.id,
                    question_id=row.question_id,
                    started_at=self._from_db_datetime(row.started_at),
                    completed_at=self._from_db_datetime(row.completed_at),
                    status=RunStatus(row.status),
                    stage_state=row.stage_state,
                    git_commit=row.git_commit
                )
                for row in results
            ]

    # SearchQueries CRUD operations
    def create_search_query(self, query: SearchQuery) -> int:
        """
        Create a new search query.

        Args:
            query: SearchQuery model instance

        Returns:
            int: ID of the created search query
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.search_queries).values(
                    run_id=query.run_id,
                    query_text=query.query_text,
                    prompt_version=query.prompt_version,
                    created_at=self._to_db_datetime(query.created_at or datetime.now())
                )
            )
            return result.lastrowid

    def get_search_queries_by_run(self, run_id: int) -> List[SearchQuery]:
        """
        Get all search queries for a run.

        Args:
            run_id: Run ID

        Returns:
            List of SearchQuery models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.search_queries).where(self.search_queries.c.run_id == run_id)
            ).fetchall()

            return [
                SearchQuery(
                    id=row.id,
                    run_id=row.run_id,
                    query_text=row.query_text,
                    prompt_version=row.prompt_version,
                    created_at=self._from_db_datetime(row.created_at)
                )
                for row in results
            ]

    # SearchResults CRUD operations
    def create_search_result(self, result: SearchResult) -> int:
        """
        Create a new search result.

        Args:
            result: SearchResult model instance

        Returns:
            int: ID of the created search result
        """
        with self.transaction() as conn:
            res = conn.execute(
                insert(self.search_results).values(
                    query_id=result.query_id,
                    url=result.url,
                    title=result.title,
                    snippet=result.snippet,
                    rank=result.rank
                )
            )
            return res.lastrowid

    def get_search_results_by_query(self, query_id: int) -> List[SearchResult]:
        """
        Get all search results for a query.

        Args:
            query_id: Query ID

        Returns:
            List of SearchResult models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.search_results).where(self.search_results.c.query_id == query_id)
                .order_by(self.search_results.c.rank)
            ).fetchall()

            return [
                SearchResult(
                    id=row.id,
                    query_id=row.query_id,
                    url=row.url,
                    title=row.title,
                    snippet=row.snippet,
                    rank=row.rank
                )
                for row in results
            ]

    # Documents CRUD operations
    def create_document(self, document: Document) -> int:
        """
        Create a new document.

        Args:
            document: Document model instance

        Returns:
            int: ID of the created document
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.documents).values(
                    run_id=document.run_id,
                    url=document.url,
                    content_hash=document.content_hash,
                    raw_content=document.raw_content,
                    cleaned_content=document.cleaned_content,
                    fetched_at=self._to_db_datetime(document.fetched_at or datetime.now()),
                    status=document.status
                )
            )
            return result.lastrowid

    def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document model or None if not found
        """
        with self.get_connection() as conn:
            result = conn.execute(
                select(self.documents).where(self.documents.c.id == document_id)
            ).fetchone()

            if result:
                return Document(
                    id=result.id,
                    run_id=result.run_id,
                    url=result.url,
                    content_hash=result.content_hash,
                    raw_content=result.raw_content,
                    cleaned_content=result.cleaned_content,
                    fetched_at=self._from_db_datetime(result.fetched_at),
                    status=result.status
                )
            return None

    def get_documents_by_run(self, run_id: int) -> List[Document]:
        """
        Get all documents for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Document models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.documents).where(self.documents.c.run_id == run_id)
            ).fetchall()

            return [
                Document(
                    id=row.id,
                    run_id=row.run_id,
                    url=row.url,
                    content_hash=row.content_hash,
                    raw_content=row.raw_content,
                    cleaned_content=row.cleaned_content,
                    fetched_at=self._from_db_datetime(row.fetched_at),
                    status=row.status
                )
                for row in results
            ]

    def get_document_by_hash(self, run_id: int, content_hash: str) -> Optional[Document]:
        """
        Get a document by content hash within a run.

        Args:
            run_id: Run ID
            content_hash: Content hash

        Returns:
            Document model or None if not found
        """
        with self.get_connection() as conn:
            result = conn.execute(
                select(self.documents).where(
                    and_(
                        self.documents.c.run_id == run_id,
                        self.documents.c.content_hash == content_hash
                    )
                )
            ).fetchone()

            if result:
                return Document(
                    id=result.id,
                    run_id=result.run_id,
                    url=result.url,
                    content_hash=result.content_hash,
                    raw_content=result.raw_content,
                    cleaned_content=result.cleaned_content,
                    fetched_at=self._from_db_datetime(result.fetched_at),
                    status=result.status
                )
            return None

    # Events CRUD operations
    def create_event(self, event: Event) -> int:
        """
        Create a new event.

        Args:
            event: Event model instance

        Returns:
            int: ID of the created event
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.events).values(
                    document_id=event.document_id,
                    event_time=event.event_time,
                    headline=event.headline,
                    body=event.body,
                    actors=event.actors,
                    confidence=event.confidence,
                    raw_response=event.raw_response
                )
            )
            return result.lastrowid

    def get_event_by_id(self, event_id: int) -> Optional[Event]:
        """
        Get an event by ID.

        Args:
            event_id: Event ID

        Returns:
            Event model or None if not found
        """
        with self.get_connection() as conn:
            result = conn.execute(
                select(self.events).where(self.events.c.id == event_id)
            ).fetchone()

            if result:
                return Event(
                    id=result.id,
                    document_id=result.document_id,
                    event_time=result.event_time,
                    headline=result.headline,
                    body=result.body,
                    actors=result.actors,
                    confidence=result.confidence,
                    raw_response=result.raw_response
                )
            return None

    def get_events_by_document(self, document_id: int) -> List[Event]:
        """
        Get all events for a document.

        Args:
            document_id: Document ID

        Returns:
            List of Event models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.events).where(self.events.c.document_id == document_id)
            ).fetchall()

            return [
                Event(
                    id=row.id,
                    document_id=row.document_id,
                    event_time=row.event_time,
                    headline=row.headline,
                    body=row.body,
                    actors=row.actors,
                    confidence=row.confidence,
                    raw_response=row.raw_response
                )
                for row in results
            ]

    def get_events_for_run(self, run_id: int) -> List[Event]:
        """
        Get all events for a run (across all documents).

        Args:
            run_id: Run ID

        Returns:
            List of Event models
        """
        with self.get_connection() as conn:
            # Join events with documents to filter by run_id
            query = select(self.events).select_from(
                self.events.join(self.documents, self.events.c.document_id == self.documents.c.id)
            ).where(self.documents.c.run_id == run_id)

            results = conn.execute(query).fetchall()

            return [
                Event(
                    id=row.id,
                    document_id=row.document_id,
                    event_time=row.event_time,
                    headline=row.headline,
                    body=row.body,
                    actors=row.actors,
                    confidence=row.confidence,
                    raw_response=row.raw_response
                )
                for row in results
            ]

    # Embeddings CRUD operations
    def create_embedding(self, embedding: Embedding) -> int:
        """
        Create a new embedding.

        Args:
            embedding: Embedding model instance

        Returns:
            int: ID of the created embedding
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.embeddings).values(
                    event_id=embedding.event_id,
                    vector=embedding.vector,
                    model=embedding.model,
                    dimensions=embedding.dimensions,
                    created_at=self._to_db_datetime(embedding.created_at or datetime.now())
                )
            )
            return result.lastrowid

    def get_embedding_by_event(self, event_id: int, model: Optional[str] = None) -> Optional[Embedding]:
        """
        Get embedding for an event.

        Args:
            event_id: Event ID
            model: Model name (optional, gets first if not specified)

        Returns:
            Embedding model or None if not found
        """
        with self.get_connection() as conn:
            query = select(self.embeddings).where(self.embeddings.c.event_id == event_id)
            if model:
                query = query.where(self.embeddings.c.model == model)

            result = conn.execute(query).fetchone()

            if result:
                return Embedding(
                    id=result.id,
                    event_id=result.event_id,
                    vector=result.vector,
                    model=result.model,
                    dimensions=result.dimensions,
                    created_at=self._from_db_datetime(result.created_at)
                )
            return None

    def get_embeddings_for_run(self, run_id: int) -> List[Tuple[int, Embedding]]:
        """
        Get all embeddings for a run with their event IDs.

        Args:
            run_id: Run ID

        Returns:
            List of tuples (event_id, Embedding)
        """
        with self.get_connection() as conn:
            # Join through events and documents to get embeddings for a run
            query = select(self.embeddings).select_from(
                self.embeddings.join(self.events, self.embeddings.c.event_id == self.events.c.id)
                .join(self.documents, self.events.c.document_id == self.documents.c.id)
            ).where(self.documents.c.run_id == run_id)

            results = conn.execute(query).fetchall()

            return [
                (row.event_id, Embedding(
                    id=row.id,
                    event_id=row.event_id,
                    vector=row.vector,
                    model=row.model,
                    dimensions=row.dimensions,
                    created_at=self._from_db_datetime(row.created_at)
                ))
                for row in results
            ]

    # EventClusters CRUD operations
    def create_event_cluster(self, cluster: EventCluster) -> int:
        """
        Create a new event cluster.

        Args:
            cluster: EventCluster model instance

        Returns:
            int: ID of the created cluster
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.event_clusters).values(
                    run_id=cluster.run_id,
                    label=cluster.label,
                    centroid_event_id=cluster.centroid_event_id,
                    member_ids=cluster.member_ids
                )
            )
            return result.lastrowid

    def get_event_clusters_by_run(self, run_id: int) -> List[EventCluster]:
        """
        Get all event clusters for a run.

        Args:
            run_id: Run ID

        Returns:
            List of EventCluster models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.event_clusters).where(self.event_clusters.c.run_id == run_id)
            ).fetchall()

            return [
                EventCluster(
                    id=row.id,
                    run_id=row.run_id,
                    label=row.label,
                    centroid_event_id=row.centroid_event_id,
                    member_ids=row.member_ids
                )
                for row in results
            ]

    # Timeline CRUD operations
    def create_timeline_entry(self, entry: TimelineEntry) -> int:
        """
        Create a new timeline entry.

        Args:
            entry: TimelineEntry model instance

        Returns:
            int: ID of the created timeline entry
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.timeline).values(
                    run_id=entry.run_id,
                    cluster_id=entry.cluster_id,
                    event_time=entry.event_time,
                    summary=entry.summary,
                    citations=entry.citations,
                    tags=entry.tags
                )
            )
            return result.lastrowid

    def get_timeline_for_run(self, run_id: int) -> List[TimelineEntry]:
        """
        Get the timeline for a run, ordered by event time.

        Args:
            run_id: Run ID

        Returns:
            List of TimelineEntry models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.timeline).where(self.timeline.c.run_id == run_id)
                .order_by(self.timeline.c.event_time)
            ).fetchall()

            return [
                TimelineEntry(
                    id=row.id,
                    run_id=row.run_id,
                    cluster_id=row.cluster_id,
                    event_time=row.event_time,
                    summary=row.summary,
                    citations=row.citations,
                    tags=row.tags
                )
                for row in results
            ]

    # Forecasts CRUD operations
    def create_forecast(self, forecast: Forecast) -> int:
        """
        Create a new forecast.

        Args:
            forecast: Forecast model instance

        Returns:
            int: ID of the created forecast
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.forecasts).values(
                    run_id=forecast.run_id,
                    probability=forecast.probability,
                    reasoning=forecast.reasoning,
                    caveats=forecast.caveats,
                    raw_response=forecast.raw_response,
                    created_at=self._to_db_datetime(forecast.created_at or datetime.now())
                )
            )
            return result.lastrowid

    def get_forecast_by_run(self, run_id: int) -> Optional[Forecast]:
        """
        Get the forecast for a run.

        Args:
            run_id: Run ID

        Returns:
            Forecast model or None if not found
        """
        with self.get_connection() as conn:
            result = conn.execute(
                select(self.forecasts).where(self.forecasts.c.run_id == run_id)
                .order_by(self.forecasts.c.created_at.desc())
            ).fetchone()

            if result:
                return Forecast(
                    id=result.id,
                    run_id=result.run_id,
                    probability=result.probability,
                    reasoning=result.reasoning,
                    caveats=result.caveats,
                    raw_response=result.raw_response,
                    created_at=self._from_db_datetime(result.created_at)
                )
            return None

    # RunMetrics CRUD operations
    def create_run_metric(self, metric: RunMetric) -> int:
        """
        Create a new run metric.

        Args:
            metric: RunMetric model instance

        Returns:
            int: ID of the created metric
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.run_metrics).values(
                    run_id=metric.run_id,
                    metric_name=metric.metric_name,
                    metric_value=metric.metric_value
                )
            )
            return result.lastrowid

    def get_metrics_by_run(self, run_id: int) -> List[RunMetric]:
        """
        Get all metrics for a run.

        Args:
            run_id: Run ID

        Returns:
            List of RunMetric models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.run_metrics).where(self.run_metrics.c.run_id == run_id)
            ).fetchall()

            return [
                RunMetric(
                    id=row.id,
                    run_id=row.run_id,
                    metric_name=row.metric_name,
                    metric_value=row.metric_value
                )
                for row in results
            ]

    # Errors CRUD operations
    def create_error(self, error: Error) -> int:
        """
        Create a new error record.

        Args:
            error: Error model instance

        Returns:
            int: ID of the created error
        """
        with self.transaction() as conn:
            result = conn.execute(
                insert(self.errors).values(
                    run_id=error.run_id,
                    stage=error.stage,
                    reference=error.reference,
                    error_type=error.error_type,
                    message=error.message,
                    created_at=self._to_db_datetime(error.created_at or datetime.now())
                )
            )
            return result.lastrowid

    def get_errors_by_run(self, run_id: int) -> List[Error]:
        """
        Get all errors for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Error models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.errors).where(self.errors.c.run_id == run_id)
                .order_by(self.errors.c.created_at)
            ).fetchall()

            return [
                Error(
                    id=row.id,
                    run_id=row.run_id,
                    stage=row.stage,
                    reference=row.reference,
                    error_type=row.error_type,
                    message=row.message,
                    created_at=self._from_db_datetime(row.created_at)
                )
                for row in results
            ]

    def get_errors_by_stage(self, run_id: int, stage: str) -> List[Error]:
        """
        Get errors for a specific stage in a run.

        Args:
            run_id: Run ID
            stage: Stage name

        Returns:
            List of Error models
        """
        with self.get_connection() as conn:
            results = conn.execute(
                select(self.errors).where(
                    and_(
                        self.errors.c.run_id == run_id,
                        self.errors.c.stage == stage
                    )
                )
                .order_by(self.errors.c.created_at)
            ).fetchall()

            return [
                Error(
                    id=row.id,
                    run_id=row.run_id,
                    stage=row.stage,
                    reference=row.reference,
                    error_type=row.error_type,
                    message=row.message,
                    created_at=self._from_db_datetime(row.created_at)
                )
                for row in results
            ]
