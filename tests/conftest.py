"""
Pytest configuration and shared fixtures for AI Forecasting Pipeline tests.

Provides:
- In-memory SQLite database fixtures
- Mock LLM responses
- Mock search results
- Sample documents, events, and embeddings
- Service mocks with predictable behavior
"""

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import sqlite3

import pytest
import numpy as np
from sqlalchemy import create_engine

from db.models import (
    Question, Run, RunStatus, SearchQuery, SearchResult, Document, Event,
    Embedding as EmbeddingModel, EventCluster, TimelineEntry, Forecast,
    RunMetric, Error
)
from db.repository import DatabaseRepository
from services import (
    Embedding, ExtractedEvent, TimestampSpecificity,
    EventCluster, ClusteringResult
)


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_forecast.db"
        yield str(db_path)


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    # Create in-memory database
    db_url = "sqlite:///:memory:"
    engine = create_engine(db_url, echo=False)

    # Import after engine creation to use same connection
    from db.migrate import create_tables

    # Create all tables
    with engine.begin() as conn:
        from db.repository import DatabaseRepository
        repo = DatabaseRepository.__new__(DatabaseRepository)
        repo.engine = engine
        repo.metadata = __import__('sqlalchemy').MetaData()
        repo._define_tables()
        repo.metadata.create_all(engine)

    yield engine


@pytest.fixture
def db_repository(temp_db_path):
    """Create a DatabaseRepository with a test database."""
    # Ensure data directory exists
    db_dir = Path(temp_db_path).parent
    db_dir.mkdir(exist_ok=True)

    repo = DatabaseRepository(temp_db_path)

    # Create tables
    from db.migrate import create_tables
    create_tables(temp_db_path)

    yield repo

    # Cleanup is automatic with temp_db_path


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_question() -> Question:
    """Create a sample forecasting question."""
    return Question(
        id=1,
        question_text="Will the global temperature increase by more than 1.5°C by 2030?",
        resolution_criteria="Based on NASA GISS temperature records for annual global mean temperature",
        created_at=datetime.now()
    )


@pytest.fixture
def sample_run(sample_question) -> Run:
    """Create a sample pipeline run."""
    return Run(
        id=1,
        question_id=sample_question.id,
        status=RunStatus.PENDING,
        started_at=datetime.now(),
        git_commit="abc123def456"
    )


@pytest.fixture
def sample_search_queries() -> List[SearchQuery]:
    """Create sample search queries."""
    return [
        SearchQuery(
            id=1,
            run_id=1,
            query_text="global temperature increase 2030",
            created_at=datetime.now()
        ),
        SearchQuery(
            id=2,
            run_id=1,
            query_text="climate change 1.5 degrees celsius",
            created_at=datetime.now()
        ),
        SearchQuery(
            id=3,
            run_id=1,
            query_text="NASA GISS temperature records 2030",
            created_at=datetime.now()
        ),
    ]


@pytest.fixture
def sample_search_results(sample_search_queries) -> List[SearchResult]:
    """Create sample search results."""
    results = []
    urls = [
        ("https://climate.nasa.gov/news/2024/", "NASA Climate Change News"),
        ("https://example.com/climate-data", "Climate Data Report"),
        ("https://ipcc.ch/report/", "IPCC Climate Report"),
    ]

    for query_idx, query in enumerate(sample_search_queries):
        for url_idx, (url, title) in enumerate(urls):
            results.append(SearchResult(
                id=len(results) + 1,
                query_id=query.id,
                url=url,
                title=title,
                snippet=f"Relevant snippet about {query.query_text}",
                rank=url_idx + 1
            ))

    return results


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample documents from scraped content."""
    return [
        Document(
            id=1,
            run_id=1,
            url="https://climate.nasa.gov/news/2024/",
            content_hash="hash1",
            cleaned_content="Global temperatures have risen by 1.2 degrees since 2000. Scientists predict further increases.",
            raw_content="<html>...</html>",
            fetched_at=datetime.now(),
            status="processed"
        ),
        Document(
            id=2,
            run_id=1,
            url="https://example.com/climate-data",
            content_hash="hash2",
            cleaned_content="Recent analysis shows accelerating warming trends in the Arctic and tropical regions.",
            raw_content="<html>...</html>",
            fetched_at=datetime.now(),
            status="processed"
        ),
        Document(
            id=3,
            run_id=1,
            url="https://ipcc.ch/report/",
            content_hash="hash3",
            cleaned_content="IPCC reports indicate 1.5°C threshold will likely be exceeded by 2030 under current policies.",
            raw_content="<html>...</html>",
            fetched_at=datetime.now(),
            status="processed"
        ),
    ]


@pytest.fixture
def sample_events(sample_documents) -> List[Event]:
    """Create sample extracted events."""
    return [
        Event(
            id=1,
            document_id=sample_documents[0].id,
            event_time="2024-01-15",
            headline="Global Temperature Record Set",
            body="2024 marked the warmest year on record with a 1.35°C increase.",
            actors='["NASA", "Climate Scientists"]',
            confidence=0.95,
            raw_response='{"event": "temperature_record"}'
        ),
        Event(
            id=2,
            document_id=sample_documents[1].id,
            event_time="2024-02-20",
            headline="Arctic Warming Accelerates",
            body="Arctic temperatures rising twice as fast as global average.",
            actors='["Climate Researchers", "Arctic Council"]',
            confidence=0.88,
            raw_response='{"event": "arctic_warming"}'
        ),
        Event(
            id=3,
            document_id=sample_documents[2].id,
            event_time="2024-03-10",
            headline="IPCC Issues Updated Climate Assessment",
            body="New analysis suggests 1.5°C warming will be reached by 2029.",
            actors='["IPCC"]',
            confidence=0.92,
            raw_response='{"event": "ipcc_update"}'
        ),
    ]


@pytest.fixture
def sample_embeddings(sample_events) -> List[Embedding]:
    """Create sample embeddings for events."""
    embeddings = []

    for event in sample_events:
        # Create a realistic embedding vector
        vector = np.random.randn(1536).astype(np.float32)
        # Normalize to unit vector
        vector = vector / np.linalg.norm(vector)

        embedding = Embedding(
            text=f"{event.headline} {event.body}",
            vector=vector,
            model="text-embedding-3-large",
            dimensions=1536,
            text_hash=f"hash_{event.id}"
        )
        embeddings.append(embedding)

    return embeddings


@pytest.fixture
def sample_event_clusters(sample_events) -> List[EventCluster]:
    """Create sample event clusters."""
    return [
        EventCluster(
            id=1,
            run_id=1,
            label="Temperature Records and Warming Trends",
            centroid_event_id=sample_events[0].id,
            member_ids=json.dumps([sample_events[0].id, sample_events[1].id])
        ),
        EventCluster(
            id=2,
            run_id=1,
            label="Scientific Assessments",
            centroid_event_id=sample_events[2].id,
            member_ids=json.dumps([sample_events[2].id])
        ),
    ]


@pytest.fixture
def sample_timeline_entries() -> List[TimelineEntry]:
    """Create sample timeline entries."""
    return [
        TimelineEntry(
            id=1,
            run_id=1,
            cluster_id=1,
            event_time="2024-01-15",
            summary="Global temperature record set in January 2024.",
            citations=json.dumps([{"url": "https://climate.nasa.gov", "title": "NASA Climate News"}]),
            tags=json.dumps(["temperature", "record"])
        ),
        TimelineEntry(
            id=2,
            run_id=1,
            cluster_id=1,
            event_time="2024-02-20",
            summary="Arctic temperatures continue to rise at twice the global rate.",
            citations=json.dumps([{"url": "https://example.com/climate-data", "title": "Climate Data"}]),
            tags=json.dumps(["arctic", "warming"])
        ),
        TimelineEntry(
            id=3,
            run_id=1,
            cluster_id=2,
            event_time="2024-03-10",
            summary="IPCC updates assessment, 1.5°C threshold likely by 2029.",
            citations=json.dumps([{"url": "https://ipcc.ch/report/", "title": "IPCC Report"}]),
            tags=json.dumps(["ipcc", "assessment"])
        ),
    ]


@pytest.fixture
def sample_forecast() -> Forecast:
    """Create a sample forecast."""
    return Forecast(
        id=1,
        run_id=1,
        probability=0.72,
        reasoning="Based on current warming trends and IPCC assessments, the probability of exceeding 1.5°C by 2030 is high.",
        caveats=json.dumps(["Assumes current policy continuation", "Unknown impact of climate interventions"]),
        created_at=datetime.now()
    )


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with predictable responses."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="Mock LLM response")
    mock.generate_sync = Mock(return_value="Mock LLM response")
    mock.request_count = 0
    mock.total_tokens = 0
    mock.get_stats = Mock(return_value={"request_count": 0, "total_tokens": 0})
    return mock


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client."""
    mock = Mock()
    mock.generate = Mock(return_value="Mock Gemini response")
    mock.generate_async = AsyncMock(return_value="Mock Gemini response")
    mock.model_name = "gemini-2.0-flash-exp"
    return mock


@pytest.fixture
def mock_search_service():
    """Create a mock Google Search service."""
    mock = AsyncMock()
    mock.search_async = AsyncMock(return_value=[
        {"url": "https://example.com/1", "title": "Result 1", "snippet": "Snippet 1"},
        {"url": "https://example.com/2", "title": "Result 2", "snippet": "Snippet 2"},
        {"url": "https://example.com/3", "title": "Result 3", "snippet": "Snippet 3"},
    ])
    mock.search = Mock(return_value=[
        {"url": "https://example.com/1", "title": "Result 1", "snippet": "Snippet 1"},
    ])
    return mock


@pytest.fixture
def mock_scraper():
    """Create a mock web scraper."""
    mock = AsyncMock()

    # Create mock scrape results
    class MockScrapeResult:
        def __init__(self, url, content):
            self.url = url
            self.success = True
            self.raw_content = f"<html>{content}</html>"
            self.cleaned_content = content
            self.error = None

    async def scrape_urls_impl(urls):
        return [MockScrapeResult(url, f"Content for {url}") for url in urls]

    mock.scrape_urls = scrape_urls_impl
    return mock


@pytest.fixture
def mock_query_generator():
    """Create a mock query generator."""
    mock = AsyncMock()
    mock.generate_queries_async = AsyncMock(return_value=[
        "query 1 about topic",
        "query 2 about topic",
        "query 3 about topic",
    ])
    mock.generate_queries = Mock(return_value=[
        "query 1 about topic",
    ])
    mock.prompt_version = "1.0"
    return mock


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = AsyncMock()

    async def generate_embeddings_async_impl(texts):
        embeddings = []
        for text in texts:
            vector = np.random.randn(1536).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            embeddings.append(Embedding(
                text=text,
                vector=vector,
                model="text-embedding-3-large",
                dimensions=1536,
                text_hash=f"hash_{len(embeddings)}"
            ))
        return embeddings

    mock.generate_embeddings_async = generate_embeddings_async_impl
    return mock


@pytest.fixture
def mock_clustering_service():
    """Create a mock clustering service."""
    mock = Mock()

    def cluster_events_impl(event_ids, embeddings, events=None):
        # Create dummy clusters
        if len(event_ids) > 0:
            clusters = [
                EventCluster(
                    event_ids=event_ids[:len(event_ids)//2] if len(event_ids) > 1 else event_ids,
                    label="Cluster 1",
                    centroid_event_id=event_ids[0]
                ),
            ]
            if len(event_ids) > 1:
                clusters.append(EventCluster(
                    event_ids=event_ids[len(event_ids)//2:],
                    label="Cluster 2",
                    centroid_event_id=event_ids[len(event_ids)//2] if len(event_ids)//2 < len(event_ids) else event_ids[-1]
                ))
        else:
            clusters = []

        return ClusteringResult(
            clusters=clusters,
            silhouette_score=0.65,
            n_clusters=len(clusters)
        )

    mock.cluster_events = cluster_events_impl
    return mock


@pytest.fixture
def mock_doc_processor():
    """Create a mock document processor."""
    mock = Mock()

    def chunk_document_impl(content, url=None):
        # Simple chunking by sentences
        sentences = content.split(". ")
        return [
            type('DocumentChunk', (), {
                'content': s.strip() + ".",
                'url': url,
                'start_idx': idx,
                'end_idx': idx + len(s)
            })()
            for idx, s in enumerate(sentences) if s.strip()
        ]

    mock.chunk_document = chunk_document_impl
    return mock


@pytest.fixture
def mock_event_extractor():
    """Create a mock event extractor."""
    mock = AsyncMock()

    async def extract_events_impl(chunks, question=None):
        events = []
        for idx, chunk in enumerate(chunks):
            events.append(ExtractedEvent(
                event_time="2024-01-01",
                headline=f"Event {idx+1}",
                body=f"Event body for chunk {idx+1}",
                actors=["Actor1", "Actor2"],
                confidence=0.85 + (idx * 0.05),
                specificity=TimestampSpecificity.SPECIFIC_DATE,
                raw_response="{}"
            ))
        return events

    mock.extract_events_from_chunks_async = extract_events_impl
    return mock


@pytest.fixture
def mock_timeline_builder():
    """Create a mock timeline builder."""
    mock = Mock()

    def build_timeline_impl(clusters, events_by_id, documents_by_id):
        # Create a mock Timeline object
        timeline = type('Timeline', (), {
            'entries': [
                type('TimelineEntry', (), {
                    'event_time': "2024-01-01",
                    'summary': "Test event summary",
                    'citations': [],
                    'tags': [],
                    'to_model': lambda run_id: TimelineEntry(
                        run_id=run_id,
                        event_time="2024-01-01",
                        summary="Test event summary",
                        citations="[]",
                        tags="[]"
                    )
                })()
            ],
            'coverage_level': type('CoverageLevel', (), {'value': 'HIGH'})()
        })()
        return timeline

    mock.build_timeline = build_timeline_impl
    return mock


@pytest.fixture
def mock_forecast_generator():
    """Create a mock forecast generator."""
    mock = AsyncMock()

    async def generate_forecast_impl(question, timeline):
        return type('Forecast', (), {
            'probability': 0.75,
            'reasoning': "Mock forecast reasoning based on timeline",
            'caveats': ["Caveat 1", "Caveat 2"],
            'confidence_level': type('ConfidenceLevel', (), {'value': 'HIGH'})(),
            'raw_output': "{}"
        })()

    mock.generate_forecast_async = generate_forecast_impl
    return mock


# ============================================================================
# Mock Dependency Fixtures
# ============================================================================

@pytest.fixture
def mock_settings():
    """Create a mock settings object."""
    mock = Mock()
    mock.google_api_key = "test_google_key"
    mock.google_cse_id = "test_cse_id"
    mock.google_gemini_api_key = "test_gemini_key"
    mock.openai_api_key = "test_openai_key"
    mock.database_url = "sqlite:///:memory:"
    mock.debug = True
    mock.max_concurrent_scrapes = 5
    mock.request_timeout = 30
    mock.user_agent = "Test Agent"
    mock.gemini_model = "gemini-2.0-flash-exp"
    mock.openai_embedding_model = "text-embedding-3-large"
    mock.embedding_dimensions = 1536
    mock.max_search_queries = 10
    mock.search_results_per_query = 10
    mock.clustering_threshold = 0.2
    mock.project_root = Path(__file__).parent.parent
    mock.data_dir = mock.project_root / "data"
    mock.outputs_dir = mock.project_root / "outputs"
    mock.cache_dir = mock.project_root / ".cache"
    return mock


# ============================================================================
# Context/Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "test_cse_id")
    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("DEBUG", "true")


@pytest.fixture
def async_context():
    """Provide async context for async tests."""
    import asyncio
    return asyncio


# ============================================================================
# Helper Fixtures
# ============================================================================

@pytest.fixture
def create_sample_data(db_repository, sample_question, sample_run,
                       sample_search_queries, sample_search_results,
                       sample_documents, sample_events, sample_embeddings,
                       sample_event_clusters, sample_timeline_entries, sample_forecast):
    """Create all sample data in the database."""
    def _create():
        # Create question
        question_id = db_repository.create_question(sample_question)
        sample_question.id = question_id

        # Create run
        sample_run.question_id = question_id
        run_id = db_repository.create_run(sample_run)
        sample_run.id = run_id

        # Create search queries
        query_ids = []
        for query in sample_search_queries:
            query.run_id = run_id
            qid = db_repository.create_search_query(query)
            query.id = qid
            query_ids.append(qid)

        # Create search results
        for i, result in enumerate(sample_search_results):
            result.query_id = query_ids[i % len(query_ids)]
            db_repository.create_search_result(result)

        # Create documents
        doc_ids = []
        for doc in sample_documents:
            doc.run_id = run_id
            doc_id = db_repository.create_document(doc)
            doc.id = doc_id
            doc_ids.append(doc_id)

        # Create events
        event_ids = []
        for event in sample_events:
            event.document_id = doc_ids[event.document_id - 1] if event.document_id <= len(doc_ids) else doc_ids[0]
            event_id = db_repository.create_event(event)
            event.id = event_id
            event_ids.append(event_id)

        # Create embeddings
        for embedding, event_id in zip(sample_embeddings, event_ids):
            emb_model = EmbeddingModel(
                event_id=event_id,
                vector=embedding.to_bytes(),
                model=embedding.model,
                dimensions=embedding.dimensions
            )
            db_repository.create_embedding(emb_model)

        # Create event clusters
        for cluster in sample_event_clusters:
            cluster.run_id = run_id
            db_repository.create_event_cluster(cluster)

        # Create timeline entries
        for entry in sample_timeline_entries:
            entry.run_id = run_id
            db_repository.create_timeline_entry(entry)

        # Create forecast
        sample_forecast.run_id = run_id
        db_repository.create_forecast(sample_forecast)

        return {
            'question_id': question_id,
            'run_id': run_id,
            'query_ids': query_ids,
            'doc_ids': doc_ids,
            'event_ids': event_ids
        }

    return _create


# ============================================================================
# Pytest Hooks and Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks after each test."""
    yield
    # Cleanup happens automatically with pytest's mock fixture management
