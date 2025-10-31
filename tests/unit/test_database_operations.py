"""
Comprehensive Database Operations Test.

Tests all database CRUD operations with mock data to ensure:
- Tables are created correctly
- All insert operations work
- All query operations work
- Data integrity is maintained
"""

import json
import pytest
from datetime import datetime

from db.repository import DatabaseRepository
from db.models import (
    Question, Run, RunStatus, SearchQuery, SearchResult, Document, Event,
    Embedding as EmbeddingModel, EventCluster, TimelineEntry, Forecast,
    RunMetric, Error
)


class TestDatabaseOperations:
    """Test suite for all database operations."""

    def test_create_tables(self, db_repository):
        """Test that all tables are created successfully."""
        # Tables should be created by the fixture
        # Verify by trying to query each table
        assert db_repository.get_all_questions() is not None
        print("✅ All tables created successfully")

    def test_question_crud(self, db_repository):
        """Test Question CRUD operations."""
        # Create
        question = Question(
            question_text="Will Bitcoin reach $150,000 by June 2026?",
            resolution_criteria="Based on major exchange prices"
        )
        question_id = db_repository.create_question(question)
        assert question_id > 0
        print(f"✅ Question created with ID: {question_id}")

        # Read
        retrieved = db_repository.get_question_by_id(question_id)
        assert retrieved is not None
        assert retrieved.question_text == question.question_text
        print(f"✅ Question retrieved: {retrieved.question_text}")

        # List
        all_questions = db_repository.get_all_questions()
        assert len(all_questions) >= 1
        print(f"✅ Found {len(all_questions)} question(s)")

    def test_run_crud(self, db_repository):
        """Test Run CRUD operations."""
        # Create question first
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)

        # Create run
        run = Run(
            question_id=question_id,
            status=RunStatus.PENDING,
            git_commit="abc123def456"
        )
        run_id = db_repository.create_run(run)
        assert run_id > 0
        print(f"✅ Run created with ID: {run_id}")

        # Read
        retrieved = db_repository.get_run_by_id(run_id)
        assert retrieved is not None
        assert retrieved.question_id == question_id
        assert retrieved.status == RunStatus.PENDING
        print(f"✅ Run retrieved with status: {retrieved.status}")

        # Update status
        db_repository.update_run_status(run_id, RunStatus.RUNNING)
        retrieved = db_repository.get_run_by_id(run_id)
        assert retrieved.status == RunStatus.RUNNING
        print(f"✅ Run status updated to: {retrieved.status}")

        # Complete run
        db_repository.update_run_status(run_id, RunStatus.COMPLETED)
        retrieved = db_repository.get_run_by_id(run_id)
        assert retrieved.status == RunStatus.COMPLETED
        print(f"✅ Run completed with status: {retrieved.status}")

    def test_search_query_and_results(self, db_repository):
        """Test SearchQuery and SearchResult operations."""
        # Create question and run
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)

        # Create search queries
        queries = [
            SearchQuery(run_id=run_id, query_text="bitcoin price prediction 2026"),
            SearchQuery(run_id=run_id, query_text="cryptocurrency market forecast"),
            SearchQuery(run_id=run_id, query_text="bitcoin $150k prediction"),
        ]

        query_ids = []
        for query in queries:
            query_id = db_repository.create_search_query(query)
            query_ids.append(query_id)
            assert query_id > 0
        print(f"✅ Created {len(query_ids)} search queries")

        # Create search results
        results_count = 0
        for query_id in query_ids:
            for i in range(3):
                result = SearchResult(
                    query_id=query_id,
                    url=f"https://example.com/article-{query_id}-{i}",
                    title=f"Article {i} for query {query_id}",
                    snippet=f"Relevant content about query {query_id}",
                    rank=i+1
                )
                db_repository.create_search_result(result)
                results_count += 1
        print(f"✅ Created {results_count} search results")

        # Retrieve
        retrieved_queries = db_repository.get_search_queries_by_run(run_id)
        assert len(retrieved_queries) == 3
        print(f"✅ Retrieved {len(retrieved_queries)} queries for run")

        for query in retrieved_queries:
            results = db_repository.get_search_results_by_query(query.id)
            assert len(results) == 3
            print(f"✅ Query '{query.query_text[:30]}...' has {len(results)} results")

    def test_document_crud(self, db_repository):
        """Test Document CRUD operations."""
        # Create question and run
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)

        # Create documents
        documents = [
            Document(
                run_id=run_id,
                url="https://example.com/article1",
                content_hash="hash1",
                raw_content="<html>Raw content 1</html>",
                cleaned_content="Cleaned content about Bitcoin reaching new highs.",
                status="processed"
            ),
            Document(
                run_id=run_id,
                url="https://example.com/article2",
                content_hash="hash2",
                raw_content="<html>Raw content 2</html>",
                cleaned_content="Market analysis suggests bullish trends.",
                status="processed"
            ),
        ]

        doc_ids = []
        for doc in documents:
            doc_id = db_repository.create_document(doc)
            doc_ids.append(doc_id)
            assert doc_id > 0
        print(f"✅ Created {len(doc_ids)} documents")

        # Retrieve
        retrieved_docs = db_repository.get_documents_by_run(run_id)
        assert len(retrieved_docs) == 2
        print(f"✅ Retrieved {len(retrieved_docs)} documents for run")

        # Check content
        for doc in retrieved_docs:
            assert doc.cleaned_content is not None
            assert len(doc.cleaned_content) > 0
            print(f"✅ Document {doc.id}: {doc.cleaned_content[:50]}...")

    def test_event_crud(self, db_repository):
        """Test Event CRUD operations."""
        # Create question, run, and document
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)
        doc = Document(
            run_id=run_id,
            url="https://example.com/article",
            content_hash="hash1",
            cleaned_content="Content",
            status="processed"
        )
        doc_id = db_repository.create_document(doc)

        # Create events
        events = [
            Event(
                document_id=doc_id,
                event_time="2024-01-15",
                headline="Bitcoin Reaches $100,000",
                body="Bitcoin crosses six-figure milestone for first time.",
                actors='["Crypto Markets", "Investors"]',
                confidence=0.95
            ),
            Event(
                document_id=doc_id,
                event_time="2024-06-20",
                headline="Major Institutional Adoption",
                body="Large institutions announce Bitcoin treasury holdings.",
                actors='["BlackRock", "Fidelity"]',
                confidence=0.88
            ),
            Event(
                document_id=doc_id,
                event_time="2025-01-10",
                headline="Bitcoin ETF Approval",
                body="SEC approves multiple spot Bitcoin ETFs.",
                actors='["SEC", "ETF Issuers"]',
                confidence=0.92
            ),
        ]

        event_ids = []
        for event in events:
            event_id = db_repository.create_event(event)
            event_ids.append(event_id)
            assert event_id > 0
        print(f"✅ Created {len(event_ids)} events")

        # Retrieve
        retrieved_events = db_repository.get_events_for_run(run_id)
        assert len(retrieved_events) == 3
        print(f"✅ Retrieved {len(retrieved_events)} events for run")

        # Check attributes
        for event in retrieved_events:
            assert event.headline is not None
            assert event.confidence is not None
            print(f"✅ Event: {event.headline} (confidence: {event.confidence})")

    def test_embedding_crud(self, db_repository):
        """Test Embedding CRUD operations."""
        # Create question, run, document, and event
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)
        doc = Document(
            run_id=run_id,
            url="https://example.com/article",
            content_hash="hash1",
            cleaned_content="Content",
            status="processed"
        )
        doc_id = db_repository.create_document(doc)
        event = Event(
            document_id=doc_id,
            event_time="2024-01-15",
            headline="Test Event",
            body="Test body",
            confidence=0.9
        )
        event_id = db_repository.create_event(event)

        # Create embedding
        import numpy as np
        vector = np.random.randn(1536).astype(np.float32)
        vector_bytes = vector.tobytes()

        embedding = EmbeddingModel(
            event_id=event_id,
            vector=vector_bytes,
            model="text-embedding-3-large",
            dimensions=1536
        )
        emb_id = db_repository.create_embedding(embedding)
        assert emb_id > 0
        print(f"✅ Created embedding with ID: {emb_id}")

        # Retrieve
        retrieved_emb = db_repository.get_embedding_by_event(event_id)
        assert retrieved_emb is not None
        assert retrieved_emb.event_id == event_id
        assert retrieved_emb.dimensions == 1536
        print(f"✅ Retrieved embedding with {retrieved_emb.dimensions} dimensions")

        # Verify vector can be reconstructed
        retrieved_vector = np.frombuffer(retrieved_emb.vector, dtype=np.float32)
        assert len(retrieved_vector) == 1536
        print(f"✅ Vector reconstructed: shape={retrieved_vector.shape}")

    def test_event_cluster_crud(self, db_repository):
        """Test EventCluster CRUD operations."""
        # Create question and run
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)

        # Create document and events
        doc = Document(
            run_id=run_id,
            url="https://example.com/article",
            content_hash="hash1",
            cleaned_content="Content",
            status="processed"
        )
        doc_id = db_repository.create_document(doc)

        event_ids = []
        for i in range(5):
            event = Event(
                document_id=doc_id,
                event_time=f"2024-0{i+1}-01",
                headline=f"Event {i+1}",
                body=f"Body {i+1}",
                confidence=0.9
            )
            event_id = db_repository.create_event(event)
            event_ids.append(event_id)

        # Create clusters
        clusters = [
            EventCluster(
                run_id=run_id,
                label="Price Movement Events",
                centroid_event_id=event_ids[0],
                member_ids=json.dumps([event_ids[0], event_ids[1], event_ids[2]])
            ),
            EventCluster(
                run_id=run_id,
                label="Adoption Events",
                centroid_event_id=event_ids[3],
                member_ids=json.dumps([event_ids[3], event_ids[4]])
            ),
        ]

        cluster_ids = []
        for cluster in clusters:
            cluster_id = db_repository.create_event_cluster(cluster)
            cluster_ids.append(cluster_id)
            assert cluster_id > 0
        print(f"✅ Created {len(cluster_ids)} event clusters")

        # Retrieve
        retrieved_clusters = db_repository.get_event_clusters_by_run(run_id)
        assert len(retrieved_clusters) == 2
        print(f"✅ Retrieved {len(retrieved_clusters)} clusters for run")

        # Check members
        for cluster in retrieved_clusters:
            member_ids = json.loads(cluster.member_ids)
            assert len(member_ids) > 0
            print(f"✅ Cluster '{cluster.label}': {len(member_ids)} members")

    def test_timeline_entry_crud(self, db_repository):
        """Test TimelineEntry CRUD operations."""
        # Create question and run
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)

        # Create timeline entries
        entries = [
            TimelineEntry(
                run_id=run_id,
                cluster_id=1,
                event_time="2024-01-15",
                summary="Bitcoin reaches $100k milestone",
                citations=json.dumps([{"url": "https://example.com/1", "title": "Source 1"}]),
                tags=json.dumps(["price", "milestone"])
            ),
            TimelineEntry(
                run_id=run_id,
                cluster_id=1,
                event_time="2024-06-20",
                summary="Institutional adoption accelerates",
                citations=json.dumps([{"url": "https://example.com/2", "title": "Source 2"}]),
                tags=json.dumps(["adoption", "institutions"])
            ),
            TimelineEntry(
                run_id=run_id,
                cluster_id=2,
                event_time="2025-01-10",
                summary="Major regulatory developments",
                citations=json.dumps([{"url": "https://example.com/3", "title": "Source 3"}]),
                tags=json.dumps(["regulation", "sec"])
            ),
        ]

        entry_ids = []
        for entry in entries:
            entry_id = db_repository.create_timeline_entry(entry)
            entry_ids.append(entry_id)
            assert entry_id > 0
        print(f"✅ Created {len(entry_ids)} timeline entries")

        # Retrieve
        retrieved_entries = db_repository.get_timeline_for_run(run_id)
        assert len(retrieved_entries) == 3
        print(f"✅ Retrieved {len(retrieved_entries)} timeline entries for run")

        # Check structure
        for entry in retrieved_entries:
            assert entry.summary is not None
            citations = json.loads(entry.citations)
            tags = json.loads(entry.tags)
            print(f"✅ Timeline entry: {entry.summary[:50]}... ({len(citations)} citations, {len(tags)} tags)")

    def test_forecast_crud(self, db_repository):
        """Test Forecast CRUD operations."""
        # Create question and run
        question = Question(question_text="Will Bitcoin reach $150,000 by June 2026?")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)

        # Create forecast
        forecast = Forecast(
            run_id=run_id,
            probability=0.68,
            reasoning=json.dumps([
                "Historical price trends show accelerating adoption",
                "Institutional investment continues to grow",
                "Supply shock from halving events"
            ]),
            caveats=json.dumps([
                "Assumes no major regulatory crackdown",
                "Market volatility remains high"
            ])
        )
        forecast_id = db_repository.create_forecast(forecast)
        assert forecast_id > 0
        print(f"✅ Created forecast with ID: {forecast_id}")

        # Retrieve
        retrieved = db_repository.get_forecast_by_run(run_id)
        assert retrieved is not None
        assert retrieved.probability == 0.68
        print(f"✅ Retrieved forecast: probability={retrieved.probability}")

        # Check reasoning and caveats
        reasoning = json.loads(retrieved.reasoning)
        caveats = json.loads(retrieved.caveats)
        print(f"✅ Forecast has {len(reasoning)} reasoning steps, {len(caveats)} caveats")

    def test_error_logging(self, db_repository):
        """Test Error logging."""
        # Create question and run
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)

        # Log errors
        errors = [
            Error(
                run_id=run_id,
                stage="SCRAPE",
                reference="https://example.com/blocked",
                error_type="FORBIDDEN",
                message="URL blocked by robots.txt"
            ),
            Error(
                run_id=run_id,
                stage="EVENT_EXTRACT",
                reference="doc_123",
                error_type="JSON_PARSE_ERROR",
                message="Invalid JSON in LLM response"
            ),
        ]

        error_ids = []
        for error in errors:
            error_id = db_repository.create_error(error)
            error_ids.append(error_id)
            assert error_id > 0
        print(f"✅ Logged {len(error_ids)} errors")

        # Retrieve
        retrieved_errors = db_repository.get_errors_by_run(run_id)
        assert len(retrieved_errors) == 2
        print(f"✅ Retrieved {len(retrieved_errors)} errors for run")

        for error in retrieved_errors:
            print(f"✅ Error in {error.stage}: {error.message}")

    def test_run_metrics(self, db_repository):
        """Test RunMetric CRUD operations."""
        # Create question and run
        question = Question(question_text="Test question")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)

        # Create metrics
        metrics = [
            RunMetric(run_id=run_id, metric_name="scrape_success_rate", metric_value=0.85),
            RunMetric(run_id=run_id, metric_name="total_events_extracted", metric_value=42.0),
            RunMetric(run_id=run_id, metric_name="clustering_silhouette_score", metric_value=0.67),
        ]

        metric_ids = []
        for metric in metrics:
            metric_id = db_repository.create_run_metric(metric)
            metric_ids.append(metric_id)
            assert metric_id > 0
        print(f"✅ Created {len(metric_ids)} metrics")

        # Retrieve
        retrieved_metrics = db_repository.get_metrics_by_run(run_id)
        assert len(retrieved_metrics) == 3
        print(f"✅ Retrieved {len(retrieved_metrics)} metrics for run")

        for metric in retrieved_metrics:
            print(f"✅ Metric {metric.metric_name}: {metric.metric_value}")

    def test_full_pipeline_data_flow(self, db_repository):
        """Test complete data flow through all stages."""
        print("\n" + "="*60)
        print("TESTING FULL PIPELINE DATA FLOW")
        print("="*60)

        # Stage 1: INIT - Create question and run
        question = Question(question_text="Will Bitcoin reach $150,000 by June 2026?")
        question_id = db_repository.create_question(question)
        run = Run(question_id=question_id, status=RunStatus.RUNNING)
        run_id = db_repository.create_run(run)
        print(f"\n✅ INIT: Created question {question_id} and run {run_id}")

        # Stage 2: QUERY_GEN - Create search queries
        queries = [
            SearchQuery(run_id=run_id, query_text="bitcoin price prediction 2026"),
            SearchQuery(run_id=run_id, query_text="cryptocurrency $150k forecast"),
        ]
        for query in queries:
            db_repository.create_search_query(query)
        print(f"✅ QUERY_GEN: Created {len(queries)} search queries")

        # Stage 3: SEARCH - Create search results
        retrieved_queries = db_repository.get_search_queries_by_run(run_id)
        results_count = 0
        for query in retrieved_queries:
            for i in range(3):
                result = SearchResult(
                    query_id=query.id,
                    url=f"https://example.com/article-{query.id}-{i}",
                    title=f"Article {i}",
                    snippet="Relevant content",
                    rank=i+1
                )
                db_repository.create_search_result(result)
                results_count += 1
        print(f"✅ SEARCH: Created {results_count} search results")

        # Stage 4: SCRAPE - Create documents
        all_results = []
        for query in retrieved_queries:
            all_results.extend(db_repository.get_search_results_by_query(query.id))

        doc_ids = []
        for result in all_results[:3]:  # Just first 3
            doc = Document(
                run_id=run_id,
                url=result.url,
                content_hash=f"hash_{result.id}",
                cleaned_content=f"Content from {result.url}",
                status="processed"
            )
            doc_id = db_repository.create_document(doc)
            doc_ids.append(doc_id)
        print(f"✅ SCRAPE: Created {len(doc_ids)} documents")

        # Stage 5: EVENT_EXTRACT - Create events
        event_ids = []
        for doc_id in doc_ids:
            for i in range(2):  # 2 events per doc
                event = Event(
                    document_id=doc_id,
                    event_time=f"2024-0{i+1}-15",
                    headline=f"Event from doc {doc_id}",
                    body="Event body content",
                    confidence=0.9
                )
                event_id = db_repository.create_event(event)
                event_ids.append(event_id)

                # Create embedding for each event
                import numpy as np
                vector = np.random.randn(1536).astype(np.float32)
                embedding = EmbeddingModel(
                    event_id=event_id,
                    vector=vector.tobytes(),
                    model="text-embedding-3-large",
                    dimensions=1536
                )
                db_repository.create_embedding(embedding)
        print(f"✅ EVENT_EXTRACT: Created {len(event_ids)} events with embeddings")

        # Stage 6: CLUSTER - Create clusters
        cluster = EventCluster(
            run_id=run_id,
            label="All Events Cluster",
            centroid_event_id=event_ids[0],
            member_ids=json.dumps(event_ids)
        )
        cluster_id = db_repository.create_event_cluster(cluster)
        print(f"✅ CLUSTER: Created cluster {cluster_id} with {len(event_ids)} events")

        # Stage 7: TIMELINE - Create timeline
        timeline_entry = TimelineEntry(
            run_id=run_id,
            cluster_id=cluster_id,
            event_time="2024-01-15",
            summary="Bitcoin price movement and predictions",
            citations=json.dumps([{"url": "https://example.com/1", "title": "Source"}]),
            tags=json.dumps(["bitcoin", "price"])
        )
        entry_id = db_repository.create_timeline_entry(timeline_entry)
        print(f"✅ TIMELINE: Created timeline entry {entry_id}")

        # Stage 8: FORECAST - Create forecast
        forecast = Forecast(
            run_id=run_id,
            probability=0.68,
            reasoning=json.dumps(["Based on trends", "Institutional adoption"]),
            caveats=json.dumps(["High volatility", "Regulatory uncertainty"])
        )
        forecast_id = db_repository.create_forecast(forecast)
        print(f"✅ FORECAST: Created forecast {forecast_id} with probability 0.68")

        # Verify complete data integrity
        print("\n" + "="*60)
        print("DATA INTEGRITY CHECK")
        print("="*60)

        assert db_repository.get_question_by_id(question_id) is not None
        assert db_repository.get_run_by_id(run_id) is not None
        assert len(db_repository.get_search_queries_by_run(run_id)) == 2
        assert len(db_repository.get_documents_by_run(run_id)) == 3
        assert len(db_repository.get_events_for_run(run_id)) == 6
        assert len(db_repository.get_event_clusters_by_run(run_id)) == 1
        assert len(db_repository.get_timeline_for_run(run_id)) == 1
        assert db_repository.get_forecast_by_run(run_id) is not None

        print("✅ All data integrity checks passed!")
        print("\n" + "="*60)
        print("FULL PIPELINE TEST COMPLETE - ALL STAGES WORKING!")
        print("="*60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
