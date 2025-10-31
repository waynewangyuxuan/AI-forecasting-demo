"""
Pipeline Orchestrator for AI Forecasting Pipeline.

Implements the main orchestration layer with state machine, idempotent stages,
resumability, and comprehensive error handling.
"""

import asyncio
import json
import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import structlog

from config.settings import settings
from db.repository import DatabaseRepository
from db.models import (
    Question, Run, RunStatus, SearchQuery, SearchResult, Document, Event,
    Embedding, EventCluster, TimelineEntry as TimelineEntryModel,
    RunMetric, Error
)
from services import (
    QueryGenerator, GoogleSearchService, WebScraper, DocumentProcessor,
    EventExtractor, EmbeddingService, ClusteringService, TimelineBuilder,
    ForecastGenerator, create_llm_client, scrape_urls
)
from services.clustering import (
    EventCluster as ServiceEventCluster, ClusteringResult, ClusteringAlgorithm
)

logger = structlog.get_logger(__name__)


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    INIT = "INIT"
    QUERY_GEN = "QUERY_GEN"
    SEARCH = "SEARCH"
    SCRAPE = "SCRAPE"
    EVENT_EXTRACT = "EVENT_EXTRACT"
    CLUSTER = "CLUSTER"
    TIMELINE = "TIMELINE"
    FORECAST = "FORECAST"


@dataclass
class StageMetrics:
    """Metrics collected during stage execution."""
    stage: str
    duration_seconds: float
    success_count: int = 0
    fail_count: int = 0
    items_processed: int = 0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """
    Context for a pipeline run.

    Maintains all state needed for pipeline execution including database access,
    service instances, and configuration.
    """
    question_id: int
    run_id: int
    question_text: str
    resume: bool = False
    max_urls: Optional[int] = None
    max_events: Optional[int] = None
    dry_run: bool = False
    verbose: bool = False
    output_dir: Optional[Path] = None

    # Database
    repo: Optional[DatabaseRepository] = None

    # Service instances (created lazily)
    llm_client: Optional[Any] = None
    query_generator: Optional[QueryGenerator] = None
    search_service: Optional[GoogleSearchService] = None
    scraper: Optional[WebScraper] = None
    doc_processor: Optional[DocumentProcessor] = None
    event_extractor: Optional[EventExtractor] = None
    embedding_service: Optional[EmbeddingService] = None
    clustering_service: Optional[ClusteringService] = None
    timeline_builder: Optional[TimelineBuilder] = None
    forecast_generator: Optional[ForecastGenerator] = None

    # Progress callback
    progress_callback: Optional[Callable[[str, str], None]] = None

    # Stage state tracking
    completed_stages: List[str] = field(default_factory=list)
    current_stage: Optional[str] = None
    stage_metrics: List[StageMetrics] = field(default_factory=list)


class PipelineOrchestrator:
    """
    Main orchestrator for the forecasting pipeline.

    Implements state machine with 8 stages, idempotent and resumable execution,
    structured logging, error handling, and progress tracking.
    """

    def __init__(self, repo: Optional[DatabaseRepository] = None):
        """
        Initialize the orchestrator.

        Args:
            repo: Database repository (creates new if not provided)
        """
        self.repo = repo or DatabaseRepository()
        self.logger = logger.bind(component="orchestrator")

    def _initialize_services(self, ctx: RunContext) -> None:
        """
        Initialize all service instances in the context.

        Args:
            ctx: Run context
        """
        # LLM client
        if not ctx.llm_client:
            ctx.llm_client = create_llm_client(provider=settings.llm_provider)

        # Query generator
        if not ctx.query_generator:
            ctx.query_generator = QueryGenerator(
                llm_client=ctx.llm_client,
                max_queries=settings.max_search_queries
            )

        # Search service
        if not ctx.search_service:
            ctx.search_service = GoogleSearchService(
                api_key=settings.google_api_key,
                cse_id=settings.google_cse_id
            )

        # Web scraper
        if not ctx.scraper:
            ctx.scraper = WebScraper(
                max_concurrent=settings.max_concurrent_scrapes,
                timeout=settings.request_timeout
            )

        # Document processor
        if not ctx.doc_processor:
            ctx.doc_processor = DocumentProcessor()

        # Event extractor
        if not ctx.event_extractor:
            ctx.event_extractor = EventExtractor(llm_client=ctx.llm_client)

        # Embedding service
        if not ctx.embedding_service:
            ctx.embedding_service = EmbeddingService(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model
            )

        # Clustering service
        if not ctx.clustering_service:
            ctx.clustering_service = ClusteringService(
                distance_threshold=settings.clustering_threshold
            )

        # Timeline builder
        if not ctx.timeline_builder:
            ctx.timeline_builder = TimelineBuilder()

        # Forecast generator
        if not ctx.forecast_generator:
            ctx.forecast_generator = ForecastGenerator(llm_client=ctx.llm_client)

    def _update_progress(self, ctx: RunContext, stage: str, message: str) -> None:
        """
        Update progress and call progress callback if provided.

        Args:
            ctx: Run context
            stage: Current stage name
            message: Progress message
        """
        self.logger.info("progress", stage=stage, message=message, run_id=ctx.run_id)
        if ctx.progress_callback:
            ctx.progress_callback(stage, message)

    def _log_error(self, ctx: RunContext, stage: str, error: Exception,
                   reference: Optional[str] = None) -> None:
        """
        Log error to database and structured logs.

        Args:
            ctx: Run context
            stage: Stage where error occurred
            error: Exception that was raised
            reference: Optional reference ID (e.g., document_id, url)
        """
        error_type = type(error).__name__
        message = str(error)

        self.logger.error(
            "stage_error",
            stage=stage,
            error_type=error_type,
            message=message,
            reference=reference,
            run_id=ctx.run_id
        )

        if not ctx.dry_run:
            error_record = Error(
                run_id=ctx.run_id,
                stage=stage,
                reference=reference,
                error_type=error_type,
                message=message
            )
            self.repo.create_error(error_record)

    def _save_stage_metrics(self, ctx: RunContext, metrics: StageMetrics) -> None:
        """
        Save stage metrics to database.

        Args:
            ctx: Run context
            metrics: Stage metrics to save
        """
        if ctx.dry_run:
            return

        # Save primary metrics
        self.repo.create_run_metric(RunMetric(
            run_id=ctx.run_id,
            metric_name=f"{metrics.stage}_duration_seconds",
            metric_value=metrics.duration_seconds
        ))

        self.repo.create_run_metric(RunMetric(
            run_id=ctx.run_id,
            metric_name=f"{metrics.stage}_success_count",
            metric_value=float(metrics.success_count)
        ))

        self.repo.create_run_metric(RunMetric(
            run_id=ctx.run_id,
            metric_name=f"{metrics.stage}_fail_count",
            metric_value=float(metrics.fail_count)
        ))

        # Save extra metrics
        for key, value in metrics.extra_metrics.items():
            if isinstance(value, (int, float)):
                self.repo.create_run_metric(RunMetric(
                    run_id=ctx.run_id,
                    metric_name=f"{metrics.stage}_{key}",
                    metric_value=float(value)
                ))

    def _is_stage_completed(self, ctx: RunContext, stage: PipelineStage) -> bool:
        """
        Check if a stage has already been completed.

        Args:
            ctx: Run context
            stage: Stage to check

        Returns:
            True if stage already completed
        """
        if not ctx.resume:
            return False

        # Load run from database
        run = self.repo.get_run_by_id(ctx.run_id)
        if not run or not run.stage_state:
            return False

        try:
            stage_state = json.loads(run.stage_state)
            return stage.value in stage_state.get('completed_stages', [])
        except (json.JSONDecodeError, KeyError):
            return False

    def _update_stage_state(self, ctx: RunContext, stage: PipelineStage) -> None:
        """
        Update stage state in database.

        Args:
            ctx: Run context
            stage: Completed stage
        """
        if ctx.dry_run:
            return

        ctx.completed_stages.append(stage.value)
        stage_state = {
            'completed_stages': ctx.completed_stages,
            'current_stage': stage.value,
            'last_updated': datetime.now().isoformat()
        }

        self.repo.update_run_status(
            ctx.run_id,
            status=RunStatus.RUNNING,
            stage_state=json.dumps(stage_state)
        )

    async def _stage_init(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 0: Initialization.

        Create/load question and run, initialize services.
        """
        start_time = datetime.now()
        stage = PipelineStage.INIT.value

        self._update_progress(ctx, stage, "Initializing pipeline")

        # Initialize services
        self._initialize_services(ctx)

        # Set output directory
        if not ctx.output_dir:
            ctx.output_dir = settings.outputs_dir / f"run_{ctx.run_id}"
        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info("stage_complete", stage=stage, duration=duration)

        return StageMetrics(
            stage=stage,
            duration_seconds=duration,
            success_count=1
        )

    async def _stage_query_gen(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 1: Query Generation.

        Generate diversified search queries using LLM.
        """
        start_time = datetime.now()
        stage = PipelineStage.QUERY_GEN.value

        if self._is_stage_completed(ctx, PipelineStage.QUERY_GEN):
            self._update_progress(ctx, stage, "Skipping (already completed)")
            queries = self.repo.get_search_queries_by_run(ctx.run_id)
            return StageMetrics(
                stage=stage,
                duration_seconds=0,
                success_count=len(queries),
                extra_metrics={'queries_loaded': len(queries)}
            )

        self._update_progress(ctx, stage, f"Generating search queries for: {ctx.question_text}")

        try:
            # Generate queries (synchronous call)
            query_objects = ctx.query_generator.generate_queries(
                question=ctx.question_text,
                run_id=ctx.run_id,
                prior_queries=None
            )

            # Save to database
            if not ctx.dry_run:
                for query in query_objects:
                    self.repo.create_search_query(query)

            # Extract query texts for next stage
            queries = [q.query_text for q in query_objects]

            # Save to context for later stages
            ctx.generated_queries = query_objects

            self._update_stage_state(ctx, PipelineStage.QUERY_GEN)

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info("stage_complete", stage=stage, duration=duration, queries=len(queries))

            return StageMetrics(
                stage=stage,
                duration_seconds=duration,
                success_count=len(queries),
                extra_metrics={'queries_generated': len(queries)}
            )

        except Exception as e:
            self._log_error(ctx, stage, e)
            raise

    async def _stage_search(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 2: Web Search.

        Execute search queries and collect URLs.
        """
        start_time = datetime.now()
        stage = PipelineStage.SEARCH.value

        if self._is_stage_completed(ctx, PipelineStage.SEARCH):
            self._update_progress(ctx, stage, "Skipping (already completed)")
            # Count existing results
            queries = self.repo.get_search_queries_by_run(ctx.run_id)
            total_results = sum(len(self.repo.get_search_results_by_query(q.id)) for q in queries)
            return StageMetrics(
                stage=stage,
                duration_seconds=0,
                success_count=total_results,
                extra_metrics={'results_loaded': total_results}
            )

        # Get queries from database
        queries = self.repo.get_search_queries_by_run(ctx.run_id)
        if not queries:
            raise ValueError("No search queries found. Run query generation first.")

        self._update_progress(ctx, stage, f"Searching {len(queries)} queries")

        success_count = 0
        fail_count = 0
        total_results = 0

        for i, query in enumerate(queries, 1):
            try:
                self._update_progress(ctx, stage, f"Query {i}/{len(queries)}: {query.query_text[:50]}...")

                # Execute search (synchronous)
                results = ctx.search_service.search(
                    query=query.query_text,
                    query_id=query.id,
                    num_results=ctx.max_urls or settings.search_results_per_query
                )

                # Save results to database (results are already SearchResult objects)
                if not ctx.dry_run:
                    for result in results:
                        self.repo.create_search_result(result)

                total_results += len(results)
                success_count += 1

            except Exception as e:
                fail_count += 1
                self._log_error(ctx, stage, e, reference=query.query_text)
                # Continue with next query

        self._update_stage_state(ctx, PipelineStage.SEARCH)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "stage_complete",
            stage=stage,
            duration=duration,
            success=success_count,
            failed=fail_count,
            total_results=total_results
        )

        return StageMetrics(
            stage=stage,
            duration_seconds=duration,
            success_count=success_count,
            fail_count=fail_count,
            extra_metrics={'total_urls': total_results}
        )

    async def _stage_scrape(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 3: Scraping & Content Extraction.

        Fetch and extract content from URLs.
        """
        start_time = datetime.now()
        stage = PipelineStage.SCRAPE.value

        if self._is_stage_completed(ctx, PipelineStage.SCRAPE):
            self._update_progress(ctx, stage, "Skipping (already completed)")
            documents = self.repo.get_documents_by_run(ctx.run_id)
            return StageMetrics(
                stage=stage,
                duration_seconds=0,
                success_count=len(documents),
                extra_metrics={'documents_loaded': len(documents)}
            )

        # Get all URLs from search results
        queries = self.repo.get_search_queries_by_run(ctx.run_id)
        all_urls = []
        for query in queries:
            results = self.repo.get_search_results_by_query(query.id)
            all_urls.extend([r.url for r in results])

        # Deduplicate URLs
        unique_urls = list(dict.fromkeys(all_urls))

        if not unique_urls:
            raise ValueError("No URLs found. Run search first.")

        self._update_progress(ctx, stage, f"Scraping {len(unique_urls)} unique URLs")

        # Scrape URLs (using module-level function)
        scrape_results = await scrape_urls(
            urls=unique_urls,
            run_id=ctx.run_id,
            max_concurrent=5,
            timeout=30,
            check_robots=True
        )

        success_count = 0
        fail_count = 0

        for result in scrape_results:
            if result.success and result.document:
                doc = result.document

                # Check if document already exists
                if not ctx.dry_run:
                    existing = self.repo.get_document_by_hash(ctx.run_id, doc.content_hash)
                    if existing:
                        continue  # Skip duplicate

                    # Create document record
                    document = Document(
                        run_id=ctx.run_id,
                        url=doc.url,
                        content_hash=doc.content_hash,
                        raw_content=doc.raw_content,
                        cleaned_content=doc.cleaned_content,
                        status='success'
                    )
                    self.repo.create_document(document)

                success_count += 1
            else:
                fail_count += 1
                if result.error_message:
                    self._log_error(ctx, stage, Exception(result.error_message), reference=result.url)

        self._update_stage_state(ctx, PipelineStage.SCRAPE)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "stage_complete",
            stage=stage,
            duration=duration,
            success=success_count,
            failed=fail_count
        )

        return StageMetrics(
            stage=stage,
            duration_seconds=duration,
            success_count=success_count,
            fail_count=fail_count,
            extra_metrics={
                'urls_scraped': len(unique_urls),
                'documents_saved': success_count
            }
        )

    async def _stage_event_extract(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 4: Event Extraction.

        Extract timestamped events from documents.
        """
        start_time = datetime.now()
        stage = PipelineStage.EVENT_EXTRACT.value

        # Check if already completed
        if self._is_stage_completed(ctx, PipelineStage.EVENT_EXTRACT):
            self._update_progress(ctx, stage, "Skipping (already completed)")
            events = self.repo.get_events_for_run(ctx.run_id)
            return StageMetrics(
                stage=stage,
                duration_seconds=0,
                success_count=len(events),
                extra_metrics={'events_loaded': len(events)}
            )

        # Get documents
        documents = self.repo.get_documents_by_run(ctx.run_id)
        if not documents:
            raise ValueError("No documents found. Run scraping first.")

        self._update_progress(ctx, stage, f"Extracting events from {len(documents)} documents")

        total_events = 0
        success_count = 0
        fail_count = 0

        for i, doc in enumerate(documents, 1):
            try:
                self._update_progress(ctx, stage, f"Document {i}/{len(documents)}")

                # Chunk document (pass Document object, not string)
                chunks = ctx.doc_processor.chunk_document(doc)

                # Log document extraction start with chunk info
                avg_chunk_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
                self.logger.info(
                    "document_event_extraction_start",
                    doc_id=doc.id,
                    doc_num=f"{i}/{len(documents)}",
                    url=doc.url[:100] if doc.url else None,
                    chunk_count=len(chunks),
                    avg_chunk_tokens=round(avg_chunk_tokens, 0),
                    total_doc_tokens=sum(c.token_count for c in chunks),
                )

                # Extract events from chunks (async method)
                events = await ctx.event_extractor.extract_from_chunks(chunks=chunks)

                # Log document extraction completion
                self.logger.info(
                    "document_event_extraction_complete",
                    doc_id=doc.id,
                    doc_num=f"{i}/{len(documents)}",
                    chunk_count=len(chunks),
                    event_count=len(events),
                    events_per_chunk=round(len(events) / len(chunks), 2) if chunks else 0,
                )

                # Apply max_events limit if specified
                if ctx.max_events and total_events + len(events) > ctx.max_events:
                    events = events[:ctx.max_events - total_events]

                # Save events to database
                if not ctx.dry_run:
                    for event in events:
                        event_record = Event(
                            document_id=doc.id,
                            event_time=event.timestamp,  # ExtractedEvent uses 'timestamp' attribute
                            headline=event.headline,
                            body=event.body,
                            actors=json.dumps(event.actors) if event.actors else None,
                            confidence=event.confidence,
                            raw_response=event.raw_response
                        )
                        self.repo.create_event(event_record)

                total_events += len(events)
                success_count += 1

                # Stop if we've reached max_events
                if ctx.max_events and total_events >= ctx.max_events:
                    break

            except Exception as e:
                fail_count += 1
                self._log_error(ctx, stage, e, reference=str(doc.id))
                # Continue with next document

        self._update_stage_state(ctx, PipelineStage.EVENT_EXTRACT)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "stage_complete",
            stage=stage,
            duration=duration,
            documents_processed=success_count,
            documents_failed=fail_count,
            total_events=total_events
        )

        return StageMetrics(
            stage=stage,
            duration_seconds=duration,
            success_count=success_count,
            fail_count=fail_count,
            extra_metrics={
                'documents_processed': len(documents),
                'total_events': total_events
            }
        )

    async def _stage_cluster(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 5: Embedding, Clustering, Deduplication.

        Generate embeddings and cluster similar events.
        """
        start_time = datetime.now()
        stage = PipelineStage.CLUSTER.value

        if self._is_stage_completed(ctx, PipelineStage.CLUSTER):
            self._update_progress(ctx, stage, "Skipping (already completed)")
            clusters = self.repo.get_event_clusters_by_run(ctx.run_id)
            return StageMetrics(
                stage=stage,
                duration_seconds=0,
                success_count=len(clusters),
                extra_metrics={'clusters_loaded': len(clusters)}
            )

        # Get events
        events = self.repo.get_events_for_run(ctx.run_id)
        if not events:
            raise ValueError("No events found. Run event extraction first.")

        self._update_progress(ctx, stage, f"Generating embeddings for {len(events)} events")

        # Generate embeddings
        event_texts = [f"{e.headline or ''} {e.body or ''}" for e in events]
        embeddings = await ctx.embedding_service.embed_batch_async(event_texts)

        # Save embeddings to database
        if not ctx.dry_run:
            for event, emb in zip(events, embeddings):
                embedding_record = Embedding(
                    event_id=event.id,
                    vector=emb.to_bytes(),  # Convert numpy array to bytes
                    model=emb.model,
                    dimensions=emb.dimensions
                )
                self.repo.create_embedding(embedding_record)

        self._update_progress(ctx, stage, f"Clustering {len(events)} events")

        # Cluster events (pass Event objects and Embedding objects)
        clustering_result = ctx.clustering_service.cluster_events(
            events=events,
            embeddings=embeddings
        )

        # Save clusters to database
        if not ctx.dry_run:
            for cluster in clustering_result.clusters:
                cluster_record = EventCluster(
                    run_id=ctx.run_id,
                    label=cluster.label,
                    centroid_event_id=cluster.centroid_event_id,
                    member_ids=json.dumps(cluster.member_event_ids)
                )
                self.repo.create_event_cluster(cluster_record)

        self._update_stage_state(ctx, PipelineStage.CLUSTER)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "stage_complete",
            stage=stage,
            duration=duration,
            events=len(events),
            clusters=len(clustering_result.clusters)
        )

        return StageMetrics(
            stage=stage,
            duration_seconds=duration,
            success_count=len(clustering_result.clusters),
            extra_metrics={
                'events': len(events),
                'clusters': len(clustering_result.clusters),
                'silhouette_score': clustering_result.silhouette_score or 0.0
            }
        )

    async def _stage_timeline(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 6: Timeline Construction.

        Build chronological timeline from clustered events.
        """
        start_time = datetime.now()
        stage = PipelineStage.TIMELINE.value

        if self._is_stage_completed(ctx, PipelineStage.TIMELINE):
            self._update_progress(ctx, stage, "Skipping (already completed)")
            timeline_entries = self.repo.get_timeline_for_run(ctx.run_id)
            return StageMetrics(
                stage=stage,
                duration_seconds=0,
                success_count=len(timeline_entries),
                extra_metrics={'entries_loaded': len(timeline_entries)}
            )

        self._update_progress(ctx, stage, "Building timeline")

        # Get clusters and events
        db_clusters = self.repo.get_event_clusters_by_run(ctx.run_id)
        if not db_clusters:
            raise ValueError("No clusters found. Run clustering first.")

        # Get all events for the run
        all_events = self.repo.get_events_for_run(ctx.run_id)

        # Get documents for URLs
        documents = self.repo.get_documents_by_run(ctx.run_id)

        # Convert database EventCluster models to service EventCluster dataclasses
        service_clusters = []
        for db_cluster in db_clusters:
            # Parse member_ids from JSON
            member_ids = json.loads(db_cluster.member_ids) if db_cluster.member_ids else []

            service_cluster = ServiceEventCluster(
                cluster_id=db_cluster.id,
                centroid_event_id=db_cluster.centroid_event_id,
                member_event_ids=member_ids,
                label=db_cluster.label,
                confidence_score=0.0,  # Not stored in DB, use default
                merged_citations=[],   # Will be computed by timeline builder
                merged_actors=[],      # Will be computed by timeline builder
            )
            service_clusters.append(service_cluster)

        # Reconstruct ClusteringResult
        clustering_result = ClusteringResult(
            clusters=service_clusters,
            algorithm=ClusteringAlgorithm.AGGLOMERATIVE,  # Default, not stored in DB
            n_clusters=len(service_clusters),
            silhouette_score=0.0,  # Not stored in DB
            cluster_size_distribution={},  # Not critical for timeline
            original_count=len(all_events),
            deduplicated_count=len(service_clusters),
        )

        # Build timeline
        timeline = ctx.timeline_builder.build_timeline(
            clustering_result=clustering_result,
            events=list(all_events),
            documents=documents
        )

        # Save timeline to database
        if not ctx.dry_run:
            for entry in timeline.entries:
                timeline_entry = entry.to_model(ctx.run_id)
                self.repo.create_timeline_entry(timeline_entry)

        self._update_stage_state(ctx, PipelineStage.TIMELINE)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "stage_complete",
            stage=stage,
            duration=duration,
            entries=len(timeline.entries),
            coverage=timeline.coverage_level.value
        )

        return StageMetrics(
            stage=stage,
            duration_seconds=duration,
            success_count=len(timeline.entries),
            extra_metrics={
                'timeline_entries': len(timeline.entries),
                'coverage_level': timeline.coverage_level.value
            }
        )

    async def _stage_forecast(self, ctx: RunContext) -> StageMetrics:
        """
        Stage 7: Forecast Generation.

        Generate final forecast with reasoning and probability.
        """
        start_time = datetime.now()
        stage = PipelineStage.FORECAST.value

        if self._is_stage_completed(ctx, PipelineStage.FORECAST):
            self._update_progress(ctx, stage, "Skipping (already completed)")
            forecast = self.repo.get_forecast_by_run(ctx.run_id)
            return StageMetrics(
                stage=stage,
                duration_seconds=0,
                success_count=1 if forecast else 0,
                extra_metrics={'forecast_loaded': bool(forecast)}
            )

        self._update_progress(ctx, stage, "Generating forecast")

        # Get timeline entries
        timeline_entries = self.repo.get_timeline_for_run(ctx.run_id)
        if not timeline_entries:
            raise ValueError("No timeline entries found. Run timeline building first.")

        # Get timeline builder to reconstruct Timeline object
        all_events = self.repo.get_events_for_run(ctx.run_id)
        documents = self.repo.get_documents_by_run(ctx.run_id)
        db_clusters = self.repo.get_event_clusters_by_run(ctx.run_id)

        # Convert database EventCluster models to service EventCluster dataclasses
        service_clusters = []
        for db_cluster in db_clusters:
            # Parse member_ids from JSON
            member_ids = json.loads(db_cluster.member_ids) if db_cluster.member_ids else []

            service_cluster = ServiceEventCluster(
                cluster_id=db_cluster.id,
                centroid_event_id=db_cluster.centroid_event_id,
                member_event_ids=member_ids,
                label=db_cluster.label,
                confidence_score=0.0,  # Not stored in DB, use default
                merged_citations=[],   # Will be computed by timeline builder
                merged_actors=[],      # Will be computed by timeline builder
            )
            service_clusters.append(service_cluster)

        # Reconstruct ClusteringResult
        clustering_result = ClusteringResult(
            clusters=service_clusters,
            algorithm=ClusteringAlgorithm.AGGLOMERATIVE,  # Default, not stored in DB
            n_clusters=len(service_clusters),
            silhouette_score=0.0,  # Not stored in DB
            cluster_size_distribution={},  # Not critical for timeline
            original_count=len(all_events),
            deduplicated_count=len(service_clusters),
        )

        timeline = ctx.timeline_builder.build_timeline(
            clustering_result=clustering_result,
            events=list(all_events),
            documents=documents
        )

        # Generate forecast (synchronous method)
        forecast = ctx.forecast_generator.generate_forecast(
            question=ctx.question_text,
            timeline=timeline
        )

        # Save forecast to database
        if not ctx.dry_run:
            forecast_record = forecast.to_model(ctx.run_id)
            self.repo.create_forecast(forecast_record)

        self._update_stage_state(ctx, PipelineStage.FORECAST)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "stage_complete",
            stage=stage,
            duration=duration,
            prediction=str(forecast.prediction),
            prediction_type=forecast.prediction_type.value
        )

        return StageMetrics(
            stage=stage,
            duration_seconds=duration,
            success_count=1,
            extra_metrics={
                'prediction': str(forecast.prediction),
                'prediction_type': forecast.prediction_type.value,
                'confidence_level': forecast.confidence_level.value
            }
        )

    async def _save_outputs(self, ctx: RunContext) -> None:
        """
        Save final outputs (JSON and Markdown).

        Args:
            ctx: Run context
        """
        if ctx.dry_run:
            return

        self._update_progress(ctx, "OUTPUT", "Saving outputs")

        # Load all data
        question = self.repo.get_question_by_id(ctx.question_id)
        run = self.repo.get_run_by_id(ctx.run_id)
        queries = self.repo.get_search_queries_by_run(ctx.run_id)
        timeline_entries = self.repo.get_timeline_for_run(ctx.run_id)
        forecast = self.repo.get_forecast_by_run(ctx.run_id)
        metrics = self.repo.get_metrics_by_run(ctx.run_id)
        errors = self.repo.get_errors_by_run(ctx.run_id)

        # Prepare JSON output
        output_data = {
            'question': {
                'id': question.id,
                'text': question.question_text,
                'resolution_criteria': question.resolution_criteria
            },
            'run': {
                'id': run.id,
                'started_at': run.started_at.isoformat() if run.started_at else None,
                'completed_at': run.completed_at.isoformat() if run.completed_at else None,
                'status': run.status.value,
                'git_commit': run.git_commit
            },
            'queries': [q.query_text for q in queries],
            'timeline': [
                {
                    'event_time': e.event_time,
                    'summary': e.summary,
                    'citations': json.loads(e.citations) if e.citations else [],
                    'tags': json.loads(e.tags) if e.tags else {}
                }
                for e in timeline_entries
            ],
            'forecast': {
                'probability': forecast.probability if forecast else None,
                'reasoning': forecast.reasoning if forecast else None,
                'caveats': json.loads(forecast.caveats) if forecast and forecast.caveats else []
            } if forecast else None,
            'metrics': {m.metric_name: m.metric_value for m in metrics},
            'errors': [
                {
                    'stage': e.stage,
                    'type': e.error_type,
                    'message': e.message
                }
                for e in errors
            ]
        }

        # Save JSON
        json_path = ctx.output_dir / "forecast_output.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Generate Markdown report
        md_content = self._generate_markdown_report(output_data)
        md_path = ctx.output_dir / "forecast_report.md"
        with open(md_path, 'w') as f:
            f.write(md_content)

        self.logger.info("outputs_saved", json=str(json_path), markdown=str(md_path))

    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """
        Generate Markdown report from output data.

        Args:
            data: Output data dictionary

        Returns:
            Markdown-formatted report
        """
        lines = [
            "# Forecast Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Question",
            "",
            f"{data['question']['text']}",
            ""
        ]

        if data['question']['resolution_criteria']:
            lines.extend([
                "### Resolution Criteria",
                "",
                f"{data['question']['resolution_criteria']}",
                ""
            ])

        # Forecast
        if data['forecast']:
            lines.extend([
                "## Forecast",
                "",
                f"**Probability:** {data['forecast']['probability']}",
                "",
                "### Reasoning",
                "",
                f"{data['forecast']['reasoning']}",
                ""
            ])

            if data['forecast']['caveats']:
                lines.extend([
                    "### Caveats",
                    ""
                ])
                for caveat in data['forecast']['caveats']:
                    lines.append(f"- {caveat}")
                lines.append("")

        # Timeline
        if data['timeline']:
            lines.extend([
                "## Timeline",
                "",
                f"**Total Events:** {len(data['timeline'])}",
                ""
            ])

            for entry in data['timeline']:
                lines.extend([
                    f"### {entry['event_time']}",
                    "",
                    f"{entry['summary']}",
                    ""
                ])

                if entry['citations']:
                    lines.append("**Sources:**")
                    for citation in entry['citations']:
                        lines.append(f"- [{citation.get('url', 'N/A')}]({citation.get('url', '#')})")
                    lines.append("")

        # Metrics
        if data['metrics']:
            lines.extend([
                "## Pipeline Metrics",
                ""
            ])
            for key, value in sorted(data['metrics'].items()):
                lines.append(f"- **{key}:** {value:.2f}")
            lines.append("")

        # Errors
        if data['errors']:
            lines.extend([
                "## Errors",
                "",
                f"**Total Errors:** {len(data['errors'])}",
                ""
            ])
            for error in data['errors']:
                lines.append(f"- **{error['stage']}:** {error['message']}")
            lines.append("")

        return "\n".join(lines)

    async def run_pipeline(self, ctx: RunContext) -> None:
        """
        Run the complete pipeline.

        Args:
            ctx: Run context with question and configuration

        Raises:
            Exception: If any stage fails critically
        """
        ctx.repo = self.repo

        # Update run status
        if not ctx.dry_run:
            self.repo.update_run_status(ctx.run_id, RunStatus.RUNNING)

        try:
            # Execute all stages
            stages = [
                self._stage_init,
                self._stage_query_gen,
                self._stage_search,
                self._stage_scrape,
                self._stage_event_extract,
                self._stage_cluster,
                self._stage_timeline,
                self._stage_forecast
            ]

            for stage_func in stages:
                ctx.current_stage = stage_func.__name__
                metrics = await stage_func(ctx)
                ctx.stage_metrics.append(metrics)
                self._save_stage_metrics(ctx, metrics)

            # Save outputs
            await self._save_outputs(ctx)

            # Mark run as completed
            if not ctx.dry_run:
                self.repo.update_run_status(
                    ctx.run_id,
                    RunStatus.COMPLETED,
                    completed_at=datetime.now()
                )

            self.logger.info("pipeline_complete", run_id=ctx.run_id)

        except Exception as e:
            self.logger.error("pipeline_failed", run_id=ctx.run_id, error=str(e))
            if not ctx.dry_run:
                self.repo.update_run_status(ctx.run_id, RunStatus.FAILED)
            raise


async def run_forecast_pipeline(
    question_text: str,
    resume: bool = False,
    max_urls: Optional[int] = None,
    max_events: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> int:
    """
    High-level function to run the forecast pipeline.

    Args:
        question_text: The forecasting question
        resume: Resume from last checkpoint if True
        max_urls: Maximum URLs to scrape per query
        max_events: Maximum events to extract
        dry_run: Don't save to database if True
        verbose: Enable verbose logging
        output_dir: Custom output directory
        progress_callback: Callback for progress updates

    Returns:
        Run ID
    """
    repo = DatabaseRepository()
    orchestrator = PipelineOrchestrator(repo)

    # Create or load question
    if not dry_run:
        question = Question(question_text=question_text)
        question_id = repo.create_question(question)

        # Get git commit
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            git_commit = None

        # Create run
        run = Run(
            question_id=question_id,
            status=RunStatus.PENDING,
            git_commit=git_commit
        )
        run_id = repo.create_run(run)
    else:
        question_id = 1  # Dummy for dry run
        run_id = 1

    # Create context
    ctx = RunContext(
        question_id=question_id,
        run_id=run_id,
        question_text=question_text,
        resume=resume,
        max_urls=max_urls,
        max_events=max_events,
        dry_run=dry_run,
        verbose=verbose,
        output_dir=output_dir,
        progress_callback=progress_callback
    )

    # Run pipeline
    await orchestrator.run_pipeline(ctx)

    return run_id
