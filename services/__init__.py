"""
Services package for AI Forecasting Pipeline.

This package contains all service modules including:
- llm_client: LLM provider wrapper with retry logic and error handling
- query_generation: LLM-based search query generation
- search: Google Custom Search API integration
- scraper: Web content scraping and extraction
- doc_processor: Document chunking and preprocessing
- event_extractor: Event extraction from documents
- embedding: Text embedding generation
- clustering: Event clustering and deduplication
- timeline: Timeline construction
- forecast: Forecast generation with LLM
"""

# LLM Client
from services.llm_client import (
    LLMClient,
    GeminiClient,
    create_llm_client,
    LLMError,
    QuotaExhaustedError as LLMQuotaExhaustedError,
    ParsingError,
    RateLimitError as LLMRateLimitError,
)

# Query Generation
from services.query_generation import (
    QueryGenerator,
    generate_queries,
    get_query_generation_prompt,
    QUERY_GENERATION_PROMPT_VERSION,
)

# Search Service
from services.search import (
    GoogleSearchService,
    search_query,
    search_multiple_queries,
    SearchError,
    SearchErrorType,
    QuotaExceededError as SearchQuotaExceededError,
    RateLimitError as SearchRateLimitError,
)

# Web Scraper
from services.scraper import (
    WebScraper,
    scrape_urls,
    DocumentFetchResult,
    ScraperError,
    ScraperErrorType,
    RobotsChecker,
    extract_urls_from_text,
)

# Document Processor
from services.doc_processor import (
    DocumentProcessor,
    DocumentChunk,
    chunk_document,
    chunk_documents,
)

# Event Extractor
from services.event_extractor import (
    EventExtractor,
    ExtractedEvent,
    TimestampSpecificity,
    extract_events_from_chunks,
    filter_events_by_confidence,
    filter_events_by_specificity,
    get_event_extraction_prompt,
    EVENT_EXTRACTION_PROMPT_VERSION,
)

# Embedding Service
from services.embedding import (
    EmbeddingService,
    Embedding,
    EmbeddingError,
    EmbeddingQuotaError,
    EmbeddingRateLimitError,
    generate_embeddings,
    generate_embeddings_async,
    cosine_similarity,
    cosine_distance,
    compute_similarity_matrix,
    compute_distance_matrix,
)

# Clustering Service
from services.clustering import (
    ClusteringService,
    EventCluster,
    ClusteringResult,
    ClusteringAlgorithm,
    ClusteringError,
    cluster_events,
    deduplicate_events,
)

# Timeline Service
from services.timeline import (
    TimelineBuilder,
    Timeline,
    TimelineEntry,
    Citation,
    CoverageLevel,
    TimelineError,
    build_timeline,
    format_timeline_markdown,
)

# Forecast Service
from services.forecast import (
    ForecastGenerator,
    Forecast,
    ForecastOutput,
    ForecastError,
    ConfidenceLevel,
    PredictionType,
    generate_forecast,
    get_forecast_prompt_template,
    FORECAST_PROMPT_VERSION,
)

__all__ = [
    # LLM Client
    "LLMClient",
    "GeminiClient",
    "create_llm_client",
    "LLMError",
    "LLMQuotaExhaustedError",
    "ParsingError",
    "LLMRateLimitError",
    # Query Generation
    "QueryGenerator",
    "generate_queries",
    "get_query_generation_prompt",
    "QUERY_GENERATION_PROMPT_VERSION",
    # Search
    "GoogleSearchService",
    "search_query",
    "search_multiple_queries",
    "SearchError",
    "SearchErrorType",
    "SearchQuotaExceededError",
    "SearchRateLimitError",
    # Scraper
    "WebScraper",
    "scrape_urls",
    "DocumentFetchResult",
    "ScraperError",
    "ScraperErrorType",
    "RobotsChecker",
    "extract_urls_from_text",
    # Document Processor
    "DocumentProcessor",
    "DocumentChunk",
    "chunk_document",
    "chunk_documents",
    # Event Extractor
    "EventExtractor",
    "ExtractedEvent",
    "TimestampSpecificity",
    "extract_events_from_chunks",
    "filter_events_by_confidence",
    "filter_events_by_specificity",
    "get_event_extraction_prompt",
    "EVENT_EXTRACTION_PROMPT_VERSION",
    # Embedding
    "EmbeddingService",
    "Embedding",
    "EmbeddingError",
    "EmbeddingQuotaError",
    "EmbeddingRateLimitError",
    "generate_embeddings",
    "generate_embeddings_async",
    "cosine_similarity",
    "cosine_distance",
    "compute_similarity_matrix",
    "compute_distance_matrix",
    # Clustering
    "ClusteringService",
    "EventCluster",
    "ClusteringResult",
    "ClusteringAlgorithm",
    "ClusteringError",
    "cluster_events",
    "deduplicate_events",
    # Timeline
    "TimelineBuilder",
    "Timeline",
    "TimelineEntry",
    "Citation",
    "CoverageLevel",
    "TimelineError",
    "build_timeline",
    "format_timeline_markdown",
    # Forecast
    "ForecastGenerator",
    "Forecast",
    "ForecastOutput",
    "ForecastError",
    "ConfidenceLevel",
    "PredictionType",
    "generate_forecast",
    "get_forecast_prompt_template",
    "FORECAST_PROMPT_VERSION",
]
