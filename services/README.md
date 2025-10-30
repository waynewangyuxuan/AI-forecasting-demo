# Services Module Documentation

This module contains the core services for the AI Forecasting Pipeline's query generation and search functionality.

## Overview

The services module provides three main components:

1. **LLM Client** (`llm_client.py`): Abstract interface and concrete implementation for LLM providers
2. **Query Generation** (`query_generation.py`): LLM-powered search query generation with diversification
3. **Search Service** (`search.py`): Google Custom Search API integration with rate limiting and error handling

## Components

### 1. LLM Client (`services/llm_client.py`)

Provides a unified interface for interacting with LLM providers with built-in retry logic, rate limiting, and error handling.

#### Key Classes

- **`LLMClient`**: Abstract base class defining the interface for LLM providers
- **`GeminiClient`**: Concrete implementation for Google Gemini API
- **Exception Types**: `LLMError`, `QuotaExhaustedError`, `ParsingError`, `RateLimitError`

#### Features

- Exponential backoff retry logic (up to 3 attempts)
- Rate limiting with configurable intervals
- Response caching for development
- Usage statistics tracking (request count, token usage)
- Comprehensive error categorization
- JSON response parsing with validation
- Safety settings configuration for Gemini

#### Usage Example

```python
from services import GeminiClient

# Initialize client
client = GeminiClient()

# Generate text
response = client.generate(
    prompt="What are the latest developments in robotics?",
    temperature=0.7,
    max_tokens=1000
)

# Generate JSON response
json_data = client.generate_json(
    prompt="List 5 robotics companies. Return as JSON array.",
    temperature=0.5
)

# Get usage statistics
stats = client.get_stats()
print(f"Requests: {stats['request_count']}, Tokens: {stats['total_tokens']}")
```

### 2. Query Generation (`services/query_generation.py`)

Generates diverse, optimized search queries for forecasting questions using LLM-powered query expansion.

#### Key Classes

- **`QueryGenerator`**: Main service for generating search queries
- **Helper Functions**: `generate_queries()`, `get_query_generation_prompt()`

#### Features

- LLM-powered query generation (~10 queries per question)
- Multi-angle coverage (temporal, technical, market, regulatory)
- Intelligent deduplication (exact and fuzzy matching)
- Keyword coverage validation
- Length validation (3-150 characters)
- Prompt versioning for tracking
- Post-processing filters

#### Query Diversification Strategy

The prompt template instructs the LLM to generate queries covering:
- **Temporal aspects**: specific years, "recent", "latest", timeline keywords
- **Technical details**: implementation specifics, technology capabilities
- **Market trends**: industry analysis, production capacity, forecasts
- **Regulatory/policy**: government policies, regulations, legal frameworks
- **Expert opinions**: analyst reports, research papers, industry commentary

#### Usage Example

```python
from services import QueryGenerator

# Initialize generator
generator = QueryGenerator()

# Generate queries
queries = generator.generate_queries(
    question="Will China be able to mass-produce humanoid robots by end of 2025?",
    run_id=1
)

# Queries are returned as SearchQuery objects ready for database persistence
for query in queries:
    print(f"Query ID: {query.id}, Text: {query.query_text}")
```

#### Prompt Template

The query generation prompt:
1. Explains the forecasting context
2. Requests exactly 10 diverse queries
3. Provides specific guidelines (temporal specificity, varied terminology, etc.)
4. Includes prior queries to avoid duplication
5. Requests JSON output format

### 3. Search Service (`services/search.py`)

Integrates with Google Custom Search JSON API to retrieve search results with comprehensive error handling and rate limiting.

#### Key Classes

- **`GoogleSearchService`**: Main service for executing searches
- **Exception Types**: `SearchError`, `QuotaExceededError`, `RateLimitError`
- **Enums**: `SearchErrorType` (quota, rate_limit, transient, fatal, network)

#### Features

- Rate limiting to respect API quotas (100 queries/day free tier)
- Exponential backoff retry logic (up to 3 attempts)
- Comprehensive error categorization
- Raw API response storage for debugging
- Top-N URL extraction per query
- Batch search with configurable delays
- Usage statistics tracking

#### Error Handling

The service categorizes errors into:
- **Quota Exceeded**: API daily limit reached (non-retryable)
- **Rate Limit**: Too many requests (retryable with backoff)
- **Transient**: Temporary server errors (retryable)
- **Fatal**: Invalid requests (non-retryable)
- **Network**: Connection issues (retryable)

#### Usage Example

```python
from services import GoogleSearchService

# Initialize service
search_service = GoogleSearchService()

# Search single query
results = search_service.search(
    query="China humanoid robot production 2025",
    query_id=1
)

# Results are SearchResult objects
for result in results:
    print(f"Rank {result.rank}: {result.title}")
    print(f"URL: {result.url}")
    print(f"Snippet: {result.snippet}\n")

# Batch search
queries = [
    ("robotics manufacturing China", 1),
    ("humanoid robot timeline 2025", 2),
]
results_map = search_service.search_multiple(queries, batch_delay=1.0)

# Get service statistics
stats = search_service.get_stats()
print(f"Requests: {stats['request_count']}")
print(f"Quota exhausted: {stats['quota_exhausted']}")
```

## Configuration

All services use settings from `config.settings.py`:

```python
from config.settings import settings

# LLM Configuration
settings.google_gemini_api_key    # Gemini API key
settings.gemini_model              # Model name (default: gemini-2.0-flash-exp)

# Search Configuration
settings.google_api_key            # Google Custom Search API key
settings.google_cse_id             # Custom Search Engine ID
settings.search_results_per_query  # Results per query (default: 10)
settings.max_search_queries        # Max queries to generate (default: 10)

# HTTP Configuration
settings.request_timeout           # Request timeout in seconds (default: 30)
settings.user_agent               # User agent string
```

## Complete Workflow Example

Here's how the services work together in the forecasting pipeline:

```python
from services import GeminiClient, QueryGenerator, GoogleSearchService
from config.settings import settings

# 1. Initialize services
llm_client = GeminiClient()
query_generator = QueryGenerator(llm_client=llm_client)
search_service = GoogleSearchService()

# 2. Generate search queries
question = "Will China be able to mass-produce humanoid robots by end of 2025?"
queries = query_generator.generate_queries(
    question=question,
    run_id=1
)

print(f"Generated {len(queries)} queries:")
for q in queries:
    print(f"  - {q.query_text}")

# 3. Execute searches (after persisting queries to database)
search_results = []
for query in queries:
    # In real usage, query.id would come from database
    results = search_service.search(
        query=query.query_text,
        query_id=query.id or 0
    )
    search_results.extend(results)

print(f"\nRetrieved {len(search_results)} total search results")

# 4. Get statistics
llm_stats = llm_client.get_stats()
search_stats = search_service.get_stats()

print(f"\nLLM Stats: {llm_stats}")
print(f"Search Stats: {search_stats}")
```

## Error Handling Best Practices

### Handling Quota Exhaustion

```python
from services import (
    QueryGenerator,
    GoogleSearchService,
    LLMQuotaExhaustedError,
    SearchQuotaExceededError
)

try:
    queries = query_generator.generate_queries(question, run_id)
except LLMQuotaExhaustedError as e:
    logger.error("LLM quota exhausted", error=str(e))
    # Fallback: use cached queries or manual query list
    queries = get_fallback_queries(question)

try:
    results = search_service.search(query_text, query_id)
except SearchQuotaExceededError as e:
    logger.error("Search quota exhausted", error=str(e))
    # Stop processing remaining queries
    break
```

### Handling Rate Limits

Rate limits are automatically handled with retry logic, but you can also add manual delays:

```python
import time

for query in queries:
    try:
        results = search_service.search(query.query_text, query.id)
        time.sleep(1.0)  # Additional delay between queries
    except SearchRateLimitError as e:
        logger.warning("Rate limited, waiting longer", error=str(e))
        time.sleep(5.0)  # Wait longer before retry
```

## Testing

The services include comprehensive error handling and logging. To test:

```python
# Enable debug mode for detailed logging
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)

# Enable response caching for development
client = GeminiClient(cache_responses=True)

# Enable raw response storage for debugging
search_service = GoogleSearchService()
# ... perform searches ...
raw_responses = search_service.get_raw_responses()
```

## Dependencies

Required packages (defined in `requirements.txt`):
- `google-generativeai>=0.3.0` - Gemini API client
- `httpx>=0.25.0` - Async HTTP client
- `tenacity>=8.2.0` - Retry logic
- `structlog>=23.2.0` - Structured logging
- `pydantic>=2.4.0` - Data validation

## Architecture Notes

### Atomic File Structure

Each service module follows the atomic file structure principle:
- Self-contained functionality
- Clear separation of concerns
- Minimal cross-dependencies
- Easy to test in isolation

### Extensibility

The abstract `LLMClient` base class allows easy addition of other providers:

```python
class OpenAIClient(LLMClient):
    def generate(self, prompt, temperature=0.7, max_tokens=None, response_format=None):
        # Implementation for OpenAI API
        pass
```

### Type Safety

All services use:
- Type hints for parameters and return values
- Pydantic models for data validation
- Enum types for error categorization

## Logging

All services use structured logging with `structlog`:

```python
logger.info(
    "query_generation_complete",
    run_id=run_id,
    query_count=len(queries),
    duration_seconds=elapsed_time
)
```

Log events include:
- Operation start/completion
- Error details with context
- Performance metrics
- Quota/rate limit warnings

## Future Enhancements

Potential improvements for future iterations:
1. Add OpenAI GPT-4 client implementation
2. Implement response streaming for long-running LLM calls
3. Add persistent caching layer (Redis/SQLite)
4. Implement query quality scoring
5. Add A/B testing for different prompt templates
6. Implement circuit breaker pattern for API failures
7. Add metrics export (Prometheus format)
