"""
Google Search Service for AI Forecasting Pipeline.

Integrates with Google Custom Search JSON API to retrieve relevant URLs
for forecasting questions with rate limiting, error handling, and result storage.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from enum import Enum
import structlog
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config.settings import settings
from db.models import SearchResult

logger = structlog.get_logger(__name__)


class SearchErrorType(str, Enum):
    """Categories of search API errors."""
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    FATAL = "fatal"
    NETWORK = "network"


class SearchError(Exception):
    """Base exception for search-related errors."""

    def __init__(self, message: str, error_type: SearchErrorType):
        super().__init__(message)
        self.error_type = error_type


class QuotaExceededError(SearchError):
    """Raised when API quota is exhausted."""

    def __init__(self, message: str):
        super().__init__(message, SearchErrorType.QUOTA_EXCEEDED)


class RateLimitError(SearchError):
    """Raised when rate limit is hit."""

    def __init__(self, message: str):
        super().__init__(message, SearchErrorType.RATE_LIMIT)


class GoogleSearchService:
    """
    Service for executing Google Custom Search queries.

    Provides:
    - Rate limiting to respect quota (100 queries/day free tier)
    - Exponential backoff for transient errors
    - Comprehensive error categorization
    - Raw response storage for debugging
    - Top-N URL extraction
    """

    # Google Custom Search API endpoint
    API_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        results_per_query: Optional[int] = None,
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize Google Search service.

        Args:
            api_key: Google API key (defaults to settings)
            cse_id: Custom Search Engine ID (defaults to settings)
            results_per_query: Number of results per query (defaults to settings)
            rate_limit_delay: Minimum seconds between requests
        """
        self.api_key = api_key or settings.google_api_key
        self.cse_id = cse_id or settings.google_cse_id
        self.results_per_query = results_per_query or settings.search_results_per_query
        self.rate_limit_delay = rate_limit_delay

        # HTTP client with timeout
        self.client = httpx.Client(
            timeout=settings.request_timeout,
            headers={"User-Agent": settings.user_agent},
        )

        # Rate limiting state
        self._last_request_time = 0.0
        self._request_count = 0
        self._quota_exhausted = False

        # Storage for raw responses (debugging)
        self._raw_responses: List[Dict[str, Any]] = []

        self.logger = structlog.get_logger(__name__)
        self.logger.info(
            "search_service_initialized",
            cse_id=self.cse_id[:10] + "...",
            results_per_query=self.results_per_query,
            rate_limit_delay=rate_limit_delay,
        )

    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self.client.close()
        except:
            pass

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            self.logger.debug("rate_limiting", sleep_seconds=sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _categorize_error(self, response: httpx.Response) -> SearchErrorType:
        """
        Categorize API error based on response.

        Args:
            response: HTTP response object

        Returns:
            SearchErrorType enum value
        """
        status_code = response.status_code

        # Check response body for specific errors
        try:
            error_data = response.json()
            error_message = str(error_data.get("error", {})).lower()

            # Check for quota exhaustion
            if "quota" in error_message or "limit exceeded" in error_message:
                return SearchErrorType.QUOTA_EXCEEDED

            # Check for rate limiting
            if "rate" in error_message or "too many requests" in error_message:
                return SearchErrorType.RATE_LIMIT

        except Exception:
            pass

        # Categorize by status code
        if status_code == 429:
            return SearchErrorType.RATE_LIMIT
        elif status_code == 403:
            # Could be quota or auth issue
            return SearchErrorType.QUOTA_EXCEEDED
        elif 500 <= status_code < 600:
            return SearchErrorType.TRANSIENT
        elif status_code == 400:
            return SearchErrorType.FATAL
        else:
            return SearchErrorType.TRANSIENT

    def _handle_api_error(self, response: httpx.Response) -> None:
        """
        Handle API error response.

        Args:
            response: HTTP response object

        Raises:
            QuotaExceededError: For quota exhaustion
            RateLimitError: For rate limiting
            SearchError: For other errors
        """
        error_type = self._categorize_error(response)

        # Try to extract error details
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text

        self.logger.error(
            "search_api_error",
            status_code=response.status_code,
            error_type=error_type.value,
            error_message=error_message[:200],
        )

        # Raise appropriate exception
        if error_type == SearchErrorType.QUOTA_EXCEEDED:
            self._quota_exhausted = True
            raise QuotaExceededError(
                f"Google Search API quota exceeded: {error_message}"
            )
        elif error_type == SearchErrorType.RATE_LIMIT:
            raise RateLimitError(
                f"Google Search API rate limited: {error_message}"
            )
        elif error_type == SearchErrorType.FATAL:
            raise SearchError(
                f"Fatal search API error: {error_message}",
                SearchErrorType.FATAL,
            )
        else:
            raise SearchError(
                f"Search API error: {error_message}",
                error_type,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, SearchError)),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def search(
        self,
        query: str,
        query_id: int,
        num_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Execute a Google search query.

        Args:
            query: Search query string
            query_id: Database ID of the SearchQuery record
            num_results: Number of results to retrieve (defaults to service config)

        Returns:
            List of SearchResult objects

        Raises:
            QuotaExceededError: When quota is exhausted
            RateLimitError: When rate limited (will retry)
            SearchError: For other API errors (may retry)
        """
        # Check if quota already exhausted
        if self._quota_exhausted:
            raise QuotaExceededError("Search quota already exhausted")

        num_results = num_results or self.results_per_query

        self.logger.info(
            "search_start",
            query=query,
            query_id=query_id,
            num_results=num_results,
        )

        # Rate limiting
        self._rate_limit()

        # Build request parameters
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": num_results,
            "fields": "items(title,link,snippet,displayLink)",
        }

        try:
            # Execute search request
            response = self.client.get(self.API_ENDPOINT, params=params)

            # Update request count
            self._request_count += 1

            # Check for errors
            if response.status_code != 200:
                self._handle_api_error(response)

            # Parse response
            data = response.json()

            # Store raw response for debugging
            if settings.debug:
                self._raw_responses.append({
                    "query": query,
                    "query_id": query_id,
                    "response": data,
                    "timestamp": time.time(),
                })

            # Extract search results
            items = data.get("items", [])

            if not items:
                self.logger.warning(
                    "search_no_results",
                    query=query,
                    query_id=query_id,
                )
                return []

            # Convert to SearchResult objects
            search_results = []
            for rank, item in enumerate(items, start=1):
                search_result = SearchResult(
                    query_id=query_id,
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    rank=rank,
                )
                search_results.append(search_result)

            self.logger.info(
                "search_complete",
                query=query,
                query_id=query_id,
                result_count=len(search_results),
                request_count=self._request_count,
            )

            return search_results

        except httpx.RequestError as e:
            self.logger.error(
                "search_network_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SearchError(
                f"Network error during search: {e}",
                SearchErrorType.NETWORK,
            )

    def search_multiple(
        self,
        queries: List[tuple],
        batch_delay: float = 1.0,
    ) -> Dict[int, List[SearchResult]]:
        """
        Execute multiple search queries with delay between batches.

        Args:
            queries: List of (query_text, query_id) tuples
            batch_delay: Additional delay between queries (seconds)

        Returns:
            Dictionary mapping query_id to list of SearchResults
        """
        results_map = {}

        for i, (query_text, query_id) in enumerate(queries):
            try:
                results = self.search(query_text, query_id)
                results_map[query_id] = results

                # Additional delay between queries
                if i < len(queries) - 1 and batch_delay > 0:
                    time.sleep(batch_delay)

            except QuotaExceededError as e:
                self.logger.error(
                    "search_quota_exhausted_batch",
                    query_index=i,
                    total_queries=len(queries),
                )
                # Stop processing remaining queries
                break

            except SearchError as e:
                if e.error_type == SearchErrorType.FATAL:
                    self.logger.error(
                        "search_fatal_error_batch",
                        query=query_text,
                        query_id=query_id,
                        error=str(e),
                    )
                    # Skip this query but continue with others
                    results_map[query_id] = []
                else:
                    raise

        return results_map

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with request_count and quota_exhausted status
        """
        return {
            "request_count": self._request_count,
            "quota_exhausted": self._quota_exhausted,
            "raw_responses_stored": len(self._raw_responses),
        }

    def get_raw_responses(self) -> List[Dict[str, Any]]:
        """
        Get stored raw API responses for debugging.

        Returns:
            List of raw response dictionaries
        """
        return self._raw_responses

    def reset_stats(self) -> None:
        """Reset statistics and quota status."""
        self._request_count = 0
        self._quota_exhausted = False
        self._raw_responses = []


def search_query(
    query: str,
    query_id: int,
    api_key: Optional[str] = None,
    cse_id: Optional[str] = None,
) -> List[SearchResult]:
    """
    Convenience function to search a single query.

    Args:
        query: Search query string
        query_id: Database ID of SearchQuery
        api_key: Optional Google API key
        cse_id: Optional Custom Search Engine ID

    Returns:
        List of SearchResult objects
    """
    service = GoogleSearchService(api_key=api_key, cse_id=cse_id)
    return service.search(query, query_id)


def search_multiple_queries(
    queries: List[tuple],
    api_key: Optional[str] = None,
    cse_id: Optional[str] = None,
    batch_delay: float = 1.0,
) -> Dict[int, List[SearchResult]]:
    """
    Convenience function to search multiple queries.

    Args:
        queries: List of (query_text, query_id) tuples
        api_key: Optional Google API key
        cse_id: Optional Custom Search Engine ID
        batch_delay: Delay between queries (seconds)

    Returns:
        Dictionary mapping query_id to SearchResults
    """
    service = GoogleSearchService(api_key=api_key, cse_id=cse_id)
    return service.search_multiple(queries, batch_delay)
