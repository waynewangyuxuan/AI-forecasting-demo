"""
Web Scraping Module for AI Forecasting Pipeline.

Provides async web scraping with content extraction, robots.txt compliance,
rate limiting, retry logic, and comprehensive error handling.
"""

import asyncio
import hashlib
import logging
import re
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urlparse
from datetime import datetime
from enum import Enum

import httpx
import structlog
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from robotexclusionrulesparser import RobotExclusionRulesParser
    ROBOTS_PARSER_AVAILABLE = True
except ImportError:
    ROBOTS_PARSER_AVAILABLE = False

from config.settings import settings
from db.models import Document

logger = structlog.get_logger(__name__)


class ScraperErrorType(str, Enum):
    """Error categories for web scraping."""
    FORBIDDEN = "FORBIDDEN"  # 403
    NOT_FOUND = "NOT_FOUND"  # 404
    TIMEOUT = "TIMEOUT"  # Request timeout
    RATE_LIMITED = "RATE_LIMITED"  # 429
    SERVER_ERROR = "SERVER_ERROR"  # 5xx
    PARSE_ERROR = "PARSE_ERROR"  # HTML parsing failed
    ROBOTS_BLOCKED = "ROBOTS_BLOCKED"  # Blocked by robots.txt
    CONNECTION_ERROR = "CONNECTION_ERROR"  # Network issues
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"  # Insufficient content
    UNKNOWN = "UNKNOWN"  # Other errors


class ScraperError(Exception):
    """Base exception for scraper errors."""
    def __init__(self, message: str, error_type: ScraperErrorType, url: str):
        super().__init__(message)
        self.error_type = error_type
        self.url = url


class DocumentFetchResult:
    """Result of fetching a document."""
    def __init__(
        self,
        url: str,
        success: bool,
        document: Optional[Document] = None,
        error_type: Optional[ScraperErrorType] = None,
        error_message: Optional[str] = None,
    ):
        self.url = url
        self.success = success
        self.document = document
        self.error_type = error_type
        self.error_message = error_message


class RobotsChecker:
    """Checks robots.txt compliance for URLs."""

    def __init__(self, user_agent: str):
        """
        Initialize robots checker.

        Args:
            user_agent: User agent string for checking rules
        """
        self.user_agent = user_agent
        self._cache: Dict[str, RobotExclusionRulesParser] = {}
        self.logger = structlog.get_logger(__name__)

    def _get_robots_url(self, url: str) -> str:
        """Get robots.txt URL for a given URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    async def is_allowed(self, url: str, client: httpx.AsyncClient) -> bool:
        """
        Check if URL is allowed by robots.txt.

        Args:
            url: URL to check
            client: HTTP client for fetching robots.txt

        Returns:
            True if allowed, False if blocked
        """
        if not ROBOTS_PARSER_AVAILABLE:
            # If parser not available, allow by default
            return True

        robots_url = self._get_robots_url(url)

        # Check cache
        if robots_url in self._cache:
            parser = self._cache[robots_url]
            allowed = parser.is_allowed(self.user_agent, url)
            self.logger.debug(
                "robots_cache_check",
                url=url,
                robots_url=robots_url,
                allowed=allowed,
            )
            return allowed

        # Fetch robots.txt
        try:
            response = await client.get(robots_url, timeout=10.0)
            if response.status_code == 200:
                parser = RobotExclusionRulesParser()
                parser.parse(response.text)
                self._cache[robots_url] = parser

                allowed = parser.is_allowed(self.user_agent, url)
                self.logger.info(
                    "robots_txt_fetched",
                    robots_url=robots_url,
                    url=url,
                    allowed=allowed,
                )
                return allowed
            else:
                # No robots.txt or error, allow by default
                self.logger.debug(
                    "robots_txt_not_found",
                    robots_url=robots_url,
                    status_code=response.status_code,
                )
                return True
        except Exception as e:
            self.logger.warning(
                "robots_txt_fetch_error",
                robots_url=robots_url,
                error=str(e),
            )
            # On error, allow by default
            return True


class WebScraper:
    """
    Async web scraper with content extraction and error handling.

    Features:
    - Semaphore-limited concurrency
    - Robots.txt compliance
    - User-agent rotation
    - Retry logic with exponential backoff
    - Multi-layered content extraction (readability -> trafilatura -> beautifulsoup)
    - Content cleaning and validation
    - Comprehensive error categorization
    """

    # Minimum content length threshold (characters)
    MIN_CONTENT_LENGTH = 200

    # User agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (compatible; ForecastBot/1.0; +https://github.com/yourrepo)",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]

    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: int = 30,
        check_robots: bool = True,
    ):
        """
        Initialize web scraper.

        Args:
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            check_robots: Whether to check robots.txt
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.check_robots = check_robots
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = structlog.get_logger(__name__)
        self.robots_checker = RobotsChecker(self.USER_AGENTS[0])
        self._user_agent_index = 0

        # Statistics
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "robots_blocked": 0,
        }

    def _get_next_user_agent(self) -> str:
        """Get next user agent from rotation."""
        ua = self.USER_AGENTS[self._user_agent_index]
        self._user_agent_index = (self._user_agent_index + 1) % len(self.USER_AGENTS)
        return ua

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove common noise patterns
        text = re.sub(r'(Accept all cookies|Privacy Policy|Terms of Service)', '', text, flags=re.IGNORECASE)

        return text

    def _extract_with_readability(self, html: str, url: str) -> Optional[Dict[str, str]]:
        """
        Extract content using readability-lxml.

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            Dict with title and content, or None if extraction fails
        """
        try:
            doc = ReadabilityDocument(html, url=url)
            title = doc.title()
            content = doc.summary(html_partial=True)

            # Parse HTML summary to get text
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

            if text and len(text) >= self.MIN_CONTENT_LENGTH:
                self.logger.debug(
                    "readability_extraction_success",
                    url=url,
                    title_length=len(title) if title else 0,
                    content_length=len(text),
                )
                return {"title": title, "content": text}

            return None
        except Exception as e:
            self.logger.debug(
                "readability_extraction_failed",
                url=url,
                error=str(e),
            )
            return None

    def _extract_with_trafilatura(self, html: str, url: str) -> Optional[Dict[str, str]]:
        """
        Extract content using trafilatura (fallback).

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            Dict with title and content, or None if extraction fails
        """
        if not TRAFILATURA_AVAILABLE:
            return None

        try:
            # Extract with trafilatura
            content = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                url=url,
            )

            if content and len(content) >= self.MIN_CONTENT_LENGTH:
                # Try to extract title from HTML
                soup = BeautifulSoup(html, 'html.parser')
                title_tag = soup.find('title')
                title = title_tag.get_text() if title_tag else None

                self.logger.debug(
                    "trafilatura_extraction_success",
                    url=url,
                    content_length=len(content),
                )
                return {"title": title, "content": content}

            return None
        except Exception as e:
            self.logger.debug(
                "trafilatura_extraction_failed",
                url=url,
                error=str(e),
            )
            return None

    def _extract_with_beautifulsoup(self, html: str, url: str) -> Optional[Dict[str, str]]:
        """
        Extract content using beautifulsoup heuristics (final fallback).

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            Dict with title and content, or None if extraction fails
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else None

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # Try to find main content areas
            main_content = None
            for selector in ['article', 'main', '[role="main"]', '.content', '#content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            # Fallback to body if no main content found
            if not main_content:
                main_content = soup.find('body')

            if main_content:
                text = main_content.get_text(separator=' ', strip=True)

                if text and len(text) >= self.MIN_CONTENT_LENGTH:
                    self.logger.debug(
                        "beautifulsoup_extraction_success",
                        url=url,
                        content_length=len(text),
                    )
                    return {"title": title, "content": text}

            return None
        except Exception as e:
            self.logger.debug(
                "beautifulsoup_extraction_failed",
                url=url,
                error=str(e),
            )
            return None

    def _extract_content(self, html: str, url: str) -> Dict[str, str]:
        """
        Extract content using multiple extraction methods with fallbacks.

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            Dict with title and cleaned_content

        Raises:
            ScraperError: If all extraction methods fail
        """
        # Try readability first (primary)
        result = self._extract_with_readability(html, url)
        if result:
            return {
                "title": result["title"],
                "cleaned_content": self._clean_text(result["content"]),
            }

        # Try trafilatura (fallback)
        result = self._extract_with_trafilatura(html, url)
        if result:
            return {
                "title": result["title"],
                "cleaned_content": self._clean_text(result["content"]),
            }

        # Try beautifulsoup (final fallback)
        result = self._extract_with_beautifulsoup(html, url)
        if result:
            return {
                "title": result["title"],
                "cleaned_content": self._clean_text(result["content"]),
            }

        # All extraction methods failed
        raise ScraperError(
            f"Failed to extract content from {url}",
            ScraperErrorType.CONTENT_TOO_SHORT,
            url,
        )

    def _categorize_http_error(self, status_code: int, url: str) -> ScraperErrorType:
        """Categorize HTTP error by status code."""
        if status_code == 403:
            return ScraperErrorType.FORBIDDEN
        elif status_code == 404:
            return ScraperErrorType.NOT_FOUND
        elif status_code == 429:
            return ScraperErrorType.RATE_LIMITED
        elif 500 <= status_code < 600:
            return ScraperErrorType.SERVER_ERROR
        else:
            return ScraperErrorType.UNKNOWN

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _fetch_url(self, url: str, client: httpx.AsyncClient) -> str:
        """
        Fetch URL content with retry logic.

        Args:
            url: URL to fetch
            client: HTTP client

        Returns:
            HTML content

        Raises:
            ScraperError: On fetch failure
        """
        # Check robots.txt
        if self.check_robots:
            allowed = await self.robots_checker.is_allowed(url, client)
            if not allowed:
                self.stats["robots_blocked"] += 1
                raise ScraperError(
                    f"URL blocked by robots.txt: {url}",
                    ScraperErrorType.ROBOTS_BLOCKED,
                    url,
                )

        # Check known blocking domains (fail fast without network call)
        blocked_domains = [
            'sciencedirect.com',
            'researchgate.net',
            'ark-invest.com',
            'jstor.org',
            'springer.com',
            'ieee.org',
        ]
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        for blocked in blocked_domains:
            if blocked in domain:
                self.stats["robots_blocked"] += 1
                self.logger.warning(
                    "known_blocker_skipped",
                    url=url,
                    domain=blocked,
                )
                raise ScraperError(
                    f"URL from known blocking domain: {domain}",
                    ScraperErrorType.FORBIDDEN,
                    url,
                )

        # Fetch URL
        user_agent = self._get_next_user_agent()
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            response = await client.get(url, headers=headers, timeout=self.timeout, follow_redirects=True)
            response.raise_for_status()
            return response.text
        except httpx.TimeoutException as e:
            self.logger.warning("fetch_timeout", url=url)
            raise ScraperError(
                f"Timeout fetching {url}: {e}",
                ScraperErrorType.TIMEOUT,
                url,
            )
        except httpx.HTTPStatusError as e:
            error_type = self._categorize_http_error(e.response.status_code, url)
            self.logger.warning(
                "fetch_http_error",
                url=url,
                status_code=e.response.status_code,
                error_type=error_type.value,
            )
            raise ScraperError(
                f"HTTP {e.response.status_code} fetching {url}",
                error_type,
                url,
            )
        except httpx.ConnectError as e:
            self.logger.warning("fetch_connection_error", url=url, error=str(e))
            raise ScraperError(
                f"Connection error fetching {url}: {e}",
                ScraperErrorType.CONNECTION_ERROR,
                url,
            )
        except Exception as e:
            self.logger.error("fetch_unknown_error", url=url, error=str(e))
            raise ScraperError(
                f"Unknown error fetching {url}: {e}",
                ScraperErrorType.UNKNOWN,
                url,
            )

    async def fetch_document(
        self,
        url: str,
        run_id: int,
        client: httpx.AsyncClient,
    ) -> DocumentFetchResult:
        """
        Fetch and process a single document.

        Args:
            url: URL to fetch
            run_id: Run ID for database reference
            client: HTTP client

        Returns:
            DocumentFetchResult with outcome
        """
        async with self.semaphore:
            self.stats["total"] += 1

            try:
                # Fetch HTML
                html = await self._fetch_url(url, client)

                # Extract content
                extracted = self._extract_content(html, url)

                # Create document
                content_hash = self._compute_content_hash(extracted["cleaned_content"])
                document = Document(
                    run_id=run_id,
                    url=url,
                    content_hash=content_hash,
                    raw_content=html[:10000] if settings.debug else None,  # Store snippet in debug mode
                    cleaned_content=extracted["cleaned_content"],
                    fetched_at=datetime.utcnow(),
                    status="SUCCESS",
                )

                self.stats["success"] += 1
                self.logger.info(
                    "document_fetched",
                    url=url,
                    content_length=len(extracted["cleaned_content"]),
                    title=extracted.get("title", "")[:100],
                )

                return DocumentFetchResult(
                    url=url,
                    success=True,
                    document=document,
                )

            except ScraperError as e:
                self.stats["failed"] += 1
                self.logger.warning(
                    "document_fetch_failed",
                    url=url,
                    error_type=e.error_type.value,
                    error=str(e),
                )
                return DocumentFetchResult(
                    url=url,
                    success=False,
                    error_type=e.error_type,
                    error_message=str(e),
                )
            except Exception as e:
                self.stats["failed"] += 1
                self.logger.error(
                    "document_fetch_unexpected_error",
                    url=url,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return DocumentFetchResult(
                    url=url,
                    success=False,
                    error_type=ScraperErrorType.UNKNOWN,
                    error_message=str(e),
                )

    async def fetch_documents_batch(
        self,
        urls: List[str],
        run_id: int,
    ) -> List[DocumentFetchResult]:
        """
        Fetch multiple documents concurrently.

        Args:
            urls: List of URLs to fetch
            run_id: Run ID for database reference

        Returns:
            List of DocumentFetchResult objects
        """
        # Reset stats for this batch
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "robots_blocked": 0,
        }

        self.logger.info(
            "batch_scrape_start",
            url_count=len(urls),
            max_concurrent=self.max_concurrent,
        )

        # Create HTTP client
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=self.max_concurrent * 2),
        ) as client:
            # Fetch all documents concurrently
            tasks = [
                self.fetch_document(url, run_id, client)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            "batch_scrape_complete",
            total=len(urls),
            success=success_count,
            failed=len(urls) - success_count,
            robots_blocked=self.stats["robots_blocked"],
            success_rate=success_count / len(urls) if urls else 0,
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        return dict(self.stats)


async def scrape_urls(
    urls: List[str],
    run_id: int,
    max_concurrent: int = 5,
    timeout: int = 30,
    check_robots: bool = True,
) -> List[DocumentFetchResult]:
    """
    Convenience function to scrape multiple URLs.

    Args:
        urls: List of URLs to scrape
        run_id: Run ID for database reference
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout in seconds
        check_robots: Whether to check robots.txt

    Returns:
        List of DocumentFetchResult objects
    """
    scraper = WebScraper(
        max_concurrent=max_concurrent,
        timeout=timeout,
        check_robots=check_robots,
    )
    return await scraper.fetch_documents_batch(urls, run_id)


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text using regex.

    Args:
        text: Text containing URLs

    Returns:
        List of extracted URLs
    """
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)
