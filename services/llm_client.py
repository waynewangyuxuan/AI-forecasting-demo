"""
LLM Client Wrapper for AI Forecasting Pipeline.

Provides an abstract base class and concrete implementations for LLM providers
with retry logic, rate limiting, and error handling.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

from config.settings import settings

logger = structlog.get_logger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class QuotaExhaustedError(LLMError):
    """Raised when API quota is exhausted."""
    pass


class ParsingError(LLMError):
    """Raised when LLM response cannot be parsed."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is hit."""
    pass


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Provides a common interface for different LLM providers with built-in
    retry logic, rate limiting, and error handling.
    """

    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider
            model_name: Name/identifier of the model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.request_count = 0
        self.total_tokens = 0
        self.logger = structlog.get_logger(__name__)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Expected response format (e.g., "json", "text")

        Returns:
            Generated text from the LLM

        Raises:
            QuotaExhaustedError: When API quota is exhausted
            RateLimitError: When rate limit is hit
            ParsingError: When response cannot be parsed
            LLMError: For other LLM-related errors
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for this client.

        Returns:
            Dictionary with request_count and total_tokens
        """
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.request_count = 0
        self.total_tokens = 0


class GeminiClient(LLMClient):
    """
    Concrete implementation for Google Gemini API.

    Provides retry logic with exponential backoff, rate limiting,
    and comprehensive error handling for the Gemini API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_responses: bool = False,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key (defaults to settings)
            model_name: Model name (defaults to settings)
            cache_responses: Whether to cache responses for development
        """
        api_key = api_key or settings.google_gemini_api_key
        model_name = model_name or settings.gemini_model
        super().__init__(api_key, model_name)

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Safety settings - allow most content for factual extraction
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings=self.safety_settings,
        )

        # Response caching for development
        self.cache_responses = cache_responses
        self._response_cache: Dict[str, str] = {}

        # Rate limiting state
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests

        self.logger.info(
            "gemini_client_initialized",
            model=self.model_name,
            cache_enabled=self.cache_responses,
        )

    def _rate_limit(self) -> None:
        """Enforce minimum time between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_cache_key(self, prompt: str, temperature: float, max_tokens: Optional[int]) -> str:
        """Generate cache key for a request."""
        return f"{hash(prompt)}_{temperature}_{max_tokens}"

    def _handle_api_error(self, error: Exception) -> None:
        """
        Categorize and re-raise API errors.

        Args:
            error: The caught exception

        Raises:
            QuotaExhaustedError: For quota/resource exhaustion
            RateLimitError: For rate limiting
            LLMError: For other errors
        """
        error_str = str(error).lower()

        # Check for quota exhaustion
        if "quota" in error_str or "resource_exhausted" in error_str:
            self.logger.error("gemini_quota_exhausted", error=str(error))
            raise QuotaExhaustedError(f"Gemini API quota exhausted: {error}")

        # Check for rate limiting
        if "rate" in error_str or "too many requests" in error_str:
            self.logger.warning("gemini_rate_limited", error=str(error))
            raise RateLimitError(f"Gemini API rate limited: {error}")

        # Check for specific Google API errors
        if isinstance(error, google_exceptions.ResourceExhausted):
            self.logger.error("gemini_quota_exhausted", error=str(error))
            raise QuotaExhaustedError(f"Gemini API quota exhausted: {error}")

        if isinstance(error, google_exceptions.TooManyRequests):
            self.logger.warning("gemini_rate_limited", error=str(error))
            raise RateLimitError(f"Gemini API rate limited: {error}")

        # Generic error
        self.logger.error("gemini_api_error", error=str(error), error_type=type(error).__name__)
        raise LLMError(f"Gemini API error: {error}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, LLMError)),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """
        Generate text from Gemini API with retry logic.

        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Expected response format ("json" or "text")

        Returns:
            Generated text from Gemini

        Raises:
            QuotaExhaustedError: When API quota is exhausted
            RateLimitError: When rate limit is hit (retried)
            ParsingError: When response cannot be parsed
            LLMError: For other API errors (retried)
        """
        # Check cache first
        if self.cache_responses:
            cache_key = self._get_cache_key(prompt, temperature, max_tokens)
            if cache_key in self._response_cache:
                self.logger.debug("gemini_cache_hit", cache_key=cache_key)
                return self._response_cache[cache_key]

        # Rate limiting
        self._rate_limit()

        # Prepare generation config
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Add JSON mode if requested
        if response_format == "json":
            generation_config.response_mime_type = "application/json"

        try:
            self.logger.debug(
                "gemini_generate_start",
                prompt_length=len(prompt),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
            )

            # Extract text from response
            if not response.text:
                if response.prompt_feedback:
                    self.logger.error(
                        "gemini_blocked_prompt",
                        feedback=str(response.prompt_feedback),
                    )
                    raise ParsingError(f"Prompt was blocked: {response.prompt_feedback}")

                raise ParsingError("Empty response from Gemini")

            result = response.text.strip()

            # Update stats
            self.request_count += 1
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self.total_tokens += (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )

            self.logger.info(
                "gemini_generate_success",
                request_count=self.request_count,
                response_length=len(result),
                total_tokens=self.total_tokens,
            )

            # Cache if enabled
            if self.cache_responses:
                cache_key = self._get_cache_key(prompt, temperature, max_tokens)
                self._response_cache[cache_key] = result

            return result

        except Exception as e:
            self._handle_api_error(e)

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON response from Gemini.

        Args:
            prompt: Input prompt (should request JSON output)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON object

        Raises:
            ParsingError: When response is not valid JSON
            Other exceptions from generate()
        """
        response = self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to fix common Gemini JSON issues
            self.logger.warning(
                "gemini_json_parse_error_attempting_fix",
                error=str(e),
                response=response[:500],
            )

            # Fix: Gemini sometimes double-escapes special chars like \$
            # Replace \\$ with $ and other common issues
            fixed_response = response.replace('\\$', '$')
            fixed_response = fixed_response.replace('\\"', '"')  # Fix double-escaped quotes

            try:
                result = json.loads(fixed_response)
                self.logger.info("gemini_json_fixed", original_error=str(e))
                return result
            except json.JSONDecodeError as e2:
                self.logger.error(
                    "gemini_json_parse_error_unfixable",
                    error=str(e2),
                    original_response=response[:500],
                    fixed_response=fixed_response[:500],
                )
                raise ParsingError(f"Failed to parse JSON response: {e2}")

    def generate_with_retry_on_parse_error(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        max_retries: int = 2,
    ) -> str:
        """
        Generate with automatic retry on parsing errors.

        Useful for extracting structured data where the LLM might occasionally
        produce malformed output.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            response_format: Expected format
            max_retries: Number of retries on parse errors

        Returns:
            Generated text
        """
        for attempt in range(max_retries + 1):
            try:
                return self.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            except ParsingError as e:
                if attempt < max_retries:
                    self.logger.warning(
                        "gemini_parse_error_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    time.sleep(1)  # Brief pause before retry
                else:
                    raise


def create_llm_client(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM clients.

    Args:
        provider: Provider name ("gemini" currently supported)
        api_key: API key for the provider
        model_name: Model name
        **kwargs: Additional provider-specific arguments

    Returns:
        Initialized LLM client

    Raises:
        ValueError: If provider is not supported
    """
    if provider.lower() == "gemini":
        return GeminiClient(api_key=api_key, model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
