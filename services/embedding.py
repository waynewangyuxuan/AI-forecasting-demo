"""
Embedding Service for AI Forecasting Pipeline.

Provides text embedding generation using OpenAI's embedding models with
caching, batch processing, and async API calls for efficiency.
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from openai import AsyncOpenAI, OpenAI
from openai import APIError, RateLimitError as OpenAIRateLimitError

from config.settings import settings
from db.models import Embedding as EmbeddingModel

logger = structlog.get_logger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class EmbeddingQuotaError(EmbeddingError):
    """Raised when embedding API quota is exhausted."""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when embedding API rate limit is hit."""
    pass


@dataclass
class Embedding:
    """
    Represents a text embedding vector.

    Attributes:
        text: Original text that was embedded
        vector: The embedding vector as numpy array
        model: Model name used for embedding
        dimensions: Vector dimensionality
        text_hash: Hash of the text for caching
    """
    text: str
    vector: np.ndarray
    model: str
    dimensions: int
    text_hash: str

    def to_bytes(self) -> bytes:
        """Convert vector to bytes for database storage."""
        return self.vector.astype(np.float32).tobytes()

    @classmethod
    def from_bytes(cls, text: str, vector_bytes: bytes, model: str, dimensions: int) -> 'Embedding':
        """
        Reconstruct Embedding from database bytes.

        Args:
            text: Original text
            vector_bytes: Serialized vector bytes
            model: Model name
            dimensions: Vector dimensions

        Returns:
            Embedding instance
        """
        vector = np.frombuffer(vector_bytes, dtype=np.float32)
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return cls(
            text=text,
            vector=vector,
            model=model,
            dimensions=dimensions,
            text_hash=text_hash
        )

    def to_model(self, event_id: int) -> EmbeddingModel:
        """
        Convert to database model.

        Args:
            event_id: ID of the event this embedding belongs to

        Returns:
            EmbeddingModel instance
        """
        return EmbeddingModel(
            event_id=event_id,
            vector=self.to_bytes(),
            model=self.model,
            dimensions=self.dimensions
        )


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

    # Compute dot product
    return float(np.dot(vec1_norm, vec2_norm))


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine distance (0-2, lower is more similar)
    """
    return 1.0 - cosine_similarity(vec1, vec2)


class EmbeddingService:
    """
    Service for generating and caching text embeddings.

    Features:
    - Async batch embedding generation
    - Disk caching by text hash
    - Retry logic for transient failures
    - Rate limiting
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize embedding service.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
            cache_dir: Directory for caching embeddings
            enable_cache: Whether to enable disk caching
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self.enable_cache = enable_cache

        # Setup cache directory
        if cache_dir is None:
            cache_dir = settings.cache_dir / "embeddings"
        self.cache_dir = Path(cache_dir)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0

        self.logger = structlog.get_logger(__name__)
        self.logger.info(
            "embedding_service_initialized",
            model=self.model,
            cache_enabled=self.enable_cache,
            cache_dir=str(self.cache_dir),
        )

    def _get_cache_path(self, text_hash: str) -> Path:
        """Get cache file path for a text hash."""
        return self.cache_dir / f"{text_hash}.npy"

    def _get_text_hash(self, text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _load_from_cache(self, text: str) -> Optional[Embedding]:
        """
        Load embedding from disk cache.

        Args:
            text: Text to look up

        Returns:
            Embedding if found in cache, None otherwise
        """
        if not self.enable_cache:
            return None

        text_hash = self._get_text_hash(text)
        cache_path = self._get_cache_path(text_hash)

        if cache_path.exists():
            try:
                # Load vector from .npy file
                vector = np.load(cache_path)

                # Load metadata from .json file
                meta_path = cache_path.with_suffix('.json')
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    model = meta.get('model', self.model)
                    dimensions = meta.get('dimensions', len(vector))
                else:
                    model = self.model
                    dimensions = len(vector)

                self.cache_hits += 1
                self.logger.debug(
                    "embedding_cache_hit",
                    text_hash=text_hash,
                    cache_hits=self.cache_hits,
                )

                return Embedding(
                    text=text,
                    vector=vector,
                    model=model,
                    dimensions=dimensions,
                    text_hash=text_hash,
                )
            except Exception as e:
                self.logger.warning(
                    "embedding_cache_load_error",
                    text_hash=text_hash,
                    error=str(e),
                )

        return None

    def _save_to_cache(self, embedding: Embedding) -> None:
        """
        Save embedding to disk cache.

        Args:
            embedding: Embedding to cache
        """
        if not self.enable_cache:
            return

        try:
            cache_path = self._get_cache_path(embedding.text_hash)

            # Save vector as .npy
            np.save(cache_path, embedding.vector)

            # Save metadata as .json
            meta_path = cache_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'model': embedding.model,
                    'dimensions': embedding.dimensions,
                    'text_preview': embedding.text[:100],
                }, f)

            self.logger.debug(
                "embedding_cached",
                text_hash=embedding.text_hash,
                cache_path=str(cache_path),
            )
        except Exception as e:
            self.logger.warning(
                "embedding_cache_save_error",
                text_hash=embedding.text_hash,
                error=str(e),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((EmbeddingRateLimitError, EmbeddingError)),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _generate_embedding_async(self, text: str) -> np.ndarray:
        """
        Generate embedding via OpenAI API (async).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingQuotaError: When quota is exhausted
            EmbeddingRateLimitError: When rate limited
            EmbeddingError: For other API errors
        """
        try:
            # Clean and truncate text if needed (max ~8k tokens for ada-002)
            clean_text = text.strip().replace('\n', ' ')[:32000]

            if not clean_text:
                raise EmbeddingError("Empty text provided for embedding")

            response = await self.async_client.embeddings.create(
                model=self.model,
                input=clean_text,
            )

            self.api_calls += 1

            # Extract embedding vector
            vector = np.array(response.data[0].embedding, dtype=np.float32)

            self.logger.debug(
                "embedding_generated",
                model=self.model,
                dimensions=len(vector),
                api_calls=self.api_calls,
            )

            return vector

        except OpenAIRateLimitError as e:
            self.logger.warning("embedding_rate_limited", error=str(e))
            raise EmbeddingRateLimitError(f"OpenAI rate limited: {e}")
        except APIError as e:
            if "quota" in str(e).lower():
                self.logger.error("embedding_quota_exhausted", error=str(e))
                raise EmbeddingQuotaError(f"OpenAI quota exhausted: {e}")
            self.logger.error("embedding_api_error", error=str(e))
            raise EmbeddingError(f"OpenAI API error: {e}")
        except Exception as e:
            self.logger.error("embedding_unexpected_error", error=str(e), error_type=type(e).__name__)
            raise EmbeddingError(f"Unexpected embedding error: {e}")

    async def embed_async(self, text: str) -> Embedding:
        """
        Generate embedding for text with caching (async).

        Args:
            text: Text to embed

        Returns:
            Embedding object with vector and metadata
        """
        # Check cache first
        cached = self._load_from_cache(text)
        if cached:
            return cached

        # Generate new embedding
        self.cache_misses += 1
        text_hash = self._get_text_hash(text)

        self.logger.debug(
            "embedding_cache_miss",
            text_hash=text_hash,
            text_preview=text[:100],
            cache_misses=self.cache_misses,
        )

        vector = await self._generate_embedding_async(text)

        embedding = Embedding(
            text=text,
            vector=vector,
            model=self.model,
            dimensions=len(vector),
            text_hash=text_hash,
        )

        # Cache for future use
        self._save_to_cache(embedding)

        return embedding

    def embed(self, text: str) -> Embedding:
        """
        Generate embedding for text with caching (synchronous).

        Args:
            text: Text to embed

        Returns:
            Embedding object with vector and metadata
        """
        # Check cache first
        cached = self._load_from_cache(text)
        if cached:
            return cached

        # Generate new embedding using sync client
        self.cache_misses += 1
        text_hash = self._get_text_hash(text)

        try:
            clean_text = text.strip().replace('\n', ' ')[:32000]

            if not clean_text:
                raise EmbeddingError("Empty text provided for embedding")

            response = self.client.embeddings.create(
                model=self.model,
                input=clean_text,
            )

            self.api_calls += 1
            vector = np.array(response.data[0].embedding, dtype=np.float32)

            embedding = Embedding(
                text=text,
                vector=vector,
                model=self.model,
                dimensions=len(vector),
                text_hash=text_hash,
            )

            # Cache for future use
            self._save_to_cache(embedding)

            return embedding

        except Exception as e:
            self.logger.error("embedding_sync_error", error=str(e))
            raise EmbeddingError(f"Embedding generation failed: {e}")

    async def embed_batch_async(
        self,
        texts: List[str],
        max_concurrent: int = 10,
    ) -> List[Embedding]:
        """
        Generate embeddings for multiple texts with concurrency control.

        Args:
            texts: List of texts to embed
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of Embedding objects in same order as input
        """
        if not texts:
            return []

        self.logger.info(
            "embed_batch_start",
            total_texts=len(texts),
            max_concurrent=max_concurrent,
        )

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_with_semaphore(text: str) -> Embedding:
            async with semaphore:
                return await self.embed_async(text)

        # Process all texts concurrently (up to max_concurrent)
        embeddings = await asyncio.gather(
            *[embed_with_semaphore(text) for text in texts],
            return_exceptions=False,
        )

        self.logger.info(
            "embed_batch_complete",
            total_embeddings=len(embeddings),
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            api_calls=self.api_calls,
        )

        return embeddings

    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """
        Generate embeddings for multiple texts (synchronous wrapper).

        Args:
            texts: List of texts to embed

        Returns:
            List of Embedding objects in same order as input
        """
        return asyncio.run(self.embed_batch_async(texts))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with cache hits, misses, and API calls
        """
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
            "cache_hit_rate": cache_hit_rate,
            "model": self.model,
        }

    def clear_cache(self) -> None:
        """Delete all cached embeddings."""
        if not self.enable_cache:
            return

        deleted_count = 0
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
            # Also delete metadata
            meta_file = cache_file.with_suffix('.json')
            if meta_file.exists():
                meta_file.unlink()
            deleted_count += 1

        self.logger.info("embedding_cache_cleared", deleted_count=deleted_count)


# Convenience functions for common use cases

async def generate_embeddings_async(
    texts: List[str],
    model: Optional[str] = None,
    max_concurrent: int = 10,
) -> List[Embedding]:
    """
    Convenience function to generate embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        model: Model to use (defaults to settings)
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of Embedding objects
    """
    service = EmbeddingService(model=model)
    return await service.embed_batch_async(texts, max_concurrent=max_concurrent)


def generate_embeddings(
    texts: List[str],
    model: Optional[str] = None,
) -> List[Embedding]:
    """
    Convenience function to generate embeddings for a list of texts (sync).

    Args:
        texts: List of texts to embed
        model: Model to use (defaults to settings)

    Returns:
        List of Embedding objects
    """
    service = EmbeddingService(model=model)
    return service.embed_batch(texts)


def compute_similarity_matrix(embeddings: List[Embedding]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for embeddings.

    Args:
        embeddings: List of Embedding objects

    Returns:
        NxN similarity matrix where element [i,j] is similarity between i and j
    """
    n = len(embeddings)
    vectors = np.array([emb.vector for emb in embeddings])

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-10)

    # Compute similarity matrix via matrix multiplication
    similarity = np.dot(normalized, normalized.T)

    return similarity


def compute_distance_matrix(embeddings: List[Embedding]) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix for embeddings.

    Args:
        embeddings: List of Embedding objects

    Returns:
        NxN distance matrix where element [i,j] is distance between i and j
    """
    similarity_matrix = compute_similarity_matrix(embeddings)
    return 1.0 - similarity_matrix
