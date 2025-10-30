"""
Document Processing Module for AI Forecasting Pipeline.

Provides semantic chunking, language detection, token counting, and text
preprocessing for downstream LLM operations.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import structlog

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

from db.models import Document

logger = structlog.get_logger(__name__)


@dataclass
class DocumentChunk:
    """
    A chunk of a document for processing.

    Attributes:
        chunk_id: Unique identifier for the chunk
        source_doc_id: ID of the source document
        content: The text content of the chunk
        start_char: Starting character index in original document
        end_char: Ending character index in original document
        token_count: Approximate number of tokens
        metadata: Additional metadata about the chunk
    """
    chunk_id: str
    source_doc_id: int
    content: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """
    Document processor with semantic chunking and language detection.

    Features:
    - Semantic chunking with configurable target size and overlap
    - Language detection to filter non-English content
    - Token counting approximation
    - Text normalization and cleaning
    - Preservation of context through chunk overlap
    """

    # Token estimation multiplier (words * 1.3 approximates tokens for English)
    TOKEN_MULTIPLIER = 1.3

    # Default chunk size in tokens
    DEFAULT_CHUNK_SIZE = 500

    # Default overlap in tokens
    DEFAULT_OVERLAP = 50

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        filter_language: bool = True,
        target_language: str = "en",
        min_chunk_tokens: int = 50,
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            filter_language: Whether to filter by language
            target_language: Target language code (default: "en" for English)
            min_chunk_tokens: Minimum tokens required for a valid chunk
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.filter_language = filter_language
        self.target_language = target_language
        self.min_chunk_tokens = min_chunk_tokens
        self.logger = structlog.get_logger(__name__)

        if filter_language and not LANGDETECT_AVAILABLE:
            self.logger.warning(
                "langdetect_not_available",
                message="langdetect not installed, language filtering disabled",
            )
            self.filter_language = False

        self.logger.info(
            "doc_processor_initialized",
            chunk_size=chunk_size,
            overlap=overlap,
            filter_language=filter_language,
            target_language=target_language,
        )

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count from text.

        Uses a simple heuristic: word_count * TOKEN_MULTIPLIER
        This approximates tokenization without requiring an actual tokenizer.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Count words (split by whitespace)
        words = text.split()
        word_count = len(words)

        # Apply multiplier
        estimated_tokens = int(word_count * self.TOKEN_MULTIPLIER)

        return estimated_tokens

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Language code (e.g., "en", "fr") or None if detection fails
        """
        if not LANGDETECT_AVAILABLE:
            return None

        try:
            # Use a sample if text is very long (for performance)
            sample = text[:5000] if len(text) > 5000 else text
            lang = detect(sample)
            self.logger.debug("language_detected", language=lang, text_length=len(text))
            return lang
        except LangDetectException as e:
            self.logger.debug("language_detection_failed", error=str(e))
            return None

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for processing.

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove excessive whitespace within lines
        text = re.sub(r'[ \t]+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _find_sentence_boundaries(self, text: str, start: int, target_end: int) -> int:
        """
        Find the nearest sentence boundary to target_end.

        Args:
            text: Full text
            start: Start position
            target_end: Target end position

        Returns:
            Actual end position at sentence boundary
        """
        # Sentence ending patterns
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

        # Look for sentence boundary within a window around target_end
        window = 100  # characters to look around
        search_start = max(start, target_end - window)
        search_end = min(len(text), target_end + window)

        # Find all sentence endings in the window
        best_pos = target_end
        best_distance = float('inf')

        for ending in sentence_endings:
            pos = text.find(ending, search_start, search_end)
            if pos != -1:
                # Position after the ending
                end_pos = pos + len(ending)
                distance = abs(end_pos - target_end)
                if distance < best_distance:
                    best_distance = distance
                    best_pos = end_pos

        # If no sentence boundary found, use target_end
        if best_distance == float('inf'):
            # Try to at least break at a space
            for i in range(target_end, search_end):
                if i < len(text) and text[i] == ' ':
                    return i
            return target_end

        return best_pos

    def _chunk_by_tokens(self, text: str, target_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
        """
        Chunk text by token count with overlap.

        Args:
            text: Text to chunk
            target_tokens: Target tokens per chunk
            overlap_tokens: Overlap in tokens

        Returns:
            List of chunk dicts with content, start_char, end_char, token_count
        """
        chunks = []
        text_length = len(text)

        # Estimate characters per token (inverse of TOKEN_MULTIPLIER)
        chars_per_token = 1.0 / self.TOKEN_MULTIPLIER * 5  # ~4 chars per word, 1.3 words per token

        target_chars = int(target_tokens * chars_per_token)
        overlap_chars = int(overlap_tokens * chars_per_token)

        start = 0
        chunk_num = 0

        while start < text_length:
            chunk_num += 1

            # Calculate end position
            end = min(start + target_chars, text_length)

            # Find sentence boundary
            if end < text_length:
                end = self._find_sentence_boundaries(text, start, end)

            # Extract chunk
            chunk_text = text[start:end].strip()

            # Skip if too small
            token_count = self.estimate_token_count(chunk_text)
            if token_count >= self.min_chunk_tokens:
                chunks.append({
                    "content": chunk_text,
                    "start_char": start,
                    "end_char": end,
                    "token_count": token_count,
                    "chunk_num": chunk_num,
                })

            # Move to next chunk with overlap
            if end >= text_length:
                break

            # Calculate next start position (with overlap)
            start = max(start + target_chars - overlap_chars, end - overlap_chars)

            # Ensure we make progress
            if start <= chunks[-1]["start_char"] if chunks else 0:
                start = end

        return chunks

    def chunk_document(
        self,
        document: Document,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a document into semantic chunks.

        Args:
            document: Document to chunk
            chunk_size: Override default chunk size (tokens)
            overlap: Override default overlap (tokens)

        Returns:
            List of DocumentChunk objects
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap

        # Validate document
        if not document.cleaned_content:
            self.logger.warning(
                "empty_document",
                doc_id=document.id,
                url=document.url,
            )
            return []

        # Check language if filtering enabled
        if self.filter_language:
            detected_lang = self.detect_language(document.cleaned_content)
            if detected_lang and detected_lang != self.target_language:
                self.logger.info(
                    "document_filtered_by_language",
                    doc_id=document.id,
                    url=document.url,
                    detected_language=detected_lang,
                    target_language=self.target_language,
                )
                return []

        # Normalize text
        normalized_text = self.normalize_text(document.cleaned_content)

        # Check if document is too short
        total_tokens = self.estimate_token_count(normalized_text)
        if total_tokens < self.min_chunk_tokens:
            self.logger.info(
                "document_too_short",
                doc_id=document.id,
                url=document.url,
                token_count=total_tokens,
                min_tokens=self.min_chunk_tokens,
            )
            return []

        # Chunk the text
        raw_chunks = self._chunk_by_tokens(normalized_text, chunk_size, overlap)

        # Create DocumentChunk objects
        chunks = []
        for i, raw_chunk in enumerate(raw_chunks):
            chunk = DocumentChunk(
                chunk_id=f"doc{document.id}_chunk{i}",
                source_doc_id=document.id,
                content=raw_chunk["content"],
                start_char=raw_chunk["start_char"],
                end_char=raw_chunk["end_char"],
                token_count=raw_chunk["token_count"],
                metadata={
                    "chunk_num": raw_chunk["chunk_num"],
                    "source_url": document.url,
                    "total_chunks": len(raw_chunks),
                    "document_token_count": total_tokens,
                },
            )
            chunks.append(chunk)

        self.logger.info(
            "document_chunked",
            doc_id=document.id,
            url=document.url,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            avg_chunk_tokens=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
        )

        return chunks

    def chunk_documents_batch(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Dict[int, List[DocumentChunk]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk
            chunk_size: Override default chunk size (tokens)
            overlap: Override default overlap (tokens)

        Returns:
            Dictionary mapping document IDs to lists of chunks
        """
        self.logger.info(
            "batch_chunking_start",
            document_count=len(documents),
        )

        results = {}
        total_chunks = 0

        for document in documents:
            chunks = self.chunk_document(document, chunk_size, overlap)
            if chunks:
                results[document.id] = chunks
                total_chunks += len(chunks)

        self.logger.info(
            "batch_chunking_complete",
            document_count=len(documents),
            processed_count=len(results),
            total_chunks=total_chunks,
            avg_chunks_per_doc=total_chunks / len(results) if results else 0,
        )

        return results


def chunk_document(
    document: Document,
    chunk_size: int = DocumentProcessor.DEFAULT_CHUNK_SIZE,
    overlap: int = DocumentProcessor.DEFAULT_OVERLAP,
    filter_language: bool = True,
    target_language: str = "en",
) -> List[DocumentChunk]:
    """
    Convenience function to chunk a single document.

    Args:
        document: Document to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        filter_language: Whether to filter by language
        target_language: Target language code

    Returns:
        List of DocumentChunk objects
    """
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        overlap=overlap,
        filter_language=filter_language,
        target_language=target_language,
    )
    return processor.chunk_document(document)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DocumentProcessor.DEFAULT_CHUNK_SIZE,
    overlap: int = DocumentProcessor.DEFAULT_OVERLAP,
    filter_language: bool = True,
    target_language: str = "en",
) -> Dict[int, List[DocumentChunk]]:
    """
    Convenience function to chunk multiple documents.

    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        filter_language: Whether to filter by language
        target_language: Target language code

    Returns:
        Dictionary mapping document IDs to lists of chunks
    """
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        overlap=overlap,
        filter_language=filter_language,
        target_language=target_language,
    )
    return processor.chunk_documents_batch(documents)
