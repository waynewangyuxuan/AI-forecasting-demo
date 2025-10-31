"""
Event Extraction Module for AI Forecasting Pipeline.

Provides LLM-based event extraction from document chunks with structured
output, confidence scoring, and batch processing capabilities.
"""

import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum

import structlog

from config.settings import settings
from services.llm_client import LLMClient, create_llm_client, ParsingError
from services.doc_processor import DocumentChunk
from db.models import Event

logger = structlog.get_logger(__name__)


# Prompt version for tracking changes
EVENT_EXTRACTION_PROMPT_VERSION = "v1.0"


class TimestampSpecificity(str, Enum):
    """Specificity levels for event timestamps."""
    EXACT_DATE = "exact_date"  # Specific date (e.g., "2024-01-15")
    MONTH = "month"  # Month and year (e.g., "January 2024")
    YEAR = "year"  # Year only (e.g., "2024")
    VAGUE = "vague"  # Vague or no date (e.g., "recently", "soon")


@dataclass
class ExtractedEvent:
    """
    An event extracted from a document chunk.

    Attributes:
        timestamp: Event timestamp in ISO format or natural language
        headline: Brief 1-2 sentence headline
        body: 2-3 sentence detailed description
        actors: List of entities/actors involved
        quote: Supporting quote from source text
        confidence: Confidence score (0.0-1.0)
        timestamp_specificity: How specific the timestamp is
        source_chunk_id: ID of source chunk
        raw_response: Raw LLM response for debugging
    """
    timestamp: str
    headline: str
    body: str
    actors: List[str]
    quote: str
    confidence: float
    timestamp_specificity: TimestampSpecificity
    source_chunk_id: str
    raw_response: Optional[str] = None

    def to_event_model(self, document_id: int) -> Event:
        """
        Convert to database Event model.

        Args:
            document_id: ID of source document

        Returns:
            Event model instance
        """
        return Event(
            document_id=document_id,
            event_time=self.timestamp,
            headline=self.headline,
            body=self.body,
            actors=json.dumps(self.actors),
            confidence=self.confidence,
            raw_response=self.raw_response,
        )


def get_event_extraction_prompt(chunk_text: str, question_context: Optional[str] = None) -> str:
    """
    Generate event extraction prompt for LLM.

    Args:
        chunk_text: Text chunk to extract events from
        question_context: Optional forecasting question for context

    Returns:
        Formatted prompt string
    """
    context_section = ""
    if question_context:
        context_section = f"""
Forecasting Context:
The events you extract will be used to forecast: "{question_context}"
Focus on events that are relevant to this forecasting question.
"""

    prompt = f"""You are an expert at extracting timestamped events from news articles and documents for forecasting purposes. Your task is to identify and extract concrete, factual events with clear temporal anchors.

{context_section}

INSTRUCTIONS:
1. Extract ONLY the TOP 3-5 MOST IMPORTANT factual events that have actually occurred or are scheduled to occur
2. Prioritize events most relevant to the forecasting question
3. Each event MUST have a temporal anchor (date, timeframe, or clear temporal reference)
4. Focus on events involving concrete actions, decisions, announcements, or measurable changes
5. Include the key actors/entities involved
6. Extract a supporting quote directly from the source text
7. Provide confidence score based on:
   - Timestamp specificity (exact date > month > year > vague)
   - Clarity of the event description
   - Presence of supporting details
   - Source credibility indicators
8. LIMIT: Extract maximum 5 events per chunk to ensure complete, valid JSON responses

WHAT TO EXTRACT:
- Policy announcements and decisions
- Economic indicators and forecasts
- Political events and elections
- Scientific breakthroughs or studies
- Business deals and financial transactions
- Regulatory changes
- Military or geopolitical developments
- Technology launches or adoptions

WHAT TO AVOID:
- Opinions or speculation without factual basis
- General statements without temporal anchors
- Background information or context
- Hypothetical scenarios
- Duplicate events (only extract once)

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
    "events": [
        {{
            "timestamp": "YYYY-MM-DD or natural language date",
            "headline": "Brief 1-2 sentence headline of what happened",
            "body": "2-3 sentence detailed description with key facts",
            "actors": ["list", "of", "key", "entities", "involved"],
            "quote": "Direct quote from source text supporting this event",
            "confidence": 0.85,
            "timestamp_specificity": "exact_date|month|year|vague"
        }}
    ]
}}

If no clear events with temporal anchors are found, return: {{"events": []}}

IMPORTANT: Extract maximum 5 events to keep response concise and prevent truncation.

TEXT TO ANALYZE:
{chunk_text}

Extract the top 3-5 most important events in JSON format:"""

    return prompt


class EventExtractor:
    """
    Event extraction service with LLM-based extraction and confidence scoring.

    Features:
    - Batched LLM requests for efficiency
    - Structured JSON output with schema validation
    - Confidence scoring with multiple factors
    - Retry logic for malformed JSON
    - Support for streaming results
    - Comprehensive error handling
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        batch_size: int = 5,
        temperature: float = 0.3,
        max_retries: int = 2,
        max_tokens: Optional[int] = None,
        question_context: Optional[str] = None,
    ):
        """
        Initialize event extractor.

        Args:
            llm_client: LLM client to use (defaults to Gemini)
            batch_size: Number of chunks to process in parallel
            temperature: LLM temperature for extraction
            max_retries: Maximum retries for malformed JSON
            max_tokens: Maximum tokens for LLM responses (defaults to settings)
            question_context: Optional forecasting question for context
        """
        self.llm_client = llm_client or create_llm_client()
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens or settings.event_extraction_max_tokens
        self.question_context = question_context
        self.logger = structlog.get_logger(__name__)

        self.stats = {
            "chunks_processed": 0,
            "events_extracted": 0,
            "parse_errors": 0,
            "empty_results": 0,
        }

        self.logger.info(
            "event_extractor_initialized",
            batch_size=batch_size,
            temperature=temperature,
            has_question_context=question_context is not None,
        )

    def _determine_timestamp_specificity(self, timestamp: str) -> TimestampSpecificity:
        """
        Determine how specific a timestamp is.

        Args:
            timestamp: Timestamp string

        Returns:
            TimestampSpecificity enum value
        """
        # Check for ISO date format (YYYY-MM-DD)
        if re.match(r'^\d{4}-\d{2}-\d{2}', timestamp):
            return TimestampSpecificity.EXACT_DATE

        # Check for month/year
        if re.match(r'^\d{4}-\d{2}', timestamp) or re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}', timestamp.lower()):
            return TimestampSpecificity.MONTH

        # Check for year only
        if re.match(r'^\d{4}$', timestamp) or re.search(r'in\s+\d{4}|during\s+\d{4}', timestamp.lower()):
            return TimestampSpecificity.YEAR

        # Default to vague
        return TimestampSpecificity.VAGUE

    def _calculate_confidence(
        self,
        event_data: Dict[str, Any],
        chunk: DocumentChunk,
    ) -> float:
        """
        Calculate confidence score for an extracted event.

        Factors:
        - LLM-provided confidence (if available)
        - Timestamp specificity
        - Quote quality (presence and length)
        - Body detail level
        - Actor specificity

        Args:
            event_data: Raw event data from LLM
            chunk: Source chunk

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start with LLM confidence if provided
        base_confidence = event_data.get("confidence", 0.7)

        # Timestamp specificity bonus
        specificity = self._determine_timestamp_specificity(event_data.get("timestamp", ""))
        specificity_bonus = {
            TimestampSpecificity.EXACT_DATE: 0.15,
            TimestampSpecificity.MONTH: 0.10,
            TimestampSpecificity.YEAR: 0.05,
            TimestampSpecificity.VAGUE: -0.10,
        }
        confidence = base_confidence + specificity_bonus[specificity]

        # Quote quality bonus
        quote = event_data.get("quote", "")
        if quote and len(quote) > 20:
            # Verify quote actually appears in source
            if quote.lower() in chunk.content.lower():
                confidence += 0.10
            else:
                confidence -= 0.05  # Penalty for hallucinated quote

        # Body detail bonus
        body = event_data.get("body", "")
        if body and len(body.split()) >= 20:  # At least 20 words
            confidence += 0.05

        # Actor specificity bonus
        actors = event_data.get("actors", [])
        if actors and len(actors) >= 1:
            confidence += 0.05

        # Clamp to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        return confidence

    def _parse_llm_response(
        self,
        response: str,
        chunk: DocumentChunk,
    ) -> List[ExtractedEvent]:
        """
        Parse LLM response and create ExtractedEvent objects.

        Args:
            response: Raw LLM response
            chunk: Source chunk

        Returns:
            List of ExtractedEvent objects

        Raises:
            ParsingError: If response cannot be parsed
        """
        try:
            # Try to parse JSON
            data = json.loads(response)

            # Validate structure
            if "events" not in data:
                raise ParsingError("Response missing 'events' key")

            events = []
            for event_data in data["events"]:
                # Validate required fields
                required_fields = ["timestamp", "headline", "body", "actors", "quote"]
                for field in required_fields:
                    if field not in event_data:
                        self.logger.warning(
                            "event_missing_field",
                            field=field,
                            chunk_id=chunk.chunk_id,
                        )
                        continue

                # Determine timestamp specificity
                specificity = self._determine_timestamp_specificity(event_data["timestamp"])

                # Calculate confidence
                confidence = self._calculate_confidence(event_data, chunk)

                # Create ExtractedEvent
                event = ExtractedEvent(
                    timestamp=event_data["timestamp"],
                    headline=event_data["headline"],
                    body=event_data["body"],
                    actors=event_data["actors"] if isinstance(event_data["actors"], list) else [],
                    quote=event_data["quote"],
                    confidence=confidence,
                    timestamp_specificity=specificity,
                    source_chunk_id=chunk.chunk_id,
                    raw_response=response,
                )
                events.append(event)

            return events

        except json.JSONDecodeError as e:
            # Try to extract JSON from response (sometimes LLMs add extra text)
            # NO RECURSION - just try to parse directly to avoid error log spam
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    # If we got valid JSON with events array, process it
                    if "events" in data and isinstance(data["events"], list):
                        events = []
                        for event_data in data["events"]:
                            specificity = event_data.get("timestamp_specificity", "day")
                            confidence = self._calculate_confidence(event_data, chunk)
                            event = ExtractedEvent(
                                timestamp=event_data["timestamp"],
                                headline=event_data["headline"],
                                body=event_data["body"],
                                actors=event_data["actors"] if isinstance(event_data["actors"], list) else [],
                                quote=event_data["quote"],
                                confidence=confidence,
                                timestamp_specificity=specificity,
                                source_chunk_id=chunk.chunk_id,
                                raw_response=response,
                            )
                            events.append(event)
                        return events
                except Exception:
                    pass

            # Log error only once
            self.logger.error(
                "json_parse_error",
                chunk_id=chunk.chunk_id,
                error=str(e),
                response_preview=response[:200],
            )
            raise ParsingError(f"Failed to parse JSON response: {e}")

    async def extract_from_chunk(
        self,
        chunk: DocumentChunk,
        retry_count: int = 0,
    ) -> List[ExtractedEvent]:
        """
        Extract events from a single chunk.

        Args:
            chunk: Document chunk to process
            retry_count: Current retry count

        Returns:
            List of extracted events
        """
        import time
        start_time = time.time()

        # Log chunk extraction start with metadata
        content_preview = chunk.content[:100].replace('\n', ' ') if chunk.content else ""
        self.logger.info(
            "chunk_extraction_start",
            chunk_id=chunk.chunk_id,
            doc_id=chunk.source_doc_id,
            token_count=chunk.token_count,
            retry_count=retry_count,
            content_preview=content_preview,
            source_url=chunk.metadata.get("source_url") if chunk.metadata else None,
        )

        try:
            # Generate prompt
            prompt = get_event_extraction_prompt(
                chunk.content,
                self.question_context,
            )

            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format="json",
            )

            # Parse response
            events = self._parse_llm_response(response, chunk)

            self.stats["chunks_processed"] += 1
            self.stats["events_extracted"] += len(events)

            if not events:
                self.stats["empty_results"] += 1

            duration = time.time() - start_time
            self.logger.info(
                "chunk_extraction_complete",
                chunk_id=chunk.chunk_id,
                event_count=len(events),
                doc_id=chunk.source_doc_id,
                duration_seconds=round(duration, 2),
                tokens_per_second=round(chunk.token_count / duration, 1) if duration > 0 else 0,
            )

            return events

        except ParsingError as e:
            self.stats["parse_errors"] += 1
            duration = time.time() - start_time

            # Retry if we haven't exceeded max retries
            if retry_count < self.max_retries:
                self.logger.warning(
                    "event_extraction_parse_error_retry",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.source_doc_id,
                    retry_count=retry_count + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                    duration_before_retry=round(duration, 2),
                    source_url=chunk.metadata.get("source_url") if chunk.metadata else None,
                )
                # Add a brief delay before retry
                import asyncio
                await asyncio.sleep(1)
                return await self.extract_from_chunk(chunk, retry_count + 1)
            else:
                self.logger.error(
                    "event_extraction_parse_error_max_retries",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.source_doc_id,
                    error=str(e),
                    total_duration=round(duration, 2),
                    source_url=chunk.metadata.get("source_url") if chunk.metadata else None,
                )
                return []

        except Exception as e:
            self.logger.error(
                "event_extraction_error",
                chunk_id=chunk.chunk_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

    async def extract_from_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> List[ExtractedEvent]:
        """
        Extract events from multiple chunks in batches.

        Args:
            chunks: List of chunks to process

        Returns:
            List of all extracted events
        """
        self.logger.info(
            "batch_extraction_start",
            chunk_count=len(chunks),
            batch_size=self.batch_size,
        )

        all_events = []

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            self.logger.debug(
                "processing_batch",
                batch_num=i // self.batch_size + 1,
                batch_size=len(batch),
            )

            # Process batch
            import asyncio
            tasks = [self.extract_from_chunk(chunk) for chunk in batch]
            batch_results = await asyncio.gather(*tasks)

            # Flatten results
            for events in batch_results:
                all_events.extend(events)

        self.logger.info(
            "batch_extraction_complete",
            chunks_processed=self.stats["chunks_processed"],
            events_extracted=self.stats["events_extracted"],
            empty_results=self.stats["empty_results"],
            parse_errors=self.stats["parse_errors"],
            avg_events_per_chunk=self.stats["events_extracted"] / self.stats["chunks_processed"] if self.stats["chunks_processed"] > 0 else 0,
        )

        return all_events

    async def extract_from_chunks_streaming(
        self,
        chunks: List[DocumentChunk],
    ) -> AsyncIterator[ExtractedEvent]:
        """
        Extract events from chunks with streaming support.

        Yields events as they are extracted for real-time processing.

        Args:
            chunks: List of chunks to process

        Yields:
            ExtractedEvent objects as they are extracted
        """
        self.logger.info(
            "streaming_extraction_start",
            chunk_count=len(chunks),
        )

        for chunk in chunks:
            events = await self.extract_from_chunk(chunk)
            for event in events:
                yield event

        self.logger.info(
            "streaming_extraction_complete",
            chunks_processed=self.stats["chunks_processed"],
            events_extracted=self.stats["events_extracted"],
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return dict(self.stats)


async def extract_events_from_chunks(
    chunks: List[DocumentChunk],
    llm_client: Optional[LLMClient] = None,
    batch_size: int = 5,
    temperature: float = 0.3,
    question_context: Optional[str] = None,
) -> List[ExtractedEvent]:
    """
    Convenience function to extract events from chunks.

    Args:
        chunks: List of document chunks
        llm_client: Optional LLM client
        batch_size: Batch size for processing
        temperature: LLM temperature
        question_context: Optional forecasting question

    Returns:
        List of extracted events
    """
    extractor = EventExtractor(
        llm_client=llm_client,
        batch_size=batch_size,
        temperature=temperature,
        question_context=question_context,
    )
    return await extractor.extract_from_chunks(chunks)


def filter_events_by_confidence(
    events: List[ExtractedEvent],
    min_confidence: float = 0.5,
) -> List[ExtractedEvent]:
    """
    Filter events by minimum confidence threshold.

    Args:
        events: List of events
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list of events
    """
    filtered = [e for e in events if e.confidence >= min_confidence]

    logger.info(
        "events_filtered_by_confidence",
        original_count=len(events),
        filtered_count=len(filtered),
        min_confidence=min_confidence,
        filtered_ratio=len(filtered) / len(events) if events else 0,
    )

    return filtered


def filter_events_by_specificity(
    events: List[ExtractedEvent],
    min_specificity: TimestampSpecificity = TimestampSpecificity.YEAR,
) -> List[ExtractedEvent]:
    """
    Filter events by minimum timestamp specificity.

    Args:
        events: List of events
        min_specificity: Minimum timestamp specificity

    Returns:
        Filtered list of events
    """
    specificity_order = {
        TimestampSpecificity.EXACT_DATE: 3,
        TimestampSpecificity.MONTH: 2,
        TimestampSpecificity.YEAR: 1,
        TimestampSpecificity.VAGUE: 0,
    }

    min_level = specificity_order[min_specificity]
    filtered = [e for e in events if specificity_order[e.timestamp_specificity] >= min_level]

    logger.info(
        "events_filtered_by_specificity",
        original_count=len(events),
        filtered_count=len(filtered),
        min_specificity=min_specificity.value,
        filtered_ratio=len(filtered) / len(events) if events else 0,
    )

    return filtered
