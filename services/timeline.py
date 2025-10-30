"""
Timeline Builder Service for AI Forecasting Pipeline.

Constructs chronological event timelines from clustered events with
metadata enrichment, citation aggregation, and coverage checks.
"""

import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import structlog

from services.clustering import EventCluster, ClusteringResult
from db.models import Event, TimelineEntry as TimelineEntryModel

logger = structlog.get_logger(__name__)


class CoverageLevel(str, Enum):
    """Timeline coverage quality levels."""
    HIGH = "HIGH"           # 30+ events
    ADEQUATE = "ADEQUATE"   # 15-29 events
    LOW = "LOW"            # <15 events


class TimelineError(Exception):
    """Base exception for timeline-related errors."""
    pass


@dataclass
class Citation:
    """
    A source citation for a timeline entry.

    Attributes:
        url: Source URL
        quote: Relevant quote from source
        event_id: ID of event this citation supports
    """
    url: str
    quote: str
    event_id: Optional[int] = None


@dataclass
class TimelineEntry:
    """
    A single entry in the timeline.

    Attributes:
        event_time: Timestamp of the event (ISO format or natural language)
        summary: Summary text of the event
        citations: List of source citations
        actor_tags: List of actors/entities involved
        topic_tags: List of topic/theme tags
        geographic_tags: List of geographic locations mentioned
        confidence_score: Confidence score from clustering
        cluster_id: ID of source cluster
        canonical_event_id: ID of canonical event
        timestamp_quality: Quality of timestamp (exact, month, year, vague)
    """
    event_time: str
    summary: str
    citations: List[Citation] = field(default_factory=list)
    actor_tags: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    geographic_tags: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    cluster_id: Optional[int] = None
    canonical_event_id: Optional[int] = None
    timestamp_quality: str = "vague"

    def to_model(self, run_id: int) -> TimelineEntryModel:
        """
        Convert to database model.

        Args:
            run_id: ID of the pipeline run

        Returns:
            TimelineEntryModel instance
        """
        # Serialize citations
        citations_json = json.dumps([
            {'url': c.url, 'quote': c.quote, 'event_id': c.event_id}
            for c in self.citations
        ])

        # Combine all tags
        all_tags = {
            'actors': self.actor_tags,
            'topics': self.topic_tags,
            'geographic': self.geographic_tags,
        }
        tags_json = json.dumps(all_tags)

        return TimelineEntryModel(
            run_id=run_id,
            cluster_id=self.cluster_id,
            event_time=self.event_time,
            summary=self.summary,
            citations=citations_json,
            tags=tags_json,
        )


@dataclass
class Timeline:
    """
    Complete timeline with metadata.

    Attributes:
        entries: List of TimelineEntry objects in chronological order
        coverage_level: Quality of coverage (HIGH/ADEQUATE/LOW)
        total_events: Total number of events
        date_range: Tuple of (earliest_date, latest_date)
        actor_summary: Dict of actor counts
        topic_summary: Dict of topic counts
        geographic_summary: Dict of geographic location counts
    """
    entries: List[TimelineEntry]
    coverage_level: CoverageLevel
    total_events: int
    date_range: Optional[Tuple[Optional[str], Optional[str]]] = None
    actor_summary: Dict[str, int] = field(default_factory=dict)
    topic_summary: Dict[str, int] = field(default_factory=dict)
    geographic_summary: Dict[str, int] = field(default_factory=dict)


class TimelineBuilder:
    """
    Service for building chronological timelines from clustered events.

    Features:
    - Temporal sorting with fallbacks
    - Metadata extraction (actors, topics, locations)
    - Citation aggregation
    - Coverage quality assessment
    """

    def __init__(
        self,
        min_adequate_events: int = 15,
        min_high_coverage_events: int = 30,
    ):
        """
        Initialize timeline builder.

        Args:
            min_adequate_events: Minimum events for adequate coverage
            min_high_coverage_events: Minimum events for high coverage
        """
        self.min_adequate_events = min_adequate_events
        self.min_high_coverage_events = min_high_coverage_events
        self.logger = structlog.get_logger(__name__)

        # Compile regex patterns for metadata extraction
        self._compile_patterns()

        self.logger.info(
            "timeline_builder_initialized",
            min_adequate=min_adequate_events,
            min_high_coverage=min_high_coverage_events,
        )

    def _compile_patterns(self):
        """Compile regex patterns for metadata extraction."""
        # Geographic patterns (basic country/city names)
        self.geo_pattern = re.compile(
            r'\b(China|USA|United States|Japan|Germany|France|UK|Britain|'
            r'Russia|India|Brazil|Canada|Australia|Korea|Taiwan|'
            r'Beijing|Shanghai|Tokyo|London|Paris|Berlin|Moscow|Delhi|'
            r'Seoul|Singapore|Hong Kong|Shenzhen|New York|Washington|'
            r'California|Silicon Valley|Europe|Asia|Africa|Americas)\b',
            re.IGNORECASE
        )

        # Topic keywords (technology-focused)
        self.topic_keywords = {
            'robotics': ['robot', 'robotics', 'humanoid', 'automation'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'neural'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'assembly'],
            'policy': ['law', 'regulation', 'policy', 'government', 'legislation'],
            'business': ['company', 'firm', 'corporation', 'startup', 'investment'],
            'research': ['research', 'development', 'r&d', 'university', 'laboratory'],
            'market': ['market', 'sales', 'revenue', 'growth', 'forecast'],
        }

    def _parse_timestamp(self, event_time: Optional[str]) -> Tuple[Optional[datetime], str]:
        """
        Parse event timestamp with quality assessment.

        Args:
            event_time: Timestamp string (ISO format or natural language)

        Returns:
            Tuple of (parsed_datetime, quality_level)
            quality_level: "exact", "month", "year", or "vague"
        """
        if not event_time:
            return None, "vague"

        event_time = event_time.strip()

        # Try ISO format first (exact date)
        for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
            try:
                dt = datetime.strptime(event_time, fmt)
                return dt, "exact"
            except ValueError:
                continue

        # Try month-year format
        for fmt in ['%B %Y', '%b %Y', '%Y-%m']:
            try:
                dt = datetime.strptime(event_time, fmt)
                return dt, "month"
            except ValueError:
                continue

        # Try year only
        if re.match(r'^\d{4}$', event_time):
            try:
                dt = datetime(int(event_time), 1, 1)
                return dt, "year"
            except ValueError:
                pass

        # Vague or unparseable
        return None, "vague"

    def _extract_actor_tags(self, event: Event, cluster: EventCluster) -> List[str]:
        """
        Extract actor tags from event and cluster.

        Args:
            event: Event object
            cluster: EventCluster object

        Returns:
            List of actor names
        """
        actors = set()

        # From event
        if event.actors:
            try:
                event_actors = json.loads(event.actors)
                if isinstance(event_actors, list):
                    actors.update(event_actors)
            except json.JSONDecodeError:
                pass

        # From cluster (merged actors)
        if cluster.merged_actors:
            actors.update(cluster.merged_actors)

        return sorted(list(actors))

    def _extract_topic_tags(self, text: str) -> List[str]:
        """
        Extract topic tags from text using keyword matching.

        Args:
            text: Text to analyze

        Returns:
            List of topic tags
        """
        text_lower = text.lower()
        topics = set()

        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.add(topic)
                    break

        return sorted(list(topics))

    def _extract_geographic_tags(self, text: str) -> List[str]:
        """
        Extract geographic location tags from text using regex.

        Args:
            text: Text to analyze

        Returns:
            List of location names
        """
        matches = self.geo_pattern.findall(text)
        # Deduplicate and normalize
        locations = set(match.title() for match in matches)
        return sorted(list(locations))

    def _build_timeline_entry(
        self,
        cluster: EventCluster,
        events_dict: Dict[int, Event],
        documents_dict: Optional[Dict[int, Any]] = None,
    ) -> TimelineEntry:
        """
        Build a timeline entry from a cluster.

        Args:
            cluster: EventCluster object
            events_dict: Dict mapping event_id to Event
            documents_dict: Optional dict mapping document_id to document data

        Returns:
            TimelineEntry object
        """
        # Get canonical event
        canonical_event = events_dict[cluster.centroid_event_id]

        # Parse timestamp
        dt, timestamp_quality = self._parse_timestamp(canonical_event.event_time)

        # Build summary from canonical event
        summary = canonical_event.headline or canonical_event.body or "Event"

        # Combine text for metadata extraction
        combined_text = f"{canonical_event.headline or ''} {canonical_event.body or ''}"

        # Extract metadata
        actor_tags = self._extract_actor_tags(canonical_event, cluster)
        topic_tags = self._extract_topic_tags(combined_text)
        geographic_tags = self._extract_geographic_tags(combined_text)

        # Build citations from cluster
        citations = []
        for citation_dict in cluster.merged_citations:
            citations.append(Citation(
                url=citation_dict['url'],
                quote=citation_dict['quote'],
                event_id=citation_dict.get('event_id'),
            ))

        entry = TimelineEntry(
            event_time=canonical_event.event_time or "Unknown date",
            summary=summary,
            citations=citations,
            actor_tags=actor_tags,
            topic_tags=topic_tags,
            geographic_tags=geographic_tags,
            confidence_score=cluster.confidence_score,
            cluster_id=cluster.cluster_id,
            canonical_event_id=cluster.centroid_event_id,
            timestamp_quality=timestamp_quality,
        )

        return entry

    def _sort_entries(
        self,
        entries: List[TimelineEntry],
        fallback_dates: Optional[Dict[int, datetime]] = None,
    ) -> List[TimelineEntry]:
        """
        Sort timeline entries chronologically.

        Args:
            entries: List of TimelineEntry objects
            fallback_dates: Optional dict mapping event_id to fallback date

        Returns:
            Sorted list of entries
        """
        def get_sort_key(entry: TimelineEntry) -> Tuple:
            # Parse timestamp
            dt, quality = self._parse_timestamp(entry.event_time)

            if dt is not None:
                # Valid date: sort by date, then by quality
                quality_order = {"exact": 0, "month": 1, "year": 2, "vague": 3}
                return (0, dt, quality_order.get(quality, 4))

            # No valid date: check fallback
            if fallback_dates and entry.canonical_event_id in fallback_dates:
                fallback_dt = fallback_dates[entry.canonical_event_id]
                return (1, fallback_dt, 0)

            # No date at all: sort to end
            return (2, datetime.max, 0)

        return sorted(entries, key=get_sort_key)

    def _compute_date_range(self, entries: List[TimelineEntry]) -> Tuple[Optional[str], Optional[str]]:
        """
        Compute date range for timeline.

        Args:
            entries: List of TimelineEntry objects

        Returns:
            Tuple of (earliest_date, latest_date) as strings
        """
        dates = []
        for entry in entries:
            dt, _ = self._parse_timestamp(entry.event_time)
            if dt:
                dates.append(dt)

        if not dates:
            return None, None

        earliest = min(dates).strftime('%Y-%m-%d')
        latest = max(dates).strftime('%Y-%m-%d')

        return earliest, latest

    def _compute_metadata_summaries(
        self,
        entries: List[TimelineEntry]
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Compute summary statistics for metadata.

        Args:
            entries: List of TimelineEntry objects

        Returns:
            Tuple of (actor_counts, topic_counts, geo_counts)
        """
        actor_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}
        geo_counts: Dict[str, int] = {}

        for entry in entries:
            for actor in entry.actor_tags:
                actor_counts[actor] = actor_counts.get(actor, 0) + 1

            for topic in entry.topic_tags:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            for geo in entry.geographic_tags:
                geo_counts[geo] = geo_counts.get(geo, 0) + 1

        return actor_counts, topic_counts, geo_counts

    def _assess_coverage(self, n_events: int) -> CoverageLevel:
        """
        Assess timeline coverage quality.

        Args:
            n_events: Number of timeline events

        Returns:
            CoverageLevel enum
        """
        if n_events >= self.min_high_coverage_events:
            return CoverageLevel.HIGH
        elif n_events >= self.min_adequate_events:
            return CoverageLevel.ADEQUATE
        else:
            return CoverageLevel.LOW

    def build_timeline(
        self,
        clustering_result: ClusteringResult,
        events: List[Event],
        documents: Optional[List[Any]] = None,
        fallback_dates: Optional[Dict[int, datetime]] = None,
    ) -> Timeline:
        """
        Build a chronological timeline from clustering results.

        Args:
            clustering_result: ClusteringResult with clusters
            events: List of Event objects
            documents: Optional list of document objects for fallback dates
            fallback_dates: Optional dict mapping event_id to fallback date

        Returns:
            Timeline object with sorted entries and metadata

        Raises:
            TimelineError: If timeline construction fails
        """
        self.logger.info(
            "timeline_build_start",
            n_clusters=len(clustering_result.clusters),
            n_events=len(events),
        )

        # Build event lookup dict
        events_dict = {event.id: event for event in events}

        # Build document lookup dict if provided
        documents_dict = None
        if documents:
            documents_dict = {doc.id: doc for doc in documents}

        # Build timeline entries
        entries = []
        for cluster in clustering_result.clusters:
            try:
                entry = self._build_timeline_entry(cluster, events_dict, documents_dict)
                entries.append(entry)
            except Exception as e:
                self.logger.warning(
                    "timeline_entry_build_failed",
                    cluster_id=cluster.cluster_id,
                    error=str(e),
                )
                # Continue with other entries

        # Sort chronologically
        entries = self._sort_entries(entries, fallback_dates)

        # Compute metadata
        date_range = self._compute_date_range(entries)
        actor_summary, topic_summary, geo_summary = self._compute_metadata_summaries(entries)

        # Assess coverage
        coverage = self._assess_coverage(len(entries))

        timeline = Timeline(
            entries=entries,
            coverage_level=coverage,
            total_events=len(entries),
            date_range=date_range,
            actor_summary=actor_summary,
            topic_summary=topic_summary,
            geographic_summary=geo_summary,
        )

        self.logger.info(
            "timeline_build_complete",
            total_entries=timeline.total_events,
            coverage_level=coverage,
            date_range=date_range,
            n_actors=len(actor_summary),
            n_topics=len(topic_summary),
            n_locations=len(geo_summary),
        )

        if coverage == CoverageLevel.LOW:
            self.logger.warning(
                "timeline_low_coverage",
                total_events=timeline.total_events,
                min_adequate=self.min_adequate_events,
            )

        return timeline


# Convenience functions

def build_timeline(
    clustering_result: ClusteringResult,
    events: List[Event],
    documents: Optional[List[Any]] = None,
    min_adequate_events: int = 15,
    min_high_coverage_events: int = 30,
) -> Timeline:
    """
    Convenience function to build a timeline.

    Args:
        clustering_result: ClusteringResult object
        events: List of Event objects
        documents: Optional list of documents
        min_adequate_events: Minimum events for adequate coverage
        min_high_coverage_events: Minimum events for high coverage

    Returns:
        Timeline object
    """
    builder = TimelineBuilder(
        min_adequate_events=min_adequate_events,
        min_high_coverage_events=min_high_coverage_events,
    )

    return builder.build_timeline(
        clustering_result=clustering_result,
        events=events,
        documents=documents,
    )


def format_timeline_markdown(timeline: Timeline) -> str:
    """
    Format timeline as Markdown for display.

    Args:
        timeline: Timeline object

    Returns:
        Markdown-formatted string
    """
    lines = []

    # Header
    lines.append("# Event Timeline")
    lines.append("")
    lines.append(f"**Coverage:** {timeline.coverage_level.value}")
    lines.append(f"**Total Events:** {timeline.total_events}")

    if timeline.date_range and timeline.date_range[0]:
        lines.append(f"**Date Range:** {timeline.date_range[0]} to {timeline.date_range[1]}")

    lines.append("")

    # Top actors
    if timeline.actor_summary:
        lines.append("**Key Actors:**")
        top_actors = sorted(timeline.actor_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        for actor, count in top_actors:
            lines.append(f"- {actor} ({count} events)")
        lines.append("")

    # Top topics
    if timeline.topic_summary:
        lines.append("**Topics:**")
        top_topics = sorted(timeline.topic_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        for topic, count in top_topics:
            lines.append(f"- {topic} ({count} events)")
        lines.append("")

    # Timeline entries
    lines.append("## Events")
    lines.append("")

    for i, entry in enumerate(timeline.entries, 1):
        lines.append(f"### {i}. {entry.event_time}")
        lines.append(f"**{entry.summary}**")
        lines.append("")

        if entry.actor_tags:
            lines.append(f"*Actors:* {', '.join(entry.actor_tags)}")

        if entry.citations:
            lines.append(f"*Sources:* {len(entry.citations)} citation(s)")
            for citation in entry.citations[:3]:  # Show max 3
                lines.append(f"- {citation.url}")

        lines.append("")

    return "\n".join(lines)
