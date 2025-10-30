"""
Pydantic models for AI Forecasting Pipeline database entities.

These models provide type-safe data validation and serialization for all
database entities in the forecasting pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict


class RunStatus(str, Enum):
    """Status enumeration for pipeline runs."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class Question(BaseModel):
    """Model for forecasting questions."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    question_text: str = Field(..., min_length=1, description="The question to forecast")
    resolution_criteria: Optional[str] = Field(None, description="Criteria for resolving the question")
    created_at: Optional[datetime] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string if needed."""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        return v


class Run(BaseModel):
    """Model for pipeline execution runs."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    question_id: int = Field(..., description="Reference to the question being forecasted")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current status of the run")
    stage_state: Optional[str] = Field(None, description="JSON state of current stage")
    git_commit: Optional[str] = Field(None, description="Git commit hash at time of run")

    @field_validator('started_at', 'completed_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        return v


class SearchQuery(BaseModel):
    """Model for search queries generated during the pipeline."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: int = Field(..., description="Reference to the run that generated this query")
    query_text: str = Field(..., min_length=1, description="The search query text")
    prompt_version: Optional[str] = Field(None, description="Version of prompt used to generate query")
    created_at: Optional[datetime] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        return v


class SearchResult(BaseModel):
    """Model for search results from search engines."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    query_id: int = Field(..., description="Reference to the search query")
    url: str = Field(..., description="URL of the search result")
    title: Optional[str] = Field(None, description="Title of the search result")
    snippet: Optional[str] = Field(None, description="Snippet/description from search result")
    rank: Optional[int] = Field(None, ge=0, description="Rank position in search results")


class Document(BaseModel):
    """Model for fetched and processed documents."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: int = Field(..., description="Reference to the run that fetched this document")
    url: str = Field(..., description="URL of the document")
    content_hash: str = Field(..., description="Hash of the document content for deduplication")
    raw_content: Optional[str] = Field(None, description="Raw HTML/text content")
    cleaned_content: Optional[str] = Field(None, description="Cleaned and processed content")
    fetched_at: Optional[datetime] = None
    status: Optional[str] = Field(None, description="Processing status of the document")

    @field_validator('fetched_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        return v


class Event(BaseModel):
    """Model for extracted events from documents."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    document_id: int = Field(..., description="Reference to the source document")
    event_time: Optional[str] = Field(None, description="When the event occurred (ISO format or natural language)")
    headline: Optional[str] = Field(None, description="Brief headline of the event")
    body: Optional[str] = Field(None, description="Detailed description of the event")
    actors: Optional[str] = Field(None, description="JSON array of actors involved in the event")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score for the extraction")
    raw_response: Optional[str] = Field(None, description="Raw LLM response for debugging")


class Embedding(BaseModel):
    """Model for vector embeddings of events."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    event_id: int = Field(..., description="Reference to the event")
    vector: bytes = Field(..., description="Binary encoded vector embedding")
    model: Optional[str] = Field(None, description="Model used to generate the embedding")
    dimensions: Optional[int] = Field(None, gt=0, description="Dimensionality of the vector")
    created_at: Optional[datetime] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        return v


class EventCluster(BaseModel):
    """Model for clusters of similar events."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: int = Field(..., description="Reference to the run")
    label: Optional[str] = Field(None, description="Human-readable label for the cluster")
    centroid_event_id: Optional[int] = Field(None, description="ID of the centroid event")
    member_ids: Optional[str] = Field(None, description="JSON array of event IDs in this cluster")


class TimelineEntry(BaseModel):
    """Model for timeline entries in the synthesized timeline."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: int = Field(..., description="Reference to the run")
    cluster_id: Optional[int] = Field(None, description="Reference to the event cluster")
    event_time: Optional[str] = Field(None, description="When this timeline entry occurred")
    summary: Optional[str] = Field(None, description="Summary of the timeline entry")
    citations: Optional[str] = Field(None, description="JSON array of source citations")
    tags: Optional[str] = Field(None, description="JSON array of tags/categories")


class Forecast(BaseModel):
    """Model for final forecast predictions."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: int = Field(..., description="Reference to the run")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Predicted probability")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the forecast")
    caveats: Optional[str] = Field(None, description="Caveats and limitations")
    raw_response: Optional[str] = Field(None, description="Raw LLM response for debugging")
    created_at: Optional[datetime] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        return v


class RunMetric(BaseModel):
    """Model for performance metrics collected during runs."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: int = Field(..., description="Reference to the run")
    metric_name: Optional[str] = Field(None, description="Name of the metric")
    metric_value: Optional[float] = Field(None, description="Value of the metric")


class Error(BaseModel):
    """Model for errors that occur during pipeline execution."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: int = Field(..., description="Reference to the run")
    stage: Optional[str] = Field(None, description="Pipeline stage where error occurred")
    reference: Optional[str] = Field(None, description="Reference ID (e.g., document_id, event_id)")
    error_type: Optional[str] = Field(None, description="Type/category of the error")
    message: Optional[str] = Field(None, description="Error message")
    created_at: Optional[datetime] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        return v
