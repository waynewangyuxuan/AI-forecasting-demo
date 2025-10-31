"""
Configuration and Settings Management
Uses pydantic-settings to load and validate environment variables.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All sensitive configuration (API keys) should be provided via environment
    variables or a .env file. Non-sensitive defaults are defined here.
    """

    # API Keys (required)
    google_api_key: str = Field(..., description="Google Custom Search API key")
    google_cse_id: str = Field(..., description="Google Custom Search Engine ID")
    google_gemini_api_key: str = Field(..., description="Google Gemini API key for LLM")
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///data/forecast.db",
        description="SQLite database URL"
    )

    # Application Settings
    debug: bool = Field(default=False, description="Enable debug mode")

    # HTTP and Scraping Configuration
    max_concurrent_scrapes: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent HTTP requests for scraping"
    )
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="HTTP request timeout in seconds"
    )
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; ForecastBot/1.0; +https://github.com/yourrepo)",
        description="User agent for HTTP requests"
    )

    # LLM Configuration
    llm_provider: str = Field(
        default="gemini",
        description="LLM provider to use: 'gemini' or 'openai'"
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Gemini model to use for LLM operations"
    )
    openai_llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for LLM operations"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model"
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions"
    )
    event_extraction_max_tokens: int = Field(
        default=4000,
        ge=1000,
        le=16000,
        description="Maximum tokens for event extraction LLM responses"
    )

    # Query Generation
    max_search_queries: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of search queries to generate per question"
    )
    search_results_per_query: int = Field(
        default=10,
        ge=1,
        le=10,
        description="Number of search results to retrieve per query"
    )

    # Clustering Configuration
    clustering_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Distance threshold for event clustering (cosine distance)"
    )

    # Project Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory"
    )

    @property
    def data_dir(self) -> Path:
        """Directory for data files (database)."""
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        """Directory for forecast outputs."""
        return self.project_root / "outputs"

    @property
    def cache_dir(self) -> Path:
        """Directory for HTTP cache."""
        return self.project_root / ".cache"

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL is SQLite."""
        if not v.startswith("sqlite"):
            raise ValueError("Only SQLite databases are supported")
        return v

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Ensure LLM provider is valid."""
        v = v.lower()
        if v not in ("gemini", "openai"):
            raise ValueError("LLM provider must be 'gemini' or 'openai'")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
# Import this in other modules: from config.settings import settings
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency injection helper for FastAPI or other frameworks.

    Returns:
        Settings instance
    """
    return settings
