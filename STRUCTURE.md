# Project Structure Documentation

This document provides an overview of the AI Forecasting Pipeline project structure.

## Directory Layout

```
AI-forecasting-demo/
├── .cache/                 # HTTP response caching during development
│   └── .gitkeep           # Preserve empty directory in git
├── config/                 # Configuration management
│   ├── __init__.py        # Package initialization
│   └── settings.py        # Pydantic settings with env validation
├── data/                   # Data storage (SQLite database)
│   └── .gitkeep           # Preserve empty directory in git
├── db/                     # Database schema and repository layer
│   └── __init__.py        # Package initialization
├── outputs/                # Forecast results (JSON, Markdown)
│   └── .gitkeep           # Preserve empty directory in git
├── pipeline/               # Orchestration layer
│   └── __init__.py        # Package initialization
├── services/               # Service modules (business logic)
│   └── __init__.py        # Package initialization
├── META/                   # Project documentation
│   ├── META.md            # Meta documentation index
│   ├── PRODUCT.md         # Product requirements and design
│   ├── TECHNICAL.md       # Technical architecture
│   ├── REGULATION.md      # Development principles
│   ├── PROGRESS.md        # Development progress tracking
│   └── TODO.md            # Task tracking
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore patterns
├── README.md               # Project overview and quick start
├── SETUP.md                # Detailed setup instructions
├── STRUCTURE.md            # This file - project structure documentation
├── requirements.txt        # Python dependencies
└── verify_setup.py         # Setup verification script
```

## Package Purposes

### `/config` - Configuration Management
- **Purpose**: Centralized configuration and settings
- **Key File**: `settings.py` uses pydantic-settings for type-safe environment loading
- **Usage**: `from config.settings import settings`

### `/pipeline` - Orchestration Layer
- **Purpose**: Coordinates execution of all pipeline stages
- **Responsibilities**:
  - Pipeline runner
  - Task graph management
  - State tracking and resumption
  - Stage-by-stage execution

### `/services` - Service Modules
- **Purpose**: Atomic, single-purpose service implementations
- **Planned Services**:
  - `query_generation.py` - LLM-based search query generation
  - `search.py` - Google Custom Search API integration
  - `scraper.py` - Web content scraping and extraction
  - `event_extractor.py` - Event extraction from documents
  - `embedding.py` - Text embedding generation (OpenAI)
  - `clustering.py` - Event clustering and deduplication
  - `timeline.py` - Timeline construction
  - `forecast.py` - Forecast generation with LLM

### `/db` - Database Layer
- **Purpose**: Data persistence and schema management
- **Planned Components**:
  - `schema.sql` - SQLite schema definitions
  - `repository.py` - Repository pattern for data access
  - `models.py` - SQLAlchemy models (optional)
  - `migrate.py` - Database migration utilities

### `/data` - Data Storage
- **Purpose**: SQLite database file location
- **File**: `forecast.db` (created on first run)
- **Note**: Excluded from git via .gitignore

### `/outputs` - Results Output
- **Purpose**: Store forecast results and reports
- **Format**: JSON and Markdown files
- **Organization**: `outputs/<question_id>/`

### `/.cache` - HTTP Cache
- **Purpose**: Cache HTTP responses during development
- **Use**: Reduces API calls when testing
- **Note**: Excluded from git via .gitignore

## Key Files

### `requirements.txt`
Python package dependencies for the project. Includes:
- CLI/API: typer, fastapi, uvicorn
- HTTP: httpx, beautifulsoup4, readability-lxml, trafilatura
- Database: sqlalchemy
- ML: numpy, scikit-learn
- LLM: google-generativeai, openai
- Config: pydantic, pydantic-settings, python-dotenv
- Logging: structlog

### `.env.example`
Template for environment variables. Copy to `.env` and fill with actual values:
- `GOOGLE_API_KEY` - Google Custom Search API
- `GOOGLE_CSE_ID` - Custom Search Engine ID
- `GOOGLE_GEMINI_API_KEY` - Gemini LLM API
- `OPENAI_API_KEY` - OpenAI embeddings API
- `DEBUG` - Enable debug logging (optional)

### `verify_setup.py`
Automated setup verification script. Checks:
- Python version (3.11+)
- All dependencies installed
- Directory structure exists
- `.env` file configured
- Settings load successfully

## Design Principles

Following `META/REGULATION.md`:

1. **Atomic File Structure**: Each file has single, well-defined purpose
2. **Atomic Code**: Functions and classes do one thing well
3. **Co-located Documentation**: Complex features documented in same directory
4. **Proper Organization**: Logical folder hierarchy, not too deep

## Next Steps

1. Implement database schema (`db/schema.sql`)
2. Create service modules in `services/`
3. Build orchestration layer in `pipeline/`
4. Add CLI entry point
5. Implement testing framework

## Development Workflow

1. **Setup**: Follow `SETUP.md`
2. **Verify**: Run `python verify_setup.py`
3. **Develop**: Implement components per `META/TECHNICAL.md`
4. **Test**: Write tests alongside features
5. **Document**: Update META docs as needed

## Notes

- All absolute paths should resolve from project root
- Use `settings.project_root` for path resolution
- Keep `.env` file secret (never commit)
- Empty directories tracked with `.gitkeep` files
