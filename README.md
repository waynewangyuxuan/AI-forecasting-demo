# AI Forecasting Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that combines real-time web information with large language model reasoning to produce evidence-based forecasts for complex prediction questions.

## Overview

This system automates the process of:
1. Generating optimized search queries from forecasting questions
2. Retrieving and scraping relevant web content
3. Extracting and deduplicating events using LLM and embeddings
4. Constructing chronological timelines
5. Generating probabilistic forecasts with transparent reasoning

## Quick Start

### Prerequisites

- Python 3.11 or higher
- API keys for:
  - Google Custom Search API
  - Google Gemini API
  - OpenAI API

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-forecasting-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Edit the `.env` file with your API credentials:

```bash
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Usage

(Usage instructions will be added as CLI commands are implemented)

## Project Structure

```
AI-forecasting-demo/
├── pipeline/           # Orchestration layer
├── services/          # Service modules
│   ├── query_generation.py
│   ├── search.py
│   ├── scraper.py
│   ├── event_extractor.py
│   ├── embedding.py
│   ├── clustering.py
│   ├── timeline.py
│   └── forecast.py
├── db/                # Database schema and repository
├── config/            # Configuration management
├── data/              # SQLite database file
├── outputs/           # Forecast results
├── .cache/            # HTTP caching (dev)
├── META/              # Project documentation
├── requirements.txt   # Python dependencies
└── .env.example       # Environment template
```

## Architecture

The pipeline follows a multi-stage architecture:

1. **Query Generation**: LLM generates diverse search queries
2. **Web Search**: Google Custom Search API retrieves URLs
3. **Scraping**: Async HTTP requests fetch and clean content
4. **Event Extraction**: LLM extracts timestamped events
5. **Clustering**: Embeddings and clustering deduplicate events
6. **Timeline**: Events organized chronologically with citations
7. **Forecasting**: LLM generates probability estimates with reasoning
8. **Reporting**: Results exported as JSON and Markdown

## Technology Stack

- **Language**: Python 3.11+
- **CLI**: Typer
- **API (Optional)**: FastAPI + Uvicorn
- **HTTP**: httpx (async)
- **Web Scraping**: BeautifulSoup4, readability-lxml, trafilatura
- **Database**: SQLite + SQLAlchemy
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: OpenAI text-embedding-3-large
- **ML**: scikit-learn, numpy
- **Logging**: structlog

## Development

This project follows the atomic file structure principle from `META/REGULATION.md`:
- Each file has a single, well-defined purpose
- Functions are small and focused
- Complex components are documented with co-located META files

## Documentation

- [PRODUCT.md](META/PRODUCT.md) - Product design and requirements
- [TECHNICAL.md](META/TECHNICAL.md) - Technical architecture and design
- [REGULATION.md](META/REGULATION.md) - Development principles and guidelines

## License

(License information to be added)

## Contributing

(Contributing guidelines to be added)
