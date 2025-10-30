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

#### Option 1: Automated Setup (Recommended)

**macOS/Linux:**
```bash
git clone <repository-url>
cd AI-forecasting-demo
chmod +x setup_venv.sh
./setup_venv.sh
```

**Windows:**
```cmd
git clone <repository-url>
cd AI-forecasting-demo
setup_venv.bat
```

The setup script will:
- Check Python version (3.11+ required)
- Create virtual environment in `./venv`
- Install all dependencies from requirements.txt
- Provide next steps guidance

#### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-forecasting-demo
```

2. Create and activate a virtual environment:
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

📖 **Detailed virtual environment guide**: See [VENV_SETUP.md](VENV_SETUP.md) for troubleshooting and IDE integration.

### Configuration

Edit the `.env` file with your API credentials:

```bash
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

### First Run

1. **Activate the virtual environment** (if not already active):
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

2. **Initialize the database**:
```bash
python cli.py init
```

3. **Run your first forecast**:
```bash
python cli.py run "Will China mass-produce humanoid robots by the end of 2025?"
```

4. **Check the results**:
```bash
python cli.py status 1
python cli.py report 1
```

### Common Commands

```bash
# Run a forecast
python cli.py run "Your forecasting question"

# Resume a failed run
python cli.py resume <run_id>

# Check run status
python cli.py status <run_id>

# List all runs
python cli.py list

# Generate detailed report
python cli.py report <run_id>

# Show help
python cli.py --help
```

📖 **Full usage guide**: See [USAGE.md](USAGE.md) for complete CLI reference and examples.

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
