# Quick Start Guide

Get up and running with the AI Forecasting Pipeline in 5 minutes.

## Prerequisites

- Python 3.11+
- API keys for:
  - Google Custom Search
  - Google Gemini
  - OpenAI

## Setup (3 minutes)

### Option 1: Automated Setup (Recommended)

**macOS/Linux:**
```bash
cd AI-forecasting-demo
./setup_venv.sh
```

**Windows:**
```cmd
cd AI-forecasting-demo
setup_venv.bat
```

Then configure and initialize:
```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# Initialize database
python cli.py init
```

### Option 2: Manual Setup

```bash
# 1. Clone and enter directory
cd AI-forecasting-demo

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# 5. Initialize database
python cli.py init
```

📖 **Having issues?** See [VENV_SETUP.md](VENV_SETUP.md) for detailed troubleshooting.

## Run Your First Forecast (2 minutes)

```bash
python cli.py run "Will Bitcoin reach $50,000 by June 2024?"
```

That's it! The pipeline will:
1. Generate search queries (10 sec)
2. Search the web (30 sec)
3. Scrape content (60 sec)
4. Extract events (2 min)
5. Cluster and deduplicate (30 sec)
6. Build timeline (10 sec)
7. Generate forecast (20 sec)

## View Results

```bash
# Check status
python cli.py status 1

# View report
python cli.py report 1

# List all runs
python cli.py list
```

## Output Files

Find your results in:
- `outputs/run_1/forecast_report.md` - Human-readable report
- `outputs/run_1/forecast_output.json` - Structured data

## Common Commands

```bash
# Basic run
python cli.py run "Your question here"

# Limited data (faster, cheaper)
python cli.py run "Question" --max-urls 5 --max-events 20

# Resume failed run
python cli.py resume 1

# Check progress
python cli.py status 1

# See all runs
python cli.py list
```

## Next Steps

- Read [USAGE.md](USAGE.md) for comprehensive documentation
- See [TECHNICAL.md](META/TECHNICAL.md) for architecture details
- Check [examples/](examples/) for sample forecasts

## Troubleshooting

**"Module not found" error?**
```bash
pip install -r requirements.txt
```

**"API key not found" error?**
```bash
# Check your .env file has all required keys
cat .env
```

**Run failed mid-way?**
```bash
# Resume from last checkpoint
python cli.py resume <run_id>
```

## Help

```bash
# Get help on any command
python cli.py --help
python cli.py run --help
python cli.py status --help
```

---

Need more details? See [USAGE.md](USAGE.md) for complete documentation.
