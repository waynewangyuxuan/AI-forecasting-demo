# Getting Started with AI Forecasting Pipeline

## 🚀 Quick Start (5 Minutes)

### 1. Set Up Virtual Environment

**macOS/Linux:**
```bash
cd AI-forecasting-demo
./setup_venv.sh
source venv/bin/activate
```

**Windows:**
```cmd
cd AI-forecasting-demo
setup_venv.bat
venv\Scripts\activate
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```bash
GOOGLE_API_KEY=your_key
GOOGLE_CSE_ID=your_cse_id
GOOGLE_GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

### 3. Initialize Database

```bash
python cli.py init
```

### 4. Run Your First Forecast

```bash
python cli.py run "Will Bitcoin reach $100,000 by end of 2025?"
```

### 5. View Results

```bash
python cli.py status 1
python cli.py report 1
```

---

## 📚 Documentation Index

| Document | Purpose | Read When |
|----------|---------|-----------|
| [README.md](README.md) | Project overview | First time |
| [QUICKSTART.md](QUICKSTART.md) | 5-min setup guide | Getting started |
| [VENV_SETUP.md](VENV_SETUP.md) | Virtual env help | Setup issues |
| [USAGE.md](USAGE.md) | Complete CLI reference | Daily use |
| [META/TECHNICAL.md](META/TECHNICAL.md) | Architecture details | Understanding code |
| [META/PRODUCT.md](META/PRODUCT.md) | Product design | Understanding features |
| [META/TODO.md](META/TODO.md) | Next steps | Contributing |
| [META/PROGRESS.md](META/PROGRESS.md) | What's been built | Status check |

---

## 🛠️ Common Commands

```bash
# Run a forecast
python cli.py run "Your question"

# Run with limits (faster)
python cli.py run "Question" --max-urls 5 --max-events 50

# Check status
python cli.py status <run_id>

# List all runs
python cli.py list

# View detailed report
python cli.py report <run_id>

# Resume failed run
python cli.py resume <run_id>

# Get help
python cli.py --help
```

---

## 📁 Project Structure

```
AI-forecasting-demo/
├── cli.py              # CLI entry point
├── pipeline/           # Orchestration
│   └── orchestrator.py # 8-stage pipeline
├── services/           # Core services
│   ├── llm_client.py
│   ├── query_generation.py
│   ├── search.py
│   ├── scraper.py
│   ├── doc_processor.py
│   ├── event_extractor.py
│   ├── embedding.py
│   ├── clustering.py
│   ├── timeline.py
│   └── forecast.py
├── db/                 # Database layer
│   ├── schema.sql
│   ├── models.py
│   ├── repository.py
│   └── migrate.py
├── config/             # Configuration
│   └── settings.py
├── data/               # SQLite database
├── outputs/            # Results
├── .cache/             # HTTP/embedding cache
└── META/               # Project docs
```

---

## 🔑 Where to Get API Keys

### Google Custom Search
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable "Custom Search API"
3. Create credentials → API Key
4. Visit [cse.google.com](https://cse.google.com/) to create Search Engine
5. Copy Search Engine ID

### Google Gemini
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key

---

## 🧪 Testing Your Setup

### Verify Installation
```bash
python verify_setup.py
```

### Test with Sample Question
```bash
# Quick test (5 URLs, 20 events, ~2 minutes)
python cli.py run "Will AI surpass humans in 2030?" --max-urls 5 --max-events 20

# Full run (~10 minutes)
python cli.py run "Will Bitcoin reach $100K by 2025?"
```

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
source venv/bin/activate  # Make sure venv is active
pip install -r requirements.txt
```

### "API key not configured"
```bash
cat .env  # Check your .env file exists and has keys
```

### "Database not found"
```bash
python cli.py init  # Initialize database
```

### Virtual environment issues
See [VENV_SETUP.md](VENV_SETUP.md) for detailed troubleshooting.

---

## 💡 Tips for Best Results

1. **Start with limited scope** for faster testing:
   ```bash
   python cli.py run "Question" --max-urls 5 --max-events 20
   ```

2. **Use specific questions** with clear timeframes:
   - ✅ "Will Bitcoin reach $100K by Dec 31, 2025?"
   - ❌ "Will Bitcoin go up?"

3. **Enable verbose mode** to see detailed progress:
   ```bash
   python cli.py run "Question" --verbose
   ```

4. **Check outputs** folder for detailed results after each run

5. **Monitor API costs** - see [META/TODO.md](META/TODO.md) for cost tracking

---

## 📊 What to Expect

### Runtime (typical question)
- Query generation: 5-10 seconds
- Web search: 30-60 seconds
- Scraping: 2-5 minutes (100 URLs)
- Event extraction: 3-7 minutes
- Clustering: 30-60 seconds
- Timeline building: 10-20 seconds
- Forecast generation: 30-60 seconds
- **Total: 6-15 minutes**

### Cost (per question)
- Google Search: $0 (free tier, 100/day)
- Gemini API: ~$0.10-0.20
- OpenAI Embeddings: ~$0.02
- **Total: ~$0.12-0.22** (well under $0.50 budget)

### Output Files
Each run creates:
- `outputs/run_<id>/forecast_output.json` - Structured data
- `outputs/run_<id>/forecast_report.md` - Human-readable report

---

## 🤝 Getting Help

- **Setup issues**: See [VENV_SETUP.md](VENV_SETUP.md)
- **Usage questions**: See [USAGE.md](USAGE.md)
- **Technical details**: See [META/TECHNICAL.md](META/TECHNICAL.md)
- **Bug reports**: Open an issue (if repository public)

---

## 🎯 Next Steps After Setup

1. **Run all 3 seed questions** from [META/PRODUCT.md](META/PRODUCT.md)
2. **Experiment with parameters** (--max-urls, --max-events)
3. **Test resume functionality** (Ctrl+C during run, then resume)
4. **Review output quality** and adjust prompts if needed
5. **See [META/TODO.md](META/TODO.md)** for future enhancements

---

**Ready to forecast? Run your first question now!**

```bash
python cli.py run "Your forecasting question here"
```

Happy forecasting! 🔮
