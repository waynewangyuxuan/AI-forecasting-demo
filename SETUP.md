# Setup Guide

This guide walks you through setting up the AI Forecasting Pipeline for development.

## Prerequisites

1. **Python 3.11 or higher**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **API Keys** - You'll need accounts and API keys for:
   - **Google Cloud Platform**: For Custom Search API and Gemini API
   - **OpenAI**: For embeddings

## Step-by-Step Setup

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd AI-forecasting-demo
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API Keys

#### Get Google Custom Search Credentials

1. **Google API Key**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable "Custom Search API"
   - Create credentials (API Key)

2. **Custom Search Engine ID**:
   - Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Create a new search engine
   - Configure to "Search the entire web"
   - Copy the Search Engine ID (cx parameter)

#### Get Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create API key
3. Copy the key

#### Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create new secret key
3. Copy the key

### 5. Set Up Environment File

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your favorite editor
nano .env  # or vim, code, etc.
```

Add your actual API keys:

```bash
GOOGLE_API_KEY=AIza...your-key-here
GOOGLE_CSE_ID=abc123...your-cse-id
GOOGLE_GEMINI_API_KEY=AIza...your-gemini-key
OPENAI_API_KEY=sk-...your-openai-key
```

### 6. Verify Setup

Test that settings load correctly:

```bash
python -c "from config.settings import settings; print('Settings loaded successfully!')"
```

If you see "Settings loaded successfully!" - you're all set!

### 7. Initialize Database

(Database initialization commands will be added once the schema is implemented)

## Directory Structure

After setup, your project structure should look like:

```
AI-forecasting-demo/
├── .venv/              # Virtual environment (created by you)
├── .env                # Your API keys (created by you, not in git)
├── .cache/             # HTTP cache (auto-created)
├── data/               # Database files (auto-created)
│   └── forecast.db    # SQLite database (auto-created on first run)
├── outputs/            # Forecast results (auto-created)
├── config/             # Configuration
├── pipeline/           # Orchestration
├── services/           # Service modules
├── db/                 # Database layer
├── META/               # Documentation
└── ...
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. All dependencies installed: `pip install -r requirements.txt`
3. You're running from project root

### API Key Errors

If you see validation errors about API keys:
1. Check `.env` file exists and has correct keys
2. Ensure no extra spaces around `=` signs
3. Keys should not be in quotes

### Database Errors

If database errors occur:
1. Ensure `data/` directory exists
2. Check write permissions on `data/` directory
3. Delete `data/forecast.db` and reinitialize if corrupted

## Next Steps

Once setup is complete:
1. Review [TECHNICAL.md](META/TECHNICAL.md) for architecture details
2. Check [PRODUCT.md](META/PRODUCT.md) for feature specifications
3. Start implementing pipeline components

## Development Tips

- Keep virtual environment activated when working
- Never commit `.env` file (it's in `.gitignore`)
- Use `python -m module_name` to run modules as scripts
- Check logs in debug mode: Set `DEBUG=true` in `.env`
