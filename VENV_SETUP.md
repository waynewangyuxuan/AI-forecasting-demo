# Virtual Environment Setup Guide

This guide explains how to set up and use a Python virtual environment for the AI Forecasting Pipeline project.

## Why Use a Virtual Environment?

A virtual environment provides:
- **Isolated dependencies**: Project dependencies don't conflict with system Python packages
- **Reproducible environment**: Ensures everyone uses the same package versions
- **Clean installation**: Easy to remove or recreate without affecting other projects
- **Version control**: Requirements.txt tracks exact versions needed

## Quick Setup

### macOS/Linux

```bash
# Run the setup script
chmod +x setup_venv.sh
./setup_venv.sh

# Activate the environment
source venv/bin/activate
```

### Windows

```cmd
# Run the setup script
setup_venv.bat

# Or activate manually
venv\Scripts\activate
```

## Manual Setup (Alternative)

If you prefer to set up manually:

### Step 1: Create Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv

# Windows
python -m venv venv
```

### Step 2: Activate Virtual Environment

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

You should see `(venv)` prefix in your terminal prompt.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- typer (CLI framework)
- fastapi, uvicorn (REST API)
- httpx (async HTTP)
- beautifulsoup4, readability-lxml, trafilatura (web scraping)
- robotexclusionrulesparser (robots.txt)
- tenacity (retry logic)
- sqlalchemy (database)
- pydantic, pydantic-settings (data models)
- numpy, scikit-learn (clustering)
- google-generativeai (Gemini API)
- openai (embeddings)
- python-dotenv (environment variables)
- structlog (logging)
- rich (terminal formatting)
- langdetect (language detection)

## Using the Virtual Environment

### Activating

Before working on the project, always activate the virtual environment:

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Running Commands

With the environment activated, run all Python commands normally:

```bash
# Initialize database
python cli.py init

# Run a forecast
python cli.py run "Your question here"

# Check status
python cli.py status 1
```

### Deactivating

When you're done working:

```bash
deactivate
```

## Verifying Installation

Check that everything is installed correctly:

```bash
# Activate venv first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run verification
python verify_setup.py
```

This will check:
- Python version (3.11+)
- All required packages installed
- Directory structure
- Configuration files

## Troubleshooting

### "python3: command not found" (macOS/Linux)

Try using `python` instead of `python3`:
```bash
python -m venv venv
```

### "permission denied" when running setup script

Make the script executable:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### Package installation fails

1. Make sure you're using Python 3.11+:
   ```bash
   python --version
   ```

2. Try upgrading pip:
   ```bash
   pip install --upgrade pip
   ```

3. Install packages one at a time to identify the problematic package:
   ```bash
   pip install typer
   pip install fastapi
   # ... etc
   ```

### Virtual environment not activating

**macOS/Linux:**
- Make sure you're using `source` command: `source venv/bin/activate`
- Check your shell (bash/zsh) and use appropriate activation script

**Windows:**
- If using PowerShell, you may need to change execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- Then use: `venv\Scripts\Activate.ps1`

### "ModuleNotFoundError" when running scripts

Make sure:
1. Virtual environment is activated (you see `(venv)` in prompt)
2. Dependencies are installed: `pip install -r requirements.txt`
3. You're in the project root directory

## Development Workflow

Typical development session:

```bash
# 1. Navigate to project
cd AI-forecasting-demo

# 2. Activate virtual environment
source venv/bin/activate

# 3. Work on the project
python cli.py run "Question"
# ... edit code, test, etc ...

# 4. If you add new dependencies
pip install <new-package>
pip freeze > requirements.txt  # Update requirements

# 5. Deactivate when done
deactivate
```

## IDE Integration

### VS Code

VS Code should automatically detect the virtual environment. If not:

1. Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `./venv/bin/python`

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select "Existing environment"
4. Choose `./venv/bin/python`

### Cursor/Claude Code

The virtual environment should be detected automatically. Run commands in the integrated terminal with venv activated.

## Best Practices

1. **Always activate before working**: `source venv/bin/activate`
2. **Don't commit venv/**: Already in .gitignore
3. **Update requirements.txt**: After installing new packages
4. **Recreate if corrupted**: Delete `venv/` and run setup script again
5. **Use same Python version**: All team members should use Python 3.11+

## Adding New Dependencies

When you need a new package:

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Install the package
pip install <package-name>

# 3. Update requirements.txt
pip freeze > requirements.txt

# 4. Commit the updated requirements.txt
git add requirements.txt
git commit -m "Add <package-name> dependency"
```

## Removing the Virtual Environment

If you need to start fresh:

```bash
# Deactivate if active
deactivate

# Remove the directory
rm -rf venv

# Recreate
./setup_venv.sh
```

## GitHub Actions / CI/CD

For automated testing, use this workflow:

```yaml
steps:
  - uses: actions/checkout@v3
  - uses: actions/setup-python@v4
    with:
      python-version: '3.11'
  - name: Install dependencies
    run: |
      python -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
  - name: Run tests
    run: |
      source venv/bin/activate
      pytest
```

## Additional Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [pip documentation](https://pip.pypa.io/en/stable/)
- [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/)

---

**Questions?** Check the main [README.md](README.md) or open an issue.
