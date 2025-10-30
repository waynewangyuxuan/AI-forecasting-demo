#!/bin/bash
# Virtual Environment Setup Script for AI Forecasting Pipeline
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "=========================================="
echo "AI Forecasting Pipeline - Virtual Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $PYTHON_VERSION"

# Verify Python 3.11+
REQUIRED_VERSION="3.11"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Error: Python 3.11 or higher is required"
    echo "   Current version: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python version check passed"
echo ""

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at ./$VENV_DIR"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
else
    echo "✓ Using existing virtual environment"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded to latest version"
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
echo "This may take a few minutes..."
echo ""
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment is ready at: ./$VENV_DIR"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and add your API keys"
echo "  2. Initialize the database: python cli.py init"
echo "  3. Run your first forecast: python cli.py run \"Your question\""
echo ""
