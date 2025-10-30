#!/usr/bin/env python3
"""Quick setup checker for AI Forecasting Pipeline."""

import sys

print("=" * 60)
print("AI Forecasting Pipeline - Setup Checker")
print("=" * 60)
print()

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 11):
    print("  ⚠️  WARNING: Python 3.11+ recommended")
print()

# Check required packages
required_packages = [
    "typer",
    "fastapi",
    "httpx",
    "beautifulsoup4",
    "readability",
    "trafilatura",
    "tenacity",
    "sqlalchemy",
    "pydantic",
    "numpy",
    "sklearn",
    "google.generativeai",
    "openai",
    "structlog",
    "rich",
]

missing = []
installed = []

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        installed.append(package)
    except ImportError:
        missing.append(package)

print(f"Packages installed: {len(installed)}/{len(required_packages)}")
print()

if installed:
    print("✓ Installed:")
    for pkg in installed:
        print(f"  - {pkg}")
    print()

if missing:
    print("✗ Missing:")
    for pkg in missing:
        print(f"  - {pkg}")
    print()
    print("To install missing packages:")
    print("  pip install -r requirements.txt")
else:
    print("✓ All required packages installed!")

print()
print("=" * 60)
