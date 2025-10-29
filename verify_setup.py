#!/usr/bin/env python3
"""
Setup Verification Script

This script verifies that the AI Forecasting Pipeline is properly configured.
Run this after completing setup to ensure all dependencies and configurations are correct.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.11 or higher."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (3.11+ required)")
        return False


def check_dependencies():
    """Verify all required dependencies are installed."""
    print("\nChecking dependencies...")
    required = [
        "typer",
        "fastapi",
        "httpx",
        "beautifulsoup4",
        "readability",
        "trafilatura",
        "tenacity",
        "sqlalchemy",
        "pydantic",
        "pydantic_settings",
        "numpy",
        "sklearn",
        "google.generativeai",
        "openai",
        "dotenv",
        "structlog",
    ]

    all_installed = True
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            all_installed = False

    return all_installed


def check_directory_structure():
    """Verify project directory structure."""
    print("\nChecking directory structure...")
    project_root = Path(__file__).parent
    required_dirs = [
        "pipeline",
        "services",
        "db",
        "config",
        "data",
        "outputs",
        ".cache",
    ]

    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            all_exist = False

    return all_exist


def check_env_file():
    """Verify .env file exists."""
    print("\nChecking environment configuration...")
    project_root = Path(__file__).parent
    env_file = project_root / ".env"

    if env_file.exists():
        print("  ✓ .env file exists")
        return True
    else:
        print("  ✗ .env file missing (copy from .env.example)")
        return False


def check_settings():
    """Try to load settings from environment."""
    print("\nChecking settings configuration...")
    try:
        from config.settings import settings

        # Check required API keys
        required_keys = [
            ("google_api_key", "GOOGLE_API_KEY"),
            ("google_cse_id", "GOOGLE_CSE_ID"),
            ("google_gemini_api_key", "GOOGLE_GEMINI_API_KEY"),
            ("openai_api_key", "OPENAI_API_KEY"),
        ]

        all_valid = True
        for attr_name, env_name in required_keys:
            value = getattr(settings, attr_name, None)
            if value and not value.startswith("your_"):
                print(f"  ✓ {env_name} configured")
            else:
                print(f"  ✗ {env_name} not configured")
                all_valid = False

        return all_valid

    except Exception as e:
        print(f"  ✗ Error loading settings: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("AI Forecasting Pipeline - Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Environment File", check_env_file),
        ("Settings Configuration", check_settings),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! Your setup is complete.")
        print("\nNext steps:")
        print("  1. Review META/TECHNICAL.md for architecture")
        print("  2. Start implementing pipeline components")
        print("  3. Run tests as you develop")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nRefer to SETUP.md for detailed setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
