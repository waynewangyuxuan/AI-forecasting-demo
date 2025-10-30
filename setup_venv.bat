@echo off
REM Virtual Environment Setup Script for AI Forecasting Pipeline (Windows)
REM This script creates a virtual environment and installs all dependencies

echo ==========================================
echo AI Forecasting Pipeline - Virtual Environment Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)
echo.

REM Create virtual environment
set VENV_DIR=venv
if exist "%VENV_DIR%" (
    echo Warning: Virtual environment already exists at .\%VENV_DIR%
    set /p RECREATE="Do you want to remove and recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Using existing virtual environment
    )
)

if not exist "%VENV_DIR%" (
    echo Creating virtual environment in .\%VENV_DIR%...
    python -m venv "%VENV_DIR%"
    echo Virtual environment created
) else (
    echo Using existing virtual environment
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo pip upgraded to latest version
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
echo This may take a few minutes...
echo.
pip install -r requirements.txt

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Virtual environment is ready at: .\%VENV_DIR%
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate
echo.
echo To deactivate when done:
echo   deactivate
echo.
echo Next steps:
echo   1. Copy .env.example to .env and add your API keys
echo   2. Initialize the database: python cli.py init
echo   3. Run your first forecast: python cli.py run "Your question"
echo.
pause
