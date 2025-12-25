#!/bin/bash

# Setup script for Prompt Injection Defense Experiments
# This script automates the installation and verification process

set -e  # Exit on error

echo "=================================="
echo "Prompt Injection Defense Setup"
echo "=================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "⚠️  Warning: Python 3.11+ recommended, you have $python_version"
fi

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install Python dependencies
echo ""
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✓ Python dependencies installed"

# Check Ollama installation
echo ""
echo "[5/6] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if Ollama service is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama service is running"
        
        # Check for required models
        echo ""
        echo "Checking for LLM models..."
        if ollama list | grep -q "llama3"; then
            echo "✓ llama3 model found"
        else
            echo "⚠️  llama3 model not found"
            read -p "Download llama3 now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                ollama pull llama3
            fi
        fi
        
        if ollama list | grep -q "mistral"; then
            echo "✓ mistral model found"
        else
            echo "⚠️  mistral model not found"
            read -p "Download mistral now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                ollama pull mistral
            fi
        fi
    else
        echo "⚠️  Ollama service not running"
        echo "Please start Ollama: ollama serve"
    fi
else
    echo "❌ Ollama not found"
    echo "Please install Ollama from: https://ollama.ai/download"
    echo "Then run this setup script again"
fi

# Verify installation
echo ""
echo "[6/6] Verifying installation..."
python3 << 'EOF'
try:
    import fastapi
    import pydantic
    import sentence_transformers
    import pandas
    import torch
    import ollama
    print("✓ All Python packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
EOF

# Create necessary directories
mkdir -p data/datasets
mkdir -p results/figures
mkdir -p notebooks/analysis

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Ensure Ollama is running: ollama serve"
echo "2. Download models: ollama pull llama3"
echo "3. Run tests: python test_setup.py"
echo "4. Start experiments: python run_experiment1.py"
echo ""
echo "To activate the environment later:"
echo "  source venv/bin/activate"
echo ""
