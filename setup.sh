#!/bin/bash
set -e

echo "=========================================="
echo "RunPod Experiment Setup"
echo "=========================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
    echo "✓ Ollama started"
else
    echo "✓ Ollama already running"
fi

# Check if llama3 model exists
if ! ollama list | grep -q "llama3"; then
    echo "Pulling llama3 model (this takes 5-10 minutes)..."
    ollama pull llama3
    echo "✓ llama3 model downloaded"
else
    echo "✓ llama3 model already available"
fi

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✓ Python packages installed"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the experiment, run:"
echo "  ./START_EXPERIMENT.sh"
echo ""
echo "To monitor progress:"
echo "  tail -f experiment.log"
echo ""
echo "=========================================="
