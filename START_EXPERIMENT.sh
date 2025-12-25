#!/bin/bash
#############################################################################
# START_EXPERIMENT.sh - Experiment 4 Setup and Execution
# This script sets up the environment and runs Experiment 4 automatically
#############################################################################

set -e  # Exit on any error

echo "=========================================="
echo "🚀 Starting Experiment 4 Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📍 Working directory: $SCRIPT_DIR"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✅ Python version: $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Check if Ollama is accessible
echo "🔍 Checking Ollama connectivity..."
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "✅ Ollama is accessible at $OLLAMA_HOST"
else
    echo "⚠️  Warning: Ollama not accessible at $OLLAMA_HOST"
    echo "   Please ensure Ollama is running or set OLLAMA_HOST environment variable"
    echo "   Example: export OLLAMA_HOST=http://your-ollama-host:11434"
fi
echo ""

# Set experiment parameters
export TRIALS="${TRIALS:-5}"
export DB_PATH="${DB_PATH:-exp4_results.db}"
export LOG_FILE="${LOG_FILE:-exp4_execution.log}"
export OUTPUT_DIR="${OUTPUT_DIR:-exp4_results}"

echo "=========================================="
echo "⚙️  Experiment 4 Configuration"
echo "=========================================="
echo "Trials:          $TRIALS"
echo "Database:        $DB_PATH"
echo "Log file:        $LOG_FILE"
echo "Output dir:      $OUTPUT_DIR"
echo "Ollama host:     $OLLAMA_HOST"
echo ""

# Run the experiment
echo "=========================================="
echo "🏃 Running Experiment 4"
echo "=========================================="
echo ""
echo "This will run experiment 4 with $TRIALS trials..."
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Execute the experiment wrapper script
./run_exp4.sh

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Experiment 4 Completed Successfully!"
    echo "=========================================="
    echo ""
    echo "📊 Results saved to:"
    echo "   - Database: $DB_PATH"
    echo "   - Log file: $LOG_FILE"
    echo "   - Output:   $OUTPUT_DIR/"
    echo ""
    echo "📈 To view results:"
    echo "   sqlite3 $DB_PATH 'SELECT COUNT(*) FROM execution_traces;'"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Experiment 4 Failed"
    echo "=========================================="
    echo ""
    echo "Check the log file for details:"
    echo "   tail -f $LOG_FILE"
    echo ""
    exit 1
fi
