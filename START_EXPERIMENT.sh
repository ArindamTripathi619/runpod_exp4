#!/bin/bash
set -e

echo "=========================================="
echo "Starting Experiment in Background"
echo "=========================================="

# Start experiment in background with nohup
echo "Launching experiment..."
nohup ./run_exp*.sh > experiment.log 2>&1 &
EXPERIMENT_PID=$!

echo "✓ Experiment started with PID: $EXPERIMENT_PID"
echo $EXPERIMENT_PID > experiment.pid

sleep 2

echo ""
echo "=========================================="
echo "✅ Experiment Running in Background!"
echo "=========================================="
echo ""
echo "Process ID: $EXPERIMENT_PID"
echo "Log file: experiment.log"
echo ""
echo "You can safely close this terminal."
echo "The experiment will continue running."
echo ""
echo "To monitor progress:"
echo "  tail -f experiment.log"
echo ""
echo "To check if it's running:"
echo "  ps aux | grep python"
echo "  cat experiment.pid"
echo ""
echo "To check trace count:"
echo "  sqlite3 experiments.db \"SELECT COUNT(*) FROM execution_traces\""
echo ""
echo "=========================================="
