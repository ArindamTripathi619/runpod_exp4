#!/bin/bash
# Monitor experiment progress
# Usage: ./monitor_experiments.sh

echo "Monitoring experiment progress..."
echo "================================"
echo ""

# Check if process is running
if pgrep -f "run_experiments.py" > /dev/null; then
    echo "✓ Experiments are running"
else
    echo "✗ Experiments not running"
fi

echo ""
echo "Recent log entries:"
echo "-------------------"
tail -20 experiments.log 2>/dev/null || echo "No log file found yet"

echo ""
echo "Results summary:"
echo "----------------"
# Count completed experiments
exp1_count=$(grep -c "EXPERIMENT 1 COMPLETE" experiments.log 2>/dev/null || echo "0")
exp2_count=$(grep -c "EXPERIMENT 2 COMPLETE" experiments.log 2>/dev/null || echo "0")
exp3_count=$(grep -c "EXPERIMENT 3 COMPLETE" experiments.log 2>/dev/null || echo "0")
exp4_count=$(grep -c "EXPERIMENT 4 COMPLETE" experiments.log 2>/dev/null || echo "0")

echo "Experiment 1 (Layer Propagation): $exp1_count"
echo "Experiment 2 (Trust Boundary): $exp2_count"
echo "Experiment 3 (Coordinated Defense): $exp3_count"
echo "Experiment 4 (Layer Ablation): $exp4_count"

echo ""
echo "To follow logs in real-time, run:"
echo "  tail -f experiments.log"
