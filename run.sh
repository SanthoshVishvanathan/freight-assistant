#!/usr/bin/env bash
# FreightTiger Smart Assistant Runner (Linux/macOS)
set -e

echo "============================================================"
echo "Running FreightTiger Smart Assistant Pipeline..."
echo "============================================================"
python3 run_assistant.py run

echo ""
echo "============================================================"
echo "Running Automated Evaluation Harness..."
echo "============================================================"
python3 eval_harness.py

echo ""
echo "[SUCCESS] All tasks completed successfully!"
