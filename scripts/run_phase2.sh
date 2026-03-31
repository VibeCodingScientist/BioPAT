#!/usr/bin/env bash
# Phase 2: T3 re-runs + context ablation
# Run with: bash scripts/run_phase2.sh
#
# Checkpoint map (old preserved, new created alongside):
#   OLD  t3_*_ctx        → cached, no predicted labels
#   NEW  t3_*_ctx_k10    → fresh, with predicted labels
#   NEW  t3_*_ctx_k1     → ablation
#   NEW  t3_*_ctx_k3     → ablation
#   NEW  t3_*_ctx_k5     → ablation
#   NEW  t3_*_ctx_k20    → ablation
#
# Est. cost: ~$8-10  |  Est. time: 30-60min

set -euo pipefail
cd "$(dirname "$0")/.."

LOG="logs/phase2_$(date +%Y%m%d_%H%M%S).log"

echo "Phase 2 starting — log at $LOG"
echo "Tail with:  tail -f $LOG"

nohup env PYTHONPATH=src /opt/homebrew/bin/python3.12 scripts/run_phase2.py \
    >> "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo "$PID" > logs/phase2.pid
echo "Kill with:  kill $PID"
