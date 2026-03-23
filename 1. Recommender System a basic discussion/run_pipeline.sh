#!/usr/bin/env bash
#
# Full training pipeline — runs all models sequentially.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                          # defaults
#   ./run_pipeline.sh --epochs 100 --lr 0.001  # custom
#
# The script pipes all output to both terminal and a timestamped log file.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "========================================"
echo " Recommender System Training Pipeline"
echo " Log: $LOG_FILE"
echo "========================================"

# Ensure data is in place
if [ ! -f "data/ratings.csv" ]; then
    echo "Symlinking data files to data/ ..."
    mkdir -p data
    # Try local files first, then Movie_Recommender/data/
    for f in movies.csv ratings.csv; do
        if [ -f "$f" ]; then
            ln -sf "$SCRIPT_DIR/$f" "data/$f"
        elif [ -f "Movie_Recommender/data/$f" ]; then
            ln -sf "$SCRIPT_DIR/Movie_Recommender/data/$f" "data/$f"
        else
            echo "ERROR: Cannot find $f"
            exit 1
        fi
    done
fi

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run full pipeline — all models, results piped to log
python -m src.train "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo " Pipeline complete"
echo " Log saved to: $LOG_FILE"
echo " Results JSON: checkpoints/results.json"
echo "========================================"
