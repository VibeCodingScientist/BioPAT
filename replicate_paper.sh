#!/bin/bash
# Replicate all key BioPAT experiments end-to-end.
#
# Prerequisites:
#   - Python 3.11+
#   - API keys exported in environment (only what you need):
#       OPENAI_API_KEY        (for primary GPT-5.2 evaluation)
#       ANTHROPIC_API_KEY     (for Claude Sonnet 4.6 + Haiku 4.5)
#       GOOGLE_API_KEY        (for Gemini 3 Pro)
#       OPENROUTER_API_KEY    (for hold-out models: Llama, DeepSeek, Qwen, Mistral)
#
# Usage:
#   ./replicate_paper.sh                  # run everything
#   ./replicate_paper.sh tier1            # just Tier 1
#   ./replicate_paper.sh holdout          # just hold-out evaluation
#
# All experiments are resumable; re-running picks up from checkpoints.

set -e
cd "$(dirname "$0")"

STAGE="${1:-all}"

# -------- 0. Environment setup (idempotent) -------------------------------
if [ ! -d venv ]; then
    echo ">>> Creating virtualenv..."
    python -m venv venv
fi
source venv/bin/activate

if ! python -c "import biopat" 2>/dev/null; then
    echo ">>> Installing dependencies..."
    pip install -e ".[evaluation]" -q
fi

# -------- 1. Download benchmark data --------------------------------------
if [ ! -f "data/benchmark/corpus.jsonl" ]; then
    echo ">>> Downloading corpus from GitHub Release..."
    mkdir -p data
    gh release download v1.0.0 --repo VibeCodingScientist/BioPAT \
        --pattern "biopat-benchmark.tar.gz"
    tar xzf biopat-benchmark.tar.gz
    rm biopat-benchmark.tar.gz
fi

# -------- 2. Tier 1: BM25, dense, agent retrieval -------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "tier1" ]; then
    echo ""
    echo "=================================================="
    echo "TIER 1: Document retrieval"
    echo "=================================================="

    # BM25 baseline (fast, no API key needed)
    PYTHONPATH=src python scripts/run_novex.py --tier 1 --method bm25

    # Dense retrieval (BGE + SPECTER2) — local encoding, ~90 min on M1
    PYTHONPATH=src python scripts/run_dense_experiments.py \
        --models "BAAI/bge-base-en-v1.5" "allenai/specter2_base" \
        --batch-size 32

    # LLM rerank (requires API keys)
    if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ]; then
        PYTHONPATH=src python scripts/run_novex.py --tier 1 --method rerank
    fi

    # Agent retrieval (requires API keys)
    if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ]; then
        PYTHONPATH=src python scripts/run_novex.py --tier 1 --method agent
    fi
fi

# -------- 3. Tier 2: Relevance grading ------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "tier2" ]; then
    echo ""
    echo "=================================================="
    echo "TIER 2: Relevance grading"
    echo "=================================================="
    PYTHONPATH=src python scripts/run_novex.py --tier 2
fi

# -------- 4. Tier 3: Novelty determination --------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "tier3" ]; then
    echo ""
    echo "=================================================="
    echo "TIER 3: Novelty determination (3 main models)"
    echo "=================================================="
    PYTHONPATH=src python scripts/run_novex.py --tier 3 --context-k 10
    # Zero-shot ablation
    PYTHONPATH=src python scripts/run_novex.py --tier 3 --mode zero_shot
    # Context k ablation (k=1,3,5,20)
    for k in 1 3 5 20; do
        PYTHONPATH=src python scripts/run_novex.py --tier 3 --context-k $k
    done
fi

# -------- 5. WP1: Hold-out models via OpenRouter --------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "holdout" ]; then
    echo ""
    echo "=================================================="
    echo "WP1: Hold-out evaluation (out-of-family models)"
    echo "=================================================="
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo "WARNING: OPENROUTER_API_KEY not set — skipping hold-out evaluation"
    else
        for model_provider in \
            "meta-llama/llama-3.3-70b-instruct,Novita" \
            "deepseek/deepseek-chat,Novita" \
            "qwen/qwen-2.5-72b-instruct,DeepInfra" \
            "mistralai/mistral-large,Mistral"
        do
            IFS=',' read -r model provider <<< "$model_provider"
            for mode in ctx_k10_cot ctx_k10_nocot; do
                python scripts/run_holdout.py \
                    --model "$model" \
                    --provider "$provider" \
                    --mode "$mode" \
                    --output-dir data/novex/results/holdout
            done
        done
    fi
fi

# -------- 6. Analysis -----------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "analysis" ]; then
    echo ""
    echo "=================================================="
    echo "ANALYSIS: Statistics, agreement, figures"
    echo "=================================================="
    PYTHONPATH=src python scripts/analyze_novex.py --analysis all
    PYTHONPATH=src python scripts/generate_figures.py --format pdf
fi

echo ""
echo "=================================================="
echo "DONE. Results in data/novex/results/"
echo "=================================================="
