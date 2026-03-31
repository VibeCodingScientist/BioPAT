#!/bin/bash
# VPS Runner — NovEx 3-Tier Evaluation (300 statements x 3 LLMs)
# ===============================================================
# Runs the full evaluation: BM25, rerank, agent, relevance, novelty
# Estimated: ~$120 cost, ~8-12 hours runtime
#
# Usage (on VPS):
#   nohup bash scripts/vps_run_novex.sh >> ~/novex_eval.log 2>&1 &
#   tail -f ~/novex_eval.log

set -euo pipefail

PROJECT_DIR="$HOME/BioPAT"
cd "$PROJECT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
die() { log "FATAL: $1"; exit 1; }

# ============================================================
# PHASE 0: Environment setup
# ============================================================
log "=== PHASE 0: Environment setup ==="

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
    log "Loaded .env"
else
    die "No .env file found"
fi

[ -n "${OPENAI_API_KEY:-}" ]    || die "OPENAI_API_KEY not set"
[ -n "${ANTHROPIC_API_KEY:-}" ] || die "ANTHROPIC_API_KEY not set"
[ -n "${GOOGLE_API_KEY:-}" ]    || die "GOOGLE_API_KEY not set"
log "API keys verified"

if [ ! -d "$PROJECT_DIR/venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/venv"
fi
source "$PROJECT_DIR/venv/bin/activate"

log "Installing dependencies..."
pip install --upgrade pip wheel setuptools -q
pip install -e ".[dev,llm]" -q 2>&1 | tail -5
log "Dependencies installed"

python -c "from biopat.novex.evaluator import NovExEvaluator; print('Import check OK')" \
    || die "Import failed"

# ============================================================
# PHASE 1: Preflight checks
# ============================================================
log "=== PHASE 1: Preflight checks ==="

python -c "
import json, sys
from biopat.novex.benchmark import NovExBenchmark

b = NovExBenchmark('data/novex', corpus_dir='data/benchmark')
b.load()

errors = []

# Check statement count
if len(b.statements) != 300:
    errors.append(f'Expected 300 statements, got {len(b.statements)}')

# Check corpus size (dual_corpus should be ~164K)
if len(b.corpus) < 160000:
    errors.append(f'Corpus too small: {len(b.corpus)} (expected ~164K)')

# Check tier1 qrels exist
t1_pairs = sum(len(v) for v in b.tier1_qrels.values())
if t1_pairs == 0:
    errors.append('No tier1 qrels loaded')

# Check tier2 qrels populated from tier1
t2_pairs = sum(len(v) for v in b.tier2_qrels.values())
if t2_pairs == 0:
    errors.append('No tier2 qrels (fallback from tier1 failed)')

# Check tier3 labels
if len(b.tier3_labels) < 290:
    errors.append(f'Only {len(b.tier3_labels)} tier3 labels (expected ~300)')

# Check no missing docs in tier1 qrels
missing = 0
for qid, docs in b.tier1_qrels.items():
    for did in docs:
        if did not in b.corpus:
            missing += 1
if missing > 0:
    errors.append(f'{missing} tier1 qrel docs missing from corpus')

# Check no stale NX-* IDs
nx_ids = [qid for qid in b.queries if qid.startswith('NX')]
if nx_ids:
    errors.append(f'{len(nx_ids)} stale NX-* query IDs found')

if errors:
    print('PREFLIGHT FAILED:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)

print(f'Preflight OK: {len(b.statements)} stmts, {len(b.corpus)} docs, '
      f'T1={t1_pairs} T2={t2_pairs} T3={len(b.tier3_labels)}, '
      f'missing_docs={missing}')
" || die "Preflight failed"

# ============================================================
# PHASE 2: Clear stale checkpoints
# ============================================================
log "=== PHASE 2: Clearing stale checkpoints ==="

CKPT_DIR="data/novex/results/checkpoints"
if [ -d "$CKPT_DIR" ] && [ "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
    STALE_COUNT=$(ls "$CKPT_DIR" | wc -l)
    log "Removing $STALE_COUNT stale checkpoints from $CKPT_DIR"
    rm -f "$CKPT_DIR"/*.json
else
    log "No stale checkpoints to clear"
fi

# Also clear stale all_results.json
rm -f data/novex/results/all_results.json
rm -f data/novex/results/costs.json
log "Stale results cleared"

# ============================================================
# PHASE 3: Run evaluation
# ============================================================
log "=== PHASE 3: Running full NovEx evaluation ==="
log "Expected: 16 checkpoints (1 BM25 + 6 rerank/agent + 3 tier2 + 6 tier3)"
log "Estimated runtime: 8-12 hours, ~\$120"

python -c "
import logging, yaml
from biopat.novex.benchmark import NovExBenchmark
from biopat.novex.evaluator import NovExEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

with open('configs/novex.yaml') as f:
    cfg = yaml.safe_load(f)

b = NovExBenchmark('data/novex', corpus_dir='data/benchmark')
b.load()

ev = NovExEvaluator(b, results_dir='data/novex/results', budget_usd=cfg['evaluation']['budget_usd'])

results = ev.run_all(cfg['evaluation'])

print(f'Completed {len(results)} evaluations')
for r in results:
    top_metric = next(iter(r.metrics.items())) if r.metrics else ('N/A', 0)
    print(f'  {r.tier}/{r.method}/{r.model}: {top_metric[0]}={top_metric[1]:.4f} (\${r.cost_usd:.2f})')
print(f'Total cost: \${ev.cost_tracker.total_cost:.2f}')
"

log "Evaluation complete"

# ============================================================
# PHASE 4: Verification
# ============================================================
log "=== PHASE 4: Verification ==="

python -c "
import json, sys

with open('data/novex/results/all_results.json') as f:
    results = json.load(f)

errors = []

# Check expected count
if len(results) != 16:
    errors.append(f'Expected 16 results, got {len(results)}')

# Check no zero-metric results
for r in results:
    key = f\"t{r['tier']}/{r['method']}/{r['model']}\"
    if not r['metrics']:
        errors.append(f'{key}: empty metrics')
    elif all(v == 0 for v in r['metrics'].values() if isinstance(v, (int, float))):
        errors.append(f'{key}: all metrics are zero')

# Check BM25 baseline exists
bm25 = [r for r in results if r['method'] == 'bm25']
if not bm25:
    errors.append('BM25 baseline missing')

# Check tier2 has pairs
t2 = [r for r in results if r['tier'] == 2]
for r in t2:
    pairs = r['metrics'].get('num_pairs', 0)
    if pairs == 0:
        errors.append(f\"t2/{r['model']}: num_pairs=0\")

# Check no stale NX-* IDs in per_query
for r in results:
    nx = [q for q in r.get('per_query', {}) if q.startswith('NX')]
    if nx:
        errors.append(f\"t{r['tier']}/{r['method']}/{r['model']}: {len(nx)} stale NX-* IDs\")

if errors:
    print('VERIFICATION FAILED:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)

print(f'Verification OK: {len(results)} results, all metrics non-zero')

# Print summary
print()
print('=== Results Summary ===')
for r in results:
    key = f\"t{r['tier']}/{r['method']}/{r['model']}\"
    top3 = list(r['metrics'].items())[:3]
    metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in top3)
    print(f'  {key}: {metrics_str} (\${r.get(\"cost_usd\", 0):.2f})')
" || log "WARNING: Verification had issues — check results manually"

log "=== ALL PHASES COMPLETE ==="
log "Results: data/novex/results/all_results.json"
log "Costs:   data/novex/results/costs.json"
log "Checkpoints: data/novex/results/checkpoints/"
