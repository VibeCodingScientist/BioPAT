#!/bin/bash
# VPS Runner — Fix outliers, run novelty, scale to 300
# =====================================================
# Run via nohup so SSH disconnects don't kill it.
#
# Usage (on VPS):
#   nohup bash scripts/vps_run_all.sh >> ~/vps_run_all.log 2>&1 &
#   tail -f ~/vps_run_all.log

set -euo pipefail

PROJECT_DIR="$HOME/BioPAT"
BACKUP_DIR="$HOME/backups/novex_$(date +%Y%m%d_%H%M%S)"

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

python -c "from biopat.novex.annotation import AnnotationProtocol; print('Import check OK')" || die "Import failed"

# ============================================================
# PHASE 1: Backup
# ============================================================
log "=== PHASE 1: Creating backups ==="
mkdir -p "$BACKUP_DIR/qrels"
cp data/novex/reverse/step7_selected.json  "$BACKUP_DIR/"
cp data/novex/reverse/step8_novelty.json  "$BACKUP_DIR/"
cp data/novex/statements.jsonl             "$BACKUP_DIR/"
cp data/novex/queries.jsonl                "$BACKUP_DIR/"
cp data/novex/qrels/tier1.tsv             "$BACKUP_DIR/qrels/"
cp data/novex/qrels/tier3.tsv             "$BACKUP_DIR/qrels/"
log "Backup saved to $BACKUP_DIR"

# ============================================================
# PHASE 2: Fix outliers
# ============================================================
log "=== PHASE 2: Fix outliers ==="
python scripts/fix_outliers.py --apply --queue
log "Outliers fixed"

# ============================================================
# PHASE 3: Novelty for 3 replacement patents (~$0.30, ~5 min)
# ============================================================
log "=== PHASE 3: Novelty for 3 outlier replacements ==="
python scripts/run_novelty_batch.py data/novex/reverse/novelty_queue_outliers.json
log "Phase 3 complete"

# ============================================================
# PHASE 4: Scale to 300
# ============================================================
log "=== PHASE 4: Scale benchmark to 300 ==="
python scripts/scale_benchmark.py --apply
log "Scaled to 300"

# ============================================================
# PHASE 5: Novelty for 200 new patents (~$6, ~6-8 hours)
# ============================================================
log "=== PHASE 5: Novelty for 200 new patents ==="
python scripts/run_novelty_batch.py data/novex/reverse/novelty_queue_scale.json
log "Phase 5 complete"

# ============================================================
# PHASE 6: Final verification
# ============================================================
log "=== PHASE 6: Final verification ==="
python -c "
import json
from collections import Counter

with open('data/novex/statements.jsonl') as f:
    stmts = [json.loads(l) for l in f]
with open('data/novex/queries.jsonl') as f:
    queries = [json.loads(l) for l in f]
with open('data/novex/qrels/tier3.tsv') as f:
    t3 = [l for l in f if not l.startswith('query_id') and l.strip()]

cats = Counter(s['category'] for s in stmts)
doms = Counter(s['domain'] for s in stmts)
novelty = Counter(s['ground_truth']['tier3_novelty_label'] for s in stmts)
pending = novelty.get('PENDING', 0)
pids = [s['source_patent_id'] for s in stmts]

print(f'Statements: {len(stmts)}')
print(f'Queries: {len(queries)}')
print(f'Tier3 entries: {len(t3)}')
print(f'Categories: {dict(cats)}')
print(f'Domains: {dict(doms)}')
print(f'Novelty: {dict(novelty)}')
print(f'Pending: {pending}')
print(f'Unique patents: {len(set(pids))}/{len(pids)}')

assert len(stmts) == 300, f'Expected 300 statements, got {len(stmts)}'
assert len(queries) == 300, f'Expected 300 queries, got {len(queries)}'
assert len(set(pids)) == len(pids), 'Duplicate patent IDs!'
assert pending == 0, f'{pending} still pending!'
print('ALL VERIFICATIONS PASSED')
"

log "=== ALL PHASES COMPLETE ==="
log "Backup at: $BACKUP_DIR"
