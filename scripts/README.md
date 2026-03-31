# BioPAT Benchmark Scripts

Quick reference for running BioPAT on a VPS.

## Quick Start

```bash
# 1. Clone the repository on your VPS
git clone <your-repo-url> BioPAT
cd BioPAT

# 2. Set up environment variables
export PATENTSVIEW_API_KEYS="key1,key2,key3"  # Optional but recommended
export OPENALEX_MAILTO="your@email.com"        # Recommended

# 3. Run the deployment script
chmod +x scripts/vps_deploy.sh
./scripts/vps_deploy.sh
```

## Scripts

### `vps_deploy.sh`

Full deployment script that:
- Installs system dependencies (Ubuntu/Debian/CentOS/macOS)
- Creates Python virtual environment
- Installs project dependencies
- Creates default configuration
- Runs the pipeline
- Generates a summary report

**Usage:**
```bash
# Full setup and run
./scripts/vps_deploy.sh

# Skip system setup (if already done)
./scripts/vps_deploy.sh --skip-setup

# Skip baseline evaluation (faster)
./scripts/vps_deploy.sh --skip-baseline

# Run a specific phase
./scripts/vps_deploy.sh --phase phase2
```

### NovEx Scripts

```bash
# Run NovEx 3-tier evaluation
PYTHONPATH=src python scripts/run_novex.py --tier 1 --method bm25
PYTHONPATH=src python scripts/run_novex.py --tier 3 --context-k 10

# Analyze results
PYTHONPATH=src python scripts/analyze_novex.py --analysis all

# Generate figures (PDF + PNG)
PYTHONPATH=src python scripts/generate_figures.py --format png

# Build expert validation spreadsheet
PYTHONPATH=src python scripts/build_expert_sheet.py
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PATENTSVIEW_API_KEYS` | Comma-separated API keys | No (but recommended) |
| `OPENALEX_MAILTO` | Email for polite pool | No (but recommended) |
| `EPO_CONSUMER_KEY` | EPO OPS consumer key | Only for Phase 6 |
| `EPO_CONSUMER_SECRET` | EPO OPS consumer secret | Only for Phase 6 |

Create a `.env` file in the project root:
```bash
PATENTSVIEW_API_KEYS=your_key_1,your_key_2
OPENALEX_MAILTO=your@email.com
```

## Running in Background

Using tmux (recommended):
```bash
# Start new tmux session
tmux new -s biopat

# Run the script
./scripts/vps_deploy.sh

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t biopat
```

Using nohup:
```bash
nohup python scripts/run_benchmark.py --phase phase1 > output.log 2>&1 &
```

## Output Structure

After running, you'll find:
```
data/
├── raw/           # Downloaded source data (RoS, etc.)
├── processed/     # Intermediate processed data
├── cache/         # API response cache
└── benchmark/     # Final BEIR-format output
    ├── corpus.jsonl
    ├── queries.jsonl
    └── qrels/
        ├── train.tsv
        ├── dev.tsv
        └── test.tsv

logs/
├── phase1_YYYYMMDD_HHMMSS.log
└── phase1_results_YYYYMMDD_HHMMSS.json
```

## Troubleshooting

**Rate limiting errors:**
- Add PatentsView API keys (get free keys at https://patentsview.org/apis/keyrequest)
- Add OpenAlex email for polite pool access

**Memory issues:**
- Reduce `target_patent_count` in config
- Use a VPS with at least 4GB RAM

**Disk space:**
- Full benchmark needs ~10-20GB
- Use `--skip-baseline` to save space on initial runs
