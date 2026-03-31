# BioPAT-NovEx: A 3-Tier Benchmark for LLM-Based Patent Prior Art Discovery

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BEIR Compatible](https://img.shields.io/badge/format-BEIR-orange.svg)](https://github.com/beir-cellar/beir)

**A benchmark for evaluating how well LLMs can retrieve, assess, and determine novelty of biomedical patent prior art.**

BioPAT-NovEx constructs 300 expert-curated patent-derived technical statements and evaluates LLM performance across three tiers: document retrieval, relevance grading, and novelty determination. It includes a 164K dual corpus of scientific papers and patents, 5,352 graded relevance judgments, and ground truth novelty labels from a 3-LLM consensus protocol.

---

## Benchmark at a Glance

| Stat | Value |
|------|-------|
| **Statements** | 300 patent-derived technical claims |
| **Dual corpus** | 164,649 documents (158,850 papers + 5,799 patents) |
| **Relevance judgments** | 5,352 graded qrels (0-3 scale) |
| **Novelty labels** | 300 (ANTICIPATED 70% / PARTIAL 24% / NOVEL 6%) |
| **Domains** | A61 (medical, 121) / C07 (organic chemistry, 88) / C12 (biochemistry, 91) |
| **LLMs evaluated** | GPT-5.2, Claude Sonnet 4.6, Gemini 3 Pro, Claude Haiku 4.5 |
| **Dense models** | BGE-base-en-v1.5, SPECTER2 |
| **Evaluation runs** | 36 (T1: 9 methods, T2: 3 models, T3: 6 conditions + 15 ablation + 1 bias control) |
| **Format** | BEIR-compatible (corpus.jsonl, queries.jsonl, qrels/*.tsv) |

Ground truth novelty labels are derived from a 3-LLM consensus annotation protocol (Fleiss' kappa = 0.342), with 59% unanimous agreement, 33% majority vote, and 8% requiring override rules.

## Results

### Tier 1: Document Retrieval

| Method | Model | Recall@10 | NDCG@10 | Recall@100 | MAP |
|--------|-------|-----------|---------|------------|-----|
| **BM25** | -- | 0.481 | 0.675 | **1.000** | **0.719** |
| **BM25 + Rerank** | GPT-5.2 | **0.564** | **0.775** | 0.814 | 0.661 |
| **BM25 + Rerank** | Claude Sonnet 4.6 | 0.563 | 0.772 | 0.814 | 0.662 |
| **BM25 + Rerank** | Gemini 3 Pro | 0.481 | 0.675 | 0.814 | 0.600 |
| **Dense** | BGE-base-en-v1.5 | 0.118 | 0.176 | 0.317 | 0.117 |
| **Dense** | SPECTER2 | 0.085 | 0.142 | 0.202 | 0.073 |
| **Agent** | Claude Sonnet 4.6 | 0.289 | 0.464 | 0.711 | 0.353 |
| **Agent** | Gemini 3 Pro | 0.396 | 0.595 | 0.587 | 0.426 |
| **Agent** | GPT-5.2 | 0.186 | 0.313 | 0.533 | 0.224 |

BM25 achieves perfect Recall@100 owing to the citation-derived qrels. LLM reranking improves top-of-list precision (NDCG@10 +15%). Dense retrieval underperforms BM25, with general-purpose BGE outperforming biomedical SPECTER2 -- suggesting the patent-publication vocabulary gap is structural, not a domain knowledge problem. Agent retrieval explores the dual corpus iteratively but trades recall for breadth.

### Tier 2: Relevance Grading (5,352 pairs)

| Model | Accuracy | MAE | Weighted Kappa |
|-------|----------|-----|----------------|
| **Claude Sonnet 4.6** | **0.791** | **0.209** | **0.873** |
| GPT-5.2 | 0.767 | 0.236 | 0.853 |
| Gemini 3 Pro | 0.725 | 0.285 | 0.827 |

All models show strong agreement with ground truth (weighted kappa > 0.82).

### Tier 3: Novelty Determination (300 statements)

| Model | Mode | Accuracy | Macro-F1 | F1-NOVEL | F1-PARTIAL | F1-ANTIC |
|-------|------|----------|----------|----------|------------|----------|
| **Claude Sonnet 4.6** | ctx (k=10) | **0.773** | 0.650 | 0.529 | **0.552** | **0.869** |
| GPT-5.2 | ctx (k=10) | 0.753 | **0.661** | **0.615** | 0.521 | 0.848 |
| Gemini 3 Pro | ctx (k=10) | 0.753 | 0.550 | 0.515 | 0.250 | 0.886 |
| **Haiku 4.5 (bias ctrl)** | ctx + CoT | 0.690 | 0.548 | 0.480 | 0.364 | 0.799 |
| Claude Sonnet 4.6 | zero-shot | 0.515 | 0.426 | 0.170 | 0.444 | 0.663 |
| GPT-5.2 | zero-shot | 0.193 | 0.195 | 0.119 | 0.190 | 0.274 |
| Gemini 3 Pro | zero-shot | 0.064 | 0.041 | 0.114 | 0.000 | 0.010 |

Context (providing top-k prior art documents) is essential -- all models collapse without it. Claude leads accuracy (77.3%), GPT-5.2 leads macro-F1 (0.661) due to stronger NOVEL class detection.

### Haiku 4.5 Bias Control Experiment

The 3 annotator models that created the ground truth were also the ones evaluated against it, raising a shared-bias concern. To quantify this, we ran Claude Haiku 4.5 (not involved in annotation) with chain-of-thought reasoning:

| Model | Accuracy | Macro F1 [95% CI] | Role |
|-------|----------|-------------------|------|
| GPT-5.2 | 75.3% | 0.661 | Annotator + Eval |
| Claude Sonnet 4.6 | 77.3% | 0.650 | Annotator + Eval |
| Gemini 3 Pro | 75.3% | 0.550 | Annotator + Eval |
| **Haiku 4.5 (CoT)** | **69.0%** | **0.548 [0.441, 0.635]** | **Control only** |

The 7pp accuracy gap is consistent with a model capability difference rather than large shared-bias inflation. Without chain-of-thought reasoning, Haiku scores only 44.7%, confirming that extended reasoning is critical for this task.

### Context Ablation (k = number of prior art docs)

| Model | k=1 | k=3 | k=5 | k=10 | k=20 |
|-------|-----|-----|-----|------|------|
| GPT-5.2 | 0.563 | 0.703 | 0.747 | 0.757 | 0.753 |
| Claude Sonnet 4.6 | 0.706 | 0.746 | 0.759 | 0.756 | 0.756 |
| Gemini 3 Pro | 0.693 | 0.747 | 0.739 | 0.728 | 0.745 |

Performance plateaus around k=5 for all models.

### Ground Truth Quality

| Metric | Value |
|--------|-------|
| Fleiss' kappa (3 LLMs) | 0.342 (fair agreement) |
| GPT-5.2 <-> Claude kappa | 0.574 (moderate) |
| Unanimous agreement | 178/300 (59%) |
| Majority agreement | 98/300 (33%) |

### Cost

| Phase | Cost |
|-------|------|
| Benchmark construction (curation + grading + novelty) | $38.56 |
| 3-tier evaluation (36 runs) | ~$346 |
| **Total** | **~$385** |

## Architecture

```
src/biopat/
├── pipeline.py              # Phase 1: 8-step benchmark construction
├── pipeline_novelty.py      # End-to-end novelty assessment
├── config.py                # Configuration management
│
├── ingestion/               # Data acquisition (PatentsView, OpenAlex, RoS)
├── processing/              # Patent/paper processing, chemical/sequence indexing
├── groundtruth/             # Relevance judgment creation, temporal validation
├── benchmark/               # BEIR formatting, train/dev/test splitting
│
├── evaluation/              # Retrieval evaluation
│   ├── bm25.py              #   BM25 baseline
│   ├── dense.py             #   Dense retrieval (13 models, FAISS)
│   ├── hybrid.py            #   BM25 + dense fusion
│   ├── reranker.py          #   Cross-encoder and LLM reranking
│   ├── agent_retrieval.py   #   Agentic dual-corpus retrieval
│   └── statistical_tests.py #   Bootstrap CIs, paired significance tests
│
├── retrieval/               # SOTA retrieval methods
│   ├── dense.py, hybrid.py, hyde.py, splade.py, colbert.py
│   └── molecular.py, sequence.py  # Chemical/sequence retrieval
│
├── reasoning/               # LLM novelty reasoning
│   ├── claim_parser.py, novelty_reasoner.py, explanation_generator.py
│
├── novex/                   # NovEx 3-tier benchmark
│   ├── evaluator.py         #   Tier 1/2/3 harness with checkpointing
│   ├── annotation.py        #   3-LLM consensus annotation protocol
│   └── analysis.py          #   24 statistical analyses
│
└── llm/                     # Unified LLM provider (OpenAI, Anthropic, Google)
    ├── providers.py         #   Consistent API with thinking/adaptive reasoning
    └── cost_tracker.py      #   Per-token cost tracking and budget enforcement
```

## Quick Start

### Installation

```bash
git clone https://github.com/VibeCodingScientist/BioPAT.git
cd BioPAT

python -m venv venv
source venv/bin/activate

# Core only
pip install -e .

# With evaluation (sentence-transformers, FAISS, PyTorch)
pip install -e ".[evaluation]"

# Everything
pip install -e ".[all]"
```

### Download Benchmark Data

The full corpus (158K documents) is available via GitHub Release:

```bash
gh release download v1.0.0 --pattern 'biopat-benchmark.tar.gz'
tar xzf biopat-benchmark.tar.gz
```

### Running Experiments

```bash
# NovEx 3-tier evaluation
PYTHONPATH=src python scripts/run_novex.py --tier 1 --method bm25
PYTHONPATH=src python scripts/run_novex.py --tier 3 --context-k 10

# Dense retrieval with checkpointing and progress
PYTHONPATH=src python scripts/run_dense_experiments.py \
  --models "BAAI/bge-base-en-v1.5" "allenai/specter2_base"

# Analyze results
PYTHONPATH=src python scripts/analyze_novex.py --analysis all

# Cost estimate (no API calls)
PYTHONPATH=src python scripts/run_novex.py --dry-run
```

### Environment Variables

```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key
```

## Data

NovEx evaluation data is included in the repository:

```
data/novex/
├── statements.jsonl         # 300 curated patent-derived statements
├── queries.jsonl            # BEIR-format queries
├── qrels/                   # tier1.tsv (relevance), tier3.tsv (novelty)
├── results/                 # All evaluation checkpoints and metrics
│   ├── all_results.json     # 36 experiment results
│   └── checkpoints/         # Per-experiment checkpoint files
├── analysis/                # 24 statistical analysis JSON files
└── expert_validation.*      # 100-statement expert review package
```

The full corpus (158K papers) is available via [GitHub Release](https://github.com/VibeCodingScientist/BioPAT/releases/tag/v1.0.0).

## Expert Validation

An expert validation package is included for human review of 100 strategically sampled statements:

- `expert_validation.xlsx` -- Annotatable Excel workbook
- `expert_validation.csv` -- Flat CSV version
- `expert_validation_guidelines.txt` -- Annotation guidelines

The 100 statements are stratified: all 11 no-consensus items, all 13 override items, 46 majority items, and 30 unanimous controls.

## Configuration

| Config file | Purpose |
|-------------|---------|
| `configs/default.yaml` | Phase 1 pipeline settings |
| `configs/novex.yaml` | NovEx benchmark (curation, annotation, evaluation) |
| `configs/experiments.yaml` | Full experiment suite |
| `configs/experiments_agent.yaml` | Agent retrieval experiments |

## LLM Support

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | GPT-4o, GPT-5.2 | Reasoning effort control |
| **Anthropic** | Claude Opus/Sonnet 4.6, Haiku 4.5 | Adaptive thinking |
| **Google** | Gemini 3 Pro, 2.5 Pro | Thinking config |

## Dependencies

**Core**: httpx, polars, pyyaml, pydantic, rank-bm25, tqdm

**Evaluation**: sentence-transformers, faiss-cpu, torch

Python 3.11+ required.

## Citation

```bibtex
@software{biopat,
  author    = {BioPAT Contributors},
  title     = {{BioPAT-NovEx}: A 3-Tier Benchmark for LLM-Based Patent Prior Art Discovery},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/VibeCodingScientist/BioPAT}
}
```

## License

MIT License -- see [LICENSE](LICENSE) for details.
