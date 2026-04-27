#!/usr/bin/env python3
"""WP3: Upload BioPAT-NovEx dataset to HuggingFace Hub.

Creates two parallel uploads:
  - Identified: VibeCodingScientist/BioPAT-NovEx
  - Anonymous (for TACL review): biopat-anon-2026/BioPAT-NovEx (when ready)

Usage:
    HUGGINGFACE_TOKEN=hf_... python scripts/upload_to_huggingface.py \
        --org VibeCodingScientist \
        --repo BioPAT-NovEx \
        --version v1.0
    HUGGINGFACE_TOKEN=hf_... python scripts/upload_to_huggingface.py \
        --org biopat-anon-2026 \
        --repo BioPAT-NovEx \
        --version v1.0 \
        --anonymous

Requires: pip install huggingface_hub datasets
"""

import argparse
import json
import os
import sys
from pathlib import Path


def build_dataset_card(dataset_name: str, anonymous: bool = False) -> str:
    """Generate the HuggingFace dataset card markdown."""
    authors_block = (
        "## Authors\n\n*[anonymized for review]*\n"
        if anonymous
        else "## Authors\n\nBioPAT Contributors\n"
    )
    citation_block = (
        '```bibtex\n'
        '@misc{biopat_anon_2026,\n'
        '  title  = {BioPAT-NovEx: A 3-Tier Benchmark for LLM-Based Patent Prior Art Discovery},\n'
        '  author = {anonymous},\n'
        '  year   = {2026},\n'
        '  note   = {Under review}\n'
        '}\n'
        '```\n'
        if anonymous
        else (
            '```bibtex\n'
            '@software{biopat,\n'
            '  author    = {BioPAT Contributors},\n'
            '  title     = {BioPAT-NovEx: A 3-Tier Benchmark for LLM-Based Patent Prior Art Discovery},\n'
            '  year      = {2026},\n'
            '  publisher = {GitHub},\n'
            '  url       = {https://github.com/VibeCodingScientist/BioPAT}\n'
            '}\n'
            '```\n'
        )
    )

    return f"""---
license: cc-by-4.0
language:
- en
tags:
- patents
- prior-art
- novelty
- biomedical
- retrieval
- BEIR
- benchmark
size_categories:
- 100K<n<1M
task_categories:
- text-retrieval
- text-classification
pretty_name: BioPAT-NovEx
configs:
- config_name: default
  data_files:
  - split: corpus
    path: corpus.jsonl
  - split: queries
    path: queries.jsonl
- config_name: statements
  data_files:
  - split: train
    path: statements.jsonl
---

# BioPAT-NovEx: A 3-Tier Benchmark for LLM-Based Patent Prior Art Discovery

A benchmark for evaluating how well LLMs can retrieve, assess, and determine
novelty of biomedical patent prior art.

{authors_block}

## Dataset Description

BioPAT-NovEx contains:
- **300 expert-curated patent-derived technical statements**
- **158,850-document scientific paper corpus** (OpenAlex-derived)
- **5,352 graded relevance judgments** (0-3 scale) for Tier 1 retrieval
- **300 ground-truth novelty labels** (NOVEL / PARTIALLY_ANTICIPATED / ANTICIPATED)
  produced via 3-LLM consensus annotation (Fleiss' kappa = 0.342)

## Files

| File | Description | Format |
|------|-------------|--------|
| `corpus.jsonl` | 158,850 documents (papers + cited patent IDs) | BEIR `_id`/`title`/`text` |
| `queries.jsonl` | 300 patent-derived statements | BEIR `_id`/`text` |
| `statements.jsonl` | Statements with full metadata, source patent, GT labels | JSON Lines |
| `qrels/tier1.tsv` | Graded relevance judgments | `query_id\\tdoc_id\\tscore` |
| `qrels/tier3.tsv` | Novelty labels | `query_id\\tdoc_id\\t{{0,1,2}}` |

## Statistics

- **Domains**: A61 (medical), C07 (organic chemistry), C12 (biochemistry)
- **Novelty distribution**: ANTICIPATED 70% / PARTIALLY_ANTICIPATED 24% / NOVEL 6%
- **Inter-annotator agreement**: Fleiss' κ = 0.342 (fair); 59% unanimous, 33% majority

## Loading

```python
from datasets import load_dataset

# Load corpus + queries (BEIR-compatible)
corpus = load_dataset("{dataset_name}", "default", split="corpus")
queries = load_dataset("{dataset_name}", "default", split="queries")

# Load full statements with metadata
statements = load_dataset("{dataset_name}", "statements", split="train")
```

## Citation

{citation_block}

## License

CC BY 4.0. Patent text and metadata are USPTO public domain. OpenAlex abstracts
are CC0 where available. Composed annotations and curation are CC BY 4.0.

## Limitations

See the original paper for known limitations including:
- Override-rule construction of NOVEL class
- Single-language (English) coverage
- Domain skew toward biomedical patents (A61/C07/C12)
"""


def stage_files(staging_dir: Path, source_dirs: dict) -> None:
    """Copy required files to staging dir."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        (source_dirs["corpus"] / "corpus.jsonl", staging_dir / "corpus.jsonl"),
        (source_dirs["novex"] / "queries.jsonl", staging_dir / "queries.jsonl"),
        (source_dirs["novex"] / "statements.jsonl", staging_dir / "statements.jsonl"),
        (source_dirs["novex"] / "qrels" / "tier1.tsv", staging_dir / "qrels" / "tier1.tsv"),
        (source_dirs["novex"] / "qrels" / "tier3.tsv", staging_dir / "qrels" / "tier3.tsv"),
    ]
    for src, dst in files_to_copy:
        if not src.exists():
            print(f"WARN: missing {src}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Stream-copy to handle large files
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            while True:
                chunk = fin.read(1 << 20)
                if not chunk:
                    break
                fout.write(chunk)
        print(f"  staged: {dst.relative_to(staging_dir)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--org", required=True, help="HF org/username")
    p.add_argument("--repo", default="BioPAT-NovEx", help="Dataset repo name")
    p.add_argument("--version", default="v1.0", help="Dataset version tag")
    p.add_argument("--anonymous", action="store_true",
                   help="Generate anonymous version (no author info)")
    p.add_argument("--corpus-dir", default="data/benchmark",
                   help="Directory containing corpus.jsonl")
    p.add_argument("--novex-dir", default="data/novex",
                   help="Directory containing queries/statements/qrels")
    p.add_argument("--staging-dir", default=".hf_staging",
                   help="Local staging directory")
    p.add_argument("--dry-run", action="store_true",
                   help="Stage files and write README locally; skip the upload")
    args = p.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HUGGINGFACE_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    dataset_name = f"{args.org}/{args.repo}"
    staging = Path(args.staging_dir) / args.repo
    print(f"Dataset: {dataset_name} (anonymous={args.anonymous})")
    print(f"Staging: {staging}")

    # 1. Stage files
    print("\n>>> Staging files...")
    stage_files(staging, {
        "corpus": Path(args.corpus_dir),
        "novex": Path(args.novex_dir),
    })

    # 2. Write dataset card
    card = build_dataset_card(dataset_name, anonymous=args.anonymous)
    (staging / "README.md").write_text(card, encoding="utf-8")
    print(f"  staged: README.md")

    # 3. Write Croissant metadata stub (HF auto-generates the rest)
    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "ml": "http://mlcommons.org/croissant/",
        },
        "@type": "Dataset",
        "name": "BioPAT-NovEx",
        "description": "3-tier benchmark for LLM-based patent prior art discovery",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "version": args.version,
        "url": f"https://huggingface.co/datasets/{dataset_name}",
        "citeAs": (
            "anonymous (2026). BioPAT-NovEx: A 3-Tier Benchmark for LLM-Based "
            "Patent Prior Art Discovery."
            if args.anonymous
            else "BioPAT Contributors (2026). BioPAT-NovEx."
        ),
    }
    (staging / "croissant.json").write_text(
        json.dumps(croissant, indent=2), encoding="utf-8"
    )
    print(f"  staged: croissant.json")

    if args.dry_run:
        print("\n--dry-run: stopping before upload. Files in", staging)
        return 0

    # 4. Upload via HuggingFace Hub
    print(f"\n>>> Uploading to {dataset_name}...")
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    create_repo(
        repo_id=dataset_name,
        repo_type="dataset",
        exist_ok=True,
        private=False,
        token=token,
    )
    api.upload_folder(
        folder_path=str(staging),
        repo_id=dataset_name,
        repo_type="dataset",
        commit_message=f"BioPAT-NovEx {args.version} initial upload",
        token=token,
    )

    # 5. Tag the version
    api.create_tag(
        repo_id=dataset_name,
        repo_type="dataset",
        tag=args.version,
        token=token,
    )

    print(f"\nDone: https://huggingface.co/datasets/{dataset_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
