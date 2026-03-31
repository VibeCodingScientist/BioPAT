#!/usr/bin/env python3
"""Run dense retrieval experiments (NovEx Tier 1) with checkpointing and live progress.

Usage:
    python scripts/run_dense_experiments.py
    python scripts/run_dense_experiments.py --models BAAI/bge-base-en-v1.5 allenai/specter2
    python scripts/run_dense_experiments.py --batch-size 16 -v

Features:
    - Embedding cache: 158K doc encodings survive restarts (~45 min saved)
    - Per-model checkpoints: completed models are skipped on re-run
    - Live progress: batch-level ETA and throughput printed to stdout
    - Fallback: if one model fails, others still complete
    - Progress file: dense_progress.json updated every 30s for external monitoring
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Writes dense_progress.json periodically for external monitoring."""

    def __init__(self, path: Path, interval: float = 30.0):
        self.path = path
        self.interval = interval
        self._last_write = 0.0
        self.state: Dict[str, Any] = {}

    def update(self, **kwargs) -> None:
        self.state.update(kwargs, updated_at=datetime.now().isoformat())
        now = time.time()
        if now - self._last_write >= self.interval:
            self._flush()
            self._last_write = now

    def _flush(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception:
            pass

    def finish(self) -> None:
        self.state["status"] = "done"
        self._flush()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Metrics (standalone, matches evaluator._tier1_metrics)
# ---------------------------------------------------------------------------

def compute_tier1_metrics(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute NDCG, Recall, MAP from ranked results against qrels."""
    all_recall = {k: [] for k in k_values}
    all_ndcg = {k: [] for k in k_values}
    all_ap = []

    for qid, rel in qrels.items():
        ranked = results.get(qid, [])
        n_rel = len(rel)
        if n_rel == 0:
            continue

        for k in k_values:
            top = set(d for d, _ in ranked[:k])
            all_recall[k].append(len(top & set(rel)) / n_rel)
            dcg = sum(rel[d] / math.log2(r + 2) for r, (d, _) in enumerate(ranked[:k]) if d in rel)
            idcg = sum(s / math.log2(r + 2) for r, s in enumerate(sorted(rel.values(), reverse=True)[:k]))
            all_ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

        hits, ap = 0, 0.0
        for r, (d, _) in enumerate(ranked):
            if d in rel:
                hits += 1
                ap += hits / (r + 1)
        all_ap.append(ap / n_rel)

    def _mean(v):
        return sum(v) / len(v) if v else 0.0

    m: Dict[str, float] = {}
    for k in k_values:
        m[f"recall@{k}"] = _mean(all_recall[k])
        m[f"ndcg@{k}"] = _mean(all_ndcg[k])
    m["map"] = _mean(all_ap)
    return m


def compute_per_query(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    pq: Dict[str, Dict[str, float]] = {}
    for qid, rel in qrels.items():
        if not rel or qid not in results:
            continue
        ranked = results[qid]
        pq[qid] = {}
        for k in k_values:
            top = set(d for d, _ in ranked[:k])
            pq[qid][f"recall@{k}"] = len(top & set(rel)) / len(rel)
    return pq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_model(
    model_name: str,
    corpus: Dict[str, dict],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    results_dir: Path,
    cache_dir: Path,
    batch_size: int,
    top_k: int,
    k_values: List[int],
    progress: ProgressTracker,
) -> Optional[Dict[str, Any]]:
    """Run dense retrieval for a single model. Returns TierResult dict or None on failure."""

    safe = model_name.replace("/", "_")
    cp_path = results_dir / "checkpoints" / f"t1_dense_{safe}.json"

    # Check checkpoint
    if cp_path.exists():
        print(f"[{_ts()}]   [cached] Loaded from checkpoint: {cp_path.name}")
        with open(cp_path) as f:
            return json.load(f)

    from biopat.evaluation.dense import DenseRetriever, DenseRetrieverConfig

    t0 = time.time()

    # Load model
    print(f"[{_ts()}]   Loading model...")
    progress.update(model=model_name, phase="loading_model", progress=0.0)
    config = DenseRetrieverConfig(
        model_name=model_name,
        batch_size=batch_size,
        cache_embeddings=True,
        cache_dir=str(cache_dir),
    )
    retriever = DenseRetriever(config=config)
    retriever.load_model()
    dim = retriever.model.get_sentence_embedding_dimension()
    print(f"[{_ts()}]   Model loaded (dim={dim})")

    # Encode documents
    n_docs = len(corpus)
    n_batches = math.ceil(n_docs / batch_size)
    print(f"[{_ts()}]   Encoding {n_docs} documents ({n_batches} batches, batch_size={batch_size})...")
    progress.update(phase="encoding_docs", progress=0.0, total_docs=n_docs)

    doc_ids = list(corpus.keys())
    texts = [f"{corpus[d].get('title', '')} {corpus[d].get('text', '')}".strip() for d in doc_ids]

    # Check embedding cache first
    import numpy as np
    import gc
    cache_key = f"docs_{len(texts)}_{doc_ids[0] if doc_ids else 'empty'}"
    cache_path = retriever._get_cache_path(cache_key)
    has_cache = cache_path and cache_path.exists()

    if has_cache:
        # Free model BEFORE loading 465MB embeddings — we'll reload for queries later
        print(f"[{_ts()}]   Cache found. Freeing model to load embeddings...")
        del retriever.model
        retriever.model = None
        gc.collect()
        doc_embeddings = np.load(cache_path)
        print(f"[{_ts()}]   Embeddings loaded from cache ({doc_embeddings.shape})")
    else:
        # Encode in chunks with periodic saves to survive crashes
        chunk_size = 10000  # save every 10K docs
        partial_cache = cache_dir / f"partial_{model_name.replace('/', '_')}.npy"
        start_idx = 0

        # Resume from partial cache if it exists
        if partial_cache.exists():
            partial = np.load(partial_cache)
            start_idx = partial.shape[0]
            print(f"[{_ts()}]   Resuming from partial cache ({start_idx}/{n_docs} docs already encoded)")
            all_embeddings = [partial]
        else:
            all_embeddings = []

        encode_start = time.time()
        for chunk_start in range(start_idx, n_docs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_docs)
            chunk_texts = texts[chunk_start:chunk_end]
            print(f"[{_ts()}]   Encoding chunk {chunk_start}-{chunk_end} ({chunk_end}/{n_docs}, {chunk_end*100//n_docs}%)...")

            chunk_emb = retriever.model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=config.normalize_embeddings,
                convert_to_numpy=True,
            )
            all_embeddings.append(chunk_emb)

            # Save partial progress after each chunk
            partial_so_far = np.vstack(all_embeddings)
            np.save(partial_cache, partial_so_far)
            elapsed = time.time() - encode_start
            rate = (chunk_end - start_idx) / elapsed if elapsed > 0 else 0
            remaining = (n_docs - chunk_end) / rate if rate > 0 else 0
            print(f"[{_ts()}]   Partial saved ({chunk_end}/{n_docs}). Rate: {rate:.1f} docs/sec, ETA: {remaining/60:.0f}min")
            progress.update(phase="encoding_docs", progress=chunk_end / n_docs,
                           docs_encoded=chunk_end, rate=f"{rate:.1f} docs/sec",
                           eta_minutes=f"{remaining/60:.0f}")

        doc_embeddings = np.vstack(all_embeddings)
        encode_time = time.time() - encode_start
        print(f"[{_ts()}]   Encoded {n_docs} docs in {encode_time:.0f}s ({n_docs/encode_time:.1f} docs/sec)")

        # Save final cache and clean up partial
        retriever._save_cached_embeddings(cache_key, doc_embeddings)
        if partial_cache.exists():
            partial_cache.unlink()
        print(f"[{_ts()}]   Embeddings cached to disk")

    # Step 1: Encode queries WHILE model is loaded (small — 300 queries)
    print(f"[{_ts()}]   Encoding {len(queries)} queries...")
    progress.update(phase="encoding_queries", progress=0.8)
    if retriever.model is None:
        print(f"[{_ts()}]   Reloading model for query encoding...")
        retriever.load_model()
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = retriever.model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )
    print(f"[{_ts()}]   Queries encoded ({query_embeddings.shape})")

    # Step 2: Free model (~400MB) before search
    del retriever.model
    retriever.model = None
    gc.collect()
    print(f"[{_ts()}]   Freed model")

    # Step 3: Search using numpy dot product (avoids FAISS segfault on Apple Silicon)
    # Process one query at a time to keep memory low
    print(f"[{_ts()}]   Searching top-{top_k} via numpy cosine similarity...")
    progress.update(phase="searching", progress=0.9)

    raw_results = {}
    for i, qid in enumerate(query_ids):
        # Cosine similarity = dot product for normalized vectors
        scores = doc_embeddings @ query_embeddings[i]
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        raw_results[qid] = {}
        for idx in top_indices:
            raw_results[qid][doc_ids[idx]] = float(scores[idx])
        if (i + 1) % 50 == 0:
            print(f"[{_ts()}]   Searched {i+1}/{len(query_ids)} queries")

    del doc_embeddings, query_embeddings
    gc.collect()
    print(f"[{_ts()}]   Search complete ({len(raw_results)} queries), memory freed")

    # Normalize to List[Tuple] format
    results = {}
    for qid, doc_scores in raw_results.items():
        if isinstance(doc_scores, dict):
            results[qid] = sorted(doc_scores.items(), key=lambda x: -x[1])
        else:
            results[qid] = doc_scores

    # Compute metrics
    metrics = compute_tier1_metrics(results, qrels, k_values)
    per_query = compute_per_query(results, qrels, k_values)
    elapsed = time.time() - t0

    print(f"[{_ts()}]   Results:")
    for k, v in sorted(metrics.items()):
        print(f"[{_ts()}]     {k}: {v:.4f}")
    print(f"[{_ts()}]   Elapsed: {elapsed:.0f}s")

    # Build TierResult
    tier_result = {
        "tier": 1,
        "method": f"dense_{safe}",
        "model": "N/A",
        "metrics": metrics,
        "per_query": per_query,
        "cost_usd": 0.0,
        "elapsed_seconds": elapsed,
        "metadata": {"model_name": model_name, "batch_size": batch_size, "dimension": dim},
    }

    # Save checkpoint
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cp_path, "w") as f:
        json.dump(tier_result, f, indent=2, default=str)
    print(f"[{_ts()}]   Checkpoint saved: {cp_path.name}")
    progress.update(phase="done", progress=1.0)

    return tier_result


def main():
    parser = argparse.ArgumentParser(description="Run dense retrieval experiments on NovEx Tier 1")
    parser.add_argument("--data-dir", default="data/novex", help="NovEx data directory")
    parser.add_argument("--results-dir", default="data/novex/results", help="Results output directory")
    parser.add_argument("--models", nargs="+", default=["BAAI/bge-base-en-v1.5", "allenai/specter2"],
                        help="Embedding model names (HuggingFace IDs)")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--top-k", type=int, default=100, help="Number of docs to retrieve per query")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = Path(args.results_dir)
    cache_dir = results_dir / "cache"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    k_values = [10, 50, 100]
    progress = ProgressTracker(results_dir / "dense_progress.json")

    # Load benchmark
    print(f"[{_ts()}] Loading benchmark...")
    from biopat.novex.benchmark import NovExBenchmark
    b = NovExBenchmark(data_dir=args.data_dir)
    b.load()
    print(f"[{_ts()}] Loaded: {len(b.queries)} queries, {len(b.corpus)} docs, {sum(len(v) for v in b.tier1_qrels.values())} qrels")

    # Run models
    completed = []
    failed = []

    for i, model_name in enumerate(args.models):
        print(f"\n[{_ts()}] {'='*55}")
        print(f"[{_ts()}] === Model {i+1}/{len(args.models)}: {model_name} ===")
        print(f"[{_ts()}] {'='*55}")
        progress.update(model=model_name, model_index=i + 1, total_models=len(args.models))

        try:
            result = run_single_model(
                model_name=model_name,
                corpus=b.corpus,
                queries=b.queries,
                qrels=b.tier1_qrels,
                results_dir=results_dir,
                cache_dir=cache_dir,
                batch_size=args.batch_size,
                top_k=args.top_k,
                k_values=k_values,
                progress=progress,
            )
            if result:
                completed.append(result)
        except Exception as e:
            logger.error(f"FAILED {model_name}: {e}", exc_info=True)
            print(f"[{_ts()}]   ERROR: {e}")
            failed.append({"model": model_name, "error": str(e)})

    # Summary
    print(f"\n[{_ts()}] {'='*55}")
    print(f"[{_ts()}] SUMMARY: {len(completed)}/{len(args.models)} models completed, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"[{_ts()}]   FAILED: {f['model']} — {f['error']}")
    print(f"[{_ts()}] {'='*55}")

    # Merge into all_results.json
    if completed:
        all_results_path = results_dir / "all_results.json"
        existing = []
        if all_results_path.exists():
            with open(all_results_path) as f:
                existing = json.load(f)

        # Remove any existing dense results for these models (avoid duplicates)
        new_methods = {r["method"] for r in completed}
        existing = [r for r in existing if r.get("method") not in new_methods]
        merged = existing + completed

        with open(all_results_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"[{_ts()}] Merged {len(completed)} new results into all_results.json ({len(merged)} total)")

    progress.finish()

    if not completed:
        print(f"[{_ts()}] No models completed successfully.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
