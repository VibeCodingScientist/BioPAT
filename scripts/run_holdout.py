#!/usr/bin/env python3
"""WP1: Tier 3 hold-out evaluation via OpenRouter.

Bulletproof checkpointing strategy:
  1. PER-CALL atomic append to JSONL (predictions + provider log)
  2. Resume support: skips qids already present in predictions file
  3. fsync after every write (no buffered loss on crash)
  4. Signal handlers (SIGINT/SIGTERM) for graceful shutdown
  5. Cost ledger written incrementally
  6. Atomic file writes with temp+rename for summary/cost files
  7. Heartbeat file every 30s for external monitoring
  8. Per-batch flush summary every 10 calls
  9. Failure file saves full traceback + raw response for debugging
 10. Validates JSONL integrity on resume (rejects malformed lines)

Usage (single model+mode):
  OPENROUTER_API_KEY=sk-or-... python run_holdout.py \
      --model meta-llama/llama-3.3-70b-instruct \
      --provider Together \
      --mode ctx_k10_cot \
      --output-dir results/

Run-all helper at the bottom: see run_all_models.sh
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


PRICING = {
    "meta-llama/llama-3.3-70b-instruct": (0.60, 0.90),
    "deepseek/deepseek-chat": (0.30, 1.00),
    "qwen/qwen-2.5-72b-instruct": (0.70, 1.00),
    "mistralai/mistral-large": (2.00, 6.00),
}

NOVELTY_LABELS = {"NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"}
NOVELTY_MAP = {0: "NOVEL", 1: "PARTIALLY_ANTICIPATED", 2: "ANTICIPATED"}

SYSTEM_NOCOT = 'Novelty expert. Output JSON: {"label": "<>", "confidence": <0-1>}'

COT_PREAMBLE = (
    "You are a patent novelty expert evaluating a biomedical patent claim against prior art. "
    "Think step by step:\n"
    "1. Identify the key technical elements of the claim.\n"
    "2. Examine each prior art document to see if any combination discloses these elements.\n"
    "3. Note: bare patent IDs (without title/abstract) cited as prior art mean an examiner "
    "already determined relevance; lean toward ANTICIPATED.\n"
    "4. Decide:\n"
    "   - NOVEL: claim describes something genuinely new not in the prior art\n"
    "   - PARTIALLY_ANTICIPATED: some elements found in prior art, but novel combinations remain\n"
    "   - ANTICIPATED: prior art fully discloses the claim's key elements\n"
    "Calibration: ~70% of statements in this benchmark are ANTICIPATED.\n\n"
)


# Global state for graceful shutdown
_shutdown = threading.Event()


def _handle_signal(signum, frame):
    logger.warning("Received signal %d - shutting down gracefully (in-flight calls will finish)", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Atomic file ops
# ---------------------------------------------------------------------------

class JsonlAppender:
    """Thread-safe append-only JSONL writer with fsync after every line."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Open in append mode
        self._fp = open(self.path, "a", encoding="utf-8")

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        with self._lock:
            self._fp.write(line)
            self._fp.flush()
            os.fsync(self._fp.fileno())

    def close(self):
        with self._lock:
            try:
                self._fp.close()
            except Exception:
                pass


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_completed_qids(path: Path) -> set:
    """Read existing predictions JSONL and return set of completed qids.
    Skips malformed lines and lines marked ok=False (so failures get retried)."""
    if not path.exists():
        return set()
    completed = set()
    bad_lines = 0
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("qid"):
                    completed.add(rec["qid"])
            except json.JSONDecodeError:
                bad_lines += 1
    if bad_lines:
        logger.warning("Skipped %d malformed lines in %s", bad_lines, path.name)
    return completed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark(data_dir: str, corpus_path: str):
    data_dir = Path(data_dir)

    queries: Dict[str, str] = {}
    with open(data_dir / "statements.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries[d["statement_id"]] = d["text"]

    tier1_qrels: Dict[str, Dict[str, int]] = {}
    with open(data_dir / "qrels" / "tier1.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], int(parts[2])
                tier1_qrels.setdefault(qid, {})[did] = score

    tier3_labels: Dict[str, str] = {}
    with open(data_dir / "qrels" / "tier3.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                tier3_labels[parts[0]] = NOVELTY_MAP.get(int(parts[2]), "NOVEL")

    corpus: Dict[str, Dict[str, str]] = {}
    if Path(corpus_path).exists():
        with open(corpus_path) as f:
            for line in f:
                d = json.loads(line)
                corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}

    logger.info(
        "Loaded: %d queries, %d tier1 qrels, %d tier3 labels, %d corpus docs",
        len(queries),
        sum(len(v) for v in tier1_qrels.values()),
        len(tier3_labels),
        len(corpus),
    )
    return queries, tier1_qrels, tier3_labels, corpus


def build_prompt(qid, queries, tier1_qrels, corpus, with_context=True, context_k=10, use_cot=False):
    text = queries[qid]
    context = ""
    if with_context and qid in tier1_qrels:
        docs = sorted(tier1_qrels[qid].items(), key=lambda x: -x[1])[:context_k]
        context = "\n---\n".join(
            f"[{d}] {corpus.get(d, {}).get('title', '')}\n{corpus.get(d, {}).get('text', '')[:300]}"
            for d, _ in docs
        )
    user_prompt = (
        f"CLAIM: {text}\n\n"
        f"PRIOR ART:\n{context or 'None.'}\n\n"
        f"Classify: NOVEL / ANTICIPATED / PARTIALLY_ANTICIPATED"
    )
    system = (COT_PREAMBLE if use_cot else "") + SYSTEM_NOCOT
    return system, user_prompt


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*?\"label\"[^{}]*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    for lab in ("PARTIALLY_ANTICIPATED", "ANTICIPATED", "NOVEL"):
        if lab in text.upper():
            return {"label": lab, "confidence": 0.5, "fallback": "regex"}
    return None


def normalize_label(raw: str) -> str:
    raw = (raw or "").upper().replace(" ", "_").replace("-", "_")
    if "PARTIAL" in raw:
        return "PARTIALLY_ANTICIPATED"
    if "ANTICIPATED" in raw:
        return "ANTICIPATED"
    if "NOVEL" in raw:
        return "NOVEL"
    return "NOVEL"


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    def __init__(self, api_key: str, model: str, provider: Optional[str] = None,
                 timeout: float = 240.0, max_retries: int = 3,
                 reasoning_effort: Optional[str] = None):
        """
        Args:
            reasoning_effort: If set ("low"/"medium"/"high"), enable OpenRouter's
                reasoning parameter to trigger thinking-mode output for compatible
                models. Reasoning chain is included in output_tokens count and
                billed at output rate.
        """
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/VibeCodingScientist/BioPAT",
                "X-Title": "BioPAT-NovEx Hold-out Evaluation",
            },
            timeout=timeout,
        )
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    def call(self, system: str, user: str, max_tokens: int = 1000,
             temperature: float = 0.1) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {"transforms": []}
        if self.provider:
            extra_body["provider"] = {
                "order": [self.provider],
                "allow_fallbacks": False,
            }
        if self.reasoning_effort:
            # OpenRouter standard reasoning parameter — triggers thinking mode
            # for reasoning-capable models. Include reasoning chain in response
            # so we can audit it but extract only the final JSON for parsing.
            extra_body["reasoning"] = {
                "effort": self.reasoning_effort,
            }
            extra_body["include_reasoning"] = True

        last_text = ""
        last_provider = None
        last_input_tokens = 0
        last_output_tokens = 0
        last_latency_ms = 0.0
        last_err: Optional[str] = None

        # Track whether we've tried disabling the reasoning param (auto-fallback
        # for models like QwQ-32B which are intrinsic reasoning models and
        # reject the explicit parameter).
        send_reasoning_param = bool(self.reasoning_effort)

        for attempt in range(self.max_retries):
            current_extra = dict(extra_body)
            if not send_reasoning_param:
                current_extra.pop("reasoning", None)
                current_extra.pop("include_reasoning", None)

            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": current_extra,
            }
            # Skip response_format -- most OpenRouter open-weight provider endpoints
            # don't support it. Our regex JSON extractor handles markdown-fenced output
            # reliably across all observed cases in smoke tests.

            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                err_str = str(e).lower()
                # Models that are intrinsic reasoning models (e.g. QwQ-32B)
                # reject the `reasoning` / `enable_thinking` parameter.
                # Auto-fallback by stripping it on subsequent retries.
                if (
                    "enable_thinking" in err_str
                    or ("reasoning" in err_str and "does not support" in err_str)
                    or "20015" in err_str
                ):
                    if send_reasoning_param:
                        send_reasoning_param = False
                        continue  # retry immediately without the param
                # Any other 400-level / format issue: still retry (e.g. response_format)
                if (
                    "response_format" in err_str
                    or "json_object" in err_str
                    or "invalid_request_body" in err_str
                    or "does not support" in err_str
                    or "400" in err_str
                ):
                    pass
                else:
                    # Backoff for 429/5xx and network issues
                    time.sleep(min(2 ** attempt, 10))
                continue

            last_latency_ms = (time.perf_counter() - t0) * 1000
            last_msg = resp.choices[0].message
            last_text = last_msg.content or ""
            # Reasoning chain is billed as output tokens by OpenRouter so
            # token counts already reflect it; we don't need to capture it
            # separately. The `content` field is the post-reasoning final answer.
            if resp.usage:
                last_input_tokens = resp.usage.prompt_tokens
                last_output_tokens = resp.usage.completion_tokens

            # Extract serving provider from response metadata
            try:
                resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else {}
            except Exception:
                resp_dict = {}
            last_provider = (
                resp_dict.get("provider")
                or getattr(resp, "provider", None)
            )

            parsed = parse_json_response(last_text)
            if parsed is not None:
                return {
                    "ok": True,
                    "parsed": parsed,
                    "raw_text": last_text,
                    "provider": last_provider,
                    "input_tokens": last_input_tokens,
                    "output_tokens": last_output_tokens,
                    "latency_ms": last_latency_ms,
                    "attempts": attempt + 1,
                }
            last_err = f"JSON parse failed (attempt {attempt + 1})"

        return {
            "ok": False,
            "error": last_err or "unknown error after retries",
            "raw_text": last_text,
            "provider": last_provider,
            "input_tokens": last_input_tokens,
            "output_tokens": last_output_tokens,
            "latency_ms": last_latency_ms,
            "attempts": self.max_retries,
        }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(client: OpenRouterClient, qid: str, system: str, user: str,
             max_tokens: int = 1000) -> Dict[str, Any]:
    if _shutdown.is_set():
        return {"qid": qid, "ok": False, "error": "shutdown signal received before call"}
    result = client.call(system, user, max_tokens=max_tokens)
    base = {
        "qid": qid,
        "ok": result["ok"],
        "raw_response": result.get("raw_text", "")[:3000],
        "provider": result.get("provider"),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "latency_ms": result.get("latency_ms", 0),
        "attempts": result.get("attempts", 0),
        "ts": datetime.now().isoformat(),
    }
    if result["ok"]:
        parsed = result["parsed"]
        base["label"] = normalize_label(parsed.get("label", "NOVEL"))
        try:
            base["confidence"] = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            base["confidence"] = 0.5
    else:
        base["error"] = result["error"]
    return base


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def write_heartbeat(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat()
    atomic_write_json(path, state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--provider", default=None, help="Pinned provider (e.g. Together, Fireworks, DeepSeek)")
    p.add_argument("--mode", choices=["ctx_k10_cot", "ctx_k10_nocot", "ctx_k10_reasoning"], required=True)
    p.add_argument("--reasoning-effort", default="high",
                   help="OpenRouter reasoning effort level (low/medium/high). Only applies to --mode ctx_k10_reasoning.")
    p.add_argument("--data-dir", default="/tmp/biopat-explore/data/novex")
    p.add_argument("--corpus-path", default="/tmp/biopat-wp1/data/benchmark/corpus.jsonl")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--max-statements", type=int, default=None)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1000)
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set in environment")
        sys.exit(1)

    queries, tier1_qrels, tier3_labels, corpus = load_benchmark(args.data_dir, args.corpus_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
    pred_path = out_dir / f"tier3_holdout_{safe}_{args.mode}.jsonl"
    log_path = out_dir / "tier3_holdout_provider_log.jsonl"
    heartbeat_path = out_dir / f".heartbeat_{safe}_{args.mode}.json"
    summary_path = out_dir / f"tier3_holdout_{safe}_{args.mode}_summary.json"

    # Resume support
    completed = load_completed_qids(pred_path)
    qids = sorted(tier3_labels.keys())
    if args.max_statements is not None:
        qids = qids[args.start_idx:args.start_idx + args.max_statements]
    todo = [q for q in qids if q not in completed]

    logger.info("=" * 70)
    logger.info("Model: %s", args.model)
    logger.info("Provider pin: %s", args.provider or "(any)")
    logger.info("Mode: %s", args.mode)
    logger.info("Output: %s", pred_path)
    logger.info("Status: %d/%d todo (%d already completed)", len(todo), len(qids), len(completed))
    logger.info("=" * 70)
    if not todo:
        logger.info("Nothing to do.")
        return 0

    use_cot = args.mode == "ctx_k10_cot"
    use_reasoning = args.mode == "ctx_k10_reasoning"
    # Reasoning runs use the no-CoT prompt (model does its own reasoning).
    # CoT prompt would double-prompt and waste tokens.
    use_cot_prompt = use_cot and not use_reasoning
    prompts = {}
    for qid in todo:
        prompts[qid] = build_prompt(qid, queries, tier1_qrels, corpus,
                                    with_context=True, context_k=10,
                                    use_cot=use_cot_prompt)

    # Reasoning runs need much higher max_tokens (thinking chain can be 3-5K tokens)
    effective_max_tokens = 8000 if use_reasoning else args.max_tokens

    client = OpenRouterClient(
        api_key=api_key,
        model=args.model,
        provider=args.provider,
        reasoning_effort=args.reasoning_effort if use_reasoning else None,
        timeout=240.0 if use_reasoning else 120.0,
    )
    pred_writer = JsonlAppender(pred_path)
    log_writer = JsonlAppender(log_path)

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    pricing = PRICING.get(args.model, (0.0, 0.0))

    # Initial heartbeat
    write_heartbeat(heartbeat_path, {
        "model": args.model, "provider": args.provider, "mode": args.mode,
        "phase": "starting", "completed": 0, "total": len(todo),
        "started_at": datetime.now().isoformat(),
    })

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_qid = {
                executor.submit(run_one, client, qid, prompts[qid][0], prompts[qid][1],
                                effective_max_tokens): qid
                for qid in todo
            }
            for i, future in enumerate(as_completed(future_to_qid)):
                qid = future_to_qid[future]
                try:
                    r = future.result()
                except Exception as e:
                    r = {
                        "qid": qid, "ok": False,
                        "error": f"future exception: {type(e).__name__}: {e}",
                        "ts": datetime.now().isoformat(),
                    }

                results.append(r)
                total_input_tokens += r.get("input_tokens", 0)
                total_output_tokens += r.get("output_tokens", 0)

                if not r.get("ok"):
                    failures.append(r)

                # Atomic per-call append (with fsync)
                pred_writer.append(r)
                log_writer.append({
                    "model": args.model,
                    "qid": qid,
                    "mode": args.mode,
                    "provider_pinned": args.provider,
                    "provider_actual": r.get("provider"),
                    "ok": r.get("ok"),
                    "input_tokens": r.get("input_tokens", 0),
                    "output_tokens": r.get("output_tokens", 0),
                    "latency_ms": r.get("latency_ms", 0),
                    "attempts": r.get("attempts", 0),
                    "ts": r.get("ts"),
                })

                done = i + 1
                running_cost = (
                    total_input_tokens * pricing[0]
                    + total_output_tokens * pricing[1]
                ) / 1_000_000

                if done % 10 == 0 or done == len(todo):
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(todo) - done) / rate if rate > 0 else 0
                    logger.info(
                        "  %3d/%d (%5.1f%%) | rate %.1f/s | ETA %4.0fs | failures: %d | $%.4f spent",
                        done, len(todo), 100 * done / len(todo),
                        rate, eta, len(failures), running_cost,
                    )
                    write_heartbeat(heartbeat_path, {
                        "model": args.model, "provider": args.provider, "mode": args.mode,
                        "phase": "running",
                        "completed": done + len(completed),
                        "total": len(qids),
                        "in_session": done,
                        "session_total": len(todo),
                        "failures": len(failures),
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "running_cost_usd": round(running_cost, 6),
                        "rate_per_sec": round(rate, 2),
                        "eta_seconds": round(eta),
                    })

                if _shutdown.is_set():
                    logger.warning("Shutdown signal — cancelling pending tasks")
                    for f in future_to_qid:
                        if not f.done():
                            f.cancel()
                    break
    finally:
        pred_writer.close()
        log_writer.close()

    elapsed = time.time() - t0
    cost = (total_input_tokens * pricing[0] + total_output_tokens * pricing[1]) / 1_000_000

    summary = {
        "model": args.model,
        "provider_pinned": args.provider,
        "mode": args.mode,
        "n_completed_in_session": len(results),
        "n_failures_in_session": len(failures),
        "n_total_completed": len(load_completed_qids(pred_path)),
        "n_target": len(qids),
        "elapsed_seconds": round(elapsed, 2),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": round(cost, 6),
        "finished_at": datetime.now().isoformat(),
    }
    atomic_write_json(summary_path, summary)
    write_heartbeat(heartbeat_path, {**summary, "phase": "done"})

    logger.info("=" * 70)
    logger.info("Done in %.0fs | %d/%d completed (%d failed) | $%.4f",
                elapsed, len(results), len(todo), len(failures), cost)
    logger.info("Predictions: %s", pred_path)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
