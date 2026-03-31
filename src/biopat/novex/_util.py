"""Shared utilities for NovEx modules — checkpoint, JSON parsing, voting."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointMixin:
    """Mixin providing JSON checkpoint save/load. Requires self.checkpoint_dir: Path."""

    checkpoint_dir: Path

    def _checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.json"

    def _has_checkpoint(self, name: str) -> bool:
        return self._checkpoint_path(name).exists()

    def _save_checkpoint(self, name: str, data: Any) -> None:
        with open(self._checkpoint_path(name), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_checkpoint(self, name: str) -> Optional[Any]:
        """Load checkpoint if it exists, else return None."""
        path = self._checkpoint_path(name)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)


def parse_llm_json(provider, prompt: str, system: str, cost_tracker, query_id: str, task: str,
                    thinking: bool = False) -> Dict[str, Any]:
    """Call provider.generate(), track cost, parse JSON from response."""
    import re
    # Thinking tokens share the max_tokens budget — need much more headroom
    max_tok = 4000 if thinking else 300
    resp = provider.generate(prompt=prompt, system_prompt=system, max_tokens=max_tok,
                             temperature=0.0, thinking=thinking)
    cost_tracker.record_response(resp, task=task, query_id=query_id)
    text = resp.text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: extract first JSON object via brace matching
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in response: {text[:200]}")


def majority_vote(labels: Dict[str, Any]) -> tuple:
    """Majority vote over label dict. Returns (winner, agreement_type).

    agreement_type: 'unanimous' | 'majority' | 'no_consensus'
    """
    counts: Dict[Any, int] = {}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, c in counts.items() if c == max_count]
    if max_count == len(labels):
        return winners[0], "unanimous"
    elif max_count >= 2:
        return winners[0], "majority"
    else:
        # No consensus — return median for ints, first for strings
        vals = sorted(labels.values()) if all(isinstance(v, int) for v in labels.values()) else list(labels.values())
        return vals[len(vals) // 2], "no_consensus"


def read_qrels_tsv(path: Path) -> Dict[str, Dict[str, int]]:
    """Read BEIR-format qrels TSV."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("query_id"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    return qrels


def setup_logging(verbose: bool = False) -> None:
    """Shared logging setup for CLI scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml_config(path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
