"""API cost tracking and budget enforcement for LLM experiments."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when the cost budget is exceeded."""


@dataclass
class CallRecord:
    """Record of a single LLM API call."""

    provider: str
    model: str
    task: str
    query_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class CostTracker:
    """Tracks LLM API costs across experiments with optional budget enforcement.

    Example:
        tracker = CostTracker(max_budget_usd=500.0)
        tracker.record("openai", "gpt-4o", "hyde", "Q001", 500, 200, 0.003, 1200)
        print(tracker.get_summary())
    """

    def __init__(self, max_budget_usd: Optional[float] = None):
        self.max_budget_usd = max_budget_usd
        self.records: List[CallRecord] = []
        self._total_cost: float = 0.0

    def record(
        self,
        provider: str,
        model: str,
        task: str,
        query_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """Record an API call.

        Raises:
            BudgetExceededError: If adding this cost exceeds the budget.
        """
        if self.max_budget_usd is not None and (self._total_cost + cost_usd) > self.max_budget_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${self._total_cost + cost_usd:.2f} > ${self.max_budget_usd:.2f}"
            )

        self.records.append(
            CallRecord(
                provider=provider,
                model=model,
                task=task,
                query_id=query_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )
        )
        self._total_cost += cost_usd

    def record_response(self, response: Any, task: str, query_id: str) -> None:
        """Record from an LLMResponse object."""
        self.record(
            provider=response.provider,
            model=response.model,
            task=task,
            query_id=query_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_calls(self) -> int:
        return len(self.records)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics grouped by task and model."""
        by_task: Dict[str, Dict[str, Any]] = {}
        by_model: Dict[str, Dict[str, Any]] = {}

        for r in self.records:
            # Per task
            if r.task not in by_task:
                by_task[r.task] = {"calls": 0, "cost": 0.0, "tokens": 0}
            by_task[r.task]["calls"] += 1
            by_task[r.task]["cost"] += r.cost_usd
            by_task[r.task]["tokens"] += r.input_tokens + r.output_tokens

            # Per model
            key = f"{r.provider}/{r.model}"
            if key not in by_model:
                by_model[key] = {"calls": 0, "cost": 0.0, "tokens": 0}
            by_model[key]["calls"] += 1
            by_model[key]["cost"] += r.cost_usd
            by_model[key]["tokens"] += r.input_tokens + r.output_tokens

        return {
            "total_cost_usd": self._total_cost,
            "total_calls": len(self.records),
            "total_input_tokens": sum(r.input_tokens for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "budget_usd": self.max_budget_usd,
            "budget_remaining_usd": (
                self.max_budget_usd - self._total_cost if self.max_budget_usd else None
            ),
            "by_task": by_task,
            "by_model": by_model,
        }

    def get_per_query_cost(self, task: Optional[str] = None) -> Dict[str, float]:
        """Get cost per query, optionally filtered by task."""
        costs: Dict[str, float] = {}
        for r in self.records:
            if task and r.task != task:
                continue
            costs[r.query_id] = costs.get(r.query_id, 0.0) + r.cost_usd
        return costs

    def save(self, path: str) -> None:
        """Save records to JSON file."""
        data = {
            "summary": self.get_summary(),
            "records": [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "task": r.task,
                    "query_id": r.query_id,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": r.cost_usd,
                    "latency_ms": r.latency_ms,
                }
                for r in self.records
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Cost tracker saved to %s (total: $%.2f)", path, self._total_cost)

    @classmethod
    def load(cls, path: str) -> "CostTracker":
        """Load records from JSON file."""
        with open(path) as f:
            data = json.load(f)

        tracker = cls(max_budget_usd=data.get("summary", {}).get("budget_usd"))
        for r in data.get("records", []):
            tracker.record(
                provider=r["provider"],
                model=r["model"],
                task=r["task"],
                query_id=r["query_id"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                cost_usd=r["cost_usd"],
                latency_ms=r["latency_ms"],
            )
        return tracker
