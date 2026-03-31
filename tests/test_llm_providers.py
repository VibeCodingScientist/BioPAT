"""Tests for unified LLM provider module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from biopat.llm.providers import (
    LLMResponse,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    create_provider,
    _estimate_cost,
    PRICING,
)
from biopat.llm.cost_tracker import CostTracker, BudgetExceededError


# --- LLMResponse tests ---

class TestLLMResponse:
    def test_total_tokens(self):
        r = LLMResponse(
            text="hello", model="gpt-4o", provider="openai",
            input_tokens=100, output_tokens=50,
        )
        assert r.total_tokens == 150

    def test_defaults(self):
        r = LLMResponse(text="x", model="m", provider="p")
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.cost_usd == 0.0
        assert r.latency_ms == 0.0


# --- Cost estimation ---

class TestCostEstimation:
    def test_known_model(self):
        cost = _estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(12.50, abs=0.01)

    def test_unknown_model(self):
        cost = _estimate_cost("unknown-model-xyz", 1000, 1000)
        assert cost == 0.0

    def test_partial_match(self):
        cost = _estimate_cost("gpt-4o-mini", 1000, 1000)
        assert cost > 0

    def test_pricing_table_not_empty(self):
        assert len(PRICING) > 5


# --- OpenAI Provider ---

class TestOpenAIProvider:
    @patch("biopat.llm.providers.openai")
    def test_generate(self, mock_openai_module):
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  test response  "
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            result = provider.generate("hello", system_prompt="sys", max_tokens=100)

        assert result.text == "test response"
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost_usd > 0

    @patch("biopat.llm.providers.openai")
    def test_generate_json(self, mock_openai_module):
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            result = provider.generate_json("give me json")

        assert result == {"key": "value"}


# --- Anthropic Provider ---

class TestAnthropicProvider:
    @patch("biopat.llm.providers.anthropic")
    def test_generate(self, mock_anthropic_module):
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "  claude response  "
        mock_response.usage.input_tokens = 80
        mock_response.usage.output_tokens = 40
        mock_client.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            provider = AnthropicProvider(model="claude-sonnet-4-5-20250929", api_key="test-key")
            result = provider.generate("hello")

        assert result.text == "claude response"
        assert result.provider == "anthropic"
        assert result.input_tokens == 80


# --- Factory function ---

class TestCreateProvider:
    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown_provider")

    @patch("biopat.llm.providers.openai")
    def test_create_openai(self, mock_openai):
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            p = create_provider("openai", model="gpt-4o", api_key="key")
        assert p.provider_name == "openai"
        assert p.model == "gpt-4o"

    @patch("biopat.llm.providers.anthropic")
    def test_create_anthropic(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            p = create_provider("anthropic", model="claude-sonnet-4-5-20250929", api_key="key")
        assert p.provider_name == "anthropic"


# --- Cost Tracker ---

class TestCostTracker:
    def test_basic_tracking(self):
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", "hyde", "Q1", 100, 50, 0.05, 500)
        tracker.record("openai", "gpt-4o", "hyde", "Q2", 120, 60, 0.06, 600)

        assert tracker.total_cost == pytest.approx(0.11)
        assert tracker.total_calls == 2

    def test_budget_enforcement(self):
        tracker = CostTracker(max_budget_usd=0.10)
        tracker.record("openai", "gpt-4o", "hyde", "Q1", 100, 50, 0.08, 500)

        with pytest.raises(BudgetExceededError):
            tracker.record("openai", "gpt-4o", "hyde", "Q2", 100, 50, 0.05, 500)

    def test_summary(self):
        tracker = CostTracker(max_budget_usd=100.0)
        tracker.record("openai", "gpt-4o", "hyde", "Q1", 100, 50, 0.05, 500)
        tracker.record("anthropic", "claude", "rerank", "Q1", 200, 100, 0.10, 800)

        summary = tracker.get_summary()
        assert summary["total_cost_usd"] == pytest.approx(0.15)
        assert summary["total_calls"] == 2
        assert "hyde" in summary["by_task"]
        assert "rerank" in summary["by_task"]
        assert summary["budget_remaining_usd"] == pytest.approx(99.85)

    def test_per_query_cost(self):
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", "hyde", "Q1", 100, 50, 0.05, 500)
        tracker.record("openai", "gpt-4o", "rerank", "Q1", 200, 100, 0.10, 800)
        tracker.record("openai", "gpt-4o", "hyde", "Q2", 100, 50, 0.03, 500)

        per_query = tracker.get_per_query_cost()
        assert per_query["Q1"] == pytest.approx(0.15)
        assert per_query["Q2"] == pytest.approx(0.03)

        per_query_hyde = tracker.get_per_query_cost(task="hyde")
        assert per_query_hyde["Q1"] == pytest.approx(0.05)

    def test_save_load(self, tmp_path):
        tracker = CostTracker(max_budget_usd=50.0)
        tracker.record("openai", "gpt-4o", "hyde", "Q1", 100, 50, 0.05, 500)
        tracker.record("anthropic", "claude", "rerank", "Q2", 200, 100, 0.10, 800)

        save_path = str(tmp_path / "costs.json")
        tracker.save(save_path)

        loaded = CostTracker.load(save_path)
        assert loaded.total_cost == pytest.approx(0.15)
        assert loaded.total_calls == 2
        assert loaded.max_budget_usd == 50.0

    def test_record_response(self):
        tracker = CostTracker()
        response = LLMResponse(
            text="test", model="gpt-4o", provider="openai",
            input_tokens=100, output_tokens=50, cost_usd=0.05, latency_ms=500,
        )
        tracker.record_response(response, task="hyde", query_id="Q1")
        assert tracker.total_calls == 1
        assert tracker.total_cost == pytest.approx(0.05)
