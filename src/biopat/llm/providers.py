"""Unified LLM provider abstraction for OpenAI, Anthropic, and Google APIs.

Provides a consistent interface across providers with token tracking,
cost estimation, and latency measurement.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# Token pricing per million tokens (input, output) in USD
PRICING = {
    # OpenAI
    "gpt-5.2": (1.75, 14.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Google Gemini
    "gemini-3-pro-preview": (2.00, 12.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw_response: Optional[Any] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts."""
    pricing = PRICING.get(model)
    if pricing is None:
        # Try partial match
        for key, val in PRICING.items():
            if key in model or model in key:
                pricing = val
                break
    if pricing is None:
        return 0.0
    input_price, output_price = pricing
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient API error worth retrying."""
    # HTTP status-based errors from all three providers
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (429, 500, 502, 503, 529):
        return True
    # Connection / timeout errors
    err_name = type(exc).__name__.lower()
    for token in ("timeout", "connection", "reset", "broken", "overloaded"):
        if token in err_name or token in str(exc).lower():
            return True
    return False


def _retry_call(fn, *, max_retries: int = 5, base_delay: float = 2.0):
    """Call fn() with exponential backoff on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt < max_retries and _is_retryable(exc):
                delay = base_delay * (2 ** attempt)
                logger.warning("Retryable error (attempt %d/%d), waiting %.0fs: %s",
                               attempt + 1, max_retries, delay, exc)
                time.sleep(delay)
            else:
                raise


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_name: str = "base"

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        thinking: bool = False,
    ) -> LLMResponse:
        """Generate a text completion. If thinking=True, enable extended reasoning."""

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate and parse a JSON response."""
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-5.2, GPT-4o, etc.)."""

    provider_name = "openai"

    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None):
        super().__init__(model, api_key or os.environ.get("OPENAI_API_KEY"))
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        thinking: bool = False,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # GPT-5+ uses max_completion_tokens; older models use max_tokens
        token_key = "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            token_key: max_tokens,
        }
        if thinking:
            kwargs["reasoning_effort"] = "high"
        else:
            kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        response = _retry_call(lambda: self.client.chat.completions.create(**kwargs))
        latency = (time.perf_counter() - t0) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=self.model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            raw_response=response,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude Sonnet 4.6, Opus 4.6, etc.)."""

    provider_name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        super().__init__(model, api_key or os.environ.get("ANTHROPIC_API_KEY"))
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        thinking: bool = False,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            # Temperature must not be set with adaptive thinking
        elif temperature > 0:
            kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        response = _retry_call(lambda: self.client.messages.create(**kwargs))
        latency = (time.perf_counter() - t0) * 1000

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # With thinking enabled, response has multiple content blocks
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text.strip()
                break
        if not text and response.content:
            text = response.content[-1].text.strip() if hasattr(response.content[-1], "text") else ""

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            raw_response=response,
        )


class GoogleProvider(LLMProvider):
    """Google Gemini API provider."""

    provider_name = "google"

    def __init__(self, model: str = "gemini-3-pro-preview", api_key: Optional[str] = None):
        super().__init__(model, api_key or os.environ.get("GOOGLE_API_KEY"))
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")
        self._genai = genai
        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        thinking: bool = False,
    ) -> LLMResponse:
        config_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
        if thinking:
            tc_fields = self._genai.types.ThinkingConfig.model_fields
            if "thinking_level" in tc_fields:
                config_kwargs["thinking_config"] = self._genai.types.ThinkingConfig(thinking_level="HIGH")
            elif "thinking_budget" in tc_fields:
                config_kwargs["thinking_config"] = self._genai.types.ThinkingConfig(thinking_budget=-1)
            else:
                config_kwargs["thinking_config"] = self._genai.types.ThinkingConfig(include_thoughts=True)
        else:
            config_kwargs["temperature"] = temperature
        config = self._genai.types.GenerateContentConfig(**config_kwargs)
        if system_prompt:
            config.system_instruction = system_prompt

        t0 = time.perf_counter()
        response = _retry_call(lambda: self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        ))
        latency = (time.perf_counter() - t0) * 1000

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            text=response.text.strip() if response.text else "",
            model=self.model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            raw_response=response,
        )


# Provider registry
_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def create_provider(
    name: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        name: Provider name ("openai", "anthropic", "google").
        model: Model ID. If None, uses provider default.
        api_key: API key. If None, reads from environment.

    Returns:
        Configured LLMProvider instance.
    """
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDERS.keys())}")

    kwargs: Dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key

    return provider_cls(**kwargs)
