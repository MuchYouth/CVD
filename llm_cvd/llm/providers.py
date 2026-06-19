"""Experiment adapters around the shared ../llm_api clients.

The original llm_api package is left untouched. It expects modules.core.config
and modules.core.metrics to exist, so this adapter supplies those modules at
runtime from environment variables, then wraps provider clients with the richer
metadata interface needed by the RAG experiment.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class LLMResult:
    provider: str
    model: str
    text: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_sec: float
    error: str | None = None
    routed_provider: str | None = None
    routed_model: str | None = None
    fallback_attempts: int | None = None


class UsageTracker:
    """Small metrics object compatible with llm_api's get_metrics_tracker."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.model: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def record_llm_call(self, model: str, input_tokens: int | None, output_tokens: int | None) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


_TRACKER_LOCAL = threading.local()


def get_usage_tracker() -> UsageTracker:
    tracker = getattr(_TRACKER_LOCAL, "tracker", None)
    if tracker is None:
        tracker = UsageTracker()
        _TRACKER_LOCAL.tracker = tracker
    return tracker


MODEL_ENV_BY_PROVIDER = {
    "chatgpt": "OPENAI_MODEL",
    "claude": "CLAUDE_MODEL",
    "gemini": "GEMINI_MODEL",
    "grok": "GROK_MODEL",
    "freellm": "FREELLMAPI_MODEL",
}


def install_llm_api_shims() -> None:
    """Install modules.core.config and modules.core.metrics shims."""
    if "modules.core.config" in sys.modules and "modules.core.metrics" in sys.modules:
        return

    modules_pkg = sys.modules.setdefault("modules", types.ModuleType("modules"))
    core_pkg = sys.modules.setdefault("modules.core", types.ModuleType("modules.core"))
    config_mod = types.ModuleType("modules.core.config")
    metrics_mod = types.ModuleType("modules.core.metrics")

    config_mod.get_openai_api_key = lambda: os.environ["OPENAI_API_KEY"]
    config_mod.get_anthropic_api_key = lambda: os.environ["ANTHROPIC_API_KEY"]
    config_mod.get_google_api_key = lambda: os.environ["GOOGLE_API_KEY"]
    config_mod.get_grok_api_key = lambda: os.environ["XAI_API_KEY"]
    config_mod.get_openai_model = lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    config_mod.get_claude_model = lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    config_mod.get_gemini_model = lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    config_mod.get_grok_model = lambda: os.getenv("GROK_MODEL", "grok-4-1-fast-non-reasoning")
    config_mod.get_llm_provider = lambda: os.getenv("LLM_PROVIDER", "claude")
    metrics_mod.get_metrics_tracker = get_usage_tracker

    modules_pkg.core = core_pkg
    core_pkg.config = config_mod
    core_pkg.metrics = metrics_mod
    sys.modules["modules.core.config"] = config_mod
    sys.modules["modules.core.metrics"] = metrics_mod


class ExperimentClient:
    """Wrap an llm_api client and expose experiment metadata."""

    def __init__(
        self,
        provider: str,
        base_client: Any,
        model_override: str | None = None,
    ) -> None:
        self.provider = normalize_provider(provider)
        self.base_client = base_client
        if model_override:
            if hasattr(base_client, "model"):
                base_client.model = model_override
            if hasattr(base_client, "model_name"):
                base_client.model_name = model_override
                try:
                    import google.generativeai as genai

                    base_client.model = genai.GenerativeModel(model_override)
                except Exception:
                    pass
        self.model = get_client_model(base_client)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 8,
    ) -> LLMResult:
        start = time.perf_counter()
        tracker = get_usage_tracker()
        tracker.reset()
        try:
            text = self.base_client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - start
            input_tokens = tracker.input_tokens
            output_tokens = tracker.output_tokens
            total_tokens = (
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            )
            self.model = get_client_model(self.base_client)
            return LLMResult(
                provider=self.provider,
                model=self.model,
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_sec=latency,
            )
        except Exception as exc:
            return LLMResult(
                provider=self.provider,
                model=self.model,
                text=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_sec=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            pass


class FreeLLMAPIClient:
    """OpenAI-compatible experiment client for a local FreeLLMAPI proxy."""

    def __init__(self, model: str | None = None) -> None:
        try:
            import openai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The openai package is required for FreeLLMAPI. "
                "Install project requirements before running this provider."
            ) from exc

        self.provider = "freellm"
        self.model = model or os.getenv("FREELLMAPI_MODEL", "gpt-4o")
        self.base_url = os.getenv("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
        api_key = os.getenv("FREELLMAPI_API_KEY")
        if not api_key:
            raise RuntimeError("Set FREELLMAPI_API_KEY before using the freellm provider.")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 8,
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response, headers = self._create_chat_completion(messages, max_tokens)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens

            routed_provider, routed_model = parse_routed_via(headers.get("x-routed-via"))
            fallback_attempts = parse_int_header(headers.get("x-fallback-attempts"))
            return LLMResult(
                provider=self.provider,
                model=self.model,
                text=response.choices[0].message.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_sec=time.perf_counter() - start,
                routed_provider=routed_provider,
                routed_model=routed_model,
                fallback_attempts=fallback_attempts,
            )
        except Exception as exc:
            return LLMResult(
                provider=self.provider,
                model=self.model,
                text=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_sec=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _create_chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[Any, dict[str, str]]:
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        completions = self.client.chat.completions
        raw_completions = getattr(completions, "with_raw_response", None)
        if raw_completions is None:
            return completions.create(**create_kwargs), {}

        raw_response = raw_completions.create(**create_kwargs)
        headers = {key.lower(): value for key, value in raw_response.headers.items()}
        return raw_response.parse(), headers


def make_client(provider: str, model: str | None = None) -> Any:
    if normalize_provider(provider) == "freellm":
        return FreeLLMAPIClient(model=model)

    install_llm_api_shims()
    from llm_api.factory import get_llm_client

    base_client = get_llm_client(provider)
    return ExperimentClient(provider=provider, base_client=base_client, model_override=model)


def resolve_provider_model(alias: str, model_override: str | None = None) -> tuple[str, str | None]:
    """Resolve a provider alias or model name into provider plus optional model."""
    normalized = normalize_provider(alias)
    if normalized in MODEL_ENV_BY_PROVIDER:
        return normalized, model_override

    provider = infer_provider_from_model(alias)
    if provider:
        env_model = os.getenv(MODEL_ENV_BY_PROVIDER[provider], "")
        if model_override:
            return provider, model_override
        if env_model and alias == env_model:
            return provider, None
        return provider, alias

    return normalized, model_override


def normalize_provider(provider: str) -> str:
    lower = provider.lower()
    if lower in {"openai", "gpt"}:
        return "chatgpt"
    if lower in {"freellmapi", "free"}:
        return "freellm"
    if lower == "google":
        return "gemini"
    if lower == "xai":
        return "grok"
    return lower


def infer_provider_from_model(model: str) -> str | None:
    lower = model.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "chatgpt"
    if lower.startswith("claude-"):
        return "claude"
    if lower.startswith(("gemini-", "models/gemini-")):
        return "gemini"
    if lower.startswith("grok-"):
        return "grok"

    for provider, env_name in MODEL_ENV_BY_PROVIDER.items():
        env_model = os.getenv(env_name)
        if env_model and lower == env_model.lower():
            return provider
    return None


def get_client_model(client: Any) -> str:
    return str(getattr(client, "model_name", getattr(client, "model", "")))


def parse_routed_via(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if "/" not in value:
        return value, None
    provider, model = value.split("/", 1)
    return provider or None, model or None


def parse_int_header(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
