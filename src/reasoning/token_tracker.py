"""
Token usage and computational cost tracking for audit compliance.
"""

from dataclasses import dataclass, asdict
import json
import time
from typing import Dict, Any, Optional


@dataclass
class LLMUsageReport:
    model_name: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    execution_time_seconds: float
    rate_card_per_million_input: float
    rate_card_per_million_output: float


class TokenCostTracker:
    """
    Tracks and audits LLM inference consumption and financial expense.
    Supports free tier and published model rate cards.
    """

    # Published rates in USD per 1M tokens (as of standard public tiers)
    RATE_CARDS = {
        "gemini-1.5-flash-free": {"input": 0.0, "output": 0.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "llama-3-8b": {"input": 0.05, "output": 0.05},
        "deterministic-rule-engine": {"input": 0.0, "output": 0.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self, model_name: str = "deterministic-rule-engine"):
        self.model_name = model_name
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.start_time = time.time()

        rates = self.RATE_CARDS.get(model_name, {"input": 0.0, "output": 0.0})
        self.rate_in = rates["input"]
        self.rate_out = rates["output"]

    def record_call(self, input_tokens: int, output_tokens: int) -> None:
        """Logs tokens consumed by a single inference call."""
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_report(self) -> LLMUsageReport:
        """Calculates final aggregate usage and cost."""
        total_toks = self.total_input_tokens + self.total_output_tokens
        cost_in = (self.total_input_tokens / 1_000_000.0) * self.rate_in
        cost_out = (self.total_output_tokens / 1_000_000.0) * self.rate_out
        total_cost = cost_in + cost_out

        elapsed = time.time() - self.start_time

        return LLMUsageReport(
            model_name=self.model_name,
            total_calls=self.total_calls,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_tokens=total_toks,
            estimated_cost_usd=round(total_cost, 6),
            execution_time_seconds=round(elapsed, 3),
            rate_card_per_million_input=self.rate_in,
            rate_card_per_million_output=self.rate_out,
        )

    def save_log(self, filepath: str) -> None:
        """Exports report to JSON."""
        report = self.get_report()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
