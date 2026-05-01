"""Shared protocols / type aliases for ob-eval.

The eval modules originally accepted ``ModelFactory`` instances (upstream's
LLM client wrapper). In this fork that's replaced with a generic
``JudgeCallable`` so consumers can plug in any judge implementation —
the harness has its own multi-judge infrastructure, retry/backoff, etc.

A ``JudgeCallable`` takes a list of OpenAI-style chat messages
(``[{"role": "system" | "user" | "assistant", "content": str}, ...]``)
and returns the assistant's reply as a string.
"""

from typing import Callable, Dict, List

JudgeCallable = Callable[[List[Dict[str, str]]], str]
