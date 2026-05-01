"""ob-eval — evaluation-only fork of kakao/OrchestrationBench.

Public surface:
- ``JudgeCallable``: protocol consumers implement to plug in a judge LLM
- ``build_judge_requests``: split the judge-prompt building out so consumers
  can drive their own judge pipeline (e.g. the harness's pre-cache pattern)
- ``score_with_judgements``: same return shape as
  ``evaluate_sub_agent_history_f1`` but consumes pre-cached judge responses
- the three eval modules are exposed as submodules
"""

from ob_eval._types import JudgeCallable
from ob_eval.evaluate_arguments import (
    build_judge_requests,
    evaluate_sub_agent_history_f1,
    score_with_judgements,
)

__all__ = [
    "JudgeCallable",
    "build_judge_requests",
    "evaluate_sub_agent_history_f1",
    "score_with_judgements",
]
__version__ = "0.2.0"
