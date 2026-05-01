"""ob-eval — evaluation-only fork of kakao/OrchestrationBench.

Public surface:
- ``JudgeCallable``: protocol consumers implement to plug in a judge LLM
- the three eval modules (``evaluate_arguments``, ``evaluate_workflow_as_DAG``,
  ``eval_utils``) are exposed as submodules
"""

from ob_eval._types import JudgeCallable

__all__ = ["JudgeCallable"]
__version__ = "0.1.0"
