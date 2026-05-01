# ob-eval

Evaluation-only fork of [kakao/OrchestrationBench](https://github.com/kakao/OrchestrationBench)
intended for use as a third-party submodule by harnesses such as
[nemo-skills-harness](https://github.daumkakao.com/lmt/nemo-skills-harness)
(BFCL-eval pattern).

## What's here

The three evaluation modules from upstream's `src/utils/evaluation/`:

| Module | Purpose |
|---|---|
| `ob_eval.evaluate_arguments` | F1 scoring of predicted vs. labelled function-call arguments (key-level + value-level via judge LLM) |
| `ob_eval.evaluate_workflow_as_DAG` | DAG-based workflow planning evaluation |
| `ob_eval.eval_utils` | Comprehensive analysis aggregation across scenarios |

## What's NOT here (vs. upstream)

- Generation pipeline (`src/orchestration_engine.py`, `src/step_history_generator.py`,
  `src/stepwise_scenario_processor.py`) — the consumer harness drives generation
- Agent + model implementations (`src/agents/`, `src/models/`)
- Config loading, token tracking, tool-call fetching
- The orchestration `loguru` setup — replaced with stdlib `logging`
- Upstream's `ModelFactory` — replaced with a generic `JudgeCallable`
  (`Callable[[list[dict]], str]`) so the consumer plugs in its own judge

## Refactor notes

The eval modules originally took `judge_model: ModelFactory`. In this fork,
that parameter is replaced with `judge_call: JudgeCallable` — a callable that
accepts a list of OpenAI-style chat messages and returns the assistant string.
The harness wraps its own judge infrastructure (multi-judge averaging,
back-off, etc.) into such a callable and injects it.

## Install

```bash
pip install -e /path/to/ob-eval
```

## License

Apache-2.0 (inherited from upstream).
