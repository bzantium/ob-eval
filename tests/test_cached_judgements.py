import asyncio

import pytest

from ob_eval import evaluate_arguments as scoring


@pytest.mark.parametrize(
    "eligible, results, key_scores, expected_value, expected_perfect",
    [
        ([1], [{"call_index": 0, "f1": 1.0}], [0.0, 1.0], 1.0, 1),
        (
            [1, 3],
            [{"call_index": 1, "f1": 0.5}, {"call_index": 0, "f1": 1.0}],
            [0.0, 1.0, 0.0, 1.0],
            0.75,
            1,
        ),
        (
            [0, 1],
            [{"call_index": 1, "f1": 0.5}, {"call_index": 0, "f1": 1.0}],
            [1.0, 1.0],
            0.75,
            1,
        ),
    ],
)
def test_cached_scores_survive_filtered_calls(
    monkeypatch, eligible, results, key_scores, expected_value, expected_perfect
):
    # Earlier rejected/failed calls remain in the original arrays, while the
    # judge sees compact indices. Response order need not equal call order.
    ctx = {
        "predicted_calls": [{} for _ in key_scores],
        "actual_calls": [{} for _ in key_scores],
        "llm_eligible_indices": eligible,
        "llm_eligible_predicted": [{} for _ in eligible],
        "llm_eligible_actual": [{} for _ in eligible],
        "llm_eligible_history": [[] for _ in eligible],
        "system_info": "",
        "key_score_result": {"detailed_key_score": key_scores},
    }
    monkeypatch.setattr(
        scoring,
        "build_evaluation_requests",
        lambda *a, **k: {
            "metadata": [{} for _ in eligible],
            "auto_only_results": [],
        },
    )
    monkeypatch.setattr(
        scoring, "process_evaluation_responses", lambda *a, **k: results
    )
    monkeypatch.setattr(scoring, "is_rejection_case", lambda *a: False)
    value, perfect, success = asyncio.run(
        scoring._value_score_from_judgements(
            ctx,
            {},
            {},
            ["answer" for _ in eligible],
        )
    )
    assert value == expected_value
    assert perfect == expected_perfect
    assert success
