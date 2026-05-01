import asyncio
import glob
import json
import logging
import os
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Set

import yaml

from ob_eval._types import JudgeCallable
from ob_eval.eval_utils import (
    _call_llm_with_semaphore,
    _process_tool_call_arguments,
    analyze_argument_matches,
    calculate_f1_score,
    combine_analysis_with_llm_results,
    create_history,
    create_llm_judge_prompt,
    extract_both_tool_calls,
    extract_content_from_llm_result,
    is_failed_case,
    is_rejection_case,
    parse_arguments_to_keys,
    save_llm_results,
)

logger = logging.getLogger(__name__)

def process_single_llm_result(llm_result, metadata, call_index):
    """Process individual LLM result and return evaluation metrics"""
    try:
        # Parse LLM judgment results
        llm_judge_results = {}
        if llm_result and not isinstance(llm_result, Exception):
            try:
                # Handle different response structures
                content = extract_content_from_llm_result(llm_result)                
                # Extract JSON from content - try multiple patterns
                json_str = ""
                
                # Pattern 1: Look for {key: value} format
                content = content.strip().split("</think>")[-1]
                json_matches = re.findall(r'\{.*\}', content, re.DOTALL| re.MULTILINE)
                if json_matches:
                    json_str = json_matches[-1]
                    # Try to parse as-is first
                    try:
                        llm_judge_results = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        fixed_json = json_str.replace("'", '"')
                        try:
                            llm_judge_results = json.loads(fixed_json)
                        except json.JSONDecodeError as e2:
                            llm_judge_results = {}
                else:
                    print(f"No JSON matches found in content")
                
            except Exception as e:
                print(f"Exception while parsing LLM response: {e}")
                llm_judge_results = {}
        
        elif isinstance(llm_result, Exception):
            logger.error(f"LLM call failed for call {call_index}: {llm_result}")
            llm_judge_results = {}
        
        analysis = {
            'auto_results': metadata['auto_results'],
            'llm_needed': metadata['llm_needed']
        }

        final_result = combine_analysis_with_llm_results(analysis, llm_judge_results, metadata['number_mapping'])
        result = {
            'call_index': call_index,
            'precision': final_result['metrics']['precision'],
            'recall': final_result['metrics']['recall'],
            'f1': final_result['metrics']['f1']
        }

        # Create eval_data for logging
        eval_data = {
            'call_index': call_index, 
            'p_call': metadata['p_call'], 
            'a_call': metadata['a_call'], 
            'auto_results': metadata['auto_results'], 
            'llm_needed': metadata['llm_needed'], 
            'number_mapping': metadata['number_mapping'], 
            'messages': metadata['history']
        }   
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing call {call_index}: {e}")
        return {
            'call_index': call_index,
            'error': str(e),
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }

def build_evaluation_requests(predicted_calls: List[Dict],
                              actual_calls: List[Dict],
                              eval_config: Dict,
                              system_info: str = "",
                              llm_history: List[Dict] = None,
                              tool_descriptions: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build per-call LLM judge requests without invoking the judge.

    Returns a dict with three keys:
    - ``requests``: list of OpenAI-style ``messages`` lists, one per call that
      needs LLM judging
    - ``metadata``: parallel list of metadata dicts that
      ``process_evaluation_responses`` consumes to merge auto-results with
      the judge's verdicts
    - ``auto_only_results``: pre-computed result dicts for calls that didn't
      need LLM judging (rejection cases, failed predictions, or all-auto
      argument matches)
    """
    skip_llm_eval = eval_config.get("skip_llm_eval", False)
    requests: List[List[Dict]] = []
    metadata: List[Dict] = []
    auto_only_results: List[Dict] = []
    if not predicted_calls or not actual_calls:
        return {"requests": requests, "metadata": metadata, "auto_only_results": auto_only_results}

    for global_idx in range(len(predicted_calls)):
        p_call = predicted_calls[global_idx]
        a_call = actual_calls[global_idx]
        history = llm_history[global_idx] if llm_history else None
        call_index = global_idx
        applied_tool_desc = {
            function_name: tool_descriptions[function_name]
            for function_name in p_call["function_name"]
            if function_name in tool_descriptions
        }

        if (
            is_rejection_case(a_call, "label")
            or is_rejection_case(p_call, "prediction")
            or is_failed_case(p_call)
        ):
            continue

        analysis = analyze_argument_matches(p_call, a_call, applied_tool_desc)
        auto_results = analysis["auto_results"]
        llm_needed = analysis["llm_needed"]

        if not llm_needed or skip_llm_eval:
            final_result = combine_analysis_with_llm_results(analysis, {}, {})
            auto_only_results.append({
                "call_index": call_index,
                "precision": final_result["metrics"]["precision"],
                "recall": final_result["metrics"]["recall"],
                "f1": final_result["metrics"]["f1"],
            })
            continue

        llm_prompt, number_mapping = create_llm_judge_prompt(llm_needed)
        system_prompt = eval_config["prompts"]["arguments_evaluation"]["prompt"]
        system_prompt = system_prompt.replace("%%system_info%%", system_info)
        system_prompt = system_prompt.replace(
            "%%tool_description%%", json.dumps(applied_tool_desc, indent=2, ensure_ascii=False)
        )
        if history:
            history.insert(0, {"role": "system", "content": system_prompt})
        else:
            history = [{"role": "system", "content": system_prompt}]
        llm_prompt = "</history>\n\n" + llm_prompt
        history = history + [{"role": "user", "content": llm_prompt}]

        # ``[ID: ...]`` pairs the response back to the request — when the
        # judge processes a batch of requests, the IDs let us re-align by
        # content rather than relying on the call-time ordering.
        batch_id = f"EVAL_BATCH_{call_index}_{global_idx}"
        history_with_id = history.copy()
        history_with_id[-1]["content"] = (
            f"[ID: {batch_id}]\n\n"
            + history_with_id[-1]["content"]
            + f"\n\n**IMPORTANT: Always start your response with [ID: {batch_id}]**"
        )

        requests.append(history_with_id)
        metadata.append({
            "call_index": call_index,
            "batch_id": batch_id,
            "p_call": p_call,
            "a_call": a_call,
            "auto_results": auto_results,
            "llm_needed": llm_needed,
            "number_mapping": number_mapping,
            "applied_tool_desc": applied_tool_desc,
            "history": history,
        })

    return {"requests": requests, "metadata": metadata, "auto_only_results": auto_only_results}


def process_evaluation_responses(metadata: List[Dict],
                                 llm_results: List[Any],
                                 auto_only_results: List[Dict] = None,
                                 save_results: bool = False,
                                 output_dir: str = None,
                                 file_identifier: str = None,
                                 input_data_for_saving: List[Dict] = None) -> List[Dict]:
    """Merge auto-results with judge responses to produce per-call F1 dicts.

    ``llm_results[i]`` is the judge's response for ``metadata[i]``'s request.
    A response can be the OpenAI-style dict from ``_call_llm`` or a raw
    string — ``extract_content_from_llm_result`` handles both.
    """
    all_results: List[Dict] = list(auto_only_results or [])
    llm_responses: List[Dict] = []
    for llm_result, meta in zip(llm_results, metadata):
        call_index = meta["call_index"]
        batch_id = meta["batch_id"]
        try:
            response_id = None
            if llm_result and not isinstance(llm_result, Exception):
                content = extract_content_from_llm_result(llm_result)
                id_match = re.search(r"\[ID:\s*([^\]]+)\]", content)
                if id_match:
                    response_id = id_match.group(1).strip()
            if response_id != batch_id:
                logger.error(
                    "ID mismatch for call %s. Expected %s, got %s. "
                    "Falling back to position-based matching.",
                    call_index, batch_id, response_id,
                )
            result = process_single_llm_result(llm_result, meta, call_index)
            all_results.append(result)
            if save_results:
                llm_responses.append({
                    "call_index": call_index,
                    "llm_response": llm_result,
                    "parsed_judgement": result.get("parsed_judgement", {}),
                    "final_metrics": result,
                })
        except Exception as e:
            logger.error("Error processing batch result %s: %s", call_index, e)
            all_results.append({
                "call_index": call_index,
                "error": str(e),
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            })

    if save_results and output_dir and file_identifier and input_data_for_saving is not None:
        save_llm_results(input_data_for_saving, llm_responses, output_dir, file_identifier)

    return all_results


async def call_evaluation_llm(predicted_calls: List[Dict],
                                actual_calls: List[Dict],
                                eval_config: Dict,
                                judge_call: JudgeCallable,
                                system_info: str = "",
                                llm_history: List[Dict] = None,
                                tool_descriptions: Dict[str, Any] = None,
                                save_results: bool = False,
                                output_dir: str = None,
                                file_identifier: str = None) -> List[Dict]:
    """Live-judge variant: build → call judge → process. Kept for backwards
    compatibility; consumers preferring a pre-cached judge pipeline should
    call ``build_evaluation_requests`` and ``process_evaluation_responses``
    directly.
    """
    built = build_evaluation_requests(
        predicted_calls, actual_calls, eval_config,
        system_info=system_info, llm_history=llm_history,
        tool_descriptions=tool_descriptions,
    )
    requests = built["requests"]
    metadata = built["metadata"]
    auto_only_results = built["auto_only_results"]

    if not requests:
        return auto_only_results

    batch_size = eval_config.get("batch_size", 8)
    llm_results: List[Any] = []
    input_data_for_saving: List[Dict] = []
    if save_results:
        for meta in metadata:
            input_data_for_saving.append({
                "call_index": meta["call_index"],
                "predicted_call": meta["p_call"],
                "actual_call": meta["a_call"],
                "auto_results": meta["auto_results"],
                "messages": meta["history"],
            })
    for batch_start in range(0, len(requests), batch_size):
        batch_end = min(batch_start + batch_size, len(requests))
        tasks = [_call_llm_with_semaphore(req, judge_call) for req in requests[batch_start:batch_end]]
        llm_results.extend(await asyncio.gather(*tasks, return_exceptions=True))

    return process_evaluation_responses(
        metadata, llm_results, auto_only_results=auto_only_results,
        save_results=save_results, output_dir=output_dir, file_identifier=file_identifier,
        input_data_for_saving=input_data_for_saving,
    )

def calculate_function_name_score(predicted_calls: List[Dict], actual_calls: List[Dict], total_label_function_call_count: int) -> Dict[str, Any]:
    """Calculate function name accuracy score by comparing predicted and actual function calls.
    This function evaluates the accuracy of function name predictions by computing F1 scores
    for valid function calls and tracking rejection cases. The input lists contain only 
    function calls that have matching function names between predictions and labels to 
    ensure valid comparisons.
    Args:
        predicted_calls (List[Dict]): List of predicted function call dictionaries.
        actual_calls (List[Dict]): List of actual function call dictionaries.
        total_label_function_call_count (int): Total number of function calls in the label for reference."""
    total_f1 = 0.0
    valid_function_calls = 0
    total_cnt = 0
    for idx in range(len(predicted_calls)):
        p_call = predicted_calls[idx]
        a_call = actual_calls[idx]
        
        # Only calculate F1 for valid function calls (not rejection or failed cases)
        should_reject = is_rejection_case(a_call, "label")
        predicted_reject = is_rejection_case(p_call, "prediction")
        is_failed = is_failed_case(p_call)
        
        if not should_reject and not predicted_reject and not is_failed:
            p_call_function_name = [p_call["function_name"]] if isinstance(p_call["function_name"], str) else p_call["function_name"]
            a_call_function_name = [a_call["function_name"]] if isinstance(a_call["function_name"], str) else a_call["function_name"]
            temp_scores = calculate_f1_score(p_call_function_name, a_call_function_name)
            total_f1 += temp_scores["f1-score"]
            valid_function_calls += 1
            total_cnt += 1

    # Calculate average F1 score only counting succeeded function calls
    avg_f1_score = total_f1 / total_cnt if total_cnt > 0 else 0.0
    
    return {
        "f1-score": round(avg_f1_score, 4),
        "details": {
            "valid_function_calls": valid_function_calls,
            "total_label_function_call_count": total_label_function_call_count
        }
    }
def calculate_key_score(predicted_calls: List[Dict], actual_calls: List[Dict]) -> Dict[str, float]:
    """
    Calculate key score for matched tool call pairs except rejection/failed cases.
    """
    total_f1 = 0.0    
    actual_function_calls = 0
    detailed_key_score = []
    
    for idx in range(len(predicted_calls)):
        p_call = predicted_calls[idx]
        a_call = actual_calls[idx]

        # Check if this should be a rejection case
        should_reject = is_rejection_case(a_call, "label")
        predicted_reject = is_rejection_case(p_call, "prediction")

        # Only calculate for non-rejected cases
        if not should_reject and not predicted_reject:
            p_keys = parse_arguments_to_keys(p_call["arguments"])
            a_keys = parse_arguments_to_keys(a_call["arguments"])
            scores = calculate_f1_score(p_keys, a_keys)
            total_f1 += scores["f1-score"]
            actual_function_calls += 1
            detailed_key_score.append(scores["f1-score"])
        else:
            # rejection case에도 점수를 추가 (일관성을 위해)
            detailed_key_score.append(0.0)
    
    average_f1 = total_f1 / actual_function_calls if actual_function_calls > 0 else 0.0
    return {
        "avg_f1": round(average_f1, 4),
        "detailed_key_score": detailed_key_score
    }


async def calculate_argument_value_scores(predicted_calls: List[Dict],
                                         actual_calls: List[Dict],
                                         histories: List[Dict],
                                         eval_config: Dict,
                                         system_info: Dict,
                                         tool_descriptions: Dict,
                                         save_llm_results: bool, 
                                         output_dir: str, 
                                         file_identifier: str, 
                                         trial: int, 
                                         key_score_result: Dict[str, float],
                                         judge_call: JudgeCallable = None,
                                         ):
    """
    Calculate value scores for predicted and actual calls using LLM evaluation.
    """
    detailed_key_score = key_score_result["detailed_key_score"]
    detailed_value_score = []
    
    # 모든 케이스에 대해 초기 스코어 리스트 생성
    for idx in range(len(predicted_calls)):
        detailed_value_score.append(0.0)
    
    try:
        # Filter out rejection cases for LLM evaluation
        llm_eval_pairs = []
        llm_eval_indices = []  # LLM 평가할 인덱스 추적
        
        for idx in range(len(predicted_calls)):
            p_call = predicted_calls[idx]
            a_call = actual_calls[idx]
            history = histories[idx] 
            should_reject = is_rejection_case(a_call, "label")
            predicted_reject = is_rejection_case(p_call, "prediction")
            is_predicted_failed = is_failed_case(p_call)
            
            if not should_reject and not predicted_reject:
                if not is_predicted_failed:
                    # Both are valid function calls - evaluate with LLM
                    llm_eval_pairs.append((p_call, a_call, history))
                    llm_eval_indices.append(idx)
                # else: failed case는 이미 0.0으로 초기화됨
            # else: rejection case는 이미 0.0으로 초기화됨

        # Evaluate valid function call pairs with LLM
        llm_success = False
        if llm_eval_pairs:
            try:
                llm_predicted = [pair[0] for pair in llm_eval_pairs]
                llm_actual = [pair[1] for pair in llm_eval_pairs]
                llm_history = [pair[2] for pair in llm_eval_pairs]

                # Create file identifier for this trial
                trial_file_identifier = f"{file_identifier}_trial_{trial}" if file_identifier else f"trial_{trial}"
                argument_eval = await call_evaluation_llm(
                    llm_predicted,
                    llm_actual,
                    eval_config,
                    judge_call,
                    system_info=system_info,
                    llm_history=llm_history,
                    tool_descriptions=tool_descriptions,
                    save_results=save_llm_results,
                    output_dir=output_dir,
                    file_identifier=trial_file_identifier
                )
                
                if argument_eval:
                    # LLM 결과를 해당 인덱스에 할당
                    for i, result in enumerate(argument_eval):
                        if i < len(llm_eval_indices):
                            idx = llm_eval_indices[i]
                            detailed_value_score[idx] = result.get('f1', 0.0)
                    llm_success = True
                    
            except Exception as llm_error:
                logger.warning(f"LLM evaluation failed, using fallback: {llm_error}")
                logger.debug(f"Detailed traceback: {traceback.format_exc()}")
                llm_success = False

        # LLM이 실패한 경우 fallback 처리
        if not llm_success and llm_eval_pairs:
            for i, (p_call, a_call, _) in enumerate(llm_eval_pairs):
                if i < len(llm_eval_indices):
                    idx = llm_eval_indices[i]
                    p_keys = parse_arguments_to_keys(p_call["arguments"])
                    a_keys = parse_arguments_to_keys(a_call["arguments"])
                    scores = calculate_f1_score(p_keys, a_keys)
                    detailed_value_score[idx] = scores["f1-score"]


        # Calculate average value score (except rejection/failed cases)
        valid_value_scores = [score for score in detailed_value_score if score > 0]
        avg_value_score = sum(valid_value_scores) / len(valid_value_scores) if valid_value_scores else 0.0
        
        perfect_fc_call_result = 0
        min_length = min(len(detailed_key_score), len(detailed_value_score))

        for i in range(min_length):
            if (detailed_key_score[i] == 1.0 and 
                detailed_value_score[i] == 1.0 and
                not is_rejection_case(actual_calls[i], "label")):
                perfect_fc_call_result += 1

        return avg_value_score, perfect_fc_call_result, llm_success
            
    except Exception as e:
        logger.error(f"Error in argument evaluation for trial {trial}: {e}")
        logger.error(traceback.format_exc())
        
        # 완전한 fallback: key-based scoring만 사용
        fallback_value_scores = [0.0] * len(predicted_calls)
        function_call_count = 0
        total_fallback_score = 0.0
        
        for idx in range(len(predicted_calls)):
            p_call = predicted_calls[idx]
            a_call = actual_calls[idx]
            should_reject = is_rejection_case(a_call, "label")
            predicted_reject = is_rejection_case(p_call, "prediction")
            is_predicted_failed = is_failed_case(p_call)
            
            if not should_reject and not predicted_reject and not is_predicted_failed:
                # Both are valid function calls
                p_keys = parse_arguments_to_keys(p_call["arguments"])
                a_keys = parse_arguments_to_keys(a_call["arguments"])
                scores = calculate_f1_score(p_keys, a_keys)
                fallback_value_scores[idx] = scores["f1-score"]
                total_fallback_score += scores["f1-score"]
                function_call_count += 1
        
        # Count successful calls
        min_length = min(len(detailed_key_score), len(fallback_value_scores))
        for i in range(min_length):
            if detailed_key_score[i] == 1.0 and fallback_value_scores[i] == 1.0:
                all_fc_successful_calls += 1
        
        avg_fallback_score = total_fallback_score / function_call_count if function_call_count > 0 else 0.0
        return avg_fallback_score, all_fc_successful_calls, False


async def evaluate_sub_agent_history_f1(data: Dict,
                                        eval_config: Dict,
                                        tool_descriptions: Dict,
                                        judge_call: JudgeCallable = None,
                                        save_llm_results: bool = False,
                                        output_dir: str = None,
                                        file_identifier: str = None,) -> Dict:
    """Live-judge variant of the F1 scoring.

    For pre-cached judge responses (the harness's standard judge-pipeline
    pattern), call ``build_judge_requests`` to obtain the prompts and
    ``score_with_judgements`` to compute the same metrics from the
    responses without an active judge connection.
    """
    return await _aggregate_trials(
        data, eval_config, tool_descriptions,
        judge_call=judge_call,
        save_llm_results=save_llm_results,
        output_dir=output_dir,
        file_identifier=file_identifier,
    )

def _walk_trials(data: Dict, eval_config: Dict, tool_descriptions: Dict):
    """Yield ``(trial, ctx)`` where ``ctx`` collects everything a per-trial
    scoring or request-build step needs: extracted tool calls, histories,
    pre-computed key/function-name scores, plus the rejection / FC
    confusion-matrix counters that aggregate at the end of
    ``evaluate_sub_agent_history_f1`` / ``score_with_judgements``.
    """
    for trial in data["history"]:
        data_run = data["history"][trial]
        sub_agent_history = data_run["sub_agent_history"]
        label_history = data_run["label"]
        system_info = label_history["1"]["content"] if label_history["1"]["role"] == "system" else ""

        (predicted_calls, actual_calls, step_list,
            false_negative_reject, false_positive_reject,
            true_positive_reject, true_negative_reject,
            true_positive_fc, false_positive_fc, false_negative_fc, true_negative_fc,
            total_label_reject_cases, total_label_fc_cases,
            rejection_type_mismatch, failed_generation) = \
                extract_both_tool_calls(sub_agent_history, label_history)
        assert len(predicted_calls) == len(actual_calls), \
            f"Mismatch in number of tool calls for trial {trial}"

        histories = [create_history(data_run, step) for step in step_list]
        key_score_result = calculate_key_score(predicted_calls, actual_calls)
        function_name_result = calculate_function_name_score(
            predicted_calls, actual_calls, total_label_fc_cases
        )

        # Filter to pairs that the LLM judge will actually evaluate (same
        # rejection/failed gates ``calculate_argument_value_scores`` uses).
        llm_eligible_indices: List[int] = []
        llm_eligible_predicted: List[Dict] = []
        llm_eligible_actual: List[Dict] = []
        llm_eligible_history: List[Dict] = []
        for idx in range(len(predicted_calls)):
            p_call = predicted_calls[idx]
            a_call = actual_calls[idx]
            if (
                not is_rejection_case(a_call, "label")
                and not is_rejection_case(p_call, "prediction")
                and not is_failed_case(p_call)
            ):
                llm_eligible_indices.append(idx)
                llm_eligible_predicted.append(p_call)
                llm_eligible_actual.append(a_call)
                llm_eligible_history.append(histories[idx])

        ctx = {
            "predicted_calls": predicted_calls,
            "actual_calls": actual_calls,
            "histories": histories,
            "system_info": system_info,
            "key_score_result": key_score_result,
            "function_name_result": function_name_result,
            "llm_eligible_indices": llm_eligible_indices,
            "llm_eligible_predicted": llm_eligible_predicted,
            "llm_eligible_actual": llm_eligible_actual,
            "llm_eligible_history": llm_eligible_history,
            # confusion-matrix counters (forwarded as-is to the aggregator)
            "false_negative_reject": false_negative_reject,
            "false_positive_reject": false_positive_reject,
            "true_positive_reject": true_positive_reject,
            "true_negative_reject": true_negative_reject,
            "true_positive_fc": true_positive_fc,
            "false_positive_fc": false_positive_fc,
            "false_negative_fc": false_negative_fc,
            "true_negative_fc": true_negative_fc,
            "total_label_reject_cases": total_label_reject_cases,
            "total_label_fc_cases": total_label_fc_cases,
            "rejection_type_mismatch": rejection_type_mismatch,
            "failed_generation": failed_generation,
        }
        yield trial, ctx


def build_judge_requests(data: Dict, eval_config: Dict, tool_descriptions: Dict) -> Dict[str, Dict[str, Any]]:
    """Per-scenario, build all argument-evaluation judge prompts grouped by trial.

    Returns ``{trial: {"requests": [...messages...], "metadata": [...]}}``.
    The harness fans the requests out through its standard judge pipeline
    (one call per request) and then passes the responses to
    ``score_with_judgements`` along with the same ``data`` / ``eval_config``
    / ``tool_descriptions`` triple.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for trial, ctx in _walk_trials(data, eval_config, tool_descriptions):
        if not ctx["llm_eligible_indices"]:
            out[trial] = {"requests": [], "metadata": []}
            continue
        built = build_evaluation_requests(
            ctx["llm_eligible_predicted"], ctx["llm_eligible_actual"], eval_config,
            system_info=ctx["system_info"],
            llm_history=ctx["llm_eligible_history"],
            tool_descriptions=tool_descriptions,
        )
        out[trial] = {"requests": built["requests"], "metadata": built["metadata"]}
    return out


async def score_with_judgements(
    data: Dict,
    eval_config: Dict,
    tool_descriptions: Dict,
    judgements: Dict[str, List[Any]],
) -> Dict:
    """Same return shape as ``evaluate_sub_agent_history_f1`` but consumes
    pre-cached judge responses instead of calling a live judge.

    ``judgements[trial]`` is the response list aligned with the request list
    from ``build_judge_requests(...)[trial]["requests"]`` — element ``i`` is
    the judge's reply to request ``i``. Each response can be a raw string,
    an OpenAI-style ``{"choices": [{"message": {"content": ...}}]}`` dict,
    or any value ``extract_content_from_llm_result`` accepts.
    """
    return await _aggregate_trials(
        data, eval_config, tool_descriptions,
        judge_call=None, judgements=judgements,
    )


async def _aggregate_trials(
    data: Dict,
    eval_config: Dict,
    tool_descriptions: Dict,
    judge_call: JudgeCallable = None,
    judgements: Dict[str, List[Any]] = None,
    save_llm_results: bool = False,
    output_dir: str = None,
    file_identifier: str = None,
) -> Dict:
    """Shared trial-walking aggregation behind both the live-judge path
    (``evaluate_sub_agent_history_f1``) and the pre-cached path
    (``score_with_judgements``)."""
    num_runs = len(data["history"])
    totals = {
        "key": 0.0, "fname": 0.0, "value": 0.0,
        "rej_cases": 0.0, "fc_cases": 0.0,
        "tp_rej": 0.0, "tn_rej": 0.0, "fp_rej": 0.0, "fn_rej": 0.0,
        "tp_fc": 0, "fp_fc": 0, "fn_fc": 0, "tn_fc": 0,
        "rej_mismatch": 0, "failed_gen": 0,
        "perfect_calls": 0.0, "fc_succ": 0.0,
    }
    llm_failed_trials: List[str] = []

    for trial, ctx in _walk_trials(data, eval_config, tool_descriptions):
        totals["key"] += ctx["key_score_result"]["avg_f1"]
        totals["fname"] += ctx["function_name_result"]["f1-score"]
        totals["rej_cases"] += ctx["total_label_reject_cases"]
        totals["fc_cases"] += ctx["total_label_fc_cases"]
        totals["tp_rej"] += ctx["true_positive_reject"]
        totals["tn_rej"] += ctx["true_negative_reject"]
        totals["fp_rej"] += ctx["false_positive_reject"]
        totals["fn_rej"] += ctx["false_negative_reject"]
        totals["tp_fc"] += ctx["true_positive_fc"]
        totals["fp_fc"] += ctx["false_positive_fc"]
        totals["fn_fc"] += ctx["false_negative_fc"]
        totals["tn_fc"] += ctx["true_negative_fc"]
        totals["rej_mismatch"] += ctx["rejection_type_mismatch"]
        totals["failed_gen"] += ctx["failed_generation"]

        if judgements is not None:
            value_score, perfect, success = await _value_score_from_judgements(
                ctx, eval_config, tool_descriptions, judgements.get(trial, []),
            )
        else:
            value_score, perfect, success = await calculate_argument_value_scores(
                ctx["predicted_calls"], ctx["actual_calls"], ctx["histories"],
                eval_config, ctx["system_info"], tool_descriptions,
                save_llm_results, output_dir, file_identifier, trial,
                ctx["key_score_result"], judge_call,
            )
        totals["perfect_calls"] += perfect
        totals["value"] += value_score
        totals["fc_succ"] += ctx["true_negative_reject"]
        if not success:
            llm_failed_trials.append(trial)

    return {
        "total_perfect_function_calls": totals["perfect_calls"],
        "total_fc_successful_calls": totals["fc_succ"],
        "key_score": {"avg_f1": round(totals["key"] / num_runs, 4)},
        "value_score_not_count_rejection": {"avg_f1": round(totals["value"] / num_runs, 4)},
        "function_name_score": {"f1-score": round(totals["fname"] / num_runs, 4)},
        "total_rejection_cases": totals["rej_cases"],
        "total_true_positive_reject": totals["tp_rej"],
        "total_true_negative_reject": totals["tn_rej"],
        "total_false_positive_reject": totals["fp_rej"],
        "total_false_negative_reject": totals["fn_rej"],
        "total_true_positive_fc": totals["tp_fc"],
        "total_false_positive_fc": totals["fp_fc"],
        "total_false_negative_fc": totals["fn_fc"],
        "total_true_negative_fc": totals["tn_fc"],
        "total_rejection_type_mismatch": totals["rej_mismatch"],
        "total_failed_generation": totals["failed_gen"],
        "llm_failed_trials": llm_failed_trials,
        "total_function_call_cases": totals["fc_cases"],
    }


async def _value_score_from_judgements(
    ctx: Dict,
    eval_config: Dict,
    tool_descriptions: Dict,
    trial_judgements: List[Any],
):
    """Per-trial scoring path for ``score_with_judgements``: rebuild the
    request metadata so we can pair pre-cached responses to call indices,
    then merge with auto-only results.
    """
    detailed_value_score = [0.0] * len(ctx["predicted_calls"])
    if not ctx["llm_eligible_indices"]:
        return 0.0, 0, True

    built = build_evaluation_requests(
        ctx["llm_eligible_predicted"], ctx["llm_eligible_actual"], eval_config,
        system_info=ctx["system_info"],
        llm_history=ctx["llm_eligible_history"],
        tool_descriptions=tool_descriptions,
    )
    metadata = built["metadata"]
    auto_only = built["auto_only_results"]

    if len(trial_judgements) < len(metadata):
        logger.warning(
            "trial received %d judgements but expected %d; missing entries scored as failures",
            len(trial_judgements), len(metadata),
        )
        trial_judgements = list(trial_judgements) + [None] * (len(metadata) - len(trial_judgements))

    results = process_evaluation_responses(metadata, trial_judgements[: len(metadata)], auto_only_results=auto_only)

    # Map per-call results back onto the original eligible indices
    sub_agent_index_to_orig = {p_idx: orig_idx for orig_idx, p_idx in enumerate(ctx["llm_eligible_indices"])}
    for r in results:
        sub_idx = r["call_index"]
        if sub_idx in sub_agent_index_to_orig:
            detailed_value_score[ctx["llm_eligible_indices"][sub_idx]] = r.get("f1", 0.0)

    valid = [s for s in detailed_value_score if s > 0]
    avg_value = sum(valid) / len(valid) if valid else 0.0

    perfect = 0
    detailed_key = ctx["key_score_result"]["detailed_key_score"]
    for i in range(min(len(detailed_key), len(detailed_value_score))):
        if (
            detailed_key[i] == 1.0
            and detailed_value_score[i] == 1.0
            and not is_rejection_case(ctx["actual_calls"][i], "label")
        ):
            perfect += 1
    return avg_value, perfect, True


if __name__ == "__main__":
    import asyncio
    
    async def main():
        file_path = 'data/results/step_wise_evaluation_EN/claude-sonnet-4/16_out.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        eval_prompt_path = "config/base_config/eval_config.yaml" 
        with open(eval_prompt_path, "r", encoding="utf-8") as file:
            eval_config = yaml.safe_load(file)
        
        
        tool_descriptions = {}
        agent_card_pathes = glob.glob("data/EN/multiagent_cards/*.json")
        for path in agent_card_pathes:
            with open(path, 'r', encoding='utf-8') as f:
                agent_card = json.load(f)
            tools = agent_card['tools']
            for tool in tools:
                tool_descriptions[tool['name']] = tool['parameters']
        print(tool_descriptions)
        results = await evaluate_sub_agent_history_f1(data, eval_config, tool_descriptions)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    asyncio.run(main())