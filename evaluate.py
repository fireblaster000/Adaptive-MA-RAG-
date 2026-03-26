import argparse
import json
import os
import re
import string
from collections import Counter

import numpy as np

from src.utils import load_dataset


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, gold_answers: list[str]) -> float:
    norm_pred = normalize_text(prediction)
    return float(any(norm_pred == normalize_text(g) for g in gold_answers if g is not None))


def _word_boundary_substring(shorter: str, longer: str) -> bool:
    """True if shorter appears in longer as a whole phrase (avoids e.g. 'no' matching inside 'not')."""
    if not shorter or not longer or len(shorter) > len(longer):
        return False
    return re.search(rf"\b{re.escape(shorter)}\b", longer) is not None


def exact_match_score_relaxed(prediction: str, gold_answers: list[str]) -> float:
    """
    Like strict EM, but also counts a match when the normalized gold answer appears as a
    word-boundary-bounded phrase inside the normalized prediction (or vice versa). This matches
    common cases where gold is a short span (yes/no, entity, year) and the model answers with a
    full sentence that still contains that span. Official HotpotQA leaderboard EM is strict only.
    """
    norm_pred = normalize_text(prediction)
    for g in gold_answers:
        if g is None:
            continue
        norm_g = normalize_text(g)
        if not norm_g:
            continue
        if norm_pred == norm_g:
            return 1.0
        if len(norm_pred) >= len(norm_g):
            if _word_boundary_substring(norm_g, norm_pred):
                return 1.0
        else:
            if _word_boundary_substring(norm_pred, norm_g):
                return 1.0
    return 0.0


def f1_score(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold_answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, gold_answers: list[str]) -> float:
    return max(metric_fn(prediction, g) for g in gold_answers if g is not None)


def extract_gold_answers(item: dict, dataset_name: str) -> list[str]:
    if dataset_name == "2wiki":
        return item.get("golden_answers", [])
    answers = []
    for o in item.get("output", []):
        ans = o.get("answer")
        if ans:
            answers.append(ans)
    return answers


def extract_prediction_from_output(output_obj: dict) -> str:
    # Priority order: top-level answer fields, then latest plan summary.
    for key in ("final_answer", "answer", "output"):
        value = output_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    past_exp = output_obj.get("past_exp", [])
    if isinstance(past_exp, list) and past_exp:
        last_exp = past_exp[-1]
        if isinstance(last_exp, dict):
            summary = last_exp.get("plan_summary", {})
            if isinstance(summary, dict):
                ans = summary.get("answer")
                if isinstance(ans, str) and ans.strip():
                    return ans.strip()
                out = summary.get("output")
                if isinstance(out, str) and out.strip():
                    return out.strip()
    return ""


def _question_text(item: dict) -> str:
    return (item.get("input") or item.get("question") or "").strip()


def evaluate(
    dataset_name: str,
    pred_dir: str,
    start_index: int = 0,
    end_index: int = 1000000,
    show_predictions: bool = False,
) -> None:
    dataset = load_dataset(dataset_name)
    total = 0
    found = 0
    missing = 0
    skipped_no_gold = 0
    em_sum = 0.0
    em_relaxed_sum = 0.0
    f1_sum = 0.0

    # Efficiency metrics (from `profile` written by main.py).
    latency_ms_vals: list[float] = []
    prompt_tokens_vals: list[float] = []
    completion_tokens_vals: list[float] = []
    total_tokens_vals: list[float] = []
    total_cost_vals: list[float] = []
    llm_calls_count_vals: list[float] = []

    for idx, item in enumerate(dataset):
        if idx < start_index or idx > end_index:
            continue
        qid = str(item["id"])
        gold_answers = extract_gold_answers(item, dataset_name)
        if not gold_answers:
            skipped_no_gold += 1
            continue
        total += 1
        pred_path = os.path.join(pred_dir, f"{qid}.json")
        if not os.path.exists(pred_path):
            missing += 1
            if show_predictions:
                print(f"\n{'=' * 72}")
                print(f"dataset row={idx}  id={qid}  — missing prediction file")
                q = _question_text(item)
                if q:
                    print(f"Question: {q}")
                print(f"Gold:     {gold_answers}")
            continue
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_obj = json.load(f)
        except Exception:
            missing += 1
            if show_predictions:
                print(f"\n{'=' * 72}")
                print(f"dataset row={idx}  id={qid}  — invalid JSON")
            continue
        pred_answer = extract_prediction_from_output(pred_obj)
        if not pred_answer:
            missing += 1
            if show_predictions:
                print(f"\n{'=' * 72}")
                print(f"dataset row={idx}  id={qid}  — empty extracted prediction")
                q = _question_text(item)
                if q:
                    print(f"Question: {q}")
                print(f"Gold:     {gold_answers}")
            continue
        found += 1

        # Parse efficiency profiling if present.
        profile = pred_obj.get("profile", {})
        if isinstance(profile, dict):
            lat = profile.get("total_latency_ms")
            if isinstance(lat, (int, float)):
                latency_ms_vals.append(float(lat))
            pt = profile.get("total_prompt_tokens")
            if isinstance(pt, (int, float)):
                prompt_tokens_vals.append(float(pt))
            ct = profile.get("total_completion_tokens")
            if isinstance(ct, (int, float)):
                completion_tokens_vals.append(float(ct))
            tt = profile.get("total_tokens")
            if isinstance(tt, (int, float)):
                total_tokens_vals.append(float(tt))
            cost = profile.get("total_cost")
            if isinstance(cost, (int, float)):
                total_cost_vals.append(float(cost))
            calls = profile.get("llm_calls_count")
            if isinstance(calls, (int, float)):
                llm_calls_count_vals.append(float(calls))

        em = exact_match_score(pred_answer, gold_answers)
        em_r = exact_match_score_relaxed(pred_answer, gold_answers)
        f1 = metric_max_over_ground_truths(f1_score, pred_answer, gold_answers)
        em_sum += em
        em_relaxed_sum += em_r
        f1_sum += f1

        if show_predictions:
            print(f"\n{'=' * 72}")
            print(
                f"dataset row={idx}  id={qid}  "
                f"EM_strict={'1' if em else '0'}  EM_relaxed={'1' if em_r else '0'}  "
                f"Token-F1={100.0 * f1:.1f}"
            )
            q = _question_text(item)
            if q:
                print(f"Question: {q}")
            print(f"Gold:     {gold_answers}")
            print(f"Pred:     {pred_answer}")

    denom = max(total, 1)
    if show_predictions:
        print(f"\n{'=' * 72}\n")
    print(f"Dataset: {dataset_name}")
    print(f"Prediction dir: {pred_dir}")
    print(f"Dataset index range (inclusive): [{start_index}, {end_index}]")
    print(f"Evaluated questions: {total}")
    print(f"Predictions found: {found}")
    print(f"Missing/invalid predictions: {missing}")
    print(f"Skipped (no gold): {skipped_no_gold}")
    print(f"EM (strict): {100.0 * em_sum / denom:.2f}")
    print(f"EM (relaxed, gold span in pred or vice versa): {100.0 * em_relaxed_sum / denom:.2f}")
    print(f"Token-F1: {100.0 * f1_sum / denom:.2f}")

    # Efficiency summary
    if total_tokens_vals:
        latency_arr = np.array(latency_ms_vals, dtype=float) if latency_ms_vals else None
        print(f"Avg llm calls/query: {np.mean(llm_calls_count_vals):.2f}" if llm_calls_count_vals else "Avg llm calls/query: N/A")
        if latency_arr is not None and latency_arr.size > 0:
            print(
                f"Avg latency (ms): {np.mean(latency_arr):.1f}  "
                f"p50: {np.percentile(latency_arr, 50):.1f}  "
                f"p90: {np.percentile(latency_arr, 90):.1f}"
            )
        print(
            f"Avg tokens/query: total={np.mean(total_tokens_vals):.1f}  "
            f"prompt={np.mean(prompt_tokens_vals) if prompt_tokens_vals else 0.0:.1f}  "
            f"completion={np.mean(completion_tokens_vals) if completion_tokens_vals else 0.0:.1f}"
        )
        if total_cost_vals:
            print(f"Avg cost/query: {np.mean(total_cost_vals):.4f}")
    if dataset_name == "fever":
        print("FEVER note: EM here is equivalent to exact-label accuracy.")
    print(
        "Note: Paper-style HotpotQA EM is strict (normalized full string equality). "
        "Relaxed EM helps when gold is a short span embedded in a longer correct answer."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MA-RAG predictions.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nq", "hotpotqa", "triviaqa", "2wiki", "fever"],
        help="Dataset name used in main.py",
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing prediction JSON files, one per question id.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="First dataset row index to evaluate (same inclusive semantics as main.py).",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=1000000,
        help="Last dataset row index to evaluate, inclusive (default: full dev set).",
    )
    parser.add_argument(
        "--show_predictions",
        action="store_true",
        help="Print per-example question, gold answer(s), and prediction (useful for smoke slices).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args.dataset,
        args.pred_dir,
        args.start_index,
        args.end_index,
        show_predictions=args.show_predictions,
    )
