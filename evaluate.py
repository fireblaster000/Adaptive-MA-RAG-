import argparse
import json
import os
import re
import string
from collections import Counter

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


def evaluate(dataset_name: str, pred_dir: str) -> None:
    dataset = load_dataset(dataset_name)
    total = 0
    found = 0
    missing = 0
    skipped_no_gold = 0
    em_sum = 0.0
    f1_sum = 0.0

    for item in dataset:
        qid = str(item["id"])
        gold_answers = extract_gold_answers(item, dataset_name)
        if not gold_answers:
            skipped_no_gold += 1
            continue
        total += 1
        pred_path = os.path.join(pred_dir, f"{qid}.json")
        if not os.path.exists(pred_path):
            missing += 1
            continue
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_obj = json.load(f)
        except Exception:
            missing += 1
            continue
        pred_answer = extract_prediction_from_output(pred_obj)
        if not pred_answer:
            missing += 1
            continue
        found += 1

        em_sum += exact_match_score(pred_answer, gold_answers)
        f1_sum += metric_max_over_ground_truths(f1_score, pred_answer, gold_answers)

    denom = max(total, 1)
    print(f"Dataset: {dataset_name}")
    print(f"Prediction dir: {pred_dir}")
    print(f"Evaluated questions: {total}")
    print(f"Predictions found: {found}")
    print(f"Missing/invalid predictions: {missing}")
    print(f"Skipped (no gold): {skipped_no_gold}")
    print(f"EM: {100.0 * em_sum / denom:.2f}")
    print(f"Token-F1: {100.0 * f1_sum / denom:.2f}")
    if dataset_name == "fever":
        print("FEVER note: EM here is equivalent to exact-label accuracy.")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.dataset, args.pred_dir)
