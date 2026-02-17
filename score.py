import json
import re
from collections import defaultdict

def norm(s: str) -> str:
    """Normalize string for comparison."""
    s = (s or "").strip()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?:;\"'""''()[]{}")
    return s.lower()

def count_matches(pred: str, gold: str) -> tuple:
    """
    Count how many gold parts are matched by prediction.

    Returns: (matched_count, total_gold_parts)

    - If gold has 2 answers and prediction matches 1 → (1, 2)
    - If gold has 2 answers and prediction matches 2 → (2, 2)
    """
    pred_norm = norm(pred)

    # Handle empty cases
    if not gold.strip():
        # Gold is empty - prediction should also be empty
        return (1, 1) if not pred_norm else (0, 1)

    if not pred_norm:
        # Prediction is empty but gold is not
        gold_parts = [g.strip() for g in gold.split(";") if g.strip()]
        return (0, len(gold_parts) if gold_parts else 1)

    # Split gold into parts
    gold_parts = [g.strip() for g in gold.split(";") if g.strip()]
    total_parts = len(gold_parts)

    if total_parts == 0:
        return (1, 1) if not pred_norm else (0, 1)

    # Check each gold part
    matched_count = 0

    for gold_part in gold_parts:
        gold_part_norm = norm(gold_part)

        # Check if gold matches exactly or appears at the end of prediction
        # (allows explanation before, but no extra words after)
        if pred_norm == gold_part_norm:
            matched_count += 1
        elif pred_norm.endswith(gold_part_norm):
            matched_count += 1

    return (matched_count, total_parts)

def exact_match(pred: str, gold: str) -> int:
    """
    Exact match - returns 1 if prediction exactly matches gold (or any gold part).
    """
    pred_norm = norm(pred)

    # Handle empty
    if not gold.strip():
        return 1 if not pred_norm else 0

    if not pred_norm:
        return 0

    # Check full match
    if pred_norm == norm(gold):
        return 1

    # Check if matches any individual part exactly
    gold_parts = [norm(g.strip()) for g in gold.split(";") if g.strip()]
    if pred_norm in gold_parts:
        return 1

    return 0

def main():
    # benchmark yükle
    bench = {}
    with open("benchmark.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            bench[item["id"]] = item

    # predictions yükle
    preds = {}
    with open("predictions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds[obj["id"]] = obj.get("prediction", "")

    # skorla
    by_field = defaultdict(lambda: {
        "n": 0,
        "em_sum": 0,
        "parts_matched": 0,
        "parts_total": 0
    })
    overall = {
        "n": 0,
        "em_sum": 0,
        "parts_matched": 0,
        "parts_total": 0
    }

    missing = 0
    for _id, item in bench.items():
        gold = item["gold"]
        pred = preds.get(_id, None)
        if pred is None:
            missing += 1
            pred = ""

        field = _id.split("__", 1)[1] if "__" in _id else "UNKNOWN"

        # Exact match (binary)
        em = exact_match(pred, gold)

        # Parts matched (count how many gold answers found)
        matched, total = count_matches(pred, gold)

        by_field[field]["n"] += 1
        by_field[field]["em_sum"] += em
        by_field[field]["parts_matched"] += matched
        by_field[field]["parts_total"] += total

        overall["n"] += 1
        overall["em_sum"] += em
        overall["parts_matched"] += matched
        overall["parts_total"] += total

    def pct(x): return 100.0 * x

    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Total questions: {overall['n']}  |  Missing predictions: {missing}")
    print(f"")
    print(f"Exact-Match:    {pct(overall['em_sum']/overall['n']):6.2f}%")
    print(f"Parts Score:    {overall['parts_matched']}/{overall['parts_total']} ({pct(overall['parts_matched']/overall['parts_total']):5.2f}%)")

    print("\n" + "="*60)
    print("BY GRAMMATICAL ELEMENT")
    print("="*60)
    print(f"{'Element':<20} | {'N':>5} | {'EM%':>7} | {'Parts':>12} | {'Parts%':>7}")
    print("-"*60)

    for field in sorted(by_field.keys()):
        data = by_field[field]
        n = data["n"]
        em = data["em_sum"] / n
        parts_pct = data["parts_matched"] / data["parts_total"] if data["parts_total"] > 0 else 0
        parts_str = f"{data['parts_matched']}/{data['parts_total']}"
        print(f"{field:<20} | {n:>5} | {pct(em):>6.2f}% | {parts_str:>12} | {pct(parts_pct):>6.2f}%")

if __name__ == "__main__":
    main()
