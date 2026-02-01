"""
Multi-model benchmark runner with Hugging Face integration.

Usage:
    python run_benchmark.py --models "gemini-2.5-flash-lite,gpt-4o-mini" --hf_dataset_id "your-username/turkish-syntax-benchmark"
"""

import json
import pandas as pd
import argparse
from collections import defaultdict
from pathlib import Path
import re

# Import scoring functions from score.py
def norm(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?:;\"'""''()[]{}")
    return s.lower()

def tokens(s: str):
    s = norm(s)
    if not s:
        return []
    return re.findall(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+", s)

def exact_match(pred: str, gold: str) -> int:
    return 1 if norm(pred) == norm(gold) else 0

def token_f1(pred: str, gold: str) -> float:
    pt = tokens(pred)
    gt = tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pset = pt[:]
    gset = gt[:]
    common = 0
    for w in pt:
        if w in gset:
            common += 1
            gset.remove(w)
    prec = common / len(pt) if pt else 0.0
    rec = common / len(gt) if gt else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def score_model(predictions_file: str) -> dict:
    """Score a model's predictions and return results by field."""
    # Load benchmark
    bench = {}
    with open("benchmark.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            bench[item["id"]] = item

    # Load predictions
    preds = {}
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds[obj["id"]] = obj.get("prediction", "")

    # Score
    by_field = defaultdict(lambda: {"n": 0, "em_sum": 0, "f1_sum": 0})
    overall = {"n": 0, "em_sum": 0, "f1_sum": 0}

    for _id, item in bench.items():
        gold = item["gold"]
        pred = preds.get(_id, "")

        field = _id.split("__", 1)[1] if "__" in _id else "UNKNOWN"
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)

        by_field[field]["n"] += 1
        by_field[field]["em_sum"] += em
        by_field[field]["f1_sum"] += f1

        overall["n"] += 1
        overall["em_sum"] += em
        overall["f1_sum"] += f1

    # Calculate percentages
    results = {}
    for field, stats in by_field.items():
        n = stats["n"]
        results[field] = {
            "exact_match": 100.0 * stats["em_sum"] / n if n > 0 else 0.0,
            "token_f1": 100.0 * stats["f1_sum"] / n if n > 0 else 0.0,
        }

    results["Overall"] = {
        "exact_match": 100.0 * overall["em_sum"] / overall["n"] if overall["n"] > 0 else 0.0,
        "token_f1": 100.0 * overall["f1_sum"] / overall["n"] if overall["n"] > 0 else 0.0,
    }

    return results


def create_results_dataframe(model_results: dict, num_sentences: int = None) -> pd.DataFrame:
    """
    Create a dataframe with columns:
    Model, Testsayısı, Mean, Yuklem, ozne, gizliözne, belirtilinesne, belirtisiznesne, dolaylıtümleç, zarftümleci

    Args:
        model_results: Dict mapping model_name -> field_name -> {"exact_match": float, "token_f1": float}
        num_sentences: Number of test sentences in the benchmark
    """
    # Field name mapping from benchmark to desired column names
    field_mapping = {
        "Yüklem": "Yuklem",
        "Özne": "ozne",
        "Gizli Özne": "gizliözne",
        "Belirtili Nesne": "belirtilinesne",
        "Belirtisiz Nesne": "belirtisiznesne",
        "Dolaylı Tümleç": "dolaylıtümleç",
        "Zarf Tümleci": "zarftümleci",
        "Edat Tümleci": "edattümleci",
    }

    rows = []
    for model_name, results in model_results.items():
        # Using exact_match as the metric
        row = {"Model": model_name}

        # Add number of sentences (Testsayısı)
        if num_sentences is not None:
            row["Testsayısı"] = num_sentences
        else:
            # Fallback: calculate from gold.json if available
            try:
                with open("gold.json", "r", encoding="utf-8") as f:
                    gold_data = json.load(f)
                    row["Testsayısı"] = len(gold_data)
            except:
                row["Testsayısı"] = 0

        # Add mean score
        row["Mean"] = results.get("Overall", {}).get("exact_match", 0.0)

        # Add individual field scores
        for benchmark_field, column_name in field_mapping.items():
            score = results.get(benchmark_field, {}).get("exact_match", 0.0)
            row[column_name] = score

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure column order
    column_order = ["Model", "Testsayısı", "Mean", "Yuklem", "ozne", "gizliözne",
                    "belirtilinesne", "belirtisiznesne", "dolaylıtümleç", "zarftümleci"]

    # Add edattümleci if it exists
    if "edattümleci" in df.columns:
        column_order.append("edattümleci")

    # Only keep columns that exist
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]

    return df


def push_to_huggingface(df: pd.DataFrame, dataset_id: str, token: str = None):
    """Push the results dataframe to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        from datasets import Dataset
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "huggingface_hub", "datasets"])
        from huggingface_hub import HfApi, create_repo
        from datasets import Dataset

    # Convert pandas dataframe to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Create repo if it doesn't exist
    try:
        create_repo(dataset_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"✓ Repository created/verified: {dataset_id}")
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")

    # Push to hub
    dataset.push_to_hub(dataset_id, token=token)
    print(f"✓ Results pushed to https://huggingface.co/datasets/{dataset_id}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-model benchmark and push to HF")
    parser.add_argument(
        "--predictions",
        nargs="+",
        help="Prediction files in format 'model_name:predictions.jsonl'",
        required=True
    )
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        help="Hugging Face dataset ID (e.g., 'username/dataset-name')",
        default=None
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face API token (or set HF_TOKEN env var)",
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path",
        default="benchmark_results.csv"
    )

    args = parser.parse_args()

    # Score all models
    model_results = {}
    print("\n=== Scoring Models ===")
    for pred_spec in args.predictions:
        if ":" in pred_spec:
            model_name, pred_file = pred_spec.split(":", 1)
        else:
            # Use filename as model name
            model_name = Path(pred_spec).stem
            pred_file = pred_spec

        print(f"\nScoring {model_name}...")
        results = score_model(pred_file)
        model_results[model_name] = results

        # Print summary
        print(f"  Mean Score: {results['Overall']['exact_match']:.2f}%")

    # Get number of sentences from gold.json
    num_sentences = 0
    try:
        with open("gold.json", "r", encoding="utf-8") as f:
            gold_data = json.load(f)
            num_sentences = len(gold_data)
    except:
        pass

    # Create dataframe
    print("\n=== Creating Results DataFrame ===")
    df = create_results_dataframe(model_results, num_sentences=num_sentences)
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")

    # Push to Hugging Face if requested
    if args.hf_dataset_id:
        print(f"\n=== Pushing to Hugging Face ===")
        import os
        token = args.hf_token or os.environ.get("HF_TOKEN")
        push_to_huggingface(df, args.hf_dataset_id, token)


if __name__ == "__main__":
    main()
