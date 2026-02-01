#!/usr/bin/env python3
"""
Add a new model result to an existing Hugging Face dataset.

Usage:
    python add_model_to_dataset.py \
        --prediction "New-Model:predictions.jsonl" \
        --hf_dataset_id "username/dataset-name"
"""

import argparse
import json
import pandas as pd
from run_benchmark import score_model, create_results_dataframe


def main():
    parser = argparse.ArgumentParser(description="Add new model to existing HF dataset")
    parser.add_argument(
        "--prediction",
        type=str,
        required=True,
        help="Prediction file in format 'model_name:predictions.jsonl'"
    )
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        required=True,
        help="Hugging Face dataset ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Local CSV file to save"
    )

    args = parser.parse_args()

    # Parse prediction argument
    if ":" in args.prediction:
        model_name, pred_file = args.prediction.split(":", 1)
    else:
        import os
        model_name = os.path.basename(args.prediction).replace(".jsonl", "")
        pred_file = args.prediction

    print(f"\n{'='*60}")
    print(f"Adding Model: {model_name}")
    print(f"{'='*60}\n")

    # Download existing dataset from HF
    print("Downloading existing dataset from Hugging Face...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "huggingface_hub", "datasets"])
        from datasets import load_dataset

    import os
    token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        existing_dataset = load_dataset(args.hf_dataset_id, token=token)
        existing_df = existing_dataset['train'].to_pandas()
        print(f"✓ Loaded {len(existing_df)} existing model(s)")
        print("Current models:", existing_df['Model'].tolist())
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure the dataset exists and you have access.")
        return

    # Score the new model
    print(f"\nScoring {model_name}...")
    results = score_model(pred_file)
    print(f"  Mean Score: {results['Overall']['token_f1']:.2f}%")

    # Get number of sentences from gold.json
    num_sentences = 0
    try:
        with open("gold.json", "r", encoding="utf-8") as f:
            gold_data = json.load(f)
            num_sentences = len(gold_data)
    except:
        # Fallback to existing dataset
        if 'Cumle' in existing_df.columns:
            num_sentences = existing_df['Cumle'].iloc[0]

    # Create dataframe for new model
    model_results = {model_name: results}
    new_df = create_results_dataframe(model_results, num_sentences=num_sentences)

    # Check if model already exists
    if model_name in existing_df['Model'].values:
        print(f"\n⚠️  Warning: {model_name} already exists in dataset.")
        response = input("Do you want to replace it? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        # Remove old entry
        existing_df = existing_df[existing_df['Model'] != model_name]

    # Combine dataframes
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Sort by Mean score (descending)
    combined_df = combined_df.sort_values('Mean', ascending=False).reset_index(drop=True)

    print("\n=== Updated Results DataFrame ===")
    print(combined_df.to_string(index=False))

    # Save to CSV
    combined_df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")

    # Push to Hugging Face
    print(f"\n=== Pushing to Hugging Face ===")
    from datasets import Dataset

    dataset = Dataset.from_pandas(combined_df)
    dataset.push_to_hub(args.hf_dataset_id, token=token)

    print(f"\n✅ Model added successfully!")
    print(f"🔗 View at: https://huggingface.co/datasets/{args.hf_dataset_id}")
    print(f"\nTotal models: {len(combined_df)}")


if __name__ == "__main__":
    main()
