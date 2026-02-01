"""
Model runner utilities for different API providers.
Supports: OpenAI, Gemini, Anthropic, OpenRouter
"""

import os
import json
import time
from typing import Optional


def make_prompt(sentence: str, question: str) -> str:
    """Standard prompt for all models."""
    return f"""Aşağıdaki cümlede sorulan ögeyi yaz.

KURALLAR:
- SADECE cevabı yaz.
- BAŞINA veya SONUNA hiçbir işaret koyma.
- Cümlede sorulan öge yoksa -(tire) koy.
- Kısaltma Yapma.
- Soru gizli özne ise cümleden almana gerek yok. Eğer gizli özne yok ise boş bırak.
- Diğer öğeler için cümleden kopyala yapıştır.



Cümle:
"{sentence}"

Soru:
{question}

Cevap:
"""


def clean_prediction(pred: str) -> str:
    """Clean model output."""
    if not pred:
        return ""
    p = pred.replace("\u200b", "").replace("\ufeff", "").strip()
    p = p.splitlines()[0].strip()

    if p.lower() in {"", "yok", "yok.", "yoktur", "(boş)", "(boş bırakılır)", "boş", "-", "—", ":"}:
        return ""
    return p.strip(" :.-")


def run_openai_model(
    model_name: str,
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    sleep: float = 0.5
):
    """Run OpenAI model on benchmark."""
    try:
        from openai import OpenAI
    except ImportError:
        import subprocess
        print("Installing openai...")
        subprocess.check_call(["pip", "install", "openai"])
        from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    with open(input_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            prompt = make_prompt(item["sentence"], item["question"])

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=512
                )
                pred_raw = response.choices[0].message.content or ""
            except Exception as e:
                print(f"[ERROR] id={item['id']} -> {e}")
                pred_raw = ""

            pred = clean_prediction(pred_raw)
            out.write(json.dumps({"id": item["id"], "prediction": pred}, ensure_ascii=False) + "\n")

            if i % 20 == 0:
                print(f"{i}/{len(items)} done")

            time.sleep(sleep)

    print(f"✓ Completed: {output_file}")


def run_anthropic_model(
    model_name: str,
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    sleep: float = 0.5
):
    """Run Anthropic model on benchmark."""
    try:
        import anthropic
    except ImportError:
        import subprocess
        print("Installing anthropic...")
        subprocess.check_call(["pip", "install", "anthropic"])
        import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    with open(input_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            prompt = make_prompt(item["sentence"], item["question"])

            try:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=512,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                pred_raw = response.content[0].text if response.content else ""
            except Exception as e:
                print(f"[ERROR] id={item['id']} -> {e}")
                pred_raw = ""

            pred = clean_prediction(pred_raw)
            out.write(json.dumps({"id": item["id"], "prediction": pred}, ensure_ascii=False) + "\n")

            if i % 20 == 0:
                print(f"{i}/{len(items)} done")

            time.sleep(sleep)

    print(f"✓ Completed: {output_file}")


def run_gemini_model(
    model_name: str,
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    sleep: float = 0.4
):
    """Run Gemini model on benchmark."""
    try:
        import google.generativeai as genai
    except ImportError:
        import subprocess
        print("Installing google-generativeai...")
        subprocess.check_call(["pip", "install", "google-generativeai"])
        import google.generativeai as genai

    genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)
    gen_config = {"temperature": temperature, "max_output_tokens": 512}

    with open(input_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    def get_text(resp) -> str:
        try:
            cands = getattr(resp, "candidates", None) or []
            if not cands:
                return ""
            content = getattr(cands[0], "content", None)
            if content is None:
                return ""
            parts = getattr(content, "parts", None) or []
            if not parts:
                return ""
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    return t
            return ""
        except Exception:
            return ""

    with open(output_file, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            prompt = make_prompt(item["sentence"], item["question"])

            try:
                resp = model.generate_content(prompt, generation_config=gen_config)
                pred_raw = get_text(resp)
            except Exception as e:
                print(f"[ERROR] id={item['id']} -> {e}")
                pred_raw = ""

            pred = clean_prediction(pred_raw)
            out.write(json.dumps({"id": item["id"], "prediction": pred}, ensure_ascii=False) + "\n")

            if i % 20 == 0:
                print(f"{i}/{len(items)} done")

            time.sleep(sleep)

    print(f"✓ Completed: {output_file}")


def run_openrouter_model(
    model_name: str,
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    sleep: float = 0.5
):
    """Run model via OpenRouter."""
    try:
        from openai import OpenAI
    except ImportError:
        import subprocess
        print("Installing openai...")
        subprocess.check_call(["pip", "install", "openai"])
        from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY")
    )

    with open(input_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            prompt = make_prompt(item["sentence"], item["question"])

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=512
                )
                pred_raw = response.choices[0].message.content or ""
            except Exception as e:
                print(f"[ERROR] id={item['id']} -> {e}")
                pred_raw = ""

            pred = clean_prediction(pred_raw)
            out.write(json.dumps({"id": item["id"], "prediction": pred}, ensure_ascii=False) + "\n")

            if i % 20 == 0:
                print(f"{i}/{len(items)} done")

            time.sleep(sleep)

    print(f"✓ Completed: {output_file}")


def run_huggingface_model(
    model_name: str,
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    sleep: float = 0.0
):
    """Run Hugging Face model locally (downloads model)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        import subprocess
        print("Installing transformers and torch...")
        subprocess.check_call(["pip", "install", "torch", "transformers", "accelerate"])
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")
    print("(First run will download the model - this may take a while)")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=api_key or os.environ.get("HF_TOKEN"),
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=api_key or os.environ.get("HF_TOKEN"),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        if device == "cpu":
            model = model.to(device)

        print(f"✓ Model loaded successfully on {device}")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("\nTip: Make sure you have access to the model on HuggingFace.")
        print("For gated models (like Llama), accept terms at: https://huggingface.co/{model_name}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(items)} items...")

    with open(output_file, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {item['id']}...", end=" ", flush=True)
            prompt = make_prompt(item["sentence"], item["question"])

            try:
                # Tokenize with attention mask
                if hasattr(tokenizer, "apply_chat_template"):
                    messages = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    encoded = tokenizer(text, return_tensors="pt", return_attention_mask=True)
                else:
                    encoded = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)

                inputs = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)

                # Generate
                print("generating...", end=" ", flush=True)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                print("done.", end=" ", flush=True)

                # Decode only the NEW tokens (skip the input prompt)
                input_len = inputs.shape[-1] if hasattr(inputs, 'shape') else len(inputs[0])
                pred_raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

                # Clean up chat model turn markers (e.g., Gemma's "model\n" prefix)
                pred_raw = pred_raw.strip()
                if pred_raw.lower().startswith("model"):
                    # Remove "model" prefix and any following newline
                    pred_raw = pred_raw[5:].strip()

                # Take only the first line of actual answer
                pred_raw = pred_raw.split("\n")[0].strip()

            except Exception as e:
                import traceback
                print(f"[ERROR] id={item['id']} -> {type(e).__name__}: {str(e)}")
                if i <= 3:  # Print full traceback for first few errors
                    traceback.print_exc()
                pred_raw = ""

            pred = clean_prediction(pred_raw)
            out.write(json.dumps({"id": item["id"], "prediction": pred}, ensure_ascii=False) + "\n")
            out.flush()  # Write immediately
            print(f"-> {pred[:50] if pred else '(empty)'}")

            time.sleep(sleep)

    print(f"✓ Completed: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a model on the benchmark")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "openrouter", "huggingface"], required=True)
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--input", type=str, default="benchmark.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output predictions file")
    parser.add_argument("--api_key", type=str, default=None, help="API key (or use env var)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests")

    args = parser.parse_args()

    runners = {
        "openai": run_openai_model,
        "anthropic": run_anthropic_model,
        "gemini": run_gemini_model,
        "openrouter": run_openrouter_model,
        "huggingface": run_huggingface_model,
    }

    runner = runners[args.provider]
    runner(
        model_name=args.model,
        input_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        temperature=args.temperature,
        sleep=args.sleep
    )
