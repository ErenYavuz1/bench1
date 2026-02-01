#!/bin/bash
# Setup script - activate virtual environment and install dependencies

echo "Activating virtual environment..."
source bin/activate

echo "Installing required packages..."
pip install pandas huggingface_hub datasets openai anthropic google-generativeai

echo ""
echo "Installing packages for local HuggingFace models..."
pip install torch transformers accelerate

echo ""
echo "✅ Setup complete!"
echo ""
echo "Virtual environment is now active."
echo "You can now run:"
echo "  python model_runners.py --provider gemini --model gemini-2.5-flash-lite --output predictions.jsonl"
