#!/bin/zsh

# Simple runner for OpenAI GPT-5 with custom settings

# Configuration
PROMPT_TEMPLATE="# Complete the function above. Output in a python ```python ... ``` block."
SUFFIX="output_formatting-$(date +%Y%m%d-%H%M%S).jsonl"
USE_SUBSET=False

# Run the generator
cd "$(dirname "$0")"

../../APR_env/bin/python simple_generator.py generate \
    --llm_model="gpt-5" \
    --llm_provider="openai" \
    --prompt_template="$PROMPT_TEMPLATE" \
    --filename_suffix="$OUTPUT_FILENAME" \
    --use_subset=$USE_SUBSET \
    --verbose=true