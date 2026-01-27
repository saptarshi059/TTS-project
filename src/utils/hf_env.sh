#!/bin/bash

# ==========================================
# HuggingFace Environment Fixes
# Prevents slow / stuck downloads (XET issue)
# ==========================================

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

echo "✅ HuggingFace environment loaded:"
echo "   - XET disabled"
echo "   - HF transfer enabled"
echo "   - Tokenizer warnings disabled"
