#!/usr/bin/env bash
set -euo pipefail

VENV=/Users/hwc/.workbuddy/binaries/python/envs/mlx
LLAMA=/Users/hwc/LycheeAI/llama.cpp
SRC=/Users/hwc/LycheeAI/LycheeAI-coder-2b-II-pro-v7-f16
WORK=/Users/hwc/LycheeAI/gguf-build
NAME=LycheeAI-coder-2b-II-pro

mkdir -p "$WORK"
echo "=== 磁盘检查 ==="
df -g /System/Volumes/Data | tail -1 | awk '{print "  可用: " $4 " GB"}'

echo ""
echo "=== 步骤 1/3  转 GGUF f16（源：v7 f16 全精度）==="
cd "$LLAMA"
"$VENV/bin/python" convert_hf_to_gguf.py "$SRC" \
    --outfile "$WORK/$NAME-f16.gguf" \
    --outtype f16
echo "  ✓ f16 GGUF 完成"

echo ""
echo "=== 步骤 2/3  量化 q8_0 ==="
"$LLAMA/build/bin/llama-quantize" "$WORK/$NAME-f16.gguf" "$WORK/$NAME-q8_0.gguf" Q8_0
echo "  ✓ q8_0 完成"

echo ""
echo "=== 步骤 3/3  量化 q4_k_m ==="
"$LLAMA/build/bin/llama-quantize" "$WORK/$NAME-f16.gguf" "$WORK/$NAME-q4_k_m.gguf" Q4_K_M
echo "  ✓ q4_k_m 完成"

echo ""
echo "=== 产物 ==="
ls -lh "$WORK"/$NAME-*.gguf 2>/dev/null || true
echo ""
df -g /System/Volumes/Data | tail -1 | awk '{print "剩余可用: " $4 " GB"}'
echo "ALL_DONE"
