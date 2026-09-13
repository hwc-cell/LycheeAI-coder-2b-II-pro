#!/usr/bin/env bash
# ============================================================
#  小规模试训（sanity run）—— v7 方法改变的核心
#
#  ★ 为什么要这个脚本
#    v6 系列的教训：每次「发现问题 → 补 5% 数据 → 全量重训」，
#    把整个能力分布重洗一遍，结果修一个坏一个（①b ↔ ⑨g/⑧a 跷跷板）。
#    v6.2 修好①b坏⑧a⑨g；v6.3 修好⑧a⑨g又坏①b，且逐项与 v6.1 相同（白跑）；
#    v6.4 加 K8 后调用率从 60%+ 崩到 12%。
#
#  ★ 新流程
#    1) 改数据
#    2) 跑这个脚本（~200 步，几分钟）
#    3) 用 eval_v6.py 验收 + 跟基线比
#    4) 任何一项退化 → 回退改法，**不进全量**
#    5) 全部通过 → 才跑全量
#
#  用法:
#    bash try_run_v7.sh            # 200 步试训
#    ITERS=400 bash try_run_v7.sh  # 改步数
# ============================================================
set -euo pipefail

VENV=${VENV:-/Users/hwc/.workbuddy/binaries/python/envs/mlx}
MLX_BASE=${MLX_BASE:-/Users/hwc/LycheeAI/MiniCPM5-2B-MLX}
DATA=${DATA:-/Users/hwc/LycheeAI/data_coder2b_pro_v6_final}
BASE_ADAPTER=${BASE_ADAPTER:-/Users/hwc/LycheeAI/adapters-coder2b-pro-v6-1}
OUT=${OUT:-/Users/hwc/LycheeAI/adapters-tryrun}
ITERS=${ITERS:-200}

echo "========================================="
echo "  小规模试训（sanity run）"
echo "========================================="
echo "  基座      : $MLX_BASE"
echo "  数据      : $DATA"
echo "  起步权重  : $BASE_ADAPTER"
echo "  输出      : $OUT"
echo "  步数      : $ITERS"
echo ""

for p in "$MLX_BASE" "$DATA/train.jsonl" "$BASE_ADAPTER/adapters.safetensors"; do
    if [ ! -e "$p" ]; then
        echo "  ✗ 找不到: $p"
        exit 1
    fi
done
N=$(wc -l < "$DATA/train.jsonl" | tr -d ' ')
echo "  数据条数  : $N"
echo ""

rm -rf "$OUT"
mkdir -p "$OUT"

"$VENV/bin/mlx_lm.lora" \
  --model "$MLX_BASE" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size 2 \
  --max-seq-length 2048 \
  --iters "$ITERS" \
  --learning-rate 5e-5 \
  --resume-adapter-file "$BASE_ADAPTER/adapters.safetensors" \
  --adapter-path "$OUT" \
  --grad-checkpoint \
  --save-every "$ITERS" 2>&1 | tee /tmp/tryrun.log | tail -20

echo ""
echo "========================================="
echo "  试训完成，下一步：验收"
echo "========================================="
echo ""
echo "  # 跟 v6.1 基线逐项对比（基线 25 通过 / 2 失败）"
echo "  $VENV/bin/python /Users/hwc/LycheeAI/eval_v6.py $MLX_BASE $OUT | tee /tmp/tryrun_eval.txt"
echo ""
echo "  ★ 判据：任何一项比基线差 → 回退改法，不进全量"
echo ""
echo "  基线 v6.1 的 2 个失败项（这两个不算退化）："
echo "    ❌ ①b 第二道算术题"
echo "    ❌ ⑧b 多轮追问"
echo ""
