#!/usr/bin/env bash
# ============================================================
#  v7 训练：精选 980 条 × 1 epoch = 490 步
#
#  ★ 与 v6 系列的根本区别
#    v6 系列是「全量 5341 条 × 22% epoch」——
#    每条样本只被见到 0.22 次，加上 51.4% 是单轮纯文本，
#    梯度被拉向"纯聊天"，多轮链路（A 类）学不到。
#
#    v7 是「精选 980 条 × 100% epoch」——
#    每一类行为都拿到完整的 1 次更新，配额由重要性决定：
#        A  多轮链路    201 (20.5%)   ← 最值钱
#        C2 user结构调  243 (24.8%)
#        C4 有不调      216 (22.0%)
#        C1 sys结构调   179 (18.3%)
#        F  单轮文本    130 (13.3%)   ← 从 51.4% 砍到 13.3%
#        E  多轮对话     11 ( 1.1%)
#
#  ★ 算术：两头夹
#    一头 K9 补稀缺类型（add/sub/div/pow/float_*）
#    一头对基线里的过剩乘法下采样
#    结果：乘法占比 50.5% → 20.0%，7 类极差 30.5x → 2.4x
#
#  ★ 源数据已修缺陷
#    data_coder2b_pro_v6_fixed：修掉 K4 段「system 只定义 calculate
#    却调用 read_file/get_weather」的 9 处自相矛盾样本，
#    并加入干扰工具保持「多工具选择」的训练信号。
#    用 build_v7.py 从该源重建精选集。
#
#  用法:
#    bash train_v7.sh
# ============================================================
set -euo pipefail

VENV=${VENV:-/Users/hwc/.workbuddy/binaries/python/envs/mlx}
MLX_BASE=${MLX_BASE:-/Users/hwc/LycheeAI/MiniCPM5-2B-MLX}
DATA=${DATA:-/Users/hwc/LycheeAI/data_coder2b_pro_v7}
BASE_ADAPTER=${BASE_ADAPTER:-/Users/hwc/LycheeAI/adapters-coder2b-pro-v6-1}
OUT=${OUT:-/Users/hwc/LycheeAI/adapters-coder2b-pro-v7}
ITERS=${ITERS:-490}

echo "========================================="
echo "  v7 训练（精选 980 条 × 1 epoch）"
echo "========================================="
echo "  基座     : $MLX_BASE"
echo "  数据     : $DATA"
echo "  起步权重 : $BASE_ADAPTER"
echo "  输出     : $OUT"
echo "  步数     : $ITERS"
echo ""

for p in "$MLX_BASE" "$DATA/train.jsonl" "$BASE_ADAPTER/adapters.safetensors"; do
    if [ ! -e "$p" ]; then echo "  ✗ 找不到: $p"; exit 1; fi
done
N=$(wc -l < "$DATA/train.jsonl" | tr -d ' ')
echo "  数据条数 : $N"
echo ""

rm -rf "$OUT"; mkdir -p "$OUT"

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
  --save-every "$ITERS" 2>&1 | tee /tmp/train_v7.log | tail -25

echo ""
echo "========================================="
echo "  训练完成，验收："
echo "========================================="
echo "  $VENV/bin/python /Users/hwc/LycheeAI/eval_v6.py $MLX_BASE $OUT"
echo ""
