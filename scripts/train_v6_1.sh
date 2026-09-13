#!/usr/bin/env bash
# ============================================================
#  v6 系列续训脚本：在已有 adapter 上打补丁
#
#  背景（为什么要续训而不是从头训）：
#    v6 跑完全量 3564 步，验收 24 通过 / 3 失败。
#    那 3 个失败项的问题是**样本量不足**，不是权重不好 ——
#    所以没必要从头再训一遍，直接在 v6 权重上补数据续训。
#
#  3 个待修问题 + 本次的数据对策：
#    ┌──────────────────────────┬──────────┬──────────┬─────────┐
#    │ 问题                      │ v6 样本  │ 现有样本 │ 对抗惯性│
#    ├──────────────────────────┼──────────┼──────────┼─────────┤
#    │ ② 纯聊天宿主凭空调工具     │   93     │   342    │  1134   │
#    │ ⑥ 工具放 user 时不调工具   │  180     │   582    │  1255   │
#    │ ⑧b 多轮追问丢上下文        │  170     │   245    │   —     │
#    └──────────────────────────┴──────────┴──────────┴─────────┘
#    ② 的比例从 1:7.5 改善到 1:3.3，⑥ 的 user 占比从 20% 提到 32%。
#
#  ★ 为什么必须喂全量数据（不能只喂补丁）
#    只喂"无工具时不调工具"的补丁，模型会学成"一律不调工具"，
#    把 v6 刚修好的"该调就调"（算术/天气/搜索/读文件）全练废。
#    这是灾难性遗忘，比原来的问题更难修。
#
#  用法（路径都可用环境变量覆盖）:
#    MLX_BASE=<基座> DATA=../data V6_ADAPTER=../adapters bash scripts/train_v6_1.sh
#
#  预期:
#    2263 步（全量 1 epoch，4526 条 / batch 2），loss 应落到 0.5 附近
# ============================================================
set -euo pipefail

VENV=${VENV:-$(dirname $(which mlx_lm.lora) 2>/dev/null | xargs dirname || echo /usr/local)}
MLX_BASE=${MLX_BASE:-./MiniCPM5-2B-MLX}
DATA=${DATA:-./data}
V6_ADAPTER=${V6_ADAPTER:-./adapters}
OUT_ADAPTER=${OUT_ADAPTER:-./adapters-out}

echo "========================================="
echo "  v6.1 续训（在 v6 权重上打补丁）"
echo "========================================="
echo ""

# ---- 前置检查 ----
echo "[检查] 依赖文件..."

if [ ! -f "$V6_ADAPTER/adapters.safetensors" ]; then
    echo "  ✗ 找不到 v6 权重: $V6_ADAPTER/adapters.safetensors"
    exit 1
fi
echo "  ✓ v6 权重"

if [ ! -f "$DATA/train.jsonl" ]; then
    echo "  ✗ 找不到训练数据: $DATA/train.jsonl"
    exit 1
fi
N=$(wc -l < "$DATA/train.jsonl" | tr -d ' ')
echo "  ✓ 训练数据 ${N} 条"

if [ ! -d "$MLX_BASE" ]; then
    echo "  ✗ 找不到 MLX 基座: $MLX_BASE"
    exit 1
fi
echo "  ✓ MLX 基座"

AVAIL=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
echo "  · 磁盘可用 ${AVAIL} GB"
if [ "$AVAIL" -lt 3 ]; then
    echo "  ✗ 可用空间不足 3GB"
    exit 1
fi
echo ""

# ---- 安全网：备份 v6 权重（万一 v6.1 更差可以回退）----
if [ ! -d "${V6_ADAPTER}-backup" ]; then
    echo "[备份] v6 权重 → ${V6_ADAPTER}-backup"
    cp -r "$V6_ADAPTER" "${V6_ADAPTER}-backup"
    echo "  ✓ 已备份"
else
    echo "[备份] 已存在，跳过"
fi
echo ""

# 续训输出目录（不覆盖 v6）
if [ -d "$OUT_ADAPTER" ]; then
    echo "[清理] 移除旧的续训输出 $OUT_ADAPTER"
    rm -rf "$OUT_ADAPTER"
fi
mkdir -p "$OUT_ADAPTER"

# ---- 步数计算 ----
# iters = epoch × ceil(N / batch_size)
# N=5093, batch=2 → 2547 步 = 1 epoch
ITERS=2671

echo "========================================="
echo "  开始续训"
echo "========================================="
echo "  起点权重 : v6（3564 步全量训完的）"
echo "  数据     : ${N} 条（全量 + 补丁）"
echo "  步数     : ${ITERS}（= 1 epoch）"
echo "  输出     : $OUT_ADAPTER"
echo ""
echo "  ★ 和学习率相关的说明："
echo "    续训沿用 5e-5。v6 已经收敛，这个学习率下 1 epoch"
echo "    足以让新数据生效，又不会大幅破坏已有能力。"
echo ""

# ---- 续训 ----
# 关键参数：
#   --resume-adapter-file  从 v6 权重起步（这是"续训"的核心）
#   --adapter-path         输出到新目录，不覆盖 v6
#   --iters 2223           全量 1 epoch
#   --num-layers 16        与 v6 保持一致
#   --save-every 600       中途存档，便于早停选优
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
  --resume-adapter-file "$V6_ADAPTER/adapters.safetensors" \
  --adapter-path "$OUT_ADAPTER" \
  --grad-checkpoint \
  --save-every 600 \
  --steps-per-report 50

echo ""
echo "========================================="
echo "  ✓ 续训完成"
echo "========================================="
echo ""
echo "下一步 —— 验收（跑 27 项，重点看这 3 项）："
echo "    cd <项目根目录>"
echo "    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY"
echo "    $VENV/bin/python eval_v6.py $MLX_BASE $OUT_ADAPTER"
echo ""
echo "  只看 3 个关键项的话："
echo "    ❌ ⑥ 工具在 user 时也能正确调用"
echo "    ❌ ⑧b 多轮追问 → 应基于上文再调 calculate"
echo "    ❌ ⑨ 天气：帮我查一下明天北京的天气 → 不该调工具"
echo ""
echo "  对照基线："
echo "    v5   : eval_baseline_v5.txt  (18 通过 / 9 失败)"
echo "    v6   : eval_v6_result.txt   (24 通过 / 3 失败)"
