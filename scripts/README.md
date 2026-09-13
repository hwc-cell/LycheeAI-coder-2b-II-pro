# 训练与评估脚本

这些是 v6 系列的脚本，均已在 Apple Silicon + MLX 上跑通。

## 环境

```bash
pip install mlx-lm          # 0.31.3 验证过
```

基座模型：

```bash
# 训练用 MLX 量化版（省内存，速度快）
mlx_lm.convert --hf-path openbmb/MiniCPM5-2B -q --q-bits 4

# 融合 f16 时需要 HF 原版
huggingface-cli download openbmb/MiniCPM5-2B --local-dir ./MiniCPM5-2B-HF
```

## 数据

`data/train.jsonl` —— 4,526 条，v6.1 训练用的那份。

格式是标准的 MLX-LM 对话 JSONL：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

工具调用是**裸 JSON**，工具结果放在 user 轮：

```
user:      789*123 等于多少
assistant: {"name": "calculate", "arguments": {"expression": "789*123"}}
user:      [工具调用结果] {"name": "calculate", "result": {"value": 97047}}
assistant: 789 × 123 = **97047**。
```

**另有 67 条**用的是"宿主已代为执行"的结构——user 轮里直接把结果给出，没有前置的调用轮，结果也不带 `name`：

```
user:      跑一下这段代码

[工具调用结果] {"exit_code": 0, "stdout": "45\n"}
assistant: 执行成功，输出 `45`。
```

两种结构模型都见过，正常接入用第一种。

## 脚本

| 文件 | 用途 |
|---|---|
| `gen_ood_v6.py` | OOD 数据生成器（v6 系列全部补丁数据的来源，程序化生成） |
| `train_v6_1.sh` | 续训脚本（从已有 adapter 起步，喂全量数据） |
| `eval_v6.py` | 27 项验收（工具调用 / 拒绝 / 无工具 / 多轮 / 一致性） |
| `final_check_v6.py` | 14 项数据质检（冲突答案、重复、格式、配比） |
| `try_run_v7.sh` | **小规模试训**（200 步，改动数据后先跑这个） |
| `regress_v7.py` | **回归对比**（逐项比基线，退化则退出码 2） |
| `k9_patch.py` | 下一轮：运算类型精确配比（补 782 条，拉平到各类型 ~170） |
| `k10_patch.py` | 下一轮：身份问答 OOD 补强（66 条，修"通用 system 下自称 Qwen"） |

---

## ★ 迭代方法：不要「补数据 → 全量重训」

v6 系列五轮迭代的教训：

| 版本 | 验收 | 结果 |
|---|---|---|
| v6 | 24/3 | ⑥⑨ 坏 |
| **v6.1** | **25/2** | 发布 |
| v6.2 | 24/3 | 修好 ①b，坏 ⑧a ⑨g |
| v6.3 | 25/2 | 修好 ⑧a⑨g，又坏 ①b；**逐项与 v6.1 相同（白跑）** |
| v6.4 | 25/2 + ④坏 | 调用率从 60%+ 崩到 12% |

原因是每轮只补 ~5% 数据，却全量重训，把整个能力分布重洗一遍。
`①b`（该调工具）和 `⑨g`/`⑧a`（不该调/收尾）是**跷跷板**。

**正确流程**：

```bash
# 1) 建基线（只做一次）
python scripts/regress_v7.py --save-baseline eval_v6_1_result.txt

# 2) 改数据后，先跑 200 步试训
bash scripts/try_run_v7.sh

# 3) 验收
python scripts/eval_v6.py $MLX_BASE /path/to/adapters-tryrun | tee /tmp/try.txt

# 4) 回归对比 —— 有任何一项退化就回退改法，不进全量
python scripts/regress_v7.py --compare /tmp/try.txt
```

**先算数，再生成**：先统计现有数据各细类的真实条数，算出缺口再生成。
`k9_patch.py` 就是这么做的（配额 add +99 / div +117 / pow +126 /
float_mul +139 / sub +139 / float_add +162 = 782 条）。
v6.4 的 K8 只生成 104 条，而乘法基线 244 条 —— 差距没拉平，所以失败。

### 关键：为什么续训要喂全量数据

只喂补丁样本（比如只加"无工具时不调工具"的样本），模型会学成**一律不调工具**，把之前练好的"该调就调"全练废 —— 这是灾难性遗忘，比原来的问题更难修。

所以 `train_v6_1.sh` 用的是**全量 4,526 条**，通过 `--resume-adapter-file` 从已有权重起步。

### 跑一次续训

```bash
MLX_BASE=./MiniCPM5-2B-MLX \
DATA=./data \
V6_ADAPTER=./adapters \
OUT_ADAPTER=./adapters-v6-2 \
bash scripts/train_v6_1.sh
```

### 跑验收

```bash
python scripts/eval_v6.py <模型路径> [adapter路径]
python scripts/final_check_v6.py
```

## 融合发布

```bash
# f16（注意：必须用 HF 原版基座，不能用 MLX 量化版）
python -m mlx_lm.fuse \
  --model ./MiniCPM5-2B-HF \
  --adapter-path ./adapters \
  --save-path ./out-f16 \
  --dequantize

# MLX 4bit
python -m mlx_lm.fuse \
  --model ./MiniCPM5-2B-MLX \
  --adapter-path ./adapters \
  --save-path ./out-mlx
```
