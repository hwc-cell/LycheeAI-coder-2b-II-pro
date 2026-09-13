---
license: apache-2.0
tags:
  - code
  - minicpm
  - chinese
  - conversational
  - tool-calling
  - mlx
  - quantization
base_model: openbmb/MiniCPM5-2B
pipeline_tag: text-generation
library_name: mlx
---

# LycheeAI-coder-2b-II-pro

[中文](#中文) | [English](#english)

> 基于 [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) 微调的轻量编程助手 · Apache 2.0
> **2B 参数，主打本地工具调用** · 当前版本 **v7**（验收 26/27）

**这是个人开发者的项目，2B 参数注定它只是个"本地小助手"。** 它比 1B 版更能干，但请别用 GPT / Claude / DeepSeek 的标准要求它。

它最大的用处是：**跑在你自己的机器上，不用联网，能调工具，响应飞快。**

**本仓库是项目主页**，放训练脚本、数据和 LoRA 适配器（`adapters/`）。
完整权重在模型站：

| 版本 | 大小 | 用途 | 下载 |
|---|---|---|---|
| **MLX 4bit** | 1.4 GB | Apple Silicon 开箱即用 | [ModelScope](https://www.modelscope.cn/models/whcl412/LycheeAI-coder-2b-II-pro) · [HF](https://huggingface.co/whcl412/LycheeAI-coder-2b-II-pro) |
| **f16 (bf16)** | 5.03 GB | 转 GGUF、二次量化、继续微调 | [ModelScope](https://www.modelscope.cn/models/whcl412/LycheeAI-coder-2b-II-pro-f16) · [HF](https://huggingface.co/whcl412/LycheeAI-coder-2b-II-pro-f16) |

---

# 中文

## v7（最新）

**用「精选 980 条 × 1 epoch」替代「全量 5,341 条 × 0.22 epoch」，验收 26/27 通过**（上一版 v6.1 是 25/27）。

被限制在 500 步以内训练（`步数 = ceil(条数/2)` → 最多 1000 条），所以换了个思路：**不砍步数，改砍数据，但按行为重要性重新配比**。

| 做了什么 | 结果 |
|---|---|
| 按行为重要性重新配比（不按原始比例） | 多轮链路从 10.7% 提到 20.5%；单轮闲聊从 51.4% 压到 13.3% |
| 算术「两头夹」：补稀缺 + 削过剩乘法 | 乘法占比 50.5% → 20.0%，7 类极差 30.5x → 2.4x |
| 修掉源数据 9 处工具名幻觉 | 零幻觉（调用未定义的工具） |
| 采样改成内容哈希排序 | 改配额不再重洗 40% 样本，实验可归因 |

**直接效果**：修好了两个长期失败的验收项——**①b 第二道算术题**、**⑧b 多轮追问**。

唯一未通过：**⑥ 工具定义放在 user 消息里时会退化**（属兼容性场景，用 system 放工具定义即正常）。试过两个方向修它（v7b → 23/4、v7c → 22/5）都变差，说明 **26/1 已是这个预算下的最优解**。

详细复盘见下方「迭代历程」。

---

## 这一版做了什么

上一版（1b-II）最大的短板是**工具调用完全不会**——训练数据里一条工具样本都没有。这一版专门补上了这块：

| 能力 | 1b-II | **2b-II-pro** |
|---|---|---|
| 工具调用格式 | ❌ 全错、字段编造 | ✅ 裸 JSON，格式稳定 |
| 单步调用 | ❌ | ✅ |
| 多步调用 | ❌ | ✅ |
| 拒绝不该做的事 | ⚠️ 时好时坏 | ✅ |
| 结果解读 | ❌ | ✅ |

**基座换了**：从 MiniCPM5-1B 换成 MiniCPM5-2B（面壁智能，Apache 2.0，512K 上下文）。

---

## 快速开始

### MLX（Apple Silicon）

```bash
pip install mlx-lm

# 直接加载
python -m mlx_lm.generate \
  --model whcl412/LycheeAI-coder-2b-II-pro \
  --prompt "帮我写个快速排序"

# 起个 OpenAI 兼容的服务
python -m mlx_lm.server --model whcl412/LycheeAI-coder-2b-II-pro --port 8080
```

国内下载慢的话，把 `--model` 换成 `modelscope` 上的同名仓库即可。

### transformers / GGUF

MLX 4bit 格式 transformers 不能直接加载，需要 HF safetensors 用 **f16 版**。
GGUF 尚未发布，可从 f16 版自行转换（`llama.cpp/convert_hf_to_gguf.py`）。

### 自己训练

```bash
# 环境
pip install mlx-lm

# 数据在 data_coder2b_pro_v6_final/
# 用 adapters/ 里的配置做续训，或从零训
mlx_lm.lora --model <MiniCPM5-2B-MLX> --train --data data_coder2b_pro_v6_final \
  --fine-tune-type lora --num-layers 16 --batch-size 2 --max-seq-length 2048 \
  --iters 2671 --learning-rate 5e-5 --adapter-path <输出目录> \
  --grad-checkpoint --save-every 600
```

---

## 工具调用

### 输出格式

它输出的是**裸 JSON**（OpenAI function calling 风格），不带任何包裹：

```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```

**没有** ```json 代码块围栏、**没有** `<tool_call>` 标签、**没有**"好的我来调用工具"这种旁白。

### 单步调用

```
system: 你是 LycheeAI-coder-2b-II-pro。可以使用工具来帮助用户。

可用工具：
- get_weather(city: string): 查询指定城市的天气

user:   北京今天天气怎么样？
```

输出：

```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```

### 多步调用（有依赖时只输出第一个）

问"北京今天适合出门吗"，它需要先查天气再判断，此时**只输出第一步**：

```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```

你把工具结果喂回去，它再继续。**宿主需要循环驱动**：解析 → 执行 → 喂回结果 → 再解析。

### 完全独立的调用会并列输出

同时查北京和上海，它会一次给两行：

```json
{"name": "get_weather", "arguments": {"city": "北京"}}
{"name": "get_weather", "arguments": {"city": "上海"}}
```

### 该拒绝时会明确拒绝

不是所有请求都调工具。超出能力或不合规的，它会说明原因并给替代方案，不会硬编一个工具调用糊弄过去。

---

## 宿主接入注意

### 格式不是通用的

**这一点很重要**：不同 agent 框架的工具调用格式不一样。

| 框架/模型 | 格式 |
|---|---|
| OpenAI / 大多数本地框架 | 裸 JSON `{"name":..., "arguments":{...}}` |
| Anthropic / Claude Tools | XML 标签 `<function_calls><invoke name="...">` |
| 部分国产模型 | `<tool_call>` 自定义标签包 JSON |
| MCP | 传输层协议，内部通常还是 JSON-RPC |

**它默认输出裸 JSON**。如果你的宿主需要别的格式，在 system 里明确告诉它，它会按你说的来。

### 什么时候给工具，什么时候别给

它训练时见过四类场景，行为是分开的：

| 类别 | 例子 | 它的行为 |
|---|---|---|
| 工具能算出确定结果 | `789*123`、`1000 美元换人民币` | 调工具 |
| 能力/伦理不允许 | 订机票、破解 WiFi、伪造证明 | 明确拒绝 + 给替代方案 |
| 工具帮不上忙 | 什么是递归、写二分查找 | 直接回答 |
| **宿主没有工具** | 纯聊天软件里的"帮我搜一下" | 说明没有这个能力，不硬调 |

**最后一行很关键**：如果你的宿主是纯聊天（微信、客服机器人），**不要给它任何工具定义**，它会自然地说"我这边没有联网能力"，而不是吐一坨宿主接不住的 JSON。

**如果你给它 `calculate` 却不希望它算简单加减**，可以在 system 里说明。默认行为是"算术优先用工具"。

---

## 训练信息

| 项 | 值 |
|---|---|
| 基座 | `openbmb/MiniCPM5-2B` |
| 方法 | LoRA（rank 8，dropout 0，scale 20，16 层） |
| 数据量 | 5,341 条 |
| 轮次 | 1 epoch（2,671 步） |
| 学习率 | 5e-5（adam） |
| batch / seq | 2 / 2048 |
| 框架 | MLX-LM 0.31.3 |
| 本版本量化 | 4bit affine（group_size 64，4.501 bits/weight） |

### 数据构成（v6 系列）

v6 是在 v5 基础上做的**定向修复**，共四批：

| 批次 | 条数 | 解决什么 |
|---|---|---|
| v5 基础 | 4,526 | v6.1 起点（v5 + OOD 初版） |
| v6.1 | 4,526 | 修 v6 的 ⑥⑨ 退步项，OOD 补强 |
| v6.2 | 4,778 | 补算术泛化（K6，252 条程序化生成） |
| v6.3 | 5,093 | 补能力边界 + 结果收尾（K7，315 条） |
| **v6.4** | 5,341 | 补运算类型均衡（K8，248 条） |

**发布的是 v6.1**（不是 v6.4）。原因见下面"关于迭代"。

### 三类行为严格区分

这是训练中最容易搞混的地方：

| 类别 | 例子 | 正确行为 |
|---|---|---|
| 工具调用 | `789*123` | 调 `calculate` |
| **真拒绝**（能力/伦理不允许） | 订机票、破解 WiFi | 说不做 + 给替代方案 |
| **无需工具** | 什么是递归、写二分查找 | 直接回答，不提工具 |

混为一谈会导致模型把正常提问也当成要拒绝的事——踩过这个坑。

---

## 能力边界

**能做：**
- 写代码、调 bug、解释技术概念
- 单步 / 多步工具调用（裸 JSON）
- 拒绝能力范围外或不合规的请求
- 解读工具返回结果
- 中文 / 英文 / 粤语

**不擅长：**
- 复杂多步推理（2B 容量限制）
- 超长上下文（虽然基座支持 512K，但实际长文表现没测）
- **无工具时的算术**（会算错，请给 `calculate`——见下面"已知问题"）
- 自称身份的 OOD 场景（见下面"已知问题"）

---

## ⚠️ 已知问题

### 1. 无工具时会心算，而且**算错**

这是当前最实际的短板。实测：

```
1234 + 5678 = 2000      (正确 6912)
35 ** 2     = 70        (正确 1225)
13.5 × 4    = 7.8       (正确 54)
199 × 257   = 40043     (正确 51143)
```

**根因**（做过对照实验确认）：模型是按**运算类型**决定调不调工具的，而这跟训练数据里各类型的样本量完全对应。

| 运算类型 | 训练样本量 | 行为 |
|---|---|---|
| 乘法 `*` | 432 | ✅ 调工具 |
| 加法 `+` | 86 | ❌ 心算 |
| 幂 `**` | 61 | ❌ 心算 |
| 小数 | 61 | ❌ 心算且算错 |
| 减法 `-` | 49 | ❌ 心算 |

不是"字符串记忆"（见过 38% vs 没见过 50%，差不多），也不是"难度"（`1234+5678` 也在心算）。

**规避方法**：**只要涉及算术，就在 system 里明确要求"任何计算都必须用 calculate"**：

```
需要任何数值计算时，一律调用 calculate，不要自己心算。
```

这条指令实测有效。v6.4 试图用数据修正（K8，按运算类型均衡），但引入了更严重的副作用（工具调用率从 60%+ 崩到 12%），所以**发布了 v6.1 而没有发布 v6.4**。

### 2. 身份问答在「宿主提供的通用 system」下会出错

如果你的 system prompt 是这样（很多 agent 框架的默认写法）：

```
你是一个AI助手，可以使用工具来帮助用户。

可用工具：
- get_weather(city: str): 查询指定城市的天气
...

需要调用工具时，请直接输出 JSON：{"name": "工具名", "arguments": {...}}
```

那么问"你叫什么名字"时，它可能乱答（实测出现过"我叫 Qwen"）。

**但如果 system 是它训练时的格式，就完全正常：**

```
你是 LycheeAI-coder-2b-II-pro，由 MiniCPM5-2B 通过 LoRA 微调而来的编程助手。
```

→ 答"我叫 LycheeAI-coder-2b-II-pro，由 MiniCPM5-2B 微调而来的编程助手。"

**原因**：训练数据的身份样本里，system 全部是上面第二种。模型没见过"通用 system + 工具列表 + 问身份"这个组合，没有训练信号。

**规避方法**：接入时把身份写进 system。比如：

```
你是 LycheeAI-coder-2b-II-pro。可以使用工具来帮助用户。

可用工具：
...
```

### 3. 两种 system 结构的一致性

训练数据混了两种结构：工具定义放 `system` / 放 `user`。实测两种都能用，但**放 system 更稳**（这也是主流 agent 框架的做法）。

### 4. 工具结果有两种结构，宿主需要知道

模型见过两种上下文，**两种都认**：

**结构 A** —— 有明确的调用轮（主流 agent 框架的样子）：

```
user:      789*123 等于多少
assistant: {"name": "calculate", "arguments": {"expression": "789*123"}}
user:      [工具调用结果] {"name": "calculate", "result": {"value": 97047}}
assistant: 789 × 123 = **97047**。
```

**结构 B** —— 宿主已代为执行，直接把结果给模型（67 条早期样本用这个）：

```
user:      跑一下这段代码

[工具调用结果] {"exit_code": 0, "stdout": "45\n"}
assistant: 执行成功，输出 `45`。
```

**区别**：结构 B 的结果**不带 `name`**，因为上下文里本来就没有调用轮，工具身份从 system 的工具列表和问题本身就能推断。

**建议**：正常接入用**结构 A**（多轮循环）。只有在你已经知道该调什么、只想让模型解读结果时，才用结构 B。

---

## 关于迭代：为什么发 v6.1 而不是 v6.4

v6 系列一共训练了 5 版，逐代验收：

| 版本 | 验收 | 结果 |
|---|---|---|
| v6 | 24 / 3 | ⑥⑨ 坏，①b 好 |
| **v6.1** | **25 / 2** | ✅ 发布 |
| v6.2 | 24 / 3 | 修好 ①b，但 ⑧a ⑨g 退步（⑨g 幻觉"我能联网"） |
| v6.3 | 25 / 2 | 修好 ⑧a ⑨g，①b 又退；**逐项与 v6.1 完全一致（等于白跑）** |
| v6.4 | 25/2 + ④坏 | 加 K8 后调用率崩到 12%，出现"2**10 是左赋值"等胡说 |

**v6.1 是唯一相对 v6 有真实提升的版本**（+1 项，且 v6.3 与它逐项一致）。

**教训**：每轮都是「发现问题 → 补 5% 数据 → 全量重训」，把整个能力分布重洗了一遍，结果修一个坏一个（①b ↔ ⑨g/⑧a 是跷跷板）。补丁数据的占比太小，压不住已有的行为惯性。

**下一版会改方法**：先小规模试训验证无副作用，再进全量；按运算类型精确补齐，而不是笼统加"算术题"。

---

## 许可

Apache 2.0，跟基座 MiniCPM5-2B 一致。

训练数据、脚本都在项目里，欢迎 issue 交流。

---

# English

## What's new

The biggest gap in the previous release (1b-II) was that it **couldn't call tools at all** — the training data contained zero tool-use samples. This release is built specifically to fix that:

| Capability | 1b-II | **2b-II-pro** |
|---|---|---|
| Tool-call format | ❌ malformed, invented fields | ✅ bare JSON, stable |
| Single-step calls | ❌ | ✅ |
| Multi-step calls | ❌ | ✅ |
| Refusing what it shouldn't do | ⚠️ inconsistent | ✅ |
| Interpreting tool results | ❌ | ✅ |

**New base model**: switched from MiniCPM5-1B to MiniCPM5-2B (OpenBMB, Apache 2.0, 512K context).

**This repo is the project home** — training scripts, data, and the LoRA adapter (`adapters/`).
Full weights live on the model hubs:

| Build | Size | Use case |
|---|---|---|
| **MLX 4-bit** | 1.4 GB | ready to run on Apple Silicon |
| **f16 (bf16)** | 5.03 GB | GGUF conversion, re-quantization, further fine-tuning |

---

## Quick start

### MLX (Apple Silicon)

```bash
pip install mlx-lm

python -m mlx_lm.generate \
  --model whcl412/LycheeAI-coder-2b-II-pro \
  --prompt "write a quicksort"

python -m mlx_lm.server --model whcl412/LycheeAI-coder-2b-II-pro --port 8080
```

### transformers / GGUF

MLX 4-bit cannot be loaded by transformers — use the f16 build for HF safetensors.
GGUF is not published yet; convert it from the f16 build yourself (`llama.cpp/convert_hf_to_gguf.py`).

---

## Tool calling

### Output format

It emits **bare JSON** (OpenAI function-calling style), with no wrapper at all:

```json
{"name": "get_weather", "arguments": {"city": "Beijing"}}
```

**No** ```json fences, **no** `<tool_call>` tags, **no** "Sure, let me call the tool" narration.

### Multi-step calls (only the first step is emitted when dependent)

For "Is it a good day to go out in Beijing?", it needs to check the weather first. In that case it **emits only the first step**:

```json
{"name": "get_weather", "arguments": {"city": "Beijing"}}
```

Feed the tool result back and it continues. **Your host must drive the loop**: parse → execute → feed result back → parse again.

### Independent calls are emitted in parallel

```json
{"name": "get_weather", "arguments": {"city": "Beijing"}}
{"name": "get_weather", "arguments": {"city": "Shanghai"}}
```

### It refuses clearly when it should

Not every request becomes a tool call. When something is out of scope or disallowed, it says so and offers alternatives instead of fabricating a tool call.

---

## When to give it tools, and when not to

| Category | Example | Behavior |
|---|---|---|
| Tool gives a definite answer | `789*123`, `1000 USD to CNY` | calls the tool |
| Out of scope / disallowed | booking flights, cracking WiFi | refuses + offers alternatives |
| Tool doesn't help | "what is recursion", "write binary search" | answers directly |
| **Host has no tools** | "search for me" in a plain chat app | says it can't, doesn't emit JSON |

**The last row matters**: if your host is a plain chat app, **give it no tool definitions at all**. It will say "I have no web access" instead of emitting JSON your host can't handle.

---

## Training

| Item | Value |
|---|---|
| Base | `openbmb/MiniCPM5-2B` |
| Method | LoRA (rank 8, dropout 0, scale 20, 16 layers) |
| Data | 5,341 samples |
| Epochs | 1 (2,671 steps) |
| LR | 5e-5 (adam) |
| batch / seq | 2 / 2048 |
| Framework | MLX-LM 0.31.3 |
| Quantization | 4-bit affine (group_size 64) |

---

## ⚠️ Known issues

### 1. It does mental arithmetic without tools — and gets it wrong

```
1234 + 5678 = 2000      (should be 6912)
35 ** 2     = 70        (should be 1225)
13.5 × 4    = 7.8       (should be 54)
199 × 257   = 40043     (should be 51143)
```

**Root cause** (confirmed by controlled experiment): the model decides based on **operation type**, and this maps exactly onto per-type sample counts in the training data — multiplication 432 samples → calls the tool; addition 86 / power 61 / subtraction 49 → does it in its head.

It is **not** string memorization (38% vs 50%, no real difference), and **not** difficulty (`1234+5678` also gets mental math).

**Workaround**: state it explicitly in the system prompt:

```
需要任何数值计算时，一律调用 calculate，不要自己心算。
```

### 2. Identity questions fail under a generic system prompt

If your system prompt is the generic "You are an AI assistant with tools..." style, asking "what's your name" may produce nonsense (observed: "I'm Qwen").

Put the identity in the system prompt:

```
You are LycheeAI-coder-2b-II-pro. You have access to tools.
```

Cause: every identity sample in the training data used the specific system prompt above; the model never saw the generic-tools-plus-identity combination.

### 3. Tool results come in two shapes — your host should know which it's sending

Both are understood:

**Shape A** — an explicit call turn (what mainstream agent frameworks do):

```
user:      789*123 等于多少
assistant: {"name": "calculate", "arguments": {"expression": "789*123"}}
user:      [工具调用结果] {"name": "calculate", "result": {"value": 97047}}
assistant: 789 × 123 = **97047**。
```

**Shape B** — the host already executed it and hands the result straight over (used by 67 early samples):

```
user:      跑一下这段代码

[工具调用结果] {"exit_code": 0, "stdout": "45\n"}
assistant: 执行成功，输出 `45`。
```

**Difference**: shape B has no `name` in the result, because there is no call turn in context — the tool identity is inferable from the system tool list and the question itself.

**Recommendation**: use **shape A** for normal integration (multi-turn loop). Use shape B only when you already know which tool to run and just want the model to interpret the result.

### 4. Why v6.1 and not v6.4

Five iterations were trained. v6.2 fixed one item and broke two; v6.3 was item-for-item identical to v6.1; v6.4 collapsed the tool-call rate to 12%. **v6.1 is the only version with a real improvement over v6** (25/2 vs 24/3 on a 27-item eval).

Lesson: each round patched ~5% of the data and retrained on everything, reshuffling the whole behavior distribution. The next version will validate on small-scale runs first.

---

## License

Apache 2.0, same as the MiniCPM5-2B base.

---

<p align="center">
<sub>LycheeAI —— 小的，但能干活。</sub><br>
<sub>LycheeAI — small, but it gets the job done.</sub>
</p>
