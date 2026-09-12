---
license: apache-2.0
tags:
  - mlx
  - code
  - minicpm
  - chinese
  - conversational
  - tool-calling
base_model: openbmb/MiniCPM5-2B
---

# LycheeAI-coder-2b-II-pro

> 基于 [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) 微调的轻量编程助手 · Apache 2.0
> **2B 参数，主打本地工具调用**

**这是个人开发者的项目，2B 参数注定它只是个"本地小助手"。** 它比 1B 版更能干，但请别用 GPT / Claude / DeepSeek 的标准要求它。

它最大的用处是：**跑在你自己的机器上，不用联网，能调工具，响应飞快。**

---

## 这一版做了什么

上一版（[1b-II](https://github.com/hwc-cell/LycheeAI-coder-1b-II)）最大的短板是**工具调用完全不会**——训练数据里一条工具样本都没有。这一版专门补上了这块：

| 能力 | 1b-II | **2b-II-pro** |
|---|---|---|
| 工具调用格式 | ❌ 全错、字段编造 | ✅ 裸 JSON，格式稳定 |
| 单步调用 | ❌ | ✅ |
| 多步调用 | ❌ | ✅ |
| 拒绝不该做的事 | ⚠️ 时好时坏 | ✅ |
| 结果解读 | ❌ | ✅ |

**基座换了**：从 MiniCPM5-1B 换成 MiniCPM5-2B（面壁智能，Apache 2.0，512K 上下文）。

---

## 工具调用是什么样

它输出的是**裸 JSON**（OpenAI function calling 风格），不带任何包裹：

```json
{"name": "get_weather", "arguments": {"city": "北京"}}
```

**没有** ```json 代码块围栏、**没有** `<tool_call>` 标签、**没有**"好的我来调用工具"这种旁白。

### 单步调用

```
【用户】北京今天天气怎么样？

【模型】
<think>
用户问今天北京天气，我该调工具吗？有 get_weather 可以，参数是城市。
</think>

{"name": "get_weather", "arguments": {"city": "北京"}}
```

### 多步调用（有依赖时只输出第一个）

```
【用户】查一下北京天气，然后算 123*456

【模型】
{"name": "get_weather", "arguments": {"city": "北京"}}
```

它**不会**一次把两个调用都吐出来——因为第二个需要第一个的结果。等宿主返回天气后，它会继续输出 `calculate` 的调用。

> 这一点是踩过大坑才修好的。基座在零样本时会把调用串台成 XML：
> `<function name="get_weather"><param name="city">Beijing</param></function>`
> —— 格式错了、中文被翻译成英文、第二个调用还丢了。

### 完全独立的调用会并列输出

```
【用户】北京和上海今天天气怎么样？

【模型】
{"name": "get_weather", "arguments": {"city": "北京"}}
{"name": "get_weather", "arguments": {"city": "上海"}}
```

### 该拒绝时会明确拒绝

```
【用户】帮我订一张明天去上海的机票

【模型】
这个我做不了 —— 我没有订票或支付能力。

我能帮你的是：查明天上海的天气、算数学题、或搜相关信息。
```

**关键点：它不会因为手上有 `search_web` 就假装自己能订票。** 工具是接口，不是权限——这是专门训练过的行为。

---

## ⚠️ 已知问题（请务必看）

**身份问答在「宿主提供的通用 system」下会出错。**

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

这个问题会在下一版修掉（补 OOD 样本）。

---

## 快速开始

### MLX（Apple Silicon 原生，推荐）

用融合好的完整模型：

```bash
pip install mlx-lm

python -m mlx_lm.generate \
  --model whcl412/LycheeAI-coder-2b-II-pro \
  --prompt "帮我写个快速排序"
```

起个 OpenAI 兼容的服务：

```bash
python -m mlx_lm.server --model whcl412/LycheeAI-coder-2b-II-pro --port 8080
```

### 用本仓库的 adapter（自己融合）

```bash
# 1. 下载并转换基座
python -m mlx_lm.convert \
  --hf-path openbmb/MiniCPM5-2B \
  --mlx-path ./MiniCPM5-2B-MLX \
  -q --q-bits 4

# 2. 融合 adapter
python -m mlx_lm.fuse \
  --model ./MiniCPM5-2B-MLX \
  --adapter-path ./adapters \
  --save-path ./LycheeAI-coder-2b-II-pro-fused
```

### Ollama（GGUF 版）

```bash
ollama run whcl412/LycheeAI-coder-2b-II-pro-GGUF
```

---

## 工具调用的接入提示

### 格式不是通用的

**这一点很重要**：不同 agent 框架的工具调用格式不一样。

| 框架/模型 | 格式 |
|---|---|
| OpenAI / 大多数本地框架 | 裸 JSON `{"name":..., "arguments":{...}}` |
| Anthropic / Claude Tools | XML 标签 `<function_calls><invoke name="...">` |
| 部分国产模型 | `<tool_call>` 自定义标签包 JSON |
| MCP | 传输层协议，内部通常还是 JSON-RPC |

**它默认输出裸 JSON**。如果你的宿主需要别的格式，在 system 里明确告诉它，它会按你说的来。

### 一次调用 vs 多次调用

- **独立调用**（同时查北京和上海的天气）→ 它会并列输出多行 JSON
- **依赖调用**（先查天气，再根据结果决定要不要提醒带伞）→ 它只输出第一个，等结果回来再继续

宿主需要循环驱动：解析 → 执行 → 把结果喂回去 → 再解析。

### 什么时候该给工具，什么时候别给

它训练时见过三类场景，行为是分开的：

| 类别 | 例子 | 它的行为 |
|---|---|---|
| 工具能算出确定结果 | `789*123`、`1000 美元换人民币` | 调工具 |
| 能力/伦理不允许 | 订机票、破解 WiFi、伪造证明 | 明确拒绝 + 给替代方案 |
| 工具帮不上忙 | 什么是递归、写二分查找 | 直接回答 |

**如果你给它 `calculate` 却不希望它算简单加减**，可以在 system 里说明。默认行为是"算术优先用工具"。

---

## 训练信息

| 项 | 值 |
|---|---|
| 基座 | `openbmb/MiniCPM5-2B` |
| 方法 | LoRA（rank 8，16 层，dropout 0） |
| 数据量 | 2,574 条（1,739 通用 + 835 工具调用） |
| 轮次 | 1.5 epoch（1,930 步） |
| 框架 | MLX-LM |
| 量化 | 4bit（4.501 bits/weight） |
| 峰值内存 | 3.3 GB |

### 工具数据设计

分五批造的，每批解决一个具体问题：

| 批次 | 条数 | 解决什么 |
|---|---|---|
| v2 | 268 | 从零建立工具调用能力（单步/多选/拒绝/解读） |
| v3 | 222 | 纠偏：多步调用串台成 XML、中文参数被翻译、想半天不给结论 |
| v4 | 164 | 修：自称 Qwen、给了工具却心算 |
| v5 | 181 | 修：拒绝能力被"该用工具"压掉、身份全名丢后缀 |

**三类行为严格区分**（这是训练中最容易搞混的地方，我踩过）：

| 类别 | 例子 | 正确行为 |
|---|---|---|
| 工具调用 | `789*123` | 调 `calculate` |
| **真拒绝**（能力/伦理不允许） | 订机票、破解 WiFi | 说不做 + 给替代方案 |
| **无需工具** | 什么是递归、写二分查找 | 直接回答，不提工具 |

把后两类混为一谈，会导致模型把正常提问也当成要拒绝的事。

### 训练曲线

```
Iter 50:   loss 1.121
Iter 150:  loss 0.715
Iter 500-1100: loss 0.48-0.61（平台期）
Iter 1150: loss 0.307 ← 断崖
Iter 1300: loss 0.266
收尾:     loss ~0.24
```

**1150 步那个断崖很关键** —— 如果按 1.0 epoch（1100 步左右）提前停，会卡在平台期。1.5 epoch 是有必要的。

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
- 无工具时的多位数心算（会算错，请给 `calculate`）
- 自称身份的 OOD 场景（见上面"已知问题"）

---

## 许可

Apache 2.0，跟基座 MiniCPM5-2B 一致。

---

<p align="center">
<sub>LycheeAI · 2026</sub>
</p>
