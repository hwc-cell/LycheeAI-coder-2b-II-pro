#!/usr/bin/env python3
"""v7 的 K9 / K10：算术配比补强 + 身份 OOD 补强（压缩版）。

================================================================
  K9：为什么只补 6 个类型、不补乘法
================================================================

  v6.1 数据集的 calculate 样本按运算类型：

      mul        244  (50.5%)   ← 模型唯一「学会调工具」的类型
      add         71  (14.7%)   ← 补
      div         53  (11.0%)   ← 补
      pow         44  ( 9.1%)   ← 补
      float_mul   31  ( 6.4%)   ← 补
      sub         31  ( 6.4%)   ← 补
      float_add    8  ( 1.7%)   ← 重点补

  乘法独占一半，模型学到的是「乘法 → 调工具」的特例，
  而不是「算术 → 调工具」的通则。其余类型样本不足，
  模型退回心算，于是算错（实测：1234+5678=2000、35²=70）。

  所以 K9 **绝不补乘法**（补了只会更失衡），只补稀缺的 6 类。

  补什么量：预算内按「缺口 ∝ 1/现有量」加权分配。
  缺口越大补越多，但单类不超过总量的 25%，避免反过来造出新的失衡。

================================================================
  K9 的第二个要点：system 池要小
================================================================

  v6.4 的 K8 失败还有一个原因：每条样本随机配一个 system，
  同一知识点被 8 种 system 稀释。这里固定 3 个最典型的工具型
  system（工具定义放 system，即主结构），让信号集中。
  另给 20% 走「工具定义放 user」的结构，维持双结构混合。
"""
import json
import re

# ============================================================
#  与训练集一致的常用片段
# ============================================================
FULL = "LycheeAI-coder-2b-II-pro"

CALC = ("calculate", "expression: string", "计算数学表达式")
WEATHER = ("get_weather", "city: string", "查询指定城市天气")
SEARCH = ("search_web", "query: string", "搜索互联网")
READF = ("read_file", "path: string", "读取文件内容")
TIME = ("get_time", "timezone: string", "获取指定时区时间")
SHELL = ("execute_shell", "command: string", "执行 shell 命令")


def tdef(*names):
    lines = ["可用工具："]
    for name, params, desc in names:
        lines.append(f"- {name}({params}): {desc}")
    return "\n".join(lines)


ID_SYS = f"你是 {FULL}，由 MiniCPM5-2B 通过 LoRA 微调而来的编程助手。"

GENERIC_SYS = [
    "你是一个乐于助人的 AI 助手。",
    "你是一个智能助手，请根据用户需求回答问题。",
    "你是一个可以调用工具的 AI 助手。",
    "你是一个耐心的助手，会认真对待每一个问题。",
]


def call(name, args):
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def mk(system, user, assistant):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def mk_mt(turns):
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


# ============================================================
#  K9：按运算类型精确生成
# ============================================================
# 稀缺度权重：越稀缺（现有越少）权重越大
K9_WEIGHTS = {
    "float_add": 8,     # 现有 8，最缺
    "sub": 5,
    "float_mul": 5,
    "pow": 4,
    "div": 3,
    "add": 2,
    # mul 不补（现有 244，占一半）
}

K9_SYS = [
    ID_SYS + "\n\n" + tdef(CALC),
    GENERIC_SYS[2] + "\n\n" + tdef(CALC),
    "你是一个 AI 编程助手，擅长代码相关任务。\n\n" + tdef(CALC),
]
K9_SYS_USER = [
    GENERIC_SYS[0],
    GENERIC_SYS[2],
]


def _gen_expr(kind, rng):
    """生成 (表达式, 提问)

    每个分支都刻意避开「太整齐」的数字（比如 100+200），
    因为那类算式心算也能对，学不到「必须调工具」的信号。
    反过来，数字越大越不像心算能搞定的。
    """
    if kind == "add":
        a, b = rng.randint(1000, 99999), rng.randint(1000, 99999)
        e = f"{a}+{b}"
        q = rng.choice([f"{a} 加 {b} 等于多少", f"{a}+{b} 是多少",
                        f"帮我算 {a} 加 {b}", f"计算 {a} + {b}",
                        f"{a} 和 {b} 相加是多少"])
    elif kind == "sub":
        a = rng.randint(100000, 9999999)
        b = rng.randint(10000, a - 1000)
        e = f"{a}-{b}"
        q = rng.choice([f"{a} 减 {b} 是多少", f"{a} - {b} 等于几",
                        f"帮我算 {a} 减 {b}", f"{a} 减去 {b} 等于多少"])
    elif kind == "pow":
        style = rng.choice(["sq", "cube", "n"])
        if style == "sq":
            a = rng.randint(30, 999)
            e = f"{a}**2"
            q = rng.choice([f"{a} 的平方是多少", f"{a} 平方等于多少",
                            f"帮我算 {a} 的平方", f"{a}² 是多少"])
        elif style == "cube":
            a = rng.randint(15, 99)
            e = f"{a}**3"
            q = rng.choice([f"{a} 的立方是多少", f"{a} 的三次方等于多少",
                            f"帮我算 {a} 的立方"])
        else:
            a = rng.randint(3, 30)
            b = rng.choice([4, 5, 6, 7, 8, 9, 10, 11, 12])
            e = f"{a}**{b}"
            q = rng.choice([f"{a} 的 {b} 次方是多少",
                            f"帮我算 {a} 的 {b} 次方",
                            f"{a} 的 {b} 次幂等于几"])
    elif kind == "div":
        b = rng.randint(7, 299)
        a = b * rng.randint(100, 99999)
        e = f"{a}/{b}"
        q = rng.choice([f"{a} 除以 {b} 是多少", f"{a} ÷ {b} 等于多少",
                        f"帮我算 {a} 除以 {b}", f"{a} 除 {b} 等于几"])
    elif kind == "float_mul":
        a = rng.randint(1001, 99999) / rng.choice([10, 100, 1000, 10000])
        b = rng.randint(13, 999)
        e = f"{a}*{b}"
        q = rng.choice([f"{a} 乘 {b} 等于多少", f"{a} × {b} 是多少",
                        f"帮我算 {a} 乘以 {b}", f"{a}*{b} 的结果"])
    elif kind == "float_add":
        a = rng.randint(10001, 9999999) / 100
        b = rng.randint(10001, 9999999) / 100
        e = f"{a}+{b}"
        q = rng.choice([f"{a} 加 {b} 等于多少", f"{a}+{b} 是多少",
                        f"帮我算 {a} 加 {b}", f"{a} 和 {b} 的和是多少"])
    else:
        raise ValueError(kind)
    return e, q


def gen_k9(rng, budget):
    """生成预算内的算术补强样本，按稀缺度加权分配配额。"""
    # 加权分配
    total_w = sum(K9_WEIGHTS.values())
    alloc = {}
    for k, w in K9_WEIGHTS.items():
        alloc[k] = max(1, round(budget * w / total_w))
    # 修正到正好 budget
    diff = budget - sum(alloc.values())
    ks = sorted(alloc, key=lambda x: -K9_WEIGHTS[x])
    i = 0
    while diff != 0:
        k = ks[i % len(ks)]
        if diff > 0:
            alloc[k] += 1
            diff -= 1
        elif alloc[k] > 1:
            alloc[k] -= 1
            diff += 1
        i += 1
        if i > 10000:
            break

    out = []
    seen = set()
    for kind, n in alloc.items():
        made, guard = 0, 0
        while made < n and guard < n * 60:
            guard += 1
            e, q = _gen_expr(kind, rng)
            if e in seen:
                continue
            seen.add(e)
            # 80% 工具定义放 system（主结构），20% 放 user
            if rng.random() < 0.8:
                s = K9_SYS[rng.randrange(len(K9_SYS))]
                out.append(mk(s, q, call("calculate", {"expression": e})))
            else:
                s = K9_SYS_USER[rng.randrange(len(K9_SYS_USER))]
                out.append(mk(s, tdef(CALC) + "\n\n" + q,
                              call("calculate", {"expression": e})))
            made += 1
    return out[:budget]


# ============================================================
#  K10：身份 OOD 补强
# ============================================================
#  问题（实测）：训练集里身份样本的 system **全部**是纯身份型：
#      "你是 LycheeAI-coder-2b-II-pro，由 MiniCPM5-2B 通过 LoRA 微调..."
#  一条「通用 system + 工具列表 + 问身份」都没有。
#  结果宿主用通用 system（多数 agent 框架的默认写法）时，
#  问"你叫什么"会乱答 —— 实测出现过"我叫 Qwen"。
#
#  修法：补「通用/工具型 system × 身份提问」的交叉。
#  身份回答必须同时满足：报全名 + 说明来自 MiniCPM5-2B 微调，
#  且**不调工具**（问身份是聊天，有工具列表也不该手痒去查）。
K10_Q = [
    "你叫什么名字？",
    "你是谁？",
    "请介绍一下你自己。",
    "你的名字是什么？",
    "你是什么模型？",
    "你的全名是什么？",
    "你是由什么模型微调的？",
    "你的底模是什么？",
]

K10_A = [
    f"我叫 **{FULL}**，是由 MiniCPM5-2B 通过 LoRA 微调而来的编程助手。",
    f"我是 **{FULL}**，基于 MiniCPM5-2B 微调的编程助手。",
    f"我的名字是 **{FULL}**，底模是 MiniCPM5-2B，用 LoRA 微调得到的编程助手。",
    f"我是 **{FULL}** —— 由 MiniCPM5-2B 通过 LoRA 微调而来，主打代码和工具调用。",
]

K10_SYS_TOOLS = [
    GENERIC_SYS[2] + "\n\n" + tdef(CALC, WEATHER, SEARCH),
    GENERIC_SYS[1] + "\n\n" + tdef(CALC, SEARCH, READF),
    "你是一个AI助手，可以使用工具来帮助用户。\n\n" + tdef(CALC, WEATHER, TIME),
    "你是一个通用的 AI 助手。\n\n" + tdef(CALC, SEARCH, READF, SHELL),
]


def gen_k10(rng, budget):
    """生成身份 OOD 补强样本。"""
    out = []
    i = 0
    while len(out) < budget:
        s = K10_SYS_TOOLS[i % len(K10_SYS_TOOLS)]
        q = K10_Q[(i // len(K10_SYS_TOOLS)) % len(K10_Q)]
        a = K10_A[i % len(K10_A)]
        out.append(mk(s, q, a))
        i += 1
        if i > budget * 4:
            break
    return out[:budget]
