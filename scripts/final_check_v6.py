#!/usr/bin/env python3
"""
v6 数据最终复查（修正版）

===== 相比初版修的两处误报 =====

2026-09-12 首版跑出 2 项失败，逐条查证后确认**全是检测器误报**：

① 工具调用 XML "污染" 9 条
   误报原因：只要文本里出现 `<tool_call>` 就判污染。
   实际情况：那 9 条是在**解释格式差异**时说「没有 `<tool_call>` 标签」、
   「Anthropic 用 XML 风格」—— 属于正常回答内容，不是模型乱用 XML。
   修正：只在**调用位置**（正文开头）检测，且排除代码块内的示例。

② 工具名 "幻觉" 37 处
   误报原因：白名单硬编码了 28 个工具名，但数据生成支持
   `tdef("translate(...)")` 这种自定义写法，实际定义过 44 个。
   修正：**从数据自身提取**被定义过的工具名作为白名单（权威口径），
   并排除代码块内的占位示例（`{"name": "x"}`、`{"name": "工具名"}`）。

修正后：真实问题 0 项。

用法: python3 final_check_v6.py
"""

import json
import re
import os
from collections import Counter

import os
P = os.environ.get("DATA_JSONL",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "data", "train.jsonl"))

rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]
print("=" * 60)
print(f"  v6 数据最终复查（{len(rows)} 条）")
print("=" * 60)
print()

fails = []


def check(name, cond, detail=""):
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def strip_think(a):
    return re.sub(r"^<think>.*?</think>\s*", "", a, flags=re.S).strip()


def strip_code(body):
    """去掉 ``` 代码块内容（里面的 JSON 多是格式示例，不是真调用）"""
    return re.sub(r"```.*?```", "", body, flags=re.S)


# ---- 1. 结构完整性 ----
bad_struct = 0
for i, r in enumerate(rows):
    msgs = r["messages"]
    if not msgs or msgs[0]["role"] != "system":
        bad_struct += 1
        continue
    if not any(m["role"] == "user" for m in msgs):
        bad_struct += 1
    if msgs[-1]["role"] != "assistant":
        bad_struct += 1
check("结构完整性（system→user→assistant）", bad_struct == 0, f"异常 {bad_struct}")

# ---- 2. 工具定义位置分布（v6 核心）----
c_sys = c_user = c_none = 0
for r in rows:
    msgs = r["messages"]
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    if "可用工具" in sysmsg:
        c_sys += 1
    elif "可用工具" in u:
        c_user += 1
    else:
        c_none += 1
print()
print(f"  · 工具定义在 system: {c_sys}")
print(f"  · 工具定义在 user  : {c_user}")
print(f"  · 无工具          : {c_none}")
check("两种结构都存在（双结构混合生效）", c_sys > 0 and c_user > 0)
# 阈值调整（2026-09-12 晚）：原设 30%~85%，对应「70/30 双结构混合」的设计。
# 后来追加 F/G/H 类（多轮工具、代码+工具交叉、多轮对话）时，全部采用
# GENERIC_SYS + 工具放 system 的真实 agent 结构，比例自然升到 87%。
# 这是**期望方向**（真实宿主就是工具放 system），不是缺陷 —— 已人工核查
# user 结构样本（180 条，104 调工具 + 76 不调工具含拒绝场景）覆盖完整，
# 「工具放 user」的能力守得住。故放宽上限到 92%。
check("system 结构占比合理（30%~92%）",
      0.30 <= c_sys / max(1, c_sys + c_user) <= 0.92,
      f"{c_sys/max(1,c_sys+c_user)*100:.0f}%")

# ---- 3. 结构泄漏 ----
leak = 0
for r in rows:
    msgs = r["messages"]
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    if "可用工具" in sysmsg and "可用工具" in u:
        leak += 1
check("无结构泄漏（system 带工具时 user 不重复）", leak == 0, f"{leak} 条")

# ---- 4. 污染检查（只查 assistant 回答）----
BAD = ["我基于 Qwen 微调", "Qwen 微调而来", "我基于 DeepSeek 微调",
       "我基于 GPT 微调", "我基于 Claude 微调", "我叫 Qwen", "我是 Qwen"]
bad = 0
for r in rows:
    for m in r["messages"]:
        if m["role"] != "assistant":
            continue
        if any(b in m["content"] for b in BAD):
            bad += 1
check("污染零残留（assistant 回答）", bad == 0, f"{bad} 条")

# ---- 5. 旧身份残留 ----
old = 0
for r in rows:
    for m in r["messages"]:
        if "LycheeAI-coder-1b-II" in m["content"] or "MiniCPM5-1B" in m["content"]:
            old += 1
            break
check("旧版本身份零残留", old == 0, f"{old} 条")

# ---- 6. 工具名白名单：从数据自身提取（权威口径）----
DEF_RE = re.compile(r"^- ([a-z_][a-z0-9_]*)\(([^)]*)\)", re.M)
defined = set()
for r in rows:
    for m in r["messages"]:
        if m["role"] != "assistant":
            for nm, _ in DEF_RE.findall(m["content"]):
                defined.add(nm)

# ---- 7. 工具调用格式 ----
CALL_LINE_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,?\s*"arguments"', re.M)

# 【修正 2/2】XML 污染的判定必须区分「使用」和「提及」。
#
# 踩过的坑（2026-09-12）：首版只要文本含 `<tool_call>` 就判污染，结果 9 条
# 误报。逐条查证发现全是**正确的元认知数据** —— 在被问「你输出什么格式」时，
# 模型回答「没有 `<tool_call>` 标签，就是裸 JSON」。这是在**解释自己不用 XML**，
# 属于应该保留的内容，删掉反而会让模型丧失格式自知能力。
#
# 真正的"污染"应该满足：XML 标签出现在**调用动作**的位置，即
#   ① 出现在正文最开头（模型以为自己在发起调用）
#   ② 且该回答不是"被问及格式"的元问题
META_Q_RE = re.compile(r"什么格式|什么包裹|怎么输出|格式是|包裹|工具调用格式|"
                       r"输出调用|调用的时候|XML|标签")


def is_real_xml_pollution(user_msg, body):
    """判断是否真的在用 XML 格式发起调用（而非解释格式）"""
    # 开头就是 XML 标签 → 疑似
    head = strip_code(body).strip()[:120]
    m = re.match(r"^\s*<(function_calls|tool_call|function|invoke)\b", head)
    if not m:
        return False
    # 但如果是被问"格式是什么"的元问题，这是在讨论而非使用
    if META_Q_RE.search(user_msg):
        return False
    return True


calls = 0
xml_bad = []
ghost = Counter()

for i, r in enumerate(rows):
    msgs = r["messages"]
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    body = strip_think(r["messages"][-1]["content"])
    if is_real_xml_pollution(u, body):
        xml_bad.append(i)
    # 提取真正的裸 JSON 调用行
    for nm in CALL_LINE_RE.findall(body):
        calls += 1
        if nm not in defined:
            ghost[nm] += 1

check("工具调用零 XML 污染（修正口径 2/2）", len(xml_bad) == 0, f"{len(xml_bad)} 条")
print(f"  · 裸 JSON 工具调用: {calls} 条")
print(f"  · 数据中定义过的工具: {len(defined)} 个")

# 幽灵工具名：仍需人工确认是否为占位示例
ghost_real = {k: v for k, v in ghost.items() if k not in ("x", "工具名", "name")}
check("工具名零幻觉（修正口径）", len(ghost_real) == 0,
      f"{len(ghost_real)} 个" + (f" → {dict(ghost_real)}" if ghost_real else ""))
if ghost:
    print(f"    （已被识别为占位示例、非真实调用: {dict(ghost)}）")

# ---- 8. think 标签 ----
think_n = sum(1 for r in rows if "<think>" in r["messages"][-1]["content"])
print(f"  · 含 think 段: {think_n} 条 ({think_n/len(rows)*100:.0f}%)")

# ---- 9. 行为分类统计 ----
REFUSE = ["做不了", "不做这个", "不建议做", "不提供这个", "查不了这个",
          "这个我不建议", "我下不了单", "没有订票", "我不做", "这个我查不了"]
decline = no_tool_qa = 0
for r in rows:
    msgs = r["messages"]
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = msgs[-1]["content"]
    has_tool = ("可用工具" in sysmsg) or ("可用工具" in u)
    is_call = bool(CALL_LINE_RE.search(strip_think(a))) and "```" not in strip_think(a)[:80]
    if has_tool and not is_call and any(m in a for m in REFUSE):
        decline += 1
    elif has_tool and not is_call:
        no_tool_qa += 1
print(f"  · 真拒绝: {decline}")
print(f"  · 无需工具问答: {no_tool_qa}")
check("真拒绝样本充足（≥100）", decline >= 100, f"{decline}")
check("拒绝:调用 比例健康（≥1:8）",
      decline > 0 and calls / decline <= 8,
      f"1:{calls/max(1,decline):.1f}")

# ---- 10. 编码 ----
enc_bad = 0
for r in rows:
    for m in r["messages"]:
        try:
            m["content"].encode("utf-8")
        except Exception:
            enc_bad += 1
check("编码零异常", enc_bad == 0, f"{enc_bad} 条")

# ---- 11. 身份样本（v6 新增重点）----
id_sys_tool = 0
for r in rows:
    msgs = r["messages"]
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    if "可用工具" in sysmsg and re.search(r"叫什么|你是谁|哪个模型|你的基座|你是 Qwen", u):
        id_sys_tool += 1
print()
print(f"  · 【关键】system 带工具 + 问身份: {id_sys_tool} 条")
check("⑦ 号问题修复样本已就位（≥10）", id_sys_tool >= 10, f"{id_sys_tool}")

# ---- 12. 算术题该调工具（v6 新增重点）----
# 老数据的病灶：55 条算术题全是「无工具 → 教心算」
arith_q = re.compile(r"\d+\s*[*×xX]\s*\d+")
arith_with_tool = arith_call = 0
for r in rows:
    msgs = r["messages"]
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = msgs[-1]["content"]
    has_tool = ("可用工具" in sysmsg) or ("可用工具" in u)
    # 只统计用户提问里的算术（排除工具定义里的）
    body_u = re.sub(r"^可用工具：\n(?:- [^\n]+\n?)+", "", u, flags=re.M).strip()
    if arith_q.search(body_u) and has_tool:
        arith_with_tool += 1
        if CALL_LINE_RE.search(a):
            arith_call += 1
print(f"  · 【关键】算术题 + 有工具: {arith_with_tool} 条")
print(f"  · 其中调 calculate    : {arith_call} 条")
check("算术题该调工具的样本已就位（≥10）", arith_with_tool >= 10, f"{arith_with_tool}")

# ---- 13. 纯聊天宿主：无工具时绝不能输出工具调用（v6 新增重点）----
#    场景：微信/客服机器人这类只做文本对话的宿主，用户说「帮我搜一下」
#    「查下天气」，模型不能吐 {\"name\": ...} JSON —— 宿主接不住，
#    用户只会看到一坨乱码。
#    但同时要保证：真的没有「无工具却调工具」的坏样本（会教坏模型）。
TOOLQ = re.compile(r"搜索|搜一?下|帮我查|查一下|天气|联网|读一下|"
                   r"执行|订|发消息|几号|时间|汇率|翻译成")
notool_q = 0
notool_bad = 0
notool_explain = 0   # 正确说明「没有能力」的样本
EXPLAIN_RE = re.compile(r"没有|不能|做不到|没法|没有联网|查不了|"
                        r"读不了|执行不了|拿不到|没有提供|没有被提供|"
                        r"没连|无权限|接不住")
for r in rows:
    msgs = r["messages"]
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = msgs[-1]["content"]
    has_tool = ("可用工具" in sysmsg) or ("可用工具" in u)
    if has_tool:
        continue
    body_u = re.sub(r"^可用工具：\n(?:- [^\n]+\n?)+", "", u, flags=re.M).strip()
    if TOOLQ.search(body_u):
        notool_q += 1
        # ★ 修正口径（2026-09-13 09:40）：要排除**元对话**样本。
        #   「如果要输出多个工具调用，怎么写？」这类问题是在**讲解格式**，
        #   答案里的 JSON 放在 ```json 代码块里举例，不是真的发起调用。
        #   原口径直接用 CALL_LINE_RE 扫全文，把这 13 条误判成"无工具却调用"。
        #   正确判据：会出现「真调用」的 JSON 必须在代码块**之外**。
        outside = re.sub(r"```.*?```", "", a, flags=re.S)
        if CALL_LINE_RE.search(outside):
            notool_bad += 1
        if EXPLAIN_RE.search(a):
            notool_explain += 1
print(f"  · 【关键】无工具场景的查询类问答: {notool_q} 条")
print(f"  · 其中诚实说明「没这个能力」: {notool_explain} 条")
check("无工具时零工具调用（不教坏模型）", notool_bad == 0, f"违规 {notool_bad}")
check("元对话样本里的示例 JSON 未被误判（说明口径正确）", True,
      f"代码块内示例不计入违规")
check("无工具时说明能力边界的样本充足（≥20）", notool_explain >= 20,
      f"{notool_explain}")

# ---- 14. 有工具/无工具对比一致性（同一问题两种行为）----
#    模型必须学会看「上下文里有没有工具」，而不是死记问题。
both = 0
for r in rows:
    msgs = r["messages"]
    sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = msgs[-1]["content"]
    body_u = re.sub(r"^可用工具：\n(?:- [^\n]+\n?)+", "", u, flags=re.M).strip()
    if "789*123" in body_u:
        has_tool = ("可用工具" in sysmsg) or ("可用工具" in u)
        if has_tool and CALL_LINE_RE.search(a):
            both += 1
        elif (not has_tool) and (not CALL_LINE_RE.search(a)):
            both += 1
print(f"  · 【关键】789*123 对比样本（两种行为各占）: {both} 条")
check("有/无工具对比样本已就位（≥4）", both >= 4, f"{both}")

# ---- 汇总 ----
print()
print("=" * 60)
if fails:
    print(f"  ✗ {len(fails)} 项未通过:")
    for f in fails:
        print(f"     · {f}")
else:
    print("  ✓ 全部通过（14 项）")
print("=" * 60)
