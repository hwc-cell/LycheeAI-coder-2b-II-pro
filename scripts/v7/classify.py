#!/usr/bin/env python3
"""把 v6.1 数据集按「行为类别」分类，并回答关键疑点。

★ 为什么会问「D 类 478 条无工具却调工具是不是污染」

  初次粗分类时，D 类的判定条件是「system 里没有工具定义，
  但 assistant 却输出了工具调用 JSON」。听起来像 bug。

  但复看数据生成脚本 K3/K5 后明白了：
  训练集里有**三分之一**的样本是「工具定义放 user 消息里」的结构
  （为了双结构均衡）。粗分类只看 system，于是把这一大类
  全误判成了「无工具却调工具」。

  所以 D 类的真实构成需要拆开看。这个脚本就是干这个的：
      D1 = system 无工具 + user 无工具 + 却调了工具   ← 真污染
      D2 = system 无工具 + user 有工具 + 调了工具     ← 合法（结构 B）
      D3 = system 无工具 + 无工具 + 没调工具          ← 纯聊天，正常

  输出每类的条数和样例，据此决定取舍。
"""
import json
import os
import re
import collections

DATA = os.environ.get(
    "DATA_JSONL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "data_coder2b_pro_v6_final", "train.jsonl"))

# 工具调用的判定：assistant 的消息体是 {"name":..., "arguments":...}
CALL_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:')
# 工具定义段的判定：出现「可用工具：」或 Available tools 之类
TDEF_RE = re.compile(r'可用工具[：:]|Available tools|工具列表[：:]')


def has_calls(messages):
    """返回 assistant 消息里的工具调用列表"""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            mm = CALL_RE.match(m["content"] or "")
            if mm:
                out.append(mm.group(1))
    return out


def classify(item):
    msgs = item["messages"]
    sys_c = " ".join(m["content"] or "" for m in msgs if m["role"] == "system")
    usr_c = " ".join(m["content"] or "" for m in msgs if m["role"] == "user")
    sys_has = bool(TDEF_RE.search(sys_c))
    usr_has = bool(TDEF_RE.search(usr_c))
    calls = has_calls(msgs)
    n_user_turns = sum(1 for m in msgs if m["role"] == "user")
    # 去掉工具结果型的 user 消息（那些是 JSON 对象）
    n_plain_user = sum(1 for m in msgs
                       if m["role"] == "user"
                       and not (m["content"] or "").strip().startswith("{"))
    has_tool_result = any(
        (m["content"] or "").strip().startswith("{") and '"result"' in (m["content"] or "")
        for m in msgs if m["role"] == "user")

    if calls and has_tool_result:
        return "A_多轮_调用→结果→收尾"
    if has_tool_result and not calls:
        return "B_结果解读_无调用轮"
    if calls:
        if sys_has and not usr_has:
            return "C1_工具有_sys工具_调"
        if usr_has and not sys_has:
            return "C2_工具有_user工具_调"
        if sys_has and usr_has:
            return "C3_工具有_双放_调"
        # 都没有工具定义，却调了
        return "D1_真污染_无工具却调"
    # 没调用
    if sys_has or usr_has:
        return "C4_工具有_但不调"
    if n_plain_user >= 2:
        return "E_多轮纯对话"
    return "F_单轮纯文本"


def main():
    items = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print("=" * 78)
    print(f"  数据集: {DATA}")
    print(f"  总条数: {len(items)}")
    print("=" * 78)
    print()

    cats = collections.Counter()
    samples = collections.defaultdict(list)
    for it in items:
        c = classify(it)
        cats[c] += 1
        if len(samples[c]) < 3:
            samples[c].append(it)

    total = len(items)
    for c, n in cats.most_common():
        print(f"  {c:<32} {n:>6}  ({n/total*100:>5.1f}%)")

    print()
    print("=" * 78)
    print("  D1「真污染」详情（system 无工具 + user 无工具 + 却调了工具）")
    print("=" * 78)
    d1 = samples.get("D1_真污染_无工具却调", [])
    if not d1:
        print("  ✓ 没有！说明 D 类全部是「工具定义放 user」的合法结构，")
        print("    不存在污染。之前的担忧是分类器只看 system 造成的误报。")
    else:
        for i, it in enumerate(d1[:3]):
            print(f"\n  --- 样例 {i+1} ---")
            for m in it["messages"]:
                print(f"  [{m['role']}] {(m['content'] or '')[:150]}")

    # 额外：把「工具定义放 user」的总量也统计出来，确认双结构比例
    n_user_struct = sum(1 for it in items if classify(it).startswith("C2"))
    n_sys_struct = sum(1 for it in items if classify(it).startswith("C1"))
    print()
    print("=" * 78)
    print("  双结构比例（这个数决定 1000 条子集里两种结构各占多少）")
    print("=" * 78)
    print(f"  工具定义放 system : {n_sys_struct:>5}")
    print(f"  工具定义放 user   : {n_user_struct:>5}")
    s = n_sys_struct + n_user_struct
    if s:
        print(f"  → sys:user = {n_sys_struct/s*100:.0f} : {n_user_struct/s*100:.0f}")


if __name__ == "__main__":
    main()
