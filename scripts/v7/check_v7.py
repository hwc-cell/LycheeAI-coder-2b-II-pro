#!/usr/bin/env python3
"""v7 数据质检：查「工具名幻觉」—— 调用了 system/user 里没定义的工具。

================================================================
  这个检查为什么重要
================================================================

  v6 系列有一个验收项 K2「工具名幻觉」：不许编不存在的工具。

  数据生成时会造很多虚构工具名（execute_sql / send_slack / ...），
  这些**本身不是问题** —— 它们都在对应的 system 里定义过，
  属于"适配不同宿主"的正常样本。

  真正的问题样本是：assistant 调了 `calculate`，
  但 system 和 user 里都**没有**定义 `calculate`。
  这才是幻觉，会教模型"凭空编工具"。

  之前踩过的分类误报（2026-09-13）：
    只看 system 找工具定义，得出"478 条无工具却调工具"的错误结论。
    实际上 66% 的样本把工具定义放在 user 消息里。
    所以这里必须 **system + user 一起找**。
================================================================
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get(
    "DATA_JSONL",
    os.path.join(os.path.dirname(HERE), "data_coder2b_pro_v7", "train.jsonl"))

# assistant 里的工具调用
CALL_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:')
# 工具定义行：- name(params): desc
DEF_LINE = re.compile(r'^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(', re.M)


def main():
    items = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    print("=" * 70)
    print("  v7 数据质检")
    print("=" * 70)
    print(f"  文件: {DATA}")
    print(f"  条数: {len(items)}")
    print()

    halluc = []
    no_def_at_all = []
    n_with_call = 0
    defs_per_item = []

    for idx, it in enumerate(items):
        ms = it["messages"]
        # 收集上下文里定义过的所有工具名（system + user 都要看）
        ctx = " ".join(m["content"] or "" for m in ms
                       if m["role"] in ("system", "user"))
        defined = set(DEF_LINE.findall(ctx))
        # 本样本的调用
        calls = [CALL_RE.match(m["content"] or "").group(1)
                 for m in ms if m["role"] == "assistant"
                 and CALL_RE.match(m["content"] or "")]
        if calls:
            n_with_call += 1
            defs_per_item.append(len(defined))
            if not defined:
                no_def_at_all.append(idx)
            for c in calls:
                if c not in defined:
                    halluc.append((idx, c, sorted(defined)[:5]))

    print("=" * 70)
    print("  ① 工具名幻觉检查（调了没定义的工具）")
    print("=" * 70)
    if not halluc:
        print("  ✅ 零幻觉。所有调用都能在上下文里找到对应定义。")
    else:
        print(f"  ✗ 发现 {len(halluc)} 处幻觉：")
        by_tool = collections.Counter(c for _, c, _ in halluc)
        for t, n in by_tool.most_common(20):
            print(f"      {t:<26} {n} 次")
        print()
        for idx, c, d in halluc[:5]:
            print(f"  --- 样例 (行 {idx+1}) 调了 {c}，"
                  f"上下文定义的工具有 {d} ---")
            for m in items[idx]["messages"]:
                print(f"    [{m['role']}] {(m['content'] or '')[:140]}")
            print()

    print()
    print("=" * 70)
    print("  ② 有调用但完全没定义工具（严重）")
    print("=" * 70)
    if not no_def_at_all:
        print("  ✅ 零条。每个有调用的样本都在上下文里给出了工具定义。")
    else:
        print(f"  ✗ {len(no_def_at_all)} 条样本调了工具但上下文无任何定义：")
        for idx in no_def_at_all[:3]:
            for m in items[idx]["messages"]:
                print(f"    [{m['role']}] {(m['content'] or '')[:140]}")
            print()

    print()
    print("=" * 70)
    print("  ③ 上下文工具定义数量分布（关系到「泛化到任意宿主」）")
    print("=" * 70)
    if defs_per_item:
        c = collections.Counter(defs_per_item)
        for k in sorted(c):
            print(f"    {k:>2} 个工具: {c[k]:>4} 条带调用的样本")
        print(f"    → 平均 {sum(defs_per_item)/len(defs_per_item):.1f} 个工具/样本")
        print(f"    → 见过 ≥3 种工具数："
              f"{sum(v for k,v in c.items() if k>=3)} 条")

    print()
    print("=" * 70)
    print("  ④ 结构完整性")
    print("=" * 70)
    bad_struct = 0
    bad_tail = 0
    for it in items:
        ms = it["messages"]
        if not ms or ms[0]["role"] not in ("system", "user"):
            bad_struct += 1
        if ms[-1]["role"] != "assistant":
            bad_tail += 1
    print(f"    首条非 system/user : {bad_struct}")
    print(f"    末条非 assistant   : {bad_tail}  "
          f"（MLX 只对 assistant 计算 loss，末条必须是它）")

    # 统计 tool result 结构
    n_res_with_name = 0
    n_res_without = 0
    for it in items:
        for m in it["messages"]:
            if m["role"] == "user" and (m["content"] or "").strip().startswith("{"):
                if '"result"' in (m["content"] or ""):
                    if '"name"' in (m["content"] or ""):
                        n_res_with_name += 1
                    else:
                        n_res_without += 1
    print()
    print("=" * 70)
    print("  ⑤ 工具结果结构（两种都认，见 README 已知问题 4）")
    print("=" * 70)
    print(f"    结构 A（带 name，有调用轮）: {n_res_with_name}")
    print(f"    结构 B（不带 name）        : {n_res_without}")

    ok = (not halluc) and (not no_def_at_all) and not bad_struct and not bad_tail
    print()
    print("=" * 70)
    print(f"  结论: {'✅ 全部通过，可以训练' if ok else '✗ 有问题，见上'}")
    print("=" * 70)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
