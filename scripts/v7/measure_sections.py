#!/usr/bin/env python3
"""按段落执行 gen_common.py，测量每段产出的样本条数。

做法：把 gen_common.py 按 print("X. ...") 切成段，逐段 exec，
每次 exec 前记录 len(samples)，exec 后差值即该段产出。

为什么要这个（不是好奇心）：
    500 步约束 = 最多 1000 条样本（batch=2）。
    v6.1 数据集 4526 条，必须砍到 ≤1000，所以要知道
    **每一段各占多少条**，才能做减法而不是瞎砍。
"""
import io
import os
import re
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "gen_common.py")

src = open(COMMON, encoding="utf-8").read()
lines = src.split("\n")

# 找所有段落起始行（print("X. ...") 形式的顶层语句）
sec_starts = []
for i, l in enumerate(lines):
    if re.match(r'^print\("[A-K]', l):
        sec_starts.append(i)

print(f"发现 {len(sec_starts)} 个段落标记")

# 构建全局命名空间并逐段执行
g = {"__name__": "__main__", "__file__": COMMON}

# 先执行 prelude（第一段之前的所有代码 + def 定义）
prelude_end = sec_starts[0]
prelude = "\n".join(lines[:prelude_end])

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(prelude, COMMON, "exec"), g)

print(f"prelude 执行完，samples = {len(g['samples'])}")
print()
print("=" * 70)
print(f"{'段落':<46} {'条数':>6} {'占比':>8}")
print("=" * 70)

total_before = len(g["samples"])
rows = []

for idx, start in enumerate(sec_starts):
    end = sec_starts[idx + 1] if idx + 1 < len(sec_starts) else len(lines)
    chunk = "\n".join(lines[start:end])

    # 段落名
    m = re.match(r'^print\("([^"]+)"', lines[start])
    name = m.group(1) if m else f"line{start}"

    before = len(g["samples"])
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile(chunk, COMMON, "exec"), g)
        except Exception as e:
            print(f"  ✗ {name} 执行失败: {type(e).__name__}: {e}")
            raise
    after = len(g["samples"])
    rows.append((name, after - before))

total = len(g["samples"])
for name, n in rows:
    print(f"{name[:46]:<46} {n:>6} {n/total*100:>7.1f}%")

print("=" * 70)
print(f"{'合计':<46} {total:>6} {'100.0%':>8}")
print()

# 同时输出每段的唯一性（重复率）
uniq = set()
for item in g["samples"]:
    key = tuple((x["role"], x["content"]) for x in item["messages"])
    uniq.add(key)
print(f"唯一（去重后）= {len(uniq)}，重复率 = {(1-len(uniq)/total)*100:.1f}%")
print(f"加权膨胀系数 = {total/max(len(uniq),1):.2f}x")
