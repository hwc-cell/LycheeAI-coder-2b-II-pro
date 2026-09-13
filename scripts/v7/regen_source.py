#!/usr/bin/env python3
"""从修补后的 gen_common.py 重新生成源数据集（修正版）。

与 gen_ood_v6.py 的区别：
    只修了一处 —— K4 多轮样本的 system 工具定义不再写死 tdef(CALC)，
    而是按轨迹实际调用的工具动态生成。

    原版有 9 处自相矛盾的样本（system 只定义 calculate，
    却调用 read_file / get_weather）——这正是"工具名幻觉"的反面教材，
    必须清掉。

输出: data_coder2b_pro_v6_fixed/train.jsonl
"""
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "gen_common.py")
TAIL = os.path.join(HERE, "_out_tail.py")

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(HERE),
                                                "data_coder2b_pro_v6_fixed"))
OUT = os.path.join(OUT_DIR, "train.jsonl")

src = open(COMMON, encoding="utf-8").read()
tail = open(TAIL, encoding="utf-8").read()

g = {"__name__": "__main__", "__file__": COMMON}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, COMMON, "exec"), g)

samples = g["samples"]
print(f"生成条数: {len(samples)}")

os.makedirs(OUT_DIR, exist_ok=True)
import random
random.shuffle(samples)
with open(OUT, "w", encoding="utf-8") as f:
    for it in samples:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

print(f"输出: {OUT}")
print(f"大小: {os.path.getsize(OUT)/1024:.1f} KB")

# 立即质检
sys.path.insert(0, HERE)
import subprocess
r = subprocess.run([sys.executable, os.path.join(HERE, "check_v7.py")],
                   env={**os.environ, "DATA_JSONL": OUT},
                   capture_output=True, text=True)
print(r.stdout[-2500:])
sys.exit(r.returncode)
