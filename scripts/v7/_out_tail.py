# ============================================================
# 输出
#
# ★ 关键：必须打散，不能按原顺序写。
#
#   问题（2026-09-13 发现）：weight 加权会产生大量**连号重复** ——
#   同一条样本连续写 N 遍。实测最大连续 6 行完全相同，
#   全文件有 516 组连号（2连 290 组、3连 169 组、4连 27 组、6连 30 组）。
#
#   危害：MLX 的 DataLoader 是按顺序取 batch 的，同一条连续出现
#   3~6 次会被**同一个 batch 反复看到**，等效于局部放大学习率，
#   容易把这少数几条过拟合，而其他样本被稀释。
#
#   修法：全局 shuffle 后再写。weight 的"多出现几次"效果保留了
#   （该样本在文件里仍占 N 行），但分散到不同位置、不同 batch 里。
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)
random.shuffle(samples)          # ← 打散，消除连号
with open(OUT, "w", encoding="utf-8") as f:
    for item in samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 校验打散效果
maxrun = 0
run = 1
prev = None
for item in samples:
    key = json.dumps(item, ensure_ascii=False)
    if key == prev:
        run += 1
        maxrun = max(maxrun, run)
    else:
        run = 1
    prev = key

unique = set()
for item in samples:
    m = item["messages"]
    key = tuple((x["role"], x["content"]) for x in m)
    unique.add(key)

# 多轮统计（之前只统计三段式，漏掉了多轮样本）
n_multi = sum(1 for i in samples
              if sum(1 for x in i["messages"] if x["role"] == "user") >= 2)
n_tool_multi = sum(1 for i in samples
                   if sum(1 for x in i["messages"] if x["role"] == "user") >= 2
                   and any('"name"' in x["content"]
                           for x in i["messages"] if x["role"] == "assistant"))

print()
print("=" * 60)
print("  生成完成")
print("=" * 60)
print(f"总条数（含加权）: {len(samples)}")
print(f"唯一条数:         {len(unique)}")
print(f"其中多轮 (>=2轮): {n_multi}")
print(f"其中多轮含工具调用: {n_tool_multi}")
print(f"最大连号重复:     {maxrun} 行  (打散前是 6)")
print(f"输出: {OUT}")
print(f"大小: {os.path.getsize(OUT)/1024:.1f} KB")
