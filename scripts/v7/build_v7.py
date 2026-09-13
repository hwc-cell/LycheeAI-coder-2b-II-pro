#!/usr/bin/env python3
"""v7 精选器：把 5341 条数据集精选到 ≤1000 条（500 步约束）。

================================================================
  为什么必须精选（而不是全量跑 0.22 epoch）
================================================================

  用户约束：下次训练 ≤500 步。batch=2 → iters = ceil(N/2)
  所以 N ≤ 1000。

  两条路：
    (a) 5341 条 × 250 步 × 2 = 22.1% epoch —— 每条样本只被见到 0.22 次
    (b) 1000 条 × 500 步 × 2 = 100% epoch —— 每条都被完整学到一次

  (a) 的问题不只是"学得少"，而是**行为惯性按样本条数加权**：
      5341 条里 51.4% 是单轮纯文本（F 类），它们把梯度拉向"纯聊天"。
      而 A 类（多轮完整链路，最难学）只占 10.7%，在 22% epoch 下
      平均每条 A 类只贡献 0.22 次更新 —— 学不到。

  (b) 让每一类行为都拿到完整的 1 次更新，且**配额由重要性决定而非原始比例**。

  这正是 v6.4 崩盘的教训：K8 补了 104 条去对冲 244 条乘法基线，
  比例没拉平，结果调用率从 60%+ 砸到 12%。
  **补丁的效力取决于它占新数据集的比例，不是它的绝对条数。**

================================================================
  配额设计（总额 1000）
================================================================
  类别                        原始    目标    理由
  --------------------------- -----   ----   --------------------------
  A  多轮 调用→结果→收尾        569    260    最值钱：宿主真实交互形态
  C1 sys 放工具 + 调            397    110    主结构，保住
  C2 user 放工具 + 调           780    190    占原始 66%，结构不能丢
  C4 有工具但不调               819    150    压住"手痒调工具"的反面样本
  F  单轮纯文本                2746    130    从 51.4% 砍到 13%
  E  多轮纯对话                  30     30    全留（本来就稀缺）
  K9 算术（压缩重配）            —      90    见下
  K10 身份（压缩）               —      40    见下
  --------------------------- -----   ----
                                        1000

================================================================
  K9 的重做：从 782 条压到 90 条，但配比才是关键
================================================================

  v6.1 数据集的 calculate 样本按运算类型分布：
      mul        244  (50.5%)   ← 模型唯一学会调工具的类型
      add         71  (14.7%)
      div         53  (11.0%)
      pow         44  ( 9.1%)
      float_mul   31  ( 6.4%)
      sub         31  ( 6.4%)
      float_add    8  ( 1.7%)

  **乘法占一半**，于是模型学到的是「乘法 → 调工具」这个特例，
  而不是「算术 → 调工具」这个通则。其余类型它没足够样本，
  就退回心算，算错。

  原 K9 方案是"把每个类型补到 ~170"，总计 +782 —— 在 5341 条里
  占 12.7%，有效但超预算。

  在 1000 条预算里，正确做法是**不追求绝对条数，只追求配比均衡**：
  让 7 个类型各占 1/7 ≈ 14%。因为配比决定行为，绝对条数决定强度。
  90 条 / 7 类 ≈ 每类 9 条，加上原有基线，最小类型的占比也能被
  显著抬升。

  注意 K9 只补「稀缺类型」，乘法和大的基线已经有 244+ 条，
  再补只会加剧失衡。所以 K9 的目标配额是：
      float_add / sub / float_mul / pow / div / add  ← 补这 6 个
      mul ← 不补
"""
import collections
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.environ.get("SRC_JSONL",
                     os.path.join(ROOT, "data_coder2b_pro_v6_final", "train.jsonl"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ROOT, "data_coder2b_pro_v7"))
OUT = os.path.join(OUT_DIR, "train.jsonl")

BUDGET = int(os.environ.get("BUDGET", "1000"))
rng = random.Random(20260914)

# ============================================================
#  配额表
# ============================================================
QUOTA = {
    "A_多轮_调用→结果→收尾":  260,
    "C1_工具有_sys工具_调":   110,
    "C2_工具有_user工具_调":  190,
    "C4_工具有_但不调":       150,
    "F_单轮纯文本":           130,
    "E_多轮纯对话":            30,
    # 下面两类由本脚本现场生成，不从原数据集采样
    "K9_算术补强":             90,
    "K10_身份补强":            40,
}

# ============================================================
#  分类器（从 classify.py 复用判定逻辑，但更严格）
# ============================================================
CALL_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:')
TDEF_RE = re.compile(r'可用工具[：:]|Available tools|工具列表[：:]')
CALL_NAME = re.compile(r'^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:')


def has_calls(msgs):
    out = []
    for m in msgs:
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
    has_result = any(
        (m["content"] or "").strip().startswith("{")
        and '"result"' in (m["content"] or "")
        for m in msgs if m["role"] == "user")
    n_plain_user = sum(
        1 for m in msgs if m["role"] == "user"
        and not (m["content"] or "").strip().startswith("{"))

    if calls and has_result:
        return "A_多轮_调用→结果→收尾"
    if calls:
        if sys_has and not usr_has:
            return "C1_工具有_sys工具_调"
        if usr_has and not sys_has:
            return "C2_工具有_user工具_调"
        return "C2_工具有_user工具_调"      # 双放算 user 类
    if sys_has or usr_has:
        return "C4_工具有_但不调"
    if n_plain_user >= 2:
        return "E_多轮纯对话"
    return "F_单轮纯文本"


# ============================================================
#  A 类细分：按工具名分子类，保证「每类工具都见过完整链路」
#
#  为什么：A 类是「调用→结果→收尾」的完整轨迹，是宿主接入后
#  最常走的路径。如果随机采样导致某类工具（比如 read_file）
#  在 A 类里没有代表，模型对该工具的"链路收尾"就学不到。
# ============================================================
def tool_of(item):
    calls = has_calls(item["messages"])
    return calls[0] if calls else "_none"


def stratified_pick(pool, n, rng, keyfn=None):
    """分层采样：按 keyfn 分组，组内轮转取，保证覆盖面。

    比单纯 rng.sample 好在：小类不会因为随机性被整个漏掉。
    """
    if len(pool) <= n:
        return list(pool)
    groups = collections.OrderedDict()
    for it in pool:
        k = keyfn(it) if keyfn else "_"
        groups.setdefault(k, []).append(it)
    for k in groups:
        rng.shuffle(groups[k])
    # 轮转：每轮从每个组取一个，直到取满
    out = []
    keys = list(groups.keys())
    idx = 0
    while len(out) < n and any(groups[k] for k in keys):
        k = keys[idx % len(keys)]
        if groups[k]:
            out.append(groups[k].pop())
        idx += 1
        if idx > n * len(keys) * 2:
            break
    return out


# ============================================================
#  ★ 乘法下采样：算术配比的真正关键
#
#    问题链：
#      v6.1 数据里 calculate 的运算类型，乘法独占 50.5%。
#      模型于是学到「乘法 → 调工具」的特例，而非「算术 → 调工具」的通则。
#      其余类型退回心算，算错（实测 1234+5678=2000、35²=70）。
#
#    第一轮 v7 的教训：
#      只靠 K9 补稀缺类型是不够的 —— 基线里还有大量乘法。
#      实测补完 K9 后配比反而变差（12.4x vs 源数据 8.0x），
#      因为基线采样把乘法一起带进来了。
#
#    正确做法：**两头夹** —— 一头补稀缺（K9），一头削乘法（这里）。
#      对含 calculate 调用的样本，若表达式是乘法且属于"过剩"类型，
#      按概率丢弃，直到乘法占比降到阈值以下。
#
#    阈值取 20%：7 个类型平均 14.3%，留一点余量让乘法保持"常见但不独占"。
#    这比"砍到 14.3%"稳 —— 太整齐反而不像自然分布。
# ============================================================
MUL_EQ_RE = re.compile(r'"expression"\s*:\s*"([^"]*)"')


def calc_kind(item):
    """若该样本含 calculate 调用，返回其运算类型；否则 None"""
    for m in item["messages"]:
        if m["role"] != "assistant":
            continue
        mm = CALL_RE.match(m["content"] or "")
        if not mm or mm.group(1) != "calculate":
            continue
        e = MUL_EQ_RE.search(m["content"] or "")
        if not e:
            return None
        expr = e.group(1).replace(" ", "")
        if "**" in expr:
            return "pow"
        if "." in expr:
            if "*" in expr:
                return "float_mul"
            if "+" in expr:
                return "float_add"
            return "float_other"
        if "*" in expr:
            return "mul"
        if "/" in expr:
            return "div"
        if "+" in expr and "-" in expr:
            return "mixed"
        if "+" in expr:
            return "add"
        if "-" in expr:
            return "sub"
        return "other"
    return None


def trim_mul(items, rng, target_ratio=0.20, protect_ratio=0.5):
    """对乘法样本下采样，把 mul 占比压到 target_ratio 以下。

    protect_ratio：被丢弃的乘法样本里，有多少比例其实是"多轮链路"样本，
    这类要保护（它们承担 A 类的链路教学，不该因为运算是乘法就被删）。

    返回 (新列表, 删掉的条数)
    """
    mul_idx = [i for i, it in enumerate(items) if calc_kind(it) == "mul"]
    n_calc = sum(1 for it in items if calc_kind(it) is not None)
    if not mul_idx or n_calc == 0:
        return items, 0

    # 目标：mul / n_calc ≤ target_ratio
    # 需要保留的乘法数上限 = target * (n_calc - 需要删的mul) / (1-target)
    # 解：keep ≤ target/(1-target) * (n_calc - n_mul + keep)
    #    keep*(1 + t/(1-t)) ≤ t/(1-t) * (n_calc - n_mul) + t/(1-t)*keep ... 化简：
    #    keep / (n_calc - n_mul + keep) ≤ t
    #    keep ≤ t*(n_calc - n_mul) / (1-t)
    t = target_ratio
    n_mul = len(mul_idx)
    n_other_calc = n_calc - n_mul
    keep_max = int(t * n_other_calc / (1 - t))
    if keep_max >= n_mul:
        return items, 0

    # 优先保护"多轮链路"乘法（含 tool result）
    def is_chain(it):
        return any((m["content"] or "").strip().startswith("{")
                   and '"result"' in (m["content"] or "")
                   for m in it["messages"] if m["role"] == "user")

    chained = [i for i in mul_idx if is_chain(items[i])]
    plain = [i for i in mul_idx if not is_chain(items[i])]
    rng.shuffle(plain)

    # 先保链路乘法（上限 keep_max 的 protect_ratio 部分），再补普通乘法
    n_prot = min(len(chained), int(keep_max * protect_ratio))
    keep = set(chained[:n_prot])
    rest = keep_max - n_prot
    if rest > 0:
        keep.update(plain[:rest])

    drop = set(mul_idx) - keep
    return [it for i, it in enumerate(items) if i not in drop], len(drop)


def dedup_key(item):
    ms = item["messages"]
    return tuple((m["role"], (m["content"] or "")[:200]) for m in ms)


# ============================================================
#  ★ 结构转换：user 放工具 → system 放工具
#
#    为什么需要：
#      README 的结论是「工具定义放 system 更稳」（主流 agent 框架的做法）。
#      但源数据里 user 结构占 66%，system 结构只有 34% —— 主结构反而是少数。
#
#      精选时若老实按配额从两个池子各取，结果就是
#      C1(119) : C2(303)，与「推荐 system」的结论背道而驰。
#
#    修法：
#      从 user 结构样本里取一部分，把工具定义段从 user 消息
#      **搬到 system 消息**，内容一字不改。这样同一条轨迹
#      以两种结构各出现一次，既补了 C1 的池子，又天然形成对照样本。
#
#    只搬「有 system 消息」的样本 —— 没有 system 的搬不了（那是结构 B）。
# ============================================================
TDEF_BLOCK = re.compile(r'(可用工具[：:]\n(?:- [^\n]+\n?)+)')


def user_to_sys_struct(item):
    """把工具定义从 user 消息搬到 system 消息。搬不了则返回 None。"""
    ms = item["messages"]
    sys_idx = next((i for i, m in enumerate(ms) if m["role"] == "system"), None)
    if sys_idx is None:
        return None
    # 找第一条含工具定义的 user 消息
    for i, m in enumerate(ms):
        if m["role"] != "user":
            continue
        c = m["content"] or ""
        mm = TDEF_BLOCK.search(c)
        if not mm:
            continue
        block = mm.group(1).rstrip("\n")
        new_user = c.replace(mm.group(1), "").strip()
        if not new_user:
            return None                     # 搬完 user 就空了，不合法
        out = []
        for j, x in enumerate(ms):
            if j == sys_idx:
                out.append({"role": "system",
                            "content": (x["content"] or "").rstrip() + "\n\n" + block})
            elif j == i:
                out.append({"role": "user", "content": new_user})
            else:
                out.append(x)
        return {"messages": out}
    return None


# ============================================================
#  主流程
# ============================================================
def main():
    with open(SRC, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    print("=" * 70)
    print("  v7 精选器")
    print("=" * 70)
    print(f"  源数据 : {SRC}")
    print(f"  总条数 : {len(items)}")
    print(f"  预算   : {BUDGET}")

    # ---- 先去重，再分桶 ----
    #
    #  ★ 顺序很重要（踩过的坑）
    #    第一版是「先按配额采样 → 再去重」，结果 870 条被去重成 671 条，
    #    A 类从 260 掉到 128 —— 配额完全失效。
    #    原因：源数据集里 43.7% 的行是**加权重复**（同一条样本被写 N 遍，
    #    gen_ood_v6.py 的 weight 机制），采样时把多份副本都选进来了，
    #    去重后数量自然腰斩。
    #
    #    正确顺序：源数据先按内容去重 → 每类得到「真实唯一条数」→
    #    再按配额采样。这样配额才是真的。
    seen0 = set()
    uniq_src = []
    for it in items:
        k = dedup_key(it)
        if k in seen0:
            continue
        seen0.add(k)
        uniq_src.append(it)
    print()
    print(f"  源数据去重: {len(items)} → {len(uniq_src)} 条唯一 "
          f"(加权膨胀 {len(items)/len(uniq_src):.2f}x)")
    items = uniq_src

    # 按类别分桶
    buckets = collections.defaultdict(list)
    for it in items:
        buckets[classify(it)].append(it)

    print()
    print("  源数据分布（唯一）:")
    for c in sorted(buckets, key=lambda x: -len(buckets[x])):
        print(f"    {c:<32} {len(buckets[c]):>5}")

    # 按配额采样
    picked = []
    print()
    print("  采样结果:")
    for cat, n in QUOTA.items():
        if cat.startswith("K"):
            continue                      # K9/K10 现场生成
        pool = buckets.get(cat, [])
        keyfn = tool_of if cat.startswith("A") else None
        got = stratified_pick(pool, n, rng, keyfn)
        picked.extend(got)
        flag = "✓" if len(got) >= n else "!"
        short = "" if len(got) >= n else f"  (源池只有 {len(pool)})"
        print(f"    {flag} {cat:<32} {len(got):>4} / {n}{short}")

    n_base = len(picked)
    print()
    print(f"  基线精选 : {n_base} 条")

    # ---- 结构转换：把 user 放工具的样本改写成 system 放工具 ----
    #
    #  C1 池子只有 49 条（源数据里 system 结构本来就少），
    #  达不到 110 的配额。从 C2 池里取样本做结构转换来补。
    c1_got = sum(1 for it in picked if classify(it) == "C1_工具有_sys工具_调")
    c1_need = QUOTA["C1_工具有_sys工具_调"] - c1_got
    if c1_need > 0:
        pool_c2 = buckets.get("C2_工具有_user工具_调", [])
        used = set(dedup_key(it) for it in picked)
        avail = [it for it in pool_c2 if dedup_key(it) not in used]
        rng.shuffle(avail)
        converted = []
        for it in avail:
            if len(converted) >= c1_need:
                break
            conv = user_to_sys_struct(it)
            if conv is None:
                continue
            # 转换后必须真的被判成 C1（自检，防止搬错）
            if classify(conv) != "C1_工具有_sys工具_调":
                continue
            converted.append(conv)
        picked.extend(converted)
        print(f"  结构转换 : {len(converted)} 条 user→system "
              f"(补 C1 池子不足 {c1_need} 条)")
        n_base = len(picked)

    # ---- K9 / K10 生成 ----
    sys.path.insert(0, HERE)
    from k9_k10_v7 import gen_k9, gen_k10
    k9 = gen_k9(rng, QUOTA["K9_算术补强"])
    k10 = gen_k10(rng, QUOTA["K10_身份补强"])
    print(f"  K9 算术  : {len(k9)} 条")
    print(f"  K10 身份 : {len(k10)} 条")

    # ---- 合并 ----
    all_items = picked + k9 + k10

    # ---- 乘法下采样（与 K9 的补稀缺形成「两头夹」）----
    all_items, n_dropped = trim_mul(all_items, rng, target_ratio=0.20)
    if n_dropped:
        print()
        print(f"  乘法下采样: 删除 {n_dropped} 条过剩的纯乘法样本")

    # ---- 补足到预算 ----
    #
    #  ★ 为什么要用满预算
    #    步数 = ceil(N/2)，N 越接近 1000 步数越接近 500。
    #    预算没用满 = 白送算力。缺口优先从高价值类补：
    #    A（多轮链路）> C4（不调工具的反面）> C1 > C2 > F
    #
    #  ★ 注意：补充时也要过乘法下采样，否则补进来的又会让配比失衡。
    #    所以这里是个循环：补 → 削 → 再补，直到收敛。
    for _round in range(8):
        if len(all_items) >= BUDGET:
            break
        gap = BUDGET - len(all_items)
        used = set(dedup_key(it) for it in all_items)
        added = 0
        for cat in ["A_多轮_调用→结果→收尾", "C4_工具有_但不调",
                    "C1_工具有_sys工具_调", "C2_工具有_user工具_调",
                    "F_单轮纯文本", "E_多轮纯对话"]:
            if gap <= 0:
                break
            pool = [it for it in buckets.get(cat, [])
                    if dedup_key(it) not in used]
            rng.shuffle(pool)
            # 补充时同样避开乘法（除非池子里只剩乘法）
            non_mul = [it for it in pool if calc_kind(it) != "mul"]
            take = (non_mul or pool)[:gap]
            for it in take:
                used.add(dedup_key(it))
            all_items.extend(take)
            added += len(take)
            gap = BUDGET - len(all_items)
        # 削一次，看是否又超
        all_items, _d = trim_mul(all_items, rng, target_ratio=0.20)
        if added == 0:
            break
    if len(all_items) < BUDGET:
        print(f"  ⓘ 补到 {len(all_items)} 条（源池已取尽，"
              f"步数 = {(len(all_items)+1)//2}）")

    # ---- 若仍超预算，从 F 类削（F 类最不值钱）----
    if len(all_items) > BUDGET:
        over = len(all_items) - BUDGET
        print()
        print(f"  ⚠ 超预算 {over} 条，从 F 类削减")
        f_items = [it for it in all_items if classify(it) == "F_单轮纯文本"]
        cut = set(id(x) for x in rng.sample(f_items, min(over, len(f_items))))
        all_items = [it for it in all_items if id(it) not in cut]
        over = len(all_items) - BUDGET
        if over > 0:
            print(f"  ⚠ F 类削完仍超 {over} 条，从 C4 削减")
            c4 = [it for it in all_items if classify(it) == "C4_工具有_但不调"]
            cut = set(id(x) for x in rng.sample(c4, min(over, len(c4))))
            all_items = [it for it in all_items if id(it) not in cut]

    rng.shuffle(all_items)

    # ---- 输出 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    final = collections.Counter(classify(it) for it in all_items)
    N = len(all_items)
    iters = (N + 1) // 2

    print()
    print("=" * 70)
    print("  最终数据集")
    print("=" * 70)
    for c in sorted(final, key=lambda x: -final[x]):
        print(f"    {c:<32} {final[c]:>5}  ({final[c]/N*100:>5.1f}%)")
    print(f"    {'合计':<32} {N:>5}")
    print()
    print(f"  步数 = ceil({N}/2) = {iters} 步  "
          f"{'✓ ≤500' if iters <= 500 else '✗ 超 500'}")
    print(f"  epoch = {iters*2/N*100:.0f}%")
    print()

    # 连号检查
    maxrun, run, prev = 1, 1, None
    for it in all_items:
        k = json.dumps(it, ensure_ascii=False)
        if k == prev:
            run += 1
            maxrun = max(maxrun, run)
        else:
            run = 1
        prev = k
    print(f"  最大连号重复: {maxrun} 行")

    # 运算类型配比复核
    kinds = collections.Counter()
    for it in all_items:
        for m in it["messages"]:
            if m["role"] == "assistant":
                mm = CALL_NAME.match(m["content"] or "")
                if mm and mm.group(1) == "calculate":
                    e = re.search(r'"expression"\s*:\s*"([^"]+)"',
                                  m["content"] or "")
                    if e:
                        kinds[kind_of(e.group(1))] += 1
    if kinds:
        print()
        print("  calculate 运算类型配比:")
        tk = sum(kinds.values())
        for k, v in kinds.most_common():
            print(f"    {k:<12} {v:>4}  ({v/tk*100:>5.1f}%)")
        mx = max(kinds.values())
        mn_ = min(kinds.values())
        print(f"    → 最热/最冷 = {mx/max(mn_,1):.1f}x  "
              f"（源数据是 8.0x，越接近 1 越好）")

    print()
    print(f"  输出: {OUT}")
    print(f"  大小: {os.path.getsize(OUT)/1024:.1f} KB")
    return 0


def kind_of(expr):
    """判定表达式属于哪个运算类型（与 K9 的配额口径一致）"""
    e = expr.replace(" ", "")
    if "**" in e:
        return "pow"
    if "(" in e or "+" in e and "-" in e and "*" in e:
        return "mixed"
    has_dot = "." in e
    if has_dot:
        if "*" in e:
            return "float_mul"
        if "+" in e:
            return "float_add"
        if "-" in e:
            return "float_sub"
        if "/" in e:
            return "float_div"
    if "*" in e:
        return "mul"
    if "/" in e:
        return "div"
    if "+" in e:
        return "add"
    if "-" in e:
        return "sub"
    return "other"


if __name__ == "__main__":
    sys.exit(main())
