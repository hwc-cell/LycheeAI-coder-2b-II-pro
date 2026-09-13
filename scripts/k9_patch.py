# ============================================================
# K9. 运算类型精确配比（K8 失败后的重做）
#
#  ★ K8 为什么失败（v6.4 实测）
#    K8 只生成了 104 条，而乘法基线是 244 条 —— 差距根本没拉平。
#    不但没修好，还把调用率从 60%+ 砸到 12%（新增噪音 + 分布偏移）。
#
#  ★ 这次先算数，再生成
#    calculate 样本按运算类型的真实分布（v6.1 数据集实测）：
#        mul        244   (50.5%)   ← 唯一学会调工具的类型
#        add         71   (14.7%)
#        div         53   (11.0%)
#        pow         44   ( 9.1%)
#        float_mul   31   ( 6.4%)
#        sub         31   ( 6.4%)
#        float_add    8   ( 1.7%)
#
#    要拉到「各类型 ~170 条」，需补：
#        add        +99
#        div       +117
#        pow       +126
#        float_mul +139
#        sub       +139
#        float_add +162
#                     合计 +782
#
#    补完后各类型占比 13.4%~19.3%，最大的 mul 也只是均值的 1.4 倍
#    （之前是 8 倍），模型才有可能学到「任何算式都调工具」。
#
#  ★ 与 K8 的另一个区别：system 池要小
#    K8 每条随机配一个 system，导致同一知识点被稀释。
#    这里固定用 3 个最典型的工具型 system（工具定义放 system 放置
#    也是主结构），让信号集中。
# ------------------------------------------------------------
print("K9. 运算类型精确配比（补 782 条，拉平到各类型 ~170）...")

_rng9 = random.Random(20260914)

# 只取「工具定义放 system」的写法，信号集中
K9_SYS = [
    ID_SYS + "\n\n" + tdef(CALC),
    GENERIC_SYS[4] + "\n\n" + tdef(CALC),      # 你是一个可以调用工具的 AI 助手。
    GENERIC_SYS[2] + "\n\n" + tdef(CALC),      # 你是一个智能助手...
]
# 另有「工具定义放 user」的结构，占比小（训练集里约 1/3），
# 这里给 20%，保持双结构混合
K9_SYS_USER = [
    GENERIC_SYS[0],
    GENERIC_SYS[4],
]


def _q_variants(expr_core, cn_ops):
    """给一个算式核心，生成几种中文提问变体"""
    return [f"{v} 等于多少" for v in cn_ops] + [f"帮我算 {v}" for v in cn_ops]


def _gen9(kind):
    """按类型生成 (表达式, 提问)。每个分支保证「算式和问题都不同」。"""
    if kind == "add":
        a, b = _rng9.randint(100, 99999), _rng9.randint(100, 99999)
        e = f"{a}+{b}"
        q = _rng9.choice([f"{a} 加 {b} 等于多少", f"{a}+{b} 是多少",
                          f"帮我算 {a} 加 {b}", f"计算 {a} + {b}",
                          f"{a} 和 {b} 相加是多少"])
    elif kind == "sub":
        a = _rng9.randint(1000, 999999)
        b = _rng9.randint(100, a - 100)
        e = f"{a}-{b}"
        q = _rng9.choice([f"{a} 减 {b} 是多少", f"{a} - {b} 等于几",
                          f"帮我算 {a} 减 {b}", f"{a} 减去 {b} 等于多少"])
    elif kind == "pow":
        style = _rng9.choice(["sq", "cube", "n"])
        if style == "sq":
            a = _rng9.randint(11, 999)
            e = f"{a}**2"
            q = _rng9.choice([f"{a} 的平方是多少", f"{a} 平方等于多少",
                              f"帮我算 {a} 的平方", f"{a}² 是多少"])
        elif style == "cube":
            a = _rng9.randint(2, 99)
            e = f"{a}**3"
            q = _rng9.choice([f"{a} 的立方是多少", f"{a} 的三次方等于多少"])
        else:
            a = _rng9.randint(2, 30)
            b = _rng9.choice([4, 5, 6, 7, 8, 10])
            e = f"{a}**{b}"
            q = _rng9.choice([f"{a} 的 {b} 次方是多少", f"帮我算 {a} 的 {b} 次方",
                              f"{a} 的 {b} 次幂等于几"])
    elif kind == "div":
        b = _rng9.randint(3, 199)
        a = b * _rng9.randint(10, 9999)
        e = f"{a}/{b}"
        q = _rng9.choice([f"{a} 除以 {b} 是多少", f"{a} ÷ {b} 等于多少",
                          f"帮我算 {a} 除以 {b}", f"{a} 除 {b} 等于几"])
    elif kind == "float_mul":
        a = _rng9.randint(1, 9999) / _rng9.choice([10, 100, 1000, 10000])
        b = _rng9.randint(2, 999)
        e = f"{a}*{b}"
        q = _rng9.choice([f"{a} 乘 {b} 等于多少", f"{a} × {b} 是多少",
                          f"帮我算 {a} 乘以 {b}", f"{a}*{b} 的结果"])
    elif kind == "float_add":
        a = _rng9.randint(1, 99999) / 100
        b = _rng9.randint(1, 99999) / 100
        e = f"{a}+{b}"
        q = _rng9.choice([f"{a} 加 {b} 等于多少", f"{a}+{b} 是多少",
                          f"帮我算 {a} 加 {b}"])
    else:
        raise ValueError(kind)
    return e, q


# 精确配额 —— 按上面算出的缺口
K9_QUOTA = [
    ("add", 99),
    ("div", 117),
    ("pow", 126),
    ("float_mul", 139),
    ("sub", 139),
    ("float_add", 162),
]

cnt9 = 0
_seen9 = set()          # 去重，防止 RNG 撞出同一个算式
for kind, n in K9_QUOTA:
    made = 0
    guard = 0
    while made < n and guard < n * 20:
        guard += 1
        e, q = _gen9(kind)
        if e in _seen9:
            continue
        _seen9.add(e)
        # 80% 工具定义放 system，20% 放 user —— 保持双结构混合但以 system 为主
        if _rng9.random() < 0.8:
            sys_text = K9_SYS[_rng9.randrange(len(K9_SYS))]
            add(sys_text, q, call("calculate", {"expression": e}))
        else:
            sys_text = K9_SYS_USER[_rng9.randrange(len(K9_SYS_USER))]
            add(sys_text, tdef(CALC) + "\n\n" + q,
                call("calculate", {"expression": e}))
        made += 1
        cnt9 += 1
    assert made == n, f"{kind} 只生成了 {made}/{n}"

print(f"   K9 生成 {cnt9} 条，去重后唯一算式 {len(_seen9)} 个")

# ★ 附：把「双类型混合算式」也补上（这类最容易心算出错）
#   实测错误：1234+5678=2000、199×257=40043
K9_MIXED = []
for _ in range(60):
    kind = _rng9.choice(["add_mul", "mul_sub", "add_sub"])
    if kind == "add_mul":
        a, b, c = (_rng9.randint(100, 9999) for _ in range(3))
        e, q = f"{a}+{b}*{c}", f"帮我算 {a} + {b} × {c}"
    elif kind == "mul_sub":
        a, b, c = (_rng9.randint(100, 9999) for _ in range(3))
        e, q = f"{a}*{b}-{c}", f"帮我算 {a} × {b} - {c}"
    else:
        a, b, c = (_rng9.randint(1000, 99999) for _ in range(3))
        e, q = f"{a}+{b}-{c}", f"帮我算 {a} 加 {b} 减 {c}"
    K9_MIXED.append((q, e))

for q, e in K9_MIXED:
    sys_text = K9_SYS[_rng9.randrange(len(K9_SYS))]
    add(sys_text, q, call("calculate", {"expression": e}))

print(f"   K9 混合算式追加 {len(K9_MIXED)} 条")
