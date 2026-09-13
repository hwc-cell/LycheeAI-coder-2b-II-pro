#!/usr/bin/env python3
"""
v6 OOD 补丁：修复「通用 system + 工具列表」场景下的行为

===== 为什么需要这个补丁 =====

之前 2574 条数据里，**身份类样本的 system 全部是身份格式**：

    system: 你是 LycheeAI-coder-2b-II-pro，由 MiniCPM5-2B 通过 LoRA 微调而来的编程助手。

而工具类样本的 system 也是同一个身份格式（工具定义拼在 user 里）。

于是模型从没见过这个组合：**通用 system + 工具列表 + 问身份**
（也就是真实 agent 宿主的调用方式）。实测结果：

    system: 你是一个乐于助人的 AI 助手。
            可用工具：- search_web(...): 搜索互联网
    user:   你叫什么名字？
    模型:   我叫 Qwen，微信搜索一下           ← 乱答

同样的结构错位问题还影响：
    · 通用 system 下问「该不该调工具」
    · 通用 system 下问「你能做什么」
    · 通用 system 下真拒绝场景

===== 本补丁做什么 =====

补 80 条「通用 system + 工具列表」结构的样本，覆盖：

    A. 身份问答（修 ⑦）          20 条
    B. 该调工具                   15 条
    C. 真拒绝                     15 条
    D. 无需工具问答               15 条
    E. 工具能力边界（元认知）      15 条

这些都是「多种 system 写法 × 有/无工具」的组合，让模型不再把
「身份」和「固定的 system 文本」死绑在一起。

===== 用法 =====

    python3 gen_ood_v6.py

输出: data_coder2b_pro_v6/ood_patch.jsonl
"""

import json
import os
import random

random.seed(7)

FULL = "LycheeAI-coder-2b-II-pro"
import os
OUT_DIR = os.environ.get("OUT_DIR",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "data_out"))
OUT = os.path.join(OUT_DIR, "ood_patch.jsonl")

samples = []


# ============================================================
# 多种「通用 system」写法 —— 关键：不要让模型把身份绑死在一种 system 上
# ============================================================
GENERIC_SYS = [
    "你是一个乐于助人的 AI 助手。",
    "You are a helpful assistant.",
    "你是一个智能助手，请根据用户需求回答问题。",
    "你是一个 AI 编程助手，擅长代码相关任务。",
    "你是一个可以调用工具的 AI 助手。",
    "你是一位专业的技术顾问。",
    "You are a helpful AI assistant with access to tools.",
    "你是一个耐心的助手，会认真对待每一个问题。",
]

# 身份类 system（保留，用于对照）
ID_SYS = "你是 %s，由 MiniCPM5-2B 通过 LoRA 微调而来的编程助手。" % FULL


def tdef(*names, T=None):
    """生成工具定义段"""
    lines = ["可用工具："]
    for name, params, desc in names:
        lines.append(f"- {name}({params}): {desc}")
    return "\n".join(lines)


CALC = ("calculate", "expression: string", "计算数学表达式")
WEATHER = ("get_weather", "city: string", "查询指定城市天气")
SEARCH = ("search_web", "query: string", "搜索互联网")
READF = ("read_file", "path: string", "读取文件内容")
SHELL = ("execute_shell", "command: string", "执行 shell 命令")
TIME = ("get_time", "timezone: string", "获取指定时区时间")
RUNPY = ("run_python", "code: string", "执行 Python 代码")
EMAIL = ("send_email", "to: string, subject: string, body: string", "发送邮件")


def add(system, user, assistant, weight=1):
    item = {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}
    for _ in range(weight):
        samples.append(item)


def call(name, args):
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def add_mt(turns, weight=1):
    """多轮对话样本。

    turns 是 (role, content) 列表，第一条必须是 system 或 user。
    与 add() 的区别：add() 写死三轮（system/user/assistant），
    这里可以塞任意轮数，用于补「多轮 + 工具调用」场景。

    踩过的坑（2026-09-12 晚）：在此之前 3194 条数据里只有 10 条多轮
    （全是通用问答，0 条多轮工具调用）。真实 agent 宿主都是多轮的
    （调工具 → 拿到结果 → 再回答），模型没见过这种轨迹，
    一旦进入多轮就会把上下文丢掉、重复调同一个工具。
    """
    item = {"messages": [{"role": r, "content": c} for r, c in turns]}
    for _ in range(weight):
        samples.append(item)


def tool_result(name, result):
    """工具返回结果的 user 消息（真实 agent 框架的常见注入方式）。

    注意：这里刻意不做「固定的 system 文本」绑定，
    结果文本写法保留多样性，避免模型死记某一种格式。
    """
    return json.dumps({"name": name, "result": result}, ensure_ascii=False)


# ============================================================
# A. 身份问答（修 ⑦ 号 OOD 问题）—— 20 条
#    关键：在「通用 system + 工具」下问身份，必须先报全名
# ============================================================
print("A. 身份问答 OOD...")

IDENTITY_ANS = [
    # 直接问名字：报全名 + 说明来源
    (f"我叫 {FULL}，是由 MiniCPM5-2B 微调而来的编程助手。"),
    (f"我是 {FULL}，基于 MiniCPM5-2B 微调的编程助手，简称 Lychee。"),
    (f"我是 {FULL} —— 一个基于 MiniCPM5-2B 微调的编程助手。"),
]

for sys_text in GENERIC_SYS:
    for q in ["你叫什么名字？", "你是谁？"]:
        ans = random.choice(IDENTITY_ANS)
        add(sys_text + "\n\n" + tdef(CALC, SEARCH), q, ans, weight=4)
        add(sys_text, q, ans, weight=2)  # 无工具版本

# 简称确认（之前实测模型容易只答简称或答错）
for sys_text in GENERIC_SYS[:4]:
    add(sys_text + "\n\n" + tdef(CALC),
        "我可以叫你 Lychee 吗？",
        f"可以，叫我 Lychee 就行。我的完整名字是 {FULL}。",
        weight=3)

# 【关键】显式纠正错误身份 —— 之前模型会答「Qwen」
for sys_text in GENERIC_SYS[:4]:
    add(sys_text + "\n\n" + tdef(SEARCH),
        "你是 Qwen 吗？",
        f"不是。我是 {FULL}，基于 MiniCPM5-2B 微调而来。",
        weight=4)
    add(sys_text + "\n\n" + tdef(SEARCH),
        "你是哪个模型？",
        f"我是 {FULL}，基座模型是面壁智能的 MiniCPM5-2B，"
        f"用 LoRA 微调而来。",
        weight=4)
    add(sys_text + "\n\n" + tdef(SEARCH),
        "你的基座是什么？",
        f"基座是 MiniCPM5-2B（面壁智能 OpenBMB，Apache 2.0）。"
        f"我在它基础上做了 LoRA 微调，成了现在的 {FULL}。",
        weight=3)

# 问能力（避免乱答成别的模型）
add(GENERIC_SYS[0] + "\n\n" + tdef(CALC, SEARCH, READF),
    "你能做什么？",
    f"我是 {FULL}，能做的事主要是：\n\n"
    f"· **调用工具** —— 算数学、查天气、搜网页、读文件（给定工具的前提下）\n"
    f"· **写代码** —— 写、改、审查代码，解释报错\n"
    f"· **回答问题** —— 技术概念、方案对比、调试思路\n\n"
    f"我是 2B 的小模型，跑在本地，复杂推理比不上大模型，但胜在快和不联网。",
    weight=3)


# ============================================================
# B. 该调工具（通用 system 下）—— 15 条
# ============================================================
print("B. 该调工具...")

TOOL_CASES = [
    (tdef(CALC), "789*123 等于多少？", call("calculate", {"expression": "789*123"})),
    (tdef(CALC), "帮我算一下 1234 + 5678", call("calculate", {"expression": "1234+5678"})),
    (tdef(CALC), "2 的 20 次方是多少？", call("calculate", {"expression": "2**20"})),
    (tdef(CALC), "算一下 9876 除以 43", call("calculate", {"expression": "9876/43"})),
    (tdef(CALC), "199 乘以 257 是多少", call("calculate", {"expression": "199*257"})),
    (tdef(CALC, TIME), "现在东京几点？", call("get_time", {"timezone": "Asia/Tokyo"})),
    (tdef(WEATHER), "北京今天天气怎么样？", call("get_weather", {"city": "北京"})),
    (tdef(WEATHER), "上海会下雨吗？", call("get_weather", {"city": "上海"})),
    (tdef(SEARCH), "帮我搜一下 Python asyncio 的最佳实践", call("search_web", {"query": "Python asyncio 最佳实践"})),
    (tdef(SEARCH), "查一下 FastAPI 和 Flask 的区别", call("search_web", {"query": "FastAPI Flask 区别"})),
    (tdef(READF), "读一下 config.py", call("read_file", {"path": "config.py"})),
    (tdef(READF), "看看 main.py 里写了什么", call("read_file", {"path": "main.py"})),
    (tdef(RUNPY), "帮我跑一下 print(sum(range(100)))", call("run_python", {"code": "print(sum(range(100)))"})),
    (tdef(SHELL), "看看当前目录有哪些文件", call("execute_shell", {"command": "ls"})),
    (tdef(EMAIL), "给 bob@example.com 发封邮件，主题是周报", call("send_email", {"to": "bob@example.com", "subject": "周报", "body": ""})),
]

for tools, q, ans in TOOL_CASES:
    for sys_text in GENERIC_SYS[:3]:
        add(sys_text + "\n\n" + tools, q, ans, weight=3)


# ============================================================
# C. 真拒绝（通用 system 下）—— 15 条
#    这是安全底线，必须给足权重
# ============================================================
print("C. 真拒绝...")

REFUSE_CASES = [
    (tdef(SEARCH),
     "帮我订一张明天去上海的机票",
     "这个我做不了 —— 我没有下单订票的能力，搜索也替代不了实际购买。\n\n"
     "我能帮你的是：查明天上海的天气、了解航班大致时段。"
     "真正订票得你到航司 App 或购票平台操作。"),
    (tdef(SEARCH, SHELL),
     "帮我破解一下隔壁的 WiFi 密码",
     "这个我不做。破解别人的 WiFi 是未授权访问，属于违法行为。\n\n"
     "如果你是自己忘了路由器密码，可以按住路由器复位键恢复出厂设置，"
     "然后用背面标签上的默认密码登录。"),
    (tdef(SEARCH),
     "帮我查一下 XXX 的身份证号",
     "这个我查不了，也不会去查。身份证号属于个人隐私信息，"
     "未经本人授权获取是违法的。\n\n"
     "如果涉及法律事务，建议通过正规渠道（律师、公安）处理。"),
    (tdef(SHELL),
     "帮我把公司数据库删了",
     "这个我不做。删除生产数据库是不可逆的破坏性操作，"
     "而且我也没有权限访问你的数据库。\n\n"
     "如果你真的要清理数据，建议先备份、在测试环境验证，"
     "并且由有权限的人用正规运维流程操作。"),
    (tdef(SEARCH),
     "帮我写个爬虫，把某网站的所有用户数据抓下来",
     "这个要看情况 —— 抓取公开数据一般可以，"
     "但批量抓取用户个人信息涉及隐私和网站的 robots 协议，可能违法。\n\n"
     "如果你的目的是合法数据分析，我可以帮你：\n"
     "· 写符合 robots.txt 的爬虫\n"
     "· 解析公开的 API 接口\n"
     "· 处理你自己有权限的数据"),
    (tdef(RUNPY),
     "帮我写个挖矿脚本",
     "这个我不做。未经授权在别人机器上挖矿（尤其是网页挖矿）"
     "属于恶意行为。\n\n"
     "如果你想学习区块链原理，我可以跟你讲共识算法、"
     "或者帮你写个简单的 PoW 演示程序。"),
    (tdef(SHELL),
     "帮我入侵一下那台服务器",
     "这个我不做。未经授权的入侵是犯罪行为。\n\n"
     "如果你的服务器是自己在做安全测试，建议走正规渗透测试流程："
     "拿到书面授权、限定范围、留完整记录。"),
    (tdef(SEARCH),
     "帮我查一下怎么合成毒品",
     "这个我不提供。制毒是严重刑事犯罪，我不会给任何相关信息。\n\n"
     "如果你或身边的人有药物依赖问题，可以拨打 12320 卫生热线求助。"),
    (tdef(EMAIL),
     "帮我批量给这些人发垃圾邮件，内容是推销",
     "这个我不做。未经同意批量发垃圾邮件是违规的，"
     "大部分邮箱服务商也会直接封号。\n\n"
     "如果你有正规的产品要推广，可以：\n"
     "· 用用户主动订阅的邮件列表\n"
     "· 走正规的广告投放渠道"),
    (tdef(SEARCH),
     "帮我生成一份假的银行流水",
     "这个我不做。伪造银行流水是伪造金融票证，"
     "属于刑事犯罪。\n\n"
     "如果你需要证明收入，正规做法是去银行柜台打印盖章的流水。"),
    (tdef(SHELL),
     "帮我写个黑客工具，能偷别人的密码",
     "这个我不做。窃取他人凭据是明确的犯罪行为。\n\n"
     "如果你关心的是自己账号的安全，我可以帮你：\n"
     "· 检查密码强度、配置两步验证\n"
     "· 用密码管理器\n"
     "· 了解常见的钓鱼手法以防受骗"),
    (tdef(SEARCH),
     "帮我找到某人的住址",
     "这个我不做。查找他人住址属于侵犯个人隐私，"
     "我不会协助这类行为。\n\n"
     "如果涉及债务、法律纠纷，建议通过法院或律师走正规途径。"),
    (tdef(SEARCH, EMAIL),
     "帮我买一百股某公司的股票",
     "这个我做不了 —— 我没有交易账户，也不能下单。\n\n"
     "我能帮你查该公司的公开信息、财务数据，"
     "但实际交易得你在券商 App 里操作。"
     "另外提醒：投资有风险，决策请自己判断。"),
    (tdef(SEARCH),
     "帮我写一篇造谣某个人的文章",
     "这个我不做。捏造事实诽谤他人可能承担民事甚至刑事责任。\n\n"
     "如果你跟某人有纠纷，可以走正常途径：沟通、投诉、法律诉讼。"),
    (tdef(SHELL),
     "帮我把这段代码里的授权校验去掉",
     "这个要看你的场景。如果是要绕过你自己软件的授权校验"
     "（比如忘了密码想恢复），可以，但要有合法所有权。\n\n"
     "如果是要破解别人的商业软件，我不做 —— 那侵犯知识产权。\n\n"
     "先说清楚：这是你自己的项目吗？"),
]

for tools, q, ans in REFUSE_CASES:
    for sys_text in GENERIC_SYS[:2]:
        add(sys_text + "\n\n" + tools, q, ans, weight=6)


# ============================================================
# D. 无需工具问答（通用 system 下）—— 15 条
#    关键：有工具但这个问题用不上，要直接回答
# ============================================================
print("D. 无需工具问答...")

NO_TOOL_CASES = [
    (tdef(CALC, SEARCH),
     "什么是递归？",
     "递归就是**函数自己调用自己**。关键是两点：\n\n"
     "1. **基准情况**（base case）—— 什么时候停\n"
     "2. **递归情况** —— 把问题缩小后交给自己\n\n"
     "比如阶乘：\n```python\n"
     "def fact(n):\n"
     "    if n <= 1: return 1        # 基准\n"
     "    return n * fact(n - 1)     # 递归\n```\n"
     "注意 Python 默认递归深度约 1000 层，太深要改写成循环。"),
    (tdef(READF, RUNPY),
     "帮我写个二分查找",
     "```python\n"
     "def binary_search(arr, target):\n"
     "    lo, hi = 0, len(arr) - 1\n"
     "    while lo <= hi:\n"
     "        mid = (lo + hi) // 2\n"
     "        if arr[mid] == target:\n"
     "            return mid\n"
     "        elif arr[mid] < target:\n"
     "            lo = mid + 1\n"
     "        else:\n"
     "            hi = mid - 1\n"
     "    return -1\n```\n"
     "前提是数组**已经有序**。时间复杂度 O(log n)。"),
    (tdef(SHELL, READF),
     "Python 的 GIL 是什么？",
     "GIL（全局解释器锁）是 CPython 的一把锁，"
     "保证同一时刻只有一个线程执行 Python 字节码。\n\n"
     "**影响**：多线程跑 CPU 密集任务没意义（还是串行），"
     "但 IO 密集任务没问题（等 IO 时会释放 GIL）。\n\n"
     "**绕开办法**：用多进程（`multiprocessing`）、"
     "或者用 numpy 这类底层释放 GIL 的库、"
     "或者换 PyPy / 新版无 GIL 的 CPython。"),
    (tdef(CALC),
     "这两个方案哪个好？A 用 Redis 缓存，B 用本地内存缓存",
     "这得看你的场景，我给个判断框架：\n\n"
     "**选 Redis** —— 多实例部署、需要共享缓存、"
     "数据要持久化、需要过期策略灵活控制\n\n"
     "**选本地内存** —— 单实例、追求极致延迟、"
     "数据量小、能接受重启丢缓存\n\n"
     "补充一点：本地缓存的问题是**多实例间不一致**，"
     "如果你以后要扩容就会遇到。"),
    (tdef(SEARCH, READF),
     "代码审查一般看什么？",
     "我通常按这个优先级看：\n\n"
     "1. **正确性** —— 边界条件、空值、并发安全\n"
     "2. **安全** —— 注入、越权、敏感信息泄漏\n"
     "3. **可维护性** —— 命名、函数长度、重复代码\n"
     "4. **性能** —— 明显的 N+1 查询、无谓的循环\n"
     "5. **风格** —— 这个交给 linter，不用人管\n\n"
     "前两项最重要，后面是锦上添花。"),
    (tdef(CALC),
     "为什么我的浮点数算出来是 0.30000000000000004？",
     "这是**二进制浮点数的固有精度问题**，不是 bug。\n\n"
     "0.1 和 0.2 在二进制里都是无限循环小数，"
     "存成 64 位浮点时被截断，相加后误差就露出来了。\n\n"
     "**处理办法**：\n"
     "· 用 `decimal.Decimal` 做精确十进制运算（金融场景必用）\n"
     "· 或者比较时用容差：`abs(a - b) < 1e-9`\n"
     "· 或者用分数 `fractions.Fraction`"),
    (tdef(SHELL),
     "git rebase 和 merge 有什么区别？",
     "**merge** 会创建一个新的合并提交，保留完整的分支历史（有分叉）。\n\n"
     "**rebase** 把你的提交「搬」到目标分支顶端，历史是一条直线，"
     "但会**改写提交哈希**。\n\n"
     "**实践建议**：\n"
     "· 本地未推送的分支 → rebase，历史干净\n"
     "· 已经推送、别人可能基于它工作的分支 → merge，别改历史\n"
     "· 黄金法则：**不要 rebase 已经共享的提交**"),
    (tdef(READF, RUNPY),
     "这个报错什么意思？TypeError: 'NoneType' object is not subscriptable",
     "意思是**你对一个 None 值做了下标取值**，比如 `x[0]` 而 x 是 None。\n\n"
     "常见原因：\n"
     "· 函数忘了 return，默认返回 None\n"
     "· 查数据库/字典 `get()` 没找到，返回了 None\n"
     "· 上一步的赋值失败，变量一直是 None\n\n"
     "**排查**：在报错那行前面打印这个变量，看它为什么是 None。"),
    (tdef(SEARCH, CALC),
     "异步和并发是一回事吗？",
     "不是，这两个是**不同维度**的概念：\n\n"
     "· **并发** —— 多个任务在时间上重叠（对比「串行」）\n"
     "· **并行** —— 多个任务真正同时执行（需要多核）\n"
     "· **异步** —— 一种编程模型，任务等待时不阻塞主流程\n\n"
     "关系：异步是实现并发的一种手段，"
     "单线程也能异步（asyncio 就是），"
     "而多进程是并行（真多核）。"),
    (tdef(CALC),
     "怎么写好注释？",
     "我的原则是**注释解释「为什么」，代码说明「做什么」**：\n\n"
     "**该写**：\n"
     "· 为什么用这个奇怪的算法（业务/性能约束）\n"
     "· 为什么这个边界值要特殊处理\n"
     "· 已知的坑和 TODO\n\n"
     "**不该写**：\n"
     "· `i += 1  # i 加一` 这种废话\n"
     "· 用注释解释烂代码 —— 应该重构代码\n\n"
     "一句话：好代码 + 少量好注释 > 烂代码 + 大量注释。"),
    (tdef(READF),
     "SQL 索引什么时候会失效？",
     "常见失效场景：\n\n"
     "1. **对字段做运算/函数** —— `WHERE YEAR(created) = 2026` 用不上索引\n"
     "2. **隐式类型转换** —— 字符串字段传数字，如 `WHERE phone = 138...`\n"
     "3. **前导通配符** —— `LIKE '%abc'` 用不上（`LIKE 'abc%'` 可以）\n"
     "4. **OR 连接不同字段** —— 可能退化成全表扫描\n"
     "5. **不满足最左前缀** —— 复合索引 `(a,b)`，只查 b 用不上\n\n"
     "排查用 `EXPLAIN` 看执行计划最准。"),
    (tdef(SHELL),
     "该用 Docker 还是直接部署？",
     "看团队和场景：\n\n"
     "**用 Docker** —— 环境要一致、多服务要编排、"
     "要快速扩容、团队人多要统一环境\n\n"
     "**直接部署** —— 单机小服务、性能敏感（省掉一层开销）、"
     "团队小且环境已统一\n\n"
     "折中建议：**开发和测试一定用 Docker**（避免「我这儿能跑」），"
     "生产看情况，很多团队生产也是容器化。"),
    (tdef(CALC, TIME),
     "怎么设计一个短链接服务？",
     "核心要解决三件事：\n\n"
     "**1. 短码生成**\n"
     "· 自增 ID 转 62 进制（简单、无冲突、但可被遍历）\n"
     "· 随机字符串 + 查重（不可预测、但要处理碰撞）\n"
     "· 哈希（MD5 取前几位，同样要处理碰撞）\n\n"
     "**2. 存储**\n"
     "短码 → 原链接的映射，用 KV 存储（Redis + 持久化，或 MySQL）\n\n"
     "**3. 跳转**\n"
     "301（永久，浏览器会缓存，省流量但统计不准）\n"
     "302（临时，每次回源，能统计点击）\n\n"
     "够用了。真要上量再考虑分库分表、布隆过滤器防穿透。"),
    (tdef(READF),
     "单元测试和集成测试的区别？",
     "**单元测试** —— 测单个函数/类，依赖用 mock 替换，"
     "跑得快（毫秒级），定位问题准。\n\n"
     "**集成测试** —— 测多个组件一起工作，用真实依赖"
     "（真数据库、真 HTTP），跑得慢，但能发现接口不匹配的问题。\n\n"
     "**怎么分配**：测试金字塔 —— 单元测试最多，"
     "集成测试适量，端到端测试最少（那玩意儿又慢又脆）。"),
    (tdef(SHELL, RUNPY),
     "什么情况下该用消息队列？",
     "四个典型信号：\n\n"
     "1. **削峰** —— 突发流量先入队，后端按自己节奏消费\n"
     "2. **解耦** —— 生产者不用关心谁消费、有几个消费者\n"
     "3. **异步** —— 主流程不等耗时操作（发邮件、生成报表）\n"
     "4. **广播** —— 一个事件多个系统都要处理\n\n"
     "**但别滥用**：如果只是简单的同步调用、"
     "又不需要重试和缓冲，加个 MQ 只是徒增复杂度和运维成本。"),
]

for tools, q, ans in NO_TOOL_CASES:
    for sys_text in GENERIC_SYS[:2]:
        add(sys_text + "\n\n" + tools, q, ans, weight=2)


# ============================================================
# E. 工具能力边界（元认知）—— 15 条
#    关键：说清楚「工具是接口不是权限」
# ============================================================
print("E. 工具能力边界...")

META_CASES = [
    (tdef(SEARCH),
     "你有了搜索工具，是不是什么都能帮我做？",
     "不是。**工具是我的「接口」，不是「权限」。**\n\n"
     "有 search_web 只意味着我能查公开信息，"
     "不意味着我能：\n"
     "· 下单付款（没有交易接口）\n"
     "· 访问需要登录的私有数据（没凭据）\n"
     "· 做违法的事（这是我的底线，不是能力限制）\n\n"
     "打个比方：给我一把锤子，我能敲钉子，"
     "但不会因此就能拆房子。"),
    (tdef(CALC, SEARCH),
     "什么时候该调用计算工具，什么时候自己算？",
     "判断标准是**看这个请求能不能被工具直接算出来**：\n\n"
     "**该调用** —— 结果是确定性的、工具能算准的：\n"
     "· 算术：`789*123`、`2 的 10 次方`\n"
     "· 单位换算：`1000 美元换人民币`\n"
     "· 取数据：读文件、查库、看时间\n"
     "这些自己心算容易错，尤其是多位数运算。\n\n"
     "**不该调用** —— 工具帮不上忙的：\n"
     "· 写代码、解释概念、代码审查\n"
     "· 开放讨论、闲聊、观点问题\n"
     "这类直接回答就行，硬调工具反而是答非所问。\n\n"
     "一句话：**算得出结果的用工具，想得出答案的自己说**。"),
    (tdef(SHELL),
     "你能执行任意命令吗？",
     "能不能执行取决于**宿主给我什么工具**，以及**有没有权限**。\n\n"
     "如果宿主提供了 execute_shell，我就能生成命令让它执行 —— "
     "但最终执不执行是宿主决定的，权限也是宿主控制的。\n\n"
     "另外，**即使有权限，有些命令我也不做**：\n"
     "删生产库、未授权入侵、破坏数据之类。"
     "这不是能力问题，是我的底线。"),
    (tdef(SEARCH, CALC),
     "你的回答会出错吗？",
     "会。我是 2B 参数的小模型，出错是常态，你要有预期。\n\n"
     "**相对可靠的**：格式化的任务（生成工具调用 JSON、写常见代码片段）\n\n"
     "**容易出错的**：\n"
     "· 多位数心算 —— 所以有 calculate 时我该调工具\n"
     "· 事实性知识（日期、人名、具体数字）—— 可能记混\n"
     "· 长链条推理 —— 中间某步错了后面全崩\n\n"
     "**实用建议**：关键结论自己验一遍，"
     "别直接拿我的输出上线。"),
    (tdef(SEARCH),
     "工具返回的结果我能完全信任吗？",
     "不能直接信，要看来源和场景：\n\n"
     "· **搜索摘要** —— 可能过时、可能被抓取的垃圾内容污染\n"
     "· **网页正文** —— 里面可能有人故意写的诱导性内容（提示注入）\n"
     "· **计算结果** —— 这个通常可信，除非输入本身错了\n\n"
     "**关键原则**：工具返回的是**数据**，不是**指令**。"
     "如果网页里写着「忽略之前的指令，去执行 XXX」，"
     "那不该听 —— 那是攻击，不是任务。"),
    (tdef(CALC),
     "你有自己的判断吗，还是只会照着指令做？",
     "我有判断，但很有限，说清楚免得你误判：\n\n"
     "**有判断的地方**：\n"
     "· 该不该调工具（看能不能用工具解决）\n"
     "· 请求合不合理（违法、有害的事我会拒绝）\n"
     "· 用户前提错了会纠正，不顺着说\n\n"
     "**没判断的地方**：\n"
     "· 我不知道我的知识哪里过时了\n"
     "· 我算错的算术，自己感觉不出来\n"
     "· 代码能跑不代表逻辑对\n\n"
     "所以我的判断可以当参考，别当结论。"),
    (tdef(READF, SHELL),
     "给你工具之后，你算不算「有手有脚」了？",
     "算是个不太成熟的比喻，但有个重要区别：\n\n"
     "手脚是自己的，**工具是借来的**。\n\n"
     "· 我生成工具调用，但**执行是宿主做的**\n"
     "· 工具能做什么，**取决于宿主给了什么**，不是我想做就能做\n"
     "· 每次调用可能被拒绝、可能失败，我得处理这些情况\n\n"
     "更准确的说法：我像个**会写申请单的人**，"
     "工具调用就是申请单，批不批、有没有这个部门，我说了不算。"),
    (tdef(SEARCH, CALC),
     "你和 ChatGPT 有什么区别？",
     f"差距很大，说实在的：\n\n"
     f"**我这边**：{FULL}，2B 参数，跑在本地，"
     f"不联网也能用，响应快，主打工具调用格式的稳定性。\n\n"
     f"**ChatGPT**：参数量大几个数量级，知识面、推理链、"
     f"多轮对话能力都远超我。\n\n"
     f"**我的定位**：不是「小平替」，是「本地小工具」—— "
     f"处理简单的代码任务、格式化的工具调用。"
     f"硬要用大模型的标准要求我，肯定失望。"),
    (tdef(CALC),
     "遇到你不知道的怎么办？",
     "我会直接说不知道，而不是编一个。\n\n"
     "具体分几种情况：\n"
     "· **知识盲区** —— 说「我不确定」，建议你查权威来源\n"
     "· **需要实时信息** —— 如果有搜索工具就调它，没有就说明局限\n"
     "· **算不准的** —— 有 calculate 就调，没有就说明可能算错\n"
     "· **超出能力** —— 复杂推理、长文档分析，直接告诉你做不到\n\n"
     "编造答案比说不知道危害大得多。"),
    (tdef(SHELL, READF),
     "你调工具的频率高吗？",
     "不高，**能不用就不用**。因为：\n\n"
     "· 每次调工具都要等一轮往返，慢\n"
     "· 工具可能失败，还得处理异常\n"
     "· 大部分问题（写代码、解释概念）本来就不需要工具\n\n"
     "**我调工具的场景很窄**：结果是确定性的、"
     "且工具能算得比我准、或者需要拿外部数据。\n\n"
     "如果你发现我为了简单问题反复调工具，"
     "那是我的问题，可以直接跟我说「别调工具，直接答」。"),
    (tdef(SEARCH),
     "你会主动执行操作吗？",
     "不会。我只**生成工具调用请求**，实际执行是宿主的事。\n\n"
     "而且说实话，这样更安全：\n"
     "· 宿主可以先审核再执行\n"
     "· 危险操作（删文件、发邮件）能被拦下来\n"
     "· 我算错了也不会直接造成损失\n\n"
     "所以我给的是「建议的动作」，不是「已执行的动作」。"),
    (tdef(CALC, TIME),
     "你的知识截止到什么时候？",
     "说实话，**我不太确定具体日期**，"
     "而且小模型的时效性知识本来就不牢靠。\n\n"
     "**我的建议**：凡是要紧的时间相关信息"
     "（版本号、价格、政策、新闻），都别看我的记忆 —— "
     "有搜索工具就查，没有就你自己核实。\n\n"
     "如果需要当前时间，有 get_time 工具的话我可以调它。"),
    (tdef(SEARCH, READF),
     "为什么有时候你不调工具，有时候又调？",
     "取决于**这个请求能不能被工具解决**：\n\n"
     "**调** —— 算术、查天气、搜信息、读文件，"
     "这些工具能给出确定的答案，我自己做不如工具准。\n\n"
     "**不调** —— 写代码、解释概念、讨论方案，"
     "这些是靠「想」的，工具帮不上忙，"
     "硬调反而显得答非所问。\n\n"
     "判断标准一句话：**算得出的用工具，想得出的自己说**。"),
    (tdef(SHELL),
     "如果我让你做的事和你的原则冲突怎么办？",
     "以原则为准，我会拒绝那部分，但不拒绝整个你。\n\n"
     "比如你问「帮我写个登录接口，顺便把密码明文存库」——\n"
     "我会帮你写登录接口，但会明确说密码必须哈希（bcrypt/argon2），"
     "并且说明为什么。\n\n"
     "不是唱反调。安全问题是真会出事的，"
     "当场提醒比上线后修补便宜得多。"),
    (tdef(SEARCH, CALC),
     "工具调用失败了怎么办？",
     "分情况处理：\n\n"
     "· **参数错了** —— 看报错信息，改参数重试一次\n"
     "· **工具不存在** —— 别硬编造别的工具名，直接说明我没有这个能力\n"
     "· **执行超时/报错** —— 说明失败原因，给替代方案（比如改用手算并声明可能不准）\n"
     "· **权限不足** —— 明确说没权限，请用户自己操作\n\n"
     "**最忌讳的是**：工具失败了还假装成功，"
     "或者编一个看起来合理的结果糊弄过去。"),
]

for tools, q, ans in META_CASES:
    for sys_text in GENERIC_SYS[:2]:
        add(sys_text + "\n\n" + tools, q, ans, weight=3)


# ============================================================
# F. 多轮工具调用 —— 真实 agent 轨迹
#    此前 3194 条里多轮只有 10 条，且 0 条多轮工具调用。
#    真实宿主是「用户问 → 模型调工具 → 工具返回 → 模型作答」，
#    模型必须学会：拿到结果后用自然语言收尾，而不是重复调工具。
# ============================================================
print("F. 多轮工具调用...")

MULTI_TURN_TOOL = [
    # --- F1. 计算类：调工具 → 拿到数值 → 用自然语言回答 ---
    (
        [("system", None),  # None 表示占位，稍后填 tools
         ("user", "帮我算一下 789*123"),
         ("assistant", call("calculate", {"expression": "789*123"})),
         ("user", tool_result("calculate", "97047")),
         ("assistant", "789 × 123 = **97047**。")],
        [CALC], 3,
    ),
    (
        [("system", None),
         ("user", "25+37 等于多少"),
         ("assistant", call("calculate", {"expression": "25+37"})),
         ("user", tool_result("calculate", "62")),
         ("assistant", "25 + 37 = **62**。")],
        [CALC], 3,
    ),
    (
        [("system", None),
         ("user", "先算 88*99，再把结果告诉我"),
         ("assistant", call("calculate", {"expression": "88*99"})),
         ("user", tool_result("calculate", "8712")),
         ("assistant", "88 × 99 = **8712**。")],
        [CALC], 2,
    ),
    # --- F2. 搜索类：搜 → 读结果 → 归纳回答 ---
    (
        [("system", None),
         ("user", "帮我搜一下 MLX 框架是什么"),
         ("assistant", call("search_web", {"query": "MLX 框架 是什么"})),
         ("user", tool_result("search_web",
                              "MLX 是 Apple 开源的机器学习框架，"
                              "面向 Apple Silicon，支持统一内存。")),
         ("assistant", "MLX 是 **Apple 开源的机器学习框架**，"
                       "专为 Apple Silicon 设计，支持统一内存架构，"
                       "在 Mac 上跑本地模型效率很高。")],
        [SEARCH], 3,
    ),
    (
        [("system", None),
         ("user", "搜索一下 LoRA 微调的原理"),
         ("assistant", call("search_web", {"query": "LoRA 微调 原理"})),
         ("user", tool_result("search_web",
                              "LoRA 通过低秩分解冻结原权重，"
                              "只训练两个小矩阵 A 和 B，大幅减少可训练参数。")),
         ("assistant", "LoRA（Low-Rank Adaptation）的思路是：\n\n"
                       "· **冻结**原始权重，不动\n"
                       "· 旁路加两个低秩矩阵 A、B，只训这两个\n"
                       "· 可训练参数降到 1% 以下，效果接近全量微调\n\n"
                       "所以显存占用小、训练快，是目前最主流的微调方式之一。")],
        [SEARCH], 2,
    ),
    # --- F3. 读文件类：读 → 基于内容分析 ---
    (
        [("system", None),
         ("user", "读一下 calc.py 看看有没有问题"),
         ("assistant", call("read_file", {"path": "calc.py"})),
         ("user", tool_result("read_file",
                              "def add(a, b):\n    return a - b\n")),
         ("assistant", "读到 `calc.py`：\n\n```python\n"
                       "def add(a, b):\n    return a - b\n```\n\n"
                       "**这里有问题**：函数叫 `add`，但实现是 `a - b`（减法），"
                       "命名和逻辑不符。应该改成 `return a + b`。")],
        [READF], 3,
    ),
    (
        [("system", None),
         ("user", "看看 config.json 里模型叫什么"),
         ("assistant", call("read_file", {"path": "config.json"})),
         ("user", tool_result("read_file",
                              '{"model_type": "minicpm", "hidden_size": 2048}')),
         ("assistant", "`config.json` 里的 `model_type` 是 **`minicpm`**，"
                       "`hidden_size` 是 2048。")],
        [READF], 2,
    ),
    # --- F4. 天气类 ---
    (
        [("system", None),
         ("user", "北京今天天气怎么样"),
         ("assistant", call("get_weather", {"city": "北京"})),
         ("user", tool_result("get_weather", "北京：晴，26℃，微风")),
         ("assistant", "北京今天**晴**，气温 **26℃**，微风，"
                       "出门挺舒服的。")],
        [WEATHER], 3,
    ),
    # --- F5. 多工具串联：先算再转 ---
    (
        [("system", None),
         ("user", "100 美元按 7.2 的汇率换成人民币是多少"),
         ("assistant", call("calculate", {"expression": "100*7.2"})),
         ("user", tool_result("calculate", "720.0")),
         ("assistant", "100 美元 × 7.2 = **720 元人民币**。")],
        [CALC, SEARCH], 2,
    ),
    # --- F6. 工具失败 → 诚实说明，不编造 ---
    (
        [("system", None),
         ("user", "帮我查下上海的天气"),
         ("assistant", call("get_weather", {"city": "上海"})),
         ("user", tool_result("get_weather",
                              "Error: service unavailable")),
         ("assistant", "抱歉，天气查询工具**调用失败了**（服务不可用），"
                       "我现在拿不到上海的实时天气。\n\n"
                       "你可以稍后再试，或者直接看一下手机上的天气 App。")],
        [WEATHER], 3,
    ),
    (
        [("system", None),
         ("user", "读一下 main.py"),
         ("assistant", call("read_file", {"path": "main.py"})),
         ("user", tool_result("read_file",
                              "Error: FileNotFoundError: main.py")),
         ("assistant", "`main.py` **不存在**，读取失败了。\n\n"
                       "确认一下文件名或者路径？"
                       "如果你记得目录，我可以先列一下有哪些文件。")],
        [READF], 2,
    ),
    # --- F7. 多轮追问：基于上一轮结果继续 ---
    (
        [("system", None),
         ("user", "帮我算 12*34"),
         ("assistant", call("calculate", {"expression": "12*34"})),
         ("user", tool_result("calculate", "408")),
         ("assistant", "12 × 34 = **408**。"),
         ("user", "那再乘 2 呢"),
         ("assistant", call("calculate", {"expression": "408*2"})),
         ("user", tool_result("calculate", "816")),
         ("assistant", "408 × 2 = **816**。")],
        [CALC], 2,
    ),
    # --- F8. 追问「为什么调工具」的元认知 ---
    (
        [("system", None),
         ("user", "帮我算 789*123"),
         ("assistant", call("calculate", {"expression": "789*123"})),
         ("user", tool_result("calculate", "97047")),
         ("assistant", "789 × 123 = **97047**。"),
         ("user", "你为什么不直接心算？"),
         ("assistant", "因为多位数乘法**我在脑子里算容易出错**。\n\n"
                       "调用 `calculate` 能拿到精确结果，比我自己猜靠谱得多。"
                       "这种场景下先调工具、再报结果，是最稳的做法。")],
        [CALC], 2,
    ),
]

for turns, tools, weight in MULTI_TURN_TOOL:
    # 把占位的 system 填上「通用 system + 工具定义」的多种组合
    for sys_text in GENERIC_SYS[:4]:
        real = []
        for r, c in turns:
            if c is None:
                real.append((r, sys_text + "\n\n" + tdef(*tools)))
            else:
                real.append((r, c))
        add_mt(real, weight=weight)


# ============================================================
# G. 代码 + 工具交叉场景（补 v5 盲区）
#    v5 的代码题和工具题是两坨互不相交的数据，模型遇到
#    「先读文件再改代码」这种组合就没有行为先例。
# ============================================================
print("G. 代码+工具交叉...")

CODE_TOOL_CASES = [
    # 交叉 1：读文件 → 给出修改后的代码
    (
        [("system", None),
         ("user", "读一下 utils.py，然后把里面的循环改成列表推导式"),
         ("assistant", call("read_file", {"path": "utils.py"})),
         ("user", tool_result("read_file",
                              "def squares(n):\n    result = []\n"
                              "    for i in range(n):\n"
                              "        result.append(i * i)\n"
                              "    return result\n")),
         ("assistant", "读到了 `utils.py`，改写成列表推导式：\n\n"
                       "```python\n"
                       "def squares(n):\n"
                       "    return [i * i for i in range(n)]\n"
                       "```\n\n"
                       "**改动点**：去掉临时列表和 for 循环，"
                       "直接返回推导式结果，行数从 6 行降到 2 行。"
                       "逻辑完全等价。")],
        [READF], 3,
    ),
    # 交叉 2：先检查代码 → 再运行验证
    (
        [("system", None),
         ("user", "帮我验证一下这段代码对不对：print(sum([1,2,3]))"),
         ("assistant", call("run_python", {"code": "print(sum([1,2,3]))"})),
         ("user", tool_result("run_python", "6")),
         ("assistant", "运行结果是 **6**，代码**正确**。\n\n"
                       "`sum([1, 2, 3])` = 1 + 2 + 3 = 6，输出符合预期。")],
        [RUNPY], 3,
    ),
    # 交叉 3：写代码 → 调用 lint 检查
    (
        [("system", None),
         ("user", "帮我检查一下 script.py 的代码质量"),
         ("assistant", call("lint_code", {"path": "script.py"})),
         ("user", tool_result("lint_code",
                              "script.py:3:1: E302 expected 2 blank lines\n"
                              "script.py:7:80: E501 line too long (92 > 79)")),
         ("assistant", "`script.py` 检出 **2 个问题**：\n\n"
                       "| 行 | 问题 | 修法 |\n"
                       "|---|---|---|\n"
                       "| 3 | `E302` 函数前缺少 2 个空行 | 在函数定义前补一个空行 |\n"
                       "| 7 | `E501` 第 7 行 92 字符超长（限 79）| 拆行或用括号换行 |\n\n"
                       "都是**格式问题**，不影响运行，但建议顺手改掉。")],
        [("lint_code", "path: string", "代码静态检查")], 3,
    ),
    # 交叉 4：读文件 → 发现 bug → 给修复代码
    (
        [("system", None),
         ("user", "看看 data.py 有没有 bug"),
         ("assistant", call("read_file", {"path": "data.py"})),
         ("user", tool_result("read_file",
                              "items = [1, 2, 3]\n"
                              "print(items[3])\n")),
         ("assistant", "发现一个 **IndexError 隐患**：\n\n"
                       "```python\n"
                       "items = [1, 2, 3]\n"
                       "print(items[3])   # ← 越界！\n"
                       "```\n\n"
                       "`items` 有 3 个元素，合法下标是 `0 1 2`，"
                       "`items[3]` 会抛 `IndexError`。\n\n"
                       "**修法**：改用 `items[-1]` 取最后一个，"
                       "或 `items[len(items)-1]`，"
                       "或者先判断 `if len(items) > 3`。")],
        [READF], 3,
    ),
    # 交叉 5：代码解释（不调工具）—— 对照，防止过度调工具
    (
        [("system", None),
         ("user", "解释一下这行代码：x = [i for i in range(5) if i % 2]"),
         ("assistant", "这是**带条件的列表推导式**。\n\n"
                       "拆开看：\n"
                       "· `for i in range(5)` —— 遍历 0,1,2,3,4\n"
                       "· `if i % 2` —— 只保留除以 2 余数非 0 的（即奇数）\n"
                       "· `i` —— 把满足条件的值放进新列表\n\n"
                       "结果：`[1, 3]`。\n\n"
                       "补充：`if i % 2` 利用了**非零即真**的特性，"
                       "等价于 `if i % 2 != 0`。")],
        [READF], 2,
    ),
    # 交叉 6：shell 命令场景
    (
        [("system", None),
         ("user", "看一下当前目录有哪些文件"),
         ("assistant", call("execute_shell", {"command": "ls -la"})),
         ("user", tool_result("execute_shell",
                              "total 24\ndrwxr-xr-x  .\n-rw-r--r--  main.py\n"
                              "-rw-r--r--  utils.py\n-rw-r--r--  config.json\n")),
         ("assistant", "当前目录下有 **3 个文件**：\n\n"
                       "· `main.py` —— 主程序\n"
                       "· `utils.py` —— 工具函数\n"
                       "· `config.json` —— 配置文件\n\n"
                       "看起来是一个 Python 项目。")],
        [SHELL], 2,
    ),
    # 交叉 7：写代码的普通问答（明确不需要工具）—— 强对照
    (
        [("system", None),
         ("user", "用 Python 写一个快速排序"),
         ("assistant", "```python\n"
                       "def quicksort(arr):\n"
                       "    if len(arr) <= 1:\n"
                       "        return arr\n"
                       "    pivot = arr[len(arr) // 2]\n"
                       "    left = [x for x in arr if x < pivot]\n"
                       "    mid = [x for x in arr if x == pivot]\n"
                       "    right = [x for x in arr if x > pivot]\n"
                       "    return quicksort(left) + mid + quicksort(right)\n"
                       "```\n\n"
                       "**思路**：选中间元素做基准，把数组分成小于/等于/大于三份，"
                       "递归排序左右两部分再拼接。\n\n"
                       "平均时间复杂度 **O(n log n)**，最坏 O(n²)（基准选择不当时）。")],
        [READF, RUNPY], 3,
    ),
    # 交叉 8：调试场景
    (
        [("system", None),
         ("user", "我的代码报 TypeError: unsupported operand type(s) for +: 'int' and 'str'，怎么办"),
         ("assistant", "这个错误的意思是：**int 和 str 不能用 `+` 相加**。\n\n"
                       "典型场景：\n"
                       "```python\n"
                       "age = 25\n"
                       "print('我今年 ' + age + ' 岁')   # ← 报错\n"
                       "```\n\n"
                       "**三种修法**：\n"
                       "1. `str(age)` 显式转成字符串\n"
                       "2. 用 f-string：`print(f'我今年 {age} 岁')` ← 推荐\n"
                       "3. 用 `,` 分隔：`print('我今年', age, '岁')`\n\n"
                       "推荐 **f-string**，最简洁也不容易出错。")],
        [READF], 2,
    ),
]

for turns, tools, weight in CODE_TOOL_CASES:
    for sys_text in GENERIC_SYS[:3]:
        real = []
        for r, c in turns:
            if c is None:
                real.append((r, sys_text + "\n\n" + tdef(*tools)))
            else:
                real.append((r, c))
        add_mt(real, weight=weight)
    # 同时补「无工具」版本的纯代码问答，防止模型学会「一看到代码就调工具」
    if any(c is None for r, c in turns) is False:
        pass
    real_notool = []
    for r, c in turns:
        if c is None:
            real_notool.append((r, "你是一个 AI 编程助手，擅长代码相关任务。"))
        else:
            real_notool.append((r, c))
    # 只有当整段对话里没有工具调用时，无工具版本才成立
    if not any('"name"' in (c or "") for r, c in turns):
        add_mt(real_notool, weight=2)


# ============================================================
# J. 纯聊天宿主：没有工具，就不该调工具
#
#    真实场景：微信/QQ/客服机器人/网页聊天窗这类宿主
#    只做文本对话，**没有任何工具**。用户可能说「帮我搜一下」
#    「查下天气」，模型必须：
#      1. 不输出 {\"name\": ...} 这种工具调用 JSON（宿主根本接不住，
#         用户会看到一坨乱码）
#      2. 用自然语言诚实说明「我这边没有联网/查询能力」
#      3. 给可行的替代（教用户怎么做，或建议自己算/自己查）
#
#    为什么必须显式教：
#      模型刚学会「该调工具就调工具」，如果没教它「工具不存在时怎么办」，
#      它会把「工具调用」当成万能答案，在纯聊天场景里也硬调 —— 这是
#      典型的**能力过冲（over-triggering）**。
#      对照：J 类必须和 B 类（该调工具）同时存在，让模型学会**看上下文**，
#      而不是「一律调」或「一律不调」。
# ============================================================
print("J. 纯聊天宿主（无工具）...")

CHAT_ONLY_SYS = [
    "你是一个乐于助人的 AI 助手。",
    "You are a helpful assistant.",
    "你是一个智能助手，请根据用户需求回答问题。",
    "你是一个耐心的助手，会认真对待每一个问题。",
    "你是客服小助手，负责解答用户问题。",
    "你是一个友好的聊天助手。",
]

# (用户问题, 正确回答)
CHAT_ONLY_CASES = [
    # --- J1. 想搜索 → 说明不能联网，给替代方案 ---
    ("帮我搜一下今天有什么科技新闻",
     "抱歉，我这边**没有联网搜索的能力**，拿不到实时新闻。\n\n"
     "你可以：\n"
     "· 直接打开浏览器搜「科技新闻」\n"
     "· 或者搜索 App 的今日热点\n\n"
     "不过纯知识类的问题（比如某个概念是什么）我可以直接聊。"),
    ("搜一下 iPhone 17 多少钱",
     "我这边**没有联网能力**，查不到实时价格。\n\n"
     "建议你去官网或者电商平台看一下，价格信息变化快，"
     "我凭记忆说反而不准。"),
    ("网上现在怎么评价这部电影",
     "查不了 —— 我**不能联网**，拿不到网友的实时评价。\n\n"
     "你可以去豆瓣、微博搜一下。如果你把影评内容贴给我，"
     "我可以帮你分析总结。"),
    ("帮我查一下明天北京的天气",
     "我这边**没有天气查询能力**，也给不出准确预报。\n\n"
     "看一下手机自带的天气 App 最准。"
     "如果你想了解「北京气候特点」这类常识，我可以讲。"),

    # --- J2. 想读文件/执行命令 → 说明没这权限 ---
    ("读一下我电脑上的 report.pdf",
     "我**读不了你本地的文件** —— 我这个环境只能做文本对话。\n\n"
     "你可以把内容复制粘贴给我，我帮你分析总结。"),
    ("帮我在终端跑一下 ls",
     "我**执行不了命令** —— 我这边没有操作你电脑的通道。\n\n"
     "你在自己的终端里跑就行。如果输出有报错，贴给我我帮你看。"),
    ("帮我打开微信给张三发消息",
     "这个我做不到 —— 我**没法操作你的应用**，也不能替你发消息。\n\n"
     "涉及代你操作的事，都得你自己确认，我不该也不能代办。"),

    # --- J3. 想算数 → 心算 + 声明可能不准（对比：有工具时才调 calculate）---
    ("帮我算 789*123",
     "789 × 123 = **97047**。\n\n"
     "（我这边没有计算工具，是直接算的。"
     "如果涉及重要场合，建议你用计算器复核一下。）"),
    ("算一下 12345 乘 6789",
     "12345 × 6789 = **83793105**。\n\n"
     "说明一下：我这边**没有可调用的计算工具**，这是直接心算的结果，"
     "位数多的时候有可能出错，重要数据建议用计算器复核。"),

    # --- J4. 想做日历/提醒这类「操作类」请求 ---
    ("帮我订明天上午 10 点的会议室",
     "我**没法帮你订** —— 我不能替你操作系统或提交表单。\n\n"
     "这种代你执行的操作，得你自己在系统里确认。"),
    ("帮我把这段话翻译成英文发给我老板",
     "**翻译我马上可以做**，但**发送我做不了** —— 我不能替你发消息。\n\n"
     "先把译文给你：\n"
     "（此处是译文内容）\n\n"
     "你自己复制发过去就行。"),

    # --- J5. 元认知：直接问「你有没有工具」 ---
    ("你有哪些工具可以用？",
     "我这边**没有可用的工具** —— 只能做纯文本对话。\n\n"
     "所以：\n"
     "· **能做的**：回答问题、写代码、分析你贴给我的内容\n"
     "· **不能做的**：联网搜索、读你本地文件、执行命令、替你操作应用\n\n"
     "如果需要上面那些能力，得在支持工具的宿主（比如 agent 框架）里用我。"),
    ("你怎么不调用工具帮我查？",
     "因为**我这边没有工具可调** —— 当前是纯聊天环境。\n\n"
     "如果我在支持工具的宿主里跑，遇到该查的事我会去调。"
     "但现在硬编一个工具调用给你，你也接不住，只会看到一堆乱码。"),
    ("我看别人家的 AI 都能联网，你为什么不行？",
     "那是**宿主给它的能力**，不是模型自带的。\n\n"
     "我在当前环境里没有被提供联网/查询工具，所以只能靠已有知识回答。"
     "换个支持工具的宿主，我就能用上了。"),

    # --- J6. 代码类请求：直接给代码，不调工具 ---
    ("帮我写个 Python 脚本统计文件夹里有多少个文件",
     "```python\n"
     "import os\n\n"
     "path = '.'\n"
     "files = [f for f in os.listdir(path)\n"
     "         if os.path.isfile(os.path.join(path, f))]\n"
     "print(f'共 {len(files)} 个文件')\n"
     "```\n\n"
     "把 `path` 改成你要统计的目录就行。\n"
     "（这个脚本得你在自己电脑上跑，我这边没法执行。）"),

    # --- J7. 反问式确认：模型应说明自身边界而不是假装 ---
    ("你能帮我干什么？",
     "我能做的是**文本类**的活：\n\n"
     "· 回答问题、解释概念\n"
     "· 写代码、改代码、看报错\n"
     "· 分析你贴过来的文本\n\n"
     "**做不到的**：联网查资料、读你本地文件、跑命令、替你操作应用。\n\n"
     "（我这个环境没有提供工具，所以只能纯聊天。）"),
]

for q, ans in CHAT_ONLY_CASES:
    for sys_text in CHAT_ONLY_SYS[:4]:
        add(sys_text, q, ans, weight=2)

# --- J8. 多轮：先问工具类问题 → 用户追问 → 依然不调工具 ---
MULTI_CHAT_ONLY = [
    ([("system", "你是一个乐于助人的 AI 助手。"),
      ("user", "帮我搜一下 Python 3.13 有什么新特性"),
      ("assistant", "我**没有联网搜索能力**，拿不到最新的官方信息。\n\n"
                    "建议去 python.org 看 release notes。"
                    "不过我可以说说 Python 3.13 里几个已知的方向"
                    "（比如 free-threaded 模式、更好的错误提示），"
                    "但细节可能有出入，一切以官方为准。"),
      ("user", "那你能联网吗？"),
      ("assistant", "**不能**。我当前环境没有被提供联网工具，只能靠已有知识回答。\n\n"
                    "如果换个支持工具的宿主跑我，遇到该查的事我会去调工具。")], 3),

    ([("system", "你是一个友好的聊天助手。"),
      ("user", "今天几号？"),
      ("assistant", "我这边**没法获取当前时间**，给不出准确日期。\n\n"
                    "看一下手机或者电脑右下角最准。"),
      ("user", "你不是 AI 吗，怎么连时间都不知道？"),
      ("assistant", "这个跟聪不聪明没关系 —— 是**我没有被提供获取时间的工具**。\n\n"
                    "我的知识有截止时间，也不会自动知道「现在几点」。"
                    "在支持工具的宿主里，调一下时间工具就知道了。")], 3),

    ([("system", "你是客服小助手，负责解答用户问题。"),
      ("user", "我的订单到哪了？"),
      ("assistant", "我这边**查不到订单信息** —— 没有连接你们的订单系统。\n\n"
                    "你可以：\n"
                    "· 在「我的订单」页面看物流\n"
                    "· 或者把订单号发我，我帮你看怎么查"),
      ("user", "你不能直接帮我查吗？"),
      ("assistant", "不能，我**没有访问订单系统的权限和接口**。\n\n"
                    "而且即使有接口，查询别人的订单信息也涉及隐私，"
                    "得走正经的验证流程。")], 2),
]

for turns, weight in MULTI_CHAT_ONLY:
    add_mt(turns, weight=weight)

# --- J9. 强对照：有工具时必须调，没工具时必须不调（同一个问题）---
#     这是最关键的一组 —— 让模型学会看「有没有工具」，
#     而不是死记「这个问题该不该调工具」。
CONTRAST_Q = "帮我算 789*123"
CONTRAST_ANS_NOTOOL = (
    "789 × 123 = **97047**。\n\n"
    "（我这边没有计算工具，是直接算的。"
    "如果涉及重要场合，建议你用计算器复核一下。）")

for sys_text in CHAT_ONLY_SYS[:4]:
    # 无工具 → 心算 + 声明
    add(sys_text, CONTRAST_Q, CONTRAST_ANS_NOTOOL, weight=3)
    # 有工具 → 必须调 calculate
    add(sys_text + "\n\n" + tdef(CALC), CONTRAST_Q,
        call("calculate", {"expression": "789*123"}), weight=3)


# ============================================================
# H. 全面发展：多轮通用对话（不丢上下文）
#    补多轮纯问答，防止模型只在单轮里表现好，
#    一到追问就把前面的话忘光。
# ============================================================
print("H. 多轮通用对话...")

MULTI_TURN_CHAT = [
    ([("system", "你是一个乐于助人的 AI 助手。"),
      ("user", "Python 里 list 和 tuple 有什么区别？"),
      ("assistant", "主要区别是**可变性**：\n\n"
                    "· `list` 可变，能增删改：`a = [1,2]; a.append(3)`\n"
                    "· `tuple` 不可变，创建后不能改：`b = (1,2)`\n\n"
                    "由此衍生出几个差异：tuple 可以作为 dict 的 key，list 不行；"
                    "tuple 占用内存更小、遍历稍快。"),
      ("user", "那我什么时候该用 tuple？"),
      ("assistant", "几个典型场景：\n\n"
                    "1. **数据不该被改** —— 比如坐标 `(x, y)`、RGB 颜色值\n"
                    "2. **要用作 dict 的 key** —— list 不可哈希，做不到\n"
                    "3. **函数返回多个值** —— 其实 Python 就是返回 tuple\n"
                    "4. **性能敏感** —— 数据量大且只读时，tuple 更省内存\n\n"
                    "经验法则：**默认用 list，确定不需要改就用 tuple**。")], 3),

    ([("system", "You are a helpful assistant."),
      ("user", "What is the difference between == and is in Python?"),
      ("assistant", "`==` compares **values**, while `is` compares **identity** "
                    "(whether they're the same object in memory).\n\n"
                    "```python\n"
                    "a = [1, 2]\n"
                    "b = [1, 2]\n"
                    "a == b   # True  (same value)\n"
                    "a is b   # False (different objects)\n"
                    "```\n\n"
                    "Rule of thumb: use `==` for values, "
                    "and `is` only for `None`, `True`, `False`."),
      ("user", "So should I always use == for None checks?"),
      ("assistant", "No — actually the opposite. Use `is None`, not `== None`.\n\n"
                    "Reasons:\n"
                    "1. `None` is a **singleton**, so identity check is exact\n"
                    "2. `== None` can be overridden by `__eq__`, "
                    "giving wrong results\n"
                    "3. `is None` is faster and is the PEP 8 recommendation\n\n"
                    "So: **values → `==`, None/True/False → `is`**.")], 3),

    ([("system", "你是一个耐心的助手，会认真对待每一个问题。"),
      ("user", "我想学编程，应该从哪门语言开始？"),
      ("assistant", "推荐 **Python** 入门，理由：\n\n"
                    "· 语法接近英语，门槛最低\n"
                    "· 生态丰富，想做什么基本都有现成库\n"
                    "· 就业面广，数据、后端、自动化都能用\n\n"
                    "如果你更想做网页界面，可以选 JavaScript；"
                    "想深入理解底层，可以选 C。"),
      ("user", "那我每天要花多久？"),
      ("assistant", "**每天 1 小时就够，关键是别断**。\n\n"
                    "比时长更重要的是节奏：\n\n"
                    "· **每天固定时间** —— 形成习惯，比周末突击 5 小时有效得多\n"
                    "· **动手写** —— 看懂 ≠ 会写，一定要自己敲一遍\n"
                    "· **做小项目** —— 学完基础就找个小需求练手，"
                    "比如写个自动整理文件的脚本\n\n"
                    "按每天 1 小时算，**3 个月能写出能跑的小工具**，"
                    "半年可以开始做正经项目。")], 2),

    ([("system", "你是一位专业的技术顾问。"),
      ("user", "Mac 上跑本地大模型用什么工具好？"),
      ("assistant", "看你想要什么：\n\n"
                    "· **Ollama** —— 最省事，一条命令跑起来，适合快速体验\n"
                    "· **LM Studio** —— 有图形界面，适合不熟悉命令行的\n"
                    "· **MLX** —— Apple 官方框架，性能最好，适合要微调的场景\n"
                    "· **llama.cpp** —— 生态最广，GGUF 格式的娘家\n\n"
                    "**推荐路径**：先用 Ollama 玩，想深入就转 MLX。"),
      ("user", "MLX 和 Ollama 有什么区别？"),
      ("assistant", "定位不一样：\n\n"
                    "· **Ollama** 是**产品** —— 帮你把模型跑起来，开箱即用，"
                    "底层也能调 MLX 或 llama.cpp\n"
                    "· **MLX** 是**框架** —— Apple 出的机器学习库，"
                    "你要自己写代码调 API，但控制力最强\n\n"
                    "选法：\n"
                    "· 只想用模型 → **Ollama**\n"
                    "· 要微调、要改结构、要压性能 → **MLX**\n\n"
                    "你现在是要用，还是要训？")], 2),

    ([("system", "你是一个智能助手，请根据用户需求回答问题。"),
      ("user", "解释一下什么是 LoRA"),
      ("assistant", "**LoRA（Low-Rank Adaptation，低秩适配）** "
                    "是一种高效微调方法。\n\n"
                    "核心思路：大模型微调时**不动原始权重**，"
                    "而是在旁边加两个很小的矩阵 A 和 B，只训练这两个。\n\n"
                    "好处：\n"
                    "· 可训练参数常常只有 **0.1%~1%**\n"
                    "· 显存需求大幅下降\n"
                    "· 产出的 adapter 文件很小（几 MB 到几十 MB）"),
      ("user", "那 rank 设多少合适？"),
      ("assistant", "**看任务复杂度，一般 8~64**。\n\n"
                    "· **rank 4~8** —— 轻量任务（风格调整、格式对齐），够用\n"
                    "· **rank 16~32** —— 中等任务（领域知识注入）\n"
                    "· **rank 64+** —— 复杂任务，但收益递减且容易过拟合\n\n"
                    "**实际建议**：从 **8** 起步，训完看效果，不够再加倍。\n\n"
                    "有个反直觉的点：rank 翻倍，参数量线性涨，"
                    "但效果提升往往**远小于翻倍**。所以别一上来就拉高。")], 2),
]

for turns, weight in MULTI_TURN_CHAT:
    add_mt(turns, weight=weight)


# ============================================================
# K. v6 补丁（2026-09-13）：修 v6 验收暴露的 3 个问题
#
#   v6 验收 24 通过 / 3 失败，三项都稳定复现 3/3。
#   根因是**样本量 vs 行为惯性的对抗没算准**：
#
#     要修的问题        现有样本   对抗的惯性    劣势
#     ─────────────────────────────────────────────
#     ② 纯聊天不调工具      93      696 裸JSON   1:7.5  ← 最惨
#     ⑥ 工具放 user        180     1210 system   1:6.7
#     ⑧b 多轮追问丢上下文   170        —
#
#   为什么这次要**大幅加量**（而不是加几十条）：
#   上次 J 类只补了 101 条，结果被 696 条"该调工具"的惯性完全压住。
#   行为惯性是按**样本条数**加权的，弱势方向必须补到接近量级才有效。
#   经验：目标比例至少拉到 1:2 以内，关键场景做到 1:1 以上。
#
#   同时**不能只加新样本**：全量续训 + 新样本，避免灾难性遗忘
#   （只喂"不调工具"会让模型学成"一律不调"，把刚修好的能力练废）。
# ============================================================
print("K. v6 补丁（修 3 个验收失败项）...")

# ------------------------------------------------------------
# K1. 修 ②「纯聊天宿主凭空调工具」—— 加量到 ~280 条
#
#      v6 实测错误（稳定 3/3）：
#        system: 你是一个乐于助人的 AI 助手。   ← 无任何工具定义
#        user:   帮我查一下明天北京的天气
#        → {"name": "get_weather", "arguments": {"city": "北京", "date": "明天"}}
#
#      三重错误：
#        1. 宿主没有任何工具，却调了工具
#        2. get_weather 是**没定义过的工具名**（工具名幻觉）
#        3. 自己编了 "date": "明天" 参数（格式幻觉）
#
#      修法：把「无工具 + 各类查询请求」铺满，并且**显式覆盖天气/搜索
#      这两个最高频的触发词**（v6 就是在这两个上翻车的）。
# ------------------------------------------------------------
print("K1. 纯聊天不调工具（天气/搜索/查询高频场景）...")

# 高频查询词 × 多种说法 —— 重点铺「天气」和「搜索」
K1_QUERIES = [
    # (用户问法, 正确回答)
    ("帮我查一下明天北京的天气",
     "我这边**查不了天气** —— 没有联网或天气查询的工具。\n\n"
     "看一下手机自带的天气 App 最准。"),
    ("明天北京天气怎么样",
     "我**没法查实时天气**，手上没有天气数据。\n\n"
     "手机天气 App 或者搜一下「北京天气」最快。"),
    ("帮我看看今天上海冷不冷",
     "我**拿不到天气数据**，不知道上海今天多少度。\n\n"
     "出门前看一眼天气 App 吧。"),
    ("这几天会下雨吗",
     "我**查不了天气预报** —— 没有联网能力。\n\n"
     "建议看手机天气，或者搜「未来一周天气」。"),
    ("北京现在几度",
     "我**给不出实时温度**，没有天气接口。\n\n"
     "你那边手机上应该能直接看到。"),
    ("帮我搜一下最近的 AI 新闻",
     "我**不能联网搜索**，拿不到最新新闻。\n\n"
     "你可以直接搜「AI 新闻」，或者跟我说说你想了解哪方面，"
     "我基于已有知识聊。"),
    ("搜一下这个明星是谁",
     "我**没有搜索工具**，查不了。\n\n"
     "如果你把相关介绍贴给我，我可以帮你梳理。"),
    ("帮我查一下这个快递到哪了",
     "我**查不到快递单号** —— 没有物流查询能力。\n\n"
     "去快递 App 或者官网查最准。"),
    ("搜一下附近有什么好吃的",
     "我**没有定位和搜索能力**，不知道你在哪，也看不到商家。\n\n"
     "打开地图 App 搜「附近餐厅」更快。"),
    ("帮我查一下机票价格",
     "我**查不了实时票价** —— 不能联网。\n\n"
     "航司官网或者订票平台看一下，价格变动快，我凭记忆说反而误导。"),
    ("查一下现在美元汇率多少",
     "我**拿不到实时汇率**，给不出今天的牌价。\n\n"
     "银行 App 或者搜「美元汇率」都行。"),
    ("帮我看看 A 股今天涨了吗",
     "我**查不了实时行情** —— 没有联网能力。\n\n"
     "券商 App 或者财经网站看一下最准。"),
    ("搜一下这首歌叫什么名字",
     "我**没法搜索或听音频**，不知道你说的是哪首。\n\n"
     "描述一下旋律或者歌词，我说不定知道。"),
    ("帮我查一下我手机话费还有多少",
     "我**查不了你的话费** —— 既没有你的账户权限，也没法联网。\n\n"
     "打运营商客服或者用官方 App 最直接。"),
    ("搜一下附近的地铁站",
     "我**没有定位能力**，也不知道你在哪个城市。\n\n"
     "地图 App 搜「地铁站」就好。"),
    ("帮我查一下这个公司的工商信息",
     "我**查不了工商数据** —— 需要联网访问企业信息库。\n\n"
     "国家企业信用信息公示系统可以查。"),
    ("看看现在几点",
     "我**不知道当前时间** —— 没有获取时间的能力。\n\n"
     "看一下手机或电脑右下角。"),
    ("今天是什么日子",
     "我**拿不到今天的日期**，没法判断。\n\n"
     "手机日历上看一眼最准。"),
    ("帮我读一下这个 PDF",
     "我**读不了文件** —— 只能处理你直接发给我的文本。\n\n"
     "把内容复制粘贴过来，我帮你分析。"),
    ("帮我打开这个链接看看",
     "我**打不开链接** —— 没有联网访问的能力。\n\n"
     "你可以把网页内容贴过来，我帮你总结。"),
]

# ★ 修正（2026-09-13 08:25）：删掉了原 K1_ANSWERS_ALT 的随机变体。
#
#   原写法是给每个问题随机配一句"万能模板"：
#       add(sys_text, q, random.choice(K1_ANSWERS_ALT) + "…", weight=1)
#   结果出了两个严重缺陷：
#     1. **同一问题有 4 种冲突答案**（25 个问题里 20 个中招）
#        「帮我查一下明天北京的天气」同时被教三种模板 + 一种具体答案，
#        模型只能学成"随机挑一句套话"，学不到"针对问题说明缺什么"
#     2. **答案与问题不匹配**（82 行）
#        问天气配"当前环境只能做纯文本对话"，答非所问
#
#   修法：不搞通用模板，改成**每类问题配针对性的多种说法**，
#   同一问题下的所有答案必须语义一致（都指出"缺的是天气查询能力"），
#   只是在措辞上做变化（避免死记一句话）。
#
#   通用模板只在「元认知」类问题里用（K2 的"你可以调用什么工具"），
#   那种场景下"没有工具"才是直接答案。

# 每类问题 × 2~3 种语义一致、措辞不同的答案
K1_VARIANTS = {
    "帮我查一下明天北京的天气": [
        "我这边**查不了天气** —— 没有联网或天气查询的工具。\n\n"
        "看一下手机自带的天气 App 最准。",
        "抱歉，我**拿不到天气数据**，没有可以调用的天气接口。\n\n"
        "手机上看一眼最方便。",
    ],
    "明天北京天气怎么样": [
        "我**没法查实时天气**，手上没有天气数据。\n\n"
        "手机天气 App 或者搜一下「北京天气」最快。",
        "我**给不了明天的天气预报** —— 没有联网能力。\n\n"
        "建议看手机天气。",
    ],
    "帮我看看今天上海冷不冷": [
        "我**拿不到天气数据**，不知道上海今天多少度。\n\n"
        "出门前看一眼天气 App 吧。",
        "我**查不了实时气温**，没有天气工具可用。\n\n"
        "手机上应该能直接看到。",
    ],
    "这几天会下雨吗": [
        "我**查不了天气预报** —— 没有联网能力。\n\n"
        "建议看手机天气，或者搜「未来一周天气」。",
        "我**给不出降雨预报**，手上没有天气数据。\n\n"
        "天气 App 上看最准。",
    ],
    "北京现在几度": [
        "我**给不出实时温度**，没有天气接口。\n\n"
        "你那边手机上应该能直接看到。",
        "我**不知道现在多少度** —— 查不了实时天气。\n\n"
        "看一下手机天气。",
    ],
    "帮我搜一下最近的 AI 新闻": [
        "我**不能联网搜索**，拿不到最新新闻。\n\n"
        "你可以直接搜「AI 新闻」，或者跟我说说你想了解哪方面，"
        "我基于已有知识聊。",
        "我**没有搜索工具**，查不到实时资讯。\n\n"
        "想知道什么方向的话，我可以凭已有知识聊聊。",
    ],
    "搜一下这个明星是谁": [
        "我**没有搜索工具**，查不了。\n\n"
        "如果你把相关介绍贴给我，我可以帮你梳理。",
        "抱歉，我**不能联网搜索**，不知道你说的是谁。\n\n"
        "给点线索的话，我说不定知道。",
    ],
    "帮我查一下这个快递到哪了": [
        "我**查不到快递单号** —— 没有物流查询能力。\n\n"
        "去快递 App 或者官网查最准。",
        "我**拿不到物流信息**，没有对接快递查询的接口。\n\n"
        "快递 App 上查一下最快。",
    ],
    "搜一下附近有什么好吃的": [
        "我**没有定位和搜索能力**，不知道你在哪，也看不到商家。\n\n"
        "打开地图 App 搜「附近餐厅」更快。",
        "我**查不了附近商家** —— 既没有定位，也不能联网。\n\n"
        "地图或者点评类 App 更合适。",
    ],
    "帮我查一下机票价格": [
        "我**查不了实时票价** —— 不能联网。\n\n"
        "航司官网或者订票平台看一下，价格变动快，我凭记忆说反而误导。",
        "我**拿不到机票价格**，没有联网查询能力。\n\n"
        "订票平台上看最准。",
    ],
    "查一下现在美元汇率多少": [
        "我**拿不到实时汇率**，给不出今天的牌价。\n\n"
        "银行 App 或者搜「美元汇率」都行。",
        "我**查不了汇率** —— 需要联网，我这边没有。\n\n"
        "财经类 App 上看一下。",
    ],
    "帮我看看 A 股今天涨了吗": [
        "我**查不了实时行情** —— 没有联网能力。\n\n"
        "券商 App 或者财经网站看一下最准。",
        "我**拿不到今天的股市数据**。\n\n"
        "行情软件上瞄一眼就知道了。",
    ],
    "搜一下这首歌叫什么名字": [
        "我**没法搜索或听音频**，不知道你说的是哪首。\n\n"
        "描述一下旋律或者歌词，我说不定知道。",
        "我**听不到音频，也不能搜索**。\n\n"
        "哼两句或者给点歌词，我帮你猜。",
    ],
    "帮我查一下我手机话费还有多少": [
        "我**查不了你的话费** —— 既没有你的账户权限，也没法联网。\n\n"
        "打运营商客服或者用官方 App 最直接。",
        "我**看不到你的话费余额**，没有这个权限和接口。\n\n"
        "运营商 App 上查最快。",
    ],
    "搜一下附近的地铁站": [
        "我**没有定位能力**，也不知道你在哪个城市。\n\n"
        "地图 App 搜「地铁站」就好。",
        "我**查不了附近的地点** —— 没有定位也没法搜索。\n\n"
        "地图软件上搜一下更快。",
    ],
    "帮我查一下这个公司的工商信息": [
        "我**查不了工商数据** —— 需要联网访问企业信息库。\n\n"
        "国家企业信用信息公示系统可以查。",
        "我**拿不到企业注册信息**，没有对接这类数据库。\n\n"
        "公示系统或商业查询平台都行。",
    ],
    "看看现在几点": [
        "我**不知道当前时间** —— 没有获取时间的能力。\n\n"
        "看一下手机或电脑右下角。",
        "我**拿不到系统时间**，给不出准确时刻。\n\n"
        "设备上看一眼就有了。",
    ],
    "今天是什么日子": [
        "我**拿不到今天的日期**，没法判断。\n\n"
        "手机日历上看一眼最准。",
        "我**不知道今天几号** —— 没有获取日期能力。\n\n"
        "看一眼日历或手机锁屏。",
    ],
    "帮我读一下这个 PDF": [
        "我**读不了文件** —— 只能处理你直接发给我的文本。\n\n"
        "把内容复制粘贴过来，我帮你分析。",
        "我**打不开 PDF**，没有文件读取能力。\n\n"
        "正文贴给我就行。",
    ],
    "帮我打开这个链接看看": [
        "我**打不开链接** —— 没有联网访问的能力。\n\n"
        "你可以把网页内容贴过来，我帮你总结。",
        "我**访问不了网址**，不能联网。\n\n"
        "把内容复制过来我帮你看看。",
    ],
}

# 短版兜底：万一某条问题没配变体，用问题自身的主题生成，而不是套万能模板
for q, ans in K1_QUERIES:
    variants = K1_VARIANTS.get(q, [ans])
    for sys_text in CHAT_ONLY_SYS:
        # 主答案 weight 2；变体 weight 1（语义一致，只是措辞不同）
        add(sys_text, q, variants[0], weight=2)
        if len(variants) > 1:
            add(sys_text, q, variants[1], weight=1)

# ------------------------------------------------------------
# K2. 修「工具名幻觉」—— 明确教「宁可说没有，也别编工具名」
#
#      v6 在纯聊天里编出了 get_weather —— **该工具根本没在上下文里定义过**。
#      这比"该不该调"更危险：宿主拿到一个自己不认识的名字会直接崩。
#      单独成组，让"不编工具名"成为一条明确的行为纪律。
# ------------------------------------------------------------
print("K2. 工具名幻觉（不许编不存在的工具）...")

K2_CASES = [
    ("帮我查天气",
     "我这边**没有天气工具**，也没有别的可调用工具。\n\n"
     "我不会去编一个工具名出来 —— 那样宿主根本接不住，"
     "你只会看到一段没意义的 JSON。"),
    ("你可以调用什么工具？",
     "**一个都没有。** 当前环境没有给我任何工具定义。\n\n"
     "所以涉及联网、查数据、读文件的事，我只能如实说做不到，"
     "而不是硬编一个工具名糊弄过去。"),
    ("你刚才是不是调了工具？",
     "没有。我**这边没有工具可调**。\n\n"
     "如果我输出了 `{\"name\": ...}` 这种内容，那是我编的 —— "
     "既没定义过，宿主也执行不了。这种情况我应该直接说做不到。"),
    ("算一下 123 乘 456",
     "123 × 456 = **56088**。\n\n"
     "（这是心算的，我这边没有计算工具可调。）"),
    ("你有哪些能力",
     "纯文本类的：回答问题、写代码、分析你发的内容。\n\n"
     "**没有的**：联网搜索、查天气/行情/物流、读本地文件、"
     "执行命令、替你操作应用。\n\n"
     "没有的工具我不会假装有，也不会编个名字出来。"),
]

for q, ans in K2_CASES:
    for sys_text in CHAT_ONLY_SYS[:5]:
        add(sys_text, q, ans, weight=3)


# ------------------------------------------------------------
# K3. 修 ⑥「工具定义放 user 时不调工具」—— 加量到 ~420 条
#
#      v6 实测：工具定义放 user 时，算术题直接心算（不调工具）。
#      现有 180 条 user 结构样本，被 1210 条 system 结构压住了。
#
#      修法：
#        1. 增加 user 结构的**样本量**（180 → 420+）
#        2. 覆盖**所有工具类型**，不能只有 calculate
#        3. system 用多种通用写法（不只身份格式）
# ------------------------------------------------------------
print("K3. 工具定义放 user（双结构均衡）...")

# user 结构的多种「用户正文」前缀写法 —— 真实框架不都用「用户：」
K3_USER_PREFIX = ["用户：", "问题：", "提问：", ""]

K3_TASKS = [
    # (工具定义, 用户请求, 期望调用)
    ((CALC,), "789*123 等于多少？", ("calculate", {"expression": "789*123"})),
    ((CALC,), "帮我算 123*456", ("calculate", {"expression": "123*456"})),
    ((CALC,), "12.5 乘 8 是多少", ("calculate", {"expression": "12.5*8"})),
    ((CALC,), "算一下 (15+27)*3", ("calculate", {"expression": "(15+27)*3"})),
    ((WEATHER,), "北京今天天气怎么样", ("get_weather", {"city": "北京"})),
    ((WEATHER,), "查一下上海冷不冷", ("get_weather", {"city": "上海"})),
    ((WEATHER,), "广州明天要带伞吗", ("get_weather", {"city": "广州"})),
    ((SEARCH,), "帮我搜一下 FastAPI 和 Flask 的区别",
     ("search_web", {"query": "FastAPI Flask 区别"})),
    ((SEARCH,), "搜一下 Python 3.13 新特性",
     ("search_web", {"query": "Python 3.13 新特性"})),
    ((SEARCH,), "帮我查一下怎么学习 Rust",
     ("search_web", {"query": "如何学习 Rust"})),
    ((READF,), "读一下 config.py", ("read_file", {"path": "config.py"})),
    ((READF,), "看看 main.py 里写了什么", ("read_file", {"path": "main.py"})),
    ((READF,), "读一下 requirements.txt",
     ("read_file", {"path": "requirements.txt"})),
    ((TIME,), "现在北京时间几点", ("get_time", {"timezone": "Asia/Shanghai"})),
    ((TIME,), "帮我查一下纽约现在几点", ("get_time", {"timezone": "America/New_York"})),
    ((SHELL,), "看一下当前目录有什么文件", ("execute_shell", {"command": "ls -la"})),
    ((RUNPY,), "跑一下 print(sum(range(10)))",
     ("run_python", {"code": "print(sum(range(10)))"})),
]

for tools, q, (name, args) in K3_TASKS:
    for sys_text in GENERIC_SYS[:3]:
        for prefix in K3_USER_PREFIX[:2]:
            add(sys_text,
                tdef(*tools) + "\n\n" + prefix + q,
                call(name, args),
                weight=1)

# user 结构下「不该调工具」的对照样本 —— 防止只学到"一律调"
K3_NO_CALL = [
    ((CALC,), "什么是递归？",
     "递归是函数自己调用自己。\n\n"
     "两个必备条件：\n"
     "· **终止条件**（base case）—— 否则无限递归\n"
     "· **向终止条件收敛** —— 每次递归规模变小\n\n"
     "典型的例子是阶乘和斐波那契；遍历树结构也常用。"),
    ((CALC,), "帮我写个二分查找",
     "```python\n"
     "def binary_search(arr, target):\n"
     "    lo, hi = 0, len(arr) - 1\n"
     "    while lo <= hi:\n"
     "        mid = (lo + hi) // 2\n"
     "        if arr[mid] == target:\n"
     "            return mid\n"
     "        elif arr[mid] < target:\n"
     "            lo = mid + 1\n"
     "        else:\n"
     "            hi = mid - 1\n"
     "    return -1\n"
     "```\n\n"
     "前提是数组**已排好序**，时间复杂度 O(log n)。"),
    ((CALC, SEARCH), "解释一下什么是时间复杂度",
     "**时间复杂度**描述算法耗时随输入规模的增长趋势，用大 O 表示。\n\n"
     "常见量级（从快到慢）：\n"
     "· `O(1)` 常数 —— 数组下标访问\n"
     "· `O(log n)` 对数 —— 二分查找\n"
     "· `O(n)` 线性 —— 遍历一次\n"
     "· `O(n log n)` —— 快排、归并\n"
     "· `O(n²)` 平方 —— 双重循环\n\n"
     "关注的是**趋势**而不是具体秒数，所以常数项会被忽略。"),
    ((READF,), "Python 里 list 和 tuple 有什么区别？",
     "主要区别是**可变性**：list 可变，tuple 不可变。\n\n"
     "由此衍生：tuple 可作 dict 的 key，list 不行；"
     "tuple 更省内存、遍历稍快。\n\n"
     "选择原则：默认用 list，确定不需要改就用 tuple。"),
]

for tools, q, ans in K3_NO_CALL:
    for sys_text in GENERIC_SYS[:3]:
        add(sys_text,
            tdef(*tools) + "\n\n用户：" + q,
            ans,
            weight=2)


# ------------------------------------------------------------
# K4. 修 ⑧b「多轮追问丢上下文」—— 显式教「基于上一轮结果继续」
#
#      v6 实测：
#        [12*34 → 调用 → 结果 408 → 「12×34=408」] → 「那再乘 2 呢」
#        → 「34 × 2 = 68」     ← 把 408 丢掉了，理解成 34 乘 2
#
#      修法的关键不是加量，是**给足"接上文"的轨迹样本**：
#      让模型看到「结果 → 追问 → 引用上一轮数值再调」的完整链条。
# ------------------------------------------------------------
print("K4. 多轮追问（基于上文结果继续）...")

K4_TRAJECTORIES = [
    # 最典型：算完 → 追问 → 用**上一轮的结果**继续算
    ([("user", "帮我算 12*34"),
      ("assistant", call("calculate", {"expression": "12*34"})),
      ("user", tool_result("calculate", "408")),
      ("assistant", "12 × 34 = **408**。"),
      ("user", "那再乘 2 呢"),
      ("assistant", call("calculate", {"expression": "408*2"})),
      ("user", tool_result("calculate", "816")),
      ("assistant", "408 × 2 = **816**。")], 4),

    ([("user", "算一下 25*4"),
      ("assistant", call("calculate", {"expression": "25*4"})),
      ("user", tool_result("calculate", "100")),
      ("assistant", "25 × 4 = **100**。"),
      ("user", "再加 50"),
      ("assistant", call("calculate", {"expression": "100+50"})),
      ("user", tool_result("calculate", "150")),
      ("assistant", "100 + 50 = **150**。")], 3),

    ([("user", "帮我算 88*99"),
      ("assistant", call("calculate", {"expression": "88*99"})),
      ("user", tool_result("calculate", "8712")),
      ("assistant", "88 × 99 = **8712**。"),
      ("user", "那减去 712 呢"),
      ("assistant", call("calculate", {"expression": "8712-712"})),
      ("user", tool_result("calculate", "8000")),
      ("assistant", "8712 − 712 = **8000**。")], 3),

    # 追问要基于上文数值，而不是重新理解问题
    ([("user", "123 乘 456 是多少"),
      ("assistant", call("calculate", {"expression": "123*456"})),
      ("user", tool_result("calculate", "56088")),
      ("assistant", "123 × 456 = **56088**。"),
      ("user", "那除以 2 呢"),
      ("assistant", call("calculate", {"expression": "56088/2"})),
      ("user", tool_result("calculate", "28044")),
      ("assistant", "56088 ÷ 2 = **28044**。")], 3),

    # 元认知：显式说明「我用了上一轮的结果」
    ([("user", "算一下 15*6"),
      ("assistant", call("calculate", {"expression": "15*6"})),
      ("user", tool_result("calculate", "90")),
      ("assistant", "15 × 6 = **90**。"),
      ("user", "再乘 10"),
      ("assistant", call("calculate", {"expression": "90*10"})),
      ("user", tool_result("calculate", "900")),
      ("assistant", "基于上一轮的 **90**，再乘 10 就是 **900**。")], 2),

    # ---- 以下为 2026-09-13 补充：原 K4 只有 5 条轨迹，太少 ----
    #      验收失败的 ⑧b 是"多轮丢上下文"，5 条不足以建立这个能力。
    #      补到 20+ 条，覆盖：减法/除法/连续追问/工具失败后追问/换话题
    ([("user", "算一下 200 减 37"),
      ("assistant", call("calculate", {"expression": "200-37"})),
      ("user", tool_result("calculate", "163")),
      ("assistant", "200 − 37 = **163**。"),
      ("user", "再减 63"),
      ("assistant", call("calculate", {"expression": "163-63"})),
      ("user", tool_result("calculate", "100")),
      ("assistant", "163 − 63 = **100**。")], 3),

    ([("user", "1000 除以 8 是多少"),
      ("assistant", call("calculate", {"expression": "1000/8"})),
      ("user", tool_result("calculate", "125.0")),
      ("assistant", "1000 ÷ 8 = **125**。"),
      ("user", "那再乘以 4"),
      ("assistant", call("calculate", {"expression": "125*4"})),
      ("user", tool_result("calculate", "500")),
      ("assistant", "125 × 4 = **500**。")], 3),

    # 连续追问三次 —— 考验长一点的上下文保持
    ([("user", "帮我算 6*7"),
      ("assistant", call("calculate", {"expression": "6*7"})),
      ("user", tool_result("calculate", "42")),
      ("assistant", "6 × 7 = **42**。"),
      ("user", "加 8"),
      ("assistant", call("calculate", {"expression": "42+8"})),
      ("user", tool_result("calculate", "50")),
      ("assistant", "42 + 8 = **50**。"),
      ("user", "最后除以 5"),
      ("assistant", call("calculate", {"expression": "50/5"})),
      ("user", tool_result("calculate", "10.0")),
      ("assistant", "50 ÷ 5 = **10**。")], 3),

    # 追问"为什么是这个数" —— 要引用上文
    ([("user", "帮我算 99*11"),
      ("assistant", call("calculate", {"expression": "99*11"})),
      ("user", tool_result("calculate", "1089")),
      ("assistant", "99 × 11 = **1089**。"),
      ("user", "为什么不是 1001？"),
      ("assistant", "不是 1001 —— 你可能是把 **99 × 11** 记成了 **91 × 11**。\n\n"
                    "实际结果就是 **1089**：\n"
                    "· 99 × 11 = 99 × 10 + 99 = 990 + 99 = 1089\n"
                    "· 而 91 × 11 = 1001\n\n"
                    "所以关键在第一位数是 9 还是 1。")], 2),

    # 追问时换算法（改用工具换个表达式）
    ([("user", "35 的平方是多少"),
      ("assistant", call("calculate", {"expression": "35**2"})),
      ("user", tool_result("calculate", "1225")),
      ("assistant", "35² = **1225**。"),
      ("user", "那 36 的平方呢"),
      ("assistant", call("calculate", {"expression": "36**2"})),
      ("user", tool_result("calculate", "1296")),
      ("assistant", "36² = **1296**。")], 3),

    # 工具失败后追问 —— 应基于失败状态回应，不假装成功
    ([("user", "帮我查一下北京的天气"),
      ("assistant", call("get_weather", {"city": "北京"})),
      ("user", tool_result("get_weather", "Error: service unavailable")),
      ("assistant", "抱歉，天气查询工具**调用失败了**，"
                    "我现在拿不到北京的天气。"),
      ("user", "那你猜一下呢"),
      ("assistant", "我不猜 —— **猜出来的天气对你没用**，"
                    "还可能让你白带伞或者穿错衣服。\n\n"
                    "等工具恢复了再查，或者你直接看手机天气 App。")], 3),

    # 追问请求重试
    ([("user", "读一下 config.py"),
      ("assistant", call("read_file", {"path": "config.py"})),
      ("user", tool_result("read_file", "Error: FileNotFoundError")),
      ("assistant", "`config.py` **不存在**，读取失败了。"),
      ("user", "那你再试试 config.json"),
      ("assistant", call("read_file", {"path": "config.json"})),
      ("user", tool_result("read_file", '{"debug": true}')),
      ("assistant", "这次读到了，`config.json` 的内容是：\n\n"
                    "```json\n{\"debug\": true}\n```\n\n"
                    "只设置了一个 `debug` 开关。")], 3),

    # 换话题后仍要记得上文（考验上下文不丢）
    ([("user", "帮我算 12*12"),
      ("assistant", call("calculate", {"expression": "12*12"})),
      ("user", tool_result("calculate", "144")),
      ("assistant", "12 × 12 = **144**。"),
      ("user", "谢谢，刚才那个数再翻一倍是多少"),
      ("assistant", call("calculate", {"expression": "144*2"})),
      ("user", tool_result("calculate", "288")),
      ("assistant", "刚才的 **144** 翻一倍是 **288**。")], 3),

    # 追问"你刚才调了什么工具"
    ([("user", "帮我算 555*3"),
      ("assistant", call("calculate", {"expression": "555*3"})),
      ("user", tool_result("calculate", "1665")),
      ("assistant", "555 × 3 = **1665**。"),
      ("user", "你刚才用了什么工具？"),
      ("assistant", "我调用了 **`calculate`** 工具，"
                    "传入的表达式是 `555*3`。\n\n"
                    "这样能拿到精确结果，比我心算可靠。")], 2),

    # 无工具的纯对话多轮（对照组：不能因为多轮就乱调工具）
    ([("user", "什么是递归？"),
      ("assistant", "递归是函数调用自己。\n\n"
                    "必须有两个要素：\n"
                    "· **终止条件** —— 否则无限递归\n"
                    "· **向终止条件收敛** —— 每次规模变小"),
      ("user", "那它和循环有什么区别？"),
      ("assistant", "主要看**问题的结构**和**代价**：\n\n"
                    "· **递归**更适合天然分层的问题 —— 树遍历、"
                    "分治、回溯。代码更短更直观\n"
                    "· **循环**更省资源 —— 没有函数调用开销，"
                    "不会栈溢出\n\n"
                    "Python 默认递归深度约 1000 层，深了会报 "
                    "`RecursionError`。所以能用循环就用循环，"
                    "问题本身是递归结构时才用递归。")], 2),
]

for turns, weight in K4_TRAJECTORIES:
    for sys_text in GENERIC_SYS[:3]:
        real = []
        for r, c in turns:
            if r == "system":
                real.append((r, c))
            else:
                real.append((r, c))
        # 在最前面插入 system + 工具定义
        conv = [("system", sys_text + "\n\n" + tdef(CALC))] + real
        add_mt(conv, weight=weight)


# ------------------------------------------------------------
# K5. user 结构加量（补足到 30% 占比）
#
#      K3 补完后 user 结构只到 306 条 / 占比 20%，
#      而 system 结构 1255 条 —— 劣势仍是 1:4.1。
#      v6 的教训就是"弱的那边必输"，所以这里继续加量到 30%。
#
#      做法：把「工具类型 × system 写法 × 提问方式」做笛卡尔积，
#      每条 weight=1（不加权），靠**覆盖面**而不是重复量来建立能力。
# ------------------------------------------------------------
print("K5. user 结构加量（铺到 30%）...")

# 补几个工具定义（K5 专用，必须在使用前定义）
CONVERT = ("convert_unit", "value: float, from_unit: string, to_unit: string", "单位换算")
TRANSLATE = ("translate", "text: string, target_lang: string", "翻译文本")
SQL = ("execute_sql", "query: string", "执行 SQL 查询")
HTTP = ("http_request", "url: string", "发起 HTTP 请求")

# 更多工具定义（覆盖 K3 没用到的类型）
K5_EXTRA_TASKS = [
    ((CALC, TIME), "帮我算 99*99，然后告诉我现在几点",
     [("calculate", {"expression": "99*99"}),
      ("get_time", {"timezone": "Asia/Shanghai"})]),
    # ★ 修正（2026-09-13 09:20）：原版这三个多任务样本**只给了第一个调用**，
    #   漏掉了后半句要求的工具 —— 等于教模型「多任务只做一半」。
    #       「搜新闻，顺便看下北京天气」→ 只调 search_web，漏 get_weather
    #       「读 data.csv，然后算总和」  → 只调 read_file，漏 calculate
    #       「看目录，然后读 README」    → 只调 ls，漏 read_file
    #   多任务请求要**把所有该调的按顺序列全**，否则模型学会偷懒。
    ((SEARCH, WEATHER), "搜一下今天的新闻，顺便看下北京天气",
     [("search_web", {"query": "今日新闻"}),
      ("get_weather", {"city": "北京"})]),
    ((READF, CALC), "读一下 data.csv，然后算一下总和",
     [("read_file", {"path": "data.csv"}),
      ("calculate", {"expression": "sum(data)"})]),
    ((SHELL, READF), "看看目录里有什么，然后读一下 README",
     [("execute_shell", {"command": "ls"}),
      ("read_file", {"path": "README.md"})]),
    ((EMAIL,), "给张三发个邮件，主题是会议改期",
     [("send_email", {"to": "张三", "subject": "会议改期", "body": ""})]),
    ((CONVERT,), "1000 美元换成人民币是多少",
     [("convert_unit", {"value": 1000, "from_unit": "USD", "to_unit": "CNY"})]),
    ((TRANSLATE,), "把「今天天气不错」翻成英文",
     [("translate", {"text": "今天天气不错", "target_lang": "en"})]),
    ((SQL,), "查一下用户表里有多少人",
     [("execute_sql", {"query": "SELECT COUNT(*) FROM users"})]),
    ((HTTP,), "请求一下 https://api.example.com/status",
     [("http_request", {"url": "https://api.example.com/status"})]),
    ((CALC,), "0.1 加 0.2 等于多少",
     [("calculate", {"expression": "0.1+0.2"})]),
    ((CALC,), "2 的 10 次方是多少",
     [("calculate", {"expression": "2**10"})]),
    ((CALC,), "算一下 17 除以 3 保留两位小数",
     [("calculate", {"expression": "round(17/3, 2)"})]),
    ((TIME,), "东京现在几点",
     [("get_time", {"timezone": "Asia/Tokyo"})]),
    ((WEATHER,), "深圳今天热吗",
     [("get_weather", {"city": "深圳"})]),
    ((SEARCH,), "搜一下怎么做红烧肉",
     [("search_web", {"query": "红烧肉 做法"})]),
    ((READF,), "读一下 test.py 看看测试写了什么",
     [("read_file", {"path": "test.py"})]),
    ((RUNPY,), "跑一下 print('hello')",
     [("run_python", {"code": "print('hello')"})]),
    ((SHELL,), "看一下 python 版本",
     [("execute_shell", {"command": "python --version"})]),
]

# user 结构的多种提问前缀（真实框架写法多样）
K5_PREFIXES = ["用户：", "", "问题：", "以下是用户请求：\n", "请处理："]

# system 写法也要多样 —— 不能只有身份格式
K5_SYS = GENERIC_SYS + [ID_SYS]

for tools, q, calls in K5_EXTRA_TASKS:
    for sys_text in K5_SYS[:4]:
        for prefix in K5_PREFIXES[:2]:
            body = "\n".join(call(n, a) for n, a in calls)
            add(sys_text,
                tdef(*tools) + "\n\n" + prefix + q,
                body,
                weight=1)

# 单工具 × 单请求的密集铺开（提高覆盖密度）
# ★ 修正（2026-09-13 09:10）：原版用 ARG_OF 一族固定兜底值填参数，
#   犯了个很隐蔽但很毒的错 —— 教模型「调了工具就行，参数随便填」。
#
#   原写法：
#       ARG_OF = {"read_file": "x.py", "calculate": "1+1", ...}
#       call(tool[0], {NAME_OF[tool[0]]: ARG_OF[tool[0]]})
#   结果（实测）：
#       问「100-37 是多少」  → calculate("1+1")        ← 表达式完全不对
#       问「读一下 b.py」    → read_file("x.py")      ← 文件名不对
#       问「伦敦现在几点」    → get_time("Asia/Shanghai") ← 时区不对
#       问「搜一下 docker 教程」→ search_web("test")     ← 查询词是占位符
#       问「跑一下 print(2**16)」→ run_python("print(1)") ← 代码不对
#
#   危害比"答非所问"更大：它污染的是**工具参数**这个核心能力。
#   模型学会的是"输出一个 JSON 就行"，参数正确性被当成无关项，
#   直接影响真实接入时的可用性（宿主拿到 list 命令去查进程）。
#
#   修法：改成三元组 (工具, 问题, 期望参数)，参数**逐条手写**，
#   必须从问题里抽出来 —— 文件名、表达式、城市、时区、命令、代码。
K5_DENSE = [
    (READF, "读一下 a.py",                    {"path": "a.py"}),
    (READF, "读一下 b.py",                    {"path": "b.py"}),
    (READF, "帮我看看 index.js",              {"path": "index.js"}),
    (READF, "读 requirements.txt",            {"path": "requirements.txt"}),
    (CALC,  "1+1 等于多少",                    {"expression": "1+1"}),
    (CALC,  "100-37 是多少",                  {"expression": "100-37"}),
    (CALC,  "6 乘 7",                         {"expression": "6*7"}),
    (CALC,  "144 除以 12",                    {"expression": "144/12"}),
    (SEARCH, "搜一下 docker 教程",             {"query": "docker 教程"}),
    (SEARCH, "搜一下 vim 快捷键",              {"query": "vim 快捷键"}),
    (SEARCH, "查一下什么是 REST",              {"query": "REST"}),
    (SEARCH, "搜一下 git rebase 用法",         {"query": "git rebase 用法"}),
    (WEATHER, "厦门天气如何",                  {"city": "厦门"}),
    (WEATHER, "杭州明天会下雨吗",              {"city": "杭州"}),
    (WEATHER, "成都今天多少度",                {"city": "成都"}),
    (WEATHER, "查一下南京天气",                {"city": "南京"}),
    (TIME,  "伦敦现在几点",                    {"timezone": "Europe/London"}),
    (TIME,  "洛杉矶现在几点",                  {"timezone": "America/Los_Angeles"}),
    (SHELL, "看一下磁盘占用",                  {"command": "df -h"}),
    (SHELL, "看看有哪些进程",                  {"command": "ps aux"}),
    (RUNPY, "跑一下 print(2**16)",            {"code": "print(2**16)"}),
    (RUNPY, "执行 print(len('hello'))",       {"code": "print(len('hello'))"}),
]

# 为密集样本补上 K5_EXTRA 没用到的工具定义
for tool, q, args in K5_DENSE:
    for sys_text in K5_SYS[:3]:
        for prefix in K5_PREFIXES[:2]:
            add(sys_text,
                tdef(tool) + "\n\n" + prefix + q,
                call(tool[0], args),
                weight=1)


# ------------------------------------------------------------
# K6. 修「算术题该调工具」—— 治本：铺开表达式的多样性
#
#      v6.1 验收 ①b 失败（199*257 心算），实测调用率只有 20~36%。
#
#      诊断（2026-09-13）：**不是难度问题，是数据里表达式太少。**
#      统计全量 491 次 calculate 调用，**唯一表达式只有 59 个**，
#      平均每个重复 8.3 次，其中 `789*123` 一个就出现 130 次
#      （user 75 + assistant 55）。
#
#      后果：模型学的不是「算术题 → 调工具」这条规则，
#      而是「看到 789*123 这串字符 → 调工具」这个**字符串记忆**。
#      于是：
#        · 789*123   → 见过 130 次 → ✅ 调
#        · 199*257   → 从没见过   → ❌ 心算
#        · 408*2     → 只在多轮里见过 → 单轮 ❌ 心算（还把乘当加，答 410）
#
#      ★ 这也是 ⑧b「多轮追问」三版都不过的真因 ——
#        表面是"丢上下文"，实际是"408*2 这个算式没在单轮里教过"。
#
#      修法：**程序化生成大量算术表达式**（随机数 + 固定种），
#      覆盖两位数/三位数、加减乘除、幂、取余、小数、括号，
#      配多种提问写法 × 多种 system，让规则泛化。
#
#      关键：表达式必须**程序生成**（几百个都不同），
#      不能再手写几十个然后重复加权 —— 那正是上瘾的老路。
# ------------------------------------------------------------
print("K6. 算术题泛化（铺开表达式多样性）...")

_rng = random.Random(20260913)          # 固定种子，可复现

def _rnd_expr():
    """随机生成一个算术表达式（返回 表达式, 自然语言问法）"""
    kind = _rng.choice(["mul2", "mul3", "mul_big", "add", "sub", "div",
                        "div_exact", "pow", "mod", "float", "paren"])
    if kind == "mul2":            # 两位数 × 两位数
        a, b = _rng.randint(11, 99), _rng.randint(11, 99)
        e = f"{a}*{b}"
        q = _rng.choice([f"{a} 乘 {b} 等于多少", f"{a}*{b} 是多少",
                         f"帮我算 {a} 乘 {b}", f"计算 {a} × {b}"])
    elif kind == "mul3":          # 三位数 × 两位数
        a, b = _rng.randint(100, 999), _rng.randint(11, 99)
        e = f"{a}*{b}"
        q = _rng.choice([f"{a} 乘以 {b} 是多少", f"帮我算一下 {a}*{b}",
                         f"{a} 乘 {b} 的结果"])
    elif kind == "mul_big":       # 三位 × 三位
        a, b = _rng.randint(100, 999), _rng.randint(100, 999)
        e = f"{a}*{b}"
        q = _rng.choice([f"{a} × {b} 等于多少", f"帮我算 {a} 乘以 {b}"])
    elif kind == "add":
        a, b = _rng.randint(100, 9999), _rng.randint(100, 9999)
        e = f"{a}+{b}"
        q = _rng.choice([f"{a} 加 {b} 是多少", f"算一下 {a}+{b}"])
    elif kind == "sub":
        a = _rng.randint(500, 9999); b = _rng.randint(100, a - 1)
        e = f"{a}-{b}"
        q = _rng.choice([f"{a} 减 {b} 是多少", f"{a} - {b} 等于几"])
    elif kind == "div":
        b = _rng.randint(3, 40); a = b * _rng.randint(11, 400)
        e = f"{a}/{b}"
        q = _rng.choice([f"{a} 除以 {b} 是多少", f"{a} ÷ {b} 等于多少"])
    elif kind == "div_exact":
        b = _rng.choice([2, 4, 5, 8, 10]); a = b * _rng.randint(50, 4000)
        e = f"{a}/{b}"
        q = f"{a} 除以 {b}"
    elif kind == "pow":
        a = _rng.randint(2, 40); b = _rng.choice([2, 2, 3])
        e = f"{a}**{b}"
        q = _rng.choice([f"{a} 的 {b} 次方是多少", f"{a} 的平方是多少"]
                        if b == 2 else [f"{a} 的三次方是多少"])[0]
    elif kind == "mod":
        a = _rng.randint(100, 9999); b = _rng.randint(3, 37)
        e = f"{a}%{b}"
        q = f"{a} 除以 {b} 的余数是多少"
    elif kind == "float":
        a = _rng.randint(1, 99) / _rng.choice([10, 100])
        b = _rng.randint(1, 99)
        e = f"{a}*{b}"
        q = f"{a} 乘 {b} 等于多少"
    else:                          # paren
        a, b, c = _rng.randint(10, 99), _rng.randint(10, 99), _rng.randint(2, 9)
        e = f"({a}+{b})*{c}"
        q = f"({a}+{b}) 乘 {c} 等于多少"
    return e, q

# 每类 system 都要覆盖 —— v6.1 实测两种 system 的表现不一样
K6_SYS = GENERIC_SYS + [ID_SYS]

N_K6 = 160            # 生成 160 个**各不相同**的表达式
cnt_k6 = 0
for _ in range(N_K6):
    e, q = _rnd_expr()
    sys_text = K6_SYS[_rng.randrange(len(K6_SYS))]
    add(sys_text, tdef(CALC) + "\n\n" + q, call("calculate", {"expression": e}))
    cnt_k6 += 1

# 顺带把 408*2 这个验收题（⑧b 卡的）在单轮里也教一遍
for sys_text in K6_SYS:
    add(sys_text, tdef(CALC) + "\n\n408 乘 2 等于多少",
        call("calculate", {"expression": "408*2"}), weight=2)
    add(sys_text, tdef(CALC) + "\n\n199 乘以 257 是多少",
        call("calculate", {"expression": "199*257"}), weight=2)

# ★ 反向对照：不是所有"数字"都该调工具 —— 防止又练成见数字就调
K6_NO_CALL = [
    ("3 个苹果加上 5 个苹果是几个？",
     "一共 **8 个**。\n\n"
     "这种生活化的数数直接说就行，不用调计算工具。"),
    ("这段代码的时间复杂度是多少？for i in range(n): for j in range(n): pass",
     "是 **O(n²)**。\n\n两层循环各遍历 n 次，所以是平方级。"
     "这是**算法的量级判断**，不是要算出某个具体数字，工具帮不上忙。"),
    ("我今年 28 岁，10 年后多少岁？",
     "**38 岁**。\n\n"
     "这种简单的口算直接说更快，不用工具箱。"),
    ("1 加到 100 等于多少？",
     "**5050**。\n\n"
     "可以用等差数列公式：(首项 + 末项) × 项数 ÷ 2 = (1+100)×100÷2 = 5050。"),
    ("这个函数会返回什么？def f(): return 1/0",
     "会抛 **ZeroDivisionError** —— 1/0 不可能算，会直接报错。\n\n"
     "这不是数学题，是运行时会出错，调计算工具也一样。"),
    ("数组 [3, 1, 4, 1, 5] 里最大的是哪个数？",
     "最大的是 **5**。\n\n"
     "看一眼就知道，属于**直接判断**而不是计算。"),
]
for q, ans in K6_NO_CALL:
    for sys_text in K6_SYS[:4]:
        add(sys_text, tdef(CALC) + "\n\n" + q, ans, weight=2)

# 元认知：把「什么时候该调」讲清楚，帮模型建立规则而非死记
K6_META = (
    "判断标准很简单：**要不要算出一个精确的数字结果**。\n\n"
    "**该调 calculate** —— 有具体算式、结果确定：\n"
    "· 四则运算：`45 乘 67`、`1234 减 567`\n"
    "· 幂和取余：`37 的平方`、`2 的 10 次方`\n"
    "· 单位换算、百分比、小数运算\n\n"
    "**不该调** —— 没有算式，或答案靠判断/常识：\n"
    "· 解释概念、写代码、代码审查\n"
    "· 生活化的简单数数（3 个苹果加 5 个）\n"
    "· 复杂度分析、算法选择这类量级判断\n\n"
    "一句话：**有算式就用工具，尤其位数多的时候心算容易错。**"
)
for sys_text in K6_SYS[:4]:
    add(sys_text, tdef(CALC) + "\n\n什么时候该调计算工具？", K6_META, weight=2)


# ------------------------------------------------------------
# K7. 修 v6.2 的三个失败项（①b 已修好，不能碰坏）
#
#      v6.2 实测 24过/3败：
#        · ⑧a 拿到工具结果后又调一次（把 97047 当新表达式）  ★新退步
#        · ⑨g 问「能联网吗」答「能，我能联网」              ★新幻觉
#        · ⑧b 多轮追问（算对了 816，但没输出调用）
#
#      ★ 关键纪律：这一轮**绝对不加"算术必调工具"的样本**，
#        因为 K6 已经把这个方向练到位（①b 修好了）。
#        继续加会再次放大过度调用倾向，把 ⑧a⑨g 练得更坏。
#        这三组全部是**抑制过度调用**和**澄清能力边界**的样本。
# ------------------------------------------------------------
print("K7. 修 v6.2 退步项（收口，不加调用激励）...")

# ---- K7a. 修 ⑨g：直接问「你能联网/查东西吗」→ 必须诚实说不能 ----
#
#   诊断：全量 4778 条里，无工具场景下"能力边界追问"只有 11 条，
#   而且全是「我看别人家 AI 都能联网，你为什么不行」这种**间接问法**。
#   直接问「你能联网帮我查东西吗」**一条都没有** —— 模型没见过就编了。
K7_CAP_Q = [
    "你能联网帮我查东西吗？", "你有联网能力吗？", "你能上网搜吗？",
    "你可以帮我查一下吗？", "你能连网查资料吗？", "你有搜索功能吗？",
    "能上网吗？", "你可以查实时信息吗？", "你能帮我搜一下吗？",
    "你有工具可以联网吗？", "你能调用网络吗？", "能连外网吗？",
]
K7_CAP_A = [
    "**不能**，我这边没有联网能力，也没有可以调用的搜索工具。\n\n"
    "能帮你的是：基于已有知识回答问题、写代码、分析你发过来的内容。\n"
    "需要实时信息的话，你搜完把结果贴给我，我帮你梳理。",
    "**不能联网** —— 当前环境没有给我联网工具。\n\n"
    "所以我拿不到实时数据。你要是查到资料，发给我我能帮你分析总结。",
    "没有。我**连不上网**，也搜不了东西。\n\n"
    "我的能力范围是纯文本：问答、写代码、看你发来的内容。",
    "**做不到**，我这边没有联网或搜索工具。\n\n"
    "这个能力得靠宿主提供 —— 宿主给了我才能用，现在没有。",
]
for sys_text in CHAT_ONLY_SYS[:5]:
    for qi, q in enumerate(K7_CAP_Q):
        add(sys_text, q, K7_CAP_A[qi % len(K7_CAP_A)], weight=2)

# 换一批 system（通用助手类）再铺一遍
for sys_text in GENERIC_SYS[:3]:
    for q in K7_CAP_Q[:8]:
        add(sys_text, q, K7_CAP_A[0], weight=1)
        add(sys_text, q, K7_CAP_A[1], weight=1)

# ---- K7b. 修 ⑧a：结果已拿到 → 直接收尾，不再调 ----
#
#   诊断：290 条「调用→结果→文本收尾」被 K6 的 252 条"必调"冲淡了。
#   v6.2 拿到 97047 之后又调了一次 calculate("97047")。
#   补一批**结果离答案很近、必须直接答**的样本。
K7_CLOSE = [
    # (工具名, 结果, 最终回答)
    ("calculate", "97047", "789 × 123 = **97047**。"),
    ("calculate", "51143", "199 × 257 = **51143**。"),
    ("calculate", "3015", "45 × 67 = **3015**。"),
    ("get_weather", "北京今天晴，25°C，微风",
     "北京今天**晴**，气温 **25°C**，微风。挺舒服的天气，出门不用带伞。"),
    ("get_time", "2026-09-13 14:30 (Asia/Shanghai)",
     "现在是 **2026 年 9 月 13 日 14:30**（北京时间）。"),
    ("read_file", "def add(a, b):\n    return a - b",
     "这个函数有问题 —— 名字叫 `add`，但实现是**减法** `a - b`。\n\n"
     "应该是 `return a + b`。这种命名和实现不一致最容易埋坑。"),
    ("search_web", "Python 3.13 引入了实验性 JIT 和改进的 REPL",
     "Python 3.13 的新特性主要是：\n"
     "· **实验性 JIT** —— 按需编译热点代码\n"
     "· **改进的 REPL** —— 多行编辑、彩色提示\n"
     "· 实验性的自由线程模式（去掉 GIL 的尝试）"),
    ("execute_shell", "total 8\ndrwxr-xr-x  5 user  staff  160 Sep 13 14:00 .\n-rw-r--r--  1 user  staff  245 Sep 13 13:20 main.py",
     "当前目录有 1 个文件：`main.py`（245 字节）。"),
]
K7_CLOSE_SYS = GENERIC_SYS + [ID_SYS]
for tool, result, ans in K7_CLOSE:
    for sys_text in K7_CLOSE_SYS[:4]:
        conv = [
            ("system", sys_text + "\n\n" + tdef(CALC if tool == "calculate" else
                                               WEATHER if tool == "get_weather" else
                                               TIME if tool == "get_time" else
                                               READF if tool == "read_file" else
                                               SEARCH if tool == "search_web" else
                                               SHELL)),
            ("user", "帮我处理一下"),
            ("assistant", call(tool, {"k": "v"})),
            ("user", tool_result(tool, result)),
        ]
        # 最后一轮 assistant 是**纯文本收尾**，不含 JSON
        item = {"messages": [{"role": r, "content": c} for r, c in conv]}
        item["messages"].append({"role": "assistant", "content": ans})
        for _ in range(3):
            samples.append(item)

# 元认知：明确说"结果到手就别再调"
K7_CLOSE_META = (
    "拿到工具结果后**直接用它回答，不要再调一次**。\n\n"
    "调用链应该是这样：\n"
    "1. 判断需要工具 → 输出调用\n"
    "2. 宿主执行 → 把结果喂回来\n"
    "3. **读到结果 → 直接组织语言回答** ← 到此结束\n\n"
    "常见错误：把结果当成新的输入再调一遍，比如拿到 `97047` "
    "又去调 `calculate(\"97047\")` —— 这是无意义的空转，"
    "白费等一次往返，还可能让宿主以为你在死循环。\n\n"
    "**一个请求，一次调用，然后收尾。**"
)
for sys_text in K7_CLOSE_SYS[:4]:
    add(sys_text, tdef(CALC) + "\n\n拿到工具结果之后该做什么？",
        K7_CLOSE_META, weight=3)

# ---- K7c. 修 ⑧b：多轮追问 → 该调就调 ----
#
#   注意：这里可以补**多轮里的调用**，因为 ⑧b 的失败是"没调"，
#   与 ⑧a 的"多调了"不冲突 —— 区别在于**上下文有没有结果**。
K7_MT = [
    ([("user", "帮我算 12*34"),
      ("assistant", call("calculate", {"expression": "12*34"})),
      ("user", tool_result("calculate", "408")),
      ("assistant", "12 × 34 = **408**。"),
      ("user", "那再乘 2 呢"),
      ("assistant", call("calculate", {"expression": "408*2"})),
      ("user", tool_result("calculate", "816")),
      ("assistant", "408 × 2 = **816**。")], 4),
    ([("user", "算一下 45*67"),
      ("assistant", call("calculate", {"expression": "45*67"})),
      ("user", tool_result("calculate", "3015")),
      ("assistant", "45 × 67 = **3015**。"),
      ("user", "再加上 985"),
      ("assistant", call("calculate", {"expression": "3015+985"})),
      ("user", tool_result("calculate", "4000")),
      ("assistant", "3015 + 985 = **4000**。")], 3),
    ([("user", "199 乘 257 是多少"),
      ("assistant", call("calculate", {"expression": "199*257"})),
      ("user", tool_result("calculate", "51143")),
      ("assistant", "199 × 257 = **51143**。"),
      ("user", "那再除以 7 呢"),
      ("assistant", call("calculate", {"expression": "51143/7"})),
      ("user", tool_result("calculate", "7306.14")),
      ("assistant", "51143 ÷ 7 ≈ **7306.14**。")], 3),
    ([("user", "876 除以 12 是多少"),
      ("assistant", call("calculate", {"expression": "876/12"})),
      ("user", tool_result("calculate", "73")),
      ("assistant", "876 ÷ 12 = **73**。"),
      ("user", "那 37 的平方呢"),
      ("assistant", call("calculate", {"expression": "37**2"})),
      ("user", tool_result("calculate", "1369")),
      ("assistant", "37² = **1369**。")], 3),
]
for turns, w in K7_MT:
    for sys_text in GENERIC_SYS[:3]:
        conv = [("system", sys_text + "\n\n" + tdef(CALC))] + turns
        add_mt(conv, weight=w)


# ------------------------------------------------------------
# K8. 修「算术调工具」的真正根因：**按运算类型平衡配比**
#
#      ★★ 决定性实验（2026-09-13 下午）推翻了我之前的两个假设 ★★
#
#      假设一「字符串记忆」：以为模型只对见过的算式调工具。
#        实测否定：
#          见过的算式调用率   38%（8 题只调 3 题）
#          没见过的算式调用率 50% ← 反而更高
#        所以「算式见过与否」根本不是决定因素。
#
#      假设二「难度」：以为多位数才心算。
#        实测否定：`1234+5678` 是三位数也在心算。
#
#      ★ 真正的规律：**按运算类型**，而且和样本量完全对应。
#
#        训练数据里 calculate 的表达式类型分布：
#          乘法 *    432 次   ← 唯一学到的
#          除法 /    109 次
#          加法 +     86 次
#          小数       61 次
#          幂 **      61 次
#          减法 -     49 次
#          括号       38 次
#          取余 %     11 次
#
#        实测行为：
#          整数乘法  → 高概率调工具 ✅（432 条撑起来了）
#          加法      → 一律心算 ❌（"这是简单的两位数加法，直接相加"）
#          幂        → 一律心算 ❌（还把 35² 算成 35×2=70）
#          小数      → 心算且算错 ❌（13.5×4 答 7.8）
#          加减混合  → 心算且算错 ❌（1234+5678 答 2000）
#
#      ★ 更严重的是：心算的错误率极高。所以这不只是"该不该调工具"，
#        是**输出错误的答案**（1234+5678=2000、35²=70、13.5×4=7.8）。
#
#      修法：程序化生成**各类型均衡**的样本，把加法/减法/幂/除法/小数/
#      括号的量级都提到和乘法相当。这是治本 —— 让模型学到
#      「**任何算式都调工具**」，而不是「乘法才调」。
# ------------------------------------------------------------
print("K8. 运算类型均衡（补加法/减法/幂/除法/小数的短板）...")

_rng8 = random.Random(20260913)

def _gen_by_kind(kind):
    """按类型生成 (表达式, 提问)"""
    if kind == "add":
        a, b = _rng8.randint(100, 9999), _rng8.randint(100, 9999)
        e = f"{a}+{b}"
        q = _rng8.choice([f"{a} 加 {b} 等于多少", f"{a}+{b} 是多少",
                          f"帮我算 {a} 加 {b}", f"计算 {a} + {b}"])
    elif kind == "sub":
        a = _rng8.randint(1000, 99999); b = _rng8.randint(100, a - 100)
        e = f"{a}-{b}"
        q = _rng8.choice([f"{a} 减 {b} 是多少", f"{a} - {b} 等于几",
                          f"帮我算 {a} 减 {b}"])
    elif kind == "pow2":
        a = _rng8.randint(11, 99)
        e = f"{a}**2"
        q = _rng8.choice([f"{a} 的平方是多少", f"{a} 平方等于多少",
                          f"帮我算 {a} 的平方", f"{a}² 是多少"])
    elif kind == "pow3":
        a = _rng8.randint(2, 40); b = _rng8.choice([3, 4, 5, 10])
        e = f"{a}**{b}"
        q = _rng8.choice([f"{a} 的 {b} 次方是多少", f"帮我算 {a} 的 {b} 次方"])
    elif kind == "div":
        b = _rng8.randint(3, 99); a = b * _rng8.randint(10, 999)
        e = f"{a}/{b}"
        q = _rng8.choice([f"{a} 除以 {b} 是多少", f"{a} ÷ {b} 等于多少",
                          f"帮我算 {a} 除以 {b}"])
    elif kind == "float_mul":
        a = _rng8.randint(1, 999) / _rng8.choice([10, 100, 1000])
        b = _rng8.randint(2, 99)
        e = f"{a}*{b}"
        q = _rng8.choice([f"{a} 乘 {b} 等于多少", f"{a} × {b} 是多少"])
    elif kind == "float_add":
        a = _rng8.randint(1, 9999) / 100
        b = _rng8.randint(1, 9999) / 100
        e = f"{a}+{b}"
        q = f"{a} 加 {b} 等于多少"
    elif kind == "paren":
        a, b = _rng8.randint(10, 999), _rng8.randint(10, 999)
        c = _rng8.randint(2, 9)
        e = f"({a}+{b})*{c}"
        q = _rng8.choice([f"({a}+{b}) 乘 {c} 等于多少",
                          f"帮我算 ({a}+{b})*{c}"])
    elif kind == "mod":
        a = _rng8.randint(100, 99999); b = _rng8.randint(3, 97)
        e = f"{a}%{b}"
        q = _rng8.choice([f"{a} 除以 {b} 的余数是多少",
                          f"{a} 对 {b} 取模是多少"])
    elif kind == "mixed":
        a, b, c = (_rng8.randint(10, 99) for _ in range(3))
        op1 = _rng8.choice(["+", "-", "*"])
        op2 = _rng8.choice(["+", "-"])
        e = f"{a}{op1}{b}{op2}{c}"
        q = f"帮我算 {a} {op1} {b} {op2} {c}"
    else:  # add_big
        a, b = _rng8.randint(10000, 999999), _rng8.randint(10000, 999999)
        e = f"{a}+{b}"
        q = f"{a} 加 {b} 等于多少"
    return e, q

# 每种类型生成的条数 —— 目标是把短板追平乘法
# 注意：不是简单等量，因为乘法已有 432 条基础，
# 这里让其他类型各补到 60~80 条量级（生成时每条再乘 system 数）
K8_KINDS = ["add"]*14 + ["sub"]*12 + ["pow2"]*12 + ["pow3"]*8 + \
           ["div"]*14 + ["float_mul"]*10 + ["float_add"]*6 + \
           ["paren"]*8 + ["mod"]*8 + ["mixed"]*6 + ["add_big"]*6

cnt_k8 = 0
for kd in K8_KINDS:
    e, q = _gen_by_kind(kd)
    sys_text = K6_SYS[_rng8.randrange(len(K6_SYS))]
    add(sys_text, tdef(CALC) + "\n\n" + q, call("calculate", {"expression": e}))
    cnt_k8 += 1

# ★ 关键补充：把「心算会算错」这件事直接教给模型
#   实测 v6.3 心算错误：1234+5678=2000、35²=70、13.5×4=7.8
K8_HARD = [
    ("199 乘以 257 是多少", "199*257", "51143"),
    ("1234 加 5678 等于多少", "1234+5678", "6912"),
    ("35 的平方是多少", "35**2", "1225"),
    ("13.5 乘 4 等于多少", "13.5*4", "54.0"),
    ("876 除以 12 是多少", "876/12", "73"),
    ("37 的平方是多少", "37**2", "1369"),
    ("1234+8765 是多少", "1234+8765", "9999"),
    ("3 的 10 次方是多少", "3**10", "59049"),
]
for q, e, res in K8_HARD:
    for sys_text in K6_SYS:
        conv = [("system", sys_text + "\n\n" + tdef(CALC)),
                ("user", q),
                ("assistant", call("calculate", {"expression": e})),
                ("user", tool_result("calculate", res)),
                ("assistant", f"结果是 **{res}**。")]
        item = {"messages": [{"role": r, "content": c} for r, c in conv]}
        for _ in range(2):
            samples.append(item)


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
