#!/usr/bin/env python3
"""回归对比：把新版本和基线逐项比对，任何一项退化都报出来。

★ 为什么需要这个
    v6 系列五次迭代，每次都是「改数据 → 全量重训 → 发现修一个坏一个」。
    根本原因是没有固化「已通过项」，退化项只能靠人肉看 27 项清单。

    这个脚本把基线的验收结果存成 JSON，之后每次跑 eval_v6.py 都能
    自动比出「哪些项从 ✅ 变 ❌」「哪些从 ❌ 变 ✅」。

用法:
    # 1) 先给基线存个档（只做一次）
    python regress_v7.py --save-baseline eval_v6_1_result.txt

    # 2) 之后每次验收后对比
    python regress_v7.py --compare /tmp/tryrun_eval.txt
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, 'regress_baseline.json')

LINE_RE = re.compile(r'^\s*(✅|❌|⚠️)\s+(\S+)\s+(.*)$')


def parse_eval(path):
    """从 eval_v6.py 的输出里抽出 {项目名: ('ok'|'bad'|'warn', 描述)}"""
    items = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = LINE_RE.match(line.rstrip('\n'))
            if not m:
                continue
            mark, key, desc = m.groups()
            status = {'✅': 'ok', '❌': 'bad', '⚠️': 'warn'}[mark]
            # ★ 同一个编号会出现多次（④ 有 3 条、⑨ 有 6 条），
            #   光用编号做键会互相覆盖（27 项被压成 19 项）。
            #   用「编号 + 描述前 14 字」做唯一键。
            desc = desc.strip()
            uid = f"{key} {desc[:14]}" if desc else key
            items[uid] = {'key': key, 'status': status, 'desc': desc[:60]}
    return items


def summary(items):
    ok = sum(1 for v in items.values() if v['status'] == 'ok')
    bad = sum(1 for v in items.values() if v['status'] == 'bad')
    warn = sum(1 for v in items.values() if v['status'] == 'warn')
    return ok, bad, warn


def cmd_save_baseline(args):
    items = parse_eval(args.save_baseline)
    ok, bad, warn = summary(items)
    data = {'source': os.path.basename(args.save_baseline), 'items': items}
    with open(BASELINE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ 基线已存: {BASELINE}")
    print(f"  来源: {data['source']}")
    print(f"  {ok} 通过 / {bad} 失败 / {warn} 可疑  （共 {len(items)} 项）")


def cmd_compare(args):
    if not os.path.exists(BASELINE):
        print(f"✗ 还没有基线。先跑: python {os.path.basename(__file__)} "
              f"--save-baseline <基线结果文件>")
        sys.exit(1)

    with open(BASELINE, encoding='utf-8') as f:
        base = json.load(f)['items']
    cur = parse_eval(args.compare)

    b_ok, b_bad, b_warn = summary(base)
    c_ok, c_bad, c_warn = summary(cur)

    print("=" * 64)
    print("  回归对比")
    print("=" * 64)
    print(f"  基线: {b_ok} 通过 / {b_bad} 失败 / {b_warn} 可疑")
    print(f"  当前: {c_ok} 通过 / {c_bad} 失败 / {c_warn} 可疑")
    print()

    regressed, improved, new_items, gone = [], [], [], []

    for k, v in base.items():
        if k not in cur:
            gone.append(k)
            continue
        if v['status'] == 'ok' and cur[k]['status'] != 'ok':
            regressed.append((k, v, cur[k]))
        elif v['status'] == 'bad' and cur[k]['status'] == 'ok':
            improved.append((k, v, cur[k]))

    for k, v in cur.items():
        if k not in base:
            new_items.append((k, v))

    if regressed:
        print(f"  ⛔ 退化 {len(regressed)} 项（这些是发布阻塞项）:")
        for k, b, c in regressed:
            print(f"     {k}  {b['status']} → {c['status']}")
            print(f"        {c['desc']}")
        print()
    if improved:
        print(f"  ✅ 改善 {len(improved)} 项:")
        for k, b, c in improved:
            print(f"     {k}")
            print(f"        {c['desc']}")
        print()
    if new_items:
        print(f"  ➕ 新增 {len(new_items)} 项（不在基线里，需人看）:")
        for k, v in new_items:
            print(f"     {k}  {v['status']}  {v['desc']}")
        print()
    if gone:
        print(f"  ➖ 基线里有但当前没有 {len(gone)} 项（用例被删了？）: {gone}")
        print()

    if not regressed and not improved and not new_items and not gone:
        print("  逐项完全一致（v6.3 就出现过这种情况 —— 等于白跑）")
        print()

    print("=" * 64)
    if regressed:
        print(f"  结论: ❌ 有 {len(regressed)} 项退化，不要进全量")
        sys.exit(2)
    elif improved:
        print(f"  结论: ✅ 有 {len(improved)} 项改善且无退化，可以进全量")
    else:
        print("  结论: ⚠️ 无变化，这次改动没起作用 —— 检查数据是否真的变了")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--save-baseline', metavar='FILE',
                   help='把某次验收结果存为基线')
    g.add_argument('--compare', metavar='FILE',
                   help='把某次验收结果和基线对比')
    args = ap.parse_args()

    if args.save_baseline:
        cmd_save_baseline(args)
    else:
        cmd_compare(args)


if __name__ == '__main__':
    main()
