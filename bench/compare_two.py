#!/usr/bin/env python3
"""对拍两个 cytotrace2 结果文件（fast-vs-fast）。

判据（与官方 gate 同构）：
- 细胞集合与顺序一致
- 三个数值列 max|Δ|、有符号统计
- 两个 potency 列精确匹配率
- Score 的 Spearman（自守恒：两次应完全同序）

用法:
  python3 compare_two.py REF.txt NEW.txt [--label 名] [--byte-check]
"""
import argparse, sys, hashlib

def load(p):
    rows = []
    with open(p) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(line.rstrip("\n").split("\t"))
    return header, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("new")
    ap.add_argument("--label", default="cmp")
    ap.add_argument("--byte-check", action="store_true")
    a = ap.parse_args()

    h1, r1 = load(a.ref)
    h2, r2 = load(a.new)
    n1, n2 = len(r1), len(r2)
    out = {"label": a.label, "n_ref": n1, "n_new": n2, "same_header": h1 == h2}
    if n1 != n2:
        out["cell_set_match"] = False
        print("FAIL: row count differs", n1, n2); sys.exit(1)
    same_cells = all(r1[i][0] == r2[i][0] for i in range(n1))
    out["cell_order_match"] = same_cells

    if a.byte_check:
        m1 = hashlib.md5(open(a.ref,'rb').read()).hexdigest()
        m2 = hashlib.md5(open(a.new,'rb').read()).hexdigest()
        out["md5_ref"], out["md5_new"], out["byte_identical"] = m1, m2, m1 == m2

    # 列: 1=Score 2=Potency 3=Relative 4=preKNN_Score 5=preKNN_Potency
    res = {}
    for col, name in [(1, "Score"), (3, "Relative"), (4, "preKNN_Score")]:
        maxd, sgn = 0.0, 0.0
        for i in range(n1):
            v1, v2 = float(r1[i][col]), float(r2[i][col])
            d = abs(v1 - v2)
            if d > maxd: maxd = d
            sgn += (v2 - v1)
        res[f"max|d|_{name}"] = maxd
        res[f"mean_signed_d_{name}"] = sgn / n1
    for col, name in [(2, "Potency"), (5, "preKNN_Potency")]:
        match = sum(1 for i in range(n1) if r1[i][col] == r2[i][col])
        res[f"match_{name}"] = match / n1
    # Spearman of Score between the two files（同序性：应=1）
    import math
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0]*len(vals)
        i = 0
        while i < len(order):
            j = i
            while j+1 < len(order) and vals[order[j+1]] == vals[order[i]]:
                j += 1
            avg = (i + j)/2 + 1
            for k in range(i, j+1): rk[order[k]] = avg
            i = j+1
        return rk
    s1 = ranks([float(r[1]) for r in r1]); s2 = ranks([float(r[1]) for r in r2])
    n = n1
    m1 = sum(s1)/n; m2 = sum(s2)/n
    num = sum((s1[i]-m1)*(s2[i]-m2) for i in range(n))
    d1 = math.sqrt(sum((x-m1)**2 for x in s1)); d2 = math.sqrt(sum((x-m2)**2 for x in s2))
    res["spearman_score_pair"] = num/(d1*d2) if d1>0 and d2>0 else float('nan')
    out.update(res)
    print(out)
    import json
    with open(a.new + ".cmp.json", "w") as f:
        json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
