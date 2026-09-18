#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CytoTRACE 2 output parity gate.

Compares a candidate implementation's results file against the official
expected output (or any baseline run) cell by cell.

File format (tab-separated, first column = cell barcode, no header name
for the first column):
    <cell_id>  CytoTRACE2_Score  CytoTRACE2_Potency  CytoTRACE2_Relative
               preKNN_CytoTRACE2_Score  preKNN_CytoTRACE2_Potency

Pass criteria (ALL must hold):
  - identical cell sets
  - max |diff| <= atol for CytoTRACE2_Score, CytoTRACE2_Relative,
    preKNN_CytoTRACE2_Score
  - 100% exact match for both potency columns
  - Spearman(baseline Score, candidate Score) >= 1 - 1e-9

Usage:
    python parity_check.py BASELINE.txt CANDIDATE.txt [--atol 1e-6]
                            [--label NAME] [--json OUT.json] [--quiet]

Exit code 0 = pass, 1 = fail. The --json dump is meant to be attached to
the campaign record as the candidate's parity evidence.
"""

import argparse
import json
import sys

NUMERIC_COLS = [
    "CytoTRACE2_Score",
    "CytoTRACE2_Relative",
    "preKNN_CytoTRACE2_Score",
]
CATEGORICAL_COLS = [
    "CytoTRACE2_Potency",
    "preKNN_CytoTRACE2_Potency",
]


def load_results(path):
    """Read a cytotrace2 results txt into {cell_id: {col: value}} keeping file order."""
    rows = {}
    order = []
    with open(path, "r", encoding="utf-8") as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                # First row: columns[0] is the (unnamed) cell id column.
                header = parts
                continue
            cell = parts[0]
            rec = {}
            for i, col in enumerate(header[1:], start=1):
                rec[col] = parts[i] if i < len(parts) else ""
            rows[cell] = rec
            order.append(cell)
    if header is None:
        raise ValueError("empty file: %s" % path)
    missing = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c not in header[1:]]
    if missing:
        raise ValueError("%s: missing expected columns %s (header=%s)" % (path, missing, header))
    return rows, order


def ranks(values):
    """Average ranks (1-based) with tie handling."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[indexed[k]] = avg
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def main():
    ap = argparse.ArgumentParser(description="CytoTRACE 2 parity gate")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--spearman-min", type=float, default=1.0 - 1e-9)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base, base_order = load_results(args.baseline)
    cand, _ = load_results(args.candidate)

    common = [c for c in base_order if c in cand]
    only_base = [c for c in base_order if c not in cand]
    only_cand = [c for c in cand if c not in base]

    report = {
        "label": args.label,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "atol": args.atol,
        "n_cells_baseline": len(base),
        "n_cells_candidate": len(cand),
        "n_cells_common": len(common),
        "cells_missing_in_candidate": len(only_base),
        "extra_cells_in_candidate": len(only_cand),
    }

    ok = not only_base and not only_cand

    for col in NUMERIC_COLS:
        diffs = [abs(float(base[c][col]) - float(cand[c][col])) for c in common]
        worst = max(diffs) if diffs else float("inf")
        worst_cell = common[diffs.index(worst)] if diffs else ""
        passed = worst <= args.atol
        ok = ok and passed
        report["max_abs_diff_%s" % col] = worst
        report["worst_cell_%s" % col] = worst_cell
        report["pass_%s" % col] = passed

    for col in CATEGORICAL_COLS:
        match = sum(1 for c in common if base[c][col] == cand[c][col])
        frac = match / len(common) if common else 0.0
        passed = frac == 1.0
        ok = ok and passed
        report["match_frac_%s" % col] = frac
        report["pass_%s" % col] = passed

    if common:
        rho = spearman(
            [float(base[c]["CytoTRACE2_Score"]) for c in common],
            [float(cand[c]["CytoTRACE2_Score"]) for c in common],
        )
    else:
        rho = 0.0
    passed_rho = rho >= args.spearman_min
    ok = ok and passed_rho
    report["spearman_score"] = rho
    report["pass_spearman"] = passed_rho

    report["parity_ok"] = ok

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    if not args.quiet:
        print("parity_ok: %s" % ("PASS" if ok else "FAIL"))
        for col in NUMERIC_COLS:
            print("  max|diff| %-28s %.3e  (worst: %s)" % (
                col, report["max_abs_diff_%s" % col], report["worst_cell_%s" % col]))
        for col in CATEGORICAL_COLS:
            print("  match   %-28s %.6f" % (col, report["match_frac_%s" % col]))
        print("  spearman(Score) %.9f" % rho)
        if only_base or only_cand:
            print("  cell sets differ: %d missing / %d extra" % (len(only_base), len(only_cand)))

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
