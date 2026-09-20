#!/usr/bin/env python3
"""Strict, NaN-aware fast-vs-fast result gate (stdlib only).

Exit 0: every requested check passed. Exit 1: mismatch or invalid input.
Empty fields and case-insensitive 'nan' are missing, never silently numeric.
Missing masks must match; numerical metrics use common finite entries only.
--byte-check additionally requires identical file bytes (SHA-256).
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

COLUMNS = (
    'CytoTRACE2_Score', 'CytoTRACE2_Potency', 'CytoTRACE2_Relative',
    'preKNN_CytoTRACE2_Score', 'preKNN_CytoTRACE2_Potency',
)
NUMERIC = (COLUMNS[0], COLUMNS[2], COLUMNS[3])
CATEGORICAL = (COLUMNS[1], COLUMNS[4])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader, None)
        if header is None or len(header) != 6 or tuple(header[1:]) != COLUMNS:
            raise ValueError(f'{path}: unexpected header; expected cell ID + {COLUMNS}')
        rows, seen = [], set()
        for line, row in enumerate(reader, 2):
            if len(row) != 6:
                raise ValueError(f'{path}:{line}: expected 6 fields, found {len(row)}')
            if not row[0] or row[0] in seen:
                raise ValueError(f'{path}:{line}: empty or duplicate cell ID {row[0]!r}')
            seen.add(row[0])
            record = dict(zip(header[1:], row[1:]))
            for name in NUMERIC:
                text = record[name].strip()
                if not text or text.lower() == 'nan':
                    record[name] = None
                else:
                    try:
                        value = float(text)
                    except ValueError as exc:
                        raise ValueError(f'{path}:{line}: invalid {name}={text!r}') from exc
                    if not math.isfinite(value):
                        raise ValueError(f'{path}:{line}: non-finite {name}={text!r}')
                    record[name] = value
            rows.append((row[0], record))
    if not rows:
        raise ValueError(f'{path}: no cell rows')
    return header, rows


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j - 1) / 2.0
        i = j
    return result


def spearman(left, right):
    if len(left) < 2:
        return None
    x, y = ranks(left), ranks(right)
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    xx = math.fsum((v - mx) ** 2 for v in x)
    yy = math.fsum((v - my) ** 2 for v in y)
    if xx == 0 or yy == 0:
        return None  # Constant ranks: correlation is undefined, not 1.
    xy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    return max(-1.0, min(1.0, xy / math.sqrt(xx * yy)))


def compare(ref, new, *, atol=1e-6, rtol=0.0, byte_check=False, min_spearman=None):
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError('atol and rtol must be finite and non-negative')
    if min_spearman is not None and not (-1 <= min_spearman <= 1):
        raise ValueError('min_spearman must be in [-1, 1]')
    h1, rows1 = load(ref)
    h2, rows2 = load(new)
    ids1, ids2 = [r[0] for r in rows1], [r[0] for r in rows2]
    failures = []
    out = dict(n_ref=len(rows1), n_new=len(rows2), same_header=h1 == h2,
               cell_set_match=set(ids1) == set(ids2), cell_order_match=ids1 == ids2,
               sha256_ref=sha256(ref), sha256_new=sha256(new),
               atol=atol, rtol=rtol, min_spearman=min_spearman,
               byte_check=byte_check, columns={})
    out['byte_identical'] = out['sha256_ref'] == out['sha256_new']
    for key in ('same_header', 'cell_set_match', 'cell_order_match'):
        if not out[key]:
            failures.append(key)
    if byte_check and not out['byte_identical']:
        failures.append('byte_identical')
    # Never compare values attached to different cells.
    if ids1 == ids2 and h1 == h2:
        for name in NUMERIC:
            x = [r[1][name] for r in rows1]
            y = [r[1][name] for r in rows2]
            mismatches = [i for i, (a, b) in enumerate(zip(x, y))
                          if (a is None) != (b is None)]
            pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
            diffs = [b - a for a, b in pairs]
            if not all(math.isfinite(d) for d in diffs):
                raise ValueError(f'{name}: numerical difference overflow')
            bad = sum(not abs(b - a) <= atol + rtol * abs(a) for a, b in pairs)
            metric = dict(n_finite=len(pairs), n_missing_ref=x.count(None),
                          n_missing_new=y.count(None), mask_mismatch_count=len(mismatches),
                          mask_mismatch_cells=[ids1[i] for i in mismatches[:20]],
                          outside_tolerance=bad,
                          max_abs_delta=max((abs(d) for d in diffs), default=None),
                          mean_signed_delta=math.fsum(diffs) / len(diffs) if diffs else None)
            out['columns'][name] = metric
            if mismatches or bad or not pairs:
                failures.append(name)
            if name == COLUMNS[0]:
                rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
                out['spearman_score_pair'] = rho
                if min_spearman is not None and (rho is None or rho < min_spearman):
                    failures.append('spearman_score_pair')
        for name in CATEGORICAL:
            count = sum(a[1][name] != b[1][name] for a, b in zip(rows1, rows2))
            out['columns'][name] = dict(mismatch_count=count,
                                       match_fraction=1.0 - count / len(rows1))
            if count:
                failures.append(name)
    out.update(passed=not failures, failures=failures)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ref', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('--label', default='cmp')
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--min-spearman', type=float)
    parser.add_argument('--byte-check', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        out = compare(args.ref, args.new, atol=args.atol, rtol=args.rtol,
                      byte_check=args.byte_check, min_spearman=args.min_spearman)
    except (OSError, ValueError, OverflowError) as exc:
        out = dict(passed=False, failures=['invalid_input'], error=str(exc))
    out['label'] = args.label
    text = json.dumps(out, indent=2, allow_nan=False)
    print(text)
    try:
        (args.output or Path(str(args.new) + '.cmp.json')).write_text(text + '\n', encoding='utf-8')
    except OSError as exc:
        print(f'FAIL: cannot write comparison report: {exc}')
        return 1
    return 0 if out['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
