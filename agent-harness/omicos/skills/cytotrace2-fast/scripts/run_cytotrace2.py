#!/usr/bin/env python3
"""Portable Omicos adapter for the cytotrace2-fast >=1.3.1 Rust CLI.

Standard library only. Native genes-by-cells TSV is passed through unchanged.
Full numeric validation is opt-in; no implicit normalization or ID rewriting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

COLUMNS = ["CytoTRACE2_Score", "CytoTRACE2_Potency", "CytoTRACE2_Relative",
           "preKNN_CytoTRACE2_Score", "preKNN_CytoTRACE2_Potency"]
DEFAULTS = {"binary": "cytotrace2", "seed": 14, "batch_size": 10000,
            "smooth_batch_size": 1000, "max_cores": 4, "timeout_seconds": None}
REQUIRED = {"input_path", "output_dir", "species", "expression_scale", "assets"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_info(path):
    stat = Path(path).stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_request(path):
    path = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(raw, dict), "request must be a JSON object")
    require(not (REQUIRED - raw.keys()), f"missing fields: {sorted(REQUIRED - raw.keys())}")
    require(not (raw.keys() - REQUIRED - DEFAULTS.keys()), "unknown request fields")
    request = {**DEFAULTS, **raw}
    for key in REQUIRED | {"binary"}:
        require(isinstance(request[key], str) and bool(request[key].strip()), f"{key} must be nonempty text")
    require(request["species"] in {"human", "mouse"}, "species must be human or mouse")
    require(request["expression_scale"] in {"counts", "CPM", "TPM"},
            "expression_scale must be counts, CPM or TPM; log/scaled data are unsupported")
    for key in ("seed", "batch_size", "smooth_batch_size", "max_cores"):
        value = request[key]
        require(type(value) is int and value >= (0 if key == "seed" else 1), f"invalid {key}")
    require(request["seed"] <= 2**32 - 1, "seed must be <= 2^32-1")
    timeout = request["timeout_seconds"]
    require(timeout is None or (type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0),
            "timeout_seconds must be null or a finite positive number")
    for key in ("input_path", "output_dir", "assets"):
        p = Path(request[key]).expanduser()
        request[key] = str((p if p.is_absolute() else path.parent / p).resolve())
    binary = request["binary"]
    if "/" in binary or "\\" in binary:
        p = Path(binary).expanduser()
        binary = str((p if p.is_absolute() else path.parent / p).resolve())
    request["binary"] = binary
    return request


def identifiers(values, kind):
    require(bool(values), f"no {kind} IDs")
    require(all(v and v.strip() == v and not any(c in v for c in '\t\r\n\"\x00') for v in values),
            f"{kind} IDs must be nonempty, unquoted and free of tabs/newlines or edge whitespace")
    require(len(set(values)) == len(values), f"duplicate {kind} IDs")


def inspect_input(path, full=False):
    """Check native header/first row, or stream every value when full=True."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        cells = stream.readline().rstrip("\r\n").split("\t")
        identifiers(cells, "cell")
        n_genes, genes, max_value = 0, set(), 0.0
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            require(len(fields) == len(cells) + 1,
                    "expected native TSV: header contains ONLY cell IDs, each data row gene + all cells")
            identifiers([fields[0]], "gene")
            require(fields[0] not in genes, "duplicate gene IDs")
            genes.add(fields[0])
            n_genes += 1
            if full:
                for token in fields[1:]:
                    require(token == token.strip() and bool(token), "empty or whitespace-padded expression value")
                    value = float(token)
                    require(math.isfinite(value) and value >= 0, "expression must be finite and nonnegative")
                    max_value = max(max_value, value)
            else:
                break
        require(n_genes > 0, "expression matrix has no gene rows")
    if full:
        require(max_value > 0, "all-zero expression matrix")
    return cells, {"cells": len(cells), "genes": n_genes if full else None,
                   "validation": "full" if full else "header_and_first_row",
                   "expression_scale_verified": False}


def runtime(request):
    binary = shutil.which(request["binary"])
    require(binary is not None, "Rust binary not found; build it once and set binary in the request")
    binary = str(Path(binary).resolve())
    probe = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15, check=True)
    version = probe.stdout.strip()
    match = re.fullmatch(r"cytotrace2-fast (\d+)\.(\d+)\.(\d+)", version)
    require(match is not None and tuple(map(int, match.groups())) >= (1, 3, 1),
            "requires cytotrace2-fast >=1.3.1 (not the upstream Python CLI)")
    assets = Path(request["assets"])
    names = ["MANIFEST.json", "features.txt", "background.npz", "map_human_alias.tsv",
             "map_human_ortholog.tsv", "map_mouse_alias.tsv"] + [f"model{i}.npz" for i in range(1, 20)]
    for name in names:
        p = assets / name
        require(p.is_file() and p.stat().st_size > 0, f"missing/empty asset: {p}")
    return {"binary": binary, "version": version, "binary_sha256": digest(binary),
            "assets_manifest_sha256": digest(assets / "MANIFEST.json"),
            "assets_check": "required_files_present; asset payload hashes not verified"}


def command(request, binary):
    args = [binary]
    mapping = [("input_path", "--input-path"), ("output_dir", "--output-dir"),
               ("assets", "--assets"), ("species", "--species"), ("seed", "--seed"),
               ("batch_size", "--batch-size"), ("smooth_batch_size", "--smooth-batch-size"),
               ("max_cores", "--max-cores")]
    for key, flag in mapping:
        args.extend([flag, str(request[key])])
    return args + ["--disable-plotting"]


def result_summary(path, cells):
    missing = {name: 0 for name in (COLUMNS[0], COLUMNS[2], COLUMNS[3])}
    potency = {}
    with Path(path).open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        require(header == [""] + COLUMNS, "unexpected result schema")
        count = 0
        for count, line in enumerate(stream, 1):
            row = line.rstrip("\r\n").split("\t")
            require(len(row) == 6 and count <= len(cells) and row[0] == cells[count - 1],
                    "result cell IDs/order or row width differ from input")
            for index, name in ((1, COLUMNS[0]), (3, COLUMNS[2]), (4, COLUMNS[3])):
                token = row[index]
                if token == "" or token.lower() == "nan":
                    missing[name] += 1
                else:
                    value = float(token)
                    require(math.isfinite(value) and -1e-8 <= value <= 1 + 1e-8,
                            f"invalid {name}: {token}")
            require(bool(row[2]) and bool(row[5]), "missing potency label")
            potency[row[2]] = potency.get(row[2], 0) + 1
        require(count == len(cells), "result cell count differs from input")
        require(missing[COLUMNS[0]] < count, "no finite final scores")
    return {"cells": count, "missing_numeric": missing, "potency_counts": potency,
            "warnings": ["Missing predictions are retained, not imputed."] if any(missing.values()) else []}


def execute(request, full=False):
    cells, info = inspect_input(request["input_path"], full=full)
    require(not (len(cells) > 30000 and request["batch_size"] >= len(cells)),
            "whole-input single batches above 30000 cells are outside this harness's support boundary; use bounded batches")
    require(not Path(request["output_dir"]).exists(), "output_dir already exists; use a fresh run directory")
    engine = runtime(request)
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "results_manifest.json"
    args = command(request, engine["binary"])
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(request["max_cores"])
    engine_env = {k: v for k, v in env.items() if k.startswith("C2RUST_") or k == "RAYON_NUM_THREADS"}
    record = {"schema_version": 1, "status": "running", "started_at": now(),
              "request": request, "input": file_info(request["input_path"]),
              "input_validation": info, "runtime": engine, "command": args,
              "engine_environment": engine_env,
              "artifacts": {"stdout": "stdout.log", "stderr": "stderr.log"}}
    start = time.monotonic()
    try:
        write_json(manifest_path, record)
        with (output / "stdout.log").open("w", encoding="utf-8") as stdout, \
                (output / "stderr.log").open("w", encoding="utf-8") as stderr:
            # Popen context waits after timeout/interrupt cleanup; never leave an orphan worker.
            with subprocess.Popen(args, stdout=stdout, stderr=stderr, env=env) as worker:
                try:
                    record["returncode"] = worker.wait(timeout=request["timeout_seconds"])
                except BaseException:
                    worker.terminate()
                    try:
                        worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
                    raise
        require(record["returncode"] == 0, f"Rust process exited {record['returncode']}; inspect stderr.log")
        summary = result_summary(output / "cytotrace2_results.txt", cells)
        require(file_info(request["input_path"]) == record["input"], "input changed during the run")
        write_json(output / "summary.json", summary)
        record["status"] = "completed_with_warnings" if summary["warnings"] else "completed"
        record["warnings"] = summary["warnings"]
        record["artifacts"].update(results="cytotrace2_results.txt", summary="summary.json")
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        record["finished_at"] = now()
        write_json(manifest_path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "validate", "doctor"):
        sub = subs.add_parser(action)
        sub.add_argument("--request", required=True)
        if action != "doctor":
            sub.add_argument("--full", action="store_true", help="stream all input values; can be expensive")
    status = subs.add_parser("status", help="read recorded state, not a liveness probe")
    status.add_argument("--output-dir", required=True)
    opts = parser.parse_args(argv)
    try:
        if opts.action == "status":
            result = json.loads((Path(opts.output_dir) / "results_manifest.json").read_text(encoding="utf-8"))
        else:
            request = load_request(opts.request)
            if opts.action == "run":
                result = execute(request, opts.full)
            elif opts.action == "doctor":
                result = runtime(request)
            else:
                _, result = inspect_input(request["input_path"], opts.full)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('{"status":"interrupted"}', file=sys.stderr)
        return 130


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
