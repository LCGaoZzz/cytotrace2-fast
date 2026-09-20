#!/usr/bin/env python3
"""Run one benchmark; preserve the child failure code and never reuse run output.

Logs go straight to files (not undrained pipes). GNU time supplies child wall,
CPU time and peak RSS; /proc samples are supplementary snapshots. No input
matrix is hashed or reread here. Binary and result SHA-256 identify artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_proc(pid: int):
    pids = [pid]
    try:
        pids.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
    except OSError:
        pass
    cpu = rss = threads = 0
    seen = False
    for child in pids:
        try:
            raw = Path(f"/proc/{child}/stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            cpu += int(fields[11]) + int(fields[12])
            threads += int(fields[17])
            # Maximum process RSS snapshot, NOT aggregate unique process-tree RSS.
            rss = max(rss, int(Path(f"/proc/{child}/statm").read_text().split()[1]))
            seen = True
        except (OSError, ValueError, IndexError):
            continue
    if not seen:
        return None
    return cpu / os.sysconf("SC_CLK_TCK"), rss * os.sysconf("SC_PAGE_SIZE") / 2**30, threads


def parse_time(path: Path) -> dict:
    # GNU time may prepend an error line on child failure. Its final line is ours.
    line = path.read_text().splitlines()[-1]
    result = json.loads(line)
    for key in ("wall_s", "user_s", "system_s", "peak_rss_kib", "exit_code"):
        if key not in result:
            raise ValueError(f"GNU time did not report {key}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tier", required=True, type=Path)
    parser.add_argument("--anno", required=True, type=Path)
    parser.add_argument("--species", choices=("mouse", "human"), default="mouse")
    parser.add_argument("--bs", type=int, default=10000)
    parser.add_argument("--sbs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--max-cores", type=int)
    parser.add_argument("--extra-env", action="append", default=[])
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "target/release/cytotrace2")
    parser.add_argument("--time-sub", action="store_true", default=True)
    args = parser.parse_args()
    if not args.label or args.label in (".", "..") or Path(args.label).name != args.label:
        parser.error("label must be a single nonempty directory name")
    if args.bs < 1 or args.sbs < 1 or (args.max_cores is not None and args.max_cores < 1):
        parser.error("batch sizes and worker limits must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be in [0, 2^32-1]")
    extra = {}
    for item in args.extra_env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error("--extra-env requires KEY=VALUE")
        extra[key] = value
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"binary is not executable: {binary}")
    timer = Path("/usr/bin/time")
    if not timer.is_file():
        parser.error("GNU /usr/bin/time is required")
    for source in (args.tier, args.anno):
        if not source.is_file():
            parser.error(f"input file does not exist: {source}")
    directory = args.outdir.resolve() / args.label
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"refusing to reuse run directory: {directory}")
    result_path = directory / "cytotrace2_results/cytotrace2_results.txt"
    time_path = directory / "time.json.txt"
    environment = dict(os.environ)
    environment.update(extra)
    environment["LC_ALL"] = "C"
    environment["C2RUST_TIME_SUB"] = "1"
    time_format = '{"wall_s":%e,"user_s":%U,"system_s":%S,"peak_rss_kib":%M,"exit_code":%x}'
    command = [str(timer), "-f", time_format, "-o", str(time_path), str(binary),
               "-f", str(args.tier.resolve()), "-a", str(args.anno.resolve()),
               "-sp", args.species, "--seed", str(args.seed), "-bs", str(args.bs),
               "-sbs", str(args.sbs), "-o", str(result_path.parent)]
    if args.max_cores is not None:
        command.extend(("-mc", str(args.max_cores)))
    binary_hash = sha256(binary)
    started = time.monotonic()
    samples = []
    with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=environment,
                                   start_new_session=True)
        try:
            while True:
                sample = read_proc(process.pid)
                if sample is not None:
                    samples.append((time.monotonic() - started, *sample))
                try:
                    process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    wrapper_wall = time.monotonic() - started
    failures = []
    try:
        measured = parse_time(time_path)
    except (OSError, ValueError, IndexError) as exc:
        measured = {}
        failures.append(f"invalid timing report: {exc}")
    if process.returncode != 0:
        failures.append(f"child returned {process.returncode}")
    if not result_path.is_file() or result_path.stat().st_size == 0:
        failures.append("successful nonempty results file not produced")
    if sha256(binary) != binary_hash:
        failures.append("binary changed during the run")
    stage_timings = None
    for line in (directory / "stdout.txt").read_text(errors="replace").splitlines():
        if line.startswith("STAGE_TIMINGS "):
            try:
                stage_timings = json.loads(line[len("STAGE_TIMINGS "):])
            except ValueError:
                failures.append("malformed STAGE_TIMINGS JSON")
    sub_lines = [line[4:] for line in (directory / "stderr.txt").read_text(errors="replace").splitlines()
                 if line.startswith("SUB ")]
    with (directory / "samples.csv").open("w") as handle:
        handle.write("t_s,cpu_s_cum,sampled_max_process_rss_gib,threads\n")
        for sample in samples:
            handle.write(",".join(map(str, sample)) + "\n")
    wall = measured.get("wall_s")
    report = {
        "schema_version": 1, "label": args.label, "binary": str(binary),
        "binary_sha256": binary_hash, "command": command,
        "tier": str(args.tier.resolve()), "anno": str(args.anno.resolve()),
        "species": args.species, "bs": args.bs, "sbs": args.sbs, "seed": args.seed,
        "max_cores": args.max_cores, "extra_env": extra,
        "wall_s": wall, "wrapper_wall_s": wrapper_wall, "rc": process.returncode,
        "peak_rss_gib_time": measured.get("peak_rss_kib", 0) / 2**20 if measured else None,
        "mean_cores_lifetime": ((measured["user_s"] + measured["system_s"]) / wall) if wall else None,
        "stage_timings": stage_timings, "sub_lines": sub_lines,
        "results_file": str(result_path) if result_path.is_file() else None,
        "results_sha256": sha256(result_path) if result_path.is_file() else None,
        "passed": not failures, "failures": failures,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (directory / (args.label + ".json")).write_text(payload)
    print(payload, end="")
    if process.returncode:
        return process.returncode if process.returncode > 0 else 128 - process.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
