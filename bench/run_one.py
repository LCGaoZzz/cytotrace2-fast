#!/usr/bin/env python3
"""cytotrace2-fast 基准 harness：包装二进制，采样 /proc，收集 STAGE_TIMINGS/SUB 行。
用法: python3 run_one.py --label <名> --tier <expr路径> --anno <注释路径> [--species mouse|human]
      [--bs N] [--sbs N] [--seed N] [--extra-env K=V ...] [--outdir DIR] [--max-cores N]
输出: <outdir>/<label>/<label>.json + .stdout + .stderr + .samples.csv
"""
import argparse, os, subprocess, sys, time, json, signal, threading, hashlib

def read_proc(pid):
    """返回 (utime+stime ticks, resident_pages, threads)，聚合 time 父进程与真正的子进程"""
    pids = [pid]
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            pids += [int(x) for x in f.read().split()]
    except Exception:
        pass
    tot_cpu, tot_res, tot_thr = 0, 0, 0
    seen = False
    for p in pids:
        try:
            with open(f"/proc/{p}/stat", "rb") as f:
                raw = f.read()
            rp = raw.rfind(b")")
            parts = raw[rp+2:].split()
            # parts 索引：stat 字段号 = 索引+3（state 是字段3 → parts[0]）
            tot_cpu += int(parts[11]) + int(parts[12])   # utime(14), stime(15)
            tot_thr += int(parts[17])                    # threads(20)
            with open(f"/proc/{p}/statm") as f:
                tot_res = max(tot_res, int(f.read().split()[1]))
            seen = True
        except Exception:
            continue
    return (tot_cpu, tot_res, tot_thr) if seen else (None, None, None)

CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--anno", required=True)
    ap.add_argument("--species", default="mouse")
    ap.add_argument("--bs", type=int, default=10000)
    ap.add_argument("--sbs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-cores", type=int, default=None)
    ap.add_argument("--extra-env", action="append", default=[])
    ap.add_argument("--binary", default="/data/users/lianchong/workspace3/cytotrace2-fast/target/release/cytotrace2")
    ap.add_argument("--time-sub", action="store_true", default=True)
    a = ap.parse_args()

    run_dir = os.path.join(a.outdir, a.label)
    os.makedirs(run_dir, exist_ok=True)
    out_results = os.path.join(run_dir, "cytotrace2_results")

    env = dict(os.environ)
    env["C2RUST_TIME_SUB"] = "1"
    for kv in a.extra_env:
        k, v = kv.split("=", 1)
        env[k] = v

    cmd = ["/usr/bin/time", "-v", "-o", os.path.join(run_dir, "time_v.txt"),
           a.binary,
           "-f", a.tier, "-a", a.anno, "-sp", a.species, "--seed", str(a.seed),
           "-bs", str(a.bs), "-sbs", str(a.sbs), "-o", out_results]
    if a.max_cores:
        cmd += ["-mc", str(a.max_cores)]

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    samples = []
    while proc.poll() is None:
        cpu, resident, threads = read_proc(proc.pid)
        if cpu is not None:
            samples.append((time.time() - t0, cpu / CLK_TCK, resident * PAGE / 2**30, threads))
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    wall = time.time() - t0

    # 采样派生量
    import statistics
    cpu_cores_series = []
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i-1][0]
        if dt > 0:
            cpu_cores_series.append((samples[i][0], (samples[i][1] - samples[i-1][1]) / dt))
    peak_rss = max((s[2] for s in samples), default=0.0)
    # 稳态窗口：连续 60s 以上 CPU 核数波动 <0.5 的平台段
    plateaus = []
    if cpu_cores_series:
        i = 0
        while i < len(cpu_cores_series):
            j = i
            while j + 1 < len(cpu_cores_series) and abs(cpu_cores_series[j+1][1] - cpu_cores_series[i][1]) < 0.5:
                j += 1
            dur = cpu_cores_series[j][0] - cpu_cores_series[i][0]
            if dur >= 30:
                plateaus.append({"t_start": round(cpu_cores_series[i][0],1),
                                 "t_end": round(cpu_cores_series[j][0],1),
                                 "dur_s": round(dur,1),
                                 "cores": round(statistics.median([c for _,c in cpu_cores_series[i:j+1]]),2)})
            i = j + 1

    stage_timings, sub_lines = None, []
    for line in stdout.splitlines():
        if line.startswith("STAGE_TIMINGS "):
            body = line[len("STAGE_TIMINGS "):].strip().strip("{}")
            stage_timings = {}
            for part in body.split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    try: stage_timings[k.strip().strip('"')] = float(v)
                    except ValueError: pass
    for line in stderr.splitlines():
        if line.startswith("SUB "):
            sub_lines.append(line[4:])

    with open(os.path.join(run_dir, "samples.csv"), "w") as f:
        f.write("t_s,cpu_ticks_cum,rss_gib,threads\n")
        for s in samples:
            f.write(f"{s[0]:.2f},{s[1]},{s[2]:.4f},{s[3]}\n")

    res_file = os.path.join(out_results, "cytotrace2_results.txt")
    rec = {"label": a.label, "tier": a.tier, "species": a.species, "bs": a.bs, "sbs": a.sbs,
           "seed": a.seed, "extra_env": dict(kv.split("=",1) for kv in a.extra_env),
           "binary": a.binary,
           "binary_sha256": hashlib.sha256(open(a.binary, "rb").read()).hexdigest(),
           "wall_s": round(wall, 2), "rc": proc.returncode,
           "peak_rss_gib_samples": round(peak_rss, 2),
           "mean_cores_lifetime": round(cpu_cores_series and statistics.mean(c for _,c in cpu_cores_series) or 0, 2),
           "plateaus": plateaus, "stage_timings": stage_timings, "sub_lines": sub_lines,
           "results_file": res_file if os.path.exists(res_file) else None}
    # /usr/bin/time -v 的 peak RSS
    try:
        tv = open(os.path.join(run_dir, "time_v.txt")).read()
        for line in tv.splitlines():
            if "Maximum resident set size" in line:
                rec["peak_rss_gib_time"] = round(int(line.split(":")[1].strip()) / 2**20, 2)
    except Exception: pass
    with open(os.path.join(run_dir, f"{a.label}.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    with open(os.path.join(run_dir, "stdout.txt"), "w") as f: f.write(stdout)
    with open(os.path.join(run_dir, "stderr.txt"), "w") as f: f.write(stderr)
    print(json.dumps({k: rec[k] for k in ("label","wall_s","rc","peak_rss_gib_time","mean_cores_lifetime")}, ensure_ascii=False))
    print("plateaus:", json.dumps(plateaus, ensure_ascii=False))
    if stage_timings: print("stages:", json.dumps(stage_timings, ensure_ascii=False))

if __name__ == "__main__":
    main()
