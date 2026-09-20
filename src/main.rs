mod fmtutil;
mod knn;
mod npsum;
mod npyio;
mod pipeline;
mod rng;

use pipeline::LABELS;
use rayon::prelude::*;
use rng::Mt19937;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn mark(timings: &mut Vec<(String, f64)>, name: &str, t: Instant) {
    timings.push((name.to_string(), t.elapsed().as_secs_f64()));
}

struct Args {
    input_path: String,
    annotation_path: String,
    species: String,
    batch_size: usize,
    smooth_batch_size: usize,
    disable_parallelization: bool,
    max_cores: Option<usize>,
    seed: i64,
    output_dir: String,
    disable_plotting: bool,
    disable_verbose: bool,
    dump_stage: Option<String>,
    assets: String,
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn default_assets_dir() -> String {
    // Prefer an explicit environment override so packaged binaries can keep
    // the model bundle outside the executable directory.
    if let Ok(value) = std::env::var("CYTOTRACE2_ASSETS") {
        if !value.trim().is_empty() {
            return value;
        }
    }

    // A checkout is usable directly from its root (`cargo run -- ...`) and a
    // copied binary is usable when `assets/` sits next to the distribution
    // root.  Keep the search runtime-relative; never bake a developer's
    // machine path into a public binary.
    let mut candidates = Vec::new();
    candidates.push(PathBuf::from("assets"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            candidates.push(bin_dir.join("assets"));
            if let Some(target_dir) = bin_dir.parent() {
                if let Some(repo_dir) = target_dir.parent() {
                    candidates.push(repo_dir.join("assets"));
                }
            }
        }
    }
    candidates
        .into_iter()
        .find(|p| p.join("MANIFEST.json").is_file())
        .unwrap_or_else(|| PathBuf::from("assets"))
        .to_string_lossy()
        .into_owned()
}

fn usage_err(msg: &str) -> ! {
    eprintln!("cytotrace2: {}", msg);
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args {
        input_path: String::new(),
        annotation_path: String::new(),
        species: "mouse".to_string(),
        batch_size: 10000,
        smooth_batch_size: 1000,
        disable_parallelization: false,
        max_cores: None,
        seed: 14,
        output_dir: "cytotrace2_results".to_string(),
        disable_plotting: false,
        disable_verbose: false,
        dump_stage: None,
        assets: default_assets_dir(),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let t = argv[i].clone();
        let (flag, inline_val) = if let Some(eqpos) = t.find('=') {
            if t.starts_with("--") {
                (t[..eqpos].to_string(), Some(t[eqpos + 1..].to_string()))
            } else {
                (t.clone(), None)
            }
        } else {
            (t.clone(), None)
        };
        let mut next = |what: &str| -> String {
            if let Some(v) = inline_val.clone() {
                return v;
            }
            i += 1;
            if i >= argv.len() {
                usage_err(&format!("expected value for {}", what));
            }
            argv[i].clone()
        };
        match flag.as_str() {
            "-f" | "--input-path" => a.input_path = next("input-path"),
            "-a" | "--annotation-path" => a.annotation_path = next("annotation-path"),
            "-sp" | "--species" => {
                a.species = next("species");
                if a.species != "mouse" && a.species != "human" {
                    usage_err("species must be mouse or human");
                }
            }
            "-bs" | "--batch-size" => a.batch_size = next("batch-size").parse().unwrap(),
            "-sbs" | "--smooth-batch-size" => {
                a.smooth_batch_size = next("smooth-batch-size").parse().unwrap()
            }
            "-dpa" | "--disable-parallelization" => a.disable_parallelization = true,
            "-mc" | "--max-cores" => a.max_cores = Some(next("max-cores").parse().unwrap()),
            "-r" | "--seed" => a.seed = next("seed").parse().unwrap(),
            "-o" | "--output-dir" => a.output_dir = next("output-dir"),
            "-dpl" | "--disable-plotting" => a.disable_plotting = true,
            "-dv" | "--disable-verbose" => a.disable_verbose = true,
            "--dump-stage" => a.dump_stage = Some(next("dump-stage")),
            "--assets" => a.assets = next("assets"),
            "-h" | "--help" => {
                println!("cytotrace2-fast {VERSION} — a Rust implementation of the CytoTRACE 2 CLI");
                println!("Usage: cytotrace2 -f EXPRESSION -a ANNOTATIONS [options]");
                println!("Run `cytotrace2 --help` with the official package for the upstream CLI reference.");
                std::process::exit(0);
            }
            "--version" => {
                println!("cytotrace2-fast {VERSION}");
                std::process::exit(0);
            }
            other => usage_err(&format!("unrecognized argument {}", other)),
        }
        i += 1;
    }
    if a.input_path.is_empty() {
        usage_err("the -f/--input-path argument is required");
    }
    if a.seed < 0 || a.seed > u32::MAX as i64 {
        usage_err("seed must be in [0, 2^32-1]");
    }
    a
}

fn load_lines(p: &Path) -> Vec<String> {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e));
    s.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn load_tsv_dict(p: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for l in load_lines(p) {
        let mut it = l.splitn(2, '\t');
        let k = it.next().unwrap_or("").to_string();
        let v = it.next().unwrap_or("").to_string();
        m.insert(k, v);
    }
    m
}

fn parse_manifest_model_order(p: &Path) -> Vec<String> {
    let s = std::fs::read_to_string(p).expect("MANIFEST.json");
    let key = "\"model_order\"";
    let ki = s.find(key).expect("model_order in manifest") + key.len();
    let rest = &s[ki..];
    let start = rest.find('[').unwrap() + 1;
    let end = rest.find(']').unwrap();
    rest[start..end]
        .split(',')
        .map(|t| t.trim().trim_matches('"').to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn dump_f64(d: &Option<PathBuf>, name: &str, shape: &[usize], data: &[f64]) {
    if let Some(dir) = d {
        npyio::write_npy_f64(&dir.join(name), shape, data);
    }
}
fn dump_f32(d: &Option<PathBuf>, name: &str, shape: &[usize], data: &[f32]) {
    if let Some(dir) = d {
        npyio::write_npy_f32(&dir.join(name), shape, data);
    }
}
fn dump_i64(d: &Option<PathBuf>, name: &str, shape: &[usize], data: &[i64]) {
    if let Some(dir) = d {
        npyio::write_npy_i64(&dir.join(name), shape, data);
    }
}

fn main() {
    let t_start = Instant::now();
    let args = parse_args();
    let mut timings: Vec<(String, f64)> = Vec::new();

    if !args.disable_verbose {
        println!("cytotrace2: Input parameters");
        println!("    Input file: {}", args.input_path);
        println!("    Species: {}", args.species);
        println!(
            "    Parallelization enabled: {}",
            !args.disable_parallelization
        );
        println!("    Batch size: {}", args.batch_size);
        println!("    Smoothing batch size: {}", args.smooth_batch_size);
        println!("    Seed: {}", args.seed);
        println!("    Output directory: {}", args.output_dir);
        println!("    Plotting enabled: {}", !args.disable_plotting);
    }
    if args.disable_plotting {
        // MVP: plotting not implemented; -dpl accepted
    }

    let dump_dir: Option<PathBuf> = args.dump_stage.as_ref().map(|d| {
        std::fs::create_dir_all(d).unwrap();
        PathBuf::from(d)
    });

    // ---------- background asset thread (r6 micro-opt) ----------
    // background.npz read + dense B^T + CSR, then the 19 model npz loads
    // (rayon-parallel across files), all on a background thread overlapping
    // io_parse / preprocessing. The thread only produces the same BtCsr /
    // ModelWeights values the sequential path built — RNG stream, asset
    // bytes and all downstream numerics are untouched.
    let assets = PathBuf::from(&args.assets);
    let model_order = parse_manifest_model_order(&assets.join("MANIFEST.json"));
    let bg_assets = args.assets.clone();
    let bg_model_order = model_order.clone();
    let bg_spawn = Instant::now();
    // r8 io-polish: the asset preload is split into two INDEPENDENT chains
    // that run concurrently (and off the rayon global pool, so they no longer
    // contend with io_parse's parallel tokenizer):
    //   A) background.npz (parallel pread) -> B^T CSR built directly from the
    //      COO triples; the CSR is the identical (col, val) stream the
    //      dense-build + row-scan produced before
    //   B) the 19 model npz loads, one std thread per file (concurrent reads)
    // Both produce the same BtCsr / ModelWeights values the single chained
    // thread built — RNG stream, asset bytes and downstream numerics are
    // untouched.
    let a_assets = bg_assets.clone();
    let h_bt: std::thread::JoinHandle<(pipeline::BtCsr, f64)> = std::thread::spawn(move || {
        let t0 = Instant::now();
        let t_r = Instant::now();
        let raw = pipeline::pread_all(&Path::new(&a_assets).join("background.npz"), 8)
            .expect("read background.npz");
        let t_read = t_r.elapsed().as_secs_f64();
        let t_n = Instant::now();
        let bt_entries = npyio::read_npz_bytes(raw);
        let t_npz = t_n.elapsed().as_secs_f64();
        let bt_csr = pipeline::background_csr_from_coo(&bt_entries);
        if std::env::var("C2RUST_TIME_SUB").is_ok() {
            eprintln!(
                "SUB bga {{\"pread_s\":{:.4},\"npz_s\":{:.4}}}",
                t_read, t_npz
            );
        }
        (bt_csr, t0.elapsed().as_secs_f64())
    });
    let h_models: std::thread::JoinHandle<(Vec<pipeline::ModelWeights>, f64)> =
        std::thread::spawn(move || {
            let t1 = Instant::now();
            let models: Vec<pipeline::ModelWeights> = std::thread::scope(|s| {
                let handles: Vec<_> = bg_model_order
                    .iter()
                    .map(|mname| {
                        let p = Path::new(&bg_assets)
                            .join(format!("{}.npz", mname.trim_end_matches(".pt")));
                        s.spawn(move || {
                            let entries = npyio::read_npz(&p);
                            pipeline::load_model(&entries)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("model load thread panicked"))
                    .collect()
            });
            (models, t1.elapsed().as_secs_f64())
        });

    // ---------- read input ----------
    let t = Instant::now();
    let input = pipeline::read_matrix(Path::new(&args.input_path));
    mark(&mut timings, "io_parse", t);
    let n_cells = input.cell_names.len();
    let n_genes_in = input.gene_names.len();
    if !args.disable_verbose {
        println!("cytotrace2: Dataset characteristics");
        println!("    Number of input genes:  {}", n_genes_in);
        println!("    Number of input cells:  {}", n_cells);
    }
    if let Some(d) = &dump_dir {
        std::fs::write(d.join("cell_names.txt"), input.cell_names.join("\n") + "\n").unwrap();
    }

    // log2 heuristic warning
    // micro-opt: chunked parallel max — each chunk folds from 0.0 with the same
    // `v > acc` comparison, so the result is identical to the serial scan
    // (NaN tokens are skipped exactly as before; the seed stays 0.0).
    {
        let mx = input
            .x
            .par_chunks(1 << 20)
            .map(|ch| {
                ch.iter()
                    .cloned()
                    .fold(0.0f64, |acc, v| if v > acc { v } else { acc })
            })
            .reduce(|| 0.0f64, |acc, v| if v > acc { v } else { acc });
        if mx <= 20.0 {
            eprintln!("Input expression data seem to be log2-transformed. Please provide data as raw counts or CPM/TPM.");
        }
    }

    // ---------- assets ----------
    // (background.npz and the 19 model npz files moved to the background
    // thread spawned above; only the small text assets stay here)
    let t = Instant::now();
    let features = load_lines(&assets.join("features.txt"));
    assert_eq!(features.len(), pipeline::N_FEATURES);
    let feat_set: HashSet<String> = features.iter().cloned().collect();
    let mouse_alias = load_tsv_dict(&assets.join("map_mouse_alias.tsv"));
    let human_ortho = load_tsv_dict(&assets.join("map_human_ortholog.tsv"));
    let human_alias = load_tsv_dict(&assets.join("map_human_alias.tsv"));
    mark(&mut timings, "asset_load_meta", t);

    // ---------- batch logic (official semantics) ----------
    let num_proc = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let mut batch_size = args.batch_size;
    if batch_size > n_cells {
        println!("cytotrace2: The passed batch_size is greater than the number of cells in the subsample. \n    Now setting batch_size to {}.", n_cells);
        // official: batch_size <- len(expression) is a no-op in python
    } else if n_cells > 50000 && batch_size > 50000 {
        println!("cytotrace2: Please consider reducing the batch_size to 50000 for runtime and memory efficiency.");
    }
    let mut chunk_number = (n_cells + batch_size - 1) / batch_size;
    let mut smooth_chunk_number = (batch_size + args.smooth_batch_size - 1) / args.smooth_batch_size;
    if n_cells < 1000 {
        chunk_number = 1;
        smooth_chunk_number = 1;
    }
    // calculate cores
    let (pred_cores, smooth_cores) = if smooth_chunk_number == 1 {
        println!("cytotrace2: The number of cells in your dataset is less than the specified smoothing batch size.\n");
        println!("    Model prediction will not be parallelized.");
        (1usize, 1usize)
    } else if args.disable_parallelization {
        (1usize, 1usize)
    } else {
        let half = std::cmp::max(1, num_proc / 2);
        match args.max_cores {
            None => {
                let p = std::cmp::max(1, half);
                let s = std::cmp::min(smooth_chunk_number, p);
                println!("cytotrace2: Running {} prediction batch(es) sequentially using {} cores per batch.", chunk_number, s);
                (p, s)
            }
            Some(mc) => {
                let mc = std::cmp::min(mc, half);
                (
                    std::cmp::min(chunk_number, mc),
                    std::cmp::min(smooth_chunk_number, mc),
                )
            }
        }
    };

    // ---------- background dense (joined from the background thread) ----------
    // The dense B^T + CSR build and the model loads ran concurrently with
    // io_parse / preprocessing; the marks below carry the thread's own
    // internal wall times (t_bt, t_models) plus the main-side span, so the
    // overlap is visible in the stage timings.
    let (bt_csr, t_bt_s) = h_bt.join().expect("background asset thread panicked");
    let (models, t_models_s) = h_models.join().expect("model load thread panicked");
    let bg_span_s = bg_spawn.elapsed().as_secs_f64();
    timings.push(("background_dense".to_string(), t_bt_s));
    timings.push(("model_load".to_string(), t_models_s));
    timings.push(("bg_thread_total_s".to_string(), t_bt_s.max(t_models_s)));
    timings.push(("bg_span_wall_s".to_string(), bg_span_s));

    // ---------- top-level RNG: seed ----------
    let mut rng = Mt19937::new(args.seed as u32);
    let mut top_perm: Vec<u32> = (0..n_cells as u32).collect();
    if chunk_number > 1 {
        rng.shuffle_indices(&mut top_perm);
    }
    let starts = split_starts(n_cells, chunk_number);

    let mut batch_results: Vec<BatchOut> = Vec::new();
    for bi in 0..chunk_number {
        let s = starts[bi];
        let e = starts[bi + 1];
        println!(
            "cytotrace2: Initiated processing batch {}/{} with {} cells",
            bi + 1,
            chunk_number,
            e - s
        );
        let sel: Vec<usize> = top_perm[s..e].iter().map(|&v| v as usize).collect();
        let out = process_batch(
            &args,
            &input,
            &sel,
            &features,
            &feat_set,
            &mouse_alias,
            &human_ortho,
            &human_alias,
            &models,
            &bt_csr,
            &mut rng,
            &dump_dir,
            smooth_cores,
            pred_cores,
            &mut timings,
        );
        batch_results.push(out);
    }

    // ---------- assemble (original cell order) ----------
    let t = Instant::now();
    let mut final_score = vec![f64::NAN; n_cells];
    let mut final_preknn = vec![f64::NAN; n_cells];
    let mut final_preknn_pot = vec![0usize; n_cells];
    for out in batch_results {
        for (ci, &g) in out.global_idx.iter().enumerate() {
            final_score[g] = out.score[ci];
            final_preknn[g] = out.binned[ci];
            final_preknn_pot[g] = out.potency[ci];
        }
    }
    // final potency via pd.cut + relative min-max
    let mut final_potency = vec![String::new(); n_cells];
    for i in 0..n_cells {
        final_potency[i] = match knn::cut_potency(final_score[i]) {
            Some(k) => LABELS[k].to_string(),
            None => String::new(),
        };
    }
    let mn = final_score.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = final_score.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let final_relative: Vec<f64> = final_score.iter().map(|&v| (v - mn) / (mx - mn)).collect();
    mark(&mut timings, "final_assemble", t);

    if let Some(d) = &dump_dir {
        dump_f64(&dump_dir, "final_score.npy", &[n_cells], &final_score);
        dump_f64(&dump_dir, "final_relative.npy", &[n_cells], &final_relative);
        dump_f64(&dump_dir, "final_preknn_score.npy", &[n_cells], &final_preknn);
        std::fs::write(d.join("final_potency.txt"), final_potency.join("\n") + "\n").unwrap();
        std::fs::write(
            d.join("final_preknn_potency.txt"),
            final_preknn_pot
                .iter()
                .map(|&k| LABELS[k])
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .unwrap();
    }

    // ---------- write output ----------
    let t = Instant::now();
    std::fs::create_dir_all(&args.output_dir).expect("mkdir output");
    let mut buf = String::with_capacity(n_cells * 110);
    buf.push_str("\tCytoTRACE2_Score\tCytoTRACE2_Potency\tCytoTRACE2_Relative\tpreKNN_CytoTRACE2_Score\tpreKNN_CytoTRACE2_Potency\n");
    for i in 0..n_cells {
        buf.push_str(&input.cell_names[i]);
        buf.push('\t');
        buf.push_str(&fmtutil::fmt_py(final_score[i]));
        buf.push('\t');
        buf.push_str(&final_potency[i]);
        buf.push('\t');
        buf.push_str(&fmtutil::fmt_py(final_relative[i]));
        buf.push('\t');
        buf.push_str(&fmtutil::fmt_py(final_preknn[i]));
        buf.push('\t');
        buf.push_str(LABELS[final_preknn_pot[i]]);
        buf.push('\n');
    }
    let outp = Path::new(&args.output_dir).join("cytotrace2_results.txt");
    let mut f = std::fs::File::create(&outp).unwrap();
    f.write_all(buf.as_bytes()).unwrap();
    mark(&mut timings, "output_write", t);

    println!("cytotrace2: Plotting disabled");
    println!("cytotrace2: Finished.");
    let total = t_start.elapsed().as_secs_f64();
    // stage timings JSON line
    let parts: Vec<String> = timings
        .iter()
        .map(|(k, v)| format!("\"{}\":{}", k, v))
        .collect();
    println!("STAGE_TIMINGS {{{},\"total\":{}}}", parts.join(","), total);
}

fn split_starts(n: usize, chunks: usize) -> Vec<usize> {
    let base = n / chunks;
    let rem = n % chunks;
    let mut st = Vec::with_capacity(chunks + 1);
    let mut acc = 0usize;
    for c in 0..chunks {
        st.push(acc);
        acc += base + if c < rem { 1 } else { 0 };
    }
    st.push(acc);
    st
}

struct BatchOut {
    global_idx: Vec<usize>,
    score: Vec<f64>,
    binned: Vec<f64>,
    potency: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn process_batch(
    args: &Args,
    input: &pipeline::InputMatrix,
    sel: &[usize],
    features: &[String],
    feat_set: &HashSet<String>,
    mouse_alias: &HashMap<String, String>,
    human_ortho: &HashMap<String, String>,
    human_alias: &HashMap<String, String>,
    models: &[pipeline::ModelWeights],
    bt_csr: &pipeline::BtCsr,
    rng: &mut Mt19937,
    dump_dir: &Option<PathBuf>,
    smooth_cores: usize,
    pred_cores: usize,
    timings: &mut Vec<(String, f64)>,
) -> BatchOut {
    let nb = sel.len();
    let n_genes_all = input.gene_names.len();
    // ---- preprocess: species mapping on the batch's genes ----
    let t = Instant::now();
    // human path: ortholog + alias mapping, then pandas-like duplicate column drop
    // (keep first occurrence of each mapped name; NaN columns dropped downstream)
    let mapped_names: Vec<String> = if args.species == "human" {
        let opt = pipeline::apply_human_map(&input.gene_names, human_ortho, human_alias);
        let mut seen: HashSet<String> = HashSet::new();
        opt.iter()
            .map(|o| match o {
                Some(name) => {
                    if seen.insert(name.clone()) {
                        name.clone()
                    } else {
                        String::new() // dropped duplicate (MVP approximation of official idx-drop)
                    }
                }
                None => String::new(),
            })
            .collect()
    } else {
        pipeline::apply_mouse_alias(&input.gene_names, mouse_alias, feat_set)
    };
    // ---- align to features ----
    let f = pipeline::N_FEATURES;
    let mut aligned = vec![0.0f64; nb * f];
    {
        let mut idx: HashMap<&str, usize> = HashMap::with_capacity(n_genes_all);
        for (g, name) in mapped_names.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            if idx.insert(name.as_str(), g).is_some() {
                panic!("duplicate gene after mapping: {} (unsupported in MVP)", name);
            }
        }
        // micro-opt: resolve each feature's input column once (14271 map
        // lookups) instead of once per (cell, feature) — each aligned cell row
        // then gathers input.x[g0 * n_genes_all + col] for exactly the same
        // (row, feature) slots, bit-identical to the per-cell map lookups.
        let col_of_feat: Vec<i32> = features
            .iter()
            .map(|feat| idx.get(feat.as_str()).map(|&c| c as i32).unwrap_or(-1))
            .collect();
        drop(idx);
        aligned.par_chunks_mut(f).enumerate().for_each(|(r, orow)| {
            let src = &input.x[sel[r] * n_genes_all..(sel[r] + 1) * n_genes_all];
            for (j, &c) in col_of_feat.iter().enumerate() {
                if c >= 0 {
                    orow[j] = src[c as usize];
                }
            }
        });
    }
    let n_in_features = {
        let s: HashSet<&str> = mapped_names
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str())
            .collect();
        features.iter().filter(|x| s.contains(x.as_str())).count()
    };
    println!(
        "    {} input genes are present in the model features.",
        n_in_features
    );
    mark(timings, "preprocess_align", t);

    let t = Instant::now();
    // r8 io-polish: one fused sweep produces denom, log2d, the f32 rank
    // matrix (and the f64 ranks only when a dump needs them) plus the QC
    // positive-gene counts — previously three full passes over the 325 MB
    // aligned matrix. Values and per-element op order are unchanged.
    let cr = pipeline::cpm_log2_rank(&aligned, nb, dump_dir.is_some());
    let denom = cr.denom;
    let log2d = cr.log2d;
    let rank32 = cr.rank32;
    let rank64 = cr.rank64;
    let pos_counts = cr.pos_counts;
    mark(timings, "preprocess_cpm_rank", t);
    if let Some(d) = dump_dir {
        dump_f64(dump_dir, "aligned_expression.npy", &[nb, f], &aligned);
        dump_f64(dump_dir, "cpm_denom.npy", &[nb], &denom);
        dump_f64(dump_dir, "log2_data.npy", &[nb, f], &log2d);
        if let Some(rd) = &rank64 {
            dump_f64(dump_dir, "rank_data.npy", &[nb, f], rd);
        }
    }
    drop(aligned);

    // QC gene counts (fused count of produced log2d values > 0 per row —
    // identical to the previous post-hoc scan)
    {
        let lowc = pos_counts.iter().filter(|&&c| c < 500).count();
        if (lowc as f64) / nb as f64 >= 0.2 {
            eprintln!(
                "cytotrace2: {:.1}% of input cells express fewer than 500 genes.",
                100.0 * lowc as f64 / nb as f64
            );
        }
    }

    // ---- top var genes ----
    let t = Instant::now();
    let disp = pipeline::dispersion(&log2d, nb);
    let top_col = pipeline::top_var_genes(&disp);
    mark(timings, "top_var_genes", t);
    if let Some(_) = dump_dir {
        dump_f64(dump_dir, "dispersion.npy", &[f], &disp);
        dump_i64(
            dump_dir,
            "top_col_inds.npy",
            &[1000],
            &top_col.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        );
    }

    // ---- ensemble inference ----
    // r6 micro-opt: the 19 model npz files are preloaded on the background
    // thread in main() (rayon-parallel across files, overlapping io_parse);
    // the model_load stage time is reported there. The f64->f32 conversions
    // below are now rayon-parallel per-element casts (same values, same order
    // — collect over par_iter preserves sequence order).
    let t = Instant::now();
    // xrank32 is produced directly by the fused preprocess pass (same f64
    // group-average -> f32 cast the separate conversion performed); only the
    // log2d cast remains here.
    let xrank32: Vec<f32> = rank32;
    let xlog32: Vec<f32> = log2d.par_iter().map(|&v| v as f32).collect();
    drop(rank64);
    mark(timings, "f32_convert", t);
    let t = Instant::now();
    let ens = pipeline::run_ensemble(&xrank32, &xlog32, nb, models, bt_csr, pred_cores.max(1));
    mark(timings, "ensemble_inference", t);
    if let Some(_) = dump_dir {
        for (mi, om) in ens.order_models.iter().enumerate() {
            dump_f32(dump_dir, &format!("model_order_{:02}.npy", mi), &[nb], om);
        }
        for (mi, pm) in ens.prob_models.iter().enumerate() {
            dump_f32(dump_dir, &format!("model_prob_{:02}.npy", mi), &[nb, 6], pm);
        }
        dump_f32(dump_dir, "predicted_order.npy", &[nb], &ens.predicted_order);
        dump_i64(
            dump_dir,
            "predicted_potency.npy",
            &[nb],
            &ens.predicted_potency
                .iter()
                .map(|&v| v as i64)
                .collect::<Vec<_>>(),
        );
        dump_f32(dump_dir, "prob_mean6.npy", &[nb, 6], &ens.prob_mean);
    }

    // ---- smoothing by diffusion (reseed inside) ----
    println!("cytotrace2: Performing smoothing by diffusion");
    let t = Instant::now();
    // official: np.random.seed(seed) inside smoothing_by_diffusion
    let mut s_rng = Mt19937::new(args.seed as u32);
    let preknn_f32 = ens.predicted_order.clone();
    let smooth = pipeline::smoothing_by_diffusion(
        &preknn_f32,
        &log2d,
        &top_col,
        nb,
        args.smooth_batch_size,
        &mut s_rng,
        pipeline::SmoothDump {
            dir: dump_dir.as_deref(),
        },
    );
    mark(timings, "smoothing_diffusion", t);
    dump_f64(dump_dir, "smooth_score_concat.npy", &[nb], &smooth);
    // advance the pipeline RNG to the same state the official run has after v0 draw
    {
        let mut v0 = vec![0.0f64; nb];
        s_rng.uniform_m1_1(&mut v0);
        dump_f64(dump_dir, "v0.npy", &[nb], &v0);
    }

    // ---- binning ----
    let t = Instant::now();
    let (binned, _groups) = pipeline::binning(&ens.predicted_potency, &smooth);
    mark(timings, "binning", t);
    dump_f64(dump_dir, "binned_preknn.npy", &[nb], &binned);

    // ---- KNN smoothing ----
    let score: Vec<f64>;
    if nb < 100 {
        println!("cytotrace2: Fewer than 100 cells in dataset. Skipping KNN smoothing step.");
        score = binned.clone();
    } else {
        println!("cytotrace2: Performing smoothing by adaptive KNN");
        let t = Instant::now();
        let data_scale = knn::scale_rows(&log2d, nb);
        dump_f64(dump_dir, "data_scale.npy", &[nb, f], &data_scale);
        mark(timings, "knn_scale", t);
        let t = Instant::now();
        let embed = knn::pca_embedding(&data_scale, nb).unwrap_or_else(|error| {
            eprintln!("cytotrace2: {error}");
            std::process::exit(1);
        });
        dump_f64(dump_dir, "pca_embedding.npy", &[nb, 30.min(nb - 1)], &embed);
        mark(timings, "knn_pca", t);
        let t = Instant::now();
        let res = knn::knn_smooth(&binned, &embed, nb, smooth_cores, dump_dir.as_deref());
        score = res.score;
        mark(timings, "knn_smooth", t);
        dump_f64(dump_dir, "knn_score_final.npy", &[nb], &score);
    }

    BatchOut {
        global_idx: sel.to_vec(),
        score,
        binned,
        potency: ens.predicted_potency,
    }
}
