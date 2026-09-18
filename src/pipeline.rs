// cytotrace2 pipeline stages: preprocess, top_var_genes, ensemble inference,
// smoothing by diffusion, binning. Numerics follow the official cytotrace2-py
// 1.1.0.4 (installed in envs/official) stage by stage.
use crate::npsum::*;
use crate::npyio::Arr;
use crate::rng::Mt19937;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

pub const N_FEATURES: usize = 14271;

// ---------------- r6 micro-opt: exact fast token parse ----------------
// Fast path for count-style tokens: optional sign, ASCII digits, at most one
// '.', no exponent, <= 15 total significant digits. Values are produced either
// as exact integers (m < 2^53) or as ONE correctly-rounded IEEE division
// m / 10^k with both operands exactly representable — bit-identical to what
// str::parse::<f64>() returns for the same token. Anything outside this
// grammar (exponent notation, >15 digits, non-ASCII bytes, inf/nan, empty)
// falls back to the original str::parse path with unchanged error semantics.
const POW10_U64: [u64; 16] = [
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000,
    100000000000,
    1000000000000,
    10000000000000,
    100000000000000,
    1000000000000000,
];
const POW10_F64: [f64; 16] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
];

static FASTPARSE_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FASTPARSE_FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FASTPARSE_MISMATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn fastparse_verify() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("C2RUST_VERIFY_FASTPARSE").is_ok())
}

#[inline]
fn parse_tok_f64_fast(t: &[u8]) -> Option<f64> {
    let (neg, b) = match t.first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let mut int_v: u64 = 0;
    let mut n_int: usize = 0;
    let mut i: usize = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        int_v = int_v * 10 + (b[i] - b'0') as u64;
        n_int += 1;
        i += 1;
    }
    let mut frac_v: u64 = 0;
    let mut n_frac: usize = 0;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            frac_v = frac_v * 10 + (b[i] - b'0') as u64;
            n_frac += 1;
            i += 1;
        }
    }
    if i != b.len() || n_int + n_frac == 0 || n_int + n_frac > 15 {
        return None;
    }
    let v = if n_frac == 0 {
        int_v as f64
    } else {
        (int_v * POW10_U64[n_frac] + frac_v) as f64 / POW10_F64[n_frac]
    };
    Some(if neg { -v } else { v })
}

/// Parse one value token exactly as the pre-micro-opt code did:
/// str::from_utf8(tok).unwrap_or("").trim_end_matches('\r').parse::<f64>(),
/// but with the bit-identical fast path in front.
#[inline]
fn parse_value_tok(tok: &[u8]) -> Result<f64, String> {
    let mut tb = tok;
    while tb.last() == Some(&b'\r') {
        tb = &tb[..tb.len() - 1];
    }
    if let Some(v) = parse_tok_f64_fast(tb) {
        // counters only run in verification mode — a per-token atomic RMW on a
        // shared line costs seconds across 80M tokens on 32 threads
        if fastparse_verify() {
            use std::sync::atomic::Ordering::Relaxed;
            let s = std::str::from_utf8(tb).unwrap_or("");
            match s.parse::<f64>() {
                Ok(sv) if sv.to_bits() == v.to_bits() => {}
                _ => {
                    FASTPARSE_MISMATCH.fetch_add(1, Relaxed);
                }
            }
            FASTPARSE_N.fetch_add(1, Relaxed);
        }
        return Ok(v);
    }
    if fastparse_verify() {
        use std::sync::atomic::Ordering::Relaxed;
        FASTPARSE_FALLBACK.fetch_add(1, Relaxed);
    }
    let s = std::str::from_utf8(tb).unwrap_or("");
    s.parse::<f64>().map_err(|_| s.to_string())
}

// ---------------- input parsing ----------------

pub struct InputMatrix {
    pub cell_names: Vec<String>,
    pub gene_names: Vec<String>,
    /// cells x genes, row-major f64
    pub x: Vec<f64>,
}

/// Parse tab-delimited genes x cells text (pandas read_csv(sep='\t', index_col=0).T semantics)
///
/// r5 determinism fix for the round-4 defect (intermittent "cannot parse float"
/// and transpose index-out-of-bounds panics on synth10k / synth200k).
/// Diagnosis: the old parser ran safe, per-line parallel code over immutable
/// borrows of the input buffer — safe code cannot shorten a row — so the bytes
/// entering the parser occasionally differed from the file on disk (WSL2
/// drvfs/9p intermittently corrupts very large reads; post-hoc re-reads of the
/// same inputs were structurally clean). The reader is therefore rebuilt to be
/// self-verifying and panic-free:
///   1. the file is read in bounded chunks into one exact-size buffer
///      (chunk size varies per attempt; no single multi-hundred-MB read
///      syscall chain),
///   2. every body line is structurally validated: exactly n_cells value
///      fields, each of which must parse as f64,
///   3. values are scattered straight into the cells-major (transposed) layout
///      through disjoint per-gene-block slots — the genes-major intermediate
///      and its parallel transpose are gone (this also halves peak parse
///      memory: the 11 GB synth200k input no longer materializes a second
///      45 GB genes-major buffer before transposing),
///   4. on any structural failure the whole buffer is discarded and the file
///      re-read from scratch (bounded attempts); only a clean read commits.
/// Bit-identity: on a clean read every value is produced by the same per-token
/// `str::parse::<f64>()` on the same bytes as before, written to the same
/// cell-major slot the old transpose produced. Gene/cell name handling,
/// duplicate checks and all downstream numerics are untouched.
const READ_ATTEMPTS: usize = 8;

enum ReadErr {
    /// buffer unreadable or structurally invalid — a fresh read may fix it
    Retryable(String),
    /// deterministic property of the input bytes — retrying cannot help
    Fatal(String),
}

fn read_file_chunked(path: &Path, attempt: usize) -> Result<(Vec<u8>, usize), ReadErr> {
    use std::os::unix::fs::FileExt;
    // per-attempt read geometry: (reader threads, chunk bytes) plus a small
    // buffer pad (shifts the allocation's virtual/physical page layout) so
    // each retry re-reads through a different low-level path/page mapping
    // than the failed attempt before it.
    //
    // r8 io-polish: the sequential read loop is replaced by a fixed-offset
    // parallel pread — reader threads own DISJOINT [t*seg, (t+1)*seg) byte
    // ranges of the same exact-size buffer, each filled by bounded read_at
    // calls. Which thread returns when is scheduling-dependent, but the
    // assembled byte content is a pure function of (attempt geometry, file
    // bytes): every byte is read exactly once from one fixed offset, so the
    // read path introduces no nondeterminism in the data. The 9p corruption
    // defense is untouched: the whole buffer still has to pass full
    // structural validation in read_matrix_once before it is accepted, and
    // any failure discards it and re-reads with a different geometry above.
    const GEO: [(usize, usize); 8] = [
        (32, 8 << 20),
        (16, 1 << 20),
        (24, 256 << 10),
        (32, 4 << 20),
        (12, 512 << 10),
        (20, 2 << 20),
        (8, 128 << 10),
        (32, 8 << 20),
    ];
    let (readers, chunk) = GEO[(attempt - 1) % GEO.len()];
    let pad = (attempt - 1) * (2 << 20);
    let len = std::fs::metadata(path)
        .map_err(|e| ReadErr::Retryable(format!("stat {}: {}", path.display(), e)))?
        .len() as usize;
    let f = std::fs::File::open(path)
        .map_err(|e| ReadErr::Retryable(format!("open {}: {}", path.display(), e)))?;
    let mut buf = vec![0u8; len + pad];
    let seg = (len + readers - 1) / readers;
    let first_err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    std::thread::scope(|s| {
        for (ci, slice) in buf[..len].chunks_mut(seg).enumerate() {
            let base = ci * seg;
            let f = &f;
            let first_err = &first_err;
            s.spawn(move || {
                let mut pos = 0usize;
                while pos < slice.len() {
                    let want = chunk.min(slice.len() - pos);
                    match f.read_at(&mut slice[pos..pos + want], (base + pos) as u64) {
                        Ok(0) => {
                            let mut g = first_err.lock().unwrap();
                            if g.is_none() {
                                *g = Some(format!(
                                    "short read on {}: eof at {} of {} bytes",
                                    path.display(),
                                    base + pos,
                                    len
                                ));
                            }
                            return;
                        }
                        Ok(n) => pos += n,
                        Err(e) => {
                            let mut g = first_err.lock().unwrap();
                            if g.is_none() {
                                *g = Some(format!("read {}: {}", path.display(), e));
                            }
                            return;
                        }
                    }
                }
            });
        }
    });
    if let Some(msg) = first_err.into_inner().unwrap() {
        return Err(ReadErr::Retryable(msg));
    }
    Ok((buf, len))
}

/// Whole-file parallel pread at fixed disjoint offsets (same reader scheme as
/// read_file_chunked, without the retry/validation harness). Deterministic
/// byte content; used only for the small asset npz files, whose zip/npy
/// structure is fully asserted inside read_npz_bytes anyway.
pub fn pread_all(path: &Path, readers: usize) -> std::io::Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let len = std::fs::metadata(path)?.len() as usize;
    let f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; len];
    if len == 0 {
        return Ok(buf);
    }
    let seg = (len + readers - 1) / readers;
    let first_err: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    std::thread::scope(|s| {
        for (ci, slice) in buf.chunks_mut(seg).enumerate() {
            let base = ci * seg;
            let f = &f;
            let first_err = &first_err;
            s.spawn(move || {
                let mut pos = 0usize;
                while pos < slice.len() {
                    match f.read_at(&mut slice[pos..], (base + pos) as u64) {
                        Ok(0) => {
                            let mut g = first_err.lock().unwrap();
                            if g.is_none() {
                                *g = Some(std::io::Error::new(
                                    std::io::ErrorKind::UnexpectedEof,
                                    format!("eof at {} of {}", base + pos, len),
                                ));
                            }
                            return;
                        }
                        Ok(n) => pos += n,
                        Err(e) => {
                            let mut g = first_err.lock().unwrap();
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(buf)
}

pub fn read_matrix(path: &Path) -> InputMatrix {
    for attempt in 1..=READ_ATTEMPTS {
        match read_matrix_once(path, attempt) {
            Ok(m) => {
                if attempt > 1 {
                    eprintln!(
                        "cytotrace2: input passed structural validation on re-read attempt {}",
                        attempt
                    );
                }
                if fastparse_verify() {
                    use std::sync::atomic::Ordering::Relaxed;
                    eprintln!(
                        "FASTPARSE {{\"fast\":{},\"fallback\":{},\"mismatch\":{}}}",
                        FASTPARSE_N.load(Relaxed),
                        FASTPARSE_FALLBACK.load(Relaxed),
                        FASTPARSE_MISMATCH.load(Relaxed)
                    );
                }
                return m;
            }
            Err(ReadErr::Retryable(msg)) => {
                eprintln!(
                    "cytotrace2: input read failed structural validation (attempt {}/{}): {}",
                    attempt, READ_ATTEMPTS, msg
                );
                if attempt < READ_ATTEMPTS {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            }
            Err(ReadErr::Fatal(msg)) => {
                eprintln!("cytotrace2: {}", msg);
                std::process::exit(1);
            }
        }
    }
    eprintln!(
        "cytotrace2: input still fails structural validation after {} reads; aborting",
        READ_ATTEMPTS
    );
    std::process::exit(1);
}

fn read_matrix_once(path: &Path, attempt: usize) -> Result<InputMatrix, ReadErr> {
    let sub = std::env::var("C2RUST_TIME_SUB").is_ok();
    let t_read0 = std::time::Instant::now();
    let (buf, dlen) = read_file_chunked(path, attempt)?;
    let t_read = t_read0.elapsed().as_secs_f64();
    let raw: &[u8] = &buf[..dlen];
    let mut lines: Vec<&[u8]> = raw.split(|&b| b == b'\n').collect();
    if let Some(last) = lines.last() {
        if last.is_empty() {
            lines.pop();
        }
    }
    if lines.len() < 2 {
        return Err(ReadErr::Fatal("input too small".to_string()));
    }
    // header: this file has NO gene-name column header — the first line is all cell names
    // (pandas read_csv auto-detects the extra data column and uses it as index)
    let hdr: Vec<&[u8]> = lines[0].split(|&b| b == b'\t').collect();
    let cell_names: Vec<String> = hdr
        .iter()
        .map(|s| {
            String::from_utf8_lossy(s)
                .trim_end_matches('\r')
                .trim_end_matches('\n')
                .to_string()
        })
        .collect();
    let n_cells = cell_names.len();
    if n_cells == 0 {
        return Err(ReadErr::Fatal(
            "header line carries no cell names".to_string(),
        ));
    }
    let body: &[&[u8]] = &lines[1..];
    let n_genes = body.len();

    let gene_names: Vec<String> = body
        .iter()
        .map(|ln| {
            let mut it = ln.split(|&b| b == b'\t');
            let g = it.next().unwrap_or(&[]);
            String::from_utf8_lossy(g)
                .trim_end_matches('\r')
                .to_string()
        })
        .collect();

    // deterministic wave-parallel parse: waves of consecutive gene lines are
    // parsed in parallel into per-line buffers (each validated: exactly
    // n_cells values, every value parseable as f64), then scattered into
    // disjoint cells-major rows of x (par_chunks_mut over cell rows — safe,
    // each slot written exactly once). Wave buffering bounds the transient
    // genes-major copy to WAVE_BYTES, so the 11 GB synth200k input never
    // materializes a second full-size matrix. Any structural anomaly fails
    // the whole attempt; the buffer is then discarded and the file re-read.
    const WAVE_BYTES: usize = 4 << 30;
    let mut x = vec![0.0f64; n_cells * n_genes];
    let failed = std::sync::atomic::AtomicBool::new(false);
    let first_err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    let lines_per_wave = std::cmp::max(1, WAVE_BYTES / (n_cells * 8));
    let mut t_parse_acc = 0.0f64;
    let mut t_scatter_acc = 0.0f64;
    for (wi, wave) in body.chunks(lines_per_wave).enumerate() {
        let t_wave0 = std::time::Instant::now();
        let l0 = wi * lines_per_wave;
        let rows: Vec<Option<Vec<f64>>> = wave
            .par_iter()
            .enumerate()
            .map(|(j, ln)| {
                use std::sync::atomic::Ordering::Relaxed;
                if failed.load(Relaxed) {
                    return None; // another line already failed; wind down quickly
                }
                let g = l0 + j;
                let mut vals = Vec::with_capacity(n_cells);
                let mut it = ln.split(|&b| b == b'\t');
                it.next(); // gene name
                for tok in it {
                    match parse_value_tok(tok) {
                        Ok(v) => vals.push(v),
                        Err(t) => {
                            let mut e = first_err.lock().unwrap();
                            if e.is_none() {
                                *e = Some(format!(
                                    "gene row {} ({}): cannot parse float {:?}",
                                    g, gene_names[g], t
                                ));
                            }
                            drop(e);
                            failed.store(true, Relaxed);
                            return None;
                        }
                    }
                }
                if vals.len() != n_cells {
                    let mut e = first_err.lock().unwrap();
                    if e.is_none() {
                        *e = Some(format!(
                            "gene row {} ({}): {} value fields, expected {}",
                            g,
                            gene_names[g],
                            vals.len(),
                            n_cells
                        ));
                    }
                    drop(e);
                    failed.store(true, Relaxed);
                    return None;
                }
                Some(vals)
            })
            .collect();
        if failed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(ReadErr::Retryable(
                first_err
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| "structural validation failed".to_string()),
            ));
        }
        t_parse_acc += t_wave0.elapsed().as_secs_f64();
        let t_scat0 = std::time::Instant::now();
        // scatter the wave: blocked transpose into the cells-major rows of x.
        // Parallel over 64-cell blocks; inside a block, 16-gene tiles keep
        // both the source reads (16 rows at one cell offset — one cache line
        // each, reused across the next 8 cells) and the destination writes
        // (16 contiguous f64 = 128 B per cell row) cache-line aligned.
        // Pure data movement: every x slot receives exactly the value the
        // per-cell gather wrote before (all entries of `rows` are Some here).
        const CB: usize = 64;
        const JB: usize = 16;
        let nw = rows.len();
        x.par_chunks_mut(n_genes * CB)
            .enumerate()
            .for_each(|(cb, xblk)| {
                let c0 = cb * CB;
                let nc = xblk.len() / n_genes;
                let mut j0 = 0usize;
                while j0 < nw {
                    let j1 = (j0 + JB).min(nw);
                    for c in 0..nc {
                        let xrow = &mut xblk[c * n_genes..(c + 1) * n_genes];
                        for j in j0..j1 {
                            xrow[l0 + j] = rows[j].as_ref().unwrap()[c0 + c];
                        }
                    }
                    j0 = j1;
                }
            });
        t_scatter_acc += t_scat0.elapsed().as_secs_f64();
    }
    if sub {
        eprintln!(
            "SUB io {{\"read_s\":{:.4},\"parse_s\":{:.4},\"scatter_s\":{:.4}}}",
            t_read, t_parse_acc, t_scatter_acc
        );
    }

    // duplicate checks (official raises; classified retryable because a
    // corrupted read could also fabricate a duplicate — a clean input has none)
    {
        let mut seen = HashMap::new();
        for (i, g) in gene_names.iter().enumerate() {
            if seen.insert(g.clone(), i).is_some() {
                return Err(ReadErr::Retryable(
                    "Please make sure the gene names are unique.".to_string(),
                ));
            }
        }
        let mut seen = HashMap::new();
        for c in cell_names.iter() {
            if seen.insert(c.clone(), ()).is_some() {
                return Err(ReadErr::Retryable(
                    "Please make sure the cell names are unique.".to_string(),
                ));
            }
        }
    }
    Ok(InputMatrix {
        cell_names,
        gene_names,
        x,
    })
}

// ---------------- species mapping ----------------

/// mouse alias normalization (gen_utils.py preprocess mouse branch)
pub fn apply_mouse_alias(
    gene_names: &[String],
    alias: &HashMap<String, String>,
    features: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mapped: Vec<String> = gene_names.to_vec();
    let mapped_set: std::collections::HashSet<&String> = mapped.iter().collect();
    let mut to_replace: Vec<usize> = Vec::new();
    for (i, g) in mapped.iter().enumerate() {
        if !features.contains(g) {
            if let Some(mm) = alias.get(g) {
                if !mapped_set.contains(mm) {
                    to_replace.push(i);
                }
            }
        }
    }
    let mut out = mapped;
    for &i in &to_replace {
        let g = out[i].clone();
        out[i] = alias.get(&g).cloned().unwrap_or(g);
    }
    out
}

/// human ortholog mapping (returns per-gene Option<mouse name>)
pub fn apply_human_map(
    gene_names: &[String],
    ortho: &HashMap<String, String>,
    alias: &HashMap<String, String>,
) -> Vec<Option<String>> {
    gene_names
        .iter()
        .map(|g| {
            if let Some(m) = ortho.get(g) {
                Some(m.clone())
            } else if let Some(m) = alias.get(g) {
                Some(m.clone())
            } else {
                None
            }
        })
        .collect()
}

/// align expression to the 14271 feature space (join + fillna(0))
pub fn align_features(
    x: &[f64],
    n_cells: usize,
    n_genes_in: usize,
    gene_names: &[String],
    features: &[String],
) -> Vec<f64> {
    let mut idx: HashMap<&str, usize> = HashMap::with_capacity(n_genes_in);
    for (i, g) in gene_names.iter().enumerate() {
        let e = idx.insert(g.as_str(), i);
        if e.is_some() {
            panic!("duplicate gene after mapping: {} (unsupported in MVP)", g);
        }
    }
    let f = N_FEATURES;
    let mut out = vec![0.0f64; n_cells * f];
    out.par_chunks_mut(f)
        .zip(x.par_chunks(n_genes_in))
        .for_each(|(orow, irow)| {
            for (j, feat) in features.iter().enumerate() {
                if let Some(&c) = idx.get(feat.as_str()) {
                    orow[j] = irow[c];
                }
            }
        });
    out
}

// ---------------- CPM + log2 + ranks ----------------

pub struct CpmRankOut {
    pub denom: Vec<f64>,
    pub log2d: Vec<f64>,
    /// per-cell descending average ranks, already materialized as f32 (the
    /// only f64->f32 cast the ensemble consumes; identical to casting the
    /// f64 rank matrix element-wise)
    pub rank32: Vec<f32>,
    /// f64 rank matrix, produced only when --dump-stage needs rank_data.npy
    pub rank64: Option<Vec<f64>>,
    /// per-row count of produced log2d values > 0 (the QC "expressed genes"
    /// count; identical to the previous post-hoc scan over log2d)
    pub pos_counts: Vec<usize>,
}

/// Fused CPM/log2 + descending-average-rank pass (r8 io-polish). One
/// streaming sweep per cell row over `aligned` now produces denom, log2d,
/// the ranks and the QC positive count together — previously three separate
/// full passes (cpm denom+log2, rank sort, QC scan). Per-element arithmetic
/// is unchanged everywhere:
///   - denom: np_sum_f64 over the same row bytes (same pairwise order)
///   - log2d: the same v==0 shortcut / log2 expression, evaluated in the
///     same order
///   - ranks: see rank_row_into
/// One cell row of the fused pass: fills the log2d row and the rank sinks,
/// returns (row CPM denominator, count of produced log2d values > 0).
fn cpm_rank_row(
    irow: &[f64],
    lrow: &mut [f64],
    r32: &mut [f32],
    r64: Option<&mut [f64]>,
) -> (f64, usize) {
    let f = irow.len();
    let s = np_sum_f64(irow);
    let mut cnt = 0usize;
    for t in 0..f {
        // micro-opt kept from r6: for v == 0.0 and s > 0 the original
        // expression evaluates exactly to log2(0/s + 1) = log2(1.0) =
        // +0.0, so the multiply/divide/log2 can be skipped for the
        // (dominant) zero entries; s == 0 keeps the original NaN
        // semantics.
        let v = irow[t];
        let lv = if v == 0.0 {
            if s == 0.0 {
                f64::NAN
            } else {
                0.0
            }
        } else {
            ((v * 1e6f64 / s) + 1.0f64).log2()
        };
        lrow[t] = lv;
        if lv > 0.0 {
            cnt += 1;
        }
    }
    rank_row_into(irow, r32, r64);
    (s, cnt)
}

pub fn cpm_log2_rank(aligned: &[f64], n_cells: usize, keep_f64_rank: bool) -> CpmRankOut {
    let f = N_FEATURES;
    let mut denom = vec![0.0f64; n_cells];
    let mut log2d = vec![0.0f64; n_cells * f];
    let mut rank32 = vec![0.0f32; n_cells * f];
    let mut rank64 = if keep_f64_rank {
        Some(vec![0.0f64; n_cells * f])
    } else {
        None
    };
    let mut pos_counts = vec![0usize; n_cells];
    if let Some(rank64) = &mut rank64 {
        aligned
            .par_chunks(f)
            .zip(log2d.par_chunks_mut(f))
            .zip(rank32.par_chunks_mut(f))
            .zip(rank64.par_chunks_mut(f))
            .zip(denom.par_iter_mut())
            .zip(pos_counts.par_iter_mut())
            .for_each(|(((((irow, lrow), r32), r64), d), pc)| {
                let (s, cnt) = cpm_rank_row(irow, lrow, r32, Some(r64));
                *d = s;
                *pc = cnt;
            });
    } else {
        aligned
            .par_chunks(f)
            .zip(log2d.par_chunks_mut(f))
            .zip(rank32.par_chunks_mut(f))
            .zip(denom.par_iter_mut())
            .zip(pos_counts.par_iter_mut())
            .for_each(|((((irow, lrow), r32), d), pc)| {
                let (s, cnt) = cpm_rank_row(irow, lrow, r32, None);
                *d = s;
                *pc = cnt;
            });
    }
    CpmRankOut {
        denom,
        log2d,
        rank32,
        rank64,
        pos_counts,
    }
}

/// per-cell descending average ranks (scipy.stats.rankdata(-X, axis=1,
/// method='average')) for one row, written into an f32 sink (and optionally
/// the f64 sink for dumps).
///
/// Fast path for raw-count rows (r8 io-polish): every exact 0.0 (including
/// -0.0, which == 0.0 and bit-normalizes to +0.0 under the tie semantics)
/// forms ONE tie group occupying the tail positions zeros..n of the
/// descending order — the full sort would give all of them
/// 0.5*((n-zeros) + n + 1) — so only the nonzero prefix is sorted, with the
/// same comparator and the same grouping loop over identical positions. The
/// branch depends only on the row's values, never on timing. Rows with any
/// negative value (never produced by count inputs, but possible in theory)
/// take the original full-sort path.
fn rank_row_into(irow: &[f64], or32: &mut [f32], mut or64: Option<&mut [f64]>) {
    let n = irow.len();
    let mut has_neg = false;
    let mut nzidx: Vec<u32> = Vec::with_capacity(64);
    for (j, &v) in irow.iter().enumerate() {
        if v != 0.0 {
            nzidx.push(j as u32);
            if v < 0.0 {
                has_neg = true;
            }
        }
    }
    let zeros = n - nzidx.len();
    if !has_neg && zeros >= 64 {
        nzidx.sort_unstable_by(|&a, &b| {
            let va = irow[a as usize];
            let vb = irow[b as usize];
            vb.partial_cmp(&va).unwrap() // descending by value (rank of -X ascending)
        });
        let mut p = 0usize;
        while p < nzidx.len() {
            let v0 = irow[nzidx[p] as usize];
            let mut q = p + 1;
            while q < nzidx.len() && irow[nzidx[q] as usize] == v0 {
                q += 1;
            }
            let avg = 0.5 * (p + q + 1) as f64;
            for k in p..q {
                let j = nzidx[k] as usize;
                or32[j] = avg as f32;
                if let Some(o) = or64.as_deref_mut() {
                    o[j] = avg;
                }
            }
            p = q;
        }
        let zavg = 0.5 * (zeros + n + 1) as f64;
        let zavg32 = zavg as f32;
        for (j, &v) in irow.iter().enumerate() {
            if v == 0.0 {
                or32[j] = zavg32;
                if let Some(o) = or64.as_deref_mut() {
                    o[j] = zavg;
                }
            }
        }
    } else {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.sort_unstable_by(|&a, &b| {
            let va = irow[a as usize];
            let vb = irow[b as usize];
            vb.partial_cmp(&va).unwrap() // descending by value (rank of -X ascending)
        });
        // group equal values (sorted descending); positions p..q 0-based -> avg rank 0.5*(p+q+1)
        let mut p = 0usize;
        while p < n {
            let v0 = irow[idx[p] as usize];
            let mut q = p + 1;
            while q < n && irow[idx[q] as usize] == v0 {
                q += 1;
            }
            let avg = 0.5 * (p + q + 1) as f64;
            for k in p..q {
                let j = idx[k] as usize;
                or32[j] = avg as f32;
                if let Some(o) = or64.as_deref_mut() {
                    o[j] = avg;
                }
            }
            p = q;
        }
    }
}

// ---------------- dispersion / top var genes ----------------

/// disp_fn per gene over cells: 0 if all values equal else var_pop/mean.
/// Column sums use numpy pairwise summation (strided reductions are buffered
/// into one contiguous block for n < 8192, then the pairwise kernel applies).
pub fn dispersion(log2: &[f64], n_cells: usize) -> Vec<f64> {
    let f = N_FEATURES;
    (0..f)
        .into_par_iter()
        .map(|j| {
            let col: Vec<f64> = (0..n_cells).map(|i| log2[i * f + j]).collect();
            let first = col[0];
            if col.iter().all(|&v| v == first) {
                0.0
            } else {
                let mean = np_sum_f64(&col) / n_cells as f64;
                let dev: Vec<f64> = col.iter().map(|&v| v - mean).collect();
                let mut var = np_sum_f64(&dev.iter().map(|d| d * d).collect::<Vec<_>>());
                var /= n_cells as f64;
                var / mean
            }
        })
        .collect()
}

pub fn top_var_genes(disp: &[f64]) -> Vec<usize> {
    let n = disp.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_unstable_by(|&a, &b| disp[a as usize].partial_cmp(&disp[b as usize]).unwrap());
    idx[n - 1000..].iter().map(|&v| v as usize).collect()
}

// ---------------- ensemble inference ----------------

pub struct ModelWeights {
    pub layers: Vec<LayerW>,
}
pub struct LayerW {
    pub wbits: Vec<f32>, // 14271*24 (0/1)
    pub maxrank: f32,
    pub rm: Vec<f32>, // 48
    pub rv: Vec<f32>, // 48
    pub ow: Vec<f32>, // 48 (single head row)
    pub ob: f32,
}

pub struct LayerPrecomp {
    rowmask: Vec<u32>, // per gene j: bit k set iff wbits[j*24+k] > 0
    n: Vec<f32>,
    maxrank: f32,
    half: Vec<f32>,
    denom2: Vec<f32>,
    gs: Vec<f32>, // 14271*24
    bg_denom: Vec<f32>,
    rm: Vec<f32>,
    rv: Vec<f32>,
    ow: Vec<f32>,
    ob: f32,
}

pub fn load_model(entries: &[(String, Arr)]) -> ModelWeights {
    let mut layers = Vec::with_capacity(6);
    for l in 0..6usize {
        let mut get_f32 = |name: String| -> Vec<f32> {
            let e = entries.iter().find(|(k, _)| *k == name).expect("npz key");
            e.1.clone().as_f32()
        };
        let w = get_f32(format!("layers.{}.weight", l));
        let maxrank = get_f32(format!("layers.{}.maxrank", l));
        let rm = get_f32(format!("layers.{}.batchnorm.running_mean", l));
        let rv = get_f32(format!("layers.{}.batchnorm.running_var", l));
        let ow = get_f32(format!("layers.{}.out.weight", l));
        let ob = get_f32(format!("layers.{}.out.bias", l));
        layers.push(LayerW {
            wbits: w
                .iter()
                .map(|&v| if v > 0.0 { 1.0f32 } else { 0.0f32 })
                .collect(),
            maxrank: maxrank[0],
            rm,
            rv,
            ow,
            ob: ob[0],
        });
    }
    ModelWeights { layers }
}

/// background.pt COO -> dense B^T (row-major 14271*14271 f32); B^T[i][j] = B[j][i]
pub fn background_dense(entries: &[(String, Arr)]) -> Vec<f32> {
    let idx = entries
        .iter()
        .find(|(k, _)| k == "indices")
        .unwrap()
        .1
        .clone()
        .as_i64();
    let vals = entries
        .iter()
        .find(|(k, _)| k == "values")
        .unwrap()
        .1
        .clone()
        .as_f32();
    let shape = entries
        .iter()
        .find(|(k, _)| k == "shape")
        .unwrap()
        .1
        .clone()
        .as_i64();
    assert_eq!(shape[0] as usize, N_FEATURES);
    assert_eq!(shape[1] as usize, N_FEATURES);
    let nnz = vals.len();
    let mut dense = vec![0.0f32; N_FEATURES * N_FEATURES];
    for k in 0..nnz {
        let r = idx[k] as usize; // indices[0, k]
        let c = idx[nnz + k] as usize; // indices[1, k]
        let v = vals[k];
        dense[c * N_FEATURES + r] = v; // B^T[c][r] = B[r][c]
    }
    dense
}

fn debug_fwd() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("C2RUST_DEBUG_FWD").is_ok())
}

/// CSR of the dense B^T: per row, (col, val) of nonzeros in ascending column
/// order — exactly the traversal order of the previous dense-row scan, so the
/// per-element accumulation order of the gs matmul is preserved bit-for-bit.
pub struct BtCsr {
    pub row_ptr: Vec<u32>,
    pub cols: Vec<u32>,
    pub vals: Vec<f32>,
}

pub fn background_csr(bt: &[f32]) -> BtCsr {
    let f = N_FEATURES;
    let rows: Vec<(Vec<u32>, Vec<f32>)> = bt
        .par_chunks(f)
        .map(|brow| {
            let mut c = Vec::new();
            let mut v = Vec::new();
            for (j, &a) in brow.iter().enumerate() {
                if a != 0.0 {
                    c.push(j as u32);
                    v.push(a);
                }
            }
            (c, v)
        })
        .collect();
    let mut row_ptr = Vec::with_capacity(f + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0u32);
    for (c, v) in &rows {
        cols.extend_from_slice(c);
        vals.extend_from_slice(v);
        row_ptr.push(cols.len() as u32);
    }
    BtCsr {
        row_ptr,
        cols,
        vals,
    }
}

/// B^T CSR built directly from the background COO triples (r8 io-polish) —
/// the 14271x14271 dense f32 intermediate (~815 MB written + rescanned) is
/// gone. The result is exactly what background_dense + background_csr
/// produced: dense[c][r] held the LAST triple with coordinates (r, c); the
/// CSR row c then listed ascending r with dense value != 0.0. Here triples
/// are bucketed by row c (bucket order = input order), each bucket is sorted
/// by (r, sequence) with the LAST occurrence of each r kept, and exact zeros
/// dropped — the identical (col, val) stream, so the gs matmul downstream is
/// bit-identical.
pub fn background_csr_from_coo(entries: &[(String, Arr)]) -> BtCsr {
    let t0 = std::time::Instant::now();
    let idx = entries
        .iter()
        .find(|(k, _)| k == "indices")
        .unwrap()
        .1
        .clone()
        .as_i64();
    let vals = entries
        .iter()
        .find(|(k, _)| k == "values")
        .unwrap()
        .1
        .clone()
        .as_f32();
    let shape = entries
        .iter()
        .find(|(k, _)| k == "shape")
        .unwrap()
        .1
        .clone()
        .as_i64();
    assert_eq!(shape[0] as usize, N_FEATURES);
    assert_eq!(shape[1] as usize, N_FEATURES);
    let nnz = vals.len();
    let f = N_FEATURES;
    // histogram of CSR rows (c = indices[1, k]) -> prefix offsets
    let mut row_ptr = vec![0u32; f + 1];
    for k in 0..nnz {
        row_ptr[(idx[nnz + k] as usize) + 1] += 1;
    }
    let mut acc = 0u32;
    for c in 0..f {
        let cnt = row_ptr[c + 1];
        row_ptr[c + 1] = acc + cnt;
        acc += cnt;
    }
    // scatter (r, seq, v) into per-row scratch, preserving input order per row
    let mut cols = vec![0u32; nnz];
    let mut seqs = vec![0u32; nnz];
    let mut svals = vec![0.0f32; nnz];
    {
        let mut pos = row_ptr.clone();
        for k in 0..nnz {
            let c = idx[nnz + k] as usize;
            let p = pos[c] as usize;
            cols[p] = idx[k] as u32;
            seqs[p] = k as u32;
            svals[p] = vals[k];
            pos[c] += 1;
        }
    }
    // per row: sort by (col, seq), keep the LAST triple per col, drop v == 0.0
    let rows: Vec<(Vec<u32>, Vec<f32>)> = (0..f)
        .into_par_iter()
        .map(|c| {
            let s = row_ptr[c] as usize;
            let e = row_ptr[c + 1] as usize;
            let mut trip: Vec<(u32, u32)> = Vec::with_capacity(e - s);
            for p in s..e {
                trip.push((cols[p], seqs[p]));
            }
            trip.sort_unstable();
            let mut oc: Vec<u32> = Vec::with_capacity(e - s);
            let mut ov: Vec<f32> = Vec::with_capacity(e - s);
            let mut i = 0usize;
            while i < trip.len() {
                let col = trip[i].0;
                let mut j = i + 1;
                while j < trip.len() && trip[j].0 == col {
                    j += 1;
                }
                let v = svals[trip[j - 1].1 as usize]; // last occurrence wins (dense overwrite)
                if v != 0.0 {
                    oc.push(col);
                    ov.push(v);
                }
                i = j;
            }
            (oc, ov)
        })
        .collect();
    let mut row_ptr2 = Vec::with_capacity(f + 1);
    let mut ccols = Vec::new();
    let mut cvals = Vec::new();
    row_ptr2.push(0u32);
    for (c, v) in &rows {
        ccols.extend_from_slice(c);
        cvals.extend_from_slice(v);
        row_ptr2.push(ccols.len() as u32);
    }
    if std::env::var("C2RUST_TIME_SUB").is_ok() {
        eprintln!("SUB bgcsr {{\"csr_s\":{:.4}}}", t0.elapsed().as_secs_f64());
    }
    BtCsr {
        row_ptr: row_ptr2,
        cols: ccols,
        vals: cvals,
    }
}

fn precomp_layer(lay: &LayerW, gs: Vec<f32>) -> LayerPrecomp {
    let f = N_FEATURES;
    let h = 24usize;
    let mut n = vec![0.0f32; h];
    for j in 0..f {
        for k in 0..h {
            n[k] += lay.wbits[j * h + k];
        }
    }
    let nmax = n.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let maxrank = (nmax + 10.0f32) + lay.maxrank.clamp(0.0, f32::INFINITY) * 1000.0f32;
    let half: Vec<f32> = n.iter().map(|&v| (v * (v + 1.0f32)) / 2.0f32).collect();
    let denom2: Vec<f32> = n.iter().map(|&v| v * maxrank).collect();
    if debug_fwd() {
        eprintln!(
            "DBG precomp: maxrank={} n[:6]={:?} half[:3]={:?} maxrank_param={}",
            maxrank,
            &n[..6],
            &half[..3],
            lay.maxrank
        );
    }

    // gs arrives precomputed (see run_ensemble): all 114 (model, layer) blocks of
    // gs = B^T @ [W1|W2|...|W114] are produced in ONE streaming pass over B^T
    // (CSR rows, columns ascending). Per output element the accumulation order
    // (j ascending over nonzeros, exact f64 products, f32 rounding at the torch
    // tensor boundary) is unchanged, so gs is bit-identical to the per-layer
    // dense-row scan this replaces.
    let mut bg_denom = vec![0.0f32; h];
    {
        // torch gs.sum(0): column sums of the f32 gs (use f64 accumulation, round to f32)
        let mut acc = [0.0f64; 24];
        for i in 0..f {
            for k in 0..h {
                acc[k] += gs[i * h + k] as f64;
            }
        }
        for k in 0..h {
            bg_denom[k] = acc[k] as f32;
        }
    }
    if debug_fwd() {
        eprintln!(
            "DBG gs colsum[:6]={:?} gs_col0_first3={:?}",
            &bg_denom[..6],
            [gs[0], gs[h], gs[2 * h]]
        );
    }
    // row bitmask of the 0/1 wbits: bit k of rowmask[j] set iff w[j][k] == 1
    let mut rowmask = vec![0u32; f];
    for j in 0..f {
        let mut m = 0u32;
        for k in 0..h {
            if lay.wbits[j * h + k] > 0.0 {
                m |= 1u32 << k;
            }
        }
        rowmask[j] = m;
    }
    LayerPrecomp {
        rowmask,
        n,
        maxrank,
        half,
        denom2,
        gs,
        bg_denom,
        rm: lay.rm.clone(),
        rv: lay.rv.clone(),
        ow: lay.ow.clone(),
        ob: lay.ob,
    }
}

#[inline]
fn head_finish(r32: [f32; 24], bg32: [f32; 24], raw32: [f32; 24], pc: &LayerPrecomp) -> f32 {
    let mut feats = [0.0f32; 48];
    for k in 0..24 {
        feats[k] = 1.0f32 - (r32[k] - pc.half[k]) / pc.denom2[k];
        feats[24 + k] = (raw32[k] / pc.n[k]) - (bg32[k] / pc.bg_denom[k]);
    }
    for k in 0..48 {
        feats[k] = (feats[k] - pc.rm[k]) / (pc.rv[k] + 1e-5f32).sqrt();
    }
    let mut z = pc.ob;
    for k in 0..48 {
        z += feats[k] * pc.ow[k];
    }
    z
}

pub struct EnsembleOut {
    pub order_models: Vec<Vec<f32>>,
    pub prob_models: Vec<Vec<f32>>,
    pub predicted_order: Vec<f32>,
    pub predicted_potency: Vec<usize>,
    pub prob_mean: Vec<f32>,
}

pub fn run_ensemble(
    xrank32: &[f32],
    xlog32: &[f32],
    n: usize,
    models: &[ModelWeights],
    csr: &BtCsr,
    threads: usize,
) -> EnsembleOut {
    let f = N_FEATURES;
    let hw = 24usize;
    let nm = models.len();
    let w = nm * 6 * hw; // 19 models x 6 layers x 24 columns = 2736 wide
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let time_sub = std::env::var("C2RUST_TIME_SUB").is_ok();
    let t0 = std::time::Instant::now();
    // ---- (A) batched background matmul ----
    // Concatenate the per-layer first-layer-space weights of all 19 models into
    // one wide matrix W (f x w, model-major blocks); stream B^T ONCE via its CSR
    // rows and produce gs_all = B^T @ [W1|W2|...|W114]. Previously B^T was
    // streamed 114 times (once per model-layer). Column blocking is only a
    // layout concern: every output element still accumulates its j-ascending,
    // zero-skipping f64 products in the same order and rounds to f32 at the
    // torch tensor boundary, so gs is bit-identical to the MVP.
    let mut w_all = vec![0.0f32; f * w];
    {
        let mut b = 0usize;
        for mw in models {
            for lay in &mw.layers {
                w_all
                    .par_chunks_mut(w)
                    .zip(lay.wbits.par_chunks(hw))
                    .for_each(|(dst, src)| {
                        dst[b * hw..b * hw + hw].copy_from_slice(src);
                    });
                b += 1;
            }
        }
    }
    let t_w = t0.elapsed().as_secs_f64();
    let mut gs_all = vec![0.0f32; f * w];
    gs_all.par_chunks_mut(w).enumerate().for_each(|(i, grow)| {
        let s = csr.row_ptr[i] as usize;
        let e = csr.row_ptr[i + 1] as usize;
        let mut acc = vec![0.0f64; w];
        for t in s..e {
            let j = csr.cols[t] as usize;
            let a = csr.vals[t] as f64;
            let wrow = &w_all[j * w..j * w + w];
            for c in 0..w {
                acc[c] += a * (wrow[c] as f64);
            }
        }
        for c in 0..w {
            grow[c] = acc[c] as f32;
        }
    });
    drop(w_all);
    let t_g = t0.elapsed().as_secs_f64();
    // ---- (A2) gene-major x workspace (f x npad f32, zero-padded) ----
    // The forward below streams each gene row once per model for a TILE-wide
    // column of cells. Padded lanes (i >= n) hold exact zeros: xd == 0.0
    // contributes nothing to bg, min(0.0, maxrank) == 0.0 adds nothing to r,
    // and raw += 0.0 is an identity on non-negative accumulators — so padding
    // cannot perturb any lane that reaches head_finish.
    const TILE: usize = 16;
    let npad = (n + TILE - 1) / TILE * TILE;
    let mut xt = vec![0.0f32; f * npad];
    let mut xl_t = vec![0.0f32; f * npad];
    {
        // blocked transpose, parallel over gene-row blocks: each task moves
        // 16-gene x 16-cell tiles with contiguous 64 B accesses on both sides
        // (source line = 16 genes of one cell row; destination lines = 16
        // cells of one gene row, reused across the cell block)
        const JB: usize = 16;
        xt.par_chunks_mut(JB * npad)
            .zip(xl_t.par_chunks_mut(JB * npad))
            .enumerate()
            .for_each(|(bi, (xtt, xltt))| {
                let j0 = bi * JB;
                let nj = xtt.len() / npad;
                for i0 in (0..npad).step_by(16) {
                    let ie = (i0 + 16).min(npad);
                    for i in i0..ie {
                        if i < n {
                            let xr = &xrank32[i * f + j0..i * f + j0 + nj];
                            let xlv = &xlog32[i * f + j0..i * f + j0 + nj];
                            for j in 0..nj {
                                xtt[j * npad + i] = xr[j];
                                xltt[j * npad + i] = xlv[j];
                            }
                        }
                    }
                }
            });
    }
    let t_x = t0.elapsed().as_secs_f64();
    let mut order_m: Vec<Vec<f32>> = Vec::with_capacity(models.len());
    let mut prob_m: Vec<Vec<f32>> = Vec::with_capacity(models.len());
    for (mi, mw) in models.iter().enumerate() {
        // slice this model's 6 layer blocks out of gs_all (pure copy)
        let pcs: Vec<LayerPrecomp> = mw
            .layers
            .iter()
            .enumerate()
            .map(|(li, lay)| {
                let b = (mi * 6 + li) * hw;
                let mut gs = vec![0.0f32; f * hw];
                gs.par_chunks_mut(hw).enumerate().for_each(|(i, grow)| {
                    grow.copy_from_slice(&gs_all[i * w + b..i * w + b + hw]);
                });
                precomp_layer(lay, gs)
            })
            .collect();
        let mut logits = vec![0.0f32; npad * 6];
        // cell-tiled forward (k-major accumulators, one fused j-sweep over the
        // model's 6 layer blocks):
        //  - bg[k][g] accumulates xd*(gs row) at j where xl[g][j] != 0, ascending
        //    (dense gs values — cannot be masked). Lanes with xd == 0.0 add
        //    xd*g == ±0.0 — an exact identity on the non-negative accumulators.
        //  - r[k][g]/raw[k][g] accumulate only where wbits[j][k] == 1 (bit set),
        //    j ascending; raw's unconditional add of xd == 0.0 is an identity.
        //  - per output element the sequence of f64 additions is unchanged, so
        //    r/bg/raw (and the head logits) are bit-identical to the group-of-8
        //    scalar loop this replaces.
        //  - parallelism is over disjoint cell tiles; padded lanes (i >= n) hold
        //    exact zeros in xt/xl_t and never reach head_finish.
        logits
            .par_chunks_mut(TILE * 6)
            .enumerate()
            .for_each(|(ti, ztile)| {
                let g0 = ti * TILE;
                let gvalid = TILE.min(n - g0.min(n));
                let mut r = vec![0.0f64; 6 * 24 * TILE];
                let mut bg = vec![0.0f64; 6 * 24 * TILE];
                let mut raw = vec![0.0f64; 6 * 24 * TILE];
                let mut xd = [0.0f64; TILE];
                let mut vd = [0.0f64; TILE];
                for j in 0..f {
                    let xlrow = &xl_t[j * npad + g0..j * npad + g0 + TILE];
                    let xtrow = &xt[j * npad + g0..j * npad + g0 + TILE];
                    let mut any = false;
                    for g in 0..TILE {
                        let xv = xlrow[g];
                        if xv != 0.0 {
                            xd[g] = xv as f64;
                            any = true;
                        } else {
                            xd[g] = 0.0;
                        }
                    }
                    for (li, pc) in pcs.iter().enumerate() {
                        if any {
                            let gr = &pc.gs[j * 24..j * 24 + 24];
                            for k in 0..24 {
                                let gk = gr[k] as f64;
                                let bk = &mut bg[(li * 24 + k) * TILE..(li * 24 + k + 1) * TILE];
                                for g in 0..TILE {
                                    bk[g] += xd[g] * gk;
                                }
                            }
                        }
                        let mut mask = pc.rowmask[j];
                        if mask != 0 {
                            let mr = pc.maxrank;
                            for g in 0..TILE {
                                vd[g] = xtrow[g].min(mr) as f64;
                            }
                            while mask != 0 {
                                let k = mask.trailing_zeros() as usize;
                                mask &= mask - 1;
                                let rk = &mut r[(li * 24 + k) * TILE..(li * 24 + k + 1) * TILE];
                                let rw = &mut raw[(li * 24 + k) * TILE..(li * 24 + k + 1) * TILE];
                                for g in 0..TILE {
                                    rk[g] += vd[g];
                                    rw[g] += xd[g];
                                }
                            }
                        }
                    }
                }
                for g in 0..gvalid {
                    let i = g0 + g;
                    for (hi, pc) in pcs.iter().enumerate() {
                        let mut r32 = [0.0f32; 24];
                        let mut bg32 = [0.0f32; 24];
                        let mut raw32 = [0.0f32; 24];
                        for k in 0..24 {
                            r32[k] = r[(hi * 24 + k) * TILE + g] as f32;
                            bg32[k] = bg[(hi * 24 + k) * TILE + g] as f32;
                            raw32[k] = raw[(hi * 24 + k) * TILE + g] as f32;
                        }
                        if debug_fwd() && i < 4 {
                            eprintln!(
                                "DBG cell{} layer{} bg[:4]={:?} raw[:4]={:?} r[:4]={:?}",
                                i,
                                hi,
                                &bg32[..4],
                                &raw32[..4],
                                &r32[..4]
                            );
                        }
                        ztile[g * 6 + hi] = head_finish(r32, bg32, raw32, pc);
                    }
                }
            });
        let logits = &logits[..n * 6];
        let mut order = vec![0.0f32; n];
        let mut prob = vec![0.0f32; n * 6];
        pool.install(|| {
            prob.par_chunks_mut(6)
                .zip(order.par_chunks_mut(1))
                .zip(logits.par_chunks(6))
                .for_each(|((prow, orow), zrow)| {
                    let m = zrow.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut e = [0.0f32; 6];
                    let mut s = 0.0f32;
                    for h in 0..6 {
                        e[h] = (zrow[h] - m).exp();
                        s += e[h];
                    }
                    let mut o = 0.0f32;
                    for h in 0..6 {
                        prow[h] = e[h] / s;
                        o += prow[h] * ORDER_VEC_F32[h];
                    }
                    orow[0] = o;
                });
        });
        order_m.push(order);
        prob_m.push(prob);
    }
    drop(gs_all);
    if time_sub {
        eprintln!(
            "SUB ensemble {{\"w_all_build_s\":{:.4},\"wide_pass_s\":{:.4},\"x_transpose_s\":{:.4},\"forward_s\":{:.4}}}",
            t_w,
            t_g - t_w,
            t_x - t_g,
            t0.elapsed().as_secs_f64() - t_x
        );
    }

    let nm = models.len();
    let predicted_order: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let col: Vec<f32> = (0..nm).map(|m| order_m[m][i]).collect();
            np_sum_f32(&col) / nm as f32
        })
        .collect();
    let prob_mean: Vec<f32> = (0..n * 6)
        .into_par_iter()
        .map(|k| {
            let col: Vec<f32> = (0..nm).map(|m| prob_m[m][k]).collect();
            np_sum_f32(&col) / nm as f32
        })
        .collect();
    let predicted_potency: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut best = 0usize;
            let mut bv = prob_mean[i * 6];
            for h in 1..6 {
                if prob_mean[i * 6 + h] > bv {
                    bv = prob_mean[i * 6 + h];
                    best = h;
                }
            }
            best
        })
        .collect();
    EnsembleOut {
        order_models: order_m,
        prob_models: prob_m,
        predicted_order,
        predicted_potency,
        prob_mean,
    }
}

pub const ORDER_VEC_F32: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

// ---------------- smoothing by diffusion ----------------

pub struct SmoothDump<'a> {
    pub dir: Option<&'a Path>,
}

pub fn smoothing_by_diffusion(
    preknn_f32: &[f32],
    log2: &[f64],
    top_col: &[usize],
    n: usize,
    smooth_batch_size: usize,
    rng: &mut Mt19937,
    dump: SmoothDump,
) -> Vec<f64> {
    let f = N_FEATURES;
    // reseed + unconditional shuffle
    // caller seeds; here we shuffle arange(n)
    let mut perm: Vec<u32> = (0..n as u32).collect();
    rng.shuffle_indices(&mut perm);
    let chunk_number = if smooth_batch_size > n {
        1
    } else {
        (n + smooth_batch_size - 1) / smooth_batch_size
    };
    // np.array_split semantics
    let base = n / chunk_number;
    let rem = n % chunk_number;
    let mut starts = Vec::with_capacity(chunk_number + 1);
    let mut acc = 0usize;
    for c in 0..chunk_number {
        starts.push(acc);
        acc += base + if c < rem { 1 } else { 0 };
    }
    starts.push(acc);

    let results: Vec<(Vec<f64>, usize)> = (0..chunk_number)
        .into_par_iter()
        .map(|c| {
            let s = starts[c];
            let e = starts[c + 1];
            let cn = e - s;
            // gather chunk log2 rows (shuffled order) restricted to top1000
            let mut sub = vec![0.0f64; cn * top_col.len()];
            for (ri, &pi) in perm[s..e].iter().enumerate() {
                let src = &log2[pi as usize * f..(pi as usize + 1) * f];
                for (tj, &gj) in top_col.iter().enumerate() {
                    sub[ri * top_col.len() + tj] = src[gj];
                }
            }
            // corrcoef
            // micro-opt: row-parallel — every cmat[i][j] still accumulates its
            // j-ascending dot in the same order; whole-matrix np_sum stays serial
            // (pairwise order must not change).
            let t_corr0 = std::time::Instant::now();
            let g = top_col.len();
            let mut xc = sub.clone();
            xc.par_chunks_mut(g)
                .zip(sub.par_chunks(g))
                .for_each(|(xrow, srow)| {
                    let m = np_sum_f64(srow) / g as f64;
                    for t in 0..g {
                        xrow[t] = srow[t] - m;
                    }
                });
            let mut cmat = vec![0.0f64; cn * cn];
            cmat.par_chunks_mut(cn).enumerate().for_each(|(i, crow)| {
                let ri = &xc[i * g..(i + 1) * g];
                for j in 0..cn {
                    let rj = &xc[j * g..(j + 1) * g];
                    let mut d = 0.0f64;
                    for t in 0..g {
                        d += ri[t] * rj[t];
                    }
                    crow[j] = d;
                }
            });
            let fact = (g as f64 - 1.0).recip();
            cmat.par_iter_mut().for_each(|v| {
                *v *= fact;
            });
            let sd: Vec<f64> = (0..cn).map(|i| cmat[i * cn + i].sqrt()).collect();
            cmat.par_chunks_mut(cn).enumerate().for_each(|(i, crow)| {
                for j in 0..cn {
                    crow[j] /= sd[i];
                }
            });
            cmat.par_chunks_mut(cn).enumerate().for_each(|(_i, crow)| {
                for j in 0..cn {
                    crow[j] /= sd[j];
                }
            });
            // np.clip(c.real, -1, 1)
            cmat.par_iter_mut().for_each(|v| {
                if *v > 1.0 {
                    *v = 1.0;
                } else if *v < -1.0 {
                    *v = -1.0;
                }
            });
            let t_corr = t_corr0.elapsed().as_secs_f64();
            if let Some(d) = dump.dir {
                crate::npyio::write_npy_f64(&d.join(format!("corr_{}.npy", c)), &[cn, cn], &cmat);
            }
            // diag=0, NaN->0, cutoff, threshold (official order)
            for i in 0..cn {
                cmat[i * cn + i] = 0.0;
            }
            cmat.par_iter_mut().for_each(|v| {
                if !v.is_finite() {
                    *v = 0.0;
                }
            });
            let mean_all = np_sum_f64(&cmat) / (cn * cn) as f64;
            let cutoff = if mean_all > 0.0 { mean_all } else { 0.0 };
            cmat.par_iter_mut().for_each(|v| {
                if *v < cutoff {
                    *v = 0.0;
                }
            });
            let mut a = vec![0.0f64; cn * cn];
            a.par_chunks_mut(cn)
                .zip(cmat.par_chunks(cn))
                .for_each(|(arow, row)| {
                    let rs = np_sum_f64(row);
                    let d = rs + 1e-5;
                    for j in 0..cn {
                        arow[j] = row[j] / d;
                    }
                });
            if let Some(d) = dump.dir {
                crate::npyio::write_npy_f64(&d.join(format!("markov_{}.npy", c)), &[cn, cn], &a);
                crate::npyio::write_npy_f64(
                    &d.join(format!("corr_cutoff_{}.npy", c)),
                    &[1],
                    &[cutoff],
                );
            }
            // iterate
            let init: Vec<f64> = perm[s..e]
                .iter()
                .map(|&pi| preknn_f32[pi as usize] as f64)
                .collect();
            let init32: Vec<f32> = perm[s..e]
                .iter()
                .map(|&pi| preknn_f32[pi as usize])
                .collect();
            let mean_init32 = np_sum_f32(&init32) / cn as f32;
            let denom_conv = mean_init32 as f64 + 1e-6;
            let t_iter0 = std::time::Instant::now();
            let mut prev = init.clone();
            let mut cur = init.clone();
            let mut iters = 0usize;
            for _ in 0..10000usize {
                let mut next = vec![0.0f64; cn];
                const IB: usize = 64;
                next.par_chunks_mut(IB).enumerate().for_each(|(bi, nrow)| {
                    for (ii, n) in nrow.iter_mut().enumerate() {
                        let i = bi * IB + ii;
                        let arow = &a[i * cn..(i + 1) * cn];
                        let mut acc = 0.0f64;
                        for j in 0..cn {
                            acc += arow[j] * prev[j];
                        }
                        *n = 0.9 * acc + 0.1 * init[i];
                    }
                });
                cur = next;
                iters += 1;
                let mut diff: Vec<f64> = vec![0.0f64; cur.len()];
                diff.par_iter_mut()
                    .zip(cur.par_iter().zip(prev.par_iter()))
                    .for_each(|(d, (&a, &b))| {
                        *d = (a - b).abs();
                    });
                let dm = np_sum_f64(&diff) / cn as f64;
                if dm / denom_conv < 1e-6 {
                    break;
                }
                prev = cur.clone();
            }
            let t_iter = t_iter0.elapsed().as_secs_f64();
            if std::env::var("C2RUST_TIME_SUB").is_ok() {
                eprintln!(
                    "SUB smooth {{\"chunk\":{},\"corr_s\":{:.4},\"iter_s\":{:.4},\"iters\":{}}}",
                    c, t_corr, t_iter, iters
                );
            }
            if let Some(d) = dump.dir {
                crate::npyio::write_npy_f64(
                    &d.join(format!("smooth_score_{}.npy", c)),
                    &[cn],
                    &cur,
                );
                crate::npyio::write_npy_f64(
                    &d.join(format!("smooth_iters_{}.npy", c)),
                    &[1],
                    &[iters as f64],
                );
            }
            (cur, iters)
        })
        .collect();
    // concatenate in chunk order then reorder to original cell order
    let mut concat = Vec::with_capacity(n);
    for c in 0..chunk_number {
        concat.extend_from_slice(&results[c].0);
    }
    let mut out = vec![0.0f64; n];
    for (pos, &pi) in perm.iter().enumerate() {
        out[pi as usize] = concat[pos];
    }
    out
}

// ---------------- binning ----------------

pub const LABELS: [&str; 6] = [
    "Differentiated",
    "Unipotent",
    "Oligopotent",
    "Multipotent",
    "Pluripotent",
    "Totipotent",
];

// np.arange(7)/6 exact f64 (division each)
pub const LIMITS_DIV6: [f64; 7] = [
    0.0,
    1.0 / 6.0,
    2.0 / 6.0,
    3.0 / 6.0,
    4.0 / 6.0,
    5.0 / 6.0,
    6.0 / 6.0,
];

pub fn binning(preknn_potency: &[usize], smooth: &[f64]) -> (Vec<f64>, Vec<Vec<usize>>) {
    let n = smooth.len();
    let mut binned = vec![0.0f64; n];
    let mut group_orders: Vec<Vec<usize>> = Vec::new();
    for (li, &lab) in LABELS.iter().enumerate() {
        let mut idxs: Vec<usize> = (0..n).filter(|&i| preknn_potency[i] == li).collect();
        if idxs.is_empty() {
            continue;
        }
        // pandas sort_values ascending (quicksort)
        idxs.sort_unstable_by(|&a, &b| smooth[a].partial_cmp(&smooth[b]).unwrap());
        let m = idxs.len();
        let lower = LIMITS_DIV6[li] + 1e-8;
        let upper = LIMITS_DIV6[li + 1] - 1e-8;
        let scale = (upper - lower) / (m as f64 - 1.0);
        for (k, &ci) in idxs.iter().enumerate() {
            binned[ci] = k as f64 * scale + lower;
        }
        group_orders.push(idxs);
    }
    (binned, group_orders)
}
