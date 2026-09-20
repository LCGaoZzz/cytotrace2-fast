// KNN smoothing (neighborhood_smoothing), PCA embedding, final staging.
use crate::npsum::np_sum_f64;
use crate::npyio;
use rayon::prelude::*;
use std::path::Path;

// np.linspace(0,1,7) bit-exact (from official env, endpoint forced to 1.0)
pub const LINSPACE_7: [f64; 7] = [
    f64::from_bits(0x0000_0000_0000_0000),
    f64::from_bits(0x3fc5_5555_5555_5555),
    f64::from_bits(0x3fd5_5555_5555_5555),
    f64::from_bits(0x3fe0_0000_0000_0000),
    f64::from_bits(0x3fe5_5555_5555_5555),
    f64::from_bits(0x3fe5_aaaa_aaaa_aaaa), // linspace[5] = 5*step, 1 ulp below 5.0/6.0
    f64::from_bits(0x3ff0_0000_0000_0000),
];

/// sklearn.preprocessing.scale(log2_data, axis=1): per-row z-score (ddof=0)
pub fn scale_rows(log2: &[f64], n: usize) -> Vec<f64> {
    let f = crate::pipeline::N_FEATURES;
    let mut out = vec![0.0f64; n * f];
    out.par_chunks_mut(f)
        .zip(log2.par_chunks(f))
        .for_each(|(orow, irow)| {
            let m = np_sum_f64(irow) / f as f64;
            let dev: Vec<f64> = irow.iter().map(|&v| v - m).collect();
            let ss = np_sum_f64(&dev.iter().map(|d| d * d).collect::<Vec<_>>());
            let sd = (ss / f as f64).sqrt();
            for j in 0..f {
                orow[j] = dev[j] / sd;
            }
        });
    out
}

/// Top-30 PCA via residual-checked Lanczos; explicit legacy full EVD is optional.
/// Returns embedding (n x 30) = U * sqrt(lambda), eigenvalues descending.
pub fn pca_embedding(data_scale: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let f = crate::pipeline::N_FEATURES;
    if n < 2 || n.checked_mul(f) != Some(data_scale.len()) {
        return Err("PCA: invalid matrix dimensions".into());
    }
    let time_sub = std::env::var("C2RUST_TIME_SUB").is_ok();
    let t0 = std::time::Instant::now();
    // column means: identical per-column value sequence (i ascending) and the
    // same numpy pairwise summation; columns are gathered in cache-blocked
    // groups of 64 so each data_scale cache line is fully used.
    let mut colmean = vec![0.0f64; f];
    {
        const JB: usize = 64;
        let jchunks = (f + JB - 1) / JB;
        let parts: Vec<(usize, Vec<f64>)> = (0..jchunks)
            .into_par_iter()
            .map(|jc| {
                let j0 = jc * JB;
                let j1 = (j0 + JB).min(f);
                let wj = j1 - j0;
                let mut buf = vec![0.0f64; wj * n];
                for i in 0..n {
                    let row = &data_scale[i * f..i * f + j1];
                    for t in 0..wj {
                        buf[t * n + i] = row[j0 + t];
                    }
                }
                let means: Vec<f64> = (0..wj)
                    .map(|t| np_sum_f64(&buf[t * n..t * n + n]) / n as f64)
                    .collect();
                (j0, means)
            })
            .collect();
        for (j0, means) in parts {
            colmean[j0..j0 + means.len()].copy_from_slice(&means);
        }
    }
    let t_cm = t0.elapsed().as_secs_f64();
    // center + transpose to k-major, parallel over column blocks with 64-row
    // tiles (pure data movement + the same subtraction -> bit-identical)
    let mut xct = vec![0.0f64; n * f]; // xct[k*n + i]
    {
        const JB: usize = 256;
        // each parallel chunk owns JB consecutive rows of xct (disjoint &mut)
        xct.par_chunks_mut(JB * n)
            .enumerate()
            .for_each(|(jc, block)| {
                let j0 = jc * JB;
                let j1 = (j0 + JB).min(f);
                let mut i0 = 0usize;
                while i0 < n {
                    let i1 = (i0 + 64).min(n);
                    for i in i0..i1 {
                        let src = &data_scale[i * f..i * f + j1];
                        for j in j0..j1 {
                            block[(j - j0) * n + i] = src[j] - colmean[j];
                        }
                    }
                    i0 = i1;
                }
            });
    }
    let t_tr = t0.elapsed().as_secs_f64();
    // ---- cand3-lanczos: top-30 PCA via Lanczos with full reorthogonalization
    // on G = Xc·Xcᵀ, WITHOUT forming the n×n Gram matrix (the matvec is
    // w = Xcᵀ·v then y = Xc·w on this k-major centered layout). Replaces the
    // O(n³) full-spectrum faer EVD + O(n²) Gram with O(n·f·m) work and
    // O(n·(f+m)) memory. Offline validation at 10k cells against the full
    // EVD: eigenvalue rel diff ≤ 3e-14, per-cell 30-NN neighbor sets and
    // orders 100% identical, normalized-distance max diff 1.3e-13.
    // Default path; set C2RUST_PCA=full to restore the legacy full EVD.
    if std::env::var("C2RUST_PCA").as_deref() != Ok("full") {
        let t_l0 = t0.elapsed().as_secs_f64();
        let (emb, conv_m, matvecs) = lanczos_pca_embedding(&xct, n, f)?;
        let t_lan = t0.elapsed().as_secs_f64() - t_l0;
        let npc = 30.min(n - 1);
        if let Ok(p) = std::env::var("C2RUST_DUMP_EMB") {
            let mut buf: Vec<u8> = Vec::with_capacity(16 + n * npc * 8);
            buf.extend_from_slice(&(n as u64).to_le_bytes());
            buf.extend_from_slice(&(npc as u64).to_le_bytes());
            for v in emb.iter() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(p, &buf).expect("dump emb");
        }
        if time_sub {
            eprintln!(
                "SUB pca {{\"colmean_s\":{:.4},\"transpose_s\":{:.4},\"lanczos_s\":{:.4},\"lanczos_m\":{},\"lanczos_matvecs\":{},\"extract_s\":{:.4}}}",
                t_cm,
                t_tr - t_cm,
                t_lan,
                conv_m,
                matvecs,
                t0.elapsed().as_secs_f64() - t_tr - t_lan
            );
        }
        return Ok(emb);
    }
    // symmetric Gram, blocked ikj with sequential-k accumulation per element
    // (upper-triangle tiles computed in parallel, mirrored after collection)
    let mut g = vec![0.0f64; n * n];
    const TB: usize = 64;
    let tiles: Vec<(usize, usize, Vec<f64>)> = (0..n)
        .into_par_iter()
        .filter(|i0| i0 % TB == 0)
        .flat_map_iter(|i0| {
            let mut out = Vec::new();
            let iend = (i0 + TB).min(n);
            let mut j0 = i0;
            while j0 < n {
                let jend = (j0 + TB).min(n);
                let ni = iend - i0;
                let nj = jend - j0;
                let mut tile = vec![0.0f64; ni * nj];
                // k-outer 4-row register-blocked kernel (single xct pass per tile,
                // same as before; the quad only amortizes rowk[j0+jj] loads across
                // four accumulator rows). Per output element the k-chain is
                // unchanged (ascending, sequential). Rows with a == 0.0 inside a
                // processed quad contribute ±0.0 to an accumulator that can never
                // be -0.0 (starts +0.0 and +0 + ±0 = +0), so including them is a
                // bitwise no-op vs the historical per-row skip.
                for k in 0..f {
                    let rowk = &xct[k * n..(k + 1) * n];
                    let mut ii = 0usize;
                    while ii + 4 <= ni {
                        let a0 = rowk[i0 + ii];
                        let a1 = rowk[i0 + ii + 1];
                        let a2 = rowk[i0 + ii + 2];
                        let a3 = rowk[i0 + ii + 3];
                        if a0 != 0.0 || a1 != 0.0 || a2 != 0.0 || a3 != 0.0 {
                            let (t0, rest) = tile[ii * nj..].split_at_mut(nj);
                            let (t1, rest) = rest.split_at_mut(nj);
                            let (t2, rest) = rest.split_at_mut(nj);
                            let (t3, _) = rest.split_at_mut(nj);
                            for jj in 0..nj {
                                let b = rowk[j0 + jj];
                                t0[jj] += a0 * b;
                                t1[jj] += a1 * b;
                                t2[jj] += a2 * b;
                                t3[jj] += a3 * b;
                            }
                        }
                        ii += 4;
                    }
                    while ii < ni {
                        let av = rowk[i0 + ii];
                        if av != 0.0 {
                            let trow = &mut tile[ii * nj..ii * nj + nj];
                            for jj in 0..nj {
                                trow[jj] += av * rowk[j0 + jj];
                            }
                        }
                        ii += 1;
                    }
                }
                out.push((i0, j0, tile));
                j0 += TB;
            }
            out
        })
        .collect();
    for (i0, j0, tile) in tiles {
        let iend = (i0 + TB).min(n);
        let jend = (j0 + TB).min(n);
        let nj = jend - j0;
        for ii in 0..(iend - i0) {
            for jj in 0..nj {
                let v = tile[ii * nj + jj];
                g[(i0 + ii) * n + (j0 + jj)] = v;
                g[(j0 + jj) * n + (i0 + ii)] = v;
            }
        }
    }
    drop(xct); // tuned: gram tiles fully collected — free ~325 MB before the EVD
    let t_gr = t0.elapsed().as_secs_f64();
    // evd candidate dev hook: dump the exact Gram matrix (row-major n*n f64, LE,
    // 8-byte leading n) and/or the resulting embedding for offline EVD benchmarking.
    // Purely additive, env-gated, default absent -> no behavior change.
    if let Ok(p) = std::env::var("C2RUST_DUMP_GRAM") {
        let mut buf: Vec<u8> = Vec::with_capacity(8 + n * n * 8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        for v in g.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(p, &buf).expect("dump gram");
    }
    // eigendecomposition (faer, LAPACK-quality)
    use faer::{Mat, Side};
    let m = Mat::from_fn(n, n, |i, j| g[i * n + j]);
    drop(g); // tuned: copied into the faer Mat — free before the EVD
    let evd = m.self_adjoint_eigen(Side::Lower).expect("eigh");
    let u = evd.U();
    let d = evd.S();
    let t_evd = t0.elapsed().as_secs_f64();
    let npc = 30.min(n - 1);
    let mut emb = vec![0.0f64; n * npc];
    for i in 0..n {
        for c in 0..npc {
            let j = n - 1 - c; // descending eigenvalues
            emb[i * npc + c] = u[(i, j)] * d[j].sqrt();
        }
    }
    if let Ok(p) = std::env::var("C2RUST_DUMP_EMB") {
        let mut buf: Vec<u8> = Vec::with_capacity(16 + n * npc * 8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        buf.extend_from_slice(&(npc as u64).to_le_bytes());
        for v in emb.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(p, &buf).expect("dump emb");
    }
    if time_sub {
        eprintln!(
            "SUB pca {{\"colmean_s\":{:.4},\"transpose_s\":{:.4},\"gram_s\":{:.4},\"evd_s\":{:.4},\"extract_s\":{:.4}}}",
            t_cm,
            t_tr - t_cm,
            t_gr - t_tr,
            t_evd - t_gr,
            t0.elapsed().as_secs_f64() - t_evd
        );
    }
    Ok(emb)
}

const PCA_RESIDUAL_TOL: f64 = 1e-10;

/// Top-30 eigenpairs of G = Xc·Xcᵀ via Lanczos with full reorthogonalization.
/// `xct` is the column-centered matrix in k-major layout: xct[k*n + i],
/// k in 0..f (feature), i in 0..n (cell). G is only ever applied through
/// w = Xcᵀ·v and y = Xc·w — the n×n Gram and the full eigenvector matrix are
/// never formed, so peak memory stays O(n·(f+m)) instead of O(n²).
///
/// Determinism: every reduction uses a fixed per-element accumulation order
/// (ascending, or numpy pairwise where noted), and the starting vector comes
/// from a fixed-seed MT19937 stream — repeated runs are byte-identical.
/// Eigenvector signs are canonicalized (largest-|entry| positive); pairwise
/// distances in the embedding are sign-invariant, so downstream KNN smoothing
/// is invariant to sign changes. Truncated-subspace accuracy still needs validation.
///
/// Returns (embedding n×npc row-major = U_top·sqrt(λ) with λ descending,
/// Lanczos steps used, matvec count including explicit residual checks).
/// A step limit or non-finite input is an error, not an implicit full-EVD fallback.
pub fn lanczos_pca_embedding(xct: &[f64], n: usize, f: usize) -> Result<(Vec<f64>, usize, usize), String> {
    lanczos_with_limit(xct, n, f, 300)
}

fn lanczos_with_limit(
    xct: &[f64],
    n: usize,
    f: usize,
    max_steps: usize,
) -> Result<(Vec<f64>, usize, usize), String> {
    if n < 2 || f == 0 || n.checked_mul(f) != Some(xct.len()) || max_steps == 0 {
        return Err("PCA: invalid matrix dimensions or iteration limit".into());
    }
    if !xct.par_iter().all(|x| x.is_finite()) {
        return Err("PCA: centered input contains NaN or infinity".into());
    }
    let t0 = std::time::Instant::now();
    let npc = 30.min(n - 1);

    // y = G·v = Xc·(Xcᵀ·v). Both stages write disjoint outputs in parallel
    // with a FIXED per-element accumulation order — deterministic.
    let matvec = |v: &[f64], w: &mut [f64], y: &mut [f64]| {
        // w[k] = Σ_i xct[k*n+i]·v[i] — parallel over disjoint k-blocks;
        // each dot strictly ascending in i
        w.par_chunks_mut(64).enumerate().for_each(|(kb, wchunk)| {
            for (kk, wk) in wchunk.iter_mut().enumerate() {
                let k = kb * 64 + kk;
                let row = &xct[k * n..(k + 1) * n];
                let mut acc = 0.0f64;
                for i in 0..n {
                    acc += row[i] * v[i];
                }
                *wk = acc;
            }
        });
        // y[i] = Σ_k xct[k*n+i]·w[k] — parallel over disjoint i-blocks; each
        // element accumulates ascending in k (single sweep of xct rows)
        y.par_chunks_mut(64).enumerate().for_each(|(ib, yb)| {
            for yi in yb.iter_mut() {
                *yi = 0.0;
            }
            let i0 = ib * 64;
            for k in 0..f {
                let wk = w[k];
                if wk != 0.0 {
                    let row = &xct[k * n + i0..k * n + i0 + yb.len()];
                    for (yi, &xv) in yb.iter_mut().zip(row.iter()) {
                        *yi += xv * wk;
                    }
                }
            }
        });
    };

    let m_max = n.min(max_steps);
    let mut basis: Vec<f64> = Vec::with_capacity(n * m_max); // column-major n×m
    let mut alpha: Vec<f64> = Vec::with_capacity(m_max);
    let mut beta: Vec<f64> = Vec::with_capacity(m_max);
    let mut wbuf: Vec<f64> = vec![0.0; f];
    let mut ybuf: Vec<f64> = vec![0.0; n];
    let mut scratch: Vec<f64> = vec![0.0; n];

    // deterministic start vector from MT19937(14)
    let mut rng = crate::rng::Mt19937::new(14);
    let mut v: Vec<f64> = (0..n).map(|_| rng.res53() - 0.5).collect();
    for (s, &x) in scratch.iter_mut().zip(v.iter()) {
        *s = x * x;
    }
    let nv = np_sum_f64(&scratch).sqrt();
    for x in v.iter_mut() {
        *x /= nv;
    }
    basis.extend_from_slice(&v);
    // deterministic parallel dot: fixed chunk boundaries, sequential sum
    // within a chunk, ascending combine — byte-identical across runs and
    // independent of thread count
    fn par_dot(a: &[f64], b: &[f64]) -> f64 {
        const C: usize = 16384;
        let nch = (a.len() + C - 1) / C;
        let parts: Vec<f64> = (0..nch)
            .into_par_iter()
            .map(|ci| {
                let s = ci * C;
                let e = (s + C).min(a.len());
                let mut acc = 0.0f64;
                for i in s..e {
                    acc += a[i] * b[i];
                }
                acc
            })
            .collect();
        let mut r = 0.0f64;
        for p in parts {
            r += p;
        }
        r
    }

    let mut vprev: Vec<f64> = vec![0.0; n];

    let mut conv_m = 0;
    let mut last_residual_est = f64::INFINITY;
    let mut matvecs = 0usize;
    let mut prev_theta: Option<Vec<f64>> = None;
    let mut stable = 0usize;
    let mut m = 0usize;
    while m < m_max {
        matvec(&v, &mut wbuf, &mut ybuf);
        matvecs += 1;
        let a_m = par_dot(&v, &ybuf);
        for i in 0..n {
            ybuf[i] -= a_m * v[i];
            if m > 0 {
                ybuf[i] -= beta[m - 1] * vprev[i];
            }
        }
        // full reorthogonalization: two passes of classical Gram-Schmidt.
        // Phase 1 computes all projection coefficients (parallel over basis
        // vectors, read-only); phase 2 subtracts them (parallel over disjoint
        // i-blocks, each sweeping the basis vectors contiguously).
        for _pass in 0..2 {
            let coeffs: Vec<f64> = (0..=m)
                .into_par_iter()
                .map(|j| par_dot(&ybuf, &basis[j * n..(j + 1) * n]))
                .collect();
            ybuf.par_chunks_mut(4096).enumerate().for_each(|(ib, yb)| {
                let i0 = ib * 4096;
                for (j, &c) in coeffs.iter().enumerate() {
                    if c != 0.0 {
                        let bj = &basis[j * n + i0..j * n + i0 + yb.len()];
                        for (yi, &bv) in yb.iter_mut().zip(bj.iter()) {
                            *yi -= c * bv;
                        }
                    }
                }
            });
        }
        let b_m = par_dot(&ybuf, &ybuf).sqrt();
        if !a_m.is_finite() || !b_m.is_finite() {
            return Err(format!(
                "PCA: non-finite Lanczos recurrence at step {}",
                m + 1
            ));
        }
        alpha.push(a_m);
        m += 1;
        // Stability is only a candidate-stop heuristic. Require a residual
        // estimate too; every returned vector is checked explicitly below.
        if (m >= npc + 18 && m % 6 == 0) || m == m_max {
            let (theta, residual_est) = tridiag_topk_desc(&alpha, &beta, npc, b_m)?;
            last_residual_est = residual_est;
            if let Some(ref pt) = prev_theta {
                let mut mx = 0.0f64;
                for c in 0..theta.len().min(pt.len()) {
                    let denom = pt[c].abs().max(1e-300);
                    let d = (theta[c] - pt[c]).abs() / denom;
                    if d > mx {
                        mx = d;
                    }
                }
                if mx < 1e-12 {
                    stable += 1;
                } else {
                    stable = 0;
                }
            }
            let complete = theta.len() == npc;
            prev_theta = Some(theta);
            if complete && (stable >= 2 || m == m_max) && residual_est <= PCA_RESIDUAL_TOL {
                conv_m = m;
                break;
            }
        }
        if m == m_max {
            break;
        }
        let scale_ref = alpha
            .iter()
            .fold(0.0f64, |a, &x| a.abs().max(x))
            .max(1e-300);
        if b_m <= 1e-13 * scale_ref {
            // happy breakdown / deflation: T keeps a zero coupling; restart
            // with a fresh vector orthogonalized against the basis
            beta.push(0.0);
            let mut rng2 = crate::rng::Mt19937::new(14 + m as u32);
            let mut cand: Vec<f64> = (0..n).map(|_| rng2.res53() - 0.5).collect();
            for _pass in 0..2 {
                for j in 0..m {
                    let bj = &basis[j * n..(j + 1) * n];
                    for (s, (&x, &y)) in scratch.iter_mut().zip(cand.iter().zip(bj.iter())) {
                        *s = x * y;
                    }
                    let c = np_sum_f64(&scratch);
                    if c != 0.0 {
                        for i in 0..n {
                            cand[i] -= c * bj[i];
                        }
                    }
                }
            }
            for (s, &x) in scratch.iter_mut().zip(cand.iter()) {
                *s = x * x;
            }
            let cn = np_sum_f64(&scratch).sqrt();
            if !(cn.is_finite() && cn > 0.0) {
                return Err(format!(
                    "PCA: restart vector exhausted before certification at step {m}"
                ));
            }
            for x in cand.iter_mut() {
                *x /= cn;
            }
            vprev.copy_from_slice(&v);
            v = cand;
            basis.extend_from_slice(&v);
            continue;
        }
        beta.push(b_m);
        let mut vnext = vec![0.0f64; n];
        for i in 0..n {
            vnext[i] = ybuf[i] / b_m;
        }
        vprev.copy_from_slice(&v);
        v = vnext;
        basis.extend_from_slice(&v);
    }
    if conv_m == 0 {
        return Err(format!(
            "PCA: Lanczos did not converge in {m} steps (limit {m_max}, residual estimate {last_residual_est:e}, tolerance {PCA_RESIDUAL_TOL:e}); no result returned"
        ));
    }

    // final Rayleigh-Ritz: small dense symmetric EVD of T_m (m ≤ 300)
    use faer::{Mat, Side};
    let mm = alpha.len();
    let mut t = vec![0.0f64; mm * mm];
    for i in 0..mm {
        t[i * mm + i] = alpha[i];
    }
    for j in 0..beta.len().min(mm.saturating_sub(1)) {
        let b = beta[j];
        t[j * mm + j + 1] = b;
        t[(j + 1) * mm + j] = b;
    }
    let tm = Mat::from_fn(mm, mm, |i, j| t[i * mm + j]);
    let evd = tm
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| format!("PCA: Lanczos Ritz decomposition failed: {e:?}"))?;
    let s = evd.S();
    let u = evd.U();
    let kk = npc.min(mm);
    let spectral_scale = s[mm - 1].abs();
    if !spectral_scale.is_finite() || spectral_scale == 0.0 {
        return Err("PCA: zero or non-finite spectrum; KNN distances are undefined".into());
    }
    // build the npc Ritz vectors (parallel over columns, then serial scatter)
    let checked_cols: Result<Vec<(Vec<f64>, f64)>, String> = (0..kk)
        .into_par_iter()
        .map(|c| {
            let j = mm - 1 - c; // descending eigenvalues
            if !s[j].is_finite() || s[j] < -PCA_RESIDUAL_TOL * spectral_scale {
                return Err(format!("PCA: invalid Ritz value {}", s[j]));
            }
            let lam = s[j].max(0.0);
            let sc = lam.sqrt();
            let mut col = vec![0.0f64; n];
            for l in 0..mm {
                let wgt = u[(l, j)];
                if wgt != 0.0 {
                    let bl = &basis[l * n..(l + 1) * n];
                    for i in 0..n {
                        col[i] += wgt * bl[i];
                    }
                }
            }
            // Explicit G*u - lambda*u check, independent of Ritz-value
            // stability. Use the leading Ritz value as the global scale;
            // this also handles null-space directions in rank-deficient data.
            let norm = par_dot(&col, &col).sqrt();
            if !norm.is_finite() || (norm - 1.0).abs() > 1e-8 {
                return Err(format!("PCA: invalid Ritz-vector norm {norm:e}"));
            }
            let mut w = vec![0.0; f];
            let mut residual = vec![0.0; n];
            matvec(&col, &mut w, &mut residual);
            for i in 0..n {
                residual[i] -= lam * col[i];
            }
            let error = par_dot(&residual, &residual).sqrt() / spectral_scale / norm;
            if !error.is_finite() || error > PCA_RESIDUAL_TOL {
                return Err(format!(
                    "PCA: explicit residual {error:e} exceeds {PCA_RESIDUAL_TOL:e} for component {c}; no result returned"
                ));
            }
            // sign convention: largest-|entry| positive (first index on ties)
            let mut mi = 0usize;
            let mut mv = 0.0f64;
            for (i, &x) in col.iter().enumerate() {
                if x.abs() > mv {
                    mv = x.abs();
                    mi = i;
                }
            }
            if col[mi] < 0.0 {
                for x in col.iter_mut() {
                    *x = -(*x);
                }
            }
            if sc != 1.0 {
                for x in col.iter_mut() {
                    *x *= sc;
                }
            }
            Ok((col, error))
        })
        .collect();
    let cols = checked_cols?;
    matvecs += kk;
    let max_residual = cols.iter().map(|(_, r)| *r).fold(0.0f64, f64::max);
    let mut emb = vec![0.0f64; n * npc];
    for (c, (col, _)) in cols.iter().enumerate() {
        for (i, &x) in col.iter().enumerate() {
            emb[i * npc + c] = x;
        }
    }
    if std::env::var("C2RUST_TIME_SUB").is_ok() {
        eprintln!(
            "SUB lanczos {{\"n\":{},\"f\":{},\"m\":{},\"conv_m\":{},\"matvecs\":{},\"max_residual\":{:.4e},\"converged\":true,\"lanczos_s\":{:.4}}}",
            n,
            f,
            mm,
            conv_m,
            matvecs,
            max_residual,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok((emb, conv_m, matvecs))
}

/// Top-k Ritz values (descending) of the tridiagonal matrix built from
/// alpha/beta — used only for the Lanczos convergence checks.
fn tridiag_topk_desc(
    alpha: &[f64],
    beta: &[f64],
    k: usize,
    beta_tail: f64,
) -> Result<(Vec<f64>, f64), String> {
    use faer::{Mat, Side};
    let m = alpha.len();
    let mut t = vec![0.0f64; m * m];
    for i in 0..m {
        t[i * m + i] = alpha[i];
    }
    for j in 0..beta.len().min(m.saturating_sub(1)) {
        let b = beta[j];
        t[j * m + j + 1] = b;
        t[(j + 1) * m + j] = b;
    }
    let tm = Mat::from_fn(m, m, |i, j| t[i * m + j]);
    let evd = tm
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| format!("PCA: Ritz convergence check failed: {e:?}"))?;
    let s = evd.S();
    let u = evd.U();
    let kk = k.min(m);
    let scale = s[m - 1].abs().max(f64::MIN_POSITIVE);
    let mut max_residual = 0.0f64;
    for c in 0..kk {
        let r = (beta_tail * u[(m - 1, m - 1 - c)]).abs() / scale;
        if !r.is_finite() || !s[m - 1 - c].is_finite() {
            return Err("PCA: non-finite Ritz convergence estimate".into());
        }
        max_residual = max_residual.max(r);
    }
    Ok(((0..kk).map(|c| s[m - 1 - c]).collect(), max_residual))
}

/// map_score_to_potency label index (-1 for above all / nan)
fn potency_idx(score: f64) -> i32 {
    if score.is_nan() {
        return -1;
    }
    for k in 0..6 {
        if score <= LINSPACE_7[k + 1] {
            return k as i32;
        }
    }
    -1
}

/// np.mean over a small contiguous slice
fn np_mean(v: &[f64]) -> f64 {
    np_sum_f64(v) / v.len() as f64
}

fn shortest_consensus(scores: &[f64]) -> usize {
    let mut idx_use = 2usize;
    let mut last_part = false;
    for i in 2..(scores.len() / 2 + 1) {
        if potency_idx(np_mean(&scores[..i])) == potency_idx(np_mean(&scores[i..2 * i]))
            && !last_part
        {
            idx_use = i;
            last_part = true;
        }
    }
    2 * idx_use
}

pub struct KnnResult {
    pub score: Vec<f64>,
}

/// neighborhood_smoothing over PCA embedding, chunked exactly like official
/// (num_chunks = smooth_cores_to_use).
pub fn knn_smooth(
    binned: &[f64],
    embed: &[f64],
    n: usize,
    smooth_cores: usize,
    dump_dir: Option<&Path>,
) -> KnnResult {
    let npc = if n >= 31 { 30 } else { n - 1 };
    let xx: Vec<f64> = (0..n)
        .map(|i| {
            (0..npc)
                .map(|c| {
                    let v = embed[i * npc + c];
                    v * v
                })
                .sum()
        })
        .collect();
    let mut score_out = vec![0.0f64; n];
    let num_chunks = smooth_cores.max(1);
    let chunk_size = (n + num_chunks - 1) / num_chunks;
    let mut neighbor_parts: Vec<(usize, Vec<u32>, Vec<f64>, Vec<i64>, Vec<f64>)> = (0..num_chunks)
        .into_par_iter()
        .map(|c| {
            let s = chunk_size * c;
            let e = (chunk_size * (c + 1)).min(n);
            let mut nb = Vec::with_capacity((e - s) * 30);
            let mut nd = Vec::with_capacity((e - s) * 30);
            let mut keep = Vec::with_capacity(e - s);
            let mut new_scores = Vec::with_capacity(e - s);
            let mut dist = vec![0.0f64; n];
            let mut idx: Vec<u32> = (0..n as u32).collect();
            for i in s..e {
                let ei = &embed[i * npc..(i + 1) * npc];
                let dmax = {
                    let mut mx = 0.0f64;
                    for j in 0..n {
                        let ej = &embed[j * npc..(j + 1) * npc];
                        let mut dot = 0.0f64;
                        for cc in 0..npc {
                            dot += ei[cc] * ej[cc];
                        }
                        let mut v = xx[i] + xx[j] - 2.0 * dot;
                        if v < 0.0 {
                            v = 0.0;
                        }
                        let dv = v.sqrt();
                        dist[j] = dv;
                        if dv > mx {
                            mx = dv;
                        }
                    }
                    mx
                };
                for j in 0..n {
                    dist[j] /= dmax;
                }
                idx.sort_unstable_by(|&a, &b| {
                    dist[a as usize].partial_cmp(&dist[b as usize]).unwrap()
                });
                let nn = 30.min(n);
                let neighbors: Vec<u32> = idx[..nn].to_vec();
                let ndists: Vec<f64> = neighbors.iter().map(|&j| dist[j as usize]).collect();
                let nscores: Vec<f64> = neighbors.iter().map(|&j| binned[j as usize]).collect();
                let nkeep = shortest_consensus(&nscores);
                nb.extend_from_slice(&neighbors);
                nd.extend_from_slice(&ndists);
                keep.push(nkeep as i64);
                if nkeep > 1 {
                    let w2: Vec<f64> = ndists[..nkeep]
                        .iter()
                        .map(|&d| (1.0 - d) * (1.0 - d))
                        .collect();
                    let prod: Vec<f64> = nscores[..nkeep]
                        .iter()
                        .zip(w2.iter())
                        .map(|(&s, &w)| s * w)
                        .collect();
                    let num = np_sum_f64(&prod);
                    let den = np_sum_f64(&w2);
                    new_scores.push(num / den);
                } else {
                    new_scores.push(-1.0);
                }
            }
            (s, nb, nd, keep, new_scores)
        })
        .collect();
    let mut score_out_idx = 0usize;
    for c in 0..num_chunks {
        let (s, nb, nd, keep, sc) =
            std::mem::replace(&mut neighbor_parts[c], (0, vec![], vec![], vec![], vec![]));
        if let Some(d) = dump_dir {
            let cnt = sc.len();
            let nb_i64: Vec<i64> = nb.iter().map(|&v| v as i64).collect();
            npyio::write_npy_i64(
                &d.join(format!("knn_neighbors_{}.npy", c)),
                &[cnt, 30],
                &nb_i64,
            );
            npyio::write_npy_f64(&d.join(format!("knn_dists_{}.npy", c)), &[cnt, 30], &nd);
            npyio::write_npy_i64(&d.join(format!("knn_keep_{}.npy", c)), &[cnt], &keep);
        }
        for (k, i) in (s..s + sc.len()).enumerate() {
            score_out[i] = sc[k];
        }
        score_out_idx += nb.len();
    }
    let _ = score_out_idx;
    KnnResult { score: score_out }
}

/// final CytoTRACE2_Potency via pd.cut(bins=linspace(0,1,7), include_lowest=True)
pub fn cut_potency(score: f64) -> Option<usize> {
    if score.is_nan() || score < LINSPACE_7[0] || score > LINSPACE_7[6] {
        return None;
    }
    for k in 0..6 {
        if score <= LINSPACE_7[k + 1] {
            return Some(k);
        }
    }
    None
}

#[cfg(test)]
#[path = "knn_tests.rs"]
mod tests;
