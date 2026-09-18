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

/// PCA (arpack top-30) via dense symmetric eigendecomposition of the Gram matrix.
/// Returns embedding (n x 30) = U * sqrt(lambda), eigenvalues descending.
pub fn pca_embedding(data_scale: &[f64], n: usize) -> Vec<f64> {
    let f = crate::pipeline::N_FEATURES;
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
    emb
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
