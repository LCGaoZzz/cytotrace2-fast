use super::*;
use faer::{Mat, Side};

fn centered_random(n: usize, f: usize) -> Vec<f64> {
    let mut rng = crate::rng::Mt19937::new(1729);
    let mut x: Vec<f64> = (0..n * f).map(|_| rng.res53() - 0.5).collect();
    for row in x.chunks_mut(n) {
        let mean = row.iter().sum::<f64>() / n as f64;
        for v in row {
            *v -= mean;
        }
    }
    x
}

fn prescribed_spectrum(n: usize, values: &[f64]) -> Vec<f64> {
    // Non-constant DCT columns are orthonormal and centered. G has the
    // prescribed nonzero eigenvalues; the remaining eigenvalues are zero.
    let mut x = vec![0.0; n * values.len()];
    for (k, &lambda) in values.iter().enumerate() {
        for i in 0..n {
            let angle = std::f64::consts::PI * (i as f64 + 0.5) * (k + 1) as f64 / n as f64;
            x[k * n + i] = (2.0 * lambda / n as f64).sqrt() * angle.cos();
        }
    }
    x
}

fn full_embedding(x: &[f64], n: usize, f: usize) -> Vec<f64> {
    let g = Mat::from_fn(n, n, |i, j| {
        (0..f).map(|k| x[k * n + i] * x[k * n + j]).sum::<f64>()
    });
    let evd = g.self_adjoint_eigen(Side::Lower).unwrap();
    let npc = 30.min(n - 1);
    let mut out = vec![0.0; n * npc];
    for i in 0..n {
        for c in 0..npc {
            let j = n - 1 - c;
            out[i * npc + c] = evd.U()[(i, j)] * evd.S()[j].max(0.0).sqrt();
        }
    }
    out
}

fn sqdist(e: &[f64], n: usize, i: usize, j: usize) -> f64 {
    let k = 30.min(n - 1);
    (0..k)
        .map(|c| (e[i * k + c] - e[j * k + c]).powi(2))
        .sum()
}

fn check_distances(x: &[f64], n: usize, f: usize, tol: f64) {
    let (new, m, matvecs) = lanczos_pca_embedding(x, n, f).unwrap();
    assert!(m <= n.min(300));
    assert_eq!(matvecs, m + 30.min(n - 1));
    let old = full_embedding(x, n, f);
    let mut error = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let reference = sqdist(&old, n, i, j);
            error = error.max((sqdist(&new, n, i, j) - reference).abs());
            scale = scale.max(reference);
        }
    }
    assert!(
        error <= tol * scale,
        "distance error {error:e}, scale {scale:e}"
    );
}

#[test]
fn lanczos_matches_full_distances_neighbors_and_scores() {
    let (n, f) = (128, 70);
    let x = centered_random(n, f);
    check_distances(&x, n, f, 1e-8);
    let (new, _, _) = lanczos_pca_embedding(&x, n, f).unwrap();
    let old = full_embedding(&x, n, f);
    for i in 0..n {
        let order = |e: &[f64]| {
            let mut ids: Vec<usize> = (0..n).collect();
            ids.sort_by(|&a, &b| sqdist(e, n, i, a).total_cmp(&sqdist(e, n, i, b)));
            ids[..30].to_vec()
        };
        assert_eq!(order(&new), order(&old));
    }
    let binned: Vec<f64> = (0..n).map(|i| 0.17 + (i % 30) as f64 * 0.01).collect();
    let a = knn_smooth(&binned, &old, n, 1, None);
    let b = knn_smooth(&binned, &new, n, 1, None);
    for (x, y) in a.score.iter().zip(&b.score) {
        assert!((x - y).abs() < 1e-8);
        assert_eq!(cut_potency(*x), cut_potency(*y));
    }
}

#[test]
fn lanczos_small_dimension_never_exceeds_ambient_space() {
    check_distances(&centered_random(12, 9), 12, 9, 1e-8);
}

#[test]
fn lanczos_rank_deficient_input() {
    check_distances(&centered_random(96, 8), 96, 8, 1e-8);
}

#[test]
fn lanczos_cluster_near_truncation_boundary() {
    let mut values: Vec<f64> = (0..29).map(|i| 100.0 - i as f64).collect();
    values.extend([20.000001, 19.999999]);
    values.extend((1..40).map(|i| 5.0 / i as f64));
    check_distances(&prescribed_spectrum(128, &values), 128, values.len(), 1e-6);
}

#[test]
fn lanczos_repeated_eigenvalues_inside_retained_subspace() {
    let mut values: Vec<f64> = (0..24).map(|i| 100.0 - i as f64).collect();
    values.extend([20.0; 6]); // Whole multiplicity is above the cutoff.
    values.extend((1..40).map(|i| 5.0 / i as f64));
    check_distances(&prescribed_spectrum(128, &values), 128, values.len(), 1e-8);
}

#[test]
fn lanczos_iteration_cap_fails_instead_of_returning_partial_result() {
    let x = centered_random(128, 100);
    let err = lanczos_with_limit(&x, 128, 100, 31).unwrap_err();
    assert!(err.contains("did not converge"), "{err}");
}

#[test]
fn lanczos_rejects_invalid_and_degenerate_input() {
    assert!(lanczos_pca_embedding(&[], 0, 0).is_err());
    assert!(lanczos_pca_embedding(&[1.0], 1, 1).is_err());
    assert!(lanczos_pca_embedding(&[1.0], 2, 2).is_err());
    assert!(lanczos_pca_embedding(&[f64::NAN, 1.0], 2, 1).is_err());
    assert!(lanczos_pca_embedding(&[f64::INFINITY, 1.0], 2, 1).is_err());
    assert!(lanczos_pca_embedding(&vec![0.0; 64 * 4], 64, 4).is_err());
}

#[test]
fn lanczos_repeatability_and_thread_width() {
    let x = centered_random(96, 48);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let two = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let first = one.install(|| lanczos_pca_embedding(&x, 96, 48).unwrap().0);
    let repeat = one.install(|| lanczos_pca_embedding(&x, 96, 48).unwrap().0);
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let other = two.install(|| lanczos_pca_embedding(&x, 96, 48).unwrap().0);
    for (a, b) in first.iter().zip(other.iter()) {
        assert!((a - b).abs() < 1e-8);
    }
}
