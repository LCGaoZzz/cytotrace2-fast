// numpy pairwise summation (f64 and f32), matching numpy's pairwise_sum C code:
//  n < 8        -> sequential
//  n <= 128     -> 8 accumulators (r[i]=a[i]; i=8..n-(n%8) step 8; combine; tail)
//  n > 128      -> n2 = n/2 (floor to multiple of 8); recursive halves
#[inline]
pub fn np_sum_f64(a: &[f64]) -> f64 {
    pairwise_f64(a, a.len())
}
fn pairwise_f64(a: &[f64], n: usize) -> f64 {
    if n < 8 {
        let mut r = 0.0f64;
        for i in 0..n {
            r += a[i];
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        for j in 0..8 {
            r[j] = a[j];
        }
        let n2 = n - (n % 8);
        let mut i = 8;
        while i < n2 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_f64(&a[..n2], n2) + pairwise_f64(&a[n2..], n - n2)
    }
}

#[inline]
pub fn np_sum_f32(a: &[f32]) -> f32 {
    pairwise_f32(a, a.len())
}
fn pairwise_f32(a: &[f32], n: usize) -> f32 {
    if n < 8 {
        let mut r = 0.0f32;
        for i in 0..n {
            r += a[i];
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        for j in 0..8 {
            r[j] = a[j];
        }
        let n2 = n - (n % 8);
        let mut i = 8;
        while i < n2 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_f32(&a[..n2], n2) + pairwise_f32(&a[n2..], n - n2)
    }
}

/// sequential summation (used where numpy reduces over non-contiguous strides)
pub fn seq_sum_f64(a: &[f64]) -> f64 {
    let mut r = 0.0f64;
    for &v in a {
        r += v;
    }
    r
}
