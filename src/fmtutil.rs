// Python str(float) compatible formatting (shortest round-trip).
pub fn fmt_py(v: f64) -> String {
    if v.is_nan() {
        return String::new(); // pandas to_csv writes '' for NaN
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    let a = v.abs();
    if v != 0.0 && (a < 1e-4 || a >= 1e16) {
        // Python switches to scientific notation
        let e = format!("{:e}", v); // e.g. "1.5e-7", "1e20", "-2.5e-9"
        let (m, ex) = match e.split_once('e') {
            Some(p) => p,
            None => return e,
        };
        let exv: i32 = ex.parse().unwrap_or(0);
        let sign = if exv < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", m, sign, exv.abs())
    } else {
        let s = format!("{}", v);
        if s.contains('.') {
            s
        } else {
            s + ".0"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fmt_py;
    #[test]
    fn basics() {
        assert_eq!(fmt_py(0.0), "0.0");
        assert_eq!(fmt_py(1.0), "1.0");
        assert_eq!(fmt_py(-1.0), "-1.0");
        assert_eq!(fmt_py(0.05548854984130043), "0.05548854984130043");
        assert_eq!(fmt_py(1e-5), "1e-05");
        assert_eq!(fmt_py(1.5e-7), "1.5e-07");
        assert_eq!(fmt_py(1e20), "1e+20");
        assert_eq!(fmt_py(f64::NAN), "");
    }
}
