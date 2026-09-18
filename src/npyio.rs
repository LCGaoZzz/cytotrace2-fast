// Minimal npz (zip) reader + npy writer.
use std::fs;
use std::io::Read;
use std::path::Path;

#[derive(Clone, Debug)]
pub enum Arr {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

impl Arr {
    pub fn len(&self) -> usize {
        match self {
            Arr::F32(v) => v.len(),
            Arr::F64(v) => v.len(),
            Arr::I64(v) => v.len(),
        }
    }
    pub fn as_f32(self) -> Vec<f32> {
        match self {
            Arr::F32(v) => v,
            _ => panic!("expected f32"),
        }
    }
    pub fn as_f64(self) -> Vec<f64> {
        match self {
            Arr::F64(v) => v,
            _ => panic!("expected f64"),
        }
    }
    pub fn as_i64(self) -> Vec<i64> {
        match self {
            Arr::I64(v) => v,
            _ => panic!("expected i64"),
        }
    }
}

#[derive(Debug)]
struct NpyHeader {
    dtype: String, // "<f4" | "<f8" | "<i8"
    shape: Vec<usize>,
    data_offset: usize,
}

fn parse_npy(buf: &[u8]) -> NpyHeader {
    assert_eq!(&buf[0..6], b"\x93NUMPY", "npy magic");
    let major = buf[6];
    let hlen = if major == 1 {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    } else {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    };
    let hs = if major == 1 { 10 } else { 12 };
    let hdr = std::str::from_utf8(&buf[hs..hs + hlen]).unwrap();
    // parse 'descr': '<f4' and 'shape': (2850, 24) by scanning quotes/parens
    let di = hdr.find("descr").unwrap();
    let ci = hdr[di..].find(':').unwrap() + di;
    let q1 = hdr[ci..].find('\'').unwrap() + ci;
    let q2 = hdr[q1 + 1..].find('\'').unwrap() + q1 + 1;
    let dtype = hdr[q1 + 1..q2].to_string();
    let si = hdr.find("shape").unwrap();
    let p1 = hdr[si..].find('(').unwrap() + si;
    let p2 = hdr[p1..].find(')').unwrap() + p1;
    let shape: Vec<usize> = hdr[p1 + 1..p2]
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse().unwrap())
        .collect();
    NpyHeader {
        dtype,
        shape,
        data_offset: hs + hlen,
    }
}

fn read_entry_npy(bytes: Vec<u8>) -> Arr {
    let h = parse_npy(&bytes);
    let data = &bytes[h.data_offset..];
    let nel: usize = h.shape.iter().product();
    match h.dtype.as_str() {
        "<f4" | "|f4" => {
            assert_eq!(data.len(), nel * 4, "f4 size");
            let mut v = Vec::with_capacity(nel);
            for c in data.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Arr::F32(v)
        }
        "<f8" => {
            assert_eq!(data.len(), nel * 8, "f8 size");
            let mut v = Vec::with_capacity(nel);
            for c in data.chunks_exact(8) {
                v.push(f64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ]));
            }
            Arr::F64(v)
        }
        "<i8" => {
            let mut v = Vec::with_capacity(nel);
            for c in data.chunks_exact(8) {
                v.push(i64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ]));
            }
            Arr::I64(v)
        }
        other => panic!("unsupported npy dtype {}", other),
    }
}

/// Read all entries of an npz file into (name, Arr).
pub fn read_npz(path: &Path) -> Vec<(String, Arr)> {
    let raw = fs::read(path).expect("read npz");
    read_npz_bytes(raw)
}

/// Read all entries of an already-loaded npz byte buffer into (name, Arr).
pub fn read_npz_bytes(raw: Vec<u8>) -> Vec<(String, Arr)> {
    // locate EOCD
    let mut eocd = None;
    let scan_start = raw.len().saturating_sub(66_000);
    for i in (scan_start..raw.len().saturating_sub(21)).rev() {
        if &raw[i..i + 4] == b"PK\x05\x06" {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.expect("EOCD not found");
    let n_entries = u16::from_le_bytes([raw[eocd + 10], raw[eocd + 11]]) as usize;
    let cd_off = u32::from_le_bytes([
        raw[eocd + 16],
        raw[eocd + 17],
        raw[eocd + 18],
        raw[eocd + 19],
    ]) as usize;
    let mut out = Vec::with_capacity(n_entries);
    let mut p = cd_off;
    for _ in 0..n_entries {
        assert_eq!(&raw[p..p + 4], b"PK\x01\x02", "central dir magic");
        let method = u16::from_le_bytes([raw[p + 10], raw[p + 11]]);
        let csize =
            u32::from_le_bytes([raw[p + 20], raw[p + 21], raw[p + 22], raw[p + 23]]) as usize;
        let usize_ =
            u32::from_le_bytes([raw[p + 24], raw[p + 25], raw[p + 26], raw[p + 27]]) as usize;
        let nlen = u16::from_le_bytes([raw[p + 28], raw[p + 29]]) as usize;
        let elen = u16::from_le_bytes([raw[p + 30], raw[p + 31]]) as usize;
        let lho = u32::from_le_bytes([raw[p + 42], raw[p + 43], raw[p + 44], raw[p + 45]]) as usize;
        let name_raw = String::from_utf8_lossy(&raw[p + 46..p + 46 + nlen]).to_string();
        let name = name_raw
            .strip_suffix(".npy")
            .unwrap_or(&name_raw)
            .to_string();
        // local header
        assert_eq!(&raw[lho..lho + 4], b"PK\x03\x04", "local magic");
        let lnlen = u16::from_le_bytes([raw[lho + 26], raw[lho + 27]]) as usize;
        let lelen = u16::from_le_bytes([raw[lho + 28], raw[lho + 29]]) as usize;
        let dstart = lho + 30 + lnlen + lelen;
        let comp = &raw[dstart..dstart + csize];
        let bytes: Vec<u8> = if method == 0 {
            comp.to_vec()
        } else if method == 8 {
            let mut dec = flate2::read::DeflateDecoder::new(comp);
            let mut v = Vec::with_capacity(usize_);
            dec.read_to_end(&mut v).expect("inflate");
            v
        } else {
            panic!("unsupported zip method {}", method);
        };
        out.push((name, read_entry_npy(bytes)));
        p += 46 + nlen + elen;
    }
    out
}

fn dtype_letter<T>() -> &'static str {
    unreachable!()
}

pub fn write_npy_f64(path: &Path, shape: &[usize], data: &[f64]) {
    write_npy(path, shape, "<f8", 8, |i, out| {
        out.copy_from_slice(&data[i].to_le_bytes())
    });
}
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) {
    write_npy(path, shape, "<f4", 4, |i, out| {
        out.copy_from_slice(&data[i].to_le_bytes())
    });
}
pub fn write_npy_i64(path: &Path, shape: &[usize], data: &[i64]) {
    write_npy(path, shape, "<i8", 8, |i, out| {
        out.copy_from_slice(&data[i].to_le_bytes())
    });
}

fn write_npy<F: Fn(usize, &mut [u8])>(
    path: &Path,
    shape: &[usize],
    dtype: &str,
    elsz: usize,
    put: F,
) {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let inner: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        format!("({})", inner.join(", "))
    };
    let mut hdr = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype, shape_str
    );
    // pad so that total header (10 + hdr + newline) is multiple of 64
    let mut total = 10 + hdr.len() + 1;
    let pad = (64 - total % 64) % 64;
    for _ in 0..pad {
        hdr.push(' ');
    }
    hdr.push('\n');
    total = 10 + hdr.len();
    debug_assert!(total % 64 == 0);
    let nel: usize = shape.iter().product();
    let mut buf = Vec::with_capacity(total + nel * elsz);
    buf.extend_from_slice(b"\x93NUMPY");
    buf.extend_from_slice(&[1u8, 0u8]);
    buf.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
    buf.extend_from_slice(hdr.as_bytes());
    let mut piece = vec![0u8; elsz];
    for i in 0..nel {
        put(i, &mut piece);
        buf.extend_from_slice(&piece);
    }
    fs::write(path, buf).expect("write npy");
}
