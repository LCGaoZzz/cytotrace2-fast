// numpy legacy RandomState (MT19937) reproduction — verified against numpy 1.26.4:
//  - seed(int) = init_genrand
//  - shuffle(contiguous array) = downward Fisher-Yates with rk_interval (mask-smeared rejection)
//  - random_sample = genrand_res53
//  - uniform(-1,1,n) = -1 + 2*res53
pub struct Mt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut s = Mt19937 {
            mt: [0u32; 624],
            mti: 624,
        };
        s.mt[0] = seed;
        for i in 1..624usize {
            let prev = s.mt[i - 1];
            s.mt[i] = (1812433253u32.wrapping_mul(prev ^ (prev >> 30))).wrapping_add(i as u32);
        }
        s
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.mti >= 624 {
            for i in 0..624usize {
                let y = (self.mt[i] & 0x8000_0000u32) | (self.mt[(i + 1) % 624] & 0x7fff_ffffu32);
                self.mt[i] = self.mt[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ (if y & 1 != 0 { 0x9908_b0dfu32 } else { 0 });
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// genrand_res53 — one random_sample() double (2 draws)
    #[inline]
    pub fn res53(&mut self) -> f64 {
        let a = self.next_u32();
        let b = self.next_u32();
        ((a >> 5) as f64 * 67108864.0 + (b >> 6) as f64) * (1.0 / 9007199254740992.0)
    }

    /// rk_interval(max) — bounded int in [0, max] via mask rejection
    #[inline]
    pub fn rk_interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// np.random.shuffle on a 0..n index array (int64/float64 contiguous path)
    pub fn shuffle_indices(&mut self, perm: &mut [u32]) {
        let n = perm.len();
        for i in (1..n).rev() {
            let j = self.rk_interval(i as u32) as usize;
            perm.swap(i, j);
        }
    }

    /// np.random.uniform(-1, 1, size) on the continuing stream
    pub fn uniform_m1_1(&mut self, out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = -1.0 + 2.0 * self.res53();
        }
    }
}
