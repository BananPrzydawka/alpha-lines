//! Seedable PRNG standing in for numba's `np.random` inside the kernels.
//!
//! numba's per-thread Mersenne Twister cannot be reproduced bit-for-bit here, so the port
//! keeps the *structure* of the random draws identical (same number of draws, in the same
//! order, at the same points) and swaps the bit source. Everything the cross-check against
//! Python compares is driven by explicit actions, never by these draws.

/// xoshiro256++, seeded through SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// `np.random.random()`: a double in [0, 1) with 53 bits of entropy.
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `np.random.randint(0, n)`: a uniform integer in [0, n), debiased (Lemire).
    #[inline]
    pub fn randint(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randint bound must be positive");
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let threshold = n.wrapping_neg() % n;
            while l < threshold {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }
}
