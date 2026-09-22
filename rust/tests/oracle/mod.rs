//! The oracle: a port of the old `main/game_kernels.py` (deleted in the Rust port;
//! retained as an independent reference for game-engine tests).
//!
//! This is the reference the incremental engine is checked against. It lives in `tests/`
//! rather than `src/` because it is not part of the product — the engine ships alone — but
//! it is not throwaway either: it is verified byte-for-byte against the Python by
//! `xcheck/xcheck.py`, which is what makes agreeing with it evidence of anything.
//!
//! It shares no code with the engine, deliberately, down to carrying its own copy of the
//! RNG. Two implementations that agree are only interesting if they are actually two.
//!
//! Same algorithms, same iteration order, same quirks — no improvements. The numba kernels
//! `parallel` flag, which was `False` in the old `main/config.py`,
//! so every loop here is serial too.
//!
//! Array layouts match the numpy shapes, flattened row-major:
//!   boards   (N, H, W) i8      -> `boards[g * H * W + r * W + c]`
//!   masks    (N, H, W) f32     -> same indexing
//!   scores   (N, 2)    f32     -> `scores[g * 2 + player]`
//!   dist     (N, H, W) f32     -> same as masks

// Kept self-contained: the oracle must not share code with the thing it is checking.
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


pub const NON_PLAYABLE_SQUARE: i8 = 0;
pub const PLAYABLE_SQUARE: i8 = 1;
pub const REMOVED_SQUARE: i8 = 2;
pub const PLAYER_0_MARK: i8 = 3;
pub const PLAYER_1_MARK: i8 = 4;

/// `_score_player`: 8-connected flood fill from the border + diagonal run-length scan,
/// for one game/player.
pub fn score_player(boards: &[i8], g: usize, player_mark: i8, height: usize, width: usize) -> f32 {
    let hw = height * width;
    let base = g * hw;

    let mut player_mask = vec![false; hw];
    for r in 0..height {
        for c in 0..width {
            player_mask[r * width + c] = boards[base + r * width + c] == player_mark;
        }
    }

    let mut reached = vec![false; hw];
    let mut stack_r = vec![0i64; hw];
    let mut stack_c = vec![0i64; hw];
    let mut sp: usize = 0;
    for r in 0..height {
        for c in 0..width {
            if player_mask[r * width + c]
                && (r == 0 || r == height - 1 || c == 0 || c == width - 1)
            {
                reached[r * width + c] = true;
                stack_r[sp] = r as i64;
                stack_c[sp] = c as i64;
                sp += 1;
            }
        }
    }
    while sp > 0 {
        sp -= 1;
        let r = stack_r[sp];
        let c = stack_c[sp];
        for dr in -1i64..2 {
            for dc in -1i64..2 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = r + dr;
                let nc = c + dc;
                if nr >= 0 && (nr as usize) < height && nc >= 0 && (nc as usize) < width {
                    let ni = nr as usize * width + nc as usize;
                    if player_mask[ni] && !reached[ni] {
                        reached[ni] = true;
                        stack_r[sp] = nr;
                        stack_c[sp] = nc;
                        sp += 1;
                    }
                }
            }
        }
    }

    let mut score: f64 = 0.0;

    // anti-diagonals: r + c = d
    for d in 0..(height + width - 1) {
        let mut cur_len: i64 = 0;
        let mut cur_reach = false;
        let r0 = if d >= width - 1 { d - (width - 1) } else { 0 };
        let r1 = if height - 1 < d { height - 1 } else { d };
        for r in r0..=r1 {
            let c = d - r;
            if player_mask[r * width + c] {
                cur_len += 1;
                cur_reach = cur_reach || reached[r * width + c];
            } else {
                if cur_len >= 2 && cur_reach {
                    score += cur_len as f64;
                }
                cur_len = 0;
                cur_reach = false;
            }
        }
        if cur_len >= 2 && cur_reach {
            score += cur_len as f64;
        }
    }

    // main diagonals: c - r = e
    let h = height as i64;
    let w = width as i64;
    for e in -(h - 1)..w {
        let mut cur_len: i64 = 0;
        let mut cur_reach = false;
        let r0 = if -e > 0 { -e } else { 0 };
        let r1 = if h - 1 < w - 1 - e { h - 1 } else { w - 1 - e };
        let mut r = r0;
        while r <= r1 {
            let c = r + e;
            let i = r as usize * width + c as usize;
            if player_mask[i] {
                cur_len += 1;
                cur_reach = cur_reach || reached[i];
            } else {
                if cur_len >= 2 && cur_reach {
                    score += cur_len as f64;
                }
                cur_len = 0;
                cur_reach = false;
            }
            r += 1;
        }
        if cur_len >= 2 && cur_reach {
            score += cur_len as f64;
        }
    }

    score as f32
}

/// `legal_masks_kernel`: (N, H, W) float legal-move masks + (N,) counts for both players,
/// honoring the first-move half-board rule.
pub fn legal_masks_kernel(
    boards: &[i8],
    n: usize,
    move_counts: &[i32],
    half_width: usize,
    height: usize,
    width: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hw = height * width;
    let mut mask_0 = vec![0.0f32; n * hw];
    let mut mask_1 = vec![0.0f32; n * hw];
    let mut count_0 = vec![0.0f32; n];
    let mut count_1 = vec![0.0f32; n];
    for g in 0..n {
        let first = move_counts[g] == 0;
        let mut c0_total: f64 = 0.0;
        let mut c1_total: f64 = 0.0;
        let base = g * hw;
        for r in 0..height {
            for c in 0..width {
                if boards[base + r * width + c] == PLAYABLE_SQUARE {
                    if !(first && c >= half_width) {
                        mask_0[base + r * width + c] = 1.0;
                        c0_total += 1.0;
                    }
                    if !(first && c < half_width) {
                        mask_1[base + r * width + c] = 1.0;
                        c1_total += 1.0;
                    }
                }
            }
        }
        count_0[g] = c0_total as f32;
        count_1[g] = c1_total as f32;
    }
    (mask_0, mask_1, count_0, count_1)
}

/// `sample_move_kernel`: one flat (row, col) move per active game, sampled from `dist`
/// restricted to legal squares. Inactive games get (0, 0) placeholders; the caller never
/// applies moves for inactive games.
pub fn sample_move_kernel(
    dist: &[f32],
    mask: &[f32],
    active: &[bool],
    n: usize,
    height: usize,
    width: usize,
    rng: &mut Rng,
) -> (Vec<i64>, Vec<i64>) {
    let hw = height * width;
    let mut r_out = vec![0i64; n];
    let mut c_out = vec![0i64; n];
    for g in 0..n {
        if !active[g] {
            continue;
        }
        let base = g * hw;
        let mut total: f64 = 0.0;
        for r in 0..height {
            for c in 0..width {
                let i = base + r * width + c;
                total += (dist[i] * mask[i]) as f64;
            }
        }

        if total < 1e-8 {
            let idx = rng.randint(hw as u64) as i64;
            r_out[g] = idx / width as i64;
            c_out[g] = idx % width as i64;
            continue;
        }

        let threshold = rng.random() * total;
        let mut cum: f64 = 0.0;
        let mut chosen_r: i64 = 0;
        let mut chosen_c: i64 = 0;
        let mut found = false;
        for r in 0..height {
            for c in 0..width {
                let i = base + r * width + c;
                let v = (dist[i] * mask[i]) as f64;
                if v > 0.0 {
                    cum += v;
                    if !found && cum >= threshold {
                        chosen_r = r as i64;
                        chosen_c = c as i64;
                        found = true;
                    }
                }
            }
        }
        r_out[g] = chosen_r;
        c_out[g] = chosen_c;
    }
    (r_out, c_out)
}

/// `apply_and_score_kernel`: for each active game, marks the move (or resolves a collision),
/// increments `move_counts`, rescores both players, and updates `finished`.
#[allow(clippy::too_many_arguments)]
pub fn apply_and_score_kernel(
    boards: &mut [i8],
    move_counts: &mut [i32],
    finished: &mut [bool],
    scores: &mut [f32],
    r0: &[i64],
    c0: &[i64],
    r1: &[i64],
    c1: &[i64],
    active: &[bool],
    n: usize,
    height: usize,
    width: usize,
) {
    let hw = height * width;
    for g in 0..n {
        if !active[g] {
            continue;
        }
        let base = g * hw;

        let rr0 = r0[g];
        let cc0 = c0[g];
        let rr1 = r1[g];
        let cc1 = c1[g];

        if rr0 == rr1 && cc0 == cc1 {
            boards[base + rr0 as usize * width + cc0 as usize] = REMOVED_SQUARE;
            for dr in -1i64..2 {
                for dc in -1i64..2 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = rr0 + dr;
                    let nc = cc0 + dc;
                    if nr >= 0
                        && (nr as usize) < height
                        && nc >= 0
                        && (nc as usize) < width
                        && (nr + nc) % 2 == 0
                    {
                        boards[base + nr as usize * width + nc as usize] = REMOVED_SQUARE;
                    }
                }
            }
        } else {
            boards[base + rr0 as usize * width + cc0 as usize] = PLAYER_0_MARK;
            boards[base + rr1 as usize * width + cc1 as usize] = PLAYER_1_MARK;
        }

        move_counts[g] += 1;

        scores[g * 2] = score_player(boards, g, PLAYER_0_MARK, height, width);
        scores[g * 2 + 1] = score_player(boards, g, PLAYER_1_MARK, height, width);

        let mut has_playable = false;
        for r in 0..height {
            for c in 0..width {
                if boards[base + r * width + c] == PLAYABLE_SQUARE {
                    has_playable = true;
                    break;
                }
            }
            if has_playable {
                break;
            }
        }
        if !has_playable {
            finished[g] = true;
        }
    }
}
