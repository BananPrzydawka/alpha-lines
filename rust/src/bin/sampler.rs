//! Three questions about the sampler, measured on the incremental implementation only.
//!
//! 1. How does sampling compare with the other way to make a move (`action_step`, where the
//!    indices come from outside and only have to be validated)?
//! 2. Can the sampling algorithm be improved beyond SIMD?
//! 3. Does using i64 for indices that never exceed 160 actually cost anything?
//!
//! Every variant below must produce byte-identical moves to the faithful one; that is
//! asserted, so a "speedup" cannot come from quietly sampling something different.

use std::time::Instant;

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::{sample_move_kernel, PLAYABLE_SQUARE};
use alpha_lines_game::incremental::HW;
use alpha_lines_game::rng::Rng;
use alpha_lines_game::IncrementalGame;

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric"))
        .unwrap_or(default)
}

/// The 80 squares that can ever hold a mark: (r + c) even. The other 80 are structurally
/// dead and the faithful sampler still walks them, twice, every single call.
fn playable_cells() -> Vec<u16> {
    (0..HW).filter(|i| ((i / WIDTH) + (i % WIDTH)) % 2 == 0).map(|i| i as u16).collect()
}

// ------------------------------------------------------------------ sampler variants

/// v1: the faithful kernel plus an early exit from the second pass. Identical output —
/// everything after the chosen cell only accumulates into `cum`, which is never read again.
fn sample_early_break(
    dist: &[f32], mask: &[f32], active: &[bool], n: usize, rng: &mut Rng,
    r_out: &mut [i64], c_out: &mut [i64],
) {
    for g in 0..n {
        if !active[g] {
            r_out[g] = 0;
            c_out[g] = 0;
            continue;
        }
        let base = g * HW;
        let mut total = 0.0f64;
        for i in 0..HW {
            total += (dist[base + i] * mask[base + i]) as f64;
        }
        if total < 1e-8 {
            let idx = rng.randint(HW as u64) as i64;
            r_out[g] = idx / WIDTH as i64;
            c_out[g] = idx % WIDTH as i64;
            continue;
        }
        let threshold = rng.random() * total;
        let mut cum = 0.0f64;
        let mut chosen = 0usize;
        for i in 0..HW {
            let v = (dist[base + i] * mask[base + i]) as f64;
            if v > 0.0 {
                cum += v;
                if cum >= threshold {
                    chosen = i;
                    break;
                }
            }
        }
        r_out[g] = (chosen / WIDTH) as i64;
        c_out[g] = (chosen % WIDTH) as i64;
    }
}

/// v2: as v1, but walking only the 80 playable squares. The other 80 always have mask 0, so
/// they contribute nothing to either pass — visiting them is pure waste.
fn sample_playable_only(
    dist: &[f32], mask: &[f32], active: &[bool], n: usize, cells: &[u16], rng: &mut Rng,
    r_out: &mut [i64], c_out: &mut [i64],
) {
    for g in 0..n {
        if !active[g] {
            r_out[g] = 0;
            c_out[g] = 0;
            continue;
        }
        let base = g * HW;
        let mut total = 0.0f64;
        for &ci in cells {
            let i = base + ci as usize;
            total += (dist[i] * mask[i]) as f64;
        }
        if total < 1e-8 {
            let idx = rng.randint(HW as u64) as i64;
            r_out[g] = idx / WIDTH as i64;
            c_out[g] = idx % WIDTH as i64;
            continue;
        }
        let threshold = rng.random() * total;
        let mut cum = 0.0f64;
        let mut chosen = 0usize;
        for &ci in cells {
            let i = base + ci as usize;
            let v = (dist[i] * mask[i]) as f64;
            if v > 0.0 {
                cum += v;
                if cum >= threshold {
                    chosen = ci as usize;
                    break;
                }
            }
        }
        r_out[g] = (chosen / WIDTH) as i64;
        c_out[g] = (chosen % WIDTH) as i64;
    }
}

/// v3: no mask at all. The game maintains a compact list of squares that are still playable,
/// so the sampler walks only those. The first-move half-board rule becomes one comparison
/// instead of a materialized mask.
///
/// `live[g]` shrinks from 80 to 0 over a game, so this walks ~40 cells on average against
/// the faithful kernel's 160, twice.
#[allow(clippy::too_many_arguments)]
fn sample_live_list(
    dist: &[f32], live: &[Vec<u16>], active: &[bool], first: &[bool], n: usize,
    player: usize, half_width: usize, rng: &mut Rng,
    r_out: &mut [i64], c_out: &mut [i64],
) {
    for g in 0..n {
        if !active[g] {
            r_out[g] = 0;
            c_out[g] = 0;
            continue;
        }
        let base = g * HW;
        let allowed = |ci: u16| -> bool {
            if !first[g] {
                return true;
            }
            let c = ci as usize % WIDTH;
            if player == 0 { c < half_width } else { c >= half_width }
        };

        let mut total = 0.0f64;
        for &ci in &live[g] {
            if allowed(ci) {
                total += dist[base + ci as usize] as f64;
            }
        }
        if total < 1e-8 {
            let idx = rng.randint(HW as u64) as i64;
            r_out[g] = idx / WIDTH as i64;
            c_out[g] = idx % WIDTH as i64;
            continue;
        }
        let threshold = rng.random() * total;
        let mut cum = 0.0f64;
        let mut chosen = 0usize;
        for &ci in &live[g] {
            if !allowed(ci) {
                continue;
            }
            let v = dist[base + ci as usize] as f64;
            if v > 0.0 {
                cum += v;
                if cum >= threshold {
                    chosen = ci as usize;
                    break;
                }
            }
        }
        r_out[g] = (chosen / WIDTH) as i64;
        c_out[g] = (chosen % WIDTH) as i64;
    }
}

/// v4: v3 writing u16 cell indices instead of two i64 row/col arrays, to see whether the
/// index width shows up at all on a scalar CPU.
#[allow(clippy::too_many_arguments)]
fn sample_live_list_u16(
    dist: &[f32], live: &[Vec<u16>], active: &[bool], first: &[bool], n: usize,
    player: usize, half_width: usize, rng: &mut Rng, out: &mut [u16],
) {
    for g in 0..n {
        if !active[g] {
            out[g] = 0;
            continue;
        }
        let base = g * HW;
        let allowed = |ci: u16| -> bool {
            if !first[g] {
                return true;
            }
            let c = ci as usize % WIDTH;
            if player == 0 { c < half_width } else { c >= half_width }
        };
        let mut total = 0.0f64;
        for &ci in &live[g] {
            if allowed(ci) {
                total += dist[base + ci as usize] as f64;
            }
        }
        if total < 1e-8 {
            out[g] = rng.randint(HW as u64) as u16;
            continue;
        }
        let threshold = rng.random() * total;
        let mut cum = 0.0f64;
        let mut chosen = 0u16;
        for &ci in &live[g] {
            if !allowed(ci) {
                continue;
            }
            let v = dist[base + ci as usize] as f64;
            if v > 0.0 {
                cum += v;
                if cum >= threshold {
                    chosen = ci;
                    break;
                }
            }
        }
        out[g] = chosen;
    }
}

// --------------------------------------------------------------------------- harness

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Faithful,
    EarlyBreak,
    PlayableOnly,
    LiveList,
    LiveListU16,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::Faithful => "v0 faithful (mask, 2x160, no break)",
            Variant::EarlyBreak => "v1  + early break",
            Variant::PlayableOnly => "v2  + only the 80 playable squares",
            Variant::LiveList => "v3  live list, no mask read",
            Variant::LiveListU16 => "v4  live list, u16 indices",
        }
    }
    /// Does this variant still need `legal_masks_kernel` run every step?
    fn needs_masks(self) -> bool {
        matches!(self, Variant::Faithful | Variant::EarlyBreak | Variant::PlayableOnly)
    }
}

struct Timings {
    sample: f64,
    masks: f64,
    upkeep: f64,
}

/// One full rollout of `n` games, timing the sampler (and, separately, whatever the sampler
/// needs kept up to date). All variants share the RNG stream and must agree exactly.
fn rollout(v: Variant, n: usize, seed: u64, d0: &[f32], d1: &[f32], cells: &[u16]) -> (Timings, Vec<i8>) {
    let mut g = IncrementalGame::new(n, seed);
    let mut rng = Rng::new(seed ^ 0xabc_def);
    let mut live: Vec<Vec<u16>> = (0..n).map(|_| cells.to_vec()).collect();

    let mut r0 = vec![0i64; n];
    let mut c0 = vec![0i64; n];
    let mut r1 = vec![0i64; n];
    let mut c1 = vec![0i64; n];
    let mut u0 = vec![0u16; n];
    let mut u1 = vec![0u16; n];

    let (mut t_sample, mut t_masks, mut t_upkeep) = (0.0, 0.0, 0.0);

    loop {
        let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
        if !active.iter().any(|&a| a) {
            break;
        }
        let first: Vec<bool> = g.move_counts.iter().map(|&m| m == 0).collect();

        let (m0, m1) = if v.needs_masks() {
            let t = Instant::now();
            let m = g.to_reference().raw_masks();
            t_masks += t.elapsed().as_secs_f64();
            m
        } else {
            (Vec::new(), Vec::new())
        };

        let t = Instant::now();
        match v {
            Variant::Faithful => {
                let (a, b) = sample_move_kernel(d0, &m0, &active, n, HEIGHT, WIDTH, &mut rng);
                let (c, d) = sample_move_kernel(d1, &m1, &active, n, HEIGHT, WIDTH, &mut rng);
                r0.copy_from_slice(&a);
                c0.copy_from_slice(&b);
                r1.copy_from_slice(&c);
                c1.copy_from_slice(&d);
            }
            Variant::EarlyBreak => {
                sample_early_break(d0, &m0, &active, n, &mut rng, &mut r0, &mut c0);
                sample_early_break(d1, &m1, &active, n, &mut rng, &mut r1, &mut c1);
            }
            Variant::PlayableOnly => {
                sample_playable_only(d0, &m0, &active, n, cells, &mut rng, &mut r0, &mut c0);
                sample_playable_only(d1, &m1, &active, n, cells, &mut rng, &mut r1, &mut c1);
            }
            Variant::LiveList => {
                sample_live_list(d0, &live, &active, &first, n, 0, g.half_width, &mut rng, &mut r0, &mut c0);
                sample_live_list(d1, &live, &active, &first, n, 1, g.half_width, &mut rng, &mut r1, &mut c1);
            }
            Variant::LiveListU16 => {
                sample_live_list_u16(d0, &live, &active, &first, n, 0, g.half_width, &mut rng, &mut u0);
                sample_live_list_u16(d1, &live, &active, &first, n, 1, g.half_width, &mut rng, &mut u1);
                for i in 0..n {
                    r0[i] = (u0[i] / WIDTH as u16) as i64;
                    c0[i] = (u0[i] % WIDTH as u16) as i64;
                    r1[i] = (u1[i] / WIDTH as u16) as i64;
                    c1[i] = (u1[i] % WIDTH as u16) as i64;
                }
            }
        }
        t_sample += t.elapsed().as_secs_f64();

        g.apply_step(&r0, &c0, &r1, &c1, &active);

        // whatever the sampler needs kept current, charged separately
        if !v.needs_masks() {
            let t = Instant::now();
            for gi in 0..n {
                if active[gi] {
                    let cells_g = &g.boards[gi * HW..(gi + 1) * HW];
                    live[gi].retain(|&ci| cells_g[ci as usize] == PLAYABLE_SQUARE);
                }
            }
            t_upkeep += t.elapsed().as_secs_f64();
        }
    }

    (
        Timings { sample: t_sample, masks: t_masks, upkeep: t_upkeep },
        g.boards,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = arg_usize(&args, "--seed", 12345) as u64;
    let target = arg_usize(&args, "--target-games", 2048);
    let cells = playable_cells();

    // ---------------- part 1: sampling vs. supplying the moves from outside -------------
    println!("=== part 1: distribution_step (samples) vs action_step (validates) ===");
    println!("incremental scorer only, microseconds per completed game\n");
    println!(
        "{:>7} {:>9} {:>9} {:>9} {:>9}  {:>9} {:>9} {:>9}  {:>9}",
        "games", "d:masks", "d:sample", "d:apply", "d:TOTAL", "a:masks", "a:valid", "a:apply", "a:TOTAL"
    );
    println!("{}", "-".repeat(96));
    for n in [1usize, 16, 64, 256, 1024, 2048, 8192] {
        let reps = (target / n).max(1);
        let mut rng = Rng::new(seed);
        let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

        // distribution_step, decomposed
        let (mut dm, mut ds, mut da) = (0.0, 0.0, 0.0);
        let mut recorded: Vec<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<bool>)> = Vec::new();
        for k in 0..reps {
            let mut g = IncrementalGame::new(n, seed + k as u64);
            let mut r = Rng::new(seed ^ 0x99);
            loop {
                let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
                if !active.iter().any(|&a| a) {
                    break;
                }
                let t = Instant::now();
                let (m0, m1) = g.to_reference().raw_masks();
                dm += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let (r0, c0) = sample_move_kernel(&d0, &m0, &active, n, HEIGHT, WIDTH, &mut r);
                let (r1, c1) = sample_move_kernel(&d1, &m1, &active, n, HEIGHT, WIDTH, &mut r);
                ds += t.elapsed().as_secs_f64();
                let t = Instant::now();
                g.apply_step(&r0, &c0, &r1, &c1, &active);
                da += t.elapsed().as_secs_f64();
                if k == 0 {
                    recorded.push((r0, c0, r1, c1, active));
                }
            }
        }

        // action_step, decomposed, replaying the moves the sampler chose
        let (mut am, mut av, mut aa) = (0.0, 0.0, 0.0);
        for k in 0..reps {
            let mut g = IncrementalGame::new(n, seed + k as u64);
            for (r0, c0, r1, c1, active) in &recorded {
                let t = Instant::now();
                let (m0, m1) = g.to_reference().raw_masks();
                am += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mut bad = 0usize;
                for gi in 0..n {
                    if active[gi] {
                        let a0 = (r0[gi] * WIDTH as i64 + c0[gi]) as usize;
                        let a1 = (r1[gi] * WIDTH as i64 + c1[gi]) as usize;
                        if m0[gi * HW + a0] != 1.0 || m1[gi * HW + a1] != 1.0 {
                            bad += 1;
                        }
                    }
                }
                assert_eq!(bad, 0, "replayed move was illegal");
                av += t.elapsed().as_secs_f64();
                let t = Instant::now();
                g.apply_step(r0, c0, r1, c1, active);
                aa += t.elapsed().as_secs_f64();
            }
        }

        let games = (n * reps) as f64;
        let us = |t: f64| t / games * 1e6;
        println!(
            "{:>7} {:>8.1}us {:>8.1}us {:>8.1}us {:>8.1}us  {:>8.1}us {:>8.1}us {:>8.1}us  {:>8.1}us",
            n, us(dm), us(ds), us(da), us(dm + ds + da), us(am), us(av), us(aa), us(am + av + aa)
        );
    }

    // ---------------- part 2 + 3: sampler variants ------------------------------------
    println!("\n=== part 2: sampler variants, microseconds per completed game ===");
    println!("all variants must produce identical games; asserted below\n");
    for n in [256usize, 2048] {
        let reps = (target / n).max(1);
        let mut rng = Rng::new(seed);
        let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

        println!("n = {n}");
        println!(
            "  {:<38} {:>9} {:>9} {:>9} {:>9}  {:>8}",
            "variant", "sample", "masks", "upkeep", "sum", "vs v0"
        );
        let mut baseline = 0.0;
        let mut reference_boards: Option<Vec<i8>> = None;
        for v in [Variant::Faithful, Variant::EarlyBreak, Variant::PlayableOnly,
                  Variant::LiveList, Variant::LiveListU16] {
            let mut best = f64::INFINITY;
            let mut best_parts = (0.0, 0.0, 0.0);
            let mut boards = Vec::new();
            for k in 0..reps.min(5).max(1) {
                let (t, b) = rollout(v, n, seed + k as u64, &d0, &d1, &cells);
                let parts = (t.sample, t.masks, t.upkeep);
                let sum = t.sample + t.masks + t.upkeep;
                if sum < best {
                    best = sum;
                    best_parts = parts;
                }
                if k == 0 {
                    boards = b;
                }
            }
            match &reference_boards {
                None => reference_boards = Some(boards),
                Some(r) => assert_eq!(*r, boards, "{} produced different games", v.name()),
            }
            let us = |t: f64| t / n as f64 * 1e6;
            if v == Variant::Faithful {
                baseline = best;
            }
            println!(
                "  {:<38} {:>8.1}us {:>8.1}us {:>8.1}us {:>8.1}us  {:>7.2}x",
                v.name(), us(best_parts.0), us(best_parts.1), us(best_parts.2), us(best),
                baseline / best
            );
        }
        println!();
    }
}
