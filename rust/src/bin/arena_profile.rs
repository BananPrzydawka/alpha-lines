//! Arena calibration with the grounded statistical model from `chain`.
//! cargo run --release --bin arena_profile -- --games 1024 --buffer 512 --t 512 --sims 100 --steps 10
//! Add `--exp3` to use EXP3 (gamma=0.1) instead of PUCT.
//! `--lifetimes PATH` writes live ages and removal lifetimes in one-sweep buckets.
//! `--numerics PATH` records maximum accumulated statistics before each sweep.
//! CSV timings are cumulative; steps count MCTS stepping batches.

use alpha_lines_game::game::{Rng, SQUARES, PLAYER_0_MARK, PLAYER_1_MARK};
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::variant::{Exp3, Exp3Stats, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::Config;

struct Stub {
    rng: Rng,
}

/// One standard normal, by Box-Muller.
fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.random().max(f64::MIN_POSITIVE);
    let u2 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Prior decay base: square s has weight R^s, so the field spans 80 squares without
/// underflowing and peaks visibly at the target corner.
const GRADIENT_R: f32 = 0.9;
/// Per-eval noise: lognormal jitter on priors (sigma in log space), Gaussian on
/// values. Keeps the gradient direction but stops the stub being a perfect oracle —
/// same position reads slightly differently every eval, like a real model.
const PRIOR_NOISE: f64 = 0.75;
const VALUE_NOISE: f32 = 0.3;

/// Gradient weight table plus prefix sums: GRAD[s] = R^s, TOP_K[k] = sum of the k
/// largest weights (R^0 + ... + R^(k-1)), the denominator of the coverage ratio.
struct Gradient {
    w: [f32; SQUARES],
    top_k: [f32; SQUARES + 1],
}

impl Gradient {
    fn build() -> Self {
        let mut w = [0.0f32; SQUARES];
        let mut acc = 1.0f32;
        for s in 0..SQUARES {
            w[s] = acc;
            acc *= GRADIENT_R;
        }
        let mut top_k = [0.0f32; SQUARES + 1];
        for k in 1..=SQUARES {
            top_k[k] = top_k[k - 1] + w[k - 1];
        }
        Gradient { w, top_k }
    }
}

/// Coverage ratio for one player: gradient mass on their marks over the best
/// achievable mass with that many marks. Empty board reads 0.5 (a draw).
fn coverage(cells: &[u8; SQUARES], mark: u8, g: &Gradient, flip: bool) -> f32 {
    let mut num = 0.0f32;
    let mut k = 0usize;
    for sq in 0..SQUARES {
        if cells[sq] == mark {
            let d = if flip { SQUARES - 1 - sq } else { sq };
            num += g.w[d];
            k += 1;
        }
    }
    if k == 0 {
        return 0.5;
    }
    num / g.top_k[k].max(1e-8)
}

impl Evaluate for Stub {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        let g = Gradient::build();
        // Player-major rows: P0 targets square 0, P1 square 79.
        // Fresh noise per evaluation; games sharing a row share its noise.
        let npos = positions.len().min(priors.len() / SQUARES / 2);
        let (p0_half, p1_half) = priors.split_at_mut(npos * SQUARES);
        for (r, cells) in positions.iter().enumerate().take(npos) {
            let p0_row = &mut p0_half[r * SQUARES..][..SQUARES];
            let p1_row = &mut p1_half[r * SQUARES..][..SQUARES];
            for sq in 0..SQUARES {
                let j0 = (-0.5 * PRIOR_NOISE * PRIOR_NOISE
                    + PRIOR_NOISE * normal(&mut self.rng))
                .exp() as f32;
                let j1 = (-0.5 * PRIOR_NOISE * PRIOR_NOISE
                    + PRIOR_NOISE * normal(&mut self.rng))
                .exp() as f32;
                p0_row[sq] = g.w[sq] * j0;
                p1_row[sq] = g.w[SQUARES - 1 - sq] * j1;
            }
            let n0 = VALUE_NOISE * normal(&mut self.rng) as f32;
            let n1 = VALUE_NOISE * normal(&mut self.rng) as f32;
            values[0 * npos + r] = (((coverage(cells, PLAYER_0_MARK, &g, false) - 0.5)
                * 2.5)
                .tanh()
                + n0)
                .clamp(-1.0, 1.0);
            values[1 * npos + r] = (((coverage(cells, PLAYER_1_MARK, &g, true) - 0.5)
                * 2.5)
                .tanh()
                + n1)
                .clamp(-1.0, 1.0);
        }
    }
}

struct TimedModel<'a> {
    model: &'a mut Stub,
    elapsed: std::time::Duration,
}

impl Evaluate for TimedModel<'_> {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        let start = std::time::Instant::now();
        self.model.evaluate(positions, priors, values);
        self.elapsed += start.elapsed();
    }
}

#[derive(Default)]
struct NumericPeaks {
    total: u64,
    edge: u64,
    abs_q: f64,
    log_w: f64,
    strategy: f64,
}

trait InspectStats {
    fn inspect(&self, peaks: &mut NumericPeaks);
}

impl InspectStats for PuctStats {
    fn inspect(&self, peaks: &mut NumericPeaks) {
        for p in 0..2 {
            peaks.total = peaks.total.max(self.total[p] as u64);
            for sq in 0..SQUARES {
                peaks.edge = peaks.edge.max(self.visit[p][sq] as u64);
                peaks.abs_q = peaks.abs_q.max(self.q[p][sq].abs() as f64);
            }
        }
    }
}

impl InspectStats for Exp3Stats {
    fn inspect(&self, peaks: &mut NumericPeaks) {
        for p in 0..2 {
            for sq in 0..SQUARES {
                if self.log_w[p][sq].is_finite() {
                    peaks.log_w = peaks.log_w.max(self.log_w[p][sq] as f64);
                }
                peaks.strategy = peaks.strategy.max(self.strategy_sum[p][sq] as f64);
            }
        }
    }
}

/// Run `steps` MCTS stepping batches, honoring T each cycle.
fn benchmark<V: Variant>(cfg: Config, steps: usize, model: &mut Stub, lifetimes: Option<&str>, numerics: Option<&str>) where V::Stats: InspectStats {
    use std::time::{Duration, Instant};
    use std::io::{BufWriter, Write};
    let mut numeric_file = numerics.map(|path| BufWriter::new(std::fs::File::create(path).expect("create numeric CSV")));
    if let Some(file) = &mut numeric_file {
        writeln!(file, "step,population,max_total,max_edge,max_abs_q,max_log_weight,max_strategy_sum").unwrap();
    }
    let mut lifetime_file = lifetimes.map(|path| BufWriter::new(std::fs::File::create(path).expect("create lifetime CSV")));
    if let Some(file) = &mut lifetime_file {
        writeln!(file, "step,status,sweeps_survived,node_count").unwrap();
    }
    assert!(cfg.g > 0 && cfg.g <= u16::MAX as usize && cfg.b > 0);
    assert!(cfg.t > 0 && cfg.t <= cfg.g && cfg.s > 0 && steps > 0);
    let start = Instant::now();
    let mut search = Search::<V>::new(cfg, 0xA1FA);
    // Defer the cycle's step to measure occupancy before and after the sweep.
    search.cfg.t = usize::MAX;
    let mut model = TimedModel { model, elapsed: Duration::ZERO };
    let mut moves = vec![0usize; cfg.g];
    let mut cycles = 0;
    let mut batches = 0;
    let mut peak = 0;
    let mut search_time = Duration::ZERO;
    let mut step_time = Duration::ZERO;
    println!("# G={} B={} T={} S={} capacity={} node_bytes={} seed=0xA1FA model_seed=0xE1A4", cfg.g, cfg.b, cfg.t, cfg.s, cfg.node_capacity, std::mem::size_of::<alpha_lines_game::mcts::arena::Node<V::Stats>>());
    println!("batch,cycles,games_stepped,min_moves,max_moves,mean_moves,before_sweep,after_sweep,dropped,peak_live,slots_allocated,elapsed_s,model_s,search_s,step_s");
    while batches < steps {
        let tick = Instant::now();
        let model_before = model.elapsed;
        search.cycle(&mut model);
        search_time += tick.elapsed().saturating_sub(model.elapsed - model_before);
        cycles += 1;
        peak = peak.max(search.arena.len());
        assert!(search.arena.slot_count() < cfg.node_capacity, "arena capacity reached; increase --capacity");
        assert!(cycles <= 100_000, "cycle limit reached");
        if search.ready() < cfg.t { continue; }
        if let Some(file) = &mut numeric_file {
            let mut arena_peaks = NumericPeaks::default();
            for slot in search.arena.live_slots() {
                search.arena.node(slot).stats.inspect(&mut arena_peaks);
            }
            let mut root_peaks = NumericPeaks::default();
            for slot in &search.slots { slot.root.stats.inspect(&mut root_peaks); }
            for (name, p) in [("arena", arena_peaks), ("roots", root_peaks)] {
                writeln!(file, "{},{name},{},{},{},{},{}", batches + 1, p.total, p.edge, p.abs_q, p.log_w, p.strategy).unwrap();
            }
        }
        let before = search.arena.len();
        let mut ages_before = vec![0usize; batches + 1];
        if lifetime_file.is_some() {
            for slot in search.arena.live_slots() {
                ages_before[search.arena.node(slot).sweep_age as usize] += 1;
            }
        }
        let tick = Instant::now();
        let records = search.step();
        step_time += tick.elapsed();
        for record in &records { moves[record.id as usize] += 1; }
        batches += 1;
        if let Some(file) = &mut lifetime_file {
            let mut survivors = vec![0usize; batches + 1];
            for slot in search.arena.live_slots() {
                survivors[search.arena.node(slot).sweep_age as usize] += 1;
            }
            let mut removed = 0;
            for age in 0..batches {
                let dead = ages_before[age].checked_sub(survivors[age + 1]).expect("invalid lifetime accounting");
                removed += dead;
                writeln!(file, "{batches},removed,{age},{dead}").unwrap();
            }
            assert_eq!(removed, before - search.arena.len());
            for (age, count) in survivors.iter().enumerate() {
                writeln!(file, "{batches},live,{age},{count}").unwrap();
            }
        }
        println!("{batches},{cycles},{},{},{},{:.3},{before},{},{},{peak},{},{:.6},{:.6},{:.6},{:.6}",
            records.len(), moves.iter().min().unwrap(), moves.iter().max().unwrap(),
            moves.iter().sum::<usize>() as f64 / cfg.g as f64,
            search.arena.len(), before - search.arena.len(), search.arena.slot_count(),
            start.elapsed().as_secs_f64(), model.elapsed.as_secs_f64(),
            search_time.as_secs_f64(), step_time.as_secs_f64());
    }
    if let Some(file) = &mut numeric_file { file.flush().expect("flush numeric CSV"); }
    if let Some(file) = &mut lifetime_file { file.flush().expect("flush lifetime CSV"); }
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let number = |key: &str, default: usize| arg(&a, key).map(|s| s.parse().expect("invalid integer argument")).unwrap_or(default);
    let g = number("--games", 1024);
    let b = number("--buffer", 512);
    let sims = number("--sims", 100);
    let cfg = Config {
        g, b, t: number("--t", b), s: sims.try_into().expect("sims exceeds u32"),
        node_capacity: number("--capacity", Config::recommended_node_capacity(g, sims.try_into().expect("sims exceeds u32"))),
        ..Config::default()
    };
    let mut model = Stub { rng: Rng::new(0xE1A4) };
    let steps = number("--steps", 10);
    let lifetimes = arg(&a, "--lifetimes");
    let numerics = arg(&a, "--numerics");
    if a.iter().any(|a| a == "--exp3") {
        println!("# variant=EXP3 gamma={}", cfg.exp3_gamma);
        benchmark::<Exp3>(cfg, steps, &mut model, lifetimes.as_deref(), numerics.as_deref());
    } else {
        println!("# variant=PUCT c={} epsilon={} alpha={}", cfg.c_puct, cfg.epsilon, cfg.alpha);
        benchmark::<Puct>(cfg, steps, &mut model, lifetimes.as_deref(), numerics.as_deref());
    }
}
