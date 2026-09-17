//! End-to-end MCTS speed run; records are counted and immediately discarded.
use alpha_lines_game::{config as defaults, game::SQUARES, inference::CompiledModel,
    mcts::{Config, Evaluate, Exp3, Puct, Search, Variant}};
use std::{env, time::{Duration, Instant}};

struct TimedModel {
    model: CompiledModel,
    elapsed: Duration,
}
impl Evaluate for TimedModel {
    fn evaluate(&mut self, cells: &[[u8; SQUARES]], scores: &[[i32; 2]],
                priors: &mut [f32], values: &mut [f32]) {
        let start = Instant::now();
        self.model.evaluate(cells, scores, priors, values);
        self.elapsed += start.elapsed();
    }
}

#[derive(Default)]
struct Report {
    steps: u64,
    cycles: u64,
    moves: u64,
    finished: u64,
    elapsed: f64,
    inference: f64,
}

fn advance<V: Variant>(search: &mut Search<V>, model: &mut TimedModel, report: &mut Report) {
    let before = (search.arena.len() == search.cfg.node_capacity).then(|| {
        let sims: u64 = search.slots.iter().map(|s| s.sim_count as u64).sum();
        (sims, search.slots.iter().any(|s| s.root.pending))
    });
    let records = search.cycle(model);
    report.cycles += 1;
    if !records.is_empty() {
        report.steps += 1;
        report.moves += records.len() as u64;
        report.finished += records.iter().filter(|r| r.finished).count() as u64;
    } else if let Some((before, pending)) = before {
        let after: u64 = search.slots.iter().map(|s| s.sim_count as u64).sum();
        assert!(pending || after > before,
            "search made no progress: increase --node-capacity (arena exhausted)");
    }
}

#[derive(Clone)]
struct Run {
    cfg: Config,
    path: String,
    cuda: bool,
    workers: usize,
    steps: u64,
    seconds: Option<f64>,
    seed: u64,
    warmup: usize,
    warmup_steps: u64,
}

fn panic_message(value: Box<dyn std::any::Any + Send>) -> String {
    value.downcast_ref::<String>().cloned()
        .or_else(|| value.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "worker panicked".into())
}

fn run<V: Variant>(options: Run) -> Result<(), String> {
    use std::sync::{mpsc, Arc, Barrier, atomic::{AtomicBool, Ordering}};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let (ready_tx, ready_rx) = mpsc::channel();
    let finished = Arc::new(Barrier::new(options.workers));
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut starts = Vec::new();
    let mut handles = Vec::new();
    for worker in 0..options.workers {
        let options = options.clone();
        let ready = ready_tx.clone();
        let finished = finished.clone();
        let cancelled = cancelled.clone();
        let (start_tx, start_rx) = mpsc::channel::<Option<Instant>>();
        starts.push(start_tx);
        handles.push(std::thread::spawn(move || {
            // Construct, use, and destroy each native model on its owning thread.
            // No unsafe Send implementation or shared mutable search state is needed.
            let setup = catch_unwind(AssertUnwindSafe(|| -> Result<_, String> {
                let cfg = options.cfg;
                let mut model = TimedModel {
                    model: CompiledModel::load(&options.path, cfg.b, options.cuda)?,
                    elapsed: Duration::ZERO,
                };
                let cells = vec![[0; SQUARES]; cfg.b];
                let scores = vec![[0; 2]; cfg.b];
                let mut priors = vec![0.0; 2 * cfg.b * SQUARES];
                let mut values = vec![0.0; 2 * cfg.b];
                for _ in 0..options.warmup {
                    model.evaluate(&cells, &scores, &mut priors, &mut values);
                }
                let mut search = Search::<V>::new(cfg, options.seed.wrapping_add(worker as u64));
                let mut report = Report::default();
                while report.steps < options.warmup_steps {
                    advance(&mut search, &mut model, &mut report);
                }
                model.elapsed = Duration::ZERO;
                Ok((model, search))
            })).map_err(panic_message).and_then(|x| x);
            ready.send(setup.as_ref().map(|_| ()).map_err(Clone::clone)).unwrap();
            drop(ready);
            let (mut model, mut search) = setup?;
            let Some(start) = start_rx.recv().map_err(|e| e.to_string())? else {
                return Err("another worker failed initialization".into());
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut report = Report::default();
                while !cancelled.load(Ordering::Relaxed) {
                    if let Some(seconds) = options.seconds {
                        if start.elapsed().as_secs_f64() >= seconds { break; }
                    } else if report.steps >= options.steps { break; }
                    advance(&mut search, &mut model, &mut report);
                }
                report.elapsed = start.elapsed().as_secs_f64();
                report.inference = model.elapsed.as_secs_f64();
                report
            })).map_err(panic_message);
            if result.is_err() { cancelled.store(true, Ordering::Relaxed); }
            // Keep native runtimes alive until every timed worker finishes. Destruction
            // must not synchronize the GPU while other workers are still being measured.
            finished.wait();
            result
        }));
    }
    drop(ready_tx);
    let mut setup_error = None;
    for _ in 0..options.workers {
        if let Err(error) = ready_rx.recv().map_err(|e| e.to_string())? {
            setup_error = Some(error);
        }
    }
    if setup_error.is_none() { println!("BENCHMARK_START"); }
    let start = setup_error.is_none().then(Instant::now);
    for sender in starts { let _ = sender.send(start); }
    let mut total = Report::default();
    for (worker, handle) in handles.into_iter().enumerate() {
        match handle.join().map_err(panic_message).and_then(|x| x) {
            Ok(r) => {
                println!("worker={worker} steps={} cycles={} root_moves={} elapsed_s={:.3} inference_s={:.3}",
                    r.steps, r.cycles, r.moves, r.elapsed, r.inference);
                total.steps += r.steps;
                total.cycles += r.cycles;
                total.moves += r.moves;
                total.finished += r.finished;
                total.elapsed = total.elapsed.max(r.elapsed);
            }
            Err(e) => { setup_error.get_or_insert(e); }
        }
    }
    if let Some(error) = setup_error { return Err(error); }
    let model_rows = total.cycles * options.cfg.b as u64 * 2;
    println!("RESULT {{\"workers\":{},\"elapsed_s\":{:.6},\"steps\":{},\"cycles\":{},\"root_moves\":{},\"finished_games\":{},\"model_rows\":{},\"model_rows_per_s\":{:.3}}}",
        options.workers, total.elapsed, total.steps, total.cycles, total.moves, total.finished,
        model_rows, model_rows as f64 / total.elapsed);
    Ok(())
}

fn main() -> Result<(), String> {
    let mut cfg = Config::default();
    let (mut steps, mut seed, mut warmup) = (defaults::BENCHMARK_STEPS, defaults::BENCHMARK_SEED, defaults::BENCHMARK_WARMUP);
    let (mut workers, mut warmup_steps, mut seconds) = (1usize, defaults::BENCHMARK_WARMUP_STEPS, Some(defaults::BENCHMARK_SECONDS));
    let (mut path, mut variant, mut device) = (None, String::from(defaults::BENCHMARK_VARIANT), String::from(defaults::BENCHMARK_DEVICE));
    let mut explicit_capacity = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!("speed --model FILE.pt2 [--variant puct|exp3] [--steps N | --seconds N]\n  [--sims N] [--g N] [--b N] [--t N] [--node-capacity N]\n  [--c-puct X] [--alpha X] [--epsilon X] [--exp3-gamma X]\n  [--seed N] [--warmup N] [--device cuda|cpu]\n  [--workers N] [--warmup-steps N]\nDefaults come from config.json; --steps selects a fixed-step run.\nSteps count threshold-triggered batches of root moves. Sims are completed simulations per root move.");
            return Ok(());
        }
        let value = args.next().ok_or_else(|| format!("missing value for {arg}"))?;
        macro_rules! parse { () => { value.parse().map_err(|_| format!("invalid value for {arg}: {value}"))? }; }
        match arg.as_str() {
            "--model" => path = Some(value),
            "--variant" => variant = value,
            "--device" => device = value,
            "--steps" => { steps = parse!(); seconds = None; }
            "--workers" => workers = parse!(),
            "--seconds" => seconds = Some(parse!()),
            "--warmup-steps" => warmup_steps = parse!(),
            "--sims" => cfg.s = parse!(),
            "--g" => cfg.g = parse!(),
            "--b" => cfg.b = parse!(),
            "--t" => cfg.t = parse!(),
            "--node-capacity" => { cfg.node_capacity = parse!(); explicit_capacity = true; }
            "--c-puct" => cfg.c_puct = parse!(),
            "--alpha" => cfg.alpha = parse!(),
            "--epsilon" => cfg.epsilon = parse!(),
            "--exp3-gamma" => cfg.exp3_gamma = parse!(),
            "--seed" => seed = parse!(),
            "--warmup" => warmup = parse!(),
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    let path = path.ok_or("--model is required")?;
    if !matches!(variant.as_str(), "puct" | "exp3") || !matches!(device.as_str(), "cuda" | "cpu") {
        return Err("invalid --variant or --device".into());
    }
    if workers == 0 || seconds.is_some_and(|s| !s.is_finite() || s <= 0.0) {
        return Err("workers and seconds must be positive".into());
    }
    if cfg.g == 0 || cfg.g > 65536 || cfg.b == 0 || cfg.t == 0 || cfg.t > cfg.g || cfg.s == 0 || steps == 0 {
        return Err("require 1 <= g <= 65536, b > 0, 1 <= t <= g, sims > 0, steps > 0".into());
    }
    if !cfg.c_puct.is_finite() || cfg.c_puct < 0.0 || !cfg.alpha.is_finite() || cfg.alpha <= 0.0
        || !(0.0..=1.0).contains(&cfg.epsilon) || !(0.0..=1.0).contains(&cfg.exp3_gamma) || cfg.exp3_gamma == 0.0 {
        return Err("require finite c-puct >= 0, alpha > 0, epsilon in [0,1], exp3-gamma in (0,1]".into());
    }
    if !explicit_capacity { cfg.node_capacity = Config::recommended_node_capacity(cfg.g, cfg.s); }
    if cfg.node_capacity == 0 || cfg.node_capacity >= u32::MAX as usize {
        return Err("node capacity must be in 1..u32::MAX".into());
    }
    println!("variant={variant} workers={workers} steps={steps} seconds={seconds:?} device={device} {cfg:?}");
    let options = Run { cfg, path, cuda: device == "cuda", workers, steps, seconds, seed, warmup, warmup_steps };
    match variant.as_str() {
        "puct" => run::<Puct>(options),
        "exp3" => run::<Exp3>(options),
        _ => unreachable!(),
    }
}
