//! Benchmark preencoded zero planes through the native bridge, including transfers.
use alpha_lines_game::{game::SQUARES, inference::ZeroInputModel};
use std::{env, time::Instant};

fn main() -> Result<(), String> {
    let mut path = None;
    let mut batch = 0usize;
    let mut seconds = 20.0f64;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--model" => path = Some(value),
            "--batch" => batch = value.parse().map_err(|_| "invalid batch")?,
            "--seconds" => seconds = value.parse().map_err(|_| "invalid seconds")?,
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    if batch == 0 || !seconds.is_finite() || seconds <= 0.0 {
        return Err("batch and seconds must be positive and seconds finite".into());
    }
    let mut model = ZeroInputModel::load(&path.ok_or("--model is required")?, batch, true)?;
    let mut priors = vec![0.0; 2 * batch * SQUARES];
    let mut values = vec![0.0; 2 * batch];
    for _ in 0..10 {
        model.evaluate(&mut priors, &mut values);
    }
    assert!(priors.iter().chain(&values).all(|v| v.is_finite()));
    println!("BENCHMARK_START: {seconds} seconds, {} evals per call", 2 * batch);
    let started = Instant::now();
    let mut calls = 0u64;
    while started.elapsed().as_secs_f64() < seconds {
        // evaluate returns only after outputs have been copied back to CPU.
        model.evaluate(&mut priors, &mut values);
        calls += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let evals = calls * batch as u64 * 2;
    println!("RESULT {{\"batch_evals\":{},\"batch_positions\":{},\"elapsed_s\":{:.9},\"calls\":{},\"evals\":{},\"evals_per_s\":{:.3},\"ms_per_call\":{:.6}}}",
        2 * batch, batch, elapsed, calls, evals, evals as f64 / elapsed,
        elapsed * 1000.0 / calls as f64);
    Ok(())
}
