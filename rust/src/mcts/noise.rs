//! Dirichlet noise for roots.
//!
//! A root's priors are mixed with a fresh Dirichlet sample so that a game explores moves the
//! network is confident are bad. It applies only to roots, which is why a root is a private
//! copy of a position rather than the shared node for it: noise written into a shared node
//! would push one game's exploration into every other game that reaches the state.

use crate::game::Rng;

/// One standard normal, by Box-Muller.
fn normal(rng: &mut Rng) -> f64 {
    // `random` is in [0, 1); nudge off zero so the log is finite
    let u1 = rng.random().max(f64::MIN_POSITIVE);
    let u2 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// One Gamma(`shape`, 1) sample, by Marsaglia-Tsang.
///
/// Shapes below 1 — which is the interesting range here, `alpha` is 0.3 — are drawn at
/// `shape + 1` and scaled down, since the method needs `shape >= 1`.
fn gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let g = gamma(rng, shape + 1.0);
        return g * rng.random().max(f64::MIN_POSITIVE).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.random().max(f64::MIN_POSITIVE);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Write a Dirichlet(`alpha`) sample over `n` components into `out[..n]`.
pub fn dirichlet(rng: &mut Rng, alpha: f32, out: &mut [f32]) {
    debug_assert!(alpha > 0.0, "dirichlet needs a positive concentration");
    let mut sum = 0.0f64;
    for slot in out.iter_mut() {
        let g = gamma(rng, f64::from(alpha));
        *slot = g as f32;
        sum += g;
    }
    // all-zero is possible only if every gamma underflowed; fall back to uniform
    if sum <= 0.0 {
        let u = 1.0 / out.len() as f32;
        out.iter_mut().for_each(|v| *v = u);
        return;
    }
    let norm = sum as f32;
    out.iter_mut().for_each(|v| *v /= norm);
}
