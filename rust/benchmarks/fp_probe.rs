//! Numerical probes using the production PUCT backup and f32 EXP3 additions.
use alpha_lines_game::mcts::variant::{Choice, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::Config;

fn main() {
    let cfg = Config::default();
    let choices = [Choice { action: 0, prob: 1.0 }; 2];
    println!("kind,n_or_accumulator,input,actual_delta,ideal_delta");
    for n in [10_000u32, 100_000, 1_000_000, 1 << 20, 1 << 22, 1 << 23, 1 << 24, 100_000_000, 1_000_000_000] {
        for value in [0.6f32, -0.5, 1.0] {
            let mut stats = PuctStats::default();
            stats.visit[0][0] = n as u64;
            stats.total[0] = n as u64;
            stats.q[0][0] = 0.5;
            Puct::backup(&mut stats, choices, [value, 0.0], [80, 80], &cfg);
            println!("puct,{n},{value},{:.12e},{:.12e}", stats.q[0][0] as f64 - 0.5, (value as f64 - 0.5) / (n as f64 + 1.0));
        }
    }
    for acc in [1024f32, 8192.0, 16384.0, 65536.0, 1_048_576.0, 16_777_216.0] {
        for prob in [0.9f32, 0.1, 0.00125] {
            let delta = cfg.exp3_gamma * (0.5 / prob) / 80.0;
            println!("exp3_log,{acc},{prob},{:.12e},{:.12e}", (acc + delta) as f64 - acc as f64, delta);
            println!("exp3_sum,{acc},{prob},{:.12e},{:.12e}", (acc + prob) as f64 - acc as f64, prob);
        }
    }
    // Accumulated error, with a known exact mean for the same input values.
    for constant in [true, false] {
        let mut stats = PuctStats::default();
        let mut exact_sum = 0.0f64;
        for n in 1u32..=20_000_000 {
            let value = if constant || n % 2 == 0 { 0.6f32 } else { 0.4f32 };
            exact_sum += value as f64;
            Puct::backup(&mut stats, choices, [value, 0.0], [80, 80], &cfg);
            if [10_000, 100_000, 1_000_000, 4_194_304, 16_777_216, 20_000_000].contains(&n) {
                println!("sequence_{constant},{n},{value},{:.12e},{:.12e}", stats.q[0][0], exact_sum / n as f64);
            }
        }
    }
    // Fixed-strategy accumulation to the first rounded-away addition.
    for prob in [0.9f32, 0.1, 0.0125, 0.00125] {
        let mut acc = 0.0f32;
        for n in 1u64..100_000_000 {
            let next = acc + prob;
            if next == acc {
                println!("sum_stops,{n},{prob},{acc},{}", n as f64 * prob as f64);
                break;
            }
            acc = next;
        }
    }
}
