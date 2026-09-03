//! Both search rules against naive transcriptions of `main/mcts_puct.py` and
//! `main/mcts_exp3.py`, plus the properties each rule is supposed to have.
//!
//! The references here compute in `f64` over dense arrays, the way numpy does, and share no
//! code with the engine's versions — which is what makes agreeing with them evidence.

use alpha_lines_game::game::{Rng, Scratch, LEGAL_WORDS, SQUARES};
use alpha_lines_game::mcts::variant::{squares, Choice, Exp3, Exp3Stats, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::Config;
use alpha_lines_game::Game;

fn cfg() -> Config {
    Config::default()
}

/// A legal mask from a real position, so the tests run on sets the game actually produces.
fn position(plies: u32, seed: u64) -> Game {
    let mut g = Game::new();
    let mut rng = Rng::new(seed);
    let mut scratch = Scratch::new();
    for _ in 0..plies {
        if g.finished {
            break;
        }
        let d = vec![1.0f32; SQUARES];
        g.distribution_step(&d, &d, &mut rng, &mut scratch);
    }
    g
}

fn random_stats_puct(legal: [u64; LEGAL_WORDS], rng: &mut Rng) -> PuctStats {
    let mut s = PuctStats::default();
    for player in 0..2 {
        for sq in squares(legal) {
            s.prior[player][sq] = rng.random() as f32;
            s.visit[player][sq] = (rng.randint(30)) as u16;
            s.q[player][sq] = rng.random() as f32 * 2.0 - 1.0;
        }
    }
    s
}

/// `mcts_puct.py`'s `select`, transcribed.
fn puct_reference(s: &PuctStats, legal: [u64; LEGAL_WORDS], player: usize, c: f64) -> usize {
    let total: f64 = (0..SQUARES).map(|sq| f64::from(s.visit[player][sq])).sum();
    let mut best = f64::NEG_INFINITY;
    let mut action = usize::MAX;
    for sq in 0..SQUARES {
        let is_legal = legal[sq >> 6] >> (sq & 63) & 1 == 1;
        if !is_legal {
            continue;
        }
        let score = f64::from(s.q[player][sq])
            + c * f64::from(s.prior[player][sq]) * total.sqrt()
                / (1.0 + f64::from(s.visit[player][sq]));
        if score > best {
            best = score;
            action = sq;
        }
    }
    action
}

#[test]
fn puct_selection_matches_the_python_formula() {
    let mut rng = Rng::new(0x9C7);
    let c = cfg();
    let mut checked = 0;
    for trial in 0..2000u64 {
        let g = position((trial % 30) as u32, trial);
        if g.finished {
            continue;
        }
        let stats = random_stats_puct(g.legal_moves(0), &mut rng);
        for player in 0..2 {
            let legal = g.legal_moves(player);
            let got = Puct::select(&mut stats.clone(), legal, player, &c, &mut rng);
            let want = puct_reference(&stats, legal, player, f64::from(c.c_puct));
            assert_eq!(got.action as usize, want, "trial {trial} player {player}");
            assert_eq!(got.prob, 1.0, "puct is deterministic");
            checked += 1;
        }
    }
    assert!(checked > 3000, "only {checked} selections compared");
}

/// Only the legal squares may ever be chosen. The visit counts and priors of illegal squares
/// are deliberately left large to make sure they are not merely losing on score.
#[test]
fn puct_never_picks_an_illegal_square() {
    let mut rng = Rng::new(0x1117);
    let c = cfg();
    for trial in 0..500u64 {
        let g = position((trial % 35) as u32 + 1, trial + 7);
        if g.finished {
            continue;
        }
        let mut stats = PuctStats::default();
        for player in 0..2 {
            for sq in 0..SQUARES {
                stats.q[player][sq] = 100.0; // illegal squares would win on score
                stats.prior[player][sq] = 1.0;
            }
            let legal = g.legal_moves(player);
            for sq in squares(legal) {
                stats.q[player][sq] = -100.0;
            }
            let got = Puct::select(&mut stats.clone(), legal, player, &c, &mut rng);
            let sq = got.action as usize;
            assert!(legal[sq >> 6] >> (sq & 63) & 1 == 1, "picked illegal square {sq}");
        }
    }
}

/// A running mean, updated one value at a time, must equal the mean of the values.
#[test]
fn puct_backup_keeps_a_running_mean() {
    let c = cfg();
    let mut stats = PuctStats::default();
    let mut rng = Rng::new(0x5EE);
    let mut fed = [Vec::new(), Vec::new()];

    for _ in 0..500 {
        let v = [rng.random() as f32 * 2.0 - 1.0, rng.random() as f32 * 2.0 - 1.0];
        let choice = [Choice { action: 3, prob: 1.0 }, Choice { action: 9, prob: 1.0 }];
        Puct::backup(&mut stats, choice, v, [40, 40], &c);
        fed[0].push(f64::from(v[0]));
        fed[1].push(f64::from(v[1]));
    }
    for (player, sq) in [(0usize, 3usize), (1, 9)] {
        let want: f64 = fed[player].iter().sum::<f64>() / fed[player].len() as f64;
        assert_eq!(stats.visit[player][sq] as usize, fed[player].len());
        assert!(
            (f64::from(stats.q[player][sq]) - want).abs() < 1e-4,
            "player {player}: running mean {} vs true mean {want}",
            stats.q[player][sq]
        );
    }
    assert_eq!(stats.saturated, 0);
}

/// The `u16` visit ceiling must degrade quietly and be counted, not wrap.
#[test]
fn puct_backup_saturates_rather_than_wrapping() {
    let c = cfg();
    let mut stats = PuctStats::default();
    stats.visit[0][5] = u16::MAX;
    stats.visit[1][5] = u16::MAX - 1;
    let choice = [Choice { action: 5, prob: 1.0 }, Choice { action: 5, prob: 1.0 }];

    Puct::backup(&mut stats, choice, [1.0, 1.0], [40, 40], &c);
    assert_eq!(stats.visit[0][5], u16::MAX, "must not wrap");
    assert_eq!(stats.visit[1][5], u16::MAX);
    assert_eq!(stats.saturated, 1, "the dropped backup is counted");

    Puct::backup(&mut stats, choice, [1.0, 1.0], [40, 40], &c);
    assert_eq!(stats.saturated, 3);
}

/// `mcts_exp3.py`'s `_mixed_strategy`, transcribed.
fn exp3_reference(s: &Exp3Stats, legal: [u64; LEGAL_WORDS], player: usize, gamma: f64) -> Vec<f64> {
    let live: Vec<usize> = squares(legal).collect();
    let n = live.len() as f64;
    let top = live
        .iter()
        .map(|&sq| f64::from(s.log_w[player][sq]))
        .fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = live
        .iter()
        .map(|&sq| (f64::from(s.log_w[player][sq]) - top).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    let mut out = vec![0.0f64; SQUARES];
    for (k, &sq) in live.iter().enumerate() {
        out[sq] = (1.0 - gamma) * (exp[k] / sum) + gamma / n;
    }
    out
}

#[test]
fn the_exp3_mixed_strategy_matches_the_python_formula() {
    let mut rng = Rng::new(0xE3E3);
    let c = cfg();
    let gamma = f64::from(c.exp3_gamma);

    for trial in 0..1000u64 {
        let g = position((trial % 30) as u32, trial + 11);
        if g.finished {
            continue;
        }
        let mut stats = Exp3Stats::default();
        for player in 0..2 {
            for sq in squares(g.legal_moves(player)) {
                stats.log_w[player][sq] = rng.random() as f32 * 8.0 - 4.0;
            }
        }
        for player in 0..2 {
            let legal = g.legal_moves(player);
            let mut got = [0.0f32; SQUARES];
            Exp3::mixed(&stats, legal, player, c.exp3_gamma, &mut got);
            let want = exp3_reference(&stats, legal, player, gamma);

            let mut total = 0.0f64;
            for sq in 0..SQUARES {
                assert!(
                    (f64::from(got[sq]) - want[sq]).abs() < 1e-6,
                    "trial {trial} player {player} square {sq}: {} vs {}",
                    got[sq],
                    want[sq]
                );
                total += f64::from(got[sq]);
            }
            assert!((total - 1.0).abs() < 1e-4, "strategy sums to {total}, not 1");
        }
    }
}

/// A huge log weight must not overflow the softmax, which is why it is shifted by the max.
#[test]
fn the_exp3_softmax_survives_extreme_weights() {
    let c = cfg();
    let g = position(6, 3);
    let legal = g.legal_moves(0);
    let live: Vec<usize> = squares(legal).collect();

    let mut stats = Exp3Stats::default();
    stats.log_w[0][live[0]] = 1e30;
    stats.log_w[0][live[1]] = -1e30;

    let mut out = [0.0f32; SQUARES];
    Exp3::mixed(&stats, legal, 0, c.exp3_gamma, &mut out);
    let total: f32 = live.iter().map(|&sq| out[sq]).sum();
    assert!(total.is_finite(), "the strategy went non-finite");
    assert!((total - 1.0).abs() < 1e-3, "strategy sums to {total}");
    let floor = c.exp3_gamma / live.len() as f32;
    for &sq in &live {
        assert!(out[sq] >= floor * 0.999, "square {sq} fell below the exploration floor");
    }
}

/// Sampling must follow the strategy: over many draws the empirical frequencies converge to
/// it, and every draw reports the probability that square actually had.
#[test]
fn exp3_sampling_follows_the_mixed_strategy() {
    let mut rng = Rng::new(0x5A11);
    let c = cfg();
    let g = position(8, 21);
    let legal = g.legal_moves(0);
    let live: Vec<usize> = squares(legal).collect();

    let mut stats = Exp3Stats::default();
    for (k, &sq) in live.iter().enumerate() {
        stats.log_w[0][sq] = (k % 5) as f32;
    }
    let mut want = [0.0f32; SQUARES];
    Exp3::mixed(&stats, legal, 0, c.exp3_gamma, &mut want);

    let draws = 400_000;
    let mut hits = vec![0u32; SQUARES];
    for _ in 0..draws {
        let ch = Exp3::select(&mut stats, legal, 0, &c, &mut rng);
        let sq = ch.action as usize;
        assert!(legal[sq >> 6] >> (sq & 63) & 1 == 1, "sampled illegal square {sq}");
        assert!(
            (ch.prob - want[sq]).abs() < 1e-6,
            "reported prob {} but the strategy says {}",
            ch.prob,
            want[sq]
        );
        hits[sq] += 1;
    }
    for &sq in &live {
        let seen = hits[sq] as f64 / draws as f64;
        let expect = f64::from(want[sq]);
        assert!(
            (seen - expect).abs() < 0.004,
            "square {sq}: sampled {seen:.4}, strategy {expect:.4}"
        );
    }
}

/// A positive reward has to make its square more likely, and the step size has to be the
/// Python's `gamma * (reward / prob) / num_legal`.
#[test]
fn exp3_backup_matches_the_python_update() {
    let c = cfg();
    let g = position(8, 5);
    let counts = [g.legal_count(0), g.legal_count(1)];

    let mut stats = Exp3Stats::default();
    let choice = [Choice { action: 12, prob: 0.25 }, Choice { action: 30, prob: 0.5 }];
    let values = [0.6f32, -0.2];
    let before = [stats.log_w[0][12], stats.log_w[1][30]];

    Exp3::backup(&mut stats, choice, values, counts, &c);

    for (player, ch, sq) in [(0usize, choice[0], 12usize), (1, choice[1], 30)] {
        let reward = f64::from((values[player] + 1.0) / 2.0);
        let want = f64::from(before[player])
            + f64::from(c.exp3_gamma) * (reward / f64::from(ch.prob))
                / f64::from(counts[player]);
        assert!(
            (f64::from(stats.log_w[player][sq]) - want).abs() < 1e-5,
            "player {player}: {} vs {want}",
            stats.log_w[player][sq]
        );
    }
    assert!(stats.log_w[0][12] > before[0], "a positive reward must raise the weight");
}

/// Priors reach the two variants in the shapes each documents: PUCT renormalised over the
/// legal squares, EXP3 as log weights with illegal squares at -inf.
#[test]
fn priors_are_adopted_the_way_each_variant_documents() {
    let mut rng = Rng::new(0xB0B);
    let g = position(9, 77);
    let priors: Vec<f32> = (0..SQUARES).map(|_| rng.random() as f32).collect();

    for player in 0..2 {
        let legal = g.legal_moves(player);

        let mut p = PuctStats::default();
        Puct::set_priors(&mut p, &priors, legal, player);
        let total: f64 = (0..SQUARES).map(|sq| f64::from(p.prior[player][sq])).sum();
        assert!((total - 1.0).abs() < 1e-5, "puct priors sum to {total}");
        for sq in 0..SQUARES {
            let is_legal = legal[sq >> 6] >> (sq & 63) & 1 == 1;
            assert_eq!(
                p.prior[player][sq] == 0.0,
                !is_legal || priors[sq] == 0.0,
                "square {sq} legality and prior disagree"
            );
        }

        let mut e = Exp3Stats::default();
        Exp3::set_priors(&mut e, &priors, legal, player);
        for sq in 0..SQUARES {
            let is_legal = legal[sq >> 6] >> (sq & 63) & 1 == 1;
            if is_legal {
                let want = f64::from(priors[sq].max(1e-8)).ln();
                assert!((f64::from(e.log_w[player][sq]) - want).abs() < 1e-5);
            } else {
                assert_eq!(e.log_w[player][sq], f32::NEG_INFINITY, "square {sq} not masked");
            }
        }
        // and the resulting strategy is the priors, renormalised, mixed with the floor
        let mut mixed = [0.0f32; SQUARES];
        Exp3::mixed(&e, legal, player, 0.0, &mut mixed);
        for sq in squares(legal) {
            assert!(
                (mixed[sq] - p.prior[player][sq]).abs() < 1e-5,
                "square {sq}: exp3 gives {} where puct gives {}",
                mixed[sq],
                p.prior[player][sq]
            );
        }
    }
}

/// An untouched node plays uniformly, which is what an unevaluated one is supposed to do.
#[test]
fn default_exp3_weights_give_the_uniform_strategy() {
    let c = cfg();
    let g = position(4, 99);
    let stats = Exp3Stats::default();
    for player in 0..2 {
        let legal = g.legal_moves(player);
        let n = g.legal_count(player) as f32;
        let mut out = [0.0f32; SQUARES];
        Exp3::mixed(&stats, legal, player, c.exp3_gamma, &mut out);
        for sq in squares(legal) {
            assert!((out[sq] - 1.0 / n).abs() < 1e-6, "square {sq} is not uniform");
        }
    }
}

/// The average strategy is what a root emits as its training target, and it is accumulated
/// on every node — not only on roots — so that promoting an interior node into a root keeps
/// the search work already done at that state instead of restarting the average.
#[test]
fn exp3_selection_accumulates_the_average_strategy() {
    let c = cfg();
    let mut rng = Rng::new(0xA7E);
    let g = position(7, 31);
    let legal = g.legal_moves(0);
    let mut stats = Exp3Stats::default();

    let mut want = [0.0f32; SQUARES];
    Exp3::mixed(&stats, legal, 0, c.exp3_gamma, &mut want);

    let draws = 500;
    for _ in 0..draws {
        Exp3::select(&mut stats, legal, 0, &c, &mut rng);
    }
    // no backup ran, so the strategy never moved and the sum is that strategy, `draws` times
    for sq in squares(legal) {
        let got = f64::from(stats.strategy_sum[0][sq]);
        let expect = f64::from(want[sq]) * f64::from(draws);
        assert!(
            (got - expect).abs() < 0.05,
            "square {sq}: summed {got}, expected {expect}"
        );
    }
    // the other player was never selected for, so its sum is untouched
    assert!(stats.strategy_sum[1].iter().all(|&v| v == 0.0));

    // normalised, the sum is the strategy back again — which is what the target is
    let total: f32 = squares(legal).map(|sq| stats.strategy_sum[0][sq]).sum();
    for sq in squares(legal) {
        assert!((stats.strategy_sum[0][sq] / total - want[sq]).abs() < 1e-4);
    }
}

/// Backups move the strategy, so the average must lag behind the current one — that is the
/// whole point of averaging rather than reading the latest.
#[test]
fn the_average_strategy_lags_the_current_one() {
    let c = cfg();
    let mut rng = Rng::new(0x1A6);
    let g = position(7, 32);
    let legal = g.legal_moves(0);
    let counts = [g.legal_count(0), g.legal_count(1)];
    let favourite = squares(legal).next().unwrap();
    let mut stats = Exp3Stats::default();

    for _ in 0..300 {
        Exp3::select(&mut stats, legal, 0, &c, &mut rng);
        // reward one square hard, every time
        let choice = [
            Choice { action: favourite as u8, prob: 0.1 },
            Choice { action: squares(g.legal_moves(1)).next().unwrap() as u8, prob: 0.1 },
        ];
        Exp3::backup(&mut stats, choice, [1.0, 1.0], counts, &c);
    }

    let mut current = [0.0f32; SQUARES];
    Exp3::mixed(&stats, legal, 0, c.exp3_gamma, &mut current);
    let total: f32 = squares(legal).map(|sq| stats.strategy_sum[0][sq]).sum();
    let average = stats.strategy_sum[0][favourite] / total;

    // the mixture caps any square at (1 - gamma) + gamma / n, so it can never reach 1
    let uniform = 1.0 / counts[0] as f32;
    let ceiling = 1.0 - c.exp3_gamma + c.exp3_gamma * uniform;
    assert!(
        current[favourite] > 5.0 * uniform && current[favourite] <= ceiling,
        "the rewarded square is at {} against uniform {uniform} and a ceiling of {ceiling}",
        current[favourite]
    );
    assert!(
        average < current[favourite],
        "the average ({average}) should lag the current strategy ({})",
        current[favourite]
    );
    assert!(average > 1.0 / counts[0] as f32, "but it should still have moved off uniform");
}
