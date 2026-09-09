use alpha_lines_game::game::{Game, Rng, SQUARES};
use alpha_lines_game::mcts::arena::Arena;
use alpha_lines_game::mcts::config::{K, OVERFLOWED};
use alpha_lines_game::mcts::variant::{Choice, Exp3, Exp3Stats, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::Config;

#[test]
fn puct_constant_values_do_not_drift() {
    let cfg = Config::default();
    let mut stats = PuctStats::default();
    for _ in 0..100_000 {
        Puct::backup(&mut stats, [Choice { action: 0, prob: 1.0 }; 2], [0.6; 2], [80; 2], &cfg);
    }
    assert_eq!(stats.q[0][0], 0.6f32 as f64);
}

#[test]
fn puct_late_updates_still_change_q() {
    for n in [1u64 << 24, 20_000_000, 100_000_000_000] {
        let mut stats = PuctStats::default();
        stats.visit[0][0] = n;
        stats.total[0] = n;
        stats.q[0][0] = 0.5;
        Puct::backup(&mut stats, [Choice { action: 0, prob: 1.0 }; 2], [0.6; 2], [80; 2], &Config::default());
        let delta = stats.q[0][0] - 0.5;
        let expected = (0.6f32 as f64 - 0.5) / (n + 1) as f64;
        assert!(delta > 0.0);
        assert!((delta - expected).abs() < f64::EPSILON);
    }
}

#[test]
fn exp3_recentering_preserves_policy_and_mask() {
    let cfg = Config::default();
    let mut shifted = Exp3Stats::default();
    let mut reference = Exp3Stats::default();
    for p in 0..2 {
        shifted.log_w[p].fill(f64::NEG_INFINITY);
        reference.log_w[p].fill(f64::NEG_INFINITY);
        shifted.log_w[p][0] = 16384.0;
        shifted.log_w[p][1] = 16383.0;
        reference.log_w[p][0] = 0.0;
        reference.log_w[p][1] = -1.0;
    }
    let choice = [Choice { action: 0, prob: 0.9 }; 2];
    Exp3::backup(&mut shifted, choice, [0.0; 2], [2; 2], &cfg);
    Exp3::backup(&mut reference, choice, [0.0; 2], [2; 2], &cfg);
    assert_eq!(shifted.log_w, reference.log_w);
    let mut actual = [0.0; SQUARES];
    let mut expected = [0.0; SQUARES];
    Exp3::mixed(&shifted, [3, 0], 0, cfg.exp3_gamma, &mut actual);
    Exp3::mixed(&reference, [3, 0], 0, cfg.exp3_gamma, &mut expected);
    assert_eq!(actual, expected);
    assert!(shifted.log_w[0][2].is_infinite());
}

#[test]
fn exp3_large_strategy_sum_keeps_small_contributions() {
    let cfg = Config::default();
    let mut stats = Exp3Stats::default();
    stats.strategy_sum[0][0] = 16_777_216.0;
    Exp3::select(&mut stats, [[3, 0]; 2], &cfg, &mut Rng::new(5));
    assert_eq!(stats.strategy_sum[0][0], 16_777_216.5);
}

#[test]
fn node_keeps_64_distinct_ids_before_overwriting() {
    assert_eq!(K, 64);
    let mut arena = Arena::<()>::new(&Config { node_capacity: 1, ..Config::default() });
    let slot = arena.insert(1, Game::new(), 0, 0).unwrap();
    for id in 1..64 { arena.touch(slot, id); }
    assert_eq!(arena.node(slot).ids(), &(0..64u16).collect::<Vec<_>>());
    assert_eq!(arena.node(slot).flags & OVERFLOWED, 0);
    arena.touch(slot, 20);
    assert_eq!(arena.node(slot).id_count, 64);
    arena.touch(slot, 64);
    assert_eq!(arena.node(slot).id_count, 64);
    assert!(!arena.node(slot).ids().contains(&0));
    assert!(arena.node(slot).ids().contains(&64));
    assert_ne!(arena.node(slot).flags & OVERFLOWED, 0);
}
