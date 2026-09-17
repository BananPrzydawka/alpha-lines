use alpha_lines_game::game::{Game, SQUARES};
use alpha_lines_game::mcts::arena::Arena;
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::variant::Puct;
use alpha_lines_game::mcts::Config;
use std::collections::HashMap;

struct Uniform;
impl Evaluate for Uniform {
    fn evaluate(&mut self, _: &[[u8; SQUARES]], _scores: &[[i32; 2]], priors: &mut [f32], values: &mut [f32]) {
        priors.fill(1.0);
        values.fill(0.0);
    }
}

#[test]
fn reused_slot_starts_at_age_zero() {
    let cfg = Config { node_capacity: 1, ..Config::default() };
    let mut arena = Arena::<()>::new(&cfg);
    let slot = arena.insert(1, Game::new(), 0, 0).unwrap();
    assert_eq!(arena.node(slot).sweep_age, 0);
    arena.node_mut(slot).sweep_age = 42;
    arena.remove(slot);
    let reused = arena.insert(2, Game::new(), 0, 0).unwrap();
    assert_eq!(slot, reused);
    assert_eq!(arena.node(reused).sweep_age, 0);
}

#[test]
fn age_changes_only_for_surviving_sweeps() {
    let cfg = Config { g: 8, b: 4, t: usize::MAX, s: 8, node_capacity: 2048, ..Config::default() };
    let mut search = Search::<Puct>::new(cfg, 123);
    let mut model = Uniform;
    let mut survivors_seen = 0;
    for _ in 0..6 {
        let mut cycles = 0;
        while search.ready() < 4 {
            let before: HashMap<_, _> = search.arena.live_slots().map(|slot| {
                let n = search.arena.node(slot);
                (n.key, n.sweep_age)
            }).collect();
            search.cycle(&mut model);
            for slot in search.arena.live_slots() {
                let n = search.arena.node(slot);
                assert_eq!(n.sweep_age, before.get(&n.key).copied().unwrap_or(0));
            }
            cycles += 1;
            assert!(cycles < 1000);
        }
        let before: HashMap<_, _> = search.arena.live_slots().map(|slot| {
            let n = search.arena.node(slot);
            (n.key, n.sweep_age)
        }).collect();
        assert!(!search.step().is_empty());
        for slot in search.arena.live_slots() {
            let n = search.arena.node(slot);
            assert_eq!(n.sweep_age, before[&n.key] + 1);
            survivors_seen += 1;
        }
    }
    assert!(survivors_seen > 0);
}
