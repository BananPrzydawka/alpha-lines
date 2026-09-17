//! The whole loop: cycles, roots, steps, sweeps and reseeding, against a stand-in model.
//!
//! What is being checked is the bookkeeping the spec makes load-bearing — that no game is
//! left holding a path, that the sweep leaves no unreferenced node behind, and that a node is
//! only ever kept by a game that can still reach it.

use alpha_lines_game::game::{Rng, SQUARES};
use alpha_lines_game::mcts::search::{Evaluate, Search, StepRecord};
use alpha_lines_game::mcts::variant::{squares, Exp3, Puct, Variant};
use alpha_lines_game::mcts::Config;
use alpha_lines_game::Game;

/// An untrained network: near-uniform priors, noisy values.
struct Stub(Rng);
impl Evaluate for Stub {
    fn evaluate(&mut self, _positions: &[[u8; SQUARES]], _scores: &[[i32; 2]], priors: &mut [f32], values: &mut [f32]) {
        for p in priors.iter_mut() {
            *p = 1.0 + self.0.random() as f32 * 0.05;
        }
        for v in values.iter_mut() {
            *v = self.0.random() as f32 * 2.0 - 1.0;
        }
    }
}

/// Small enough that games finish inside a test.
fn cfg() -> Config {
    Config { g: 64, b: 128, t: 16, s: 8, node_capacity: 40_000, ..Config::default() }
}

/// Everything that must be true between cycles.
fn check<V: Variant>(s: &Search<V>) {
    for (i, slot) in s.slots.iter().enumerate() {
        assert!(slot.occupied, "slot {i} went unoccupied");
        assert_eq!(slot.waiting_on, None, "slot {i} still holds an unresolved path");
        assert!(slot.path.is_empty(), "slot {i} left a path behind");
        assert!(slot.sim_count <= s.cfg.s, "slot {i} overran S at {}", slot.sim_count);
        assert!(!slot.game_over, "a finished game was not reseeded");
    }
    // §9 leak assertion: the sweep deletes every node nobody claims
    for i in 0..s.arena.slot_count() as u32 {
        if s.arena.is_unused(i) {
            continue;
        }
        let n = s.arena.node(i);
        assert!(!n.ids().is_empty(), "node {i} survived with no ids");
        assert!(n.ids().len() <= alpha_lines_game::mcts::K);
        // a node is kept only by games that can still reach it
        for &id in n.ids() {
            let root = &s.slots[id as usize].root.game;
            assert!(
                n.game.reachable_from(root),
                "node {i} is kept by game {id}, which can no longer reach it"
            );
        }
    }
}

fn run<V: Variant>(cycles: usize, seed: u64) -> (Search<V>, Vec<StepRecord>) {
    let mut s = Search::<V>::new(cfg(), seed);
    let mut model = Stub(Rng::new(seed ^ 0x5B));
    let mut all = Vec::new();
    for c in 0..cycles {
        all.extend(s.cycle(&mut model));
        if c % 25 == 0 {
            check(&s);
        }
    }
    check(&s);
    (s, all)
}

#[test]
fn a_cycle_leaves_no_game_waiting_and_no_node_orphaned() {
    let (s, _) = run::<Puct>(200, 0x11);
    assert!(s.diag.cycles == 200);
    assert!(s.diag.steps > 0, "no game ever stepped");
    assert_eq!(s.diag.exhausted, 0, "ran out of nodes on a small board budget");
}

/// All slots open at the same position, so the pending-root table must collapse them to one
/// evaluation rather than 64.
#[test]
fn identical_roots_cost_one_evaluation() {
    let mut s = Search::<Puct>::new(cfg(), 0x22);
    let mut model = Stub(Rng::new(3));
    s.cycle(&mut model);
    assert_eq!(s.diag.root_evals, 1, "64 identical opening roots should share one row");
    for slot in s.slots.iter() {
        assert!(!slot.root.pending, "a waiter did not receive the root evaluation");
    }
}

/// The first cycle queues the roots; the second has to fan out. If selection ignores
/// priors when nothing is visited, every game picks the same square and the search
/// never widens.
#[test]
fn the_first_cycle_does_not_collapse_onto_one_move() {
    let mut s = Search::<Puct>::new(cfg(), 0x33);
    let mut model = Stub(Rng::new(4));
    s.cycle(&mut model);
    assert_eq!(s.arena.len(), 0, "cycle 0 queues roots, it builds nothing");
    s.cycle(&mut model);
    assert!(s.arena.len() > 8, "only {} distinct children from 64 games", s.arena.len());
}

/// Games must actually reach the end, be marked over, and have their slots reseeded.
#[test]
fn games_finish_and_their_slots_are_reused() {
    let (s, records) = run::<Puct>(900, 0x44);
    let finished = records.iter().filter(|r| r.finished).count();
    assert!(finished > 0, "no game finished in {} moves", records.len());
    for r in records.iter().filter(|r| r.finished) {
        assert!(r.values == [1.0, -1.0] || r.values == [-1.0, 1.0] || r.values == [0.0, 0.0]);
    }
    assert_eq!(
        s.diag.games_finished as usize, finished,
        "every finished game must have had its slot reseeded"
    );
}

/// A step's policy target is a distribution over exactly the legal squares, and the move it
/// reports was drawn from it.
#[test]
fn training_targets_are_distributions_over_legal_moves() {
    let (_, records) = run::<Puct>(300, 0x55);
    assert!(!records.is_empty());
    for r in &records {
        let mut scratch = alpha_lines_game::Scratch::new();
        let g = Game::from_cells(r.cells, &mut scratch);
        for player in 0..2 {
            let legal = g.legal_moves(player);
            let total: f32 = squares(legal).map(|sq| r.policy[player][sq]).sum();
            assert!((total - 1.0).abs() < 1e-3, "policy sums to {total}");
            for sq in 0..SQUARES {
                let is_legal = legal[sq >> 6] >> (sq & 63) & 1 == 1;
                assert!(is_legal || r.policy[player][sq] == 0.0, "weight on illegal {sq}");
            }
            let played = r.played[player] as usize;
            assert!(legal[played >> 6] >> (played & 63) & 1 == 1, "played illegal square");
        }
    }
}

/// The sweep is the only thing that keeps the arena bounded; without it the node count would
/// climb without limit.
#[test]
fn the_sweep_reclaims_what_the_games_have_left_behind() {
    let (s, _) = run::<Puct>(600, 0x66);
    assert!(s.diag.nodes_deleted > 0, "the sweep never deleted anything");
    assert!(
        s.arena.len() < s.diag.buffer_unique as usize,
        "nothing was reclaimed: {} live against {} ever created",
        s.arena.len(),
        s.diag.buffer_unique
    );
    assert!(s.arena.free_depth() > 0, "no slot was returned to the free list");
}

/// EXP3 drives the same loop.
#[test]
fn exp3_runs_the_whole_loop_too() {
    let (s, records) = run::<Exp3>(400, 0x77);
    assert!(s.diag.steps > 0);
    assert!(!records.is_empty());
    for r in &records {
        for player in 0..2 {
            let total: f32 = r.policy[player].iter().sum();
            assert!((total - 1.0).abs() < 1e-3, "exp3 policy sums to {total}");
        }
    }
}

/// A node budget too small to hold the search must degrade, not panic or corrupt.
#[test]
fn a_tight_node_budget_degrades_instead_of_failing() {
    let c = Config { node_capacity: 500, ..cfg() };
    let mut s = Search::<Puct>::new(c, 0x88);
    let mut model = Stub(Rng::new(9));
    for _ in 0..150 {
        s.cycle(&mut model);
    }
    check(&s);
    assert!(s.diag.exhausted > 0, "500 nodes should not have been enough");
    assert!(s.arena.len() <= 500);
}
