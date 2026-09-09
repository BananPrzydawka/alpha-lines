//! Descent against the engine: the claim under test is that walking the tree by Zobrist key
//! alone lands on exactly the positions playing the moves would have produced, and that a
//! game holds at most one path at a time.

use alpha_lines_game::game::{Rng, Scratch, SQUARES};
use alpha_lines_game::mcts::arena::Arena;
use alpha_lines_game::mcts::config::PENDING;
use alpha_lines_game::mcts::descent::{back_up, descend, Descent};
use alpha_lines_game::mcts::slot::{Slot, ROOT};
use alpha_lines_game::mcts::variant::{squares, Exp3, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::Config;
use alpha_lines_game::{zobrist, Game};

fn cfg() -> Config {
    Config { node_capacity: 4096, ..Config::default() }
}

/// A slot playing from the opening.
fn opening<S: Default>() -> Slot<S> {
    let mut slot = Slot::<S>::default();
    slot.occupied = true;
    slot.root.game = Game::new();
    slot.root.key = zobrist::hash(&slot.root.game.cells);
    slot.root.pending = false;
    slot
}

/// What a cycle does for one game: hand the awaited node priors, clear `pending`, back up.
fn resolve<V: Variant>(
    slot: &mut Slot<V::Stats>,
    arena: &mut Arena<V::Stats>,
    values: [f32; 2],
    cfg: &Config,
    rng: &mut Rng,
) {
    let node = slot.waiting_on.expect("nothing to resolve");
    let priors: Vec<f32> = (0..SQUARES).map(|_| rng.random() as f32 + 0.01).collect();
    let n = arena.node_mut(node);
    n.flags &= !PENDING;
    let legal = [n.game.legal_moves(0), n.game.legal_moves(1)];
    for player in 0..2 {
        V::set_priors(&mut n.stats, &priors, legal[player], player);
    }
    back_up::<V>(slot, arena, values, cfg);
}

/// Descend, resolve, repeat — the search a single game would run on its own.
fn run<V: Variant>(sims: usize, seed: u64) -> (Slot<V::Stats>, Arena<V::Stats>) {
    let c = cfg();
    let mut arena = Arena::<V::Stats>::new(&c);
    let mut slot = opening::<V::Stats>();
    let mut rng = Rng::new(seed);
    let mut scratch = Scratch::new();

    while (slot.sim_count as usize) < sims {
        match descend::<V>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch) {
            Descent::Entry { .. } => {
                let v = [rng.random() as f32 * 2.0 - 1.0, 0.0];
                resolve::<V>(&mut slot, &mut arena, [v[0], -v[0]], &c, &mut rng);
            }
            Descent::NoEntry => {}
            Descent::Exhausted => panic!("ran out of nodes at {} sims", slot.sim_count),
        }
    }
    (slot, arena)
}

#[test]
fn the_first_descent_builds_one_child_and_waits_on_it() {
    let c = cfg();
    let mut arena = Arena::<PuctStats>::new(&c);
    let mut slot = opening::<PuctStats>();
    let mut rng = Rng::new(1);
    let mut scratch = Scratch::new();

    let got = descend::<Puct>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch);
    let Descent::Entry { node, fresh } = got else { panic!("expected an entry, got {got:?}") };
    assert!(fresh, "the game that created the node owes the buffer an entry");
    assert_eq!(arena.len(), 1);
    assert_eq!(slot.waiting_on, Some(node));
    assert_eq!(slot.path.len(), 1, "one step: the root");
    assert_eq!(slot.path.step(0).node, ROOT);
    assert_eq!(slot.sim_count, 0, "nothing is backed up until the evaluation lands");
}

/// The whole point of hashing the move instead of playing it: the node the key finds holds
/// the board the engine would have produced.
#[test]
fn nodes_hold_the_position_their_key_names() {
    let (slot, arena) = run::<Puct>(400, 0xC0FFEE);
    let mut checked = 0;
    for i in arena.live_slots() {
        let n = arena.node(i);
        assert_eq!(n.key, zobrist::hash(&n.game.cells), "node {i} key does not match its board");
        assert!(n.game.reachable_from(&slot.root.game), "node {i} is not reachable from the root");
        assert!(n.game.move_count > slot.root.game.move_count, "node {i} is not below the root");
        checked += 1;
    }
    assert!(checked > 100, "only {checked} nodes to check");
}

/// A second game reaching a node that is already queued must ride that entry rather than
/// duplicating the evaluation, and must still be recorded as a user of the node.
#[test]
fn a_pending_hit_makes_no_second_entry() {
    let c = cfg();
    let mut arena = Arena::<PuctStats>::new(&c);
    let mut a = opening::<PuctStats>();
    let mut b = opening::<PuctStats>();
    let mut scratch = Scratch::new();

    // the same seed, so both games select the same joint move from the same root
    let first = descend::<Puct>(0, &mut a, &mut arena, &c, &mut Rng::new(7), &mut scratch);
    let second = descend::<Puct>(1, &mut b, &mut arena, &c, &mut Rng::new(7), &mut scratch);

    let Descent::Entry { node: n0, fresh: f0 } = first else { panic!("{first:?}") };
    let Descent::Entry { node: n1, fresh: f1 } = second else { panic!("{second:?}") };
    assert_eq!(n0, n1, "both games should have landed on the same state");
    assert!(f0 && !f1, "only the first arrival owes an entry");
    assert_eq!(arena.len(), 1, "the second descent must not allocate");
    assert_eq!(arena.node(n0).ids(), &[0, 1], "both games are recorded on the node");
    assert_eq!(a.waiting_on, Some(n0));
    assert_eq!(b.waiting_on, Some(n0));
}

/// Once a node is evaluated, descents pass through it instead of stopping there.
#[test]
fn descents_deepen_as_nodes_become_evaluated() {
    let c = cfg();
    let mut arena = Arena::<PuctStats>::new(&c);
    let mut slot = opening::<PuctStats>();
    let mut rng = Rng::new(0x5EED);
    let mut scratch = Scratch::new();

    let mut deepest = 0;
    for _ in 0..300 {
        match descend::<Puct>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch) {
            Descent::Entry { .. } => {
                deepest = deepest.max(slot.path.len());
                resolve::<Puct>(&mut slot, &mut arena, [0.5, -0.5], &c, &mut rng);
            }
            Descent::NoEntry => {}
            Descent::Exhausted => panic!("out of nodes"),
        }
    }
    assert!(deepest >= 3, "the search never went deeper than {deepest}");
}

/// Every step of a path names a move that was legal at that node, and no position repeats,
/// which is what makes the order of backup irrelevant.
#[test]
fn paths_are_legal_and_never_revisit_a_position() {
    let c = cfg();
    let mut arena = Arena::<PuctStats>::new(&c);
    let mut slot = opening::<PuctStats>();
    let mut rng = Rng::new(0xBEEF);
    let mut scratch = Scratch::new();

    for _ in 0..400 {
        let got = descend::<Puct>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch);
        let mut seen: Vec<u64> = Vec::new();
        for k in 0..slot.path.len() {
            let step = slot.path.step(k);
            let (key, game) = if step.node == ROOT {
                (slot.root.key, &slot.root.game)
            } else {
                (arena.node(step.node).key, &arena.node(step.node).game)
            };
            assert!(!seen.contains(&key), "position repeated on one path");
            seen.push(key);
            for player in 0..2 {
                let sq = step.choice[player].action as usize;
                let mask = game.legal_moves(player);
                assert!(mask[sq >> 6] >> (sq & 63) & 1 == 1, "played illegal square {sq}");
            }
        }
        assert_eq!(slot.path.step(0).node, ROOT, "a path starts at the root");
        if let Descent::Entry { .. } = got {
            resolve::<Puct>(&mut slot, &mut arena, [0.1, -0.1], &c, &mut rng);
        }
    }
}

/// A terminal is backed up on the spot: it needs no network, so the game stays in the cycle.
#[test]
fn terminals_are_counted_without_an_evaluation() {
    let (slot, _) = run::<Puct>(600, 0x7E12);
    assert!(slot.sim_count >= 600);
    assert!(slot.path.is_empty(), "a retired simulation leaves no path");
    assert_eq!(slot.waiting_on, None);
}

/// Backup must reach every node on the route, not only the leaf.
#[test]
fn backup_credits_the_whole_path() {
    let c = cfg();
    let mut arena = Arena::<PuctStats>::new(&c);
    let mut slot = opening::<PuctStats>();
    let mut rng = Rng::new(0x8AC);
    let mut scratch = Scratch::new();

    // build some depth first
    for _ in 0..200 {
        if let Descent::Entry { .. } =
            descend::<Puct>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch)
        {
            resolve::<Puct>(&mut slot, &mut arena, [0.2, -0.2], &c, &mut rng);
        }
    }
    // then take one descent and watch its whole path move
    let got = descend::<Puct>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch);
    assert!(matches!(got, Descent::Entry { .. }));
    assert!(slot.path.len() >= 2, "need a path with an interior node");

    let before: Vec<u64> = (0..slot.path.len())
        .map(|k| {
            let s = slot.path.step(k);
            let a = s.choice[0].action as usize;
            if s.node == ROOT { slot.root.stats.visit[0][a] } else { arena.node(s.node).stats.visit[0][a] }
        })
        .collect();
    let steps: Vec<_> = (0..slot.path.len()).map(|k| slot.path.step(k)).collect();
    resolve::<Puct>(&mut slot, &mut arena, [1.0, -1.0], &c, &mut rng);

    for (k, s) in steps.iter().enumerate() {
        let a = s.choice[0].action as usize;
        let now = if s.node == ROOT {
            slot.root.stats.visit[0][a]
        } else {
            arena.node(s.node).stats.visit[0][a]
        };
        assert_eq!(now, before[k] + 1, "step {k} was not credited");
    }
}

/// The recorded depth must be the depth the descent actually reached, including for descents
/// that back up on the way out. A terminal clears its path before returning, so anything that
/// measures the path afterwards reads zero and silently loses every terminal from the profile.
#[test]
fn a_descent_records_the_depth_it_reached_even_when_it_backs_up() {
    let c = cfg();
    let mut arena = Arena::<PuctStats>::new(&c);
    let mut rng = Rng::new(0xD3B7);
    let mut scratch = Scratch::new();

    // a root a few plies from the end, so descents actually reach terminals
    let mut history = vec![Game::new().cells];
    let mut g = Game::new();
    let d = vec![1.0f32; SQUARES];
    while !g.finished {
        g.distribution_step(&d, &d, &mut rng, &mut scratch);
        history.push(g.cells);
    }
    let near_end = history[history.len() - 4];

    let mut slot = opening::<PuctStats>();
    slot.root.game = Game::from_cells(near_end, &mut scratch);
    slot.root.key = zobrist::hash(&slot.root.game.cells);
    assert!(!slot.root.game.finished, "the chosen root is already over");

    let (mut terminals, mut entries) = (0, 0);
    for _ in 0..2000 {
        match descend::<Puct>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch) {
            Descent::Entry { .. } => {
                // the path is still in flight, so it can be compared directly
                assert_eq!(
                    slot.last_depth as usize,
                    slot.path.len(),
                    "recorded depth disagrees with the path it came from"
                );
                assert!(slot.last_depth >= 1, "a descent selected at least at the root");
                entries += 1;
                resolve::<Puct>(&mut slot, &mut arena, [0.4, -0.4], &c, &mut rng);
            }
            Descent::NoEntry => {
                assert!(slot.path.is_empty(), "a backed-up descent should have cleared its path");
                assert!(
                    slot.last_depth >= 1,
                    "a terminal recorded depth {} -- it was measured after the backup cleared \
                     the path",
                    slot.last_depth
                );
                terminals += 1;
            }
            Descent::Exhausted => panic!("out of nodes"),
        }
    }
    assert!(terminals > 0, "no terminal was reached, so the case is untested");
    assert!(entries > 0);
}

/// A full node stack is a budget reached, not a fault: nothing is created, nothing is left
/// in flight, and the search can carry on.
#[test]
fn a_full_stack_ends_the_descent_cleanly() {
    let c = Config { node_capacity: 24, ..Config::default() };
    let mut arena = Arena::<PuctStats>::new(&c);
    let mut slot = opening::<PuctStats>();
    let mut rng = Rng::new(0xF0F0);
    let mut scratch = Scratch::new();

    let mut hit = false;
    for _ in 0..500 {
        match descend::<Puct>(0, &mut slot, &mut arena, &c, &mut rng, &mut scratch) {
            Descent::Entry { .. } => resolve::<Puct>(&mut slot, &mut arena, [0.0, 0.0], &c, &mut rng),
            Descent::NoEntry => {}
            Descent::Exhausted => {
                hit = true;
                assert_eq!(arena.len(), arena.capacity(), "exhausted below capacity");
                assert!(slot.path.is_empty(), "an exhausted descent left a path behind");
                assert_eq!(slot.waiting_on, None);
                break;
            }
        }
    }
    assert!(hit, "24 nodes should not have been enough");
}

/// EXP3 descends the same shape as PUCT: different rule, same contract.
#[test]
fn exp3_descends_and_shares_nodes_too() {
    let (slot, arena) = run::<Exp3>(400, 0x3E3);
    assert!(slot.sim_count >= 400);
    assert!(arena.len() > 50, "only {} nodes", arena.len());
    for i in arena.live_slots() {
        let n = arena.node(i);
        assert_eq!(n.key, zobrist::hash(&n.game.cells));
        let legal = n.game.legal_moves(0);
        assert!(squares(legal).count() as u32 == n.game.legal_count(0));
    }
}
