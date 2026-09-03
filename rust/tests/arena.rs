//! The node stack: lookup, deletion, slot reuse, tombstone rehashing, and the id rules.

use alpha_lines_game::game::SQUARES;
use alpha_lines_game::mcts::arena::IdWrite;
use alpha_lines_game::mcts::{Arena, Config, K};

fn cfg(capacity: usize) -> Config {
    Config { node_capacity: capacity, ..Config::default() }
}

fn board(seed: u8) -> [u8; SQUARES] {
    let mut b = [0u8; SQUARES];
    b[0] = seed;
    b
}

#[test]
fn a_node_can_be_found_by_key_and_not_by_another() {
    let mut a: Arena<()> = Arena::new(&cfg(64));
    let n = a.insert(0xABCD, board(1), 0, 7).unwrap();
    assert_eq!(a.get(0xABCD), Some(n));
    assert_eq!(a.get(0xABCE), None);
    assert_eq!(a.len(), 1);
    assert_eq!(a.node(n).ids(), &[7]);
}

/// Keys that land in the same bucket must still be told apart, and deleting one must not
/// hide the other behind a tombstone.
#[test]
fn colliding_keys_probe_past_each_other_and_past_tombstones() {
    let mut a: Arena<()> = Arena::new(&cfg(64));
    let slots = 128u64; // capacity 64 -> 128 index slots
    let (k1, k2, k3) = (5, 5 + slots, 5 + 2 * slots);
    let n1 = a.insert(k1, board(1), 0, 1).unwrap();
    let n2 = a.insert(k2, board(2), 0, 2).unwrap();
    let n3 = a.insert(k3, board(3), 0, 3).unwrap();
    assert_eq!((a.get(k1), a.get(k2), a.get(k3)), (Some(n1), Some(n2), Some(n3)));

    a.remove(n2);
    assert_eq!(a.get(k2), None, "a removed key must not be found");
    assert_eq!(a.get(k1), Some(n1), "a tombstone must not hide an earlier key");
    assert_eq!(a.get(k3), Some(n3), "a tombstone must not hide a later key");
}

#[test]
fn a_removed_slot_is_reused_and_carries_none_of_the_old_node() {
    let mut a: Arena<u32> = Arena::new(&cfg(4));
    let n = a.insert(1, board(1), 0, 11).unwrap();
    a.node_mut(n).stats = 999;
    a.remove(n);
    assert_eq!(a.len(), 0);
    assert_eq!(a.free_depth(), 1);

    let m = a.insert(2, board(2), 0, 22).unwrap();
    assert_eq!(m, n, "the freed slot should come back");
    assert_eq!(a.node(m).stats, 0, "stats must be reset");
    assert_eq!(a.node(m).ids(), &[22]);
    assert_eq!(a.node(m).board, board(2));
    assert_eq!(a.get(1), None, "the old key must be gone");
}

#[test]
fn the_stack_refuses_to_grow_past_capacity() {
    let mut a: Arena<()> = Arena::new(&cfg(3));
    for k in 0..3 {
        assert!(a.insert(k, board(0), 0, 0).is_some(), "insert {k} should fit");
    }
    assert!(a.insert(99, board(0), 0, 0).is_none(), "capacity must be a hard limit");
    a.remove(a.get(1).unwrap());
    assert!(a.insert(99, board(0), 0, 0).is_some(), "freeing one makes room for one");
}

/// Churn the arena well past the tombstone limit and require every surviving key to remain
/// findable, which is what the rehash is for.
#[test]
fn rehashing_after_many_deletions_keeps_every_surviving_key() {
    let mut a: Arena<()> = Arena::new(&cfg(256));
    let mut live: Vec<u64> = Vec::new();
    for round in 0..40u64 {
        for i in 0..20u64 {
            let key = round * 1000 + i + 1;
            if a.insert(key, board(0), 0, 0).is_some() {
                live.push(key);
            }
        }
        // drop half of what is live, oldest first
        let drop = live.len() / 2;
        for key in live.drain(..drop).collect::<Vec<_>>() {
            let n = a.get(key).expect("live key must be present");
            a.remove(n);
        }
        for &key in &live {
            assert!(a.get(key).is_some(), "key {key} lost in round {round}");
        }
    }
    assert_eq!(a.len(), live.len());
}

#[test]
fn ids_append_then_wrap_and_never_duplicate() {
    let mut a: Arena<()> = Arena::new(&cfg(4));
    let n = a.insert(1, board(0), 0, 0).unwrap();

    assert_eq!(a.touch(n, 0), IdWrite::Present, "the seeding id is already there");
    for id in 1..K as u16 {
        assert_eq!(a.touch(n, id), IdWrite::Appended);
    }
    assert_eq!(a.node(n).id_count as usize, K);
    assert_eq!(a.touch(n, 3), IdWrite::Present, "a listed id must not be written twice");

    // full: the next new id displaces the oldest, and the cursor walks
    assert_eq!(a.touch(n, 100), IdWrite::Displaced(0));
    assert_eq!(a.touch(n, 101), IdWrite::Displaced(1));
    assert_eq!(a.node(n).id_count as usize, K, "the list stays full");
    assert_eq!(a.overflows, 2);
    assert!(a.node(n).ids().contains(&100) && a.node(n).ids().contains(&101));
    assert!(!a.node(n).ids().contains(&0), "the displaced id is gone");
}

#[test]
fn dropping_the_last_id_reports_the_node_as_unused() {
    let mut a: Arena<()> = Arena::new(&cfg(4));
    let n = a.insert(1, board(0), 0, 5).unwrap();
    a.touch(n, 6);
    assert!(a.drop_id(n, 5), "5 was listed");
    assert_eq!(a.node(n).ids(), &[6]);
    assert!(!a.is_unused(n), "one id left");
    assert!(a.drop_id(n, 6), "6 was listed");
    assert!(a.is_unused(n), "none left");
    assert!(!a.drop_id(n, 99), "an absent id was never there to drop");
}

#[test]
fn live_slots_lists_exactly_the_live_nodes() {
    let mut a: Arena<()> = Arena::new(&cfg(8));
    let keys: Vec<u64> = (1..=5).collect();
    let slots: Vec<u32> = keys.iter().map(|&k| a.insert(k, board(0), 0, 0).unwrap()).collect();
    a.remove(slots[1]);
    a.remove(slots[3]);
    let live: Vec<u32> = a.live_slots().collect();
    assert_eq!(live, vec![slots[0], slots[2], slots[4]]);
    assert_eq!(live.len(), a.len());
}

#[test]
fn clearing_drops_everything_and_the_arena_is_reusable() {
    let mut a: Arena<()> = Arena::new(&cfg(8));
    for k in 1..=5 {
        a.insert(k, board(0), 0, 0).unwrap();
    }
    a.clear();
    assert_eq!(a.len(), 0);
    assert_eq!(a.live_slots().count(), 0);
    for k in 1..=5 {
        assert_eq!(a.get(k), None, "key {k} survived a clear");
    }
    let n = a.insert(42, board(0), 0, 0).unwrap();
    assert_eq!(a.get(42), Some(n));
}
