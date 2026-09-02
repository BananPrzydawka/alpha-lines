//! The incremental hash must always equal a from-scratch hash of the board it describes,
//! and equal boards must hash equal whatever order they were reached in.

use alpha_lines_game::game::{Rng, Scratch, LEGAL_WORDS, SQUARES};
use alpha_lines_game::{zobrist, Game};

/// A uniformly random legal square for `player`.
fn uniform_move(game: &Game, player: usize, rng: &mut Rng) -> usize {
    let w: [u64; LEGAL_WORDS] = game.legal_moves(player);
    let count = w[0].count_ones() + w[1].count_ones();
    assert!(count > 0, "no legal move for player {player}");
    let mut k = rng.randint(count as u64) as u32;
    for (wi, &word) in w.iter().enumerate() {
        let c = word.count_ones();
        if k < c {
            let mut x = word;
            for _ in 0..k {
                x &= x - 1;
            }
            return (wi << 6) + x.trailing_zeros() as usize;
        }
        k -= c;
    }
    unreachable!("select past the end of the legal set")
}

#[test]
fn the_opening_position_hashes_to_the_documented_constant() {
    assert_eq!(zobrist::hash(&Game::new().cells), zobrist::OPENING);
}

/// The whole contract, checked move by move: stepping the hash has to land exactly where
/// rehashing the resulting board lands, on every move of a lot of games.
///
/// Collisions are the only case `step` reads the board at all, and free play produces them
/// on ~7% of moves; every sixth step is nudged on top of that so the five-square path is
/// exercised hard rather than incidentally.
#[test]
fn stepping_the_hash_matches_rehashing_the_board() {
    let games: usize = std::env::var("PARITY_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000);
    let mut rng = Rng::new(0x2013_1257);
    let mut scratch = Scratch::new();
    let (mut moves, mut collisions) = (0usize, 0usize);

    for _ in 0..games {
        let mut game = Game::new();
        let mut h = zobrist::OPENING;
        let mut step = 0usize;

        while !game.finished {
            let i0 = uniform_move(&game, 0, &mut rng);
            let mut i1 = uniform_move(&game, 1, &mut rng);
            // force a collision where the square is legal for both, which it never is on the
            // opening move because the halves are disjoint
            if step % 6 == 0 {
                let w = game.legal_moves(1);
                if w[i0 >> 6] >> (i0 & 63) & 1 == 1 {
                    i1 = i0;
                }
            }

            h = zobrist::step(h, &game.cells, i0, i1);
            game.action_step(i0, i1, &mut scratch);
            assert_eq!(
                h,
                zobrist::hash(&game.cells),
                "hash drifted at step {step} (move {i0}, {i1})"
            );

            moves += 1;
            collisions += usize::from(i0 == i1);
            step += 1;
        }
        assert_eq!(h, zobrist::hash(&game.cells), "hash of the finished board");
    }

    assert!(collisions * 6 > moves, "only {collisions} collisions in {moves} moves");
    println!("{games} games, {moves} moves, {collisions} collisions, hash never drifted");
}

/// The reason the hash exists: two move orders reaching the same board are one node.
#[test]
fn positions_reached_by_different_move_orders_hash_the_same() {
    let mut rng = Rng::new(0x7A05);
    let mut scratch = Scratch::new();
    let mut checked = 0;

    for trial in 0..400 {
        // a shared prefix, so the opening half-board rule is behind both branches and each
        // of the two moves below is legal in either order
        let mut root = Game::new();
        let mut h = zobrist::OPENING;
        for _ in 0..(trial % 7 + 1) {
            let (i0, i1) = (uniform_move(&root, 0, &mut rng), uniform_move(&root, 1, &mut rng));
            h = zobrist::step(h, &root.cells, i0, i1);
            root.action_step(i0, i1, &mut scratch);
        }
        if root.finished {
            continue;
        }

        // two moves that do not touch each other, so playing them in either order lands on
        // the same board
        let a = (uniform_move(&root, 0, &mut rng), uniform_move(&root, 1, &mut rng));
        let b = (uniform_move(&root, 0, &mut rng), uniform_move(&root, 1, &mut rng));
        let touched = [a.0, a.1, b.0, b.1];
        let distinct = touched.iter().all(|&x| touched.iter().filter(|&&y| y == x).count() == 1);
        if !distinct {
            continue; // overlapping or colliding moves genuinely do not commute
        }

        let mut first = root.clone();
        let mut hf = zobrist::step(h, &first.cells, a.0, a.1);
        first.action_step(a.0, a.1, &mut scratch);
        hf = zobrist::step(hf, &first.cells, b.0, b.1);
        first.action_step(b.0, b.1, &mut scratch);

        let mut second = root.clone();
        let mut hs = zobrist::step(h, &second.cells, b.0, b.1);
        second.action_step(b.0, b.1, &mut scratch);
        hs = zobrist::step(hs, &second.cells, a.0, a.1);
        second.action_step(a.0, a.1, &mut scratch);

        assert_eq!(first.cells, second.cells, "the two orders did not transpose");
        assert_eq!(hf, hs, "a transposition hashed to two different keys");
        assert_eq!(hf, zobrist::hash(&first.cells), "and neither matched a rehash");
        checked += 1;
    }
    assert!(checked > 200, "only {checked} transpositions tested");
    println!("{checked} transpositions hashed identically");
}

/// Different boards must hash differently. Not a proof — 64 bits cannot promise it — but a
/// duplicate over this many positions would mean the keys are not doing their job.
#[test]
fn distinct_positions_get_distinct_hashes() {
    use std::collections::HashMap;

    let mut rng = Rng::new(0xD15C);
    let mut scratch = Scratch::new();
    let mut seen: HashMap<u64, [i8; SQUARES]> = HashMap::new();
    let mut equal = 0usize;

    for _ in 0..3_000 {
        let mut game = Game::new();
        let mut h = zobrist::OPENING;
        while !game.finished {
            let (i0, i1) = (uniform_move(&game, 0, &mut rng), uniform_move(&game, 1, &mut rng));
            h = zobrist::step(h, &game.cells, i0, i1);
            game.action_step(i0, i1, &mut scratch);
            match seen.get(&h) {
                Some(prev) => {
                    assert_eq!(*prev, game.cells, "two different boards share a hash");
                    equal += 1;
                }
                None => {
                    seen.insert(h, game.cells);
                }
            }
        }
    }
    println!("{} distinct positions, {equal} repeats, no collisions", seen.len());
    assert!(seen.len() > 100_000, "only {} positions visited", seen.len());
}
