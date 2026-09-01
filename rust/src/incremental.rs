//! Incremental, level-based scoring.
//!
//! The reference implementation (`game_kernels::score_player`) rescores a board from
//! scratch after every move: a flood fill from the border plus two full diagonal scans.
//! This module maintains the same score incrementally instead, so a move usually costs a
//! handful of neighbour lookups rather than an O(H*W) rescan.
//!
//! # What is stored
//!
//! Per cell, alongside the board value, a **level**: the number of steps to the nearest
//! border cell, walking only over that player's own marks.
//!
//! * a mark on the border has level 0
//! * any other mark has level `1 + min(level of its own-player diagonal neighbours)`
//! * a mark with no route to the border has level [`INF`]
//!
//! Only playable squares (`(r + c)` even) can hold marks, and orthogonal neighbours always
//! have the opposite parity, so the reference's 8-connected flood fill is exactly
//! 4-connectivity over the diagonals. That is what [`DIAG`] encodes.
//!
//! # Why levels rather than "reachable" bits
//!
//! A single reachable bit cannot survive deletion: a line anchored to the border at both
//! ends, cut in the middle, leaves every surviving cell still reachable, but a bit gives no
//! way to know that without re-running the flood fill. Storing *which neighbour I depend
//! on* does not work either — recording alternate routes puts cycles in the dependency
//! graph, and a cycle is a set of cells that mutually justify each other with nothing
//! underneath.
//!
//! Levels fix this because the provider relation is **derived, not stored**: neighbour `n`
//! is a valid provider for `x` exactly when `level[n] < level[x]`. Strict inequality makes
//! cycles impossible by construction — every mark is held up by something strictly closer
//! to the border, and the chain has to bottom out at the border.
//!
//! # The invariant everything rests on
//!
//! `level[x] != INF` if and only if `x` has a path to the border. It is uniform across a
//! connected component, and therefore uniform along any diagonal run (consecutive run cells
//! are diagonal neighbours). **The score only moves when a cell crosses between INF and
//! finite.** Level changes that stay finite are pure bookkeeping — they cost work and
//! change no score. That is the price of never running a global flood fill.
//!
//! The tests re-derive all of it from scratch after every move and compare.

use crate::config::{HEIGHT, WIDTH};
// Only the board encoding is shared with the reference implementation; none of its kernels
// are, and this module does not know the reference type exists.
use crate::game_kernels::{
    NON_PLAYABLE_SQUARE, PLAYABLE_SQUARE, PLAYER_0_MARK, PLAYER_1_MARK, REMOVED_SQUARE,
};
use crate::rng::Rng;

/// Cells per board.
pub const HW: usize = HEIGHT * WIDTH;

/// "No route to the border." Also acts as the saturating top of the level range.
pub const INF: u8 = 255;

/// The largest level a real path can have: a shortest path visits distinct playable
/// squares, and only half the board is playable. Anything above this is unreachable, which
/// is what stops a severed cycle from climbing forever.
pub const MAX_LEVEL: u8 = (HW / 2) as u8;

const _: () = assert!(
    HW / 2 < INF as usize,
    "board is too large for a u8 level: MAX_LEVEL would collide with INF"
);

/// The four diagonal neighbours. On this board these are the *only* neighbours a mark can
/// have, since orthogonal neighbours are always non-playable parity.
const DIAG: [(i64, i64); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];

/// The two diagonal families that score, as forward steps.
/// Anti-diagonals hold `r + c` constant; main diagonals hold `c - r` constant.
const ANTI: (i64, i64) = (1, -1);
const MAIN: (i64, i64) = (1, 1);
const FAMILIES: [(i64, i64); 2] = [ANTI, MAIN];

#[inline]
fn on_border(i: usize) -> bool {
    let (r, c) = (i / WIDTH, i % WIDTH);
    r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1
}

/// The cell one step `(dr, dc)` from `i`, or `None` if that leaves the board.
#[inline]
fn step(i: usize, d: (i64, i64)) -> Option<usize> {
    let (r, c) = ((i / WIDTH) as i64, (i % WIDTH) as i64);
    let (nr, nc) = (r + d.0, c + d.1);
    if nr < 0 || nr >= HEIGHT as i64 || nc < 0 || nc >= WIDTH as i64 {
        None
    } else {
        Some(nr as usize * WIDTH + nc as usize)
    }
}

/// Is the cell one step `d` from `i` a mark of player `p`?
#[inline]
fn mark_at(cells: &[i8], i: usize, d: (i64, i64), p: i8) -> bool {
    matches!(step(i, d), Some(n) if cells[n] == p)
}

#[inline]
fn player_index(p: i8) -> usize {
    if p == PLAYER_0_MARK {
        0
    } else {
        1
    }
}

// ---------------------------------------------------------------------------- scratch

/// Reusable working memory, so a move allocates nothing.
///
/// `stamp`/`epoch` is a generation-stamped visited set: bumping `epoch` "clears" it in O(1)
/// instead of rewriting 160 bytes.
///
/// `buckets` is a monotone bucket queue keyed by level, which is how both level passes get
/// "process in increasing level order" cheaply. `cursor` only ever moves forward, which is
/// sound because neither pass ever needs to enqueue *below* the level it is currently
/// working on.
pub struct Scratch {
    stamp: Vec<u32>,
    epoch: u32,
    stack: Vec<u16>,
    queue: Vec<u16>,
    comp: Vec<u16>,
    deps: Vec<u16>,
    buckets: Vec<Vec<u16>>,
    cursor: usize,
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Scratch {
    pub fn new() -> Self {
        Scratch {
            stamp: vec![0; HW],
            epoch: 0,
            stack: Vec::with_capacity(HW),
            queue: Vec::with_capacity(HW),
            comp: Vec::with_capacity(HW),
            deps: Vec::with_capacity(4),
            buckets: (0..=INF as usize).map(|_| Vec::new()).collect(),
            cursor: 0,
        }
    }

    /// Start a fresh component walk; every cell reads as unvisited again.
    #[inline]
    fn begin_walk(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // wrapped: the only time we pay for a real clear
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
        self.stack.clear();
    }

    #[inline]
    fn visit(&mut self, i: usize) -> bool {
        if self.stamp[i] == self.epoch {
            false
        } else {
            self.stamp[i] = self.epoch;
            true
        }
    }

    #[inline]
    fn queue_reset(&mut self) {
        debug_assert!(self.buckets.iter().all(|b| b.is_empty()), "queue left dirty");
        self.cursor = 0;
    }

    #[inline]
    fn push(&mut self, cell: usize, level: u8) {
        self.buckets[level as usize].push(cell as u16);
    }

    /// Pop the lowest-level entry, returning the cell and the bucket it came from.
    #[inline]
    fn pop(&mut self) -> Option<(usize, u8)> {
        while self.cursor < self.buckets.len() {
            if let Some(c) = self.buckets[self.cursor].pop() {
                return Some((c as usize, self.cursor as u8));
            }
            self.cursor += 1;
        }
        None
    }
}

// ------------------------------------------------------------------- level maintenance

/// What `i`'s level *should* be, read straight off the invariant.
///
/// `i` itself is not consulted, so this is equally valid for a cell that is about to become
/// a mark and for one that already is.
fn computed_level(cells: &[i8], level: &[u8], i: usize, p: i8) -> u8 {
    if on_border(i) {
        return 0;
    }
    let mut best = INF;
    for d in DIAG {
        if let Some(n) = step(i, d) {
            if cells[n] == p && level[n] < best {
                best = level[n];
            }
        }
    }
    if best == INF {
        INF
    } else if best + 1 > MAX_LEVEL {
        // no real path can be this long, so there is no path at all
        INF
    } else {
        best + 1
    }
}

/// Push improvements outward after an insertion: levels only ever *fall* here.
///
/// Cells that go from INF to finite are exactly the blob being revived; the caller detects
/// that case up front and takes the slow scoring path.
fn relax_down(cells: &[i8], level: &mut [u8], seed: usize, p: i8, s: &mut Scratch) {
    s.queue_reset();
    s.push(seed, level[seed]);
    while let Some((x, bucket)) = s.pop() {
        if level[x] != bucket {
            continue; // stale entry: x was lowered again after this push
        }
        if level[x] >= MAX_LEVEL {
            continue;
        }
        let nl = level[x] + 1;
        for d in DIAG {
            if let Some(n) = step(x, d) {
                if cells[n] == p && level[n] > nl {
                    level[n] = nl;
                    s.push(n, nl);
                }
            }
        }
    }
}

/// Re-derive the levels of one whole component from scratch, instead of walking them up.
///
/// The whole of the slow-removal path's level repair. `comp` must be the component
/// the deleted cell belonged to, collected *before* the deletion — which the slow path
/// already has in hand, because it needs it to score the component anyway.
///
/// Two facts make this a complete answer rather than a local patch:
///
/// * Every level that can change is inside `comp`. A level only changes if its route ran
///   through the deleted cell, and every such route lies inside the component.
/// * Nothing outside `comp` can rescue anything inside it. A mark diagonally adjacent to a
///   mark of the same player is *by definition* in the same component, so `comp` has no
///   neighbouring marks at all. The BFS therefore cannot leak out, and no external ground
///   exists that we would be failing to consider.
///
/// So: blank the component, seed from whichever of its cells sit on the border, and let a
/// plain shortest-path BFS fill the rest. Cost is O(|comp|), bounded by the 80 playable
/// squares, and completely independent of how tangled the levels were before.
///
/// The cap that [`computed_level`] and [`relax_down`] carry is not needed here and never
/// fires: a shortest path visits distinct cells, so no real distance can reach `MAX_LEVEL`
/// in the first place. Anything left at `INF` is genuinely unreachable.
fn rebuild_component_levels(cells: &[i8], level: &mut [u8], comp: &[u16], p: i8, s: &mut Scratch) {
    let mut q = std::mem::take(&mut s.queue);
    q.clear();

    // blank the component, then seed from its border cells
    for &ci in comp {
        let i = ci as usize;
        if cells[i] != p {
            continue; // the cell we just deleted, and anything else no longer a mark
        }
        if on_border(i) {
            level[i] = 0;
        } else {
            level[i] = INF;
        }
    }
    // seeding is a second pass on purpose: the first pass is still blanking cells that a
    // border cell found early would otherwise have been compared against
    for &ci in comp {
        let i = ci as usize;
        if cells[i] == p && level[i] == 0 {
            q.push(ci);
        }
    }

    // A plain FIFO is enough — every step costs exactly one, so cells come off the queue in
    // nondecreasing level order for free. That is also why no cell is ever queued twice:
    // by the time it is reached, the level it gets is already the smallest available.
    let mut head = 0;
    while head < q.len() {
        let x = q[head] as usize;
        head += 1;
        let nl = level[x] + 1;
        for d in DIAG {
            if let Some(n) = step(x, d) {
                if cells[n] == p && level[n] > nl {
                    level[n] = nl;
                    q.push(n as u16);
                }
            }
        }
    }
    s.queue = q;
}

/// Rebuild every level from scratch. Used at construction, after an import, and as the
/// ground truth the incremental passes are checked against. This is the only global BFS.
pub fn rebuild_levels(cells: &[i8], level: &mut [u8], s: &mut Scratch) {
    level.iter_mut().for_each(|l| *l = INF);
    for p in [PLAYER_0_MARK, PLAYER_1_MARK] {
        s.queue_reset();
        for i in 0..HW {
            if cells[i] == p && on_border(i) {
                level[i] = 0;
                s.push(i, 0);
            }
        }
        while let Some((x, bucket)) = s.pop() {
            if level[x] != bucket || level[x] >= MAX_LEVEL {
                continue;
            }
            let nl = level[x] + 1;
            for d in DIAG {
                if let Some(n) = step(x, d) {
                    if cells[n] == p && level[n] > nl {
                        level[n] = nl;
                        s.push(n, nl);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------- component walk

/// Collect every cell reachable from `seed` through diagonal steps on `p`'s marks.
///
/// Appends to `out` and honours the current walk epoch, so several seeds can be unioned
/// into one buffer without double-counting.
fn walk_component(cells: &[i8], seed: usize, p: i8, s: &mut Scratch, out: &mut Vec<u16>) {
    if cells[seed] != p || !s.visit(seed) {
        return;
    }
    s.stack.clear();
    s.stack.push(seed as u16);
    out.push(seed as u16);
    while let Some(x) = s.stack.pop() {
        for d in DIAG {
            if let Some(n) = step(x as usize, d) {
                if cells[n] == p && s.visit(n) {
                    s.stack.push(n as u16);
                    out.push(n as u16);
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------ scoring

/// Total score generated by the given cells.
///
/// `cells_of_interest` must be a union of *whole* components. That matters: a run never
/// leaves its component (consecutive run cells are diagonal neighbours), so every run is
/// either wholly inside the set or wholly outside it, never half — which is what makes it
/// valid to score the set in isolation.
///
/// Each run is measured exactly once, from its first cell: a cell whose backward neighbour
/// along that diagonal is also a mark is not a run start and is skipped. Runs shorter than
/// 2 score nothing; a qualifying run scores its full length, but only if it is reachable.
/// Reachability is uniform along a run, so testing the start cell is enough. A cell that is
/// in a qualifying run of *both* families is counted twice — that is intended, and matches
/// the reference.
fn contribution(cells: &[i8], level: &[u8], cells_of_interest: &[u16], p: i8) -> i32 {
    cells_of_interest
        .iter()
        .map(|&ci| runs_starting_at(cells, level, ci as usize, p))
        .sum()
}

/// The same scorer applied to every square, which is the whole board's score for `p`.
///
/// Only used when adopting a board the engine did not build itself; the running score is
/// maintained incrementally everywhere else.
fn score_board(cells: &[i8], level: &[u8], p: i8) -> i32 {
    (0..HW).map(|i| runs_starting_at(cells, level, i, p)).sum()
}

/// Score of the runs that *begin* at `i`, in either family. Zero if `i` holds no mark of
/// `p`, if the run continues backwards through `i` (so some earlier cell owns it), if the
/// run is shorter than 2, or if it cannot reach the border.
fn runs_starting_at(cells: &[i8], level: &[u8], i: usize, p: i8) -> i32 {
    if cells[i] != p {
        return 0; // empty, removed, the other player's, or a cell dropped since collection
    }
    let mut total = 0i32;
    for v in FAMILIES {
        let back = (-v.0, -v.1);
        if mark_at(cells, i, back, p) {
            continue; // not the first cell of this run
        }
        let mut run_len = 1i32;
        let mut cur = i;
        while let Some(n) = step(cur, v) {
            if cells[n] != p {
                break;
            }
            run_len += 1;
            cur = n;
        }
        if run_len >= 2 && level[i] != INF {
            total += run_len;
        }
    }
    total
}

fn run_side(cells: &[i8], i: usize, v: (i64, i64), p: i8) -> i32 {
    match step(i, v) {
        Some(a) if cells[a] == p => match step(a, v) {
            Some(b) if cells[b] == p => 2,
            _ => 1,
        },
        _ => 0,
    }
}

/// The score change from filling (or emptying) the single square `i`, given that
/// reachability does not change.
///
/// A run of length L scores L if L >= 2, else 0. Filling the gap between runs of length
/// `la` and `lb` therefore gains `(la + 1 + lb) - [la if la>=2] - [lb if lb>=2]`, which
/// collapses to `1 + (la == 1) + (lb == 1)` — one point for extending a run at all, plus
/// one more for each side that was a lone mark, since a lone mark scored nothing before.
/// Both sides empty means a run of length 1, which scores nothing.
#[inline]
fn local_delta(cells: &[i8], i: usize, p: i8) -> i32 {
    let mut d = 0;
    for v in FAMILIES {
        let back = (-v.0, -v.1);
        let la = run_side(cells, i, back, p);
        let lb = run_side(cells, i, v, p);
        if la > 0 || lb > 0 {
            d += 1 + (la == 1) as i32 + (lb == 1) as i32;
        }
    }
    d
}

// ---------------------------------------------------------------------------- insertion

/// Place `p`'s mark on the empty square `i`, updating levels and the running score.
pub fn insert(cells: &mut [i8], level: &mut [u8], score: &mut [i32], i: usize, p: i8, s: &mut Scratch) {
    debug_assert_eq!(cells[i], PLAYABLE_SQUARE, "insert onto a non-playable square");

    let lvl = computed_level(cells, level, i, p);

    if lvl == INF {
        // every mark it touches was already unreachable, so nothing scored before and
        // nothing scores now
        cells[i] = p;
        level[i] = INF;
        return;
    }

    let attaches_dead = DIAG
        .iter()
        .filter_map(|&d| step(i, d))
        .any(|n| cells[n] == p && level[n] == INF);

    if !attaches_dead {
        // Fast path, and the common one: a mark dropped into live territory. Reachability
        // is unchanged everywhere, so the whole score change is local to the two runs
        // through this square.
        cells[i] = p;
        level[i] = lvl;
        relax_down(cells, level, i, p, s);
        score[player_index(p)] += local_delta(cells, i, p);
        return;
    }

    // Slow path: this mark revives a dead blob, so a whole region flips from scoring
    // nothing to scoring everything. There is no local shortcut — measure before, measure
    // after, apply the difference.
    let mut comp = std::mem::take(&mut s.comp);
    comp.clear();
    s.begin_walk();
    for d in DIAG {
        if let Some(n) = step(i, d) {
            // dead components contribute zero by definition, so only the live ones matter
            if cells[n] == p && level[n] != INF {
                walk_component(cells, n, p, s, &mut comp);
            }
        }
    }
    let old = contribution(cells, level, &comp, p);

    cells[i] = p;
    level[i] = lvl;
    relax_down(cells, level, i, p, s);

    comp.clear();
    s.begin_walk();
    walk_component(cells, i, p, s, &mut comp);
    let new = contribution(cells, level, &comp, p);
    s.comp = comp;

    score[player_index(p)] += new - old;
}

// ----------------------------------------------------------------------------- removal

/// Clear square `i`, updating levels and the running score. Handles empty and already
/// removed squares as no-ops beyond the state write.
pub fn remove(cells: &mut [i8], level: &mut [u8], score: &mut [i32], i: usize, s: &mut Scratch) {
    let p = cells[i];
    if p != PLAYER_0_MARK && p != PLAYER_1_MARK {
        cells[i] = REMOVED_SQUARE;
        return;
    }

    if level[i] == INF {
        // its component scored nothing, so losing a cell changes nothing
        cells[i] = REMOVED_SQUARE;
        level[i] = INF;
        return;
    }

    // Only a neighbour sitting exactly one level above could have been routing through
    // this cell. If there are none, no level anywhere changes and nothing is cut off.
    let here = level[i];
    let mut deps = std::mem::take(&mut s.deps);
    deps.clear();
    for d in DIAG {
        if let Some(n) = step(i, d) {
            if cells[n] == p && level[n] == here + 1 {
                deps.push(n as u16);
            }
        }
    }

    if deps.is_empty() {
        // Fast path: same local calculation as insertion, negated. The flanking runs keep
        // their levels, so they stay reachable and their contributions still count.
        cells[i] = REMOVED_SQUARE;
        level[i] = INF;
        score[player_index(p)] -= local_delta(cells, i, p);
        s.deps = deps;
        return;
    }

    // Slow path: the structure may have been cut. Measure while it is still whole.
    let mut comp = std::mem::take(&mut s.comp);
    comp.clear();
    s.begin_walk();
    walk_component(cells, i, p, s, &mut comp);
    let old = contribution(cells, level, &comp, p);

    cells[i] = REMOVED_SQUARE;
    level[i] = INF;
    rebuild_component_levels(cells, level, &comp, p, s);

    // `comp` may now be several disconnected pieces, some alive, some INF. `contribution`
    // handles that on its own: a piece on INF simply contributes nothing, and the removed
    // cell is skipped because it no longer holds a mark.
    let new = contribution(cells, level, &comp, p);

    s.comp = comp;
    s.deps = deps;
    score[player_index(p)] += new - old;
}

/// Apply one move for one game, mirroring the reference's `apply_and_score_kernel`.
///
/// A collision (both players choosing the same square) clears that square and its four
/// diagonal neighbours. Those five are handled as independent single-cell removals: the
/// four neighbours are mutually non-adjacent, and the two sharing a diagonal have the empty
/// centre between them, so they lie in different runs and the order cannot matter. Each
/// victim is handled under whichever player happens to own it.
#[allow(clippy::too_many_arguments)]
pub fn apply_move(
    cells: &mut [i8],
    level: &mut [u8],
    score: &mut [i32],
    legal: &mut [u64],
    (r0, c0): (usize, usize),
    (r1, c1): (usize, usize),
    s: &mut Scratch,
) {
    if (r0, c0) == (r1, c1) {
        let centre = r0 * WIDTH + c0;
        clear_legal(legal, centre);
        remove(cells, level, score, centre, s);
        for d in DIAG {
            if let Some(n) = step(centre, d) {
                clear_legal(legal, n);
                remove(cells, level, score, n, s);
            }
        }
    } else {
        let (i0, i1) = (r0 * WIDTH + c0, r1 * WIDTH + c1);
        clear_legal(legal, i0);
        clear_legal(legal, i1);
        insert(cells, level, score, i0, PLAYER_0_MARK, s);
        insert(cells, level, score, i1, PLAYER_1_MARK, s);
    }
}

// -------------------------------------------------------------------------- validation

// ------------------------------------------------------------------- legality bitboard

/// Which squares are still playable, as a bitboard: two `u64` per game.
///
/// Only the 80 squares with `(r + c)` even are ever playable, and `i / 2` maps exactly those
/// onto `0..80` densely *and in row-major order* — row `r`'s playable columns are
/// `2j + (r & 1)`, so `i / 2 == r * 8 + j`. The whole legal set of a game is therefore 80
/// bits, 16 bytes, against the 1280 bytes a pair of f32 masks needs. A 2048-game batch's
/// legality is 32 KB and lives in L2; the mask form is 2.6 MB and does not.
///
/// Row-major order is not incidental: the sampler's cumulative scan picks the first square
/// whose running total crosses the threshold, so traversal order is part of the result.
/// Walking bits low to high reproduces the reference kernel's scan exactly.
///
/// The set only ever shrinks and every internal board change goes through a move, so this is
/// maintained rather than recomputed: one bit cleared per square a move consumes.
pub const LEGAL_WORDS: usize = 2;

const _: () = assert!(HW / 2 == 80, "the legality bitboard assumes 80 playable squares");

/// The 80 valid bits — the opening position.
const LEGAL_ALL: [u64; LEGAL_WORDS] = [!0u64, 0xFFFF];

/// The opening-move half-board rule, precomputed. `c < half_width` is `2j + (r & 1) < 8`,
/// which is `j < 4` for both row parities — so the rule is the low nibble of every byte and
/// applying it is one `AND`, not a column test per square.
const LEGAL_LEFT: [u64; LEGAL_WORDS] = [0x0F0F_0F0F_0F0F_0F0F, 0x0F0F];
const LEGAL_RIGHT: [u64; LEGAL_WORDS] = [0xF0F0_F0F0_F0F0_F0F0, 0xF0F0];

/// Word and bit for a board index. A non-playable index aliases onto its even neighbour,
/// which is harmless because its bit is never set in the first place.
#[inline]
fn legal_bit(i: usize) -> (usize, u64) {
    let k = i >> 1;
    (k >> 6, 1u64 << (k & 63))
}

/// The board index a bit position stands for; the inverse of `i >> 1` over playable cells.
///
/// It reduces to two instructions. Position `k` is `r * 8 + j`, and the cell it names is
/// `r * 16 + 2j + (r & 1)` — and `r * 16 + 2j` is exactly `2k`, so all that is left is the
/// row's parity, which is bit 3 of `k`.
#[inline]
fn legal_cell(k: usize) -> usize {
    (k << 1) | ((k >> 3) & 1)
}

/// Mark a square as no longer playable. Called for every square a move writes, whether it
/// took a mark or was blasted away, which is exactly the set that stops being playable.
#[inline]
fn clear_legal(legal: &mut [u64], i: usize) {
    let (w, b) = legal_bit(i);
    legal[w] &= !b;
}

/// Sample one move per active game from `dist`, restricted to the still-playable squares
/// plus the first-move half-board rule.
///
/// This replaces `game_kernels::sample_move_kernel` for the incremental engine. Same
/// distribution, same RNG draws in the same order — but instead of walking all 160 squares
/// twice against a materialized f32 mask, it walks only the set bits, and the half-board
/// rule is folded into the words before the loop rather than tested per square.
#[allow(clippy::too_many_arguments)]
fn sample_bits(
    dist: &[f32],
    legal: &[u64],
    active: &[bool],
    move_counts: &[i32],
    n: usize,
    player: usize,
    rng: &mut Rng,
    r_out: &mut [i64],
    c_out: &mut [i64],
) {
    for g in 0..n {
        r_out[g] = 0;
        c_out[g] = 0;
        if !active[g] {
            continue;
        }
        let base = g * HW;
        let half = if move_counts[g] == 0 {
            if player == 0 {
                LEGAL_LEFT
            } else {
                LEGAL_RIGHT
            }
        } else {
            LEGAL_ALL
        };
        let words = [
            legal[g * LEGAL_WORDS] & half[0],
            legal[g * LEGAL_WORDS + 1] & half[1],
        ];

        // Two words, unrolled: the second holds only 16 bits and is usually empty by the
        // midgame, so a loop over them would spend its time on the branch, not the work.
        let mut total: f64 = 0.0;
        let mut w = words[0];
        while w != 0 {
            let k = w.trailing_zeros() as usize;
            w &= w - 1;
            total += dist[base + legal_cell(k)] as f64;
        }
        let mut w = words[1];
        while w != 0 {
            let k = 64 + w.trailing_zeros() as usize;
            w &= w - 1;
            total += dist[base + legal_cell(k)] as f64;
        }

        // Same degenerate branch as the reference kernel: an unmasked uniform draw over the
        // whole board. Kept so the RNG is consumed identically.
        if total < 1e-8 {
            let idx = rng.randint(HW as u64) as i64;
            r_out[g] = idx / WIDTH as i64;
            c_out[g] = idx % WIDTH as i64;
            continue;
        }

        let threshold = rng.random() * total;
        let mut cum: f64 = 0.0;
        let mut chosen: usize = 0;
        'scan: for (wi, &word) in words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let k = (wi << 6) + w.trailing_zeros() as usize;
                w &= w - 1;
                let cell = legal_cell(k);
                let v = dist[base + cell] as f64;
                if v > 0.0 {
                    cum += v;
                    if cum >= threshold {
                        chosen = cell;
                        break 'scan;
                    }
                }
            }
        }
        r_out[g] = (chosen / WIDTH) as i64;
        c_out[g] = (chosen % WIDTH) as i64;
    }
}

// ------------------------------------------------------------------------ batched game


/// A batch of games scored incrementally.
///
/// Everything that does not involve scoring — the legal-move masks, the sampler, the
/// first-move rule, `move_counts`, `finished` — is the reference code, reused unchanged.
/// The only difference from [`BatchedLinesGame`] is that the board is scored by the level
/// machinery above instead of by a from-scratch rescan after every move.
pub struct IncrementalGame {
    pub n: usize,
    /// (N, H, W) row-major, same encoding as the reference.
    pub boards: Vec<i8>,
    /// (N, H, W) row-major levels, parallel to `boards`.
    pub levels: Vec<u8>,
    /// (N, 2) running score. Integer because every score is a sum of run lengths.
    pub scores: Vec<i32>,
    pub move_counts: Vec<i32>,
    pub finished: Vec<bool>,
    pub half_width: usize,
    pub rng: Rng,
    /// Squares still playable, as `[g * LEGAL_WORDS ..][.. LEGAL_WORDS]`. This is what the
    /// sampler walks instead of a mask, what `get_legal_masks` expands from, and what makes
    /// the finished check two comparisons instead of a board scan.
    legal: Vec<u64>,
    scratch: Scratch,
}

impl IncrementalGame {
    /// Derive one game's legality bitboard from its board. Only needed when adopting a
    /// position from outside; during play the bitboard is maintained, never rebuilt.
    fn rebuild_legal(&mut self, g: usize) {
        let cells = &self.boards[g * HW..(g + 1) * HW];
        let mut w = [0u64; LEGAL_WORDS];
        for i in 0..HW {
            if cells[i] == PLAYABLE_SQUARE {
                let (wi, b) = legal_bit(i);
                w[wi] |= b;
            }
        }
        self.legal[g * LEGAL_WORDS..(g + 1) * LEGAL_WORDS].copy_from_slice(&w);
    }

    /// An empty bitboard is exactly "no playable square left".
    #[inline]
    fn no_moves_left(&self, g: usize) -> bool {
        self.legal[g * LEGAL_WORDS] == 0 && self.legal[g * LEGAL_WORDS + 1] == 0
    }

    /// This game's legality words. The cheap form of [`Self::get_legal_masks`] for a caller
    /// that can consume 16 bytes instead of 1280 — count is `count_ones`, a legality check
    /// is a bit test, and the opening-move restriction is an `AND` with `LEGAL_LEFT` or
    /// `LEGAL_RIGHT`.
    pub fn legal_bits(&self, g: usize) -> [u64; LEGAL_WORDS] {
        [self.legal[g * LEGAL_WORDS], self.legal[g * LEGAL_WORDS + 1]]
    }
    pub fn new(num_games: usize, seed: u64) -> Self {
        let mut boards = vec![NON_PLAYABLE_SQUARE; num_games * HW];
        for g in 0..num_games {
            for r in 0..HEIGHT {
                for c in 0..WIDTH {
                    if (r + c) % 2 == 0 {
                        boards[g * HW + r * WIDTH + c] = PLAYABLE_SQUARE;
                    }
                }
            }
        }
        let mut g = IncrementalGame {
            n: num_games,
            boards,
            levels: vec![INF; num_games * HW],
            scores: vec![0; num_games * 2],
            move_counts: vec![0; num_games],
            finished: vec![false; num_games],
            half_width: WIDTH / 2,
            rng: Rng::new(seed),
            legal: vec![0; num_games * LEGAL_WORDS],
            scratch: Scratch::new(),
        };
        for i in 0..num_games {
            g.rebuild_legal(i);
        }
        g
    }

    /// Adopt arbitrary boards. The board *is* the state, so nothing else is asked for.
    ///
    /// Levels come from the one global BFS, scores from a full rescan, legality from a scan
    /// for playable squares. The two flags the engine also carries are derived rather than
    /// supplied: a game is finished exactly when no playable square is left, and it is on
    /// its opening move exactly when nothing has been played — no marks and no blasted
    /// squares. `move_counts` is only ever compared against zero, never counted with, so
    /// 0-or-1 carries everything that depends on it; this is the same rule the reference
    /// applies when it imports a printed board.
    pub fn from_state(boards: Vec<i8>, seed: u64) -> Self {
        let n = boards.len() / HW;
        let mut move_counts = vec![0i32; n];
        let mut finished = vec![false; n];
        for i in 0..n {
            let cells = &boards[i * HW..(i + 1) * HW];
            let played = cells
                .iter()
                .any(|&v| v != PLAYABLE_SQUARE && v != NON_PLAYABLE_SQUARE);
            move_counts[i] = i32::from(played);
            finished[i] = !cells.iter().any(|&v| v == PLAYABLE_SQUARE);
        }
        let mut g = IncrementalGame {
            n,
            boards,
            levels: vec![INF; n * HW],
            scores: vec![0; n * 2],
            move_counts,
            finished,
            half_width: WIDTH / 2,
            rng: Rng::new(seed),
            legal: vec![0; n * LEGAL_WORDS],
            scratch: Scratch::new(),
        };
        for i in 0..n {
            g.rebuild_legal(i);
        }
        for i in 0..n {
            let cells = &g.boards[i * HW..(i + 1) * HW];
            let level = &mut g.levels[i * HW..(i + 1) * HW];
            rebuild_levels(cells, level, &mut g.scratch);
            g.scores[i * 2] = score_board(cells, level, PLAYER_0_MARK);
            g.scores[i * 2 + 1] = score_board(cells, level, PLAYER_1_MARK);
        }
        g
    }

    fn active(&self) -> Vec<bool> {
        self.finished.iter().map(|f| !f).collect()
    }

    /// Apply one sampled move per active game. Private: [`Self::distribution_step`] and
    /// [`Self::action_step`] are the two ways in, and they differ only in where the moves
    /// come from.
    fn apply_step(&mut self, r0: &[i64], c0: &[i64], r1: &[i64], c1: &[i64], active: &[bool]) {
        for g in 0..self.n {
            if !active[g] {
                continue;
            }
            let cells = &mut self.boards[g * HW..(g + 1) * HW];
            let level = &mut self.levels[g * HW..(g + 1) * HW];
            let score = &mut self.scores[g * 2..g * 2 + 2];
            let legal = &mut self.legal[g * LEGAL_WORDS..(g + 1) * LEGAL_WORDS];
            apply_move(
                cells,
                level,
                score,
                legal,
                (r0[g] as usize, c0[g] as usize),
                (r1[g] as usize, c1[g] as usize),
                &mut self.scratch,
            );
            self.move_counts[g] += 1;
            if self.no_moves_left(g) {
                self.finished[g] = true;
            }
        }
    }

    /// Copy the named games into a fresh batch.
    ///
    /// Every derived quantity is copied rather than recomputed — levels and the legality
    /// bitboard come across verbatim — so a clone costs a memcpy per game instead of the
    /// global BFS and full rescore [`Self::from_state`] would pay for the same boards.
    /// Naming a game twice is allowed and yields two independent copies.
    ///
    /// The clone draws a fresh RNG from the parent's stream, so a seeded parent still
    /// determines everything its clones go on to do.
    pub fn clone_states_to_batch(&mut self, indices: &[usize]) -> Self {
        let mut t = IncrementalGame::new(indices.len(), self.rng.next_u64());
        for (d, &g) in indices.iter().enumerate() {
            t.boards[d * HW..(d + 1) * HW].copy_from_slice(&self.boards[g * HW..(g + 1) * HW]);
            t.levels[d * HW..(d + 1) * HW].copy_from_slice(&self.levels[g * HW..(g + 1) * HW]);
            t.legal[d * LEGAL_WORDS..(d + 1) * LEGAL_WORDS]
                .copy_from_slice(&self.legal[g * LEGAL_WORDS..(g + 1) * LEGAL_WORDS]);
            t.scores[d * 2] = self.scores[g * 2];
            t.scores[d * 2 + 1] = self.scores[g * 2 + 1];
            t.move_counts[d] = self.move_counts[g];
            t.finished[d] = self.finished[g];
        }
        t
    }

    /// Sample one move per active game for `player`, without applying it.
    ///
    /// Exposed so a caller can measure or reuse the sampler on its own;
    /// [`Self::distribution_step`] is this twice plus [`Self::apply_step`].
    pub fn sample_moves(
        &mut self,
        dist: &[f32],
        player: usize,
        active: &[bool],
        r_out: &mut [i64],
        c_out: &mut [i64],
    ) {
        sample_bits(
            dist, &self.legal, active, &self.move_counts,
            self.n, player, &mut self.rng, r_out, c_out,
        );
    }

    /// One move for all active games, sampled from `dist_p0`/`dist_p1`.
    ///
    /// Unlike the reference, this never materializes the (N, H, W) legal masks: the live
    /// list already says which squares are available, so the mask kernel — 2.6 MB of writes
    /// per step at 2048 games — is skipped entirely.
    pub fn distribution_step(&mut self, dist_p0: &[f32], dist_p1: &[f32]) {
        let active = self.active();
        if !active.iter().any(|&a| a) {
            return;
        }
        let n = self.n;
        let mut r0 = vec![0i64; n];
        let mut c0 = vec![0i64; n];
        let mut r1 = vec![0i64; n];
        let mut c1 = vec![0i64; n];
        self.sample_moves(dist_p0, 0, &active, &mut r0, &mut c0);
        self.sample_moves(dist_p1, 1, &active, &mut r1, &mut c1);
        self.apply_step(&r0, &c0, &r1, &c1, &active);
    }

    pub fn action_step(&mut self, idx_0: &[i64], idx_1: &[i64]) -> Result<(), String> {
        let active = self.active();
        if !active.iter().any(|&a| a) {
            return Ok(());
        }
        // The reference builds both full masks just to read two entries per game. The same
        // predicate reads straight off the board: a square is legal if it is still playable
        // and, on the opening move, in the half this player is confined to.
        let half = self.half_width;
        let legal = |cells: &[i8], idx: usize, first: bool, player: usize| -> bool {
            if cells[idx] != PLAYABLE_SQUARE {
                return false;
            }
            if !first {
                return true;
            }
            let c = idx % WIDTH;
            if player == 0 { c < half } else { c >= half }
        };

        let mut invalid_indices: Vec<usize> = Vec::new();
        for g in 0..self.n {
            if !active[g] {
                continue;
            }
            let cells = &self.boards[g * HW..(g + 1) * HW];
            let first = self.move_counts[g] == 0;
            if !legal(cells, idx_0[g] as usize, first, 0)
                || !legal(cells, idx_1[g] as usize, first, 1)
            {
                invalid_indices.push(g);
            }
        }
        if !invalid_indices.is_empty() {
            return Err(format!(
                "Invalid move detected in batch at game indices: {:?}. \
                 Execution aborted; no games updated.",
                invalid_indices
            ));
        }

        let w = WIDTH as i64;
        let r0: Vec<i64> = idx_0.iter().map(|&i| i / w).collect();
        let c0: Vec<i64> = idx_0.iter().map(|&i| i % w).collect();
        let r1: Vec<i64> = idx_1.iter().map(|&i| i / w).collect();
        let c1: Vec<i64> = idx_1.iter().map(|&i| i % w).collect();
        self.apply_step(&r0, &c0, &r1, &c1, &active);
        Ok(())
    }
}
