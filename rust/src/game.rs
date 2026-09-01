//! The game: one board, scored incrementally.
//!
//! [`Game`] is the whole public surface — a position, the moves that are legal in it, and
//! the two ways to pick one. Everything above it in this file is the scorer it is built on.
//!
//! The reference implementation this replaced (`game_kernels::score_player`) rescores a
//! board from scratch after every move: a flood fill from the border plus two full diagonal
//! scans. This module maintains the same score incrementally instead, so a move usually
//! costs a handful of neighbour lookups rather than an O(H*W) rescan.
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

// This module has no dependencies on the rest of the crate: the board shape, the square
// encoding and the RNG are all defined here, so the file can be lifted out whole. They are
// duplicated in the reference port under `tests/`; the tests assert the two definitions
// still agree, so they cannot drift apart while both exist.

/// Board shape. Every square with `(r + c)` even is playable — 80 of the 160.
pub const HEIGHT: usize = 10;
pub const WIDTH: usize = 16;

/// Cells per board.
pub const HW: usize = HEIGHT * WIDTH;

/// Square encoding. A square is one of these five things and nothing else.
pub const NON_PLAYABLE_SQUARE: i8 = 0;
pub const PLAYABLE_SQUARE: i8 = 1;
pub const REMOVED_SQUARE: i8 = 2;
pub const PLAYER_0_MARK: i8 = 3;
pub const PLAYER_1_MARK: i8 = 4;

/// "No route to the border." Also acts as the saturating top of the level range.
pub const INF: u8 = 255;

/// The largest level a real path can have.
///
/// A level counts steps through *one player's* marks, and a shortest path visits each cell
/// at most once, so the bound is that player's mark count. Half the board is playable — 80
/// squares — and every non-colliding move gives both players exactly one mark, so neither
/// can ever hold more than half of those. Collisions only take marks away. So a player has
/// at most `HW / 4` = 40 marks and no real level can exceed 39.
///
/// Nothing legitimate is ever capped by this; it is a bound, not a mechanism. It used to be
/// load-bearing, back when a removal repaired levels by walking them upward and a severed
/// cycle would climb in lockstep until something stopped it. The slow-removal path now
/// rebuilds its component with a BFS, which assigns true distances and terminates on its
/// own, so the cap only guards against a level running away somewhere it cannot.
pub const MAX_LEVEL: u8 = (HW / 4) as u8;

const _: () = assert!(
    HW / 4 < INF as usize,
    "board is too large for a u8 level: MAX_LEVEL would collide with INF"
);

// ---------------------------------------------------------------------------------- rng

/// Seedable PRNG for the sampler, standing in for numba's hidden `np.random` state.
///
/// numba's per-thread Mersenne Twister cannot be reproduced bit-for-bit, so the port keeps
/// the *structure* of the draws identical — same count, same order, same places — and swaps
/// the bit source. Nothing the cross-check against Python compares depends on these draws.
/// xoshiro256++, seeded through SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// `np.random.random()`: a double in [0, 1) with 53 bits of entropy.
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `np.random.randint(0, n)`: a uniform integer in [0, n), debiased (Lemire).
    #[inline]
    pub fn randint(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randint bound must be positive");
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let threshold = n.wrapping_neg() % n;
            while l < threshold {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

// -------------------------------------------------------------------------------- geometry

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
/// `buckets` is a monotone bucket queue keyed by level, sized to the level range rather
/// than to `INF`, since nothing finite is ever queued above `MAX_LEVEL`. That also bounds
/// how far `pop` can advance the cursor looking for the next non-empty bucket.
/// It is how the outward relaxation gets
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
            buckets: (0..=MAX_LEVEL as usize).map(|_| Vec::new()).collect(),
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
        debug_assert!(level <= MAX_LEVEL, "queued a level above the cap: {level}");
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
fn insert(cells: &mut [i8], level: &mut [u8], score: &mut [i32], i: usize, p: i8, s: &mut Scratch) {
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
fn remove(cells: &mut [i8], level: &mut [u8], score: &mut [i32], i: usize, s: &mut Scratch) {
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
fn apply_move(
    cells: &mut [i8],
    level: &mut [u8],
    score: &mut [i32],
    legal: &mut [u64],
    i0: usize,
    i1: usize,
    s: &mut Scratch,
) {
    if i0 == i1 {
        let centre = i0;
        clear_legal(legal, centre);
        remove(cells, level, score, centre, s);
        for d in DIAG {
            if let Some(n) = step(centre, d) {
                clear_legal(legal, n);
                remove(cells, level, score, n, s);
            }
        }
    } else {
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
/// Public because it is the decoder for [`Game::legal_moves`]: the mask is 80 bit positions,
/// and without this it says nothing about which squares they are. Walking a mask is
/// `while w != 0 { let cell = legal_cell(base + w.trailing_zeros() as usize); w &= w - 1; }`.
///
/// It reduces to two instructions. Position `k` is `r * 8 + j`, and the cell it names is
/// `r * 16 + 2j + (r & 1)` — and `r * 16 + 2j` is exactly `2k`, so all that is left is the
/// row's parity, which is bit 3 of `k`.
#[inline]
pub fn legal_cell(k: usize) -> usize {
    (k << 1) | ((k >> 3) & 1)
}

/// Mark a square as no longer playable. Called for every square a move writes, whether it
/// took a mark or was blasted away, which is exactly the set that stops being playable.
#[inline]
fn clear_legal(legal: &mut [u64], i: usize) {
    let (w, b) = legal_bit(i);
    legal[w] &= !b;
}


// ------------------------------------------------------------------------- single game

/// The opening position, laid out once at compile time so `Game::new` is a memcpy.
const OPENING: [i8; HW] = {
    let mut a = [NON_PLAYABLE_SQUARE; HW];
    let mut r = 0;
    while r < HEIGHT {
        let mut c = 0;
        while c < WIDTH {
            if (r + c) % 2 == 0 {
                a[r * WIDTH + c] = PLAYABLE_SQUARE;
            }
            c += 1;
        }
        r += 1;
    }
    a
};

/// One game, as a plain value.
///
/// Every field is a fixed-size array, so a position is ~350 bytes with no indirection and no
/// allocation: clone it into a tree node, play it forward, throw it away. The [`Scratch`] is
/// deliberately *not* part of it — that is working memory, not state, and a search holding
/// thousands of positions wants one workspace, not thousands, so it is lent to every call
/// that needs it.
///
/// The board also carries the derived state the scorer needs, which is what makes a fork a
/// memcpy rather than a rebuild: `levels` and the running `scores` come across with the
/// cells, so a clone costs nothing beyond the copy while [`Self::from_cells`] — adopting a
/// bare board — has to pay for a global BFS and a full rescore.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Game {
    /// The board, row-major, in the encoding at the top of this file.
    pub cells: [i8; HW],
    /// Steps to the border through own marks, parallel to `cells`. See the module docs.
    pub levels: [u8; HW],
    /// Running score, integer because every score is a sum of run lengths.
    pub scores: [i32; 2],
    /// Only ever compared against zero, to apply the opening-move half-board rule.
    pub move_count: u32,
    pub finished: bool,
    legal: [u64; LEGAL_WORDS],
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    /// The opening position: 80 playable squares, nothing played.
    pub fn new() -> Self {
        Game {
            cells: OPENING,
            levels: [INF; HW],
            scores: [0; 2],
            move_count: 0,
            finished: false,
            legal: LEGAL_ALL,
        }
    }

    /// Adopt an arbitrary board. The board *is* the state: levels come from the one global
    /// BFS, scores from a full rescan, legality from a scan for playable squares.
    ///
    /// The two flags are derived rather than supplied. A game is finished exactly when no
    /// playable square is left, and it is on its opening move exactly when nothing has been
    /// played — no marks and no blasted squares. So `move_count` comes back as 0 or 1 and
    /// not as the number of moves that actually made this board, which the board does not
    /// record; the half-board rule only ever asks which of the two it is. This is the same
    /// rule the Python applies when it imports a printed board.
    pub fn from_cells(cells: [i8; HW], s: &mut Scratch) -> Self {
        let mut g = Game {
            cells,
            levels: [INF; HW],
            scores: [0; 2],
            move_count: 0,
            finished: false,
            legal: [0; LEGAL_WORDS],
        };
        for i in 0..HW {
            if g.cells[i] == PLAYABLE_SQUARE {
                let (w, b) = legal_bit(i);
                g.legal[w] |= b;
            }
        }
        let played = g
            .cells
            .iter()
            .any(|&v| v != PLAYABLE_SQUARE && v != NON_PLAYABLE_SQUARE);
        g.move_count = u32::from(played);
        g.finished = g.no_moves_left();
        rebuild_levels(&g.cells, &mut g.levels, s);
        g.scores = [
            score_board(&g.cells, &g.levels, PLAYER_0_MARK),
            score_board(&g.cells, &g.levels, PLAYER_1_MARK),
        ];
        g
    }

    /// An empty bitboard is exactly "no playable square left".
    #[inline]
    fn no_moves_left(&self) -> bool {
        self.legal[0] == 0 && self.legal[1] == 0
    }

    /// The squares `player` may play right now, as a bitboard: two words, 80 bits, one per
    /// playable square, decoded by [`legal_cell`].
    ///
    /// This is the playable set narrowed to this player's half on the opening move, which
    /// costs one `AND` per word — see [`LEGAL_LEFT`]. There is no separate accessor for the
    /// un-narrowed set, because a caller cannot legally use one: on the opening move the
    /// halves are the rule, and after it both players see the same set, so `legal_moves(0)`
    /// and `legal_moves(1)` are the same two words.
    #[inline]
    pub fn legal_moves(&self, player: usize) -> [u64; LEGAL_WORDS] {
        let half = if self.move_count != 0 {
            LEGAL_ALL
        } else if player == 0 {
            LEGAL_LEFT
        } else {
            LEGAL_RIGHT
        };
        [self.legal[0] & half[0], self.legal[1] & half[1]]
    }

    /// How many moves `player` has. `popcount`, not a board scan.
    #[inline]
    pub fn legal_count(&self, player: usize) -> u32 {
        let w = self.legal_moves(player);
        w[0].count_ones() + w[1].count_ones()
    }

    /// The board index of the `k`th set bit of `w`, counting from the low end.
    #[inline]
    fn select(w: [u64; LEGAL_WORDS], mut k: u32) -> usize {
        for (wi, &word) in w.iter().enumerate() {
            let c = word.count_ones();
            if k < c {
                let mut x = word;
                for _ in 0..k {
                    x &= x - 1; // drop the lowest set bit
                }
                return legal_cell((wi << 6) + x.trailing_zeros() as usize);
            }
            k -= c;
        }
        unreachable!("select past the end of the legal set")
    }

    /// A uniform draw over a mask. Only the degenerate branch of [`Self::sample_move`] needs
    /// it: picking moves uniformly is a driver's business, not the engine's, and a driver
    /// that wants it already has the mask and can `popcount`, draw and select over it.
    #[inline]
    fn uniform(w: [u64; LEGAL_WORDS], rng: &mut Rng) -> usize {
        let count = w[0].count_ones() + w[1].count_ones();
        assert!(count > 0, "no legal move to sample");
        Self::select(w, rng.randint(count as u64) as u32)
    }

    /// Sample a legal move for `player` from a weight per square, `dist[0..HW]`.
    ///
    /// Accumulate the weight of the legal squares, draw a threshold, then walk the set bits
    /// again until the running sum crosses it. Weights need not be normalized. Set bits are
    /// walked low to high, which is board order, so this consumes the same draw and makes the
    /// same choice as the numba `sample_move_kernel` scanning all 160 squares against a
    /// materialized mask — the tests hold it to that.
    ///
    /// One deliberate exception: an all-zero distribution falls back to a uniform draw over
    /// the *legal* squares, where the numba version draws uniformly over the whole board and
    /// can land on a square that is not playable at all. Nothing here has to stay
    /// bit-compatible with that quirk, and a softmax policy never triggers it.
    pub fn sample_move(&self, dist: &[f32], player: usize, rng: &mut Rng) -> usize {
        debug_assert!(dist.len() >= HW, "distribution must cover the board");
        let words = self.legal_moves(player);

        let mut total = 0.0f64;
        for (wi, &word) in words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let k = (wi << 6) + w.trailing_zeros() as usize;
                w &= w - 1;
                total += dist[legal_cell(k)] as f64;
            }
        }
        if total < 1e-8 {
            return Self::uniform(words, rng);
        }

        let threshold = rng.random() * total;
        let mut cum = 0.0f64;
        let mut chosen = usize::MAX;
        'scan: for (wi, &word) in words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let k = (wi << 6) + w.trailing_zeros() as usize;
                w &= w - 1;
                let cell = legal_cell(k);
                let v = dist[cell] as f64;
                if v > 0.0 {
                    cum += v;
                    if cum >= threshold {
                        chosen = cell;
                        break 'scan;
                    }
                }
            }
        }
        // only reachable if rounding put the threshold past the whole sum
        if chosen == usize::MAX {
            return Self::uniform(words, rng);
        }
        chosen
    }

    /// Play one move: `i0` for player 0, `i1` for player 1, as board indices.
    ///
    /// Equal indices are a collision, which clears that square and its four diagonal
    /// neighbours — see [`apply_move`]. Both moves must be legal; that is checked only in
    /// debug builds, because the caller picked them out of [`Self::legal_moves`] and paying
    /// for a re-check on every node of a search is not worth it.
    #[inline]
    pub fn action_step(&mut self, i0: usize, i1: usize, s: &mut Scratch) {
        debug_assert!(!self.finished, "move played on a finished game");
        debug_assert!(self.holds_bit(i0, 0), "illegal move {i0} for player 0");
        debug_assert!(self.holds_bit(i1, 1), "illegal move {i1} for player 1");
        apply_move(
            &mut self.cells,
            &mut self.levels,
            &mut self.scores,
            &mut self.legal,
            i0,
            i1,
            s,
        );
        self.move_count += 1;
        self.finished = self.no_moves_left();
    }

    /// Does `player`'s mask hold the bit for board index `i`? For the debug assertions in
    /// [`Self::action_step`] — a caller wanting this reads its own mask.
    #[inline]
    fn holds_bit(&self, i: usize, player: usize) -> bool {
        if i >= HW || legal_cell(i >> 1) != i {
            return false; // not a playable-parity square, so it holds no bit of its own
        }
        let k = i >> 1;
        self.legal_moves(player)[k >> 6] >> (k & 63) & 1 == 1
    }
}
