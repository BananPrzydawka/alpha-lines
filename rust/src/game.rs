//! The alpha-lines game: one board, scored incrementally.
//!
//! Every cell carries a *level*: the number of steps to the nearest border cell, walking
//! only over that player's own marks. A border mark is level 0, any other mark is
//! `1 + min(own-player diagonal neighbours)`, and [`INF`] means no route to the border.
//!
//! A run only scores if it can reach the border, and reachability is exactly
//! `level != INF`. So the score moves only when a cell crosses between INF and finite;
//! level changes that stay finite are bookkeeping. Levels rather than a reachable bit
//! because a bit cannot survive a deletion in the middle of a line anchored at both ends.
//!
//! Only `(r + c)` even squares are playable, so a mark's only neighbours are its four
//! diagonals — that is what [`DIAG`] encodes.
//!
//! [`Game`] is the public surface. Everything above it is the scorer it sits on. The module
//! has no dependencies on the rest of the crate and can be lifted out as a single file.

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

/// The largest level a real path can have: a player holds at most `HW / 4` marks and a
/// shortest path visits each once. A bound, not a mechanism — nothing legitimate hits it.
pub const MAX_LEVEL: u8 = (HW / 4) as u8;

const _: () = assert!(
    HW / 4 < INF as usize,
    "board is too large for a u8 level: MAX_LEVEL would collide with INF"
);

// ---------------------------------------------------------------------------------- rng

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

/// The four diagonal neighbours: the only neighbours a mark can have.
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

/// Working memory for one move, so a move allocates nothing: a generation-stamped visited
/// set (bump `epoch` to clear it in O(1)), a stack, a component buffer, and a bucket queue
/// keyed by level for processing cells in increasing level order.
///
/// The caller owns this and lends it to every call that needs it, which is what keeps a
/// [`Game`] plain data and a fork a memcpy. Create one per thread and pass it around.
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

/// What `i`'s level should be. `i` itself is not consulted, so this is equally valid for a
/// cell about to become a mark and for one that already is.
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

/// Push improvements outward after an insertion; levels only ever fall here.
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

/// Re-derive one whole component's levels: blank it, seed from its border cells, BFS.
///
/// `comp` must be the component the deleted cell belonged to, collected *before* the
/// deletion. Every level that can change is inside it, and nothing outside can rescue
/// anything inside it — a same-player diagonal neighbour would be in the component by
/// definition — so this is complete, not a local patch. O(|comp|).
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
    // seeding is a second pass: the first is still blanking cells a border cell found
    // early would have been compared against
    for &ci in comp {
        let i = ci as usize;
        if cells[i] == p && level[i] == 0 {
            q.push(ci);
        }
    }

    // a plain FIFO suffices: every step costs one, so cells come off in nondecreasing
    // level order, and no cell is ever queued twice
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

/// Rebuild every level from scratch. The only global BFS; used when adopting a board, and
/// by the tests as ground truth.
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

/// Collect every cell reachable from `seed` through `p`'s marks. Appends to `out` and
/// honours the current walk epoch, so several seeds union into one buffer cleanly.
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

/// Total score generated by the given cells. `cells_of_interest` must be a union of *whole*
/// components, since a run never leaves its component and so is never split by the set.
fn contribution(cells: &[i8], level: &[u8], cells_of_interest: &[u16], p: i8) -> i32 {
    cells_of_interest
        .iter()
        .map(|&ci| runs_starting_at(cells, level, ci as usize, p))
        .sum()
}

/// The whole board's score for `p`. Only for adopting a board; otherwise the score is
/// maintained incrementally.
fn score_board(cells: &[i8], level: &[u8], p: i8) -> i32 {
    (0..HW).map(|i| runs_starting_at(cells, level, i, p)).sum()
}

/// Score of the runs that *begin* at `i`, in either family. Zero if `i` holds no mark of
/// `p`, if an earlier cell owns the run, if the run is shorter than 2, or if it is
/// unreachable. A cell in a qualifying run of both families counts twice, as intended.
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

/// The score change from filling or emptying the single square `i`, given that reachability
/// does not change. A run of length L scores L if L >= 2, so joining runs of length `la` and
/// `lb` gains `1 + (la == 1) + (lb == 1)`.
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
        // fast path: dropped into live territory, so the score change is local
        cells[i] = p;
        level[i] = lvl;
        relax_down(cells, level, i, p, s);
        score[player_index(p)] += local_delta(cells, i, p);
        return;
    }

    // slow path: this revives a dead blob, so measure before, measure after, apply the
    // difference
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

    // only a neighbour one level above could have been routing through this cell
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
        // fast path: the insertion calculation, negated
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

    // `comp` may now be several pieces, some INF; `contribution` handles that
    let new = contribution(cells, level, &comp, p);

    s.comp = comp;
    s.deps = deps;
    score[player_index(p)] += new - old;
}

/// Apply one move. A collision — both players on the same square — clears that square and
/// its four diagonal neighbours, as five independent single-cell removals; they lie in
/// different runs, so the order cannot matter.
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

// ------------------------------------------------------------------- legality bitboard

/// `u64`s in a legality mask. The 80 playable squares pack into 80 bits: `i / 2` maps them
/// onto `0..80` densely and in row-major order, since row `r`'s playable columns are
/// `2j + (r & 1)`, so `i / 2 == r * 8 + j`. Order matters — the sampler's cumulative scan
/// walks bits low to high, which is board order.
///
/// This is a count of words, not a width; the words are `u64` at every use site.
pub const LEGAL_WORDS: usize = 2;

const _: () = assert!(HW / 2 == 80, "the legality bitboard assumes 80 playable squares");
const _: () = assert!(LEGAL_WORDS * 64 >= HW / 2, "the mask cannot hold every playable square");

/// The 80 valid bits — the opening position.
const LEGAL_ALL: [u64; LEGAL_WORDS] = [!0u64, 0xFFFF];

/// The opening-move half-board rule, precomputed: `c < 8` is `j < 4` for both row parities,
/// so the rule is the low nibble of every byte and applying it is one `AND`.
const LEGAL_LEFT: [u64; LEGAL_WORDS] = [0x0F0F_0F0F_0F0F_0F0F, 0x0F0F];
const LEGAL_RIGHT: [u64; LEGAL_WORDS] = [0xF0F0_F0F0_F0F0_F0F0, 0xF0F0];

/// Word and bit for a board index. A non-playable index aliases onto its even neighbour,
/// harmless because that bit is never set.
#[inline]
fn legal_bit(i: usize) -> (usize, u64) {
    let k = i >> 1;
    (k >> 6, 1u64 << (k & 63))
}

/// The board index a bit position stands for — the decoder for [`Game::legal_moves`], which
/// is otherwise 80 opaque bit positions. Walk a mask with
/// `while w != 0 { let cell = legal_cell(base + w.trailing_zeros() as usize); w &= w - 1; }`.
#[inline]
pub fn legal_cell(k: usize) -> usize {
    (k << 1) | ((k >> 3) & 1)
}

/// Mark a square as no longer playable: every square a move writes, marked or blasted.
#[inline]
fn clear_legal(legal: &mut [u64], i: usize) {
    let (w, b) = legal_bit(i);
    legal[w] &= !b;
}


// ------------------------------------------------------------------------- single game

/// The opening position, laid out at compile time so `Game::new` is a memcpy.
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

/// One game: ~350 bytes of fixed-size arrays, no indirection, no allocation. Clone it into
/// a tree node, play it forward, throw it away.
///
/// It carries the scorer's derived state — `levels` and the running `scores` — which is why
/// a clone is a memcpy while [`Self::from_cells`] has to pay for a BFS and a full rescore.
/// The [`Scratch`] is not part of it; pass one in.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Game {
    /// The board, row-major, in the encoding at the top of this file.
    pub cells: [i8; HW],
    /// Steps to the border through own marks, parallel to `cells`. See the module docs.
    pub levels: [u8; HW],
    /// Running score, integer because every score is a sum of run lengths.
    pub scores: [i32; 2],
    /// Only ever compared against zero, for the opening-move half-board rule.
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

    /// Adopt an arbitrary board: levels from the global BFS, scores from a full rescan,
    /// legality from a scan for playable squares.
    ///
    /// `move_count` comes back as 0 or 1, not the real count — the board records whether a
    /// move was made, not how many, and the half-board rule only asks which. Same rule the
    /// Python applies when importing a printed board.
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

    /// The squares `player` may play, as a mask: 80 bits, decoded by [`legal_cell`]. On the
    /// opening move this is narrowed to the player's half; after it, both players see the
    /// same two words, so `legal_moves(0) | legal_moves(1)` is the playable set.
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

    /// How many moves `player` has: two `popcount`s, not a board scan.
    #[inline]
    pub fn legal_count(&self, player: usize) -> u32 {
        let w = self.legal_moves(player);
        w[0].count_ones() + w[1].count_ones()
    }

    /// The board index of the `k`th set bit of `w`, from the low end.
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

    /// A uniform draw over a mask, for the degenerate all-zero-distribution case.
    #[inline]
    fn uniform(w: [u64; LEGAL_WORDS], rng: &mut Rng) -> usize {
        let count = w[0].count_ones() + w[1].count_ones();
        assert!(count > 0, "no legal move to sample");
        Self::select(w, rng.randint(count as u64) as u32)
    }

    /// Draw one legal move for `player`, weighting each square by `dist`. Weights need not
    /// be normalized; an all-zero distribution falls back to a uniform draw over the mask.
    fn sample(&self, dist: &[f32], player: usize, rng: &mut Rng) -> usize {
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

    /// Play the moves `i0` and `i1`, as board indices. Equal indices are a collision, which
    /// clears that square and its four diagonal neighbours.
    ///
    /// Both moves must be legal — checked in debug builds only, so pick them out of
    /// [`Self::legal_moves`].
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

    /// Play one move drawn from a weight per square for each player, `dist_0[0..HW]` and
    /// `dist_1[0..HW]`. The other way in besides [`Self::action_step`]; they differ only in
    /// where the moves come from.
    pub fn distribution_step(
        &mut self,
        dist_0: &[f32],
        dist_1: &[f32],
        rng: &mut Rng,
        s: &mut Scratch,
    ) {
        debug_assert!(!self.finished, "move played on a finished game");
        let i0 = self.sample(dist_0, 0, rng);
        let i1 = self.sample(dist_1, 1, rng);
        self.action_step(i0, i1, s);
    }

    /// Does `player`'s mask hold the bit for `i`? For the debug assertions only; a caller
    /// wanting this reads its own mask.
    #[inline]
    fn holds_bit(&self, i: usize, player: usize) -> bool {
        if i >= HW || legal_cell(i >> 1) != i {
            return false; // not a playable-parity square, so it holds no bit of its own
        }
        let k = i >> 1;
        self.legal_moves(player)[k >> 6] >> (k & 63) & 1 == 1
    }
}
