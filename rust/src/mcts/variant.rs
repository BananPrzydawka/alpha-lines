//! The two search rules. Everything else about the search is shared.
//!
//! A variant supplies a statistic block per node and three operations on it: adopt the
//! model's priors, choose a move, and fold a value back into the edge that was chosen.
//!
//! Values are absolute — `values[0]` is always player 0's — never relative to a side to move.
//! That is what makes a node's statistics safe to accumulate across games.

use crate::game::{Rng, LEGAL_WORDS, SQUARES};
use crate::mcts::config::Config;
use crate::mcts::noise;

/// A move and the probability it was drawn with. PUCT is deterministic and leaves `prob` at
/// 1; EXP3 needs it for the importance weight in its backup.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Choice {
    pub action: u8,
    pub prob: f32,
}

pub trait Variant {
    type Stats: Clone + Default;

    /// Adopt the model's output for one player. `priors` is at least `SQUARES` long.
    fn set_priors(
        stats: &mut Self::Stats,
        priors: &[f32],
        legal: [u64; LEGAL_WORDS],
        player: usize,
    );

    /// Choose the joint move: one square per player, out of that player's mask, neither of
    /// which may be empty.
    ///
    /// Joint because the players move simultaneously and the search only ever wants both,
    /// which also keeps a caller from pairing one player's mask with the other's index.
    ///
    /// Takes the statistics mutably because EXP3 accumulates its average strategy here, which
    /// is the thing a root eventually emits as a training target.
    fn select(
        stats: &mut Self::Stats,
        legal: [[u64; LEGAL_WORDS]; 2],
        cfg: &Config,
        rng: &mut Rng,
    ) -> [Choice; 2];

    /// The training target for a root: a distribution per player over the legal squares.
    ///
    /// PUCT emits the visit distribution, EXP3 the average strategy. Both are read only from
    /// roots, but both are accumulated everywhere, so promoting a node into a root keeps the
    /// search already done at that state instead of restarting it.
    fn target(
        stats: &Self::Stats,
        legal: [[u64; LEGAL_WORDS]; 2],
        out: &mut [[f32; SQUARES]; 2],
    );

    /// Treat this statistic block as a root's.
    ///
    /// Called when a game adopts a position as its root: on a fresh seed, on promotion, and
    /// when a pending root's evaluation lands. PUCT mixes in fresh Dirichlet noise — on an
    /// all-zero prior that is pure noise, which is what §7 wants of an unevaluated root.
    fn make_root(
        stats: &mut Self::Stats,
        legal: [[u64; LEGAL_WORDS]; 2],
        cfg: &Config,
        rng: &mut Rng,
    );

    /// Fold `values` into the edges named by `choice`.
    ///
    /// `legal_counts` is the node's own legal-move count per player. It is a property of the
    /// position, so it is the same at backup as it was at selection.
    fn backup(
        stats: &mut Self::Stats,
        choice: [Choice; 2],
        values: [f32; 2],
        legal_counts: [u32; 2],
        cfg: &Config,
    );
}

/// The set squares of a mask, low to high — which is board order.
#[derive(Clone, Copy)]
pub struct MaskIter {
    words: [u64; LEGAL_WORDS],
    /// Which of the two words is being drained.
    word: usize,
}

impl Iterator for MaskIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.word < LEGAL_WORDS {
            let w = self.words[self.word];
            if w != 0 {
                self.words[self.word] = w & (w - 1);
                return Some((self.word << 6) + w.trailing_zeros() as usize);
            }
            self.word += 1;
        }
        None
    }
}

#[inline]
pub fn squares(legal: [u64; LEGAL_WORDS]) -> MaskIter {
    MaskIter { words: legal, word: 0 }
}

#[inline]
fn count(legal: [u64; LEGAL_WORDS]) -> u32 {
    legal[0].count_ones() + legal[1].count_ones()
}

// ------------------------------------------------------------------------------------ puct

#[derive(Clone, Debug)]
pub struct PuctStats {
    pub prior: [[f32; SQUARES]; 2],
    pub visit: [[u32; SQUARES]; 2],
    pub q: [[f32; SQUARES]; 2],
    /// Sum of `visit` per player, maintained by backup so selection reads the arrays once
    /// rather than twice. Over every square, not just the legal ones — the same number, since
    /// an illegal square is never selected and so never accrues a visit.
    pub total: [u32; 2],
    /// Backups dropped because an edge hit the `u32` visit ceiling. Unreachable in practice;
    /// it is here so that if it ever happens it is visible rather than silent.
    pub saturated: u32,
}

impl Default for PuctStats {
    fn default() -> Self {
        PuctStats {
            prior: [[0.0; SQUARES]; 2],
            visit: [[0; SQUARES]; 2],
            q: [[0.0; SQUARES]; 2],
            total: [0; 2],
            saturated: 0,
        }
    }
}

pub struct Puct;

impl Variant for Puct {
    type Stats = PuctStats;

    /// Mask the priors to the legal squares and renormalise, as `mcts_puct.py` does.
    fn set_priors(stats: &mut PuctStats, priors: &[f32], legal: [u64; LEGAL_WORDS], player: usize) {
        debug_assert!(priors.len() >= SQUARES);
        let dst = &mut stats.prior[player];
        *dst = [0.0; SQUARES];
        let mut sum = 0.0f32;
        for sq in squares(legal) {
            dst[sq] = priors[sq];
            sum += priors[sq];
        }
        let norm = sum.max(1e-8);
        for sq in squares(legal) {
            dst[sq] /= norm;
        }
    }

    /// `argmax(q + c * prior * sqrt(1 + total) / (1 + visit))` over the legal squares.
    ///
    /// `total` is the node's own edge counts and is never read from a parent or a child, so
    /// backups arriving from several games compose here.
    fn select(
        stats: &mut PuctStats,
        legal: [[u64; LEGAL_WORDS]; 2],
        cfg: &Config,
        _rng: &mut Rng,
    ) -> [Choice; 2] {
        let mut out = [Choice { action: 0, prob: 1.0 }; 2];
        for player in 0..2 {
            let mask = legal[player];
            // `1 +` deviates from `mcts_puct.py`, which uses sqrt(total) and so scores every
            // square zero on a node's first visit — priors, and the root's Dirichlet noise,
            // multiplied away. Sequentially that self-corrects after one backup; here
            // thousands of games select at the same fresh root in one cycle and all choose
            // the same square. The offset makes the first visit follow the priors.
            let explore = cfg.c_puct * (1.0 + stats.total[player] as f32).sqrt();
            let (q, visit, prior) =
                (&stats.q[player], &stats.visit[player], &stats.prior[player]);
            debug_assert_eq!(
                stats.total[player],
                visit.iter().sum::<u32>(),
                "the cached visit total drifted from the counts it stands for"
            );

            let mut best = f32::NEG_INFINITY;
            let mut action = usize::MAX;
            for sq in squares(mask) {
                let score = q[sq] + explore * prior[sq] / (1.0 + visit[sq] as f32);
                if score > best {
                    best = score;
                    action = sq;
                }
            }
            debug_assert!(action != usize::MAX, "select over an empty legal set");
            out[player] = Choice { action: action as u8, prob: 1.0 };
        }
        out
    }

    /// `visit / sum(visit)` over the legal squares.
    fn target(
        stats: &PuctStats,
        legal: [[u64; LEGAL_WORDS]; 2],
        out: &mut [[f32; SQUARES]; 2],
    ) {
        for player in 0..2 {
            out[player] = [0.0; SQUARES];
            let total = stats.total[player] as f32;
            if total == 0.0 {
                // never searched: fall back to uniform rather than emit nothing
                let n = count(legal[player]) as f32;
                for sq in squares(legal[player]) {
                    out[player][sq] = 1.0 / n;
                }
                continue;
            }
            for sq in squares(legal[player]) {
                out[player][sq] = stats.visit[player][sq] as f32 / total;
            }
        }
    }

    /// Mix `epsilon` of a fresh Dirichlet sample into the priors, legal squares only.
    fn make_root(
        stats: &mut PuctStats,
        legal: [[u64; LEGAL_WORDS]; 2],
        cfg: &Config,
        rng: &mut Rng,
    ) {
        let mut noise = [0.0f32; SQUARES];
        for player in 0..2 {
            let n = count(legal[player]) as usize;
            debug_assert!(n > 0, "a root with no legal move");
            noise::dirichlet(rng, cfg.alpha, &mut noise[..n]);
            for (k, sq) in squares(legal[player]).enumerate() {
                let p = &mut stats.prior[player][sq];
                *p = (1.0 - cfg.epsilon) * *p + cfg.epsilon * noise[k];
            }
        }
    }

    fn backup(
        stats: &mut PuctStats,
        choice: [Choice; 2],
        values: [f32; 2],
        _legal_counts: [u32; 2],
        _cfg: &Config,
    ) {
        for player in 0..2 {
            let sq = choice[player].action as usize;
            let n = stats.visit[player][sq];
            // the total is the sum, so it hits the ceiling long before any single edge does;
            // low-ply nodes are never flushed and run for the whole deployment
            if n == u32::MAX || stats.total[player] == u32::MAX {
                stats.saturated += 1;
                continue;
            }
            let q = &mut stats.q[player][sq];
            *q = (*q * n as f32 + values[player]) / (n as f32 + 1.0);
            stats.visit[player][sq] = n + 1;
            stats.total[player] += 1;
        }
    }
}

// ------------------------------------------------------------------------------------ exp3

#[derive(Clone, Debug)]
pub struct Exp3Stats {
    pub log_w: [[f32; SQUARES]; 2],
    /// Every mixed strategy this node has played, summed. Normalised, it is the average
    /// strategy — the training target a root emits.
    ///
    /// Carried on interior nodes, not only roots, because promotion turns an interior node
    /// into a root: the sum it built up while interior is exactly the search work that would
    /// otherwise be thrown away when the game steps into it.
    pub strategy_sum: [[f32; SQUARES]; 2],
}

impl Default for Exp3Stats {
    /// All zero, which is the uniform strategy: the softmax of a flat vector is flat, and
    /// `(1 - gamma) / n + gamma / n` is `1 / n`. That is what an unevaluated node plays.
    fn default() -> Self {
        Exp3Stats {
            log_w: [[0.0; SQUARES]; 2],
            strategy_sum: [[0.0; SQUARES]; 2],
        }
    }
}

pub struct Exp3;

impl Exp3 {
    /// The largest log weight over the legal squares, which the softmax is shifted by so a
    /// large weight cannot overflow.
    #[inline]
    fn top(log_w: &[f32; SQUARES], legal: [u64; LEGAL_WORDS]) -> f32 {
        let mut top = f32::NEG_INFINITY;
        for sq in squares(legal) {
            if log_w[sq] > top {
                top = log_w[sq];
            }
        }
        top
    }

    /// The full mixed strategy for `player`, written into `out` at the legal squares.
    ///
    /// Only a root needs this, for its `strategy_sum`; [`Variant::select`] samples without
    /// materialising the vector.
    pub fn mixed(
        stats: &Exp3Stats,
        legal: [u64; LEGAL_WORDS],
        player: usize,
        gamma: f32,
        out: &mut [f32; SQUARES],
    ) {
        let log_w = &stats.log_w[player];
        let n = count(legal);
        debug_assert!(n > 0, "mixed strategy over an empty legal set");
        let top = Self::top(log_w, legal);

        let mut sum = 0.0f32;
        for sq in squares(legal) {
            let e = (log_w[sq] - top).exp();
            out[sq] = e;
            sum += e;
        }
        let floor = gamma / n as f32;
        for sq in squares(legal) {
            out[sq] = (1.0 - gamma) * (out[sq] / sum) + floor;
        }
    }
}

impl Variant for Exp3 {
    type Stats = Exp3Stats;

    /// `log(clip(prior, 1e-8))` on legal squares, `-inf` elsewhere, as `mcts_exp3.py` does.
    /// The softmax is shift-invariant, so it does not matter whether `priors` is normalised.
    fn set_priors(stats: &mut Exp3Stats, priors: &[f32], legal: [u64; LEGAL_WORDS], player: usize) {
        debug_assert!(priors.len() >= SQUARES);
        let dst = &mut stats.log_w[player];
        *dst = [f32::NEG_INFINITY; SQUARES];
        for sq in squares(legal) {
            dst[sq] = priors[sq].max(1e-8).ln();
        }
    }

    /// Build the mixed strategy, add it to the running average, and sample from it by
    /// inverse CDF in board order.
    fn select(
        stats: &mut Exp3Stats,
        legal: [[u64; LEGAL_WORDS]; 2],
        cfg: &Config,
        rng: &mut Rng,
    ) -> [Choice; 2] {
        // one buffer for both players: `mixed` writes only the squares it is about to read
        let mut probs = [0.0f32; SQUARES];
        let mut out = [Choice { action: 0, prob: 0.0 }; 2];

        for player in 0..2 {
            let mask = legal[player];
            Self::mixed(stats, mask, player, cfg.exp3_gamma, &mut probs);

            let sum = &mut stats.strategy_sum[player];
            let threshold = rng.random() as f32; // the strategy sums to 1 by construction
            let mut cum = 0.0f32;
            let mut chosen = None;
            let mut last = (0usize, 0.0f32);
            for sq in squares(mask) {
                sum[sq] += probs[sq];
                last = (sq, probs[sq]);
                cum += probs[sq];
                if chosen.is_none() && cum > threshold {
                    chosen = Some(last);
                }
            }
            // `chosen` is None only if rounding left the running sum just short of the draw
            let (action, prob) = chosen.unwrap_or(last);
            out[player] = Choice { action: action as u8, prob };
        }
        out
    }

    /// `strategy_sum / sum` over the legal squares: the average strategy, which is what EXP3
    /// converges on rather than its latest one.
    fn target(
        stats: &Exp3Stats,
        legal: [[u64; LEGAL_WORDS]; 2],
        out: &mut [[f32; SQUARES]; 2],
    ) {
        for player in 0..2 {
            out[player] = [0.0; SQUARES];
            let total: f32 = squares(legal[player]).map(|sq| stats.strategy_sum[player][sq]).sum();
            if total <= 0.0 {
                let n = count(legal[player]) as f32;
                for sq in squares(legal[player]) {
                    out[player][sq] = 1.0 / n;
                }
                continue;
            }
            for sq in squares(legal[player]) {
                out[player][sq] = stats.strategy_sum[player][sq] / total;
            }
        }
    }

    /// Nothing to do.
    ///
    /// EXP3 has no priors to disturb — exploration is the `gamma / n` floor, which is already
    /// in every mixed strategy. And the inherited `strategy_sum` is deliberately kept: it was
    /// accumulated by searches through this very state, so a promoted node starts with a
    /// target already part-built rather than from nothing.
    fn make_root(
        _stats: &mut Exp3Stats,
        _legal: [[u64; LEGAL_WORDS]; 2],
        _cfg: &Config,
        _rng: &mut Rng,
    ) {
    }

    /// `log_w[a] += gamma * (reward / prob) / num_legal`, with `reward = (value + 1) / 2`.
    ///
    /// `prob` is the probability the sample was actually drawn with, so the importance weight
    /// stays unbiased even when another game has moved `log_w` in between.
    fn backup(
        stats: &mut Exp3Stats,
        choice: [Choice; 2],
        values: [f32; 2],
        legal_counts: [u32; 2],
        cfg: &Config,
    ) {
        for player in 0..2 {
            let Choice { action, prob } = choice[player];
            // the mixed strategy puts a floor of gamma / n under every legal square, so a
            // drawn move never has probability zero
            debug_assert!(prob > 0.0, "a move was drawn with probability zero");
            debug_assert!(legal_counts[player] > 0, "a move was drawn from an empty set");
            let reward = (values[player] + 1.0) / 2.0;
            stats.log_w[player][action as usize] +=
                cfg.exp3_gamma * (reward / prob) / legal_counts[player] as f32;
        }
    }
}
