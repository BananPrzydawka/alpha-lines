//! The two search rules. Everything else about the search is shared.
//!
//! A variant supplies a statistic block per node and three operations on it: adopt the
//! model's priors, choose a move, and fold a value back into the edge that was chosen.
//!
//! Values are absolute — `values[0]` is always player 0's — never relative to a side to move.
//! That is what makes a node's statistics safe to accumulate across games.

use crate::game::{Rng, LEGAL_WORDS, SQUARES};
use crate::mcts::config::Config;

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

    /// `argmax(q + c * prior * sqrt(total) / (1 + visit))` over the legal squares.
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
            let (q, visit, prior) =
                (&stats.q[player], &stats.visit[player], &stats.prior[player]);
            let total: u64 = squares(mask).map(|sq| u64::from(visit[sq])).sum();
            let explore = cfg.c_puct * (total as f32).sqrt();

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
            if n == u32::MAX {
                stats.saturated += 1;
                continue;
            }
            let q = &mut stats.q[player][sq];
            *q = (*q * n as f32 + values[player]) / (n as f32 + 1.0);
            stats.visit[player][sq] = n + 1;
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
