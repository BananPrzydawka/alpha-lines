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

    /// Choose a move for `player` out of `legal`, which must not be empty.
    fn select(
        stats: &Self::Stats,
        legal: [u64; LEGAL_WORDS],
        player: usize,
        cfg: &Config,
        rng: &mut Rng,
    ) -> Choice;

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
    at: usize,
}

impl Iterator for MaskIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.at < LEGAL_WORDS {
            let w = self.words[self.at];
            if w != 0 {
                self.words[self.at] = w & (w - 1);
                return Some((self.at << 6) + w.trailing_zeros() as usize);
            }
            self.at += 1;
        }
        None
    }
}

#[inline]
pub fn squares(legal: [u64; LEGAL_WORDS]) -> MaskIter {
    MaskIter { words: legal, at: 0 }
}

#[inline]
fn count(legal: [u64; LEGAL_WORDS]) -> u32 {
    legal[0].count_ones() + legal[1].count_ones()
}

// ------------------------------------------------------------------------------------ puct

#[derive(Clone, Debug)]
pub struct PuctStats {
    pub prior: [[f32; SQUARES]; 2],
    pub visit: [[u16; SQUARES]; 2],
    pub q: [[f32; SQUARES]; 2],
    /// Backups dropped because an edge hit the `u16` visit ceiling.
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
        stats: &PuctStats,
        legal: [u64; LEGAL_WORDS],
        player: usize,
        cfg: &Config,
        _rng: &mut Rng,
    ) -> Choice {
        let (q, visit, prior) = (&stats.q[player], &stats.visit[player], &stats.prior[player]);
        let total: u32 = squares(legal).map(|sq| u32::from(visit[sq])).sum();
        let explore = cfg.c_puct * (total as f32).sqrt();

        let mut best = f32::NEG_INFINITY;
        let mut action = usize::MAX;
        for sq in squares(legal) {
            let score = q[sq] + explore * prior[sq] / (1.0 + f32::from(visit[sq]));
            if score > best {
                best = score;
                action = sq;
            }
        }
        debug_assert!(action != usize::MAX, "select over an empty legal set");
        Choice { action: action as u8, prob: 1.0 }
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
            if n == u16::MAX {
                stats.saturated += 1;
                continue;
            }
            let q = &mut stats.q[player][sq];
            *q = (*q * f32::from(n) + values[player]) / (f32::from(n) + 1.0);
            stats.visit[player][sq] = n + 1;
        }
    }
}

// ------------------------------------------------------------------------------------ exp3

#[derive(Clone, Debug)]
pub struct Exp3Stats {
    pub log_w: [[f32; SQUARES]; 2],
}

impl Default for Exp3Stats {
    /// All zero, which is the uniform strategy: the softmax of a flat vector is flat, and
    /// `(1 - gamma) / n + gamma / n` is `1 / n`. That is what an unevaluated node plays.
    fn default() -> Self {
        Exp3Stats { log_w: [[0.0; SQUARES]; 2] }
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

    /// Sample from the mixed strategy by inverse CDF, in board order.
    ///
    /// Three passes over the legal set and no `SQUARES`-wide buffer: the strategy is rebuilt
    /// term by term inside the scan, which is all the sampler needs.
    fn select(
        stats: &Exp3Stats,
        legal: [u64; LEGAL_WORDS],
        player: usize,
        cfg: &Config,
        rng: &mut Rng,
    ) -> Choice {
        let log_w = &stats.log_w[player];
        let n = count(legal);
        debug_assert!(n > 0, "select over an empty legal set");
        let gamma = cfg.exp3_gamma;
        let top = Self::top(log_w, legal);

        let mut sum = 0.0f32;
        for sq in squares(legal) {
            sum += (log_w[sq] - top).exp();
        }

        // the strategy sums to 1 by construction, so the threshold is drawn against 1
        let threshold = rng.random() as f32;
        let floor = gamma / n as f32;
        let mut cum = 0.0f32;
        let mut chosen = usize::MAX;
        let mut prob = 0.0f32;
        let mut last = (0usize, 0.0f32);
        for sq in squares(legal) {
            let p = (1.0 - gamma) * ((log_w[sq] - top).exp() / sum) + floor;
            last = (sq, p);
            cum += p;
            if cum > threshold {
                chosen = sq;
                prob = p;
                break;
            }
        }
        // only reachable if rounding left the running sum just short of the draw
        if chosen == usize::MAX {
            (chosen, prob) = last;
        }
        Choice { action: chosen as u8, prob }
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
