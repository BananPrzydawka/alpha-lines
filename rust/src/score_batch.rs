//! Single-threaded random games for the score demo. No search or inference.
use crate::{game::Rng, Game, Scratch, SQUARES};

pub struct ScoreBatch {
    games: Vec<Game>,
    rng: Rng,
    scratch: Scratch,
}

impl ScoreBatch {
    pub fn new(batch: usize, seed: u64) -> Self {
        assert!(batch > 0);
        Self { games: vec![Game::new(); batch], rng: Rng::new(seed), scratch: Scratch::new() }
    }

    /// Return post-move boards, including terminal boards exactly once. A finished
    /// slot starts a new game on the next call. Both players sample independently
    /// from their pre-move legal set; zero weights invoke exact uniform sampling.
    pub fn step(&mut self, cells: &mut [u8], scores: &mut [i64]) -> u64 {
        assert_eq!(cells.len(), self.games.len() * SQUARES);
        assert_eq!(scores.len(), self.games.len() * 2);
        let mut finished = 0;
        for (i, game) in self.games.iter_mut().enumerate() {
            if game.finished { *game = Game::new(); }
            game.distribution_step(&[0.0; SQUARES], &[0.0; SQUARES], &mut self.rng, &mut self.scratch);
            cells[i * SQUARES..(i + 1) * SQUARES].copy_from_slice(&game.cells);
            for p in 0..2 {
                assert!((0..=80).contains(&game.scores[p]));
                scores[2 * i + p] = game.scores[p] as i64;
            }
            finished += u64::from(game.finished);
        }
        finished
    }
}

// Private-to-this-project C ABI, used by python/score_data.py. The Python wrapper
// owns the handle and allocates contiguous CPU tensors of precisely these sizes.
#[no_mangle]
pub extern "C" fn score_batch_new(batch: usize, seed: u64) -> *mut ScoreBatch {
    if batch == 0 { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(ScoreBatch::new(batch, seed)))
}

/// # Safety
/// Handle must be live and exclusively owned; outputs must have batch*80 u8 and
/// batch*2 i64 writable elements, respectively, and must not alias.
#[no_mangle]
pub unsafe extern "C" fn score_batch_step(handle: *mut ScoreBatch, cells: *mut u8, scores: *mut i64) -> u64 {
    let batch = &mut *handle;
    let n = batch.games.len();
    batch.step(std::slice::from_raw_parts_mut(cells, n * SQUARES),
               std::slice::from_raw_parts_mut(scores, n * 2))
}

/// # Safety
/// Handle must have been returned by score_batch_new and not previously freed.
#[no_mangle]
pub unsafe extern "C" fn score_batch_free(handle: *mut ScoreBatch) {
    if !handle.is_null() { drop(Box::from_raw(handle)); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_batches_match_rescoring_and_recycle() {
        let mut batch = ScoreBatch::new(16, 42);
        let mut replay = ScoreBatch::new(16, 42);
        let mut cells = vec![0; 16 * SQUARES];
        let mut targets = vec![0; 32];
        let mut other_cells = cells.clone();
        let mut other_targets = targets.clone();
        let mut scratch = Scratch::new();
        let mut completed = 0;
        for _ in 0..160 {
            let ended = batch.step(&mut cells, &mut targets);
            assert_eq!(ended, replay.step(&mut other_cells, &mut other_targets));
            assert_eq!(cells, other_cells);
            assert_eq!(targets, other_targets);
            completed += ended;
            for i in 0..16 {
                let game = Game::from_cells(cells[i*SQUARES..(i+1)*SQUARES].try_into().unwrap(), &mut scratch);
                assert_eq!(game.scores.map(i64::from), targets[2*i..2*i+2]);
                assert_eq!(game.finished, batch.games[i].finished);
            }
        }
        assert!(completed >= 16 * 2);
    }
}
