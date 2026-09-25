//! C ABI for collecting PUCT targets with the current KataGo model.
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::game::{board_index, SQUARES};
use crate::klent::encode;
use crate::mcts::{Config, Puct, Search, StepRecord};

pub struct Trainer {
    search: Search<Puct>,
    records: Vec<StepRecord>,
    model_calls: usize,
    evaluated: usize,
    finished_games: usize,
}

fn coordinate(index: usize, transform: u8) -> usize {
    let mut r = index / 16;
    let mut c = index % 16;
    if transform & 1 != 0 { r = 9 - r; }
    if transform & 2 != 0 { c = 15 - c; }
    r * 16 + c
}

impl Trainer {
    fn new(cfg: Config, seed: u64) -> Self {
        assert!(cfg.g > 0 && cfg.g <= u16::MAX as usize);
        assert!(cfg.b > 0 && cfg.b <= cfg.g && cfg.t > 0 && cfg.t <= cfg.g);
        assert!(cfg.s > 0 && cfg.node_capacity > 0);
        assert!(cfg.c_puct.is_finite() && cfg.c_puct > 0.0);
        assert!(cfg.alpha.is_finite() && cfg.alpha > 0.0);
        assert!(cfg.epsilon.is_finite() && (0.0..=1.0).contains(&cfg.epsilon));
        Self { search: Search::new(cfg, seed), records: Vec::new(),
            model_calls: 0, evaluated: 0, finished_games: 0 }
    }

    fn inputs(&mut self, boards: &mut [f32]) -> usize {
        boards.fill(0.0);
        let b = self.search.cfg.b;
        let (positions, len) = self.search.prepare_evaluation();
        for row in 0..len {
            for player in 0..2 {
                let start = (player * b + row) * 800;
                encode(&positions[row], player, &mut boards[start..start + 800]);
            }
        }
        self.model_calls += 1;
        self.evaluated += len;
        len
    }

    fn advance(&mut self, logits: &[f32], q: &[f32]) -> usize {
        let emitted = self.search.complete_evaluation(logits, q);
        let count = emitted.len();
        self.records.extend(emitted);
        self.finished_games += self.search.take_completed_histories().len();
        count
    }

    fn batch(&self, ids: &[u32], transforms: &[u8], boards: &mut [f32],
             policies: &mut [f32], action_values: &mut [f32], visited: &mut [f32]) {
        boards.fill(0.0);
        policies.fill(0.0);
        action_values.fill(0.0);
        visited.fill(0.0);
        for (i, (&id, &transform)) in ids.iter().zip(transforms).enumerate() {
            assert!(transform < 4);
            let record = &self.records[id as usize];
            for player in 0..2 {
                let row = 2 * i + player;
                let mut source_board = [0.0f32; 800];
                encode(&record.cells, player, &mut source_board);
                for plane in 0..5 {
                    for cell in 0..160 {
                        boards[row * 800 + plane * 160 + coordinate(cell, transform)] =
                            source_board[plane * 160 + cell];
                    }
                }
                for sq in 0..SQUARES {
                    let target = coordinate(board_index(sq), transform);
                    policies[row * 160 + target] = record.policy[player][sq];
                    action_values[row * 160 + target] = record.action_value[player][sq];
                    visited[row * 160 + target] = record.visits[player][sq] as f32;
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn mcts_new(g: usize, b: usize, t: usize, s: u32, node_capacity: usize,
                            c_puct: f32, alpha: f32, epsilon: f32, seed: u64) -> *mut Trainer {
    catch_unwind(|| {
        let cfg = Config { g, b, t, s, node_capacity, c_puct, alpha, epsilon,
            ..Config::default() };
        Box::into_raw(Box::new(Trainer::new(cfg, seed)))
    }).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn mcts_free(handle: *mut Trainer) {
    if !handle.is_null() { drop(Box::from_raw(handle)); }
}

#[no_mangle]
pub unsafe extern "C" fn mcts_inputs(handle: *mut Trainer, boards: *mut f32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let trainer = &mut *handle;
        let out = std::slice::from_raw_parts_mut(boards, trainer.search.cfg.b * 1600);
        trainer.inputs(out) as i32
    })).unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn mcts_advance(handle: *mut Trainer, logits: *const f32, q: *const f32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let trainer = &mut *handle;
        let len = trainer.search.cfg.b * 160;
        trainer.advance(std::slice::from_raw_parts(logits, len),
                        std::slice::from_raw_parts(q, len)) as i32
    })).unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn mcts_stats(handle: *const Trainer, out: *mut usize) {
    let trainer = &*handle;
    std::slice::from_raw_parts_mut(out, 6).copy_from_slice(&[
        trainer.records.len(), trainer.model_calls, trainer.evaluated,
        trainer.finished_games, trainer.search.arena.len(), trainer.search.exhausted,
    ]);
}

#[no_mangle]
pub unsafe extern "C" fn mcts_batch(handle: *const Trainer, count: usize, ids: *const u32,
    transforms: *const u8, boards: *mut f32, policies: *mut f32,
    action_values: *mut f32, visited: *mut f32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let trainer = &*handle;
        let ids = std::slice::from_raw_parts(ids, count);
        let transforms = std::slice::from_raw_parts(transforms, count);
        let boards = std::slice::from_raw_parts_mut(boards, count * 1600);
        let policies = std::slice::from_raw_parts_mut(policies, count * 320);
        let action_values = std::slice::from_raw_parts_mut(action_values, count * 320);
        let visited = std::slice::from_raw_parts_mut(visited, count * 320);
        trainer.batch(ids, transforms, boards, policies, action_values, visited);
        count as i32
    })).unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn mcts_network_update(handle: *mut Trainer) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let trainer = &mut *handle;
        trainer.search.clear_for_new_model();
        trainer.records.clear();
        trainer.model_calls = 0;
        trainer.evaluated = 0;
        trainer.finished_games = 0;
        0
    })).unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_step_emits_exactly_t_records_and_masks_unvisited_values() {
        let cfg = Config { g: 8, b: 8, t: 4, s: 2, node_capacity: 1000,
            ..Config::default() };
        let mut trainer = Trainer::new(cfg, 23);
        let mut boards = vec![0.0; cfg.b * 1600];
        let logits = vec![0.0; cfg.b * 160];
        let q = vec![0.5; cfg.b * 160];
        let mut emitted = 0;
        for _ in 0..20 {
            trainer.inputs(&mut boards);
            emitted = trainer.advance(&logits, &q);
            if emitted > 0 { break; }
        }
        assert_eq!(emitted, cfg.t);
        assert_eq!(trainer.records.len(), cfg.t);
        assert!(trainer.search.ready() > 0, "extra ready games should wait");
        for record in &trainer.records {
            for player in 0..2 {
                assert_eq!(record.visits[player].iter().sum::<u64>(), cfg.s as u64);
                for sq in 0..SQUARES {
                    if record.visits[player][sq] > 0 {
                        assert!((record.action_value[player][sq] - 0.5).abs() < 1e-5);
                    }
                }
            }
        }
    }
}
