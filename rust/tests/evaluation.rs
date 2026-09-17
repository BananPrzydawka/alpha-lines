use alpha_lines_game::{game::SQUARES, mcts::{Config, Evaluate, Exp3, Puct, Search, Variant}, Game, Scratch};

struct CheckInputs {
    scratch: Scratch,
    saw_scores: bool,
}
impl Evaluate for CheckInputs {
    fn evaluate(&mut self, cells: &[[u8; SQUARES]], scores: &[[i32; 2]],
                priors: &mut [f32], values: &mut [f32]) {
        assert_eq!(cells.len(), scores.len());
        for (&board, &score) in cells.iter().zip(scores) {
            assert_eq!(Game::from_cells(board, &mut self.scratch).scores, score);
            self.saw_scores |= score != [0; 2];
        }
        priors.fill(1.0);
        values.fill(0.0);
    }
}

fn check<V: Variant>() {
    let cfg = Config { g: 8, b: 3, t: 4, s: 3, node_capacity: 4096, ..Config::default() };
    let mut search = Search::<V>::new(cfg, 7);
    let mut model = CheckInputs { scratch: Scratch::new(), saw_scores: false };
    let mut steps = 0;
    for _ in 0..1000 {
        if !search.cycle(&mut model).is_empty() { steps += 1; }
        if steps == 50 { break; }
    }
    assert_eq!(steps, 50);
    assert!(model.saw_scores, "must cover scored positions as well as opening/padded rows");
}

#[test]
fn puct_buffer_keeps_scores_aligned() { check::<Puct>(); }
#[test]
fn exp3_buffer_keeps_scores_aligned() { check::<Exp3>(); }
