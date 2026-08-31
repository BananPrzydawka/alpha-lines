//! Cross-check driver: replays a move file produced by the Python side and dumps every
//! observable piece of state, so `rust/xcheck/xcheck.py` can diff the two implementations
//! field by field.
//!
//! Usage:
//!   verify --moves <moves.bin> --out-dir <dir> --tag <tag> [--impl reference|incremental]
//!
//! `--impl incremental` runs the same replay through the level-based incremental scorer,
//! so it is checked against the Python directly and not only against the reference port.
//!
//! Binary formats (little-endian throughout) are documented in `rust/xcheck/README.md`.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

const STATE_MAGIC: &[u8; 4] = b"ALST";
const MOVES_MAGIC: &[u8; 4] = b"ALMV";
const VERSION: u32 = 1;

struct Moves {
    n: usize,
    n_steps: usize,
    /// n_steps * n * 2, flattened (step, game, player)
    data: Vec<i32>,
}

fn read_moves(path: &PathBuf) -> Moves {
    let mut buf = Vec::new();
    File::open(path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()))
        .read_to_end(&mut buf)
        .expect("read moves");
    assert_eq!(&buf[0..4], MOVES_MAGIC, "bad moves magic");
    let u32_at = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
    assert_eq!(u32_at(4), VERSION, "bad moves version");
    let n = u32_at(8) as usize;
    let h = u32_at(12) as usize;
    let w = u32_at(16) as usize;
    let n_steps = u32_at(20) as usize;
    assert_eq!((h, w), (HEIGHT, WIDTH), "board geometry mismatch");
    let body = &buf[24..];
    let count = n_steps * n * 2;
    assert_eq!(body.len(), count * 4, "moves body length mismatch");
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        data.push(i32::from_le_bytes(body[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    Moves { n, n_steps, data }
}

struct StateWriter {
    w: BufWriter<File>,
    records: u32,
    path: PathBuf,
    n: usize,
}

impl StateWriter {
    fn create(path: PathBuf, n: usize) -> Self {
        let mut w = BufWriter::new(File::create(&path).expect("create state file"));
        w.write_all(STATE_MAGIC).unwrap();
        for v in [VERSION, n as u32, HEIGHT as u32, WIDTH as u32, 0u32] {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        StateWriter { w, records: 0, path, n }
    }

    fn push(&mut self, game: &BatchedLinesGame) {
        assert_eq!(game.n, self.n, "record batch size mismatch");
        let (m0, m1, c0, c1) = game.get_legal_masks();
        let e0 = game.get_encoded_states(0);
        let e1 = game.get_encoded_states(1);

        write_i8(&mut self.w, &game.boards);
        write_f32(&mut self.w, &game.scores);
        write_i32(&mut self.w, &game.move_counts);
        let fin: Vec<u8> = game.finished.iter().map(|&f| f as u8).collect();
        self.w.write_all(&fin).unwrap();
        write_f32(&mut self.w, &m0);
        write_f32(&mut self.w, &m1);
        write_f32(&mut self.w, &c0);
        write_f32(&mut self.w, &c1);
        write_f32(&mut self.w, &e0);
        write_f32(&mut self.w, &e1);
        self.records += 1;
    }

    /// Optional trailer: `1u8` + the two terminal-outcome arrays, or `0u8`.
    fn finish(mut self, outcomes: Option<(Vec<i64>, Vec<i64>)>) {
        match outcomes {
            Some((p0, p1)) => {
                self.w.write_all(&[1u8]).unwrap();
                write_i64(&mut self.w, &p0);
                write_i64(&mut self.w, &p1);
            }
            None => self.w.write_all(&[0u8]).unwrap(),
        }
        self.w.flush().unwrap();
        drop(self.w);
        // patch the record count into the header
        let mut f = std::fs::OpenOptions::new().write(true).open(&self.path).unwrap();
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(20)).unwrap();
        f.write_all(&self.records.to_le_bytes()).unwrap();
        f.flush().unwrap();
    }
}

fn write_i8<W: Write>(w: &mut W, v: &[i8]) {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
    w.write_all(bytes).unwrap();
}
fn write_f32<W: Write>(w: &mut W, v: &[f32]) {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf).unwrap();
}
fn write_i32<W: Write>(w: &mut W, v: &[i32]) {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf).unwrap();
}
fn write_i64<W: Write>(w: &mut W, v: &[i64]) {
    let mut buf = Vec::with_capacity(v.len() * 8);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf).unwrap();
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{key} needs a value"))
            .clone()
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let moves_path = PathBuf::from(arg(&args, "--moves").expect("--moves is required"));
    let out_dir = PathBuf::from(arg(&args, "--out-dir").expect("--out-dir is required"));
    let tag = arg(&args, "--tag").unwrap_or_else(|| "rs".to_string());
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let implementation = arg(&args, "--impl").unwrap_or_else(|| "reference".to_string());
    let incremental = match implementation.as_str() {
        "reference" => false,
        "incremental" => true,
        other => panic!("unknown --impl {other:?}; expected reference or incremental"),
    };

    let moves = read_moves(&moves_path);
    // Both paths dump through the reference-shaped state, so the encoding, rendering and
    // terminal-outcome code under test is identical and only the scorer differs.
    let mut driver = Driver::new(incremental, moves.n);
    let mut game = driver.snapshot();

    let mut state = StateWriter::create(out_dir.join(format!("state_{tag}.bin")), moves.n);
    let mut prints_all = String::new();

    // record 0 is the initial state, before any move
    state.push(&game);
    prints_all.push_str(&format!("=== STEP 0 ===\n"));
    prints_all.push_str(&game.format_state(None, 0));
    let prints_step0_p0 = game.format_state(None, 0);
    let prints_step0_p1 = game.format_state(None, 1);

    for step in 0..moves.n_steps {
        let off = step * moves.n * 2;
        let idx_0: Vec<i64> = (0..moves.n).map(|g| moves.data[off + g * 2] as i64).collect();
        let idx_1: Vec<i64> = (0..moves.n).map(|g| moves.data[off + g * 2 + 1] as i64).collect();
        if let Err(e) = driver.action_step(&idx_0, &idx_1) {
            eprintln!("action_step failed at step {step}: {e}");
            std::process::exit(2);
        }
        game = driver.snapshot();
        state.push(&game);
        prints_all.push_str(&format!("=== STEP {} ===\n", step + 1));
        prints_all.push_str(&game.format_state(None, 0));
    }

    let prints_final_p0 = game.format_state(None, 0);
    let prints_final_p1 = game.format_state(None, 1);

    let outcomes = if game.finished.iter().all(|&f| f) {
        Some(game.get_terminal_outcomes())
    } else {
        None
    };
    state.finish(outcomes);

    std::fs::write(out_dir.join(format!("prints_all_{tag}.txt")), &prints_all).unwrap();
    std::fs::write(out_dir.join(format!("prints_step0_p0_{tag}.txt")), &prints_step0_p0).unwrap();
    std::fs::write(out_dir.join(format!("prints_step0_p1_{tag}.txt")), &prints_step0_p1).unwrap();
    std::fs::write(out_dir.join(format!("prints_final_p0_{tag}.txt")), &prints_final_p0).unwrap();
    std::fs::write(out_dir.join(format!("prints_final_p1_{tag}.txt")), &prints_final_p1).unwrap();

    // round-trip: import_prints on each of the four rendered texts
    let mut reimport = StateWriter::create(out_dir.join(format!("reimport_{tag}.bin")), moves.n);
    for (text, player) in [
        (&prints_step0_p0, 0usize),
        (&prints_step0_p1, 1usize),
        (&prints_final_p0, 0usize),
        (&prints_final_p1, 1usize),
    ] {
        let g = BatchedLinesGame::import_prints(text, player, 0)
            .unwrap_or_else(|e| panic!("import_prints(player={player}) failed: {e}"));
        // For the incremental scorer, re-derive levels and scores from the imported board
        // alone, so `from_state` is verified against the Python too.
        let g = if incremental {
            IncrementalGame::from_state(g.boards, g.move_counts, g.finished, 0).to_reference()
        } else {
            g
        };
        reimport.push(&g);
    }
    reimport.finish(None);

    println!("rust replay ok ({}): {} games, {} steps", implementation, moves.n, moves.n_steps);
}

/// Runs the replay through whichever scorer was selected, exposing a single shape to the
/// dumping code above.
enum Driver {
    Reference(BatchedLinesGame),
    Incremental(IncrementalGame),
}

impl Driver {
    fn new(incremental: bool, n: usize) -> Self {
        if incremental {
            Driver::Incremental(IncrementalGame::new(n, 0))
        } else {
            Driver::Reference(BatchedLinesGame::new(n, 0))
        }
    }

    fn action_step(&mut self, idx_0: &[i64], idx_1: &[i64]) -> Result<(), String> {
        match self {
            Driver::Reference(g) => g.action_step(idx_0, idx_1),
            Driver::Incremental(g) => g.action_step(idx_0, idx_1),
        }
    }

    fn snapshot(&self) -> BatchedLinesGame {
        match self {
            Driver::Reference(g) => g.clone(),
            Driver::Incremental(g) => g.to_reference(),
        }
    }
}
