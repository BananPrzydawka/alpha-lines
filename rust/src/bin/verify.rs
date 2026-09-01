//! Cross-check driver: replays a move file produced by the Python side and dumps the
//! engine's whole observable state, so `rust/xcheck/xcheck.py` can diff it against
//! `main/game.py` field by field.
//!
//! Usage:
//!   verify --moves <moves.bin> --out-dir <dir> --tag <tag>
//!
//! The dump is boards, scores, move counts, terminal flags and legality — everything the
//! engine owns. The Python's encoding and rendering have no counterpart here, because the
//! engine does not implement them; they are the caller's job.
//!
//! The move file is a batch — N games advanced together — because the Python it is checking
//! against is. The engine is not: it plays one board at a time, so this holds a `Vec<Game>`
//! and steps every entry. That is the whole of what "batched" means here now.
//!
//! Binary formats are little-endian throughout.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use alpha_lines_game::game::{Scratch, HEIGHT, HW, WIDTH};
use alpha_lines_game::Game;

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

/// Expand the engine's 80-bit legality words into the dense masks the Python produces, so
/// the two can be compared at all. The engine has no reason to do this itself; a caller that
/// needs a mask for a policy network builds it once, in the shape that network wants.
fn dense_masks(games: &[Game]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = games.len();
    let (mut m0, mut m1) = (vec![0.0f32; n * HW], vec![0.0f32; n * HW]);
    let (mut c0, mut c1) = (vec![0.0f32; n], vec![0.0f32; n]);
    for (g, game) in games.iter().enumerate() {
        for i in 0..HW {
            if game.is_legal(i, 0) {
                m0[g * HW + i] = 1.0;
                c0[g] += 1.0;
            }
            if game.is_legal(i, 1) {
                m1[g * HW + i] = 1.0;
                c1[g] += 1.0;
            }
        }
    }
    (m0, m1, c0, c1)
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

    fn push(&mut self, games: &[Game]) {
        assert_eq!(games.len(), self.n, "record batch size mismatch");
        let (m0, m1, c0, c1) = dense_masks(games);

        let boards: Vec<i8> = games.iter().flat_map(|g| g.cells).collect();
        let scores: Vec<f32> = games
            .iter()
            .flat_map(|g| [g.scores[0] as f32, g.scores[1] as f32])
            .collect();
        let counts: Vec<i32> = games.iter().map(|g| g.move_count as i32).collect();
        let fin: Vec<u8> = games.iter().map(|g| g.finished as u8).collect();

        write_i8(&mut self.w, &boards);
        write_f32(&mut self.w, &scores);
        write_i32(&mut self.w, &counts);
        self.w.write_all(&fin).unwrap();
        write_f32(&mut self.w, &m0);
        write_f32(&mut self.w, &m1);
        write_f32(&mut self.w, &c0);
        write_f32(&mut self.w, &c1);
        self.records += 1;
    }

    fn finish(mut self) {
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

    let moves = read_moves(&moves_path);
    let mut games: Vec<Game> = (0..moves.n).map(|_| Game::new()).collect();
    let mut scratch = Scratch::new();
    let mut state = StateWriter::create(out_dir.join(format!("state_{tag}.bin")), moves.n);

    // record 0 is the initial state, before any move
    state.push(&games);

    for step in 0..moves.n_steps {
        let off = step * moves.n * 2;
        // The engine only debug-asserts legality — the caller is expected to have picked out
        // of `legal_moves`. Here the moves come from a file, so they are checked for real:
        // this driver exists to catch disagreements, and a bad move file must be reported
        // rather than replayed into undefined behaviour.
        for (g, game) in games.iter().enumerate() {
            if game.finished {
                continue;
            }
            let (i0, i1) = (moves.data[off + g * 2], moves.data[off + g * 2 + 1]);
            for (player, i) in [(0usize, i0), (1usize, i1)] {
                if i < 0 || !game.is_legal(i as usize, player) {
                    eprintln!("step {step}, game {g}: illegal move {i} for player {player}");
                    std::process::exit(2);
                }
            }
        }
        for (g, game) in games.iter_mut().enumerate() {
            if game.finished {
                continue;
            }
            let i0 = moves.data[off + g * 2] as usize;
            let i1 = moves.data[off + g * 2 + 1] as usize;
            game.apply(i0, i1, &mut scratch);
        }
        state.push(&games);
    }
    state.finish();

    // Adopting the final boards from scratch has to land on the same state the engine
    // reached by playing there, which is what checks `from_cells` against the Python too.
    let adopted: Vec<Game> = games
        .iter()
        .map(|g| Game::from_cells(g.cells, &mut scratch))
        .collect();
    let mut re = StateWriter::create(out_dir.join(format!("reimport_{tag}.bin")), moves.n);
    re.push(&adopted);
    re.finish();

    println!("rust replay ok: {} games, {} steps", moves.n, moves.n_steps);
}
