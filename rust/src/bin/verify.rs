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
//! Binary formats are little-endian throughout.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use alpha_lines_game::incremental::{HEIGHT, HW, WIDTH};
use alpha_lines_game::IncrementalGame;

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
fn dense_masks(inc: &IncrementalGame) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let half = WIDTH / 2;
    let (mut m0, mut m1) = (vec![0.0f32; inc.n * HW], vec![0.0f32; inc.n * HW]);
    let (mut c0, mut c1) = (vec![0.0f32; inc.n], vec![0.0f32; inc.n]);
    for g in 0..inc.n {
        let bits = inc.legal_bits(g);
        let first = inc.move_counts[g] == 0;
        for i in 0..HW {
            let k = i >> 1;
            if i % 2 != (i / WIDTH) % 2 || bits[k >> 6] >> (k & 63) & 1 == 0 {
                continue;
            }
            let c = i % WIDTH;
            if !(first && c >= half) {
                m0[g * HW + i] = 1.0;
                c0[g] += 1.0;
            }
            if !(first && c < half) {
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

    fn push(&mut self, game: &IncrementalGame) {
        assert_eq!(game.n, self.n, "record batch size mismatch");
        let (m0, m1, c0, c1) = dense_masks(game);
        let scores: Vec<f32> = game.scores.iter().map(|&s| s as f32).collect();

        write_i8(&mut self.w, &game.boards);
        write_f32(&mut self.w, &scores);
        write_i32(&mut self.w, &game.move_counts);
        let fin: Vec<u8> = game.finished.iter().map(|&f| f as u8).collect();
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
    let mut game = IncrementalGame::new(moves.n, 0);
    let mut state = StateWriter::create(out_dir.join(format!("state_{tag}.bin")), moves.n);

    // record 0 is the initial state, before any move
    state.push(&game);

    for step in 0..moves.n_steps {
        let off = step * moves.n * 2;
        let idx_0: Vec<i64> = (0..moves.n).map(|g| moves.data[off + g * 2] as i64).collect();
        let idx_1: Vec<i64> = (0..moves.n).map(|g| moves.data[off + g * 2 + 1] as i64).collect();
        if let Err(e) = game.action_step(&idx_0, &idx_1) {
            eprintln!("action_step failed at step {step}: {e}");
            std::process::exit(2);
        }
        state.push(&game);
    }
    state.finish();

    // Adopting the final boards from scratch has to land on the same state the engine
    // reached by playing there, which is what checks `from_state` against the Python too.
    let adopted = IncrementalGame::from_state(game.boards.clone(), 0);
    let mut re = StateWriter::create(out_dir.join(format!("reimport_{tag}.bin")), moves.n);
    re.push(&adopted);
    re.finish();

    println!("rust replay ok: {} games, {} steps", moves.n, moves.n_steps);
}
