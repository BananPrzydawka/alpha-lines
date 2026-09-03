//! Replays a move file from the Python side and dumps the engine's whole observable state,
//! so `rust/xcheck/xcheck.py` can diff it against `main/game.py` field by field.
//!
//! Usage: verify --moves <moves.bin> --out-dir <dir> --tag <tag>
//!
//! The move file is a batch because the Python is; the engine is not, so this holds a
//! `Vec<Game>` and steps every entry. Binary formats are little-endian.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use alpha_lines_game::game::{
    board_index, square_index, Scratch, HEIGHT, HW, PLAYABLE_SQUARE, PLAYER_0_MARK,
    PLAYER_1_MARK, REMOVED_SQUARE, SQUARES, WIDTH,
};
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

/// Expand the engine's masks into the dense board-shaped ones the Python produces.
fn dense_masks(games: &[Game]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = games.len();
    let (mut m0, mut m1) = (vec![0.0f32; n * HW], vec![0.0f32; n * HW]);
    let (mut c0, mut c1) = (vec![0.0f32; n], vec![0.0f32; n]);
    for (g, game) in games.iter().enumerate() {
        for (player, dense, count) in
            [(0usize, &mut m0, &mut c0), (1usize, &mut m1, &mut c1)]
        {
            for (wi, &word) in game.legal_moves(player).iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let sq = (wi << 6) + w.trailing_zeros() as usize;
                    w &= w - 1;
                    dense[g * HW + board_index(sq)] = 1.0;
                    count[g] += 1.0;
                }
            }
        }
    }
    (m0, m1, c0, c1)
}

/// The engine's squares as a full board, which is what the Python dumps.
/// The Python's board encoding, which the engine does not share.
///
/// It covers the whole `HEIGHT x WIDTH` grid, so it needs a value for the unplayable half
/// that has no square index, and it orders the four real values differently. The engine's
/// order is a subset lattice on purpose (see `game::PLAYABLE_SQUARE`); this maps between
/// them at the one place the two meet.
const PY_NON_PLAYABLE: i8 = 0;
const PY_PLAYABLE: i8 = 1;
const PY_REMOVED: i8 = 2;
const PY_PLAYER_0: i8 = 3;
const PY_PLAYER_1: i8 = 4;

fn to_python(v: u8) -> i8 {
    match v {
        PLAYABLE_SQUARE => PY_PLAYABLE,
        PLAYER_0_MARK => PY_PLAYER_0,
        PLAYER_1_MARK => PY_PLAYER_1,
        REMOVED_SQUARE => PY_REMOVED,
        _ => unreachable!("square value {v}"),
    }
}

fn expand(cells: &[u8; SQUARES]) -> Vec<i8> {
    let mut board = vec![PY_NON_PLAYABLE; HW];
    for (sq, &v) in cells.iter().enumerate() {
        board[board_index(sq)] = to_python(v);
    }
    board
}

/// The square a move file's board index names, if the mask holds it. Needed to police a file
/// this driver did not produce.
fn legal_square(mask: [u64; 2], board: i32) -> Option<usize> {
    let sq = square_index(usize::try_from(board).ok()?)?;
    (mask[sq >> 6] >> (sq & 63) & 1 == 1).then_some(sq)
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

        let boards: Vec<i8> = games.iter().flat_map(|g| expand(&g.cells)).collect();
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
        // the engine only debug-asserts legality, so a move file is checked for real
        for (g, game) in games.iter_mut().enumerate() {
            if game.finished {
                continue;
            }
            let mut sq = [0usize; 2];
            for (player, slot) in sq.iter_mut().enumerate() {
                let board = moves.data[off + g * 2 + player];
                match legal_square(game.legal_moves(player), board) {
                    Some(k) => *slot = k,
                    None => {
                        eprintln!(
                            "step {step}, game {g}: illegal move {board} for player {player}"
                        );
                        std::process::exit(2);
                    }
                }
            }
            game.action_step(sq[0], sq[1], &mut scratch);
        }
        state.push(&games);
    }
    state.finish();

    // adopting the final boards has to land where playing there landed
    let adopted: Vec<Game> = games
        .iter()
        .map(|g| Game::from_cells(g.cells, &mut scratch))
        .collect();
    let mut re = StateWriter::create(out_dir.join(format!("reimport_{tag}.bin")), moves.n);
    re.push(&adopted);
    re.finish();

    println!("rust replay ok: {} games, {} steps", moves.n, moves.n_steps);
}
