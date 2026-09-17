//! Node stack: a flat array with a free list, plus an open-addressed index from Zobrist key
//! to node slot.
//!
//! Generic over the statistic block so PUCT and EXP3 share the storage. Nodes are allocated
//! on demand up to `node_capacity`, so a large capacity reserves address space without
//! touching it.
//!
//! The index stores the key beside the slot, so probing never reads a node. That matters
//! because each node is ~2 KB: probing the compact index avoids loading nodes just to
//! compare keys. Index entries are 16 bytes, four to a cache line.
//!
//! Deleting repairs the probe run instead of marking it. Linear probing puts a key at the
//! first free slot at or after its home, so emptying a slot can strand a later key that had
//! probed past it. Rather than leave a tombstone saying "keep walking", the delete pulls any
//! such key back into the hole. Nothing is left behind, so probe length depends only on how
//! many entries are live, and the index never needs rebuilding.
//!
//! The catch is that entries move, which rules out concurrent readers and pins the table to
//! linear probing. Both are fine here: the search is single-threaded by design, linear
//! probing is what the cache wants anyway, and nothing outside holds an index position —
//! callers hold node slots, which never move.

use crate::game::Game;
use crate::mcts::config::{Config, K, PENDING};

/// An index slot holding no node.
const EMPTY: u32 = u32::MAX;

/// One index slot: the key and the node it names. Holding the key here is what keeps probing
/// out of the node array, and it is also what lets a delete work out where a displaced entry
/// wanted to live.
#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    slot: u32,
}

impl Entry {
    const VACANT: Entry = Entry { key: 0, slot: EMPTY };
}

/// A node owns the position it stands for.
///
/// A child is built by cloning its parent's game and stepping it, which is the only time the
/// engine is touched: descending through nodes that already exist moves by Zobrist key alone.
/// Holding the game also means `terminal` and the terminal values are read off it rather than
/// stored beside it, so they cannot drift.
#[derive(Clone, Debug)]
pub struct Node<S> {
    pub key: u64,
    pub game: Game,
    pub flags: u8,
    /// Game ids that have used this node, oldest-overwritten once full.
    pub ids: [u16; K],
    pub id_count: u8,
    pub id_cursor: u8,
    pub stats: S,
}

impl<S> Node<S> {
    /// Evaluation requested but not yet returned.
    pub fn is_pending(&self) -> bool {
        self.flags & PENDING != 0
    }
    pub fn is_terminal(&self) -> bool {
        self.game.finished
    }
    /// Win/draw/loss per player. Only meaningful when [`Self::is_terminal`].
    pub fn terminal_values(&self) -> [f32; 2] {
        self.game.terminal_values()
    }
    pub fn ids(&self) -> &[u16] {
        &self.ids[..self.id_count as usize]
    }
}


pub struct Arena<S> {
    nodes: Vec<Node<S>>,
    free: Vec<u32>,
    index: Vec<Entry>,
    mask: usize,
    capacity: usize,
    live: usize,
}

impl<S> Arena<S> {
    pub fn new(cfg: &Config) -> Self {
        let capacity = cfg.node_capacity;
        // power of two, at least 2x capacity, so the table stays under half full
        let slots = (capacity.max(1) * 2).next_power_of_two();
        Arena {
            nodes: Vec::with_capacity(capacity.min(1 << 20)),
            free: Vec::new(),
            index: vec![Entry::VACANT; slots],
            mask: slots - 1,
            capacity,
            live: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.live
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn free_depth(&self) -> usize {
        self.free.len()
    }
    /// Slots ever allocated, live or free — the range the sweep walks.
    pub fn slot_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, i: u32) -> &Node<S> {
        &self.nodes[i as usize]
    }
    pub fn node_mut(&mut self, i: u32) -> &mut Node<S> {
        &mut self.nodes[i as usize]
    }

    /// The slot holding `key`, or `None`.
    pub fn get(&self, key: u64) -> Option<u32> {
        let mut b = key as usize & self.mask;
        loop {
            let e = self.index[b];
            if e.slot == EMPTY {
                return None;
            }
            if e.key == key {
                return Some(e.slot);
            }
            b = (b + 1) & self.mask;
        }
    }

    /// Drop a node: repair its probe run and return the slot to the free list.
    pub fn remove(&mut self, i: u32) {
        let key = self.nodes[i as usize].key;
        if !self.index_remove(key, i) {
            return; // not indexed; nothing to do
        }
        self.nodes[i as usize].id_count = 0;
        self.nodes[i as usize].key = 0;
        self.free.push(i);
        self.live -= 1;
    }

    /// Record that game `id` has used node `i`. Once the list is full, the new id
    /// displaces the oldest one, round-robin.
    pub fn touch(&mut self, i: u32, id: u16) {
        let n = &mut self.nodes[i as usize];
        if n.ids[..n.id_count as usize].contains(&id) {
            return;
        }
        if (n.id_count as usize) < K {
            n.ids[n.id_count as usize] = id;
            n.id_count += 1;
        } else {
            n.ids[n.id_cursor as usize] = id;
            n.id_cursor = ((n.id_cursor as usize + 1) % K) as u8;
        }
    }

    /// Drop `id` from node `i`, compacting the list. Returns whether it was listed.
    ///
    /// Deliberately not "the node is now unused": that question is [`Self::is_unused`], and
    /// conflating the two makes dropping an absent id look like it emptied the node.
    pub fn drop_id(&mut self, i: u32, id: u16) -> bool {
        let n = &mut self.nodes[i as usize];
        let count = n.id_count as usize;
        let Some(at) = n.ids[..count].iter().position(|&x| x == id) else {
            return false;
        };
        n.ids[at] = n.ids[count - 1];
        n.id_count -= 1;
        if n.id_cursor as usize >= n.id_count as usize {
            n.id_cursor = 0;
        }
        true
    }

    /// No game references this node, so the sweep may delete it.
    pub fn is_unused(&self, i: u32) -> bool {
        self.nodes[i as usize].id_count == 0
    }

    /// Every live slot, for the sweep.
    pub fn live_slots(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.nodes.len() as u32).filter(|&i| self.nodes[i as usize].id_count > 0)
    }

    /// Drop every node. Used on a network update (spec section 11).
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.index.iter_mut().for_each(|e| *e = Entry::VACANT);
        self.live = 0;
    }

    fn index_insert(&mut self, key: u64, slot: u32) {
        let mut b = key as usize & self.mask;
        while self.index[b].slot != EMPTY {
            b = (b + 1) & self.mask;
        }
        self.index[b] = Entry { key, slot };
    }

    /// Take `key` out of the index and close the gap behind it. Returns whether it was there.
    ///
    /// Walking forward from the hole, an entry can be pulled back into it exactly when it is
    /// already displaced at least as far as the hole is behind it — that is, when its home
    /// slot is at or before the hole in probe order. Anything closer to home than that is
    /// holding its own run together and must stay. Each move makes the vacated slot the new
    /// hole, and the scan ends at the first empty slot, which nothing can have probed past.
    fn index_remove(&mut self, key: u64, slot: u32) -> bool {
        let mask = self.mask;
        let mut hole = key as usize & mask;
        loop {
            let e = self.index[hole];
            if e.slot == EMPTY {
                return false;
            }
            if e.slot == slot && e.key == key {
                break;
            }
            hole = (hole + 1) & mask;
        }

        let mut j = hole;
        loop {
            j = (j + 1) & mask;
            let e = self.index[j];
            if e.slot == EMPTY {
                break;
            }
            let home = e.key as usize & mask;
            if j.wrapping_sub(home) & mask >= j.wrapping_sub(hole) & mask {
                self.index[hole] = e;
                hole = j;
            }
        }
        self.index[hole] = Entry::VACANT;
        true
    }
}

/// `Default` is needed only to blank a new node's statistics.
impl<S: Default> Arena<S> {
    /// Add a node for `key`. The caller must have checked it is absent.
    ///
    /// Returns `None` when the stack is full, which the caller has to treat as a search
    /// budget being exhausted rather than an error.
    pub fn insert(&mut self, key: u64, game: Game, flags: u8, id: u16) -> Option<u32> {
        let slot = match self.free.pop() {
            Some(s) => {
                let n = &mut self.nodes[s as usize];
                n.key = key;
                n.game = game;
                n.flags = flags;
                n.ids[0] = id;
                n.id_count = 1;
                n.id_cursor = 0;
                n.stats = S::default();
                s
            }
            None => {
                if self.nodes.len() >= self.capacity {
                    return None;
                }
                let mut ids = [0u16; K];
                ids[0] = id;
                self.nodes.push(Node {
                    key,
                    game,
                    flags,
                    ids,
                    id_count: 1,
                    id_cursor: 0,
                    stats: S::default(),
                });
                (self.nodes.len() - 1) as u32
            }
        };
        self.index_insert(key, slot);
        self.live += 1;
        Some(slot)
    }
}
