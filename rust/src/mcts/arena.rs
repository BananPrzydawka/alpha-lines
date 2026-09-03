//! Node stack: a flat array with a free list, plus an open-addressed index from Zobrist key
//! to node slot.
//!
//! Generic over the statistic block so PUCT and EXP3 share the storage. Nodes are allocated
//! on demand up to `node_capacity`, so a large capacity reserves address space without
//! touching it.

use crate::game::SQUARES;
use crate::mcts::config::{Config, K, PENDING, TERMINAL};

/// An index slot holding no node.
const EMPTY: u32 = u32::MAX;
/// An index slot whose node was deleted; probing continues through it.
const TOMB: u32 = u32::MAX - 1;

#[derive(Clone, Debug)]
pub struct Node<S> {
    pub key: u64,
    pub board: [u8; SQUARES],
    pub flags: u8,
    /// Valid only when `flags & TERMINAL`.
    pub terminal_values: [f32; 2],
    /// Game ids that have used this node, oldest-overwritten once full.
    pub ids: [u16; K],
    pub id_count: u8,
    pub id_cursor: u8,
    pub stats: S,
}

impl<S> Node<S> {
    pub fn is_pending(&self) -> bool {
        self.flags & PENDING != 0
    }
    pub fn is_terminal(&self) -> bool {
        self.flags & TERMINAL != 0
    }
    pub fn ids(&self) -> &[u16] {
        &self.ids[..self.id_count as usize]
    }
}

/// What an id write did, for the diagnostics in spec section 9.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IdWrite {
    /// The game was already listed.
    Present,
    /// Appended into a free slot.
    Appended,
    /// The list was full; this id displaced another, which is named.
    Displaced(u16),
}

pub struct Arena<S> {
    nodes: Vec<Node<S>>,
    free: Vec<u32>,
    index: Vec<u32>,
    mask: usize,
    capacity: usize,
    live: usize,
    tombstones: usize,
    tombstone_limit: usize,
    /// Diagnostics: id writes that displaced an existing id.
    pub overflows: u64,
}

impl<S: Default + Clone> Arena<S> {
    pub fn new(cfg: &Config) -> Self {
        let capacity = cfg.node_capacity;
        // power of two, at least 2x capacity, so the table stays under half full
        let slots = (capacity.max(1) * 2).next_power_of_two();
        Arena {
            nodes: Vec::with_capacity(capacity.min(1 << 20)),
            free: Vec::new(),
            index: vec![EMPTY; slots],
            mask: slots - 1,
            capacity,
            live: 0,
            tombstones: 0,
            tombstone_limit: (slots as f32 * cfg.tombstone_ratio) as usize,
            overflows: 0,
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
    pub fn tombstones(&self) -> usize {
        self.tombstones
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
            match self.index[b] {
                EMPTY => return None,
                TOMB => {}
                n if self.nodes[n as usize].key == key => return Some(n),
                _ => {}
            }
            b = (b + 1) & self.mask;
        }
    }

    /// Add a node for `key`. The caller must have checked it is absent.
    ///
    /// Returns `None` when the stack is full, which the caller has to treat as a search
    /// budget being exhausted rather than an error.
    pub fn insert(&mut self, key: u64, board: [u8; SQUARES], flags: u8, id: u16) -> Option<u32> {
        let slot = match self.free.pop() {
            Some(s) => {
                let n = &mut self.nodes[s as usize];
                n.key = key;
                n.board = board;
                n.flags = flags;
                n.terminal_values = [0.0; 2];
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
                    board,
                    flags,
                    terminal_values: [0.0; 2],
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

    /// Drop a node: tombstone its index entry and return the slot to the free list.
    pub fn remove(&mut self, i: u32) {
        let key = self.nodes[i as usize].key;
        let mut b = key as usize & self.mask;
        loop {
            match self.index[b] {
                EMPTY => return, // not indexed; nothing to do
                n if n == i => {
                    self.index[b] = TOMB;
                    self.tombstones += 1;
                    break;
                }
                _ => {}
            }
            b = (b + 1) & self.mask;
        }
        self.nodes[i as usize].id_count = 0;
        self.nodes[i as usize].key = 0;
        self.free.push(i);
        self.live -= 1;
        if self.tombstones > self.tombstone_limit {
            self.rehash();
        }
    }

    /// Record that game `id` has used node `i`, per the spec's id-write rule.
    pub fn touch(&mut self, i: u32, id: u16) -> IdWrite {
        let n = &mut self.nodes[i as usize];
        if n.ids[..n.id_count as usize].contains(&id) {
            return IdWrite::Present;
        }
        if (n.id_count as usize) < K {
            n.ids[n.id_count as usize] = id;
            n.id_count += 1;
            IdWrite::Appended
        } else {
            let displaced = n.ids[n.id_cursor as usize];
            n.ids[n.id_cursor as usize] = id;
            n.id_cursor = ((n.id_cursor as usize + 1) % K) as u8;
            self.overflows += 1;
            IdWrite::Displaced(displaced)
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
        self.index.iter_mut().for_each(|s| *s = EMPTY);
        self.live = 0;
        self.tombstones = 0;
    }

    fn index_insert(&mut self, key: u64, slot: u32) {
        let mut b = key as usize & self.mask;
        loop {
            let e = self.index[b];
            if e == EMPTY || e == TOMB {
                if e == TOMB {
                    self.tombstones -= 1;
                }
                self.index[b] = slot;
                return;
            }
            b = (b + 1) & self.mask;
        }
    }

    /// Rebuild the index, clearing tombstones. Node slots do not move.
    fn rehash(&mut self) {
        self.index.iter_mut().for_each(|s| *s = EMPTY);
        self.tombstones = 0;
        for i in 0..self.nodes.len() as u32 {
            if self.nodes[i as usize].id_count > 0 {
                let key = self.nodes[i as usize].key;
                self.index_insert(key, i);
            }
        }
    }
}
