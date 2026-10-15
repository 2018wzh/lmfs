//! Store the model and apply ranked byte pair merges.
use crate::lexer::{Pieces, Specials};
use rayon::prelude::*;
use rustc_hash::{FxHashMap as Map, FxHashSet as Set};
use std::{cmp::Reverse, collections::BinaryHeap};
pub type Rule = (u16, u16, u16);
const MISS: u32 = u32::MAX;
const NONE: usize = usize::MAX;
pub const CACHE_CAP: usize = 131072;
#[inline]
pub fn key(a: u16, b: u16) -> u32 {
    ((a as u32) << 16) | b as u32
}

pub struct Model {
    pub vocab: Vec<Vec<u8>>,
    pub public: Vec<u64>,
    pub identity: bool,
    pub public_lookup: Map<u64, usize>,
    pub specials: Specials,
    pub arena: Vec<u8>,
    pub offsets: Vec<usize>,
    dense: Box<[u32]>,
    sparse: Map<u32, u32>,
    direct: Map<Box<[u8]>, u16>,
}
impl Model {
    pub fn new(
        vocab: Vec<Vec<u8>>,
        rules: Vec<Rule>,
        specials: Vec<(String, u16)>,
        public: Vec<u64>,
    ) -> Result<Self, String> {
        let n = vocab.len();
        let public_lookup: Map<u64, usize> =
            public.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let special_ids: Set<u16> = specials.iter().map(|&(_, id)| id).collect();
        let mut dense = vec![MISS; 65536];
        let mut sparse = Map::default();
        for (rank, &(a, b, c)) in rules.iter().enumerate() {
            let code = ((rank as u32) << 16) | c as u32;
            if a < 256 && b < 256 {
                dense[a as usize * 256 + b as usize] = code;
            } else {
                sparse.insert(key(a, b), code);
            }
        }
        let mut arena = Vec::new();
        let mut offsets = vec![0];
        for v in &vocab {
            arena.extend_from_slice(v);
            offsets.push(arena.len());
        }
        let identity = public.iter().enumerate().all(|(i, &id)| id == i as u64);
        let mut model = Self {
            vocab,
            public,
            identity,
            public_lookup,
            specials: Specials::new(specials)?,
            arena,
            offsets,
            dense: dense.into_boxed_slice(),
            sparse,
            direct: Map::default(),
        };
        let mut scratch = Scratch::default();
        for i in 0..n {
            if special_ids.contains(&(i as u16)) {
                continue;
            }
            let raw = model.vocab[i].clone();
            merge(&raw, &model, 128, &mut scratch);
            if scratch.ids.len() == 1 {
                model.direct.insert(raw.into_boxed_slice(), scratch.ids[0]);
            }
        }
        Ok(model)
    }
    #[inline]
    fn lookup(&self, a: u16, b: u16) -> u32 {
        if a < 256 && b < 256 {
            self.dense[a as usize * 256 + b as usize]
        } else {
            *self.sparse.get(&key(a, b)).unwrap_or(&MISS)
        }
    }
    #[inline]
    pub fn index(&self, id: u64) -> Option<usize> {
        if self.identity {
            usize::try_from(id).ok().filter(|&i| i < self.vocab.len())
        } else {
            self.public_lookup.get(&id).copied()
        }
    }
}
#[derive(Clone)]
struct Node {
    token: u16,
    prev: usize,
    next: usize,
    version: u32,
    alive: bool,
}
type Candidate = Reverse<(u16, usize, usize, u32, u32, u16)>;
#[derive(Default)]
pub struct Scratch {
    pub ids: Vec<u16>,
    edges: Vec<u32>,
    nodes: Vec<Node>,
    heap: BinaryHeap<Candidate>,
}
fn enqueue(m: &Model, n: &[Node], left: usize, h: &mut BinaryHeap<Candidate>) {
    if left == NONE || !n[left].alive {
        return;
    }
    let right = n[left].next;
    if right == NONE {
        return;
    }
    let v = m.lookup(n[left].token, n[right].token);
    if v != MISS {
        h.push(Reverse((
            (v >> 16) as u16,
            left,
            right,
            n[left].version,
            n[right].version,
            v as u16,
        )));
    }
}
pub fn merge(raw: &[u8], m: &Model, threshold: usize, s: &mut Scratch) {
    s.ids.clear();
    // Scan short pieces directly. Use a heap for long pieces.
    if raw.len() <= threshold {
        s.ids.extend(raw.iter().map(|&b| b as u16));
        s.edges.clear();
        s.edges
            .extend(s.ids.windows(2).map(|p| m.lookup(p[0], p[1])));
        while !s.edges.is_empty() {
            let mut best = 0;
            for i in 1..s.edges.len() {
                if s.edges[i] < s.edges[best] {
                    best = i;
                }
            }
            if s.edges[best] == MISS {
                break;
            }
            s.ids[best] = s.edges[best] as u16;
            s.ids.remove(best + 1);
            s.edges.remove(best);
            if best > 0 {
                s.edges[best - 1] = m.lookup(s.ids[best - 1], s.ids[best]);
            }
            if best < s.edges.len() {
                s.edges[best] = m.lookup(s.ids[best], s.ids[best + 1]);
            }
        }
    } else {
        s.nodes.clear();
        s.heap.clear();
        for (i, &b) in raw.iter().enumerate() {
            s.nodes.push(Node {
                token: b as u16,
                prev: if i == 0 { NONE } else { i - 1 },
                next: if i + 1 == raw.len() { NONE } else { i + 1 },
                version: 0,
                alive: true,
            });
        }
        for i in 0..s.nodes.len() {
            enqueue(m, &s.nodes, i, &mut s.heap);
        }
        while let Some(Reverse((_, left, right, lv, rv, id))) = s.heap.pop() {
            if !s.nodes[left].alive
                || !s.nodes[right].alive
                || s.nodes[left].next != right
                || s.nodes[left].version != lv
                || s.nodes[right].version != rv
            {
                continue;
            }
            let prev = s.nodes[left].prev;
            let next = s.nodes[right].next;
            s.nodes[left].token = id;
            s.nodes[left].next = next;
            s.nodes[left].version += 1;
            s.nodes[right].alive = false;
            s.nodes[right].version += 1;
            if next != NONE {
                s.nodes[next].prev = left;
            }
            enqueue(m, &s.nodes, prev, &mut s.heap);
            enqueue(m, &s.nodes, left, &mut s.heap);
        }
        if !s.nodes.is_empty() {
            let mut i = 0;
            while i != NONE {
                s.ids.push(s.nodes[i].token);
                i = s.nodes[i].next;
            }
        }
    }
}
// Use one merge loop for both cache layouts.
fn encode_part<const UNIFIED: bool>(text: &str, m: &Model) -> Vec<u16> {
    let mut out = Vec::with_capacity(text.len() / 3);
    let mut s = Scratch::default();
    let mut values = Vec::<u16>::new();
    let mut cache: Map<&[u8], u64> = if UNIFIED {
        Map::with_capacity_and_hasher(m.direct.len() + 65536, Default::default())
    } else {
        Map::default()
    };
    if UNIFIED {
        for (b, &id) in &m.direct {
            cache.insert(b.as_ref(), id as u64);
        }
    }
    let mut added = 0usize;
    let threshold = if UNIFIED { 64 } else { 128 };
    m.specials.visit(text, |part, special| {
        if let Some(id) = special {
            out.push(id);
            return;
        }
        for raw in Pieces::new(part) {
            if raw.len() == 1 {
                out.push(raw[0] as u16);
                continue;
            }
            if !UNIFIED {
                if let Some(&id) = m.direct.get(raw) {
                    out.push(id);
                    continue;
                }
            }
            if let Some(&code) = cache.get(raw) {
                let n = (code >> 32) as usize;
                if n == 0 {
                    out.push(code as u16);
                } else {
                    let start = code as u32 as usize;
                    out.extend_from_slice(&values[start..start + n]);
                }
                continue;
            }
            merge(raw, m, threshold, &mut s);
            out.extend_from_slice(&s.ids);
            if added < CACHE_CAP && raw.len() <= 256 && values.len() <= u32::MAX as usize {
                let code = if s.ids.len() == 1 {
                    s.ids[0] as u64
                } else {
                    let v = ((s.ids.len() as u64) << 32) | values.len() as u64;
                    values.extend_from_slice(&s.ids);
                    v
                };
                cache.insert(raw, code);
                added += 1;
            }
        }
    });
    out
}
pub struct Engine {
    pub model: Model,
    pub pool: Option<rayon::ThreadPool>,
}
impl Engine {
    pub fn new(model: Model, threads: usize) -> Result<Self, String> {
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let threads = if threads == 0 { available } else { threads };
        let pool = if threads == 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("bpe-worker-{i}"))
                    .build()
                    .map_err(|e| e.to_string())?,
            )
        };
        Ok(Self { model, pool })
    }
    pub fn encode(&self, text: &str) -> Vec<u16> {
        if let Some(pool) = &self.pool {
            let chunks = self.model.specials.chunks(text, 4 << 20);
            if chunks.len() < 2 {
                return encode_part::<false>(text, &self.model);
            }
            let parts: Vec<Vec<u16>> = pool.install(|| {
                chunks
                    .par_iter()
                    .map(|s| encode_part::<true>(s, &self.model))
                    .collect()
            });
            let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
            for p in parts {
                out.extend_from_slice(&p);
            }
            out
        } else {
            encode_part::<false>(text, &self.model)
        }
    }
}
