use crate::{
    core::{Rule, key},
    io::read_chunks,
    lexer::{Pieces, Specials},
};
use rustc_hash::{FxHashMap as Map, FxHashSet as Set};
use serde::{Deserialize, Serialize};
use std::{collections::BinaryHeap, sync::Arc};
#[derive(Serialize, Deserialize)]
pub struct RawModel {
    pub vocab: Vec<Vec<u8>>,
    pub rules: Vec<Rule>,
    pub specials: Vec<(String, u16)>,
    pub public: Vec<u64>,
}
struct Word {
    ids: Vec<u16>,
    weight: u64,
}
#[derive(Default)]
struct PairState {
    count: u64,
    words: Set<usize>,
}
// Compare frequency first. Resolve ties with the byte pair.
#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Candidate {
    count: u64,
    left: Arc<[u8]>,
    right: Arc<[u8]>,
    key: u32,
}
pub fn train(path: &str, limit: usize, tokens: Vec<String>) -> Result<RawModel, String> {
    let specials: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), (256 + i) as u16))
        .collect();
    let sp = Specials::new(specials.clone())?;
    let mut counts: Map<Vec<u8>, u64> = Map::default();
    read_chunks(path, &sp, 4 << 20, |text| {
        sp.visit(text, |part, id| {
            if id.is_none() {
                for p in Pieces::new(part) {
                    if let Some(n) = counts.get_mut(p) {
                        *n += 1;
                    } else {
                        counts.insert(p.to_vec(), 1);
                    }
                }
            }
        });
        Ok(())
    })?;
    let mut words: Vec<_> = counts
        .into_iter()
        .map(|(b, weight)| Word {
            ids: b.into_iter().map(u16::from).collect(),
            weight,
        })
        .collect();
    let mut vocab: Vec<Arc<[u8]>> = (0..256).map(|b| Arc::from(vec![b as u8])).collect();
    for s in &tokens {
        vocab.push(Arc::from(s.as_bytes()));
    }
    let mut states: Map<u32, PairState> = Map::default();
    for (wi, w) in words.iter().enumerate() {
        for p in w.ids.windows(2) {
            let st = states.entry(key(p[0], p[1])).or_default();
            st.count += w.weight;
            st.words.insert(wi);
        }
    }
    let candidate = |k: u32, n: u64, v: &Vec<Arc<[u8]>>| Candidate {
        count: n,
        left: v[(k >> 16) as usize].clone(),
        right: v[(k & 65535) as usize].clone(),
        key: k,
    };
    let mut heap = BinaryHeap::new();
    for (&k, s) in &states {
        heap.push(candidate(k, s.count, &vocab));
    }
    let mut rules = Vec::new();
    let mut ordinary: Map<Vec<u8>, u16> = (0..256).map(|i| (vec![i as u8], i as u16)).collect();
    while vocab.len() < limit {
        let Some(best) = heap.pop() else { break };
        if best.count == 0 || states.get(&best.key).map(|s| s.count) != Some(best.count) {
            continue;
        }
        let a = (best.key >> 16) as u16;
        let b = best.key as u16;
        let mut joined = best.left.to_vec();
        joined.extend_from_slice(&best.right);
        let id = if let Some(&id) = ordinary.get(&joined) {
            id
        } else {
            let id = vocab.len() as u16;
            ordinary.insert(joined.clone(), id);
            vocab.push(Arc::from(joined));
            id
        };
        rules.push((a, b, id));
        let affected: Vec<_> = states[&best.key].words.iter().copied().collect();
        let mut changed = Set::default();
        for wi in affected {
            let w = &mut words[wi];
            for p in w.ids.windows(2) {
                let k = key(p[0], p[1]);
                let st = states.get_mut(&k).unwrap();
                st.count -= w.weight;
                st.words.remove(&wi);
                changed.insert(k);
            }
            let (mut r, mut out) = (0, 0);
            while r < w.ids.len() {
                if r + 1 < w.ids.len() && w.ids[r] == a && w.ids[r + 1] == b {
                    w.ids[out] = id;
                    r += 2;
                } else {
                    w.ids[out] = w.ids[r];
                    r += 1;
                }
                out += 1;
            }
            w.ids.truncate(out);
            for p in w.ids.windows(2) {
                let k = key(p[0], p[1]);
                let st = states.entry(k).or_default();
                st.count += w.weight;
                st.words.insert(wi);
                changed.insert(k);
            }
        }
        for k in changed {
            let n = states[&k].count;
            if n > 0 {
                heap.push(candidate(k, n, &vocab));
            }
        }
        if heap.len() > states.len().saturating_mul(5).max(100000) {
            heap.clear();
            for (&k, st) in &states {
                if st.count > 0 {
                    heap.push(candidate(k, st.count, &vocab));
                }
            }
        }
    }
    Ok(RawModel {
        public: (0..vocab.len() as u64).collect(),
        vocab: vocab.into_iter().map(|v| v.to_vec()).collect(),
        rules,
        specials,
    })
}
