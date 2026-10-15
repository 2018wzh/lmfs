//! Split text with the assignment pre-tokenization rules.
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
pub static CLASSES: &[u8; 0x110000] = include_bytes!("../resources/unicode.bin");
#[inline]
fn class_at(s: &str, i: usize) -> (u8, usize) {
    let b = s.as_bytes()[i];
    if b < 128 {
        (CLASSES[b as usize], 1)
    } else {
        let c = s[i..].chars().next().unwrap();
        (CLASSES[c as usize], c.len_utf8())
    }
}
pub struct Pieces<'a> {
    text: &'a str,
    pos: usize,
}
impl<'a> Pieces<'a> {
    pub fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }
}
impl<'a> Iterator for Pieces<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let s = self.text;
        let b = s.as_bytes();
        let start = self.pos;
        if start == b.len() {
            return None;
        }
        if b[start] == b'\'' {
            let tail = &b[start + 1..];
            let n = if tail
                .first()
                .is_some_and(|x| matches!(x, b's' | b'd' | b'm' | b't'))
            {
                1
            } else if tail.starts_with(b"ll") || tail.starts_with(b"ve") || tail.starts_with(b"re")
            {
                2
            } else {
                0
            };
            if n > 0 {
                self.pos += n + 1;
                return Some(&b[start..self.pos]);
            }
        }
        let mut i = start;
        if b[i] == b' ' && i + 1 < b.len() && class_at(s, i + 1).0 != 3 {
            i += 1;
        }
        let (c, w) = class_at(s, i);
        i += w;
        if c != 3 {
            while i < b.len() {
                let (next, w) = class_at(s, i);
                if next != c {
                    break;
                }
                i += w;
            }
        } else {
            let mut last = start;
            let mut n = 1;
            while i < b.len() {
                let (next, w) = class_at(s, i);
                if next != 3 {
                    break;
                }
                last = i;
                i += w;
                n += 1;
            }
            if i < b.len() && n > 1 {
                i = last;
            }
        }
        self.pos = i;
        Some(&b[start..i])
    }
}
pub struct Specials {
    pub entries: Vec<(String, u16)>,
    matcher: Option<AhoCorasick>,
}
impl Specials {
    pub fn new(entries: Vec<(String, u16)>) -> Result<Self, String> {
        let matcher = if entries.is_empty() {
            None
        } else {
            Some(
                AhoCorasickBuilder::new()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(entries.iter().map(|(s, _)| s.as_bytes()))
                    .map_err(|e| e.to_string())?,
            )
        };
        Ok(Self { entries, matcher })
    }
    pub fn visit<'a>(&self, s: &'a str, mut f: impl FnMut(&'a str, Option<u16>)) {
        let mut start = 0;
        if let Some(rx) = &self.matcher {
            for m in rx.find_iter(s.as_bytes()) {
                if m.start() > start {
                    f(&s[start..m.start()], None);
                }
                f(
                    &s[m.start()..m.end()],
                    Some(self.entries[m.pattern().as_usize()].1),
                );
                start = m.end();
            }
        }
        if start < s.len() {
            f(&s[start..], None);
        }
    }
    /// Split work at special tokens or stable pre-token boundaries.
    pub fn chunks<'a>(&self, s: &'a str, target: usize) -> Vec<&'a str> {
        let mut out = Vec::new();
        let mut start = 0;
        if let Some(rx) = &self.matcher {
            for m in rx.find_iter(s.as_bytes()) {
                if m.end() - start >= target {
                    out.push(&s[start..m.end()]);
                    start = m.end();
                }
            }
        }
        if start < s.len() {
            out.push(&s[start..]);
        }
        if out.len() > 1 || s.len() < target * 2 {
            return out;
        }
        out.clear();
        start = 0;
        let origin = s.as_ptr() as usize;
        self.visit(s, |part, sp| {
            if sp.is_some() {
                let end = part.as_ptr() as usize - origin + part.len();
                if end - start >= target {
                    out.push(&s[start..end]);
                    start = end;
                }
            } else {
                for p in Pieces::new(part) {
                    let end = p.as_ptr() as usize - origin + p.len();
                    if end - start >= target && !ends_space(p) {
                        out.push(&s[start..end]);
                        start = end;
                    }
                }
            }
        });
        if start < s.len() {
            out.push(&s[start..]);
        }
        out
    }
    /// Keep the suffix that later text could change.
    pub fn stable_cut(&self, s: &str) -> usize {
        let guard = self.entries.iter().map(|(s, _)| s.len()).max().unwrap_or(0) + 4;
        let limit = s.len().saturating_sub(guard);
        let mut ends = [0usize; 3];
        let origin = s.as_ptr() as usize;
        self.visit(s, |part, sp| {
            if sp.is_some() {
                let end = part.as_ptr() as usize - origin + part.len();
                if end <= limit {
                    ends.rotate_left(1);
                    ends[2] = end;
                }
            } else {
                for p in Pieces::new(part) {
                    let end = p.as_ptr() as usize - origin + p.len();
                    if end <= limit && !ends_space(p) {
                        ends.rotate_left(1);
                        ends[2] = end;
                    }
                }
            }
        });
        ends[0]
    }
}
fn ends_space(bytes: &[u8]) -> bool {
    let c = std::str::from_utf8(bytes).unwrap().chars().last().unwrap();
    CLASSES[c as usize] == 3
}
