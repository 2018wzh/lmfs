import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from bpe_rs._native import RustBPE, train_file


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.num_threads = os.cpu_count() or 1
        self.vocab = dict(vocab) if vocab is not None else {i: bytes([i]) for i in range(256)}
        self.merges = list(merges or [])
        self.special_tokens = list(special_tokens or [])
        self._special_ids = self._find_special_ids()
        self._compile()

    def _find_special_ids(self) -> dict[str, int]:
        byte_to_id = {v: k for k, v in self.vocab.items()}
        special_ids = {}
        for token in self.special_tokens:
            raw = token.encode("utf-8")
            if raw not in byte_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = raw
                byte_to_id[raw] = token_id
            special_ids[token] = byte_to_id[raw]
        return special_ids

    def _compile(self) -> None:
        special_ids = set(self._special_ids.values())
        ordinary = {raw: token_id for token_id, raw in self.vocab.items() if token_id not in special_ids}
        base = [ordinary[bytes([i])] for i in range(256)]
        public = base + sorted(set(self.vocab) - set(base))
        compact = {token_id: idx for idx, token_id in enumerate(public)}
        rules = [
            (
                compact[ordinary[left]],
                compact[ordinary[right]],
                compact[ordinary[left + right]],
            )
            for left, right in self.merges
        ]
        specials = [(token, compact[self._special_ids[token]]) for token in self.special_tokens]
        self._native = RustBPE(
            [self.vocab[token_id] for token_id in public],
            rules,
            specials,
            public,
            self.num_threads or 0,
        )

    def train(self, input_path: str | Path, vocab_size: int, special_tokens: list[str]) -> None:
        model = json.loads(train_file(str(input_path), vocab_size, list(special_tokens)))
        values = [bytes(v) for v in model["vocab"]]
        self.vocab = dict(zip(model["public"], values))
        self.merges = [(values[left], values[right]) for left, right, _ in model["rules"]]
        self.special_tokens = [token for token, _ in model["specials"]]
        self._special_ids = {token: model["public"][index] for token, index in model["specials"]}
        self._compile()

    def encode(self, text: str) -> list[int]:
        return self._native.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._native.decode(ids)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        stream = self._native.stream()
        for chunk in iterable:
            yield from stream.push(chunk)
        yield from stream.push("", True)

    def encode_numpy(self, text: str) -> np.ndarray:
        return self._native.encode_numpy(text)

    def encode_to_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        block_bytes: int = 16 << 20,
    ) -> int:
        return self._native.encode_file(str(input_path), str(output_path), block_bytes)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "BPE":
        visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
        visible_set = set(visible)
        hidden = [b for b in range(256) if b not in visible_set]
        byte_decoder = {chr(b): b for b in visible}
        byte_decoder.update({chr(256 + i): b for i, b in enumerate(hidden)})

        raw_vocab = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        vocab = {int(idx): bytes(byte_decoder[ch] for ch in token) for token, idx in raw_vocab.items()}
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#version:"):
                    continue
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    merges.append(
                        (
                            bytes(byte_decoder[ch] for ch in parts[0]),
                            bytes(byte_decoder[ch] for ch in parts[1]),
                        )
                    )
        return cls(vocab, merges, special_tokens)

    def save(self, path_prefix: str | Path) -> None:
        path = Path(f"{path_prefix}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": [[i, value.hex()] for i, value in sorted(self.vocab.items())],
            "merges": [[left.hex(), right.hex()] for left, right in self.merges],
            "special_tokens": self.special_tokens,
            "special_ids": self._special_ids,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path_prefix: str | Path) -> None:
        data = json.loads(Path(f"{path_prefix}.json").read_text(encoding="utf-8"))
        self.vocab = {int(i): bytes.fromhex(raw) for i, raw in data["vocab"]}
        self.merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in data["merges"]]
        self.special_tokens = list(data["special_tokens"])
        self._special_ids = {k: int(v) for k, v in data["special_ids"].items()}
        self._compile()


def run_train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    tokenizer = BPE()
    tokenizer.train(input_path, vocab_size, special_tokens)
    return tokenizer.vocab, tokenizer.merges


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPE:
    return BPE(vocab, merges, special_tokens)


__all__ = ["BPE", "get_tokenizer", "run_train_bpe"]
