import json
import sys
import time
from pathlib import Path
import wandb
from wandb_record import experiment
from dotenv import load_dotenv


def main():
    config = {
        "train_file": "/lmfs2026/data/owt_train.txt",
        "valid_file": "/lmfs2026/data/owt_valid.txt",
        "vocab_size": 32000,
        "special_tokens": ["<|endoftext|>"],  # 示例：替换为本人实际清单
    }
    with experiment("bpe", config) as run:
        from bpe import BPE

        bpe = BPE()
        bpe.train(config["train_file"], config["vocab_size"], config["special_tokens"])
        prefix = str(Path("outputs") / run.id / "tokenizer")
        bpe.save(prefix)
        # 重新加载，避免使用训练或先前评估产生的编码缓存。
        bpe = BPE()
        bpe.load(prefix)
        with open(config["valid_file"], encoding="utf-8", newline="") as f:
            text = f.read()
        started = time.perf_counter()
        ids = bpe.encode(text)
        decoded_text = bpe.decode(ids)
        passed = decoded_text == text
        valid_seconds = time.perf_counter() - started
        # 以下均为展示/审查操作，不计入验证耗时。
        raw = text.encode("utf-8")
        first_ids = ids[:200]
        rows, offset = [], 0
        for i, token_id in enumerate(first_ids):
            piece = bpe.vocab[token_id]
            if not isinstance(piece, bytes):
                raise TypeError("vocab 的值必须是 bytes")
            end = offset + len(piece)
            try:
                piece.decode("utf-8", errors="strict")
                standalone_valid = True
            except UnicodeDecodeError:
                standalone_valid = False
            rows.append(
                [
                    i,
                    int(token_id),
                    piece.hex(),
                    repr(piece),
                    repr(bpe.decode([token_id])),
                    standalone_valid,
                    offset,
                    end,
                    piece == raw[offset:end],
                ]
            )
            offset = end
        columns = [
            "position",
            "token_id",
            "token_hex",
            "token_bytes",
            "decoded_token",
            "standalone_utf8_valid",
            "byte_start",
            "byte_end",
            "matches_input_bytes",
        ]
        print("前 200 个 encode ID：", first_ids, flush=True)
        print("前 200 个 token 整体 decode：", repr(bpe.decode(first_ids)), flush=True)
        print("逐 token 详情字段：", columns, flush=True)
        for row in rows:
            print(row, flush=True)
        run.log({"bpe/first_200_tokens": wandb.Table(columns=columns, data=rows)})
        result = {
            "bpe/valid_seconds": valid_seconds,
            "bpe/valid_token_count": len(ids),
            "bpe/roundtrip_passed": passed,
            "bpe/vocab_size": len(bpe.vocab),
        }
        run.summary.update(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print("任务二加载本人 BPE 的路径前缀：", prefix, flush=True)
        if len(ids) < 200 or not passed:
            raise ValueError("评估文本不足 200 个 token 或还原失败，请检查输入与实现")


if __name__ == "__main__":
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
