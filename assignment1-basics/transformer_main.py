import time

PROGRAM_STARTED_UNIX = time.time()
PROGRAM_STARTED_CLOCK = time.perf_counter()

import argparse
import json
import math
import random
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from bpe import BPE
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW
from wandb_record import experiment


@torch.no_grad()
def evaluate(model, dataset, context_length, batch_size, device, use_bf16=True):
    model.eval()
    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = 0
    num_windows, tail = divmod(len(dataset) - 1, context_length)
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and "cuda" in str(device)
        else nullcontext()
    )

    for start_idx in range(0, num_windows, batch_size):
        end_idx = min(start_idx + batch_size, num_windows)
        starts = np.arange(start_idx, end_idx) * context_length
        offsets = starts[:, None] + np.arange(context_length + 1)[None, :]
        chunk = torch.from_numpy(np.asarray(dataset[offsets], dtype=np.int64)).to(
            device
        )
        x, y = chunk[:, :-1], chunk[:, 1:]
        with ctx:
            loss = cross_entropy(model(x), y)
        total_nll.add_(loss.double() * y.numel())
        total_tokens += y.numel()

    if tail > 0:
        chunk = (
            torch.from_numpy(
                np.asarray(dataset[num_windows * context_length :], dtype=np.int64)
            )
            .to(device)
            .unsqueeze(0)
        )
        x, y = chunk[:, :-1], chunk[:, 1:]
        with ctx:
            loss = cross_entropy(model(x), y)
        total_nll.add_(loss.double() * y.numel())
        total_tokens += y.numel()

    model.train()
    return float(total_nll.item()), int(total_tokens)


def train_model(config, run):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    device = config["device"]
    out_dir = Path("outputs") / run.id
    out_dir.mkdir(parents=True, exist_ok=True)

    bpe = BPE()
    bpe.load(config["bpe_prefix"])
    vocab_size = len(bpe.vocab)

    train_npy = out_dir / "train.npy"
    valid_npy = out_dir / "valid.npy"
    bpe.encode_to_file(config["train_file"], train_npy)
    bpe.encode_to_file(config["valid_file"], valid_npy)
    train_data = np.load(train_npy, mmap_mode="r")
    valid_data = np.load(valid_npy, mmap_mode="r")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
    )

    decay_params = [p for p in model.parameters() if p.ndim >= 2]
    nodecay_params = [p for p in model.parameters() if p.ndim < 2]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config["max_learning_rate"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
    )

    global_step = 0
    if config.get("resume_checkpoint"):
        global_step = load_checkpoint(config["resume_checkpoint"], model, optimizer)

    def forward_loss(x, y):
        return cross_entropy(model(x), y)

    training_loss = torch.compile(forward_loss, dynamic=False) if config.get("compile", False) else forward_loss
    use_bf16 = config.get("bf16", False) and "cuda" in str(device)
    grad_accum_steps = config["batch_size"] // config["microbatch_size"]
    ckpt_path = out_dir / "last.pt"
    last_ckpt_time = time.perf_counter()

    # 训练循环开始前初始化
    interval_nll = None
    interval_tokens = None

    decay_started = None
    model.train()
    while True:
        remaining = config["deadline_clock"] - time.perf_counter()
        if remaining <= config["reserve_seconds"]:
            break

        if global_step < config["warmup_iters"]:
            lr = config["max_learning_rate"] * (global_step + 1) / config["warmup_iters"]
        else:
            now = time.perf_counter()
            if decay_started is None:
                decay_started = now
            end = config["deadline_clock"] - config["reserve_seconds"]
            progress = min(1.0, max(0.0, (now - decay_started) / max(end - decay_started, 1e-9)))
            lr = config["min_learning_rate"] + 0.5 * (1.0 + math.cos(math.pi * progress)) * (
                config["max_learning_rate"] - config["min_learning_rate"]
            )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            x, y = get_batch(
                train_data, config["microbatch_size"], config["context_length"], device
            )
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with ctx:
                loss = training_loss(x, y)
            num_tokens = y.numel()
            loss_sum = loss.detach() * num_tokens

            # 每个微批次计算损失后累计；detach() 避免保留计算图。
            # 独立的累计张量不会修改原始 loss_sum，也不参与反向传播。
            if interval_nll is None:
                interval_nll = torch.zeros(
                    (), device=loss_sum.device, dtype=torch.float64
                )
                interval_tokens = torch.zeros(
                    (), device=loss_sum.device, dtype=torch.int64
                )
            interval_nll.add_(loss_sum.detach().to(dtype=torch.float64))
            interval_tokens.add_(
                torch.as_tensor(
                    num_tokens, device=loss_sum.device, dtype=torch.int64
                ).detach()
            )

            (loss / grad_accum_steps).backward()

        gradient_clipping(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        global_step += 1

        # 每完成一次 optimizer 更新后检查，不放在每个微批次更新位置。
        if global_step % 100 == 0:
            # 仅在记录时取回累计值；num_tokens 可为整数或设备上的标量张量。
            logged_nll = interval_nll.item()
            logged_tokens = interval_tokens.item()
            if logged_tokens > 0:
                run.log(
                    {
                        "train/ppl": math.exp(logged_nll / logged_tokens),
                        "train/step": int(global_step),
                        "time/elapsed_seconds": config["previous_elapsed_seconds"]
                        + time.perf_counter()
                        - PROGRAM_STARTED_CLOCK,
                    }
                )
            interval_nll.zero_()
            interval_tokens.zero_()

        if (
            time.perf_counter() - last_ckpt_time
            >= config["checkpoint_interval_seconds"]
        ):
            save_checkpoint(model, optimizer, global_step, ckpt_path)
            last_ckpt_time = time.perf_counter()

    if interval_tokens is not None and interval_tokens.item() > 0:
        logged_nll = interval_nll.item()
        logged_tokens = interval_tokens.item()
        run.log(
            {
                "train/ppl": math.exp(logged_nll / logged_tokens),
                "train/step": int(global_step),
                "time/elapsed_seconds": config["previous_elapsed_seconds"]
                + time.perf_counter()
                - PROGRAM_STARTED_CLOCK,
            }
        )
        interval_nll.zero_()
        interval_tokens.zero_()

    save_checkpoint(model, optimizer, global_step, ckpt_path)
    val_nll_sum, val_token_count = evaluate(
        model,
        valid_data,
        config["context_length"],
        config["microbatch_size"],
        device,
        use_bf16,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "val_nll_sum": float(val_nll_sum),
        "val_token_count": int(val_token_count),
        "step": int(global_step),
    }


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train the transformer language model.").parse_args()
    load_dotenv()
    config = {
        "train_file": "data/owt_train.txt",
        "valid_file": "data/owt_valid.txt",
        "bpe_prefix": "outputs/a4206cc4bdcf/tokenizer",
        "max_train_seconds": 21600,
        "previous_elapsed_seconds": 0.0,
        "resume_from_run": None,
        "resume_checkpoint": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 336,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 2048,
        "context_length": 1024,
        "rope_theta": 10000.0,
        "batch_size": 64,
        "microbatch_size": 16,
        "max_learning_rate": 1.5e-3,
        "min_learning_rate": 1.5e-4,
        "warmup_iters": 2000,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "bf16": True,
        "compile": True,
        "checkpoint_interval_seconds": 600,
        "reserve_seconds": 900,
    }
    with experiment("transformer", config) as run:
        run.define_metric("train/step")
        run.define_metric("train/ppl", step_metric="train/step")

        def utc(timestamp):
            return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")

        previous = float(config["previous_elapsed_seconds"])
        if not math.isfinite(previous) or not 0 <= previous < 21600:
            raise ValueError("此前累计耗时无效或已用完预算")
        run.summary["time/program_started_utc"] = utc(PROGRAM_STARTED_UNIX)
        run.summary["time/previous_elapsed_seconds"] = previous
        # 训练循环可调用：remaining = config["deadline_clock"] - time.perf_counter()
        config["deadline_clock"] = PROGRAM_STARTED_CLOCK + 21600 - previous
        try:
            result = train_model(config, run)
        finally:
            # 正常返回或 Python 异常时都记录；强制 kill/断电无法保证执行。
            ended_unix = time.time()
            current_seconds = time.perf_counter() - PROGRAM_STARTED_CLOCK
            total_seconds = previous + current_seconds
            timing = {
                "time/program_started_utc": utc(PROGRAM_STARTED_UNIX),
                "time/experiment_ended_utc": utc(ended_unix),
                "time/current_run_seconds": current_seconds,
                "time/total_seconds": total_seconds,
                "time/difference_from_6h_seconds": total_seconds - 21600,
                "time/within_6h": total_seconds <= 21600,
            }
            run.summary.update(timing)
            print(json.dumps(timing, ensure_ascii=False), flush=True)
        if result["val_token_count"] <= 0:
            raise ValueError("验证 token 数必须大于零")
        val_loss = result["val_nll_sum"] / result["val_token_count"]
        if not math.isfinite(val_loss) or val_loss < 0:
            raise ValueError("验证平均 NLL 非有限值或为负，请检查计算")
        try:
            final_ppl = math.exp(val_loss)
        except OverflowError as error:
            raise ValueError("验证 PPL 溢出，请检查模型数值") from error
        if not math.isfinite(final_ppl):
            raise ValueError("PPL 非有限值，请检查模型数值")
        final = {
            "val/ppl": final_ppl,
            "val/token_count": result["val_token_count"],
            "train/final_step": result["step"],
            "train/total_seconds": total_seconds,
        }
        run.summary.update(final)
        print(json.dumps(final, ensure_ascii=False), flush=True)
        if not 0 <= total_seconds <= 21600:
            raise ValueError("累计训练耗时超过预算或无效，请核查")
