from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BERT MRPC demo with PyTorch DistributedDataParallel"
    )
    p.add_argument(
        "--global-batch-size",
        type=int,
        default=64,
        help="Total samples per optimizer step (summed over all ranks)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Untimed steps before benchmark (timed window starts at warmup_steps + 1)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Total training steps (inclusive; benchmark ends at this step)",
    )
    p.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit NVTX range 'backward' around loss.backward() for nsys",
    )
    return p.parse_args()


def _backward(loss: torch.Tensor, *, nvtx: bool) -> None:
    if nvtx:
        torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if nvtx:
        torch.cuda.nvtx.range_pop()


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    model_name = "bert-large-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    raw.to(device)
    model = nn.parallel.DistributedDataParallel(
        raw,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    if dist.get_rank() == 0:
        print(
            f"global_batch_size={args.global_batch_size} "
            f"warmup_steps={args.warmup_steps} max_steps={args.max_steps} "
            f"nvtx={args.nvtx}"
        )

    ds = load_dataset("glue", "mrpc")

    def tok(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_ds = ds["train"].map(tok, batched=True)
    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    global_batch_size = args.global_batch_size
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by world_size ({world_size})"
        )
    per_device_batch_size = global_batch_size // world_size

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
    loader = DataLoader(train_ds, batch_size=per_device_batch_size, sampler=sampler, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if warmup_steps >= max_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < max_steps ({max_steps}) "
            "so the benchmark window has at least one step"
        )

    model.train()
    bench_last_step = max_steps  # inclusive; timed window is steps warmup_steps+1 .. bench_last_step
    step = 0
    bench_t0: float | None = None
    bench_t1: float | None = None
    while step < max_steps:
        sampler.set_epoch(step)
        for batch in loader:
            if step >= max_steps:
                break
            if step == warmup_steps:
                torch.cuda.synchronize()
                bench_t0 = time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            out = model(**batch)
            loss = F.cross_entropy(out.logits, labels)
            _backward(loss, nvtx=args.nvtx)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == bench_last_step:
                torch.cuda.synchronize()
                bench_t1 = time.perf_counter()
            if dist.get_rank() == 0 and step % 10 == 0:
                print(f"step {step} loss {loss.item():.4f}")

    dist.destroy_process_group()
    if local_rank == 0:
        elapsed = time.perf_counter() - t0
        print(f"demo finished OK (wall time {elapsed:.2f}s)")
        if bench_t0 is not None and bench_t1 is not None:
            bench_steps = bench_last_step - warmup_steps
            dt = bench_t1 - bench_t0
            print(
                f"bench steps {warmup_steps + 1}-{bench_last_step}: {dt:.4f}s "
                f"({bench_steps} steps, {dt / bench_steps * 1000:.2f} ms/step)"
            )


if __name__ == "__main__":
    main()
