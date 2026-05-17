[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-DDuAbg5)

# Nano DDP - CSIT 5970 Project

Name: Wu Yongjin
SID: 20564741

This project is for **implementing Data Parallel LLM distributed training** by myself. The repo provides three incremental implementations (**NanoDDP V1 / V2 / V3**) in:

- [`nano_ddp.py`](nano_ddp.py)

Besides, to compare the efficiency of my own implementation of **Nano DDP** with the  **PyTorch’s built-in `DistributedDataParallel`**, I fine-tune **BERT-Large** on **GLUE MRPC** on a single machine with multiple GPUs. The scripts are:

- Nano DDP demo: [`run_nano_ddp.py`](run_nano_ddp.py) (`--path v1|v2|v3`)
- Official DDP demo: [`run_ddp.py`](run_ddp.py)


## Differences among Nano DDP V1/V2/V3

- **V1:** Wait until backward finishes for every parameter, then `all_reduce` once on flattened grads.
- **V2:** `all_reduce` each parameter’s gradient as soon as it’s ready during backward.
- **V3:** Pack parameters into byte-sized buckets and `all_reduce` when a bucket fills up — closest to official DDP.


## Requirements

- Python 3.10+
- PyTorch with CUDA and NCCL
- Single-node, multi-GPU setup

```bash
pip install torch transformers datasets
```

## Quick start

Replace `2` with your GPU count
### Official PyTorch DDP

```bash
torchrun --standalone --nproc_per_node=2 run_ddp.py
```

### Nano DDP

Use `--path` to select different implementation (see later section)

```bash
# V1
torchrun --standalone --nproc_per_node=2 run_nano_ddp.py --path v1

# V2
torchrun --standalone --nproc_per_node=2 run_nano_ddp.py --path v2

# V3
torchrun --standalone --nproc_per_node=2 run_nano_ddp.py --path v3
```


## CLI arguments

Besides `--path`, you can use the following CLI arguments for profiling under different settings:

| Flag | Default | Description |
|------|---------|-------------|
| `--global-batch-size` | `64` | Global batch size (sum over all ranks); must be divisible by `world_size` |
| `--warmup-steps` | `10` | Warmup steps excluded from timing; benchmark starts at step `warmup_steps + 1` |
| `--max-steps` | `30` | Total training steps (including warmup); requires `warmup_steps < max_steps` |
| `--nvtx` | off | Wrap `loss.backward()` in NVTX range `backward` for Nsight Systems |





## Profiling with Nsight Systems

Use NVTX on backward to compare communication vs compute overlap (`--nvtx` + a short run):

```bash
# Nano DDP
nsys profile -o profile_v1 --force-overwrite=true --trace=cuda,nccl,nvtx \
  --capture-range=nvtx --nvtx-capture="backward" \
  torchrun --standalone --nproc_per_node=2 run_nano_ddp.py --path v1 --global-batch-size 64 --nvtx \
    --warmup-steps 2 --max-steps 5

# Official DDP
nsys profile -o profile_ddp --force-overwrite=true --trace=cuda,nccl,nvtx \
  --capture-range=nvtx --nvtx-capture="backward" \
  torchrun --standalone --nproc_per_node=2 run_ddp.py --global-batch-size 64  --nvtx \
    --warmup-steps 2 --max-steps 5
```

