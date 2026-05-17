#!/usr/bin/env python3
"""Summarize NCCL AllReduce from nsys profiles."""
from __future__ import annotations

import re
import subprocess

BENCH_STEPS = 20
N_RANKS = 2

PROFILES = [
    ("DDP", "profile_ddp.nsys-rep"),
    ("V1", "profile_v1.nsys-rep"),
    ("V2", "profile_v2.nsys-rep"),
    ("V3", "profile_v3.nsys-rep"),
]


def parse_kern_sum(path: str, filter_nvtx: str | None = None) -> dict | None:
    cmd = ["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum", path]
    if filter_nvtx:
        cmd += ["--filter-nvtx", filter_nvtx]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    if "SKIPPED" in out:
        return None
    total = ar = nccl = 0
    ar_inst = 0
    for line in out.splitlines():
        if "allreduce" in line.lower() and "nccl" in line.lower():
            m = re.match(r"\s+[\d.]+\s+(\d+)\s+(\d+)", line)
            if m:
                ar += int(m.group(1))
                ar_inst += int(m.group(2))
        m = re.match(r"\s+[\d.]+\s+(\d+)\s+\d+", line)
        if not m:
            continue
        t = int(m.group(1))
        total += t
        if "nccl" in line.lower():
            nccl += t
    return {
        "total_ms": total / 1e6,
        "ar_ms": ar / 1e6,
        "ar_inst": ar_inst,
        "nccl_ms": nccl / 1e6,
        "ar_pct": 100 * ar / total if total else 0.0,
    }


def nvtx_backward_count(path: str) -> int:
    out = subprocess.check_output(
        ["nsys", "stats", "--force-export=true", "--report", "nvtx_pushpop_sum", path],
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in out.splitlines():
        if "backward" in line.lower() and "CCCL" not in line and "cublas" not in line.lower():
            m = re.match(r"\s+[\d.]+\s+(\d+)\s+(\d+)", line)
            if m and int(m.group(2)) >= 10:
                return int(m.group(2))
    return 0


def main() -> None:
    print("Setup: warmup=10, max_steps=30, bench 11-30 (20 steps), 2 GPUs, batch=64\n")
    rows: list[tuple[str, dict, dict | None, int]] = []
    for name, path in PROFILES:
        rows.append((name, parse_kern_sum(path), parse_kern_sum(path, "backward"), nvtx_backward_count(path)))

    print("## Full trace (all CUDA kernels, summed)")
    print("| | DDP | V1 | V2 | V3 |")
    print("|--|--:|--:|--:|--:|")
    for key, fmt in [
        ("total_ms", "{:.0f} ms"),
        ("ar_ms", "{:.0f} ms"),
        ("ar_pct", "{:.1f}%"),
        ("ar_inst", "{}"),
    ]:
        vals = []
        for name, full, _, _ in rows:
            v = full[key]
            vals.append(fmt.format(v) if key != "ar_inst" else str(int(v)))
        print(f"| {key} | " + " | ".join(vals) + " |")

    per = BENCH_STEPS * N_RANKS
    print(f"\n| AR / bench step / GPU ({per} calls expected for 20×2) |", end="")
    for name, full, _, _ in rows:
        print(f" {name} {full['ar_ms']/per:.1f} ms |", end="")
    print()

    print("\n| avg per AllReduce call |", end="")
    for name, full, _, _ in rows:
        avg = full["ar_ms"] / full["ar_inst"] if full["ar_inst"] else 0
        print(f" {name} {avg:.2f} ms |", end="")
    print()

    print("\n## backward NVTX slice (reference only; filter often under-counts)")
    print("| | DDP | V1 | V2 | V3 |")
    print("|--|--:|--:|--:|--:|")
    for label, key in [("kernel total", "total_ms"), ("AllReduce", "ar_ms"), ("AR %", "ar_pct"), ("AR calls", "ar_inst")]:
        vals = []
        for name, _, bwd, _ in rows:
            if bwd is None:
                vals.append("n/a")
            elif key == "ar_pct":
                vals.append(f"{bwd[key]:.1f}%")
            elif key == "ar_inst":
                vals.append(str(int(bwd[key])))
            else:
                vals.append(f"{bwd[key]:.0f} ms")
        print(f"| {label} | " + " | ".join(vals) + " |")

    print("\n| backward NVTX ranges |", end="")
    for name, _, _, nb in rows:
        print(f" {name}={nb}", end="")
    print(" (expect ~60 = 30 steps × 2 ranks)")


if __name__ == "__main__":
    main()
