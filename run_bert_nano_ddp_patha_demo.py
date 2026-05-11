# Demo: fine-tune BERT on GLUE MRPC with NanoDDPPathA (wrapper in nano_ddp.py).
#
# Put this file next to nano_ddp.py (same directory), then:
#
#   pip install transformers datasets  # if not already
#
# Single node, 2 GPUs (change 2 to your GPU count):
#   torchrun --standalone --nproc_per_node=2 run_bert_nano_ddp_patha_demo.py
#
# Copy to server (paths adjusted):
#   scp /Users/james/git/nano_ddp/nano_ddp.py /Users/james/git/nano_ddp/run_bert_nano_ddp_patha_demo.py user@host:~/nano_ddp/
#   ssh user@host 'cd ~/nano_ddp && torchrun --standalone --nproc_per_node=2 run_bert_nano_ddp_patha_demo.py'

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nano_ddp import NanoDDPPathA


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    raw.to(device)
    model = NanoDDPPathA(raw, device=device)

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

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
    loader = DataLoader(train_ds, batch_size=8, sampler=sampler, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    max_steps = 30
    step = 0
    while step < max_steps:
        sampler.set_epoch(step)
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            out = model(**batch)
            loss = F.cross_entropy(out.logits, labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if dist.get_rank() == 0 and step % 10 == 0:
                print(f"step {step} loss {loss.item():.4f}")

    dist.destroy_process_group()
    if local_rank == 0:
        print("demo finished OK")


if __name__ == "__main__":
    main()
