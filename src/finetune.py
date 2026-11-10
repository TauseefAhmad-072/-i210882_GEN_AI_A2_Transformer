"""Task 1: LoRA supervised fine-tuning (loss on the JSON answer tokens only).
Usage: python -m src.finetune --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/lora
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import build_prompt, load_split, target_text


def encode(tok, rec, max_len, answer_budget=64):
    prompt, dropped = build_prompt(tok, rec["sentences"], rec["criterion_type"], rec["criterion_text"],
                                   max_len - answer_budget)
    kept_ev = [e for e in rec["evidence"] if e not in dropped]          # cannot cite what was truncated
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(target_text(rec["label"], kept_ev) + tok.eos_token, add_special_tokens=False).input_ids
    return {"input_ids": p + t, "labels": [-100] * len(p) + t,
            "truncated": bool(dropped), "lost_evidence": len(rec["evidence"]) - len(kept_ev)}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100)
    att = torch.zeros(len(batch), L, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return ids, att, lab


@torch.no_grad()
def eval_loss(model, loader, dev, amp_dtype):
    model.eval(); tot = n = 0
    for ids, att, lab in loader:
        ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
        with torch.autocast(dev, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        k = (lab[:, 1:] != -100).sum().item(); tot += loss.item() * k; n += k
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="outputs/lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=0, help="debug: limit #train records")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        amp = None
    wdtype = amp if amp is not None else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=wdtype).to(dev)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()                                     # fp32 adapter weights
    model.print_trainable_parameters()

    tr, va = load_split("train"), load_split("val")
    if a.max_train:
        tr = tr[:a.max_train]
    enc_tr = [encode(tok, r, a.max_len) for r in tr]
    enc_va = [encode(tok, r, a.max_len) for r in va]
    trunc = {"train_truncated": sum(e["truncated"] for e in enc_tr), "train_lost_evidence": sum(e["lost_evidence"] for e in enc_tr),
             "val_truncated": sum(e["truncated"] for e in enc_va), "val_lost_evidence": sum(e["lost_evidence"] for e in enc_va),
             "max_seq_len_seen": max(len(e["input_ids"]) for e in enc_tr + enc_va)}
    print("truncation report:", trunc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ltr = DataLoader(enc_tr, batch_size=a.bs, shuffle=True, collate_fn=lambda b: collate(b, pad))
    lva = DataLoader(enc_va, batch_size=a.bs, shuffle=False, collate_fn=lambda b: collate(b, pad))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(ltr) / a.grad_accum) * a.epochs
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp == torch.float16)
    json.dump({**vars(a), "lora": cfg.to_dict() if hasattr(cfg, "to_dict") else {}, "truncation": trunc,
               "n_train": len(enc_tr), "n_val": len(enc_va), "optimizer_steps": steps,
               "amp_dtype": str(amp)}, open(out / "train_config.json", "w"), indent=2, default=str)

    log = open(out / "train_log.jsonl", "w")
    best, step, t0 = float("inf"), 0, time.time()
    model.train()
    for ep in range(a.epochs):
        run, cnt = 0.0, 0
        for i, (ids, att, lab) in enumerate(ltr):
            ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
            scaler.scale(loss).backward(); run += loss.item() * a.grad_accum; cnt += 1
            if (i + 1) % a.grad_accum == 0 or i + 1 == len(ltr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); step += 1
                if step % 5 == 0:
                    rec = {"step": step, "epoch": ep + 1, "train_loss": run / cnt, "lr": sch.get_last_lr()[0],
                           "elapsed_s": round(time.time() - t0)}
                    print(rec); log.write(json.dumps(rec) + "\n"); log.flush(); run, cnt = 0.0, 0
        vl = eval_loss(model, lva, dev, amp)
        rec = {"step": step, "epoch": ep + 1, "val_loss": vl}
        print(rec); log.write(json.dumps(rec) + "\n"); log.flush()
        model.save_pretrained(out / "last")
        if vl < best:
            best = vl; model.save_pretrained(out / "best"); print("saved best ->", out / "best")
    tok.save_pretrained(out / "best")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rows = [json.loads(l) for l in open(out / "train_log.jsonl")]
        t = [r for r in rows if "train_loss" in r]; v = [r for r in rows if "val_loss" in r]
        plt.figure(figsize=(5, 3.2)); plt.plot([r["step"] for r in t], [r["train_loss"] for r in t], label="train")
        plt.plot([r["step"] for r in v], [r["val_loss"] for r in v], "o-", label="validation")
        plt.xlabel("optimizer step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "curves.png", dpi=200)
    except Exception as e:
        print("plot skipped:", e)
    print("done. best val loss", best)


if __name__ == "__main__":
    main()
