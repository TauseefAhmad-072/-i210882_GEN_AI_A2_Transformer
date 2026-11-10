"""Benchmarks for Tasks 3, 4, 5 (warm-up, synchronized, repeated timing; peak memory; token equality).
  python -m src.bench kv   --adapter outputs/lora/best
  python -m src.bench spec --adapter outputs/lora/best --draft Qwen/Qwen2.5-0.5B-Instruct
  python -m src.bench attn
"""
import argparse
import json
import statistics as st
from pathlib import Path

import psutil
import torch

from .data import build_prompt, load_split
from .decoding import generate, sync


def measure(fn, device, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    runs, res = [], None
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        sync(device)
        res = fn()
        sync(device)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else psutil.Process().memory_info().rss / 2**20
        runs.append((res.prefill_s, res.decode_s, res.tok_per_s, res.decode_tok_per_s, peak))
    mean = lambda i: st.mean(r[i] for r in runs)
    sd = lambda i: st.pstdev(r[i] for r in runs)
    return res, {"prefill_s": mean(0), "decode_s": mean(1), "tok_per_s": mean(2), "tok_per_s_std": sd(2),
                 "decode_tok_per_s": mean(3), "peak_mem_mb": max(r[4] for r in runs),
                 "mem_metric": "cuda_peak_allocated" if device.startswith("cuda") else "process_rss"}


def prompt_ids(tok, n_tokens, device):
    """First n_tokens of a long concatenation of validation prompts (val only, never test)."""
    text = ""
    for r in load_split("val"):
        text += build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"])[0] + "\n"
        if len(text) > 40000:
            break
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :n_tokens]
    return ids.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["kv", "spec", "attn"])
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--in_lens", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--out_lens", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    Path("results").mkdir(exist_ok=True)

    if a.what == "attn":
        from .attention import benchmark
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        rows = benchmark(device=dev)
        json.dump({"device": dev, "rows": rows}, open("results/bench_attention.json", "w"), indent=2)
        for r in rows:
            print(r)
        return

    from .pipeline import load_model
    adapter = None if a.adapter.lower() == "none" else a.adapter
    model, tok = load_model(a.base, adapter)
    dev = str(next(model.parameters()).device)
    draft = load_model(a.draft)[0] if a.what == "spec" else None
    rows = []
    for n_in in a.in_lens:
        ids = prompt_ids(tok, n_in, dev)
        for n_out in a.out_lens:
            modes = ["no_cache", "kv_cache"] if a.what == "kv" else ["kv_cache"] + [("speculative", k) for k in a.k]
            ref_tokens = None
            for mode in modes:
                m, k = (mode if isinstance(mode, tuple) else (mode, 4))
                res, stats = measure(lambda: generate(m, model, ids, n_out, None, draft, k), dev, a.repeats)
                if ref_tokens is None:
                    ref_tokens = res.tokens
                first_mismatch = next((i for i, (x, y) in enumerate(zip(ref_tokens, res.tokens)) if x != y), None)
                row = {"input_len": ids.size(1), "output_len": n_out, "mode": m, "k": k if m == "speculative" else None,
                       **{kk: round(v, 4) if isinstance(v, float) else v for kk, v in stats.items()},
                       "tokens_equal_to_reference": res.tokens == ref_tokens, "first_mismatch": first_mismatch,
                       "spec_stats": res.stats if m == "speculative" else None}
                rows.append(row); print(row)
    json.dump({"device": dev, "dtype": str(next(model.parameters()).dtype), "base": a.base, "adapter": adapter,
               "draft": a.draft if draft is not None else None, "rows": rows},
              open(f"results/bench_{a.what}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
