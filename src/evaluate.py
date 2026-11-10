"""Tasks 1/2: evaluate base vs fine-tuned model on val/test with identical decoding settings.
  python -m src.evaluate --split val  --adapter none               (before fine-tuning)
  python -m src.evaluate --split val  --adapter outputs/lora/best  (after)
  python -m src.evaluate --split test --adapter outputs/lora/best --final   (ONLY once, at the end)
"""
import argparse
import json
from pathlib import Path

import torch

from .data import build_prompt, load_split, parse_output
from .metrics import evaluate_predictions
from .pipeline import load_model


@torch.no_grad()
def predict(model, tok, recs, bs=4, max_prompt_tokens=3500, max_new_tokens=64):
    dev = next(model.parameters()).device
    prompts = [build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"], max_prompt_tokens) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: len(prompts[i][0]))
    preds = [None] * len(recs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i][0] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            lab, ev, valid = parse_output(t, recs[i]["criterion_type"])
            preds[i] = {"label": lab, "evidence": ev, "valid": valid, "raw": t, "dropped": prompts[i][1]}
        print(f"  {min(s + bs, len(order))}/{len(order)}", end="\r")
    print()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        raise SystemExit("Refusing to evaluate on test without --final (do not tune on test).")
    adapter = None if a.adapter.lower() == "none" else a.adapter
    tag = a.tag or ("finetuned" if adapter else "base")
    recs = load_split(a.split)
    if a.limit:
        recs = recs[:a.limit]
    model, tok = load_model(a.base, adapter)
    preds = predict(model, tok, recs, a.bs)
    m = evaluate_predictions(recs, preds)
    out = Path("results"); out.mkdir(exist_ok=True)
    meta = {"base": a.base, "adapter": adapter, "split": a.split, "n": len(recs), "decoding": "greedy, max_new_tokens=64, rep_penalty=1.0, HF generate",
            "dtype": str(next(model.parameters()).dtype)}
    json.dump({"meta": meta, "metrics": m}, open(out / f"eval_{tag}_{a.split}.json", "w"), indent=2)
    with open(out / f"preds_{tag}_{a.split}.jsonl", "w") as f:
        for r, p in zip(recs, preds):
            f.write(json.dumps({"patient_id": r["patient_id"], "trial_id": r["trial_id"], "type": r["criterion_type"],
                                "criterion": r["criterion_text"], "gold": r["label"], "gold_evidence": r["evidence"],
                                "pred": p["label"], "pred_evidence": p["evidence"], "raw": p["raw"],
                                "truncated_ids": p["dropped"]}) + "\n")
    for k in ("overall", "inclusion", "exclusion"):
        c, e = m[k]["classification"], m[k]["evidence"]
        print(f"[{tag}/{a.split}] {k:9s} n={c.get('n')} acc={c.get('accuracy', 0):.3f} macroF1={c.get('macro_f1', 0):.3f} "
              f"invalid={c.get('invalid_outputs')} | evidence micro-F1={e['micro']['f1']:.3f} macro-F1={e['macro'].get('f1', 0):.3f}")


if __name__ == "__main__":
    main()
